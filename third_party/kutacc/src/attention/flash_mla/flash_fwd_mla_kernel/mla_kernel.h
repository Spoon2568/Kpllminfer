/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
 *
 * Licensed under a modified version of the MIT license. See LICENSE in the project root for license information.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#pragma once

#include <arm_sve.h>

#include "kutacc.h"
#include "../common.h"
#include "kernel_traits.h"
#include "kernels/mla_bf16_32x64.h"

namespace kutacc {

template <typename Kernel_traits, bool Is_causal>
void flash_fwd_splitkv_mla_kernel(const FlashMLAFwdParams &params)
{
    static constexpr int kBlockN = Kernel_traits::kBlockN;
    static constexpr int kBlockM = Kernel_traits::kBlockM;
    static constexpr int kHeadDim = Kernel_traits::kHeadDim;
    static constexpr int kHeadDimV = Kernel_traits::kHeadDimV;
    static PersistentBuffer pbuff;
    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;

    const int num_m_block = ceil_div(params.seqlen_q, Kernel_traits::kBlockM);
    const int task_num = params.num_thread_parts * params.h * num_m_block;

    char *tiling_buffer_ptr;
    if (params.tiling_buffer_ptr != nullptr) {
        tiling_buffer_ptr = (char *)params.tiling_buffer_ptr;
    } else {
        tiling_buffer_ptr = (char *)pbuff.malloc_buffer(kutacc::get_thread_num() * extra_buffer_size_per_thread);
    }

    kutacc::parallel_for(0, task_num, 1, [&](int64_t start, int64_t end) {
        for (int task_id = start; task_id < end; ++task_id) {
            const int m_block = task_id / params.h / params.num_thread_parts;
            const int partition_idx = task_id / params.h % params.num_thread_parts;
            const int bidh = task_id % params.h;

            const int *tile_scheduler_metadata_ptr = params.tile_scheduler_metadata_ptr +
                partition_idx * TileSchedulerMetaDataSize;
            int begin_idx = tile_scheduler_metadata_ptr[0];
            int begin_seqlen = tile_scheduler_metadata_ptr[1];
            int end_idx = tile_scheduler_metadata_ptr[2];
            int end_seqlen = tile_scheduler_metadata_ptr[3];
            int begin_n_split_idx = tile_scheduler_metadata_ptr[4];

            if (begin_idx < params.b) {
                for (int batch_id = begin_idx; batch_id <= end_idx; ++batch_id) {
                    const int n_split_idx = batch_id == begin_idx ? begin_n_split_idx : 0;
                    const int seqlen_k = params.cu_seqlens_k[batch_id];
                    const int n_block_min = batch_id == begin_idx ? begin_seqlen / kBlockN : 0;
                    const int n_block_max =
                        batch_id == end_idx ? ceil_div(end_seqlen, kBlockN) : ceil_div(seqlen_k, kBlockN);
                    const bool NoSplit = n_block_min == 0 && n_block_max == ceil_div(seqlen_k, kBlockN);

                    compute_attn_1rowblock_bf16_32x64<Kernel_traits, Is_causal>(
                        params, batch_id, bidh, m_block, n_split_idx, seqlen_k, n_block_min, n_block_max,
                        NoSplit, tiling_buffer_ptr);
                }
            }
        }
    });
}

template <typename Element, typename ElementAccum, typename index_t, int kHeadDimV, int kMaxSplits>
void flash_fwd_splitkv_mla_combine_kernel(const FlashMLAFwdParams &params)
{
    if (params.num_splits_ptr[params.b] == params.b) {
        return ;
    }
    const int bhs = params.b * params.h * params.seqlen_q;
    kutacc::parallel_for(0, bhs, 1, [&](int64_t start, int64_t end) {
        for (int bidx = start; bidx < end; ++bidx) {
            const int hs = params.h * params.seqlen_q;
            const int batch_idx = bidx / hs;
            const int hs_idx = bidx % hs;

            const int split_offset = params.num_splits_ptr[batch_idx];
            const int actual_num_splits = params.num_splits_ptr[batch_idx + 1] - split_offset;
            
            KUTACC_CHECK(actual_num_splits <= kMaxSplits, "");

            if (actual_num_splits == 1) {
                continue;
            }

            const index_t row_offset_lseaccum = split_offset * hs + hs_idx;
            const index_t row_offset_lse = bidx;

            const ElementAccum *LSEaccum =
                reinterpret_cast<ElementAccum *>(params.softmax_lseaccum_ptr) + row_offset_lseaccum;
            ElementAccum *LSE = reinterpret_cast<ElementAccum *>(params.softmax_lse_ptr) + row_offset_lse;

            alignas(64) ElementAccum LseScale[kMaxSplits];
            alignas(64) float local_lse[kMaxSplits];

            float max_lse = -INFINITY;
            for (int split = 0; split < actual_num_splits; ++split) {
                local_lse[split] = LSEaccum[split * hs];
                if (max_lse < local_lse[split]) {
                    max_lse = local_lse[split];
                }
            }

            float sum_lse_exp = 0;
            for (int split = 0; split < actual_num_splits; ++split) {
                sum_lse_exp += expf(local_lse[split] - max_lse);
            }

            float global_lse =
                (sum_lse_exp == 0.f || sum_lse_exp != sum_lse_exp) ? INFINITY : logf(sum_lse_exp) + max_lse;

            LSE[0] = global_lse;
            
            for (int split = 0; split < actual_num_splits; ++split) {
                LseScale[split] = expf(local_lse[split] - global_lse);
            }
            
            const index_t row_offset_oaccum = (split_offset * hs + hs_idx) * kHeadDimV;
            const ElementAccum *Oaccum = reinterpret_cast<ElementAccum *>(params.oaccum_ptr) + row_offset_oaccum;

            const int head_idx = (bidx - batch_idx * hs) / params.seqlen_q;
            const int row = bidx - batch_idx * hs - head_idx * params.seqlen_q;

            Element *O = reinterpret_cast<Element *>(params.o_ptr) + batch_idx * params.o_batch_stride +
                         head_idx * params.o_head_stride + row * params.o_row_stride;

            svbool_t ptrue = svptrue_b16();
            for (int i = 0; i < kHeadDimV; i += 32) {
                svfloat32_t sve00 = svdup_f32(0);
                svfloat32_t sve01 = svdup_f32(0);
                for (int split = 0; split < actual_num_splits; ++split) {
                    sve00 = svadd_f32_m(ptrue, sve00,
                        svmul_n_f32_m(ptrue, svld1_f32(ptrue, Oaccum + hs * kHeadDimV * split + i), LseScale[split]));
                    sve01 = svadd_f32_m(ptrue, sve01,
                        svmul_n_f32_m(ptrue,
                            svld1_f32(ptrue, Oaccum + hs * kHeadDimV * split + i + 16), LseScale[split]));
                }
                svfloat32_t sve02 = svuzp1_f32(sve00, sve01);
                svfloat32_t sve03 = svuzp2_f32(sve00, sve01);
                svbfloat16_t sve0 = svcvt_bf16_f32_x(ptrue, sve02);
                sve0 = svcvtnt_bf16_f32_x(sve0, ptrue, sve03);
                svst1_bf16(ptrue, O + i, sve0);
            }
        }
    });
}

} // namespace kutacc
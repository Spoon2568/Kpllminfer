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
#include <tuple>
#include <unordered_map>
#include <arm_sve.h>
#include <arm_sme.h>

#include "kernel.h"
#include "linear.h"
#include "gemm_tiling.h"

namespace kutacc {

void gemm_init_tiling_plan()
{
    int64_t num_threads = kutacc::get_thread_num();
    if (num_threads == 32) {
        tiling_plan = tiling_plan_32;
        bgemm_tiling_plan = bgemm_tiling_plan_32;
    }
}

inline void reduce(int64_t m, int64_t n, int64_t bk, bfloat16_t *origin_c, bfloat16_t *tmpc, int64_t start, int64_t end)
{
    svbfloat16_t zero = svdup_bf16(0);
    for (int64_t i = start; i < end; i += svcnth()) {
        svbool_t p = svwhilelt_b16(i, end);
        svbfloat16_t s = svld1(p, tmpc + i);
        svfloat32_t s0 = svreinterpret_f32(svzip1(zero, s));
        svfloat32_t s1 = svreinterpret_f32(svzip2(zero, s));
        for (int64_t k = 1; k < bk; ++k) {
            svbfloat16_t v = svld1(p, tmpc + i + k * m * n);
            s0 = svadd_m(svptrue_b32(), s0, svreinterpret_f32(svzip1(zero, v)));
            s1 = svadd_m(svptrue_b32(), s1, svreinterpret_f32(svzip2(zero, v)));
        }
        svst1(p, origin_c + i, svuzp1(svcvt_bf16_x(svptrue_b32(), s0), svcvt_bf16_x(svptrue_b32(), s1)));
    }
}

void igemm_bdq(int64_t m, int64_t n, int64_t k, MatrixTilingBlock t, int8_t *act_ptr, int8_t *weight_ptr,
    float *act_scale_ptr, float *weight_scale_ptr, bfloat16_t *output_ptr, bfloat16_t *tmpc)
{
    int64_t tile_m = std::get<0>(t);
    int64_t tile_n = std::get<1>(t);
    int64_t tile_k = std::get<2>(t);
    int64_t blocks_in_m = m / tile_m;
    int64_t blocks_in_n = n / tile_n;
    int64_t blocks_in_k = k / tile_k;
    int64_t num_threads = kutacc::get_thread_num();
    kutacc::parallel_for(0, num_threads, 1, [&](int64_t, int64_t) {
        int64_t thread_id = kutacc::get_thread_id();
        auto [idx_m, idx_n, idx_k] = get_idxs(thread_id, blocks_in_m, blocks_in_n, blocks_in_k);
        int8_t *a = act_ptr + idx_m * tile_m * k + idx_k * tile_m * tile_k;
        int8_t *b = weight_ptr + idx_n * tile_n * k + idx_k * tile_n * tile_k;
        bfloat16_t *c = output_ptr + n * m * idx_k + idx_m * tile_m * n + idx_n * tile_n;
        float *cscales = act_scale_ptr + idx_m * tile_m;
        float *rscales = weight_scale_ptr + idx_n * tile_n;
        bfloat16_t alpha = 1;
        bfloat16_t beta = 0;
        if (blocks_in_k == 1) {
            igemmbdq_prepack_ab_alpha1_beta0(tile_m, tile_n, n, tile_k, a, b, c, rscales, cscales);
        } else {
            bfloat16_t *nowc = tmpc + n * m * idx_k + idx_m * tile_m * n + idx_n * tile_n;
            igemmbdq_prepack_ab_alpha1_beta0(tile_m, tile_n, n, tile_k, a, b, nowc, rscales, cscales);
            kutacc::parallel_region_barrier();
            int64_t start = m * n / num_threads * thread_id;
            int64_t end = start + m * n / num_threads;
            reduce(m, n, blocks_in_k, output_ptr, tmpc, start, end);
        }
    });
}

void igemm_pack(int64_t r, int64_t c, int64_t split_r, int64_t split_c, int8_t *input_ptr, int8_t *output_ptr,
    bool with_idx, int *idx, int64_t ldc)
{
    int64_t num_threads = r / split_r * c / split_c;
    int64_t max_threads = kutacc::get_thread_num();
    int64_t z_threads = 1;
    if (max_threads > num_threads && split_r % (max_threads / num_threads * 16) == 0) {
        z_threads = max_threads / num_threads;
        split_r /= z_threads;
        num_threads = max_threads;
    }
    if (ldc == 0) {
        ldc = c;
    }
    kutacc::parallel_for(0, num_threads, 1, [&](int64_t s, int64_t e) {
        if (s < e) {
            int64_t thread_id = kutacc::get_thread_id();
            int64_t z = thread_id % z_threads;
            int64_t tmp = thread_id / z_threads;
            int64_t blocks = c / split_c;
            int64_t x = tmp / blocks;
            int64_t y = tmp % blocks;
            if (!with_idx)
                igemm_pack(split_r, split_c, input_ptr + (x * z_threads + z) * split_r * c + y * split_c, ldc,
                    output_ptr + thread_id * split_r * split_c);
            else
                igemm_pack_with_idx(split_r, split_c, input_ptr + y * split_c, ldc,
                    (x * z_threads + z) * split_r, idx, output_ptr + thread_id * split_r * split_c);
        }
    });
}

void bgemm(int64_t m, int64_t n, int64_t k, MatrixTilingBlock t, bfloat16_t *act_ptr, bfloat16_t *weight_ptr,
    bfloat16_t *output_ptr, bfloat16_t *tmpc)
{
    int64_t tile_m = std::get<0>(t);
    int64_t tile_n = std::get<1>(t);
    int64_t tile_k = std::get<2>(t);
    int64_t blocks_in_m = m / tile_m;
    int64_t blocks_in_n = n / tile_n;
    int64_t blocks_in_k = k / tile_k;
    int64_t num_threads = kutacc::get_thread_num();
    kutacc::parallel_for(0, num_threads, 1, [&](int64_t, int64_t) {
        int64_t thread_id = kutacc::get_thread_id();
        auto [idx_m, idx_n, idx_k] = get_idxs(thread_id, blocks_in_m, blocks_in_n, blocks_in_k);
        bfloat16_t *a = act_ptr + idx_m * tile_m * k + idx_k * tile_m * tile_k;
        bfloat16_t *b = weight_ptr + idx_n * tile_n * k + idx_k * tile_n * tile_k;
        bfloat16_t *c = output_ptr + n * m * idx_k + idx_m * tile_m * n + idx_n * tile_n;
        if (blocks_in_k == 1) {
            bgemm_prepack_ab_alpha1_beta0(tile_m, tile_n, n, tile_k, a, b, c);
        } else {
            bfloat16_t *nowc = tmpc + n * m * idx_k + idx_m * tile_m * n + idx_n * tile_n;
            bgemm_prepack_ab_alpha1_beta0(tile_m, tile_n, n, tile_k, a, b, nowc);
            kutacc::parallel_region_barrier();
            int64_t start = m * n / num_threads * thread_id;
            int64_t end = start + m * n / num_threads;
            reduce(m, n, blocks_in_k, output_ptr, tmpc, start, end);
        }
    });
}

void bgemm_pack(int64_t r, int64_t c, int64_t split_r, int64_t split_c, bfloat16_t *input_ptr,
    bfloat16_t *output_ptr)
{
    int64_t num_threads = r / split_r * c / split_c;
    int64_t max_threads = kutacc::get_thread_num();
    int64_t z_threads = 1;
    if (max_threads > num_threads && split_r % (max_threads / num_threads * 16) == 0) {
        z_threads = max_threads / num_threads;
        split_r /= z_threads;
        num_threads = max_threads;
    }
    kutacc::parallel_for(0, num_threads, 1, [&](int64_t s, int64_t e) {
        if (s < e) {
            int64_t thread_id = kutacc::get_thread_id();
            int64_t z = thread_id % z_threads;
            int64_t tmp = thread_id / z_threads;
            int64_t blocks = c / split_c;
            int64_t x = tmp / blocks;
            int64_t y = tmp % blocks;
            bgemm_pack(split_r, split_c, input_ptr + (x * z_threads + z) * split_r * c + y * split_c, c,
                output_ptr + thread_id * split_r * split_c);
        }
    });
}

void igemm_gateup(int64_t total_bs, int64_t K, int64_t N, int64_t num_experts, int64_t lda, int64_t ldas,
    int8_t *acts, int8_t *weights, float *acts_scale, float *weights_scale, int *token_ids, int *experts_offset,
    bfloat16_t *output, int8_t *tmpx, float *tmpy, float *tmp_scales)
{
    for (int64_t i = 0; i < num_experts; i++) {
        int64_t bs = experts_offset[i + 1] - experts_offset[i];
        int64_t bs_offset = experts_offset[i] - experts_offset[0];
        if (bs == 0) {
            continue;
        }
        for (int64_t step = 0; step < bs; step += FUSEDMOE_TILEBUF) {
            int64_t now_bs = std::min(FUSEDMOE_TILEBUF, bs - step);
            int *ids = token_ids + experts_offset[i] + step;
            MatrixTilingBlock t = igemm_find_optimal_tiling_plan(FUSEDMOE_TILEBUF, N, K, kutacc::get_thread_num());
            int64_t tile_m = now_bs;
            int64_t tile_n = std::get<1>(t);
            int64_t tile_k = std::get<2>(t);

            // 1. Gather scale
            for (int64_t ib = 0; ib < now_bs; ib++) {
                tmp_scales[ib] = acts_scale[ids[ib] * ldas];
            }

            // 2. Pack act with ids
            igemm_pack(now_bs, K, tile_m, tile_k, acts, tmpx, 1, ids, lda);

            // 3. Call IGEMMBDQ kernel
            igemm_bdq(now_bs, N, K, MatrixTilingBlock(tile_m, tile_n, tile_k), tmpx, weights + i * N * K, tmp_scales,
                weights_scale + i * N, output + (bs_offset + step) * N, reinterpret_cast<bfloat16_t *>(tmpy));
        }
    }
}

void igemm_down(int64_t total_bs, int64_t K, int64_t N, int64_t num_experts, int8_t *acts, int8_t *weights,
    float *acts_scale, float *weights_scale, int *experts_offset, bfloat16_t *output, int8_t *tmpx, float *tmpy)
{
    for (int64_t i = 0; i < num_experts; i++) {
        int64_t bs = experts_offset[i + 1] - experts_offset[i];
        int64_t bs_offset = experts_offset[i] - experts_offset[0];
        if (bs == 0) {
            continue;
        }
        for (int64_t step = 0; step < bs; step += FUSEDMOE_TILEBUF) {
            int64_t now_bs = std::min(FUSEDMOE_TILEBUF, bs - step);
            MatrixTilingBlock t = igemm_find_optimal_tiling_plan(FUSEDMOE_TILEBUF, N, K, kutacc::get_thread_num());
            int64_t tile_m = now_bs;
            int64_t tile_n = std::get<1>(t);
            int64_t tile_k = std::get<2>(t);
            // 1. Pack act
            igemm_pack(now_bs, K, tile_m, tile_k, acts + (bs_offset + step) * K, tmpx);

            // 2. Call IGEMMBDQ kernel
            igemm_bdq(now_bs, N, K, MatrixTilingBlock(tile_m, tile_n, tile_k), tmpx, weights + i * N * K,
                acts_scale + bs_offset + step, weights_scale + i * N, output + (bs_offset + step) * N,
                reinterpret_cast<bfloat16_t *>(tmpy));
        }
    }
}

void batched_gemm_pack(int64_t bs, int64_t m, int64_t n, int64_t stride_bs, int64_t stride_m, void *src, void *dst,
    int64_t dtype)
{
    int64_t num_threads = kutacc::get_thread_num();
    int64_t m_threads = std::min(num_threads / bs, m / 32);
    int64_t step_m = m / m_threads;

    kutacc::parallel_for(0, m_threads * bs, 1, [&](int64_t, int64_t) {
        int64_t tid = kutacc::get_thread_id();
        int64_t bi = tid / m_threads;
        int64_t m_id = tid % m_threads;

        uint8_t *src_off = (uint8_t *)src + (bi * stride_bs + m_id * step_m * stride_m) * dtype;
        uint8_t *dst_off = (uint8_t *)dst + (bi * m * n + m_id * step_m * n) * dtype;
        char trans;
        char pack_matrix;
        if (dtype == 1) {
            trans = 'T';
            pack_matrix = 'A';
        } else if (dtype == 2) {
            trans = 'N';
            pack_matrix = 'B';
        }
        gemm_ex_t extra{.nThreads = 1};
        bgemm_pack_woqs8_ex_(&pack_matrix, &trans, step_m, n, src_off, stride_m, dst_off, &extra);
    });
}

void batched_gemm_woqs8(int64_t bs, int64_t m, int64_t n, int64_t k, int64_t stride_bs, int64_t stride_m,
    bfloat16_t *act, int8_t *weight, bfloat16_t *out, float *rscale, float *cscale)
{
    int64_t num_threads = kutacc::get_thread_num();
    int64_t m_threads =
        std::min(num_threads / bs, m / std::min({m, n, 32L}));  // bgemm_woqs8_ex_ kernel support minimum value of 32
    int64_t step_m = m / m_threads;
    kutacc::parallel_for(0, m_threads * bs, 1, [&](int64_t, int64_t) {
        int64_t tid = kutacc::get_thread_id();
        int64_t m_id = tid % m_threads;
        int64_t bi = tid / m_threads;
        const float *quantArgs[2] = {};
        if (rscale) {
            quantArgs[0] = rscale + bi * n;
        }
        if (cscale) {
            quantArgs[1] = cscale + bi * k;
        }
        auto weight_data = weight + bi * n * k;
        bfloat16_t *act_data = act + bi * m * k + m_id * step_m * k;
        bfloat16_t *out_data = out + bi * stride_bs + m_id * step_m * stride_m;
        char transa = 'N';
        char transb = 'T';
        bfloat16_t alpha = 1;
        bfloat16_t beta = 0;
        gemm_ex_t extra{.nThreads = 1, .quantArgs = quantArgs, .prepackA = true, .prepackB = true};
        int N = n;
        int M = step_m;
        int K = k;
        int lda = k;
        int ldb = k;
        int ldc = stride_m;
        bgemm_woqs8_ex_(&transb, &transa, &N, &M, &K, &alpha, weight_data, &ldb, act_data, &lda, &beta,
            out_data, &ldc, &extra);
    });
}

} // namespace kutacc
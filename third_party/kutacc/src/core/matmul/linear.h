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

#include <arm_bf16.h>
#include "kutacc.h"

namespace kutacc {

/*
 * GEMM shape define:
 * Matrix A shape[m, k]
 * Matrix B shape[n, k]
 * Matrix C shape[m, n]
 *
 * gemm_tiling t[tm, tn, tk], assume m/tm * n/tn * k/tk == num_threads
 *
 */

struct tiling_block_hash {
    std::size_t operator()(const MatrixTilingBlock &t) const
    {
        std::size_t h1 = std::hash<int64_t>{}(std::get<0>(t));
        std::size_t h2 = std::hash<int64_t>{}(std::get<1>(t));
        std::size_t h3 = std::hash<int64_t>{}(std::get<2>(t));
        return h1 ^ (h2 << 4) ^ (h3 << 8);
    }
};

extern std::unordered_map<MatrixTilingBlock, MatrixTilingBlock, tiling_block_hash> tiling_plan;
extern std::unordered_map<MatrixTilingBlock, MatrixTilingBlock, tiling_block_hash> tiling_plan_32;
extern std::unordered_map<MatrixTilingBlock, MatrixTilingBlock, tiling_block_hash> bgemm_tiling_plan;
extern std::unordered_map<MatrixTilingBlock, MatrixTilingBlock, tiling_block_hash> bgemm_tiling_plan_32;

void gemm_init_tiling_plan();

MatrixTilingBlock igemm_find_optimal_tiling_plan(int64_t M, int64_t N, int64_t K, int64_t num_threads);
MatrixTilingBlock bgemm_find_optimal_tiling_plan(int64_t M, int64_t N, int64_t K, int64_t num_threads);

void igemm_bdq(int64_t m, int64_t n, int64_t k, MatrixTilingBlock t, int8_t *act_ptr, int8_t *weight_ptr,
    float *act_scale_ptr, float *weight_scale_ptr, bfloat16_t *output_ptr, bfloat16_t *tmpc);
void igemm_pack(int64_t r, int64_t c, int64_t split_r, int64_t split_c, int8_t *input_ptr, int8_t *output_ptr,
    bool with_idx, int *idx, int64_t ldc);

void bgemm(int64_t m, int64_t n, int64_t k, MatrixTilingBlock t, bfloat16_t *act_ptr, bfloat16_t *weight_ptr,
    bfloat16_t *output_ptr, bfloat16_t *tmpc);
void bgemm_pack(int64_t r, int64_t c, int64_t split_r, int64_t split_c, bfloat16_t *input_ptr,
    bfloat16_t *output_ptr);

void igemm_gateup(int64_t total_bs, int64_t K, int64_t N, int64_t num_experts, int64_t lda, int64_t ldas,
    int8_t *acts, int8_t *weights, float *acts_scale, float *weights_scale, int *token_ids, int *experts_offset,
    bfloat16_t *output, int8_t *tmpx, float *tmpy, float *tmp_scales);
void igemm_down(int64_t total_bs, int64_t K, int64_t N, int64_t num_experts, int8_t *acts, int8_t *weights,
    float *acts_scale, float *weights_scale, int *experts_offset, bfloat16_t *output, int8_t *tmpx, float *tmpy);

// dtype=1 weight int8, dtype=2 activation bfloat16
void batched_gemm_pack(int64_t bs, int64_t m, int64_t n, int64_t stride_bs, int64_t stride_m, void *src, void *dst,
    int64_t dtype);
void batched_gemm_woqs8(int64_t bs, int64_t m, int64_t n, int64_t k, int64_t stride_bs, int64_t stride_m,
    bfloat16_t *act, int8_t *weight, bfloat16_t *out, float *rscale, float *cscale);
}

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

struct gemm_ex_t {
    int nThreads;
    const void *quantArgs;
    bool prepackA;
    bool prepackB;
    int batchCount;
    int splitK;
};

__arm_new("za") void igemm_pack(int m, int n, int8_t *src, int ldc, int8_t *dst) __arm_streaming;

__arm_new("za") void igemm_pack_with_idx(int m, int n, int8_t *src, int ldc, int m_off, int *idx,
    int8_t *dst) __arm_streaming;

__arm_new("za") void igemmbdq_prepack_ab_alpha1_beta0(int m, int n, int ldc, int k, int8_t *a, int8_t *b, bfloat16_t *c,
    const float *rscales, const float *cscales, int macro_kernel_m = 256, int macro_kernel_n = 512,
    int macro_kernel_k = 1024) __arm_streaming;

__arm_new("za") void igemm_prepack_ab_alpha1_beta0(int m, int n, int ldc, int k, int8_t *a, int8_t *b, int *c,
    int macro_kernel_m = 256, int macro_kernel_n = 512, int macro_kernel_k = 1024) __arm_streaming;

__arm_new("za") void bgemm_pack_a(int lda, bfloat16_t *a, bfloat16_t *pack_a, const int m_start, const int m_end,
    const int k_start, const int k_end) __arm_streaming;

__arm_new("za") void bgemm_prepack_ab_16_mod_16_mod(int block_k, int m, int n, int ldc, int k, bfloat16_t *a,
    bfloat16_t *b, bfloat16_t *beta, bfloat16_t *c) __arm_streaming;

__arm_new("za") void bgemm_pack(int m, int n, bfloat16_t *src, int ldc, bfloat16_t *dst) __arm_streaming;

__arm_new("za") void bgemm_prepack_ab_alpha1_beta0(int m, int n, int ldc, int k, bfloat16_t *a, bfloat16_t *b,
    bfloat16_t *c, int macro_kernel_m = 256, int macro_kernel_n = 512, int macro_kernel_k = 1024) __arm_streaming;

extern "C" void bgemm_pack_woqs8_ex_(const char *matrix_, const char *trans, const int m, const int n, const void *src,
    const int ld, void *dst, const gemm_ex_t *_extra);

extern "C" void bgemm_woqs8_ex_(const char *transA, const char *transB, const int *M, const int *N, const int *K,
    const bfloat16_t *alpha, const int8_t *A, const int *LDA, const bfloat16_t *B, const int *LDB,
    const bfloat16_t *beta, bfloat16_t *C, const int *LDC, const gemm_ex_t *extra);

extern "C" void gemm_ncopy_matrix(int64_t, int64_t, const void *, int64_t, void *);

} // namespace kutacc
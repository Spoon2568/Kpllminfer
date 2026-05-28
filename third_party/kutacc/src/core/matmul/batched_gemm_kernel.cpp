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
#include <cstdint>
#include <cassert>
#include <algorithm>
#include <arm_bf16.h>
#include <arm_sve.h>
#include <arm_sme.h>

#include "kutacc.h"
#include "kernel.h"

namespace kutacc {

constexpr int64_t DEFAULT_BLOCK_M = 256;
constexpr int64_t DEFAULT_BLOCK_N = 1024;
constexpr int64_t DEFAULT_BLOCK_K = 4096;
constexpr int64_t MAX_CPU_NUMBER = 1000;

struct GemmArgs {
    int64_t m;
    int64_t n;
    int64_t k;
    bfloat16_t alpha;
    const void *a;
    int64_t lda;
    const void *b;
    int64_t ldb;
    bfloat16_t beta;
    const void *c;
    int64_t ldc;
    gemm_ex_t extra;
    GemmArgs(int64_t m, int64_t n, int64_t k, bfloat16_t alpha, const void *a, int64_t lda, const void *b, int64_t ldb,
        bfloat16_t beta, const void *c, int64_t ldc, const gemm_ex_t &extra = {})
        : m(m), n(n), k(k), alpha(alpha), a(a), lda(lda), b(b), ldb(ldb), beta(beta), c(c), ldc(ldc), extra(extra)
    {}
};

template <bool RowQuant>
static void bgemm_woqs8_kernel_16(int64_t m, int64_t n, int64_t k, const int8_t *a,
    [[maybe_unused]] int64_t lda, const __bf16 *b, [[maybe_unused]] int64_t ldb, __bf16 *c, int64_t ldc, __bf16 _alpha,
    __bf16 _beta, const float *scale) __arm_streaming
{
    svbool_t pg8 = svptrue_b8();
    svbool_t pg16 = svptrue_b16();
    svbool_t pg32 = svptrue_b32();
    float alpha = _alpha;
    [[maybe_unused]] float beta = _beta;
    const int64_t prefetch_dis = 16;
    for (int64_t ni = 0; ni < n; ni += 16) {
        for (int64_t mi = 0; mi < m; mi += 16) {
            svfloat32_t rscale00;
            svfloat32_t rscale01;
            if constexpr (RowQuant) {
                svfloat32_t rscale0 = svld1(pg32, scale + mi); // 加载个数 = 512/32 = 16 ,加载一次
                rscale00 = svzip1(rscale0, rscale0);
                rscale01 = svzip2(rscale0, rscale0);
            }
            svzero_za();
            for (int64_t ki = 0; ki < k; ki += 4) {
                svint8_t a_values = svld1(pg8, a + mi * k + 16 * ki); // 加载个数 512 / 8 = 64 个

                svint8_t zero_s8 = svdup_s8(0);
                svint16_t zero_s16 = svreinterpret_s16(zero_s8);

                svfloat32_t a00_f32 = svcvt_f32_x(pg32,
                    svextb_x(pg32, svreinterpret_s32(svzip1(svreinterpret_s16(svzip1(a_values, zero_s8)), zero_s16))));
                svfloat32_t a01_f32 = svcvt_f32_x(pg32,
                    svextb_x(pg32, svreinterpret_s32(svzip2(svreinterpret_s16(svzip1(a_values, zero_s8)), zero_s16))));
                svfloat32_t a10_f32 = svcvt_f32_x(pg32,
                    svextb_x(pg32, svreinterpret_s32(svzip1(svreinterpret_s16(svzip2(a_values, zero_s8)), zero_s16))));
                svfloat32_t a11_f32 = svcvt_f32_x(pg32,
                    svextb_x(pg32, svreinterpret_s32(svzip2(svreinterpret_s16(svzip2(a_values, zero_s8)), zero_s16))));

                if (RowQuant) {
                    a00_f32 = svmul_x(pg32, a00_f32, rscale00);
                    a01_f32 = svmul_x(pg32, a01_f32, rscale00);
                    a10_f32 = svmul_x(pg32, a10_f32, rscale01);
                    a11_f32 = svmul_x(pg32, a11_f32, rscale01);
                }
                if (!RowQuant) {
                    svfloat32_t cscale0 = svzip1(svdup_f32(scale[ki]), svdup_f32(scale[ki + 1]));
                    svfloat32_t cscale1 = svzip1(svdup_f32(scale[ki + 2]), svdup_f32(scale[ki + 3]));
                    a00_f32 = svmul_x(pg32, a00_f32, cscale0);
                    a01_f32 = svmul_x(pg32, a01_f32, cscale0);
                    a10_f32 = svmul_x(pg32, a10_f32, cscale1);
                    a11_f32 = svmul_x(pg32, a11_f32, cscale1);
                }

                svbfloat16_t a00_bf16 = svcvt_bf16_x(pg32, a00_f32);
                svbfloat16_t a01_bf16 = svcvt_bf16_x(pg32, a01_f32);
                svbfloat16_t a10_bf16 = svcvt_bf16_x(pg32, a10_f32);
                svbfloat16_t a11_bf16 = svcvt_bf16_x(pg32, a11_f32);
                svbfloat16_t a0 = svuzp1(a00_bf16, a01_bf16);
                svbfloat16_t a1 = svuzp1(a10_bf16, a11_bf16);
                svbfloat16_t b0 = svld1(pg16, b + ni * k + 8 * ki);
                svbfloat16_t b1 = svld1(pg16, b + (ni + 16) * k + 8 * ki);


                if (ki + prefetch_dis < k) {
                    __builtin_prefetch(a + mi * k + 16 * (ki + prefetch_dis), 0, 0);
                    __builtin_prefetch(b + ni * k + 16 * (ki + prefetch_dis), 0, 0);
                }

                svmopa_za32_bf16_m(0, pg16, pg16, b0, a0);
                svmopa_za32_bf16_m(1, pg16, pg16, b0, a1);
                svmopa_za32_bf16_m(2, pg16, pg16, b1, a0);
                svmopa_za32_bf16_m(3, pg16, pg16, b1, a1);
            }
            for (int64_t i = 0; i < 8; i++) {
                svfloat32_t o00 = svread_hor_za32_f32_m(svfloat32_t(), pg32, 0, i);
                svfloat32_t o01 = svread_hor_za32_f32_m(svfloat32_t(), pg32, 1, i);
                svfloat32_t o10 = svread_hor_za32_f32_m(svfloat32_t(), pg32, 2, i);
                svfloat32_t o11 = svread_hor_za32_f32_m(svfloat32_t(), pg32, 3, i);
                o00 = svmul_x(pg32, o00, alpha);
                o01 = svmul_x(pg32, o01, alpha);
                o10 = svmul_x(pg32, o10, alpha);
                o11 = svmul_x(pg32, o11, alpha);
                svbfloat16_t o0 = svuzp1(svcvt_bf16_x(pg32, o00), svcvt_bf16_x(pg32, o01));
                svbfloat16_t o1 = svuzp1(svcvt_bf16_x(pg32, o10), svcvt_bf16_x(pg32, o11));
                svst1(pg16, c + (ni + i) * ldc + mi, o0);
                svst1(pg16, c + (ni + i + 8) * ldc + mi, o1);
            }
        }
    }
}

template <bool RowQuant>
__arm_new("za") static void bgemm_woqs8_enable_matrix(int64_t m, int64_t n, int64_t k, const int8_t *a,
    [[maybe_unused]] int64_t lda, const __bf16 *b, [[maybe_unused]] int64_t ldb, __bf16 *c, int64_t ldc, __bf16 _alpha,
    __bf16 _beta, const float *scale) __arm_streaming
{
    int block_size = std::min(std::min((int)m, (int)n), 32);
    if (block_size == 16) {
        bgemm_woqs8_kernel_16<RowQuant>(m, n, k, a, lda, b, ldb, c, ldc, _alpha, _beta, scale);
        return;
    }

    assert(_beta == 0);
    svbool_t pg8 = svptrue_b8();
    svbool_t pg16 = svptrue_b16();
    svbool_t pg32 = svptrue_b32();
    float alpha = _alpha;
    [[maybe_unused]] float beta = _beta;
    const int64_t prefetch_dis = 16;
    for (int64_t ni = 0; ni < n; ni += 32) {
        for (int64_t mi = 0; mi < m; mi += 32) {
            svfloat32_t rscale00;
            svfloat32_t rscale01;
            svfloat32_t rscale10;
            svfloat32_t rscale11;
            if constexpr (RowQuant) {
                svfloat32_t rscale0 = svld1(pg32, scale + mi);
                svfloat32_t rscale1 = svld1(pg32, scale + mi + 16);
                rscale00 = svzip1(rscale0, rscale0);
                rscale01 = svzip2(rscale0, rscale0);
                rscale10 = svzip1(rscale1, rscale1);
                rscale11 = svzip2(rscale1, rscale1);
            }
            svzero_za();
            for (int64_t ki = 0; ki < k; ki += 2) {
                svint8_t a_values = svld1(pg8, a + mi * k + 32 * ki);
                svint8_t zero_s8 = svdup_s8(0);
                svint16_t zero_s16 = svreinterpret_s16(zero_s8);
                svfloat32_t a00_f32 = svcvt_f32_x(pg32,
                    svextb_x(pg32, svreinterpret_s32(svzip1(svreinterpret_s16(svzip1(a_values, zero_s8)), zero_s16))));
                svfloat32_t a01_f32 = svcvt_f32_x(pg32,
                    svextb_x(pg32, svreinterpret_s32(svzip2(svreinterpret_s16(svzip1(a_values, zero_s8)), zero_s16))));
                svfloat32_t a10_f32 = svcvt_f32_x(pg32,
                    svextb_x(pg32, svreinterpret_s32(svzip1(svreinterpret_s16(svzip2(a_values, zero_s8)), zero_s16))));
                svfloat32_t a11_f32 = svcvt_f32_x(pg32,
                    svextb_x(pg32, svreinterpret_s32(svzip2(svreinterpret_s16(svzip2(a_values, zero_s8)), zero_s16))));
                if (RowQuant) {
                    a00_f32 = svmul_x(pg32, a00_f32, rscale00);
                    a01_f32 = svmul_x(pg32, a01_f32, rscale01);
                    a10_f32 = svmul_x(pg32, a10_f32, rscale10);
                    a11_f32 = svmul_x(pg32, a11_f32, rscale11);
                }
                if (!RowQuant) {
                    svfloat32_t cscale = svzip1(svdup_f32(scale[ki]), svdup_f32(scale[ki + 1]));
                    a00_f32 = svmul_x(pg32, a00_f32, cscale);
                    a01_f32 = svmul_x(pg32, a01_f32, cscale);
                    a10_f32 = svmul_x(pg32, a10_f32, cscale);
                    a11_f32 = svmul_x(pg32, a11_f32, cscale);
                }
                svbfloat16_t a00_bf16 = svcvt_bf16_x(pg32, a00_f32);
                svbfloat16_t a01_bf16 = svcvt_bf16_x(pg32, a01_f32);
                svbfloat16_t a10_bf16 = svcvt_bf16_x(pg32, a10_f32);
                svbfloat16_t a11_bf16 = svcvt_bf16_x(pg32, a11_f32);
                svbfloat16_t a0 = svuzp1(a00_bf16, a01_bf16);
                svbfloat16_t a1 = svuzp1(a10_bf16, a11_bf16);
                svbfloat16_t b0 = svld1(pg16, b + ni * k + 16 * ki);
                svbfloat16_t b1 = svld1(pg16, b + (ni + 16) * k + 16 * ki);
                if (ki + prefetch_dis < k) {
                    __builtin_prefetch(a + mi * k + 16 * (ki + prefetch_dis), 0, 0);
                    __builtin_prefetch(a + (mi + 16) * k + 16 * (ki + prefetch_dis), 0, 0);
                    __builtin_prefetch(b + ni * k + 16 * (ki + prefetch_dis), 0, 0);
                    __builtin_prefetch(b + (ni + 16) * k + 16 * (ki + prefetch_dis), 0, 0);
                }
                svmopa_za32_bf16_m(0, pg16, pg16, b0, a0);
                svmopa_za32_bf16_m(1, pg16, pg16, b0, a1);
                svmopa_za32_bf16_m(2, pg16, pg16, b1, a0);
                svmopa_za32_bf16_m(3, pg16, pg16, b1, a1);
            }
            for (int64_t i = 0; i < 16; i++) {
                svfloat32_t o00 = svread_hor_za32_f32_m(svfloat32_t(), pg32, 0, i);
                svfloat32_t o01 = svread_hor_za32_f32_m(svfloat32_t(), pg32, 1, i);
                svfloat32_t o10 = svread_hor_za32_f32_m(svfloat32_t(), pg32, 2, i);
                svfloat32_t o11 = svread_hor_za32_f32_m(svfloat32_t(), pg32, 3, i);
                o00 = svmul_x(pg32, o00, alpha);
                o01 = svmul_x(pg32, o01, alpha);
                o10 = svmul_x(pg32, o10, alpha);
                o11 = svmul_x(pg32, o11, alpha);
                svbfloat16_t o0 = svuzp1(svcvt_bf16_x(pg32, o00), svcvt_bf16_x(pg32, o01));
                svbfloat16_t o1 = svuzp1(svcvt_bf16_x(pg32, o10), svcvt_bf16_x(pg32, o11));
                svst1(pg16, c + (ni + i) * ldc + mi, o0);
                svst1(pg16, c + (ni + i + 16) * ldc + mi, o1);
            }
        }
    }
}

int64_t GetPrepackOffset(int64_t m, int64_t n, int64_t mOff, int64_t nOff, int64_t blockN)
{
    return mOff * std::min(n - nOff, blockN) + nOff * m;
}

extern "C" void bgemm_woqs8_ex_(const char *transA, const char *transB, const int *M, const int *N, const int *K,
    const bfloat16_t *alpha, const int8_t *A, const int *LDA, const bfloat16_t *B, const int *LDB,
    const bfloat16_t *beta, bfloat16_t *C, const int *LDC, const gemm_ex_t *extra)
{
    auto args = GemmArgs(*M, *N, *K, *alpha, A, *LDA, B, *LDB, *beta, C, *LDC, *extra);
    if (args.extra.nThreads <= 0) {
        args.extra.nThreads = kutacc::get_thread_num();
    }
    assert(std::toupper(*transA) == 'T' && std::toupper(*transB) == 'N');
    assert(args.extra.quantArgs != nullptr);
    assert(args.extra.prepackA && args.extra.prepackB);
    int64_t m = args.m;
    int64_t n = args.n;
    int64_t k = args.k;
    auto a = static_cast<const int8_t *>(args.a);
    auto b = static_cast<const bfloat16_t *>(args.b);
    auto c = static_cast<bfloat16_t *>(const_cast<void *>(args.c));
    auto lda = args.lda;
    auto ldb = args.ldb;
    auto ldc = args.ldc;
    int64_t blockM = DEFAULT_BLOCK_M;
    int64_t blockN = DEFAULT_BLOCK_N;
    int64_t blockK = args.extra.splitK;
    if (blockK == 0) {
        blockK = DEFAULT_BLOCK_K / std::max(sizeof(int8_t), sizeof(bfloat16_t));
    }
    const float *quantPtr[2] = {};
    auto quantArgs = static_cast<const float *const *>(args.extra.quantArgs);
    for (int64_t blockNOff = 0; blockNOff < n; blockNOff += blockN) {
        int64_t stepN = std::min(n - blockNOff, blockN);
        bfloat16_t now_beta = args.beta;
        for (int64_t blockKOff = 0; blockKOff < k; blockKOff += blockK) {
            int64_t stepK = std::min(k - blockKOff, blockK);
            const bfloat16_t *bPtr = b + GetPrepackOffset(n, k, blockNOff, blockKOff, blockK);
            for (int64_t blockMOff = 0; blockMOff < m; blockMOff += blockM) {
                int64_t stepM = std::min(m - blockMOff, blockM);
                const int8_t *aPtr = a + GetPrepackOffset(m, k, blockMOff, blockKOff, blockK);
                auto cPtr = c + blockNOff * ldc + blockMOff;
                assert(stepK % 4 == 0);
                if (quantArgs[0]) {
                    quantPtr[0] = quantArgs[0] + blockMOff;
                    bgemm_woqs8_enable_matrix<true>(stepM, stepN, stepK, aPtr, lda, bPtr, ldb, cPtr, ldc, *alpha, now_beta,
                        quantPtr[0]);
                } else {
                    quantPtr[1] = quantArgs[1] + blockKOff;
                    bgemm_woqs8_enable_matrix<false>(stepM, stepN, stepK, aPtr, lda, bPtr, ldb, cPtr, ldc, *alpha,
                        now_beta, quantPtr[1]);
                }
            }
            now_beta = 1;
        }
    }
}

enum class Matrix : uint8_t { A, B, C };

template <int64_t blockM = 16, int64_t blockN = 4>
inline void GemmNCopyStub(int64_t m, int64_t n, const void *src_, int64_t ld, void *dst_)
{
    auto src = static_cast<const uint8_t *>(src_);
    auto dst = static_cast<uint8_t *>(dst_);
    int64_t stepM = 0;
    for (int64_t blockMOff = 0; blockMOff < m; blockMOff += stepM) {
        stepM = std::min(m - blockMOff, blockM);
        for (int64_t blockNOff = 0; blockNOff < n; blockNOff += blockN) {
            int64_t stepN = std::min(n - blockNOff, blockN);
            for (int64_t mOff = 0; mOff < stepM; mOff++) {
                for (int64_t nOff = 0; nOff < stepN; nOff++) {
                    *dst = src[(blockMOff + mOff) * ld + blockNOff + nOff];
                    dst++;
                }
            }
        }
    }
}

template <Matrix Mat>
void GemmPackOp(int64_t m, int64_t n, const int64_t *rangeM, const int64_t *rangeN, const void *src, int64_t ld,
    void *dst, int64_t splitK)
{
    int64_t blockN = splitK;
    if (blockN == 0) {
        blockN = DEFAULT_BLOCK_K / std::max(sizeof(int8_t), sizeof(bfloat16_t));
    }
    int64_t mStart = 0;
    int64_t mEnd = m;
    int64_t nStart = 0;
    int64_t nEnd = n;
    if (rangeM != nullptr && rangeN != nullptr) {
        mStart = rangeM[0];
        mEnd = rangeM[1];
        nStart = rangeN[0];
        nEnd = rangeN[1];
    }
    for (int64_t blockNOff = nStart; blockNOff < nEnd; blockNOff += blockN) {
        int64_t stepN = std::min(n - blockNOff, blockN);
        if constexpr (Mat == Matrix::A) {
            const void *srcPtr = reinterpret_cast<const int8_t *>(src) + mStart * ld + blockNOff;
            void *dstPtr = reinterpret_cast<int8_t *>(dst) + GetPrepackOffset(m, n, mStart, blockNOff, blockN);
            GemmNCopyStub<32, 2>(mEnd - mStart, stepN, srcPtr, ld, dstPtr);
        } else if constexpr (Mat == Matrix::B) {
            const void *srcPtr = reinterpret_cast<const bfloat16_t *>(src) + mStart * ld + blockNOff;
            void *dstPtr = reinterpret_cast<bfloat16_t *>(dst) + GetPrepackOffset(m, n, mStart, blockNOff, blockN);
            gemm_ncopy_matrix(mEnd - mStart, stepN * 2, srcPtr, ld * 2, dstPtr);
        }
    }
}

template <Matrix Mat>
void GemmPackThread(int64_t m, int64_t n, const void *src, int64_t ld, void *dst, const gemm_ex_t &extra)
{
    int64_t rangeM[MAX_CPU_NUMBER + 1];
    int64_t rangeN[2] = {0, n};
    int64_t unrollSz = 16;

    int64_t rangeLenM;
    {
        int64_t nThreads = extra.nThreads;
        int64_t part_m = (m + (nThreads * unrollSz) - 1) / (nThreads * unrollSz) * unrollSz;
        rangeLenM = (m + part_m - 1) / part_m;
        for (int64_t m_thread_id = 0; m_thread_id < rangeLenM; m_thread_id++) {
            rangeM[m_thread_id] = std::min(m_thread_id * part_m, m);
        }
        rangeM[rangeLenM] = m;
    }
    int64_t nThreads = rangeLenM;
    kutacc::parallel_for(0, nThreads, 1, [&](int64_t start, int64_t end) {
        for (int64_t threadId = start; threadId < end; threadId++) {
            GemmPackOp<Mat>(m, n, rangeM + threadId, rangeN, src, ld, dst, extra.splitK);
        }
    });
}

extern "C" void bgemm_pack_woqs8_ex_(const char *matrix_, const char *trans, const int m, const int n, const void *src,
    const int ld, void *dst, const gemm_ex_t *_extra)
{
    auto matrix = std::toupper(*matrix_) == 'A' ? Matrix::A : std::toupper(*matrix_) == 'B' ? Matrix::B : Matrix::C;
    bool is_trans = (std::toupper(*trans) == 'T');
    assert((matrix == Matrix::A && is_trans) || (matrix == Matrix::B && !is_trans));
    gemm_ex_t extra = *_extra;
    if (extra.nThreads <= 0) {
        extra.nThreads = kutacc::get_thread_num();
    }
    if (extra.nThreads == 1) {
        if (matrix == Matrix::A)
            GemmPackOp<Matrix::A>(m, n, nullptr, nullptr, src, ld, dst, extra.splitK);
        else if (matrix == Matrix::B)
            GemmPackOp<Matrix::B>(m, n, nullptr, nullptr, src, ld, dst, extra.splitK);
    } else {
        if (matrix == Matrix::A)
            GemmPackThread<Matrix::A>(m, n, src, ld, dst, extra);
        else if (matrix == Matrix::B)
            GemmPackThread<Matrix::B>(m, n, src, ld, dst, extra);
    }
}

} // namespace kutacc

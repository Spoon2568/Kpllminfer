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
#include <cstdlib>
#include <cassert>
#include <algorithm>
#include <arm_bf16.h>
#include <arm_sve.h>
#include <arm_sme.h>

#include "kernel.h"
#include "kernel_common.h"

namespace kutacc {

__arm_new("za") void bgemm_pack_a(int lda, bfloat16_t *a, bfloat16_t *pack_a, const int m_start, const int m_end,
    const int k_start, const int k_end) __arm_streaming
{
    svbool_t pg_16_all = svptrue_b16();
    svbool_t pg_32_all = svptrue_b32();
    const int pack_ks = 32, pack_m = 16;  // assume pack_ks % 2 == 0
    const int pack_k_step_4 = pack_ks * 4;
    int pack_pos = 0;
    int ki = k_start, mi = m_start;
    const int prefetch_dis = 4;
    while (mi < m_end) {
        int m_cur_start = mi;
        int m_cur_end = mi + pack_m;
        ki = k_start;
        while (ki < k_end) {
            mi = m_cur_start;
            if (ki + pack_k_step_4 <= k_end) {
                for (; mi < m_cur_end; mi++) {
                    const int m_index = (mi - m_cur_start) * 2;
                    svld1_hor_za16(0, m_index, pg_16_all, a + mi * lda + ki);
                    svld1_hor_za16(1, m_index, pg_16_all, a + mi * lda + ki + pack_ks);
                    svld1_hor_za16(0, m_index + 1, pg_16_all, a + mi * lda + ki + pack_ks * 2);
                    svld1_hor_za16(1, m_index + 1, pg_16_all, a + mi * lda + ki + pack_ks * 3);
                    if (mi + prefetch_dis < m_cur_end) {
                        __builtin_prefetch(a + (mi + prefetch_dis) * lda + ki, 0, 2);
                        __builtin_prefetch(a + (mi + prefetch_dis) * lda + ki + pack_ks, 0, 2);
                        __builtin_prefetch(a + (mi + prefetch_dis) * lda + ki + pack_ks * 2, 0, 2);
                        __builtin_prefetch(a + (mi + prefetch_dis) * lda + ki + pack_ks * 3, 0, 2);
                    }
                }

                for (int pki = 0; pki < (pack_ks / 2); pki++) {
                    svst1_ver_za32(0, pki, pg_32_all, pack_a + pack_pos);
                    pack_pos += pack_ks;
                }

                for (int pki = 0; pki < (pack_ks / 2); pki++) {
                    svst1_ver_za32(1, pki, pg_32_all, pack_a + pack_pos);
                    pack_pos += pack_ks;
                }

                for (int pki = 0; pki < (pack_ks / 2); pki++) {
                    svst1_ver_za32(2, pki, pg_32_all, pack_a + pack_pos);
                    pack_pos += pack_ks;
                }

                for (int pki = 0; pki < (pack_ks / 2); pki++) {
                    svst1_ver_za32(3, pki, pg_32_all, pack_a + pack_pos);
                    pack_pos += pack_ks;
                }
                ki += pack_k_step_4;
            } else {
                for (; mi < m_cur_end; mi++) {
                    const int m_index = (mi - m_cur_start) * 2;
                    svld1_hor_za16(0, m_index, pg_16_all, a + mi * lda + ki);
                    if (mi + prefetch_dis < m_cur_end) {
                        __builtin_prefetch(a + (mi + prefetch_dis) * lda + ki, 0, 2);
                    }
                }
                svbool_t pg_32_m = svwhilelt_b32(m_cur_start, m_cur_end);
                int k_num = std::min(pack_ks, k_end - ki);
                for (int pki = 0; pki < k_num / 2; pki++) {
                    svst1_ver_za32(0, pki, pg_32_m, pack_a + pack_pos);
                    pack_pos += (m_cur_end - m_cur_start) * 2;
                }
                ki += pack_ks;
            }
        }
        mi = m_cur_end;
    }
}

__arm_new("za") void bgemm_prepack_ab_16_mod_16_mod(int block_k, int m, int n, int ldc, int k, bfloat16_t *a,
    bfloat16_t *b, bfloat16_t *beta, bfloat16_t *c) __arm_streaming
{
    const int pack_step = 16;
    const int m_step = 32;
    const int n_step = 32;
    const int m_single_step = 16;
    const int n_single_step = 16;
    const int m_n_num = pack_step * 2;
    svbool_t pg_16_m_n = svwhilelt_b16(0, m_n_num);
    const int prefetch_dis = 8;
    svbool_t pg_c = svwhilelt_b32(0, pack_step);
    svbool_t pg_16_all = svptrue_b16();
    svbool_t pg_32_all = svptrue_b32();
    svbool_t pg_16_half = svwhilelt_b16(0, pack_step);
    svbfloat16_t zero_b = svdup_bf16(0);
    int ni = 0;
    while (ni < n) {
        int mi = 0;
        while (mi < m) {
            svzero_za();
            if (mi + m_step <= m && ni + n_step <= n) {
                for (int ki = 0; ki < (block_k / 2); ki++) {
                    svbfloat16_t n_data0 = svld1(pg_16_m_n, b + ni * k + m_n_num * ki);
                    svbfloat16_t n_data1 = svld1(pg_16_m_n, b + (ni + pack_step) * k + m_n_num * ki);
                    svbfloat16_t m_data0 = svld1(pg_16_m_n, a + mi * k + m_n_num * ki);
                    svbfloat16_t m_data1 = svld1(pg_16_m_n, a + (mi + pack_step) * k + m_n_num * ki);
                    if (ki + prefetch_dis < (k / 2)) {
                        __builtin_prefetch(a + mi * k + m_n_num * (ki + prefetch_dis), 0, 0);
                        __builtin_prefetch(a + (mi + pack_step) * k + m_n_num * (ki + prefetch_dis), 0, 0);
                        __builtin_prefetch(b + ni * k + m_n_num * (ki + prefetch_dis), 0, 0);
                        __builtin_prefetch(b + (ni + pack_step) * k + m_n_num * (ki + prefetch_dis), 0, 0);
                    }
                    svmopa_za32_bf16_m(0, pg_16_m_n, pg_16_m_n, m_data0, n_data0);
                    svmopa_za32_bf16_m(1, pg_16_m_n, pg_16_m_n, m_data0, n_data1);
                    svmopa_za32_bf16_m(2, pg_16_m_n, pg_16_m_n, m_data1, n_data0);
                    svmopa_za32_bf16_m(3, pg_16_m_n, pg_16_m_n, m_data1, n_data1);
                }

                if ((*beta) == 0) {
                    for (int i = 0; i < pack_step; i++) {
                        svfloat32_t out_v0_0 = svread_hor_za32_f32_m(svfloat32_t(), pg_32_all, 0, i);
                        svfloat32_t out_v1_0 = svread_hor_za32_f32_m(svfloat32_t(), pg_32_all, 1, i);
                        svfloat32_t out_v2_0 = svread_hor_za32_f32_m(svfloat32_t(), pg_32_all, 2, i);
                        svfloat32_t out_v3_0 = svread_hor_za32_f32_m(svfloat32_t(), pg_32_all, 3, i);
                        svst1(pg_16_all, c + (mi + i) * ldc + ni,
                            svuzp1(svcvt_bf16_x(pg_c, out_v0_0), svcvt_bf16_x(pg_c, out_v1_0)));
                        svst1(pg_16_all, c + (mi + i + pack_step) * ldc + ni,
                            svuzp1(svcvt_bf16_x(pg_c, out_v2_0), svcvt_bf16_x(pg_c, out_v3_0)));
                    }
                } else {
                    for (int i = 0; i < pack_step; i++) {
                        svfloat32_t out_v0_0 = svread_hor_za32_f32_m(svfloat32_t(), pg_32_all, 0, i);
                        svfloat32_t out_v1_0 = svread_hor_za32_f32_m(svfloat32_t(), pg_32_all, 1, i);
                        svfloat32_t out_v2_0 = svread_hor_za32_f32_m(svfloat32_t(), pg_32_all, 2, i);
                        svfloat32_t out_v3_0 = svread_hor_za32_f32_m(svfloat32_t(), pg_32_all, 3, i);
                        svbfloat16_t c_v0 = svld1(pg_16_all, c + (mi + i) * ldc + ni);
                        auto c_v00 = svmul_x(pg_32_all, svreinterpret_f32(svzip1(zero_b, c_v0)), (float)(*beta));
                        auto c_v01 = svmul_x(pg_32_all, svreinterpret_f32(svzip2(zero_b, c_v0)), (float)(*beta));
                        c_v00 = svadd_x(pg_32_all, c_v00, out_v0_0);
                        c_v01 = svadd_x(pg_32_all, c_v01, out_v1_0);
                        svst1(pg_16_all, c + (mi + i) * ldc + ni,
                            svuzp1(svcvt_bf16_x(pg_32_all, c_v00), svcvt_bf16_x(pg_32_all, c_v01)));

                        svbfloat16_t c_v1 = svld1(pg_16_all, c + (mi + i + pack_step) * ldc + ni);
                        auto c_v10 = svmul_x(pg_32_all, svreinterpret_f32(svzip1(zero_b, c_v1)), (float)(*beta));
                        auto c_v11 = svmul_x(pg_32_all, svreinterpret_f32(svzip2(zero_b, c_v1)), (float)(*beta));
                        c_v10 = svadd_x(pg_32_all, c_v10, out_v2_0);
                        c_v11 = svadd_x(pg_32_all, c_v11, out_v3_0);
                        svst1(pg_16_all, c + (mi + i + pack_step) * ldc + ni,
                            svuzp1(svcvt_bf16_x(pg_32_all, c_v10), svcvt_bf16_x(pg_32_all, c_v11)));
                    }
                }
                mi += m_step;
            } else if (mi + m_step <= m && ni + n_single_step <= n) {
                for (int ki = 0; ki < (block_k / 2); ki++) {
                    svbfloat16_t n_data0 = svld1(pg_16_m_n, b + ni * k + m_n_num * ki);
                    svbfloat16_t m_data0 = svld1(pg_16_m_n, a + mi * k + m_n_num * ki);
                    svbfloat16_t m_data1 = svld1(pg_16_m_n, a + (mi + pack_step) * k + m_n_num * ki);
                    if (ki + prefetch_dis < (k / 2)) {
                        __builtin_prefetch(a + mi * k + m_n_num * (ki + prefetch_dis), 0, 0);
                        __builtin_prefetch(a + (mi + pack_step) * k + m_n_num * (ki + prefetch_dis), 0, 0);
                        __builtin_prefetch(b + ni * k + m_n_num * (ki + prefetch_dis), 0, 0);
                    }
                    svmopa_za32_bf16_m(0, pg_16_m_n, pg_16_m_n, m_data0, n_data0);
                    svmopa_za32_bf16_m(2, pg_16_m_n, pg_16_m_n, m_data1, n_data0);
                }

                if ((*beta) == 0) {
                    for (int i = 0; i < pack_step; i++) {
                        svfloat32_t out_v0_0 = svread_hor_za32_f32_m(svfloat32_t(), pg_32_all, 0, i);
                        svfloat32_t out_v2_0 = svread_hor_za32_f32_m(svfloat32_t(), pg_32_all, 2, i);
                        svst1(pg_16_half, c + (mi + i) * ldc + ni, svuzp1(svcvt_bf16_x(pg_c, out_v0_0), zero_b));
                        svst1(pg_16_half, c + (mi + i + pack_step) * ldc + ni,
                            svuzp1(svcvt_bf16_x(pg_c, out_v2_0), zero_b));
                    }
                } else {
                    for (int i = 0; i < pack_step; i++) {
                        svfloat32_t out_v0_0 = svread_hor_za32_f32_m(svfloat32_t(), pg_32_all, 0, i);
                        svfloat32_t out_v2_0 = svread_hor_za32_f32_m(svfloat32_t(), pg_32_all, 2, i);
                        svbfloat16_t c_v0 = svld1(pg_16_all, c + (mi + i) * ldc + ni);
                        auto c_v00 = svmul_x(pg_32_all, svreinterpret_f32(svzip1(zero_b, c_v0)), (float)(*beta));
                        c_v00 = svadd_x(pg_32_all, c_v00, out_v0_0);
                        svst1(pg_16_half, c + (mi + i) * ldc + ni, svuzp1(svcvt_bf16_x(pg_32_all, c_v00), zero_b));

                        svbfloat16_t c_v1 = svld1(pg_16_all, c + (mi + i + pack_step) * ldc + ni);
                        auto c_v10 = svmul_x(pg_32_all, svreinterpret_f32(svzip1(zero_b, c_v1)), (float)(*beta));
                        c_v10 = svadd_x(pg_32_all, c_v10, out_v2_0);
                        svst1(pg_16_half, c + (mi + i + pack_step) * ldc + ni,
                            svuzp1(svcvt_bf16_x(pg_32_all, c_v10), zero_b));
                    }
                }
                mi += m_step;
            } else if (mi + m_single_step <= m && ni + n_step <= n) {
                for (int ki = 0; ki < (block_k / 2); ki++) {
                    svbfloat16_t n_data0 = svld1(pg_16_m_n, b + ni * k + m_n_num * ki);
                    svbfloat16_t n_data1 = svld1(pg_16_m_n, b + (ni + pack_step) * k + m_n_num * ki);
                    svbfloat16_t m_data0 = svld1(pg_16_m_n, a + mi * k + m_n_num * ki);
                    if (ki + prefetch_dis < (k / 2)) {
                        __builtin_prefetch(a + mi * k + m_n_num * (ki + prefetch_dis), 0, 0);
                        __builtin_prefetch(b + ni * k + m_n_num * (ki + prefetch_dis), 0, 0);
                        __builtin_prefetch(b + (ni + pack_step) * k + m_n_num * (ki + prefetch_dis), 0, 0);
                    }
                    svmopa_za32_bf16_m(0, pg_16_m_n, pg_16_m_n, m_data0, n_data0);
                    svmopa_za32_bf16_m(1, pg_16_m_n, pg_16_m_n, m_data0, n_data1);
                }

                if ((*beta) == 0) {
                    for (int i = 0; i < pack_step; i++) {
                        svfloat32_t out_v0_0 = svread_hor_za32_f32_m(svfloat32_t(), pg_32_all, 0, i);
                        svfloat32_t out_v1_0 = svread_hor_za32_f32_m(svfloat32_t(), pg_32_all, 1, i);
                        svst1(pg_16_all, c + (mi + i) * ldc + ni,
                            svuzp1(svcvt_bf16_x(pg_c, out_v0_0), svcvt_bf16_x(pg_c, out_v1_0)));
                    }
                } else {
                    for (int i = 0; i < pack_step; i++) {
                        svfloat32_t out_v0_0 = svread_hor_za32_f32_m(svfloat32_t(), pg_32_all, 0, i);
                        svfloat32_t out_v1_0 = svread_hor_za32_f32_m(svfloat32_t(), pg_32_all, 1, i);
                        svbfloat16_t c_v0 = svld1(pg_16_all, c + (mi + i) * ldc + ni);
                        auto c_v00 = svmul_x(pg_32_all, svreinterpret_f32(svzip1(zero_b, c_v0)), (float)(*beta));
                        auto c_v01 = svmul_x(pg_32_all, svreinterpret_f32(svzip2(zero_b, c_v0)), (float)(*beta));
                        c_v00 = svadd_x(pg_32_all, c_v00, out_v0_0);
                        c_v01 = svadd_x(pg_32_all, c_v01, out_v1_0);
                        svst1(pg_16_all, c + (mi + i) * ldc + ni,
                            svuzp1(svcvt_bf16_x(pg_32_all, c_v00), svcvt_bf16_x(pg_32_all, c_v01)));
                    }
                }
                mi += m_single_step;
            } else if (mi + m_single_step <= m && ni + n_single_step <= n) {
                for (int ki = 0; ki < (block_k / 2); ki++) {
                    svbfloat16_t n_data0 = svld1(pg_16_m_n, b + ni * k + m_n_num * ki);
                    svbfloat16_t m_data0 = svld1(pg_16_m_n, a + mi * k + m_n_num * ki);
                    if (ki + prefetch_dis < (k / 2)) {
                        __builtin_prefetch(a + mi * k + m_n_num * (ki + prefetch_dis), 0, 0);
                        __builtin_prefetch(b + ni * k + m_n_num * (ki + prefetch_dis), 0, 0);
                    }
                    svmopa_za32_bf16_m(0, pg_16_m_n, pg_16_m_n, m_data0, n_data0);
                }

                if ((*beta) == 0) {
                    for (int i = 0; i < pack_step; i++) {
                        svfloat32_t out_v0_0 = svread_hor_za32_f32_m(svfloat32_t(), pg_32_all, 0, i);
                        svst1(pg_16_half, c + (mi + i) * ldc + ni, svuzp1(svcvt_bf16_x(pg_c, out_v0_0), zero_b));
                    }
                } else {
                    for (int i = 0; i < pack_step; i++) {
                        svfloat32_t out_v0_0 = svread_hor_za32_f32_m(svfloat32_t(), pg_32_all, 0, i);
                        svbfloat16_t c_v0 = svld1(pg_16_all, c + (mi + i) * ldc + ni);
                        auto c_v00 = svmul_x(pg_32_all, svreinterpret_f32(svzip1(zero_b, c_v0)), (float)(*beta));
                        c_v00 = svadd_x(pg_32_all, c_v00, out_v0_0);
                        svst1(pg_16_half, c + (mi + i) * ldc + ni, svuzp1(svcvt_bf16_x(pg_32_all, c_v00), zero_b));
                    }
                }
                mi += m_single_step;
            } else {
                assert(-1);
                return;
            }
        }
        if (ni + n_step <= n) {
            ni += n_step;
        } else if (ni + n_single_step <= n) {
            ni += n_single_step;
        } else {
            assert(-1);
            return;
        }
    }
}

inline void bgemm_macro_kernel(int bm, int em, int bn, int en, int k, int k_offset, int k_num, int ldc, bfloat16_t *a,
    bfloat16_t *b, bfloat16_t *c) __arm_inout("za") __arm_streaming
{
    for (int mi = bm; mi < em;) {
        if (mi + 64 <= em) {
            int ni = bn;
            for (; ni < en; ni += 16) {
                int n_num = std::min(32, (en - ni) * 2);
                svzero_za();
                bgemm_4VL_1VL(a + mi * k + k_offset * 16, b + ni * k + k_offset * n_num / 2, k, k_num, n_num);
                bgemm41_save(ldc, c + mi * ldc + ni, n_num);
            }
            mi += 64;
        } else {
            int m_num = std::min(32, (em - mi) * 2), ni = bn;
            for (; ni < en; ni += 16) {
                int n_num = std::min(32, (en - ni) * 2);
                svzero_za();
                bgemm_1VL_1VL(a + mi * k + k_offset * m_num / 2, b + ni * k + k_offset * n_num / 2, k, k_num, m_num,
                    n_num);
                bgemm11_save(ldc, c + mi * ldc + ni, m_num, n_num);
            }
            mi += 16;
        }
    }
}

inline void preload_L1(int bm, int em, int k_num, int k, bfloat16_t *a)
{
    for (int mi = bm; mi + 32 <= em; mi += 32)
        for (int ki = 0; ki < k_num / 2; ki++) {
            svprfh(svptrue_b16(), a + mi * k + ki * 32, SV_PLDL1KEEP);
            svprfh(svptrue_b16(), a + (mi + 16) * k + ki * 32, SV_PLDL1KEEP);
        }
}

inline void preload_L2(int bm, int em, int k_num, int k, bfloat16_t *a)
{
    for (int mi = bm; mi + 32 <= em; mi += 32)
        for (int ki = 0; ki < k_num / 2; ki++) {
            svprfh(svptrue_b16(), a + mi * k + ki * 32, SV_PLDL2KEEP);
            svprfh(svptrue_b16(), a + (mi + 16) * k + ki * 32, SV_PLDL2KEEP);
        }
}

__arm_new("za") void bgemm_prepack_ab_alpha1_beta0(int m, int n, int ldc, int k, bfloat16_t *a, bfloat16_t *b,
    bfloat16_t *c, int macro_kernel_m, int macro_kernel_n, int macro_kernel_k) __arm_streaming
{
    if (macro_kernel_k < k)
        macro_kernel_k = k;
    for (int bm = 0; bm < m; bm += macro_kernel_m) {
        int em = std::min(bm + macro_kernel_m, m);
        for (int k_offset = 0; k_offset < k; k_offset += macro_kernel_k) {
            int k_num = std::min(macro_kernel_k, k - k_offset);
            for (int bn = 0; bn < n; bn += macro_kernel_n) {
                int en = std::min(bn + macro_kernel_n, n);
                bgemm_macro_kernel(bm, em, bn, en, k, k_offset, k_num, ldc, a, b, c);
            }
        }
    }
}

__arm_new("za") void bgemm_pack(int m, int n, bfloat16_t *src, int ldc, bfloat16_t *dst) __arm_streaming
{
    for (int mi = 0, dst_off = 0; mi < m; mi += 16) {
        int m_num = std::min(16, m - mi) * 2;
        svbool_t pg_m = svwhilelt_b32(mi, m);
        for (int ni = 0; ni < n; ni += 32) {
            svzero_za();
            svbool_t pg_n = svwhilelt_b32(0, (n - ni) / 2);
            for (int i = 0; i < 16 && mi + i < m; i++)
                svld1_hor_za32(0, i, pg_n, src + (mi + i) * ldc + ni);
            for (int i = 0; i < 16 && ni + i * 2 < n; i++, dst_off += m_num)
                svst1_ver_za32(0, i, pg_m, dst + dst_off);
        }
    }
}

} // namespace kutacc

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
#include <cmath>
#include <algorithm>
#include <arm_sve.h>
#include <arm_sme.h>

#include "kernel.h"
#include "kernel_common.h"

namespace kutacc {

inline void igemmbdq_macro_kernel(int bm, int em, int bn, int en, int k, int k_offset, int k_num, int ldc, int8_t *a,
    int8_t *b, bfloat16_t *c, const float *rscales, const float *cscales) __arm_inout("za") __arm_streaming
{
    int prefetch_dis = (k == 128 ? 128 : (k == 4096 ? 48 : 16));
    if (k >= 1024) {
        for (int ni = bn; ni < en;) {
            if (ni + 32 <= en) {
                int mi = bm;
                for (; mi + 32 <= em; mi += 32) {
                    svzero_za();
                    igemm_2VL_2VL(a + mi * k + k_offset * 16, b + ni * k + k_offset * 16, k, k_num, prefetch_dis);
                    igemm22_save_bdq(rscales + ni, cscales + mi, ldc, c + mi * ldc + ni);
                }
                for (; mi < em; mi += 16) {
                    int m_num = std::min(64, (em - mi) * 4);
                    svzero_za();
                    igemm_2VL_1VL(a + mi * k + k_offset * m_num / 4, b + ni * k + k_offset * 16, k, k_num, m_num);
                    igemm21_save_bdq(rscales + ni, cscales + mi, ldc, c + mi * ldc + ni, m_num);
                }
                ni += 32;
            } else {
                int n_num = std::min(64, (en - ni) * 4), mi = bm;
                for (; mi + 64 <= em; mi += 64) {
                    svzero_za();
                    igemm_1VL_4VL(a + mi * k + k_offset * 16, b + ni * k + k_offset * n_num / 4, k, k_num, n_num);
                    igemm14_save_bdq(rscales + ni, cscales + mi, ldc, c + mi * ldc + ni, n_num);
                }
                for (; mi < em; mi += 16) {
                    int m_num = std::min(64, (em - mi) * 4);
                    svzero_za();
                    igemm_1VL_1VL(a + mi * k + k_offset * m_num / 4, b + ni * k + k_offset * n_num / 4, k_num, m_num,
                        n_num);
                    igemm11_save_bdq(rscales + ni, cscales + mi, ldc, c + mi * ldc + ni, m_num, n_num);
                }
                ni += 16;
            }
        }
    } else {
        for (int mi = bm; mi < em;) {
            if (mi + 32 <= em) {
                int ni = bn;
                for (; ni + 32 <= en; ni += 32) {
                    svzero_za();
                    igemm_2VL_2VL(a + mi * k + k_offset * 16, b + ni * k + k_offset * 16, k, k_num, prefetch_dis);
                    igemm22_save_bdq(rscales + ni, cscales + mi, ldc, c + mi * ldc + ni);
                }
                for (; ni < en; ni += 16) {
                    int n_num = std::min(64, (en - ni) * 4);
                    svzero_za();
                    igemm_1VL_1VL(a + mi * k + k_offset * 16, b + ni * k + k_offset * n_num / 4, k_num, 64, n_num);
                    igemm11_save_bdq(rscales + ni, cscales + mi, ldc, c + mi * ldc + ni, 64, n_num);
                    svzero_za();
                    igemm_1VL_1VL(a + (mi + 16) * k + k_offset * 16, b + ni * k + k_offset * n_num / 4, k_num, 64,
                        n_num);
                    igemm11_save_bdq(rscales + ni, cscales + (mi + 16), ldc, c + (mi + 16) * ldc + ni, 64, n_num);
                }
                mi += 32;
            } else {
                int m_num = std::min(64, (em - mi) * 4), ni = bn;
                for (; ni + 32 <= en; ni += 32) {
                    svzero_za();
                    igemm_2VL_1VL(a + mi * k + k_offset * m_num / 4, b + ni * k + k_offset * 16, k, k_num, m_num);
                    igemm21_save_bdq(rscales + ni, cscales + mi, ldc, c + mi * ldc + ni, m_num);
                }
                for (; ni < en; ni += 16) {
                    int n_num = std::min(64, (en - ni) * 4);
                    svzero_za();
                    igemm_1VL_1VL(a + mi * k + k_offset * m_num / 4, b + ni * k + k_offset * n_num / 4, k_num, m_num,
                        n_num);
                    igemm11_save_bdq(rscales + ni, cscales + mi, ldc, c + mi * ldc + ni, m_num, n_num);
                }
                mi += 16;
            }
        }
    }
}

inline void igemm_macro_kernel(int bm, int em, int bn, int en, int k, int k_offset, int k_num, int ldc, int8_t *a,
    int8_t *b, int32_t *c) __arm_inout("za") __arm_streaming
{
    for (int ni = bn; ni < en;) {
        if (ni + 32 <= en) {
            int mi = bm;
            for (; mi + 32 <= em; mi += 32) {
                if (k_offset == 0) {
                    svzero_za();
                } else {
                    igemm22_load(ldc, c + mi * ldc + ni);
                }
                igemm_2VL_2VL(a + mi * k + k_offset * 16, b + ni * k + k_offset * 16, k, k_num);
                igemm22_save(ldc, c + mi * ldc + ni);
            }
            for (; mi < em; mi += 16) {
                int m_num = std::min(64, (em - mi) * 4);
                if (k_offset == 0) {
                    svzero_za();
                } else {
                    igemm21_load(ldc, c + mi * ldc + ni, m_num);
                }
                igemm_2VL_1VL(a + mi * k + k_offset * m_num / 4, b + ni * k + k_offset * 16, k, k_num, m_num);
                igemm21_save(ldc, c + mi * ldc + ni, m_num);
            }
            ni += 32;
        } else {
            int n_num = std::min(64, (en - ni) * 4), mi = bm;
            for (; mi + 64 <= em; mi += 64) {
                if (k_offset == 0) {
                    svzero_za();
                } else {
                    igemm14_load(ldc, c + mi * ldc + ni, n_num);
                }
                igemm_1VL_4VL(a + mi * k + k_offset * 16, b + ni * k + k_offset * n_num / 4, k, k_num, n_num);
                igemm14_save(ldc, c + mi * ldc + ni, n_num);
            }
            for (; mi < em; mi += 16) {
                int m_num = std::min(64, (em - mi) * 4);
                if (k_offset == 0) {
                    svzero_za();
                } else {
                    igemm11_load(ldc, c + mi * ldc + ni, m_num, n_num);
                }
                igemm_1VL_1VL(a + mi * k + k_offset * m_num / 4, b + ni * k + k_offset * n_num / 4, k_num, m_num,
                    n_num);
                igemm11_save(ldc, c + mi * ldc + ni, m_num, n_num);
            }
            ni += 16;
        }
    }
}

inline void preload_L2(int bm, int em, int k_num, int k, int8_t *a)
{
    for (int mi = bm; mi + 32 <= em; mi += 32)
        for (int ki = 0; ki < k_num / 4; ki++) {
            svprfb(svptrue_b8(), a + mi * k + ki * 64, SV_PLDL2KEEP);
            svprfb(svptrue_b8(), a + (mi + 16) * k + ki * 64, SV_PLDL2KEEP);
        }
}

__arm_new("za") void igemmbdq_prepack_ab_alpha1_beta0(int m, int n, int ldc, int k, int8_t *a, int8_t *b, bfloat16_t *c,
    const float *rscales, const float *cscales, int macro_kernel_m, int macro_kernel_n,
    int macro_kernel_k) __arm_streaming
{
    if (macro_kernel_k < k)
        macro_kernel_k = k;
    for (int bm = 0; bm < m; bm += macro_kernel_m) {
        int em = std::min(bm + macro_kernel_m, m);
        for (int k_offset = 0; k_offset < k; k_offset += macro_kernel_k) {
            int k_num = std::min(macro_kernel_k, k - k_offset);
            preload_L2(bm, em, k_num, k, a + k_offset * 16);
            for (int bn = 0; bn < n; bn += macro_kernel_n) {
                int en = std::min(bn + macro_kernel_n, n);
                igemmbdq_macro_kernel(bm, em, bn, en, k, k_offset, k_num, ldc, a, b, c, rscales, cscales);
            }
        }
    }
}

__arm_new("za") void igemm_prepack_ab_alpha1_beta0(int m, int n, int ldc, int k, int8_t *a, int8_t *b, int *c,
    int macro_kernel_m, int macro_kernel_n, int macro_kernel_k) __arm_streaming
{
    if (k >= 4096)
        macro_kernel_n = 512, macro_kernel_k = 1024;
    else if (k <= 512)
        macro_kernel_n = 512;
    for (int bm = 0; bm < m; bm += macro_kernel_m) {
        int em = std::min(bm + macro_kernel_m, m);
        for (int k_offset = 0; k_offset < k; k_offset += macro_kernel_k) {
            int k_num = std::min(macro_kernel_k, k - k_offset);
            preload_L2(bm, em, k_num, k, a + k_offset * 16);
            for (int bn = 0; bn < n; bn += macro_kernel_n) {
                int en = std::min(bn + macro_kernel_n, n);
                igemm_macro_kernel(bm, em, bn, en, k, k_offset, k_num, ldc, a, b, c);
            }
        }
    }
}

__arm_new("za") void igemm_pack(int m, int n, int8_t *src, int ldc, int8_t *dst) __arm_streaming
{
    for (int mi = 0, dst_off = 0; mi < m; mi += 16) {
        int m_num = std::min(16, m - mi) * 4;
        svbool_t pg_m = svwhilelt_b32(mi, m);
        for (int ni = 0; ni < n; ni += 64) {
            svzero_za();
            svbool_t pg_n = svwhilelt_b32(0, (n - ni) / 4);
            for (int i = 0; i < 16 && mi + i < m; i++)
                svld1_hor_za32(0, i, pg_n, src + (mi + i) * ldc + ni);
            for (int i = 0; i < 16 && ni + i * 4 < n; i++, dst_off += m_num)
                svst1_ver_za32(0, i, pg_m, dst + dst_off);
        }
    }
}

__arm_new(
    "za") void igemm_pack_with_idx(int m, int n, int8_t *src, int ldc, int m_off, int *idx, int8_t *dst) __arm_streaming
{
    for (int mi = 0, dst_off = 0; mi < m; mi += 16) {
        int m_num = std::min(16, m - mi) * 4;
        svbool_t pg_m = svwhilelt_b32(mi, m);
        for (int ni = 0; ni < n; ni += 64) {
            svzero_za();
            svbool_t pg_n = svwhilelt_b32(0, (n - ni) / 4);
            for (int i = 0; i < 16 && mi + i < m; i++)
                svld1_hor_za32(0, i, pg_n, src + idx[m_off + mi + i] * ldc + ni);
            for (int i = 0; i < 16 && ni + i * 4 < n; i++, dst_off += m_num)
                svst1_ver_za32(0, i, pg_m, dst + dst_off);
        }
    }
}

} // namespace kutacc

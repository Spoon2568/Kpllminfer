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
#include <arm_sme.h>
#include <arm_bf16.h>

namespace kutacc {

__attribute__((always_inline)) inline void igemm_2VL_2VL(const int8_t *a0, const int8_t *b0, int k, int k_num,
    int prefetch_dis = 32) __arm_inout("za") __arm_streaming
{
    svbool_t pg_8_m_n = svwhilelt_b8(0, 64);
    const int8_t *a1 = a0 + k * 16, *b1 = b0 + k * 16;
    int ki = 0;
    for (; ki + 4 <= (k_num / 4); ki += 4) {
        svprfb(pg_8_m_n, a0 + (ki + prefetch_dis) * 64, SV_PLDL1STRM);
        svprfb(pg_8_m_n, a1 + (ki + prefetch_dis) * 64, SV_PLDL1STRM);
        svprfb(pg_8_m_n, b0 + (ki + prefetch_dis) * 64, SV_PLDL1STRM);
        svprfb(pg_8_m_n, b1 + (ki + prefetch_dis) * 64, SV_PLDL1STRM);
        svint8_t m_data0 = svld1(pg_8_m_n, a0 + ki * 64);
        svint8_t n_data0 = svld1(pg_8_m_n, b0 + ki * 64);
        svmopa_za32_s8_m(0, pg_8_m_n, pg_8_m_n, m_data0, n_data0);
        svint8_t m_data1 = svld1(pg_8_m_n, a1 + ki * 64);
        svmopa_za32_s8_m(2, pg_8_m_n, pg_8_m_n, m_data1, n_data0);
        svint8_t n_data1 = svld1(pg_8_m_n, b1 + ki * 64);
        svmopa_za32_s8_m(1, pg_8_m_n, pg_8_m_n, m_data0, n_data1);
        svmopa_za32_s8_m(3, pg_8_m_n, pg_8_m_n, m_data1, n_data1);
        svprfb(pg_8_m_n, a0 + (ki + 1 + prefetch_dis) * 64, SV_PLDL1STRM);
        svprfb(pg_8_m_n, a1 + (ki + 1 + prefetch_dis) * 64, SV_PLDL1STRM);
        svprfb(pg_8_m_n, b0 + (ki + 1 + prefetch_dis) * 64, SV_PLDL1STRM);
        svprfb(pg_8_m_n, b1 + (ki + 1 + prefetch_dis) * 64, SV_PLDL1STRM);
        svint8_t m_data10 = svld1(pg_8_m_n, a0 + (ki + 1) * 64);
        svint8_t n_data10 = svld1(pg_8_m_n, b0 + (ki + 1) * 64);
        svmopa_za32_s8_m(0, pg_8_m_n, pg_8_m_n, m_data10, n_data10);
        svint8_t m_data11 = svld1(pg_8_m_n, a1 + (ki + 1) * 64);
        svmopa_za32_s8_m(2, pg_8_m_n, pg_8_m_n, m_data11, n_data10);
        svint8_t n_data11 = svld1(pg_8_m_n, b1 + (ki + 1) * 64);
        svmopa_za32_s8_m(1, pg_8_m_n, pg_8_m_n, m_data10, n_data11);
        svmopa_za32_s8_m(3, pg_8_m_n, pg_8_m_n, m_data11, n_data11);
        svprfb(pg_8_m_n, a0 + (ki + 2 + prefetch_dis) * 64, SV_PLDL1STRM);
        svprfb(pg_8_m_n, a1 + (ki + 2 + prefetch_dis) * 64, SV_PLDL1STRM);
        svprfb(pg_8_m_n, b0 + (ki + 2 + prefetch_dis) * 64, SV_PLDL1STRM);
        svprfb(pg_8_m_n, b1 + (ki + 2 + prefetch_dis) * 64, SV_PLDL1STRM);
        svint8_t m_data20 = svld1(pg_8_m_n, a0 + (ki + 2) * 64);
        svint8_t n_data20 = svld1(pg_8_m_n, b0 + (ki + 2) * 64);
        svmopa_za32_s8_m(0, pg_8_m_n, pg_8_m_n, m_data20, n_data20);
        svint8_t m_data21 = svld1(pg_8_m_n, a1 + (ki + 2) * 64);
        svmopa_za32_s8_m(2, pg_8_m_n, pg_8_m_n, m_data21, n_data20);
        svint8_t n_data21 = svld1(pg_8_m_n, b1 + (ki + 2) * 64);
        svmopa_za32_s8_m(1, pg_8_m_n, pg_8_m_n, m_data20, n_data21);
        svmopa_za32_s8_m(3, pg_8_m_n, pg_8_m_n, m_data21, n_data21);
        svprfb(pg_8_m_n, a0 + (ki + 3 + prefetch_dis) * 64, SV_PLDL1STRM);
        svprfb(pg_8_m_n, a1 + (ki + 3 + prefetch_dis) * 64, SV_PLDL1STRM);
        svprfb(pg_8_m_n, b0 + (ki + 3 + prefetch_dis) * 64, SV_PLDL1STRM);
        svprfb(pg_8_m_n, b1 + (ki + 3 + prefetch_dis) * 64, SV_PLDL1STRM);
        svint8_t m_data30 = svld1(pg_8_m_n, a0 + (ki + 3) * 64);
        svint8_t n_data30 = svld1(pg_8_m_n, b0 + (ki + 3) * 64);
        svmopa_za32_s8_m(0, pg_8_m_n, pg_8_m_n, m_data30, n_data30);
        svint8_t m_data31 = svld1(pg_8_m_n, a1 + (ki + 3) * 64);
        svmopa_za32_s8_m(2, pg_8_m_n, pg_8_m_n, m_data31, n_data30);
        svint8_t n_data31 = svld1(pg_8_m_n, b1 + (ki + 3) * 64);
        svmopa_za32_s8_m(1, pg_8_m_n, pg_8_m_n, m_data30, n_data31);
        svmopa_za32_s8_m(3, pg_8_m_n, pg_8_m_n, m_data31, n_data31);
    }
    for (; ki < (k_num / 4); ki++) {
        svint8_t m_data0 = svld1(pg_8_m_n, a0 + ki * 64);
        svint8_t n_data0 = svld1(pg_8_m_n, b0 + ki * 64);
        svmopa_za32_s8_m(0, pg_8_m_n, pg_8_m_n, m_data0, n_data0);
        svint8_t m_data1 = svld1(pg_8_m_n, a1 + ki * 64);
        svmopa_za32_s8_m(2, pg_8_m_n, pg_8_m_n, m_data1, n_data0);
        svint8_t n_data1 = svld1(pg_8_m_n, b1 + ki * 64);
        svmopa_za32_s8_m(1, pg_8_m_n, pg_8_m_n, m_data0, n_data1);
        svmopa_za32_s8_m(3, pg_8_m_n, pg_8_m_n, m_data1, n_data1);
    }
}

__attribute__((always_inline)) inline void igemm_2VL_2VL_experimental(const int8_t *a0, const int8_t *b0, int k,
    int k_num, int prefetch_dis = 32) __arm_inout("za") __arm_streaming
{
    svbool_t pg_8_m_n = svwhilelt_b8(0, 64);
    const int8_t *a1 = a0 + k * 16, *b1 = b0 + k * 16;
    int ki = 0;
    for (; ki < (k_num / 4); ki++) {
        svint8_t m_data0 = svld1(pg_8_m_n, a0 + ki * 64);
        svint8_t n_data0 = svld1(pg_8_m_n, b0 + ki * 64);
        svmopa_za32_s8_m(0, pg_8_m_n, pg_8_m_n, m_data0, n_data0);
        svint8_t m_data1 = svld1(pg_8_m_n, a1 + ki * 64);
        svmopa_za32_s8_m(2, pg_8_m_n, pg_8_m_n, m_data1, n_data0);
        svint8_t n_data1 = svld1(pg_8_m_n, b1 + ki * 64);
        svmopa_za32_s8_m(1, pg_8_m_n, pg_8_m_n, m_data0, n_data1);
        svmopa_za32_s8_m(3, pg_8_m_n, pg_8_m_n, m_data1, n_data1);
    }
}

const int prefetch_dis = 32;

__attribute__((always_inline)) inline void igemm_2VL_1VL(const int8_t *a, const int8_t *b0, int k, int k_num, int m_num)
    __arm_inout("za") __arm_streaming
{
    svbool_t pg_8_m = svwhilelt_b8(0, m_num);
    svbool_t pg_8_n = svwhilelt_b8(0, 64);
    const int8_t *b1 = b0 + k * 16;
    for (int ki = 0; ki < (k_num / 4); ki++) {
        svprfb(pg_8_n, b0 + (ki + prefetch_dis) * 64, SV_PLDL1STRM);
        svprfb(pg_8_n, b1 + (ki + prefetch_dis) * 64, SV_PLDL1STRM);
        svint8_t m_data = svld1(pg_8_m, a + ki * m_num);
        svint8_t n_data0 = svld1(pg_8_n, b0 + ki * 64);
        svint8_t n_data1 = svld1(pg_8_n, b1 + ki * 64);
        svmopa_za32_s8_m(0, pg_8_m, pg_8_n, m_data, n_data0);
        svmopa_za32_s8_m(1, pg_8_m, pg_8_n, m_data, n_data1);
    }
}

__attribute__((always_inline)) inline void igemm_1VL_1VL(const int8_t *a, const int8_t *b, int k_num, int m_num,
    int n_num) __arm_inout("za") __arm_streaming
{
    svbool_t pg_8_m = svwhilelt_b8(0, m_num);
    svbool_t pg_8_n = svwhilelt_b8(0, n_num);
    for (int ki = 0; ki < (k_num / 4); ki++) {
        svprfb(pg_8_n, b + (ki + prefetch_dis) * n_num, SV_PLDL1STRM);
        svint8_t m_data = svld1(pg_8_m, a + ki * m_num);
        svint8_t n_data = svld1(pg_8_n, b + ki * n_num);
        svmopa_za32_s8_m(0, pg_8_m, pg_8_n, m_data, n_data);
    }
}

__attribute__((always_inline)) inline void igemm_1VL_4VL(const int8_t *a0, const int8_t *b, int k, int k_num, int n_num)
    __arm_inout("za") __arm_streaming
{
    svbool_t pg_8_m = svwhilelt_b8(0, 64);
    svbool_t pg_8_n = svwhilelt_b8(0, n_num);
    const int8_t *a1 = a0 + k * 16, *a2 = a1 + k * 16, *a3 = a2 + k * 16;
    for (int ki = 0; ki < (k_num / 4); ki++) {
        svprfb(pg_8_n, b + (ki + prefetch_dis) * n_num, SV_PLDL1STRM);
        svint8_t m_data0 = svld1(pg_8_m, a0 + ki * 64);
        svint8_t m_data1 = svld1(pg_8_m, a1 + ki * 64);
        svint8_t m_data2 = svld1(pg_8_m, a2 + ki * 64);
        svint8_t m_data3 = svld1(pg_8_m, a3 + ki * 64);
        svint8_t n_data = svld1(pg_8_n, b + ki * n_num);
        svmopa_za32_s8_m(0, pg_8_m, pg_8_n, m_data0, n_data);
        svmopa_za32_s8_m(1, pg_8_m, pg_8_n, m_data1, n_data);
        svmopa_za32_s8_m(2, pg_8_m, pg_8_n, m_data2, n_data);
        svmopa_za32_s8_m(3, pg_8_m, pg_8_n, m_data3, n_data);
    }
}

__attribute__((always_inline)) inline void igemm22_save(int ldc, int *c) __arm_inout("za") __arm_streaming
{
    svint32_t out_v0 = svdup_s32(0);
    svint32_t out_v1 = svdup_s32(0);
    svint32_t out_v2 = svdup_s32(0);
    svint32_t out_v3 = svdup_s32(0);
    svbool_t pg32_all = svptrue_b32();
    for (int i = 0; i < 16; i++) {
        out_v0 = svread_hor_za32_s32_m(out_v0, pg32_all, 0, i);
        out_v1 = svread_hor_za32_s32_m(out_v1, pg32_all, 1, i);
        out_v2 = svread_hor_za32_s32_m(out_v2, pg32_all, 2, i);
        out_v3 = svread_hor_za32_s32_m(out_v3, pg32_all, 3, i);
        svst1(pg32_all, c + i * ldc, out_v0);
        svst1(pg32_all, c + i * ldc + 16, out_v1);
        svst1(pg32_all, c + (i + 16) * ldc, out_v2);
        svst1(pg32_all, c + (i + 16) * ldc + 16, out_v3);
    }
}

__attribute__((always_inline)) inline void igemm22_load(int ldc, int *c) __arm_inout("za") __arm_streaming
{
    svbool_t pg32_all = svptrue_b32();
    for (int i = 0; i < 16; i++) {
        svint32_t cv0 = svld1(pg32_all, c + i * ldc);
        svint32_t cv1 = svld1(pg32_all, c + i * ldc + 16);
        svint32_t cv2 = svld1(pg32_all, c + (i + 16) * ldc);
        svint32_t cv3 = svld1(pg32_all, c + (i + 16) * ldc + 16);
        svwrite_hor_za32_s32_m(0, i, pg32_all, cv0);
        svwrite_hor_za32_s32_m(1, i, pg32_all, cv1);
        svwrite_hor_za32_s32_m(2, i, pg32_all, cv2);
        svwrite_hor_za32_s32_m(3, i, pg32_all, cv3);
    }
}

__attribute__((always_inline)) inline void igemm21_save(int ldc, int *c, int m_num) __arm_inout("za") __arm_streaming
{
    svint32_t out_v0 = svdup_s32(0);
    svint32_t out_v1 = svdup_s32(0);
    svbool_t pg32_all = svptrue_b32();
    for (int i = 0; i < m_num / 4; i++) {
        out_v0 = svread_hor_za32_s32_m(out_v0, pg32_all, 0, i);
        out_v1 = svread_hor_za32_s32_m(out_v1, pg32_all, 1, i);
        svst1(pg32_all, c + i * ldc, out_v0);
        svst1(pg32_all, c + i * ldc + 16, out_v1);
    }
}

__attribute__((always_inline)) inline void igemm21_load(int ldc, int *c, int m_num) __arm_inout("za") __arm_streaming
{
    svbool_t pg32_all = svptrue_b32();
    for (int i = 0; i < m_num / 4; i++) {
        svint32_t cv0 = svld1(pg32_all, c + i * ldc);
        svint32_t cv1 = svld1(pg32_all, c + i * ldc + 16);
        svwrite_hor_za32_s32_m(0, i, pg32_all, cv0);
        svwrite_hor_za32_s32_m(1, i, pg32_all, cv1);
    }
}

__attribute__((always_inline)) inline void igemm11_save(int ldc, int *c, int m_num, int n_num)
    __arm_inout("za") __arm_streaming
{
    svint32_t out_v0 = svdup_s32(0);
    svbool_t pg_c = svwhilelt_b32(0, n_num / 4);
    svbool_t pg32_all = svptrue_b32();
    for (int i = 0; i < m_num / 4; i++) {
        out_v0 = svread_hor_za32_s32_m(out_v0, pg32_all, 0, i);
        svst1(pg_c, c + i * ldc, out_v0);
    }
}

__attribute__((always_inline)) inline void igemm11_load(int ldc, int *c, int m_num, int n_num)
    __arm_inout("za") __arm_streaming
{
    svbool_t pg_c = svwhilelt_b32(0, n_num / 4);
    svbool_t pg32_all = svptrue_b32();
    for (int i = 0; i < m_num / 4; i++) {
        svint32_t cv0 = svld1(pg_c, c + i * ldc);
        svwrite_hor_za32_s32_m(0, i, pg_c, cv0);
    }
}

__attribute__((always_inline)) inline void igemm14_save(int ldc, int *c, int n_num) __arm_inout("za") __arm_streaming
{
    svbool_t pg_c = svwhilelt_b32(0, n_num / 4);
    svbool_t pg32_all = svptrue_b32();
    svint32_t out_v0 = svdup_s32(0);
    svint32_t out_v1 = svdup_s32(0);
    svint32_t out_v2 = svdup_s32(0);
    svint32_t out_v3 = svdup_s32(0);
    for (int i = 0; i < 16; i++) {
        out_v0 = svread_hor_za32_s32_m(out_v0, pg32_all, 0, i);
        out_v1 = svread_hor_za32_s32_m(out_v1, pg32_all, 1, i);
        out_v2 = svread_hor_za32_s32_m(out_v2, pg32_all, 2, i);
        out_v3 = svread_hor_za32_s32_m(out_v3, pg32_all, 3, i);
        svst1(pg_c, c + (i + 0) * ldc, out_v0);
        svst1(pg_c, c + (i + 16) * ldc, out_v1);
        svst1(pg_c, c + (i + 32) * ldc, out_v2);
        svst1(pg_c, c + (i + 48) * ldc, out_v3);
    }
}

__attribute__((always_inline)) inline void igemm14_load(int ldc, int *c, int n_num) __arm_inout("za") __arm_streaming
{
    svbool_t pg_c = svwhilelt_b32(0, n_num / 4);
    svbool_t pg32_all = svptrue_b32();
    for (int i = 0; i < 16; i++) {
        svint32_t cv0 = svld1(pg_c, c + (i + 0) * ldc);
        svint32_t cv1 = svld1(pg_c, c + (i + 16) * ldc);
        svint32_t cv2 = svld1(pg_c, c + (i + 32) * ldc);
        svint32_t cv3 = svld1(pg_c, c + (i + 48) * ldc);
        svwrite_hor_za32_s32_m(0, i, pg_c, cv0);
        svwrite_hor_za32_s32_m(1, i, pg_c, cv1);
        svwrite_hor_za32_s32_m(2, i, pg_c, cv2);
        svwrite_hor_za32_s32_m(3, i, pg_c, cv3);
    }
}

__attribute__((always_inline)) inline void igemm22_save_bdq(const float *rscales, const float *cscales, int ldc,
    bfloat16_t *c) __arm_inout("za") __arm_streaming
{
    svbool_t pg16_all = svptrue_b16();
    svbool_t pg32_all = svptrue_b32();
    svint32_t zero = svdup_s32(0);
    svfloat32_t rscales_v0 = svld1(pg32_all, rscales);
    svfloat32_t rscales_v1 = svld1(pg32_all, rscales + 16);
    for (int i = 0; i < 16; i++) {
        svprfh(svptrue_b16(), c + i * ldc, SV_PSTL1STRM);
        svprfh(svptrue_b16(), c + (i + 16) * ldc, SV_PSTL1STRM);
        svint32_t out_v0 = svread_hor_za32_s32_m(zero, pg32_all, 0, i);
        svint32_t out_v1 = svread_hor_za32_s32_m(zero, pg32_all, 1, i);
        svint32_t out_v2 = svread_hor_za32_s32_m(zero, pg32_all, 2, i);
        svint32_t out_v3 = svread_hor_za32_s32_m(zero, pg32_all, 3, i);
        float cscales_v0 = *(cscales + i);
        float cscales_v1 = *(cscales + 16 + i);

        svfloat32_t out_v0_f = svcvt_f32_s32_x(pg32_all, out_v0);
        svfloat32_t out_v1_f = svcvt_f32_s32_x(pg32_all, out_v1);
        svfloat32_t out_v2_f = svcvt_f32_s32_x(pg32_all, out_v2);
        svfloat32_t out_v3_f = svcvt_f32_s32_x(pg32_all, out_v3);

        out_v0_f = svmul_x(pg32_all, out_v0_f, svmul_x(pg32_all, rscales_v0, cscales_v0));
        out_v1_f = svmul_x(pg32_all, out_v1_f, svmul_x(pg32_all, rscales_v1, cscales_v0));
        out_v2_f = svmul_x(pg32_all, out_v2_f, svmul_x(pg32_all, rscales_v0, cscales_v1));
        out_v3_f = svmul_x(pg32_all, out_v3_f, svmul_x(pg32_all, rscales_v1, cscales_v1));

        svst1(pg16_all, c + i * ldc, svuzp1(svcvt_bf16_x(pg32_all, out_v0_f), svcvt_bf16_x(pg32_all, out_v1_f)));
        svst1(pg16_all, c + (16 + i) * ldc, svuzp1(svcvt_bf16_x(pg32_all, out_v2_f), svcvt_bf16_x(pg32_all, out_v3_f)));
    }
}

__attribute__((always_inline)) inline void igemm21_save_bdq(const float *rscales, const float *cscales, int ldc,
    bfloat16_t *c, int m_num) __arm_inout("za") __arm_streaming
{
    svbool_t pg16_all = svptrue_b16();
    svbool_t pg32_all = svptrue_b32();
    for (int i = 0; i < m_num / 4; i++) {
        svint32_t out_v0 = svdup_s32(0);
        svint32_t out_v1 = svdup_s32(0);
        out_v0 = svread_hor_za32_s32_m(out_v0, pg32_all, 0, i);
        out_v1 = svread_hor_za32_s32_m(out_v1, pg32_all, 1, i);
        svfloat32_t rscales_v0 = svld1(pg32_all, rscales);
        svfloat32_t rscales_v1 = svld1(pg32_all, rscales + 16);
        float cscales_v0 = *(cscales + i);

        svfloat32_t out_v0_f = svcvt_f32_s32_x(pg32_all, out_v0);
        svfloat32_t out_v1_f = svcvt_f32_s32_x(pg32_all, out_v1);

        out_v0_f = svmul_x(pg32_all, out_v0_f, svmul_x(pg32_all, rscales_v0, cscales_v0));
        out_v1_f = svmul_x(pg32_all, out_v1_f, svmul_x(pg32_all, rscales_v1, cscales_v0));

        svst1(pg16_all, c + i * ldc, svuzp1(svcvt_bf16_x(pg32_all, out_v0_f), svcvt_bf16_x(pg32_all, out_v1_f)));
    }
}

__attribute__((always_inline)) inline void igemm14_save_bdq(const float *rscales, const float *cscales, int ldc,
    bfloat16_t *c, int n_num) __arm_inout("za") __arm_streaming
{
    svbool_t pg_c_16 = svwhilelt_b16(0, n_num / 4);
    svbool_t pg32_all = svptrue_b32();
    svbfloat16_t zero = svdup_bf16(0);
    for (int i = 0; i < 16; i++) {
        svint32_t out_v0 = svdup_s32(0);
        svint32_t out_v1 = svdup_s32(0);
        svint32_t out_v2 = svdup_s32(0);
        svint32_t out_v3 = svdup_s32(0);
        out_v0 = svread_hor_za32_s32_m(out_v0, pg32_all, 0, i);
        out_v1 = svread_hor_za32_s32_m(out_v1, pg32_all, 1, i);
        out_v2 = svread_hor_za32_s32_m(out_v2, pg32_all, 2, i);
        out_v3 = svread_hor_za32_s32_m(out_v3, pg32_all, 3, i);
        svfloat32_t rscales_v = svld1(pg32_all, rscales);
        float cscales_v0 = *(cscales + i), cscales_v1 = *(cscales + 16 + i);
        float cscales_v2 = *(cscales + 32 + i), cscales_v3 = *(cscales + 48 + i);
        svfloat32_t out_v0_f = svcvt_f32_s32_x(pg32_all, out_v0);
        svfloat32_t out_v1_f = svcvt_f32_s32_x(pg32_all, out_v1);
        svfloat32_t out_v2_f = svcvt_f32_s32_x(pg32_all, out_v2);
        svfloat32_t out_v3_f = svcvt_f32_s32_x(pg32_all, out_v3);
        out_v0_f = svmul_x(pg32_all, out_v0_f, svmul_x(pg32_all, rscales_v, cscales_v0));
        out_v1_f = svmul_x(pg32_all, out_v1_f, svmul_x(pg32_all, rscales_v, cscales_v1));
        out_v2_f = svmul_x(pg32_all, out_v2_f, svmul_x(pg32_all, rscales_v, cscales_v2));
        out_v3_f = svmul_x(pg32_all, out_v3_f, svmul_x(pg32_all, rscales_v, cscales_v3));
        svst1(pg_c_16, c + (i + 0) * ldc, svuzp1(svcvt_bf16_f32_x(pg32_all, out_v0_f), zero));
        svst1(pg_c_16, c + (i + 16) * ldc, svuzp1(svcvt_bf16_f32_x(pg32_all, out_v1_f), zero));
        svst1(pg_c_16, c + (i + 32) * ldc, svuzp1(svcvt_bf16_f32_x(pg32_all, out_v2_f), zero));
        svst1(pg_c_16, c + (i + 48) * ldc, svuzp1(svcvt_bf16_f32_x(pg32_all, out_v3_f), zero));
    }
}

__attribute__((always_inline)) inline void igemm11_save_bdq(const float *rscales, const float *cscales, int ldc,
    bfloat16_t *c, int m_num, int n_num) __arm_inout("za") __arm_streaming
{
    svbool_t pg_c_16 = svwhilelt_b16(0, n_num / 4);
    svbool_t pg32_all = svptrue_b32();
    svbfloat16_t zero = svdup_bf16(0);
    for (int i = 0; i < m_num / 4; i++) {
        svint32_t out_v0 = svdup_s32(0);
        out_v0 = svread_hor_za32_s32_m(out_v0, pg32_all, 0, i);
        svfloat32_t rscales_v = svld1(pg32_all, rscales);
        float cscales_v = *(cscales + i);
        svfloat32_t out_v0_f = svcvt_f32_s32_x(pg32_all, out_v0);
        out_v0_f = svmul_x(pg32_all, out_v0_f, svmul_x(pg32_all, rscales_v, cscales_v));
        svst1(pg_c_16, c + (i + 0) * ldc, svuzp1(svcvt_bf16_f32_x(pg32_all, out_v0_f), zero));
    }
}

__attribute__((always_inline)) inline void igemm22_load_save_bdq(const float *rscales, const float *cscales, int ldc,
    bfloat16_t *c) __arm_inout("za") __arm_streaming
{
    svbool_t pg16_all = svptrue_b16();
    svbool_t pg32_all = svptrue_b32();
    svbfloat16_t zero = svdup_bf16(0);
    svint32_t zero_i = svdup_s32(0);
    svfloat32_t rscales_v0 = svld1(pg32_all, rscales);
    svfloat32_t rscales_v1 = svld1(pg32_all, rscales + 16);
    for (int i = 0; i < 16; i++) {
        svprfh(svptrue_b16(), c + i * ldc, SV_PSTL1STRM);
        svprfh(svptrue_b16(), c + (i + 16) * ldc, SV_PSTL1STRM);
        svint32_t out_v0 = svread_hor_za32_s32_m(zero_i, pg32_all, 0, i);
        svint32_t out_v1 = svread_hor_za32_s32_m(zero_i, pg32_all, 1, i);
        svint32_t out_v2 = svread_hor_za32_s32_m(zero_i, pg32_all, 2, i);
        svint32_t out_v3 = svread_hor_za32_s32_m(zero_i, pg32_all, 3, i);
        float cscales_v0 = *(cscales + i), cscales_v1 = *(cscales + 16 + i);
        svbfloat16_t c0 = svld1(pg16_all, c + i * ldc);
        svbfloat16_t c1 = svld1(pg16_all, c + (i + 16) * ldc);
        svfloat32_t out_v0_f = svcvt_f32_s32_x(pg32_all, out_v0);
        svfloat32_t out_v1_f = svcvt_f32_s32_x(pg32_all, out_v1);
        svfloat32_t out_v2_f = svcvt_f32_s32_x(pg32_all, out_v2);
        svfloat32_t out_v3_f = svcvt_f32_s32_x(pg32_all, out_v3);
        out_v0_f =
            svmla_x(pg32_all, svreinterpret_f32(svzip1(zero, c0)), out_v0_f, svmul_x(pg32_all, rscales_v0, cscales_v0));
        out_v1_f =
            svmla_x(pg32_all, svreinterpret_f32(svzip2(zero, c0)), out_v1_f, svmul_x(pg32_all, rscales_v1, cscales_v0));
        out_v2_f =
            svmla_x(pg32_all, svreinterpret_f32(svzip1(zero, c1)), out_v2_f, svmul_x(pg32_all, rscales_v0, cscales_v1));
        out_v3_f =
            svmla_x(pg32_all, svreinterpret_f32(svzip2(zero, c1)), out_v3_f, svmul_x(pg32_all, rscales_v1, cscales_v1));
        svst1(pg16_all, c + i * ldc, svuzp1(svcvt_bf16_x(pg32_all, out_v0_f), svcvt_bf16_x(pg32_all, out_v1_f)));
        svst1(pg16_all, c + (16 + i) * ldc, svuzp1(svcvt_bf16_x(pg32_all, out_v2_f), svcvt_bf16_x(pg32_all, out_v3_f)));
    }
}

__attribute__((always_inline)) inline void igemm21_load_save_bdq(const float *rscales, const float *cscales, int ldc,
    bfloat16_t *c, int m_num) __arm_inout("za") __arm_streaming
{
    svbool_t pg16_all = svptrue_b16();
    svbool_t pg32_all = svptrue_b32();
    svbfloat16_t zero = svdup_bf16(0);
    for (int i = 0; i < m_num / 4; i++) {
        svint32_t out_v0 = svdup_s32(0);
        svint32_t out_v1 = svdup_s32(0);
        out_v0 = svread_hor_za32_s32_m(out_v0, pg32_all, 0, i);
        out_v1 = svread_hor_za32_s32_m(out_v1, pg32_all, 1, i);
        svfloat32_t rscales_v0 = svld1(pg32_all, rscales);
        svfloat32_t rscales_v1 = svld1(pg32_all, rscales + 16);
        float cscales_v0 = *(cscales + i);
        svbfloat16_t c0 = svld1(pg16_all, c + i * ldc);
        svfloat32_t out_v0_f = svcvt_f32_s32_x(pg32_all, out_v0);
        svfloat32_t out_v1_f = svcvt_f32_s32_x(pg32_all, out_v1);
        out_v0_f =
            svmla_x(pg32_all, svreinterpret_f32(svzip1(zero, c0)), out_v0_f, svmul_x(pg32_all, rscales_v0, cscales_v0));
        out_v1_f =
            svmla_x(pg32_all, svreinterpret_f32(svzip2(zero, c0)), out_v1_f, svmul_x(pg32_all, rscales_v1, cscales_v0));
        svst1(pg16_all, c + i * ldc, svuzp1(svcvt_bf16_x(pg32_all, out_v0_f), svcvt_bf16_x(pg32_all, out_v1_f)));
    }
}

__attribute__((always_inline)) inline void igemm14_load_save_bdq(const float *rscales, const float *cscales, int ldc,
    bfloat16_t *c, int n_num) __arm_inout("za") __arm_streaming
{
    svbool_t pg_c_16 = svwhilelt_b16(0, n_num / 4);
    svbool_t pg32_all = svptrue_b32();
    svbfloat16_t zero = svdup_bf16(0);
    for (int i = 0; i < 16; i++) {
        svint32_t out_v0 = svdup_s32(0);
        svint32_t out_v1 = svdup_s32(0);
        svint32_t out_v2 = svdup_s32(0);
        svint32_t out_v3 = svdup_s32(0);
        out_v0 = svread_hor_za32_s32_m(out_v0, pg32_all, 0, i);
        out_v1 = svread_hor_za32_s32_m(out_v1, pg32_all, 1, i);
        out_v2 = svread_hor_za32_s32_m(out_v2, pg32_all, 2, i);
        out_v3 = svread_hor_za32_s32_m(out_v3, pg32_all, 3, i);
        svfloat32_t rscales_v = svld1(pg32_all, rscales);
        float cscales_v0 = *(cscales + i), cscales_v1 = *(cscales + 16 + i);
        float cscales_v2 = *(cscales + 32 + i), cscales_v3 = *(cscales + 48 + i);
        svbfloat16_t c0 = svld1(pg_c_16, c + (i + 0) * ldc);
        svbfloat16_t c1 = svld1(pg_c_16, c + (i + 16) * ldc);
        svbfloat16_t c2 = svld1(pg_c_16, c + (i + 32) * ldc);
        svbfloat16_t c3 = svld1(pg_c_16, c + (i + 48) * ldc);
        svfloat32_t out_v0_f = svcvt_f32_s32_x(pg32_all, out_v0);
        svfloat32_t out_v1_f = svcvt_f32_s32_x(pg32_all, out_v1);
        svfloat32_t out_v2_f = svcvt_f32_s32_x(pg32_all, out_v2);
        svfloat32_t out_v3_f = svcvt_f32_s32_x(pg32_all, out_v3);
        out_v0_f =
            svmla_x(pg32_all, svreinterpret_f32(svzip1(zero, c0)), out_v0_f, svmul_x(pg32_all, rscales_v, cscales_v0));
        out_v1_f =
            svmla_x(pg32_all, svreinterpret_f32(svzip1(zero, c1)), out_v1_f, svmul_x(pg32_all, rscales_v, cscales_v1));
        out_v2_f =
            svmla_x(pg32_all, svreinterpret_f32(svzip1(zero, c2)), out_v2_f, svmul_x(pg32_all, rscales_v, cscales_v2));
        out_v3_f =
            svmla_x(pg32_all, svreinterpret_f32(svzip1(zero, c3)), out_v3_f, svmul_x(pg32_all, rscales_v, cscales_v3));
        svst1(pg_c_16, c + (i + 0) * ldc, svuzp1(svcvt_bf16_f32_x(pg32_all, out_v0_f), zero));
        svst1(pg_c_16, c + (i + 16) * ldc, svuzp1(svcvt_bf16_f32_x(pg32_all, out_v1_f), zero));
        svst1(pg_c_16, c + (i + 32) * ldc, svuzp1(svcvt_bf16_f32_x(pg32_all, out_v2_f), zero));
        svst1(pg_c_16, c + (i + 48) * ldc, svuzp1(svcvt_bf16_f32_x(pg32_all, out_v3_f), zero));
    }
}

__attribute__((always_inline)) inline void igemm11_load_save_bdq(const float *rscales, const float *cscales, int ldc,
    bfloat16_t *c, int m_num, int n_num) __arm_inout("za") __arm_streaming
{
    svbool_t pg_c_16 = svwhilelt_b16(0, n_num / 4);
    svbool_t pg32_all = svptrue_b32();
    svbfloat16_t zero = svdup_bf16(0);
    for (int i = 0; i < m_num / 4; i++) {
        svint32_t out_v0 = svdup_s32(0);
        out_v0 = svread_hor_za32_s32_m(out_v0, pg32_all, 0, i);
        svfloat32_t rscales_v = svld1(pg32_all, rscales);
        float cscales_v = *(cscales + i);
        svbfloat16_t c0 = svld1(pg_c_16, c + (i + 0) * ldc);
        svfloat32_t out_v0_f = svcvt_f32_s32_x(pg32_all, out_v0);
        out_v0_f =
            svmla_x(pg32_all, svreinterpret_f32(svzip1(zero, c0)), out_v0_f, svmul_x(pg32_all, rscales_v, cscales_v));
        svst1(pg_c_16, c + (i + 0) * ldc, svuzp1(svcvt_bf16_f32_x(pg32_all, out_v0_f), zero));
    }
}

__attribute__((always_inline)) inline void bgemm_4VL_1VL(const bfloat16_t *a0, const bfloat16_t *b, int k, int k_num,
    int n_num) __arm_inout("za") __arm_streaming
{
    svbool_t pg_16_m = svwhilelt_b16(0, 32);
    svbool_t pg_16_n = svwhilelt_b16(0, n_num);
    const bfloat16_t *a1 = a0 + k * 16, *a2 = a1 + k * 16, *a3 = a2 + k * 16;
    for (int ki = 0; ki < (k_num / 2); ki++) {
        svbfloat16_t m_data0 = svld1(pg_16_m, a0 + ki * 32);
        svbfloat16_t m_data1 = svld1(pg_16_m, a1 + ki * 32);
        svbfloat16_t m_data2 = svld1(pg_16_m, a2 + ki * 32);
        svbfloat16_t m_data3 = svld1(pg_16_m, a3 + ki * 32);
        svbfloat16_t n_data = svld1(pg_16_n, b + ki * n_num);
        svmopa_za32_bf16_m(0, pg_16_m, pg_16_n, m_data0, n_data);
        svmopa_za32_bf16_m(1, pg_16_m, pg_16_n, m_data1, n_data);
        svmopa_za32_bf16_m(2, pg_16_m, pg_16_n, m_data2, n_data);
        svmopa_za32_bf16_m(3, pg_16_m, pg_16_n, m_data3, n_data);
    }
}

__attribute__((always_inline)) inline void bgemm_1VL_1VL(const bfloat16_t *a, const bfloat16_t *b, int k, int k_num,
    int m_num, int n_num) __arm_inout("za") __arm_streaming
{
    svbool_t pg_16_m = svwhilelt_b16(0, m_num);
    svbool_t pg_16_n = svwhilelt_b16(0, n_num);
    for (int ki = 0; ki < (k_num / 2); ki++) {
        svbfloat16_t m_data = svld1(pg_16_m, a + ki * m_num);
        svbfloat16_t n_data = svld1(pg_16_n, b + ki * n_num);
        svmopa_za32_bf16_m(0, pg_16_m, pg_16_n, m_data, n_data);
    }
}

__attribute__((always_inline)) inline void bgemm41_save(int ldc, bfloat16_t *c, int n_num)
    __arm_inout("za") __arm_streaming
{
    svbool_t pg_c_16 = svwhilelt_b16(0, n_num / 2);
    svbool_t pg32_all = svptrue_b32();
    svfloat32_t zero_f32 = svdup_f32(0);
    svbfloat16_t zero_bf16 = svdup_bf16(0);
    for (int i = 0; i < 16; i++) {
        svfloat32_t out_v0_f = svread_hor_za32_f32_m(zero_f32, pg32_all, 0, i);
        svfloat32_t out_v1_f = svread_hor_za32_f32_m(zero_f32, pg32_all, 1, i);
        svfloat32_t out_v2_f = svread_hor_za32_f32_m(zero_f32, pg32_all, 2, i);
        svfloat32_t out_v3_f = svread_hor_za32_f32_m(zero_f32, pg32_all, 3, i);

        svst1(pg_c_16, c + (i + 00) * ldc, svuzp1(svcvt_bf16_f32_x(pg32_all, out_v0_f), zero_bf16));
        svst1(pg_c_16, c + (i + 16) * ldc, svuzp1(svcvt_bf16_f32_x(pg32_all, out_v1_f), zero_bf16));
        svst1(pg_c_16, c + (i + 32) * ldc, svuzp1(svcvt_bf16_f32_x(pg32_all, out_v2_f), zero_bf16));
        svst1(pg_c_16, c + (i + 48) * ldc, svuzp1(svcvt_bf16_f32_x(pg32_all, out_v3_f), zero_bf16));
    }
}

__attribute__((always_inline)) inline void bgemm11_save(int ldc, bfloat16_t *c, int m_num, int n_num)
    __arm_inout("za") __arm_streaming
{
    svbool_t pg_c_16 = svwhilelt_b16(0, n_num / 2);
    svbool_t pg32_all = svptrue_b32();
    svfloat32_t zero_f32 = svdup_f32(0);
    svbfloat16_t zero_bf16 = svdup_bf16(0);
    for (int i = 0; i < m_num / 2; i++) {
        svfloat32_t out_v0_f = svread_hor_za32_f32_m(zero_f32, pg32_all, 0, i);
        svst1(pg_c_16, c + i * ldc, svuzp1(svcvt_bf16_f32_x(pg32_all, out_v0_f), zero_bf16));
    }
}

} // namespace kutacc
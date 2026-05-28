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
#include <arm_sve.h>
#include <arm_sme.h>
#include <cmath>
#include "kutacc.h"
#include "utils/check.h"
#include "quant.h"

namespace kutacc {
__arm_new("za") void quant_pack_kernel(const int w_start_offset, const int w_end_offset, const int h_start_offset,
                                       const int h_end_offset, const int64_t width, const int height,
                                       bfloat16_t *__restrict__ acts, const float *quant_scales,
                                       int8_t *__restrict__ out) __arm_streaming
{
    const int64_t STEP_HALF = svcnth(), STEP_DOUBLE_HALF = svcnth() * 2;
    const int PACK_STEP = 16;
    svbool_t pg16 = svptrue_b16();
    svbool_t pg_32 = svptrue_b32();
    svbfloat16_t zero_b = svdup_bf16(0);
    for (int h_i = h_start_offset; h_i < h_end_offset; h_i += PACK_STEP) {
        int part_height = std::min(h_end_offset - h_i, PACK_STEP);
        const int H_NUM = part_height * 4;
        svbool_t pg_out = svwhilelt_b8(0, H_NUM);
        bfloat16_t *cur_acts = acts + h_i * width;
        int8_t *cur_out = out + h_i * width + w_start_offset * part_height;
        int w = w_start_offset;
        // 4
        for (; w + (STEP_DOUBLE_HALF * 4) <= w_end_offset; w += (STEP_DOUBLE_HALF * 4)) {
            for (int h = 0; h < part_height; h++) {
                float scales_inv = 1 / quant_scales[h_i + h];
                // 0
                svbfloat16_t i0_0 = svld1(pg16, cur_acts + h * width + w);
                svbfloat16_t i1_0 = svld1(pg16, cur_acts + h * width + w + STEP_HALF);
                svfloat32_t i00_0 = svreinterpret_f32(svzip1(zero_b, i0_0));
                svfloat32_t i01_0 = svreinterpret_f32(svzip2(zero_b, i0_0));
                svfloat32_t i10_0 = svreinterpret_f32(svzip1(zero_b, i1_0));
                svfloat32_t i11_0 = svreinterpret_f32(svzip2(zero_b, i1_0));
                i00_0 = svmul_x(pg_32, i00_0, scales_inv);
                i01_0 = svmul_x(pg_32, i01_0, scales_inv);
                i10_0 = svmul_x(pg_32, i10_0, scales_inv);
                i11_0 = svmul_x(pg_32, i11_0, scales_inv);

                svfloat16_t o0_0 = svuzp1(svcvt_f16_x(pg_32, i00_0), svcvt_f16_x(pg_32, i01_0));
                svfloat16_t o1_0 = svuzp1(svcvt_f16_x(pg_32, i10_0), svcvt_f16_x(pg_32, i11_0));
                svint8_t t0_0 = svqxtnb_s16(svcvt_s16_x(pg16, svrintn_x(pg16, o0_0)));
                svint8_t t1_0 = svqxtnb_s16(svcvt_s16_x(pg16, svrintn_x(pg16, o1_0)));

                svwrite_hor_za8_s8_m(0, h * 4, pg_out, svuzp1_s8(t0_0, t1_0));

                // 1
                svbfloat16_t i0_1 = svld1(pg16, cur_acts + h * width + w + 1 * STEP_DOUBLE_HALF);
                svbfloat16_t i1_1 = svld1(pg16, cur_acts + h * width + w + 1 * STEP_DOUBLE_HALF + STEP_HALF);
                svfloat32_t i00_1 = svreinterpret_f32(svzip1(zero_b, i0_1));
                svfloat32_t i01_1 = svreinterpret_f32(svzip2(zero_b, i0_1));
                svfloat32_t i10_1 = svreinterpret_f32(svzip1(zero_b, i1_1));
                svfloat32_t i11_1 = svreinterpret_f32(svzip2(zero_b, i1_1));
                i00_1 = svmul_x(pg_32, i00_1, scales_inv);
                i01_1 = svmul_x(pg_32, i01_1, scales_inv);
                i10_1 = svmul_x(pg_32, i10_1, scales_inv);
                i11_1 = svmul_x(pg_32, i11_1, scales_inv);

                svfloat16_t o0_1 = svuzp1(svcvt_f16_x(pg_32, i00_1), svcvt_f16_x(pg_32, i01_1));
                svfloat16_t o1_1 = svuzp1(svcvt_f16_x(pg_32, i10_1), svcvt_f16_x(pg_32, i11_1));
                svint8_t t0_1 = svqxtnb_s16(svcvt_s16_x(pg16, svrintn_x(pg16, o0_1)));
                svint8_t t1_1 = svqxtnb_s16(svcvt_s16_x(pg16, svrintn_x(pg16, o1_1)));

                svwrite_hor_za8_s8_m(0, h * 4 + 1, pg_out, svuzp1_s8(t0_1, t1_1));

                // 2
                svbfloat16_t i0_2 = svld1(pg16, cur_acts + h * width + w + 2 * STEP_DOUBLE_HALF);
                svbfloat16_t i1_2 = svld1(pg16, cur_acts + h * width + w + 2 * STEP_DOUBLE_HALF + STEP_HALF);
                svfloat32_t i00_2 = svreinterpret_f32(svzip1(zero_b, i0_2));
                svfloat32_t i01_2 = svreinterpret_f32(svzip2(zero_b, i0_2));
                svfloat32_t i10_2 = svreinterpret_f32(svzip1(zero_b, i1_2));
                svfloat32_t i11_2 = svreinterpret_f32(svzip2(zero_b, i1_2));
                i00_2 = svmul_x(pg_32, i00_2, scales_inv);
                i01_2 = svmul_x(pg_32, i01_2, scales_inv);
                i10_2 = svmul_x(pg_32, i10_2, scales_inv);
                i11_2 = svmul_x(pg_32, i11_2, scales_inv);

                svfloat16_t o0_2 = svuzp1(svcvt_f16_x(pg_32, i00_2), svcvt_f16_x(pg_32, i01_2));
                svfloat16_t o1_2 = svuzp1(svcvt_f16_x(pg_32, i10_2), svcvt_f16_x(pg_32, i11_2));
                svint8_t t0_2 = svqxtnb_s16(svcvt_s16_x(pg16, svrintn_x(pg16, o0_2)));
                svint8_t t1_2 = svqxtnb_s16(svcvt_s16_x(pg16, svrintn_x(pg16, o1_2)));

                svwrite_hor_za8_s8_m(0, h * 4 + 2, pg_out, svuzp1_s8(t0_2, t1_2));

                // 3
                svbfloat16_t i0_3 = svld1(pg16, cur_acts + h * width + w + 3 * STEP_DOUBLE_HALF);
                svbfloat16_t i1_3 = svld1(pg16, cur_acts + h * width + w + 3 * STEP_DOUBLE_HALF + STEP_HALF);
                svfloat32_t i00_3 = svreinterpret_f32(svzip1(zero_b, i0_3));
                svfloat32_t i01_3 = svreinterpret_f32(svzip2(zero_b, i0_3));
                svfloat32_t i10_3 = svreinterpret_f32(svzip1(zero_b, i1_3));
                svfloat32_t i11_3 = svreinterpret_f32(svzip2(zero_b, i1_3));
                i00_3 = svmul_x(pg_32, i00_3, scales_inv);
                i01_3 = svmul_x(pg_32, i01_3, scales_inv);
                i10_3 = svmul_x(pg_32, i10_3, scales_inv);
                i11_3 = svmul_x(pg_32, i11_3, scales_inv);

                svfloat16_t o0_3 = svuzp1(svcvt_f16_x(pg_32, i00_3), svcvt_f16_x(pg_32, i01_3));
                svfloat16_t o1_3 = svuzp1(svcvt_f16_x(pg_32, i10_3), svcvt_f16_x(pg_32, i11_3));
                svint8_t t0_3 = svqxtnb_s16(svcvt_s16_x(pg16, svrintn_x(pg16, o0_3)));
                svint8_t t1_3 = svqxtnb_s16(svcvt_s16_x(pg16, svrintn_x(pg16, o1_3)));

                svwrite_hor_za8_s8_m(0, h * 4 + 3, pg_out, svuzp1_s8(t0_3, t1_3));
            }

            // move za to out
            const int w_num = STEP_DOUBLE_HALF / 4;
            // 0
            for (int w_i = 0; w_i < w_num; w_i++) {
                svint32_t out_v = svdup_s32(0);
                // mova    z0.s, p1/m, za3v.s[w13, 0]
                out_v = svread_ver_za32_s32_m(out_v, svptrue_b32(), 0, w_i);
                svst1(pg_out, &cur_out[w_i * H_NUM], svreinterpret_s8(out_v));
            }
            cur_out += w_num * H_NUM;

            // 1
            for (int w_i = 0; w_i < w_num; w_i++) {
                svint32_t out_v = svdup_s32(0);
                // mova    z0.s, p1/m, za3v.s[w13, 0]
                out_v = svread_ver_za32_s32_m(out_v, svptrue_b32(), 1, w_i);
                svst1(pg_out, &cur_out[w_i * H_NUM], svreinterpret_s8(out_v));
            }
            cur_out += w_num * H_NUM;

            // 2
            for (int w_i = 0; w_i < w_num; w_i++) {
                svint32_t out_v = svdup_s32(0);
                // mova    z0.s, p1/m, za3v.s[w13, 0]
                out_v = svread_ver_za32_s32_m(out_v, svptrue_b32(), 2, w_i);
                svst1(pg_out, &cur_out[w_i * H_NUM], svreinterpret_s8(out_v));
            }
            cur_out += w_num * H_NUM;

            // 3
            for (int w_i = 0; w_i < w_num; w_i++) {
                svint32_t out_v = svdup_s32(0);
                // mova    z0.s, p1/m, za3v.s[w13, 0]
                out_v = svread_ver_za32_s32_m(out_v, svptrue_b32(), 3, w_i);
                svst1(pg_out, &cur_out[w_i * H_NUM], svreinterpret_s8(out_v));
            }
            cur_out += w_num * H_NUM;
        }

        // 3
        for (; w + (STEP_DOUBLE_HALF * 3) <= w_end_offset; w += (STEP_DOUBLE_HALF * 3)) {
            for (int h = 0; h < part_height; h++) {
                float scales_inv = 1 / quant_scales[h_i + h];
                // 0
                svbfloat16_t i0_0 = svld1(pg16, cur_acts + h * width + w);
                svbfloat16_t i1_0 = svld1(pg16, cur_acts + h * width + w + STEP_HALF);
                svfloat32_t i00_0 = svreinterpret_f32(svzip1(zero_b, i0_0));
                svfloat32_t i01_0 = svreinterpret_f32(svzip2(zero_b, i0_0));
                svfloat32_t i10_0 = svreinterpret_f32(svzip1(zero_b, i1_0));
                svfloat32_t i11_0 = svreinterpret_f32(svzip2(zero_b, i1_0));
                i00_0 = svmul_x(pg_32, i00_0, scales_inv);
                i01_0 = svmul_x(pg_32, i01_0, scales_inv);
                i10_0 = svmul_x(pg_32, i10_0, scales_inv);
                i11_0 = svmul_x(pg_32, i11_0, scales_inv);

                svfloat16_t o0_0 = svuzp1(svcvt_f16_x(pg_32, i00_0), svcvt_f16_x(pg_32, i01_0));
                svfloat16_t o1_0 = svuzp1(svcvt_f16_x(pg_32, i10_0), svcvt_f16_x(pg_32, i11_0));
                svint8_t t0_0 = svqxtnb_s16(svcvt_s16_x(pg16, svrintn_x(pg16, o0_0)));
                svint8_t t1_0 = svqxtnb_s16(svcvt_s16_x(pg16, svrintn_x(pg16, o1_0)));

                svwrite_hor_za8_s8_m(0, h * 4, pg_out, svuzp1_s8(t0_0, t1_0));

                // 1
                svbfloat16_t i0_1 = svld1(pg16, cur_acts + h * width + w + 1 * STEP_DOUBLE_HALF);
                svbfloat16_t i1_1 = svld1(pg16, cur_acts + h * width + w + 1 * STEP_DOUBLE_HALF + STEP_HALF);
                svfloat32_t i00_1 = svreinterpret_f32(svzip1(zero_b, i0_1));
                svfloat32_t i01_1 = svreinterpret_f32(svzip2(zero_b, i0_1));
                svfloat32_t i10_1 = svreinterpret_f32(svzip1(zero_b, i1_1));
                svfloat32_t i11_1 = svreinterpret_f32(svzip2(zero_b, i1_1));
                i00_1 = svmul_x(pg_32, i00_1, scales_inv);
                i01_1 = svmul_x(pg_32, i01_1, scales_inv);
                i10_1 = svmul_x(pg_32, i10_1, scales_inv);
                i11_1 = svmul_x(pg_32, i11_1, scales_inv);

                svfloat16_t o0_1 = svuzp1(svcvt_f16_x(pg_32, i00_1), svcvt_f16_x(pg_32, i01_1));
                svfloat16_t o1_1 = svuzp1(svcvt_f16_x(pg_32, i10_1), svcvt_f16_x(pg_32, i11_1));
                svint8_t t0_1 = svqxtnb_s16(svcvt_s16_x(pg16, svrintn_x(pg16, o0_1)));
                svint8_t t1_1 = svqxtnb_s16(svcvt_s16_x(pg16, svrintn_x(pg16, o1_1)));

                svwrite_hor_za8_s8_m(0, h * 4 + 1, pg_out, svuzp1_s8(t0_1, t1_1));

                // 2
                svbfloat16_t i0_2 = svld1(pg16, cur_acts + h * width + w + 2 * STEP_DOUBLE_HALF);
                svbfloat16_t i1_2 = svld1(pg16, cur_acts + h * width + w + 2 * STEP_DOUBLE_HALF + STEP_HALF);
                svfloat32_t i00_2 = svreinterpret_f32(svzip1(zero_b, i0_2));
                svfloat32_t i01_2 = svreinterpret_f32(svzip2(zero_b, i0_2));
                svfloat32_t i10_2 = svreinterpret_f32(svzip1(zero_b, i1_2));
                svfloat32_t i11_2 = svreinterpret_f32(svzip2(zero_b, i1_2));
                i00_2 = svmul_x(pg_32, i00_2, scales_inv);
                i01_2 = svmul_x(pg_32, i01_2, scales_inv);
                i10_2 = svmul_x(pg_32, i10_2, scales_inv);
                i11_2 = svmul_x(pg_32, i11_2, scales_inv);

                svfloat16_t o0_2 = svuzp1(svcvt_f16_x(pg_32, i00_2), svcvt_f16_x(pg_32, i01_2));
                svfloat16_t o1_2 = svuzp1(svcvt_f16_x(pg_32, i10_2), svcvt_f16_x(pg_32, i11_2));
                svint8_t t0_2 = svqxtnb_s16(svcvt_s16_x(pg16, svrintn_x(pg16, o0_2)));
                svint8_t t1_2 = svqxtnb_s16(svcvt_s16_x(pg16, svrintn_x(pg16, o1_2)));

                svwrite_hor_za8_s8_m(0, h * 4 + 2, pg_out, svuzp1_s8(t0_2, t1_2));
            }

            // move za to out
            const int w_num = STEP_DOUBLE_HALF / 4;
            // 0
            for (int w_i = 0; w_i < w_num; w_i++) {
                svint32_t out_v = svdup_s32(0);
                // mova    z0.s, p1/m, za3v.s[w13, 0]
                out_v = svread_ver_za32_s32_m(out_v, svptrue_b32(), 0, w_i);
                svst1(pg_out, &cur_out[w_i * H_NUM], svreinterpret_s8(out_v));
            }
            cur_out += w_num * H_NUM;

            // 1
            for (int w_i = 0; w_i < w_num; w_i++) {
                svint32_t out_v = svdup_s32(0);
                // mova    z0.s, p1/m, za3v.s[w13, 0]
                out_v = svread_ver_za32_s32_m(out_v, svptrue_b32(), 1, w_i);
                svst1(pg_out, &cur_out[w_i * H_NUM], svreinterpret_s8(out_v));
            }
            cur_out += w_num * H_NUM;

            // 2
            for (int w_i = 0; w_i < w_num; w_i++) {
                svint32_t out_v = svdup_s32(0);
                // mova    z0.s, p1/m, za3v.s[w13, 0]
                out_v = svread_ver_za32_s32_m(out_v, svptrue_b32(), 2, w_i);
                svst1(pg_out, &cur_out[w_i * H_NUM], svreinterpret_s8(out_v));
            }
            cur_out += w_num * H_NUM;
        }

        // 2
        for (; w + (STEP_DOUBLE_HALF * 2) <= w_end_offset; w += (STEP_DOUBLE_HALF * 2)) {
            for (int h = 0; h < part_height; h++) {
                float scales_inv = 1 / quant_scales[h_i + h];
                // 0
                svbfloat16_t i0_0 = svld1(pg16, cur_acts + h * width + w);
                svbfloat16_t i1_0 = svld1(pg16, cur_acts + h * width + w + STEP_HALF);
                svfloat32_t i00_0 = svreinterpret_f32(svzip1(zero_b, i0_0));
                svfloat32_t i01_0 = svreinterpret_f32(svzip2(zero_b, i0_0));
                svfloat32_t i10_0 = svreinterpret_f32(svzip1(zero_b, i1_0));
                svfloat32_t i11_0 = svreinterpret_f32(svzip2(zero_b, i1_0));
                i00_0 = svmul_x(pg_32, i00_0, scales_inv);
                i01_0 = svmul_x(pg_32, i01_0, scales_inv);
                i10_0 = svmul_x(pg_32, i10_0, scales_inv);
                i11_0 = svmul_x(pg_32, i11_0, scales_inv);

                svfloat16_t o0_0 = svuzp1(svcvt_f16_x(pg_32, i00_0), svcvt_f16_x(pg_32, i01_0));
                svfloat16_t o1_0 = svuzp1(svcvt_f16_x(pg_32, i10_0), svcvt_f16_x(pg_32, i11_0));
                svint8_t t0_0 = svqxtnb_s16(svcvt_s16_x(pg16, svrintn_x(pg16, o0_0)));
                svint8_t t1_0 = svqxtnb_s16(svcvt_s16_x(pg16, svrintn_x(pg16, o1_0)));

                svwrite_hor_za8_s8_m(0, h * 4, pg_out, svuzp1_s8(t0_0, t1_0));

                // 1
                svbfloat16_t i0_1 = svld1(pg16, cur_acts + h * width + w + 1 * STEP_DOUBLE_HALF);
                svbfloat16_t i1_1 = svld1(pg16, cur_acts + h * width + w + 1 * STEP_DOUBLE_HALF + STEP_HALF);
                svfloat32_t i00_1 = svreinterpret_f32(svzip1(zero_b, i0_1));
                svfloat32_t i01_1 = svreinterpret_f32(svzip2(zero_b, i0_1));
                svfloat32_t i10_1 = svreinterpret_f32(svzip1(zero_b, i1_1));
                svfloat32_t i11_1 = svreinterpret_f32(svzip2(zero_b, i1_1));
                i00_1 = svmul_x(pg_32, i00_1, scales_inv);
                i01_1 = svmul_x(pg_32, i01_1, scales_inv);
                i10_1 = svmul_x(pg_32, i10_1, scales_inv);
                i11_1 = svmul_x(pg_32, i11_1, scales_inv);

                svfloat16_t o0_1 = svuzp1(svcvt_f16_x(pg_32, i00_1), svcvt_f16_x(pg_32, i01_1));
                svfloat16_t o1_1 = svuzp1(svcvt_f16_x(pg_32, i10_1), svcvt_f16_x(pg_32, i11_1));
                svint8_t t0_1 = svqxtnb_s16(svcvt_s16_x(pg16, svrintn_x(pg16, o0_1)));
                svint8_t t1_1 = svqxtnb_s16(svcvt_s16_x(pg16, svrintn_x(pg16, o1_1)));

                svwrite_hor_za8_s8_m(0, h * 4 + 1, pg_out, svuzp1_s8(t0_1, t1_1));
            }

            // move za to out
            const int w_num = STEP_DOUBLE_HALF / 4;
            // 0
            for (int w_i = 0; w_i < w_num; w_i++) {
                svint32_t out_v = svdup_s32(0);
                // mova    z0.s, p1/m, za3v.s[w13, 0]
                out_v = svread_ver_za32_s32_m(out_v, svptrue_b32(), 0, w_i);
                svst1(pg_out, &cur_out[w_i * H_NUM], svreinterpret_s8(out_v));
            }
            cur_out += w_num * H_NUM;

            // 1
            for (int w_i = 0; w_i < w_num; w_i++) {
                svint32_t out_v = svdup_s32(0);
                // mova    z0.s, p1/m, za3v.s[w13, 0]
                out_v = svread_ver_za32_s32_m(out_v, svptrue_b32(), 1, w_i);
                svst1(pg_out, &cur_out[w_i * H_NUM], svreinterpret_s8(out_v));
            }
            cur_out += w_num * H_NUM;
        }

        // 1
        for (; w + STEP_DOUBLE_HALF <= w_end_offset; w += STEP_DOUBLE_HALF) {
            for (int h = 0; h < part_height; h++) {
                float scales_inv = 1 / quant_scales[h_i + h];
                // 0
                svbfloat16_t i0_0 = svld1(pg16, cur_acts + h * width + w);
                svbfloat16_t i1_0 = svld1(pg16, cur_acts + h * width + w + STEP_HALF);
                svfloat32_t i00_0 = svreinterpret_f32(svzip1(zero_b, i0_0));
                svfloat32_t i01_0 = svreinterpret_f32(svzip2(zero_b, i0_0));
                svfloat32_t i10_0 = svreinterpret_f32(svzip1(zero_b, i1_0));
                svfloat32_t i11_0 = svreinterpret_f32(svzip2(zero_b, i1_0));
                i00_0 = svmul_x(pg_32, i00_0, scales_inv);
                i01_0 = svmul_x(pg_32, i01_0, scales_inv);
                i10_0 = svmul_x(pg_32, i10_0, scales_inv);
                i11_0 = svmul_x(pg_32, i11_0, scales_inv);

                svfloat16_t o0_0 = svuzp1(svcvt_f16_x(pg_32, i00_0), svcvt_f16_x(pg_32, i01_0));
                svfloat16_t o1_0 = svuzp1(svcvt_f16_x(pg_32, i10_0), svcvt_f16_x(pg_32, i11_0));
                svint8_t t0_0 = svqxtnb_s16(svcvt_s16_x(pg16, svrintn_x(pg16, o0_0)));
                svint8_t t1_0 = svqxtnb_s16(svcvt_s16_x(pg16, svrintn_x(pg16, o1_0)));

                svwrite_hor_za8_s8_m(0, h * 4, pg_out, svuzp1_s8(t0_0, t1_0));
            }

            // move za to out
            const int w_num = STEP_DOUBLE_HALF / 4;
            // 0
            for (int w_i = 0; w_i < w_num; w_i++) {
                svint32_t out_v = svdup_s32(0);
                // mova    z0.s, p1/m, za3v.s[w13, 0]
                out_v = svread_ver_za32_s32_m(out_v, svptrue_b32(), 0, w_i);
                svst1(pg_out, &cur_out[w_i * H_NUM], svreinterpret_s8(out_v));
            }
            cur_out += w_num * H_NUM;
        }
    }
}

void quant_calculate_scale_kernel(bfloat16_t *input_data, float *scale_data, int num_tokens, int hidden_size,
                                  int64_t step_half, int64_t step_double_half)
{
    svbool_t pg16 = svptrue_b16();
    svbool_t pg_32 = svptrue_b32();
    svbfloat16_t zero_b = svdup_bf16(0);
    kutacc::parallel_for(0, num_tokens, 1, [&](int64_t start, int64_t end) {
        for (int i = start; i < end; ++i) {
            svfloat32_t mx_v = svdup_f32(std::numeric_limits<float>::lowest());
            for (int j = 0; j + step_double_half <= hidden_size; j += step_double_half) {
                svbfloat16_t i0 = svld1(pg16, input_data + i * hidden_size + j);
                svbfloat16_t i1 = svld1(pg16, input_data + i * hidden_size + j + step_half);
                svfloat32_t i00 = svreinterpret_f32(svzip1(zero_b, i0));
                svfloat32_t i01 = svreinterpret_f32(svzip2(zero_b, i0));
                svfloat32_t i10 = svreinterpret_f32(svzip1(zero_b, i1));
                svfloat32_t i11 = svreinterpret_f32(svzip2(zero_b, i1));
                i00 = svabs_f32_x(pg_32, i00);
                i01 = svabs_f32_x(pg_32, i01);
                i10 = svabs_f32_x(pg_32, i10);
                i11 = svabs_f32_x(pg_32, i11);
                mx_v = svmax_x(pg_32, i00, mx_v);
                mx_v = svmax_x(pg_32, i01, mx_v);
                mx_v = svmax_x(pg_32, i10, mx_v);
                mx_v = svmax_x(pg_32, i11, mx_v);
            }
            float max_value = svmaxv_f32(pg_32, mx_v);
            scale_data[i] = max_value / 127.0f;
        }
    });
}

void quant_pack(bfloat16_t *input_data, int8_t *output_data, float *scale_data, int hidden_size, int num_tokens)
{
    const int64_t STEP_HALF = svcnth();
    const int64_t STEP_DOUBLE_HALF = svcnth() * 2;
    KUTACC_CHECK((hidden_size % STEP_DOUBLE_HALF) == 0, "(hidden_size % STEP_DOUBLE_HALF) != 0");
    kutacc::quant_calculate_scale_kernel(input_data, scale_data, num_tokens, hidden_size, STEP_HALF, STEP_DOUBLE_HALF);

    // quant pack
    const int64_t PART_WIDTH_BASE_NUM = STEP_DOUBLE_HALF * 4;
    const int64_t PART_HEIGHT = 16;
    int64_t thread_nums = kutacc::get_thread_num();
    int64_t part_width = (hidden_size + (thread_nums * PART_WIDTH_BASE_NUM) - 1) / (thread_nums * PART_WIDTH_BASE_NUM) *
                         PART_WIDTH_BASE_NUM;  // round multiples of PART_WIDTH_BASE_NUM
    int64_t used_thread_nums = (hidden_size + part_width - 1) / part_width;
    int64_t left_thread_nums = thread_nums / used_thread_nums;
    int64_t part_height =
        (num_tokens + (left_thread_nums * PART_HEIGHT) - 1) / (left_thread_nums * PART_HEIGHT) * PART_HEIGHT;
    kutacc::parallel_for(0, thread_nums, 1, [&](int64_t start, int64_t end) {
        int64_t thread_id = kutacc::get_thread_id();
        int64_t w_thread_id = thread_id % used_thread_nums;
        int64_t h_thread_id = thread_id / used_thread_nums;
        kutacc::quant_pack_kernel(
            w_thread_id * part_width, std::min((int64_t)hidden_size, (w_thread_id + 1) * part_width),
            h_thread_id * part_height, std::min((int64_t)num_tokens, (h_thread_id + 1) * part_height), hidden_size,
            num_tokens, input_data, scale_data, output_data);
    });
}

void quant(int64_t height, int64_t width, const __bf16 *input, int64_t input_stride, int8_t *out, int64_t out_stride,
    float *scale)
{
    const int64_t STEP_HALF = svcnth();
    const int64_t STEP_32 = svcnth() * 2;
    svbool_t pg16 = svptrue_b16();
    svbool_t pg_32 = svptrue_b32();
    svbool_t pg8 = svptrue_b8();
    svbfloat16_t zero_b = svdup_bf16(0);
    kutacc::parallel_for(0, height, 1, [&](int64_t start, int64_t end) {
        for (int i = start; i < end; ++i) {
            svfloat32_t mx_v = svdup_f32(0);
            int j = 0;
            for (; j + STEP_32 <= width; j += STEP_32) {
                svbfloat16_t i0 = svld1(pg16, input + i * input_stride + j);
                svbfloat16_t i1 = svld1(pg16, input + i * input_stride + j + STEP_HALF);
                svfloat32_t i00 = svreinterpret_f32(svzip1(zero_b, i0));
                svfloat32_t i01 = svreinterpret_f32(svzip2(zero_b, i0));
                svfloat32_t i10 = svreinterpret_f32(svzip1(zero_b, i1));
                svfloat32_t i11 = svreinterpret_f32(svzip2(zero_b, i1));
                i00 = svabs_f32_x(pg_32, i00);
                i01 = svabs_f32_x(pg_32, i01);
                i10 = svabs_f32_x(pg_32, i10);
                i11 = svabs_f32_x(pg_32, i11);
                mx_v = svmax_x(pg_32, i00, mx_v);
                mx_v = svmax_x(pg_32, i01, mx_v);
                mx_v = svmax_x(pg_32, i10, mx_v);
                mx_v = svmax_x(pg_32, i11, mx_v);
            }
            float max_value = svmaxv_f32(pg_32, mx_v);
            for (; j < width; ++j) {
                max_value = std::max(max_value, std::abs((float)input[i * input_stride + j]));
            }
            float scale_val = max_value / 127.0f;
            scale[i] = scale_val;
            float scale_val_inv = 1 / scale_val;
            for (j = 0; j + STEP_32 <= width; j += STEP_32) {
                svbfloat16_t i0 = svld1(pg16, input + i * input_stride + j);
                svbfloat16_t i1 = svld1(pg16, input + i * input_stride + j + STEP_HALF);
                svfloat32_t i00 = svreinterpret_f32(svzip1(zero_b, i0));
                svfloat32_t i01 = svreinterpret_f32(svzip2(zero_b, i0));
                svfloat32_t i10 = svreinterpret_f32(svzip1(zero_b, i1));
                svfloat32_t i11 = svreinterpret_f32(svzip2(zero_b, i1));
                i00 = svmul_x(pg_32, i00, scale_val_inv);
                i01 = svmul_x(pg_32, i01, scale_val_inv);
                i10 = svmul_x(pg_32, i10, scale_val_inv);
                i11 = svmul_x(pg_32, i11, scale_val_inv);

                svfloat16_t o0 = svuzp1(svcvt_f16_x(pg_32, i00), svcvt_f16_x(pg_32, i01));
                svfloat16_t o1 = svuzp1(svcvt_f16_x(pg_32, i10), svcvt_f16_x(pg_32, i11));
                svint8_t t0 = svqxtnb_s16(svcvt_s16_x(pg16, svrintn_x(pg16, o0)));
                svint8_t t1 = svqxtnb_s16(svcvt_s16_x(pg16, svrintn_x(pg16, o1)));
                svst1(pg8, out + i * out_stride + j, svuzp1(t0, t1));
            }
            for (; j < width; ++j) {
                out[i * out_stride + j] =
                    std::nearbyint(std::clamp(input[i * input_stride + j] / scale_val, -127.0f, 127.0f));
            }
        }
    });
}

}  // namespace kutacc

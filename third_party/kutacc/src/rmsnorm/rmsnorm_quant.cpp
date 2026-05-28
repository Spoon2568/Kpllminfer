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
#include <arm_neon.h>
#include <cmath>
#include <iostream>
#include <type_traits>
#include "kutacc.h"
#include "utils/check.h"
#include "def.h"

namespace kutacc {

template <typename scalar_t>
inline void rmsnorm_quant_kernel(int64_t start, int64_t end, float quant_scale, const scalar_t *acts,
    const scalar_t *weights, int8_t *outs)
{
    if (start >= end) {
        return;
    }
    int64_t i = start;
    svbfloat16_t zero_b = svdup_bf16(0);
    svbool_t pg8 = svptrue_b8();
    svbool_t pg16 = svptrue_b16();
    svbool_t pg32 = svptrue_b32();
#pragma unroll(4)
    for (; i + STEP * 2 <= end; i += STEP * 2) {
        static_assert(std::is_same<scalar_t, bfloat16_t>::value);
        svbfloat16_t a0 = svld1(pg16, acts + i);
        svbfloat16_t a1 = svld1(pg16, acts + i + STEP);
        svfloat32_t a00 = svreinterpret_f32(svzip1(zero_b, a0));
        svfloat32_t a01 = svreinterpret_f32(svzip2(zero_b, a0));
        svfloat32_t a10 = svreinterpret_f32(svzip1(zero_b, a1));
        svfloat32_t a11 = svreinterpret_f32(svzip2(zero_b, a1));

        svbfloat16_t w0 = svld1(pg16, weights + i);
        svbfloat16_t w1 = svld1(pg16, weights + i + STEP);
        svfloat32_t w00 = svreinterpret_f32(svzip1(zero_b, w0));
        svfloat32_t w01 = svreinterpret_f32(svzip2(zero_b, w0));
        svfloat32_t w10 = svreinterpret_f32(svzip1(zero_b, w1));
        svfloat32_t w11 = svreinterpret_f32(svzip2(zero_b, w1));

        svfloat32_t o00 = svmul_x(pg32, svmul_x(pg32, a00, quant_scale), w00);
        svfloat32_t o01 = svmul_x(pg32, svmul_x(pg32, a01, quant_scale), w01);
        svfloat32_t o10 = svmul_x(pg32, svmul_x(pg32, a10, quant_scale), w10);
        svfloat32_t o11 = svmul_x(pg32, svmul_x(pg32, a11, quant_scale), w11);

        svfloat16_t o0 = svuzp1(svcvt_f16_x(pg32, o00), svcvt_f16_x(pg32, o01));
        svfloat16_t o1 = svuzp1(svcvt_f16_x(pg32, o10), svcvt_f16_x(pg32, o11));

        svint8_t t0 = svqxtnb_s16(svcvt_s16_x(pg16, svrintn_x(pg16, o0)));
        svint8_t t1 = svqxtnb_s16(svcvt_s16_x(pg16, svrintn_x(pg16, o1)));

        svst1(pg8, outs + i - start, svuzp1(t0, t1));
    }
    for (; i < end; ++i) {
        float tmp = std::clamp(std::round(to_float(acts[i]) * quant_scale * to_float(weights[i])), -128.f, 127.f);
        outs[i - start] = to_bf16(tmp);
    }
}

template <typename scalar_t, bool has_residual>
void rmsnorm_quant(int64_t height, int64_t width, scalar_t *acts, int64_t acts_stride, const scalar_t *weights,
    float eps, scalar_t *residual, int8_t *outs, float *scales)
{
    kutacc::parallel_for(0, height, 1, [&](int64_t start, int64_t end) {
        for (int64_t i = start; i < end; i++) {
            float quant_scale;
            int64_t offset = i * width;
            int64_t acts_offset = i * acts_stride;
            int64_t outs_offset = i * width;
            if constexpr (has_residual) {
                get_sum_quant<scalar_t, has_residual>(width, acts + acts_offset, weights, eps, residual + offset,
                    quant_scale, scales[i]);
                rmsnorm_quant_kernel<scalar_t>(0, width, quant_scale, residual + offset, weights, outs + outs_offset);
            } else {
                get_sum_quant<scalar_t, has_residual>(width, acts + acts_offset, weights, eps, nullptr, quant_scale,
                    scales[i]);
                rmsnorm_quant_kernel<scalar_t>(0, width, quant_scale, acts + acts_offset, weights, outs + outs_offset);
            }
        }
    });
}

template void rmsnorm_quant<__bf16, false>(int64_t height, int64_t width, __bf16 *acts, int64_t acts_stride,
    const __bf16 *weights, float eps, __bf16 *residual, int8_t *outs, float *scales);

template void rmsnorm_quant<__bf16, true>(int64_t height, int64_t width, __bf16 *acts, int64_t acts_stride,
    const __bf16 *weights, float eps, __bf16 *residual, int8_t *outs, float *scales);

} // namespace kutacc
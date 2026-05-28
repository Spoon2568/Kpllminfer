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
#include <arm_neon.h>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <type_traits>

#include "kutacc.h"
#include "def.h"

namespace kutacc {

template <typename scalar_t>
inline void rmsnorm_kernel(int64_t start, int64_t end, float sqrtsum, const scalar_t *acts, const scalar_t *weights,
    scalar_t *outs)
{
    if (start >= end) {
        return;
    }
    int64_t i = start;
    svbool_t pg16 = svptrue_b16();
    svbool_t pg32 = svptrue_b32();
    svfloat32_t s = svdup_f32(sqrtsum);
    svfloat16_t zero_h = svdup_f16(0);
    svbfloat16_t zero_b = svdup_bf16(0);
#pragma unroll(4)
    for (; i + STEP <= end; i += STEP) {
        if constexpr (std::is_same<scalar_t, float16_t>::value) {
            svfloat16_t a = svld1(pg16, acts + i);
            svfloat32_t a0 = svcvt_f32_x(pg32, svzip1(a, zero_h));
            svfloat32_t a1 = svcvt_f32_x(pg32, svzip2(a, zero_h));
            svfloat16_t w = svld1(pg16, weights + i);
            svfloat32_t w0 = svcvt_f32_x(pg32, svzip1(w, zero_h));
            svfloat32_t w1 = svcvt_f32_x(pg32, svzip2(w, zero_h));
            svfloat16_t o0 = svcvt_f16_x(pg32, svmul_x(pg32, svmul_x(pg32, a0, s), w0));
            svfloat16_t o1 = svcvt_f16_x(pg32, svmul_x(pg32, svmul_x(pg32, a1, s), w1));
            svst1(pg16, outs + i, svuzp1(o0, o1));
        } else {
            svbfloat16_t a = svld1(pg16, acts + i);
            svfloat32_t a0 = svreinterpret_f32(svzip1(zero_b, a));
            svfloat32_t a1 = svreinterpret_f32(svzip2(zero_b, a));
            svbfloat16_t w = svld1(pg16, weights + i);
            svfloat32_t w0 = svreinterpret_f32(svzip1(zero_b, w));
            svfloat32_t w1 = svreinterpret_f32(svzip2(zero_b, w));
            svbfloat16_t o0 = svcvt_bf16_x(pg32, svmul_x(pg32, svmul_x(pg32, a0, s), w0));
            svbfloat16_t o1 = svcvt_bf16_x(pg32, svmul_x(pg32, svmul_x(pg32, a1, s), w1));
            svst1(pg16, outs + i, svuzp1(o0, o1));
        }
    }
    for (; i < end; ++i) {
        float tmp = to_float(acts[i]) * sqrtsum * to_float(weights[i]);
        outs[i] = to_bf16(tmp);
    }
}

template <typename scalar_t, bool has_residual>
void rmsnorm(int64_t height, int64_t width, scalar_t *acts, int64_t acts_stride, const scalar_t *weights, float eps,
    scalar_t *residual, scalar_t *outs)
{
    kutacc::parallel_for(0, height, 1, [&](int start, int end) {
        for (int i = start; i < end; i++) {
            int64_t offset = i * width;
            int64_t acts_offset = i * acts_stride;
            if constexpr (has_residual) {
                float sqrtsum =
                    1.0f /
                    sqrt(
                        get_sum<scalar_t, has_residual>(0, width, acts + acts_offset, residual + offset) / width + eps);
                rmsnorm_kernel<scalar_t>(0, width, sqrtsum, residual + offset, weights, outs + offset);
            } else {
                float sqrtsum =
                    1.0f / sqrt(get_sum<scalar_t, has_residual>(0, width, acts + acts_offset, nullptr) / width + eps);
                rmsnorm_kernel<scalar_t>(0, width, sqrtsum, acts + acts_offset, weights, outs + offset);
            }
        }
    });
}

template <typename scalar_t>
void head_rmsnorm(int64_t token_num, int64_t head_num, int64_t head_dim, scalar_t *acts, int64_t acts_stride, const scalar_t *weights, float eps, scalar_t *outs)
{
    kutacc::parallel_for(0, token_num * head_num, 1, [&](int start, int end) {
        for (int i = start; i < end; i++) {
            int64_t token_offset = i / head_num;
            int64_t head_offset = i % head_num;
            int64_t acts_offset = token_offset * acts_stride + head_offset * head_dim;
            int64_t offset = i * head_dim;
            float sqrtsum =
                1.0f / sqrt(get_sum<scalar_t, false>(0, head_dim, acts + acts_offset, nullptr) / head_dim + eps);
            rmsnorm_kernel<scalar_t>(0, head_dim, sqrtsum, acts + acts_offset, weights, outs + offset);
        }
    });
} 


template void rmsnorm<__bf16, false>(int64_t height, int64_t width, __bf16 *acts, int64_t acts_stride,
    const __bf16 *weights, float eps, __bf16 *residual, __bf16 *outs);

template void rmsnorm<__bf16, true>(int64_t height, int64_t width, __bf16 *acts, int64_t acts_stride,
    const __bf16 *weights, float eps, __bf16 *residual, __bf16 *outs);

template void head_rmsnorm<__bf16>(int64_t token_num, int64_t head_num, int64_t head_dim, __bf16 *acts, int64_t acts_stride, const __bf16 *weights, float eps, __bf16 *outs);

} // namespace kutacc
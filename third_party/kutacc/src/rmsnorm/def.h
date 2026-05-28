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
#include <arm_sve.h>

#ifndef __GNUC__
#define UNLIKELY(x) (x)
#else
#define UNLIKELY(x) __builtin_expect(!!(x), 0)
#endif

const int64_t STEP = svcnth();

const int64_t PREFETCH_DIST = 16;
namespace kutacc {

inline __bf16 to_bf16(float x)
{
    return vcvth_bf16_f32(x);
}

inline float to_float(__bf16 x)
{
    return vcvtah_f32_bf16(x);
}

template <typename scalar_t, bool has_residual>
inline float get_sum(const int64_t start, const int64_t end, const scalar_t *acts, scalar_t *residual)
{
    if (start >= end) {
        return 0;
    }
    int64_t i = start;
    float sum = 0;
    svbool_t pg16 = svptrue_b16();
    svbool_t pg32 = svptrue_b32();
    svfloat32_t sqrsum = svdup_f32(0);
    svfloat16_t zero_h = svdup_f16(0);
    svbfloat16_t zero_b = svdup_bf16(0);
#pragma unroll(4)
    for (; i + STEP <= end; i += STEP) {
        svfloat32_t a0;
        svfloat32_t a1;
        if constexpr (std::is_same<scalar_t, float16_t>::value) {
            svfloat16_t a = svld1(pg16, acts + i);
            a0 = svcvt_f32_x(pg32, svzip1(a, zero_h)), a1 = svcvt_f32_x(pg32, svzip2(a, zero_h));
            if constexpr (has_residual) {
                svfloat16_t b = svld1(pg16, residual + i);
                a0 = svadd_x(pg32, a0, svcvt_f32_x(pg32, svzip1(b, zero_h))),
                a1 = svadd_x(pg32, a1, svcvt_f32_x(pg32, svzip2(b, zero_h)));
                svst1(pg16, residual + i, svuzp1(svcvt_f16_x(pg32, a0), svcvt_f16_x(pg32, a1)));
            }
        } else {
            svbfloat16_t a = svld1(pg16, acts + i);
            a0 = svreinterpret_f32(svzip1(zero_b, a)), a1 = svreinterpret_f32(svzip2(zero_b, a));
            if constexpr (has_residual) {
                svbfloat16_t b = svld1(pg16, residual + i);
                a0 = svadd_x(pg32, a0, svreinterpret_f32(svzip1(zero_b, b))),
                a1 = svadd_x(pg32, a1, svreinterpret_f32(svzip2(zero_b, b)));
                svst1(pg16, residual + i, svuzp1(svcvt_bf16_x(pg32, a0), svcvt_bf16_x(pg32, a1)));
            }
        }
        sqrsum = svmla_x(pg32, sqrsum, a0, a0);
        sqrsum = svmla_x(pg32, sqrsum, a1, a1);
    }
    sum = svaddv_f32(pg32, sqrsum);
    for (; i < end; i++) {
        if constexpr (has_residual) {
            residual[i] = float(acts[i]) + residual[i];
        }
        sum += float(residual[i]) * residual[i];
    }
    return sum;
}

template <typename scalar_t, bool has_residual>
inline void get_sum_quant(const int64_t width, const scalar_t *acts, const scalar_t *weights, float eps,
    scalar_t *residual, float &quant_scale, float &scale)
{
    int64_t i = 0;
    float sum = 0;
    float max = 0;
    svbool_t pg16 = svptrue_b16();
    svbool_t pg32 = svptrue_b32();
    svfloat32_t sqrsum = svdup_f32(0);
    svfloat32_t absmax = svdup_f32(0);
    svbfloat16_t zero_b = svdup_bf16(0);
    static_assert(std::is_same<scalar_t, bfloat16_t>::value);
#pragma unroll(4)
    for (; i + STEP <= width; i += STEP) {
        svfloat32_t a0;
        svfloat32_t a1;
        svbfloat16_t a = svld1(pg16, acts + i);
        a0 = svreinterpret_f32(svzip1(zero_b, a));
        a1 = svreinterpret_f32(svzip2(zero_b, a));
        if constexpr (has_residual) {
            svbfloat16_t b = svld1(pg16, residual + i);
            a0 = svadd_x(pg32, a0, svreinterpret_f32(svzip1(zero_b, b))),
            a1 = svadd_x(pg32, a1, svreinterpret_f32(svzip2(zero_b, b)));
            svst1(pg16, residual + i, svuzp1(svcvt_bf16_x(pg32, a0), svcvt_bf16_x(pg32, a1)));
        }
        sqrsum = svmla_x(pg32, sqrsum, a0, a0);
        sqrsum = svmla_x(pg32, sqrsum, a1, a1);
        svbfloat16_t w = svld1(pg16, weights + i);
        svfloat32_t w0 = svreinterpret_f32(svzip1(zero_b, w));
        svfloat32_t w1 = svreinterpret_f32(svzip2(zero_b, w));
        a0 = svmul_x(pg32, a0, w0);
        a1 = svmul_x(pg32, a1, w1);
        absmax = svmax_x(pg32, absmax, svabs_x(pg32, a0));
        absmax = svmax_x(pg32, absmax, svabs_x(pg32, a1));
    }
    sum = svaddv_f32(pg32, sqrsum);
    max = svmaxv_f32(pg32, absmax);
    for (; i < width; i++) {
        if constexpr (has_residual) {
            residual[i] = float(acts[i]) + residual[i];
        }
        sum += float(residual[i]) * residual[i];
        max = std::max(max, std::abs(float(residual[i]) * weights[i]));
    }
    float rms = sqrt(sum / width + eps);
    quant_scale = 127 / max;
    scale = max / 127 / rms;
}

}  // namespace kutacc
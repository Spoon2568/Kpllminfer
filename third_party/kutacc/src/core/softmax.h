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
#ifndef SOFTMAX_H
#define SOFTMAX_H

#include <cmath>
#include <cstdlib>
#include "../math/fast_exp.h"
#include "../wrapper/wrapper.h"

namespace kutacc {
template <typename scalar_t>
inline void softmax_with_max(float *buf, scalar_t *out, int64_t size, float max)
{
    auto vl = svcntw();
    // sub max & exp & reduce sum
    svfloat32_t reduce = svdup_f32(0.f);
    for (int64_t i = 0; i < size; i += vl) {
        svbool_t pg = svwhilelt_b32(i, size);
        svfloat32_t values = svld1_f32(pg, &buf[i]);
        values = kutacc::fast_exp(pg, svsub_x(pg, values, max));
        reduce = svadd_m(pg, reduce, values);
        svst1_f32(pg, &buf[i], values);
    }
    // mul sum_inv & convert to scalar_t
    float sum_inv = 1 / svaddv(svptrue_b32(), reduce);
    for (int64_t i = 0; i < size; i += vl) {
        svbool_t pg = svwhilelt_b32(i, size);
        svfloat32_t values = svld1_f32(pg, &buf[i]);
        values = svmul_x(pg, values, sum_inv);
        kutacc::svst1<scalar_t, float>(pg, &out[i], values);
    }
}

template <typename scalar_t>
inline void softmax(scalar_t *data, int64_t size, scalar_t *out, float *buf)
{
    auto vl = svcntw();
    // convert to float && reduce max
    svfloat32_t reduce = svdup_f32(-INFINITY);
    for (int64_t i = 0; i < size; i += vl) {
        svbool_t pg = svwhilelt_b32(i, size);
        svfloat32_t values = kutacc::svld1<float>(pg, &data[i]);
        reduce = svmax_m(pg, reduce, values);
        svst1_f32(pg, &buf[i], values);
    }
    // softmax with max
    softmax_with_max(buf, out, size, svmaxv(svptrue_b32(), reduce));
}
}   // namespace kutacc

#endif

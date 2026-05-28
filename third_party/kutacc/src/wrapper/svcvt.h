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
#ifndef SVCVT_H
#define SVCVT_H

#include <arm_sve.h>
#include "vector_of.h"

namespace kutacc {
template <typename dst_t, typename src_t>
vector_of_t<dst_t> svcvt(svbool_t pg, vector_of_t<src_t> src);

#define DEFINE_SVCVT(short_dst_t, dst_t, src_t)                                                     \
    template <>                                                                                     \
    inline __attribute__((__always_inline__)) vector_of_t<dst_t> svcvt<dst_t, src_t>(svbool_t pg,   \
        vector_of_t<src_t> src)                                                                     \
    {                                                                                               \
        return svcvt_##short_dst_t##_x(pg, src);                                                    \
    }
DEFINE_SVCVT(bf16, __bf16, float);
#undef DEFINE_SVCVT

template <>
inline __attribute__((__always_inline__)) svfloat32_t svcvt<float, __bf16>(svbool_t pg, svbfloat16_t src)
{
    return svreinterpret_f32(svlsl_x(pg, svreinterpret_u32(src), (uint32_t)16));
}
}   // namespace kutacc

#endif

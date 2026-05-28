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
#ifndef SVST1_H
#define SVST1_H

#include <type_traits>
#include "vector_of.h"

namespace kutacc {
template <typename dst_t, typename src_t>
inline void svst1(svbool_t pg, dst_t *dst, vector_of_t<src_t> src)
{
    if constexpr (std::is_same_v<dst_t, src_t>) {
        ::svst1(pg, dst, src);
    } else if constexpr (sizeof(dst_t) == SIZEOF_BF16 && sizeof(src_t) == SIZEOF_FLOAT32) {
        auto pg_dst = svuzp1_b16(pg, svpfalse());
        auto values = svcvt<dst_t, src_t>(pg, src);
        values = svuzp1(values, values);
        ::svst1(pg_dst, dst, values);
    } else {
        dst_t::not_implemented();
    }
}
}   // namespace kutacc

#endif

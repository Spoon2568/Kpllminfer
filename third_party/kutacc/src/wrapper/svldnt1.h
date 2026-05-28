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
#ifndef SVLDNT1_H
#define SVLDNT1_H

#include <type_traits>
#include "vector_of.h"

namespace kutacc {
template <typename dst_t, typename src_t>
inline vector_of_t<dst_t> svldnt1(svbool_t pg, const src_t *data)
{
    if constexpr (std::is_same_v<dst_t, src_t>) {
        return ::svldnt1(pg, data);
    } else if constexpr (sizeof(dst_t) == SIZEOF_FLOAT32 && sizeof(src_t) == SIZEOF_BF16) {
        auto pg_src = svuzp1_b16(pg, svpfalse());
        auto values = ::svldnt1(pg_src, data);
        values = svzip1(values, values);
        return kutacc::svcvt<dst_t, src_t>(pg, values);
    } else {
        dst_t::not_implemented();
    }
}
}   // namespace kutacc

#endif

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
#ifndef SVDUP_H
#define SVDUP_H

#include <arm_sve.h>
#include "vector_of.h"

namespace kutacc {
template <typename scalar_t>
vector_of_t<scalar_t> svdup(scalar_t);

#define DEFINE_WRAPPER(nbits, short_scalar_t, scalar_t, vector_t)                               \
    template <>                                                                                 \
    inline __attribute__((__always_inline__)) vector_t svdup<scalar_t>(scalar_t value)          \
    {                                                                                           \
        return svdup_##short_scalar_t(value);                                                   \
    }
#include "wrapper-incl.h"
#undef DEFINE_WRAPPER
}   // namespace kutacc

#endif

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
#ifndef SVCNT_H
#define SVCNT_H

#include <arm_sve.h>
#include <cstdint>
#include "vector_of.h"

namespace kutacc {
template <typename scalar_t>
uint64_t svcnt();

#define DEFINE_WRAPPER(nbits, short_scalar_t, scalar_t, vector_t)               \
    template <>                                                                 \
    inline __attribute__((__always_inline__)) uint64_t svcnt<scalar_t>()        \
    {                                                                           \
        return __ARM_FEATURE_SVE_BITS / (nbits);                                \
    };
#include "wrapper-incl.h"
#undef DEFINE_WRAPPER
}  // namespace kutacc

#endif

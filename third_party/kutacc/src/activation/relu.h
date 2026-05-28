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
#ifndef KUTACC_RELU
#define KUTACC_RELU

#include <algorithm>
#include <type_traits>

#include "../wrapper/wrapper.h"

namespace kutacc {
    class ReLU {
    public:
        /**
         * @brief Template function `call` can be instantiated as follows
         * 1. svfloat32_t call(svbool_t, svfloat32_t)
         * 2. svfloat16_t call(svbool_t, svfloat16_t)
         */
        template <typename T>
        static std::enable_if_t<std::is_same_v<T, svfloat32_t> || std::is_same_v<T, svfloat16_t>, T> call(
            svbool_t pg, T x)
        {
            return svmax_x(pg, x, 0);
        }

        /**
         * @brief Template function `call` can be instantiated as follows
         * 1. float call(float)
         * 2. __fp16 call(__fp16)
         */
        template <typename T>
        static std::enable_if_t<std::is_same_v<T, float> || std::is_same_v<T, __fp16>, T> call(T x)
        {
            return std::max(x, (T)0);
        }
    }
}   // namespace kutacc

#endif
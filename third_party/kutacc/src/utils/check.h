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
#ifndef KUTACC_CHECK_H
#define KUTACC_CHECK_H

#include <iostream>
#include <sstream>
#include <string>

namespace kutacc {
extern bool kutacc_check_err_set;
namespace internal {
    inline void check_fail_print(std::stringstream &stream)
    {
        stream << std::endl;
    }

    template <typename Arg, typename... Rest>
    inline void check_fail_print(std::stringstream &stream, Arg &&arg, Rest &&...rest)
    {
        stream << std::forward<Arg>(arg);
        check_fail_print(stream, rest...);
    }

    template <typename... Args>
    inline void check_fail(std::string func, std::string file, int line, Args &&...args)
    {
        std::stringstream stream;
        stream << "KUTACC_CHECK fail in " << func << " at " << file << ":" << line << " , ";
        check_fail_print(stream, std::forward<Args>(args)...);
        stream << "\n";
        std::cerr << stream.str();
    }

}   // namespace internal
}   // namespace kutacc

#define KUTACC_CHECK(condition, ...)                                                            \
    do {                                                                                        \
        if (__builtin_expect(!(condition), 0)) {                                                \
            kutacc::internal::check_fail(__func__, __FILE__, __LINE__, __VA_ARGS__);            \
            kutacc::kutacc_check_err_set = true;                                                \
        }                                                                                       \
    } while (0)

#define KUTACC_CHECK_TENSOR_SHAPE(tensor, ...)                                                                     \
    KUTACC_CHECK((tensor).sizes() == c10::IntArrayRef({__VA_ARGS__}), "invalid tensor shape: ", (tensor).sizes(),  \
        ", expect: ", c10::IntArrayRef({__VA_ARGS__}))

#define KUTACC_CHECK_TENSORWRAPPER_SHAPE(tensor, ...)                                           \
    KUTACC_CHECK(c10::IntArrayRef((tensor).sizes) == c10::IntArrayRef({__VA_ARGS__}),           \
        "invalid tensor wrapper shape: ", c10::IntArrayRef((tensor).sizes),                     \
        ", expect: ", c10::IntArrayRef({__VA_ARGS__}))

#endif
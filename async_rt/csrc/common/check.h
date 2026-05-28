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
#pragma once

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>

namespace internal {
inline void check_fail_print(std::stringstream &stream)
{}

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
    stream << "PARAMETER_CHECK fail in " << func << " at " << file << ":" << line << ", ";
    check_fail_print(stream, std::forward<Args>(args)...);
    stream << "\n";
    std::cerr << stream.str();
    exit(0);
}
}  // namespace internal

#define PARAMETER_CHECK(condition, ...)                                              \
    do {                                                                       \
        if (__builtin_expect(!(condition), 0)) {                               \
            ::internal::check_fail(__func__, __FILE__, __LINE__, __VA_ARGS__); \
        }                                                                      \
    } while (0)

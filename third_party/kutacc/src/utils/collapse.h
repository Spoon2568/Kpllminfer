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

#include "check.h"

namespace kutacc {

template <typename T>
inline T data_index_init(T offset)
{
    return offset;
}

template <typename T, typename... Args>
inline T data_index_init(T offset, T& x, const T& X, Args&&... args)
{
    offset = data_index_init(offset, std::forward<Args>(args)...);
    x = offset % X;
    return offset / X;
}

inline bool data_index_step()
{
    return true;
}

template <typename T, typename... Args>
inline bool data_index_step(T& x, const T& X, Args&&... args)
{
    if (data_index_step(std::forward<Args>(args)...)) {
        x = ((x + 1) == X) ? 0 : (x + 1);
        return x == 0;
    }
    return false;
}

template <typename F>
inline void collapse_for(int64_t start, int64_t end, const F &f)
{
    for (int64_t i = start; i < end; i ++) {
        f(i);
    }
}

template <typename F>
inline void collapse_for(int64_t start, int64_t end, int64_t n0, int64_t n1, const F &f)
{
    int64_t i0;
    int64_t i1;
    data_index_init(start, i0, n0, i1, n1);
    for ([[maybe_unused]] int64_t i = start; i < end; i ++) {
        f(i0, i1);
        data_index_step(i0, n0, i1, n1);
    }
}

template <typename F>
inline void collapse_for(int64_t start, int64_t end, int64_t n0, int64_t n1, int64_t n2, const F &f)
{
    int64_t i0;
    int64_t i1;
    int64_t i2;
    data_index_init(start, i0, n0, i1, n1, i2, n2);
    for ([[maybe_unused]] int64_t i = start; i < end; i ++) {
        f(i0, i1, i2);
        data_index_step(i0, n0, i1, n1, i2, n2);
    }
}

template <typename F>
inline void collapse_for(int64_t start, int64_t end, int64_t n0, int64_t n1, int64_t n2, int64_t n3, const F &f)
{
    int64_t i0;
    int64_t i1;
    int64_t i2;
    int64_t i3;
    data_index_init(start, i0, n0, i1, n1, i2, n2, i3, n3);
    for ([[maybe_unused]] int64_t i = start; i < end; i ++) {
        f(i0, i1, i2, i3);
        data_index_step(i0, n0, i1, n1, i2, n2, i3, n3);
    }
}

}   // namespace kutacc

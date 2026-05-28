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

#include <cstdint>
#include <algorithm>
#include <kutacc.h>

#include "check.h"

namespace utils {
inline int64_t get_thread_num()
{
    return kutacc::get_thread_id();
}

inline int64_t get_num_threads()
{
    return kutacc::get_thread_num();
}

inline void parallel_for(int64_t begin, int64_t end, int64_t grain_size, const std::function<void(int64_t, int64_t)> &f)
{
    kutacc::parallel_for(begin, end, grain_size, f);
}

inline void barrier()
{
    kutacc::parallel_region_barrier();
}
}  // namespace utils

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
#ifndef KMEMORY_H
#define KMEMORY_H

#include <cstdint>
#include <memory>
#include <malloc.h>
#include "kpmalloc.h"
#include "check.h"

namespace kutacc {
template <typename T>
struct KpMallocDeleter {
    void operator()(T *ptr) const
    {
        kp_free(ptr);
    }
};

template <typename T>
inline std::unique_ptr<T[], KpMallocDeleter<T> > alloc(int64_t size)
{
    void *ptr;
    size_t real_size = static_cast<size_t>(size);
    kp_posix_memalign(&ptr, 64, real_size * sizeof(T));
    return std::unique_ptr<T[], KpMallocDeleter<T>>(static_cast<T*>(ptr));
}
}   // namespace kutacc

#endif
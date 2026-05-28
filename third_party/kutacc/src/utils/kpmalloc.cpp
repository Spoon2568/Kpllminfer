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
#include "kpmalloc.h"
#include <dlfcn.h>
#include <cstdlib>
#ifdef USE_ON_PACKAGE_MEMORY
#include <memkind.h>
#endif

int kp_posix_memalign(void **memptr, size_t alignment, size_t size)
{
#ifdef USE_ON_PACKAGE_MEMORY
    return memkind_posix_memalign(MEMKIND_HBW_HUGETLB, memptr, alignment, size);
#else
    return posix_memalign(memptr, alignment, size);
#endif
}

void kp_free(void *ptr)
{
    if (ptr == nullptr) {
        return;
    }
#ifdef USE_ON_PACKAGE_MEMORY
    memkind_free(MEMKIND_HBW_HUGETLB, ptr);
    ptr = nullptr;
    return;
#else
    free(ptr);
    ptr = nullptr;
    return;
#endif
}

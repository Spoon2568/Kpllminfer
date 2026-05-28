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

#include <memory>
#include <numa.h>
#include <numaif.h>
#include <unistd.h>
#include <sys/mman.h>

#include "check.h"

template <bool USE_ON_PACKAGE_MEMORY = true>
struct PersistentBuffer {
    static constexpr int USE_MMAP = true;
    static constexpr int USE_HUGETLB = true;
    static constexpr int MMAP_FLAG = USE_HUGETLB ? MAP_HUGETLB : 0;
    void *g_buf{nullptr};
    size_t g_bufsize{0};
    size_t page_size = sysconf(_SC_PAGESIZE);
    size_t huge_page_size = 2 * 1024 * 1024;

    ~PersistentBuffer()
    {
        if (g_buf == nullptr) {
            return;
        }

        if (USE_MMAP) {
            munmap(g_buf, g_bufsize);
        } else {
            free(g_buf);
        }
    }

    inline void *malloc_buffer(size_t size)
    {
        if (USE_HUGETLB) {
            size = (size + huge_page_size - 1) / huge_page_size * huge_page_size;
        } else {
            size = (size + page_size - 1) / page_size * page_size;
        }
        if (size <= g_bufsize) {
            return g_buf;
        }
        if (USE_MMAP) {
            if (g_buf != nullptr) {
                munmap(g_buf, g_bufsize);
            }
            g_buf = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MMAP_FLAG, -1, 0);
            KUTACC_CHECK(g_buf != MAP_FAILED, "");
            
            int cpu = sched_getcpu();
            KUTACC_CHECK(cpu >= 0, "");
            int rnode = numa_node_of_cpu(cpu) + (USE_ON_PACKAGE_MEMORY ? 16 : 0);
            unsigned long mask = 1UL << rnode;
            int success = mbind(g_buf, size, MPOL_BIND, &mask, sizeof(mask) * 8, MPOL_MF_STRICT);
            KUTACC_CHECK(success == 0, "");
        } else {
            if (g_buf != nullptr) {
                free(g_buf);
            }
            g_buf = malloc(size);
            KUTACC_CHECK(g_buf != nullptr, "");
        }
        g_bufsize = size;
        return g_buf;
    }

    inline void free_buffer()
    {
        if (g_buf == nullptr) {
            return;
        }

        if (USE_MMAP) {
            munmap(g_buf, g_bufsize);
        } else {
            free(g_buf);
        }
        g_buf = nullptr;
        g_bufsize = 0;
    }
};
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
#include "kutacc.h"
#include "check.h"

#if defined(USE_OMP_PARALLEL)
#include <omp.h>
#elif defined(USE_KSPIN_PARALLEL)
#include "kspin/kspin.h"
#elif defined(USE_KUPL_PARALLEL)
#include "kupl.h"
#else
#error "invalid parallel backend"
#endif

namespace kutacc {

namespace internal {

inline int64_t divup(int64_t x, int64_t y)
{
    return (x + y - 1) / y;
}

} // namespace internal

int64_t get_thread_num()
{
#if defined(USE_OMP_PARALLEL)
    return omp_get_max_threads();
#elif defined(USE_KSPIN_PARALLEL)
    return kspin_get_max_threads();
#elif defined(USE_KUPL_PARALLEL)
    return kupl_get_num_executors();
#endif
}

int64_t get_thread_id()
{
#if defined(USE_OMP_PARALLEL)
    return omp_get_thread_num();
#elif defined(USE_KSPIN_PARALLEL)
    return kspin_get_thread_num();
#elif defined(USE_KUPL_PARALLEL)
    return kupl_get_executor_num();
#endif
}

#if defined(USE_KUPL_PARALLEL)
struct kupl_parallel_info {
    int exes[38];
    kupl_egroup_h eg = nullptr;

    kupl_parallel_info()
    {
        int num_threads = static_cast<int>(get_thread_num());
        for (int i = 0; i < num_threads; i++) {
            exes[i] = i;
        }
        eg = kupl_egroup_create(exes, num_threads);
    }

    ~kupl_parallel_info()
    {
        kupl_egroup_destroy(eg);
    }
};

static kupl_parallel_info info;

struct func_args {
    int64_t begin;
    int64_t end;
    int64_t chunk_size;
    const std::function<void(int64_t, int64_t)> &f;
};

static void parallel_for_kernel(kupl_nd_range_t *nd_range, void *args, int tid, int tnum)
{
    auto data = (func_args *) args;
    int64_t begin = data->begin;
    int64_t end = data->end;
    int64_t chunk_size = data->chunk_size;
    const std::function<void(int64_t, int64_t)> &f = data->f;

    int64_t begin_tid = begin + tid * chunk_size;
    if (begin_tid < end) {
        f(begin_tid, std::min(end, begin_tid + chunk_size));
    }
}
#endif

void parallel_for(int64_t begin, int64_t end, int64_t grain_size, const std::function<void(int64_t, int64_t)> &f)
{
    KUTACC_CHECK(grain_size > 0, "grain_size invalid: ", grain_size);
    if (begin >= end) {
        return;
    }
    int64_t num_threads = std::min(get_thread_num(), internal::divup(end - begin, grain_size));
    int64_t chunk_size = internal::divup(end - begin, num_threads);
    if (num_threads == 1) {
        f(begin, end);
    } else {
#if defined(USE_OMP_PARALLEL)
#pragma omp parallel
        {
            int64_t tid = get_thread_id();
            int64_t begin_tid = begin + tid * chunk_size;
            if (begin_tid < end) {
                f(begin_tid, std::min(end, chunk_size + begin_tid));
            }
        }
#elif defined(USE_KSPIN_PARALLEL)
        struct Task {
            int64_t begin;
            int64_t end;
            int64_t chunk_size;
            const std::function<void(int64_t, int64_t)> &f;
            static void call(void *_task)
            {
                auto task = static_cast<Task *>(_task);
                int64_t tid = get_thread_id();
                int64_t beginTid = task->begin + tid * task->chunk_size;
                if (beginTid < task->end) {
                    task->f(beginTid, std::min(task->end, task->chunk_size + beginTid));
                }
            }
        };
        Task task{begin, end, chunk_size, f};
        kspin_run_with_pool(Task::call, &task);
#elif defined(USE_KUPL_PARALLEL)
        kupl_parallel_for_desc_t desc = {
            .field_mask = KUPL_PARALLEL_FOR_DESC_FIELD_DEFAULT,
            .range = NULL,
            .egroup = NULL,
            .concurrency = static_cast<int>(num_threads),
            .policy = KUPL_LOOP_POLICY_STATIC
        };
        func_args args = {begin, end, chunk_size, f};
        kupl_parallel_for(&desc, parallel_for_kernel, &args);
#endif
    }
}

void parallel_region_barrier()
{
#if defined(USE_OMP_PARALLEL)
#pragma omp barrier
#elif defined(USE_KSPIN_PARALLEL)
    kspin_barrier();
#elif defined(USE_KUPL_PARALLEL)
    kupl_egroup_barrier(info.eg);
#endif
}

} // namespace kutacc
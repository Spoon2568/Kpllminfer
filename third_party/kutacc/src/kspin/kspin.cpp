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
#include <algorithm>
#include <atomic>
#include <iostream>
#include <unistd.h>
#include <cstdlib>
#include <thread>
#include <functional>
#include <vector>

#include "kspin.hpp"
#include "kspin.h"

namespace kspin {
namespace {
struct Worker;
thread_local int num_threads = -1;
thread_local int thread_num = -1;
thread_local bool in_parallel_region = false;
thread_local Worker *current_worker = nullptr;
int total_threads = 0;
int used_threads = 0;

void setaffinity(int cpuId)
{
    cpu_set_t cpus;
    CPU_ZERO(&cpus);
    CPU_SET(cpuId, &cpus);
    sched_setaffinity(0, sizeof(cpu_set_t), &cpus);
}

struct Worker {
    alignas(128) std::atomic<uint32_t> syncState{0};
    Worker *parent{nullptr};
    std::vector<Worker *> children;
    uint32_t version{0};
    int cpuId = -1;
    int localId = -1;
    int globalId = -1;
    std::thread thread;
    const std::function<void()> *task{nullptr};
    bool stop{false};

    void topDown()
    {
        if (parent != nullptr) {
            while (version != parent->syncState.load(std::memory_order_acquire))
                ;
        }
        if (!children.empty()) {
            syncState.store(version, std::memory_order_release);
        }
    }

    void bottomUp()
    {
        for (auto child : children) {
            while (version != child->syncState.load(std::memory_order_acquire))
                ;
        }
        syncState.store(version, std::memory_order_release);
    }

    void barrier()
    {
        ++version;
        bottomUp();
        ++version;
        topDown();
    }

    void run()
    {
        ++version;
        topDown();
        in_parallel_region = true;
        auto task_ = parent ? parent->task : task;
        (*task_)();
        in_parallel_region = false;
        ++version;
        bottomUp();
    }

    void start(const std::function<void()> &group_func = nullptr)
    {
        // group_func 只用于扩展线程组的第一个线程
        if (globalId == 0) {
            current_worker = this;
            thread_num = current_worker->localId;
            num_threads = current_worker->children.size() + 1;
            setaffinity(cpuId);
        } else {
            thread = std::thread([this, group_func] {
                setaffinity(cpuId);
                current_worker = this;
                thread_num = current_worker->localId;
                num_threads = (parent ? parent : current_worker)->children.size() + 1;
                children.shrink_to_fit();
                if (parent) {
                    while (!stop) {
                        run();
                    }
                } else {
                    // 扩展线程组的第一个线程
                    group_func();
                    // 停止扩展线程组
                    std::function<void()> f = [] { current_worker->stop = true; };
                    task = &f;
                    run();
                }
            });
        }
    }
};

struct ThreadPool {
    std::vector<std::unique_ptr<Worker>> workers;

    ThreadPool()
    {
        // 获取总线程数 total_threads，主线程数 main_threads
        cpu_set_t g_cpus;
        CPU_ZERO(&g_cpus);
        if (sched_getaffinity(0, sizeof(cpu_set_t), &g_cpus) == -1) {
            perror("sched_getaffinity");
            exit(EXIT_FAILURE);
        }
        int nrcpus = sysconf(_SC_NPROCESSORS_CONF);
        total_threads = 0;
        for (int cpuId = 0; cpuId < nrcpus; ++cpuId) {
            if (CPU_ISSET(cpuId, &g_cpus)) {
                total_threads++;
            }
        }
        auto env_main_threads = std::getenv("KSPIN_NUM_TRHEADS");
        int main_threads = env_main_threads ? std::stoi(env_main_threads) : total_threads;
        if (main_threads > total_threads) {
            fprintf(stderr, "main_threads too large.\n");
            exit(0);
        }

        // 初始化所有 worker 和主线程组
        int parentId = 0;
        for (int cpuId = 0; cpuId < nrcpus; ++cpuId) {
            if (CPU_ISSET(cpuId, &g_cpus)) {
                auto globalId = workers.size();
                auto worker = std::make_unique<Worker>();
                worker->cpuId = cpuId;
                worker->globalId = globalId;
                if (globalId < main_threads) {
                    worker->localId = globalId - parentId;
                    if (globalId != parentId) {
                        workers[parentId]->children.push_back(worker.get());
                        worker->parent = workers[parentId].get();
                    }
                }
                workers.push_back(std::move(worker));
            }
        }
        // 启动主线程组
        for (int globalId = 0; globalId < main_threads; globalId++) {
            workers[globalId]->start();
        }
        used_threads = main_threads;
    }

    void createGroupAndRun(int group_size, const std::function<void()> &f = nullptr)
    {
        if (used_threads + group_size > total_threads) {
            fprintf(stderr, "Failed to create thread group: too many threads.\n");
            exit(0);
        }
        int parentId = used_threads;
        // 初始化扩展线程组
        for (int globalId = parentId; globalId < parentId + group_size; globalId++) {
            workers[globalId]->localId = globalId - parentId;
            if (globalId != parentId) {
                workers[parentId]->children.push_back(workers[globalId].get());
                workers[globalId]->parent = workers[parentId].get();
            }
        }
        // 启动扩展线程组
        for (int globalId = parentId; globalId < parentId + group_size; globalId++) {
            workers[globalId]->start(globalId == parentId ? f : nullptr);
        }
        used_threads += group_size;
    }

    void run(const std::function<void()> &task)
    {
        if (__builtin_expect(in_parallel_region, 0)) {
            fprintf(stderr, "Subthreads do not support calling thread pool functions.\n");
            exit(0);
        }
        current_worker->task = &task;
        current_worker->run();
    }

    ~ThreadPool()
    {
        // 停止主线程组
        run([] { current_worker->stop = true; });
        for (int id = 1; id < used_threads; id++) {
            workers[id]->thread.join();
        }
    }
};

ThreadPool pool{};
}  // namespace

bool in_parallel()
{
    return in_parallel_region;
}

int get_thread_num()
{
    return thread_num;
}

int get_max_threads()
{
    return num_threads;
}

void run_with_pool(const std::function<void()> &f)
{
    pool.run(f);
}

void parallel_for(const int64_t start, const int64_t end, const std::function<void(int64_t, int64_t)> &f)
{
    int64_t thread_num = get_max_threads();
    int64_t item_per_thread = (end - start + thread_num - 1) / thread_num;
    std::function<void()> _func = [&]() {
        int64_t id = get_thread_num();
        int64_t task_start = start + id * item_per_thread;
        int64_t task_end = std::min(start + (id + 1) * item_per_thread, end);
        f(task_start, task_end);
    };
    pool.run(_func);
}

void barrier()
{
    current_worker->barrier();
}

void create_group_and_run(int group_size, const std::function<void()> &f)
{
    pool.createGroupAndRun(group_size, f);
}

extern "C" bool kspin_in_parallel()
{
    return in_parallel_region;
}

extern "C" int kspin_get_thread_num()
{
    return thread_num;
}

extern "C" int kspin_get_max_threads()
{
    return num_threads;
}

extern "C" void kspin_run_with_pool(void (*f)(void *), void *context)
{
    std::function<void()> _func = [&]() { f(context); };
    pool.run(_func);
}

extern "C" void kspin_barrier()
{
    current_worker->barrier();
}

extern "C" void kspin_create_group_and_run(int group_size, void (*f)(void *), void *context)
{
    std::function<void()> _func = [&]() { f(context); };
    pool.createGroupAndRun(group_size, _func);
}
}  // namespace kspin

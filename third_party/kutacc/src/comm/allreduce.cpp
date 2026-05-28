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
#include <atomic>
#include <arm_sve.h>
#include <functional>
#include <kupl.h>

#include "kutacc.h"
#include "utils/check.h"

#define kupl_aarch64_dmb(_op) asm volatile("dmb " #_op ::: "memory")

#define kupl_memory_cpu_load_fence() kupl_aarch64_dmb(ishld)

#define wait_fence(ptr)                                                                         \
    do {                                                                                        \
        int8_t flag = 0;                                                                        \
        do {                                                                                    \
            kupl_memory_cpu_load_fence();                                                       \
            flag = *(ptr);                                                                      \
        } while (flag != 1);                                                                    \
        *(ptr) = 0;                                                                             \
    } while (0)

#define set_fence(ptr)                                                                          \
    do {                                                                                        \
        std::atomic_thread_fence(std::memory_order_seq_cst);                                    \
        *(ptr) = 1;                                                                             \
    } while (0)


namespace kutacc {

namespace {

template <size_t k1, size_t k2>
inline void reduce(bfloat16_t **a, bfloat16_t **b, size_t n)
{
    static_assert(k1 > 0);
    svbfloat16_t sve_zero = svdup_bf16(0);
    svbool_t ptrue32 = svptrue_b32();
    for (size_t i = 0; i < n; i += 32) {
        svbool_t pred = svwhilelt_b16(i, n);
        svbfloat16_t sve = svld1(pred, a[0] + i);
        svfloat32_t sve0 = svreinterpret_f32(svzip1(sve_zero, sve));
        svfloat32_t sve1 = svreinterpret_f32(svzip2(sve_zero, sve));
        for (size_t k = 1; k < k1; ++k) {
            sve = svld1(pred, a[k] + i);
            sve0 = svadd_m(ptrue32, sve0, svreinterpret_f32(svzip1(sve_zero, sve)));
            sve1 = svadd_m(ptrue32, sve1, svreinterpret_f32(svzip2(sve_zero, sve)));
        }
        sve = svuzp1(svcvt_bf16_x(ptrue32, sve0), svcvt_bf16_x(ptrue32, sve1));
        for (size_t k = 0; k < k2; ++k) {
            svst1(pred, b[k] + i, sve);
        }
    }
}

template <size_t k>
inline void copy(bfloat16_t **a, const bfloat16_t *b, size_t n)
{
    for (size_t i = 0; i < n; i += 32) {
        svbool_t pred = svwhilelt_b16(i, n);
        auto sve = svld1(pred, b + i);
        for (size_t j = 0; j < k; ++j) {
            svst1(pred, a[j] + i, sve);
        }
    }
}

size_t num_threads;
size_t max_buffer_size;
size_t extra_buffer_size;
size_t fence_buffer_size;

constexpr size_t MAX_NUM_FENCES = 6;
constexpr size_t ALIGNMENT = 128;

bfloat16_t *local_extra_buffer;
bfloat16_t *inter_die_extra_buffer;
bfloat16_t *inter_socket_extra_buffer;

int8_t *local_fence_buffer;
int8_t *inter_die_fence_buffer;
int8_t *inter_socket_fence_buffer;

kupl_event_h *sdma_events;

}

#define USE_SDMA 0

void shm_allreduce_get_metadata(size_t max_num_elements, size_t &extra_size, bool &use_sdma)
{
    num_threads = kutacc::get_thread_num();
    max_buffer_size = max_num_elements * sizeof(bfloat16_t);
    extra_buffer_size = max_buffer_size / 8 * 2;
    fence_buffer_size = num_threads * MAX_NUM_FENCES * ALIGNMENT;
    extra_size = extra_buffer_size + fence_buffer_size;
    use_sdma = USE_SDMA;
}

void shm_allreduce_init(int rank, size_t max_num_elements, bfloat16_t **intra_node_extra_buffers)
{
    KUTACC_CHECK(max_buffer_size == max_num_elements * sizeof(bfloat16_t), "");

    local_extra_buffer = intra_node_extra_buffers[rank];
    inter_die_extra_buffer = intra_node_extra_buffers[rank ^ 4];
    inter_socket_extra_buffer = intra_node_extra_buffers[rank ^ 8];

    local_fence_buffer = (int8_t *)local_extra_buffer + extra_buffer_size;
    inter_die_fence_buffer = (int8_t *)inter_die_extra_buffer + extra_buffer_size;
    inter_socket_fence_buffer = (int8_t *)inter_socket_extra_buffer + extra_buffer_size;

    memset(local_fence_buffer, 0, fence_buffer_size);

    sdma_events = (kupl_event_h *)malloc(sizeof(kupl_event_h) * num_threads);
    for (int i = 0; i < num_threads; ++i) {
        sdma_events[i] = kupl_event_create();
    }
}

void shm_allreduce_finalize()
{
    for (int i = 0; i < num_threads; ++i) {
        kupl_event_destroy(sdma_events[i]);
    }
    free(sdma_events);
}

void shm_allreduce(int rank, int num_elements, bfloat16_t **intra_node_buffers, kupl_shm_win_h &kupl_win_intra_die,
    kupl_shm_win_h &kupl_win_intra_socket, int world_size)
{
    int &N = num_elements;

    KUTACC_CHECK(N * sizeof(bfloat16_t) <= max_buffer_size, "");
    KUTACC_CHECK(N % (8 * num_threads) == 0, "");

    bfloat16_t *local_buffer = intra_node_buffers[rank];
    bfloat16_t *intra_die_buffer[4];
    int local_rank = rank % 4;
    for (int i = 0; i < 4; ++i) {
        intra_die_buffer[i] = intra_node_buffers[rank - local_rank + i];
    }

    static int flip_flag = 0;
    int fence_id_offset = MAX_NUM_FENCES / 2 * flip_flag;
    int extra_buffer_offset = extra_buffer_size / 2 / sizeof(bfloat16_t) * flip_flag;
    local_extra_buffer += extra_buffer_offset;
    inter_socket_extra_buffer += extra_buffer_offset;
    flip_flag ^= 1;

    int n = N / (8 * num_threads);

    kupl_shm_fence(kupl_win_intra_die);

    if (world_size == 4) {
        kutacc::parallel_for(0, num_threads, 1, [&](int64_t start, int64_t end) {
            int tid = kutacc::get_thread_id();
            int o = tid * 2 + (rank & 4) / 4;
            int e = o ^ 1;
            bfloat16_t *ptr0[4];
            bfloat16_t *ptr1[4];     // 指向 4 个 rank 的 o 块
            bfloat16_t *ptr3[4];     // 用于广播的循环移位指针

            for (int i = 0; i < 4; ++i) {
                ptr0[i] = intra_die_buffer[i] + (N / 4) * local_rank + n * e;
                ptr1[i] = intra_die_buffer[i] + (N / 4) * local_rank + n * o;
            }
            for (int i = 0; i < 4; ++i) {
                ptr3[i] = intra_die_buffer[(local_rank + i) % 4] + (N / 4) * local_rank + n * e;
            }

            reduce<4, 1>(ptr0, (bfloat16_t *[]){local_buffer + (N / 4) * local_rank + n * e}, n);
            reduce<4, 1>(ptr1, (bfloat16_t *[]){local_extra_buffer + n * tid}, n);
            copy<4>(ptr1, local_extra_buffer + n * tid, n);
            copy<4>(ptr3, local_buffer + (N / 4) * local_rank + n * e, n);

            });
    }
    else if (world_size == 8) {
        bfloat16_t *inter_die_buffer = intra_node_buffers[rank ^ 4];
        kutacc::parallel_for(0, num_threads, 1, [&](int64_t start, int64_t end) {
            int tid = kutacc::get_thread_id();
            int o = tid * 2 + (rank & 4) / 4;
            int e = o ^ 1;
            bfloat16_t *ptr0[4];
            bfloat16_t *ptr2[5];     // 指向 4 个 rank 的 o 块
            bfloat16_t *ptr3[4];     // 用于广播的循环移位指针

            ptr2[4] = inter_die_buffer + (N / 4) * local_rank + n * o;
            for (int i = 0; i < 4; ++i) {
                ptr0[i] = intra_die_buffer[i] + (N / 4) * local_rank + n * e;
                ptr2[i] = intra_die_buffer[i] + (N / 4) * local_rank + n * o;
            }
            for (int i = 0; i < 4; ++i) {
                ptr3[i] = intra_die_buffer[(local_rank + i) % 4] + (N / 4) * local_rank + n * e;
            }
            int fence_offset = tid * MAX_NUM_FENCES + fence_id_offset;
            
            reduce<4, 1>(ptr0, (bfloat16_t *[]){local_buffer + (N / 4) * local_rank + n * e}, n);
            set_fence(inter_die_fence_buffer + fence_offset * ALIGNMENT);
            wait_fence(local_fence_buffer + fence_offset * ALIGNMENT);
            reduce<5, 1>(ptr2, (bfloat16_t *[]){local_extra_buffer + n * tid}, n);

            reduce<1, 5>((bfloat16_t *[]){local_extra_buffer + n * tid}, ptr2, n);
            set_fence(inter_die_fence_buffer + (fence_offset + 2) * ALIGNMENT);
            wait_fence(local_fence_buffer + (fence_offset + 2) * ALIGNMENT);
            copy<3>(ptr3 + 1, local_buffer + (N / 4) * local_rank + n * e, n);

        });
    }
    else if (world_size == 16) {
        bfloat16_t *inter_die_buffer = intra_node_buffers[rank ^ 4];
        bfloat16_t *inter_socket_buffer = intra_node_buffers[rank ^ 8];

        kutacc::parallel_for(0, num_threads, 1, [&](int64_t start, int64_t end) {
            int tid = kutacc::get_thread_id();
            int o = tid * 2 + (rank & 4) / 4;
            int e = o ^ 1;
            bfloat16_t *ptr0[4];
            for (int i = 0; i < 4; ++i) {
                ptr0[i] = intra_die_buffer[i] + (N / 4) * local_rank + n * e;
            }
            bfloat16_t *ptr2[5];
            ptr2[4] = inter_die_buffer + (N / 4) * local_rank + n * o;
            for (int i = 0; i < 4; ++i) {
                ptr2[i] = intra_die_buffer[i] + (N / 4) * local_rank + n * o;
            }
            bfloat16_t *ptr3[4];
            for (int i = 0; i < 4; ++i) {
                ptr3[i] = intra_die_buffer[(local_rank + i) % 4] + (N / 4) * local_rank + n * e;
            }
            int fence_offset = (tid * MAX_NUM_FENCES + fence_id_offset);

            reduce<4, 1>(ptr0, (bfloat16_t *[]){local_buffer + (N / 4) * local_rank + n * e}, n);

            set_fence(inter_die_fence_buffer + fence_offset * ALIGNMENT);
            wait_fence(local_fence_buffer + fence_offset * ALIGNMENT);

            reduce<5, 1>(ptr2, (bfloat16_t *[]){local_extra_buffer + n * tid}, n);

            set_fence(inter_socket_fence_buffer + (fence_offset + 1) * ALIGNMENT);
            wait_fence(local_fence_buffer + (fence_offset + 1) * ALIGNMENT);

    #if USE_SDMA
            kupl_memcpy_async(local_buffer + (N / 4) * local_rank + n * o, inter_socket_extra_buffer + n * tid,
                            n * sizeof(bfloat16_t), nullptr, sdma_event[tid]);
            kupl_event_wait(sdma_event[tid]);
            reduce<2, 4>(
                (bfloat16_t *[]){local_buffer + (N / 4) * local_rank + n * o, local_extra_buffer + n * tid}, ptr2, n);
    #else
            reduce<2, 5>((bfloat16_t *[]){inter_socket_extra_buffer + n * tid, local_extra_buffer + n * tid}, ptr2, n);
    #endif

            set_fence(inter_die_fence_buffer + (fence_offset + 2) * ALIGNMENT);
            wait_fence(local_fence_buffer + (fence_offset + 2) * ALIGNMENT);

    #if USE_SDMA
            copy<4>(ptr3, inter_die_buffer + (N / 4) * local_rank + n * e, n);
    #else
            copy<3>(ptr3 + 1, local_buffer + (N / 4) * local_rank + n * e, n);
    #endif
        });
    }
    #if USE_SDMA
        kupl_shm_fence(kupl_win_intra_socket);
    #else
        kupl_shm_fence(kupl_win_intra_die);
    #endif

        local_extra_buffer -= extra_buffer_offset;
        inter_socket_extra_buffer -= extra_buffer_offset;
}

} // namespace kutacc
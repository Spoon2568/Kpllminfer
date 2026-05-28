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
#include "utils/collapse.h"
#include "utils/check.h"

#include <cstdint>
#include <atomic>
#include <functional>
#include <kupl.h>
#include <iostream>
#include <cstring>
#include <arm_sve.h>

namespace kutacc {

namespace {

template <typename scalar_t>
void copy_buffer(scalar_t *dst, const scalar_t *src, int64_t numel, int64_t grain_size = 8192)
{
    int64_t real_grain_size = grain_size / sizeof(scalar_t);
    if (real_grain_size == 0) {
        real_grain_size = 1;
    }
    kutacc::parallel_for(0, numel, real_grain_size,
        [&](int64_t start, int64_t end) { memcpy(dst + start, src + start, (end - start) * sizeof(scalar_t)); });
}

template <typename scalar_t>
void shm_allgather_comm8(int64_t batch, const scalar_t *sendbuf, int64_t sendcount, scalar_t *recvbuf,
                         int64_t recvcount, int64_t rank, uint8_t **buffers, int64_t buffer_size,
                         const std::function<void()> &barrier)
{
    KUTACC_CHECK(sendbuf == (scalar_t *)buffers[rank], "no implement");
    const int64_t socket_size = 8;
    KUTACC_CHECK(recvcount == sendcount * socket_size, "allgather");
    const int64_t thread_nums = kutacc::get_thread_num();
    const int64_t total_batch = batch * socket_size;
    const int64_t total_thread_batch = total_batch / thread_nums;
    const int64_t start_rank = (rank / socket_size) * socket_size;
    barrier();
    // step 1: copy out
    kutacc::parallel_for(0, thread_nums, 1, [&](int64_t start, int64_t end) {
        const int64_t thread_id = start;
        const int64_t bs_start = thread_id * total_thread_batch;
        const int64_t bs_end = std::min(total_batch, bs_start + total_thread_batch);
        const int64_t bs_num = bs_end - bs_start;
        int64_t recv_bs_offset = bs_start % batch;
        int64_t recv_w_offset = bs_start / batch * sendcount;
        int64_t cur_rank = bs_start / batch + start_rank;
        for (int64_t i = 0; i < bs_num; i++) {
            memcpy(recvbuf + recv_bs_offset * (sendcount * socket_size) + recv_w_offset,
                ((scalar_t *)buffers[cur_rank]) + recv_bs_offset * sendcount, sendcount * sizeof(scalar_t));
            recv_bs_offset++;
            if (recv_bs_offset == batch) {
                recv_w_offset += sendcount;
                recv_bs_offset = 0;
                cur_rank++;
            }
        }
    });
    barrier();
}

template <typename scalar_t>
void shm_allgather_comm8_hierarchical(int64_t batch, const scalar_t *sendbuf, int64_t sendcount, scalar_t *recvbuf,
                                      int64_t recvcount, int64_t rank, uint8_t **buffers, int64_t buffer_size,
                                      const std::function<void()> &barrier)
{
    int64_t half_size = buffer_size / 2 / sizeof(scalar_t);
    int64_t socket_size = 8;
    int64_t die_size = 4;
    KUTACC_CHECK(sendcount * socket_size == recvcount, sendcount, " ", recvcount);
    int64_t part_numel = batch * sendcount;
    KUTACC_CHECK(half_size >= part_numel, "allgather sendcount too large");
    // Match comm16 hierarchical: sendbuf may be a separate tensor; stage into this rank's shm slot first.
    if (sendbuf != (scalar_t *)buffers[rank]) {
        copy_buffer((scalar_t *)buffers[rank], sendbuf, part_numel);
        barrier();
    }
    int64_t socket_root = rank / socket_size * socket_size;
    int64_t die_root = rank / die_size * die_size;
    // step 1: inter-die gather
    {
        int64_t remote_rank = rank ^ die_size;
        auto src = (scalar_t *)buffers[rank];
        auto dst = (scalar_t *)buffers[remote_rank] + half_size;
        kutacc::parallel_for(0, part_numel, 8192 / sizeof(scalar_t),
            [&](int64_t start, int64_t end) {
                memcpy(dst + start, src + start, (end - start) * sizeof(scalar_t)); });
        barrier();
    }
    // step 2: intra-die gather
    {
        kutacc::parallel_for(0, socket_size * batch, 1, [&](int64_t start, int64_t end) {
            kutacc::collapse_for(start, end, socket_size, batch, [&](int64_t remote_rank_off, int64_t bi) {
                int64_t remote_rank = socket_root + remote_rank_off;
                bool same_die = (rank / die_size == remote_rank / die_size);
                auto src = (scalar_t *)buffers[die_root + remote_rank % die_size] + (same_die ? 0 : half_size) +
                        bi * sendcount;
                auto dst = recvbuf + bi * recvcount + remote_rank_off * sendcount;
                memcpy(dst, src, sendcount * sizeof(scalar_t));
            });
        });
        barrier();
    }
}

template <typename scalar_t>
void shm_allgather_comm16(int64_t batch, const scalar_t *sendbuf, int64_t sendcount, scalar_t *recvbuf,
    int64_t recvcount, int64_t rank, uint8_t **buffers, int64_t buffer_size, const std::function<void()> &barrier)
{
    constexpr int world_size = 16;
    KUTACC_CHECK(sendbuf == (scalar_t *)buffers[rank], "no implement");
    KUTACC_CHECK(recvcount == sendcount * world_size, "allgather recvcount");
    const int64_t thread_nums = kutacc::get_thread_num();
    const int64_t total_batch = batch * world_size;
    const int64_t total_thread_batch = total_batch / thread_nums;
    barrier();
    kutacc::parallel_for(0, thread_nums, 1, [&](int64_t start, int64_t end) {
        const int64_t thread_id = start;
        const int64_t bs_start = thread_id * total_thread_batch;
        const int64_t bs_end = std::min(total_batch, bs_start + total_thread_batch);
        const int64_t bs_num = bs_end - bs_start;
        int64_t cur_rank = bs_start / batch;
        int64_t recv_bs_offset = bs_start % batch;
        int64_t recv_w_offset = cur_rank * sendcount;
        for (int64_t i = 0; i < bs_num; i++) {
            memcpy(recvbuf + recv_bs_offset * (sendcount * world_size) + recv_w_offset,
                ((scalar_t *)buffers[cur_rank]) + recv_bs_offset * sendcount, sendcount * sizeof(scalar_t));
            recv_bs_offset++;
            if (recv_bs_offset == batch) {
                recv_w_offset += sendcount;
                recv_bs_offset = 0;
                cur_rank++;
            }
        }
    });
    barrier();
}

template <typename scalar_t>
void shm_allgather_comm16_hierarchical(int64_t batch, const scalar_t *sendbuf, int64_t sendcount, scalar_t *recvbuf,
                                       int64_t recvcount, int64_t rank, uint8_t **buffers, int64_t buffer_size,
                                       const std::function<void()> &barrier)
{
    constexpr int world_size = 16;
    int64_t quarter_size = buffer_size / 4 / sizeof(scalar_t);
    int64_t count = sendcount;
    KUTACC_CHECK(sendbuf != (scalar_t *)buffers[rank], "no implement");
    KUTACC_CHECK(quarter_size >= count, "allgather sendcount too large");
    KUTACC_CHECK(recvcount == sendcount * world_size, "allgather recvcount");
    int64_t socket_size = world_size / 2;
    int64_t die_size = socket_size / 2;
    int64_t die_root = rank / die_size * die_size;
    int64_t batch_step = quarter_size / count;
    for (int64_t batch_start = 0; batch_start < batch; batch_start += batch_step) {
        int64_t batch_end = std::min(batch, batch_start + batch_step);
        int64_t chunk_numel = (batch_end - batch_start) * count;
        const scalar_t *send_ptr = sendbuf + batch_start * count;
        scalar_t *recv_ptr = recvbuf + batch_start * recvcount;
        // step 1 & 2: copy in & inter-socket gather
        {
            int64_t rank_offset = (rank / die_size) * quarter_size;
            int64_t remote_rank = (rank + socket_size) % world_size;
            kutacc::parallel_for(0, chunk_numel, 1, [&](int64_t start, int64_t end) {
                memcpy((scalar_t *)buffers[rank] + rank_offset + start, send_ptr + start,
                    (end - start) * sizeof(scalar_t));
                memcpy((scalar_t *)buffers[remote_rank] + rank_offset + start, send_ptr + start,
                    (end - start) * sizeof(scalar_t));
            });
            barrier();
        }
        // step 3: inter-die gather
        {
            int64_t local_rank = (rank + die_size) % socket_size + (rank >= socket_size ? socket_size : 0);
            int64_t local_offset = (1 - rank % socket_size / die_size) * quarter_size;
            copy_buffer((scalar_t *)buffers[rank] + local_offset, (scalar_t *)buffers[local_rank] + local_offset,
                chunk_numel, sizeof(scalar_t));
            copy_buffer((scalar_t *)buffers[rank] + local_offset + quarter_size * 2,
                (scalar_t *)buffers[local_rank] + local_offset + quarter_size * 2, chunk_numel, sizeof(scalar_t));
            barrier();
        }
        // step 4: copy out
        {
            if (count > 128)
                kutacc::parallel_for(0, world_size * batch_step, 1, [&](int64_t start, int64_t end) {
                    kutacc::collapse_for(start, end, world_size, batch_step, [&](int64_t remote_rank, int64_t bi) {
                        if (batch_start + bi < batch) {
                            memcpy(recv_ptr + bi * world_size * count + remote_rank * count,
                                (scalar_t *)buffers[die_root + remote_rank % die_size] +
                                    (remote_rank / die_size) * quarter_size + bi * count,
                                count * sizeof(scalar_t));
                        }
                    });
                });
            else
                kutacc::parallel_for(batch_start, batch_end, 1, [&](int64_t start, int64_t end) {
                    for (int64_t bi = start; bi < end; ++bi)
                        for (int64_t i = 0; i < world_size; ++i) {
                            int64_t now_rank = (rank + i) % world_size;
                            memcpy(recv_ptr + bi * world_size * count + now_rank * count,
                                (scalar_t *)buffers[die_root + now_rank % die_size] +
                                    (now_rank / die_size) * quarter_size + bi * count,
                                count * sizeof(scalar_t));
                        }
                });
            barrier();
        }
    }
}

}

template <typename scalar_t, int world_size, bool is_hierarchical>
void shm_allgather(int64_t batch, const scalar_t *sendbuf, int64_t sendcount, scalar_t *recvbuf,
                   int64_t recvcount, int64_t rank, uint8_t **buffers, int64_t buffer_size,
                   const std::function<void()> &barrier)
{
    if constexpr(world_size == 8) {
        if constexpr(is_hierarchical) {
            shm_allgather_comm8_hierarchical(
                batch, sendbuf, sendcount, recvbuf, recvcount, rank, buffers, buffer_size, barrier);
        } else {
            shm_allgather_comm8(batch, sendbuf, sendcount, recvbuf, recvcount, rank, buffers, buffer_size, barrier);
        }
    } else if constexpr(world_size == 16) {
        if constexpr(is_hierarchical) {
            shm_allgather_comm16_hierarchical(
                batch, sendbuf, sendcount, recvbuf, recvcount, rank, buffers, buffer_size, barrier);
        } else {
            shm_allgather_comm16(batch, sendbuf, sendcount, recvbuf, recvcount, rank, buffers, buffer_size, barrier);
        }
    } else {
        KUTACC_CHECK(false, "no implement");
    }
}

template void shm_allgather<bfloat16_t, 8, true>(int64_t, const bfloat16_t *, int64_t, bfloat16_t *, int64_t,
                                                 int64_t, uint8_t **, int64_t, const std::function<void()> &);
template void shm_allgather<bfloat16_t, 8, false>(int64_t, const bfloat16_t *, int64_t, bfloat16_t *, int64_t,
                                                  int64_t, uint8_t **, int64_t, const std::function<void()> &);
template void shm_allgather<bfloat16_t, 16, true>(int64_t, const bfloat16_t *, int64_t, bfloat16_t *, int64_t,
                                                  int64_t, uint8_t **, int64_t, const std::function<void()> &);
template void shm_allgather<bfloat16_t, 16, false>(int64_t, const bfloat16_t *, int64_t, bfloat16_t *, int64_t,
                                                   int64_t, uint8_t **, int64_t, const std::function<void()> &);
template void shm_allgather<uint8_t, 16, true>(int64_t, const uint8_t *, int64_t, uint8_t *, int64_t,
                                               int64_t, uint8_t **, int64_t, const std::function<void()> &);
template void shm_allgather<uint8_t, 16, false>(int64_t, const uint8_t *, int64_t, uint8_t *, int64_t,
                                                int64_t, uint8_t **, int64_t, const std::function<void()> &);
template void shm_allgather<uint8_t, 8, true>(int64_t, const uint8_t *, int64_t, uint8_t *, int64_t,
                                               int64_t, uint8_t **, int64_t, const std::function<void()> &);
template void shm_allgather<uint8_t, 8, false>(int64_t, const uint8_t *, int64_t, uint8_t *, int64_t,
                                                int64_t, uint8_t **, int64_t, const std::function<void()> &);

}  // namespace kutacc

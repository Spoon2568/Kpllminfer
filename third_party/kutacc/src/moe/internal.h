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
#include <kupl.h>
#include <vector>
#include <utility>
#include <memory>
#include <sys/time.h>
#include "core/kurmcl/kurmcl_impl.h"

#define kupl_aarch64_dmb(_op) asm volatile("dmb " #_op ::: "memory")
#define kupl_memory_cpu_load_fence() kupl_aarch64_dmb(ishld)
namespace kutacc {
    inline int num_ranks;
    inline int my_rank;
    inline int16_t *src_info;
    inline int16_t *catch_recv_src_info;
    inline bfloat16_t *tmpx_for_sum;
    inline int64_t num_local_experts;
    inline int64_t recv_src_info_size;
    inline int64_t recv_src_info_conut;
    inline int64_t recv_src_info_ep_bias;
    inline int64_t packed_recv_x_size;
    inline int64_t disbuf_size;
    inline void *disbuf_baseptr;

    inline int local_size, local_rank;
    inline bfloat16_t *combinedx_base_ptr;
    inline uint8_t *combinedx_meta_ptr;
    inline std::vector<bfloat16_t *> combinedx_group_ptr;
    inline std::vector<uint8_t *> combinedx_meta_group_ptr;
    inline std::vector<int> will_recv_from_rank;
    inline int dis_peer_nums = 0;
    inline int comb_peer_nums = 0;

    inline double dis_send_stage1 = 0;
    inline double dis_send_stage2 = 0;
    inline double dis_recv_stage1 = 0;
    inline double dis_start_time = 0;
    inline int dis_times = 0;
    inline double dis_thread_start_time = 0;
    inline double com_send_stage1 = 0;
    inline double com_send_stage2 = 0;
    inline double com_start_time = 0;
    inline double com_recv_stage1 = 0;
    inline double com_recv_stage2 = 0;
    inline double com_recv_stage3 = 0;
    inline int comb_times = 0;
    inline double com_thread_start_time = 0;

}
inline double get_clock_us()
{
    struct timeval clock;
    gettimeofday(&clock, NULL);
    return 1000000.0 * clock.tv_sec + clock.tv_usec;
}
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
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <arm_bf16.h>
#include <arm_neon.h>
#include <arm_sve.h>
#include <omp.h>

#include "internal.h"
#include "kutacc.h"

namespace kutacc {

buf_mr_info_t comb_token_buf_send, comb_token_buf_recv, allg_token_buf_send, allg_token_buf_recv;
buf_mr_info_t *comb_remote_buf;  // 封装rkey用于交换
buf_mr_info_t *allg_remote_buf;  // 封装rkey用于交换
std::vector<int> iovcont_per_rank_for_comb;
kurmcl_iov_t *iovlist_for_comb = NULL;
kurmcl_iov_t *iovlist_for_allgather = NULL;

std::atomic<int> counter(0);
int mynode_id = 0;
int node_nums = 0;
int64_t die_leader[16] = {4, 5, 6, 7, 0, 1, 2, 3, 12, 13, 14, 15, 8, 9, 10, 11};

void moe_combine_init(bfloat16_t *new_packed_recv_x_data, int64_t num_tokens, int16_t num_experts, int16_t num_topk,
    int16_t num_max_dispatch_tokens_per_rank, int16_t hidden, std::vector<bfloat16_t *>&& group_ptr,
    std::vector<uint8_t *>&& meta_group_ptr, int local_size_, int local_rank_, bfloat16_t *tmpx_for_sum_)
{
    tmpx_for_sum = tmpx_for_sum_;
    int64_t num_local_experts = num_experts / num_ranks;
    int64_t tmpx_for_sum_size = num_local_experts * num_ranks * num_max_dispatch_tokens_per_rank * hidden;
    mynode_id = my_rank / 16;
    node_nums = num_ranks / 16;
    int res;
    combinedx_group_ptr = std::move(group_ptr);
    combinedx_meta_group_ptr = std::move(meta_group_ptr);
    local_size = local_size_;
    local_rank = local_rank_;
    combinedx_base_ptr = combinedx_group_ptr[local_rank];
    combinedx_meta_ptr = combinedx_meta_group_ptr[local_rank];
    memset(combinedx_base_ptr, 0, num_tokens / node_nums * hidden * sizeof(bfloat16_t));
    memset(combinedx_meta_ptr, 0, 16);
    iovcont_per_rank_for_comb.resize(num_ranks);
    kurmcl_reg_mr(&comb_token_buf_send, new_packed_recv_x_data,
        num_local_experts * num_ranks * num_max_dispatch_tokens_per_rank * hidden * sizeof(int16_t),
        ds_conn_info);  // 注册发方token mr
    kurmcl_reg_mr(&comb_token_buf_recv, tmpx_for_sum, tmpx_for_sum_size * sizeof(int16_t),
        ds_conn_info);  // 注册发方meta mr
    comb_remote_buf = (buf_mr_info_t *)malloc(num_ranks * sizeof(buf_mr_info_t));
    kurmcl_exchange_mr_info(ds_conn_info, &comb_token_buf_recv, comb_remote_buf);  // allgather交换，获取所有其他进程的接收buffer
    iovlist_for_comb =
        (kurmcl_iov_t *)malloc(num_ranks * num_max_dispatch_tokens_per_rank * sizeof(kurmcl_iov_t));  // 申请iovlist
    kurmcl_reg_mr(&allg_token_buf_send, combinedx_base_ptr,
        num_tokens / node_nums * hidden * sizeof(bfloat16_t), ds_conn_info);
    kurmcl_reg_mr(&allg_token_buf_recv, combinedx_base_ptr,
        num_tokens / node_nums * hidden * sizeof(bfloat16_t), ds_conn_info);
    allg_remote_buf = (buf_mr_info_t*)malloc(num_ranks*sizeof (buf_mr_info_t));
    kurmcl_exchange_mr_info(ds_conn_info, &allg_token_buf_recv, allg_remote_buf); // allgather交换，获取所有其他进程的接收buffer
    iovlist_for_allgather = (kurmcl_iov_t *)malloc(16 * sizeof(kurmcl_iov_t));
    for (int i = 0; i < num_ranks; ++i) {
        iovcont_per_rank_for_comb[i] = 0;
    }
}


void moe_combine_send(bfloat16_t *x_data, int16_t *src_info_data, int64_t num_tokens,
    int64_t num_max_dispatch_tokens_per_rank, int64_t num_experts, int64_t hidden, int16_t *parallel_sizes_data,
    int64_t batch_id)
{
#ifdef COMM_PERF
    comb_times++;
    if (comb_times > 0) {
        com_start_time = get_clock_us();
    }
#endif
    int64_t num_local_experts = num_experts / num_ranks;
    int64_t tmpx_ep_bias = num_ranks * num_max_dispatch_tokens_per_rank * hidden;
    int64_t tmpx_rank_bias = num_max_dispatch_tokens_per_rank * hidden;
    int64_t x_data_ep_bias = num_ranks * num_max_dispatch_tokens_per_rank * hidden;
    int64_t x_data_rank_bias = num_max_dispatch_tokens_per_rank * hidden;
    int16_t flag = 1;
    int tid;
    // 接收连续输入
    int send_bias = 0;
    for (int64_t i = 0; i < num_local_experts; ++i) {
        for (int64_t j = 0; j < num_ranks; ++j) {
            // src_info_data: during dispatch, what i receive
            int recv_nums = src_info_data[i * recv_src_info_ep_bias + j * (num_max_dispatch_tokens_per_rank + 1)];
            if (recv_nums != -1) {
                int16_t peer_rank_iov_cnt = iovcont_per_rank_for_comb[j];
                int16_t iov_bias = j * num_max_dispatch_tokens_per_rank + peer_rank_iov_cnt;
                iovlist_for_comb[iov_bias].len = hidden * recv_nums * sizeof(bfloat16_t);
                iovlist_for_comb[iov_bias].local_buffer =
                    comb_token_buf_send.buffer + (send_bias * hidden) * sizeof(bfloat16_t);
                iovlist_for_comb[iov_bias].lkey = comb_token_buf_send.lkey;
                iovlist_for_comb[iov_bias].remote_buffer =
                    comb_remote_buf[j].buffer + (i * tmpx_ep_bias + my_rank * tmpx_rank_bias) * sizeof(bfloat16_t);
                iovlist_for_comb[iov_bias].rkey = comb_remote_buf[j].rkey;
                iovcont_per_rank_for_comb[j]++;
                send_bias += recv_nums;
            }
            if (i == num_local_experts - 1 && iovcont_per_rank_for_comb[j] > 0) {
                comb_peer_nums++;
            }
        }
    }

#ifdef COMM_PERF
    if (comb_times > 0) {
        com_send_stage1 += (get_clock_us() - com_start_time);
        com_start_time = get_clock_us();
    }
#endif

kutacc::parallel_for(0, num_ranks, 1, [&](int64_t start, int64_t end) {
    for (int i = start; i < end; ++i) {
        if (iovcont_per_rank_for_comb[i] > 0) {
            kurmcl_put(&(iovlist_for_comb[i * num_max_dispatch_tokens_per_rank]), iovcont_per_rank_for_comb[i], i,
                1, ds_conn_info);
        }
    }
});

#ifdef COMM_PERF
    if (comb_times > 0) {
        com_send_stage2 += (get_clock_us() - com_start_time);
    }
#endif
}


void moe_combine_recv(bfloat16_t *combined_x_data, int16_t *topk_idx_data, float *topk_weights_data, int64_t num_tokens,
    int64_t num_experts, int64_t num_max_dispatch_tokens_per_rank, int64_t num_topk, int64_t hidden,  int16_t *parallel_sizes_data, int64_t batch_id,
    kupl_shm_win_h win)
{
#ifdef COMM_PERF
    if (comb_times > 0) {
        com_start_time = get_clock_us();
    }
#endif
    int16_t tp = parallel_sizes_data[0] / parallel_sizes_data[1];
    int16_t send_tokens_per_rank = num_tokens / tp;
    int16_t start_token_idx = (my_rank % tp) * send_tokens_per_rank;
    int16_t end_token_idx = (my_rank % tp + 1) * send_tokens_per_rank;
    int64_t private_token_idx = 0;
    int64_t tid;
    int64_t num_threads = 8;
    int64_t num_threads_per_rank = num_ranks / num_threads;
    int64_t myleader_rank = local_rank < 8 ? 0 : 8;
    int64_t peerleader_rank = local_rank < 8 ? 8 : 0;

kutacc::parallel_for(0, dis_peer_nums, 1, [&](int64_t start, int64_t end) {
    for (int i = start; i < end; ++i) {
        kurmcl_recv_imm_cnt(1, will_recv_from_rank[i], ds_conn_info);
    }
});
    kurmcl_flush(comb_peer_nums, ds_conn_info);
    comb_peer_nums = 0;
    dis_peer_nums = 0;
    for (int i = 0; i < num_ranks; ++i) {
        iovcont_per_rank_for_comb[i] = 0;
    }
#ifdef COMM_PERF
    if (comb_times > 0) {
        com_recv_stage1 += (get_clock_us() - com_start_time);
        com_start_time = get_clock_us();
    }
#endif
    if (num_tokens == 128) {
        kutacc::parallel_for(0, 8, 1, [&](int64_t start, int64_t end) {
            int tid = start;
            int token_idx = start + start_token_idx;
            volatile int x;
            double com_thread_start_time = 0;
            int64_t private_token_idx = token_idx >= end_token_idx ? token_idx - 8 : token_idx;
            if (token_idx < end_token_idx) {
                bfloat16_t *token_sum = combined_x_data + token_idx * hidden;
                std::vector<std::pair<float, bfloat16_t *>> g;
                for (int64_t i = 0; i < num_local_experts; ++i) {
                    for (int64_t j = 0; j < num_ranks; ++j) {
                        int64_t target_ep = j * num_local_experts + i;
                        float target_weight;
                        for (int64_t h = 0; h < num_topk; ++h) {
                            if (topk_idx_data[token_idx * num_topk + h] == target_ep) {
                                target_weight = topk_weights_data[token_idx * num_topk + h];
                                break;
                            }
                        }
                        int64_t token_will_receive =
                            src_info[i * recv_src_info_ep_bias + j * (num_max_dispatch_tokens_per_rank + 1)];
                        for (int64_t k = 0; k < token_will_receive; ++k) {
                            int target_token = src_info[i * recv_src_info_ep_bias +
                                j * (num_max_dispatch_tokens_per_rank + 1) + k + 1];
                            if (target_token == token_idx) {
                                auto token_part = tmpx_for_sum + (i * num_ranks * num_max_dispatch_tokens_per_rank +
                                                                    j * num_max_dispatch_tokens_per_rank + k) *
                                                                    hidden;
                                g.push_back({target_weight, token_part});
                            }
                        }
                    }
                }
                int i = 0;
                svbool_t pred = svptrue_b16();
                svbfloat16_t sve_zero = svdup_bf16(0);
                for (; i + 32 <= hidden; i += 32) {
                    svbfloat16_t sve_ = sve_zero;  // svld1(pred, token_sum + i);
                    svfloat32_t sve0 = svreinterpret_f32(svzip1(sve_zero, sve_));
                    svfloat32_t sve1 = svreinterpret_f32(svzip2(sve_zero, sve_));
                    for (const auto &[v, p] : g) {
                        sve_ = svld1(pred, p + i);
                        sve0 = svadd_m(pred, sve0, svmul_m(pred, svreinterpret_f32(svzip1(sve_zero, sve_)), v));
                        sve1 = svadd_m(pred, sve1, svmul_m(pred, svreinterpret_f32(svzip2(sve_zero, sve_)), v));
                    }
                    svfloat32_t sve2 = svuzp1_f32(sve0, sve1);
                    svfloat32_t sve3 = svuzp2_f32(sve0, sve1);
                    svbfloat16_t sve = svcvt_bf16_f32_x(pred, sve2);
                    sve = svcvtnt_bf16_f32_x(sve, pred, sve3);
                    svst1(pred, token_sum + i, sve);
                }
                if (__builtin_expect((i < hidden), 0)) {
                    for (int j = i; j < hidden; ++j) {
                        token_sum[j] = 0;
                        for (const auto &[v, p] : g) {
                            token_sum[j] += p[j] * v;
                        }
                    }
                }
                ++counter;
                int peer = mynode_id * 16 + tid + peerleader_rank;
                iovlist_for_allgather[tid].len = hidden * sizeof(bfloat16_t);
                iovlist_for_allgather[tid].local_buffer = allg_token_buf_send.buffer
                    + (private_token_idx * hidden) * sizeof(bfloat16_t);
                iovlist_for_allgather[tid].lkey = allg_token_buf_send.lkey;
                iovlist_for_allgather[tid].remote_buffer = allg_remote_buf[peer].buffer
                    + (private_token_idx * hidden) * sizeof(bfloat16_t);
                iovlist_for_allgather[tid].rkey = allg_remote_buf[peer].rkey;
                kurmcl_put(&(iovlist_for_allgather[tid]), 1, peer, 1, ds_conn_info);
                for (i = 0; i < hidden; i += 32) {
                    svbool_t pred = svwhilelt_b16((uint64_t)i, (uint64_t)hidden);
                    auto sve = svld1(pred, token_sum + i);
                    for (int peer = myleader_rank; peer < myleader_rank + 8; ++peer) {
                        if (peer != local_rank) {
                            svst1(pred, combinedx_group_ptr[peer] + private_token_idx * hidden + i, sve);
                        }
                    }
                }
                kurmcl_recv_imm_cnt(1, mynode_id * 16 + peerleader_rank + tid, ds_conn_info);
                int64_t bias = ((peerleader_rank + tid) * 8 + local_rank - myleader_rank);
                bias = bias * hidden;

                for (i = 0; i < hidden; i += 32) {
                    svbool_t pred = svwhilelt_b16((uint64_t)i, (uint64_t)hidden);
                    auto sve = svld1(pred, &(combined_x_data[bias]) + i);
                    for (int peer = myleader_rank; peer < myleader_rank + 8; ++peer) {
                        if (peer != local_rank) {
                            svst1(pred, combinedx_group_ptr[peer] + bias + i, sve);
                        }
                    }
                }
            }
            do {
                kupl_memory_cpu_load_fence();
                x = counter;
            } while (x != 8);
            for (int j = tid * num_threads_per_rank; j < (tid + 1) * num_threads_per_rank; ++j) {
                for (int i = 0; i < num_local_experts; ++i) {
                    catch_recv_src_info[i * recv_src_info_ep_bias + j * (num_max_dispatch_tokens_per_rank + 1)] = -1;
                    src_info[i * recv_src_info_ep_bias + j * (num_max_dispatch_tokens_per_rank + 1)] = 0;
                }
            }
        });
        kurmcl_flush(8, ds_conn_info);
        counter = 0;
    } else {
        kutacc::parallel_for(start_token_idx, end_token_idx, 1, [&](int64_t start, int64_t end) {
            for (int token_idx = start; token_idx < end; ++token_idx) {
                bfloat16_t *token_sum = combined_x_data + token_idx * hidden;
                std::vector<std::pair<float, bfloat16_t *>> g;
                for (int64_t i = 0; i < num_local_experts; ++i) {
                    for (int64_t j = 0; j < num_ranks; ++j) {
                        int64_t target_ep = j * num_local_experts + i;
                        float target_weight;
                        for (int64_t h = 0; h < num_topk; ++h) {
                            if (topk_idx_data[token_idx * num_topk + h] == target_ep) {
                                target_weight = topk_weights_data[token_idx * num_topk + h];
                                break;
                            }
                        }
                        int64_t token_will_receive =
                            src_info[i * recv_src_info_ep_bias + j * (num_max_dispatch_tokens_per_rank + 1)];
                        for (int64_t k = 0; k < token_will_receive; ++k) {
                            int target_token = src_info[i * recv_src_info_ep_bias +
                                j * (num_max_dispatch_tokens_per_rank + 1) + k + 1];
                            if (target_token == token_idx) {
                                auto token_part = tmpx_for_sum + (i * num_ranks * num_max_dispatch_tokens_per_rank +
                                                                    j * num_max_dispatch_tokens_per_rank + k) *
                                                                    hidden;
                                g.push_back({target_weight, token_part});
                            }
                        }
                    }
                }
                int i = 0;
                svbool_t pred = svptrue_b16();
                svbfloat16_t sve_zero = svdup_bf16(0);
                for (; i + 32 <= hidden; i += 32) {
                    svbfloat16_t sve_ = sve_zero;  // svld1(pred, token_sum + i);
                    svfloat32_t sve0 = svreinterpret_f32(svzip1(sve_zero, sve_));
                    svfloat32_t sve1 = svreinterpret_f32(svzip2(sve_zero, sve_));
                    for (const auto &[v, p] : g) {
                        sve_ = svld1(pred, p + i);
                        sve0 = svadd_m(pred, sve0, svmul_m(pred, svreinterpret_f32(svzip1(sve_zero, sve_)), v));
                        sve1 = svadd_m(pred, sve1, svmul_m(pred, svreinterpret_f32(svzip2(sve_zero, sve_)), v));
                    }
                    svfloat32_t sve2 = svuzp1_f32(sve0, sve1);
                    svfloat32_t sve3 = svuzp2_f32(sve0, sve1);
                    svbfloat16_t sve = svcvt_bf16_f32_x(pred, sve2);
                    sve = svcvtnt_bf16_f32_x(sve, pred, sve3);
                    svst1(pred, token_sum + i, sve);
                }
                if (__builtin_expect((i < hidden), 0)) {
                    for (int j = i; j < hidden; ++j) {
                        token_sum[j] = 0;
                        for (const auto &[v, p] : g) {
                            token_sum[j] += p[j] * v;
                        }
                    }
                }
            }
        });
        for (int i = 0; i < num_local_experts; ++i) {
            for (int j = 0; j < num_ranks; ++j) {
                catch_recv_src_info[i * recv_src_info_ep_bias + j * (num_max_dispatch_tokens_per_rank + 1)] = -1;
                src_info[i * recv_src_info_ep_bias + j * (num_max_dispatch_tokens_per_rank + 1)] = 0;
            }
        }
        kutacc::parallel_for(0, local_size, 1, [&](int64_t start, int64_t end) {
            for (int i = start; i < end; ++i) {
                int peer = (i + local_rank) % local_size;
                if (peer != local_rank) {
                    memcpy(combinedx_group_ptr[peer] + local_rank * hidden * send_tokens_per_rank,
                        &(combined_x_data[start_token_idx * hidden]),
                        hidden * send_tokens_per_rank * sizeof(bfloat16_t));
                }
            }
        });
    }
#ifdef COMM_PERF
    if (comb_times > 0) {
        com_recv_stage2 += (get_clock_us() - com_start_time);
        com_start_time = get_clock_us();
    }
#endif
    kupl_shm_fence(win);
#ifdef COMM_PERF
    if (comb_times > 0) {
        com_recv_stage3 += (get_clock_us() - com_start_time);
    }
#endif
}

void moe_combine_get_perf(double *perf_ary)
{
    perf_ary[0] = com_send_stage1;
    perf_ary[1] = com_send_stage2;
    perf_ary[2] = com_recv_stage1;
    perf_ary[3] = com_recv_stage2;
    perf_ary[4] = com_recv_stage3;
}

void moe_combine_finalize()
{
    kurmcl_dreg_mr(&comb_token_buf_send);
    kurmcl_dreg_mr(&comb_token_buf_recv);
    kurmcl_finalize(ds_conn_info);
    free(iovlist_for_comb);
}

}  // namespace kutacc

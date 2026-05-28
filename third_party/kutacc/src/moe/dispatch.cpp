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
#include <unistd.h>
#include "internal.h"
#include "kutacc.h"

namespace kutacc {

buf_mr_info_t disp_token_buf_send, disp_meta_buf_send, disp_buf_recv;
buf_mr_info_t *disp_remote_buf;  // 封装rkey用于交换
std::vector<int> iovcont_per_rank_for_disp;
kurmcl_iov_t *iovlist_for_disp = NULL;
// node based barrier abbreviated as nbb
buf_mr_info_t disp_nbb_send, disp_nbb_recv, *disp_nbb_remote;
int nbb_ppn = 16;
int nbb_num_node;
kurmcl_iov_t nbb_iov;
// end nnb

void moe_dispatch_init(uint8_t *x_data, uint8_t *packed_recv_x_data, int16_t *recv_src_info_data, int64_t num_experts,
    int64_t num_max_dispatch_tokens_per_rank, int64_t hidden, int64_t num_tokens, int64_t num_ranks_, int64_t my_rank_,
    int16_t *src_info_, void* disbuf_baseptr_, kurmcl_conn_info_t* ds_conn_info_, int16_t *parallel_sizes_data_)
{
    num_ranks = num_ranks_;
    my_rank = my_rank_;
    disbuf_baseptr = disbuf_baseptr_;
    num_local_experts = num_experts / num_ranks == 0 ? 1 : num_experts / num_ranks;
    packed_recv_x_size =
        num_local_experts * num_ranks * num_max_dispatch_tokens_per_rank * hidden * sizeof(char);
    recv_src_info_conut =
        num_local_experts * num_ranks * (num_max_dispatch_tokens_per_rank + 1);
    recv_src_info_size = recv_src_info_conut * sizeof(int16_t);
    recv_src_info_ep_bias = num_ranks * (num_max_dispatch_tokens_per_rank + 1);
    disbuf_size = packed_recv_x_size + recv_src_info_size;
    nbb_ppn = parallel_sizes_data_[0] / parallel_sizes_data_[1];

    int res;
    catch_recv_src_info = recv_src_info_data;
    src_info = src_info_;
    if (src_info == NULL) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        return;
    }

    kurmcl_init(ds_conn_info);
    kurmcl_build_conn(ds_conn_info);
    kurmcl_recv_init(ds_conn_info);
    kurmcl_reg_mr(&disp_token_buf_send, x_data, num_tokens * hidden, ds_conn_info);  // 注册发方token mr
    kurmcl_reg_mr(&disp_meta_buf_send, src_info, recv_src_info_conut * sizeof(int16_t),
        ds_conn_info);                                                   // 注册发方meta mr
    kurmcl_reg_mr(&disp_buf_recv, disbuf_baseptr, disbuf_size, ds_conn_info);  // 注册收方mr
    disp_remote_buf = (buf_mr_info_t *)malloc(num_ranks * sizeof(buf_mr_info_t));
    kurmcl_exchange_mr_info(ds_conn_info, &disp_buf_recv, disp_remote_buf);
    iovlist_for_disp =
        (kurmcl_iov_t *)malloc(num_ranks * num_max_dispatch_tokens_per_rank * 2 * sizeof(kurmcl_iov_t));  // 申请iovlist
    iovcont_per_rank_for_disp.resize(num_ranks);
    will_recv_from_rank.resize(num_ranks);
    for (int i = 0; i < num_ranks; ++i) {
        iovcont_per_rank_for_disp[i] = 0;
    }
    // node based barrier
    int *nbb_send;
    int *nbb_recv;
    nbb_num_node = num_ranks / nbb_ppn;
    if (num_ranks % nbb_ppn != 0) {
        printf("Error: feature node_based need a unified ppn %d node_cnt %d\n", nbb_ppn, nbb_num_node);
    }
    if (nbb_ppn > 128) {
        printf("Error: feature node_based need ppn < 128 %d\n", nbb_ppn);
    }
    int nbb_byte_cnt =
        nbb_num_node * (nbb_ppn + 1) * sizeof(int);  // should send to every node and one extra Bype for count
    nbb_send = (int *)malloc(nbb_byte_cnt);
    nbb_recv = (int *)malloc(nbb_byte_cnt);
    kurmcl_reg_mr(&disp_nbb_send, nbb_send, nbb_byte_cnt, ds_conn_info);
    kurmcl_reg_mr(&disp_nbb_recv, nbb_recv, nbb_byte_cnt, ds_conn_info);
    disp_nbb_remote = (buf_mr_info_t *)malloc(num_ranks * sizeof(buf_mr_info_t));
    kurmcl_exchange_mr_info(ds_conn_info, &disp_nbb_recv, disp_nbb_remote);
    // end node based barrier
    for (int i = 0; i < recv_src_info_conut; ++i) {
        catch_recv_src_info[i] = -1;
        src_info[i] = 0;
    }
    for (int i = 0; i < num_local_experts; ++i) {
        for (int j = 0; j < num_ranks; ++j) {
            recv_src_info_data[i * recv_src_info_ep_bias + j * (num_max_dispatch_tokens_per_rank + 1)] = -1;
            src_info[i * recv_src_info_ep_bias + j * (num_max_dispatch_tokens_per_rank + 1)] = 0;
        }
    }
}

void moe_dispatch_send(uint8_t *x_data, int16_t *topk_idx_data, int64_t num_tokens, int64_t num_topk,
    int64_t num_experts, int64_t num_max_dispatch_tokens_per_rank, int64_t hidden, int16_t *parallel_sizes_data,
    uint8_t *packed_recv_x_data, int16_t *recv_src_info_data, int64_t batch_id)
{
#ifdef COMM_PERF
    dis_times++;
    if (dis_times > 0) {
        dis_start_time = get_clock_us();
    }
#endif
    int64_t recv_x_ep_bias = num_ranks * num_max_dispatch_tokens_per_rank * hidden;
    int64_t recv_x_rank_bias = num_max_dispatch_tokens_per_rank * hidden;
    int16_t DP = parallel_sizes_data[1];
    int16_t same_data_ranks = num_ranks / DP;
    int64_t send_tokens_per_rank = num_tokens / same_data_ranks;
    int16_t start_token_idx = (my_rank % same_data_ranks) * send_tokens_per_rank;
    int16_t end_token_idx = (my_rank % same_data_ranks + 1) * send_tokens_per_rank;
    int tid;
    int last_peer = -1;

    // node based barrier
    int16_t nbb_same_data_ranks = num_ranks / DP;
    int64_t nbb_send_tokens_per_rank = num_tokens / same_data_ranks;
    int nbb_start_token_idx = 0;
    int nbb_end_token_idx = same_data_ranks * send_tokens_per_rank;
    int nbb_peer_rank;
    int nbb_src_rank;
    int nbb_dst_exp;
    int nbb_peer_node;
    int nbb_tmp = 0;
    int *nnb_sendbuf = (int *)disp_nbb_send.buffer;
    for (int i = 0; i < nbb_num_node; i++) {
        for (int j = 0; j < nbb_ppn; j++) {
            nnb_sendbuf[i * nbb_ppn + j] = 0;  // initialize bit
        }
    }
    for (int64_t i = nbb_start_token_idx; i < nbb_end_token_idx; i++) {
        int src_rank = i / nbb_send_tokens_per_rank;
        for (int64_t j = 0; j < num_topk; ++j) {
            nbb_dst_exp = topk_idx_data[i * num_topk + j];
            nbb_peer_rank = nbb_dst_exp / num_local_experts;
            if ((nbb_peer_rank % nbb_ppn) == (my_rank % nbb_ppn)) {   // local id inside node is the same
                nbb_peer_node = nbb_peer_rank / nbb_ppn;              // node id of the peer rank
                nnb_sendbuf[nbb_peer_node * nbb_ppn + src_rank] = 1;  // store the src rank id should be globalid
            }
        }
    }
    int src_node_id = my_rank / nbb_ppn;
    for (int i = 0; i < nbb_num_node; i++) {
        int peer_rank = my_rank % nbb_ppn;
        peer_rank = peer_rank + i * nbb_ppn;  // golbal id
        int offset_per_rank = nbb_ppn * sizeof(int);
        nbb_iov.len = offset_per_rank;
        nbb_iov.local_buffer = disp_nbb_send.buffer + i * offset_per_rank;
        nbb_iov.lkey = disp_nbb_send.lkey;
        nbb_iov.remote_buffer = disp_nbb_remote[peer_rank].buffer + src_node_id * offset_per_rank;
        nbb_iov.rkey = disp_nbb_remote[peer_rank].rkey;
        kurmcl_put(&nbb_iov, 1, peer_rank, 1, ds_conn_info);
    }
    // end node based barrier

    for (int64_t i = start_token_idx; i < end_token_idx; ++i) {
        for (int64_t j = 0; j < num_topk; ++j) {
            int16_t dst_exp = topk_idx_data[i * num_topk + j];
            int16_t peer_rank = dst_exp / num_local_experts;
            int16_t local_dst_exp = dst_exp % num_local_experts;
            int16_t num =
                src_info[local_dst_exp * recv_src_info_ep_bias + peer_rank * (num_max_dispatch_tokens_per_rank + 1)];

            int16_t peer_rank_iov_cnt = iovcont_per_rank_for_disp[peer_rank];
            int16_t iov_bias = peer_rank * num_max_dispatch_tokens_per_rank * 2 + peer_rank_iov_cnt;
            iovlist_for_disp[iov_bias].len = hidden;
            iovlist_for_disp[iov_bias].local_buffer = disp_token_buf_send.buffer + i * hidden;
            iovlist_for_disp[iov_bias].lkey = disp_token_buf_send.lkey;
            iovlist_for_disp[iov_bias].remote_buffer = disp_remote_buf[peer_rank].buffer +
                                                       local_dst_exp * recv_x_ep_bias + my_rank * recv_x_rank_bias +
                                                       num * hidden;
            iovlist_for_disp[iov_bias].rkey = disp_remote_buf[peer_rank].rkey;

            src_info[local_dst_exp * recv_src_info_ep_bias + peer_rank * (num_max_dispatch_tokens_per_rank + 1)]++;
            src_info[local_dst_exp * recv_src_info_ep_bias + peer_rank * (num_max_dispatch_tokens_per_rank + 1) + num +
                     1] = i;
            iovcont_per_rank_for_disp[peer_rank]++;
        }
    }
    for (int64_t j = 0; j < num_ranks; ++j) {
        for (int64_t i = 0; i < num_local_experts; ++i) {
            int16_t meta_nums = src_info[i * recv_src_info_ep_bias + j * (num_max_dispatch_tokens_per_rank + 1)];
            if (meta_nums > 0) {
                int16_t peer_rank_iov_cnt = iovcont_per_rank_for_disp[j];
                int16_t iov_bias = j * num_max_dispatch_tokens_per_rank * 2 + peer_rank_iov_cnt;
                iovlist_for_disp[iov_bias].len = (meta_nums + 1) * 2;
                iovlist_for_disp[iov_bias].local_buffer =
                    disp_meta_buf_send.buffer +
                    (i * recv_src_info_ep_bias + j * (num_max_dispatch_tokens_per_rank + 1)) * 2;
                iovlist_for_disp[iov_bias].lkey = disp_meta_buf_send.lkey;
                iovlist_for_disp[iov_bias].remote_buffer =
                    disp_remote_buf[j].buffer + packed_recv_x_size +
                    (i * recv_src_info_ep_bias + my_rank * (num_max_dispatch_tokens_per_rank + 1)) * 2;
                iovlist_for_disp[iov_bias].rkey = disp_remote_buf[j].rkey;
                iovcont_per_rank_for_disp[j]++;
                if (last_peer != j) {
                    will_recv_from_rank[dis_peer_nums] = j;
                    dis_peer_nums++;
                    last_peer = j;
                }
            }
        }
    }
#ifdef COMM_PERF
    if (dis_times > 0) {
        dis_send_stage1 += (get_clock_us() - dis_start_time);
        dis_start_time = get_clock_us();
    }
#endif

kutacc::parallel_for(0, num_ranks, 1, [&](int64_t start, int64_t end) {
    for (int i = start; i < end; ++i) {
        if (iovcont_per_rank_for_disp[i] > 0) {
            kurmcl_put(&(iovlist_for_disp[i * num_max_dispatch_tokens_per_rank * 2]), iovcont_per_rank_for_disp[i],
                i, 1, ds_conn_info);
        }
    }
});

#ifdef COMM_PERF
    if (dis_times > 0) {
        dis_send_stage2 += (get_clock_us() - dis_start_time);
    }
#endif
}


void moe_dispatch_recv(int16_t *recv_src_info_data, int64_t num_experts, int64_t num_max_dispatch_tokens_per_rank,
    int64_t batch_id)
{
#ifdef COMM_PERF
    if (dis_times > 0) {
        dis_start_time = get_clock_us();
    }
#endif
    for (int i = 0; i < num_ranks; ++i) {
        iovcont_per_rank_for_disp[i] = 0;
    }
    int *nnb_recvbuf_tmp = (int *)disp_nbb_recv.buffer;

kutacc::parallel_for(0, nbb_num_node, 1, [&](int64_t start, int64_t end) {
    for (int i = start; i < end; ++i) {
        int src_rank = my_rank % nbb_ppn;                 // local id in node
        src_rank = i * nbb_ppn + src_rank;                // global id
        kurmcl_recv_imm_cnt(1, src_rank, ds_conn_info);  // node info recveived, recv token info
        for (int j = 0; j < nbb_ppn; j++) {
            if (nnb_recvbuf_tmp[i * nbb_ppn + j] != 0) {
                src_rank = i * nbb_ppn + j;
                kurmcl_recv_imm_cnt(1, src_rank, ds_conn_info);
            }
        }
    }
});
    kurmcl_flush(dis_peer_nums + nbb_num_node, ds_conn_info);
    // node_based_barrier

#ifdef COMM_PERF
    if (dis_times > 0) {
        dis_recv_stage1 += (get_clock_us() - dis_start_time);
    }
#endif
}

void moe_dispatch_get_perf(double *perf_ary)
{
    perf_ary[0] = dis_send_stage1;
    perf_ary[1] = dis_send_stage2;
    perf_ary[2] = dis_recv_stage1;
}

void moe_dispatch_finalize()
{
    kurmcl_dreg_mr(&disp_token_buf_send);
    kurmcl_dreg_mr(&disp_meta_buf_send);
    free(iovlist_for_disp);
}

}  // namespace kutacc

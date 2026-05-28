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

#include <cerrno>
#include <infiniband/verbs.h>
#include <netinet/in.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <arpa/inet.h>
#include <sched.h>
#include "kutacc.h"

#ifndef KURMCL_DS
#define KURMCL_DS

#define MAX_NIC_NUM 2
#define MULTI_NODE
#define CQE_MODERATION
#define TX_CQ_MODERATION 128
#define TX_MAX_POLL 64
#define VERBS_PORT_NUM 1
#define VERBS_MAX_INLINE_DATA 64
#define IB_DEVICE_INDEX 0
#define GID_INDEX 1
#define CUSTOM_SEND_WR 4096
#define TRANSFER_PACKETS ((size_t)64)

#define VERBS_CHECK(call)                      \
    do {                                       \
        int _ret = (call);                     \
        verbs_check(_ret, __FILE__, __LINE__); \
    } while (0)

#define VERBS_CHECK_PTR(ptr)                     \
    do {                                         \
        if (!(ptr)) {                            \
            verbs_check(-1, __FILE__, __LINE__); \
        }                                        \
    } while (0)
// data barrier
#define kurmcl_aarch64_dsb(_op) asm volatile("dsb " #_op ::: "memory")

namespace kutacc {

#define MAX_NIC_NUM 2

typedef struct kurmcl_ep_info {
    struct ibv_qp *qp[MAX_NIC_NUM];
    struct ibv_recv_wr *rwr[MAX_NIC_NUM];
    struct ibv_cq *rcq[MAX_NIC_NUM];
    int rwr_avai[MAX_NIC_NUM];
    int rwr_max[MAX_NIC_NUM];
    int unsignaled[MAX_NIC_NUM];
} kurmcl_ep_t;

typedef struct kurmcl_iface {
    uint16_t lid;
    struct ibv_port_attr port_attr;
    struct ibv_device_attr dev_attr;
    struct ibv_pd *pd;
    struct ibv_context *ctx;
    struct ibv_device **dev_list;
    struct ibv_srq *srq;
    union ibv_gid gid;

    struct ibv_sge *ssge;
    struct ibv_recv_wr *rwr;
    struct ibv_send_wr *swr;
    struct ibv_cq *scq;
    struct ibv_cq *rcq;
    struct ibv_wc *wc;
    int tx_available;
    int tx_max_cnt;
    int rx_available;
    int rx_max_cnt;
    int tx_moderation;
    int tx_max_poll;
} kurmcl_iface_t;

typedef struct mr_key_info {
    uint32_t key[MAX_NIC_NUM];
} mr_key_t;

typedef struct buf_mr_info {
    uint64_t buffer;
    size_t len;
    struct ibv_mr *mr[MAX_NIC_NUM];
    mr_key_t lkey;
    mr_key_t rkey;
} buf_mr_info_t;

typedef struct barrier_info {
    buf_mr_info sbuf;
    buf_mr_info rbuf;
    buf_mr_info *remote;
} kurmcl_barrier_t;

typedef struct kurmcl_buf_iov {
    size_t len;
    uint64_t local_buffer;
    mr_key_t lkey;
    uint64_t remote_buffer;
    mr_key_t rkey;
} kurmcl_iov_t;

typedef struct nic_info {
    char nic_name[INET_ADDRSTRLEN];
    int numa_id;
} nic_info_t;

typedef struct conn_data {
    uint32_t qpn[MAX_NIC_NUM];
    union ibv_gid gid[MAX_NIC_NUM];
    uint16_t lid[MAX_NIC_NUM];
} conn_data_t;

typedef struct kurmcl_conn_info {
    uint32_t my_qpn;
    uint32_t remote_qpn;
    uint16_t my_lid;
    uint16_t remote_lid;
    int comm_size;
    int my_rank;
    int group;
    int nic_used;
    union ibv_gid my_gid;
    union ibv_gid remote_gid;
    struct ibv_qp *qp;
    struct ibv_port_attr port_attr;
    struct ibv_device_attr dev_attr;
    struct ibv_pd *pd;
    struct ibv_cq *cq;
    struct ibv_wc *wc;
    struct ibv_recv_wr *rwr;
    struct ibv_send_wr *swr;
    struct ibv_context *ctx;
    struct ibv_device **dev_list;
    struct ibv_sge *ssge;
    kurmcl_ep_t *eps;
    kurmcl_iface_t *ifaces;
    kurmcl_barrier_t *barrier_info;
    kurmcl_oob_allgather_cb_t oob_allgather;
    kurmcl_oob_barrier_cb_t oob_barrier;
    kurmcl_oob_alltoall_cb_t oob_alltoall;
} *kurmcl_conn_info_h, kurmcl_conn_info_t;

inline kurmcl_conn_info_h ds_conn_info;

void modify_qp_to_init(struct ibv_qp *qp);
void verbs_check(int error_code, const char *file, int line);
void modify_qp_to_rtr(struct ibv_qp *qp, uint32_t remote_qpn, uint16_t dlid, union ibv_gid *dgid, int gid_idx,
                      enum ibv_mtu active_mtu);
void modify_qp_to_rts(struct ibv_qp *qp);
void kurmcl_fill_nic_order(int *nic_order, nic_info_t *nic_info, struct ibv_device **dev_list);
void kurmcl_init(kurmcl_conn_info_t *conn_info);
void kurmcl_finalize(kurmcl_conn_info_t *conn_info);
void kurmcl_reg_mr(buf_mr_info_t *my_buf, void *buffer, size_t len, kurmcl_conn_info_t *conn_info);
void kurmcl_dreg_mr(buf_mr_info_t *buffer);
void kurmcl_build_conn(kurmcl_conn_info_t *conn_info);
void kurmcl_recv_init(kurmcl_conn_info_t *conn_info);
void kurmcl_put(kurmcl_iov_t *iovlist, int iovcount, int rank, int enable_imm, kurmcl_conn_info_t *conn_info);
void kurmcl_exchange_mr_info(kurmcl_conn_info_t *conn_info, buf_mr_info_t *my_buf_info, buf_mr_info_t *remote_buf);
void kurmcl_wait_imm_cnt(int cnt, kurmcl_conn_info_t *conn_info);
void kurmcl_flush(int count, kurmcl_conn_info_t *conn_info);
void kurmcl_recv_imm_cnt(int cnt, int rank, kurmcl_conn_info_t *conn_info);
int kurmcl_test_imm_cnt(int cnt, int rank, kurmcl_conn_info_t *conn_info);
void kurmcl_put_nosingal(kurmcl_iov_t *iovlist, int iovcount, int rank, int enable_imm, kurmcl_conn_info_t *conn_info);
void kurmcl_barrier_init(kurmcl_conn_info_t *conn_info, int comm_size);

}

#endif


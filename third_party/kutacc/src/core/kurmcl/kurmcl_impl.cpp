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
#include "kurmcl_impl.h"
int my_rank;
int comm_size;

namespace kutacc {

void verbs_check(int error_code, const char *file, int line)
{
    if (error_code) {
        printf("Verbs ERROR code=%d errno=[%s] at %s:%d\n", error_code, strerror(errno), file, line);
        exit(1);
    }
}

void modify_qp_to_init(struct ibv_qp *qp)
{
    struct ibv_qp_attr attr;
    int attr_mask;
    int rc;

    memset(&attr, 0, sizeof(attr));

    attr.qp_state = IBV_QPS_INIT;
    attr.port_num = VERBS_PORT_NUM;
    attr.qp_access_flags = IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE;
    attr_mask = IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS;

    rc = ibv_modify_qp(qp, &attr, attr_mask);
    if (rc) {
        printf("failed to modify QP state to INIT (%s)\n", strerror(errno));
        fflush(stdout);
        exit(1);
    }
}

void modify_qp_to_rtr(struct ibv_qp *qp, uint32_t remote_qpn, uint16_t dlid, union ibv_gid *dgid, int gid_idx,
    enum ibv_mtu active_mtu)
{
    struct ibv_qp_attr attr;
    int attr_mask;
    int rc;

    memset(&attr, 0, sizeof(attr));

    attr.qp_state = IBV_QPS_RTR;
    attr.path_mtu = active_mtu;
    attr.dest_qp_num = remote_qpn;
    attr.max_dest_rd_atomic = 16;
    attr.min_rnr_timer = 0x12;

    attr.ah_attr.is_global = 0;
    attr.ah_attr.dlid = dlid;
    attr.ah_attr.port_num = VERBS_PORT_NUM;
    attr_mask = IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
                IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER;
    if (dlid == 0) {
        attr.ah_attr.is_global = 1;
        memcpy(&attr.ah_attr.grh.dgid, dgid, 16);
        attr.ah_attr.grh.flow_label = 3;
        attr.ah_attr.grh.hop_limit = 64;
        attr.ah_attr.grh.sgid_index = gid_idx;
        attr.ah_attr.grh.traffic_class = 106;
    }

    rc = ibv_modify_qp(qp, &attr, attr_mask);
    if (rc) {
        printf("failed to modify QP state to RTR (%s)\n", strerror(errno));
        fflush(stdout);
        exit(1);
    }
}

void modify_qp_to_rts(struct ibv_qp *qp)
{
    struct ibv_qp_attr attr;
    int attr_mask;
    int rc;

    memset(&attr, 0, sizeof(attr));

    attr.qp_state = IBV_QPS_RTS;
    attr.timeout = 0x12;
    attr.retry_cnt = 6;
    attr.rnr_retry = 0;
    attr.max_rd_atomic = 16;

    attr_mask =
        IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC;

    rc = ibv_modify_qp(qp, &attr, attr_mask);
    if (rc) {
        printf("failed to modify QP state to RTS (%s)\n", strerror(errno));
        fflush(stdout);
        exit(1);
    }
}

void kurmcl_fill_nic_order(int *nic_order, nic_info_t *nic_info, struct ibv_device **dev_list)
{
    int port = 1;
    int gid_idx = 1;
    int err = 0;
    char local_str[128];
    uint8_t *local_addr;
    uint8_t *remote_addr;
    size_t addr_offset;
    size_t addr_size;
    int nic_cout = 0;
    FILE *stream;
    int numa_id;
    char buf[64] = {"\0"};
    char path[64];

    int nic_num = 0;

    for (int i = 0; dev_list[i]; ++i) {
        struct ibv_context *ctx = ibv_open_device(dev_list[i]);
        if (!ctx) {
            continue;
        }

        struct ibv_device_attr device_attr;
        if (ibv_query_device(ctx, &device_attr)) {
            ibv_close_device(ctx);
            continue;
        }

        struct ibv_port_attr port_attr;
        if (ibv_query_port(ctx, port, &port_attr)) {
            continue;
        }

        if (port_attr.link_layer != IBV_LINK_LAYER_ETHERNET) {
            continue;
        }

        union ibv_gid gid;
        if (ibv_query_gid(ctx, port, gid_idx, &gid)) {
            continue;
        }

        char gid_str[INET6_ADDRSTRLEN];
        inet_ntop(AF_INET6, gid.raw, gid_str, sizeof(gid_str));

        // 检查是否为IPv4映射地址
        int is_ipv4 = 1;
        for (int j = 0; j < 10; ++j) {
            if (gid.raw[j] != 0) {
                is_ipv4 = 0;
                break;
            }
        }

        if (is_ipv4 && gid.raw[10] == 0xff && gid.raw[11] == 0xff) {
            struct in_addr ipv4_addr;
            memcpy(&ipv4_addr, &gid.raw[12], 4);
            char ipv4_str[INET_ADDRSTRLEN];
            inet_ntop(AF_INET, &ipv4_addr, ipv4_str, sizeof(ipv4_str));
            strcpy(nic_info[nic_num].nic_name, ipv4_str);
            // read numa id for this nic
            snprintf(path, sizeof(path), "cat /sys/class/infiniband/%s/device/numa_node",
                ibv_get_device_name(dev_list[i]));
            stream = popen(path, "r");
            err = fread(buf, sizeof(int), sizeof(int), stream);
            int numa = atoi(buf);
            nic_info[nic_num].numa_id = numa;
            nic_num++;
            pclose(stream);
        }

        ibv_close_device(ctx);
    }

    // find the nearest nic
    int ncore_per_numa = 38;
    int die_per_node = 4;
    int numa_num_uma = 4;
    int core_id = sched_getcpu();
    int core_numa_id = core_id / ncore_per_numa;
    int core_die_id = core_numa_id / die_per_node;
    int nic_die_id;
    int max_numa_id = 0;
    int os_mode = 0;  // 0: numa mode, 1: uma mode
    int nic_id = 0;
    for (int ii = 0; ii < nic_num; ii++) {
        if (max_numa_id < nic_info[ii].numa_id) {
            max_numa_id = nic_info[ii].numa_id;
        }
    }
    if (max_numa_id <= numa_num_uma) {  // uma
        os_mode = 1;
    }

    for (int ii = 0; ii < nic_num; ii++) {
        if (os_mode) {
            nic_die_id = nic_info[ii].numa_id;
        } else {
            nic_die_id = nic_info[ii].numa_id / die_per_node;
        }

        if (core_die_id == nic_die_id) {  // find the nearest one nic
            nic_order[nic_id] = ii;
            nic_id++;
        }
    }
    if (nic_id > 2 || nic_id <= 0) {
        printf("fatal numa_dis calculation failed nic_num %d\n", nic_id);
    }

    // reorder the nic based on the ip
    char nic_one[INET_ADDRSTRLEN];
    char nic_two[INET_ADDRSTRLEN];
    strcpy(nic_one, nic_info[nic_order[0]].nic_name);
    strcpy(nic_two, nic_info[nic_order[1]].nic_name);
    if (strcmp(nic_one, nic_two) > 0) {
        int tmp = nic_order[0];
        nic_order[0] = nic_order[1];
        nic_order[1] = tmp;
    }
}

#ifdef CQE_MODERATION
void kurmcl_txqp_init(kurmcl_ep_t *eps, int comm_size)
{
    for (int i = 0; i < comm_size; i++) {
        for (int j = 0; j < MAX_NIC_NUM; j++) {
            eps[i].unsignaled[j] = 0;
        }
    }
}
#endif

int kurmcl_comm_create(int size, int rank, kurmcl_oob_cb_h oob_cbs, int group, kurmcl_conn_info_h *conn_info_user)
{
    if (size <= 0 || rank < 0 || rank >= size) {
        printf("size, rank or pid out of range\n");
        return -1;
    }
    if (conn_info_user == nullptr || oob_cbs == nullptr || oob_cbs->oob_allgather == nullptr ||
        oob_cbs->oob_barrier == nullptr || oob_cbs->oob_alltoall == nullptr) {
        printf("conn_info, group or oob_allgather is nullptr\n");
        return -1;
    }
    *conn_info_user = (kurmcl_conn_info *)malloc(sizeof(kurmcl_conn_info));
    if (*conn_info_user == nullptr) {
        printf("kurmcl_malloc failed\n");
        return -1;
    }
    (*conn_info_user)->comm_size = size;
    (*conn_info_user)->my_rank = rank;
    (*conn_info_user)->oob_allgather = oob_cbs->oob_allgather;
    (*conn_info_user)->oob_barrier = oob_cbs->oob_barrier;
    (*conn_info_user)->oob_alltoall = oob_cbs->oob_alltoall;
    (*conn_info_user)->group = group;
    comm_size = size;
    my_rank = rank;
    ds_conn_info = *conn_info_user;
    return 0;
}

void kurmcl_init(kurmcl_conn_info_t *conn_info)
{
    struct ibv_device **dev_list = ibv_get_device_list(NULL);
    VERBS_CHECK_PTR(dev_list);

#ifdef MULTI_NODE
    kurmcl_iface_t *ifaces;
    ifaces = (kurmcl_iface_t *)malloc(MAX_NIC_NUM * sizeof(kurmcl_iface_t));
    int nic_order[MAX_NIC_NUM];
    nic_info_t nic_info[10];
    kurmcl_fill_nic_order(nic_order, nic_info, dev_list);
    for (int ii = 0; ii < MAX_NIC_NUM; ii++) {
        int nic_idx = nic_order[ii];

        struct ibv_context *ctx = ibv_open_device(dev_list[nic_idx]);
        VERBS_CHECK_PTR(ctx);

        struct ibv_device_attr dev_attr;
        VERBS_CHECK(ibv_query_device(ctx, &dev_attr));

        struct ibv_pd *pd = ibv_alloc_pd(ctx);
        VERBS_CHECK_PTR(pd);

        struct ibv_cq *scq = ibv_create_cq(ctx, dev_attr.max_cqe, NULL, NULL, 0);
        VERBS_CHECK_PTR(scq);
        struct ibv_port_attr port_attr;
        VERBS_CHECK(ibv_query_port(ctx, VERBS_PORT_NUM, &port_attr));

        union ibv_gid gid;
        if (port_attr.lid == 0) {
            VERBS_CHECK(ibv_query_gid(ctx, VERBS_PORT_NUM, GID_INDEX, &gid));
        }

        struct ibv_wc *wc = (struct ibv_wc *)malloc(sizeof(struct ibv_wc) * dev_attr.max_cqe);
        VERBS_CHECK_PTR(wc);

        struct ibv_send_wr *wr = (struct ibv_send_wr *)malloc(sizeof(struct ibv_send_wr) * CUSTOM_SEND_WR);
        VERBS_CHECK_PTR(wr);
        struct ibv_sge *sge = (struct ibv_sge *)malloc(sizeof(struct ibv_sge) * CUSTOM_SEND_WR);
        VERBS_CHECK_PTR(sge);
#ifdef ENABLE_SRQ
        struct ibv_srq_init_attr srq_init_attr;

        memset(&srq_init_attr, 0, sizeof(srq_init_attr));
        srq_init_attr.attr.max_sge = 1;
        srq_init_attr.attr.max_wr = 32768;
        srq_init_attr.attr.srq_limit = 0;
        srq_init_attr.srq_context = ctx;
        struct ibv_srq *srq = ibv_create_srq(pd, &srq_init_attr);
        VERBS_CHECK_PTR(srq);
        ifaces[ii].srq = srq;
#endif
        ifaces[ii].ctx = ctx;
        ifaces[ii].dev_attr = dev_attr;
        ifaces[ii].pd = pd;
        ifaces[ii].scq = scq;
        ifaces[ii].port_attr = port_attr;
        ifaces[ii].gid = gid;
        ifaces[ii].lid = port_attr.lid;
        ifaces[ii].wc = wc;
        ifaces[ii].swr = wr;
        ifaces[ii].ssge = sge;
#ifdef CQE_MODERATION
	ifaces[ii].tx_moderation = TX_CQ_MODERATION;
        ifaces[ii].tx_max_poll = TX_MAX_POLL;
#endif
    }
    kurmcl_ep_t *eps = (kurmcl_ep_t *)malloc(comm_size * sizeof(kurmcl_ep_t));
    VERBS_CHECK_PTR(eps);

    for (int ii = 0; ii < comm_size; ii++) {
        for (int jj = 0; jj < MAX_NIC_NUM; jj++) {
            struct ibv_context *ctx = ifaces[jj].ctx;
            struct ibv_device_attr dev_attr = ifaces[jj].dev_attr;
            struct ibv_cq *rcq = ibv_create_cq(ctx, 4096, NULL, NULL, 0);
            VERBS_CHECK_PTR(rcq);
            struct ibv_qp_init_attr qp_init_attr = {.send_cq = ifaces[jj].scq,
                .recv_cq = rcq,
#ifdef ENABLE_SRQ
                .srq = ifaces[jj].srq,
#endif
                .cap = {.max_send_wr = CUSTOM_SEND_WR,
                    .max_recv_wr = static_cast<uint32_t>(ifaces[jj].dev_attr.max_qp_wr),
                    .max_send_sge = 1,
                    .max_recv_sge = 1,
                    .max_inline_data = VERBS_MAX_INLINE_DATA},
                .qp_type = IBV_QPT_RC,
                .sq_sig_all = 0};

            struct ibv_qp *qp = ibv_create_qp(ifaces[jj].pd, &qp_init_attr);
            VERBS_CHECK_PTR(qp);
            modify_qp_to_init(qp);
            eps[ii].qp[jj] = qp;
            eps[ii].rcq[jj] = rcq;
        }
    }
    conn_info->ifaces = ifaces;
    conn_info->eps = eps;
    conn_info->comm_size = comm_size;  // global communication size
    // calculate the nic used
    int proc_per_node;
    int proc_per_die;
    int die_per_node = 4;
    int local_rankid_in_die;
    int nic_used = 0;
    int rank_local_size;
    char *mpi_local_size;
    mpi_local_size = getenv("MV2_COMM_WORLD_LOCAL_SIZE");

    proc_per_node = atoi(mpi_local_size);
    proc_per_die = proc_per_node / die_per_node;
    ;
    local_rankid_in_die = my_rank % proc_per_die;

    if (proc_per_die != 1 && (proc_per_die % 2) != 0) {
        printf("fatal: proc per die should be 1 or a even number\n");
        exit(0);
    }
    // currently support 2 nic per die, and each process use only 1 nic
    if (local_rankid_in_die < (proc_per_die / 2)) {
        nic_used = 0;
    } else {
        nic_used = 1;
    }

    conn_info->nic_used = nic_used;  // use to switch between two nics;
    conn_info->dev_list = dev_list;
#ifdef CQE_MODERATION
    kurmcl_txqp_init(eps, comm_size);
#endif
    kurmcl_barrier_init(conn_info, comm_size);
#else
    printf("rank %d choose device %s\n", uru_mpi->rank, ibv_get_device_name(dev_list[IB_DEVICE_INDEX]));
    struct ibv_context *ctx = ibv_open_device(dev_list[IB_DEVICE_INDEX]);
    VERBS_CHECK_PTR(ctx);

    struct ibv_device_attr dev_attr;
    VERBS_CHECK(ibv_query_device(ctx, &dev_attr));

    struct ibv_pd *pd = ibv_alloc_pd(ctx);
    VERBS_CHECK_PTR(pd);

    struct ibv_cq *cq = ibv_create_cq(ctx, dev_attr.max_cqe, NULL, NULL, 0);
    VERBS_CHECK_PTR(cq);

    struct ibv_port_attr port_attr;
    VERBS_CHECK(ibv_query_port(ctx, VERBS_PORT_NUM, &port_attr));

    if (port_attr.lid == 0) {
        VERBS_CHECK(ibv_query_gid(ctx, VERBS_PORT_NUM, GID_INDEX, &conn_info->my_gid));
    }

    struct ibv_qp_init_attr qp_init_attr = {.send_cq = cq,
        .recv_cq = cq,
        .cap = {.max_send_wr = CUSTOM_SEND_WR,
            .max_recv_wr = dev_attr.max_qp_wr,
            .max_send_sge = 1,
            .max_recv_sge = 1,
            .max_inline_data = VERBS_MAX_INLINE_DATA},
        .qp_type = IBV_QPT_RC};

    struct ibv_qp *qp = ibv_create_qp(pd, &qp_init_attr);
    VERBS_CHECK_PTR(qp);
    modify_qp_to_init(qp);
    conn_info->my_qpn = qp->qp_num;
    struct ibv_wc *wc = (struct ibv_wc *)malloc(sizeof(struct ibv_wc) * dev_attr.max_cqe);
    VERBS_CHECK_PTR(wc);

    conn_info->my_lid = port_attr.lid;
    conn_info->qp = qp;
    conn_info->port_attr = port_attr;
    conn_info->dev_attr = dev_attr;
    conn_info->cq = cq;
    conn_info->pd = pd;
    conn_info->wc = wc;
    conn_info->ctx = ctx;
    conn_info->dev_list = dev_list;
#endif
}

void kurmcl_finalize(kurmcl_conn_info_t *conn_info)
{
    kurmcl_iface_t *ifaces = conn_info->ifaces;
    kurmcl_ep_t *eps = conn_info->eps;
    for (int ii = 0; ii < MAX_NIC_NUM; ii++) {
        struct ibv_context *ctx = ifaces[ii].ctx;
        struct ibv_pd *pd = ifaces[ii].pd;
        struct ibv_cq *scq = ifaces[ii].scq;
        struct ibv_cq *rcq = ifaces[ii].rcq;
        struct ibv_wc *wc = ifaces[ii].wc;
        struct ibv_sge *ssge = ifaces[ii].ssge;
        struct ibv_send_wr *swr = ifaces[ii].swr;
        struct ibv_srq *srq = ifaces[ii].srq;
        for (int jj = 0; jj < comm_size; jj++) {
            struct ibv_recv_wr *rwr = eps[jj].rwr[ii];
            struct ibv_cq *rcq = eps[jj].rcq[ii];
            struct ibv_qp *qp = eps[jj].qp[ii];
            if (qp) {
                VERBS_CHECK(ibv_destroy_qp(qp));
            }
            if (rcq) {
                VERBS_CHECK(ibv_destroy_cq(rcq));
            }
            if (rwr) {
                free(rwr);
            }
        }
        if (scq) {
            VERBS_CHECK(ibv_destroy_cq(scq));
        }
        if (wc) {
            free(wc);
        }
        if (ssge) {
            free(ssge);
        }
        if (swr) {
            free(swr);
        }
        if (ctx) {
            VERBS_CHECK(ibv_close_device(ctx));
        }
    }
    struct ibv_device **dev_list = conn_info->dev_list;
    ibv_free_device_list(dev_list);
}
void kurmcl_dreg_mr(buf_mr_info_t *buffer)
{
    struct ibv_mr *mr;
    for (int ii = 0; ii < MAX_NIC_NUM; ii++) {
        mr = buffer->mr[ii];
        if (mr) {
            VERBS_CHECK(ibv_dereg_mr(mr));
        }
    }
}

void kurmcl_build_conn(kurmcl_conn_info_t *conn_info)
{
    conn_data_t *conn_data;
    conn_data_t *conn_data_remot;
    uint32_t remot_qpn;
    uint16_t remot_lid;
    union ibv_gid remot_gid;
    struct ibv_qp *qp;
    struct ibv_port_attr port_attr;

    int comm_size = conn_info->comm_size;
    kurmcl_ep_t *eps = conn_info->eps;
    kurmcl_iface_t *ifaces = conn_info->ifaces;
    conn_data = (conn_data_t *)malloc(comm_size * sizeof(conn_data_t));
    VERBS_CHECK_PTR(conn_data);
    conn_data_remot = (conn_data_t *)malloc(comm_size * sizeof(conn_data_t));
    VERBS_CHECK_PTR(conn_data_remot);
    for (int ii = 0; ii < comm_size; ii++) {
        for (int jj = 0; jj < MAX_NIC_NUM; jj++) {
            conn_data[ii].qpn[jj] = eps[ii].qp[jj]->qp_num;
            conn_data[ii].gid[jj] = ifaces[jj].gid;
            conn_data[ii].lid[jj] = ifaces[jj].lid;
        }
    }
    conn_info->oob_alltoall(conn_data, sizeof(conn_data_t), KURMCL_DATATYPE_CHAR, conn_data_remot, sizeof(conn_data_t),
                            KURMCL_DATATYPE_CHAR, conn_info->group);

    for (int ii = 0; ii < comm_size; ii++) {
        for (int jj = 0; jj < MAX_NIC_NUM; jj++) {
            remot_qpn = conn_data_remot[ii].qpn[jj];
            remot_lid = conn_data_remot[ii].lid[jj];
            remot_gid = conn_data_remot[ii].gid[jj];
            qp = eps[ii].qp[jj];
            port_attr = ifaces[jj].port_attr;
            modify_qp_to_rtr(qp, remot_qpn, remot_lid, &remot_gid, GID_INDEX, port_attr.active_mtu);
            modify_qp_to_rts(qp);
        }
    }
    free(conn_data);
    free(conn_data_remot);
}

void kurmcl_put_nosingal(kurmcl_iov_t *iovlist, int iovcount, int rank, int enable_imm, kurmcl_conn_info_t *conn_info)
{
    struct ibv_sge sge[10];
    struct ibv_send_wr wr[10];
    kurmcl_ep_t *eps = conn_info->eps;
    kurmcl_iface_t *ifaces = conn_info->ifaces;
    int nic_order = conn_info->nic_used;
    if (iovcount >= 10) {
        printf("fatal: iovcnt %d larger than 10\n", iovcount);
    }

    for (int i = 0; i < iovcount; i++) {
        sge[i].addr = iovlist[i].local_buffer;
        sge[i].length = iovlist[i].len;
        sge[i].lkey = iovlist[i].lkey.key[nic_order];

        wr[i].wr_id = 0;
        if (enable_imm && (i == iovcount - 1)) {
            wr[i].opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
        } else {
            wr[i].opcode = IBV_WR_RDMA_WRITE;
        }
        wr[i].send_flags = 0;
        wr[i].imm_data = 1;
        wr[i].wr.rdma.remote_addr = iovlist[i].remote_buffer;
        wr[i].wr.rdma.rkey = iovlist[i].rkey.key[nic_order];
        wr[i].sg_list = &sge[i];
        wr[i].num_sge = 1;
        wr[i].next = &wr[i + 1];
    }
    wr[iovcount - 1].next = NULL;
    struct ibv_send_wr *bad_wr;
    struct ibv_qp *qp = eps[rank].qp[nic_order];
    VERBS_CHECK(ibv_post_send(qp, wr, &bad_wr));
}

#ifdef CQE_MODERATION
int kurmcl_tx_moderation(kurmcl_ep_t *ep, kurmcl_iface_t *ifaces, int nic_order)
{
    if (ep->unsignaled[nic_order] >= ifaces[nic_order].tx_moderation) {
        return 1;
    } else {
        return 0;
    }
}
void kurmcl_txqp_posted(kurmcl_ep_t *ep, int send_flag, int nic_order, int iovcount)
{
    if (send_flag) { // send cqe is enabled, start from zero
        ep->unsignaled[nic_order] = 0;
    } else {
        ep->unsignaled[nic_order] += iovcount;
    }
}
#endif

void kurmcl_put(kurmcl_iov_t *iovlist, int iovcount, int rank, int enable_imm, kurmcl_conn_info_t *conn_info)
{
    struct ibv_sge sge[100];
    struct ibv_send_wr wr[100];
    kurmcl_ep_t *eps = conn_info->eps;
    kurmcl_iface_t *ifaces = conn_info->ifaces;
    int nic_order = conn_info->nic_used;

#ifdef CQE_MODERATION
    int send_flag = 0;
    kurmcl_ep_t *ep = &conn_info->eps[rank];
    send_flag = kurmcl_tx_moderation(ep, ifaces, nic_order);
#endif
    if (iovcount >= 100) {
        printf("fatal: iovcnt %d larger than 10\n", iovcount);
    }

    for (int i = 0; i < iovcount; i++) {
        sge[i].addr = iovlist[i].local_buffer;
        sge[i].length = iovlist[i].len;
        sge[i].lkey = iovlist[i].lkey.key[nic_order];

        wr[i].wr_id = 0;
        if (enable_imm && (i == iovcount - 1)) {
            wr[i].opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
        } else {
            wr[i].opcode = IBV_WR_RDMA_WRITE;
        }
#ifdef CQE_MODERATION
        if ((i == iovcount - 1) && send_flag) {
#else
        if (i == iovcount - 1) {
#endif
            wr[i].send_flags = IBV_SEND_SIGNALED;
        } else {
            wr[i].send_flags = 0;
        }
        wr[i].imm_data = 1;
        wr[i].wr.rdma.remote_addr = iovlist[i].remote_buffer;
        wr[i].wr.rdma.rkey = iovlist[i].rkey.key[nic_order];
        wr[i].sg_list = &sge[i];
        wr[i].num_sge = 1;
        wr[i].next = &wr[i + 1];
    }
    wr[iovcount - 1].next = NULL;
    struct ibv_send_wr *bad_wr;
    struct ibv_qp *qp = eps[rank].qp[nic_order];
    VERBS_CHECK(ibv_post_send(qp, wr, &bad_wr));
#ifdef CQE_MODERATION
    kurmcl_txqp_posted(ep, send_flag, nic_order, iovcount);
#endif
}

void kurmcl_recv_init(kurmcl_conn_info_t *conn_info)
{
    struct ibv_device_attr dev_attr;
    struct ibv_qp *qp;
    int comm_size = conn_info->comm_size;

    kurmcl_iface_t *ifaces = conn_info->ifaces;
    kurmcl_ep_t *eps = conn_info->eps;
#ifdef ENABLE_SRQ
    for (int jj = 0; jj < MAX_NIC_NUM; jj++) {
        dev_attr = ifaces[jj].dev_attr;
        struct ibv_srq *srq = ifaces[jj].srq;
        struct ibv_recv_wr *wr = (struct ibv_recv_wr *)malloc(sizeof(struct ibv_recv_wr) * dev_attr.max_qp_wr);
        VERBS_CHECK_PTR(wr);
        memset(wr, 0, sizeof(struct ibv_recv_wr) * dev_attr.max_qp_wr);
        for (int i = 0; i < dev_attr.max_qp_wr - 1; i++) {
            wr[i].next = &wr[i + 1];
        }
        wr[dev_attr.max_qp_wr - 1].next = NULL;
        struct ibv_recv_wr *bad_wr;
        ibv_post_srq_recv(srq, wr, &bad_wr);
        ifaces[jj].rwr = wr;
    }
#else
    for (int ii = 0; ii < comm_size; ii++) {
        for (int jj = 0; jj < MAX_NIC_NUM; jj++) {
            dev_attr = ifaces[jj].dev_attr;
            qp = eps[ii].qp[jj];
            struct ibv_recv_wr *wr = (struct ibv_recv_wr *)malloc(sizeof(struct ibv_recv_wr) * dev_attr.max_qp_wr);
            VERBS_CHECK_PTR(wr);
            memset(wr, 0, sizeof(struct ibv_recv_wr) * dev_attr.max_qp_wr);
            for (int i = 0; i < dev_attr.max_qp_wr; i++) {
                wr[i].next = &wr[i + 1];
            }
            wr[dev_attr.max_qp_wr - 1].next = NULL;
            struct ibv_recv_wr *bad_wr;
            ibv_post_recv(qp, wr, &bad_wr);
            eps[ii].rwr[jj] = wr;
            eps[ii].rwr_avai[jj] = dev_attr.max_qp_wr;  // init the recv queue
            eps[ii].rwr_max[jj] = dev_attr.max_qp_wr;   // init the recv queue
        }
    }
#endif
}

void kurmcl_reg_mr(buf_mr_info_t *my_buf, void *buffer, size_t len, kurmcl_conn_info_t *conn_info)
{
    struct ibv_pd *pd;
    kurmcl_iface_t *ifaces = conn_info->ifaces;
    my_buf->len = len;
    my_buf->buffer = (uint64_t)buffer;
    for (int ii = 0; ii < MAX_NIC_NUM; ii++) {
        pd = ifaces[ii].pd;
        struct ibv_mr *mr =
            ibv_reg_mr(pd, buffer, len, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE);
        VERBS_CHECK_PTR(mr);
        my_buf->mr[ii] = mr;
        my_buf->lkey.key[ii] = mr->lkey;
        my_buf->rkey.key[ii] = mr->rkey;
    }
}

void kurmcl_exchange_mr_info(kurmcl_conn_info_t *conn_info, buf_mr_info_t *my_buf_info, buf_mr_info_t *remote_buf)
{
    conn_info->oob_allgather(my_buf_info, remote_buf, sizeof(buf_mr_info_t), conn_info->group, KURMCL_DATATYPE_CHAR);
}

void kurmcl_wait_imm_cnt(int cnt, kurmcl_conn_info_t *conn_info)
{
    struct ibv_cq *cq;
    struct ibv_qp *qp;
    int outstanding = cnt;
    struct ibv_device_attr dev_attr;
    struct ibv_wc *wc = conn_info->wc;
    struct ibv_recv_wr *wr;
    kurmcl_iface_t *ifaces = conn_info->ifaces;
    kurmcl_ep_t *eps;
    struct ibv_srq *srq;

    size_t count = 0;
    printf("for this version wati_imm_cnt is NA please use kurmcl_recv_immcnt\n");
    while (count < outstanding) {
        for (int ii = 0; ii < MAX_NIC_NUM; ii++) {
            cq = ifaces[ii].rcq;
            dev_attr = ifaces[ii].dev_attr;
            wc = ifaces[ii].wc;
            int num_complete = ibv_poll_cq(cq, dev_attr.max_cqe, wc);
            if (num_complete < 0) {
                VERBS_CHECK(num_complete);
            }
            count += num_complete;
            if (num_complete) {
                wr = ifaces[ii].rwr;
#ifdef ENABLE_SRQ
                srq = ifaces[ii].srq;
                wr[num_complete - 1].next = NULL;
                struct ibv_recv_wr *bad_wr;
                ibv_post_srq_recv(srq, wr, &bad_wr);
                wr[num_complete - 1].next = wr + num_complete;
#endif
            }
        }
    }
}

void kurmcl_recv_imm_cnt(int cnt, int rank, kurmcl_conn_info_t *conn_info)
{
    struct ibv_cq *cq;
    struct ibv_qp *qp;
    int outstanding = cnt;
    int remain = cnt;
    int rwr_avai;
    int rwr_max;
    struct ibv_device_attr dev_attr;
    struct ibv_wc wc[10];
    struct ibv_recv_wr *wr;
    kurmcl_iface_t *ifaces = conn_info->ifaces;
    kurmcl_ep_t *eps = conn_info->eps;

    size_t count = 0;
    while (count < outstanding) {
        for (int ii = 0; ii < MAX_NIC_NUM; ii++) {
            cq = eps[rank].rcq[ii];
            qp = eps[rank].qp[ii];
            wr = eps[rank].rwr[ii];
            dev_attr = ifaces[ii].dev_attr;
            rwr_avai = eps[rank].rwr_avai[ii];
            rwr_max = eps[rank].rwr_max[ii];
            int num_complete = ibv_poll_cq(cq, remain, wc);
            if (num_complete < 0) {
                VERBS_CHECK(num_complete);
            }
            count += num_complete;
            remain = remain - num_complete;
            rwr_avai -= num_complete;
            eps[rank].rwr_avai[ii] = rwr_avai;
            if (rwr_avai < rwr_max / 16) {
                wr[rwr_max - rwr_avai - 1].next = NULL;
                struct ibv_recv_wr *bad_wr;
                ibv_post_recv(qp, wr, &bad_wr);
                wr[rwr_max - rwr_avai - 1].next = wr + rwr_max - rwr_avai;
                eps[rank].rwr_avai[ii] = rwr_max;
            }
            if (remain < 0) {
                printf("cqe num is wrong in poll recvcq remain %d\n", remain);
            }
        }
    }
}

int kurmcl_test_imm_cnt(int cnt, int rank, kurmcl_conn_info_t *conn_info)
{
    int status = 0;
    struct ibv_cq *cq;
    struct ibv_qp *qp;
    int outstanding = cnt;
    int remain = cnt;
    int rwr_avai;
    int rwr_max;
    struct ibv_device_attr dev_attr;
    struct ibv_wc wc[10];
    struct ibv_recv_wr *wr;
    kurmcl_iface_t *ifaces = conn_info->ifaces;
    kurmcl_ep_t *eps = conn_info->eps;

    size_t count = 0;
    for (int ii = 0; ii < MAX_NIC_NUM; ii++) {
        cq = eps[rank].rcq[ii];
        qp = eps[rank].qp[ii];
        wr = eps[rank].rwr[ii];
        dev_attr = ifaces[ii].dev_attr;
        rwr_avai = eps[rank].rwr_avai[ii];
        rwr_max = eps[rank].rwr_max[ii];
        int num_complete = ibv_poll_cq(cq, remain, wc);
        if (num_complete < 0) {
            VERBS_CHECK(num_complete);
        }
        count += num_complete;
        remain = remain - num_complete;
        rwr_avai -= num_complete;
        eps[rank].rwr_avai[ii] = rwr_avai;
        if (rwr_avai < rwr_max / 16) {
            wr[rwr_max - rwr_avai - 1].next = NULL;
            struct ibv_recv_wr *bad_wr;
            ibv_post_recv(qp, wr, &bad_wr);
            wr[rwr_max - rwr_avai - 1].next = wr + rwr_max - rwr_avai;
            eps[rank].rwr_avai[ii] = rwr_max;
        }
        if (remain < 0) {
            printf("cqe num is wrong in poll recvcq remain %d\n", remain);
        }
    }
    // if cqe is arrived set the status
    if (count == outstanding) {
        status = 1;
    }
    return status;
}

static inline int kurmcl_next_poweroftwo(int value)
{
    int power2;
    for (power2 = 1; value > 0; value >>= 1, power2 <<= 1) { /* empty */
        ;
    }

    return power2;
}

void kurmcl_barrier_init(kurmcl_conn_info_t *conn_info, int comm_size)
{
    char *sendbuf, *recvbuf;
    buf_mr_info_t my_buf_send;
    buf_mr_info_t my_buf_recv;
    buf_mr_info_t *remote_buf;
    int buffer_size = comm_size * sizeof(char);
    kurmcl_barrier_t *barrier_info;
    barrier_info = (kurmcl_barrier_t *)malloc(sizeof(kurmcl_barrier_t));
    VERBS_CHECK_PTR(barrier_info);

    sendbuf = (char *)malloc(buffer_size);
    VERBS_CHECK_PTR(sendbuf);
    recvbuf = (char *)malloc(buffer_size);
    VERBS_CHECK_PTR(recvbuf);
    remote_buf = (buf_mr_info_t *)malloc(comm_size * sizeof(buf_mr_info_t));
    VERBS_CHECK_PTR(remote_buf);

    kurmcl_reg_mr(&my_buf_recv, recvbuf, buffer_size, conn_info);
    kurmcl_reg_mr(&my_buf_send, sendbuf, buffer_size, conn_info);
    kurmcl_exchange_mr_info(conn_info, &my_buf_recv, remote_buf);
    barrier_info->sbuf = my_buf_send;
    barrier_info->remote = remote_buf;
    conn_info->barrier_info = barrier_info;
}

void kurmcl_barrier(kurmcl_conn_info_t *conn_info, int size, int rank)
{
    int adjsize, remote, mask;
    if (size == 1) {
        return;
    }
    adjsize = kurmcl_next_poweroftwo(size);
    adjsize >>= 1;

    buf_mr_info_t *sbuffer = &conn_info->barrier_info->sbuf;
    buf_mr_info_t *rbuf = conn_info->barrier_info->remote;
    kurmcl_iov_t iov[1];
    iov[0].len = 0;
    iov[0].local_buffer = sbuffer->buffer;
    iov[0].lkey = sbuffer->lkey;

    if (adjsize != size) {
        if (rank >= adjsize) {
            remote = rank - adjsize;
            iov[0].remote_buffer = rbuf[remote].buffer;
            iov[0].rkey = rbuf[remote].rkey;
            kurmcl_put(iov, 1, remote, 1, conn_info);
            kurmcl_flush(remote, conn_info);
            kurmcl_recv_imm_cnt(1, remote, conn_info);
        } else if (rank < (size - adjsize)) {
            remote = rank + adjsize;
            kurmcl_recv_imm_cnt(1, remote, conn_info);
        }
    }
    if (rank < adjsize) {
        mask = 0x1;
        while (mask < adjsize) {
            remote = rank ^ mask;
            mask <<= 1;
            if (remote >= adjsize) {
                continue;
            }
            iov[0].remote_buffer = rbuf[remote].buffer;
            iov[0].rkey = rbuf[remote].rkey;
            kurmcl_put(iov, 1, remote, 1, conn_info);
            kurmcl_flush(remote, conn_info);
            kurmcl_recv_imm_cnt(1, remote, conn_info);
        }
    }
    if (adjsize != size) {
        if (rank < (size - adjsize)) {
            remote = rank + adjsize;
            kurmcl_put(iov, 1, remote, 1, conn_info);
            kurmcl_flush(remote, conn_info);
        }
    }
    return;
}

#ifdef CQE_MODERATION
void kurmcl_flush_once (kurmcl_conn_info_t *conn_info) {

    kurmcl_iface_t *ifaces;
    struct ibv_cq* cq;
    ifaces = conn_info->ifaces;

    for (int ii = 0; ii < MAX_NIC_NUM; ii++) {
        cq = ifaces[ii].scq;
        int num_wcs = ifaces[ii].tx_max_poll;
        struct ibv_wc wc[num_wcs];
        int num_complete = ibv_poll_cq(cq, num_wcs, wc);
        if (num_complete < 0) {
            VERBS_CHECK(num_complete);
        }
        for (int i = 0; i < num_complete; i++) {
            if (wc[i].status != IBV_WC_SUCCESS) {
                printf("Send complete error: %s\n", ibv_wc_status_str(wc[i].status));
                exit(1);
            }
        }
    }
}
#endif

void kurmcl_flush(int count, kurmcl_conn_info_t *conn_info)
{
#ifdef CQE_MODERATION
     kurmcl_flush_once(conn_info);
#else
    kurmcl_iface_t *ifaces;

    struct ibv_cq *cq;
    struct ibv_wc *wc;
    int outstanding = count;
    ifaces = conn_info->ifaces;
    struct ibv_device_attr dev_attr;

    while (outstanding) {
        for (int ii = 0; ii < MAX_NIC_NUM; ii++) {
            cq = ifaces[ii].scq;
            dev_attr = ifaces[ii].dev_attr;
            wc = ifaces[ii].wc;
            int num_complete = ibv_poll_cq(cq, dev_attr.max_cqe, wc);
            if (num_complete < 0) {
                VERBS_CHECK(num_complete);
            }
            for (int i = 0; i < num_complete; i++) {
                if (wc[i].status != IBV_WC_SUCCESS) {
                    printf("Send complete error: %s\n", ibv_wc_status_str(wc[i].status));
                    exit(1);
                }
            }
            outstanding -= num_complete;
            if (outstanding < 0) {
                printf("some thing is wrong in sending\n");
            }
        }
    }
#endif
}

}

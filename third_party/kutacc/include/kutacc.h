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
#ifndef KUTACC_H
#define KUTACC_H

#include <vector>
#include <cstdint>
#include <functional>
#include <arm_bf16.h>
#include <kupl.h>

#ifdef __cplusplus
extern "C" {
#endif

/** @brief exposed KuTACC symbol table*/
#define kutacc_export           __attribute__((visibility("default")))

/** @brief status in KuTACC library*/
#define KUTACC_OK 0
#define KUTACC_ERROR (-1)

/** @brief KuTACC version info*/
typedef struct kutacc_version {
    const char *product_name;
    const char *product_version;
    const char *component_name;
    const char *component_version;
    const char *component_appendinfo;
} kutacc_version_t;

/*
 * @brief get the kutacc version info
 * @param [out] version     the kutacc version info
 *
 * @return KUTACC_OK for get version info success, other for failed.
 */
kutacc_export int kutacc_get_version(kutacc_version_t *version);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
namespace kutacc {
/*! \enum DType
 *  \brief datatype of tensor.
 */
typedef enum TensorDataType {
    kInt64 = 0,
    kBF16 = 1,
    kNumTypes
} DType;

/** @brief KuTACC kml extend param */
struct BlasExtendParams {
    int64_t num_threads;
    bool prepack_a;
    bool prepack_b;
    bool row_bias;
    bool col_bias;
    void *bias;
    bool relu;
};

struct Tensor;

struct kutacc_export TensorWrapper {
private:
    Tensor *tensor_;

public:
    TensorWrapper(void *data_ptr, std::vector<int64_t> sizes, std::vector<int64_t> strides, int64_t dim, DType dtype);

    void* data() const;
    std::vector<int64_t> sizes() const;
    std::vector<int64_t> strides() const;
    DType dtype() const;
    int64_t dim() const;
};

/** @brief the parameters of FlashMLA forward function */
struct FlashMLAFwdParams {
    using index_t = int64_t;

    int b, seqlen_q, d, d_v;
    int h, ngroups;
    bool is_causal;
    float scale_softmax, scale_softmax_log2;
    int *cu_seqlens_k;

    void *q_ptr;
    void *k_ptr;
    void *v_ptr;
    void *o_ptr;
    void *softmax_lse_ptr;

    index_t q_batch_stride;
    index_t k_batch_stride;
    index_t v_batch_stride;
    index_t o_batch_stride;
    index_t q_row_stride;
    index_t k_row_stride;
    index_t v_row_stride;
    index_t o_row_stride;
    index_t q_head_stride;
    index_t k_head_stride;
    index_t v_head_stride;
    index_t o_head_stride;

    int *block_table;
    index_t block_table_batch_stride;
    int page_block_size;

    int *tile_scheduler_metadata_ptr;
    int num_thread_parts;
    int *num_splits_ptr;

    void *softmax_lseaccum_ptr;
    void *oaccum_ptr;

    void *tiling_buffer_ptr = nullptr;
};

/**
 * @brief run FlashMLA forward function
 *
 * @tparam T                                        the data type of the elements
 * @tparam head_dim                                 the qk head_dim of FlashMLA
 * @param [in, out] params                          the parameters of FlashMLA forward function
 */
template <typename T, int head_dim>
kutacc_export void flash_mla_run_fwd(const FlashMLAFwdParams &params);

/** @brief the parameters of FlashMLA metadata */
struct FlashMLAMetadataParams {
    int *seqlens_k_ptr;
    int *tile_scheduler_metadata_ptr;
    int *num_splits_ptr;

    int batch_size;
    int block_size_n;
    int fixed_overhead_num_blocks = 1;
    int num_thread_parts;
};

/**
 * @brief get constant of FlashMLA
 *
 * @param [out] extra_buffer_size_per_thread        the size of extra buffer required by each thread
 * @param [out] tile_scheduler_metadata_size        the size of tile_scheduler_metadata
 * @param [out] block_size_m                        the block size of dimension "m"
 * @param [out] block_size_m                        the block size of dimension "n"
 */
kutacc_export void flash_mla_get_constant(int &extra_buffer_size_per_thread,
                                          int &tile_scheduler_metadata_size,
                                          int &block_size_m, int &block_size_n);

/**
 * @brief get metadata of FlashMLA
 *
 * @param [in, out] params                          the parameters of FlashMLA metadata
 */
kutacc_export void flash_mla_get_metadata(const FlashMLAMetadataParams &params);


constexpr int64_t FUSEDMOE_TILEBUF = 64;
using MatrixTilingBlock = std::tuple<int64_t, int64_t, int64_t>;

/**
 * @brief gemm tiling plan initialize
 */
kutacc_export void gemm_init_tiling_plan();

/**
 * @brief find igemm optimal tiling plan
 */
kutacc_export MatrixTilingBlock igemm_find_optimal_tiling_plan(int64_t M, int64_t N, int64_t K, int64_t num_threads);

/**
 * @brief find bgemm optimal tiling plan
 */
kutacc_export MatrixTilingBlock bgemm_find_optimal_tiling_plan(int64_t M, int64_t N, int64_t K, int64_t num_threads);

/**
 * @brief int8 gemm kernels
 */
kutacc_export void igemm_bdq(int64_t m, int64_t n, int64_t k, MatrixTilingBlock t, int8_t *act_ptr,
	int8_t *weight_ptr, float *act_scale_ptr, float *weight_scale_ptr, bfloat16_t *output_ptr, bfloat16_t *tmpc);

kutacc_export void igemm_pack(int64_t r, int64_t c, int64_t split_r, int64_t split_c, int8_t *input_ptr,
	int8_t *output_ptr, bool with_idx = 0, int *idx = nullptr, int64_t ldc = 0);

kutacc_export void igemm_gateup(int64_t total_bs, int64_t K, int64_t N, int64_t num_experts, int64_t lda,
	int64_t ldas, int8_t *acts, int8_t *weights, float *acts_scale, float *weights_scale, int *token_ids,
	int *experts_offset, bfloat16_t *output, int8_t *tmpx, float *tmpy, float *tmp_scales);

kutacc_export void igemm_down(int64_t total_bs, int64_t K, int64_t N, int64_t num_experts, int8_t *acts,
	int8_t *weights, float *acts_scale, float *weights_scale, int *experts_offset, bfloat16_t *output,
	int8_t *tmpx, float *tmpy);

/**
 * @brief bf16 gemm kernels
 */
kutacc_export void bgemm(int64_t m, int64_t n, int64_t k, MatrixTilingBlock t, bfloat16_t *act_ptr, bfloat16_t *weight_ptr,
	bfloat16_t *output_ptr, bfloat16_t *tmpc);

kutacc_export void bgemm_pack(int64_t r, int64_t c, int64_t split_r, int64_t split_c, bfloat16_t *input_ptr,
    bfloat16_t *output_ptr);

/**
 * @brief batched gemm kernels
 */
kutacc_export void batched_gemm_pack(int64_t bs, int64_t m, int64_t n, int64_t stride_bs, int64_t stride_m,
	void *src, void *dst, int64_t dtype);

kutacc_export void batched_gemm_woqs8(int64_t bs, int64_t m, int64_t n, int64_t k, int64_t stride_bs, int64_t stride_m,
    bfloat16_t *act, int8_t *weight, bfloat16_t *out, float *rscale, float *cscale);

/**
 * @brief quant kernels
 */
kutacc_export void quant(int64_t height, int64_t width, const __bf16 *input, int64_t input_stride,
	int8_t *out, int64_t out_stride, float *scale);

kutacc_export void quant_pack(bfloat16_t *input_data, int8_t *output_data, float *scale_data,
	int hidden_size, int num_tokens);

/**
 * @brief rmsnorm kernels
 */
template <typename scalar_t, bool has_residual>
kutacc_export void rmsnorm(int64_t height, int64_t width, scalar_t *acts, int64_t acts_stride,
	const scalar_t *weights, float eps, scalar_t *residual, scalar_t *outs);

template <typename scalar_t>
kutacc_export void head_rmsnorm(int64_t token_num, int64_t head_num, int64_t head_dim, scalar_t *acts,
    int64_t acts_stride, const scalar_t *weights, float eps, scalar_t *outs);

template <typename scalar_t, bool has_residual>
kutacc_export void rmsnorm_quant(int64_t height, int64_t width, scalar_t *acts, int64_t acts_stride,
	const scalar_t *weights, float eps, scalar_t *residual, int8_t *outs, float *scales);

/**
 * @brief embedding kernel
 */
kutacc_export void embedding(const int64_t *input, const void *weight_, void *out_,
	int64_t element_size, int64_t n_tokens, int64_t hidden, int64_t vocab_start, int64_t vocab_end);

typedef enum kurmcl_datatype {
    KURMCL_DATATYPE_CHAR = 0,
    KURMCL_DATATYPE_INT,
    KURMCL_DATATYPE_LONG,
    KURMCL_DATATYPE_FLOAT,
    KURMCL_DATATYPE_DOUBLE
} kurmcl_datatype_t;

typedef int (*kurmcl_oob_allgather_cb_t)(const void *sendbuf, void *recvbuf, int count, int group,
                                         kurmcl_datatype_t datatype);

typedef int (*kurmcl_oob_barrier_cb_t)(int group);

typedef int (*kurmcl_oob_alltoall_cb_t)(const void *sendbuf, int sendcount, kurmcl_datatype_t sendtype,
                                        void *recvbuf, int recvcount, kurmcl_datatype_t recvtype, int group);

typedef struct kurmcl_oob_cb {
    kurmcl_oob_allgather_cb_t oob_allgather;
    kurmcl_oob_barrier_cb_t oob_barrier;
    kurmcl_oob_alltoall_cb_t oob_alltoall;
} kurmcl_oob_cb_t, *kurmcl_oob_cb_h;

/** @brief the parameters of kurmcl communicator */
typedef struct kurmcl_conn_info* kurmcl_conn_info_h;

/**
 * @brief kurmcl communicator create function.
 *
 * @param [in] size        the size of communicator
 * @param [in] rank        the rank of communicator
 * @param [in] oob_cbs     out of band callback function
 * @param [in] group       mpi communicator
 * @param [out] conn_info  kurmcl communicator
 */
kutacc_export int kurmcl_comm_create(int size, int rank, kurmcl_oob_cb_h oob_cbs, int group, kurmcl_conn_info_h *conn_info);

/**
 * @brief kurmcl barrier function.
 *
 * @param [in] conn_info   kurmcl communicator
 * @param [in] size        the size of communicator
 * @param [in] rank        the rank of communicator
 */
kutacc_export void kurmcl_barrier(kurmcl_conn_info_h conn_info, int size, int rank);

/**
 * @brief initializ dispatch related data structures, register memory and address exchange
 */
kutacc_export void moe_dispatch_init(uint8_t *x_data, uint8_t *packed_recv_x_data, int16_t *recv_src_info_data,
                                     int64_t num_experts, int64_t num_max_dispatch_tokens_per_rank, int64_t hidden,
                                     int64_t num_tokens, int64_t num_ranks_, int64_t my_rank_, int16_t *src_info_,
                                     void* disbuf_baseptr_, kurmcl_conn_info_h ds_conn_info_, int16_t *parallel_sizes_data_);

/**
 * @brief based on the topk information, send the token to the corresponding expert
 */
kutacc_export void moe_dispatch_send(uint8_t *x_data, int16_t *topk_idx_data, int64_t num_tokens, int64_t num_topk,
                                     int64_t num_experts, int64_t num_max_dispatch_tokens_per_rank, int64_t hidden,
                                     int16_t *parallel_sizes_data, uint8_t *packed_recv_x_data,
                                     int16_t *recv_src_info_data, int64_t batch_id);

/**
 * @brief block until the required token is received
 */
kutacc_export void moe_dispatch_recv(int16_t *recv_src_info_data, int64_t num_experts,
                                     int64_t num_max_dispatch_tokens_per_rank, int64_t batch_id);

/**
 * @brief release dispatch related data structures
 */
kutacc_export void moe_dispatch_finalize();

/**
 * @brief initializ combine related data structures, register memory and address exchange
 */
kutacc_export void moe_combine_init(bfloat16_t *new_packed_recv_x_data, int64_t num_tokens, int16_t num_experts,
                                    int16_t num_topk, int16_t num_max_dispatch_tokens_per_rank, int16_t hidden,
                                    std::vector<bfloat16_t *>&& group_ptr, std::vector<uint8_t *>&& meta_group_ptr,
                                    int local_size_, int local_rank_, bfloat16_t *tmpx_for_sum_);

/**
 * @brief send the activation value back to the original expert
 */
kutacc_export void moe_combine_send(bfloat16_t *x_data, int16_t *src_info_data, int64_t num_tokens,
                                    int64_t num_max_dispatch_tokens_per_rank, int64_t num_experts, int64_t hidden,
                                    int16_t *parallel_sizes_data, int64_t batch_id);

/**
 * @brief confirm receipt of activation value ans multiply-add with weight
 */
kutacc_export void moe_combine_recv(bfloat16_t *combined_x_data, int16_t *topk_idx_data, float *topk_weights_data,
                                    int64_t num_tokens, int64_t num_experts, int64_t num_max_dispatch_tokens_per_rank,
                                    int64_t num_topk, int64_t hidden, int16_t *parallel_sizes_data, int64_t batch_id, kupl_shm_win_h win);

/**
 * @brief release combine related data structures
 */
kutacc_export void moe_combine_finalize();

/**
 * @brief obtain performance data during the dispatch phase
 */
kutacc_export void moe_dispatch_get_perf(double *perf_ary);

/**
 * @brief obtain performance data during the combine phase
 */
kutacc_export void moe_combine_get_perf(double *perf_ary);

/**
 * @brief get the metadate of shm_allreduce
 *
 * @param [in] max_num_elements                     the maximum number of elements
 * @param [out] extra_size                          the extra size required by shm_allreduce
 * @param [in, out] use_sdma                        if use sdma
 */
kutacc_export void shm_allreduce_get_metadata(size_t max_num_elements, size_t &extra_size, bool &use_sdma);

/**
 * @brief shm_allreduce initialize
 *
 * @param [in] rank                                 the rank of current process in the node
 * @param [in] max_num_elements                     the maximum number of elements
 * @param [in] intra_node_extra_buffers             the extra buffers in the node (guaranteed to be shared memory)
 */
kutacc_export void shm_allreduce_init(int rank, size_t max_num_elements, bfloat16_t **intra_node_extra_buffers);

/**
 * @brief shm_allreduce finalize
 */
kutacc_export void shm_allreduce_finalize();

/**
 * @brief allreduce by shared memory
 *
 * @param [in] rank                                 the rank of current process in the node
 * @param [in] num_elements                         the maximum number of elements
 * @param [in, out] intra_node_buffers              the communication buffers in the node (guaranteed to be shared memory)
 * @param [in] kupl_win_intra_die                   the kupl win of the die
 * @param [in] kupl_win_intra_socket                the kupl win of the socket
 * @param [in] world_size                         intra-node process count: 4, 8, or 16
 */
kutacc_export void shm_allreduce(int rank, int num_elements, bfloat16_t **intra_node_buffers,
                                 kupl_shm_win_h &kupl_win_intra_die, kupl_shm_win_h &kupl_win_intra_socket,
                                 int world_size = 16);

/**
 * @brief allgather by shared memory
 *
 * @tparam scalar_t                                 the data type of the elements
 * @tparam comm_size                                the communication size of allgather
 * @tparam is_hierarchical                          if use hierarchical communication
 * @param [in] batch                                the batch size of the buffer
 * @param [in] sendbuf                              the allgather sendbuf
 * @param [in] sendcount                            the number of elements in sendbuf
 * @param [out] recvbuf                             the allgather recvbuf
 * @param [in] recvcount                            the number of elements in recvbuf
 * @param [in] rank                                 the rank of current process in the node
 * @param [in] buffers                              the shared memory buffers in the node
 * @param [in] buffer_size                          the size of each shared memory buffer
 * @param [in] barrier                              the barrier function of the communication
 */
template <typename scalar_t, int comm_size, bool is_hierarchical>
kutacc_export void shm_allgather(int64_t batch, const scalar_t *sendbuf, int64_t sendcount, scalar_t *recvbuf,
                                 int64_t recvcount, int64_t rank, uint8_t **buffers, int64_t buffer_size,
                                 const std::function<void()> &barrier);

/**
 * @brief get the total number of threads
 *
 * @return int64_t                                  the total number of threads
 */
kutacc_export int64_t get_thread_num();

/**
 * @brief get the id of the current thread
 *
 * @return int64_t                                  the id of the current thread
 */
kutacc_export int64_t get_thread_id();

/**
 * @brief create a parallel for the specific function in the specific range
 *
 * @param [in] begin                                the begin of the range
 * @param [in] end                                  the end of the range
 * @param [in] grain_size                           the grain size for each thread
 * @param [in] func                                 the function for this parallel
 */
kutacc_export void parallel_for(int64_t begin, int64_t end, int64_t grain_size,
								const std::function<void(int64_t, int64_t)> &func);

/**
 * @brief barrier in parallel region
 */
kutacc_export void parallel_region_barrier();

} // namespace kutacc

#endif
#endif

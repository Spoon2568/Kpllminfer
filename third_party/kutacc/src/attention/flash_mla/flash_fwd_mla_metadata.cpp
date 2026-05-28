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
#include "common.h"

namespace kutacc {

void flash_mla_get_metadata(const FlashMLAMetadataParams &params)
{
    int *seqlens_k_ptr = params.seqlens_k_ptr;
    int *tile_scheduler_metadata_ptr = params.tile_scheduler_metadata_ptr;
    int *num_splits_ptr = params.num_splits_ptr;

    int batch_size = params.batch_size;
    int block_size_n = params.block_size_n;
    int64_t fixed_overhead_num_blocks = params.fixed_overhead_num_blocks;
    int num_thread_parts = params.num_thread_parts;

    KUTACC_CHECK(batch_size > 0, "");

    int *num_blocks = (int *)malloc(batch_size * sizeof(int));

    int64_t total_num_blocks = 0;
    for (int i = 0; i < batch_size; ++i) {
        num_blocks[i] = ceil_div(seqlens_k_ptr[i], block_size_n);
        total_num_blocks += num_blocks[i] + fixed_overhead_num_blocks;
    }
    total_num_blocks -= fixed_overhead_num_blocks;

    int64_t payload = ceil_div(total_num_blocks, (int64_t)num_thread_parts);

    int now_idx = 0;
    int now_block = 0;
    int now_n_split_idx = 0;
    int cum_num_splits = 0;
    num_splits_ptr[0] = 0;
    for (int i = 0; i < num_thread_parts; ++i) {
        int *tile_scheduler_metadata = tile_scheduler_metadata_ptr + i * TileSchedulerMetaDataSize;

        tile_scheduler_metadata[0] = now_idx;
        tile_scheduler_metadata[1] = now_block * block_size_n;
        tile_scheduler_metadata[4] = now_n_split_idx;

        int64_t remain_payload = payload;
        while (now_idx < batch_size) {
            int now_remain_blocks = num_blocks[now_idx] - now_block;
            if (remain_payload >= now_remain_blocks) {
                cum_num_splits += now_n_split_idx + 1;
                num_splits_ptr[now_idx + 1] = cum_num_splits;
                remain_payload -= now_remain_blocks + fixed_overhead_num_blocks;
                ++now_idx;
                now_block = 0;
                now_n_split_idx = 0;
            } else {
                if (remain_payload > 0) {
                    now_block += remain_payload;
                    ++now_n_split_idx;
                    remain_payload = 0;
                }
                break;
            }
        }

        tile_scheduler_metadata[2] = now_block > 0 ? now_idx : now_idx - 1;
        tile_scheduler_metadata[3] = now_block > 0 ? now_block * block_size_n : seqlens_k_ptr[now_idx - 1];
    }

    KUTACC_CHECK(now_idx == batch_size && now_block == 0 && now_n_split_idx == 0, "");

    free(num_blocks);
}

void flash_mla_get_constant(int &extra_buffer_size_per_thread_, int &tile_scheduler_metadata_size, int &block_size_m,
    int &block_size_n)
{
    extra_buffer_size_per_thread_ = extra_buffer_size_per_thread;
    tile_scheduler_metadata_size = TileSchedulerMetaDataSize;
    block_size_m = BlockSizeM;
    block_size_n = BlockSizeN;
}

} // namespace kutacc
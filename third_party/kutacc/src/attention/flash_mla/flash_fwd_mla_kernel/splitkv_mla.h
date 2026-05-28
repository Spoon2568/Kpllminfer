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

#include "kutacc.h"
#include "../common.h"
#include "mla_kernel.h"

namespace kutacc {

template <typename Kernel_traits>
void run_flash_splitkv_fwd_mla(const FlashMLAFwdParams &params)
{
    KUTACC_CHECK(params.page_block_size == Kernel_traits::kBlockN, "");
    
    BOOL_SWITCH(params.is_causal, Is_causal, [&] {
        auto kernel = &flash_fwd_splitkv_mla_kernel<Kernel_traits, Is_causal>;
        kernel(params);
    });

    MLA_NUM_SPLITS_SWITCH(params.num_thread_parts, kMaxSplits, [&] {
        auto combine_kernel = &flash_fwd_splitkv_mla_combine_kernel<typename Kernel_traits::Element,
                                                                    typename Kernel_traits::ElementAccum,
                                                                    typename Kernel_traits::index_t,
                                                                    Kernel_traits::kHeadDimV, kMaxSplits>;
        combine_kernel(params);
    });
}

template <typename T, int Headdim>
void flash_mla_run_fwd(const FlashMLAFwdParams &params)
{
    static_assert(Headdim == 576);
    KUTACC_CHECK(params.d_v == 512, "");
    KUTACC_CHECK(params.k_ptr == params.v_ptr, "");  // Shared_KV
    
    using Kernel_traits = Flash_fwd_kernel_traits_mla<576, BlockSizeM, BlockSizeN, T, 512>;

    run_flash_splitkv_fwd_mla<Kernel_traits>(params);
}

} // namespace kutacc
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

namespace kutacc {

std::unordered_map<MatrixTilingBlock, MatrixTilingBlock, tiling_block_hash> tiling_plan = {};
std::unordered_map<MatrixTilingBlock, MatrixTilingBlock, tiling_block_hash> bgemm_tiling_plan = {};
std::unordered_map<MatrixTilingBlock, MatrixTilingBlock, tiling_block_hash> tiling_plan_32 = {
    {{16, 1536, 1536}, {8, 96, 1536}},
    {{32, 1536, 1536}, {16, 96, 1536}},
    {{64, 1536, 1536}, {32, 96, 1536}},
    {{128, 1536, 1536}, {64, 96, 1536}},
    {{256, 1536, 1536}, {128, 96, 1536}},
    {{32, 4096, 7168}, {32, 512, 1792}},
    {{32, 7168, 2048}, {32, 448, 1024}},
    {{64, 4096, 7168}, {64, 512, 1792}},
    {{64, 7168, 2048}, {64, 448, 1024}},
    {{128, 4096, 7168}, {128, 512, 1792}},
    {{128, 7168, 2048}, {128, 448, 1024}},
    // {{128, 768, 4096}, {16, 192, 4096}},
    {{64, 264, 7168}, {64, 132, 448}},
    {{64, 7168, 1024}, {64, 224, 1024}},
    {{64, 256, 7168}, {64, 128, 448}},
    {{64, 7168, 128}, {64, 224, 128}},
    {{64, 2304, 7168}, {64, 288, 1792}},
    {{64, 7168, 1152}, {64, 224, 1152}},
    // {{64, 3072, 4096}, {16, 384, 4096}},
    // {{64, 4096, 1536}, {32, 256, 1536}},
    {{128, 264, 7168}, {128, 132, 448}},
    {{128, 7168, 1024}, {128, 224, 1024}},
    {{128, 256, 7168}, {128, 128, 448}},
    {{128, 7168, 128}, {128, 224, 128}},
    {{128, 2304, 7168}, {128, 288, 1792}},
    {{128, 7168, 1152}, {128, 224, 1152}},
    {{256, 264, 7168}, {256, 132, 448}},
    {{256, 7168, 1024}, {256, 224, 1024}},
    {{256, 256, 7168}, {256, 128, 448}},
    {{256, 7168, 128}, {256, 224, 128}},
    {{256, 2304, 7168}, {256, 288, 1792}},
    {{256, 7168, 1152}, {256, 224, 1152}},
};
std::unordered_map<MatrixTilingBlock, MatrixTilingBlock, tiling_block_hash> bgemm_tiling_plan_32 = {
    {{16, 16, 7168}, {16, 16, 224}},
    {{16, 8080, 7168}, {16, 1010, 1792}},
    {{16, 9496, 4096}, {16, 1187, 1024}},
    {{32, 16, 7168}, {32, 16, 224}},
    {{32, 8080, 7168}, {32, 1010, 1792}},
    {{32, 9496, 4096}, {32, 1187, 1024}},
    {{32, 18992, 4096}, {32, 1187, 2048}},
    {{64, 16, 7168}, {64, 16, 224}},
    {{64, 8080, 7168}, {64, 1010, 1792}},
    {{64, 9496, 4096}, {64, 1187, 1024}},
    {{64, 18992, 4096}, {64, 1187, 2048}},
    {{96, 16, 7168}, {96, 16, 224}},
    {{96, 8080, 7168}, {96, 1010, 1792}},
    {{96, 9496, 4096}, {96, 1187, 1024}},
    {{96, 18992, 4096}, {96, 1187, 2048}},
    {{128, 16, 7168}, {128, 16, 224}},
    {{128, 8080, 7168}, {128, 1010, 1792}},
    {{128, 9496, 4096}, {128, 1187, 1024}},
    {{128, 18992, 4096}, {128, 1187, 2048}},
};

int64_t max_power_of_two_factor(int64_t num)
{
    if (num == 0) {
        return 0;
    }
    uint64_t x = (num & (-num));
    uint64_t n = 0;
    if (!(x >> 16)) {
        n += 16, x <<= 16;
    }
    if (!(x >> 24)) {
        n += 8, x <<= 8;
    }
    if (!(x >> 28)) {
        n += 4, x <<= 4;
    }
    if (!(x >> 30)) {
        n += 2, x <<= 2;
    }
    n += (x >> 31) ^ 1;  // 处理最高位
    return 31 - n;
}

int64_t get_value(MatrixTilingBlock t)
{
    return 2LL * std::get<0>(t) * std::get<1>(t) + 1LL * std::get<1>(t) * std::get<2>(t) +
           1LL * std::get<0>(t) * std::get<2>(t);
}

MatrixTilingBlock igemm_find_optimal_tiling_plan(int64_t M, int64_t N, int64_t K, int64_t num_threads)
{
    auto save_tiling_plan = tiling_plan.find(MatrixTilingBlock(M, N, K));
    if (save_tiling_plan != tiling_plan.end()) {
        auto ret = save_tiling_plan->second;
        return save_tiling_plan->second;
    }
    MatrixTilingBlock ans;
    int64_t minarg = 1LL * M * N * K * num_threads;
    int64_t count = 0;
    for (int64_t i = 0; i <= max_power_of_two_factor(M); ++i) {
        for (int64_t j = 0; j <= max_power_of_two_factor(N); ++j) {
            for (int64_t k = 0; k <= max_power_of_two_factor(K); ++k) {
                if ((1 << i) * (1 << j) * (1 << k) != num_threads) {
                    continue;
                }
                MatrixTilingBlock tmp(M >> i, N >> j, K >> k);
                int64_t arg = get_value(tmp) * num_threads + 2l * (1 << k) * M * N;
                if (arg < minarg) {
                    minarg = arg, ans = tmp;
                }
            }
        }
    }
    tiling_plan[MatrixTilingBlock(M, N, K)] = ans;
    return ans;
}

MatrixTilingBlock bgemm_find_optimal_tiling_plan(int64_t M, int64_t N, int64_t K, int64_t num_threads)
{
    auto save_tiling_plan = bgemm_tiling_plan.find(MatrixTilingBlock(M, N, K));
    if (save_tiling_plan != bgemm_tiling_plan.end()) {
        auto ret = save_tiling_plan->second;
        return save_tiling_plan->second;
    }
    return {M, N, K / num_threads};
}

auto get_idxs(int64_t tid, int64_t bm, int64_t bn, int64_t bk)
{
    int64_t idk = tid % bk;
    tid /= bk;
    int64_t idm = tid % bm;
    tid /= bm;
    int64_t idn = tid % bn;
    tid /= bn;
    return std::tuple(idm, idn, idk);
}

} // namespace kutacc
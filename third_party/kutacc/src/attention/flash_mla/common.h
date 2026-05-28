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

#include <sys/time.h>

#include "math/fast_exp.h"
#include "utils/persistent_buffer.h"
#include "utils/check.h"

#define BOOL_SWITCH(COND, CONST_NAME, ...)                                                                  \
    [&] {                                                                                                   \
        if (COND) {                                                                                         \
            constexpr static bool CONST_NAME = true;                                                        \
            return __VA_ARGS__();                                                                           \
        } else {                                                                                            \
            constexpr static bool CONST_NAME = false;                                                       \
            return __VA_ARGS__();                                                                           \
        }                                                                                                   \
    }()


#define MLA_NUM_SPLITS_SWITCH(NUM_SPLITS, NAME, ...)                                                        \
    [&] {                                                                                                   \
        if (NUM_SPLITS <= 32) {                                                                             \
            constexpr static int NAME = 32;                                                                 \
            return __VA_ARGS__();                                                                           \
        } else if (NUM_SPLITS <= 64) {                                                                      \
            constexpr static int NAME = 64;                                                                 \
            return __VA_ARGS__();                                                                           \
        } else if (NUM_SPLITS <= 96) {                                                                      \
            constexpr static int NAME = 96;                                                                 \
            return __VA_ARGS__();                                                                           \
        } else if (NUM_SPLITS <= 128) {                                                                     \
            constexpr static int NAME = 128;                                                                \
            return __VA_ARGS__();                                                                           \
        } else if (NUM_SPLITS <= 160) {                                                                     \
            constexpr static int NAME = 160;                                                                \
            return __VA_ARGS__();                                                                           \
        } else {                                                                                            \
            KUTACC_CHECK(false, "");                                                                        \
        }                                                                                                   \
    }()

namespace kutacc {

template <typename T>
constexpr T ceil_div(const T &x, const T &y)
{
    return (x + y - 1) / y;
}

inline double get_clock_us()
{
    struct timeval clock;
    gettimeofday(&clock, NULL);
    return 1000000.0 * clock.tv_sec + clock.tv_usec;
}

static constexpr int extra_buffer_size_per_thread = 256 * 1024;

static constexpr int TileSchedulerMetaDataSize = 8;
// [begin_idx, begin_seqlen, end_idx, end_seqlen, begin_n_split_idx, _, _, _]

static constexpr int BlockSizeM = 32;
static constexpr int BlockSizeN = 64;

} // namespace kutacc
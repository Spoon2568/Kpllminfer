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
#include "embedding.h"

namespace kutacc {

void embedding_kernel(const int64_t *input, const void *weight_, void *out_, int64_t element_size, int64_t n_tokens,
    int64_t hidden, int64_t vocab_start, int64_t vocab_end)
{
    auto weight = reinterpret_cast<const uint8_t *>(weight_);
    auto out = reinterpret_cast<uint8_t *>(out_);
    auto hidden_bytes = hidden * element_size;
    for (int64_t i = 0; i < n_tokens; i++) {
        if (input[i] >= vocab_start && input[i] < vocab_end) {
            std::memcpy(out + i * hidden_bytes, weight + (input[i] - vocab_start) * hidden_bytes, hidden_bytes);
        } else {
            std::memset(out + i * hidden_bytes, 0, hidden_bytes);
        }
    }
}

void embedding(const int64_t *input, const void *weight_, void *out_, int64_t element_size, int64_t n_tokens,
    int64_t hidden, int64_t vocab_start, int64_t vocab_end)
{
    auto weight = reinterpret_cast<const uint8_t *>(weight_);
    auto out = reinterpret_cast<uint8_t *>(out_);
    auto hidden_bytes = hidden * element_size;
    kutacc::parallel_for(0, n_tokens, 1, [&](int64_t start, int64_t end) {
        embedding_kernel(input + start, weight, out + start * hidden_bytes, element_size, end - start, hidden,
            vocab_start, vocab_end);
    });
}

}
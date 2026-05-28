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
#ifndef KUTACC_TENSOR_H
#define KUTACC_TENSOR_H

#include <vector>
#include "kutacc.h"
#include "../utils/check.h"

namespace kutacc {
struct SimpleTensor {
    int64_t dim;
    void *data_ptr;
    std::vector<int64_t> sizes;
    std::vector<int64_t> strides;
    DType dtype;

    SimpleTensor(void *data_ptr, std::vector<int64_t> sizes, std::vector<int64_t> strides,
        int64_t dim, ::kutacc::DType dtype)
        : dim(dim), data_ptr(data_ptr), sizes(sizes), strides(strides), dtype(dtype)
    {
        KUTACC_CHECK(dim == (int64_t) sizes.size(), dim, " ", sizes.size());
        KUTACC_CHECK(dim == (int64_t) strides.size(), dim, " ", strides.size());
    }
};

struct Tensor {
    SimpleTensor data;

    Tensor(void *data_ptr, std::vector<int64_t> sizes, std::vector<int64_t> strides, int64_t dim, ::kutacc::DType dtype)
        : data(SimpleTensor(data_ptr, sizes, strides, dim, dtype)) {};
};
}   // namespace kutacc
#endif

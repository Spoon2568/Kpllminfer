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
#include "tensor.h"
#include <vector>
#include "kutacc.h"
#include "../utils/check.h"

namespace kutacc {
TensorWrapper::TensorWrapper(void *data_ptr, std::vector<int64_t> sizes, std::vector<int64_t> strides, int64_t dim,
    DType dtype)
{
    tensor_ = new Tensor(data_ptr, sizes, strides, dim, dtype);
}

void* TensorWrapper::data() const
{
    return tensor_->data.data_ptr;
}

std::vector<int64_t> TensorWrapper::sizes() const
{
    return tensor_->data.sizes;
}

std::vector<int64_t> TensorWrapper::strides() const
{
    return tensor_->data.strides;
}

DType TensorWrapper::dtype() const
{
    return tensor_->data.dtype;
}

int64_t TensorWrapper::dim() const
{
    return tensor_->data.dim;
}
}   // namespace kutacc

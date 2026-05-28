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

const kutacc_version_t g_version = {
    .product_name = "Kunpeng HPCKit",
    .product_version = "25.1.RC1",
    .component_name = "KuTACC",
    .component_version = "25.1.RC1",
#if defined(__clang__)
    .component_appendinfo = "bisheng",
#elif defined(__GNUC__)
    .component_appendinfo = "gcc",
#endif
};

int kutacc_get_version(kutacc_version_t *version)
{
    if (version == nullptr) {
        return KUTACC_ERROR;
    }
    *version = g_version;
    return KUTACC_OK;
}

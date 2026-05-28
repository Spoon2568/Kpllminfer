cmake_minimum_required(VERSION 3.14.1)
#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under a modified version of the MIT license. See LICENSE in the project root for license information.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#

project(kutacc VERSION 1.0.0 LANGUAGES C CXX ASM)

# colored output
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
set(${COLOR_FLAGS} "-fdiagnostics-color=always")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
set(${COLOR_FLAGS} "-fcolor-diagnostics")
endif()

# Set library's name
set(LIBKUTACC_NAME "kutacc")

# Set kutacc root path
set(KUTACC_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# Set cmake include directories
set(KUTACC_CMAKE_INCLUDE_PATH ${KUTACC_ROOT_DIR}/cmake)

# Set installation directories
set(KUTACC_INSTALL_INCLUDEDIR ${CMAKE_INSTALL_PREFIX}/include)
set(KUTACC_INSTALL_LIBDIR ${CMAKE_INSTALL_PREFIX}/lib)

# Set compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${COLOR_FLAGS}")
add_compile_options(-march=armv8.2-a+sve2+bf16+sme -msve-vector-bits=512)

if (KUTACC_PARALLEL_BACKEND STREQUAL "kspin")
    add_subdirectory(src/kspin)
endif()

add_subdirectory(src)
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

# Check debug mode
if (CMAKE_BUILD_TYPE MATCHES "Debug")
    set(ENABLE_KUTACC_DEBUG ON)
    set(ENABLE_KUTACC_ASSERT ON)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wfloat-equal -fno-common -march=armv8.2-a+bf16+sve -msve-vector-bits=512 -fPIC -fopenmp")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -Wfloat-equal -fno-common -march=armv8.2-a+bf16+sve -msve-vector-bits=512 -fPIC -fopenmp")
    set(CMAKE_ASM_FLAGS "${CMAKE_ASM_FLAGS} -Wall -Wextra -Wfloat-equal -fno-common -march=armv8.2-a+bf16+sve -msve-vector-bits=512 -fPIC -fopenmp")
    add_compile_options(-O0 -g -gdwarf-4)
    message("-- Caution : using debug mode, performance is bad, just for develop")
elseif (CMAKE_BUILD_TYPE MATCHES "Release")
    set(CMAKE_ASM_FLAGS "${CMAKE_ASM_FLAGS} -mcpu=native")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mcpu=native -Wno-undefined-arm-za -Wno-undefined-arm-streaming -rtlib=compiler-rt")
    if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-missing-field-initializers")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-missing-field-initializers")
        set(CMAKE_ASM_FLAGS "${CMAKE_ASM_FLAGS} -Wno-missing-field-initializers")
    endif()
    add_compile_options(-O3)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -stdlib=libc++")

include(${KUTACC_CMAKE_INCLUDE_PATH}/libkutacc_source_files.cmake)
libkutacc_source_files(KUTACC_SRCS)
add_library(${LIBKUTACC_NAME} SHARED ${KUTACC_SRCS})

if (KUTACC_PARALLEL_BACKEND STREQUAL "kspin")
    message("-- kutacc parallel backend is set to kspin")
    add_definitions(-DUSE_KSPIN_PARALLEL)
    target_link_libraries(${LIBKUTACC_NAME} kspin)
elseif (KUTACC_PARALLEL_BACKEND STREQUAL "omp")
    message("-- kutacc parallel backend is set to omp")
    add_definitions(-DUSE_OMP_PARALLEL)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
elseif (KUTACC_PARALLEL_BACKEND STREQUAL "kupl")
    message("-- kutacc parallel backend is set to kupl")
    add_definitions(-DUSE_KUPL_PARALLEL)
    target_link_libraries(${LIBKUTACC_NAME} kupl)
endif()

if (KUTACC_BUILD_KIND MATCHES "src")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=hidden")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fvisibility=hidden")
    set(CMAKE_ASM_FLAGS "${CMAKE_ASM_FLAGS} -fvisibility=hidden")
endif()

if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    if (KUTACC_BUILD_KIND MATCHES "test")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-inline -fprofile-arcs -ftest-coverage")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fno-inline -fprofile-arcs -ftest-coverage")
        set(CMAKE_ASM_FLAGS "${CMAKE_ASM_FLAGS} -fno-inline -fprofile-arcs -ftest-coverage")
    elseif (KUTACC_BUILD_KIND MATCHES "fuzz")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=hidden")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fvisibility=hidden")
        set(CMAKE_ASM_FLAGS "${CMAKE_ASM_FLAGS} -fvisibility=hidden")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-inline -fprofile-arcs -ftest-coverage -fsanitize-coverage=trace-pc")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fno-inline -fprofile-arcs -ftest-coverage -fsanitize-coverage=trace-pc")
        set(CMAKE_ASM_FLAGS "${CMAKE_ASM_FLAGS} -fno-inline -fprofile-arcs -ftest-coverage -fsanitize-coverage=trace-pc")
    endif()
endif()

if (CMAKE_BUILD_TYPE MATCHES "Release")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_FORTIFY_SOURCE=2")
endif()

include_directories(${KUTACC_ROOT_DIR}/include)
include_directories(${KUTACC_ROOT_DIR}/src)
include_directories(${CMAKE_CURRENT_BINARY_DIR})

target_link_libraries(${LIBKUTACC_NAME} numa kupl)

target_link_libraries(${LIBKUTACC_NAME}
    /usr/lib64/libibverbs.so
    /usr/lib64/libibverbs/libcxgb4-rdmav34.so
    /usr/lib64/libibverbs/libsiw-rdmav34.so
    /usr/lib64/libibverbs/libvmw_pvrdma-rdmav34.so
    /usr/lib64/libmlx5.so
    /usr/lib64/libibverbs/librxe-rdmav34.so
    /usr/lib64/libefa.so
    /usr/lib64/libibverbs/libipathverbs-rdmav34.so
    /usr/lib64/libibverbs/libhfi1verbs-rdmav34.so
    /usr/lib64/libhns.so
    /usr/lib64/libibverbs/libmthca-rdmav34.so
    /usr/lib64/libibverbs/libqedr-rdmav34.so
    /usr/lib64/libmlx4.so
    /usr/lib64/libibverbs/libbnxt_re-rdmav34.so
    /usr/lib64/libibverbs/libocrdma-rdmav34.so
    /usr/lib64/libibverbs/libirdma-rdmav34.so)

add_definitions(-DCOMM_PERF)

set_target_properties(${LIBKUTACC_NAME} PROPERTIES VERSION 25.1.RC1)

#Install
libkutacc_header_install()
install(TARGETS ${LIBKUTACC_NAME} DESTINATION ${KUTACC_INSTALL_LIBDIR})


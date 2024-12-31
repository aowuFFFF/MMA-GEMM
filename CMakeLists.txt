# Copyright 2023. All Rights Reserved.
# Author: Bruce-Lee-LY
# Date: 20:42:28 on Sun, Feb 12, 2023
#
# Description: cmake for cuda hgemm

cmake_minimum_required (VERSION 3.20)

project (cuda_hgemm LANGUAGES C CXX CUDA)

set (CMAKE_VERBOSE_MAKEFILE ${HGEMM_VERBOSE_MAKEFILE})

# 检查CMake政策CMP0146的版本，如果存在，则将其设置为旧行为。这通常用于控制CMake的某些特性或行为的兼容性。
# if (POLICY CMP0146)
#     cmake_policy (SET CMP0146 OLD) 
# endif ()

# 查找ccache程序。如果找到，将其设置为全局编译和链接的规则，以加速后续的编译过程。
# find_program (CCACHE_FOUND ccache)
# if (CCACHE_FOUND)
#     set_property (GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
#     set_property (GLOBAL PROPERTY RULE_LAUNCH_LINK ccache)
# endif (CCACHE_FOUND)


# 设置C语言的编译标志：
# 使用C11标准。
# 调试模式下的标志：禁用优化，包含调试信息。
# 发布模式下的标志：启用最高级别的优化。
# set (CMAKE_C_FLAGS "-std=c11")
# set (CMAKE_C_FLAGS_DEBUG "$ENV{CFLAGS} -O0 -g2 -ggdb")
# set (CMAKE_C_FLAGS_RELEASE "$ENV{CFLAGS} -O3")

# 类似地，设置C++语言的编译标志。
# set (CMAKE_CXX_FLAGS "-std=c++11")
# set (CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -g2 -ggdb")
# set (CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3")

# 设置链接器的标志，确保只链接需要的库。
# set (CMAKE_EXE_LINKER_FLAGS "-Wl,--as-needed")

# 添加了一系列编译警告选项，以增强代码质量，确保代码在不同情况下都能警示潜在问题。
# add_compile_options (
#     -Wall
#     -Werror
#     -Wextra
#     -Wswitch-default
#     -Wfloat-equal
#     -Wshadow
#     -Wcast-qual
# )

# Nvidia GPU
find_package (CUDA REQUIRED)

# 设置CUDA的编译选项，包括使用C++11标准和OpenMP支持。
set (CUDA_VERBOSE_BUILD ${HGEMM_VERBOSE_MAKEFILE})
# set (CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11 -Xcompiler -fopenmp --expt-relaxed-constexpr")
set (CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11 -Xcompiler -fopenmp")
# 根据构建类型（调试或发布）设置不同的CUDA编译选项，包括架构、调试信息和优化级别。
if (${CMAKE_BUILD_TYPE} MATCHES "Debug")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -arch=sm_90a -g -G")
else ()
    # set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_${CMAKE_CUDA_ARCHITECTURES},code=sm_${CMAKE_CUDA_ARCHITECTURES} --use_fast_math -O3")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -arch=sm_90a")
endif ()

# 设置CUDA运行时和驱动程序的系统路径。
set (SYS_CUDART_PATH "/usr/local/cuda")
set (SYS_CUDA_DRIVER_PATH "/usr/lib/x86_64-linux-gnu")

# 查找所需的gflags库和OpenMP支持。
find_package(gflags REQUIRED)
find_package(OpenMP REQUIRED)

include_directories (
    ${PROJECT_SOURCE_DIR}/src/common
    ${PROJECT_SOURCE_DIR}/src
    ${SYS_CUDART_PATH}/include
    ${GFLAGS_INCLUDE_DIR}
)

link_directories (
    ${SYS_CUDART_PATH}/lib64
    ${SYS_CUDA_DRIVER_PATH}
)

file (GLOB HGEMM_SRCS 
    ${PROJECT_SOURCE_DIR}/src/*.cu
    ${PROJECT_SOURCE_DIR}/src/kernel/*.cu
)

cuda_add_executable (hgemm ${HGEMM_SRCS})
target_link_libraries (hgemm -lcublas -lcublasLt OpenMP::OpenMP_CXX ${GFLAGS_LIBRARIES})

install (TARGETS hgemm RUNTIME DESTINATION bin)


# docker run -itd --name mma --network host --gpus all -v /home/yujie.wu/mma_test:/mma_test nvidia/cuda:12.4-base bash
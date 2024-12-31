# Description: compile script

# Examples：
# ./build.sh -a 89 -t Release -b OFF
# ./build.sh -a 89 -t Debug -b OFF



#!/bin/bash

set -euo pipefail # 确保脚本在遇到错误时停止执行。

echo "========== build enter =========="

WORK_PATH=$(cd $(dirname $0) && pwd) && cd $WORK_PATH # 定义当前工作路径，并切换到该目录。

CUDA_ARCHITECTURE=89 # a: (NVIDIA A100: 80, RTX3080Ti / RTX3090 / RTX A6000: 86, H100: 89)
BUILD_TYPE=Release # t: (Debug, Release)
VERBOSE_MAKEFILE=OFF # b: (ON, OFF)

# 使用 getopts 来解析命令行参数
while getopts ":a:t:b:" opt
do
    case $opt in
        a)
        CUDA_ARCHITECTURE=$OPTARG
        echo "CUDA_ARCHITECTURE: $CUDA_ARCHITECTURE"
        ;;
        t)
        BUILD_TYPE=$OPTARG
        echo "BUILD_TYPE: $BUILD_TYPE"
        ;;
        b)
        VERBOSE_MAKEFILE=$OPTARG
        echo "VERBOSE_MAKEFILE: $VERBOSE_MAKEFILE"
        ;;
        ?)
        echo "invalid param: $OPTARG"
        exit 1
        ;;
    esac
done

# echo_cmd函数可以在脚本中被调用。
echo_cmd() {
    echo $1   # 打印出传递给函数的第一个参数 $1
    $1        # 执行了传递给函数的第一个参数所代表的命令
}


# 开始构建
echo "========== build cuda_hgemm =========="

echo_cmd "rm -rf build output"
echo_cmd "mkdir build"

echo_cmd "cd build"
echo_cmd "cmake -DCMAKE_CUDA_ARCHITECTURES=$CUDA_ARCHITECTURE -DCMAKE_BUILD_TYPE=$BUILD_TYPE -DHGEMM_VERBOSE_MAKEFILE=$VERBOSE_MAKEFILE -DCMAKE_INSTALL_PREFIX=$WORK_PATH/output -DCMAKE_SKIP_RPATH=ON .."
echo_cmd "make -j$(nproc --ignore=2)"
echo_cmd "make install"


# 版本信息记录：脚本会将 Git 分支、提交哈希、GCC 版本、编译时间等信息写入到 hgemm_version 文件中，以便于追踪构建的版本信息。
echo "========== build info =========="

BRANCH=`git rev-parse --abbrev-ref HEAD`
COMMIT=`git rev-parse HEAD`
GCC_VERSION=`gcc -dumpversion`
COMPILE_TIME=$(date "+%H:%M:%S %Y-%m-%d")

echo "branch: $BRANCH" >> $WORK_PATH/output/hgemm_version
echo "commit: $COMMIT" >> $WORK_PATH/output/hgemm_version
echo "gcc_version: $GCC_VERSION" >> $WORK_PATH/output/hgemm_version
echo "compile_time: $COMPILE_TIME" >> $WORK_PATH/output/hgemm_version

echo "========== build exit =========="

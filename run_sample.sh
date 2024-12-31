# 编译生成exe：./build.sh -a 89 -t Debug -b OFF

# =============================================================================
# 单次测试说明
# =============================================================================
# 本脚本用于执行矩阵乘法（HGEMM）测试，执行前请根据需要调整以下参数：
#
# 1. 矩阵维度配置：
#    M、N 和 K 分别代表矩阵的行数、列数和计算维度。根据实际测试需求，调整这些值。
#
# 2. 数据类型选择：
#    FP8E4M3
#    FP8E5M2
#    TF32
#    FP16
#
# 3. 矩阵文件路径：
#    - 传入矩阵A、B、C的文件路径及文件名，确保路径和文件名正确无误。
#    - eg：
#      -A_filename="/path/to/matrix_A.txt"
#      -B_filename="/path/to/matrix_B.txt"
#      -C_filename="/path/to/matrix_C.txt"
#
# 请根据实际测试需求修改命令中的参数并确保矩阵文件路径的正确性。
# =============================================================================

#!/bin/bash

set -euo pipefail

WORK_PATH=$(cd $(dirname $0) && pwd) && cd $WORK_PATH

rm -rf log && mkdir -p log

# 单次测试：
# nohup $WORK_PATH/output/bin/hgemm -M=16 -N=8 -K=32 -FP8E4M3=true -A_filename="./matrix_A.txt" -B_filename="./matrix_B.txt" -C_filename="./matrix_C.txt" > log/hgemm.log 2>&1 &
# nohup $WORK_PATH/output/bin/hgemm -M=16 -N=8 -K=32 -FP8E5M2=true -A_filename="./matrix_A.txt" -B_filename="./matrix_B.txt" -C_filename="./matrix_C.txt" > log/hgemm.log 2>&1 &
nohup $WORK_PATH/output/bin/hgemm -M=16 -N=8 -K=4 -TF32=true -A_filename="./matrix_A.txt" -B_filename="./matrix_B.txt" -C_filename="./matrix_C.txt" > log/hgemm.log 2>&1 &
# nohup $WORK_PATH/output/bin/hgemm -M=16 -N=8 -K=16 -FP16=true -A_filename="./matrix_A.txt" -B_filename="./matrix_B.txt" -C_filename="./matrix_C.txt" > log/hgemm.log 2>&1 &



# $1: M, $2: N, $3: K
# evaluate_hgemm() {
#     echo "Evaluating $1 * $2 * $3"
#     $WORK_PATH/output/bin/hgemm -M=$1 -N=$2 -K=$3 -enable_mma=true -warmup_iterations=1 -profiling_iterations=10 -sleep_duration=100 -enable_check=true > log/hgemm_${1}_${2}_${3}.log 2>&1
#     sleep 3
# }

# benchmark_hgemm() {
#     dims=(256 512 768 1024 1536 2048 3072 4096 5120 6144 7168 8192 9216 10240 11264 12288 13312 14336 15360 16384)

#     # M == N == K
#     for M in ${dims[@]};
#     do
#         evaluate_hgemm $M $M $M
#     done
# }

# 测试不同维度矩阵：
# benchmark_hgemm



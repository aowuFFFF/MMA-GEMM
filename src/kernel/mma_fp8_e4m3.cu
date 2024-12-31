// Description: mma naive hgemm

#include "common.h"
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <iostream>
#include <bitset>

// 矩阵的每个子矩阵的尺寸分别为 16x16, 8x16。
#define MMA_M 16
#define MMA_N 8
#define MMA_K 32

#define WARP_SIZE 32

template <typename AType, typename BType, typename CType>
__global__ void mma_fp8_e4m3(const AType * A, const BType * B, float * C, int M,
                               int N, int K) {
   
    // printf("A : %f \n", float(A[0]));
    const int K_tiles = K/32;

    // 计算当前 warp（线程块）在矩阵中的起始位置。
    const int warp_row = blockIdx.y * MMA_M;
    const int warp_col = blockIdx.x * MMA_N;
    // printf("blockIdx.y:%d  warp_row: %lu  blockIdx.x:%d  warp_col: %lu  \n", blockIdx.y, warp_row, blockIdx.x, warp_col);

    // 如果当前线程块的位置超出矩阵的范围，则提前退出。
    if (warp_row >= M || warp_col >= N) {
        return;
    }

    __shared__ __nv_fp8_e4m3 A_smem[MMA_M][MMA_K];
    __shared__ __nv_fp8_e4m3 B_smem[MMA_N][MMA_K];
    __shared__ float C_smem[MMA_M][MMA_N];


    // lane_id：每个线程会根据其线程号（threadIdx.x）计算出它在 warp 中的唯一标识。一个 warp 包含 32 个线程。
    int lane_id = threadIdx.x % WARP_SIZE;
    // printf("lane_id: %lu  K_tiles: %lu \n", lane_id, K_tiles);

    float RC[4] = {0.0, 0.0, 0.0, 0.0}; // uint32_t 4 个字节

    
    #pragma unroll  1
    for (int i = 0; i <  K_tiles; i++) 
    { // 每次循环处理一个子块的乘法
     __syncwarp();
        
        // 数据加载与计算 : 数据加载是通过 int4方式进行的,int4 是一个结构体，包含四个 int 类型的整数，每个 int 占用 4 字节，一共16个字节
        *((int4 *)(&A_smem[lane_id / 2][0]) + lane_id % 2) = 
            *((int4 *)(&A[(warp_row + lane_id / 2) * K + i * MMA_K]) + lane_id % 2);

        // lane_id / 4 = [0-7]
        if (lane_id < MMA_N * 2) {
            *((int4 *)(&B_smem[lane_id / 2][0]) + lane_id % 2) =
                *((int4 *)(&B[i * MMA_K + (warp_col + lane_id / 2) * K]) + lane_id % 2); // 加载矩阵 B 的数据到共享内存 B_smem
        }
        // printf("B_smem[0][0] = %f \n", float(B_smem[0][0]));
        __syncwarp();

        uint32_t RA[4];
        uint32_t RB[2];

        // 矩阵乘法操作
        // 从共享内存加载矩阵数据到寄存器 A_smem:16*32
        uint32_t A_smem_lane_addr = __cvta_generic_to_shared(&A_smem[lane_id % 16][(lane_id / 16) * 16]);
        LDMATRIX_X4(RA[0], RA[1], RA[2], RA[3], A_smem_lane_addr);// 加载两行数据到RA

        // B_smem:8*32
        uint32_t B_smem_lane_addr = __cvta_generic_to_shared(&B_smem[lane_id % 8][((lane_id / 8) % 2) * 16]);
        LDMATRIX_X2(RB[0], RB[1], B_smem_lane_addr);// 加载一行数据到RB
   
        // 执行半精度矩阵乘法，并将结果累加到 RC 中 
        // printf("lane_id: %lu  RA[0] = %f, RA[1] = %f, RB[0] = %f \n", lane_id, *((float*)&RA[0]), *((float*)&RA[1]), *((float*)&RB[0]));
        HMMA16832E4m3(RC[0], RC[1], RC[2], RC[3], RA[0], RA[1], RA[2], RA[3], RB[0], RB[1], RC[0], RC[1], RC[2], RC[3]);

        __syncwarp();
    }

    // 结果写回到共享内存
    // C_smem:16*8
    // printf("C_smem %f \n", C_smem[0][0]);
    C_smem[lane_id / 4][(lane_id % 4)*2]            = RC[0];
    C_smem[lane_id / 4][(lane_id % 4)*2 + 1]        = RC[1];
    C_smem[lane_id / 4 + 8][(lane_id % 4)*2]        = RC[2];
    C_smem[lane_id / 4 + 8][(lane_id % 4)*2 + 1]    = RC[3];

    __syncthreads(); // 同步所有线程，确保共享内存中的数据加载和计算完成。

    // 将计算结果从共享内存写回到全局内存中的 C 矩阵。
    if (lane_id < MMA_M) { // MMA_M=16
        for (int i = 0; i < 8; ++i) {
            C[(warp_row + lane_id) * N + warp_col + i] = C_smem[lane_id][i];
            // printf("C[%lu] = C_smem[%lu][%d] = %f \n", (warp_row + lane_id) * N + warp_col + i, lane_id, i, C_smem[lane_id][i]);
        }

    }
    
}

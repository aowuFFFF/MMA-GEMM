// Description: mma naive hgemm

#include "common.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
// 矩阵的每个子矩阵的尺寸分别为 16x16, 8x16。
#define MMA_M 16
#define MMA_N 8
#define MMA_K 4

#define WARP_SIZE 32
#define CEIL_DIV(M, N) ((M) + (N)-1) / (N)

// __device__ static uint32_t convert(unsigned const & s) {
__device__ static uint32_t convert(unsigned const & s) {
    unsigned storage = 0;

    asm volatile("cvt.rn.tf32.f32 %0, %1;" : "=r"(storage) : "r"(s));

    return storage;
  }

template <typename AType, typename BType, typename CType>
__global__ void mma_tf32(const AType * A, const BType * B, float * C, size_t M,
                               size_t N, size_t K) {
    const size_t K_tiles = CEIL_DIV(K, MMA_K);

    // 计算当前 warp（线程块）在矩阵中的起始位置。
    const size_t warp_row = blockIdx.y * MMA_M;
    const size_t warp_col = blockIdx.x * MMA_N;
    // printf("blockIdx.y:%d  warp_row: %lu  blockIdx.x:%d  warp_col: %lu  \n", blockIdx.y, warp_row, blockIdx.x, warp_col);

    // 如果当前线程块的位置超出矩阵的范围，则提前退出。
    if (warp_row >= M || warp_col >= N) {
        return;
    }

    __shared__ float A_smem[MMA_M][MMA_K];
    __shared__ float B_smem[MMA_N][MMA_K];
    __shared__ float C_smem[MMA_M][MMA_N];

    // lane_id：每个线程会根据其线程号（threadIdx.x）计算出它在 warp 中的唯一标识。一个 warp 包含 32 个线程。
    const size_t lane_id = threadIdx.x % WARP_SIZE;
    // printf("lane_id: %lu  K_tiles: %lu \n", lane_id, K_tiles);

    uint32_t RC[4] = {0, 0, 0, 0}; // uint32_t 4 个字节

    
#pragma unroll
    for (size_t i = 0; i < K_tiles; ++i) { // 每次循环处理一个子块的乘法
        // 数据加载与计算 : 数据加载是通过 int4方式进行的,int4 是一个结构体，包含四个 int 类型的整数，每个 int 占用 4 字节，一共16个字节
        // lane_id=[0-31], lane_id / 2 = [0-15], lane_id % 2=[0,1]，
        if(lane_id < MMA_M){
            *((float4 *)(&A_smem[lane_id ][0])) = 
                *((float4 *)(&A[(warp_row + lane_id ) * K + i * MMA_K]));
        }
        
        // lane_id / 4 = [0-7]
        if (lane_id < MMA_N) {
            *((float4 *)(&B_smem[lane_id][0])) =
                *((float4 *)(&B[i * MMA_K + (warp_col + lane_id) * K]));
        }


        __syncthreads();

        uint32_t RA[2];
        uint32_t RB[1];


        // 矩阵乘法操作

        // 从共享内存加载矩阵数据到寄存器 A_smem:16*4
        uint32_t A_smem_lane_addr = __cvta_generic_to_shared(&A_smem[lane_id % 16][0]);
        LDMATRIX_X2(RA[0], RA[1], A_smem_lane_addr);// 加载两行数据到RA

        // B_smem:8*4
        uint32_t B_smem_lane_addr = __cvta_generic_to_shared(&B_smem[lane_id % 8][0]);
        LDMATRIX_X1(RB[0], B_smem_lane_addr);// 加载一行数据到RB

        RA[0] = convert(RA[0]);
        RA[1] = convert(RA[1]);
        RB[0] = convert(RB[0]);

        // 执行半精度矩阵乘法，并将结果累加到 RC 中 
        // printf("lane_id: %lu  RA[0] = %f, RA[1] = %f, RB[0] = %f \n", lane_id, *((float*)&RA[0]), *((float*)&RA[1]), *((float*)&RB[0]));
        HMMA1684(RC[0], RC[1], RC[2], RC[3], RA[0], RA[1], RB[0], RC[0], RC[1], RC[2], RC[3]);
        

        __syncthreads();
    }

    // 结果写回到共享内存
    // C_smem:16*8
    *((&C_smem[lane_id / 4][0]) + (lane_id % 4) * 2)        = *reinterpret_cast<float*>(&RC[0]);
    *((&C_smem[lane_id / 4][0]) + (lane_id % 4) * 2 + 1)    = *reinterpret_cast<float*>(&RC[1]);
    *((&C_smem[lane_id / 4 + 8][0]) + (lane_id % 4) * 2)    = *reinterpret_cast<float*>(&RC[2]);
    *((&C_smem[lane_id / 4 + 8][0]) + (lane_id % 4) * 2 + 1)= *reinterpret_cast<float*>(&RC[3]);

    __syncthreads(); // 同步所有线程，确保共享内存中的数据加载和计算完成。

    // 将计算结果从共享内存写回到全局内存中的 C 矩阵。
    if (lane_id < MMA_M) { // MMA_M=16
        for (int i = 0; i < 8; ++i) {
            C[(warp_row + lane_id) * N + warp_col + i] = C_smem[lane_id][i];
        }

    //     *((int4 *)(&C[(warp_row + lane_id) * N + warp_col])) = *((int4 *)(&C_smem[lane_id][0]));
    //     *((int4 *)(&C[(warp_row + lane_id) * N + warp_col + 4])) = *((int4 *)(&C_smem[lane_id][4]));
    }
}



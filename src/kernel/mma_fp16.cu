// Description: mma naive hgemm

#include "common.h"

// 矩阵的每个子矩阵的尺寸分别为 16x16, 8x16。
#define MMA_M 16
#define MMA_N 8
#define MMA_K 16

#define WARP_SIZE 32
#define CEIL_DIV(M, N) ((M) + (N)-1) / (N)

template <typename AType, typename BType, typename CType>
__global__ void mma_fp16(const AType * A, const BType * B, float * C, size_t M,
                               size_t N, size_t K) {
    const size_t K_tiles = CEIL_DIV(K, MMA_K);

    // 计算当前 warp 在矩阵中的起始位置。
    const size_t warp_row = blockIdx.y * MMA_M;
    const size_t warp_col = blockIdx.x * MMA_N;

    // 如果当前线程块的位置超出矩阵的范围，则提前退出。
    if (warp_row >= M || warp_col >= N) {
        return;
    }

    __shared__ half A_smem[MMA_M][MMA_K];
    __shared__ half B_smem[MMA_N][MMA_K];
    __shared__ half C_smem[MMA_M][MMA_N];

    // lane_id：每个线程会根据其线程号（threadIdx.x）计算出它在 warp 中的唯一标识。一个 warp 包含 32 个线程。
    const size_t lane_id = threadIdx.x % WARP_SIZE;

    uint32_t RC[2] = {0, 0};

    
#pragma unroll
    for (size_t i = 0; i < K_tiles; ++i) { // 每次循环处理一个子块的乘法
        // 数据加载与计算
        // 数据加载是通过 int4（8个半精度浮点数）方式进行的
        *((int4 *)(&A_smem[lane_id / 2][0]) + lane_id % 2) =
            *((int4 *)(&A[(warp_row + lane_id / 2) * K + i * MMA_K]) + lane_id % 2); // 将矩阵 A 的一部分数据加载到共享内存 A_smem 中

        if (lane_id < MMA_N * 2) {
            *((int4 *)(&B_smem[lane_id / 2][0]) + lane_id % 2) =
                *((int4 *)(&B[i * MMA_K + (warp_col + lane_id / 2) * K]) + lane_id % 2); // 加载矩阵 B 的数据到共享内存 B_smem
        }

        __syncthreads();

        uint32_t RA[4];
        uint32_t RB[2];


        // 矩阵乘法操作

        // 从共享内存加载矩阵数据到寄存器
        // A_smem:16*16
        uint32_t A_smem_lane_addr = __cvta_generic_to_shared(&A_smem[lane_id % 16][(lane_id / 16) * 8]);
        // printf("lane_id:%lu A_smem[%d][%lu] \n ", lane_id, lane_id % 16,  (lane_id / 16) * 8);
        LDMATRIX_X4(RA[0], RA[1], RA[2], RA[3], A_smem_lane_addr);

        // B_smem:8*16
        uint32_t B_smem_lane_addr = __cvta_generic_to_shared(&B_smem[lane_id % 8][((lane_id / 8) % 2) * 8]);
        // printf("lane_id:%lu B_smem[%lu][%lu] \n ", lane_id, lane_id % 8,  ((lane_id / 8) % 2) * 8);
        LDMATRIX_X2(RB[0], RB[1], B_smem_lane_addr);

        // if(threadIdx.x == 0)
        // {
        //     for (int i = 31; i >= 0; --i) {
        //         printf("%d", ( RA[0]>> i) & 1);
        //     }
        //     printf("\n");
        // }

        // 执行半精度矩阵乘法，并将结果累加到 RC 中
        HMMA16816(RC[0], RC[1], RA[0], RA[1], RA[2], RA[3], RB[0], RB[1], RC[0], RC[1]);

        __syncthreads();
    }

    // 结果写回到共享内存
    // C_smem:16*8
    *((uint32_t *)(&C_smem[lane_id / 4][0]) + lane_id % 4) = RC[0];
    *((uint32_t *)(&C_smem[lane_id / 4 + 8][0]) + lane_id % 4) = RC[1];

    __syncthreads(); // 同步所有线程，确保共享内存中的数据加载和计算完成。

    // 将计算结果从共享内存写回到全局内存中的 C 矩阵。
    if (lane_id < MMA_M) {
        for (int col = 0; col < MMA_N; ++col) {
            auto smem_value = C_smem[lane_id][col];
            float float_value = static_cast<float>(smem_value);
            // 将转换后的 float 值存储到 C 矩阵中
            C[(warp_row + lane_id) * N + (warp_col + col)] = float_value;
        }
        // *((int4 *)(&C[(warp_row + lane_id) * N + warp_col])) = *((int4 *)(&C_smem[lane_id][0]));
    }
}


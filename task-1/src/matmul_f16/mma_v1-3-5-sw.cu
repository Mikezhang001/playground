#include "playground/matmul.hpp"
#include "playground/system.hpp"

#include <cuda_fp16.h>
#include <mma.h>
#include <stdio.h>
#include <cuda.h>
#include <algorithm>   // for std::max

using namespace nvcuda;

#include "header/ptx.h"
#define MMA_M 16
#define MMA_N 8
#define MMA_K 16

#define BLOCK_ROWS 256
#define BLOCK_COLS 128

#define WARP_ROWS 64
#define WARP_COLS 64

#define BLOCK_ROW_WARPS 2  // BLOCK_COLS / WARP_COLS
#define BLOCK_COL_WARPS 4  // BLOCK_ROWS / WARP_ROWS

#define BLOCK_ROW_TILES 16  // BLOCK_COLS / MMA_N
#define BLOCK_COL_TILES 16  // BLOCK_ROWS / MMA_M

#define WARP_ROW_TILES 8  // WARP_COLS / MMA_N
#define WARP_COL_TILES 4  // WARP_ROWS / MMA_M

#define WARP_SIZE 32
#define WARPS_PER_BLOCK 8      // BLOCK_ROW_WARPS * BLOCK_COL_WARPS
#define THREADS_PER_BLOCK 256  // WARP_SIZE * WARPS_PER_BLOCK

#define CHUNK_K 2  // 32 / MMA_K

#define CHUNK_LINE_BYTES 64          // CHUNK_K * MMA_K * sizeof(half)
#define CHUNK_COPY_LINES_PER_WARP 8  // WARP_SIZE * sizeof(int4) / CHUNK_LINE_BYTES
#define CHUNK_COPY_LINE_LANES 4      // WARP_SIZE / CHUNK_COPY_LINES_PER_WARP

#define AB_SMEM_STRIDE 32  // CHUNK_K * MMA_K

#define C_SMEM_STRIDE 128  // BLOCK_COLS
#define C_SMEM_OFFSET 64   // WARP_COLS

#define BLOCK_STRIDE 16


#define THREAD_COPY_BYTES 16

#define SMEM_BANK_ROWS 2  // 32 * 4 / (AB_SMEM_STRIDE * sizeof(half))

#define PERMUTED_OFFSET 8
#define PERMUTED_COLS 4


#define BM (256)
#define BN (128)
#define BK (32)
#define padding_SB (8)

namespace playground
{

__global__ void matrixKernel19s(half *A, half *B, half *C, int M, int K, int N)
{
    const size_t M_tiles = M / MMA_M;
    const size_t N_tiles = N / MMA_N;
    const size_t K_tiles = K / MMA_K;

    const size_t block_tile_i = blockIdx.y * BLOCK_COL_TILES;
    const size_t block_tile_j = blockIdx.x * BLOCK_ROW_TILES;
    if (block_tile_i >= M_tiles || block_tile_j >= N_tiles) {
        return;
    }

    extern __shared__ half smem[][AB_SMEM_STRIDE];

    int tid = threadIdx.x;
    const size_t warp_id = threadIdx.x / WARP_SIZE;
    const size_t lane_id = threadIdx.x % WARP_SIZE;

    uint32_t RC[WARP_COL_TILES][WARP_ROW_TILES][2];


    //A -> S_A
    half* S_A = &smem[0][0];
    int S_A_row = lane_id / 4 + (BM / 8) * warp_id;//每个warp连续拿行
    int S_A_col = (lane_id % 4) * 8;//一次拿8个
    int row_offset = blockIdx.y * BM;

    //B -> S_B
    half* S_B = &smem[0][0] + BM * BK;
    int S_B_row = lane_id / 16 + (BK / 8) * warp_id;//每个warp连续拿行
    int S_B_col = (lane_id % 16) * 8 ;//一次拿8个
    int col_offset = blockIdx.x * BN;
    
    //S_B ->register
    int WARP_M = 64;
    int WARP_N = 64;
    int warp_y = warp_id / 2;
    int warp_x = warp_id % 2; 
    int S_AC_row_offset = warp_y * WARP_M;
    int S_BC_col_offset = warp_x * WARP_N;

    //S_C
    half* S_C = &smem[0][0];

#pragma unroll
    for (size_t i = 0; i < WARP_COL_TILES; ++i) {
#pragma unroll
        for (size_t j = 0; j < WARP_ROW_TILES; ++j) {
            RC[i][j][0] = 0;
            RC[i][j][1] = 0;
        }
    }


#pragma unroll
    for (size_t tile_k = 0; tile_k < K_tiles; tile_k += CHUNK_K) {
        int base_k = tile_k * 16;
        // #pragma unroll
        // for(int S_A_row_id = S_A_row; S_A_row_id < (BM / 8) * (warp_id + 1); S_A_row_id += (32 * 8) / 32)//每个warp可以拿8行
        // {
        //     // (float4 &)S_A[S_A_row_id][S_A_col] = (float4 &)dA[(S_A_row_id + row_offset) * K + S_A_col + base_k];
        //     // (float4 &)S_A[(S_A_row_id) * BK + S_A_col] = (float4 &)dA[(S_A_row_id + row_offset) * K + S_A_col + base_k];

        //     //每八行中的每组间偏移，其中0-1，2-3, 4-5, 6-7 这种组内不需要做偏移
        //     (float4 &)S_A[(S_A_row_id) * BK + (S_A_col / 8 + (S_A_row_id % 8) / 2) % 4 * 8] = (float4 &)A[(S_A_row_id + row_offset) * K + S_A_col + base_k];
        // }
    (float4 &)S_A[(S_A_row) * BK + (S_A_col / 8 + (S_A_row % 8) / 2) % 4 * 8] = (float4 &)A[(S_A_row + row_offset) * K + S_A_col + base_k];
    (float4 &)S_A[(S_A_row + 8) * BK + (S_A_col / 8 + ((S_A_row + 8) % 8) / 2) % 4 * 8] = (float4 &)A[(S_A_row + 8 + row_offset) * K + S_A_col + base_k];
    (float4 &)S_A[(S_A_row + 16) * BK + (S_A_col / 8 + ((S_A_row + 16) % 8) / 2) % 4 * 8] = (float4 &)A[(S_A_row + 16 + row_offset) * K + S_A_col + base_k];
    (float4 &)S_A[(S_A_row + 24) * BK + (S_A_col / 8 + ((S_A_row + 24) % 8) / 2) % 4 * 8] = (float4 &)A[(S_A_row + 24 + row_offset) * K + S_A_col + base_k];


    // #pragma unroll
    //     for(int S_B_row_id = S_B_row; S_B_row_id < (BK / 8) * (warp_id + 1); S_B_row_id += (32 * 8) / 128)//每个warp连续拿
    //     {
    //         (float4 &)S_B[(S_B_row_id) * (BN + padding_SB) + S_B_col] = (float4 &)B[(S_B_row_id + base_k) * N + S_B_col + col_offset];
    //     }
    (float4 &)S_B[(S_B_row) * (BN + padding_SB) + S_B_col] = (float4 &)B[(S_B_row + base_k) * N + S_B_col + col_offset];
    (float4 &)S_B[(S_B_row + 2) * (BN + padding_SB) + S_B_col] = (float4 &)B[(S_B_row + 2 + base_k) * N + S_B_col + col_offset];

    __syncthreads();

#pragma unroll
        for (size_t k_step = 0; k_step < CHUNK_K; ++k_step) {
            uint32_t RA[WARP_COL_TILES][4];
            uint32_t RB[WARP_ROW_TILES][2];

#pragma unroll
            for (size_t i = 0; i < WARP_COL_TILES; ++i) {
                size_t A_smem_idx = (warp_id / BLOCK_ROW_WARPS) * WARP_ROWS + i * MMA_M;
                // uint32_t A_smem_lane_addr = __cvta_generic_to_shared(
                //     &smem[A_smem_idx + lane_id % 16]
                //          [(k_step * MMA_K + (lane_id / 16) * 8 +
                //            (lane_id % 16 % (PERMUTED_COLS * SMEM_BANK_ROWS)) / SMEM_BANK_ROWS * PERMUTED_OFFSET) %
                //           AB_SMEM_STRIDE]);

                uint32_t A_smem_lane_addr = __cvta_generic_to_shared( &smem[(A_smem_idx + lane_id % 16)][((lane_id / 16 + k_step * MMA_K / 8 + (lane_id % 8) / 2)) % 4 * 8]);//
                // uint32_t A_smem_lane_addr = __cvta_generic_to_shared(&S_A[(A_smem_idx + lane_id % 16) * BK + ((lane_id / 16 + k_step * MMA_K / 8 + (lane_id % 8) / 2)) % 4 * 8]);//                  
                asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n" \
                    : "=r"(RA[i][0]), "=r"(RA[i][1]), "=r"(RA[i][2]), "=r"(RA[i][3])                     \
                    : "r"(A_smem_lane_addr));    
                
            }

#pragma unroll
            for(int j = 0; j < WARP_ROW_TILES; j++)
            {
                // uint32_t B_smem_lane = __cvta_generic_to_shared(&S_B[tid % 16][0]);//感觉这个有些问题, 16 -> 31好像不用标记
                // uint32_t B_smem_lane = __cvta_generic_to_shared(&S_B[tid % 16 + i * MMA_K][j * MMA_N + S_BC_col_offset]);
                uint32_t B_smem_lane = __cvta_generic_to_shared(&S_B[(tid % 16 + k_step * MMA_K) * (BN + padding_SB) + j * MMA_N + S_BC_col_offset]);
                asm volatile("ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0, %1}, [%2];\n" \
                    : "=r"(RB[j][0]), "=r"(RB[j][1])                                   \
                    : "r"(B_smem_lane));//r 四个字节，l 八个字节    

            }

#pragma unroll
            for (size_t i = 0; i < WARP_COL_TILES; ++i) {
#pragma unroll
                for (size_t j = 0; j < WARP_ROW_TILES; ++j) {
                    size_t j_s = (i % 2) ? (WARP_ROW_TILES - j - 1) : j;

                    HMMA16816(RC[i][j_s][0], RC[i][j_s][1], RA[i][0], RA[i][1], RA[i][2], RA[i][3], RB[j_s][0],
                              RB[j_s][1], RC[i][j_s][0], RC[i][j_s][1]);
                }
            }
        }

        __syncthreads();
    }

#pragma unroll
    for(int i = 0; i < WARP_M / MMA_M; i++)
    {
#pragma unroll
        for(int j = 0; j < WARP_N / MMA_N; j++)
        {
            ////uint32_t 4字节两个half
            // *((uint32_t *)(&S_C[tid / 4][0]) + tid % 4) = RC[0];
            // *((uint32_t *)(&S_C[tid / 4 + 8][0]) + tid % 4) = RC[1];
            // *((uint32_t *)(&S_C[lane_id / 4 + i * MMA_M + S_AC_row_offset][j * MMA_N + S_BC_col_offset]) + lane_id % 4) = RC[i][j][0];
            // *((uint32_t *)(&S_C[lane_id / 4 + 8 + i * MMA_M + S_AC_row_offset][j * MMA_N + S_BC_col_offset]) + lane_id % 4) = RC[i][j][1];

            //一、
            // *((uint32_t *)(&S_C[(lane_id / 4 + i * MMA_M + S_AC_row_offset) * BN + j * MMA_N + S_BC_col_offset]) + lane_id % 4) = RC[i][j][0];
            // *((uint32_t *)(&S_C[(lane_id / 4 + 8 + i * MMA_M + S_AC_row_offset) * BN + j * MMA_N + S_BC_col_offset]) + lane_id % 4) = RC[i][j][1];

            //二、与 一 等价
            // *(uint32_t *)(&S_C[(lane_id / 4 + i * MMA_M + S_AC_row_offset) * BN + j * MMA_N + S_BC_col_offset + lane_id % 4 * 2]) = RC[i][j][0];
            // *(uint32_t *)(&S_C[(lane_id / 4 + 8 + i * MMA_M + S_AC_row_offset) * BN + j * MMA_N + S_BC_col_offset + lane_id % 4 * 2]) = RC[i][j][1];

            //观察可知，需要将每行都偏移4个bank即为4个half，以8为周期
            *(uint32_t *)(&S_C[(lane_id / 4 + i * MMA_M + S_AC_row_offset) * BN + ((j * MMA_N + S_BC_col_offset + lane_id % 4 * 2) + (lane_id / 4) * 8) % BN]) = RC[i][j][0];
            *(uint32_t *)(&S_C[(lane_id / 4 + 8 + i * MMA_M + S_AC_row_offset) * BN + ((j * MMA_N + S_BC_col_offset + lane_id % 4 * 2) + (lane_id / 4) * 8) % BN]) = RC[i][j][1];//一样
            //  *(uint32_t *)(&S_C[(lane_id / 4 + 8 + i * MMA_M + S_AC_row_offset) * BN + ((j * MMA_N + S_BC_col_offset + lane_id % 4 * 2) + (lane_id / 4 + 8) * 8) % BN]) = RC[i][j][1];//错误的
        }
    }

    __syncthreads();

   //从S_C搬到global memory[256, 128]
    int S_C_row = tid / 16;
    int S_C_col = (tid % 16) * 8;
#pragma unroll
    for(int S_C_row_id = S_C_row; S_C_row_id < BM; S_C_row_id += 256 / 16)
    {
        // (float4 &)C[S_C_row_id + row_offset][S_C_col + col_offset] = (float4 &)S_C[S_C_row_id][S_C_col];
        // (float4 &)C[(S_C_row_id + row_offset) * N + S_C_col + col_offset] = (float4 &)S_C[S_C_row_id][S_C_col];
        
        // (float4 &)C[(S_C_row_id + row_offset) * N + S_C_col + col_offset] = (float4 &)S_C[(S_C_row_id) * BN + S_C_col];

        // //观察可知，需要将每行都偏移4个bank即为4个half，以8为周期
        (float4 &)C[(S_C_row_id + row_offset) * N + S_C_col + col_offset] = (float4 &)S_C[(S_C_row_id) * BN + (S_C_col + S_C_row_id % 8 * 8) % BN];
    }
}

PLAYGROUND_MATMUL_DEC(float16_t, 19, M, N, K, A, B, C)
{
    const int BLOCK_DIM_y = 1;
    const int BLOCK_DIM_x = 256;

    int num_block_y = (M + BM - 1) / BM;
    int num_block_x = (N + BN - 1) / BN;

    dim3 block_dim(BLOCK_DIM_x, BLOCK_DIM_y, 1);
    dim3 grid_dim(num_block_x, num_block_y, 1);

    size_t smem_max_size = std::max((BM * BK + BN * (BK +  padding_SB)) * sizeof(half), BM * BN * sizeof(half));
    cudaFuncSetAttribute(matrixKernel19s, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_max_size);
    
    matrixKernel19s<<<grid_dim, block_dim, smem_max_size>>>(
        const_cast<float16_t*>(A), 
        const_cast<float16_t*>(B), 
        const_cast<float16_t*>(C), 
        M, K, N
    );

    cudaDeviceSynchronize();
}

}  // namespace playground
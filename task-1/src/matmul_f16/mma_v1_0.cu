#include "playground/matmul.hpp"
#include "playground/system.hpp"

#include <cuda_fp16.h>
#include <mma.h>
using namespace nvcuda;

#define MMA_M (16)
#define MMA_N (8)
#define MMA_K (16)
#define BM (256)
#define BN (128)
#define BK (32)

namespace playground
{

__global__ void matrixKernel18s(half *dA, half *dB, half *dC, int M, int K, int N)
{
    // 全局偏移
    int row_offset = blockIdx.y * BM;
    int col_offset = blockIdx.x * BN;

    // 块内线程索引
    int tid = threadIdx.x + threadIdx.y * blockDim.x;

    //8个warp，即为每个warp处理[64, 32] [32, 64] = [64, 64]
    // __shared__ half S_A[BM][BK];  // [256][32]
    // __shared__ half S_B[BK][BN];  // [32][128]
    // __shared__ half S_C[BM][BN]; //  [256][128]

    extern __shared__ half smem[];//空间有点大换动态
    half* S_A = smem;
    half* S_B = S_A + BM * BK;
    half* S_C = smem;

    int S_A_row = tid / 4;
    int S_A_col = (tid % 4) * 8;//一次拿8个

    int S_B_row = tid / 16;
    int S_B_col = (tid % 16) * 8 ;//一次拿8个

    int WARP_M = 64;
    int WARP_N = 64;
    int warp_id = tid / 32;//将其映射为[4, 2]列
    int lane_id = tid % 32;//每个warp内的id
    int warp_y = warp_id / 2;
    int warp_x = warp_id % 2; 
    int S_AC_row_offset = warp_y * WARP_M;
    int S_BC_col_offset = warp_x * WARP_N;
    uint32_t RA[4][2][4];//WARP_M / MMA_M = 4; BK / MMA_K = 2;
    uint32_t RB[2][8][2];//BK / MMA_K = 2; WARP_N / MMA_N = 8;
    uint32_t RC[4][8][2] = {0, 0};//WARP_M / MMA_M = 4;WARP_N / MMA_N = 8;


    // K维度循环
    for(int split_k_id = 0; split_k_id < (K + BK - 1) / BK; split_k_id++)
    {
        int base_k = split_k_id * BK;
        
        // ===== 加载 A 矩阵到共享内存 =====
        // A 是行优先: [M][K], 我们加载 [16][16] 块
        // 32个线程协作，每个加载 8个 half (4 uint32_t)
        // (float4 &)S_A[S_A_row][S_A_col] = (float4 &)dA[(S_A_row + row) * K + S_A_col + base_k];
        
        //dA -> S_A
        for(int S_A_row_id = S_A_row; S_A_row_id < BM; S_A_row_id += 256 / 4)//每行4个thread, 256个thread一次拿64行
        {
            // (float4 &)S_A[S_A_row_id][S_A_col] = (float4 &)dA[(S_A_row_id + row_offset) * K + S_A_col + base_k];
            (float4 &)S_A[(S_A_row_id) * BK + S_A_col] = (float4 &)dA[(S_A_row_id + row_offset) * K + S_A_col + base_k];
        }
        // ===== 加载 B 矩阵到共享内存 =====
        // B 是行优先: [K][N], 我们加载 [16][8] 块
        // 32个线程协作，但只需要16个
        // if(tid < 16)
        // {
        //     (float4 &)S_B[S_B_row][S_B_col] = (float4 &)dB[(S_B_row + base_k) * N + S_B_col + col];
        // }
        //dB -> S_B
        for(int S_B_row_id = S_B_row; S_B_row_id < BK; S_B_row_id += 256 / 16)//每行16个thread, 256个thread一次拿16行
        {
            (float4 &)S_B[(S_B_row_id) * BN + S_B_col] = (float4 &)dB[(S_B_row_id + base_k) * N + S_B_col + col_offset];
        }
        __syncthreads();

        //S_A -> 左矩阵, 对S_A横着切
        //每个warp要算的[64, 64] = [WARP_M, BK] * [BK, WARP_N];

        for(int i = 0; i < WARP_M / MMA_M; i++)
        {
            for(int j = 0; j < BK / MMA_K; j++)
            {
                // uint32_t A_smem_lane = __cvta_generic_to_shared(&S_A[tid % 16][(tid / 16) * 8]);//转成共享内存的格式，布局为[16, 2]
                // uint32_t A_smem_lane = __cvta_generic_to_shared(&S_A[lane_id % 16 + i * MMA_M + S_AC_row_offset][(lane_id / 16) * 8 + j * MMA_K]);//加上偏移
                uint32_t A_smem_lane = __cvta_generic_to_shared(&S_A[(lane_id % 16 + i * MMA_M + S_AC_row_offset) * BK + (lane_id / 16) * 8 + j * MMA_K]);//加上偏移
                asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n" \
                    : "=r"(RA[i][j][0]), "=r"(RA[i][j][1]), "=r"(RA[i][j][2]), "=r"(RA[i][j][3])                     \
                    : "r"(A_smem_lane));     
            }
        }

        //S_B ->右矩阵，对S_B纵着切
       
        for(int i = 0; i < BK / MMA_K; i++)
        {
            for(int j = 0; j < WARP_N / MMA_N; j++)
            {
                // uint32_t B_smem_lane = __cvta_generic_to_shared(&S_B[tid % 16][0]);//感觉这个有些问题, 16 -> 31好像不用标记
                // uint32_t B_smem_lane = __cvta_generic_to_shared(&S_B[tid % 16 + i * MMA_K][j * MMA_N + S_BC_col_offset]);
                uint32_t B_smem_lane = __cvta_generic_to_shared(&S_B[(tid % 16 + i * MMA_K) * BN + j * MMA_N + S_BC_col_offset]);
                asm volatile("ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0, %1}, [%2];\n" \
                    : "=r"(RB[i][j][0]), "=r"(RB[i][j][1])                                   \
                    : "r"(B_smem_lane));//r 四个字节，l 八个字节    
            }
        }

        //左矩阵 × 右矩阵
        for(int i = 0; i < WARP_M / MMA_M; i++)
        {
            for(int j = 0; j < WARP_N / MMA_N; j++)
            {
                for(int k = 0; k < BK / MMA_K; k++)
                {
                    // asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%8, %9};\n" \
                    //     : "=r"(RC[0]), "=r"(RC[1])                                                                                \
                    //     : "r"(RA[0]), "r"(RA[1]), "r"(RA[2]), "r"(RA[3]), "r"(RB[0]), "r"(RB[1]), "r"(RC[0]), "r"(RC[1]));  
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%8, %9};\n" \
                        : "=r"(RC[i][j][0]), "=r"(RC[i][j][1])                                                                                \
                        : "r"(RA[i][k][0]), "r"(RA[i][k][1]), "r"(RA[i][k][2]), "r"(RA[i][k][3]), "r"(RB[k][j][0]), "r"(RB[k][j][1]), "r"(RC[i][j][0]), "r"(RC[i][j][1]));  
                }
            }
        }
        __syncthreads();

    }

    //结果矩阵搬到S_C

    for(int i = 0; i < WARP_M / MMA_M; i++)
    {
        for(int j = 0; j < WARP_N / MMA_N; j++)
        {
            ////uint32_t 4字节两个half
            // *((uint32_t *)(&S_C[tid / 4][0]) + tid % 4) = RC[0];
            // *((uint32_t *)(&S_C[tid / 4 + 8][0]) + tid % 4) = RC[1];
            // *((uint32_t *)(&S_C[lane_id / 4 + i * MMA_M + S_AC_row_offset][j * MMA_N + S_BC_col_offset]) + lane_id % 4) = RC[i][j][0];
            // *((uint32_t *)(&S_C[lane_id / 4 + 8 + i * MMA_M + S_AC_row_offset][j * MMA_N + S_BC_col_offset]) + lane_id % 4) = RC[i][j][1];
            *((uint32_t *)(&S_C[(lane_id / 4 + i * MMA_M + S_AC_row_offset) * BN + j * MMA_N + S_BC_col_offset]) + lane_id % 4) = RC[i][j][0];
            *((uint32_t *)(&S_C[(lane_id / 4 + 8 + i * MMA_M + S_AC_row_offset) * BN + j * MMA_N + S_BC_col_offset]) + lane_id % 4) = RC[i][j][1];
        }
    }

   //从S_C搬到global memory[256, 128]
    int S_C_row = tid / 16;
    int S_C_col = (tid % 16) * 8;
    for(int S_C_row_id = S_C_row; S_C_row_id < BM; S_C_row_id += 256 / 16)
    {
        // (float4 &)dC[S_C_row_id + row_offset][S_C_col + col_offset] = (float4 &)S_C[S_C_row_id][S_C_col];
        // (float4 &)dC[(S_C_row_id + row_offset) * N + S_C_col + col_offset] = (float4 &)S_C[S_C_row_id][S_C_col];
        (float4 &)dC[(S_C_row_id + row_offset) * N + S_C_col + col_offset] = (float4 &)S_C[(S_C_row_id) * BN + S_C_col];
    }
}

PLAYGROUND_MATMUL_DEC(float16_t, 18, M, N, K, A, B, C)
{
    const int BLOCK_DIM_y = 16;
    const int BLOCK_DIM_x = 16;


    int num_block_y = (M + BM - 1) / BM;
    int num_block_x = (N + BN - 1) / BN;

    dim3 block_dim(BLOCK_DIM_x, BLOCK_DIM_y, 1);
    dim3 grid_dim(num_block_x, num_block_y, 1);

    size_t smem_max_size = std::max((BM + BN) * BK * sizeof(half), BM * BN * sizeof(half));
    cudaFuncSetAttribute(matrixKernel18s, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_max_size);
    
    matrixKernel18s<<<grid_dim, block_dim, smem_max_size>>>(
        const_cast<float16_t*>(A), 
        const_cast<float16_t*>(B), 
        const_cast<float16_t*>(C), 
        M, K, N
    );

    cudaDeviceSynchronize();
}

}  // namespace playground
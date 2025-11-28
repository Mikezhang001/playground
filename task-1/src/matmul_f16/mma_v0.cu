#include "playground/matmul.hpp"
#include "playground/system.hpp"

#include <cuda_fp16.h>
#include <mma.h>
using namespace nvcuda;


#define MMA_M 16
#define MMA_N 8
#define MMA_K 16

namespace playground
{

__global__ void matrixKernel17s(half *dA, half *dB, half *dC, int M, int K, int N)
{

    // 全局偏移
    int row = blockIdx.y * MMA_M;
    int col = blockIdx.x * MMA_N;

    // 块内线程索引
    int tid = threadIdx.x + threadIdx.y * blockDim.x;

    __shared__ half S_A[MMA_M][MMA_K];  // [16][16]
    __shared__ half S_B[MMA_K][MMA_N];  // [16][8]
    __shared__ half S_C[MMA_M][MMA_N];  // [16][8]

    int S_A_row = tid / 2;
    int S_A_col = (tid % 2) * 8;//一次拿8个

    int S_B_row = tid / 1;
    int S_B_col = 0;//一次拿8个

    uint32_t RA[4];
    uint32_t RB[2];
    uint32_t RC[2] = {0, 0};


    // K维度循环
    for(int split_k_id = 0; split_k_id < (K + MMA_K - 1) / MMA_K; split_k_id++)
    {
        int base_k = split_k_id * MMA_K;
        
        // ===== 加载 A 矩阵到共享内存 =====
        // A 是行优先: [M][K], 我们加载 [16][16] 块
        // 32个线程协作，每个加载 8个 half (4 uint32_t)
        (float4 &)S_A[S_A_row][S_A_col] = (float4 &)dA[(S_A_row + row) * K + S_A_col + base_k];
        
        // ===== 加载 B 矩阵到共享内存 =====
        // B 是行优先: [K][N], 我们加载 [16][8] 块
        // 32个线程协作，但只需要16个
        if(tid < 16)
        {
            (float4 &)S_B[S_B_row][S_B_col] = (float4 &)dB[(S_B_row + base_k) * N + S_B_col + col];
        }
        __syncthreads();

        uint32_t A_smem_lane = __cvta_generic_to_shared(&S_A[tid % 16][(tid / 16) * 8]);//转成共享内存的格式，布局为[16, 2]
        asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n" \
                : "=r"(RA[0]), "=r"(RA[1]), "=r"(RA[2]), "=r"(RA[3])                     \
                : "r"(A_smem_lane));

        uint32_t B_smem_lane = __cvta_generic_to_shared(&S_B[tid % 16][0]);//感觉这个有些问题, 16 -> 31好像不用标记
        asm volatile("ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0, %1}, [%2];\n" \
                : "=r"(RB[0]), "=r"(RB[1])                                   \
                : "r"(B_smem_lane));//r 四个字节，l 八个字节    

        asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%8, %9};\n" \
                : "=r"(RC[0]), "=r"(RC[1])                                                                                \
                : "r"(RA[0]), "r"(RA[1]), "r"(RA[2]), "r"(RA[3]), "r"(RB[0]), "r"(RB[1]), "r"(RC[0]), "r"(RC[1]));  
        
        __syncthreads();
    }

    *((uint32_t *)(&S_C[tid / 4][0]) + tid % 4) = RC[0];
    *((uint32_t *)(&S_C[tid / 4 + 8][0]) + tid % 4) = RC[1];

    if(tid < 16)//1个lane搬运一整行
    {
        (float4 &)dC[(row + tid) * N + col]= (float4 &)S_C[tid][0];
    }
}

PLAYGROUND_MATMUL_DEC(float16_t, 17, M, N, K, A, B, C)
{
    const int BLOCK_DIM_y = 1;
    const int BLOCK_DIM_x = 32;


    // const int WARP_SIZE = 32;
    const int BM = 16;
    const int BN = 8;

    int num_block_y = (M + BM - 1) / BM;
    int num_block_x = (N + BN - 1) / BN;

    dim3 block_dim(BLOCK_DIM_x, BLOCK_DIM_y, 1);
    dim3 grid_dim(num_block_x, num_block_y, 1);
    
    
    
    matrixKernel17s<<<grid_dim, block_dim>>>(
        const_cast<float16_t*>(A), 
        const_cast<float16_t*>(B), 
        const_cast<float16_t*>(C), 
        M, K, N
    );

    cudaDeviceSynchronize();
}

}  // namespace playground
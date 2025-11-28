
#include "playground/matmul.hpp"
#include "playground/system.hpp"

#include <cuda_fp16.h>  // 包含半精度支持
#include <mma.h>             // ← 关键：WMMA 支持
using namespace nvcuda;      // ← 关键：让 wmma:: 可用

#define FLOAT4(ptr) (*reinterpret_cast<float4*>(&(ptr)))
namespace playground
{

//BM = 128，BN = 256，BK = 32，thread_per_block = 256
template <int BM, int BK, int BN>//代表WARP处理的维度由[16, 16] -> [16 * W_M, 16 * W_N]
__global__ void matrixKernel10s(half *dA, half *dB, half *dC, int M, int K, int N)
{

    const int padding = 8;
    //以[64, 64]对结果进行切割
    __shared__ half S_A[BM][BK + padding];
    __shared__ half S_B[BK][BN + padding];
    
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> left_frag[2][4];//对这个矩阵进行转置
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> right_frag[2][4];//[32, 64]->[2,4]
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> c_frag[4][4];

    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int warp_id = tid / 32;
    int warp_id_x = warp_id % 4;
    int warp_id_y = warp_id / 4; 
    
    int S_A_row, S_A_col, S_B_row, S_B_col;

    //更换读写连续性，每次都读连续的行
    S_A_row = (tid / (BK / 8)) * 2;//因为一个thread可以拿8个所以用BK / 8 
    S_A_col = tid % (BK / 8);

    //更换读写连续性，每次都读连续的行
    S_B_row = (tid / (BN / 8)) * 4;
    S_B_col = tid % (BN / 8);

    int row_offset, col_offset;
    row_offset = blockIdx.y * BM;
    col_offset = blockIdx.x * BN;



    //初始化
    #pragma unroll
    for(int i = 0; i < 64 / 16; i++)
    {
        #pragma unroll
        for(int j = 0; j < 64 / 16; j++)
        {
            wmma::fill_fragment(c_frag[i][j], 0.0f);
        }
    }
    for(int split_k_id = 0; split_k_id < K / BK; split_k_id++)
    {
        //dA->S_A
        FLOAT4(S_A[S_A_row    ][S_A_col * 8]) =  FLOAT4(dA[(S_A_row + row_offset) * K + (S_A_col * 8 + split_k_id * BK)]);
        FLOAT4(S_A[S_A_row + 1][S_A_col * 8]) = FLOAT4(dA[(S_A_row + 1 + row_offset) * K + (S_A_col * 8 + split_k_id * BK)]);//改成+1
        //dB->S_B
        FLOAT4(S_B[S_B_row    ][S_B_col * 8]) = FLOAT4(dB[(S_B_row + split_k_id * BK) * N + (S_B_col * 8 + col_offset)]);
        FLOAT4(S_B[S_B_row + 1][S_B_col * 8]) = FLOAT4(dB[(S_B_row + 1 + split_k_id * BK) * N + (S_B_col * 8 + col_offset)]);//改成+1
        FLOAT4(S_B[S_B_row + 2][S_B_col * 8]) = FLOAT4(dB[(S_B_row + 2 + split_k_id * BK) * N + (S_B_col * 8 + col_offset)]);//改成+2
        FLOAT4(S_B[S_B_row + 3][S_B_col * 8]) = FLOAT4(dB[(S_B_row + 3 + split_k_id * BK) * N + (S_B_col * 8 + col_offset)]);//改成+3

        __syncthreads();

        //S_A -> left_matrix[64, 32]//对这个矩阵进行转置
        #pragma unroll
        for(int row = 0; row < 64 / 16; row++)
        {
            #pragma unroll
            for(int  col = 0; col < 32 / 16; col++)
            {
                wmma::load_matrix_sync(left_frag[col][row], &S_A[16 * row + 64 * warp_id_y][16 * col], BK + padding);
            }
        }

        //S_B -> right_matrix[32, 64]
        #pragma unroll
        for(int row = 0; row < 32 / 16; row++)
        {
            #pragma unroll
            for(int  col = 0; col < 64 / 16; col++)
            {
                wmma::load_matrix_sync(right_frag[row][col], &S_B[16 * row][16 * col + 64 * warp_id_x], BN + padding);
            }
        }
        
        
        //left_matrix * right_matrix

        #pragma unroll
        for(int row = 0; row < 4; row++)
        {
            #pragma unroll
            for(int col = 0; col < 4; col++)
            {
                wmma::mma_sync(c_frag[row][col], left_frag[0][row], right_frag[0][col], c_frag[row][col]);//因为转置改变计算逻辑
                wmma::mma_sync(c_frag[row][col], left_frag[1][row], right_frag[1][col], c_frag[row][col]);//因为转置改变计算逻辑
            }
        }

        __syncthreads();
       
    }

    // //搬出
    #pragma unroll
    for(int row = 0; row < 64 / 16; row++)
    {
        #pragma unroll
        for(int col = 0; col < 64 / 16; col++)
        {
            wmma::store_matrix_sync(dC + (16 * row + 64 * warp_id_y + row_offset) * N + 16 * col + 64 * warp_id_x + col_offset, c_frag[row][col], N, wmma::mem_row_major);
        }
    }
    
}

PLAYGROUND_MATMUL_DEC(float16_t, 10, M, N, K, A, B, C)
{

    //因为还需要重排序,256 设 8 个warp
    const int BLOCK_DIM_y = 16;
    const int BLOCK_DIM_x = 16;
    
    
    //每个block的处理维度
    const int BM = 128;
    const int BK = 32;
    const int BN = 256;

    int num_block_y = (M + BM - 1) / BM;
    int num_block_x = (N + BN - 1) / BN;

    dim3 block_dim( BLOCK_DIM_x, BLOCK_DIM_y, 1);
    dim3 grid_dim(num_block_x, num_block_y, 1);
    
    matrixKernel10s<BM, BK, BN><<<grid_dim, block_dim>>>(const_cast<float16_t*>(A), const_cast<float16_t*>(B), const_cast<float16_t*>(C), M, K, N);

    cudaDeviceSynchronize();
}
}  // namespace playground
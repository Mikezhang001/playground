
#include "playground/matmul.hpp"
#include "playground/system.hpp"

#include <cuda_fp16.h>  // 包含半精度支持
#include <mma.h>             // ← 关键：WMMA 支持
using namespace nvcuda;      // ← 关键：让 wmma:: 可用

namespace playground
{

//BM = 128，BN = 256，BK = 32，thread_per_block = 256
__global__ void matrixKernel13s(half *dA, half *dB, half *dC, int M, int K, int N)
{

    const int BM = 128;
    const int BK = 32;
    const int BN = 256;

    const int padding = 8;
    //以[64, 64]对结果进行切割

    //开double buffer 超过48K需要额外配置共享内存

    // __shared__ half S_A[BM * (BK + padding)];//一维数组，且空间开为之前的两倍
    // __shared__ half S_B[BK * (BN + padding)];
    extern __shared__ half smem[];

    half *S_A = smem;
    half *S_B = smem + 2 * BM * (BK + padding);

    
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

    //采用异步拷贝后，S_A和S_B的寻址需要变成单字节寻址
    int S_A_base_addr = __cvta_generic_to_shared(S_A);//不能用S_A[0],传入的是地址
    int S_B_base_addr = __cvta_generic_to_shared(S_B);
    int S_A_addr0, S_A_addr1, S_B_addr0, S_B_addr1, S_B_addr2, S_B_addr3;

    //这部分是常量
    S_A_addr0 = S_A_base_addr + ((S_A_row) * (BK + padding) + S_A_col * 8) * sizeof(half);
    S_A_addr1 = S_A_base_addr + ((S_A_row + 1) * (BK + padding) + S_A_col * 8) * sizeof(half);
    S_B_addr0 = S_B_base_addr + ((S_B_row) * (BN + padding) + S_B_col * 8) * sizeof(half);
    S_B_addr1 = S_B_base_addr + ((S_B_row + 1) * (BN + padding) + S_B_col * 8) * sizeof(half);
    S_B_addr2 = S_B_base_addr + ((S_B_row + 2) * (BN + padding) + S_B_col * 8) * sizeof(half);
    S_B_addr3 = S_B_base_addr + ((S_B_row + 3) * (BN + padding) + S_B_col * 8) * sizeof(half);

    //double buffer部分开始

    //第0轮数据的搬入
    int split_k_id = 0;
    //dA->S_A
    asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
        : "l"(S_A_addr0 + (split_k_id % 2) * BM * (BK + padding) * sizeof(half)), "l"(&dA[(S_A_row + row_offset) * K + (S_A_col * 8 + split_k_id * BK)]));
    asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
        : "l"(S_A_addr1 + (split_k_id % 2) * BM * (BK + padding) * sizeof(half)), "l"(&dA[(S_A_row + 1 + row_offset) * K + (S_A_col * 8 + split_k_id * BK)]));
        //dB->S_B
    asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
        : "l"(S_B_addr0 + (split_k_id % 2) * BK * (BN + padding) * sizeof(half)), "l"(&dB[(S_B_row + split_k_id * BK) * N + (S_B_col * 8 + col_offset)]));
    asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
        : "l"(S_B_addr1 + (split_k_id % 2) * BK * (BN + padding) * sizeof(half)), "l"(&dB[(S_B_row + 1 + split_k_id * BK) * N + (S_B_col * 8 + col_offset)]));
    asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
        : "l"(S_B_addr2 + (split_k_id % 2) * BK * (BN + padding) * sizeof(half)), "l"(&dB[(S_B_row + 2 + split_k_id * BK) * N + (S_B_col * 8 + col_offset)]));
    asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
        : "l"(S_B_addr3 + (split_k_id % 2) * BK * (BN + padding) * sizeof(half)), "l"(&dB[(S_B_row + 3 + split_k_id * BK) * N + (S_B_col * 8 + col_offset)]));

    //PTX 汇编指令提交和同步
    asm ("cp.async.commit_group;\n" ::);
    asm ("cp.async.wait_group 0;\n" ::);
    __syncthreads();

    #pragma unroll 32
    for(split_k_id = 1; split_k_id < K / BK; split_k_id++)
    {

    
        // //dA->S_A
        // FLOAT4(S_A[S_A_row    ][S_A_col * 8]) = FLOAT4(dA[(S_A_row + row_offset) * K + (S_A_col * 8 + split_k_id * BK)]);
        // FLOAT4(S_A[S_A_row + 1][S_A_col * 8]) = FLOAT4(dA[(S_A_row + 1 + row_offset) * K + (S_A_col * 8 + split_k_id * BK)]);//改成+1
        // //dB->S_B
        // FLOAT4(S_B[S_B_row    ][S_B_col * 8]) = FLOAT4(dB[(S_B_row + split_k_id * BK) * N + (S_B_col * 8 + col_offset)]);
        // FLOAT4(S_B[S_B_row + 1][S_B_col * 8]) = FLOAT4(dB[(S_B_row + 1 + split_k_id * BK) * N + (S_B_col * 8 + col_offset)]);//改成+1
        // FLOAT4(S_B[S_B_row + 2][S_B_col * 8]) = FLOAT4(dB[(S_B_row + 2 + split_k_id * BK) * N + (S_B_col * 8 + col_offset)]);//改成+2
        // FLOAT4(S_B[S_B_row + 3][S_B_col * 8]) = FLOAT4(dB[(S_B_row + 3 + split_k_id * BK) * N + (S_B_col * 8 + col_offset)]);//改成+3


        //第split_k_id轮数据搬入shared memory
        //dA->S_A
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "l"(S_A_addr0 + (split_k_id % 2) * BM * (BK + padding) * sizeof(half)), "l"(&dA[(S_A_row + row_offset) * K + (S_A_col * 8 + split_k_id * BK)]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "l"(S_A_addr1 + (split_k_id % 2) * BM * (BK + padding) * sizeof(half)), "l"(&dA[(S_A_row + 1 + row_offset) * K + (S_A_col * 8 + split_k_id * BK)]));
        //dB->S_B
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "l"(S_B_addr0 + (split_k_id % 2) * BK * (BN + padding) * sizeof(half)), "l"(&dB[(S_B_row + split_k_id * BK) * N + (S_B_col * 8 + col_offset)]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "l"(S_B_addr1 + (split_k_id % 2) * BK * (BN + padding) * sizeof(half)), "l"(&dB[(S_B_row + 1 + split_k_id * BK) * N + (S_B_col * 8 + col_offset)]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "l"(S_B_addr2 + (split_k_id % 2) * BK * (BN + padding) * sizeof(half)), "l"(&dB[(S_B_row + 2 + split_k_id * BK) * N + (S_B_col * 8 + col_offset)]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "l"(S_B_addr3 + (split_k_id % 2) * BK * (BN + padding) * sizeof(half)), "l"(&dB[(S_B_row + 3 + split_k_id * BK) * N + (S_B_col * 8 + col_offset)]));

        

        //PTX 汇编指令提交和同步往后延
        // asm ("cp.async.commit_group;\n" ::);
        // asm ("cp.async.wait_group 0;\n" ::);

        //将第split_k_id - 1轮数据搬入矩阵
        #pragma unroll
        for(int row = 0; row < 64 / 16; row++)
        {
            #pragma unroll
            for(int  col = 0; col < 32 / 16; col++)
            {
                wmma::load_matrix_sync(left_frag[col][row], &S_A[((split_k_id - 1) % 2) * BM * (BK + padding) + (16 * row + 64 * warp_id_y) * (BK + padding) + 16 * col], BK + padding);//改成一维度的寻址
            }
        }
        #pragma unroll
        for(int row = 0; row < 32 / 16; row++)
        {
            #pragma unroll
            for(int  col = 0; col < 64 / 16; col++)
            {
                wmma::load_matrix_sync(right_frag[row][col], &S_B[((split_k_id - 1) % 2) * BK * (BN + padding) + (16 * row) * (BN + padding) + 16 * col + 64 * warp_id_x], BN + padding);//改成一维度的寻址
            }
        }
        
        //将搬入第split_k_id - 1轮搬入矩阵计算
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

        //PTX 汇编指令提交和同步后延
        asm ("cp.async.commit_group;\n" ::);
        asm ("cp.async.wait_group 0;\n" ::);
        __syncthreads();
       
    }

    //将第split_k_id - 1轮数据搬入矩阵
    #pragma unroll
    for(int row = 0; row < 64 / 16; row++)
    {
        #pragma unroll
        for(int  col = 0; col < 32 / 16; col++)
        {
            wmma::load_matrix_sync(left_frag[col][row], &S_A[((split_k_id - 1) % 2) * BM * (BK + padding) + (16 * row + 64 * warp_id_y) * (BK + padding) + 16 * col], BK + padding);//改成一维度的寻址
        }
    }
    #pragma unroll
    for(int row = 0; row < 32 / 16; row++)
    {
        #pragma unroll
        for(int  col = 0; col < 64 / 16; col++)
        {
            wmma::load_matrix_sync(right_frag[row][col], &S_B[((split_k_id - 1) % 2) * BK * (BN + padding) + (16 * row) * (BN + padding) + 16 * col + 64 * warp_id_x], BN + padding);//改成一维度的寻址
        }
    }
        
    //将搬入第split_k_id - 1轮搬入矩阵计算
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

PLAYGROUND_MATMUL_DEC(float16_t, 13, M, N, K, A, B, C)
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
    
    cudaFuncSetAttribute(matrixKernel13s, cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);//只能传不带模板的进去, 96KB
    unsigned int dsmem = 2 * (BM * (BK + 8) + BK * (BN + 8)) * sizeof(half);
    matrixKernel13s<<<grid_dim, block_dim, dsmem>>>(const_cast<float16_t*>(A), const_cast<float16_t*>(B), const_cast<float16_t*>(C), M, K, N);

    cudaDeviceSynchronize();
}
}  // namespace playground
#include "playground/matmul.hpp"
#include "playground/system.hpp"

#include <cuda_fp16.h>
#include <mma.h>
using namespace nvcuda;

namespace playground
{

// 真正的优化：增加BK到64，减少循环次数和同步开销
__global__ void matrixKernel16s(half *dA, half *dB, half *dC, int M, int K, int N)
{
    // 关键优化：BK从32增加到64
    const int BM = 128;
    const int BK = 64;  // ← 主要优化点
    const int BN = 256;
    const int padding = 8;

    extern __shared__ half smem[];
    half *S_A = smem;
    half *S_B = smem + 2 * BM * (BK + padding);

    // BK=64需要4个K方向的fragment
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> left_frag[4][4];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> right_frag[4][4];
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> c_frag[4][4];

    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int warp_id = tid / 32;
    int warp_id_x = warp_id % 4;
    int warp_id_y = warp_id / 4;
    
    // 调整加载模式以适应BK=64
    int S_A_row = (tid / (BK / 8)) * 2;
    int S_A_col = tid % (BK / 8);
    int S_B_row = (tid / (BN / 8)) * 4;
    int S_B_col = tid % (BN / 8);

    int row_offset = blockIdx.y * BM;
    int col_offset = blockIdx.x * BN;

    // 初始化
    #pragma unroll
    for(int i = 0; i < 4; i++) {
        #pragma unroll
        for(int j = 0; j < 4; j++) {
            wmma::fill_fragment(c_frag[i][j], 0.0f);
        }
    }

    int S_A_base_addr = __cvta_generic_to_shared(S_A);
    int S_B_base_addr = __cvta_generic_to_shared(S_B);
    
    int S_A_addr0 = S_A_base_addr + ((S_A_row) * (BK + padding) + S_A_col * 8) * sizeof(half);
    int S_A_addr1 = S_A_base_addr + ((S_A_row + 1) * (BK + padding) + S_A_col * 8) * sizeof(half);
    int S_B_addr0 = S_B_base_addr + ((S_B_row) * (BN + padding) + S_B_col * 8) * sizeof(half);
    int S_B_addr1 = S_B_base_addr + ((S_B_row + 1) * (BN + padding) + S_B_col * 8) * sizeof(half);
    int S_B_addr2 = S_B_base_addr + ((S_B_row + 2) * (BN + padding) + S_B_col * 8) * sizeof(half);
    int S_B_addr3 = S_B_base_addr + ((S_B_row + 3) * (BN + padding) + S_B_col * 8) * sizeof(half);

    // 预加载第0轮
    int split_k_id = 0;
    asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
        : "l"(S_A_addr0 + (split_k_id % 2) * BM * (BK + padding) * sizeof(half)), 
          "l"(&dA[(S_A_row + row_offset) * K + (S_A_col * 8 + split_k_id * BK)]));
    asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
        : "l"(S_A_addr1 + (split_k_id % 2) * BM * (BK + padding) * sizeof(half)), 
          "l"(&dA[(S_A_row + 1 + row_offset) * K + (S_A_col * 8 + split_k_id * BK)]));
    
    asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
        : "l"(S_B_addr0 + (split_k_id % 2) * BK * (BN + padding) * sizeof(half)), 
          "l"(&dB[(S_B_row + split_k_id * BK) * N + (S_B_col * 8 + col_offset)]));
    asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
        : "l"(S_B_addr1 + (split_k_id % 2) * BK * (BN + padding) * sizeof(half)), 
          "l"(&dB[(S_B_row + 1 + split_k_id * BK) * N + (S_B_col * 8 + col_offset)]));
    asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
        : "l"(S_B_addr2 + (split_k_id % 2) * BK * (BN + padding) * sizeof(half)), 
          "l"(&dB[(S_B_row + 2 + split_k_id * BK) * N + (S_B_col * 8 + col_offset)]));
    asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
        : "l"(S_B_addr3 + (split_k_id % 2) * BK * (BN + padding) * sizeof(half)), 
          "l"(&dB[(S_B_row + 3 + split_k_id * BK) * N + (S_B_col * 8 + col_offset)]));

    asm ("cp.async.commit_group;\n" ::);
    asm ("cp.async.wait_group 0;\n" ::);
    __syncthreads();

    // 主循环 - 循环次数减半 (4096/64 = 64 vs 4096/32 = 128)
    for(split_k_id = 1; split_k_id < K / BK; split_k_id++)
    {
        // 异步加载
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "l"(S_A_addr0 + (split_k_id % 2) * BM * (BK + padding) * sizeof(half)), 
              "l"(&dA[(S_A_row + row_offset) * K + (S_A_col * 8 + split_k_id * BK)]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "l"(S_A_addr1 + (split_k_id % 2) * BM * (BK + padding) * sizeof(half)), 
              "l"(&dA[(S_A_row + 1 + row_offset) * K + (S_A_col * 8 + split_k_id * BK)]));
        
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "l"(S_B_addr0 + (split_k_id % 2) * BK * (BN + padding) * sizeof(half)), 
              "l"(&dB[(S_B_row + split_k_id * BK) * N + (S_B_col * 8 + col_offset)]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "l"(S_B_addr1 + (split_k_id % 2) * BK * (BN + padding) * sizeof(half)), 
              "l"(&dB[(S_B_row + 1 + split_k_id * BK) * N + (S_B_col * 8 + col_offset)]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "l"(S_B_addr2 + (split_k_id % 2) * BK * (BN + padding) * sizeof(half)), 
              "l"(&dB[(S_B_row + 2 + split_k_id * BK) * N + (S_B_col * 8 + col_offset)]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :
            : "l"(S_B_addr3 + (split_k_id % 2) * BK * (BN + padding) * sizeof(half)), 
              "l"(&dB[(S_B_row + 3 + split_k_id * BK) * N + (S_B_col * 8 + col_offset)]));

        // 加载上一轮 - BK=64需要加载4个K fragment
        int prev_buf = ((split_k_id - 1) % 2) * BM * (BK + padding);
        int prev_buf_B = ((split_k_id - 1) % 2) * BK * (BN + padding);

        // 加载A矩阵 - 4行 × 4个K块
        #pragma unroll
        for(int row = 0; row < 4; row++) {
            #pragma unroll
            for(int k_frag = 0; k_frag < 4; k_frag++) {
                wmma::load_matrix_sync(
                    left_frag[k_frag][row], 
                    &S_A[prev_buf + (16 * row + 64 * warp_id_y) * (BK + padding) + 16 * k_frag], 
                    BK + padding
                );
            }
        }

        // 加载B矩阵 - 4个K块 × 4列
        #pragma unroll
        for(int k_frag = 0; k_frag < 4; k_frag++) {
            #pragma unroll
            for(int col = 0; col < 4; col++) {
                wmma::load_matrix_sync(
                    right_frag[k_frag][col], 
                    &S_B[prev_buf_B + (16 * k_frag) * (BN + padding) + 16 * col + 64 * warp_id_x], 
                    BN + padding
                );
            }
        }

        // 计算 - 4个K块累加
        #pragma unroll
        for(int k_frag = 0; k_frag < 4; k_frag++) {
            #pragma unroll
            for(int row = 0; row < 4; row++) {
                #pragma unroll
                for(int col = 0; col < 4; col++) {
                    wmma::mma_sync(c_frag[row][col], left_frag[k_frag][row], right_frag[k_frag][col], c_frag[row][col]);
                }
            }
        }

        asm ("cp.async.commit_group;\n" ::);
        asm ("cp.async.wait_group 0;\n" ::);
        __syncthreads();
    }

    // 最后一轮
    int prev_buf = ((split_k_id - 1) % 2) * BM * (BK + padding);
    int prev_buf_B = ((split_k_id - 1) % 2) * BK * (BN + padding);

    #pragma unroll
    for(int row = 0; row < 4; row++) {
        #pragma unroll
        for(int k_frag = 0; k_frag < 4; k_frag++) {
            wmma::load_matrix_sync(
                left_frag[k_frag][row], 
                &S_A[prev_buf + (16 * row + 64 * warp_id_y) * (BK + padding) + 16 * k_frag], 
                BK + padding
            );
        }
    }

    #pragma unroll
    for(int k_frag = 0; k_frag < 4; k_frag++) {
        #pragma unroll
        for(int col = 0; col < 4; col++) {
            wmma::load_matrix_sync(
                right_frag[k_frag][col], 
                &S_B[prev_buf_B + (16 * k_frag) * (BN + padding) + 16 * col + 64 * warp_id_x], 
                BN + padding
            );
        }
    }

    #pragma unroll
    for(int k_frag = 0; k_frag < 4; k_frag++) {
        #pragma unroll
        for(int row = 0; row < 4; row++) {
            #pragma unroll
            for(int col = 0; col < 4; col++) {
                wmma::mma_sync(c_frag[row][col], left_frag[k_frag][row], right_frag[k_frag][col], c_frag[row][col]);
            }
        }
    }

    __syncthreads();

    // 写回
    #pragma unroll
    for(int row = 0; row < 4; row++) {
        #pragma unroll
        for(int col = 0; col < 4; col++) {
            wmma::store_matrix_sync(
                dC + (16 * row + 64 * warp_id_y + row_offset) * N + 16 * col + 64 * warp_id_x + col_offset, 
                c_frag[row][col], 
                N, 
                wmma::mem_row_major
            );
        }
    }
}

PLAYGROUND_MATMUL_DEC(float16_t, 16, M, N, K, A, B, C)
{
    const int BLOCK_DIM_y = 16;
    const int BLOCK_DIM_x = 16;
    
    const int BM = 128;
    const int BK = 64;  // ← 增加BK
    const int BN = 256;

    int num_block_y = (M + BM - 1) / BM;
    int num_block_x = (N + BN - 1) / BN;

    dim3 block_dim(BLOCK_DIM_x, BLOCK_DIM_y, 1);
    dim3 grid_dim(num_block_x, num_block_y, 1);
    
    cudaFuncSetAttribute(
        matrixKernel16s, 
        cudaFuncAttributeMaxDynamicSharedMemorySize, 
        98304
    );
    
    // 增加shared memory大小
    unsigned int dsmem = 2 * (BM * (BK + 8) + BK * (BN + 8)) * sizeof(half);
    
    matrixKernel16s<<<grid_dim, block_dim, dsmem>>>(
        const_cast<float16_t*>(A), 
        const_cast<float16_t*>(B), 
        const_cast<float16_t*>(C), 
        M, K, N
    );

    cudaDeviceSynchronize();
}

}  // namespace playground
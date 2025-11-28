
#include "playground/matmul.hpp"
#include "playground/system.hpp"

#include <cuda_fp16.h>  // 包含半精度支持
#include <mma.h>             // ← 关键：WMMA 支持
using namespace nvcuda;      // ← 关键：让 wmma:: 可用

namespace playground
{

//BM = 128，BN = 256，BK = 32，thread_per_block = 256
__global__ void matrixKernel15s(half *dA, half *dB, half *dC, int M, int K, int N)
{
    const int BM = 128;
    const int BK = 32;
    const int BN = 256;
    const int padding = 8;

    extern __shared__ half smem[];
    half *S_A = smem;
    half *S_B = smem + 3 * BM * (BK + padding);  // ← 修正：3 buffers for A

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> left_frag[2][4];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> right_frag[2][4];
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> c_frag[4][4];

    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int warp_id = tid / 32;
    int warp_id_x = warp_id % 4;
    int warp_id_y = warp_id / 4;

    const int S_A_row = (tid / (BK / 8)) * 2;
    const int S_A_col = tid % (BK / 8);
    const int S_B_row = (tid / (BN / 8)) * 4;
    const int S_B_col = tid % (BN / 8);

    int row_offset = blockIdx.y * BM;
    int col_offset = blockIdx.x * BN;

    // Initialize C
    #pragma unroll
    for(int i = 0; i < 4; i++) {
        #pragma unroll
        for(int j = 0; j < 4; j++) {
            wmma::fill_fragment(c_frag[i][j], 0.0f);
        }
    }

    int S_A_base_addr = __cvta_generic_to_shared(S_A);
    int S_B_base_addr = __cvta_generic_to_shared(S_B);

    // Precompute constant offsets (byte addresses)
    int S_A_off0 = ((S_A_row + 0) * (BK + padding) + S_A_col * 8) * sizeof(half);
    int S_A_off1 = ((S_A_row + 1) * (BK + padding) + S_A_col * 8) * sizeof(half);
    int S_B_off0 = ((S_B_row + 0) * (BN + padding) + S_B_col * 8) * sizeof(half);
    int S_B_off1 = ((S_B_row + 1) * (BN + padding) + S_B_col * 8) * sizeof(half);
    int S_B_off2 = ((S_B_row + 2) * (BN + padding) + S_B_col * 8) * sizeof(half);
    int S_B_off3 = ((S_B_row + 3) * (BN + padding) + S_B_col * 8) * sizeof(half);

    int K_tiles = K / BK;  // assume K divisible by BK for simplicity (as in your code)

    // --- Prefetch k = 0 ---
    asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "l"(S_A_base_addr + 0 * BM * (BK + padding) * sizeof(half) + S_A_off0), "l"(&dA[(S_A_row + row_offset) * K + S_A_col * 8 + 0 * BK]));
    asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "l"(S_A_base_addr + 0 * BM * (BK + padding) * sizeof(half) + S_A_off1), "l"(&dA[(S_A_row + 1 + row_offset) * K + S_A_col * 8 + 0 * BK]));
    asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "l"(S_B_base_addr + 0 * BK * (BN + padding) * sizeof(half) + S_B_off0), "l"(&dB[(S_B_row + 0 * BK) * N + S_B_col * 8 + col_offset]));
    asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "l"(S_B_base_addr + 0 * BK * (BN + padding) * sizeof(half) + S_B_off1), "l"(&dB[(S_B_row + 1 + 0 * BK) * N + S_B_col * 8 + col_offset]));
    asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "l"(S_B_base_addr + 0 * BK * (BN + padding) * sizeof(half) + S_B_off2), "l"(&dB[(S_B_row + 2 + 0 * BK) * N + S_B_col * 8 + col_offset]));
    asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "l"(S_B_base_addr + 0 * BK * (BN + padding) * sizeof(half) + S_B_off3), "l"(&dB[(S_B_row + 3 + 0 * BK) * N + S_B_col * 8 + col_offset]));

    // --- Prefetch k = 1 ---
    asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "l"(S_A_base_addr + 1 * BM * (BK + padding) * sizeof(half) + S_A_off0), "l"(&dA[(S_A_row + row_offset) * K + S_A_col * 8 + 1 * BK]));
    asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "l"(S_A_base_addr + 1 * BM * (BK + padding) * sizeof(half) + S_A_off1), "l"(&dA[(S_A_row + 1 + row_offset) * K + S_A_col * 8 + 1 * BK]));
    asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "l"(S_B_base_addr + 1 * BK * (BN + padding) * sizeof(half) + S_B_off0), "l"(&dB[(S_B_row + 1 * BK) * N + S_B_col * 8 + col_offset]));
    asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "l"(S_B_base_addr + 1 * BK * (BN + padding) * sizeof(half) + S_B_off1), "l"(&dB[(S_B_row + 1 + 1 * BK) * N + S_B_col * 8 + col_offset]));
    asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "l"(S_B_base_addr + 1 * BK * (BN + padding) * sizeof(half) + S_B_off2), "l"(&dB[(S_B_row + 2 + 1 * BK) * N + S_B_col * 8 + col_offset]));
    asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "l"(S_B_base_addr + 1 * BK * (BN + padding) * sizeof(half) + S_B_off3), "l"(&dB[(S_B_row + 3 + 1 * BK) * N + S_B_col * 8 + col_offset]));

    asm ("cp.async.commit_group;\n" ::);
    asm ("cp.async.wait_group 0;\n" ::);
    __syncthreads();

    int split_k_id;

    #pragma unroll 32
    for (split_k_id = 2; split_k_id < K_tiles; split_k_id++)
    {
        // --- Prefetch current k = split_k_id ---
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "l"(S_A_base_addr + (split_k_id % 3) * BM * (BK + padding) * sizeof(half) + S_A_off0), "l"(&dA[(S_A_row + row_offset) * K + S_A_col * 8 + split_k_id * BK]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "l"(S_A_base_addr + (split_k_id % 3) * BM * (BK + padding) * sizeof(half) + S_A_off1), "l"(&dA[(S_A_row + 1 + row_offset) * K + S_A_col * 8 + split_k_id * BK]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "l"(S_B_base_addr + (split_k_id % 3) * BK * (BN + padding) * sizeof(half) + S_B_off0), "l"(&dB[(S_B_row + split_k_id * BK) * N + S_B_col * 8 + col_offset]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "l"(S_B_base_addr + (split_k_id % 3) * BK * (BN + padding) * sizeof(half) + S_B_off1), "l"(&dB[(S_B_row + 1 + split_k_id * BK) * N + S_B_col * 8 + col_offset]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "l"(S_B_base_addr + (split_k_id % 3) * BK * (BN + padding) * sizeof(half) + S_B_off2), "l"(&dB[(S_B_row + 2 + split_k_id * BK) * N + S_B_col * 8 + col_offset]));
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "l"(S_B_base_addr + (split_k_id % 3) * BK * (BN + padding) * sizeof(half) + S_B_off3), "l"(&dB[(S_B_row + 3 + split_k_id * BK) * N + S_B_col * 8 + col_offset]));

        // --- Compute k = split_k_id - 2 ---
        // int buf = (split_k_id - 2) % 3;

        wmma::load_matrix_sync(left_frag[0][0], &S_A[(split_k_id - 2) % 3 * BM * (BK + padding) + (16 * 0 + 64 * warp_id_y) * (BK + padding) + 16 * 0], BK + padding);
        wmma::load_matrix_sync(left_frag[1][0], &S_A[(split_k_id - 2) % 3 * BM * (BK + padding) + (16 * 0 + 64 * warp_id_y) * (BK + padding) + 16 * 1], BK + padding);
        wmma::load_matrix_sync(left_frag[0][1], &S_A[(split_k_id - 2) % 3 * BM * (BK + padding) + (16 * 1 + 64 * warp_id_y) * (BK + padding) + 16 * 0], BK + padding);
        wmma::load_matrix_sync(left_frag[1][1], &S_A[(split_k_id - 2) % 3 * BM * (BK + padding) + (16 * 1 + 64 * warp_id_y) * (BK + padding) + 16 * 1], BK + padding);
        wmma::load_matrix_sync(left_frag[0][2], &S_A[(split_k_id - 2) % 3 * BM * (BK + padding) + (16 * 2 + 64 * warp_id_y) * (BK + padding) + 16 * 0], BK + padding);
        wmma::load_matrix_sync(left_frag[1][2], &S_A[(split_k_id - 2) % 3 * BM * (BK + padding) + (16 * 2 + 64 * warp_id_y) * (BK + padding) + 16 * 1], BK + padding);
        wmma::load_matrix_sync(left_frag[0][3], &S_A[(split_k_id - 2) % 3 * BM * (BK + padding) + (16 * 3 + 64 * warp_id_y) * (BK + padding) + 16 * 0], BK + padding);
        wmma::load_matrix_sync(left_frag[1][3], &S_A[(split_k_id - 2) % 3 * BM * (BK + padding) + (16 * 3 + 64 * warp_id_y) * (BK + padding) + 16 * 1], BK + padding);

        wmma::load_matrix_sync(right_frag[0][0], &S_B[(split_k_id - 2) % 3 * BK * (BN + padding) + (16 * 0) * (BN + padding) + 16 * 0 + 64 * warp_id_x], BN + padding);
        wmma::load_matrix_sync(right_frag[0][1], &S_B[(split_k_id - 2) % 3 * BK * (BN + padding) + (16 * 0) * (BN + padding) + 16 * 1 + 64 * warp_id_x], BN + padding);
        wmma::load_matrix_sync(right_frag[0][2], &S_B[(split_k_id - 2) % 3 * BK * (BN + padding) + (16 * 0) * (BN + padding) + 16 * 2 + 64 * warp_id_x], BN + padding);
        wmma::load_matrix_sync(right_frag[0][3], &S_B[(split_k_id - 2) % 3 * BK * (BN + padding) + (16 * 0) * (BN + padding) + 16 * 3 + 64 * warp_id_x], BN + padding);
        wmma::load_matrix_sync(right_frag[1][0], &S_B[(split_k_id - 2) % 3 * BK * (BN + padding) + (16 * 1) * (BN + padding) + 16 * 0 + 64 * warp_id_x], BN + padding);
        wmma::load_matrix_sync(right_frag[1][1], &S_B[(split_k_id - 2) % 3 * BK * (BN + padding) + (16 * 1) * (BN + padding) + 16 * 1 + 64 * warp_id_x], BN + padding);
        wmma::load_matrix_sync(right_frag[1][2], &S_B[(split_k_id - 2) % 3 * BK * (BN + padding) + (16 * 1) * (BN + padding) + 16 * 2 + 64 * warp_id_x], BN + padding);
        wmma::load_matrix_sync(right_frag[1][3], &S_B[(split_k_id - 2) % 3 * BK * (BN + padding) + (16 * 1) * (BN + padding) + 16 * 3 + 64 * warp_id_x], BN + padding);

        #pragma unroll
        for(int row = 0; row < 4; row++) {
            #pragma unroll
            for(int col = 0; col < 4; col++) {
                wmma::mma_sync(c_frag[row][col], left_frag[0][row], right_frag[0][col], c_frag[row][col]);
                wmma::mma_sync(c_frag[row][col], left_frag[1][row], right_frag[1][col], c_frag[row][col]);
            }
        }

        asm ("cp.async.commit_group;\n" ::);
        asm ("cp.async.wait_group 0;\n" ::);
        __syncthreads();
    }

    // --- Compute k = K_tiles - 2 (i.e., split_k_id - 2 where split_k_id = K_tiles) ---
    
        // int buf = (split_k_id - 2) % 3;

        wmma::load_matrix_sync(left_frag[0][0], &S_A[(split_k_id - 2) % 3 * BM * (BK + padding) + (16 * 0 + 64 * warp_id_y) * (BK + padding) + 16 * 0], BK + padding);
        wmma::load_matrix_sync(left_frag[1][0], &S_A[(split_k_id - 2) % 3 * BM * (BK + padding) + (16 * 0 + 64 * warp_id_y) * (BK + padding) + 16 * 1], BK + padding);
        wmma::load_matrix_sync(left_frag[0][1], &S_A[(split_k_id - 2) % 3 * BM * (BK + padding) + (16 * 1 + 64 * warp_id_y) * (BK + padding) + 16 * 0], BK + padding);
        wmma::load_matrix_sync(left_frag[1][1], &S_A[(split_k_id - 2) % 3 * BM * (BK + padding) + (16 * 1 + 64 * warp_id_y) * (BK + padding) + 16 * 1], BK + padding);
        wmma::load_matrix_sync(left_frag[0][2], &S_A[(split_k_id - 2) % 3 * BM * (BK + padding) + (16 * 2 + 64 * warp_id_y) * (BK + padding) + 16 * 0], BK + padding);
        wmma::load_matrix_sync(left_frag[1][2], &S_A[(split_k_id - 2) % 3 * BM * (BK + padding) + (16 * 2 + 64 * warp_id_y) * (BK + padding) + 16 * 1], BK + padding);
        wmma::load_matrix_sync(left_frag[0][3], &S_A[(split_k_id - 2) % 3 * BM * (BK + padding) + (16 * 3 + 64 * warp_id_y) * (BK + padding) + 16 * 0], BK + padding);
        wmma::load_matrix_sync(left_frag[1][3], &S_A[(split_k_id - 2) % 3 * BM * (BK + padding) + (16 * 3 + 64 * warp_id_y) * (BK + padding) + 16 * 1], BK + padding);

        wmma::load_matrix_sync(right_frag[0][0], &S_B[(split_k_id - 2) % 3 * BK * (BN + padding) + (16 * 0) * (BN + padding) + 16 * 0 + 64 * warp_id_x], BN + padding);
        wmma::load_matrix_sync(right_frag[0][1], &S_B[(split_k_id - 2) % 3 * BK * (BN + padding) + (16 * 0) * (BN + padding) + 16 * 1 + 64 * warp_id_x], BN + padding);
        wmma::load_matrix_sync(right_frag[0][2], &S_B[(split_k_id - 2) % 3 * BK * (BN + padding) + (16 * 0) * (BN + padding) + 16 * 2 + 64 * warp_id_x], BN + padding);
        wmma::load_matrix_sync(right_frag[0][3], &S_B[(split_k_id - 2) % 3 * BK * (BN + padding) + (16 * 0) * (BN + padding) + 16 * 3 + 64 * warp_id_x], BN + padding);
        wmma::load_matrix_sync(right_frag[1][0], &S_B[(split_k_id - 2) % 3 * BK * (BN + padding) + (16 * 1) * (BN + padding) + 16 * 0 + 64 * warp_id_x], BN + padding);
        wmma::load_matrix_sync(right_frag[1][1], &S_B[(split_k_id - 2) % 3 * BK * (BN + padding) + (16 * 1) * (BN + padding) + 16 * 1 + 64 * warp_id_x], BN + padding);
        wmma::load_matrix_sync(right_frag[1][2], &S_B[(split_k_id - 2) % 3 * BK * (BN + padding) + (16 * 1) * (BN + padding) + 16 * 2 + 64 * warp_id_x], BN + padding);
        wmma::load_matrix_sync(right_frag[1][3], &S_B[(split_k_id - 2) % 3 * BK * (BN + padding) + (16 * 1) * (BN + padding) + 16 * 3 + 64 * warp_id_x], BN + padding);

        #pragma unroll
        for(int row = 0; row < 4; row++) {
            #pragma unroll
            for(int col = 0; col < 4; col++) {
                wmma::mma_sync(c_frag[row][col], left_frag[0][row], right_frag[0][col], c_frag[row][col]);
                wmma::mma_sync(c_frag[row][col], left_frag[1][row], right_frag[1][col], c_frag[row][col]);
            }
        }
        __syncthreads();
    

    // --- Compute k = K_tiles - 1 (i.e., split_k_id - 1) ---
    
        // int buf = (split_k_id - 1) % 3;

        wmma::load_matrix_sync(left_frag[0][0], &S_A[(split_k_id - 1) % 3 * BM * (BK + padding) + (16 * 0 + 64 * warp_id_y) * (BK + padding) + 16 * 0], BK + padding);
        wmma::load_matrix_sync(left_frag[1][0], &S_A[(split_k_id - 1) % 3 * BM * (BK + padding) + (16 * 0 + 64 * warp_id_y) * (BK + padding) + 16 * 1], BK + padding);
        wmma::load_matrix_sync(left_frag[0][1], &S_A[(split_k_id - 1) % 3 * BM * (BK + padding) + (16 * 1 + 64 * warp_id_y) * (BK + padding) + 16 * 0], BK + padding);
        wmma::load_matrix_sync(left_frag[1][1], &S_A[(split_k_id - 1) % 3 * BM * (BK + padding) + (16 * 1 + 64 * warp_id_y) * (BK + padding) + 16 * 1], BK + padding);
        wmma::load_matrix_sync(left_frag[0][2], &S_A[(split_k_id - 1) % 3 * BM * (BK + padding) + (16 * 2 + 64 * warp_id_y) * (BK + padding) + 16 * 0], BK + padding);
        wmma::load_matrix_sync(left_frag[1][2], &S_A[(split_k_id - 1) % 3 * BM * (BK + padding) + (16 * 2 + 64 * warp_id_y) * (BK + padding) + 16 * 1], BK + padding);
        wmma::load_matrix_sync(left_frag[0][3], &S_A[(split_k_id - 1) % 3 * BM * (BK + padding) + (16 * 3 + 64 * warp_id_y) * (BK + padding) + 16 * 0], BK + padding);
        wmma::load_matrix_sync(left_frag[1][3], &S_A[(split_k_id - 1) % 3 * BM * (BK + padding) + (16 * 3 + 64 * warp_id_y) * (BK + padding) + 16 * 1], BK + padding);

        wmma::load_matrix_sync(right_frag[0][0], &S_B[(split_k_id - 1) % 3 * BK * (BN + padding) + (16 * 0) * (BN + padding) + 16 * 0 + 64 * warp_id_x], BN + padding);
        wmma::load_matrix_sync(right_frag[0][1], &S_B[(split_k_id - 1) % 3 * BK * (BN + padding) + (16 * 0) * (BN + padding) + 16 * 1 + 64 * warp_id_x], BN + padding);
        wmma::load_matrix_sync(right_frag[0][2], &S_B[(split_k_id - 1) % 3 * BK * (BN + padding) + (16 * 0) * (BN + padding) + 16 * 2 + 64 * warp_id_x], BN + padding);
        wmma::load_matrix_sync(right_frag[0][3], &S_B[(split_k_id - 1) % 3 * BK * (BN + padding) + (16 * 0) * (BN + padding) + 16 * 3 + 64 * warp_id_x], BN + padding);
        wmma::load_matrix_sync(right_frag[1][0], &S_B[(split_k_id - 1) % 3 * BK * (BN + padding) + (16 * 1) * (BN + padding) + 16 * 0 + 64 * warp_id_x], BN + padding);
        wmma::load_matrix_sync(right_frag[1][1], &S_B[(split_k_id - 1) % 3 * BK * (BN + padding) + (16 * 1) * (BN + padding) + 16 * 1 + 64 * warp_id_x], BN + padding);
        wmma::load_matrix_sync(right_frag[1][2], &S_B[(split_k_id - 1) % 3 * BK * (BN + padding) + (16 * 1) * (BN + padding) + 16 * 2 + 64 * warp_id_x], BN + padding);
        wmma::load_matrix_sync(right_frag[1][3], &S_B[(split_k_id - 1) % 3 * BK * (BN + padding) + (16 * 1) * (BN + padding) + 16 * 3 + 64 * warp_id_x], BN + padding);

        #pragma unroll
        for(int row = 0; row < 4; row++) {
            #pragma unroll
            for(int col = 0; col < 4; col++) {
                wmma::mma_sync(c_frag[row][col], left_frag[0][row], right_frag[0][col], c_frag[row][col]);
                wmma::mma_sync(c_frag[row][col], left_frag[1][row], right_frag[1][col], c_frag[row][col]);
            }
        }
        __syncthreads();
    

    // Store result
    #pragma unroll
    for(int row = 0; row < 4; row++) {
        #pragma unroll
        for(int col = 0; col < 4; col++) {
            wmma::store_matrix_sync(
                dC + (16 * row + 64 * warp_id_y + row_offset) * N + 16 * col + 64 * warp_id_x + col_offset,
                c_frag[row][col], N, wmma::mem_row_major);
        }
    }
}

PLAYGROUND_MATMUL_DEC(float16_t, 15, M, N, K, A, B, C)
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
    
    cudaFuncSetAttribute(matrixKernel15s, cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);//只能传不带模板的进去, 96KB
    unsigned int dsmem = 3 * (BM * (BK + 8) + BK * (BN + 8)) * sizeof(half);
    matrixKernel15s<<<grid_dim, block_dim, dsmem>>>(const_cast<float16_t*>(A), const_cast<float16_t*>(B), const_cast<float16_t*>(C), M, K, N);

    cudaDeviceSynchronize();
}
}  // namespace playground
#include <stdio.h>
#include <sys/time.h>
#include <cuda.h>
#include <mma.h>
#include <assert.h>
// #include <cuda_fp16.hpp>  // 包含 half 类型支持
#include <cuda_fp16.h>  // 包含半精度支持
using namespace nvcuda;

#define OFFSET(row, col, ld) ((row) * (ld) + (col))
#define FLOAT4(ptr) (*reinterpret_cast<float4*>(&(ptr)))

//BM = 128，BN = 256，BK = 32，thread_per_block = 256

__global__ void matrixKernel(half *dA, half *dB, half *dC, int M, int K, int N)
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

double  get_walltime()
{
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return (double)(tp.tv_sec + tp.tv_usec * 1e-6);
}
void matrixSerial(half *hostA, half *hostB, half *hostC, int M, int K, int N)
{
    half tmp = 0.0f;
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            tmp = 0.0f;
            for (int s = 0; s < K; s++)
            {
               tmp += hostA[i * K + s] * hostB[s * N + j];
            }
            hostC[i * N + j] = tmp;
        }
    }
}
float compare(half *hostC, half *serialC, int M, int N)
{
    float error = 0;
    for (int i = 0; i < M * N; i++)
    {
        error = fmax(error, fabs(__half2float(hostC[i]) - __half2float(serialC[i])));
    }
    return error;
}

void hostMatrix(half *hostA, half *hostB, half *hostC, int M, int K, int N)
{
    double st, ela;
    st = get_walltime();

    half *dA, *dB, *dC;
    cudaMalloc((void **)&dA, M * K * sizeof(half));
    cudaMalloc((void **)&dB, N * K * sizeof(half));
    cudaMalloc((void **)&dC, M * N * sizeof(half));

    cudaMemcpy(dA, hostA, M * K * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hostB, N * K * sizeof(half), cudaMemcpyHostToDevice);



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
    float ker_time = 0;

    cudaFuncSetAttribute(matrixKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);//只能传不带模板的进去

    unsigned int dsmem = 3 * (BM * (BK + 8) + BK * (BN + 8)) * sizeof(half);
    matrixKernel<<<grid_dim, block_dim, dsmem>>>(dA, dB, dC, M, K, N);
    
    int repeat = 20;
    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    for (int i = 0; i < repeat; i++)
    {
       matrixKernel<<<grid_dim, block_dim, dsmem>>>(dA, dB, dC, M, K, N);
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ker_time, start, stop); // must float ker_time

    cudaMemcpy(hostC, dC, M * N * sizeof(half), cudaMemcpyDeviceToHost);

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    ela = get_walltime() - st;

    printf("GPU use time: %.4f second\n", ela);
    printf("kernel time: %.4f second, %.4f ms\n", ker_time / (repeat * 1000.), ker_time / repeat);
    printf("grid dim: %d, %d, %d\n", grid_dim.x, grid_dim.y, grid_dim.z);
    printf("block dim: %d, %d, %d\n", block_dim.x, block_dim.y, block_dim.z);
}

int main()
{
    half *hostA, *hostB, *hostC, *serialC;
    int M = 4096;
    int K = 4096;
    int N = 4096;


    hostA = (half *)malloc(M * K * sizeof(half));
    hostB = (half *)malloc(N * K * sizeof(half));
    hostC = (half *)malloc(M * N * sizeof(half));//GPU端
    serialC = (half *)malloc(M * N * sizeof(half));//CPU端
    for (int i = 0; i < M * K; i++)
    {
        hostA[i] = i % 3;
    }
    for (int i = 0; i < N * K; i++)
    {
        hostB[i] = i % 3;
    }


    // srand((unsigned)time(NULL));
    // for (int i = 0; i < M * K; i++)
    // {
    //     float r = (float)rand() / (float)RAND_MAX; // 0.0 - 1.0
    //     hostA[i] = __float2half(r);
    // }
    // for (int i = 0; i < N * K; i++)
    // {
    //     float r = (float)rand() / (float)RAND_MAX; // 0.0 - 1.0
    //     hostB[i] = __float2half(r);
    // }

    hostMatrix(hostA, hostB, hostC, M, K, N);
    // double st, ela;
    // st = get_walltime();
    // matrixSerial(hostA, hostB, serialC, M, K, N);
    // ela = get_walltime() - st;
    // float error = compare(hostC, serialC, M, N);
    // printf("CPU time:%.2f, error:%.4e\n", ela, error);

    
    // for (int i = 0; i < 20; i++) {
    //     printf("%2d: hostC=%f    serialC=%f\n", i, __half2float(hostC[i]), __half2float(serialC[i]));//累加器明明是可以float16的，为啥gpt都说是不可以的，用0-1的小数随机，确实由较大误差
    // }
    free(hostA);
    free(hostB);
    free(hostC);
    free(serialC);
    return 0;
}


#include "playground/matmul.hpp"
#include "playground/system.hpp"

#include <cuda_fp16.h>  // 包含半精度支持
#include <mma.h>             // ← 关键：WMMA 支持
using namespace nvcuda;      // ← 关键：让 wmma:: 可用

namespace playground
{

template <int WMMA_M, int WMMA_N, int WMMA_K, int WARP_SIZE, int WARP_DIM_X, int WARP_DIM_Y>
__global__ void matrixKernel2st(half *dA, half *dB, half *dC, int M, int K, int N)
{
    int ld_a = K; //行主序：i * K + j，列主序 i + j * K;
    int ld_b = N;
    int ld_c = N;
    // Declare the fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> left_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> right_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> c_frag;

    //全局偏移
    int row, col;
    row = blockIdx.y * (WMMA_M * WARP_DIM_Y);
    col = blockIdx.x * (WMMA_N * WARP_DIM_X);

    //块内偏移
    int warp_id_x, warp_id_y;
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    warp_id_y = (tid / WARP_SIZE) / WARP_DIM_X;
    warp_id_x = (tid / WARP_SIZE) % WARP_DIM_X;

    //真正偏移
    int row_offset, col_offset;
    row_offset = row + warp_id_y * WMMA_M;
    col_offset = col + warp_id_x * WMMA_N;


    //累加器初始化
    wmma::fill_fragment(c_frag, 0.0f);

    for(int split_k_id = 0; split_k_id < (K + WMMA_K - 1)/(WMMA_K); split_k_id++)
    {
        wmma::load_matrix_sync(left_frag, dA + (row_offset * K) + (split_k_id * WMMA_K), ld_a);
        wmma::load_matrix_sync(right_frag, dB + (split_k_id * WMMA_K) * N + col_offset, ld_b);
        wmma::mma_sync(c_frag, left_frag, right_frag, c_frag);
    }

    // //搬出
    wmma::store_matrix_sync(dC + (row_offset * N) + col_offset, c_frag, ld_c, wmma::mem_row_major);

    

}

PLAYGROUND_MATMUL_DEC(float16_t, 2, M, N, K, A, B, C)
{

    const int WMMA_M = 16;
    const int WMMA_N = 16;
    const int WMMA_K = 16;

    //因为还需要重排序,会将block设置为(4, 4) * 32 = 512
    const int BLOCK_DIM_x = 32;
    const int BLOCK_DIM_y = 16;
    const int WARP_SIZE = 32;
    const int warpNum = BLOCK_DIM_x * BLOCK_DIM_y / WARP_SIZE;
    const int WARP_DIM_X = 4;
    const int WARP_DIM_Y = warpNum / WARP_DIM_X;

    int num_block_x = (M + WMMA_M * WARP_DIM_X - 1) / (WMMA_M * WARP_DIM_X);
    int num_block_y = (N + WMMA_N * WARP_DIM_Y - 1) / (WMMA_N * WARP_DIM_Y);

    dim3 block_dim(BLOCK_DIM_x, BLOCK_DIM_y, 1);
    dim3 grid_dim(num_block_x, num_block_y, 1);
    
    matrixKernel2st<WMMA_M, WMMA_N, WMMA_K, WARP_SIZE, WARP_DIM_X, WARP_DIM_Y><<<grid_dim, block_dim>>>(const_cast<float16_t*>(A), const_cast<float16_t*>(B), const_cast<float16_t*>(C), M, K, N);

    cudaDeviceSynchronize();
}
}  // namespace playground
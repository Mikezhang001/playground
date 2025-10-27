
#include "playground/matmul.hpp"
#include "playground/system.hpp"

namespace playground
{

template <int split_K, int blockDim_y, int blockDim_x, int TM, int TN>// blockDim_y = real_blockDim_y * TM, blockDim_x = real_blockDim_x * TN (相较shared memory情况下)
__global__ void matrixKernel5st(float *dA, float *dB, float *dC, int M, int K, int N)
{
    int row, col;
    row = (blockIdx.y * blockDim.y) * TM;
    col = (blockIdx.x * blockDim.x) * TN;

    __shared__ float S_A[blockDim_y][split_K];
    __shared__ float S_B[split_K][blockDim_x];
    float result[TM][TN] = {0.0f};

    for(int split_K_id = 0; split_K_id < (K + split_K - 1) / split_K; split_K_id++)
    {
        for(int S_A_row = threadIdx.y * TM; S_A_row < (threadIdx.y + 1) * TM; S_A_row++)
        {
            for(int x = threadIdx.x; x < split_K; x += blockDim.x)//A的部分搬入S_A（做列切）
            {
                int k_idx = split_K_id * split_K + x;
                S_A[S_A_row][x] = (k_idx < K) ? dA[(row + S_A_row) * K + k_idx] : 0.0f;
           
            }
        }
        __syncthreads();

        for(int S_B_col = threadIdx.x * TN; S_B_col < (threadIdx.x + 1) * TN; S_B_col++)
        {
            for(int y = threadIdx.y; y < split_K; y += blockDim.y)//B的部分搬入S_B（做行切）
            {
                int k_idx = split_K_id * split_K + y;
                S_B[y][S_B_col] = (k_idx < K) ? dB[(k_idx) * N + col + S_B_col] : 0.0f;
            }
        }

        __syncthreads();

        for(int result_row = 0; result_row < TM; result_row++)
        {
            for(int result_col = 0; result_col < TN; result_col++)
            {
                for(int k = 0; k < split_K; k++)
                {
                    result[result_row][result_col] += S_A[threadIdx.y * TM + result_row][k] * S_B[k][threadIdx.x * TN + result_col];
                }
            }
        }
        __syncthreads();
    }

    for(int result_row = 0; result_row < TM; result_row++)
    {
        for(int result_col = 0; result_col < TN; result_col++)
        {
            dC[(row + threadIdx.y * TM + result_row) * N + col + threadIdx.x * TN + result_col] = result[result_row][result_col];
        }
    }
}

PLAYGROUND_MATMUL_DEC(float32_t, 5, M, N, K, A, B, C)
{

    #define TM 2
    #define TN 2

    #define BLOCK_DIM_Y 16
    #define BLOCK_DIM_X 16
    #define SPLIT_K 32
    
    #define LOGIC_BLOCK_DIM_Y  (TM * BLOCK_DIM_Y)
    #define LOGIC_BLOCK_DIM_X  (TN * BLOCK_DIM_X)
    
    int num_blocks_y = (M + LOGIC_BLOCK_DIM_Y - 1) / LOGIC_BLOCK_DIM_Y;
    int num_blocks_x = (N + LOGIC_BLOCK_DIM_X - 1) / LOGIC_BLOCK_DIM_X;

    dim3 block_dim(BLOCK_DIM_X, BLOCK_DIM_Y, 1);
    dim3 grid_dim(num_blocks_x, num_blocks_y, 1);

    matrixKernel5st<SPLIT_K, LOGIC_BLOCK_DIM_Y, LOGIC_BLOCK_DIM_X, TM, TN><<<grid_dim, block_dim>>>(const_cast<float*>(A), const_cast<float*>(B), const_cast<float*>(C), M, K, N);
    cudaDeviceSynchronize();
}
}  // namespace playground
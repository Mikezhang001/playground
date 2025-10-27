
#include "playground/matmul.hpp"
#include "playground/system.hpp"

namespace playground
{

template <int split_K, int blockDim_y, int blockDim_x>
__global__ void matrixKernel4st(float *dA, float *dB, float *dC, int M, int K, int N)
{
    int row, col;
    row = threadIdx.y + blockIdx.y * blockDim.y;
    col = threadIdx.x + blockIdx.x * blockDim.x;

    __shared__ float S_A[blockDim_y][split_K];
    __shared__ float S_B[split_K][blockDim_x];
    float result = 0;
    for(int split_K_id = 0; split_K_id < (K + split_K - 1) / split_K; split_K_id++)
    {
        for(int x = threadIdx.x; x < split_K; x += blockDim.x)//A的部分搬入S_A（做列切）
        {
            int k_idx = split_K_id * split_K + x;
            S_A[threadIdx.y][x] = (k_idx < K) ? dA[row * K + k_idx] : 0.0f;
           
        }
        for(int y = threadIdx.y; y < split_K; y += blockDim.y)//B的部分搬入S_B（做行切）
        {
            int k_idx = split_K_id * split_K + y;
            S_B[y][threadIdx.x] = (k_idx < K) ? dB[k_idx * N + col] : 0.0f;
        }
        __syncthreads();
        for(int i = 0; i < split_K; i++)
        {
            result += S_A[threadIdx.y][i] * S_B[i][threadIdx.x];
        }
        __syncthreads();
    }
    dC[row * N + col] = result;
}

PLAYGROUND_MATMUL_DEC(float32_t, 4, M, N, K, A, B, C)
{

    #define TM 1
    #define TN 1

    #define BLOCK_DIM_Y 32
    #define BLOCK_DIM_X 32
    #define SPLIT_K 32
    
    #define LOGIC_BLOCK_DIM_Y  (TM * BLOCK_DIM_Y)
    #define LOGIC_BLOCK_DIM_X  (TN * BLOCK_DIM_X)
    
    int num_blocks_y = (M + LOGIC_BLOCK_DIM_Y - 1) / LOGIC_BLOCK_DIM_Y;
    int num_blocks_x = (N + LOGIC_BLOCK_DIM_X - 1) / LOGIC_BLOCK_DIM_X;

    dim3 block_dim(BLOCK_DIM_X, BLOCK_DIM_Y, 1);
    dim3 grid_dim(num_blocks_x, num_blocks_y, 1);

    matrixKernel4st<SPLIT_K, LOGIC_BLOCK_DIM_Y, LOGIC_BLOCK_DIM_X><<<grid_dim, block_dim>>>(const_cast<float*>(A), const_cast<float*>(B), const_cast<float*>(C), M, K, N);
    cudaDeviceSynchronize();
}
}  // namespace playground
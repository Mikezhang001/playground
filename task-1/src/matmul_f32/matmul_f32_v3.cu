
#include "playground/matmul.hpp"
#include "playground/system.hpp"

namespace playground
{
__global__ void matrixKernel3st(float *dA, float *dB, float *dC, int M, int K, int N)
{
    int row, col;
    row = threadIdx.y + blockIdx.y * blockDim.y;
    col = threadIdx.x + blockIdx.x * blockDim.x;
    float sum = 0.0f;
    for(int i = 0; i < K; i++)
    {
        sum += dA[row * K + i] * dB[i * N+ col];
    }
    dC[row * N + col] = sum;
}

PLAYGROUND_MATMUL_DEC(float32_t, 3, M, N, K, A, B, C)
{

    #define TM 1
    #define TN 1

    #define BLOCK_DIM_Y 32
    #define BLOCK_DIM_X 32
    
    #define LOGIC_BLOCK_DIM_Y  (TM * BLOCK_DIM_Y)
    #define LOGIC_BLOCK_DIM_X  (TN * BLOCK_DIM_X)
    
    int num_blocks_y = (M + LOGIC_BLOCK_DIM_Y - 1) / LOGIC_BLOCK_DIM_Y;
    int num_blocks_x = (N + LOGIC_BLOCK_DIM_X - 1) / LOGIC_BLOCK_DIM_X;

    dim3 block_dim(BLOCK_DIM_X, BLOCK_DIM_Y, 1);
    dim3 grid_dim(num_blocks_x, num_blocks_y, 1);

    matrixKernel3st<<<grid_dim, block_dim>>>(const_cast<float*>(A), const_cast<float*>(B), const_cast<float*>(C), M, K, N);
    cudaDeviceSynchronize();
}
}  // namespace playground
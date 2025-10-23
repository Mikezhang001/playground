
#include "playground/matmul.hpp"
#include "playground/system.hpp"

namespace playground
{
template <int split_K, int blockDim_y, int blockDim_x, int TM, int TN>// blockDim_y = real_blockDim_y * TM, blockDim_x = real_blockDim_x * TN (相较shared memory情况下)
__global__ void matrixKernel(float *dA, float *dB, float *dC, int M, int K, int N)
{
    int row, col;
    row = (blockIdx.y * blockDim.y) * TM;
    col = (blockIdx.x * blockDim.x) * TN;

    __shared__ float S_A[split_K * blockDim_y * 2]; //将其看作为 2个连续空间的S_A[split_K][blockDim_y]
    __shared__ float S_B[split_K * blockDim_x * 2]; //将其看作为 2个连续空间的S_B[split_K][blockDim_x]
    float result[TM][TN] = {0.0f};
    float tmp[4];

    float tmp_S_A[TM];
    float tmp_S_B[TN];

    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int S_A_row = tid / (split_K / 4);
    int S_A_col = tid % (split_K / 4);

    int S_B_row = tid / (blockDim_x / 4);
    int S_B_col = tid % (blockDim_x / 4);

    //第0轮数据搬运
    int split_K_id, k_idx;
    split_K_id = 0;
    k_idx = split_K_id * split_K;
    (float4 &)(tmp[0]) = (float4 &)(dA[(S_A_row + row) * K + S_A_col * 4 + k_idx]);
    for(int i = 0; i < 4; i++)
    {
        //添加奇数和偶数的相对偏移
        S_A[((S_A_col * 4) + i) * blockDim_y + S_A_row + (split_K_id % 2) * (split_K * blockDim_y)] = tmp[i];//
    }
    //添加奇数和偶数的相对偏移
    (float4 &)(S_B[(S_B_row) * blockDim_x + S_B_col * 4 + (split_K_id % 2) * (split_K * blockDim_x)]) = (float4 &)(dB[(S_B_row + k_idx) * N + S_B_col * 4 + col]);
    __syncthreads();


    for(split_K_id = 1; split_K_id < (K + split_K - 1) / split_K; split_K_id++)
    {

        //第split_K_id - 1轮计算
        for(int k = 0; k < split_K; k++)
        {
            for(int i = 0; i < TM / 4; i++)//存S_A到寄存器
            {
                //添加奇数和偶数的相对偏移
                (float4 &)(tmp_S_A[i * 4]) = (float4 &)(S_A[k * blockDim_y + (threadIdx.y * TM + i * 4) + ((split_K_id -1) % 2) * (split_K * blockDim_y)]);
            }
            for(int i = 0; i < TN / 4; i++)//存S_B到寄存器
            {
                //添加奇数和偶数的相对偏移
                (float4 &)(tmp_S_B[i * 4]) = (float4 &)(S_B[k * blockDim_x + (threadIdx.x * TN + i * 4) + ((split_K_id -1) % 2) * (split_K * blockDim_x)]);
            }
            for(int result_row = 0; result_row < TM; result_row++)
            {
                for(int result_col = 0; result_col < TN; result_col++)
                {
                    result[result_row][result_col] += tmp_S_A[result_row] * tmp_S_B[result_col];
                }
            }
        }

        //第split_K_id轮数据搬运
        k_idx = split_K_id * split_K;
        (float4 &)(tmp[0]) = (float4 &)(dA[(S_A_row + row) * K + S_A_col * 4 + k_idx]);
        for(int i = 0; i < 4; i++)
        {
            //添加奇数和偶数的相对偏移
            S_A[((S_A_col * 4) + i) * blockDim_y + S_A_row + (split_K_id % 2) * (split_K * blockDim_y)] = tmp[i];//
        }
        //添加奇数和偶数的相对偏移
        (float4 &)(S_B[(S_B_row) * blockDim_x + S_B_col * 4 + (split_K_id % 2) * (split_K * blockDim_x)]) = (float4 &)(dB[(S_B_row + k_idx) * N + S_B_col * 4 + col]);
        __syncthreads();

    }


    //最后一轮计算
    for(int k = 0; k < split_K; k++)
    {
        for(int i = 0; i < TM / 4; i++)//存S_A到寄存器
        {
            //添加奇数和偶数的相对偏移
            (float4 &)(tmp_S_A[i * 4]) = (float4 &)(S_A[k * blockDim_y + (threadIdx.y * TM + i * 4) + ((split_K_id -1) % 2) * (split_K * blockDim_y)]);
        }
        for(int i = 0; i < TN / 4; i++)//存S_B到寄存器
        {
            //添加奇数和偶数的相对偏移
            (float4 &)(tmp_S_B[i * 4]) = (float4 &)(S_B[k * blockDim_x + (threadIdx.x * TN + i * 4) + ((split_K_id -1) % 2) * (split_K * blockDim_x)]);
        }
        for(int result_row = 0; result_row < TM; result_row++)
        {
            for(int result_col = 0; result_col < TN; result_col++)
            {
                result[result_row][result_col] += tmp_S_A[result_row] * tmp_S_B[result_col];
            }
        }
    }


    
    for(int result_row = 0; result_row < TM; result_row++)
    {
        for(int result_col = 0; result_col < TN; result_col++)
        {
            dC[(row + threadIdx.y * TM + result_row) * N + col + threadIdx.x * TN + result_col] = result[result_row][result_col];
        }
    }
}

PLAYGROUND_MATMUL_DEC(float32_t, 2, M, N, K, A, B, C)
{

    #define TM 8
    #define TN 8

    #define BLOCK_DIM_Y 16
    #define BLOCK_DIM_X 16
    #define SPLIT_K  ((4 * BLOCK_DIM_X) / TM)
    
    #define LOGIC_BLOCK_DIM_Y  (TM * BLOCK_DIM_Y)
    #define LOGIC_BLOCK_DIM_X  (TN * BLOCK_DIM_X)
    
    int num_blocks_y = (M + LOGIC_BLOCK_DIM_Y - 1) / LOGIC_BLOCK_DIM_Y;
    int num_blocks_x = (N + LOGIC_BLOCK_DIM_X - 1) / LOGIC_BLOCK_DIM_X;

    dim3 block_dim(BLOCK_DIM_X, BLOCK_DIM_Y, 1);
    dim3 grid_dim(num_blocks_x, num_blocks_y, 1);

    matrixKernel<SPLIT_K, LOGIC_BLOCK_DIM_Y, LOGIC_BLOCK_DIM_X, TM, TN><<<grid_dim, block_dim>>>(const_cast<float*>(A), const_cast<float*>(B), const_cast<float*>(C), M, K, N);
    cudaDeviceSynchronize();
}
}  // namespace playground
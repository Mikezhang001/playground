#include <stdio.h>
#include <sys/time.h>
#include <cuda.h>

double
get_walltime()
{
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return (double)(tp.tv_sec + tp.tv_usec * 1e-6);
}
void matrixSerial(float *hostA, float *hostB, float *hostC, int M, int K, int N)
{
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            float tmp = 0;
            for (int s = 0; s < K; s++)
            {
                tmp += hostA[i * K + s] * hostB[s * N + j];
            }
            hostC[i * N + j] = tmp;
        }
    }
}
float compare(float *hostC, float *serialC, int M, int N)
{
    float error = 0;
    for (int i = 0; i < M * N; i++)
    {
        error = fmax(error, fabs(hostC[i] - serialC[i]));
    }
    return error;
}

template <int split_K, int blockDim_y, int blockDim_x, int TM, int TN>// 在 V5的基础上使用外积，并用float4来加速数据读取。
__global__ void matrixKernel(float *dA, float *dB, float *dC, int M, int K, int N)
{
    int row, col;
    row = (blockIdx.y * blockDim.y) * TM;
    col = (blockIdx.x * blockDim.x) * TN;

    __shared__ float S_A[split_K][blockDim_y];
    __shared__ float S_B[split_K][blockDim_x];
    float result[TM][TN] = {0.0f};
    float tmp[4];

    //tmp_S_A长度取值TM, tmp_S_B长度取值为TN；增加
    float tmp_S_A[TM];
    float tmp_S_B[TN];

    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int S_A_row = tid / (split_K / 4);
    int S_A_col = tid % (split_K / 4);

    int S_B_row = tid / (blockDim_x / 4);
    int S_B_col = tid % (blockDim_x / 4);


    for(int split_K_id = 0; split_K_id < (K + split_K - 1) / split_K; split_K_id++)
    {
        int k_idx = split_K_id * split_K;
        // (float4 &)(S_A[S_A_row][S_A_col * 4]) = (float4 &)(dA[S_A_row + row][S_A_col * 4 + k_idx]);
        // (float4 &)(S_B[S_B_row][S_B_col * 4]) = (float4 &)(dB[S_B_row + k_idx][S_B_col * 4 + col]);

        
        (float4 &)(tmp[0]) = (float4 &)(dA[(S_A_row + row) * K + S_A_col * 4 + k_idx]);
        for(int i = 0; i < 4; i++)
        {
            S_A[(S_A_col * 4) + i][S_A_row] = tmp[i];
        }


        (float4 &)(S_B[(S_B_row)][S_B_col * 4]) = (float4 &)(dB[(S_B_row + k_idx) * N + S_B_col * 4 + col]);


        __syncthreads();

        //内积化外积，并用float4加速读取
        for(int k = 0; k < split_K; k++)
        {
            for(int i = 0; i < TM / 4; i++)//存S_A到寄存器
            {
                (float4 &)(tmp_S_A[i * 4]) = (float4 &)(S_A[k][(threadIdx.y * TM + i * 4)]);
            }
            for(int i = 0; i < TN / 4; i++)//存S_B到寄存器
            {
                (float4 &)(tmp_S_B[i * 4]) = (float4 &)(S_B[k][threadIdx.x * TN + i * 4]);
            }
            for(int result_row = 0; result_row < TM; result_row++)
            {
                for(int result_col = 0; result_col < TN; result_col++)
                {
                    result[result_row][result_col] += tmp_S_A[result_row] * tmp_S_B[result_col];
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

void hostMatrix(float *hostA, float *hostB, float *hostC, int M, int K, int N)
{
    double st, ela;
    st = get_walltime();

    float *dA, *dB, *dC;
    cudaMalloc((void **)&dA, M * K * sizeof(float));
    cudaMalloc((void **)&dB, N * K * sizeof(float));
    cudaMalloc((void **)&dC, M * N * sizeof(float));

    cudaMemcpy(dA, hostA, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hostB, N * K * sizeof(float), cudaMemcpyHostToDevice);

    // 假设 TM = TN, BLCOK_DIM_Y = BLOCK_DIM_X，LOGIC_BLOCK_DIM_Y * SPLIT_K = BLOCK_DIM_Y * BLOCK_DIM_X * 4 ------> SPLIT_K = ((4 * BLOCK_DIM_X) / TM)；
    #define TM 8
    #define TN 8

    #define BLOCK_DIM_Y 16
    #define BLOCK_DIM_X 16
    #define SPLIT_K  ((4 * BLOCK_DIM_X) / TM)
    
    #define LOGIC_BLOCK_DIM_Y  (TM * BLOCK_DIM_Y)
    #define LOGIC_BLOCK_DIM_X  (TN * BLOCK_DIM_X)
    
    int num_blocks_y = (M + LOGIC_BLOCK_DIM_Y - 1) / LOGIC_BLOCK_DIM_Y;
    int num_blocks_x = (N + LOGIC_BLOCK_DIM_X - 1) / LOGIC_BLOCK_DIM_X;

    printf("M-K-N: %d-%d-%d\n", M, K, N);
    printf("TM = %d\n", TM);
    printf("TN = %d\n", TN);
    printf("BLOCK_DIM_Y = %d\n", BLOCK_DIM_Y);
    printf("BLOCK_DIM_X = %d\n", BLOCK_DIM_X);
    printf("SPLIT_K = %d\n", SPLIT_K);
    printf("LOGIC_BLOCK_DIM_Y = %d\n", LOGIC_BLOCK_DIM_Y);
    printf("LOGIC_BLOCK_DIM_X = %d\n", LOGIC_BLOCK_DIM_X);
    printf("num_blocks_y = %d\n", num_blocks_y);
    printf("num_blocks_x = %d\n", num_blocks_x);


    dim3 block_dim(BLOCK_DIM_X, BLOCK_DIM_Y, 1);
    dim3 grid_dim(num_blocks_x, num_blocks_y, 1);
    int repeat = 20;
    
    matrixKernel<SPLIT_K, LOGIC_BLOCK_DIM_Y, LOGIC_BLOCK_DIM_X, TM, TN><<<grid_dim, block_dim>>>(dA, dB, dC, M, K, N);
    
    cudaEvent_t start, stop;
    float ker_time = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    for (int i = 0; i < repeat; i++)
    {
        matrixKernel<SPLIT_K, LOGIC_BLOCK_DIM_Y, LOGIC_BLOCK_DIM_X, TM, TN><<<grid_dim, block_dim>>>(dA, dB, dC, M, K, N);
       
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ker_time, start, stop); // must float ker_time

    cudaMemcpy(hostC, dC, M * N * sizeof(float), cudaMemcpyDeviceToHost);

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
    float *hostA, *hostB, *hostC, *serialC;
    int M = 1024;
    int K = 1024;
    int N = 1024;

    hostA = (float *)malloc(M * K * sizeof(float));
    hostB = (float *)malloc(N * K * sizeof(float));
    hostC = (float *)malloc(M * N * sizeof(float));
    serialC = (float *)malloc(M * N * sizeof(float));
    for (int i = 0; i < M * K; i++)
    {
        hostA[i] = i % 3;
    }
    for (int i = 0; i < N * K; i++)
    {
        hostB[i] = i % 3;
    }
    hostMatrix(hostA, hostB, hostC, M, K, N);
    double st, ela;
    st = get_walltime();
    matrixSerial(hostA, hostB, serialC, M, K, N);
    ela = get_walltime() - st;
    float error = compare(hostC, serialC, M, N);
    printf("CPU time:%.2f second\n", ela);
    printf("The error between CPU and GPU: %.4e\n", error);
    free(hostA);
    free(hostB);
    free(hostC);
    free(serialC);
    return 0;
}

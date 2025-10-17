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
__global__ void matrixKernel1st(float *dA, float *dB, float *dC, int M, int K, int N)
{
    int row, col;
    row = threadIdx.y + blockIdx.y * blockDim.y;
    col = threadIdx.x + blockIdx.x * blockDim.x;

    for(int i = 0; i < K; i++)
    {
        dC[row * N + col] += dA[row * K + i] * dB[i * N+ col];
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

    int BLOCK_DIM_x = 32;
    int BLOCK_DIM_y = 32;
    int num_blocks_x = (M + BLOCK_DIM_x - 1) / BLOCK_DIM_x;
    int num_blocks_y = (N + BLOCK_DIM_y - 1) / BLOCK_DIM_y;
    dim3 block_dim(BLOCK_DIM_x, BLOCK_DIM_y, 1);
    dim3 grid_dim(num_blocks_x, num_blocks_y, 1);
    int repeat = 20;
    matrixKernel1st<<<grid_dim, block_dim>>>(dA, dB, dC, M, K, N);
    
    cudaEvent_t start, stop;
    float ker_time = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    for (int i = 0; i < repeat; i++)
    {
        matrixKernel1st<<<grid_dim, block_dim>>>(dA, dB, dC, M, K, N);
       
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
    printf("M-K-N: %d-%d-%d\n", M, K, N);
    printf("GPU use time: %.4f second\n", ela);
    printf("kernel time: %.4f second, %.4f ms\n", ker_time / (repeat * 1000.), ker_time / repeat);
    printf("grid dim: %d, %d, %d\n", grid_dim.x, grid_dim.y, grid_dim.z);
    printf("block dim: %d, %d, %d\n", block_dim.x, block_dim.y, block_dim.z);
}

int main()
{
    float *hostA, *hostB, *hostC, *serialC;
    int M = 4096;
    int K = 4096;
    int N = 4096;

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
    //cpu时间太长，忽略
    // double st, ela;
    // st = get_walltime();
    // matrixSerial(hostA, hostB, serialC, M, K, N);
    // ela = get_walltime() - st;
    // float error = compare(hostC, serialC, M, N);
    // printf("CPU time:%.2f second\n", ela);
    // printf("The error between CPU and GPU: %.4e\n", error);
    free(hostA);
    free(hostB);
    free(hostC);
    free(serialC);
    return 0;
}
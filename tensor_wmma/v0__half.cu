#include <stdio.h>
#include <sys/time.h>
#include <cuda.h>
#include <mma.h>

// #include <cuda_fp16.hpp>  // 包含 half 类型支持
#include <cuda_fp16.h>  // 包含半精度支持
using namespace nvcuda;

template <int WMMA_M, int WMMA_N, int WMMA_K, int TM, int TN>// blockDim_y = real_blockDim_y * TM, blockDim_x = real_blockDim_x * TN (相较shared memory情况下)
__global__ void matrixKernel(float *dA, float *dB, float *dC, int M, int K, int N)
{
    // Declare the fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::row_major> left_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::row_major> right_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
}
double  get_walltime()
{
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return (double)(tp.tv_sec + tp.tv_usec * 1e-6);
}
void matrixSerial(__half *hostA, __half *hostB, __half *hostC, int M, int K, int N)
{
    __half tmp = __float2half(0.0f);
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            tmp = __float2half(0.0f);
            for (int s = 0; s < K; s++)
            {
               tmp = __hadd(tmp, __hmul(hostA[i * K + s], hostB[s * N + j]));
            }
            hostC[i * N + j] = tmp;
        }
    }
}
float compare(__half *hostC, __half *serialC, int M, int N)
{
    float error = 0;
    for (int i = 0; i < M * N; i++)
    {
        error = fmax(error, fabs(__half2float(hostC[i]) - __half2float(serialC[i])));
    }
    return error;
}

void hostMatrix(__half *hostA, __half *hostB, __half *hostC, int M, int K, int N)
{
    double st, ela;
    st = get_walltime();

    __half *dA, *dB, *dC;
    cudaMalloc((void **)&dA, M * K * sizeof(__half));
    cudaMalloc((void **)&dB, N * K * sizeof(__half));
    cudaMalloc((void **)&dC, M * N * sizeof(__half));

    cudaMemcpy(dA, hostA, M * K * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hostB, N * K * sizeof(__half), cudaMemcpyHostToDevice);


    const int WMMA_M = 16;
    const int WMMA_N = 16;
    const int WMMA_K = 16;

    //因为还需要重排序,会将block设置为(4, 4) * 32 = 512
    const int BLOCK_DIM_x = 32;
    const int BLOCK_DIM_y = 16;
    const int warpSize = 32;
    const int warpNum = BLOCK_DIM_x * BLOCK_DIM_y / warpSize;
    const int warpX = 4;
    const int warpY = warpNum / warpX;

    int num_block_x = (M + WMMA_M * warpX - 1) / (WMMA_M * warpX);
    int num_block_y = (N + WMMA_N * warpY - 1) / (WMMA_N * warpY);

    dim3 block_dim(BLOCK_DIM_x, BLOCK_DIM_y, 1);
    dim3 grid_dim(num_block_x, num_block_y, 1);
    float ker_time = 0;
    // row_wmma_ker<<<grid_dim, block_dim>>>(dA, dB, dC, M, K, N);
    
    int repeat = 20;
    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    for (int i = 0; i < repeat; i++)
    {
        // row_wmma_ker<<<grid_dim, block_dim>>>(dA, dB, dC, M, K, N);
        
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ker_time, start, stop); // must float ker_time

    cudaMemcpy(hostC, dC, M * N * sizeof(__half), cudaMemcpyDeviceToHost);

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
    __half *hostA, *hostB, *hostC, *serialC;
    int M = 16;
    int K = 16;
    int N = 16;


    hostA = (__half *)malloc(M * K * sizeof(__half));
    hostB = (__half *)malloc(N * K * sizeof(__half));
    hostC = (__half *)malloc(M * N * sizeof(__half));
    serialC = (__half *)malloc(M * N * sizeof(__half));
    for (int i = 0; i < M * K; i++)
    {
        hostA[i] = i % 3;
    }
    for (int i = 0; i < N * K; i++)
    {
        hostB[i] = i % 3;
    }
    // hostMatrix(hostA, hostB, hostC, M, K, N);
    // double st, ela;
    // st = get_walltime();
    matrixSerial(hostA, hostB, serialC, M, K, N);
    // ela = get_walltime() - st;
    // float error = compare(hostC, serialC, M, N);
    // printf("CPU time:%.2f, error:%.4e\n", ela, error);
    free(hostA);
    free(hostB);
    free(hostC);
    free(serialC);
    return 0;
}

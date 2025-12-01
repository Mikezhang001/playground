#include <stdio.h>
#include <sys/time.h>
#include <cuda.h>
#include <mma.h>

// #include <cuda_fp16.hpp>  // 包含 half 类型支持
#include <cuda_fp16.h>  // 包含半精度支持
using namespace nvcuda;

template <int WMMA_M, int WMMA_N, int WMMA_K, int WARP_SIZE, int WARP_DIM_X, int WARP_DIM_Y>
__global__ void matrixKernel(half *dA, half *dB, half *dC, int M, int K, int N)
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


    const int WMMA_M = 16;
    const int WMMA_N = 16;
    const int WMMA_K = 16;

    //因为还需要重排序,会将block设置为(4, 4) * 32 = 512
    const int BLOCK_DIM_x = 32;
    const int BLOCK_DIM_y = 16;
    const int WARP_SIZE = 32;
    // const int warpNum = BLOCK_DIM_x * BLOCK_DIM_y / WARP_SIZE;
    const int WARP_DIM_X = 4;
    const int WARP_DIM_Y = warpNum / WARP_DIM_X;

    int num_block_y = (M + WMMA_M * WARP_DIM_X - 1) / (WMMA_M * WARP_DIM_X);
    int num_block_x = (N + WMMA_N * WARP_DIM_Y - 1) / (WMMA_N * WARP_DIM_Y);

    dim3 block_dim(BLOCK_DIM_x, BLOCK_DIM_y, 1);
    dim3 grid_dim(num_block_x, num_block_y, 1);
    float ker_time = 0;

    matrixKernel<WMMA_M, WMMA_N, WMMA_K, WARP_SIZE, WARP_DIM_X, WARP_DIM_Y><<<grid_dim, block_dim>>>(dA, dB, dC, M, K, N);
    
    int repeat = 20;
    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    for (int i = 0; i < repeat; i++)
    {
       matrixKernel<WMMA_M, WMMA_N, WMMA_K, WARP_SIZE, WARP_DIM_X, WARP_DIM_Y><<<grid_dim, block_dim>>>(dA, dB, dC, M, K, N);
        
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
    int M = 128;
    int K = 128;
    int N = 128;


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
    double st, ela;
    st = get_walltime();
    matrixSerial(hostA, hostB, serialC, M, K, N);
    ela = get_walltime() - st;
    float error = compare(hostC, serialC, M, N);
    printf("CPU time:%.2f, error:%.4e\n", ela, error);


    
    // for (int i = 0; i < 20; i++) {
    //     printf("%2d: hostC=%f    serialC=%f\n", i, __half2float(hostC[i]), __half2float(serialC[i]));//累加器明明是可以float16的，为啥gpt都说是不可以的，用0-1的小数随机，确实由较大误差
    // }
    free(hostA);
    free(hostB);
    free(hostC);
    free(serialC);
    return 0;
}

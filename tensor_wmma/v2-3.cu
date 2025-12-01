#include <stdio.h>
#include <sys/time.h>
#include <cuda.h>
#include <mma.h>

// #include <cuda_fp16.hpp>  // 包含 half 类型支持
#include <cuda_fp16.h>  // 包含半精度支持
using namespace nvcuda;

//BM = 128，BN = 256，BK = 32，thread_per_block = 256
template <int BM, int BK, int BN>//代表WARP处理的维度由[16, 16] -> [16 * W_M, 16 * W_N]
__global__ void matrixKernel(half *dA, half *dB, half *dC, int M, int K, int N)
{

    const int padding = 8;
    //以[64, 64]对结果进行切割
    __shared__ half S_A[BM][BK + padding];
    __shared__ half S_B[BK][BN + padding];
    
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> left_frag[4][2];//[64, 32] ->[4,2]
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> right_frag[2][4];//[32, 64]->[2,4]
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> c_frag[4][4];

    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int warp_id = tid / 32;
    int warp_id_x = warp_id % 4;
    int warp_id_y = warp_id / 4; 
    
    int S_A_row, S_A_col, S_B_row, S_B_col;

    //S_A变成S_B一样的列读模式
    S_A_row = tid / (BK / 8);//因为一个thread可以拿8个所以用BK / 8
    S_A_col = tid % (BK / 8);

    S_B_row = tid / (BN / 8);
    S_B_col = tid % (BN / 8);

    int row_offset, col_offset;
    row_offset = blockIdx.y * BM;
    col_offset = blockIdx.x * BN;

    //初始化
    for(int i = 0; i < 64 / 16; i++)
    {
        for(int j = 0; j < 64 / 16; j++)
        {
            wmma::fill_fragment(c_frag[i][j], 0.0f);
        }
    }
    for(int split_k_id = 0; split_k_id < K / BK; split_k_id++)
    {
        //dA->S_A
        for(int S_A_row_id = S_A_row; S_A_row_id < BM; S_A_row_id += (256 / (BK / 8)))
        {
            *reinterpret_cast<float4*>(&(S_A[S_A_row_id][S_A_col * 8])) = *reinterpret_cast<float4*>(&(dA[(S_A_row_id + row_offset) * K + (S_A_col * 8 + split_k_id * BK)]));
        }
        //dB->S_B
        for(int S_B_row_id = S_B_row; S_B_row_id < BK; S_B_row_id += (256 / (BN / 8)))
        {
            *reinterpret_cast<float4*>(&(S_B[S_B_row_id][S_B_col * 8])) = *reinterpret_cast<float4*>(&(dB[(S_B_row_id + split_k_id * BK) * N + (S_B_col * 8 + col_offset)]));
        }
        __syncthreads();

        //S_A -> left_matrix[64, 32]
        for(int row = 0; row < 64 / 16; row++)
        {
            for(int  col = 0; col < 32 / 16; col++)
            {
                wmma::load_matrix_sync(left_frag[row][col], &S_A[16 * row + 64 * warp_id_y][16 * col], BK + padding);
            }
        }

        //S_B -> right_matrix[32, 64]
        for(int row = 0; row < 32 / 16; row++)
        {
            for(int  col = 0; col < 64 / 16; col++)
            {
                wmma::load_matrix_sync(right_frag[row][col], &S_B[16 * row][16 * col + 64 * warp_id_x], BN + padding);
            }
        }
        __syncthreads();
        
        //left_matrix * right_matrix

        for(int row = 0; row < 64 / 16; row++)
        {
            for(int col = 0; col < 64 / 16; col++)
            {
                for(int k = 0; k < 32 / 16; k++)
                {
                    wmma::mma_sync(c_frag[row][col], left_frag[row][k], right_frag[k][col], c_frag[row][col]);
                }
            }
        }

        __syncthreads();

    }

    // //搬出
    for(int row = 0; row < 64 / 16; row++)
    {
        for(int col = 0; col < 64 / 16; col++)
        {
            wmma::store_matrix_sync(dC + (16 * row + 64 * warp_id_y + row_offset) * N + 16 * col + 64 * warp_id_x + col_offset, c_frag[row][col], N, wmma::mem_row_major);
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


    matrixKernel<BM, BK, BN><<<grid_dim, block_dim>>>(dA, dB, dC, M, K, N);
    
    int repeat = 20;
    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    for (int i = 0; i < repeat; i++)
    {
       matrixKernel<BM, BK, BN><<<grid_dim, block_dim>>>(dA, dB, dC, M, K, N);
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
    int M = 512;
    int K = 512;
    int N = 512;


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

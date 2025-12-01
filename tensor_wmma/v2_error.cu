#include <stdio.h>
#include <sys/time.h>
#include <cuda.h>
#include <mma.h>

// #include <cuda_fp16.hpp>  // 包含 half 类型支持
#include <cuda_fp16.h>  // 包含半精度支持
using namespace nvcuda;

template <int WMMA_M, int WMMA_N, int WMMA_K, int WARP_SIZE, int WARP_DIM_X, int WARP_DIM_Y, int W_M, int W_N>//代表WARP处理的维度由[16, 16] -> [16 * W_M, 16 * W_N]
__global__ void matrixKernel(half *dA, half *dB, half *dC, int M, int K, int N)
{
    // int ld_a = K; //行主序：i * K + j，列主序 i + j * K;
    // int ld_b = N;
    int ld_c = N;

    //shared memory部分
    __shared__ half S_A[(WARP_DIM_Y * WARP_DIM_X) * (W_M * WMMA_M * WMMA_K)];
    __shared__ half S_B[(WARP_DIM_Y * WARP_DIM_X) * (W_N * WMMA_K * WMMA_N)];

    // Declare the fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> left_frag[W_M];
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> right_frag[W_N];
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> c_frag[W_M][W_N];

    //全局偏移
    int row, col;
    row = blockIdx.y * (W_M * WMMA_M * WARP_DIM_Y);
    col = blockIdx.x * (W_N * WMMA_N * WARP_DIM_X);

    //块内偏移
    int warp_id_x, warp_id_y;
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    warp_id_y = (tid / WARP_SIZE) / WARP_DIM_X;
    warp_id_x = (tid / WARP_SIZE) % WARP_DIM_X;
    int warp_id = tid / WARP_SIZE;

    //真正偏移
    int row_offset, col_offset;
    row_offset = row + warp_id_y * W_M * WMMA_M;
    col_offset = col + warp_id_x * W_N * WMMA_N;

    //对一个WARP内的thread重索引,映射为S_A_row, S_A_col, S_B_row, S_B_col;
    int S_A_row, S_A_col, S_B_row, S_B_col;
    S_A_row = (tid % WARP_SIZE) % WMMA_M;
    S_A_col = (tid % WARP_SIZE) / WMMA_M;//必须把每一列装满

    S_B_row = (tid % WARP_SIZE) / WMMA_N;//先把每一行放满
    S_B_col = (tid % WARP_SIZE) % WMMA_N;

    //累加器初始化
    for(int W_M_id = 0; W_M_id < W_M; W_M_id++)
    {
        for(int W_N_id = 0; W_N_id < W_N; W_N_id++)
        {
            wmma::fill_fragment(c_frag[W_M_id][W_N_id], 0.0f);
        }
    }


    for(int split_k_id = 0; split_k_id < (K + WMMA_K - 1)/(WMMA_K); split_k_id++)
    {
        //dA->S_A
        for(int S_A_row_id = S_A_row * W_M; S_A_row_id < (S_A_row + 1) * W_M; S_A_row_id++)//就多出个多循环.
        {
            for(int S_A_col_id = S_A_col; S_A_col_id < WMMA_K; S_A_col_id += (WARP_SIZE / WMMA_M))//一次拿WARP_SIZE / WMMA_M列
            {
                S_A[S_A_row_id * WMMA_K + S_A_col_id + warp_id * (W_M * WMMA_M * WMMA_K)] = dA[(row_offset + S_A_row_id) * K + (split_k_id * WMMA_K) + S_A_col_id];
            }
        }


        //dB->S_B
        for(int S_B_col_id = S_B_col * W_N; S_B_col_id < (S_B_col + 1) * W_N; S_B_col_id++)//就多出个多循环.
        {
            for(int S_B_row_id = S_B_row; S_B_row_id < WMMA_K; S_B_row_id += (WARP_SIZE / WMMA_N))//一次拿WARP_SIZE / WMMA_N行
            {
                S_B[S_B_row_id * WMMA_N + S_B_col_id + warp_id * (W_N * WMMA_K * WMMA_N)] = dB[(split_k_id * WMMA_K + S_B_row_id) * N + col_offset + S_B_col_id];
            }
        }
        __syncthreads();

        //S_A -> left_frag
        for(int W_M_id = 0; W_M_id < W_M; W_M_id++)
        {
            wmma::load_matrix_sync(left_frag[W_M_id], S_A + W_M_id * WMMA_M * WMMA_K + warp_id * (W_M * WMMA_M * WMMA_K), WMMA_K);
        }
        __syncthreads();

        //S_B -> right_frag
        for(int W_N_id = 0; W_N_id < W_N; W_N_id++)
        {
            wmma::load_matrix_sync(right_frag[W_N_id], S_B + W_N_id * WMMA_K * WMMA_N + warp_id * (W_N * WMMA_K * WMMA_N), WMMA_N);
        }

        __syncthreads();    

        //multiply
        for(int W_M_id = 0; W_M_id < W_M; W_M_id++)
        {
            for(int W_N_id = 0; W_N_id < W_N; W_N_id++)
            {
                wmma::mma_sync(c_frag[W_M_id][W_N_id], left_frag[W_M_id], right_frag[W_N_id], c_frag[W_M_id][W_N_id]);
            }
        }

        __syncthreads();
    }

    // //搬出
    for(int W_M_id = 0; W_M_id < W_M; W_M_id++)
    {
        for(int W_N_id = 0; W_N_id < W_N; W_N_id++)
        {
            wmma::store_matrix_sync(dC + (row_offset + (W_M_id * WMMA_M)) * N + col_offset + (W_N_id * WMMA_N), c_frag[W_M_id][W_N_id], ld_c, wmma::mem_row_major);
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

    const int W_M = 2;//每个warp的放缩尺度
    const int W_N = 2;

    int num_block_x = (M + W_M * WMMA_M * WARP_DIM_X  - 1) / (W_M * WMMA_M * WARP_DIM_X);
    int num_block_y = (N + W_N * WMMA_N * WARP_DIM_Y - 1) / (W_N * WMMA_N * WARP_DIM_Y);

    dim3 block_dim(BLOCK_DIM_x, BLOCK_DIM_y, 1);
    dim3 grid_dim(num_block_x, num_block_y, 1);
    float ker_time = 0;


    matrixKernel<WMMA_M, WMMA_N, WMMA_K, WARP_SIZE, WARP_DIM_X, WARP_DIM_Y, W_M, W_N><<<grid_dim, block_dim>>>(dA, dB, dC, M, K, N);
    
    int repeat = 20;
    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    for (int i = 0; i < repeat; i++)
    {
       matrixKernel<WMMA_M, WMMA_N, WMMA_K, WARP_SIZE, WARP_DIM_X, WARP_DIM_Y, W_M, W_N><<<grid_dim, block_dim>>>(dA, dB, dC, M, K, N);
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

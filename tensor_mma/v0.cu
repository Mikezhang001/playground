#include <stdio.h>
#include <sys/time.h>
#include <cuda.h>
#include <mma.h>

// #include <cuda_fp16.hpp>  // 包含 half 类型支持
#include <cuda_fp16.h>  // 包含半精度支持
using namespace nvcuda;

template <int MMA_M, int MMA_N, int MMA_K>
__global__ void matrixKernel(half *dA, half *dB, half *dC, int M, int K, int N)
{
    // 全局偏移
    int row = blockIdx.y * MMA_M;
    int col = blockIdx.x * MMA_N;

    // 块内线程索引
    int tid = threadIdx.x + threadIdx.y * blockDim.x;

    __shared__ half S_A[MMA_M][MMA_K];  // [16][16]
    __shared__ half S_B[MMA_K][MMA_N];  // [16][8]
    __shared__ half S_C[MMA_M][MMA_N];  // [16][8]

    int S_A_row = tid / 2;
    int S_A_col = (tid % 2) * 8;//一次拿8个

    int S_B_row = tid / 1;
    int S_B_col = 0;//一次拿8个

    uint32_t RA[4];
    uint32_t RB[2];
    uint32_t RC[2] = {0, 0};


    // K维度循环
    for(int split_k_id = 0; split_k_id < (K + MMA_K - 1) / MMA_K; split_k_id++)
    {
        int base_k = split_k_id * MMA_K;
        
        // ===== 加载 A 矩阵到共享内存 =====
        // A 是行优先: [M][K], 我们加载 [16][16] 块
        // 32个线程协作，每个加载 8个 half (4 uint32_t)
        (float4 &)S_A[S_A_row][S_A_col] = (float4 &)dA[(S_A_row + row) * K + S_A_col + base_k];
        
        // ===== 加载 B 矩阵到共享内存 =====
        // B 是行优先: [K][N], 我们加载 [16][8] 块
        // 32个线程协作，但只需要16个
        if(tid < 16)
        {
            (float4 &)S_B[S_B_row][S_B_col] = (float4 &)dB[(S_B_row + base_k) * N + S_B_col + col];
        }
        __syncthreads();

        uint32_t A_smem_lane = __cvta_generic_to_shared(&S_A[tid % 16][(tid / 16) * 8]);//转成共享内存的格式，布局为[16, 2]
        asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n" \
                : "=r"(RA[0]), "=r"(RA[1]), "=r"(RA[2]), "=r"(RA[3])                     \
                : "r"(A_smem_lane));

        uint32_t B_smem_lane = __cvta_generic_to_shared(&S_B[tid % 16][0]);//感觉这个有些问题, 16 -> 31好像不用标记
        asm volatile("ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0, %1}, [%2];\n" \
                : "=r"(RB[0]), "=r"(RB[1])                                   \
                : "r"(B_smem_lane));//r 四个字节，l 八个字节    

        asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%8, %9};\n" \
                : "=r"(RC[0]), "=r"(RC[1])                                                                                \
                : "r"(RA[0]), "r"(RA[1]), "r"(RA[2]), "r"(RA[3]), "r"(RB[0]), "r"(RB[1]), "r"(RC[0]), "r"(RC[1]));  
        
        __syncthreads();
    }

    *((uint32_t *)(&S_C[tid / 4][0]) + tid % 4) = RC[0];
    *((uint32_t *)(&S_C[tid / 4 + 8][0]) + tid % 4) = RC[1];

    if(tid < 16)//1个lane搬运一整行
    {
        (float4 &)dC[(row + tid) * N + col]= (float4 &)S_C[tid][0];
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


    const int MMA_M = 16;
    const int MMA_N = 8;
    const int MMA_K = 16;

    // const int WARP_SIZE = 32;
    const int BM = 16;
    const int BN = 8;
    //因为还需要重排序,会将block设置为(4, 4) * 32 = 512
    const int BLOCK_DIM_x = 32;
    const int BLOCK_DIM_y = 1;

    int num_block_y = (M + BM - 1) / (BM);
    int num_block_x = (N + BN - 1) / (BN);

    dim3 block_dim(BLOCK_DIM_x, BLOCK_DIM_y, 1);
    dim3 grid_dim(num_block_x, num_block_y, 1);
    float ker_time = 0;

    matrixKernel<MMA_M, MMA_N, MMA_K><<<grid_dim, block_dim>>>(dA, dB, dC, M, K, N);
    
    int repeat = 20;
    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    for (int i = 0; i < repeat; i++)
    {
       matrixKernel<MMA_M, MMA_N, MMA_K><<<grid_dim, block_dim>>>(dA, dB, dC, M, K, N);    
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

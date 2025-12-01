#include <stdio.h>
#include <sys/time.h>
#include <cuda.h>
#include <mma.h>

// #include <cuda_fp16.hpp>  // 包含 half 类型支持
#include <cuda_fp16.h>  // 包含半精度支持
using namespace nvcuda;

#include "header/common.h"//工具类

#define MMA_M (16)
#define MMA_N (8)
#define MMA_K (16)
#define BM (256)
#define BN (128)
#define BK (32)

#define padding_SB (8)
__global__ void matrixKernel(half *dA, half *dB, half *dC, int M, int K, int N)
{
    // 全局偏移
    int row_offset = blockIdx.y * BM;
    int col_offset = blockIdx.x * BN;
    //另一种布局和偏移方法
    // int row_offset = (blockIdx.z % 2) ? ((gridDim.y - blockIdx.y - 1) * BM) : (blockIdx.y * BM);
    // int col_offset = (blockIdx.z * gridDim.x + blockIdx.x) * BN;
    // 块内线程索引
    int tid = threadIdx.x + threadIdx.y * blockDim.x;

    //8个warp，即为每个warp处理[64, 32] [32, 64] = [64, 64]
    // __shared__ half S_A[BM][BK];  // [256][32]
    // __shared__ half S_B[BK][BN];  // [32][128]
    // __shared__ half S_C[BM][BN]; //  [256][128]

    extern __shared__ half smem[];//空间有点大换动态
    half* S_A = smem;
    half* S_B = S_A + BM * BK;
    half* S_C = smem;

    int S_A_row = tid / 4;
    int S_A_col = (tid % 4) * 8;//一次拿8个

    int S_B_row = tid / 16;
    int S_B_col = (tid % 16) * 8 ;//一次拿8个

    int WARP_M = 64;
    int WARP_N = 64;
    int warp_id = tid / 32;//将其映射为[4, 2]列
    int lane_id = tid % 32;//每个warp内的id
    int warp_y = warp_id / 2;
    int warp_x = warp_id % 2; 
    int S_AC_row_offset = warp_y * WARP_M;
    int S_BC_col_offset = warp_x * WARP_N;
    uint32_t RA[4][4];//WARP_M / MMA_M = 4; BK / MMA_K = 2;因为打开循环，所以[4][2][4] -> [4][4]
    uint32_t RB[8][2];//BK / MMA_K = 2; WARP_N / MMA_N = 8;因为打开循环，所以[2][8][2] -> [8][2]
    uint32_t RC[4][8][2] = {0, 0};//WARP_M / MMA_M = 4;WARP_N / MMA_N = 8;


    // K维度循环
#pragma unroll
    for(int split_k_id = 0; split_k_id < (K + BK - 1) / BK; split_k_id++)
    {
        int base_k = split_k_id * BK;
        
        // ===== 加载 A 矩阵到共享内存 =====
        // A 是行优先: [M][K], 我们加载 [16][16] 块
        // 32个线程协作，每个加载 8个 half (4 uint32_t)
        // (float4 &)S_A[S_A_row][S_A_col] = (float4 &)dA[(S_A_row + row) * K + S_A_col + base_k];
        
        //dA -> S_A
#pragma unroll
        for(int S_A_row_id = S_A_row; S_A_row_id < BM; S_A_row_id += 256 / 4)//每行4个thread, 256个thread一次拿64行
        {
            // (float4 &)S_A[S_A_row_id][S_A_col] = (float4 &)dA[(S_A_row_id + row_offset) * K + S_A_col + base_k];
            // (float4 &)S_A[(S_A_row_id) * BK + S_A_col] = (float4 &)dA[(S_A_row_id + row_offset) * K + S_A_col + base_k];

            //每八行中的每组间偏移，其中0-1，2-3, 4-5, 6-7 这种组内不需要做偏移
            (float4 &)S_A[(S_A_row_id) * BK + (S_A_col / 8 + (S_A_row_id % 8) / 2) % 4 * 8] = (float4 &)dA[(S_A_row_id + row_offset) * K + S_A_col + base_k];
        }
        // ===== 加载 B 矩阵到共享内存 =====
        // B 是行优先: [K][N], 我们加载 [16][8] 块
        // 32个线程协作，但只需要16个
        // if(tid < 16)
        // {
        //     (float4 &)S_B[S_B_row][S_B_col] = (float4 &)dB[(S_B_row + base_k) * N + S_B_col + col];
        // }
        //dB -> S_B
#pragma unroll
        for(int S_B_row_id = S_B_row; S_B_row_id < BK; S_B_row_id += 256 / 16)//每行16个thread, 256个thread一次拿16行
        {
            (float4 &)S_B[(S_B_row_id) * (BN + padding_SB) + S_B_col] = (float4 &)dB[(S_B_row_id + base_k) * N + S_B_col + col_offset];
        }
        __syncthreads();

        //S_A -> 左矩阵, 对S_A横着切
        //每个warp要算的[64, 64] = [WARP_M, BK] * [BK, WARP_N];
#pragma unroll
        for(int split_MMA_K_id = 0; split_MMA_K_id < BK / MMA_K; split_MMA_K_id++)
        {
#pragma unroll
            for(int i = 0; i < WARP_M / MMA_M; i++)
            {

                // uint32_t A_smem_lane = __cvta_generic_to_shared(&S_A[tid % 16][(tid / 16) * 8]);//转成共享内存的格式，布局为[16, 2]
                // uint32_t A_smem_lane = __cvta_generic_to_shared(&S_A[lane_id % 16 + i * MMA_M + S_AC_row_offset][(lane_id / 16) * 8 + j * MMA_K]);//加上偏移
                // uint32_t A_smem_lane = __cvta_generic_to_shared(&S_A[(lane_id % 16 + i * MMA_M + S_AC_row_offset) * BK + (lane_id / 16) * 8 + split_MMA_K_id * MMA_K]);//加上偏移

                //每八行中的每组间偏移，其中0-1，2-3, 4-5, 6-7 这种组内不需要做偏移：原来的列(lane_id / 16 + split_MMA_K_id * MMA_K / 8)，需要再加上偏移(lane_id % 8)/2
                uint32_t A_smem_lane = __cvta_generic_to_shared(&S_A[(lane_id % 16 + i * MMA_M + S_AC_row_offset) * BK + ((lane_id / 16 + split_MMA_K_id * MMA_K / 8 + (lane_id % 8) / 2)) % 4 * 8]);//
                                       
                asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n" \
                    : "=r"(RA[i][0]), "=r"(RA[i][1]), "=r"(RA[i][2]), "=r"(RA[i][3])                     \
                    : "r"(A_smem_lane));     

            }

        //S_B ->右矩阵，对S_B纵着切
#pragma unroll
            for(int j = 0; j < WARP_N / MMA_N; j++)
            {
                // uint32_t B_smem_lane = __cvta_generic_to_shared(&S_B[tid % 16][0]);//感觉这个有些问题, 16 -> 31好像不用标记
                // uint32_t B_smem_lane = __cvta_generic_to_shared(&S_B[tid % 16 + i * MMA_K][j * MMA_N + S_BC_col_offset]);
                uint32_t B_smem_lane = __cvta_generic_to_shared(&S_B[(tid % 16 + split_MMA_K_id * MMA_K) * (BN + padding_SB) + j * MMA_N + S_BC_col_offset]);
                asm volatile("ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0, %1}, [%2];\n" \
                    : "=r"(RB[j][0]), "=r"(RB[j][1])                                   \
                    : "r"(B_smem_lane));//r 四个字节，l 八个字节    
            }
#pragma unroll
        //左矩阵 × 右矩阵
            for(int i = 0; i < WARP_M / MMA_M; i++)
            {
#pragma unroll
                for(int j = 0; j < WARP_N / MMA_N; j++)
                {

                    // asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%8, %9};\n" \
                    //     : "=r"(RC[0]), "=r"(RC[1])                                                                                \
                    //     : "r"(RA[0]), "r"(RA[1]), "r"(RA[2]), "r"(RA[3]), "r"(RB[0]), "r"(RB[1]), "r"(RC[0]), "r"(RC[1]));  
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%8, %9};\n" \
                        : "=r"(RC[i][j][0]), "=r"(RC[i][j][1])                                                                                \
                        : "r"(RA[i][0]), "r"(RA[i][1]), "r"(RA[i][2]), "r"(RA[i][3]), "r"(RB[j][0]), "r"(RB[j][1]), "r"(RC[i][j][0]), "r"(RC[i][j][1]));  
                }
            }
        }
        __syncthreads();

    }

    //结果矩阵搬到S_C
#pragma unroll
    for(int i = 0; i < WARP_M / MMA_M; i++)
    {
#pragma unroll
        for(int j = 0; j < WARP_N / MMA_N; j++)
        {
            ////uint32_t 4字节两个half
            // *((uint32_t *)(&S_C[tid / 4][0]) + tid % 4) = RC[0];
            // *((uint32_t *)(&S_C[tid / 4 + 8][0]) + tid % 4) = RC[1];
            // *((uint32_t *)(&S_C[lane_id / 4 + i * MMA_M + S_AC_row_offset][j * MMA_N + S_BC_col_offset]) + lane_id % 4) = RC[i][j][0];
            // *((uint32_t *)(&S_C[lane_id / 4 + 8 + i * MMA_M + S_AC_row_offset][j * MMA_N + S_BC_col_offset]) + lane_id % 4) = RC[i][j][1];

            //一、
            *((uint32_t *)(&S_C[(lane_id / 4 + i * MMA_M + S_AC_row_offset) * BN + j * MMA_N + S_BC_col_offset]) + lane_id % 4) = RC[i][j][0];
            *((uint32_t *)(&S_C[(lane_id / 4 + 8 + i * MMA_M + S_AC_row_offset) * BN + j * MMA_N + S_BC_col_offset]) + lane_id % 4) = RC[i][j][1];

            //二、与 一 等价
            // *(uint32_t *)(&S_C[(lane_id / 4 + i * MMA_M + S_AC_row_offset) * BN + j * MMA_N + S_BC_col_offset + lane_id % 4 * 2]) = RC[i][j][0];
            // *(uint32_t *)(&S_C[(lane_id / 4 + 8 + i * MMA_M + S_AC_row_offset) * BN + j * MMA_N + S_BC_col_offset + lane_id % 4 * 2]) = RC[i][j][1];

            //观察可知，需要将每行都偏移4个bank即为4个half，以8为周期
            // *(uint32_t *)(&S_C[(lane_id / 4 + i * MMA_M + S_AC_row_offset) * BN + ((j * MMA_N + S_BC_col_offset + lane_id % 4 * 2) + (lane_id / 4) * 8) % BN]) = RC[i][j][0];
            // // *(uint32_t *)(&S_C[(lane_id / 4 + 8 + i * MMA_M + S_AC_row_offset) * BN + ((j * MMA_N + S_BC_col_offset + lane_id % 4 * 2) + (lane_id / 4 + 8) * 8) % BN]) = RC[i][j][1];//一样
            // *(uint32_t *)(&S_C[(lane_id / 4 + 8 + i * MMA_M + S_AC_row_offset) * BN + ((j * MMA_N + S_BC_col_offset + lane_id % 4 * 2) + (lane_id / 4) * 8) % BN]) = RC[i][j][1];//一样
        }
    }

   //从S_C搬到global memory[256, 128]
    int S_C_row = tid / 16;
    int S_C_col = (tid % 16) * 8;
#pragma unroll
    for(int S_C_row_id = S_C_row; S_C_row_id < BM; S_C_row_id += 256 / 16)
    {
        // (float4 &)dC[S_C_row_id + row_offset][S_C_col + col_offset] = (float4 &)S_C[S_C_row_id][S_C_col];
        // (float4 &)dC[(S_C_row_id + row_offset) * N + S_C_col + col_offset] = (float4 &)S_C[S_C_row_id][S_C_col];
        
        (float4 &)dC[(S_C_row_id + row_offset) * N + S_C_col + col_offset] = (float4 &)S_C[(S_C_row_id) * BN + S_C_col];

        // //观察可知，需要将每行都偏移4个bank即为4个half，以8为周期
        // (float4 &)dC[(S_C_row_id + row_offset) * N + S_C_col + col_offset] = (float4 &)S_C[(S_C_row_id) * BN + (S_C_col + S_C_row_id % 8 * 8) % BN];
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
// size_t initMmaBase(int BLOCK_ROWS, int BLOCK_COLS, int AB_SMEM_STRIDE, int C_SMEM_STRIDE) {
//     int dev_id = 0;
//     HGEMM_CHECK_CUDART_ERROR(cudaGetDevice(&dev_id));

//     cudaDeviceProp dev_prop;
//     HGEMM_CHECK_CUDART_ERROR(cudaGetDeviceProperties(&dev_prop, dev_id));

//     size_t smem_max_size =
//         std::max((BLOCK_ROWS + BLOCK_COLS) * AB_SMEM_STRIDE * sizeof(half), BLOCK_ROWS * C_SMEM_STRIDE * sizeof(half));
//     HLOG("smem_max_size: %.0f KBytes (%zu Bytes)", static_cast<double>(smem_max_size) / 1024, smem_max_size);

//     HGEMM_CHECK_GT(dev_prop.sharedMemPerMultiprocessor, smem_max_size);
//     HGEMM_CHECK_CUDART_ERROR(
//         cudaFuncSetAttribute(matrixKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_max_size));

//     return smem_max_size;
// }

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


    //因为还需要重排序,会将block设置为256
    const int BLOCK_DIM_x = 16;
    const int BLOCK_DIM_y = 16;



    dim3 block_dim(BLOCK_DIM_x, BLOCK_DIM_y, 1);
    //一种调用
    int num_block_y = (M + BM - 1) / (BM);
    int num_block_x = (N + BN - 1) / (BN);
    dim3 grid_dim(num_block_x, num_block_y, 1);
    //第二种调用
    // const int BLOCK_STRIDE = 16;
    // dim3 grid_dim(BLOCK_STRIDE, (M  + BM -1) / BM, (N + BN * BLOCK_STRIDE - 1) / (BN * BLOCK_STRIDE));

    float ker_time = 0;

    //shared memory单块开大些
    // static size_t smem_max_size = initMmaBase(BM, BN, BK, BN);
    size_t smem_max_size = std::max((BM + BN) * BK * sizeof(half), BM * BN * sizeof(half));
    cudaFuncSetAttribute(matrixKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_max_size);

    matrixKernel<<<grid_dim, block_dim, smem_max_size>>>(dA, dB, dC, M, K, N);
    
    int repeat = 20;
    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    for (int i = 0; i < repeat; i++)
    {
       matrixKernel<<<grid_dim, block_dim, smem_max_size>>>(dA, dB, dC, M, K, N);    
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
    int M = 4096;
    int K = 4096;
    int N = 4096;



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
    // double st, ela;
    // st = get_walltime();
    // matrixSerial(hostA, hostB, serialC, M, K, N);
    // ela = get_walltime() - st;
    // float error = compare(hostC, serialC, M, N);
    // printf("CPU time:%.2f, error:%.4e\n", ela, error);


    
    // for (int i = 0; i < 20; i++) {
    //     printf("%2d: hostC=%f    serialC=%f\n", i, __half2float(hostC[i]), __half2float(serialC[i]));//累加器明明是可以float16的，为啥gpt都说是不可以的，用0-1的小数随机，确实由较大误差
    // }
    free(hostA);
    free(hostB);
    free(hostC);
    free(serialC);
    return 0;
}

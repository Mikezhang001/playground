// Copyright 2023. All Rights Reserved.
// Author: Bruce-Lee-LY (Modified by Gemini for clarity and correct memory usage)
//
// Description: mma hgemm using the highly effective Padding Offset strategy (the user's optimal approach).

#include <stdio.h>
#include <sys/time.h>
#include <cuda.h>
#include <mma.h>
#include <cuda_fp16.h> // Includes half precision support
#include <algorithm>   // for std::max
#include <cmath>       // for fabs

using namespace nvcuda;

// 假设 common.h 包含了 utility macros, 这里为了代码独立性移除
#define HLOG(...) (void)0
#define HGEMM_CHECK_CUDART_ERROR(ans) do { (void)ans; } while(0)
#define HGEMM_CHECK_GT(val1, val2) do { (void)val1; (void)val2; } while(0)

#define MMA_M (16)
#define MMA_N (8)
#define MMA_K (16)
#define BM (256)
#define BN (128)
#define BK (32)

// 填充参数定义：8个 half-word (16字节)，用于打破 32-bank 冲突
#define S_A_PAD (8) // BK + 8 = 40 (非 32 的倍数)
#define S_B_PAD (8) // BN + 8 = 136 (非 32 的倍数)
#define S_A_STRIDE (BK + S_A_PAD)
#define S_B_STRIDE (BN + S_B_PAD)

__global__ void matrixKernel(half *dA, half *dB, half *dC, int M, int K, int N)
{
    // 全局偏移
    // 采用您的简单二维网格布局
    int row_offset = blockIdx.y * BM;
    int col_offset = blockIdx.x * BN;
    
    if (row_offset >= M || col_offset >= N) return;
    
    // 块内线程索引
    int tid = threadIdx.x + threadIdx.y * blockDim.x;

    extern __shared__ half smem[];//动态共享内存
    half* S_A = smem;
    // S_B 偏移计算：BM行 * S_A 的带填充步长
    half* S_B = S_A + BM * S_A_STRIDE;
    half* S_C = smem; // S_C 重用空间

    // G-Mem to S-Mem load coordinates
    int S_A_row = tid / 4;
    int S_A_col = (tid % 4) * 8; // Load 8 half-words (float4)

    int S_B_row = tid / 16;
    int S_B_col = (tid % 16) * 8; // Load 8 half-words (float4)

    // Warp setup
    int WARP_M = 64;
    int WARP_N = 64;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int warp_y = warp_id / 2;
    int warp_x = warp_id % 2; 
    int S_AC_row_offset = warp_y * WARP_M;
    int S_BC_col_offset = warp_x * WARP_N;
    
    // Registers for MMA
    uint32_t RA[4][4];  // 4 M-tiles (4 * 16) * 4 regs (16 words)
    uint32_t RB[8][2];  // 8 N-tiles (8 * 8) * 2 regs (8 words)
    uint32_t RC[4][8][2] = {0, 0}; // Accumulator fragments

    // K维度循环
    for(int split_k_id = 0; split_k_id < (K + BK - 1) / BK; split_k_id++)
    {
        int base_k = split_k_id * BK;
        
        // ===== G-Mem -> S-Mem Load (使用填充步长 S_A_STRIDE=40) =====
        
        // dA -> S_A
        #pragma unroll
        for(int S_A_row_id = S_A_row; S_A_row_id < BM; S_A_row_id += 256 / 4)
        {
            if (S_A_row_id + row_offset < M && S_A_col + base_k < K) 
            {
                // S_A 写入：使用 S_A_STRIDE (40)
                // (float4 &)S_A[(S_A_row_id) * S_A_STRIDE + S_A_col] = (float4 &)dA[(S_A_row_id + row_offset) * K + S_A_col + base_k];
        
                uint32_t dst = __cvta_generic_to_shared(&S_A[(S_A_row_id) * S_A_STRIDE + S_A_col]);
                uint64_t src = (uint64_t)dA + (uint64_t)((S_A_row_id + row_offset) * K + S_A_col + base_k) * sizeof(half);
                // cp.async.cg.shared.global [dst], [src], 16 (Bytes)
                asm volatile("cp.async.cg.shared.global [%0], [%1], %2;\n" ::"r"(dst), "l"(src), "n"(16));
            }
        }
        // dB -> S_B
        #pragma unroll
        for(int S_B_row_id = S_B_row; S_B_row_id < BK; S_B_row_id += 256 / 16)
        {
            if (S_B_row_id + base_k < K && S_B_col + col_offset < N) 
            {
                // S_B 写入：使用 S_B_STRIDE (136)
                // (float4 &)S_B[(S_B_row_id) * S_B_STRIDE + S_B_col] = (float4 &)dB[(S_B_row_id + base_k) * N + S_B_col + col_offset];

                uint32_t dst = __cvta_generic_to_shared(&S_B[(S_B_row_id) * S_B_STRIDE + S_B_col]);
                uint64_t src = (uint64_t)dB + (uint64_t)((S_B_row_id + base_k) * N + S_B_col + col_offset) * sizeof(half);
                // cp.async.cg.shared.global [dst], [src], 16 (Bytes)
                asm volatile("cp.async.cg.shared.global [%0], [%1], %2;\n" ::"r"(dst), "l"(src), "n"(16));
            }
        }

        asm volatile("cp.async.commit_group;\n" ::);
        asm volatile("cp.async.wait_group %0;\n" ::"n"(0));
        __syncthreads();

        // ===== S-Mem -> Register Load (LDMATRIX) (使用填充步长) =====
        
        // S_A -> RA (A matrix fragments)
        #pragma unroll
        for(int split_MMA_K_id = 0; split_MMA_K_id < BK / MMA_K; split_MMA_K_id++)
        {
            #pragma unroll
            for(int i = 0; i < WARP_M / MMA_M; i++)
            {
                // Calculate LDMATRIX A base row and column
                int r = lane_id % 16 + i * MMA_M + S_AC_row_offset;
                int c = (lane_id / 16) * 8 + split_MMA_K_id * MMA_K;
                
                // S_A 读取：使用 S_A_STRIDE (40)
                uint32_t A_smem_lane = __cvta_generic_to_shared(&S_A[r * S_A_STRIDE + c]);
                
                asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n" \
                    : "=r"(RA[i][0]), "=r"(RA[i][1]), "=r"(RA[i][2]), "=r"(RA[i][3])              \
                    : "r"(A_smem_lane)); 
            }

            // S_B -> RB (B matrix fragments)
            #pragma unroll
            for(int j = 0; j < WARP_N / MMA_N; j++)
            {
                // Calculate LDMATRIX B base row and column
                int r = tid % 16 + split_MMA_K_id * MMA_K;
                int c = j * MMA_N + S_BC_col_offset;
                
                // S_B 读取：使用 S_B_STRIDE (136)
                uint32_t B_smem_lane = __cvta_generic_to_shared(&S_B[r * S_B_STRIDE + c]);
                
                // Note: .trans is used for B matrix loading
                asm volatile("ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0, %1}, [%2];\n" \
                    : "=r"(RB[j][0]), "=r"(RB[j][1])                                            \
                    : "r"(B_smem_lane)); 
            }

            // 执行 MMA 操作 (累加到 RC)
            #pragma unroll
            for(int i = 0; i < WARP_M / MMA_M; i++)
            {
                #pragma unroll
                for(int j = 0; j < WARP_N / MMA_N; j++)
                {
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%8, %9};\n" \
                        : "=r"(RC[i][j][0]), "=r"(RC[i][j][1])                                                                    \
                        : "r"(RA[i][0]), "r"(RA[i][1]), "r"(RA[i][2]), "r"(RA[i][3]), "r"(RB[j][0]), "r"(RB[j][1]), "r"(RC[i][j][0]), "r"(RC[i][j][1])); 
                }
            }
        }
        __syncthreads(); // Synchronize before next K block
    } // K-Loop End

    // Register -> S_C Store (回写 S_C 不需要 Swizzle/Padding，使用原始 BN 步长 128)
    __syncthreads(); // 确保 MMA 完成
    
    #pragma unroll
    for(int i = 0; i < WARP_M / MMA_M; i++)
    {
        #pragma unroll
        for(int j = 0; j < WARP_N / MMA_N; j++)
        {
            // S_C 写入：使用原始 BN 步长 (128)
            int base_idx = (lane_id / 4 + i * MMA_M + S_AC_row_offset) * BN + j * MMA_N + S_BC_col_offset;
            int offset_idx = lane_id % 4; // uint32_t offset
            
            // Store 16x8 block (4 uint32_t units wide)
            *((uint32_t *)(&S_C[base_idx]) + offset_idx) = RC[i][j][0];
            *((uint32_t *)(&S_C[base_idx + 8 * BN]) + offset_idx) = RC[i][j][1]; // +8 rows for the second fragment
        }
    }
    
    __syncthreads();

    // S_C -> G-Mem Write
    int S_C_row = tid / 16;
    int S_C_col = (tid % 16) * 8;
    #pragma unroll
    for(int S_C_row_id = S_C_row; S_C_row_id < BM; S_C_row_id += 256 / 16)
    {
        if (S_C_row_id + row_offset < M && S_C_col + col_offset < N) {
            // S_C 读取：使用原始 BN 步长 (128)
            int smem_idx = (S_C_row_id) * BN + S_C_col;
            (float4 &)dC[(S_C_row_id + row_offset) * N + S_C_col + col_offset] = (float4 &)S_C[smem_idx];
        }
    }
}

// ===================================================
// Host-side code
// ===================================================

double get_walltime()
{
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return (double)(tp.tv_sec + tp.tv_usec * 1e-6);
}

float compare(half *hostC, half *serialC, int M, int N)
{
    float error = 0;
    for (int i = 0; i < M * N; i++)
    {
        error = std::fmax(error, fabs(__half2float(hostC[i]) - __half2float(serialC[i])));
    }
    return error;
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
void hostMatrix(half *hostA, half *hostB, half *hostC, int M, int K, int N)
{
    double st, ela;
    st = get_walltime();

    half *dA, *dB, *dC;
    // 假设 B 矩阵是 KxN 布局
    HGEMM_CHECK_CUDART_ERROR(cudaMalloc((void **)&dA, M * K * sizeof(half)));
    HGEMM_CHECK_CUDART_ERROR(cudaMalloc((void **)&dB, K * N * sizeof(half))); 
    HGEMM_CHECK_CUDART_ERROR(cudaMalloc((void **)&dC, M * N * sizeof(half)));

    HGEMM_CHECK_CUDART_ERROR(cudaMemcpy(dA, hostA, M * K * sizeof(half), cudaMemcpyHostToDevice));
    HGEMM_CHECK_CUDART_ERROR(cudaMemcpy(dB, hostB, K * N * sizeof(half), cudaMemcpyHostToDevice));

    const int BLOCK_DIM_x = 16;
    const int BLOCK_DIM_y = 16;
    dim3 block_dim(BLOCK_DIM_x, BLOCK_DIM_y, 1);
    
    int num_block_y = (M + BM - 1) / (BM);
    int num_block_x = (N + BN - 1) / (BN);
    dim3 grid_dim(num_block_x, num_block_y, 1);

    float ker_time = 0;

    // 共享内存最大尺寸修正：S_A 空间 + S_B 空间，与 S_C 空间取最大值
    // S_A 空间: BM * S_A_STRIDE (half words)
    // S_B 空间: BK * S_B_STRIDE (half words)
    // S_C 空间: BM * BN (half words)
    size_t smem_size_ab_half = (BM * S_A_STRIDE + BK * S_B_STRIDE);
    size_t smem_size_c_half = BM * BN;
    size_t smem_max_size = std::max(smem_size_ab_half, smem_size_c_half) * sizeof(half);

    HGEMM_CHECK_CUDART_ERROR(cudaFuncSetAttribute(matrixKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_max_size));

    matrixKernel<<<grid_dim, block_dim, smem_max_size>>>(dA, dB, dC, M, K, N);
    
    int repeat = 20;
    cudaEvent_t start, stop;

    HGEMM_CHECK_CUDART_ERROR(cudaEventCreate(&start));
    HGEMM_CHECK_CUDART_ERROR(cudaEventCreate(&stop));
    HGEMM_CHECK_CUDART_ERROR(cudaEventRecord(start, 0));
    for (int i = 0; i < repeat; i++)
    {
       matrixKernel<<<grid_dim, block_dim, smem_max_size>>>(dA, dB, dC, M, K, N);    
    }

    HGEMM_CHECK_CUDART_ERROR(cudaEventRecord(stop, 0));
    HGEMM_CHECK_CUDART_ERROR(cudaEventSynchronize(stop));
    HGEMM_CHECK_CUDART_ERROR(cudaEventElapsedTime(&ker_time, start, stop));

    HGEMM_CHECK_CUDART_ERROR(cudaMemcpy(hostC, dC, M * N * sizeof(half), cudaMemcpyDeviceToHost));

    HGEMM_CHECK_CUDART_ERROR(cudaFree(dA));
    HGEMM_CHECK_CUDART_ERROR(cudaFree(dB));
    HGEMM_CHECK_CUDART_ERROR(cudaFree(dC));
    HGEMM_CHECK_CUDART_ERROR(cudaEventDestroy(start));
    HGEMM_CHECK_CUDART_ERROR(cudaEventDestroy(stop));
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
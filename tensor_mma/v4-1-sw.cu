#include <cuda.h>
#include <mma.h>
#include <stdio.h>
#include <sys/time.h>

// #include <cuda_fp16.hpp>  // 包含 half 类型支持
#include <cuda_fp16.h> // 包含半精度支持
using namespace nvcuda;

#include "header/common.h" //工具类

#define MMA_M 16
#define MMA_N 8
#define MMA_K 16

#define BLOCK_ROWS 256
#define BLOCK_COLS 128

#define WARP_ROWS 64
#define WARP_COLS 64

#define BLOCK_ROW_WARPS 2 // BLOCK_COLS / WARP_COLS
#define BLOCK_COL_WARPS 4 // BLOCK_ROWS / WARP_ROWS

#define BLOCK_ROW_TILES 16 // BLOCK_COLS / MMA_N
#define BLOCK_COL_TILES 16 // BLOCK_ROWS / MMA_M

#define WARP_ROW_TILES 8 // WARP_COLS / MMA_N
#define WARP_COL_TILES 4 // WARP_ROWS / MMA_M

#define WARP_SIZE 32
#define WARPS_PER_BLOCK 8     // BLOCK_ROW_WARPS * BLOCK_COL_WARPS
#define THREADS_PER_BLOCK 256 // WARP_SIZE * WARPS_PER_BLOCK

#define CHUNK_K 2 // 32 / MMA_K

#define CHUNK_LINE_BYTES 64 // CHUNK_K * MMA_K * sizeof(half)
#define CHUNK_COPY_LINES_PER_WARP 8   // WARP_SIZE * sizeof(int4) / CHUNK_LINE_BYTES
#define CHUNK_COPY_LINE_LANES 4 // WARP_SIZE / CHUNK_COPY_LINES_PER_WARP

#define AB_SMEM_STRIDE 32 // CHUNK_K * MMA_K

#define C_SMEM_STRIDE 128 // BLOCK_COLS
#define C_SMEM_OFFSET 64  // WARP_COLS

#define BLOCK_STRIDE 16

#define THREAD_COPY_BYTES 16

#define SMEM_BANK_ROWS 2 // 32 * 4 / (AB_SMEM_STRIDE * sizeof(half))

#define PERMUTED_OFFSET 8
#define PERMUTED_COLS 4

#define BM (256)
#define BN (128)
#define BK (32)
#define padding_SB (8)

#define K_STAGE 2
__global__ void matrixKernel(half *A, half *B, half *C, int M, int K, int N)
{
    const size_t M_tiles = div_ceil(M, MMA_M);
    const size_t N_tiles = div_ceil(N, MMA_N);
    const size_t K_tiles = div_ceil(K, MMA_K);

    const size_t block_tile_i = blockIdx.y * BLOCK_COL_TILES;
    const size_t block_tile_j = blockIdx.x * BLOCK_ROW_TILES;
    if (block_tile_i >= M_tiles || block_tile_j >= N_tiles) {
        return;
    }

    extern __shared__ half smem[];

    // int tid = threadIdx.x;
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    const size_t warp_id = tid / WARP_SIZE;
    const size_t lane_id = tid % WARP_SIZE;

    uint32_t RC[WARP_COL_TILES][WARP_ROW_TILES][2];

    // A -> S_A
    half *S_A = &smem[0];
    int S_A_row = lane_id / 4 + (BM / 8) * warp_id; // 每个warp连续拿行
    int S_A_col = (lane_id % 4) * 8;                // 一次拿8个
    int row_offset = blockIdx.y * BM;

    // B -> S_B
    half *S_B = &smem[0] + BM * BK;
    int S_B_row = lane_id / 16 + (BK / 8) * warp_id; // 每个warp连续拿行
    int S_B_col = (lane_id % 16) * 8;                // 一次拿8个
    int col_offset = blockIdx.x * BN;

    // S_B ->register
    int WARP_M = 64;
    int WARP_N = 64;
    int warp_y = warp_id / 2;
    int warp_x = warp_id % 2;
    int S_AC_row_offset = warp_y * WARP_M;
    int S_BC_col_offset = warp_x * WARP_N;

    // S_C
    half *S_C = &smem[0];

#pragma unroll
    for (size_t i = 0; i < WARP_COL_TILES; ++i) {
#pragma unroll
        for (size_t j = 0; j < WARP_ROW_TILES; ++j) {
            RC[i][j][0] = 0;
            RC[i][j][1] = 0;
        }
    }
    //buffer偏移
    int BUFFER_OFFSET = BM * BK + BK * (BN + padding_SB);

    int buffer_offset_load_idx = 0;
    int buffer_offset_load = 0;

    int buffer_offset_store_idx = 0;
    int buffer_offset_store = 0;

    S_A = &smem[0];
    S_B = &smem[0] + BM * BK;
    int base_k = 0;
    // 第0轮数据读取
    uint32_t dst_1 = __cvta_generic_to_shared(&S_A[(S_A_row) * BK + (S_A_col / 8 + (S_A_row % 8) / 2) % 4 * 8]);
    uint64_t src_1 = (uint64_t)A + (uint64_t)((S_A_row + row_offset) * K + S_A_col + base_k) * sizeof(half);
    asm volatile("cp.async.cg.shared.global [%0], [%1], %2;\n" ::"r"(dst_1), "l"(src_1), "n"(16));
    
    uint32_t dst_2 = __cvta_generic_to_shared(&S_A[(S_A_row + 8) * BK + (S_A_col / 8 + ((S_A_row + 8) % 8) / 2) % 4 * 8]);
    uint64_t src_2 = (uint64_t)A + (uint64_t)((S_A_row + 8 + row_offset) * K + S_A_col + base_k) * sizeof(half);
    asm volatile("cp.async.cg.shared.global [%0], [%1], %2;\n" ::"r"(dst_2), "l"(src_2), "n"(16));

    uint32_t dst_3 = __cvta_generic_to_shared(&S_A[(S_A_row + 16) * BK + (S_A_col / 8 + ((S_A_row + 16) % 8) / 2) % 4 * 8]);
    uint64_t src_3 = (uint64_t)A + (uint64_t)((S_A_row + 16 + row_offset) * K + S_A_col + base_k) * sizeof(half);
    asm volatile("cp.async.cg.shared.global [%0], [%1], %2;\n" ::"r"(dst_3), "l"(src_3), "n"(16));

    uint32_t dst_4 = __cvta_generic_to_shared(&S_A[(S_A_row + 24) * BK + (S_A_col / 8 + ((S_A_row + 24) % 8) / 2) % 4 * 8]);
    uint64_t src_4 = (uint64_t)A + (uint64_t)((S_A_row + 24 + row_offset) * K + S_A_col + base_k) * sizeof(half);
    asm volatile("cp.async.cg.shared.global [%0], [%1], %2;\n" ::"r"(dst_4), "l"(src_4), "n"(16));


    uint32_t dst_5 = __cvta_generic_to_shared(&S_B[(S_B_row) * (BN + padding_SB) + S_B_col]);
    uint64_t src_5 = (uint64_t)B + (uint64_t)((S_B_row + base_k) * N + S_B_col + col_offset) * sizeof(half);
    asm volatile("cp.async.cg.shared.global [%0], [%1], %2;\n" ::"r"(dst_5), "l"(src_5), "n"(16));

    uint32_t dst_6 = __cvta_generic_to_shared(&S_B[(S_B_row + 2) * (BN + padding_SB) + S_B_col]);
    uint64_t src_6 = (uint64_t)B + (uint64_t)((S_B_row + 2 + base_k) * N + S_B_col + col_offset) * sizeof(half);
    asm volatile("cp.async.cg.shared.global [%0], [%1], %2;\n" ::"r"(dst_6), "l"(src_6), "n"(16));

    asm volatile("cp.async.commit_group;\n" ::);
    asm volatile("cp.async.wait_group %0;\n" ::"n"(0));
    __syncthreads();
#pragma unroll
    for (size_t tile_k = 2; tile_k < K_tiles; tile_k += CHUNK_K) {

        int base_k = tile_k * 16;

        buffer_offset_store_idx = (buffer_offset_store_idx + 1) % K_STAGE;
        buffer_offset_store = buffer_offset_store_idx * BUFFER_OFFSET;
        
        

        // 第tile_k轮数据读取
        S_A = &smem[0] + buffer_offset_store;
        S_B = &smem[0] + BM * BK + buffer_offset_store;

        uint32_t dst_1 = __cvta_generic_to_shared(&S_A[(S_A_row) * BK + (S_A_col / 8 + (S_A_row % 8) / 2) % 4 * 8]);
        uint64_t src_1 = (uint64_t)A + (uint64_t)((S_A_row + row_offset) * K + S_A_col + base_k) * sizeof(half);
        asm volatile("cp.async.cg.shared.global [%0], [%1], %2;\n" ::"r"(dst_1), "l"(src_1), "n"(16));
    
        uint32_t dst_2 = __cvta_generic_to_shared(&S_A[(S_A_row + 8) * BK + (S_A_col / 8 + ((S_A_row + 8) % 8) / 2) % 4 * 8]);
        uint64_t src_2 = (uint64_t)A + (uint64_t)((S_A_row + 8 + row_offset) * K + S_A_col + base_k) * sizeof(half);
        asm volatile("cp.async.cg.shared.global [%0], [%1], %2;\n" ::"r"(dst_2), "l"(src_2), "n"(16));

        uint32_t dst_3 = __cvta_generic_to_shared(&S_A[(S_A_row + 16) * BK + (S_A_col / 8 + ((S_A_row + 16) % 8) / 2) % 4 * 8]);
        uint64_t src_3 = (uint64_t)A + (uint64_t)((S_A_row + 16 + row_offset) * K + S_A_col + base_k) * sizeof(half);
        asm volatile("cp.async.cg.shared.global [%0], [%1], %2;\n" ::"r"(dst_3), "l"(src_3), "n"(16));

        uint32_t dst_4 = __cvta_generic_to_shared(&S_A[(S_A_row + 24) * BK + (S_A_col / 8 + ((S_A_row + 24) % 8) / 2) % 4 * 8]);
        uint64_t src_4 = (uint64_t)A + (uint64_t)((S_A_row + 24 + row_offset) * K + S_A_col + base_k) * sizeof(half);
        asm volatile("cp.async.cg.shared.global [%0], [%1], %2;\n" ::"r"(dst_4), "l"(src_4), "n"(16));


        uint32_t dst_5 = __cvta_generic_to_shared(&S_B[(S_B_row) * (BN + padding_SB) + S_B_col]);
        uint64_t src_5 = (uint64_t)B + (uint64_t)((S_B_row + base_k) * N + S_B_col + col_offset) * sizeof(half);
        asm volatile("cp.async.cg.shared.global [%0], [%1], %2;\n" ::"r"(dst_5), "l"(src_5), "n"(16));

        uint32_t dst_6 = __cvta_generic_to_shared(&S_B[(S_B_row + 2) * (BN + padding_SB) + S_B_col]);
        uint64_t src_6 = (uint64_t)B + (uint64_t)((S_B_row + 2 + base_k) * N + S_B_col + col_offset) * sizeof(half);
        asm volatile("cp.async.cg.shared.global [%0], [%1], %2;\n" ::"r"(dst_6), "l"(src_6), "n"(16));


        // 第tile_k - 1 轮数据计算
        S_A = &smem[0] + buffer_offset_load;
        S_B = &smem[0] + BM * BK + buffer_offset_load;
#pragma unroll
        for (size_t k_step = 0; k_step < CHUNK_K; ++k_step) {
            uint32_t RA[WARP_COL_TILES][4];
            uint32_t RB[WARP_ROW_TILES][2];

#pragma unroll
            for (size_t i = 0; i < WARP_COL_TILES; ++i) {
                size_t A_smem_idx = (warp_id / BLOCK_ROW_WARPS) * WARP_ROWS + i * MMA_M;

                // uint32_t A_smem_lane_addr = __cvta_generic_to_shared( &smem[(A_smem_idx + lane_id % 16)][((lane_id / 16 + k_step * MMA_K / 8 + (lane_id % 8) / 2)) % 4 * 8]);//
                uint32_t A_smem_lane_addr = __cvta_generic_to_shared(&S_A[(A_smem_idx + lane_id % 16) * BK + ((lane_id / 16 + k_step * MMA_K / 8 + (lane_id % 8) / 2)) % 4 * 8]);//                  
                asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n" \
                    : "=r"(RA[i][0]), "=r"(RA[i][1]), "=r"(RA[i][2]), "=r"(RA[i][3])                     \
                    : "r"(A_smem_lane_addr));    
                
            }

#pragma unroll
            for(int j = 0; j < WARP_ROW_TILES; j++)
            {
                // uint32_t B_smem_lane = __cvta_generic_to_shared(&S_B[tid % 16][0]);//感觉这个有些问题, 16 -> 31好像不用标记
                // uint32_t B_smem_lane = __cvta_generic_to_shared(&S_B[tid % 16 + i * MMA_K][j * MMA_N + S_BC_col_offset]);

                uint32_t B_smem_lane = __cvta_generic_to_shared(&S_B[(tid % 16 + k_step * MMA_K) * (BN + padding_SB) + j * MMA_N + S_BC_col_offset]);
                asm volatile("ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0, %1}, [%2];\n" \
                    : "=r"(RB[j][0]), "=r"(RB[j][1])                                   \
                    : "r"(B_smem_lane));//r 四个字节，l 八个字节    

            }

#pragma unroll
            for (size_t i = 0; i < WARP_COL_TILES; ++i) {
#pragma unroll
                for (size_t j = 0; j < WARP_ROW_TILES; ++j) {
                    size_t j_s = (i % 2) ? (WARP_ROW_TILES - j - 1) : j;

                    HMMA16816(RC[i][j_s][0], RC[i][j_s][1], RA[i][0], RA[i][1], RA[i][2], RA[i][3], RB[j_s][0],
                              RB[j_s][1], RC[i][j_s][0], RC[i][j_s][1]);
                }
            }
        }

        asm volatile("cp.async.commit_group;\n" ::);
        asm volatile("cp.async.wait_group %0;\n" ::"n"(0));
        __syncthreads();
    
        buffer_offset_load_idx = (buffer_offset_load_idx + 1) % K_STAGE;//到这里需要第0轮S_A和S_B的偏移已经结束了
        buffer_offset_load = buffer_offset_load_idx * BUFFER_OFFSET;
    }

    // 最后一轮数据计算
    S_A = &smem[0] + buffer_offset_store;
    S_B = &smem[0] + BM * BK + buffer_offset_store;
#pragma unroll
        for (size_t k_step = 0; k_step < CHUNK_K; ++k_step) {
            uint32_t RA[WARP_COL_TILES][4];
            uint32_t RB[WARP_ROW_TILES][2];

#pragma unroll
            for (size_t i = 0; i < WARP_COL_TILES; ++i) {
                size_t A_smem_idx = (warp_id / BLOCK_ROW_WARPS) * WARP_ROWS + i * MMA_M;

                // uint32_t A_smem_lane_addr = __cvta_generic_to_shared( &smem[(A_smem_idx + lane_id % 16)][((lane_id / 16 + k_step * MMA_K / 8 + (lane_id % 8) / 2)) % 4 * 8]);//
                uint32_t A_smem_lane_addr = __cvta_generic_to_shared(&S_A[(A_smem_idx + lane_id % 16) * BK + ((lane_id / 16 + k_step * MMA_K / 8 + (lane_id % 8) / 2)) % 4 * 8]);//                  
                asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n" \
                    : "=r"(RA[i][0]), "=r"(RA[i][1]), "=r"(RA[i][2]), "=r"(RA[i][3])                     \
                    : "r"(A_smem_lane_addr));    
                
            }

#pragma unroll
            for(int j = 0; j < WARP_ROW_TILES; j++)
            {
                // uint32_t B_smem_lane = __cvta_generic_to_shared(&S_B[tid % 16][0]);//感觉这个有些问题, 16 -> 31好像不用标记
                // uint32_t B_smem_lane = __cvta_generic_to_shared(&S_B[tid % 16 + i * MMA_K][j * MMA_N + S_BC_col_offset]);
                
                uint32_t B_smem_lane = __cvta_generic_to_shared(&S_B[(tid % 16 + k_step * MMA_K) * (BN + padding_SB) + j * MMA_N + S_BC_col_offset]);
                asm volatile("ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0, %1}, [%2];\n" \
                    : "=r"(RB[j][0]), "=r"(RB[j][1])                                   \
                    : "r"(B_smem_lane));//r 四个字节，l 八个字节    

            }

#pragma unroll
            for (size_t i = 0; i < WARP_COL_TILES; ++i) {
#pragma unroll
                for (size_t j = 0; j < WARP_ROW_TILES; ++j) {
                    size_t j_s = (i % 2) ? (WARP_ROW_TILES - j - 1) : j;

                    HMMA16816(RC[i][j_s][0], RC[i][j_s][1], RA[i][0], RA[i][1], RA[i][2], RA[i][3], RB[j_s][0],
                              RB[j_s][1], RC[i][j_s][0], RC[i][j_s][1]);
                }
            }
        }
    __syncthreads();

// 数据搬出到shared memory
#pragma unroll
    for (int i = 0; i < WARP_M / MMA_M; i++) {
#pragma unroll
        for (int j = 0; j < WARP_N / MMA_N; j++) {
            // 观察可知，需要将每行都偏移4个bank即为4个half，以8为周期
            *(uint32_t *)(&S_C[(lane_id / 4 + i * MMA_M + S_AC_row_offset) * BN + ((j * MMA_N + S_BC_col_offset + lane_id % 4 * 2) + (lane_id / 4) * 8) % BN]) = RC[i][j][0];
            *(uint32_t *)(&S_C[(lane_id / 4 + 8 + i * MMA_M + S_AC_row_offset) * BN + ((j * MMA_N + S_BC_col_offset + lane_id % 4 * 2) + (lane_id / 4) * 8) % BN]) = RC[i][j][1]; // 一样

        }
    }

    __syncthreads();

    // 从S_C搬到global memory[256, 128]
    int S_C_row = tid / 16;
    int S_C_col = (tid % 16) * 8;
#pragma unroll
    for (int S_C_row_id = S_C_row; S_C_row_id < BM; S_C_row_id += 256 / 16) {
        // //观察可知，需要将每行都偏移4个bank即为4个half，以8为周期
        (float4 &)C[(S_C_row_id + row_offset) * N + S_C_col + col_offset] = (float4 &)S_C[(S_C_row_id)*BN + (S_C_col + S_C_row_id % 8 * 8) % BN];
    }
}
double get_walltime()
{
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return (double)(tp.tv_sec + tp.tv_usec * 1e-6);
}
void matrixSerial(half *hostA, half *hostB, half *hostC, int M, int K, int N)
{
    half tmp = 0.0f;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            tmp = 0.0f;
            for (int s = 0; s < K; s++) {
                tmp += hostA[i * K + s] * hostB[s * N + j];
            }
            hostC[i * N + j] = tmp;
        }
    }
}
float compare(half *hostC, half *serialC, int M, int N)
{
    float error = 0;
    for (int i = 0; i < M * N; i++) {
        error = fmax(error,
                     fabs(__half2float(hostC[i]) - __half2float(serialC[i])));
    }
    return error;
}
// size_t initMmaBase(int BLOCK_ROWS, int BLOCK_COLS, int AB_SMEM_STRIDE, int
// C_SMEM_STRIDE) {
//     int dev_id = 0;
//     HGEMM_CHECK_CUDART_ERROR(cudaGetDevice(&dev_id));

//     cudaDeviceProp dev_prop;
//     HGEMM_CHECK_CUDART_ERROR(cudaGetDeviceProperties(&dev_prop, dev_id));

//     size_t smem_max_size =
//         std::max((BLOCK_ROWS + BLOCK_COLS) * AB_SMEM_STRIDE * sizeof(half),
//         BLOCK_ROWS * C_SMEM_STRIDE * sizeof(half));
//     HLOG("smem_max_size: %.0f KBytes (%zu Bytes)",
//     static_cast<double>(smem_max_size) / 1024, smem_max_size);

//     HGEMM_CHECK_GT(dev_prop.sharedMemPerMultiprocessor, smem_max_size);
//     HGEMM_CHECK_CUDART_ERROR(
//         cudaFuncSetAttribute(matrixKernel,
//         cudaFuncAttributeMaxDynamicSharedMemorySize, smem_max_size));

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

    // 因为还需要重排序,会将block设置为256
    const int BLOCK_DIM_x = 16;
    const int BLOCK_DIM_y = 16;

    dim3 block_dim(BLOCK_DIM_x, BLOCK_DIM_y, 1);
    // 一种调用
    int num_block_y = (M + BM - 1) / (BM);
    int num_block_x = (N + BN - 1) / (BN);
    dim3 grid_dim(num_block_x, num_block_y, 1);
    // 第二种调用
    //  const int BLOCK_STRIDE = 16;
    //  dim3 grid_dim(BLOCK_STRIDE, (M  + BM -1) / BM, (N + BN * BLOCK_STRIDE -
    //  1) / (BN * BLOCK_STRIDE));

    float ker_time = 0;
    // int buffer_num = 2;
    // shared memory单块开大些
    //  static size_t smem_max_size = initMmaBase(BM, BN, BK, BN);
    size_t smem_max_size =
        std::max((BM + BN) * (BK + padding_SB) * sizeof(half) *  K_STAGE,
                 BM * (BN) * sizeof(half));
    cudaFuncSetAttribute(matrixKernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         smem_max_size);

    matrixKernel<<<grid_dim, block_dim, smem_max_size>>>(dA, dB, dC, M, K, N);

    int repeat = 20;
    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    for (int i = 0; i < repeat; i++) {
        matrixKernel<<<grid_dim, block_dim, smem_max_size>>>(dA, dB, dC, M, K,
                                                             N);
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
    printf("kernel time: %.4f second, %.4f ms\n", ker_time / (repeat * 1000.),
           ker_time / repeat);
    printf("grid dim: %d, %d, %d\n", grid_dim.x, grid_dim.y, grid_dim.z);
    printf("block dim: %d, %d, %d\n", block_dim.x, block_dim.y, block_dim.z);
}

int main()
{
    half *hostA, *hostB, *hostC, *serialC;
    int M = 256;
    int K = 128;
    int N = 256;

    hostA = (half *)malloc(M * K * sizeof(half));
    hostB = (half *)malloc(N * K * sizeof(half));
    hostC = (half *)malloc(M * N * sizeof(half));   // GPU端
    serialC = (half *)malloc(M * N * sizeof(half)); // CPU端
    for (int i = 0; i < M * K; i++) {
        hostA[i] = i % 3;
    }
    for (int i = 0; i < N * K; i++) {
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
    //     printf("%2d: hostC=%f    serialC=%f\n", i, __half2float(hostC[i]),
    //     __half2float(serialC[i]));//累加器明明是可以float16的，为啥gpt都说是不可以的，用0-1的小数随机，确实由较大误差
    // }
    free(hostA);
    free(hostB);
    free(hostC);
    free(serialC);
    return 0;
}

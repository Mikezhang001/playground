# Task 1: CUDA Programming

High performance gemm implementation on Nvidia A100 ([internal feishu doc](https://aicarrier.feishu.cn/wiki/EvivwNtVRij2XVk0i36cBN8Bn1f)).

## 1. Target

Implement a high performance gemm (General Matrix Multiply) function with CUDA on Nvidia A100 for float32 and float16 data types.

The implementation should be able to achieve at least **90%** of the performance of cuBLAS, with the given benchmarking structure.

## 2. Quick Start with Bash Script

We provide a convenient bash script `task1.sh` that offers the same operations as the previous Makefile:

```bash
# Show all available commands
./task1.sh help

# 1) Build code with specified FLOAT type and VERSION
./task1.sh build --float f32 --ver 1

# 2) Build and run code, automatically save logs
./task1.sh run --float f16 --ver 1

# 3) Build with debug symbols (RelWithDebInfo)
./task1.sh debug --float f32 --ver 2

# 4) Profile with nsight compute, save reports
./task1.sh profile --float f16 --ver 2

# Clean build files
./task1.sh clean

# Clean log files  
./task1.sh clean-logs
```

### Key Features:
- **Automatic Logging**: Run results are saved to `logs/` directory with timestamp and version info
- **TFLOPS and Error Tracking**: Captures performance metrics and error rates automatically
- **Nsight Compute Integration**: Profile reports saved to `logs/profiles/` with timestamp
- **Version-Specific File Inclusion**: Only includes source files for the current version to avoid conflicts

## 3. Benchmark cBlas and cuBlas

Build example matmul with the following commands (`v0 -> cblas`; `v1 -> cublas`):

### Using build script directly:
```bash
# Build gemm implemented with CBLAS (CPU) under float32:
bash scripts/build-task1.sh -f32 -v0
# Build gemm implemented with CBLAS (CPU) under float16:
bash scripts/build-task1.sh -f16 -v0
# Build gemm implemented with cublas (CUDA) under float32:
bash scripts/build-task1.sh -f32 -v1
# Build gemm implemented with cublas (CUDA) under float16:
bash scripts/build-task1.sh -f16 -v1
```

### Using task1.sh script (Recommended):
```bash
# Build gemm implemented with CBLAS (CPU) under float32:
./task1.sh build --float f32 --ver 0
# Build gemm implemented with CBLAS (CPU) under float16:
./task1.sh build --float f16 --ver 0
# Build gemm implemented with cublas (CUDA) under float32:
./task1.sh build --float f32 --ver 1
# Build gemm implemented with cublas (CUDA) under float16:
./task1.sh build --float f16 --ver 1
```

For more compile options, see "[./scripts/build-task1.sh](../scripts/build-task1.sh)" or run `./task1.sh help`.

> 💡**Note**:  
> 1. Please install the following extensions in VSCode:
>    - llvm-vs-code-extensions.vscode-clangd
>    - twxs.cmake
>    - josetr.cmake-language-support-vscode
> 2. It is suggested to restart clangd server after building (to avoid some code analysis errors).  
> To restart clangd server, press `Ctrl+Shift+P` in VSCode, and select `clangd: Restart language server`.  
> ![restart-clangd](../docs/imgs/restart-clangd.png)

Run the binarys in "[./build/src](../build/src)" directory to get the benchmark results.

### Running directly:
You can set `m`, `n`, `k`, `n_warmup` and `n_test` by passing arguments to binarys built in this task. Use `-h` to print help messages:

```bash
# Run the binary but showing help messages only
./build/src/task1_float16_v0 -h
```

### Running with task1.sh script (Recommended):
```bash
# Build and run with automatic logging
./task1.sh run --float f16 --ver 0

# Run with different configurations
./task1.sh run --float f32 --ver 1
./task1.sh run --float f16 --ver 1
```

The run results will be automatically saved to `logs/` directory with timestamp and version information.

## 3. Add Your Own Implementation

Create a `.cu` file under directory "[./task-1/src](./src)" with any name you like, and implement a matmul function with macro `PLAYGROUND_MATMUL_DEC`.

For example, add the following lines in "./task-1/src/xxx/xxx/f16-v2.cu" to provide the definition for function `matmul<float16_t, 2>`:

```cpp
// @file: ./task-1/src/xxx/xxx/f16-v2.cu

#include "playground/matmul.hpp"

namespace playground {
// Implement the matmul function with DType=float16_t and Version=2
PLAYGROUND_MATMUL_DEC(float16_t, 2, A, B, C, M, N, K)
{
    // ......
}
}
```

> 💡**Note**:  
> Do not use version `0` and `1` because they are for cblas and cublas respectively.

Now you are able to build a new binary `task1_float16_v2` to with the following command:

### Using build script directly:
```bash
# Build the test binary with DType=float16 and Version=2:
bash ./scripts/build-task1.sh -v2 -f16
# Run the test binary
./build/src/task1_float16_v2
```

### Using task1.sh script (Recommended):
```bash
# Build and run with automatic logging
./task1.sh run --float f16 --ver 2
```

## 4. Profile Your Kernel with Nsight Compute

Use "[scripts/nsight-profile.sh](../scripts/nsight-profile.sh)" to profile an binary which contains **a self-defined cuda kernel**.

⚠️ **The profiled binary must be built with `RelWithDebInfo` or `RD` flag**. 

### Using build script directly:
For example, to build matmul kernel with `DType=float16`, `Version=2` and `RD` flag:

```bash
# `RD` is the same as `RelWithDebInfo`
bash ./scripts/build-task1.sh RD -f16 -v2 
```

Then you can profile the binary with `ncu` with a tool script:

```bash
bash ./scripts/nsight-profile.sh -t build/src/task1_float16_v2
```

### Using task1.sh script (Recommended):
```bash
# Build with debug symbols and profile automatically
./task1.sh profile --float f16 --ver 2
```

A `.ncu-rep` file will be generated in the `logs/profiles/` directory with timestamp. Download it to your local machine and open it with Nsight Compute GUI.

![ncu-example](../docs/imgs/ncu-example.png)

## 5. Example Target Results

### CUDA Core(FP32)
| Version | v2 | v3 | v4 | v5 | v6 | v7 | v8 | v9 | v10 | v11 |cuBLAS | Theory Peak |
| --- | --- | --- | --- | --- | --- | --- | --- |--- | --- | --- | --- | --- |  
| Average error(e-06) | 5.8  | 7.04|  8.83|  6.83|  6.78| 9.63 | 6.82 |   6.40| 7.79|  6.82| 5.94 | / |
| TFLOPS              | 0.89 | 3.04 | 3.66 |5.17 | 7.3 | 10.54 |  11.84|11.64 | 15.41 |  17.10|  18.71 |  19.5|

####  V2 这个版本非常的显而意见，与cpu做乘法的逻辑基本是一致的，记得dC = 0初始化.
####  V3 用寄存器变量来取代每次的dC[row * N + col]
####  V4 首次引入shared memory，变化split_K的维度，但感觉似乎影响并不是太大
####  V5 如果每个thread只处理自己对应位置，会发现A矩阵重复的行和B矩阵的列切有关，B也是同理，所以引入TM和TN来减少这部分的数据重复。TM = TN =4，反而慢于TM = TN =2.
####  V6 在V5的情况下, 使shared memory中元素的个数与block中实际thread数目一致，并对线程重排序，从而去除了对K切片后的TM和TN的再次循环。这个参数会更快，所以改成这个，且可以于V7形成一个比较
####  V7 在V6的情况下, 使用(float4 &)来加速dA -> S_A, dB -> S_B.  参数可以于V6对照
####  V8 对矩阵S_A进行转置，用来避免bank冲突的，运算中S_A的遍历也需要进行改变
####  V9 内积换外积（改变遍历顺序）
####  V10 内积换外积（改变遍历顺序）+ (float4 &加速S_A ->寄存器变量，float4 &加速S_B->寄存器变量)
####  V11 在V10的基础上进行一个双流水的设定，关键在于将循环id = 0的数据搬运单独拿出,循环体内进行如下循环（i - 1次前的计算，i次的数据搬运），结束前再计算一次最后的遗留

### Tensor Core(MMA)(FP16)

| Version | v0 | v1_0 | v1-3-5-sw |  v2-sw |v3-sw | v3-1sw | v4-sw| v4-1-sw|cuBLAS| Theory Peak|
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |--- |
| Average error | 0.0188 | 0.0187 | 0.0192 |0.0178 | 0.0182 | 0.0187| 0.0167| 0.0224 | 0.0264|/ |
| TFLOPS        | 9.66 | 46.81 |142.31 |138.98 | 130.84 | 164.91 | 124.08| 206.11 | 219.53|312 |

####  v0  每个block处理[16, 8]
####  v1_0 Shared memory版本
####  v1-3-5-sw 仿照样例(mma_naive.cu)改改访问方式，本质还是1-3-4-sw，随便一动优化就没了离谱(一维布局调用）
####  v2-sw  float4换async指令（没有明显速度上升，但(16,16)调用速度不会下降特别多)
####  v3-sw  double buffer预取S_A, S_B
####  v3-1sw  RA, RB开double buffer
####  v4-sw  2阶段流水线，将shared memory的流水线拿出去
####  v4-1-sw 在4-sw的基础上进行三阶段流水
> 💡**Note**:  
> Some card can reach above 250 TFLOPS using cuBLAS fp16. The target is the 90% of cuBLAS on the same card

## 6. References
See also: [feishu doc: cuda学习资料](https://aicarrier.feishu.cn/wiki/SFdnw61vHi1AfRkeJVecgMjBnrc)

### CUDA Core


- [CUDA实现矩阵乘法的性能优化](https://zhuanlan.zhihu.com/p/708583794)

##### Source code:
- [参考代码](https://github.com/xgqdut2016/cuda_code/tree/main/matrix)


### Tensor Core

#### MMA
- [关于矩阵乘加操作的四个指令（ldmatrix、mma、stmatrix、movmatrix）](https://zhuanlan.zhihu.com/p/1906775725278737888)
- [CUDA shared memory避免bank conflict的swizzling机制解析](https://zhuanlan.zhihu.com/p/4746910252) 
- [CUDA Shared Memory 在向量化指令下的访存机制](https://code.hitori.moe/post/cuda-shared-memory-access-mechanism-with-vectorized-instructions/) 
- [搞懂 CUDA Shared Memory 上的 bank conflicts 和向量化指令（LDS.128 / float4）的访存特点](https://zhuanlan.zhihu.com/p/690052715) 
- [How to understand the bank conflict of shared_mem](https://forums.developer.nvidia.com/t/how-to-understand-the-bank-conflict-of-shared-mem/260900) 
- [ldmatrix指令例子和其带来的smem bank冲突](https://zhuanlan.zhihu.com/p/697228676) 
- [Nvidia Tensor Core-CUDA HGEMM优化进阶](https://zhuanlan.zhihu.com/p/639297098) 

##### Source code:
- [参考代码](https://github.com/Bruce-Lee-LY/cuda_hgemm/blob/master/src/mma/mma_async_stage3.cu) 性能最优方案

#### WMMA
- [自己写的CUDA矩阵乘法能优化到多快？](https://www.zhihu.com/question/41060378/answer/2645323107)

##### Source code:
- [参考代码](https://github.com/nicolaswilde/cuda-tensorcore-hgemm/blob/master/gemm-tc-f16f16-128x256-public/gemm.cu)

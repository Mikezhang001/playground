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
| Version | v0 | v1 | v2 | v3 | v4 | cuBLAS | Theory Peak |
| --- | --- | --- | --- | --- | --- | --- | --- | 
| Average error | 0.0115 | 0.0115 | 0.0115 | 0.0116 | 0.0116 | / | / |
| TFLOPS | 2.41 | 3.85 | 9.24 | 15.15 | 17.16 | 18.38 | 19.5 |

### Tensor Core(FP16)

| Version | v0 | v1 | v2 |  v3 |v4 | cuBLAS | Theory Peak |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Average error | 0.0117 | 0.0117 | 0.0117 | 0.0117 | 0.0019 |0.0153 | / |
| TFLOPS | 18.09 | 53.05 |103.05 |159.35 | 213.12 |222.11 | 312 |

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

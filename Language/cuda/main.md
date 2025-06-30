# cuda

## nvcc

## 编译文件

假设你有一个名为 hello.cu 的简单 CUDA 程序，想要将其编译为名为 hello 的可执行文件，你可以使用如下命令：

```bash
nvcc -o hello hello.cu
```

## nsight-compute

###  编译

nsight-compute对编译出的文件进行profile
>ncu --set detailed -o main ./main

**权限问题**
运行ncu可能遇到月权限不足的问题将示例如下

```bash
==ERROR== ERR_NVGPUCTRPERM - The user does not have permission to access NVIDIA GPU Performance Counters on the target device 0.
For instructions on enabling permissions and to get more information see https://developer.nvidia.com/ERR_NVGPUCTRPERM
```

linux

```bash
# 创建配置文件，允许所有用户访问
echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' | sudo tee /etc/modprobe.d/nvidia-profiling.conf

# 重建initrd（某些系统需要）
# RedHat系：
sudo dracut --regenerate-all -f
# Debian系：
sudo update-initramfs -u -k all

# 重启系统使配置生效
sudo reboot
```

**代码附录**
法法试从文件

```c
#include <stdio.h>

__global__ void kernel_A(double* A, int N, int M)
{
    double d = 0.0;
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {

#pragma unroll(100)
        for (int j = 0; j < M; ++j) {
            d += A[idx];
        }

        A[idx] = d;

    }
}

__global__ void kernel_B(double* A, int N, int M)
{
    double d = 0.0;
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < N) {

#pragma unroll(100)
        for (int j = 0; j < M; ++j) {
            d += A[idx];
        }

        A[idx] = d;

    }
}

__global__ void kernel_C(double* A, const double* B, int N)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    const int stride = 16;
    int strided_idx = threadIdx.x * stride + blockIdx.x % stride + (blockIdx.x / stride) * stride * blockDim.x;

    if (strided_idx < N) {
        A[idx] = B[strided_idx] + B[strided_idx];
    }
}

int main() {

    double* A;
    double* B;

    int N = 80 * 2048 * 100;
    size_t sz = N * sizeof(double);

    cudaMalloc((void**) &A, sz);
    cudaMalloc((void**) &B, sz);


    cudaMemset(A, 0, sz);
    cudaMemset(B, 0, sz);

    int threadsPerBlock = 64;
    int numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    int M = 10000;
    kernel_A<<<numBlocks, threadsPerBlock>>>(A, N, M);

    cudaFuncSetAttribute(kernel_B, cudaFuncAttributeMaxDynamicSharedMemorySize, 48 * 1024);
    kernel_B<<<numBlocks, threadsPerBlock, 48 * 1024>>>(A, N, M);

    kernel_C<<<numBlocks, threadsPerBlock>>>(A, B, N);

    cudaDeviceSynchronize();

}

```

### Nsight Compute (ncu) 基本使用方法

1.基本命令格式

>ncu [options] &lt;application&gt; [application arguments]

2.常用选项

收集基本性能数据

```shell
# 分析所有内核，保存报告
ncu -o my_report ./my_cuda_app

# 只分析特定内核
ncu -k kernel_name -o my_report ./my_cuda_app

# 限制分析的内核数量
ncu -c 10 ./my_cuda_app  # 只分析前10个内核
```

选择分析集

```shell
# 基本分析集（默认）
ncu --set basic ./my_cuda_app

# 详细分析集
ncu --set detailed ./my_cuda_app

# 完整分析集
ncu --set full ./my_cuda_app

# 查看可用的分析集
ncu --list-sets
特定指标收集：
# 收集特定指标
ncu --metrics sm__cycles_active.avg,sm__cycles_elapsed.sum ./my_cuda_app

# 查看可用指标
ncu --query-metrics
```

3.MPI应用程序分析

```shell
# 分析所有进程
ncu --target-processes all -o report mpirun [mpi options] ./my_app

# 每个进程一个报告文件
mpirun [mpi options] ncu -o report_%q{OMPI_COMM_WORLD_RANK} ./my_app
```

4.查看报告

```shell
# 文本摘要
ncu --import report.ncu-rep

# 详细信息
ncu --print-details all --import report.ncu-rep

# 使用GUI查看（如果有Qt环境）
ncu-ui report.ncu-rep
```

## Nsight Systems 基本使用方法

1. 基本命令

```shell
# 收集系统级性能数据
nsys profile ./my_cuda_app

# 保存报告
nsys profile -o my_trace ./my_cuda_app

# 指定采样选项
nsys profile --trace=cuda,nvtx,osrt ./my_cuda_app
```

2.查看报告

```shell
# 生成报告
nsys stats my_trace.qdrep

# 使用GUI查看
nsys-ui my_trace.qdrep
```

# xmake
配置debug模式
>xmake config -m debug

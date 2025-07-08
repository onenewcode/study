#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>

const char* programSource =
"__kernel                                                         \n"
"void vecadd(__global int *A, __global int* B, __global int* C) { \n"
"    int idx = get_global_id(0);                                  \n"
"    C[idx] = A[idx] + B[idx];                                    \n"
"}                                                                \n"
;

int main() {
    // 下面的代码运行在OpenCL主机上

    // 每个数组的元素个数
    const int elements = 2048;
    // 每个数组的数据长度
    size_t datasize = sizeof(int) * elements;

    // 为主机端的输入输出数据分配内存空间
    int* A = (int*)malloc(datasize); // 输入数组
    int* B = (int*)malloc(datasize); // 输入数组
    int* C = (int*)malloc(datasize); // 输出数组

    // 初始化输入数据
    for (int i = 0; i < elements; ++i) {
        A[i] = i;
        B[i] = i;
    }

    // 例程为了简单起见忽略了错误检查，实际开发中应当在每次调用API后都检查返回值是否等于CL_SUCCESS
    cl_int status;

    // 获取第一个平台
    cl_platform_id platform;
    status = clGetPlatformIDs(1, &platform, NULL);

    // 获取第一个设备
    cl_device_id device;
    status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, NULL);

    // 创建一个上下文，并将它关联到设备
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &status);

    // 创建一个命令队列，并将它关联到设备
    cl_command_queue cmdQueue = clCreateCommandQueueWithProperties(context, device, 0, &status);

    // 创建两个输入数组和一个输出数组
    cl_mem bufA = clCreateBuffer(context, CL_MEM_READ_ONLY, datasize, NULL, &status);
    cl_mem bufB = clCreateBuffer(context, CL_MEM_READ_ONLY, datasize, NULL, &status);
    cl_mem bufC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, datasize, NULL, &status);

    // 把输入数据A和B分别写入数组对象bufA和bufB中
    status = clEnqueueWriteBuffer(cmdQueue, bufA, CL_FALSE, 0, datasize, A, 0, NULL, NULL);
    status = clEnqueueWriteBuffer(cmdQueue, bufB, CL_FALSE, 0, datasize, B, 0, NULL, NULL);

    // 使用内核源码创建程序
    cl_program program = clCreateProgramWithSource(context, 1, &programSource, NULL, &status);

    // 为设备构建(编译)程序
    status = clBuildProgram(program, 1, &device, NULL, NULL, NULL);

    // 创建内核
    cl_kernel kernel = clCreateKernel(program, "vecadd", &status);

    // 设置内核参数
    status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufA);
    status = clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufB);
    status = clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufC);

    // 定义工作项的索引空间
    // 工作组的大小不是必须的，但设置一下也无妨
    size_t indexSpaceSize[1] = { elements };
    size_t workGroupSize[1] = { 256 };

    // 执行内核
    status = clEnqueueNDRangeKernel(cmdQueue, kernel, 1, NULL, indexSpaceSize, workGroupSize, 0, NULL, NULL);

    // 把输出数组读取到主机的输出数据中
    status = clEnqueueReadBuffer(cmdQueue, bufC, CL_TRUE, 0, datasize, C, 0, NULL, NULL);

    // 释放OpenCL资源
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(cmdQueue);
    clReleaseMemObject(bufA);
    clReleaseMemObject(bufB);
    clReleaseMemObject(bufC);
    clReleaseContext(context);

    // 释放主机资源
    free(A);
    free(B);
    free(C);

    return 0;
}

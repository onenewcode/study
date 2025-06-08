# opencl

## 文档

https://registry.khronos.org/OpenCL/specs/3.0-unified/html/OpenCL_API.html

## 安装

https://github.com/KhronosGroup/OpenCL-Guide/blob/main/chapters/getting_started_linux.md

## 编译

linux
>gcc -Wall -Wextra -D CL_TARGET_OPENCL_VERSION=100 Main.c -o HelloOpenCL -lOpenCL

参数详情:

- Wall -Wextra 打开所有警告（最高敏感级别）
- D 指示预处理器创建一个带有 NAME：VALUE 的定义
  - CL_TARGET_OPENCL_VERSION 启用/禁用与定义的版本对应的 API 函数。将其设置为 100 将禁用标头中所有高于 OpenCL 1.0 的 API 函数
- I 为包含目录搜索路径设置其他路径
- Main.c 是输入源文件的名称
- o 设置输出可执行文件的名称（默认为 a.out）
- l 指示链接器链接库 OpenCL

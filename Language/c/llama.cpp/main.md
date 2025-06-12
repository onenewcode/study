# llama.cpp

## 编译

前提：

- cmake
- c编译器

> cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Debug -DLLAMA_CURL=OFF
cmake --build build --config Debug

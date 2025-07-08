# llama.cpp

## 编译

前提：

- cmake
- c编译器

> cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Debug -DLLAMA_CURL=OFF
cmake --build build --config Debug -j8

## 转换gguf

在根目录安装依赖文件
>pip install -r requirements.txt

进行转换

```shell
python convert_hf_to_gguf.py "E:\deepseek\DeepSeek-R1-Distill-Qwen-7B" --outtype f16
```

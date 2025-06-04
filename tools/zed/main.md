# 安装

下载地址：https://zed.dev/download

## linux中rust配置

zed中原生支持rust，在编写rust中会自动从github中下载`rust-analyzer`，但是在国内由于个各种限制，zed并不能在自动下载rust-analyzer，有时配置梯子也不能自动下载，会出现以下从报错

```shell
Language server error: rust-analyzer

downloading release from https://github.com/rust-lang/rust-analyzer/releases/download/2025-06-02/rust-analyzer-x86_64-unknown-linux-gnu.gz
-- stderr--

```

首先我们要从提供的网址下载对应的[rust-analyzer](https://github.com/rust-lang/rust-analyzer/releases/download)，下载完毕后，我们需要把压缩报解压到指定的目录`/home/onenewcode/.local/share/zed/languages/rust-analyzer`,其中`home`后面个填写的是用户名，需要根据自己的用户名进行修改，然后我们需要更改解压后的为文件的名称，同时赋予其可执行权限，文件名称格式为`rust-analyzer-YYYY-MM-DD`，比如我的格式为`rust-analyzer-2025-06-02`。

**注意**： 如果出现以下错误则说明程序缺少可执行权限，需要赋予用户可执行权限

```shell
Language server error: rust-analyzer

failed to spawn command. path: "/home/onenewcode/.local/share/zed/languages/rust-analyzer/rust-analyzer-2025-06-02", working directory: "/home/onenewcode/文档/rust/InfiniLM", args: []
-- stderr--

```

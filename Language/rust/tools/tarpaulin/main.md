# tarpaulin

## 安装

配置好rust之后只需要 cargo install cargo-tarpaulin 就能正常安装，但是每个系统会有自己的坑，请看下面介绍。

### Windows

Windows 上必须安装rust的msvc 的后端，安装gnu后端，不能运行tarpaulin，关于msvc工具链安装可以通过以下连接  `https://visualstudio.microsoft.com/zh-hans/downloads/`

### ubuntu(linux)

ubuntu下需要先安装 sudo apt-get install libssl-dev ，然后才能通过cargo安装

### Mac

苹果按照官方文档可以直接通过cargo安装，但是我手头没有设备没有尝试。

### 使用

直接在项目的根目录运行 cargo tarpaulin 正常输出效果如下，显示所有未完全覆盖的文件，并且显示未覆盖的行，在最后显示覆盖率

```shell
|| Uncovered Lines:
|| src\lib.rs: 148, 168
|| src\local.rs: 57, 85, 92
|| Tested/Total Lines:
|| src\flag.rs: 36/36 +0.00%
|| src\lib.rs: 39/41 +0.00%
|| src\local.rs: 34/37 +0.00%
|| src\weak.rs: 19/19 +0.00%
||
96.24% coverage, 128/133 lines covered, +0.00% change in coverage
```

**注意** tarpaulin可能存在部分match语句无法覆盖的情况，tarpaulin在windos 和ubuntu上同样的代码覆盖率测试可能略有差异

# simd

测试 使用`cargo bench`但是不能开启多个simd进行测试，否则时间将会出现混乱。同时需要开启以下特性以便于开启指定的指令集。

```toml
# Windows 专用配置
[target.x86_64-pc-windows-msvc]
# 明确禁用不兼容的指令集（避免在旧 CPU 崩溃）
# rustflags = [
#     "-C", "target-feature=+avx2,avx2,-avx512f",
#     "-C", "llvm-args=-x86-asm-syntax=intel"  # 可选：生成 Intel 风格汇编
# ]
rustflags = ["-C", "target-cpu=native"]

```

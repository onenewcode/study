# pyo3

## 安装

必要条件python安装`maturin`

## 项目配置

添加文件 pyproject.toml

```toml
[project]
name = "pyo3_example"
version = "0.1.0"
description = "A PyO3 example project"
readme = "README.md"
requires-python = ">=3.7"
classifiers = [
    "Programming Language :: Rust",
    "Programming Language :: Python :: Implementation :: CPython",
    "Programming Language :: Python :: Implementation :: PyPy",
]

# 可选作者、许可证等
authors = [{ name = "Your Name", email = "you@example.com" }]
license = { text = "MIT" }

# 如果使用 maturin 构建工具
[build-system]
requires = ["maturin>=1.0,<2.0"]
build-backend = "maturin"
```

# Rerun

## 简述

## 安装

>cargo install rerun-cli --locked

或者
>cargo binstall rerun-cli

## 开发

### demo

```rs
use rerun::{demo_util::grid, external::glam};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 创建名为 "rerun_example_minimal" 的记录流
    let rec = rerun::RecordingStreamBuilder::new("rerun_example_minimal").spawn()?;

    let points = grid(glam::Vec3::splat(-10.0), glam::Vec3::splat(5.0), 10);
    let colors = grid(glam::Vec3::ZERO, glam::Vec3::splat(255.0), 10)
        .map(|v| rerun::Color::from_rgb(v.x as u8, v.y as u8, v.z as u8));

    rec.log(
        "my_points",
        &rerun::Points3D::new(points)
            .with_colors(colors)
            .with_radii([0.5]),
    )?;

    Ok(())
}

```

### 三维向量线性插值公式

#### 给定参数

- 起点向量：$\vec{a} = (a_x, a_y, a_z)$
- 终点向量：$\vec{b} = (b_x, b_y, b_z)$
- 插值步数：$n \in \mathbb{Z}^+$（正整数）

#### 1. 插值参数计算

对于第 $i$ 个插值点（$0 \leq i < n$）：

$$
t_i = \begin{cases}
0 & \text{if } n = 1 \\
\dfrac{i}{n-1} & \text{if } n > 1
\end{cases}
$$

#### 2. 分量插值公式

$$
\begin{aligned}
x_i &= a_x + (b_x - a_x) \cdot t_i \\
y_i &= a_y + (b_y - a_y) \cdot t_i \\
z_i &= a_z + (b_z - a_z) \cdot t_i
\end{aligned}
$$

#### 3. 插值点向量

$$
\vec{p_i} = \begin{pmatrix} x_i \\ y_i \\ z_i \end{pmatrix}
$$

#### 4. 结果序列

$$
\text{Result} = \left[ \vec{p_0},\ \vec{p_1},\ \ldots,\ \vec{p_{n-1}} \right]
$$

#### 几何特性

相邻点间的欧氏距离：
$$
\Delta d = \frac{\|\vec{b} - \vec{a}\|}{n-1}
$$

其中 $\|\vec{b} - \vec{a}\|$ 是起点到终点的距离：
$$
\|\vec{b} - \vec{a}\| = \sqrt{(b_x - a_x)^2 + (b_y - a_y)^2 + (b_z - a_z)^2}
$$

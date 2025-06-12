
# Criterion

## 入门

运行命令cargo bench --bench my_benchmark

目录结构

```shell
│  Cargo.toml
│
├─benches
│      my_benchmark.rs
│
└─src
        lib.rs
        main.rs
```

因此社区 benchmark 就应运而生，其中最有名的就是 criterion.rs，它有几个重要特性:

统计分析，例如可以跟上一次运行的结果进行差异比对
图表，使用 gnuplots 展示详细的结果图表
首先，如果你需要图表，需要先安装 gnuplots，其次，我们需要引入相关的包，在 Cargo.toml 文件中新增 :

```toml
[dev-dependencies]
criterion = "0.3"

[[bench]]
name = "my_benchmark"
harness = false
```

接着，在项目中创建一个测试文件: $PROJECT/benches/my_benchmark.rs，然后加入以下内容：

```rs

use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn fibonacci(n: u64) -> u64 {
    match n {
        0 => 1,
        1 => 1,
        n => fibonacci(n-1) + fibonacci(n-2),
    }
}

fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("fib 20", |b| b.iter(|| fibonacci(black_box(20))));
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
```

最后，使用` cargo bench `运行并观察结果：

```shell
Gnuplot not found, using plotters backend
Fibonacci/10            time:   [263.10 ps 266.95 ps 270.83 ps]
                        change: [-3.0764% +0.0460% +3.0907%] (p = 0.98 > 0.05)
                        No change in performance detected.
Found 1 outliers among 100 measurements (1.00%)
  1 (1.00%) high mild
Fibonacci/20            time:   [267.64 ps 272.26 ps 277.85 ps]
                        change: [+4.0082% +6.8572% +9.9692%] (p = 0.00 < 0.05)
                        Performance has regressed.
Fibonacci/30            time:   [258.78 ps 263.52 ps 268.94 ps]
                        change: [-16.784% -10.402% -3.3094%] (p = 0.00 < 0.05)
                        Performance has improved.
Found 7 outliers among 100 measurements (7.00%)
  3 (3.00%) high mild
  4 (4.00%) high severe
```

## 概述

### Warmup

每个 Criterion.rs 基准测试都会在可配置的预热期（默认情况下为 3 秒）内自动迭代基准测试函数。对于 Rust 函数基准测试，这是为了预热处理器缓存和（如果适用）文件系统缓存。

### Collecting Samples

Criterion 通过不同的迭代次数迭代要进行基准测试的函数，以生成每次迭代所花费时间的估计值。样本数量是可配置的。它还根据预热期间每次迭代的时间打印采样过程所需时间的估计值。

### Time

```shell
time:   [2.5094 ms 2.5306 ms 2.5553 ms]
thrpt:  [391.34 MiB/s 395.17 MiB/s 398.51 MiB/s]
```

这显示了此基准测试的每次迭代测量时间的置信区间。left 和 right 值分别显示置信区间的下限和上限，而 center 值显示 Criterion.rs 对基准测试例程的每次迭代所花费的时间的最佳估计值。

置信度是可配置的。较大的置信度（例如 99%）将扩大区间，从而为用户提供有关真实斜率的信息较少。另一方面，较小的置信区间（例如 90%）会缩小区间，但用户对区间包含真实斜率的信心会降低。95% 通常是一个不错的平衡。

Criterion.rs 执行 Bootstrap 重采样以生成这些置信区间。引导样本的数量是可配置的，默认为 100,000。或者，Criterion.rs 还可以以字节数或每秒元素数为单位报告基准测试代码的吞吐量。

### Change

运行 Criterion.rs 基准时，它会将统计信息保存在 target/criterion 目录中。基准测试的后续执行将加载此数据并将其与当前样本进行比较，以显示代码更改的效果。

```shell
change: [-38.292% -37.342% -36.524%] (p = 0.00 < 0.05)
Performance has improved.
```

这显示了本次基准测试与上次基准测试之间差异的置信区间，以及测量的差异可能偶然发生的概率。如果无法读取此基准测试的保存数据，则这些行将被省略。

第二行显示快速摘要。如果有强有力的统计证据表明情况确实如此，则此行将指示性能有所improved或者regressed。它还可能指示更改在噪声阈值范围内。Criterion.rs 尝试尽可能减少噪声的影响，但基准测试环境的差异（例如，与其他进程的不同负载、内存使用等）可能会影响结果。对于高度确定性的基准测试，Criterion.rs 可能足够敏感，可以检测到这些微小的波动，因此与 +-noise_threshold 范围重叠的基准测试结果被视为噪声，并被视为无关紧要。noise 阈值是可配置的，默认为 +-2%。

### Detecting Outliers

```shell
Found 8 outliers among 100 measurements (8.00%)
  4 (4.00%) high mild
  4 (4.00%) high severe
```

Criterion.rs 尝试检测异常高或异常低的样本，并将其报告为异常值。大量异常值表明基准测试结果具有干扰性，应以适当的怀疑态度看待。在这种情况下，您可以看到有些样本花费的时间比正常情况要长得多。这可能是由于运行基准测试的计算机上的负载不可预测、线程或进程调度，或者被基准测试的代码所花费的时间不规则造成的。

为了确保可靠的结果，基准测试应在安静的计算机上运行，并且应设计为每次迭代执行大致相同的工作量。如果无法做到这一点，请考虑增加测量时间以减少异常值对结果的影响，但代价是基准测试周期更长。或者，可以延长预热期（以确保任何 JIT 编译器或类似编译器都已预热），或者可以使用其他迭代循环在每个基准测试之前执行设置，以防止这影响结果。

### Additional Statistics  其他统计数据

```shell
slope  [2.5094 ms 2.5553 ms] R^2            [0.8660614 0.8640630]
mean   [2.5142 ms 2.5557 ms] std. dev.      [62.868 us 149.50 us]
median [2.5023 ms 2.5262 ms] med. abs. dev. [40.034 us 73.259 us]
```

这将显示基于其他统计数据的其他置信区间。

Criterion.rs 执行线性回归以计算每次迭代的时间。第一行显示线性回归中斜率的置信区间，而 R^2 区域显示该置信区间的下限和上限的拟合优度值。如果 R^2 值较低，这可能表示基准测试在每次迭代中执行的工作量不同。您可能希望检查绘图输出并考虑提高基准测试例程的一致性。

第二行显示每次迭代时间的平均值和标准差的置信区间（天真地计算）。如果 std. dev. 与上面的时间值相比很大，则基准测试是嘈杂的。您可能需要更改基准测试以减少噪音。

中位数/中值绝对偏差线与平均值/标准差偏差线类似，不同之处在于它使用中位数和中位数绝对偏差。与标准差一样，如果 med. abs. dev. 很大，则表明基准测试有噪声。

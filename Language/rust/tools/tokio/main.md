# tokio

## 简介

Tokio 速度很快，构建在 Rust 编程语言之上，而 Rust 编程语言本身就是 快。这是本着 Rust 的精神完成的，目标是你不应该 能够通过手动编写等效代码来提高性能。

Tokio 是可扩展的，构建在 async/await 语言功能之上，该功能 本身是可扩展的。在处理网络时，速度是有限制的 由于延迟，您可以处理连接，因此扩展的唯一方法是 一次处理多个连接。使用 async/await 语言功能，增加并发操作的数量变得非常便宜，允许您扩展到大量并发任务。

## 快速开始

### 第一个 tokio 程序

首先，在 Cargo.toml 文件中添加 Tokio 依赖项：

```toml
[dependencies]
tokio = { version = "1", features = ["full"] }
```

然后，在 src/main.rs 中编写如下代码：

```rs
use tokio;

#[tokio::main]
async fn main() {
    println!("Hello, world!");
}
```

以上就是我们的第一个 tokio 程序，可能看以来很疑惑，为什么仅仅通过添加一行代码就能启动一个异步程序，接下来将会展示`#[tokio::main]`宏下隐藏的细节。

`#[tokio::main]`宏主要做了以下的功能。

- 启动运行时：宏会生成代码来创建并启动 Tokio 运行时（Runtime）。
进入事件循环：
启动后，运行时会进入一个事件循环，等待异步任务被唤醒或完成。这个循环是无阻塞的，并且能够高效地处理大量的并发任务。
- 调用异步函数：宏会确保你的 main 函数作为异步函数被执行，这意味着它可以使用 .await 语法来暂停和恢复执行，直到所有依赖的异步操作完成。
- 阻塞当前线程：main 函数本身是异步的，但整个程序需要有一个同步的起点。因此，#[tokio::main] 宏会阻塞主线程，直到 main 函数返回。
- 配置选项：宏还允许你通过属性指定不同的配置选项。例如，你可以选择不同的调度器类型（如多线程或单线程），或者启用或禁用各种特性。
- 错误传播：如果 main 函数返回一个结果类型（即 Result<T, E>），宏还会处理可能发生的错误，这通常意味着如果异步操作失败，程序将会以非零状态退出。
- 资源清理：当所有的异步操作都完成后，宏会确保正确地清理所有分配的资源，包括关闭运行时。

接下来展示以下我们的`#[tokio::main]` 宏展开后的代码。

```rs

fn main() {
    // 输出的异步函数
    let body = async {
        {
            println!("Hello, world!\n");
        };
    };
    {   //构建我们的异步运行时
        return tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .expect("Failed building the Runtime")
            .block_on(body); //阻塞，直到异步函数执行完成
    }
}
```

## runtime

Tokio 运行时就像是一个繁忙的机场调度中心，而异步程序则是那些需要起飞和降落的飞机。

在机场（运行时）里，调度中心负责协调所有航班（任务）的活动。它知道什么时候跑道是空闲的，可以安排一架飞机起飞或降落；它还管理着地勤服务，确保加油、乘客上下机等操作不会阻碍其他飞机的操作。同时，调度中心还处理各种意外情况，比如天气变化导致的延误，或者紧急情况下的优先降落。

异步程序就像是这些飞机。它们准备好了就请求起飞（启动一个异步任务），一旦完成某个阶段的任务（例如到达目的地），就会通知调度中心（完成一部分 Future）。如果途中遇到等待（如等待网络响应或文件 I/O 操作），它们会告诉调度中心自己正在等待，并允许其他飞机使用跑道（让出线程资源给其他任务）。

每当有飞机完成了它的飞行计划（任务完成），调度中心就会记录下来，并根据需要安排下一个航班。如果有突发状况，比如某架飞机需要立即降落（高优先级任务），调度中心也会迅速做出调整，确保一切顺利进行。

因此，Tokio 运行时就像这个高效的机场调度中心，它确保所有的异步任务（飞机）都能有序、高效地运行，最大化利用可用资源（跑道、地勤人员等），并且能够灵活应对各种不可预见的情况。通过这种方式，即使在高并发环境下，系统也能够保持流畅和平稳的运作。

所以与其他 Rust 程序不同，异步应用程序需要运行时 支持。具体而言，以下运行时服务是必需的。

可能在大多数情况下我们的程序入口是一串行的程序，只有到某一个点才会进入异步代码，这时候我们 runtime 就派上用场了，他会负责给我们的异步程序提供一个能够运行的环境。

例子：

```rs
use tokio::runtime::Builder;

fn main() {
    // 创建一个运行时构建器
    let runtime = Builder::new_multi_thread() // 或者 new_current_thread() 为单线程调度
        .worker_threads(1) // 设置工作线程的数量
        .thread_name("my-custom-thread") // 设置线程名称
        .enable_all() // 启用 I/O 和时间驱动，
        .build()
        .unwrap(); // 构建运行时

    // 使用运行时执行异步代码
    (0..100).into_iter().for_each(|i|{
        runtime.spawn(async move {
            println!("Hello, world! time: {}",i);
        });
    });
}
```

以上的的例子就是展示我们直接使用 runtime 构建我们的异步程序的例子，可能有些同学觉得有些奇怪，为什么明明看到的循环是 100 次为什么自己运行的结果却小于 100。这就涉及的异步程序的特点，在习惯同步程序中，程序都是串行的执行的，程序会逐行执行，但是在异步程序中却不是这样。我们的异步程序一般不会和同步代码在一个线程之中，这就导致了如果我们的主线程的同步代码运行的太快，就会导致我们的异步代码还来不及完成整个主线程就结束了，就如同上面的例子一样，这样该如何改进呢？答案就是我们要获取个任务的 handle，然后使用它的 await 方法来等待异步任务完成。

```rs
use tokio::runtime::Builder;

fn main() {
    // 创建一个运行时构建器
    let runtime = Builder::new_multi_thread() // 或者 new_current_thread() 为单线程调度
        .worker_threads(1) // 设置工作线程的数量
        .thread_name("my-custom-thread") // 设置线程名称
        .enable_all() // 启用 I/O 和时间驱动
        .build()
        .unwrap(); // 构建运行时
    // 使用运行时执行异步代码

    let tasks: Vec<_> = (0..100)
        .into_iter()
        .map(|i| {
            // 使用运行时执行异步代码
            runtime.spawn(async move {
                println!("Hello, world! time: {}", i);
            })
        })
        .collect();
    // 等待所有任务完成
    runtime.block_on(async {
        for task in tasks {
            // 等待每一个任务完成
            // 注意：如果你的任务中可能会有错误，这里应该处理 join 的结果
            let _ = task.await;
        }
    });
}
```

修改为以上的代码就可以正确显示我们的结果了。但是你可能觉得比较繁琐，为什么我们的 await 方法要放在 block_on 方法之中呢？这是因为 await 方法只能在异步代码块中使用，我们的 main 方法是一个同步代码块，所以必须使用 runtime 生成一个异步代码块，然后再使用 await 方法。有些同学可能觉得写循环太不优雅了和有点繁琐，当然我们也有更优雅的方式来实现。首先我们需要引入一个rust官方提供的包futures，通过`cargo add futures`来引入。然后把我们最后的一个异步代码块改成以下形式，你是不是绝的优雅了许多。

```rs
    // 等待所有任务完成
    runtime.block_on(futures::future::join_all(tasks));
```

## sync

可能会有人疑惑，在异步程序内如何进行通信呢，接下来将为你介绍几个 常用的方法。
当然，下面是每种方法的一个简单代码示例。请注意，这些例子假设你已经安装了 Tokio，并且在你的 Cargo.toml 文件中添加了必要的依赖项。

### Channels

在 Tokio 中，Channels 用于线程间或任务间的通信。以下是几种常见的 Channels 类型及其作用：

- mpsc (Multi-producer, single-consumer):
  - tokio::sync::mpsc 提供了多生产者单消费者的通道。它可以有多个发送者（producers），但只有一个接收者（consumer）。这种类型的通道适用于多个任务向同一个任务发送消息的情况。
  - 包含 unbounded 和 bounded 两种类型：
    - unbounded: 不限制通道中可以同时存在的消息数量，可能导致内存使用不受控制地增长。
    - bounded: 设置了一个容量上限，当通道满了的时候，发送操作会等待直到有足够的空间。
oneshot:
  - tokio::sync::oneshot 提供了一次性的通道，只允许发送一条消息。一旦消息被发送并接收后，该通道即关闭。这通常用于任务完成信号或者返回值传递。
- watch:
  - tokio::sync::watch 提供了一个广播式的通道，允许多个接收者监听来自一个发送者的更新。每个接收者只会看到最新的值，并且可以在接收到新值时得到通知。这非常适合用于状态广播，例如配置更新。
- broadcast:
  - tokio::sync::broadcast 提供了一个广播通道，支持多个接收者。与 watch 不同的是，所有的接收者都可以选择是否接收每条消息，并且如果它们没有及时接收，可能会错过一些消息。它有一个固定的缓冲区大小，超出的消息会被丢弃。

由于他们的用法都差不多，这里就以mpsc为例来进行说明。

```rs
use tokio::sync::mpsc;

#[tokio::main]
async fn main() {
    let (tx, mut rx) = mpsc::channel(32);

    tokio::spawn(async move {
        tx.send("hello").await.unwrap();
    });

    if let Some(message) = rx.recv().await {
        println!("Got = {}", message);
    }
}
```

这就是最简单的例子，当然其中还有很多用法的细节，就留在以后探索了。

### Shared State

在 Tokio 中，共享状态是指多个任务或线程可以同时访问和修改的状态。最常见的方式就是加锁。

```rs
use std::sync::Arc;
use tokio::sync::Mutex;

#[tokio::main]
async fn main() {
    let data = Arc::new(Mutex::new(0));
    let mut tasks = Vec::new();

    for i in 0..10 {
        let my_data = Arc::clone(&data);
        let task = tokio::spawn(async move {
            let mut data = my_data.lock().await;
            *data += i;
        });
        tasks.push(task);
    }

    for task in tasks {
        task.await.unwrap();
    }

    println!("Final count: {}", *data.lock().await);
}
```

### Synchronization Primitives (Notify)

在 Tokio 中，Notify 是一种同步原语，用于任务间的简单通知机制。它允许一个或多个任务等待某个事件的发生，而另一个任务负责通知这个事件已经发生。Notify 的设计目的是为了简化异步代码中常见的“唤醒”模式，即一个任务需要等待直到另一任务完成某个操作或者满足某个条件。

```rs
use tokio::sync::Notify;

#[tokio::main]
async fn main() {
    let notify = Notify::new();
    let notified = notify.notified();

    // Spawn a task that will notify others.
    tokio::spawn(async {
        // Do some work...
        notify.notify_one();
    });

    // Wait to be notified.
    notified.await;
    println!("Received notification!");
}
```

### Task Local Storage (TLS)

线程局部存储（Thread Local Storage, TLS）允许每个线程拥有其自己的变量副本。而在异步编程中，我们通常使用单线程或多线程的异步运行时，如 Tokio，其中多个任务可能在同一线程上复用。

```rs
use std::cell::RefCell;
use tokio::task_local;

task_local! {
    static TASK_ID: RefCell<u64>;
}

#[tokio::main]
async fn main() {
    TASK_ID.scope(RefCell::new(42), async {
        println!("Task ID is: {}", TASK_ID.with(|id| *id.borrow()));
    }).await;
}
```

### Async-aware Data Structures (Watch)

在 Tokio 中，Watch 是一个异步感知的数据结构，它用于一对多的通信模式，即一个发送者（sender）可以向多个接收者（receiver）广播更新。它的设计目的是为了简化跨任务之间的状态同步问题，特别是当这些任务可能在不同的线程上并发执行时。

```rs
use tokio::sync::watch;

#[tokio::main]
async fn main() {
    let (tx, mut rx) = watch::channel("initial");

    tokio::spawn(async move {
        // Update the value
        tx.send("updated").unwrap();
    });

    // Wait for an update
    rx.changed().await.unwrap();
    println!("Value changed to: {}", *rx.borrow());
}
```

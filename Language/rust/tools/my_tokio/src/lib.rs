use tokio::{fs, io::{self, AsyncReadExt}, runtime::Builder};


/// 异步读取目录中的所有小文件
pub async fn read_small_files(dir_path: &str) -> io::Result<Vec<(String, Vec<u8>)>> {
    let mut entries = fs::read_dir(dir_path).await?;
    let mut tasks = Vec::new();

    // 并行处理目录项
    while let Some(entry) = entries.next_entry().await? {
        let path = entry.path();

        // 只处理文件，跳过目录和符号链接等
        if !path.is_file() {
            continue;
        }

        tasks.push(async move {
            // 获取文件名
            let file_name = path.file_name()
                .and_then(|n| n.to_str())
                .map(|s| s.to_owned())
                .ok_or_else(|| io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Invalid filename"
                ))?;

            // 异步读取整个文件
            let content = fs::read(&path).await?;

            Ok::<_, io::Error>((file_name, content))
        });
    }

    // 并行执行所有任务并收集结果
    let results = futures::future::try_join_all(tasks).await?;
    Ok(results)
}
pub fn runtime_example() {
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
            let _ = task.await;
        }
    });

}
#[cfg(test)]
mod tests {
    use tokio::fs::File;

    use super::*;

    #[test]
    fn runtime_example_test() {
        runtime_example();
    }
        #[tokio::test]
    async fn test_read_small_files() {
        // 创建临时目录
        let _ = fs::read_dir("t/").await.unwrap();
    }
}

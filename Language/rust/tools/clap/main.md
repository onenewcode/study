
# clap

Clap 是一个用于命令行参数解析的 Rust 库。它提供了一种简单的方式来定义命令行参数，并使用这些参数来解析命令行输入。Clap 支持多种类型的参数，包括选项、子命令、环境变量和配置文件。
Clap 提供了多种功能，包括：

1. 命令行参数的解析：Clap 可以解析命令行参数，并自动将参数转换为指定的类型。
2. 帮助信息：Clap 可以自动生成帮助信息，包括参数的描述、默认值、示例等。

## 例子

首先我们要使用以下命令行引入依赖
>cargo add clap --features derive

然后我们开始编写第一个demo

```rs
use std::path::PathBuf;

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Cli {
    /// Optional name to operate on
    name: Option<String>,

    /// Sets a custom config file
    #[arg(short, long, value_name = "FILE")]
    config: Option<PathBuf>,
    // 使用计数动作（clap::ArgAction::Count），意味着每多指定一次 -d 或 --debug，它的值就增加 1。
    /// Turn debugging information on
    #[arg(short, long, action = clap::ArgAction::Count)]
    debug: u8,
    #[command(subcommand)]
    command: Option<Commands>,
}
 // Commands 枚举定义了子命令 Test，它本身可以接受一个布尔类型的参数 list。
#[derive(Subcommand)]
enum Commands {
    /// does testing things
    Test {
        /// lists test values
        #[arg(short, long)]
        list: bool,
    },
}

fn main() {
    let cli = Cli::parse();

    // 检查了 name 和 config 参数是否被提供
    if let Some(name) = cli.name.as_deref() {
        println!("Value for name: {name}");
    }

    if let Some(config_path) = cli.config.as_deref() {
        println!("Value for config: {}", config_path.display());
    }

    // debug 参数被指定的次数来判断调试模式的状态。
    match cli.debug {
        0 => println!("Debug mode is off"),
        1 => println!("Debug mode is kind of on"),
        2 => println!("Debug mode is on"),
        _ => println!("Don't be crazy"),
    }

    // 检查是否存在子命令 Test，并根据 list 参数的值来决定是否打印测试列表
    match &cli.command {
        Some(Commands::Test { list }) => {
            if *list {
                println!("Printing testing lists...");
            } else {
                println!("Not printing testing lists...");
            }
        }
        None => {}
    }
}
```

运行效果

```shell
./clap.exe --help
Usage: clap.exe [OPTIONS] [NAME] [COMMAND]

Commands:
  test  does testing things
  help  Print this message or the help of the given subcommand(s)

Arguments:
  [NAME]  Optional name to operate on

Options:
  -c, --config <FILE>  Sets a custom config file
  -d, --debug...       Turn debugging information on
  -h, --help           Print help
  -V, --version        Print version


./clap.exe -dd test
Debug mode is on
Not printing testing lists...
```

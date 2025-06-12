# 生成随机值

rust 中官方并没有像以他语言一样，rust 并没有官方并没有提供生成随机数的工具，所以我们要借助 rand 包进行生成随机数。这里我们使用现在使用罪为广泛的 rand 包只需要引入以下依赖就能够使用。`rand = "0.8.5"`

## 生成随机数

在随机数生成器 rand::Rng 的帮助下，通过 rand::thread_rng 生成随机数。可以开启多个线程，每个线程都有一个初始化的生成器。整数在其类型范围内均匀分布，浮点数是从 0 均匀分布到 1，但不包括 1。

```rs
use rand::Rng;

fn main() {
    let mut rng = rand::thread_rng();

    let n1: u8 = rng.gen();
    let n2: u16 = rng.gen();
    println!("Random u8: {}", n1);
    println!("Random u16: {}", n2);
    // 改变类型
    println!("Random u32: {}", rng.gen::<u32>());
    println!("Random i32: {}", rng.gen::<i32>());
    println!("Random float: {}", rng.gen::<f64>());
}
```

结果

```shell
Random u8: 247
Random u16: 46458
Random u32: 2649532043
Random i32: 1393744920
Random float: 0.5923489382636902
```

## 生成范围内随机数

使用 Rng::gen_range，在半开放的 [0, 10) 范围内（不包括 10）生成一个随机值。

```rs
use rand::Rng;

fn main() {
    let mut rng = rand::thread_rng();
    println!("Integer: {}", rng.gen_range(0..10));
    println!("Float: {}", rng.gen_range(0.0..10.0));
}
```

结果

```shell
   let mut rng = rand::thread_rng();
    println!("Integer: {}", rng.gen_range(0..10));
    println!("Float: {}", rng.gen_range(0.0..10.0));
```

使用 Uniform 模块可以得到均匀分布的值。下述代码和上述代码具有相同的效果，但在相同范围内重复生成数字时，下述代码性能可能会更好。

```rs

use rand::distributions::{Distribution, Uniform};

fn main() {
    let mut rng = rand::thread_rng();
    let die = Uniform::from(1..7);

    loop {
        let throw = die.sample(&mut rng);
        println!("Roll the die: {}", throw);
        if throw == 6 {
            break;
        }
    }
}
```

结果

```shell
Roll the die: 1
Roll the die: 2
Roll the die: 6
```

## 生成自定义类型随机值

随机生成一个元组 (i32, bool, f64) 和用户定义类型为 Point 的变量。为 Standard 实现 Distribution trait，以允许随机生成。

```rs
use rand::Rng;
use rand::distributions::{Distribution, Standard};

#[derive(Debug)]
struct Point {
    x: i32,
    y: i32,
}

impl Distribution<Point> for Standard {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Point {
        let (rand_x, rand_y) = rng.gen();
        Point {
            x: rand_x,
            y: rand_y,
        }
    }
}

fn main() {
    let mut rng = rand::thread_rng();
    let rand_tuple = rng.gen::<(i32, bool, f64)>();
    let rand_point: Point = rng.gen();
    println!("Random tuple: {:?}", rand_tuple);
    println!("Random Point: {:?}", rand_point);
}
```

结果

```shell
Random tuple: (590118681, false, 0.7548409339548463)
Random Point: Point { x: 914499268, y: 795986012 }
```

## 从一组字母数字字符创建随机密码

随机生成一个给定长度的 ASCII 字符串，范围为 A-Z，a-z，0-9，使用字母数字样本。

```rs
use rand::{thread_rng, Rng};
use rand::distributions::Alphanumeric;

fn main() {
    let rand_string: String = thread_rng()
        .sample_iter(&Alphanumeric)
        .take(30)
        .map(char::from)
        .collect();

    println!("{}", rand_string);
}
```

结果

```shell
fwaZUzdIkK1p78fyNvh44Od5gcr3BL
```

## 从一组用户定义字符创建随机密码

使用用户自定义的字节字符串，使用 gen_range 函数，随机生成一个给定长度的 ASCII 字符串。

```rs
use rand::Rng;
fn main() {

    const CHARSET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ\
                            abcdefghijklmnopqrstuvwxyz\
                            0123456789)(*&^%$#@!~";
    const PASSWORD_LEN: usize = 30;
    let mut rng = rand::thread_rng();

    let password: String = (0..PASSWORD_LEN)
        .map(|_| {
            let idx = rng.gen_range(0..CHARSET.len());
            CHARSET[idx] as char
        })
        .collect();

    println!("{:?}", password);
}
```

结果

```shell
"F@QNgOrsviJ2tqM$zOSJSR^Hjevvce"
```

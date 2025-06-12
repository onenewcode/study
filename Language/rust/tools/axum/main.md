
# axum

## 简介

Axum 是一个用于构建 Web 应用程序的 Rust 框架，它建立在 Tokio 和 Tower 之上。Axum 的设计目标是提供一种简单且灵活的方式来创建异步 Web 服务，同时保持高性能和安全性。
**特点**：

- 异步优先：Axum 是为异步编程而设计的，充分利用了 Rust 的 async/await 语法，使得编写非阻塞的 Web 服务变得简单。
- 基于 HTTP 规范：Axum 使用 http crate 来处理请求和响应，这确保了框架与标准 HTTP 协议的良好兼容性。
- 中间件支持：通过使用 Tower 提供的功能，Axum 支持中间件来处理跨多个路由的通用逻辑，如日志记录、身份验证等。
- 路由：Axum 提供了一个直观的 API 来定义路由，可以轻松地将 URL 路径映射到相应的处理函数。
- 提取器：这是 Axum 中的一个重要概念，允许从请求中提取数据（例如路径参数、查询参数、请求体等）并直接作为处理器函数的参数传递。
- 类型安全：由于 Rust 的强类型系统，Axum 的 API 设计使得许多错误可以在编译时被捕捉，从而提高了代码的安全性和可靠性。
- 社区和生态系统：作为一个活跃维护的项目，Axum 有一个不断增长的社区和丰富的插件生态系统。
- 性能：得益于 Rust 的零成本抽象和其他优化特性，Axum 可以实现非常高的性能。
- 灵活性：虽然 Axum 提供了很多开箱即用的功能，但它也允许开发者根据需要进行定制和扩展。

## 快速开始

以下是一个简单的 Axum 应用程序示例，它定义了一个根路由和一个创建用户的路由：

```rs
use axum::{
    routing::{get, post},
    http::StatusCode,
    Json, Router,
};
// 引入 serde 库中的 Deserialize 和 Serialize 特性，用于数据结构的序列化和反序列化。
use serde::{Deserialize, Serialize};

// 使用 tokio 作为异步运行时，并定义异步 main 函数。
#[tokio::main]
async fn main() {
    // 初始化日志记录格式器，以便在程序中输出跟踪信息。
    tracing_subscriber::fmt::init();

    // 创建一个新的Router实例，并定义两个路由：
    // 1. 对根路径"/"的GET请求，调用`root`处理函数；
    // 2. 对"/users"路径的POST请求，调用`create_user`处理函数。
    let app = Router::new()
        .route("/", get(root))
        .route("/users", post(create_user));

    // 绑定到TCP地址0.0.0.0:3000并开始监听传入连接。
    // 然后使用`axum::serve`来服务HTTP请求。
    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

// 定义一个异步函数`root`，它响应根路径的 GET 请求，
// 并返回一个静态字符串"Hello, World!"。
async fn root() -> &'static str {
    "Hello, World!"
}

// 定义一个异步函数`create_user`，它接受一个 JSON 格式的`CreateUser`对象作为输入，
// 并返回一个包含 HTTP 状态码和 JSON 格式的`User`对象的元组。
async fn create_user(
    // 解构出从请求体中提取的 JSON 数据，并将其转换为`CreateUser`类型。
    Json(payload): Json<CreateUser>,
) -> (StatusCode, Json<User>) {
    // 创建一个新的`User`实例，给定固定的 ID `1337`和从请求体中提取的用户名。
    let user = User {
        id: 1337,
        username: payload.username,
    };

    // 返回HTTP状态码201 Created和序列化后的`User`对象作为响应。
    (StatusCode::CREATED, Json(user))
}

// 定义了一个`CreateUser`结构体，表示创建用户的请求体，
// 其中只包含一个字段`username`，并且实现了`Deserialize`特性，
// 这使得它可以被自动地从 JSON 格式的数据中解析出来。
#[derive(Deserialize)]
struct CreateUser {
    username: String,
}

// 定义了一个`User`结构体，表示用户信息，
// 包含一个无符号 64 位整数类型的`id`和一个字符串类型的`username`，
// 并且实现了`Serialize`特性，这使得它可以被自动地序列化为 JSON 格式的数据。
#[derive(Serialize)]
struct User {
    id: u64,
    username: String,
}
```

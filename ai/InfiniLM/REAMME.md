# InfiniLM

## 整体结构

主要看llama.cu的结构。首先看`lib.rs`其中`Service`结构体是项目服务的核心，提供从gguf文件加载模型（new）和通过`Service`中的`Option<(Receiver<Output>, std::thread::JoinHandle<()>)>`管理管道和线程的销毁时机。然后通过`exec\engine.rs`中的`engine`方法启动推理引擎。然后通过构建`Worker`结构体，通过`lead`方法启动推理。

模型结构在`nn`中定义的，其中`model/llama.rs`定义了如何加载张量。同时`exec\model.rs`中的`launch`定义了模型是如何启动的。

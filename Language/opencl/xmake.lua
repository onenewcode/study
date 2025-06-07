-- 设置项目信息
set_project("OpenCLTest")
set_version("1.0")

-- 设置构建模式为 debug 或 release
set_configvar("CL_TARGET_OPENCL_VERSION", 300) -- 指定 OpenCL 目标版本为 3.0

-- 添加平台相关配置
if is_plat("linux") then
    add_packages("opencl")
elseif is_plat("windows") then
    -- 如果你在 Windows 上使用 Intel、NVIDIA 或 AMD 的 SDK，请确保已安装并设置好环境变量
    add_links("OpenCL")
    add_linkdirs("C:/Program Files/NVIDIA Corporation/OpenCL/lib/Win64") -- 示例路径，请根据你的系统修改
end

-- 默认目标
target("opencl_app")
    set_kind("binary") -- 构建为可执行文件
    add_files("src/*.c") -- 替换为你的源代码目录，如 main.c 所在位置
    add_includedirs("/usr/include/CL", {public = true}) -- 包含 OpenCL 头文件目录（Linux）
    add_links("OpenCL") -- 确保链接 OpenCL 库

    -- 自动检测并链接 OpenCL 库
    if is_plat("linux") then
        add_packages("opencl")
    elseif is_plat("windows") then
        add_links("OpenCL")
    end

-- 设置项目信息
set_project("OpenCLTest")
set_version("1.0")

-- 设置构建模式为 debug 或 release
set_configvar("CL_TARGET_OPENCL_VERSION", 300) -- 指定 OpenCL 目标版本为 3.0

-- 默认目标
target("opencl_app")
    set_kind("binary") -- 构建为可执行文件
    add_files("src/*.c")

    -- 平台相关设置
    if is_plat("linux") then
        add_includedirs("/usr/include/CL", {public = true})
        add_links("OpenCL")
        add_packages("opencl")
    elseif is_plat("windows") then
        -- 建议用户根据实际安装路径修改
        add_includedirs(os.getenv("OPENCL_INC") or "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.9\\include", {public = true})
        add_linkdirs(os.getenv("OPENCL_LIB") or "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.9\\lib\\x64")
        add_links("OpenCL")
    end

    -- 调试信息
    if is_mode("debug") then
        set_symbols("debug")
    end

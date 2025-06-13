add_rules("mode.debug", "mode.release")

target("test")
    set_kind("binary")
    set_languages("c++17")
    add_files("src/test.cu")
    add_cugencodes("native")
    add_defines("CUDA")

function(ime_core_add_mlx)
    include(FetchContent)
    set(MLX_BUILD_TESTS OFF CACHE BOOL "Build MLX upstream tests" FORCE)
    set(MLX_BUILD_EXAMPLES OFF CACHE BOOL "Build MLX examples" FORCE)
    set(MLX_BUILD_PYTHON_BINDINGS OFF CACHE BOOL "Build MLX Python bindings" FORCE)
    set(MLX_BUILD_GGUF ON CACHE BOOL "Read existing IME GGUF models" FORCE)
    set(MLX_METAL_JIT OFF CACHE BOOL "Use precompiled MLX kernels for IME cold starts" FORCE)
    set(MLX_METAL_PATH "${CMAKE_BINARY_DIR}/ime-core-metal" CACHE PATH "MLX kernel output" FORCE)
    file(MAKE_DIRECTORY "${MLX_METAL_PATH}")
    FetchContent_Declare(ime_core_mlx
        URL https://codeload.github.com/ml-explore/mlx/tar.gz/refs/tags/v0.32.3
        URL_HASH SHA256=4129039ddcb36cb860982b616c975a5b9b8e6d8fa97efa86335b4427c29c4b4b
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
    FetchContent_MakeAvailable(ime_core_mlx)
    FetchContent_GetProperties(gguflib)
    # Correct the pinned Q4_K second-half scale indices.
    file(READ "${gguflib_SOURCE_DIR}/gguflib.c" parser_source)
    set(second_half "            /* Second set: lower bits. */\n            for (uint32_t j = 0; j < 32; j++) {")
    string(FIND "${parser_source}" "${second_half}" second_half_offset)
    if(second_half_offset LESS 0)
        message(FATAL_ERROR "Pinned MLX GGUF Q4_K decoder layout changed; revalidate the scale fix")
    endif()
    string(REPLACE "${second_half}"
        "            /* Second set: lower bits. */\n            scale = scales[b+1];\n            min = mins[b+1];\n            for (uint32_t j = 0; j < 32; j++) {"
        parser_source "${parser_source}")
    set(patched_parser "${CMAKE_CURRENT_BINARY_DIR}/ime-core-gguflib.c")
    file(WRITE "${patched_parser}" "${parser_source}")
    set_property(TARGET gguflib PROPERTY SOURCES "${gguflib_SOURCE_DIR}/fp16.c" "${patched_parser}")
    target_include_directories(gguflib PRIVATE "${gguflib_SOURCE_DIR}")
    # Namespace gguf-tools symbols because llama.cpp exports incompatible names.
    set(parser_functions
        gguf_append_kv gguf_append_tensor_data gguf_append_tensor_info gguf_close
        gguf_create gguf_do_with_value gguf_get_alignment_padding gguf_get_key
        gguf_get_tensor gguf_get_tensor_type_name gguf_get_value_type_name
        gguf_init gguf_open gguf_print_value gguf_remap gguf_rewind
        gguf_skip_key_values_section gguf_tensor_to_bf16 gguf_tensor_to_f16 gguf_tensor_to_float)
    foreach(name IN LISTS parser_functions)
        target_compile_definitions(mlx PRIVATE "${name}=ime_core_mlx_${name}")
        target_compile_definitions(gguflib PRIVATE "${name}=ime_core_mlx_${name}")
    endforeach()
    # Keep the private GGUF reader available to installed static consumers.
    target_sources(mlx PRIVATE "${gguflib_SOURCE_DIR}/fp16.c" "${patched_parser}")
    set_source_files_properties("${patched_parser}" TARGET_DIRECTORY mlx gguflib
        PROPERTIES COMPILE_OPTIONS "-UNDEBUG")
    set(IME_CORE_MLX_SOURCE_DIR "${ime_core_mlx_SOURCE_DIR}" PARENT_SCOPE)
    set(IME_CORE_MLX_GGUF_SOURCE_DIR "${gguflib_SOURCE_DIR}" PARENT_SCOPE)
endfunction()

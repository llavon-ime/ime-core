cmake_minimum_required(VERSION 3.24)

project(ime_core VERSION 0.1.0 LANGUAGES CXX)

include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

option(IME_CORE_BUILD_TESTS "Build ime-core tests" OFF)
option(IME_CORE_BUILD_TOOLS "Build manual inference diagnostics (requires a model)" OFF)

find_package(reflectcpp CONFIG REQUIRED)
find_package(utf8cpp CONFIG REQUIRED)
find_package(llama CONFIG REQUIRED)

set(IME_CORE_HAS_MLX OFF)
set(IME_CORE_PROVIDER_LIBRARIES llama)
set(IME_CORE_PROVIDER_SOURCES "")
set(IME_CORE_RUNTIME_FILES "")
set(IME_CORE_PROVIDER_LICENSE_FILES "")
set(IME_CORE_PROVIDER_DEPENDENCIES "find_dependency(llama CONFIG)")

if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    if(NOT CMAKE_OSX_DEPLOYMENT_TARGET)
        set(CMAKE_OSX_DEPLOYMENT_TARGET "14.0" CACHE STRING "Minimum macOS version" FORCE)
    elseif(CMAKE_OSX_DEPLOYMENT_TARGET VERSION_LESS 14.0)
        message(FATAL_ERROR "The MLX macOS backend requires macOS 14.0 or newer")
    endif()
    include(cmake/MlxDependency.cmake)
    ime_core_add_mlx()
    set(IME_CORE_HAS_MLX ON)
    list(APPEND IME_CORE_PROVIDER_LIBRARIES mlx)
    list(APPEND IME_CORE_PROVIDER_SOURCES src/engine/mlx_backend.cpp)
    set(IME_CORE_RUNTIME_FILES "${MLX_METAL_PATH}/mlx.metallib")
    configure_file("${IME_CORE_MLX_SOURCE_DIR}/LICENSE" generated/MLX-LICENSE.txt COPYONLY)
    configure_file("${IME_CORE_MLX_GGUF_SOURCE_DIR}/LICENSE" generated/MLX-gguf-tools-LICENSE.txt COPYONLY)
    set(IME_CORE_PROVIDER_LICENSE_FILES
        "${CMAKE_CURRENT_BINARY_DIR}/generated/MLX-LICENSE.txt"
        "${CMAKE_CURRENT_BINARY_DIR}/generated/MLX-gguf-tools-LICENSE.txt")
    string(APPEND IME_CORE_PROVIDER_DEPENDENCIES "\nfind_dependency(MLX CONFIG)")
endif()

set(IME_CORE_UTF8_TARGET "")
if(TARGET utf8cpp::utf8cpp)
    set(IME_CORE_UTF8_TARGET utf8cpp::utf8cpp)
elseif(TARGET utf8::cpp)
    set(IME_CORE_UTF8_TARGET utf8::cpp)
else()
    message(FATAL_ERROR "ime-core requires an utf8cpp CMake target")
endif()

add_library(ime-core STATIC
    src/core.cpp
    src/encoding_tables.cpp
    include/ime-core/core.hpp
    include/ime-core/encoding_tables.hpp
    include/ime-core/logger.hpp
    src/bopomofo/bopomofo.hpp
    src/engine/backend_factory.cpp
    src/engine/backend_factory.hpp
    src/engine/core_state.cpp
    src/engine/core_state.hpp
    src/engine/engine.hpp
    src/engine/llama_backend.hpp
    src/engine/prediction_engine.cpp
    src/engine/prediction_engine.hpp
    ${IME_CORE_PROVIDER_SOURCES}
    src/engine/tokenizer.hpp
    src/utils/core_paths.hpp
)
add_library(ime-core::ime-core ALIAS ime-core)
set_property(TARGET ime-core PROPERTY IME_CORE_RUNTIME_FILES "${IME_CORE_RUNTIME_FILES}")
set_property(TARGET ime-core PROPERTY IME_CORE_PROVIDER_LICENSE_FILES "${IME_CORE_PROVIDER_LICENSE_FILES}")

function(ime_core_install_provider_resources)
    get_target_property(runtime_files ime-core IME_CORE_RUNTIME_FILES)
    get_target_property(license_files ime-core IME_CORE_PROVIDER_LICENSE_FILES)
    if(runtime_files)
        install(FILES ${runtime_files} DESTINATION "${CMAKE_INSTALL_BINDIR}")
    endif()
    if(license_files)
        install(FILES ${license_files} DESTINATION "${CMAKE_INSTALL_DATADIR}/licenses/ime-core")
    endif()
endfunction()
ime_core_install_provider_resources()

target_compile_features(ime-core PUBLIC cxx_std_23)
set_target_properties(ime-core PROPERTIES
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
    EXPORT_NAME ime-core
)

target_include_directories(ime-core
    PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
        "${CMAKE_CURRENT_BINARY_DIR}/generated"
)

target_link_libraries(ime-core
    PRIVATE
        ${IME_CORE_PROVIDER_LIBRARIES}
        reflectcpp::reflectcpp
        ${IME_CORE_UTF8_TARGET}
)
target_compile_definitions(ime-core PRIVATE IME_CORE_HAS_MLX=$<BOOL:${IME_CORE_HAS_MLX}>)

if(MSVC)
    target_compile_options(ime-core PRIVATE /W4 /WX /utf-8)
else()
    target_compile_options(ime-core PRIVATE -Wall -Wextra -Wpedantic
        -Wconversion -Wsign-conversion -Wshadow -Wformat=2 -Wundef
        -Wcast-align -Wswitch-enum -Wnon-virtual-dtor -Woverloaded-virtual)
endif()

install(TARGETS ime-core
    EXPORT ime-core-targets
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
)
install(DIRECTORY include/ DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
install(DIRECTORY table/ DESTINATION "${CMAKE_INSTALL_DATADIR}/ime-core/tables")
install(FILES LICENSE DESTINATION "${CMAKE_INSTALL_DATADIR}/licenses/ime-core")

configure_package_config_file(
    cmake/ime-core-config.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/ime-core-config.cmake"
    INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/ime-core"
)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/ime-core-config-version.cmake"
    VERSION "${PROJECT_VERSION}"
    COMPATIBILITY SameMajorVersion
)

install(EXPORT ime-core-targets
    FILE ime-core-targets.cmake
    NAMESPACE ime-core::
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/ime-core"
)
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/ime-core-config.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/ime-core-config-version.cmake"
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/ime-core"
)

if(IME_CORE_BUILD_TESTS)
    enable_testing()
    add_executable(ime-core-api-tests tests/core_api_tests.cpp)
    target_compile_features(ime-core-api-tests PRIVATE cxx_std_23)
    set_target_properties(ime-core-api-tests PROPERTIES CXX_EXTENSIONS NO)
    target_compile_options(ime-core-api-tests PRIVATE "$<TARGET_PROPERTY:ime-core,COMPILE_OPTIONS>")
    target_link_libraries(ime-core-api-tests PRIVATE ime-core::ime-core)
    add_test(NAME ime-core-api-tests COMMAND ime-core-api-tests)

    add_executable(ime-core-tokenizer-tests tests/tokenizer_tests.cpp)
    target_compile_features(ime-core-tokenizer-tests PRIVATE cxx_std_23)
    set_target_properties(ime-core-tokenizer-tests PROPERTIES CXX_EXTENSIONS NO)
    target_include_directories(ime-core-tokenizer-tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_compile_definitions(ime-core-tokenizer-tests PRIVATE
        IME_CORE_TEST_TABLES_DIR="${CMAKE_CURRENT_SOURCE_DIR}/table")
    target_link_libraries(ime-core-tokenizer-tests PRIVATE
        ime-core::ime-core reflectcpp::reflectcpp ${IME_CORE_UTF8_TARGET})
    if(MSVC)
        target_compile_options(ime-core-tokenizer-tests PRIVATE /W4 /WX /utf-8)
    else()
        target_compile_options(ime-core-tokenizer-tests PRIVATE
            -Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion -Wshadow
            -Wformat=2 -Wundef -Wcast-align -Wswitch-enum
            -Wnon-virtual-dtor -Woverloaded-virtual)
    endif()
    add_test(NAME ime-core-tokenizer-tests COMMAND ime-core-tokenizer-tests)

    add_executable(ime-core-prediction-tests tests/prediction_engine_tests.cpp)
    target_compile_features(ime-core-prediction-tests PRIVATE cxx_std_23)
    set_target_properties(ime-core-prediction-tests PROPERTIES CXX_EXTENSIONS NO)
    target_include_directories(ime-core-prediction-tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_compile_definitions(ime-core-prediction-tests PRIVATE
        IME_CORE_TEST_TABLES_DIR="${CMAKE_CURRENT_SOURCE_DIR}/table")
    target_link_libraries(ime-core-prediction-tests PRIVATE
        ime-core::ime-core reflectcpp::reflectcpp ${IME_CORE_UTF8_TARGET})
    target_compile_options(ime-core-prediction-tests PRIVATE
        "$<TARGET_PROPERTY:ime-core,COMPILE_OPTIONS>")
    add_test(NAME ime-core-prediction-tests COMMAND ime-core-prediction-tests)

    set(IME_CORE_MLX_TEST_ROOT "" CACHE PATH "Namespaced MLX SDK from MlxDependency.cmake for cross-backend tests")
    if(IME_CORE_MLX_TEST_ROOT AND NOT TARGET mlx)
        find_package(MLX 0.32.3 CONFIG REQUIRED PATHS "${IME_CORE_MLX_TEST_ROOT}" NO_DEFAULT_PATH)
    endif()
    if(TARGET mlx)
        add_executable(ime-core-mlx-backend-tests tests/mlx_backend_tests.cpp)
        if(NOT IME_CORE_HAS_MLX)
            target_sources(ime-core-mlx-backend-tests PRIVATE src/engine/mlx_backend.cpp)
        endif()
        target_compile_features(ime-core-mlx-backend-tests PRIVATE cxx_std_23)
        set_target_properties(ime-core-mlx-backend-tests PROPERTIES CXX_EXTENSIONS NO)
        target_include_directories(ime-core-mlx-backend-tests PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/src" "${CMAKE_CURRENT_BINARY_DIR}/generated")
        target_compile_definitions(ime-core-mlx-backend-tests PRIVATE
            IME_CORE_TEST_TABLES_DIR="${CMAKE_CURRENT_SOURCE_DIR}/table")
        target_compile_options(ime-core-mlx-backend-tests PRIVATE "$<TARGET_PROPERTY:ime-core,COMPILE_OPTIONS>")
        target_link_libraries(ime-core-mlx-backend-tests PRIVATE
            ime-core::ime-core mlx reflectcpp::reflectcpp ${IME_CORE_UTF8_TARGET})
        set(mlx_test_args "")
        if(IME_CORE_HAS_MLX)
            set(mlx_test_args --require-metal)
        endif()
        add_test(NAME ime-core-mlx-backend-tests COMMAND ime-core-mlx-backend-tests ${mlx_test_args})
        set_tests_properties(ime-core-mlx-backend-tests PROPERTIES TIMEOUT 60)
        set(IME_CORE_TEST_MODEL "" CACHE FILEPATH "GGUF model for maintained native backend parity regression")
        if(IME_CORE_TEST_MODEL)
            add_test(NAME ime-core-mlx-model-tests COMMAND ime-core-mlx-backend-tests
                ${mlx_test_args} --model "${IME_CORE_TEST_MODEL}")
            set_tests_properties(ime-core-mlx-model-tests PROPERTIES TIMEOUT 120)
        endif()
    endif()
endif()

if(IME_CORE_BUILD_TOOLS)
    add_executable(ime-core-latency-check tools/latency_check.cpp)
    target_compile_features(ime-core-latency-check PRIVATE cxx_std_23)
    set_target_properties(ime-core-latency-check PROPERTIES CXX_EXTENSIONS NO)
    target_compile_options(ime-core-latency-check PRIVATE "$<TARGET_PROPERTY:ime-core,COMPILE_OPTIONS>")
    target_link_libraries(ime-core-latency-check PRIVATE ime-core::ime-core)
endif()

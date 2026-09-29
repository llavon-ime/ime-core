if(NOT WIN32 OR NOT MSVC)
    message(FATAL_ERROR "The optional Ryzen AI backend requires Windows and MSVC")
endif()
find_package(microsoft.windows.ai.machinelearning CONFIG REQUIRED)
find_package(onnxruntime-genai-winml CONFIG REQUIRED)
target_sources(ime-core PRIVATE
    src/backends/ryzen_ai/onnx.cpp)
target_compile_definitions(ime-core PRIVATE IME_CORE_RYZENAI NOMINMAX WIN32_LEAN_AND_MEAN)
target_compile_options(ime-core PRIVATE /Zc:__cplusplus)
target_link_libraries(ime-core PRIVATE
    WindowsML::Api
    WindowsML::OnnxRuntime
    ONNXRuntimeGenAI::ONNXRuntimeGenAI)

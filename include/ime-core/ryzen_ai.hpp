#pragma once

#include <ime-core/core.hpp>
#include <ime-core/logger.hpp>

#include <cstdint>
#include <filesystem>
#include <memory>

namespace llavon::ime::core {

struct RyzenAiConfig {
    // ONNX Runtime GenAI model directory containing genai_config.json.
    std::filesystem::path model_directory;
    // Device-specific EPContext artifacts are written here after the first
    // successful VitisAI compilation.
    std::filesystem::path cache_directory;
    // A host that compiles separately can require an existing EPContext.
    // Missing or unusable caches then fail instead of compiling in this call.
    bool allow_compilation = true;
    // Explicit diagnostic sink supplied by the host. The backend never writes
    // to process standard streams.
    std::shared_ptr<Logger> logger;
};

// Windows ML acquires the certified VitisAI execution provider. Model
// preparation fails if any decoder node would fall back to the CPU.
std::shared_ptr<InferenceAccelerator> create_ryzen_ai_accelerator(RyzenAiConfig config);

} // namespace llavon::ime::core

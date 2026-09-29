#include <ime-core/ryzen_ai.hpp>

#include <stdexcept>
#include <utility>

#ifdef IME_CORE_RYZENAI
#include "onnx.hpp"
#endif

namespace llavon::ime::core {

std::shared_ptr<InferenceAccelerator> create_ryzen_ai_accelerator(RyzenAiConfig config) {
#ifdef IME_CORE_RYZENAI
    if (!config.model_directory.is_absolute() ||
        !std::filesystem::is_directory(config.model_directory) ||
        !std::filesystem::is_regular_file(config.model_directory / "genai_config.json")) {
        throw std::invalid_argument(
            "Ryzen AI model_directory must be an existing absolute ONNX GenAI model directory");
    }
    if (!config.cache_directory.is_absolute()) {
        throw std::invalid_argument("Ryzen AI cache_directory must be an absolute path");
    }
    if (!config.logger) {
        throw std::invalid_argument("Ryzen AI logger dependency is required");
    }
    return ryzen_ai::create_onnx_accelerator(std::move(config));
#else
    (void)config;
    throw std::runtime_error("ime-core was built without Ryzen AI support (IME_CORE_ENABLE_RYZENAI=OFF)");
#endif
}

} // namespace llavon::ime::core

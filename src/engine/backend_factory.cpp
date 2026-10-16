#include "backend_factory.hpp"

#include "llama_backend.hpp"

#if IME_CORE_HAS_MLX
#include "mlx_backend.hpp"
#endif

#include <stdexcept>
#include <utility>

namespace llavon::ime::core {

InferenceProvider default_inference_provider() noexcept {
#if IME_CORE_HAS_MLX
    return InferenceProvider::mlx;
#else
    return InferenceProvider::llama_cpp;
#endif
}

bool inference_provider_available(InferenceProvider provider) noexcept {
    switch (provider) {
        case InferenceProvider::llama_cpp:
            return true;
        case InferenceProvider::mlx:
#if IME_CORE_HAS_MLX
            return true;
#else
            return false;
#endif
    }
    return false;
}

std::vector<InferenceProvider> available_inference_providers() {
#if IME_CORE_HAS_MLX
    return {InferenceProvider::mlx, InferenceProvider::llama_cpp};
#else
    return {InferenceProvider::llama_cpp};
#endif
}

namespace internal {

std::shared_ptr<BackendModel> create_backend_model(
    InferenceProvider provider,
    std::shared_ptr<const CorePaths> paths) {
    switch (provider) {
        case InferenceProvider::llama_cpp:
            return std::make_shared<LlamaModel>(std::move(paths));
        case InferenceProvider::mlx:
#if IME_CORE_HAS_MLX
            return std::make_shared<MlxModel>(std::move(paths));
#else
            throw std::invalid_argument("MLX inference provider is unavailable in this build");
#endif
    }
    throw std::invalid_argument("unknown inference provider");
}

std::vector<InferenceDeviceInfo> enumerate_backend_devices(InferenceProvider provider) {
    switch (provider) {
        case InferenceProvider::llama_cpp:
            return LlamaModel::enumerate_devices();
        case InferenceProvider::mlx:
#if IME_CORE_HAS_MLX
            return MlxModel::enumerate_devices();
#else
            throw std::invalid_argument("MLX inference provider is unavailable in this build");
#endif
    }
    throw std::invalid_argument("unknown inference provider");
}

} // namespace internal
} // namespace llavon::ime::core

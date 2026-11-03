#include "core_state.hpp"

#include "../bopomofo/bopomofo.hpp"
#include "../utils/core_paths.hpp"
#include "backend_factory.hpp"
#include "tokenizer.hpp"

#include <ime-core/logger.hpp>

#include <cstdint>
#include <future>
#include <stdexcept>
#include <utility>

namespace llavon::ime::core::internal {

CoreState::CoreState(CoreConfig config) : logger_(std::move(config.logger)) {
    if (!logger_) {
        throw std::invalid_argument("ime-core logger dependency is required");
    }
    const auto paths = std::make_shared<CorePaths>(
        config.model_path,
        config.tables_dir,
        config.vulkan_pipeline_cache_dir,
        config.context_length,
        config.threads,
        config.gpu_layers,
        config.inference_device,
        config.sessions_per_context);
    // Table loading is independent of backend initialization and model preparation.
    // Return owned results so a failure in either branch cannot leave background
    // work accessing a partially constructed CoreState.
    auto tables_loading = std::async(std::launch::async, [paths] {
        return std::pair{
            std::make_shared<Tokenizer>(*paths),
            std::make_shared<HanziMapEngine>(*paths)};
    });
    model_ = create_backend_model(config.inference_provider, paths);
    auto [tokenizer, hanzi_map] = tables_loading.get();
    tokenizer_ = std::move(tokenizer);
    hanzi_map_ = std::move(hanzi_map);
    if (!tokenizer_->tokens_fit_vocabulary(
            static_cast<std::int64_t>(model_->vocabulary_size()))) {
        throw std::runtime_error("IME token tables exceed model vocabulary");
    }
}

CoreState::~CoreState() = default;

PredictionEngine CoreState::create_engine() const {
    return PredictionEngine(
        model_->create_session(), tokenizer_, hanzi_map_, logger_);
}

const InferenceRuntimeInfo& CoreState::runtime_info() const noexcept {
    return model_->runtime_info();
}

} // namespace llavon::ime::core::internal

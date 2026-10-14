#pragma once

#include <ime-core/core.hpp>
#include <ime-core/logger.hpp>

#include <memory>
#include <stdexcept>
#include <utility>

#include "prediction_engine.hpp"

namespace llavon::ime::core::internal {

template<BackendPolicy Backend>
class CoreState {
public:
    explicit CoreState(CoreConfig config) : logger_(std::move(config.logger)) {
        if (!logger_) throw std::invalid_argument("ime-core logger dependency is required");
        const auto paths = std::make_shared<CorePaths>(config.model_path, config.tables_dir,
            config.vulkan_pipeline_cache_dir, config.context_length, config.threads,
            config.gpu_layers, config.inference_device);
        tokenizer_ = std::make_shared<Tokenizer>(*paths);
        hanzi_map_ = std::make_shared<HanziMapEngine>(*paths);
        model_ = std::make_shared<typename Backend::Model>(paths);
        if (!tokenizer_->tokens_fit_vocabulary(static_cast<std::int64_t>(model_->vocabulary_size())))
            throw std::runtime_error("IME token tables exceed model vocabulary");
    }

    PredictionEngine<Backend> create_engine() const {
        return PredictionEngine<Backend>(model_, tokenizer_, hanzi_map_, logger_);
    }
    const InferenceRuntimeInfo& runtime_info() const noexcept { return model_->runtime_info(); }

private:
    std::shared_ptr<Logger> logger_;
    std::shared_ptr<typename Backend::Model> model_;
    std::shared_ptr<const Tokenizer> tokenizer_;
    std::shared_ptr<const HanziMapEngine> hanzi_map_;
};

template<BackendPolicy Backend>
struct SessionState {
    explicit SessionState(const CoreState<Backend>& core) : engine(core.create_engine()) {}
    PredictionEngine<Backend> engine;
};

} // namespace llavon::ime::core::internal

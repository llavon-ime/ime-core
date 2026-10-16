#pragma once

#include <memory>

#include <ime-core/core.hpp>

#include "prediction_engine.hpp"

namespace llavon::ime::core::internal {

class BackendModel;
class HanziMapEngine;
class Tokenizer;

class CoreState final {
public:
    explicit CoreState(CoreConfig config);
    ~CoreState();

    CoreState(const CoreState&) = delete;
    CoreState& operator=(const CoreState&) = delete;

    PredictionEngine create_engine() const;
    const InferenceRuntimeInfo& runtime_info() const noexcept;

private:
    std::shared_ptr<Logger> logger_;
    std::shared_ptr<BackendModel> model_;
    std::shared_ptr<const Tokenizer> tokenizer_;
    std::shared_ptr<const HanziMapEngine> hanzi_map_;
};

} // namespace llavon::ime::core::internal

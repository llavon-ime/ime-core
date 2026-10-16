#pragma once

#include <chrono>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "engine.hpp"

namespace llavon::ime::core::internal {

class HanziMapEngine;
class Tokenizer;
struct PaddingEntry;

class PredictionEngine final {
public:
    PredictionEngine(
        std::unique_ptr<BackendSession> backend,
        std::shared_ptr<const Tokenizer> tokenizer,
        std::shared_ptr<const HanziMapEngine> hanzi_map,
        std::shared_ptr<Logger> logger);
    ~PredictionEngine();

    PredictionEngine(PredictionEngine&&) noexcept;
    PredictionEngine& operator=(PredictionEngine&&) noexcept;
    PredictionEngine(const PredictionEngine&) = delete;
    PredictionEngine& operator=(const PredictionEngine&) = delete;

    void ready();
    std::vector<PredictResult> predict(
        const std::u16string& context,
        const std::vector<PaddingEntry>& padding);

private:
    using Clock = std::chrono::steady_clock;

    void align_cache(const std::vector<int>& prompt);
    void decode(std::span<const int> tokens);
    std::vector<std::pair<int, float>> masked_predict(
        std::span<const int> candidates) const;

    std::unique_ptr<BackendSession> backend_;
    std::shared_ptr<const Tokenizer> tokenizer_;
    std::shared_ptr<const HanziMapEngine> hanzi_map_;
    std::shared_ptr<Logger> logger_;
    std::vector<int> cached_;
    Clock::time_point last_touch_ = Clock::time_point::min();
};

} // namespace llavon::ime::core::internal

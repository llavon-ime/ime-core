#pragma once

#include <chrono>
#include <cstddef>
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
    void prefill(const std::u16string& context);
    std::vector<PredictResult> predict(
        const std::u16string& context,
        const std::vector<PaddingEntry>& padding);

private:
    using Clock = std::chrono::steady_clock;

    std::vector<int> window_prompt() const;
    void reset() noexcept;
    std::size_t select_window(std::span<const int> context) const;
    std::size_t boundary_cut(std::span<const int> context) const;
    void align_cache(const std::vector<int>& prompt);
    void decode(std::span<const int> tokens);
    std::vector<std::pair<int, float>> masked_predict(
        std::span<const int> candidates) const;

    std::unique_ptr<BackendSession> backend_;
    std::shared_ptr<const Tokenizer> tokenizer_;
    std::shared_ptr<const HanziMapEngine> hanzi_map_;
    std::shared_ptr<Logger> logger_;
    std::vector<int> cached_;
    std::vector<int> window_;
    Clock::time_point last_touch_ = Clock::time_point::min();
};

} // namespace llavon::ime::core::internal

#pragma once

#include <ime-core/logger.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <format>
#include <map>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "../bopomofo/bopomofo.hpp"
#include "engine.hpp"
#include "tokenizer.hpp"

namespace llavon::ime::core::internal {

template<BackendPolicy Backend>
class PredictionEngine {
public:
    PredictionEngine(std::shared_ptr<typename Backend::Model> model,
                     std::shared_ptr<const Tokenizer> tokenizer,
                     std::shared_ptr<const HanziMapEngine> hanzi_map,
                     std::shared_ptr<Logger> logger)
        : backend_(std::move(model)), tokenizer_(std::move(tokenizer)),
          hanzi_map_(std::move(hanzi_map)), logger_(std::move(logger)) {
        if (!tokenizer_ || !hanzi_map_ || !logger_)
            throw std::invalid_argument("prediction engine dependencies are required");
    }

    void ready() {
        const auto now = Clock::now();
        if (last_touch_ != Clock::time_point::min() && now - last_touch_ < std::chrono::milliseconds(500)) return;
        backend_.ready(); // Dedicated scratch state; never the session KV cache.
        last_touch_ = Clock::now();
    }

    std::vector<PredictResult> predict(const std::u16string& context,
                                      const std::vector<PaddingEntry>& padding) {
        const auto start = Clock::now();
        const auto prompt = tokenizer_->tokenize(context, padding);
        logger_->log(LogInformation::context,
            [tokenizer = tokenizer_, context, prompt, count = padding.size()] {
                const auto captured = utf8::utf16to8(context);
                return std::to_string(captured.size()) + '\n' + captured +
                    utf8::utf16to8(tokenizer->detokenize_context(prompt, count));
            });

        struct Prepared {
            std::vector<int> tokens;
            std::map<int, char32_t> characters;
            int chosen = -1;
        };
        std::vector<Prepared> prepared(padding.size());
        bool needs_logits = false;
        for (std::size_t i = 0; i < padding.size(); ++i) {
            auto& entry = prepared[i];
            if (padding[i].is_chosen) {
                entry.chosen = tokenizer_->map_char(padding[i].chosen_char);
                continue;
            }
            for (const auto character : hanzi_map_->lookup_all(padding[i].bpmf)) {
                const auto token = tokenizer_->map_char(character);
                if (token < 0) continue;
                entry.tokens.push_back(token);
                entry.characters.emplace(token, character);
            }
            needs_logits = needs_logits || !entry.tokens.empty();
        }

        std::vector<PredictResult> results(padding.size());
        if (!needs_logits) return results;
        try {
            align_cache(prompt);
            std::vector<int> pending;
            pending.reserve(padding.size());
            for (std::size_t i = 0; i < padding.size(); ++i) {
                const auto& entry = prepared[i];
                if (padding[i].is_chosen) {
                    if (entry.chosen >= 0) pending.push_back(entry.chosen);
                    continue;
                }
                if (entry.tokens.empty()) continue;
                // Decode guesses/chosen characters only when another unresolved
                // position needs them. The final guess is not decoded.
                decode(pending);
                pending.clear();
                const auto probabilities = masked_predict(entry.tokens);
                for (const auto& [token, probability] : probabilities)
                    results[i].candidates.emplace_back(entry.characters.at(token), probability);
                pending.push_back(probabilities.front().first);
            }
        } catch (...) {
            // No later request may trust a partially executed suffix/cache.
            cached_.clear();
            backend_.clear();
            throw;
        }
        const auto elapsed = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
        logger_->log(LogInformation::general, std::format("[TIME] predict_ms={:.3f}", elapsed));
        return results;
    }

private:
    using Clock = std::chrono::steady_clock;
    Backend backend_;
    std::shared_ptr<const Tokenizer> tokenizer_;
    std::shared_ptr<const HanziMapEngine> hanzi_map_;
    std::shared_ptr<Logger> logger_;
    std::vector<int> cached_;
    Clock::time_point last_touch_ = Clock::time_point::min();

    void align_cache(const std::vector<int>& prompt) {
        std::size_t common = 0;
        while (common < cached_.size() && common < prompt.size() && cached_[common] == prompt[common]) ++common;
        // Trimming KV does not regenerate logits. Refresh the final retained
        // prompt token when a shorter request discards cached character guesses.
        if (common == prompt.size() && common < cached_.size() && common > 0) --common;
        if (common < cached_.size()) {
            if (!backend_.rewind(common)) {
                backend_.clear();
                common = 0;
            }
            cached_.resize(common);
        }
        decode(std::span(prompt).subspan(common));
    }

    void decode(std::span<const int> tokens) {
        if (tokens.empty()) return;
        backend_.decode(tokens, cached_.size());
        cached_.insert(cached_.end(), tokens.begin(), tokens.end());
        last_touch_ = Clock::now();
    }

    std::vector<std::pair<int, float>> masked_predict(std::span<const int> candidates) const {
        const auto logits = backend_.logits();
        float maximum = -INFINITY;
        for (const int token : candidates) {
            if (token < 0 || static_cast<std::size_t>(token) >= logits.size())
                throw std::runtime_error("candidate token exceeds model vocabulary");
            const auto value = logits[static_cast<std::size_t>(token)];
            if (!std::isfinite(value)) throw std::runtime_error("model returned non-finite logits");
            maximum = std::max(maximum, value);
        }
        std::vector<std::pair<int, float>> probabilities;
        probabilities.reserve(candidates.size());
        float sum = 0;
        for (const int token : candidates) {
            const auto probability = std::exp(logits[static_cast<std::size_t>(token)] - maximum);
            probabilities.emplace_back(token, probability);
            sum += probability;
        }
        for (auto& [token, probability] : probabilities) {
            (void)token;
            probability /= sum;
        }
        std::sort(probabilities.begin(), probabilities.end(), [](const auto& a, const auto& b) {
            return a.second > b.second;
        });
        return probabilities;
    }
};

} // namespace llavon::ime::core::internal

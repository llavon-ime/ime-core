#include "prediction_engine.hpp"

#include "../bopomofo/bopomofo.hpp"
#include "tokenizer.hpp"

#include <ime-core/logger.hpp>

#include <algorithm>
#include <cmath>
#include <format>
#include <map>
#include <stdexcept>
#include <utility>

namespace llavon::ime::core::internal {

namespace {
constexpr std::size_t refresh_window_length = 256;
constexpr std::size_t maximum_window_length = 320;
constexpr std::size_t retained_window_length = 128;
constexpr std::size_t minimum_window_length = 32;
} // namespace

PredictionEngine::PredictionEngine(
    std::unique_ptr<BackendSession> backend,
    std::shared_ptr<const Tokenizer> tokenizer,
    std::shared_ptr<const HanziMapEngine> hanzi_map,
    std::shared_ptr<Logger> logger)
    : backend_(std::move(backend)),
      tokenizer_(std::move(tokenizer)),
      hanzi_map_(std::move(hanzi_map)),
      logger_(std::move(logger)) {
    if (!backend_ || !tokenizer_ || !hanzi_map_ || !logger_) {
        throw std::invalid_argument("prediction engine dependencies are required");
    }
}

PredictionEngine::~PredictionEngine() = default;
PredictionEngine::PredictionEngine(PredictionEngine&&) noexcept = default;
PredictionEngine& PredictionEngine::operator=(PredictionEngine&&) noexcept = default;

void PredictionEngine::ready() {
    const auto now = Clock::now();
    if (last_touch_ != Clock::time_point::min() &&
        now - last_touch_ < std::chrono::milliseconds(500)) {
        return;
    }
    backend_->ready();
    last_touch_ = Clock::now();
}

void PredictionEngine::prefill(const std::u16string& context) {
    const auto start = Clock::now();
    const auto context_tokens = tokenizer_->tokenize_context(context);
    auto window_start = select_window(context_tokens);
    if (context_tokens.size() - window_start >= refresh_window_length) {
        window_start += boundary_cut(std::span(context_tokens).subspan(window_start));
    }
    window_.assign(
        context_tokens.begin() + static_cast<std::ptrdiff_t>(window_start),
        context_tokens.end());
    const auto prompt = window_prompt();
    if (prompt.size() <= cached_.size() &&
        std::equal(prompt.begin(), prompt.end(), cached_.begin())) {
        return;
    }
    try {
        align_cache(prompt);
    } catch (...) {
        reset();
        throw;
    }

    const auto elapsed = std::chrono::duration<double, std::milli>(
        Clock::now() - start).count();
    logger_->log(
        LogInformation::general,
        std::format("[TIME] prefill_ms={:.3f} window={}", elapsed, window_.size()));
}

std::vector<PredictResult> PredictionEngine::predict(
    const std::u16string& context,
    const std::vector<PaddingEntry>& padding) {
    const auto start = Clock::now();
    const auto context_tokens = tokenizer_->tokenize_context(context);
    window_.assign(
        context_tokens.begin() + static_cast<std::ptrdiff_t>(select_window(context_tokens)),
        context_tokens.end());
    auto prompt = window_prompt();
    tokenizer_->append_padding(prompt, padding);
    logger_->log(
        LogInformation::context,
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
    std::size_t last_prediction = 0;
    for (std::size_t index = 0; index < padding.size(); ++index) {
        auto& entry = prepared[index];
        if (padding[index].is_chosen) {
            entry.chosen = tokenizer_->map_char(padding[index].chosen_char);
            continue;
        }
        for (const auto character : hanzi_map_->lookup_all(padding[index].bpmf)) {
            const auto token = tokenizer_->map_char(character);
            if (token < 0) {
                continue;
            }
            entry.tokens.push_back(token);
            entry.characters.emplace(token, character);
        }
        if (!entry.tokens.empty()) {
            needs_logits = true;
            last_prediction = index;
        }
    }

    std::vector<PredictResult> results(padding.size());
    if (!needs_logits) {
        draft_.clear();
        return results;
    }

    // Only feed a contiguous, valid draft. The last prediction's token is
    // never needed as input, matching the ordinary AR cache layout.
    const auto speculation = [&](std::size_t first) {
        std::vector<int> tokens;
        tokens.reserve(last_prediction - std::min(first, last_prediction));
        bool has_draft = false;
        for (std::size_t index = first; index < last_prediction; ++index) {
            const auto& entry = prepared[index];
            if (padding[index].is_chosen) {
                if (entry.chosen >= 0) tokens.push_back(entry.chosen);
            } else if (!entry.tokens.empty()) {
                if (index >= draft_.size() ||
                    std::ranges::find(entry.tokens, draft_[index]) == entry.tokens.end()) {
                    break;
                }
                tokens.push_back(draft_[index]);
                has_draft = true;
            }
        }
        if (!has_draft) tokens.clear();
        return tokens;
    };

    std::vector<int> next_draft(padding.size(), -1);
    std::size_t drafted = 0;
    std::size_t accepted = 0;
    std::size_t verify_batches = 0;
    std::size_t ar_predictions = 0;
    try {
        auto continuation = speculation(0);
        align_cache(prompt, continuation);
        bool verifying = !continuation.empty();
        if (verifying) {
            drafted += continuation.size();
            ++verify_batches;
        }
        std::size_t base = prompt.size();
        std::size_t row = 0;
        std::vector<int> pending;
        pending.reserve(padding.size());
        for (std::size_t index = 0; index < padding.size(); ++index) {
            const auto& entry = prepared[index];
            if (padding[index].is_chosen) {
                next_draft[index] = entry.chosen;
                if (entry.chosen >= 0) {
                    if (verifying) {
                        ++row;
                        ++accepted;
                    } else {
                        pending.push_back(entry.chosen);
                    }
                }
                continue;
            }
            if (entry.tokens.empty()) {
                continue;
            }
            if (!verifying) {
                continuation = pending.empty() ? std::vector<int>{} : speculation(index);
                if (continuation.empty()) {
                    decode(pending);
                    ++ar_predictions;
                } else {
                    base = cached_.size() + pending.size();
                    decode_draft(pending, continuation);
                    drafted += continuation.size();
                    ++verify_batches;
                    verifying = true;
                    row = 0;
                }
                pending.clear();
            }
            const auto probabilities = masked_predict(
                entry.tokens, verifying ? backend_->logits_at(row) : backend_->logits());
            for (const auto& [token, probability] : probabilities) {
                results[index].candidates.emplace_back(
                    entry.characters.at(token), probability);
            }
            const auto prediction = probabilities.front().first;
            next_draft[index] = prediction;
            if (verifying && row < continuation.size() && prediction == continuation[row]) {
                ++row;
                ++accepted;
            } else {
                if (verifying && row < continuation.size()) {
                    // The current row has the correct prefix. Discard the
                    // wrong draft token and every KV entry depending on it.
                    rewind_cache(base + row);
                }
                verifying = false;
                pending.push_back(prediction);
            }
        }
        draft_ = std::move(next_draft);
    } catch (...) {
        reset();
        throw;
    }

    const auto elapsed = std::chrono::duration<double, std::milli>(
        Clock::now() - start).count();
    logger_->log(
        LogInformation::general,
        std::format("[TIME] predict_ms={:.3f} draft_tokens={} accepted_tokens={} verify_batches={} ar_predictions={}",
            elapsed, drafted, accepted, verify_batches, ar_predictions));
    return results;
}

std::vector<int> PredictionEngine::window_prompt() const {
    std::vector<int> prompt{tokenizer_->bos_token_id()};
    prompt.insert(prompt.end(), window_.begin(), window_.end());
    return prompt;
}

void PredictionEngine::reset() noexcept {
    cached_.clear();
    window_.clear();
    draft_.clear();
    backend_->clear();
}

std::size_t PredictionEngine::select_window(std::span<const int> context) const {
    std::size_t best_start = 0;
    std::size_t best_length = 0;
    for (std::size_t start = 0; start < context.size() && best_length < window_.size(); ++start) {
        const auto length = static_cast<std::size_t>(
            std::ranges::mismatch(context.subspan(start), window_).in2 - window_.begin());
        if (length >= best_length) {
            best_start = start;
            best_length = length;
        }
    }
    const bool reusable = best_length >= std::min(window_.size(), minimum_window_length);
    if (reusable && context.size() - best_start <= maximum_window_length) {
        return best_start;
    }
    return context.size() > refresh_window_length ? boundary_cut(context) : 0;
}

std::size_t PredictionEngine::boundary_cut(std::span<const int> context) const {
    const auto first = context.size() - std::min(context.size(), retained_window_length);
    const auto last = context.size() - std::min(context.size(), minimum_window_length);
    const auto candidates = context.subspan(first, last - first);
    const auto boundary = std::ranges::find_if(
        candidates, [this](int token) { return tokenizer_->is_boundary(token); });
    if (boundary == candidates.end()) {
        return first;
    }
    return first + static_cast<std::size_t>(boundary - candidates.begin()) + 1;
}

void PredictionEngine::align_cache(const std::vector<int>& prompt, std::span<const int> draft) {
    std::size_t common = 0;
    while (common < cached_.size() && common < prompt.size() &&
           cached_[common] == prompt[common]) {
        ++common;
    }
    if (common == prompt.size() && (common < cached_.size() || !draft.empty()) && common > 0) {
        --common;
    }
    if (common < cached_.size()) {
        if (!backend_->rewind(common)) {
            backend_->clear();
            common = 0;
        }
        cached_.resize(common);
    }
    if (draft.empty()) {
        decode(std::span(prompt).subspan(common));
        return;
    }
    // Recompute SEP even for an unchanged prompt. Earlier prompt rows need no
    // output head; requested row zero must predict the first output token.
    decode_draft(std::span(prompt).subspan(common), draft);
}

void PredictionEngine::decode_draft(std::span<const int> prefix, std::span<const int> draft) {
    std::vector<int> feed(prefix.begin(), prefix.end());
    feed.insert(feed.end(), draft.begin(), draft.end());
    backend_->decode_with_logits(feed, cached_.size(), prefix.size() - 1);
    cached_.insert(cached_.end(), feed.begin(), feed.end());
    last_touch_ = Clock::now();
}

void PredictionEngine::rewind_cache(std::size_t position) {
    if (backend_->rewind(position)) {
        cached_.resize(position);
        return;
    }
    // Backends may reject a partial rewind. Rebuild only the accepted prefix.
    const std::vector<int> prefix(cached_.begin(), cached_.begin() + static_cast<std::ptrdiff_t>(position));
    backend_->clear();
    cached_.clear();
    decode(prefix);
}

void PredictionEngine::decode(std::span<const int> tokens) {
    if (tokens.empty()) {
        return;
    }
    backend_->decode(tokens, cached_.size());
    cached_.insert(cached_.end(), tokens.begin(), tokens.end());
    last_touch_ = Clock::now();
}

std::vector<std::pair<int, float>> PredictionEngine::masked_predict(
    std::span<const int> candidates, std::span<const float> logits) {
    float maximum = -INFINITY;
    for (const int token : candidates) {
        if (token < 0 || static_cast<std::size_t>(token) >= logits.size()) {
            throw std::runtime_error("candidate token exceeds model vocabulary");
        }
        const auto value = logits[static_cast<std::size_t>(token)];
        if (!std::isfinite(value)) {
            throw std::runtime_error("model returned non-finite logits");
        }
        maximum = std::max(maximum, value);
    }

    std::vector<std::pair<int, float>> probabilities;
    probabilities.reserve(candidates.size());
    float sum = 0;
    for (const int token : candidates) {
        const auto probability = std::exp(
            logits[static_cast<std::size_t>(token)] - maximum);
        probabilities.emplace_back(token, probability);
        sum += probability;
    }
    for (auto& [token, probability] : probabilities) {
        static_cast<void>(token);
        probability /= sum;
    }
    std::sort(
        probabilities.begin(),
        probabilities.end(),
        [](const auto& left, const auto& right) {
            if (left.second != right.second) return left.second > right.second;
            return left.first < right.first;
        });
    return probabilities;
}

} // namespace llavon::ime::core::internal

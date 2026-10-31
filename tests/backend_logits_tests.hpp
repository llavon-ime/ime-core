#pragma once

#include "engine/engine.hpp"

#include <algorithm>
#include <cmath>
#include <format>
#include <span>
#include <stdexcept>
#include <vector>

namespace llavon::ime::core::internal::tests {

inline void verify_batch_logits(BackendModel& model, float tolerance) {
    const auto require_equal = [tolerance](std::span<const float> actual, std::span<const float> expected) {
        if (actual.size() != expected.size() || actual.empty()) {
            throw std::runtime_error("incorrect batch logits extent");
        }
        float maximum_error = 0;
        for (std::size_t index = 0; index < actual.size(); ++index) {
            if (!std::isfinite(actual[index]) || !std::isfinite(expected[index])) {
                throw std::runtime_error("backend returned non-finite logits");
            }
            maximum_error = std::max(maximum_error, std::abs(actual[index] - expected[index]));
        }
        if (maximum_error > tolerance) {
            throw std::runtime_error(std::format(
                "batch logits differ from sequential decode: max_error={} tolerance={}", maximum_error, tolerance));
        }
    };
    auto batch = model.create_session();
    auto sequential = model.create_session();
    auto other = model.create_session();
    const std::vector<int> prompt{1, 300, 301, 3, 500, 501};
    batch->decode_with_logits(prompt, 0, 2);
    // Another session may reuse the underlying model/context before rows are read.
    other->decode(std::span(prompt).first(2), 0);
    for (std::size_t index = 0; index < prompt.size(); ++index) {
        sequential->decode(std::span(prompt).subspan(index, 1), index);
        if (index >= 2) require_equal(batch->logits_at(index - 2), sequential->logits());
    }
    require_equal(batch->logits(), batch->logits_at(3));
    bool invalid_row = false;
    try { (void)batch->logits_at(4); }
    catch (const std::out_of_range&) { invalid_row = true; }
    if (!invalid_row) throw std::runtime_error("batch accepted an invalid logits row");

    // A rejected draft's suffix must be replaceable without contaminating KV.
    if (!batch->rewind(3) || !sequential->rewind(3)) {
        throw std::runtime_error("backend rejected a valid cache rewind");
    }
    const std::vector<int> correction{502, 503};
    batch->decode_with_logits(correction, 3, 0);
    for (std::size_t index = 0; index < correction.size(); ++index) {
        sequential->decode(std::span(correction).subspan(index, 1), 3 + index);
        require_equal(batch->logits_at(index), sequential->logits());
    }
    batch->clear();
    bool cleared = false;
    try { (void)batch->logits_at(0); }
    catch (const std::out_of_range&) { cleared = true; }
    if (!cleared) throw std::runtime_error("clear retained verification logits");
    bool invalid_extent = false;
    try { batch->decode_with_logits(prompt, 0, prompt.size()); }
    catch (const std::invalid_argument&) { invalid_extent = true; }
    if (!invalid_extent) throw std::runtime_error("batch accepted an empty logits suffix");
    batch->decode(prompt, 0);
    other->clear();
    other->decode_with_logits(prompt, 0, 0);
    require_equal(batch->logits(), other->logits_at(prompt.size() - 1));
}

} // namespace llavon::ime::core::internal::tests

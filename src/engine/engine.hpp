#pragma once

#include <cstddef>
#include <memory>
#include <span>
#include <vector>

#include <ime-core/core.hpp>

namespace llavon::ime::core::internal {

struct PredictResult {
    std::vector<std::pair<char32_t, float>> candidates;
};

class BackendSession {
public:
    virtual ~BackendSession() = default;

    virtual void ready() = 0;
    virtual void clear() noexcept = 0;
    virtual bool rewind(std::size_t position) = 0;
    virtual void decode(std::span<const int> tokens, std::size_t position) = 0;
    virtual std::span<const float> logits() const = 0;
};

class BackendModel : public std::enable_shared_from_this<BackendModel> {
public:
    virtual ~BackendModel() = default;

    virtual const InferenceRuntimeInfo& runtime_info() const noexcept = 0;
    virtual std::size_t vocabulary_size() const noexcept = 0;
    virtual std::unique_ptr<BackendSession> create_session() = 0;
};

}  // namespace llavon::ime::core::internal

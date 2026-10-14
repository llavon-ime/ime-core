#pragma once

#include <concepts>
#include <cstddef>
#include <memory>
#include <span>
#include <utility>
#include <vector>

#include "../utils/core_paths.hpp"

namespace llavon::ime::core::internal {

struct PredictResult {
    std::vector<std::pair<char32_t, float>> candidates;
};

// Backend policies supply model execution, never IME reading/candidate rules.
// There is no virtual dispatch or platform selection in the prediction loop.
template<class Backend>
concept BackendPolicy = requires(Backend& backend, const Backend& view,
                                 const typename Backend::Model& model,
                                 std::span<const int> tokens, std::size_t position) {
    requires std::constructible_from<typename Backend::Model, std::shared_ptr<const CorePaths>>;
    requires std::constructible_from<Backend, std::shared_ptr<typename Backend::Model>>;
    { Backend::Model::enumerate_devices() } -> std::same_as<std::vector<InferenceDeviceInfo>>;
    { model.runtime_info() } -> std::same_as<const InferenceRuntimeInfo&>;
    { model.vocabulary_size() } -> std::convertible_to<std::size_t>;
    { backend.ready() } -> std::same_as<void>;
    { backend.clear() } -> std::same_as<void>;
    { backend.rewind(position) } -> std::same_as<bool>;
    { backend.decode(tokens, position) } -> std::same_as<void>;
    { view.logits() } -> std::same_as<std::span<const float>>;
};

}  // namespace llavon::ime::core::internal

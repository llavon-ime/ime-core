#include <ime-core/core.hpp>

#include "backend_selection.hpp"
#include "engine/core_state.hpp"

#include <memory>
#include <stdexcept>
#include <utility>

namespace llavon::ime::core {

class Core::Impl final : public internal::CoreState<internal::SelectedBackend> {
public:
    using CoreState::CoreState;
};

class Session::Impl final : public internal::SessionState<internal::SelectedBackend> {
public:
    using SessionState::SessionState;
};

Session::Session(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {
    if (!impl_) throw std::invalid_argument("ime-core session implementation is required");
}
Session::~Session() = default;
Session::Session(Session&&) noexcept = default;
Session& Session::operator=(Session&&) noexcept = default;

void Session::ready() { impl_->engine.ready(); }

std::vector<Prediction> Session::predict(const std::u16string& context,
                                       const std::vector<PaddingEntry>& padding) {
    std::vector<internal::PaddingEntry> entries;
    entries.reserve(padding.size());
    for (const auto& entry : padding)
        entries.push_back({entry.chosen, entry.chosen_char, entry.bopomofo});
    auto predictions = impl_->engine.predict(context, entries);
    std::vector<Prediction> result;
    result.reserve(predictions.size());
    for (auto& prediction : predictions) result.push_back({std::move(prediction.candidates)});
    return result;
}

Core::Core(CoreConfig config) : impl_(std::make_unique<Impl>(std::move(config))) {}
Core::~Core() = default;
Core::Core(Core&&) noexcept = default;
Core& Core::operator=(Core&&) noexcept = default;

std::vector<InferenceDeviceInfo> enumerate_inference_devices() {
    return internal::SelectedBackend::Model::enumerate_devices();
}

std::unique_ptr<Session> Core::create_session() const {
    if (!impl_) throw std::logic_error("ime-core has been moved from");
    return std::unique_ptr<Session>(new Session(std::make_unique<Session::Impl>(*impl_)));
}

InferenceRuntimeInfo Core::inference_runtime_info() const {
    if (!impl_) throw std::logic_error("ime-core has been moved from");
    return impl_->runtime_info();
}

} // namespace llavon::ime::core

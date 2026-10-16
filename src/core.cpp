#include <ime-core/core.hpp>

#include "engine/backend_factory.hpp"
#include "engine/core_state.hpp"
#include "engine/tokenizer.hpp"

#include <memory>
#include <stdexcept>
#include <utility>

namespace llavon::ime::core {

class Core::Impl final {
public:
    explicit Impl(CoreConfig config) : state(std::move(config)) {}

    internal::CoreState state;
};

class Session::Impl final {
public:
    explicit Impl(internal::PredictionEngine value) : engine(std::move(value)) {}

    internal::PredictionEngine engine;
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

std::vector<InferenceDeviceInfo> enumerate_inference_devices(InferenceProvider provider) {
    return internal::enumerate_backend_devices(provider);
}

std::unique_ptr<Session> Core::create_session() const {
    if (!impl_) throw std::logic_error("ime-core has been moved from");
    return std::unique_ptr<Session>(
        new Session(std::make_unique<Session::Impl>(impl_->state.create_engine())));
}

InferenceRuntimeInfo Core::inference_runtime_info() const {
    if (!impl_) throw std::logic_error("ime-core has been moved from");
    return impl_->state.runtime_info();
}

} // namespace llavon::ime::core

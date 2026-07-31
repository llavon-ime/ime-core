#include <ime-core/core.hpp>

#include "engine/llama_engine.hpp"
#include "utils/core_paths.hpp"

#include <stdexcept>

namespace llavon::ime::core {

class Session::Impl final {
public:
    internal::LlamaEngine engine;
};

class Core::Impl final {
public:
    explicit Impl(CoreConfig config) : config_(std::move(config)) {
        internal::CorePaths::configure(
            config_.model_path,
            config_.tables_dir,
            config_.context_length,
            config_.threads,
            config_.gpu_layers);
        internal::ModelManager::initialize();
    }

private:
    CoreConfig config_;
};

Session::Session(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {
    if (!impl_) {
        throw std::invalid_argument("ime-core session implementation is required");
    }
}

Session::~Session() = default;
Session::Session(Session&&) noexcept = default;
Session& Session::operator=(Session&&) noexcept = default;

void Session::ready() {
    impl_->engine.ready();
}

std::vector<Prediction> Session::predict(
    const std::u16string& context,
    const std::vector<PaddingEntry>& padding) {
    std::vector<internal::PaddingEntry> core_padding;
    core_padding.reserve(padding.size());
    for (const auto& entry : padding) {
        core_padding.push_back(
            internal::PaddingEntry{entry.chosen, entry.chosen_char, entry.bopomofo});
    }

    auto core_predictions = impl_->engine.predict(context, core_padding);
    std::vector<Prediction> predictions;
    predictions.reserve(core_predictions.size());
    for (auto& prediction : core_predictions) {
        predictions.push_back(Prediction{std::move(prediction.candidates)});
    }
    return predictions;
}

Core::Core(CoreConfig config) : impl_(std::make_unique<Impl>(std::move(config))) {}
Core::~Core() = default;
Core::Core(Core&&) noexcept = default;
Core& Core::operator=(Core&&) noexcept = default;

std::unique_ptr<Session> Core::create_session() const {
    if (!impl_) {
        throw std::logic_error("ime-core has been moved from");
    }
    return std::unique_ptr<Session>(new Session(std::make_unique<Session::Impl>()));
}

}  // namespace llavon::ime::core

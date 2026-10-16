#pragma once

#include <cstddef>
#include <memory>
#include <span>
#include <vector>

#include "../utils/core_paths.hpp"
#include "engine.hpp"

namespace llavon::ime::core::internal {

class MlxModel final : public BackendModel {
public:
    explicit MlxModel(std::shared_ptr<const CorePaths> paths);
    ~MlxModel() override;
    MlxModel(const MlxModel&) = delete;
    MlxModel& operator=(const MlxModel&) = delete;
    static std::vector<InferenceDeviceInfo> enumerate_devices();
    const InferenceRuntimeInfo& runtime_info() const noexcept override;
    std::size_t vocabulary_size() const noexcept override;
    std::unique_ptr<BackendSession> create_session() override;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
    friend class MlxBackend;
};

class MlxBackend final : public BackendSession {
public:
    explicit MlxBackend(std::shared_ptr<MlxModel> model);
    ~MlxBackend() override;
    MlxBackend(MlxBackend&&) noexcept;
    MlxBackend& operator=(MlxBackend&&) noexcept;
    MlxBackend(const MlxBackend&) = delete;
    MlxBackend& operator=(const MlxBackend&) = delete;
    void ready() override;
    void clear() noexcept override;
    bool rewind(std::size_t position) override;
    void decode(std::span<const int> tokens, std::size_t position) override;
    std::span<const float> logits() const override;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace llavon::ime::core::internal

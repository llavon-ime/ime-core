#pragma once

#include <cstddef>
#include <memory>
#include <span>
#include <vector>

#include "../utils/core_paths.hpp"

namespace llavon::ime::core::internal {

class MlxModel {
public:
    explicit MlxModel(std::shared_ptr<const CorePaths> paths);
    ~MlxModel();
    MlxModel(const MlxModel&) = delete;
    MlxModel& operator=(const MlxModel&) = delete;
    static std::vector<InferenceDeviceInfo> enumerate_devices();
    const InferenceRuntimeInfo& runtime_info() const noexcept;
    std::size_t vocabulary_size() const noexcept;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
    friend class MlxBackend;
};

class MlxBackend {
public:
    using Model = MlxModel;
    explicit MlxBackend(std::shared_ptr<Model> model);
    ~MlxBackend();
    MlxBackend(MlxBackend&&) noexcept;
    MlxBackend& operator=(MlxBackend&&) noexcept;
    MlxBackend(const MlxBackend&) = delete;
    MlxBackend& operator=(const MlxBackend&) = delete;
    void ready();
    void clear() noexcept;
    bool rewind(std::size_t position);
    void decode(std::span<const int> tokens, std::size_t position);
    std::span<const float> logits() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace llavon::ime::core::internal

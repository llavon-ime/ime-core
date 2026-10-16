#pragma once

#include <memory>
#include <vector>

#include <ime-core/core.hpp>

#include "engine.hpp"

namespace llavon::ime::core::internal {

class CorePaths;

std::shared_ptr<BackendModel> create_backend_model(
    InferenceProvider provider,
    std::shared_ptr<const CorePaths> paths);

std::vector<InferenceDeviceInfo> enumerate_backend_devices(InferenceProvider provider);

} // namespace llavon::ime::core::internal

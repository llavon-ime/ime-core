#pragma once

#include <ime-core/ryzen_ai.hpp>

namespace llavon::ime::core::ryzen_ai {

std::shared_ptr<InferenceAccelerator> create_onnx_accelerator(RyzenAiConfig config);

} // namespace llavon::ime::core::ryzen_ai

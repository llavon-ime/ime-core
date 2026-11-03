#pragma once

#include <llama.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace llavon::ime::core::internal {

// Build representative graphs and prepare their pipelines on the host. No decode,
// graph input upload, tensor evaluation, or KV-cache mutation takes place here.
inline void prepare_vulkan_pipelines(llama_context* context) {
    const auto capacity = std::min({llama_n_ctx_seq(context),
                                   llama_n_batch(context), llama_n_ubatch(context)});
    if (capacity == 0) throw std::runtime_error("invalid Vulkan preparation capacity");
    const auto begin = std::chrono::steady_clock::now();
    std::vector<std::uint32_t> sizes;
    for (std::uint32_t size = 1; size <= std::min(capacity, 8u); ++size) {
        sizes.push_back(size);
    }
    for (std::uint32_t size = 16; size < capacity; size *= 2) {
        sizes.push_back(size);
        if (size > capacity / 2) break;
    }
    if (sizes.back() != capacity) sizes.push_back(capacity);

    ggml_backend_graph_prepare_result total{};
    std::uint32_t graphs = 0;
    for (const bool full_kv : {false, true}) {
        for (const auto size : sizes) {
            for (unsigned variant = 0; variant < 2; ++variant) {
                if (variant == 1 && size == 1) continue;
                const auto outputs = variant == 0 ? 1u : size;
                ggml_backend_graph_prepare_result result{};
                const auto status = llama_prepare_pipelines(
                    context, size, outputs, full_kv, true, &result);
                if (status != GGML_STATUS_SUCCESS) {
                    throw std::runtime_error("Vulkan host pipeline preparation failed");
                }
                total.cache_hits += result.cache_hits;
                total.compile_required += result.compile_required;
                total.unprobed += result.unprobed;
                ++graphs;
            }
        }
    }
    const auto elapsed = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - begin).count();
    std::clog << "[CORE] Vulkan host preparation graphs=" << graphs
              << " cache_hits=" << total.cache_hits
              << " compile_required=" << total.compile_required
              << " unprobed=" << total.unprobed << " ms=" << elapsed << '\n';
}

} // namespace llavon::ime::core::internal

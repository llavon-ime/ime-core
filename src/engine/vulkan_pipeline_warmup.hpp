#pragma once

#include <llama.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace llavon::ime::core::internal {

// ggml's Vulkan backend specializes small matrix-vector batches separately.
// A one-token decode does not prepare the 2..8-token pipelines. Larger batches
// select matrix-matrix kernels by size. Exercise these paths once at model
// load, including attention with a populated KV cache, before accepting input.
// This does not run periodically and does not change the inference batch size.
inline void prepare_vulkan_pipelines(llama_context* context, llama_token token) {
    const auto capacity = std::min(llama_n_ctx(context), llama_n_batch(context));
    if (capacity == 0) throw std::runtime_error("invalid Vulkan warmup capacity");

    std::vector<std::uint32_t> sizes;
    for (std::uint32_t size = 1; size <= std::min(capacity, 8u); ++size) {
        sizes.push_back(size);
    }
    for (std::uint32_t size = 16; size < capacity; size *= 2) {
        sizes.push_back(size);
        if (size > capacity / 2) break;
    }
    if (sizes.back() != capacity) sizes.push_back(capacity);

    struct Batch final {
        llama_batch value;
        explicit Batch(std::uint32_t count)
            : value(llama_batch_init(static_cast<std::int32_t>(count), 0, 1)) {}
        ~Batch() { llama_batch_free(value); }
        Batch(const Batch&) = delete;
        Batch& operator=(const Batch&) = delete;
    };

    const auto memory = llama_get_memory(context);
    for (const bool populated : {false, true}) {
        for (const auto size : sizes) {
            const auto position = static_cast<llama_pos>(populated ? capacity - size : 0);
            if (!populated) {
                llama_memory_clear(memory, true);
            } else if (!llama_memory_seq_rm(memory, 0, position, -1)) {
                throw std::runtime_error("failed to reset Vulkan warmup cache");
            }

            Batch batch(size);
            if (!batch.value.token || !batch.value.pos || !batch.value.n_seq_id ||
                !batch.value.seq_id || !batch.value.logits) {
                throw std::runtime_error("failed to allocate Vulkan warmup batch");
            }
            batch.value.n_tokens = static_cast<std::int32_t>(size);
            for (std::int32_t i = 0; i < batch.value.n_tokens; ++i) {
                batch.value.token[i] = token;
                batch.value.pos[i] = position + i;
                batch.value.n_seq_id[i] = 1;
                batch.value.seq_id[i][0] = 0;
                batch.value.logits[i] = i == batch.value.n_tokens - 1;
            }
            const int status = llama_decode(context, batch.value);
            llama_synchronize(context);
            if (status != 0) throw std::runtime_error("Vulkan pipeline warmup failed");
        }
    }
    llama_memory_clear(memory, true);
}

} // namespace llavon::ime::core::internal

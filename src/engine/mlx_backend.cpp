#include "mlx_backend.hpp"

#include <mlx/mlx.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <variant>

namespace llavon::ime::core::internal {
namespace {
namespace mx = mlx::core;

// A single executor boundary protects MLX's scheduler across service workers.
// Each session still owns independent KV/logit state and all weights are shared.
std::recursive_mutex& executor_mutex() {
    static std::recursive_mutex mutex;
    return mutex;
}

InferenceDeviceInfo describe_device(const mx::Device& device) {
    const bool gpu = device.type == mx::Device::gpu;
    const auto& properties = mx::device_info(device);
    std::string name = gpu ? "MLX Metal" : "MLX CPU";
    std::uint64_t memory = 0;
    if (const auto it = properties.find("device_name"); it != properties.end())
        if (const auto* value = std::get_if<std::string>(&it->second)) name = *value;
    for (const auto* key : {"total_memory", "memory_size"}) {
        const auto it = properties.find(key);
        if (it == properties.end()) continue;
        if (const auto* value = std::get_if<std::size_t>(&it->second)) memory = static_cast<std::uint64_t>(*value);
        if (memory != 0) break;
    }
    return {gpu ? InferenceBackend::metal : InferenceBackend::cpu,
        gpu ? InferenceDeviceType::integrated_gpu : InferenceDeviceType::cpu,
        gpu ? "mlx-metal-0" : "mlx-cpu-0", std::move(name), gpu ? "MLX Metal GPU" : "MLX CPU", 0, memory};
}

int integer_metadata(const std::unordered_map<std::string, mx::GGUFMetaData>& metadata,
                     const std::string& key, int fallback = 0) {
    const auto it = metadata.find(key);
    if (it == metadata.end() && fallback > 0) return fallback;
    if (it == metadata.end()) throw std::runtime_error("GGUF missing " + key);
    const auto* value = std::get_if<mx::array>(&it->second);
    if (!value || value->size() != 1) throw std::runtime_error("invalid GGUF metadata " + key);
    const auto number = mx::astype(*value, mx::int64, mx::Device{mx::Device::cpu}).item<std::int64_t>();
    if (number <= 0 || number > std::numeric_limits<int>::max())
        throw std::runtime_error("GGUF dimension out of range: " + key);
    return static_cast<int>(number);
}

float float_metadata(const std::unordered_map<std::string, mx::GGUFMetaData>& metadata,
                     const std::string& key, float fallback) {
    const auto it = metadata.find(key);
    if (it == metadata.end()) return fallback;
    const auto* value = std::get_if<mx::array>(&it->second);
    if (!value || value->size() != 1) throw std::runtime_error("invalid GGUF metadata " + key);
    const auto number = mx::astype(*value, mx::float32, mx::Device{mx::Device::cpu}).item<float>();
    if (!std::isfinite(number) || number <= 0) throw std::runtime_error("invalid GGUF value " + key);
    return number;
}

struct LayerCache {
    std::optional<mx::array> keys;
    std::optional<mx::array> values;
};
} // namespace

class MlxModel::Impl {
public:
    std::unordered_map<std::string, mx::array> weights;
    mx::Device device{mx::Device::cpu};
    mx::Stream stream{mx::default_stream(device)};
    InferenceRuntimeInfo runtime;
    int hidden = 0;
    int layers = 0;
    int heads = 0;
    int kv_heads = 0;
    int head_dim = 0;
    int rotary_dim = 0;
    int vocabulary = 0;
    int capacity = 0;
    int bos = 1;
    float epsilon = 1e-5F;
    float rope_base = 10000.0F;

    explicit Impl(const CorePaths& paths) {
        const auto requested = paths.inference_device();
        const int gpu_layers = paths.gpu_layers();
        const bool wants_gpu = requested.backend != InferenceBackend::cpu &&
            (gpu_layers == -2 || gpu_layers == -1 || gpu_layers > 0);
        const mx::Device gpu{mx::Device::gpu};
        const bool compatible = requested.backend == InferenceBackend::automatic || requested.backend == InferenceBackend::metal;
        const bool matches_id = requested.device_id.empty() || requested.device_id == "mlx-metal-0";
        const bool use_gpu = wants_gpu && compatible && matches_id && mx::is_available(gpu);
        device = use_gpu ? gpu : mx::Device{mx::Device::cpu};
        // Ordinary MLX streams belong to their creating thread. Service requests
        // can move between workers, so use a global stream behind our executor
        // lock rather than retain another worker's thread-local encoder.
        stream = mx::new_thread_unsafe_stream(device);
        runtime = {describe_device(device), use_gpu, wants_gpu && !use_gpu};
        std::clog << "[CORE] loading MLX GGUF model: " << paths.model_path() << '\n';
        auto [arrays, metadata] = mx::load_gguf(paths.model_path().string(), stream);
        const auto architecture = metadata.find("general.architecture");
        if (architecture == metadata.end() || !std::holds_alternative<std::string>(architecture->second) ||
            std::get<std::string>(architecture->second) != "llama")
            throw std::runtime_error("MLX backend currently requires a Llama GGUF model");
        hidden = integer_metadata(metadata, "llama.embedding_length");
        layers = integer_metadata(metadata, "llama.block_count");
        heads = integer_metadata(metadata, "llama.attention.head_count");
        kv_heads = integer_metadata(metadata, "llama.attention.head_count_kv", heads);
        if (hidden % heads != 0 || heads % kv_heads != 0)
            throw std::runtime_error("unsupported Llama attention dimensions");
        head_dim = hidden / heads;
        rotary_dim = integer_metadata(metadata, "llama.rope.dimension_count", head_dim);
        if (rotary_dim > head_dim || rotary_dim % 2 != 0)
            throw std::runtime_error("unsupported Llama rotary dimensions");
        epsilon = float_metadata(metadata, "llama.attention.layer_norm_rms_epsilon", 1e-5F);
        rope_base = float_metadata(metadata, "llama.rope.freq_base", 10000.0F);
        const auto native_capacity = integer_metadata(metadata, "llama.context_length");
        if (paths.context_length() > static_cast<std::uint32_t>(std::numeric_limits<int>::max()))
            throw std::invalid_argument("MLX context length exceeds supported range");
        capacity = paths.context_length() == 0 ? native_capacity : static_cast<int>(paths.context_length());
        if (metadata.contains("llama.rope.scaling.type") &&
            std::get<std::string>(metadata.at("llama.rope.scaling.type")) != "none")
            throw std::runtime_error("MLX backend does not yet support GGUF RoPE scaling");

        // GGUF's Q/K layout uses interleaved (traditional) RoPE, unlike HF
        // checkpoints. MLX reads Q4_K/Q6_K as FP16. Q4_0/Q4_1/Q8_0 arrive packed;
        // expand those explicitly too, without changing any vocabulary rows.
        for (auto& [name, weight] : arrays) {
            if (!name.ends_with(".weight")) continue;
            if (weight.dtype() == mx::uint32) {
                const auto prefix = name.substr(0, name.size() - 7);
                const auto& scales = arrays.at(prefix + ".scales");
                const auto& biases = arrays.at(prefix + ".biases");
                if (weight.ndim() != 2 || scales.ndim() != 2 || scales.shape(-1) <= 0 ||
                    weight.shape(-1) % scales.shape(-1) != 0)
                    throw std::runtime_error("unsupported packed GGUF tensor " + name);
                const int bits = weight.shape(-1) / scales.shape(-1);
                if (bits != 4 && bits != 8) throw std::runtime_error("unsupported GGUF quantization " + name);
                weight = mx::dequantize(weight, scales, biases, 32, bits, "affine", std::nullopt, mx::float16, stream);
            } else weight = mx::astype(weight, mx::float16, stream);
        }
        weights = std::move(arrays);
        const auto& embedding = weights.at("token_embd.weight");
        if (embedding.ndim() != 2 || embedding.shape(1) != hidden)
            throw std::runtime_error("invalid Llama embedding shape");
        vocabulary = embedding.shape(0);
        if (vocabulary <= 0) throw std::runtime_error("MLX model has no vocabulary");
        bos = vocabulary > 1 ? 1 : 0;
        if (const auto it = metadata.find("tokenizer.ggml.bos_token_id"); it != metadata.end()) {
            const auto* value = std::get_if<mx::array>(&it->second);
            if (!value || value->size() != 1) throw std::runtime_error("invalid GGUF BOS token metadata");
            const auto token = mx::astype(*value, mx::int64, mx::Device{mx::Device::cpu}).item<std::int64_t>();
            if (token < 0 || token >= vocabulary) throw std::runtime_error("GGUF BOS token exceeds vocabulary");
            bos = static_cast<int>(token);
        }
        if (!weights.contains("output.weight")) weights.emplace("output.weight", embedding);
        require_shape("output.weight", {vocabulary, hidden});
        require_shape("output_norm.weight", {hidden});
        for (int layer = 0; layer < layers; ++layer) {
            const auto prefix = "blk." + std::to_string(layer) + ".";
            require_shape(prefix + "attn_norm.weight", {hidden});
            require_shape(prefix + "ffn_norm.weight", {hidden});
            require_shape(prefix + "attn_q.weight", {hidden, hidden});
            require_shape(prefix + "attn_k.weight", {kv_heads * head_dim, hidden});
            require_shape(prefix + "attn_v.weight", {kv_heads * head_dim, hidden});
            require_shape(prefix + "attn_output.weight", {hidden, hidden});
            const auto& gate = weights.at(prefix + "ffn_gate.weight");
            if (gate.ndim() != 2 || gate.shape(1) != hidden || gate.shape(0) <= 0)
                throw std::runtime_error("invalid Llama feed-forward shape");
            require_shape(prefix + "ffn_up.weight", gate.shape());
            require_shape(prefix + "ffn_down.weight", {hidden, gate.shape(0)});
        }
        std::vector<mx::array> materialize;
        for (const auto& [name, weight] : weights) {
            (void)name;
            materialize.push_back(weight);
        }
        mx::eval(materialize);
        std::erase_if(weights, [](const auto& entry) {
            return entry.first.ends_with(".scales") || entry.first.ends_with(".biases");
        });
        std::clog << "[CORE] MLX model ready device=" << runtime.device.description << '\n';
    }

    void require_shape(const std::string& name, const mx::Shape& shape) const {
        if (weights.at(name).shape() != shape) throw std::runtime_error("invalid Llama weight shape: " + name);
    }
    mx::array linear(const mx::array& x, const std::string& name) const {
        return mx::matmul(x, mx::transpose(weights.at(name), stream), stream);
    }
    mx::array forward(std::span<const int> tokens, int position, std::vector<LayerCache>& cache) const {
        const int count = static_cast<int>(tokens.size());
        const mx::array indices(tokens.data(), {1, count}, mx::int32);
        auto h = mx::take(weights.at("token_embd.weight"), indices, 0, stream);
        std::optional<mx::array> mask;
        if (count > 1) {
            const auto rows = mx::reshape(mx::arange(position, position + count, stream), {count, 1}, stream);
            const auto columns = mx::reshape(mx::arange(position + count, stream), {1, position + count}, stream);
            mask = mx::greater_equal(rows, columns, stream);
        }
        for (int i = 0; i < layers; ++i) {
            auto& state = cache[static_cast<std::size_t>(i)];
            const auto prefix = "blk." + std::to_string(i) + ".";
            const auto normalized = mx::fast::rms_norm(h, weights.at(prefix + "attn_norm.weight"), epsilon, stream);
            auto q = mx::transpose(mx::reshape(linear(normalized, prefix + "attn_q.weight"),
                {1, count, heads, head_dim}, stream), {0, 2, 1, 3}, stream);
            auto k = mx::transpose(mx::reshape(linear(normalized, prefix + "attn_k.weight"),
                {1, count, kv_heads, head_dim}, stream), {0, 2, 1, 3}, stream);
            auto v = mx::transpose(mx::reshape(linear(normalized, prefix + "attn_v.weight"),
                {1, count, kv_heads, head_dim}, stream), {0, 2, 1, 3}, stream);
            q = mx::fast::rope(q, rotary_dim, true, rope_base, 1.0F, position, std::nullopt, stream);
            k = mx::fast::rope(k, rotary_dim, true, rope_base, 1.0F, position, std::nullopt, stream);
            if (state.keys) {
                k = mx::concatenate({*state.keys, k}, 2, stream);
                v = mx::concatenate({*state.values, v}, 2, stream);
            }
            state.keys = k; state.values = v;
            const auto attended = mx::fast::scaled_dot_product_attention(q, k, v,
                1.0F / std::sqrt(static_cast<float>(head_dim)), "", mask, std::nullopt, false, stream);
            const auto attention = mx::reshape(mx::transpose(attended, {0, 2, 1, 3}, stream), {1, count, hidden}, stream);
            h = mx::add(h, linear(attention, prefix + "attn_output.weight"), stream);
            const auto n = mx::fast::rms_norm(h, weights.at(prefix + "ffn_norm.weight"), epsilon, stream);
            const auto gate = linear(n, prefix + "ffn_gate.weight");
            const auto activated = mx::multiply(mx::multiply(gate, mx::sigmoid(gate, stream), stream),
                linear(n, prefix + "ffn_up.weight"), stream);
            h = mx::add(h, linear(activated, prefix + "ffn_down.weight"), stream);
        }
        // Only the final prompt position is needed for IME candidates.
        h = mx::slice(h, {0, count - 1, 0}, {1, count, hidden}, stream);
        h = mx::fast::rms_norm(h, weights.at("output_norm.weight"), epsilon, stream);
        return mx::astype(mx::reshape(linear(h, "output.weight"), {vocabulary}, stream), mx::float32, stream);
    }
};

class MlxBackend::Impl {
public:
    std::shared_ptr<MlxModel> model;
    std::vector<LayerCache> cache;
    std::vector<float> values;
    std::size_t offset = 0;
    explicit Impl(std::shared_ptr<MlxModel> value) : model(std::move(value)) {
        if (!model) throw std::invalid_argument("MLX backend requires a model");
        cache.resize(static_cast<std::size_t>(model->impl_->layers));
    }
};

MlxModel::MlxModel(std::shared_ptr<const CorePaths> paths) {
    if (!paths) throw std::invalid_argument("MLX model paths are required");
    const std::lock_guard lock(executor_mutex());
    impl_ = std::make_unique<Impl>(*paths);
}
MlxModel::~MlxModel() { const std::lock_guard lock(executor_mutex()); impl_.reset(); }
const InferenceRuntimeInfo& MlxModel::runtime_info() const noexcept { return impl_->runtime; }
std::size_t MlxModel::vocabulary_size() const noexcept { return static_cast<std::size_t>(impl_->vocabulary); }
std::vector<InferenceDeviceInfo> MlxModel::enumerate_devices() {
    const std::lock_guard lock(executor_mutex());
    std::vector<InferenceDeviceInfo> result{describe_device(mx::Device{mx::Device::cpu})};
    if (mx::is_available(mx::Device{mx::Device::gpu})) result.push_back(describe_device(mx::Device{mx::Device::gpu}));
    return result;
}

MlxBackend::MlxBackend(std::shared_ptr<Model> model) : impl_(std::make_unique<Impl>(std::move(model))) {}
MlxBackend::~MlxBackend() { const std::lock_guard lock(executor_mutex()); impl_.reset(); }
MlxBackend::MlxBackend(MlxBackend&&) noexcept = default;
MlxBackend& MlxBackend::operator=(MlxBackend&& other) noexcept {
    const std::lock_guard lock(executor_mutex());
    impl_ = std::move(other.impl_);
    return *this;
}
void MlxBackend::clear() noexcept {
    const std::lock_guard lock(executor_mutex());
    for (auto& entry : impl_->cache) { entry.keys.reset(); entry.values.reset(); }
    impl_->values.clear(); impl_->offset = 0;
}
bool MlxBackend::rewind(std::size_t position) {
    const std::lock_guard lock(executor_mutex());
    if (position > impl_->offset) return false;
    if (position == 0) { clear(); return true; }
    for (auto& entry : impl_->cache) {
        if (!entry.keys || !entry.values) return false;
        auto shape = entry.keys->shape(); shape[2] = static_cast<int>(position);
        entry.keys = mx::slice(*entry.keys, {0, 0, 0, 0}, shape, impl_->model->impl_->stream);
        entry.values = mx::slice(*entry.values, {0, 0, 0, 0}, shape, impl_->model->impl_->stream);
    }
    impl_->offset = position; impl_->values.clear();
    return true;
}
void MlxBackend::decode(std::span<const int> tokens, std::size_t position) {
    const std::lock_guard lock(executor_mutex());
    const auto& model = *impl_->model->impl_;
    const auto capacity = static_cast<std::size_t>(model.capacity);
    if (tokens.empty() || position != impl_->offset || position > capacity || tokens.size() > capacity - position)
        throw std::invalid_argument("invalid MLX decode extent");
    for (const int token : tokens)
        if (token < 0 || token >= model.vocabulary) throw std::invalid_argument("MLX token exceeds vocabulary");
    auto next = impl_->cache;
    const auto logits = model.forward(tokens, static_cast<int>(position), next);
    std::vector<mx::array> evaluate{logits};
    for (const auto& entry : next) { evaluate.push_back(*entry.keys); evaluate.push_back(*entry.values); }
    mx::eval(evaluate);
    const auto* data = logits.data<float>();
    impl_->values.assign(data, data + model.vocabulary);
    impl_->cache = std::move(next);
    impl_->offset += tokens.size();
}
std::span<const float> MlxBackend::logits() const {
    if (impl_->values.empty()) throw std::runtime_error("MLX returned no logits");
    return impl_->values;
}
void MlxBackend::ready() {
    const std::lock_guard lock(executor_mutex());
    const auto& model = *impl_->model->impl_;
    std::vector<LayerCache> scratch(static_cast<std::size_t>(model.layers));
    mx::eval(model.forward(std::span(&model.bos, 1), 0, scratch));
}

} // namespace llavon::ime::core::internal

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
#include <vector>

namespace llavon::ime::core::internal {
namespace {
namespace mx = mlx::core;

using Metadata = std::unordered_map<std::string, mx::GGUFMetaData>;
using Weights = std::unordered_map<std::string, mx::array>;
using DeviceProperties =
    std::unordered_map<std::string, std::variant<std::string, std::size_t>>;

constexpr auto kMlxMetalDeviceId = "mlx-metal-0";
constexpr auto kMlxCpuDeviceId = "mlx-cpu-0";

std::recursive_mutex& executor_mutex() {
    static std::recursive_mutex mutex;
    return mutex;
}

std::uint64_t device_memory(const DeviceProperties& properties) {
    for (const auto* key : {"total_memory", "memory_size"}) {
        const auto entry = properties.find(key);
        if (entry == properties.end()) {
            continue;
        }
        if (const auto* value = std::get_if<std::size_t>(&entry->second)) {
            if (*value != 0) {
                return static_cast<std::uint64_t>(*value);
            }
        }
    }
    return 0;
}

InferenceDeviceInfo describe_device(const mx::Device& device) {
    const bool uses_gpu = device.type == mx::Device::gpu;
    const auto& properties = mx::device_info(device);
    std::string name = uses_gpu ? "MLX Metal" : "MLX CPU";
    if (const auto entry = properties.find("device_name"); entry != properties.end()) {
        if (const auto* value = std::get_if<std::string>(&entry->second)) {
            name = *value;
        }
    }
    return {
        uses_gpu ? InferenceBackend::metal : InferenceBackend::cpu,
        uses_gpu ? InferenceDeviceType::integrated_gpu : InferenceDeviceType::cpu,
        uses_gpu ? kMlxMetalDeviceId : kMlxCpuDeviceId,
        std::move(name),
        uses_gpu ? "MLX Metal GPU" : "MLX CPU",
        0,
        device_memory(properties),
    };
}

const mx::array& metadata_array(const Metadata& metadata, const std::string& key) {
    const auto entry = metadata.find(key);
    if (entry == metadata.end()) {
        throw std::runtime_error("GGUF missing " + key);
    }
    const auto* value = std::get_if<mx::array>(&entry->second);
    if (!value || value->size() != 1) {
        throw std::runtime_error("invalid GGUF metadata " + key);
    }
    return *value;
}

int integer_metadata(
    const Metadata& metadata,
    const std::string& key,
    std::optional<int> fallback = std::nullopt) {
    if (!metadata.contains(key) && fallback) {
        return *fallback;
    }
    const auto number = mx::astype(
        metadata_array(metadata, key), mx::int64, mx::Device{mx::Device::cpu})
                            .item<std::int64_t>();
    if (number <= 0 || number > std::numeric_limits<int>::max()) {
        throw std::runtime_error("GGUF dimension out of range: " + key);
    }
    return static_cast<int>(number);
}

float float_metadata(const Metadata& metadata, const std::string& key, float fallback) {
    if (!metadata.contains(key)) {
        return fallback;
    }
    const auto number = mx::astype(
        metadata_array(metadata, key), mx::float32, mx::Device{mx::Device::cpu})
                            .item<float>();
    if (!std::isfinite(number) || number <= 0) {
        throw std::runtime_error("invalid GGUF value " + key);
    }
    return number;
}

std::string string_metadata(const Metadata& metadata, const std::string& key) {
    const auto entry = metadata.find(key);
    if (entry == metadata.end()) {
        throw std::runtime_error("GGUF missing " + key);
    }
    const auto* value = std::get_if<std::string>(&entry->second);
    if (!value) {
        throw std::runtime_error("invalid GGUF metadata " + key);
    }
    return *value;
}

struct DeviceSelection {
    mx::Device device;
    bool wants_gpu = false;
    bool uses_gpu = false;
};

DeviceSelection select_device(const CorePaths& paths) {
    const auto& requested = paths.inference_device();
    const int gpu_layers = paths.gpu_layers();
    const bool wants_gpu = requested.backend != InferenceBackend::cpu &&
        (gpu_layers == -2 || gpu_layers == -1 || gpu_layers > 0);
    const bool accepts_metal = requested.backend == InferenceBackend::automatic ||
        requested.backend == InferenceBackend::metal;
    const bool matches_device = requested.device_id.empty() ||
        requested.device_id == kMlxMetalDeviceId;
    const mx::Device gpu{mx::Device::gpu};
    const bool uses_gpu = wants_gpu && accepts_metal && matches_device &&
        mx::is_available(gpu);
    return {uses_gpu ? gpu : mx::Device{mx::Device::cpu}, wants_gpu, uses_gpu};
}

struct ModelShape {
    int hidden = 0;
    int layers = 0;
    int heads = 0;
    int kv_heads = 0;
    int head_dim = 0;
    int rotary_dim = 0;
    int vocabulary = 0;
    int capacity = 0;
};

struct LayerCache {
    std::optional<mx::array> keys;
    std::optional<mx::array> values;
};

} // namespace

class MlxModel::Impl {
public:
    explicit Impl(const CorePaths& paths)
        : Impl(paths, select_device(paths)) {}

    const InferenceRuntimeInfo& runtime_info() const noexcept {
        return runtime_;
    }

    std::size_t vocabulary_size() const noexcept {
        return static_cast<std::size_t>(shape_.vocabulary);
    }

    int layer_count() const noexcept {
        return shape_.layers;
    }

    int capacity() const noexcept {
        return shape_.capacity;
    }

    int bos_token() const noexcept {
        return bos_token_;
    }

    const mx::Stream& stream() const noexcept {
        return stream_;
    }

    mx::array forward(
        std::span<const int> tokens,
        int position,
        std::vector<LayerCache>& cache) const {
        const int count = static_cast<int>(tokens.size());
        const mx::array indices(tokens.data(), {1, count}, mx::int32);
        auto hidden = mx::take(weights_.at("token_embd.weight"), indices, 0, stream_);
        const auto mask = causal_mask(position, count);

        for (int layer = 0; layer < shape_.layers; ++layer) {
            hidden = forward_layer(
                std::move(hidden),
                layer,
                position,
                count,
                mask,
                cache[static_cast<std::size_t>(layer)]);
        }

        hidden = mx::slice(
            hidden,
            {0, count - 1, 0},
            {1, count, shape_.hidden},
            stream_);
        hidden = mx::fast::rms_norm(
            hidden, weights_.at("output_norm.weight"), epsilon_, stream_);
        return mx::astype(
            mx::reshape(linear(hidden, "output.weight"), {shape_.vocabulary}, stream_),
            mx::float32,
            stream_);
    }

private:
    Impl(const CorePaths& paths, const DeviceSelection& selection)
        : device_(selection.device),
          stream_(mx::new_thread_unsafe_stream(device_)),
          runtime_{
              InferenceProvider::mlx,
              describe_device(device_),
              selection.uses_gpu,
              selection.wants_gpu && !selection.uses_gpu,
          } {
        std::clog << "[CORE] loading MLX GGUF model: " << paths.model_path() << '\n';
        auto [weights, metadata] = mx::load_gguf(paths.model_path().string(), stream_);
        load_configuration(paths, metadata);
        prepare_weights(weights);
        weights_ = std::move(weights);
        validate_weights();
        materialize_weights();
        std::clog << "[CORE] MLX model ready device=" << runtime_.device.description << '\n';
    }

    void load_configuration(const CorePaths& paths, const Metadata& metadata) {
        if (string_metadata(metadata, "general.architecture") != "llama") {
            throw std::runtime_error("MLX backend currently requires a Llama GGUF model");
        }

        shape_.hidden = integer_metadata(metadata, "llama.embedding_length");
        shape_.layers = integer_metadata(metadata, "llama.block_count");
        shape_.heads = integer_metadata(metadata, "llama.attention.head_count");
        shape_.kv_heads = integer_metadata(
            metadata, "llama.attention.head_count_kv", shape_.heads);
        if (shape_.hidden % shape_.heads != 0 ||
            shape_.heads % shape_.kv_heads != 0) {
            throw std::runtime_error("unsupported Llama attention dimensions");
        }

        shape_.head_dim = shape_.hidden / shape_.heads;
        shape_.rotary_dim = integer_metadata(
            metadata, "llama.rope.dimension_count", shape_.head_dim);
        if (shape_.rotary_dim > shape_.head_dim || shape_.rotary_dim % 2 != 0) {
            throw std::runtime_error("unsupported Llama rotary dimensions");
        }

        epsilon_ = float_metadata(
            metadata, "llama.attention.layer_norm_rms_epsilon", 1e-5F);
        rope_base_ = float_metadata(metadata, "llama.rope.freq_base", 10000.0F);
        const int native_capacity = integer_metadata(metadata, "llama.context_length");
        if (paths.context_length() >
            static_cast<std::uint32_t>(std::numeric_limits<int>::max())) {
            throw std::invalid_argument("MLX context length exceeds supported range");
        }
        shape_.capacity = paths.context_length() == 0
            ? native_capacity
            : static_cast<int>(paths.context_length());

        if (metadata.contains("llama.rope.scaling.type") &&
            string_metadata(metadata, "llama.rope.scaling.type") != "none") {
            throw std::runtime_error("MLX backend does not yet support GGUF RoPE scaling");
        }

        bos_token_ = read_bos_token(metadata);
    }

    int read_bos_token(const Metadata& metadata) const {
        const auto entry = metadata.find("tokenizer.ggml.bos_token_id");
        if (entry == metadata.end()) {
            return -1;
        }
        const auto token = mx::astype(
            metadata_array(metadata, "tokenizer.ggml.bos_token_id"),
            mx::int64,
            mx::Device{mx::Device::cpu})
                               .item<std::int64_t>();
        if (token < 0 || token > std::numeric_limits<int>::max()) {
            throw std::runtime_error("GGUF BOS token is out of range");
        }
        return static_cast<int>(token);
    }

    void prepare_weights(Weights& weights) const {
        for (auto& [name, weight] : weights) {
            if (!name.ends_with(".weight")) {
                continue;
            }
            if (weight.dtype() != mx::uint32) {
                weight = mx::astype(weight, mx::float16, stream_);
                continue;
            }

            const auto prefix = name.substr(0, name.size() - 7);
            const auto& scales = weights.at(prefix + ".scales");
            const auto& biases = weights.at(prefix + ".biases");
            if (weight.ndim() != 2 || scales.ndim() != 2 ||
                scales.shape(-1) <= 0 ||
                weight.shape(-1) % scales.shape(-1) != 0) {
                throw std::runtime_error("unsupported packed GGUF tensor " + name);
            }
            const int bits = weight.shape(-1) / scales.shape(-1);
            if (bits != 4 && bits != 8) {
                throw std::runtime_error("unsupported GGUF quantization " + name);
            }
            weight = mx::dequantize(
                weight,
                scales,
                biases,
                32,
                bits,
                "affine",
                std::nullopt,
                mx::float16,
                stream_);
        }
    }

    void validate_weights() {
        const auto embedding = weights_.at("token_embd.weight");
        if (embedding.ndim() != 2 || embedding.shape(1) != shape_.hidden) {
            throw std::runtime_error("invalid Llama embedding shape");
        }
        shape_.vocabulary = embedding.shape(0);
        if (shape_.vocabulary <= 0) {
            throw std::runtime_error("MLX model has no vocabulary");
        }
        if (bos_token_ < 0) {
            bos_token_ = shape_.vocabulary > 1 ? 1 : 0;
        } else if (bos_token_ >= shape_.vocabulary) {
            throw std::runtime_error("GGUF BOS token exceeds vocabulary");
        }
        if (!weights_.contains("output.weight")) {
            weights_.emplace("output.weight", embedding);
        }

        require_shape("output.weight", {shape_.vocabulary, shape_.hidden});
        require_shape("output_norm.weight", {shape_.hidden});
        for (int layer = 0; layer < shape_.layers; ++layer) {
            validate_layer(layer);
        }
    }

    void validate_layer(int layer) const {
        const auto prefix = layer_prefix(layer);
        require_shape(prefix + "attn_norm.weight", {shape_.hidden});
        require_shape(prefix + "ffn_norm.weight", {shape_.hidden});
        require_shape(prefix + "attn_q.weight", {shape_.hidden, shape_.hidden});
        require_shape(
            prefix + "attn_k.weight",
            {shape_.kv_heads * shape_.head_dim, shape_.hidden});
        require_shape(
            prefix + "attn_v.weight",
            {shape_.kv_heads * shape_.head_dim, shape_.hidden});
        require_shape(prefix + "attn_output.weight", {shape_.hidden, shape_.hidden});

        const auto& gate = weights_.at(prefix + "ffn_gate.weight");
        if (gate.ndim() != 2 || gate.shape(1) != shape_.hidden || gate.shape(0) <= 0) {
            throw std::runtime_error("invalid Llama feed-forward shape");
        }
        require_shape(prefix + "ffn_up.weight", gate.shape());
        require_shape(prefix + "ffn_down.weight", {shape_.hidden, gate.shape(0)});
    }

    void materialize_weights() {
        std::vector<mx::array> arrays;
        arrays.reserve(weights_.size());
        for (const auto& [name, weight] : weights_) {
            static_cast<void>(name);
            arrays.push_back(weight);
        }
        mx::eval(arrays);
        std::erase_if(weights_, [](const auto& entry) {
            return entry.first.ends_with(".scales") ||
                entry.first.ends_with(".biases");
        });
    }

    void require_shape(const std::string& name, const mx::Shape& shape) const {
        if (weights_.at(name).shape() != shape) {
            throw std::runtime_error("invalid Llama weight shape: " + name);
        }
    }

    mx::array linear(const mx::array& input, const std::string& name) const {
        return mx::matmul(
            input,
            mx::transpose(weights_.at(name), stream_),
            stream_);
    }

    std::optional<mx::array> causal_mask(int position, int count) const {
        if (count <= 1) {
            return std::nullopt;
        }
        const auto rows = mx::reshape(
            mx::arange(position, position + count, stream_),
            {count, 1},
            stream_);
        const auto columns = mx::reshape(
            mx::arange(position + count, stream_),
            {1, position + count},
            stream_);
        return mx::greater_equal(rows, columns, stream_);
    }

    mx::array forward_layer(
        mx::array hidden,
        int layer,
        int position,
        int count,
        const std::optional<mx::array>& mask,
        LayerCache& cache) const {
        const auto prefix = layer_prefix(layer);
        const auto normalized = mx::fast::rms_norm(
            hidden, weights_.at(prefix + "attn_norm.weight"), epsilon_, stream_);
        auto query = project_attention(
            normalized, prefix + "attn_q.weight", shape_.heads, count);
        auto key = project_attention(
            normalized, prefix + "attn_k.weight", shape_.kv_heads, count);
        auto value = project_attention(
            normalized, prefix + "attn_v.weight", shape_.kv_heads, count);

        query = mx::fast::rope(
            query, shape_.rotary_dim, true, rope_base_, 1.0F,
            position, std::nullopt, stream_);
        key = mx::fast::rope(
            key, shape_.rotary_dim, true, rope_base_, 1.0F,
            position, std::nullopt, stream_);
        if (cache.keys) {
            key = mx::concatenate({*cache.keys, key}, 2, stream_);
            value = mx::concatenate({*cache.values, value}, 2, stream_);
        }
        cache.keys = key;
        cache.values = value;

        const auto attended = mx::fast::scaled_dot_product_attention(
            query,
            key,
            value,
            1.0F / std::sqrt(static_cast<float>(shape_.head_dim)),
            "",
            mask,
            std::nullopt,
            false,
            stream_);
        const auto attention = mx::reshape(
            mx::transpose(attended, {0, 2, 1, 3}, stream_),
            {1, count, shape_.hidden},
            stream_);
        hidden = mx::add(
            hidden,
            linear(attention, prefix + "attn_output.weight"),
            stream_);

        const auto normalized_feed_forward = mx::fast::rms_norm(
            hidden, weights_.at(prefix + "ffn_norm.weight"), epsilon_, stream_);
        const auto gate = linear(normalized_feed_forward, prefix + "ffn_gate.weight");
        const auto activated = mx::multiply(
            mx::multiply(gate, mx::sigmoid(gate, stream_), stream_),
            linear(normalized_feed_forward, prefix + "ffn_up.weight"),
            stream_);
        return mx::add(
            hidden,
            linear(activated, prefix + "ffn_down.weight"),
            stream_);
    }

    mx::array project_attention(
        const mx::array& hidden,
        const std::string& weight,
        int heads,
        int count) const {
        return mx::transpose(
            mx::reshape(
                linear(hidden, weight),
                {1, count, heads, shape_.head_dim},
                stream_),
            {0, 2, 1, 3},
            stream_);
    }

    static std::string layer_prefix(int layer) {
        return "blk." + std::to_string(layer) + ".";
    }

    Weights weights_;
    mx::Device device_;
    mx::Stream stream_;
    InferenceRuntimeInfo runtime_;
    ModelShape shape_;
    int bos_token_ = -1;
    float epsilon_ = 1e-5F;
    float rope_base_ = 10000.0F;
};

class MlxBackend::Impl {
public:
    explicit Impl(std::shared_ptr<MlxModel> model) : model_(std::move(model)) {
        if (!model_) {
            throw std::invalid_argument("MLX backend requires a model");
        }
        cache_.resize(static_cast<std::size_t>(model_->impl_->layer_count()));
    }

    void clear() noexcept {
        for (auto& entry : cache_) {
            entry.keys.reset();
            entry.values.reset();
        }
        logits_.clear();
        offset_ = 0;
    }

    bool rewind(std::size_t position) {
        if (position > offset_) {
            return false;
        }
        if (position == 0) {
            clear();
            return true;
        }
        for (auto& entry : cache_) {
            if (!entry.keys || !entry.values) {
                return false;
            }
            auto shape = entry.keys->shape();
            shape[2] = static_cast<int>(position);
            entry.keys = mx::slice(
                *entry.keys, {0, 0, 0, 0}, shape, model_->impl_->stream());
            entry.values = mx::slice(
                *entry.values, {0, 0, 0, 0}, shape, model_->impl_->stream());
        }
        offset_ = position;
        logits_.clear();
        return true;
    }

    void decode(std::span<const int> tokens, std::size_t position) {
        const auto& model = *model_->impl_;
        const auto capacity = static_cast<std::size_t>(model.capacity());
        if (tokens.empty() || position != offset_ || position > capacity ||
            tokens.size() > capacity - position) {
            throw std::invalid_argument("invalid MLX decode extent");
        }
        for (const int token : tokens) {
            if (token < 0 || static_cast<std::size_t>(token) >= model.vocabulary_size()) {
                throw std::invalid_argument("MLX token exceeds vocabulary");
            }
        }

        auto next_cache = cache_;
        const auto next_logits = model.forward(
            tokens, static_cast<int>(position), next_cache);
        std::vector<mx::array> pending{next_logits};
        pending.reserve(1 + next_cache.size() * 2);
        for (const auto& entry : next_cache) {
            pending.push_back(*entry.keys);
            pending.push_back(*entry.values);
        }
        mx::eval(pending);

        const auto* data = next_logits.data<float>();
        logits_.assign(data, data + model.vocabulary_size());
        cache_ = std::move(next_cache);
        offset_ += tokens.size();
    }

    std::span<const float> logits() const {
        if (logits_.empty()) {
            throw std::runtime_error("MLX returned no logits");
        }
        return logits_;
    }

    void ready() const {
        const auto& model = *model_->impl_;
        std::vector<LayerCache> scratch(
            static_cast<std::size_t>(model.layer_count()));
        const int token = model.bos_token();
        mx::eval(model.forward(std::span(&token, 1), 0, scratch));
    }

private:
    std::shared_ptr<MlxModel> model_;
    std::vector<LayerCache> cache_;
    std::vector<float> logits_;
    std::size_t offset_ = 0;
};

MlxModel::MlxModel(std::shared_ptr<const CorePaths> paths) {
    if (!paths) {
        throw std::invalid_argument("MLX model paths are required");
    }
    const std::lock_guard lock(executor_mutex());
    impl_ = std::make_unique<Impl>(*paths);
}

MlxModel::~MlxModel() {
    const std::lock_guard lock(executor_mutex());
    impl_.reset();
}

const InferenceRuntimeInfo& MlxModel::runtime_info() const noexcept {
    return impl_->runtime_info();
}

std::size_t MlxModel::vocabulary_size() const noexcept {
    return impl_->vocabulary_size();
}

std::unique_ptr<BackendSession> MlxModel::create_session() {
    return std::make_unique<MlxBackend>(
        std::static_pointer_cast<MlxModel>(shared_from_this()));
}

std::vector<InferenceDeviceInfo> MlxModel::enumerate_devices() {
    const std::lock_guard lock(executor_mutex());
    std::vector<InferenceDeviceInfo> devices{
        describe_device(mx::Device{mx::Device::cpu}),
    };
    if (mx::is_available(mx::Device{mx::Device::gpu})) {
        devices.push_back(describe_device(mx::Device{mx::Device::gpu}));
    }
    return devices;
}

MlxBackend::MlxBackend(std::shared_ptr<MlxModel> model)
    : impl_(std::make_unique<Impl>(std::move(model))) {}

MlxBackend::~MlxBackend() {
    const std::lock_guard lock(executor_mutex());
    impl_.reset();
}

MlxBackend::MlxBackend(MlxBackend&&) noexcept = default;

MlxBackend& MlxBackend::operator=(MlxBackend&& other) noexcept {
    const std::lock_guard lock(executor_mutex());
    impl_ = std::move(other.impl_);
    return *this;
}

void MlxBackend::clear() noexcept {
    const std::lock_guard lock(executor_mutex());
    impl_->clear();
}

bool MlxBackend::rewind(std::size_t position) {
    const std::lock_guard lock(executor_mutex());
    return impl_->rewind(position);
}

void MlxBackend::decode(std::span<const int> tokens, std::size_t position) {
    const std::lock_guard lock(executor_mutex());
    impl_->decode(tokens, position);
}

std::span<const float> MlxBackend::logits() const {
    const std::lock_guard lock(executor_mutex());
    return impl_->logits();
}

void MlxBackend::ready() {
    const std::lock_guard lock(executor_mutex());
    impl_->ready();
}

} // namespace llavon::ime::core::internal

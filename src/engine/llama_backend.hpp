#pragma once

#include <ggml-backend.h>
#include <llama-cpp.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <format>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "../utils/core_paths.hpp"
#include "vulkan_pipeline_warmup.hpp"

namespace llavon::ime::core::internal {

struct LlamaOffloadDevice {
    ggml_backend_dev_t device = nullptr;
    enum ggml_backend_dev_type type = GGML_BACKEND_DEVICE_TYPE_CPU;
    InferenceBackend backend = InferenceBackend::cpu;
    std::string device_id;
    std::size_t memory_free = 0;
    std::size_t memory_total = 0;
};

class LlamaModel {
public:
    static std::vector<InferenceDeviceInfo> enumerate_devices() {
        initialize();
        std::vector<InferenceDeviceInfo> result;
        for (const auto& device : scan_devices()) result.push_back(device_info(device));
        return result;
    }

    explicit LlamaModel(std::shared_ptr<const CorePaths> paths) : paths_(std::move(paths)) {
        if (!paths_) throw std::invalid_argument("llama model paths are required");
        initialize();
        auto params = llama_model_default_params();
        std::array<ggml_backend_dev_t, 2> devices{};
        const auto& requested = paths_->inference_device();
        auto selected = requested.backend == InferenceBackend::cpu ? LlamaOffloadDevice{} : select_gpu(requested);
        const int layers = paths_->gpu_layers();
        const bool wants_gpu = requested.backend != InferenceBackend::cpu &&
            (layers == -2 || layers == -1 || layers > 0);
        const bool offload = wants_gpu && selected.device && llama_supports_gpu_offload();
        params.devices = devices.data();
        params.n_gpu_layers = 0;
        if (offload) {
            devices[0] = selected.device;
            params.n_gpu_layers = layers < 0 ? -1 : layers;
            params.split_mode = LLAMA_SPLIT_MODE_NONE;
            params.main_gpu = 0;
            if (selected.backend == InferenceBackend::vulkan && !paths_->vulkan_pipeline_cache_dir().empty()) {
                using Configure = bool (*)(ggml_backend_dev_t, const char*);
                const auto registry = ggml_backend_dev_backend_reg(selected.device);
                const auto configure = reinterpret_cast<Configure>(ggml_backend_reg_get_proc_address(
                    registry, "ggml_backend_vk_set_pipeline_cache_directory"));
                if (configure) {
                    const auto encoded = paths_->vulkan_pipeline_cache_dir().u8string();
                    const std::string directory(reinterpret_cast<const char*>(encoded.data()), encoded.size());
                    if (!configure(selected.device, directory.c_str()))
                        std::clog << "[CORE] Vulkan pipeline cache unavailable\n";
                } else std::clog << "[CORE] Vulkan pipeline cache unsupported by backend\n";
            }
        }
        std::clog << "[CORE] loading model: " << paths_->model_path() << '\n';
        model_.reset(llama_model_load_from_file(paths_->model_path().string().c_str(), params));
        if (!model_) throw std::runtime_error("Failed to load llama model");
        vocab_ = llama_model_get_vocab(model_.get());
        if (!vocab_ || llama_vocab_n_tokens(vocab_) <= 0) throw std::runtime_error("llama model has no vocabulary");
        const auto active = offload ? selected : select_cpu();
        if (!active.device) throw std::runtime_error("Failed to identify active llama inference device");
        runtime_ = {device_info(active), offload, wants_gpu && !offload};
        std::clog << "[CORE] active inference device backend=" << backend_name(runtime_.device.backend)
                  << " name=" << runtime_.device.name << " id=" << runtime_.device.device_id << '\n';
        if (offload && selected.backend == InferenceBackend::vulkan) {
            llama_context_ptr context{new_context()};
            prepare_vulkan_pipelines(context.get(), warmup_token());
            using Save = void (*)(ggml_backend_dev_t);
            const auto registry = ggml_backend_dev_backend_reg(selected.device);
            const auto save = reinterpret_cast<Save>(ggml_backend_reg_get_proc_address(
                registry, "ggml_backend_vk_save_pipeline_cache"));
            if (save) save(selected.device);
            prepared_context_ = std::move(context);
        }
    }

    const InferenceRuntimeInfo& runtime_info() const noexcept { return runtime_; }
    std::size_t vocabulary_size() const noexcept { return static_cast<std::size_t>(llama_vocab_n_tokens(vocab_)); }
    int warmup_token() const noexcept {
        for (const auto token : {llama_vocab_bos(vocab_), llama_vocab_eos(vocab_), llama_vocab_nl(vocab_), llama_vocab_pad(vocab_)})
            if (token >= 0) return token;
        return 0;
    }

    llama_context* new_context(std::uint32_t size = 0, std::uint32_t batch = 0) {
        if (size == 0 && batch == 0) {
            const std::lock_guard lock(prepared_mutex_);
            if (prepared_context_) return prepared_context_.release();
        }
        auto params = llama_context_default_params();
        if (size != 0) params.n_ctx = size;
        else if (paths_->context_length() != 0) params.n_ctx = paths_->context_length();
        if (batch != 0) { params.n_batch = batch; params.n_ubatch = batch; }
        if (paths_->threads() > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max()))
            throw std::invalid_argument("llama thread count exceeds supported range");
        params.n_threads = static_cast<std::int32_t>(paths_->threads());
        params.n_threads_batch = params.n_threads;
        auto* context = llama_init_from_model(model_.get(), params);
        if (!context) throw std::runtime_error("Failed to create llama context");
        return context;
    }

private:
    std::shared_ptr<const CorePaths> paths_;
    llama_model_ptr model_;
    const llama_vocab* vocab_ = nullptr;
    InferenceRuntimeInfo runtime_;
    std::mutex prepared_mutex_;
    llama_context_ptr prepared_context_;

    static void initialize() {
        static std::once_flag once;
        std::call_once(once, [] { ggml_backend_load_all(); llama_backend_init(); });
    }
    static std::string lowercase(const char* text) {
        std::string value = text ? text : "";
        std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
            return static_cast<char>(std::tolower(c));
        });
        return value;
    }
    static const char* backend_name(InferenceBackend backend) {
        if (backend == InferenceBackend::cpu) return "CPU";
        if (backend == InferenceBackend::cuda) return "CUDA";
        if (backend == InferenceBackend::vulkan) return "Vulkan";
        if (backend == InferenceBackend::metal) return "Metal";
        return "automatic";
    }
    static std::optional<InferenceBackend> backend_type(ggml_backend_dev_t device, enum ggml_backend_dev_type type) {
        if (type == GGML_BACKEND_DEVICE_TYPE_CPU) return InferenceBackend::cpu;
        const auto registry = ggml_backend_dev_backend_reg(device);
        const auto name = lowercase(registry ? ggml_backend_reg_name(registry) : nullptr);
        if (name.find("cuda") != std::string::npos) return InferenceBackend::cuda;
        if (name.find("vulkan") != std::string::npos) return InferenceBackend::vulkan;
        if (name == "mtl" || name.find("metal") != std::string::npos) return InferenceBackend::metal;
        return std::nullopt;
    }
    static std::vector<LlamaOffloadDevice> scan_devices() {
        std::vector<LlamaOffloadDevice> devices;
        for (std::size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            auto device = ggml_backend_dev_get(i);
            ggml_backend_dev_props props{};
            ggml_backend_dev_get_props(device, &props);
            const auto backend = backend_type(device, props.type);
            if (!backend) continue;
            const auto name = ggml_backend_dev_name(device);
            devices.push_back({device, props.type, *backend,
                props.device_id && *props.device_id ? props.device_id : (name ? name : ""),
                props.memory_free, props.memory_total});
        }
        return devices;
    }
    static InferenceDeviceInfo device_info(const LlamaOffloadDevice& device) {
        ggml_backend_dev_props props{};
        ggml_backend_dev_get_props(device.device, &props);
        const auto name = props.name ? props.name : ggml_backend_dev_name(device.device);
        const auto description = props.description ? props.description : ggml_backend_dev_description(device.device);
        const auto type = device.type == GGML_BACKEND_DEVICE_TYPE_IGPU ? InferenceDeviceType::integrated_gpu :
            device.type == GGML_BACKEND_DEVICE_TYPE_GPU ? InferenceDeviceType::gpu : InferenceDeviceType::cpu;
        return {device.backend, type, device.device_id, name ? name : "", description ? description : "",
                static_cast<std::uint64_t>(device.memory_free), static_cast<std::uint64_t>(device.memory_total)};
    }
    static int priority(InferenceBackend backend) {
        if (backend == InferenceBackend::cuda) return 0;
        if (backend == InferenceBackend::vulkan) return 1;
        if (backend == InferenceBackend::metal) return 2;
        return 3;
    }
    static LlamaOffloadDevice select_gpu(const InferenceDeviceSelection& requested) {
        auto devices = scan_devices();
        std::erase_if(devices, [&](const auto& device) {
            return (device.type != GGML_BACKEND_DEVICE_TYPE_GPU && device.type != GGML_BACKEND_DEVICE_TYPE_IGPU) ||
                (requested.backend != InferenceBackend::automatic && requested.backend != device.backend) ||
                (!requested.device_id.empty() && requested.device_id != device.device_id);
        });
        std::sort(devices.begin(), devices.end(), [](const auto& a, const auto& b) {
            if (a.type != b.type) return a.type == GGML_BACKEND_DEVICE_TYPE_GPU;
            if (a.backend != b.backend) return priority(a.backend) < priority(b.backend);
            if (a.memory_total != b.memory_total) return a.memory_total > b.memory_total;
            return a.memory_free > b.memory_free;
        });
        return devices.empty() ? LlamaOffloadDevice{} : devices.front();
    }
    static LlamaOffloadDevice select_cpu() {
        for (const auto& device : scan_devices())
            if (device.backend == InferenceBackend::cpu && device.type == GGML_BACKEND_DEVICE_TYPE_CPU) return device;
        return {};
    }
};

class LlamaBackend {
public:
    using Model = LlamaModel;
    explicit LlamaBackend(std::shared_ptr<Model> model) : model_(std::move(model)) {
        if (!model_) throw std::invalid_argument("llama backend requires a model");
        context_.reset(model_->new_context());
        clear();
    }
    void clear() noexcept { llama_memory_clear(llama_get_memory(context_.get()), true); }
    bool rewind(std::size_t position) {
        if (position > static_cast<std::size_t>(std::numeric_limits<llama_pos>::max())) return false;
        return llama_memory_seq_rm(llama_get_memory(context_.get()), 0, static_cast<llama_pos>(position), -1);
    }
    void ready() {
        if (!warmup_) warmup_.reset(model_->new_context(8, 1));
        const int token = model_->warmup_token();
        llama_set_warmup(warmup_.get(), true);
        try { execute(warmup_.get(), std::span(&token, 1), 0); }
        catch (...) {
            llama_memory_clear(llama_get_memory(warmup_.get()), true);
            llama_set_warmup(warmup_.get(), false);
            throw;
        }
        llama_memory_clear(llama_get_memory(warmup_.get()), true);
        llama_set_warmup(warmup_.get(), false);
    }
    void decode(std::span<const int> tokens, std::size_t position) {
        execute(context_.get(), tokens, position);
    }
    std::span<const float> logits() const {
        const auto* values = llama_get_logits_ith(context_.get(), -1);
        if (!values) throw std::runtime_error("llama returned no logits");
        return {values, model_->vocabulary_size()};
    }

private:
    std::shared_ptr<Model> model_;
    llama_context_ptr context_;
    llama_context_ptr warmup_;

    void execute(llama_context* context, std::span<const int> tokens, std::size_t position) {
        const auto limit = static_cast<std::size_t>(llama_n_ctx(context));
        if (tokens.empty() || position > limit || tokens.size() > limit - position ||
            tokens.size() > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()))
            throw std::invalid_argument("invalid llama decode extent");
        for (const int token : tokens)
            if (token < 0 || static_cast<std::size_t>(token) >= model_->vocabulary_size())
                throw std::invalid_argument("llama decode token exceeds vocabulary");
        struct Batch {
            llama_batch value;
            explicit Batch(std::size_t size) : value(llama_batch_init(static_cast<std::int32_t>(size), 0, 1)) {}
            ~Batch() { llama_batch_free(value); }
            Batch(const Batch&) = delete;
            Batch& operator=(const Batch&) = delete;
        } batch(tokens.size());
        auto& b = batch.value;
        if (!b.token || !b.pos || !b.n_seq_id || !b.seq_id || !b.logits)
            throw std::runtime_error("llama_batch_init failed");
        b.n_tokens = static_cast<std::int32_t>(tokens.size());
        for (std::size_t i = 0; i < tokens.size(); ++i) {
            b.token[i] = tokens[i]; b.pos[i] = static_cast<llama_pos>(position + i);
            b.n_seq_id[i] = 1; b.seq_id[i][0] = 0; b.logits[i] = static_cast<std::int8_t>(i + 1 == tokens.size());
        }
        const int result = llama_decode(context, b);
        llama_synchronize(context);
        if (result != 0) throw std::runtime_error("llama decode failed");
    }
};

} // namespace llavon::ime::core::internal

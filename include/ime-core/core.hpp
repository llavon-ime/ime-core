#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace llavon::ime::core {

class Logger;

enum class InferenceBackend : std::uint8_t {
    automatic,
    cpu,
    cuda,
    vulkan,
    metal,
    ryzen_ai,
};

enum class InferenceDeviceType : std::uint8_t {
    cpu,
    gpu,
    integrated_gpu,
    npu,
};

struct InferenceDeviceSelection {
    InferenceBackend backend = InferenceBackend::automatic;
    std::string device_id;
};

struct InferenceDeviceInfo {
    InferenceBackend backend = InferenceBackend::cpu;
    InferenceDeviceType type = InferenceDeviceType::cpu;
    std::string device_id;
    std::string name;
    std::string description;
    std::uint64_t memory_free = 0;
    std::uint64_t memory_total = 0;
};

struct InferenceRuntimeInfo {
    InferenceDeviceInfo device;
    bool gpu_offload = false;
    bool fell_back_to_cpu = false;
    bool npu_offload = false;
};

// One decoder context per IME session. Implementations retain their own KV
// cache; only tokens and final logits cross this interface.
class InferenceContext {
public:
    virtual ~InferenceContext() = default;
    virtual void decode(std::span<const std::int32_t> tokens, std::uint32_t position) = 0;
    virtual void truncate(std::uint32_t length) = 0;
    virtual std::span<const float> logits() const = 0;
};

// The model owns its accelerator and prepared weights. Preparation must finish
// before the replacement core is made visible to IME clients.
class InferenceAccelerator {
public:
    virtual ~InferenceAccelerator() = default;
    virtual void prepare(const std::filesystem::path& model, std::uint32_t context_length) = 0;
    virtual std::unique_ptr<InferenceContext> create_context() = 0;
    virtual InferenceDeviceInfo device_info() const = 0;
};

struct CoreConfig {
    std::filesystem::path model_path;
    std::filesystem::path tables_dir;
    std::uint32_t context_length = 0;
    std::uint32_t threads = 8;
    int gpu_layers = -2;
    InferenceDeviceSelection inference_device;
    std::shared_ptr<Logger> logger;
    std::shared_ptr<InferenceAccelerator> accelerator;
};

// Enumerates devices from the inference backends that are available in the
// current process. This function does not load a model or read application
// settings.
std::vector<InferenceDeviceInfo> enumerate_inference_devices();

struct PaddingEntry {
    bool chosen = false;
    char32_t chosen_char = 0;
    std::u16string bopomofo;
};

struct Prediction {
    std::vector<std::pair<char32_t, float>> candidates;
};

class Session final {
public:
    ~Session();

    Session(Session&&) noexcept;
    Session& operator=(Session&&) noexcept;

    Session(const Session&) = delete;
    Session& operator=(const Session&) = delete;

    void ready();
    std::vector<Prediction> predict(
        const std::u16string& context,
        const std::vector<PaddingEntry>& padding);

private:
    class Impl;

    explicit Session(std::unique_ptr<Impl> impl);

    std::unique_ptr<Impl> impl_;

    friend class Core;
};

class Core final {
public:
    explicit Core(CoreConfig config);
    ~Core();

    Core(Core&&) noexcept;
    Core& operator=(Core&&) noexcept;

    Core(const Core&) = delete;
    Core& operator=(const Core&) = delete;

    std::unique_ptr<Session> create_session() const;
    InferenceRuntimeInfo inference_runtime_info() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace llavon::ime::core

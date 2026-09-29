#include "onnx.hpp"
#include "provider_metadata.hpp"

#include <Windows.h>
#include <WinMLEpCatalog.h>
#include <ort_genai.h>
#include <rfl/Rename.hpp>
#include <rfl/json/write.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <format>
#include <limits>
#include <memory>
#include <ranges>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace llavon::ime::core::ryzen_ai {
namespace {

constexpr std::string_view kProviderName = "VitisAIExecutionProvider";
constexpr std::uint32_t kAmdVendorId = 0x1022;

using detail::check_hresult;
using detail::provider_string;

std::string utf8_path(const std::filesystem::path& path) {
    const auto text = path.u8string();
    return {reinterpret_cast<const char*>(text.data()), text.size()};
}

struct CatalogCloser {
    void operator()(WinMLEpCatalog* catalog) const noexcept { WinMLEpCatalogRelease(catalog); }
};
using Catalog = std::unique_ptr<WinMLEpCatalog, CatalogCloser>;

OgaHandle& oga_lifetime() {
    // This object must outlive every OGA model and generator in the process.
    static OgaHandle handle;
    return handle;
}

struct RegisteredProvider {
    std::string cache_identity;
    std::string library;
    std::string version;
    std::string version_error;
};

const RegisteredProvider& ensure_vitisai_provider() {
    static const RegisteredProvider provider = [] {
        (void)oga_lifetime();
        WinMLEpCatalogHandle raw_catalog = nullptr;
        check_hresult("WinMLEpCatalogCreate", WinMLEpCatalogCreate(&raw_catalog));
        Catalog catalog(raw_catalog);

        WinMLEpHandle ep = nullptr;
        check_hresult("Find VitisAI execution provider",
                      WinMLEpCatalogFindProvider(catalog.get(), kProviderName.data(), nullptr, &ep));

        WinMLEpCertification certification = WinMLEpCertification_Unknown;
        check_hresult("Read VitisAI certification", WinMLEpGetCertification(ep, &certification));
        if (certification != WinMLEpCertification_Certified) {
            throw std::runtime_error("Windows ML did not provide a certified VitisAI execution provider");
        }

        // This is the only download/install step. It obtains the Windows-certified
        // EP package; it does not install or invoke the AMD Ryzen AI SDK.
        check_hresult("Install VitisAI execution provider", WinMLEpEnsureReady(ep));
        WinMLEpReadyState state = WinMLEpReadyState_NotPresent;
        check_hresult("Read VitisAI readiness", WinMLEpGetReadyState(ep, &state));
        if (state != WinMLEpReadyState_Ready) {
            throw std::runtime_error("The VitisAI execution provider is not ready after installation");
        }

        const auto library = provider_string(ep, WinMLEpGetLibraryPathSize,
                                             WinMLEpGetLibraryPath, "VitisAI library path");
        std::string version;
        std::string version_error;
        try {
            version = provider_string(ep, WinMLEpGetVersionSize, WinMLEpGetVersion,
                                      "VitisAI version", detail::ProviderString::optional);
        } catch (const std::runtime_error& error) {
            version_error = error.what();
        }
        const auto identity = detail::provider_cache_identity(library);
        OgaRegisterExecutionProviderLibrary(kProviderName.data(), library.c_str());
        return RegisteredProvider{identity, library, version, version_error};
    }();
    return provider;
}

void hash_bytes(std::uint64_t& hash, std::string_view bytes) {
    constexpr std::uint64_t prime = 1099511628211ULL;
    for (const unsigned char byte : bytes) {
        hash ^= byte;
        hash *= prime;
    }
}

std::string model_cache_key(const RyzenAiConfig& config, std::uint32_t context_length,
                            std::string_view provider_identity) {
    std::uint64_t hash = 14695981039346656037ULL;
    hash_bytes(hash, utf8_path(config.model_directory.lexically_normal()));
    hash_bytes(hash, provider_identity);
    hash_bytes(hash, std::string_view(reinterpret_cast<const char*>(&context_length),
                                      sizeof(context_length)));

    std::vector<std::filesystem::path> files;
    std::error_code error;
    for (std::filesystem::recursive_directory_iterator iterator(config.model_directory, error), end;
         !error && iterator != end; iterator.increment(error)) {
        if (iterator->is_regular_file(error) && !error) files.push_back(iterator->path());
    }
    if (error) throw std::filesystem::filesystem_error("Unable to inspect ONNX model directory",
                                                       config.model_directory, error);
    std::ranges::sort(files);
    for (const auto& file : files) {
        const auto relative = file.lexically_relative(config.model_directory);
        hash_bytes(hash, utf8_path(relative));
        const auto size = std::filesystem::file_size(file);
        const auto timestamp = std::filesystem::last_write_time(file).time_since_epoch().count();
        hash_bytes(hash, std::string_view(reinterpret_cast<const char*>(&size), sizeof(size)));
        hash_bytes(hash, std::string_view(reinterpret_cast<const char*>(&timestamp), sizeof(timestamp)));
    }
    return std::format("{:016x}", hash);
}

struct StrictSessionOptions {
    rfl::Rename<"session.disable_cpu_ep_fallback", std::string> disable_cpu_fallback{"1"};
};
struct CompileSessionOptions {
    rfl::Rename<"session.disable_cpu_ep_fallback", std::string> disable_cpu_fallback{"1"};
    rfl::Rename<"ep.context_enable", std::string> context_enable{"1"};
    rfl::Rename<"ep.context_embed_mode", std::string> context_embed_mode{"1"};
    rfl::Rename<"ep.context_file_path", std::string> context_file_path;
};
template <class SessionOptions>
struct DecoderOverlay {
    SessionOptions session_options;
};
template <class SessionOptions>
struct ModelOverlay {
    DecoderOverlay<SessionOptions> decoder;
};
template <class SessionOptions>
struct ConfigOverlay {
    ModelOverlay<SessionOptions> model;
};

struct CachedDecoderOverlay {
    StrictSessionOptions session_options;
    std::string filename;
};
struct CachedModelOverlay {
    CachedDecoderOverlay decoder;
};
struct CachedConfigOverlay {
    CachedModelOverlay model;
};

std::unique_ptr<OgaConfig> base_config(const RyzenAiConfig& config) {
    const auto directory = utf8_path(config.model_directory);
    auto result = OgaConfig::Create(directory.c_str());
    result->ClearProviders();
    result->AppendProvider(kProviderName.data());
    result->SetDecoderProviderOptionsHardwareDeviceType(kProviderName.data(), "NPU");
    result->SetDecoderProviderOptionsHardwareVendorId(kProviderName.data(), kAmdVendorId);
    return result;
}

std::unique_ptr<OgaModel> create_model(const RyzenAiConfig& config,
                                       const std::filesystem::path& compiled_model,
                                       bool use_compiled_model) {
    auto runtime_config = base_config(config);
    if (use_compiled_model) {
        const auto overlay = rfl::json::write(CachedConfigOverlay{
            .model = {.decoder = {
                .session_options = {}, .filename = utf8_path(compiled_model)}}});
        runtime_config->Overlay(overlay.c_str());
    } else {
        const auto overlay = rfl::json::write(ConfigOverlay<CompileSessionOptions>{
            .model = {.decoder = {.session_options = {
                .context_file_path = utf8_path(compiled_model)}}}});
        runtime_config->Overlay(overlay.c_str());
    }
    return OgaModel::Create(*runtime_config);
}

struct Prepared {
    std::unique_ptr<OgaModel> model;
    std::uint32_t context_length;
};

std::shared_ptr<Prepared> prepare_model(const RyzenAiConfig& config,
                                       std::uint32_t context_length, bool allow_compile) {
    const auto& provider = ensure_vitisai_provider();
    if (!provider.version_error.empty()) {
        config.logger->log("[NPU] VitisAI version metadata unavailable: " + provider.version_error);
    }
    config.logger->log("[NPU] VitisAI library: " + provider.library);
    config.logger->log(std::format("[NPU] Windows ML VitisAI EP registered (version: {})",
                                   provider.version.empty() ? "not reported" : provider.version));
    if (provider.version.empty()) {
        config.logger->log("[NPU] VitisAI version not reported; using installed DLL identity for cache");
    }
    const auto key = model_cache_key(config, context_length, provider.cache_identity);
    const auto directory = config.cache_directory / key;
    std::filesystem::create_directories(directory);
    const auto compiled_model = directory / "decoder_ctx.onnx";

    if (std::filesystem::is_regular_file(compiled_model)) {
        try {
            config.logger->log("[NPU] loading compiled ONNX context: " + utf8_path(compiled_model));
            return std::make_shared<Prepared>(create_model(config, compiled_model, true),
                                              context_length);
        } catch (const std::exception& error) {
            if (!allow_compile) throw;
            config.logger->log(std::string("[NPU] compiled context is stale; recompiling: ") + error.what());
            std::error_code ignored;
            std::filesystem::remove(compiled_model, ignored);
        }
    }

    if (!allow_compile) {
        throw std::runtime_error("No matching EPContext cache is available and compilation is disabled");
    }

    config.logger->log("[NPU] compiling INT4 ONNX decoder for the AMD NPU");
    auto model = create_model(config, compiled_model, false);
    if (!std::filesystem::is_regular_file(compiled_model)) {
        throw std::runtime_error(
            "VitisAI loaded the ONNX model but did not create the required EPContext cache");
    }
    config.logger->log("[NPU] compiled ONNX context cached: " + utf8_path(compiled_model));
    return std::make_shared<Prepared>(std::move(model), context_length);
}

class OnnxContext final : public InferenceContext {
public:
    explicit OnnxContext(std::shared_ptr<Prepared> prepared) : prepared_(std::move(prepared)) {
        auto parameters = OgaGeneratorParams::Create(*prepared_->model);
        parameters->SetSearchOption("batch_size", 1);
        if (prepared_->context_length != 0) {
            parameters->SetSearchOption("max_length", prepared_->context_length);
        }
        generator_ = OgaGenerator::Create(*prepared_->model, *parameters);
    }

    void decode(std::span<const std::int32_t> tokens, std::uint32_t position) override {
        if (tokens.empty() || position != generator_->TokenCount() ||
            (prepared_->context_length != 0 &&
             (position > prepared_->context_length ||
              tokens.size() > prepared_->context_length - position))) {
            throw std::invalid_argument("Invalid ONNX NPU decoder position or context length");
        }

        valid_logits_ = false;
        generator_->AppendTokens(tokens);
        auto output = generator_->GetLogits();
        if (output->Type() != OgaElementType_float32) {
            throw std::runtime_error("ONNX NPU decoder returned logits in an unsupported type");
        }
        const auto shape = output->Shape();
        if (shape.empty() || shape.back() <= 0) {
            throw std::runtime_error("ONNX NPU decoder returned an invalid logits shape");
        }
        std::size_t elements = 1;
        for (const auto dimension : shape) {
            if (dimension <= 0 || static_cast<std::uint64_t>(dimension) >
                                      std::numeric_limits<std::size_t>::max() / elements) {
                throw std::runtime_error("ONNX NPU decoder returned an invalid logits shape");
            }
            elements *= static_cast<std::size_t>(dimension);
        }
        const auto vocabulary = static_cast<std::size_t>(shape.back());
        if (elements < vocabulary || elements % vocabulary != 0) {
            throw std::runtime_error("ONNX NPU decoder returned inconsistent logits dimensions");
        }
        const auto* data = static_cast<const float*>(output->Data());
        logits_.assign(data + elements - vocabulary, data + elements);
        if (!std::ranges::all_of(logits_, [](float value) { return std::isfinite(value); })) {
            throw std::runtime_error("ONNX NPU decoder returned non-finite logits");
        }
        valid_logits_ = true;
    }

    void truncate(std::uint32_t length) override {
        if (length > generator_->TokenCount()) {
            throw std::invalid_argument("Invalid ONNX NPU cache truncation");
        }
        if (length != generator_->TokenCount()) {
            generator_->RewindTo(length);
            valid_logits_ = false;
        }
    }

    std::span<const float> logits() const override {
        if (!valid_logits_) throw std::runtime_error("ONNX NPU logits are not available");
        return logits_;
    }

private:
    std::shared_ptr<Prepared> prepared_;
    std::unique_ptr<OgaGenerator> generator_;
    std::vector<float> logits_;
    bool valid_logits_ = false;
};

class OnnxAccelerator final : public InferenceAccelerator {
public:
    explicit OnnxAccelerator(RyzenAiConfig config) : config_(std::move(config)) {}

    void prepare(const std::filesystem::path&, std::uint32_t context_length) override {
        if (prepared_) throw std::logic_error("An NPU accelerator can prepare only one model");
        prepared_ = prepare_model(config_, context_length, config_.allow_compilation);
    }

    std::unique_ptr<InferenceContext> create_context() override {
        if (!prepared_) throw std::logic_error("ONNX NPU model is not prepared");
        return std::make_unique<OnnxContext>(prepared_);
    }

    InferenceDeviceInfo device_info() const override {
        return {.backend = InferenceBackend::ryzen_ai,
                .type = InferenceDeviceType::npu,
                .device_id = "RYZENAI-NPU",
                .name = "AMD Ryzen AI NPU",
                .description = "Windows ML / VitisAI INT4 ONNX decoder"};
    }

private:
    RyzenAiConfig config_;
    std::shared_ptr<Prepared> prepared_;
};

} // namespace

std::shared_ptr<InferenceAccelerator> create_onnx_accelerator(RyzenAiConfig config) {
    return std::make_shared<OnnxAccelerator>(std::move(config));
}

} // namespace llavon::ime::core::ryzen_ai

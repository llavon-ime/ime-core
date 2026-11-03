#include <ggml-backend.h>
#include <llama-cpp.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

std::vector<std::uint8_t> sequence_state(llama_context* context) {
    std::vector<std::uint8_t> result(llama_state_seq_get_size(context, 0));
    require(llama_state_seq_get_data(context, result.data(), result.size(), 0) == result.size(),
            "unable to snapshot sequence state");
    return result;
}

void prepare(llama_context* context) {
    const auto state = sequence_state(context);
    const auto performance = llama_perf_context(context);
    ggml_backend_graph_prepare_result total{};
    for (const bool full : {false, true}) {
        for (const auto size : {1u, 4u, 16u, 64u}) {
            ggml_backend_graph_prepare_result result{};
            const auto status = llama_prepare_pipelines(context, size, size, full, true, &result);
            require(status == GGML_STATUS_SUCCESS, "host pipeline preparation failed");
            total.cache_hits += result.cache_hits;
            total.compile_required += result.compile_required;
            total.unprobed += result.unprobed;
        }
    }
    require(sequence_state(context) == state, "preparation modified KV sequence state");
    const auto after = llama_perf_context(context);
    require(after.n_eval == performance.n_eval && after.n_p_eval == performance.n_p_eval,
            "preparation performed model evaluation");
    ggml_backend_graph_prepare_result cached{};
    const auto check = llama_prepare_pipelines(context, 16, 16, true, false, &cached);
    require(check == GGML_STATUS_SUCCESS || (check == GGML_STATUS_FAILED && total.unprobed > 0),
            "cache-only check failed after preparation");
    require(cached.compile_required == 0, "prepared graph still needs compilation");
    require(llama_prepare_pipelines(context, 0, 1, true, true, &cached) == GGML_STATUS_FAILED,
            "invalid preparation extent accepted");
    std::cout << "PREPARE,cache_hits=" << total.cache_hits
              << ",compile_required=" << total.compile_required
              << ",unprobed=" << total.unprobed << '\n';
}

void compare_decode(llama_context* prepared, llama_context* reference,
                    llama_token token, llama_pos position, std::size_t vocabulary) {
    auto batch = llama_batch_get_one(&token, 1);
    batch.pos = &position;
    require(llama_decode(prepared, batch) == 0, "prepared decode failed");
    require(llama_decode(reference, batch) == 0, "reference decode failed");
    const auto* actual = llama_get_logits_ith(prepared, -1);
    const auto* expected = llama_get_logits_ith(reference, -1);
    require(actual && expected, "missing logits after preparation");
    for (std::size_t index = 0; index < vocabulary; ++index) {
        require(std::isfinite(actual[index]) && std::abs(actual[index] - expected[index]) < 0.001F,
                "preparation changed inference logits");
    }
}
} // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "usage: ime-core-vulkan-prepare-tests MODEL CACHE_DIRECTORY\n";
        return 2;
    }
    try {
        ggml_backend_load_all();
        const auto registry = ggml_backend_reg_by_name("Vulkan");
        if (!registry || ggml_backend_reg_dev_count(registry) == 0) return 77;
        const auto device = ggml_backend_reg_dev_get(registry, 0);
        using Configure = bool (*)(ggml_backend_dev_t, const char*);
        const auto configure = reinterpret_cast<Configure>(ggml_backend_reg_get_proc_address(
            registry, "ggml_backend_vk_set_pipeline_cache_directory"));
        require(configure && configure(device, argv[2]), "cache configuration failed");
        llama_backend_init();
        auto model_parameters = llama_model_default_params();
        std::array devices{device, ggml_backend_dev_t{}};
        model_parameters.devices = devices.data();
        model_parameters.n_gpu_layers = -1;
        model_parameters.split_mode = LLAMA_SPLIT_MODE_NONE;
        llama_model_ptr model(llama_model_load_from_file(argv[1], model_parameters));
        require(static_cast<bool>(model), "model load failed");
        auto parameters = llama_context_default_params();
        parameters.n_ctx = 512;
        parameters.n_batch = 64;
        parameters.n_ubatch = 64;
        parameters.n_seq_max = 4;
        parameters.no_perf = false;
        llama_context_ptr prepared(llama_init_from_model(model.get(), parameters));
        llama_context_ptr reference(llama_init_from_model(model.get(), parameters));
        require(prepared && reference, "context creation failed");

        ggml_backend_graph_prepare_result probe{};
        const auto status = llama_prepare_pipelines(prepared.get(), 16, 16, true, false, &probe);
        require(status == GGML_STATUS_SUCCESS || status == GGML_STATUS_ABORTED ||
                    status == GGML_STATUS_FAILED, "invalid probe status");
        if (status == GGML_STATUS_ABORTED) {
            ggml_backend_graph_prepare_result retry{};
            require(llama_prepare_pipelines(prepared.get(), 16, 16, true, false, &retry) ==
                        GGML_STATUS_ABORTED && retry.compile_required > 0,
                    "repeated cache-only check forgot missing pipelines");
        }
        std::cout << "PROBE,status=" << status << ",cache_hits=" << probe.cache_hits
                  << ",compile_required=" << probe.compile_required << '\n';
        prepare(prepared.get());
        const auto* vocabulary = llama_model_get_vocab(model.get());
        const auto token = std::max(llama_vocab_bos(vocabulary), 0);
        compare_decode(prepared.get(), reference.get(), token, 0, llama_vocab_n_tokens(vocabulary));
        prepare(prepared.get());
        compare_decode(prepared.get(), reference.get(), token, 1, llama_vocab_n_tokens(vocabulary));
        using Save = void (*)(ggml_backend_dev_t);
        if (const auto save = reinterpret_cast<Save>(ggml_backend_reg_get_proc_address(
                registry, "ggml_backend_vk_save_pipeline_cache"))) save(device);
        std::cout << "[ok] cache probe, host preparation, KV preservation and inference logits\n";
    } catch (const std::exception& error) {
        std::cerr << "[FAIL] " << error.what() << '\n';
        return 1;
    }
}

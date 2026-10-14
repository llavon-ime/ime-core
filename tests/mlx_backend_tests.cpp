#include "engine/core_state.hpp"
#include "engine/mlx_backend.hpp"
#include "backend_selection.hpp"

#include <mlx/mlx.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace {
namespace mx = mlx::core;
using namespace llavon::ime::core;
using namespace llavon::ime::core::internal;

void require(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
class QuietLogger final : public Logger {
public:
    using Logger::log;
    void log(std::string) noexcept override {}
    void log(MessageFactory) noexcept override {}
};
class Fixture {
public:
    std::filesystem::path path;
    Fixture() {
        const auto unique = std::chrono::steady_clock::now().time_since_epoch().count();
        path = std::filesystem::temp_directory_path() / ("ime-core-mlx-" + std::to_string(unique) + ".gguf");
        const mx::Device cpu{mx::Device::cpu};
        std::unordered_map<std::string, mx::array> weights;
        const auto matrix = [cpu](int rows, int columns, float phase) {
            const auto indices = mx::arange(rows * columns, mx::float32, cpu);
            const auto data = mx::multiply(mx::sin(mx::add(mx::multiply(indices, mx::array(0.017F), cpu),
                mx::array(phase), cpu), cpu), mx::array(0.025F), cpu);
            return mx::astype(mx::reshape(data, {rows, columns}, cpu), mx::float16, cpu);
        };
        constexpr int hidden = 32;
        constexpr int vocabulary = 18546;
        weights.emplace("token_embd.weight", matrix(vocabulary, hidden, 0.1F));
        weights.emplace("output.weight", matrix(vocabulary, hidden, 0.7F));
        weights.emplace("output_norm.weight", mx::ones({hidden}, mx::float16, cpu));
        for (int i = 0; i < 2; ++i) {
            const auto prefix = "blk." + std::to_string(i) + ".";
            weights.emplace(prefix + "attn_norm.weight", mx::ones({hidden}, mx::float16, cpu));
            weights.emplace(prefix + "ffn_norm.weight", mx::ones({hidden}, mx::float16, cpu));
            for (const auto* name : {"attn_q", "attn_output"})
                weights.emplace(prefix + name + ".weight", matrix(hidden, hidden, static_cast<float>(i) + 0.3F));
            // GQA fixture: two KV heads and four query heads.
            for (const auto* name : {"attn_k", "attn_v"})
                weights.emplace(prefix + name + ".weight", matrix(16, hidden, static_cast<float>(i) + 0.4F));
            for (const auto* name : {"ffn_gate", "ffn_up"})
                weights.emplace(prefix + name + ".weight", matrix(64, hidden, static_cast<float>(i) + 0.5F));
            weights.emplace(prefix + "ffn_down.weight", matrix(hidden, 64, static_cast<float>(i) + 0.6F));
        }
        mx::save_gguf(path.string(), std::move(weights), {
            {"general.architecture", std::string("llama")},
            {"llama.embedding_length", mx::array(hidden)},
            {"llama.block_count", mx::array(2)},
            {"llama.attention.head_count", mx::array(4)},
            {"llama.attention.head_count_kv", mx::array(2)},
            {"llama.rope.dimension_count", mx::array(8)},
            {"llama.context_length", mx::array(64)},
            {"llama.attention.layer_norm_rms_epsilon", mx::array(1e-5F)},
            {"llama.rope.freq_base", mx::array(10000.0F)},
        });
    }
    ~Fixture() { std::error_code error; std::filesystem::remove(path, error); }
    Fixture(const Fixture&) = delete;
    Fixture& operator=(const Fixture&) = delete;
};

void q4_k_reader() {
    const auto unique = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto path = std::filesystem::temp_directory_path() / ("ime-core-q4-k-" + std::to_string(unique) + ".gguf");
    struct RemoveFile {
        std::filesystem::path path;
        ~RemoveFile() { std::error_code error; std::filesystem::remove(path, error); }
    } cleanup{path};
    std::ofstream file(path, std::ios::binary);
    const auto integer = [&file](std::uint64_t value, unsigned width) {
        for (unsigned i = 0; i < width; ++i) file.put(static_cast<char>((value >> (8 * i)) & 0xffU));
    };
    file.write("GGUF", 4); integer(3, 4); integer(1, 8); integer(0, 8);
    const std::string name = "probe.weight";
    integer(name.size(), 8); file.write(name.data(), static_cast<std::streamsize>(name.size()));
    integer(2, 4); integer(256, 8); integer(1, 8); integer(12, 4); integer(0, 8);
    const auto position = static_cast<std::uint64_t>(file.tellp());
    for (std::uint64_t i = position; i % 32 != 0; ++i) file.put(0);
    // Q4_K: d=1, dmin=0.5, distinct scales/mins, q=3 throughout.
    integer(0x3c00, 2); integer(0x3800, 2);
    for (const unsigned byte : {1U, 2U, 3U, 4U, 8U, 7U, 6U, 5U, 0x45U, 0x36U, 0x27U, 0x18U})
        file.put(static_cast<char>(byte));
    for (int i = 0; i < 128; ++i) file.put(0x33);
    file.close();
    const auto loaded = mx::load_gguf(path.string(), mx::Device{mx::Device::cpu});
    const auto values = mx::astype(loaded.first.at(name), mx::float32, mx::Device{mx::Device::cpu});
    mx::eval(values);
    require(values.size() == 256, "invalid Q4_K fixture size");
    const auto* data = values.data<float>();
    for (int i = 0; i < 256; ++i)
        require(data[i] == 3.0F * static_cast<float>(i / 32 + 1) - 0.5F * static_cast<float>(8 - i / 32),
            "Q4_K decoder reused the first sub-block scale/min");
}

void equal_logits(std::span<const float> a, std::span<const float> b) {
    require(a.size() == b.size() && !a.empty(), "incorrect logit size");
    for (std::size_t i = 0; i < a.size(); ++i)
        require(std::isfinite(a[i]) && std::isfinite(b[i]) && std::abs(a[i] - b[i]) < 0.015F,
            "incremental/rewound logits differ from full decode");
}

void run(const Fixture& fixture, bool gpu) {
    const auto paths = std::make_shared<CorePaths>(fixture.path, IME_CORE_TEST_TABLES_DIR,
        std::filesystem::path{}, 64, 2, gpu ? -2 : 0);
    const auto model = std::make_shared<MlxModel>(paths);
    require(model->vocabulary_size() == 18546, "vocabulary rows changed");
    require(model->runtime_info().gpu_offload == gpu, "unexpected inference device");
    MlxBackend full(model), incremental(model), other(model);
    const std::vector<int> prompt{1, 300, 301, 3, 500, 501};
    full.decode(prompt, 0);
    const std::vector<float> expected(full.logits().begin(), full.logits().end());
    incremental.decode(std::span(prompt).first(3), 0);
    incremental.decode(std::span(prompt).subspan(3), 3);
    equal_logits(expected, incremental.logits());
    incremental.ready();
    equal_logits(expected, incremental.logits());
    require(incremental.rewind(3), "failed to trim MLX cache");
    incremental.decode(std::span(prompt).subspan(3), 3);
    equal_logits(expected, incremental.logits());
    require(!incremental.rewind(100), "rewound beyond valid state");
    other.decode(prompt, 0);
    equal_logits(expected, other.logits());
    incremental.clear();
    bool invalid = false;
    const int bad = -1;
    try { incremental.decode(std::span(&bad, 1), 0); } catch (const std::invalid_argument&) { invalid = true; }
    require(invalid, "invalid token accepted");
    incremental.decode(prompt, 0);
    equal_logits(expected, incremental.logits());

    // Sessions can be scheduled on different service worker threads. In
    // particular Metal must not retain a thread-local command encoder.
    std::exception_ptr first_error, second_error;
    std::jthread first_worker([&] {
        try { full.clear(); full.decode(prompt, 0); equal_logits(expected, full.logits()); }
        catch (...) { first_error = std::current_exception(); }
    });
    std::jthread second_worker([&] {
        try { other.clear(); other.decode(prompt, 0); equal_logits(expected, other.logits()); }
        catch (...) { second_error = std::current_exception(); }
    });
    first_worker.join(); second_worker.join();
    if (first_error) std::rethrow_exception(first_error);
    if (second_error) std::rethrow_exception(second_error);

    CoreConfig config;
    config.model_path = fixture.path;
    config.tables_dir = IME_CORE_TEST_TABLES_DIR;
    config.context_length = 64;
    config.gpu_layers = gpu ? -2 : 0;
    config.logger = std::make_shared<QuietLogger>();
    CoreState<MlxBackend> state(config);
    auto engine = state.create_engine();
    const std::vector<llavon::ime::core::internal::PaddingEntry> readings{{false, 0, u"ㄋㄧˇ"}, {false, 0, u"ㄏㄠˇ"}};
    const auto first = engine.predict(u"", readings);
    const auto second = engine.predict(u"", readings);
    require(first.size() == 2 && second.size() == 2, "MLX IME segment count changed");
    for (std::size_t i = 0; i < first.size(); ++i) {
        require(!first[i].candidates.empty() && first[i].candidates.front().first == second[i].candidates.front().first,
            "cache changed MLX candidate ranking");
        float sum = 0;
        for (const auto& [character, probability] : first[i].candidates) {
            (void)character;
            require(std::isfinite(probability) && probability > 0, "invalid masked probability");
            sum += probability;
        }
        require(std::abs(sum - 1.0F) < 1e-5F, "masked probabilities do not sum to one");
    }
    const auto alias = engine.predict(u"", {{false, 0, u"ㄟ "}});
    require(alias.size() == 1 && alias.front().candidates.empty(), "MLX alias changed candidate character");
}

template<BackendPolicy Reference>
void installed_model(const std::filesystem::path& path, bool gpu) {
    CoreConfig config;
    config.model_path = path;
    config.tables_dir = IME_CORE_TEST_TABLES_DIR;
    config.context_length = 512;
    config.gpu_layers = gpu ? -2 : 0;
    config.logger = std::make_shared<QuietLogger>();
    CoreState<MlxBackend> state(config);
    require(state.runtime_info().gpu_offload == gpu, "installed model used an unexpected device");
    auto engine = state.create_engine();
    std::optional<PredictionEngine<Reference>> reference;
    auto reference_config = config;
    reference_config.gpu_layers = 0;
    CoreState<Reference> baseline(reference_config);
    reference.emplace(baseline.create_engine());
    const auto compare = [&](const auto& actual, const auto& readings, const std::u16string& context = u"") {
        if (!reference) return;
        const auto expected = reference->predict(context, readings);
        require(expected.size() == actual.size(), "backend segment counts differ");
        for (std::size_t i = 0; i < actual.size(); ++i) {
            require(actual[i].candidates.size() == expected[i].candidates.size(), "backend candidate masks differ");
            if (actual[i].candidates.empty()) continue;
            if (actual[i].candidates.front().first != expected[i].candidates.front().first) {
                std::cout << "[reference]";
                for (std::size_t j = 0; j < std::min<std::size_t>(3, expected[i].candidates.size()); ++j)
                    std::cout << ' ' << utf8::utf32to8(std::u32string(1, expected[i].candidates[j].first))
                              << ':' << expected[i].candidates[j].second;
                std::cout << '\n';
            }
            require(actual[i].candidates.front().first == expected[i].candidates.front().first,
                "MLX and reference candidate ranking differ");
            for (const auto& [character, probability] : actual[i].candidates) {
                const auto found = std::ranges::find_if(expected[i].candidates,
                    [character](const auto& item) { return item.first == character; });
                require(found != expected[i].candidates.end() && std::abs(probability - found->second) < 0.025F,
                    "MLX and reference masked probabilities differ");
            }
        }
    };
    const std::vector<llavon::ime::core::internal::PaddingEntry> pair{{false, 0, u"ㄋㄧˇ"}, {false, 0, u"ㄏㄠˇ"}};
    for (int repeat = 0; repeat < 2; ++repeat) {
        const auto answer = engine.predict(u"", pair);
        require(answer.size() == 2 && !answer[0].candidates.empty() && !answer[1].candidates.empty(),
            "installed MLX model returned no complete prediction");
        for (const auto& segment : answer) {
            std::cout << "[model]";
            for (std::size_t i = 0; i < std::min<std::size_t>(3, segment.candidates.size()); ++i)
                std::cout << ' ' << utf8::utf32to8(std::u32string(1, segment.candidates[i].first))
                          << ':' << segment.candidates[i].second;
            std::cout << '\n';
        }
        require(answer[1].candidates.front().first == U'好',
            "installed MLX model failed the conventional phrase regression");
        compare(answer, pair);
    }
    const std::vector<llavon::ime::core::internal::PaddingEntry> aliases{{false, 0, u"ㄟ "}, {false, 0, u"ㄋㄜ "}};
    const auto particle = engine.predict(u"", aliases);
    require(particle.size() == 2 && particle[0].candidates.empty() &&
        !particle[1].candidates.empty() && particle[1].candidates.front().first == U'呢', "installed-model aliases regressed");
    compare(particle, aliases);
    const std::vector<llavon::ime::core::internal::PaddingEntry> reading{{false, 0, u"ㄟˋ"}};
    const auto fourth = engine.predict(u"", reading);
    require(fourth.size() == 1 && !fourth[0].candidates.empty() && fourth[0].candidates.front().first == U'欸',
        "installed-model fourth-tone candidate changed");
    compare(fourth, reading);
    const std::vector<llavon::ime::core::internal::PaddingEntry> singleton{{false, 0, u"ㄋㄧˇ"}};
    compare(engine.predict(u"", pair), pair);
    compare(engine.predict(u"", singleton), singleton);
    compare(engine.predict(u"", singleton), singleton);
    const std::u16string context = u"今天天氣";
    compare(engine.predict(context, pair), pair, context);
    const std::vector<llavon::ime::core::internal::PaddingEntry> chosen{{true, U'你', {}}, {false, 0, u"ㄏㄠˇ"}};
    compare(engine.predict(u"", chosen), chosen);
    std::cout << "[ok] native MLX and CPU reference agree on ranking and masked probabilities\n";
    std::cout << "[ok] installed Q4_K_M GGUF executes through native MLX and common IME templates\n";
}
} // namespace

int main(int argc, char** argv) {
    try {
        bool require_metal = false;
        std::filesystem::path model;
        for (int i = 1; i < argc; ++i) {
            const std::string argument(argv[i]);
            if (argument == "--require-metal") require_metal = true;
            else if (argument == "--model" && i + 1 < argc) model = argv[++i];
            else throw std::invalid_argument("usage: ime-core-mlx-backend-tests [--require-metal] [--model MODEL.gguf]");
        }
        q4_k_reader();
        const Fixture fixture;
        run(fixture, false);
        const auto devices = MlxModel::enumerate_devices();
        const bool metal = std::ranges::any_of(devices, [](const auto& d) { return d.backend == InferenceBackend::metal; });
        require(!require_metal || metal, "MLX Metal device required but unavailable");
        if (metal) run(fixture, true);
        else std::cout << "[skip] MLX Metal device unavailable; CPU path verified\n";
        if (!model.empty()) {
            installed_model<SelectedBackend>(model, false);
            if (metal) installed_model<SelectedBackend>(model, true);
        }
        std::cout << "[ok] native MLX GGUF, forward, GQA, KV rewind, warmup and IME templates\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "[FAIL] MLX: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}

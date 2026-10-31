#include "backend_logits_tests.hpp"
#include "bopomofo/bopomofo.hpp"
#include "engine/llama_backend.hpp"
#include "engine/prediction_engine.hpp"
#include "engine/tokenizer.hpp"

#include <ime-core/logger.hpp>

#include <cmath>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {
using namespace llavon::ime::core;
using namespace llavon::ime::core::internal;

class QuietLogger final : public Logger {
public:
    using Logger::log;
    void log(std::string) noexcept override {}
    void log(MessageFactory) noexcept override {}
};

void run(const std::filesystem::path& model_path, bool gpu) {
    const auto paths = std::make_shared<CorePaths>(model_path, IME_CORE_TEST_TABLES_DIR,
        std::filesystem::path{}, 128, 4, gpu ? -2 : 0, InferenceDeviceSelection{}, 4);
    const auto model = std::make_shared<LlamaModel>(paths);
    if (gpu && !model->runtime_info().gpu_offload) {
        throw std::runtime_error("requested GPU backend is unavailable");
    }
    // Quantized CPU matrix/vector kernels can differ more than FP16 kernels.
    // Candidate probabilities are checked separately below with a tighter bound.
    tests::verify_batch_logits(*model, 0.25F);
    const auto tokenizer = std::make_shared<Tokenizer>(*paths);
    const auto family = std::make_shared<HanziMapEngine>(*paths);
    const auto logger = std::make_shared<QuietLogger>();
    PredictionEngine engine(model->create_session(), tokenizer, family, logger);
    using Entry = llavon::ime::core::internal::PaddingEntry;
    std::vector<Entry> padding;
    for (const auto& reading : {u"ㄋㄧˇ", u"ㄏㄠˇ", u"ㄕˋ", u"ㄐㄧㄝˋ", u"ㄋㄧˇ", u"ㄏㄠˇ"}) {
        padding.push_back({false, 0, reading});
        PredictionEngine reference(model->create_session(), tokenizer, family, logger);
        const auto expected = reference.predict(u"", padding);
        const auto actual = engine.predict(u"", padding);
        for (std::size_t index = 0; index < actual.size(); ++index) {
            if (actual[index].candidates.size() != expected[index].candidates.size()) {
                throw std::runtime_error("speculation changed candidate mask");
            }
            for (const auto& [character, probability] : actual[index].candidates) {
                const auto found = std::ranges::find(expected[index].candidates, character,
                    &std::pair<char32_t, float>::first);
                if (found == expected[index].candidates.end() ||
                    std::abs(probability - found->second) > 0.025F) {
                    throw std::runtime_error("speculative probabilities differ from AR");
                }
            }
        }
    }
}
} // namespace

int main(int argc, char** argv) {
    if ((argc != 2 && argc != 3) || (argc == 3 && std::string_view(argv[2]) != "--gpu")) {
        std::cerr << "usage: ime-core-llama-backend-tests MODEL [--gpu]\n";
        return EXIT_FAILURE;
    }
    try {
        run(argv[1], argc == 3 && std::string_view(argv[2]) == "--gpu");
        std::cout << "[ok] llama batch logits/cache/speculative verification\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "[FAIL] " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}

#include <ime-core/core.hpp>
#include <ime-core/logger.hpp>

#include <chrono>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace {
using namespace llavon::ime::core;

class QuietLogger final : public Logger {
public:
    using Logger::log;
    void log(std::string) noexcept override {}
    void log(MessageFactory) noexcept override {}
};

class TemporaryDirectory final {
public:
    TemporaryDirectory()
        : path_(std::filesystem::temp_directory_path() /
                ("ime-core-startup-" + std::to_string(
                    std::chrono::steady_clock::now().time_since_epoch().count()))) {
        if (!std::filesystem::create_directory(path_)) {
            throw std::runtime_error("unable to create startup test directory");
        }
    }

    ~TemporaryDirectory() {
        std::error_code ignored;
        std::filesystem::remove_all(path_, ignored);
    }

    TemporaryDirectory(const TemporaryDirectory&) = delete;
    TemporaryDirectory& operator=(const TemporaryDirectory&) = delete;

    const std::filesystem::path& path() const noexcept { return path_; }

private:
    std::filesystem::path path_;
};

void expect_startup_failure(CoreConfig config) {
    try {
        const Core core(std::move(config));
    } catch (const std::exception&) {
        return;
    }
    throw std::runtime_error("Core accepted invalid startup assets");
}

void run(const std::filesystem::path& model, const std::filesystem::path& tables) {
    CoreConfig config;
    config.model_path = model;
    config.tables_dir = tables;
    config.context_length = 128;
    config.threads = 4;
    config.gpu_layers = 0;
    config.inference_provider = InferenceProvider::llama_cpp;
    config.inference_device.backend = InferenceBackend::cpu;
    config.logger = std::make_shared<QuietLogger>();

    // A backend failure must join the independent table-loading task safely.
    auto invalid_model = config;
    invalid_model.model_path = tables / "tokens" / "chars.json";
    expect_startup_failure(std::move(invalid_model));

    // All required files exist, so this fails during asynchronous JSON parsing
    // rather than the synchronous path validation.
    const TemporaryDirectory temporary;
    const auto invalid_tables = temporary.path() / "tables";
    std::filesystem::copy(tables, invalid_tables, std::filesystem::copy_options::recursive);
    {
        std::ofstream corrupt(invalid_tables / "tokens" / "chars.json", std::ios::trunc);
        corrupt.exceptions(std::ios::badbit | std::ios::failbit);
        corrupt << "{";
    }
    auto invalid_table_config = config;
    invalid_table_config.tables_dir = invalid_tables;
    expect_startup_failure(std::move(invalid_table_config));

    // Failed startup must leave backend globals reusable. A constructed Core
    // must have both its model and tables ready for the first prediction.
    const Core core(std::move(config));
    const auto session = core.create_session();
    const std::vector<PaddingEntry> padding{
        {.bopomofo = u"ㄋㄧˇ"},
        {.bopomofo = u"ㄏㄠˇ"},
    };
    const auto predictions = session->predict(u"", padding);
    if (predictions.size() != padding.size()) {
        throw std::runtime_error("first prediction has the wrong size");
    }
    for (const auto& prediction : predictions) {
        if (prediction.candidates.empty()) {
            throw std::runtime_error("first prediction has no candidates");
        }
    }
}
} // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "usage: ime-core-startup-tests MODEL TABLES\n";
        return EXIT_FAILURE;
    }
    try {
        run(argv[1], argv[2]);
        std::cout << "[ok] Core startup failure recovery and first prediction\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "[FAIL] " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}

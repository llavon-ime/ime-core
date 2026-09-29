#include <ime-core/core.hpp>
#include <ime-core/logger.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace core = llavon::ime::core;
using Clock = std::chrono::steady_clock;
using namespace std::chrono_literals;

class TimingLogger final : public core::Logger {
public:
    void log(std::string message) noexcept override {
        // Collect on the inference thread; flush only after the measurement.
        try { messages.push_back(std::move(message)); } catch (...) {}
    }
    void log(MessageFactory factory) noexcept override {
        try { factories.push_back(std::move(factory)); } catch (...) {}
    }
    void flush() {
        for (const auto& message : messages) std::cerr << message << '\n';
        messages.clear();
        // Logger factories must execute on a worker, outside the measured call.
        std::jthread worker([pending = std::move(factories)] {
            for (const auto& factory : pending) std::cerr << factory() << '\n';
        });
        factories.clear();
    }
private:
    std::vector<std::string> messages;
    std::vector<MessageFactory> factories;
};

int main(int argc, char** argv) {
    const bool load_only = argc == 5 && std::string_view(argv[4]) == "--load-only";
    if (argc != 4 && !load_only) {
        std::cerr << "usage: ime-core-latency-check MODEL TABLES VULKAN_DEVICE_ID [--load-only]\n";
        return 2;
    }
    try {
        std::cout << std::unitbuf;
        auto logger = std::make_shared<TimingLogger>();
        core::CoreConfig config;
        config.model_path = argv[1];
        config.tables_dir = argv[2];
        config.inference_device = {core::InferenceBackend::vulkan, argv[3]};
        config.logger = logger;
        const auto load_begin = Clock::now();
        core::Core model(std::move(config));
        const auto runtime = model.inference_runtime_info();
        if (!runtime.gpu_offload || runtime.device.device_id != argv[3]) return 3;
        std::cout << "DEVICE," << runtime.device.description << '\n'
                  << "LOAD_MS," << std::chrono::duration<double, std::milli>(Clock::now()-load_begin).count() << '\n';
        if (load_only) return 0;
        const std::vector<int> lengths{0,1,2,3,4,5,6,7,8,9,12,15,16,24,31,32,47,63,95,127,191,255,300,360};
        std::vector<double> times;
        for (int session_index = 0; session_index < 2; ++session_index) {
            auto session = model.create_session();
            session->ready();
            logger->flush();
            for (std::size_t request = 0; request < lengths.size(); ++request) {
                // Every sample counts, including the first and resumed requests.
                std::this_thread::sleep_for(request % 12 == 0 ? 2300ms : 250ms);
                const std::u16string context(static_cast<std::size_t>(lengths[request]),
                                             request % 2 == 0 ? u'天' : u'好');
                const std::vector<core::PaddingEntry> padding{{.bopomofo = u"ㄋㄧˇ"}};
                const auto begin = Clock::now();
                const auto result = session->predict(context, padding);
                const double ms = std::chrono::duration<double, std::milli>(Clock::now()-begin).count();
                if (result.size() != padding.size() || result.front().candidates.empty()) return 4;
                times.push_back(ms);
                std::cout << "SAMPLE," << session_index << ',' << request << ',' << lengths[request] << ',' << ms << '\n';
                for (const auto& [character, probability] : result.front().candidates) {
                    if (!std::isfinite(probability) || probability < 0 || probability > 1) return 5;
                    std::cout << "CANDIDATE," << session_index << ',' << request << ','
                              << static_cast<std::uint32_t>(character) << ',' << probability << '\n';
                }
                logger->flush();
            }
        }
        std::ranges::sort(times);
        std::cout << "SUMMARY,count=" << times.size() << ",median_ms=" << times[times.size()/2]
                  << ",max_ms=" << times.back() << '\n';
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

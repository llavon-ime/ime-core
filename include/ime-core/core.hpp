#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace llavon::ime::core {

struct CoreConfig {
    std::filesystem::path model_path;
    std::filesystem::path tables_dir;
    std::uint32_t context_length = 0;
    std::uint32_t threads = 8;
    int gpu_layers = -2;
};

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

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace llavon::ime::core

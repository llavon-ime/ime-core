#pragma once

#include <ime-core/core.hpp>

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace llavon::ime::core {

// Gives offline consumers the same tokenization and candidate tables as inference.
class EncodingTables final {
public:
    explicit EncodingTables(const std::filesystem::path& tables_directory);
    ~EncodingTables();

    EncodingTables(EncodingTables&&) noexcept;
    EncodingTables& operator=(EncodingTables&&) noexcept;

    EncodingTables(const EncodingTables&) = delete;
    EncodingTables& operator=(const EncodingTables&) = delete;

    std::vector<std::int64_t> tokenize(
        const std::u16string& context,
        const std::vector<PaddingEntry>& padding) const;
    std::int64_t token_for_character(char32_t character) const;
    std::vector<std::pair<char32_t, std::int64_t>> candidates_for_reading(
        const std::u16string& reading) const;
    std::int64_t pad_token_id() const;
    bool tokens_fit_vocabulary(std::int64_t vocabulary_size) const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace llavon::ime::core

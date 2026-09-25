#include <ime-core/encoding_tables.hpp>

#include "bopomofo/bopomofo.hpp"
#include "engine/tokenizer.hpp"

#include <utility>

namespace llavon::ime::core {

class EncodingTables::Impl final {
public:
    explicit Impl(const std::filesystem::path& tables_directory)
        : tokenizer(tables_directory), hanzi_map(tables_directory) {}

    internal::Tokenizer tokenizer;
    internal::HanziMapEngine hanzi_map;
};

EncodingTables::EncodingTables(const std::filesystem::path& tables_directory)
    : impl_(std::make_unique<Impl>(tables_directory)) {}
EncodingTables::~EncodingTables() = default;
EncodingTables::EncodingTables(EncodingTables&&) noexcept = default;
EncodingTables& EncodingTables::operator=(EncodingTables&&) noexcept = default;

std::vector<std::int64_t> EncodingTables::tokenize(
    const std::u16string& context,
    const std::vector<PaddingEntry>& padding) const {
    std::vector<internal::PaddingEntry> internal_padding;
    internal_padding.reserve(padding.size());
    for (const auto& entry : padding) {
        internal_padding.push_back({entry.chosen, entry.chosen_char, entry.bopomofo});
    }
    const auto tokens = impl_->tokenizer.tokenize(context, internal_padding);
    return {tokens.begin(), tokens.end()};
}

std::int64_t EncodingTables::token_for_character(char32_t character) const {
    const int token = impl_->tokenizer.map_char(character);
    return token == -1 ? impl_->tokenizer.unknown_token_id() : token;
}

std::vector<std::pair<char32_t, std::int64_t>>
EncodingTables::candidates_for_reading(const std::u16string& reading) const {
    std::vector<std::pair<char32_t, std::int64_t>> result;
    for (const char32_t character : impl_->hanzi_map.lookup_all(reading)) {
        const int token = impl_->tokenizer.map_char(character);
        if (token != -1) result.emplace_back(character, token);
    }
    return result;
}

std::int64_t EncodingTables::pad_token_id() const {
    return impl_->tokenizer.pad_token_id();
}

bool EncodingTables::tokens_fit_vocabulary(std::int64_t vocabulary_size) const {
    return impl_->tokenizer.tokens_fit_vocabulary(vocabulary_size);
}

}  // namespace llavon::ime::core

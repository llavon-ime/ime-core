#pragma once

#include <utf8/cpp20.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <rfl/json.hpp>
#include <ranges>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <variant>
#include <vector>

#include "../utils/core_paths.hpp"

namespace llavon::ime::core::internal {

struct PaddingEntry {
    bool is_chosen;
    char32_t chosen_char = 0;
    std::u16string bpmf;
};

class Tokenizer {
    std::unordered_map<std::string, int> char_table;
    std::unordered_map<std::string, int> latin_table;
    std::unordered_map<std::string, int> special_table;
    std::unordered_map<std::string, int> bpmf_table;
    std::unordered_map<int, std::string> inverse_table;
    std::unordered_set<int> boundary_tokens;

    template <typename Table>
    void add_inverse_tokens(const Table& table) {
        for (const auto& [text, token] : table) {
            const auto [existing, inserted] = inverse_table.emplace(token, text);
            if (!inserted && existing->second != text) {
                throw std::logic_error("duplicate token id in tokenizer tables");
            }
        }
    }

public:
    explicit Tokenizer(const std::filesystem::path& tables_dir) {
        const auto tokens = tables_dir / "tokens";
        char_table =
            rfl::json::load<std::unordered_map<std::string, int>>((tokens / "chars.json").string()).value();
        latin_table =
            rfl::json::load<std::unordered_map<std::string, int>>((tokens / "latin.json").string()).value();
        special_table =
            rfl::json::load<std::unordered_map<std::string, int>>((tokens / "special_tokens.json").string())
                .value();
        bpmf_table =
            rfl::json::load<std::unordered_map<std::string, int>>((tokens / "bpmf.json").string()).value();

        add_inverse_tokens(char_table);
        add_inverse_tokens(latin_table);
        add_inverse_tokens(special_table);
        add_inverse_tokens(bpmf_table);

        boundary_tokens.insert(special_table.at("<SP>"));
        for (const auto c : std::u32string_view(U"，。、；：！？「」『』（）《》,.;:!?()")) {
            if (const auto token = map_char(c); token >= 0) boundary_tokens.insert(token);
        }
    }

    explicit Tokenizer(const CorePaths& paths) : Tokenizer(paths.tables_dir()) {}

    int pad_token_id() const { return special_table.at("<PAD>"); }
    int unknown_token_id() const { return special_table.at("<UNK>"); }

    bool tokens_fit_vocabulary(std::int64_t vocabulary_size) const {
        if (vocabulary_size <= 0) return false;
        const auto fits = [vocabulary_size](const auto& table) {
            return std::ranges::all_of(table | std::views::values,
                [vocabulary_size](int id) { return id >= 0 && id < vocabulary_size; });
        };
        return fits(char_table) && fits(latin_table) &&
               fits(special_table) && fits(bpmf_table);
    }

    static bool is_alpha(char32_t c) { return (c >= U'a' && c <= U'z') || (c >= U'A' && c <= U'Z'); }
    static bool is_digit(char32_t c) { return (c >= U'0' && c <= U'9'); }
    static bool is_latin_char(char32_t c) { return is_alpha(c) || is_digit(c) || c == U'-' || c == U'_' || c == U'+'; }
    static char to_lower(char32_t c) { return static_cast<char>((c >= U'A' && c <= U'Z') ? c ^ 0x20 : c); }

    static void remove_leading_unknown_context_tokens(std::vector<int>& context_tokens, int unknown_token) {
        size_t first_kept = 0;
        while (first_kept < context_tokens.size() && context_tokens[first_kept] == unknown_token) first_kept++;
        if (first_kept == 0) return;

        context_tokens.erase(context_tokens.begin(), context_tokens.begin() + static_cast<std::ptrdiff_t>(first_kept));
    }

    int map_char(char32_t c) const {
        std::string s;
        utf8::append(c, s);
        if (char_table.contains(s)) return char_table.at(s);
        return -1;
    }

    int bos_token_id() const { return special_table.at("<BOS>"); }

    bool is_boundary(int token) const { return boundary_tokens.contains(token); }

    std::vector<int> tokenize(const std::u16string& context16,
                              const std::vector<PaddingEntry>& padding) const {
        std::vector<int> res{bos_token_id()};
        const auto context = tokenize_context(context16);
        res.insert(res.end(), context.begin(), context.end());
        append_padding(res, padding);
        return res;
    }

    std::vector<int> tokenize_context(const std::u16string& context16) const {
        std::vector<int> context_tokens;
        std::string context8 = utf8::utf16to8(context16);
        std::u32string context = utf8::utf8to32(context8);
        for (size_t i = 0; i < context.size(); i++) {
            std::string s;
            utf8::append(context[i], s);
            if (context[i] == U' ' || context[i] == U'\n') {
                context_tokens.push_back(special_table.at("<SP>"));
            } else if (char_table.contains(s)) {
                context_tokens.push_back(char_table.at(s));
            } else if (is_latin_char(context[i])) {
                std::string str;
                for (; i < context.size(); i++) {
                    if (is_latin_char(context[i])) {
                        str.push_back(to_lower(context[i]));
                    } else {
                        i--;
                        break;
                    }
                }
                if (latin_table.contains(str)) {
                    context_tokens.push_back(latin_table.at(str));
                } else {
                    context_tokens.push_back(special_table.at("<LATIN>"));
                }
            } else {
                context_tokens.push_back(special_table.at("<UNK>"));
            }
        }
        remove_leading_unknown_context_tokens(context_tokens, special_table.at("<UNK>"));
        return context_tokens;
    }

    void append_padding(std::vector<int>& res, const std::vector<PaddingEntry>& padding) const {
        for (auto& entry : padding) {
            if (entry.is_chosen) {
                std::string s;
                utf8::append(entry.chosen_char, s);
                if (char_table.contains(s)) {
                    res.push_back(char_table.at(s));
                } else {
                    res.push_back(special_table.at("<UNK>"));
                }
            } else {
                std::string s8 = "<" + utf8::utf16to8(entry.bpmf) + ">";
                if (s8 == "<ㄋㄜ >") s8 = "<ㄋㄜ˙>";
                if (s8 == "<ㄌㄜ >") s8 = "<ㄌㄜ˙>";
                // Alias only the model token; retain the original reading for
                // candidate lookup so first-tone ㄟ still produces ㄟ, not 欸.
                if (s8 == "<ㄟ >") s8 = "<ㄟˋ>";
                if (!bpmf_table.contains(s8)) {
                    throw std::logic_error("invalid bpmf: " + s8);
                }
                res.push_back(bpmf_table.at(s8));
            }
        }
        res.push_back(special_table.at("<SEP>"));
    }

    std::u16string detokenize_context(std::span<const int> tokens,
                                      std::size_t padding_count) const {
        // tokenize() emits BOS, context, padding, SEP in that order. Decode
        // only the context range so the result is directly comparable with
        // the text acquired from TSF.
        if (tokens.size() < padding_count + 2) return {};

        const auto first = tokens.begin() + 1;
        const auto last = tokens.end() - static_cast<std::ptrdiff_t>(padding_count) - 1;
        std::string text;
        for (auto token = first; token != last; ++token) {
            if (*token == special_table.at("<SP>")) {
                text.push_back(' ');
                continue;
            }
            const auto found = inverse_table.find(*token);
            if (found != inverse_table.end()) text += found->second;
        }
        return utf8::utf8to16(text);
    }
};

}  // namespace llavon::ime::core::internal

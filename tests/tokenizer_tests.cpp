#include "bopomofo/bopomofo.hpp"
#include "engine/tokenizer.hpp"

#include <algorithm>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

int main() {
    using namespace llavon::ime::core::internal;
    try {
        const Tokenizer tokenizer(IME_CORE_TEST_TABLES_DIR);
        const HanziMapEngine candidates(IME_CORE_TEST_TABLES_DIR);
        for (const auto& [reading, alias] :
             std::vector<std::pair<std::u16string, std::u16string>>{
                 {u"ㄟ ", u"ㄟˋ"}, {u"ㄋㄜ ", u"ㄋㄜ˙"}, {u"ㄌㄜ ", u"ㄌㄜ˙"}}) {
            const std::vector<PaddingEntry> original{{false, 0, reading}};
            const std::vector<PaddingEntry> canonical{{false, 0, alias}};
            const auto tokens = tokenizer.tokenize(u"你好", original);
            if (tokens != tokenizer.tokenize(u"你好", canonical) || original.front().bpmf != reading) {
                std::cerr << "[FAIL] reading token alias or original reading changed\n";
                return EXIT_FAILURE;
            }
        }

        // The alias must not turn the original first-tone character into 欸.
        if (candidates.lookup_all(u"ㄟ ") != std::vector<char32_t>{U'ㄟ'} ||
            candidates.lookup_all(u"ㄟˋ") != std::vector<char32_t>{U'欸'}) {
            std::cerr << "[FAIL] first- and fourth-tone candidates changed\n";
            return EXIT_FAILURE;
        }

        const auto first_tone_zhi = candidates.lookup_all(u"ㄓ ");
        const auto third_tone_zhi = candidates.lookup_all(u"ㄓˇ");
        if (std::ranges::count(first_tone_zhi, U'脂') != 1 ||
            std::ranges::count(third_tone_zhi, U'脂') != 1) {
            std::cerr << "[FAIL] 脂 must support both first- and third-tone readings\n";
            return EXIT_FAILURE;
        }

        const std::vector<PaddingEntry> mixed{
            {false, 0, u"ㄟ "}, {false, 0, u"ㄅㄟ "}, {false, 0, u"ㄟˊ"}, {true, U'ㄟ', u"ㄟ "}};
        const auto tokens = tokenizer.tokenize(u"", mixed);
        if (tokens.size() != 6 ||
            tokens.at(1) != tokenizer.tokenize(u"", {{false, 0, u"ㄟˋ"}}).at(1) ||
            tokens.at(2) != tokenizer.tokenize(u"", {{false, 0, u"ㄅㄟ "}}).at(1) ||
            tokens.at(3) != tokenizer.tokenize(u"", {{false, 0, u"ㄟˊ"}}).at(1) ||
            tokens.at(4) != tokenizer.unknown_token_id() || tokens.at(1) == tokens.at(3)) {
            // A manually chosen ㄟ has no character token; preserve the existing
            // UNK behaviour rather than substituting its reading token.
            std::cerr << "[FAIL] alias affected another reading or a chosen character\n";
            return EXIT_FAILURE;
        }

        std::cout << "[ok] reading token aliases preserve readings and candidate characters\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "[FAIL] tokenizer: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}

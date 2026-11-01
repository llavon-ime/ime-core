#include "bopomofo/bopomofo.hpp"
#include "engine/prediction_engine.hpp"
#include "engine/tokenizer.hpp"

#include <ime-core/logger.hpp>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
using namespace llavon::ime::core;
using namespace llavon::ime::core::internal;
using TestPaddingEntry = llavon::ime::core::internal::PaddingEntry;

class QuietLogger final : public Logger {
public:
    using Logger::log;
    void log(std::string) noexcept override {}
    void log(MessageFactory) noexcept override {}
};

class FakeBackend;

struct FakeModel final : BackendModel {
    struct Decode { std::size_t position; std::vector<int> tokens; std::size_t logits_from; };
    std::vector<Decode> calls;
    std::vector<std::size_t> rewinds;
    int clears = 0;
    int warmups = 0;
    bool fail_decode = false;
    bool reject_rewind = false;
    std::function<void(std::span<const int>, std::span<float>)> score;
    InferenceRuntimeInfo runtime;
    const InferenceRuntimeInfo& runtime_info() const noexcept override { return runtime; }
    std::size_t vocabulary_size() const noexcept override { return 65536; }
    std::unique_ptr<BackendSession> create_session() override;
};

class FakeBackend final : public BackendSession {
public:
    explicit FakeBackend(std::shared_ptr<FakeModel> model) : model_(std::move(model)) {
        const Tokenizer tokenizer(IME_CORE_TEST_TABLES_DIR);
        ni_ = static_cast<std::size_t>(tokenizer.map_char(U'你'));
        hao_ = static_cast<std::size_t>(tokenizer.map_char(U'好'));
    }
    void ready() override { ++model_->warmups; }
    void clear() noexcept override { ++model_->clears; prefix_.clear(); values_.clear(); }
    bool rewind(std::size_t position) override {
        model_->rewinds.push_back(position);
        if (model_->reject_rewind) return false;
        if (position > prefix_.size()) throw std::logic_error("rewind exceeds cache");
        prefix_.resize(position);
        values_.clear();
        return true;
    }
    void decode(std::span<const int> tokens, std::size_t position) override {
        decode_with_logits(tokens, position, tokens.empty() ? 0 : tokens.size() - 1);
    }
    void decode_with_logits(
        std::span<const int> tokens, std::size_t position, std::size_t logits_from) override {
        if (position != prefix_.size()) throw std::logic_error("incorrect cache position");
        if (tokens.empty() || logits_from >= tokens.size()) throw std::invalid_argument("invalid logits extent");
        model_->calls.push_back({position, {tokens.begin(), tokens.end()}, logits_from});
        const auto vocabulary = model_->vocabulary_size();
        values_.assign((tokens.size() - logits_from) * vocabulary, 0);
        for (std::size_t index = 0; index < tokens.size(); ++index) {
            prefix_.push_back(tokens[index]);
            if (index < logits_from) continue;
            auto row = std::span(values_).subspan((index - logits_from) * vocabulary, vocabulary);
            row[ni_] = 10;
            row[hao_] = 9;
            if (model_->score) model_->score(prefix_, row);
        }
        if (std::exchange(model_->fail_decode, false)) throw std::runtime_error("partial decode failure");
    }
    std::span<const float> logits() const override {
        if (values_.empty()) throw std::logic_error("missing logits");
        return std::span(values_).last(model_->vocabulary_size());
    }
    std::span<const float> logits_at(std::size_t row) const override {
        if (row >= values_.size() / model_->vocabulary_size()) throw std::out_of_range("invalid logits row");
        return std::span(values_).subspan(row * model_->vocabulary_size(), model_->vocabulary_size());
    }
private:
    std::shared_ptr<FakeModel> model_;
    std::vector<float> values_;
    std::vector<int> prefix_;
    std::size_t ni_ = 0;
    std::size_t hao_ = 0;
};

std::unique_ptr<BackendSession> FakeModel::create_session() {
    return std::make_unique<FakeBackend>(
        std::static_pointer_cast<FakeModel>(shared_from_this()));
}

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void require_equal(const std::vector<PredictResult>& actual, const std::vector<PredictResult>& expected) {
    require(actual.size() == expected.size(), "prediction count differs from AR");
    for (std::size_t index = 0; index < actual.size(); ++index) {
        require(actual[index].candidates == expected[index].candidates,
            "candidate order or probabilities differ from AR");
    }
}

struct PredictionFixture {
    std::shared_ptr<Tokenizer> tokenizer = std::make_shared<Tokenizer>(IME_CORE_TEST_TABLES_DIR);
    std::shared_ptr<HanziMapEngine> family = std::make_shared<HanziMapEngine>(IME_CORE_TEST_TABLES_DIR);
    std::shared_ptr<QuietLogger> logger = std::make_shared<QuietLogger>();
    std::shared_ptr<FakeModel> model = std::make_shared<FakeModel>();
    PredictionEngine engine{model->create_session(), tokenizer, family, logger};

    std::vector<PredictResult> reference(
        const std::u16string& context, const std::vector<TestPaddingEntry>& padding) const {
        const auto reference_model = std::make_shared<FakeModel>();
        reference_model->score = model->score;
        // A fresh engine has no draft and follows the ordinary AR path.
        PredictionEngine reference_engine(reference_model->create_session(), tokenizer, family, logger);
        return reference_engine.predict(context, padding);
    }
};

void test_speculative_acceptance() {
    PredictionFixture fixture;
    std::vector<TestPaddingEntry> padding;
    for (std::size_t count = 1; count <= 8; ++count) {
        padding.push_back({false, 0, count % 2 == 0 ? u"ㄏㄠˇ" : u"ㄋㄧˇ"});
        const auto before = fixture.model->calls.size();
        require_equal(fixture.engine.predict(u"前文", padding), fixture.reference(u"前文", padding));
        require(fixture.model->calls.size() == before + 1, "accepted draft used multiple decodes");
        if (count > 1) {
            const auto& call = fixture.model->calls.back();
            require(call.logits_from == 1 && call.tokens.size() == count + 1,
                "append did not fuse the new reading, SEP and draft");
        }
    }
    const auto before = fixture.model->calls.size();
    require_equal(fixture.engine.predict(u"前文", padding), fixture.reference(u"前文", padding));
    require(fixture.model->calls.size() == before + 1 && fixture.model->calls.back().logits_from == 0,
        "unchanged prompt did not refresh verification rows");

    padding[1].bpmf = u"ㄋㄧˇ";
    const auto before_edit = fixture.model->calls.size();
    require_equal(fixture.engine.predict(u"前文", padding), fixture.reference(u"前文", padding));
    const auto& batch = fixture.model->calls[before_edit];
    require(batch.tokens.size() - batch.logits_from == 2,
        "draft outside the edited candidate set was fed to the model");
}

struct RejectionFixture : PredictionFixture {
    std::vector<int> tokens;
    std::vector<std::size_t> flips;

    RejectionFixture() {
        for (const auto character : family->lookup_all(u"ㄋㄧˇ")) {
            if (const auto token = tokenizer->map_char(character); token >= 0) tokens.push_back(token);
        }
        require(tokens.size() >= 2, "rejection fixture requires two candidates");
        const auto sep = tokenizer->tokenize(u"", {}).back();
        model->score = [this, sep](std::span<const int> prefix, std::span<float> logits) {
            std::ranges::fill(logits, 0.0F);
            std::uint32_t hash = 17;
            for (const auto token : prefix) hash = hash * 31 + static_cast<std::uint32_t>(token);
            const auto separator = std::ranges::find(prefix, sep);
            const auto step = static_cast<std::size_t>(prefix.end() - separator - 1);
            for (const auto token : tokens) {
                logits[static_cast<std::size_t>(token)] =
                    static_cast<float>((hash ^ static_cast<std::uint32_t>(token)) % 31) / 32.0F;
            }
            const auto preferred = tokens[std::ranges::contains(flips, step) ? 1 : 0];
            logits[static_cast<std::size_t>(preferred)] += 5.0F;
        };
    }
};

void test_speculative_rejection(bool reject_rewind) {
    // Row values depend on every input token, so retaining any row after a
    // rejected draft token changes probabilities even if its top-1 agrees.
    for (std::size_t mismatch = 0; mismatch < 4; ++mismatch) {
        RejectionFixture fixture;
        const std::vector<TestPaddingEntry> padding(5, {false, 0, u"ㄋㄧˇ"});
        (void)fixture.engine.predict(u"", padding);
        fixture.flips = {mismatch};
        fixture.model->reject_rewind = reject_rewind;
        const auto before = fixture.model->calls.size();
        require_equal(fixture.engine.predict(u"", padding), fixture.reference(u"", padding));
        const auto prompt_size = fixture.tokenizer->tokenize(u"", padding).size();
        require(fixture.model->rewinds.back() == prompt_size + mismatch,
            "rejected draft was rewound to the wrong position");
        require(fixture.model->calls.size() - before == 2 + (reject_rewind ? 1 : 0),
            "rejection did not resubmit the remaining draft in one batch");
        fixture.model->reject_rewind = false;
        const auto before_repeat = fixture.model->calls.size();
        require_equal(fixture.engine.predict(u"", padding), fixture.reference(u"", padding));
        require(fixture.model->calls.size() == before_repeat + 1,
            "rejection left stale draft or KV state");
    }
}

void test_speculative_resubmission() {
    RejectionFixture fixture;
    const std::vector<TestPaddingEntry> padding(6, {false, 0, u"ㄋㄧˇ"});
    (void)fixture.engine.predict(u"", padding);
    fixture.flips = {0, 2, 3};
    const auto before = fixture.model->calls.size();
    const auto rewinds = fixture.model->rewinds.size();
    require_equal(fixture.engine.predict(u"", padding), fixture.reference(u"", padding));
    require(fixture.model->calls.size() - before == 4,
        "each rejection did not resubmit the remaining draft");
    const auto prompt_size = fixture.tokenizer->tokenize(u"", padding).size();
    require(std::ranges::equal(std::span(fixture.model->rewinds).subspan(rewinds + 1),
                std::vector{prompt_size, prompt_size + 2, prompt_size + 3}),
        "resubmitted drafts were rewound to the wrong positions");
    const auto& last = fixture.model->calls.back();
    require(last.position == prompt_size + 3 && last.logits_from == 0 && last.tokens.size() == 2,
        "row-zero rejection did not resubmit the following draft");
}

void test_speculative_edits_and_failures() {
    PredictionFixture fixture;
    fixture.model->score = [](std::span<const int> prefix, std::span<float> logits) {
        std::uint32_t hash = 17;
        for (const auto token : prefix) hash = hash * 31 + static_cast<std::uint32_t>(token);
        for (std::size_t token = 0; token < logits.size(); ++token) {
            const auto mixed = hash ^ (static_cast<std::uint32_t>(token) * 2654435761U);
            logits[token] = static_cast<float>(mixed % 1021) / 128.0F;
        }
    };
    const std::vector<std::vector<TestPaddingEntry>> requests{
        {{false, 0, u"ㄋㄧˇ"}, {true, U'好', {}}, {false, 0, u"ㄟ "},
            {true, U'\U0010ffff', {}}, {false, 0, u"ㄋㄧˇ"}, {true, U'好', {}}},
        {{false, 0, u"ㄋㄧˇ"}, {true, U'好', {}}, {false, 0, u"ㄟ "},
            {true, U'\U0010ffff', {}}, {false, 0, u"ㄋㄧˇ"}, {false, 0, u"ㄏㄠˇ"}},
        {{true, U'好', {}}, {false, 0, u"ㄋㄧˇ"}, {false, 0, u"ㄏㄠˇ"}},
        {{false, 0, u"ㄋㄧˇ"}, {false, 0, u"ㄏㄠˇ"}},
        {}, {{true, U'好', {}}}, {{false, 0, u"ㄟ "}},
        {{false, 0, u"ㄋㄧˇ"}, {false, 0, u"ㄏㄠˇ"}, {false, 0, u"ㄋㄧˇ"}},
    };
    for (const auto& padding : requests) {
        for (const auto& context : {u"前文", u"修改"}) {
            require_equal(fixture.engine.predict(context, padding), fixture.reference(context, padding));
            require_equal(fixture.engine.predict(context, padding), fixture.reference(context, padding));
        }
    }

    const std::u16string context(180, u'前');
    fixture.engine.prefill(context);
    require_equal(fixture.engine.predict(context, requests.front()), fixture.reference(context, requests.front()));
    fixture.engine.prefill(context + u"文");
    require_equal(fixture.engine.predict(context + u"文", requests[1]), fixture.reference(context + u"文", requests[1]));

    const auto& padding = requests.back();
    (void)fixture.engine.predict(u"修改", padding);
    fixture.model->fail_decode = true;
    bool failed = false;
    try { (void)fixture.engine.predict(u"修改", padding); }
    catch (const std::runtime_error&) { failed = true; }
    require(failed, "failed verification was swallowed");
    const auto before = fixture.model->calls.size();
    require_equal(fixture.engine.predict(u"修改", padding), fixture.reference(u"修改", padding));
    require(fixture.model->calls.size() == before + padding.size() &&
        fixture.model->calls[before].position == 0, "failed verification retained its draft or cache");

    fixture.model->score = [](std::span<const int>, std::span<float> logits) {
        std::ranges::fill(logits, 0.0F);
    };
    const auto tied = fixture.engine.predict(u"", padding);
    require_equal(tied, fixture.reference(u"", padding));
    for (const auto& result : tied) {
        require(std::ranges::is_sorted(result.candidates, {}, [&](const auto& candidate) {
            return fixture.tokenizer->map_char(candidate.first);
        }), "tied probabilities were not ordered by token ID");
    }
}

void test() {
    const auto tokenizer = std::make_shared<Tokenizer>(IME_CORE_TEST_TABLES_DIR);
    const auto family = std::make_shared<HanziMapEngine>(IME_CORE_TEST_TABLES_DIR);
    const auto logger = std::make_shared<QuietLogger>();
    const auto model = std::make_shared<FakeModel>();
    PredictionEngine engine(model->create_session(), tokenizer, family, logger);
    engine.ready(); engine.ready();
    require(model->warmups == 1 && model->calls.empty(), "warmup touched prediction state");

    const std::vector<llavon::ime::core::internal::PaddingEntry> pair{{false, 0, u"ㄋㄧˇ"}, {false, 0, u"ㄏㄠˇ"}};
    const auto prompt = tokenizer->tokenize(u"", pair);
    auto results = engine.predict(u"", pair);
    require(results.size() == 2 && results[0].candidates.front().first == U'你' &&
        results[1].candidates.front().first == U'好', "masked ranking changed");
    require(model->calls.size() == 2 && model->calls[0].tokens == prompt &&
        model->calls[1].tokens == std::vector<int>{tokenizer->map_char(U'你')}, "incorrect decode batching");
    require(model->calls[1].position == prompt.size(), "incorrect guess position");

    (void)engine.predict(u"", pair);
    require(model->rewinds.back() == prompt.size() - 1 &&
        model->calls.size() == 3 && model->calls[2].logits_from == 0 &&
        model->calls[2].tokens == std::vector<int>{prompt.back(), tokenizer->map_char(U'你')},
        "repeat did not verify the draft with SEP in one decode");
    const std::vector<llavon::ime::core::internal::PaddingEntry> singleton{{false, 0, u"ㄋㄧˇ"}};
    (void)engine.predict(u"", singleton);
    const auto unchanged = model->calls.size();
    (void)engine.predict(u"", singleton);
    require(model->calls.size() == unchanged, "unchanged prompt was unnecessarily decoded");

    (void)engine.predict(u"新", singleton);
    require(model->calls.back().position == 1, "fresh context reused a stale suffix");
    model->reject_rewind = true;
    (void)engine.predict(u"", singleton);
    require(model->clears > 0 && model->calls.back().position == 0, "rewind rejection did not rebuild cache");
    model->reject_rewind = false;

    model->fail_decode = true;
    bool failed = false;
    try { (void)engine.predict(u"失敗", singleton); } catch (const std::runtime_error&) { failed = true; }
    require(failed, "decode failure was swallowed");
    (void)engine.predict(u"", singleton);
    require(model->calls.back().position == 0 && model->calls.back().tokens == tokenizer->tokenize(u"", singleton),
        "failed decode left reusable partial cache");

    const std::vector<llavon::ime::core::internal::PaddingEntry> chosen{{true, U'好', {}}, {false, 0, u"ㄋㄧˇ"}};
    results = engine.predict(u"", chosen);
    require(results[0].candidates.empty() && results[1].candidates.front().first == U'你', "manual choice was replaced");
    require(model->calls.back().tokens == std::vector<int>{tokenizer->map_char(U'好')}, "chosen token missing from continuation");

    const auto before_alias = model->calls.size();
    results = engine.predict(u"", {{false, 0, u"ㄟ "}});
    require(results.size() == 1 && results[0].candidates.empty() && model->calls.size() == before_alias,
        "token-only alias changed the original candidate family");
    require(model->calls.back().tokens.size() == 1, "final prediction was unnecessarily decoded");
}

std::u16string distinct_text(const Tokenizer& tokenizer, char32_t first, std::size_t length, std::size_t punctuation_every) {
    std::u16string text;
    for (auto c = first; text.size() < length; ++c) {
        if (punctuation_every != 0 && text.size() % punctuation_every == punctuation_every - 1) text.push_back(u'，');
        else if (tokenizer.map_char(c) >= 0 && !tokenizer.is_boundary(tokenizer.map_char(c)) && c <= 0xFFFF)
            text.push_back(static_cast<char16_t>(c));
    }
    return text;
}

std::size_t decoded_since(const FakeModel& model, std::size_t first) {
    std::size_t total = 0;
    for (auto index = first; index < model.calls.size(); ++index) total += model.calls[index].tokens.size();
    return total;
}

void test_window() {
    const auto tokenizer = std::make_shared<Tokenizer>(IME_CORE_TEST_TABLES_DIR);
    const auto family = std::make_shared<HanziMapEngine>(IME_CORE_TEST_TABLES_DIR);
    const auto logger = std::make_shared<QuietLogger>();
    const auto model = std::make_shared<FakeModel>();
    PredictionEngine engine(model->create_session(), tokenizer, family, logger);
    const std::vector<llavon::ime::core::internal::PaddingEntry> singleton{{false, 0, u"ㄋㄧˇ"}};
    const auto comma = tokenizer->map_char(U'，');

    const auto text = distinct_text(*tokenizer, U'一', 800, 23);
    (void)engine.predict(text.substr(0, 300), singleton);
    std::size_t refreshes = 0;
    for (std::size_t length = 301; length <= text.size(); ++length) {
        const auto visible = length > 384 ? text.substr(length - 384, 384) : text.substr(0, length);
        const auto before_prefill = model->calls.size();
        engine.prefill(visible);
        if (decoded_since(*model, before_prefill) > 1) {
            ++refreshes;
            const auto& refresh = model->calls.back();
            require(refresh.position == 1 && refresh.tokens.size() <= 128 && refresh.tokens.size() >= 32,
                "prefill refresh used an unexpected window");
            const auto context = tokenizer->tokenize_context(visible);
            const auto cut = context.end() - static_cast<std::ptrdiff_t>(refresh.tokens.size());
            require(*(cut - 1) == comma, "prefill refresh did not cut after punctuation");
        }
        const auto before_predict = model->calls.size();
        (void)engine.predict(visible, singleton);
        require(decoded_since(*model, before_predict) == 2, "prediction after prefill decoded context");
    }
    require(refreshes >= 3, "long context never refreshed its window");

    const auto before_repeat = model->calls.size();
    engine.prefill(text.substr(text.size() - 384, 384));
    require(model->calls.size() == before_repeat, "prefill discarded an aligned cache");

    const auto before_backspace = model->calls.size();
    (void)engine.predict(text.substr(text.size() - 384, 383), singleton);
    require(decoded_since(*model, before_backspace) == 2, "backspace rebuilt the window");

    const auto edited = distinct_text(*tokenizer, U'瀀', 300, 0);
    const auto before_edit = model->calls.size();
    (void)engine.predict(edited, singleton);
    require(model->calls.back().position == 1 && decoded_since(*model, before_edit) == 128 + 2,
        "edited context without punctuation did not fall back to a fixed cut");
}
} // namespace

int main() {
    try {
        test();
        test_window();
        test_speculative_acceptance();
        test_speculative_rejection(false);
        test_speculative_rejection(true);
        test_speculative_resubmission();
        test_speculative_edits_and_failures();
        std::cout << "[ok] prediction/cache/alias/window/speculative verification policies\n";
        return EXIT_SUCCESS;
    }
    catch (const std::exception& error) { std::cerr << "[FAIL] " << error.what() << '\n'; return EXIT_FAILURE; }
}

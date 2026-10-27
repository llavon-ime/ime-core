#include "bopomofo/bopomofo.hpp"
#include "engine/prediction_engine.hpp"
#include "engine/tokenizer.hpp"

#include <ime-core/logger.hpp>

#include <cstdlib>
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

class QuietLogger final : public Logger {
public:
    using Logger::log;
    void log(std::string) noexcept override {}
    void log(MessageFactory) noexcept override {}
};

class FakeBackend;

struct FakeModel final : BackendModel {
    struct Decode { std::size_t position; std::vector<int> tokens; };
    std::vector<Decode> calls;
    std::vector<std::size_t> rewinds;
    int clears = 0;
    int warmups = 0;
    bool fail_decode = false;
    bool reject_rewind = false;
    InferenceRuntimeInfo runtime;
    const InferenceRuntimeInfo& runtime_info() const noexcept override { return runtime; }
    std::size_t vocabulary_size() const noexcept override { return 65536; }
    std::unique_ptr<BackendSession> create_session() override;
};

class FakeBackend final : public BackendSession {
public:
    explicit FakeBackend(std::shared_ptr<FakeModel> model) : model_(std::move(model)) {
        values_.resize(model_->vocabulary_size());
        const Tokenizer tokenizer(IME_CORE_TEST_TABLES_DIR);
        values_.at(static_cast<std::size_t>(tokenizer.map_char(U'你'))) = 10;
        values_.at(static_cast<std::size_t>(tokenizer.map_char(U'好'))) = 9;
    }
    void ready() override { ++model_->warmups; }
    void clear() noexcept override { ++model_->clears; offset_ = 0; }
    bool rewind(std::size_t position) override {
        model_->rewinds.push_back(position);
        if (model_->reject_rewind) return false;
        if (position > offset_) throw std::logic_error("rewind exceeds cache");
        offset_ = position;
        return true;
    }
    void decode(std::span<const int> tokens, std::size_t position) override {
        if (position != offset_) throw std::logic_error("incorrect cache position");
        model_->calls.push_back({position, {tokens.begin(), tokens.end()}});
        offset_ += tokens.size();
        if (std::exchange(model_->fail_decode, false)) throw std::runtime_error("partial decode failure");
    }
    std::span<const float> logits() const override { return values_; }
private:
    std::shared_ptr<FakeModel> model_;
    std::vector<float> values_;
    std::size_t offset_ = 0;
};

std::unique_ptr<BackendSession> FakeModel::create_session() {
    return std::make_unique<FakeBackend>(
        std::static_pointer_cast<FakeModel>(shared_from_this()));
}

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
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
        model->calls[2].tokens == std::vector<int>{prompt.back()}, "shortened cache did not refresh SEP logits");
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
        std::cout << "[ok] prediction/cache/alias/window policies\n";
        return EXIT_SUCCESS;
    }
    catch (const std::exception& error) { std::cerr << "[FAIL] " << error.what() << '\n'; return EXIT_FAILURE; }
}

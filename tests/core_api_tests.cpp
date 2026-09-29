#include <ime-core/core.hpp>

#include <cstdlib>
#include <type_traits>

int main() {
    static_assert(!std::is_copy_constructible_v<llavon::ime::core::Core>);
    static_assert(std::is_move_constructible_v<llavon::ime::core::Core>);
    static_assert(!std::is_copy_constructible_v<llavon::ime::core::Session>);
    static_assert(std::is_move_constructible_v<llavon::ime::core::Session>);

    llavon::ime::core::PaddingEntry pending{
        .chosen = false,
        .chosen_char = 0,
        .bopomofo = u"\u310c\u311a",
    };
    llavon::ime::core::Prediction prediction;
    prediction.candidates.emplace_back(U'\u62c9', 1.0F);
    const llavon::ime::core::CoreConfig default_config;

    return default_config.vulkan_pipeline_cache_dir.empty() &&
                   pending.bopomofo == u"\u310c\u311a" &&
                   prediction.candidates.front().first == U'\u62c9'
               ? EXIT_SUCCESS
               : EXIT_FAILURE;
}

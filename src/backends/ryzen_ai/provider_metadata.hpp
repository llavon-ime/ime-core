#pragma once

#include <Windows.h>
#include <WinMLEpCatalog.h>

#include <cstdint>
#include <filesystem>
#include <format>
#include <stdexcept>
#include <string>
#include <string_view>

namespace llavon::ime::core::ryzen_ai::detail {

inline void check_hresult(std::string_view operation, HRESULT result) {
    if (FAILED(result)) {
        throw std::runtime_error(std::format("{} failed (HRESULT 0x{:08X})", operation,
                                             static_cast<std::uint32_t>(result)));
    }
}

enum class ProviderString { required, optional };

inline std::string provider_string(
    WinMLEpHandle provider,
    HRESULT (*size_function)(WinMLEpHandle, size_t*),
    HRESULT (*read_function)(WinMLEpHandle, size_t, char*, size_t*),
    std::string_view name,
    ProviderString requirement = ProviderString::required) {
    size_t size = 0;
    check_hresult(std::format("{} size", name), size_function(provider, &size));
    std::string result(size, '\0');
    if (size != 0) {
        size_t used = 0;
        check_hresult(name, read_function(provider, result.size(), result.data(), &used));
        if (used > result.size()) {
            throw std::runtime_error(std::format("Windows ML returned an invalid {} size", name));
        }
        if (used != 0) result.resize(used);
        while (!result.empty() && result.back() == '\0') result.pop_back();
    }
    if (result.empty() && requirement == ProviderString::required) {
        throw std::runtime_error(std::format("Windows ML returned an empty {}", name));
    }
    return result;
}

inline std::string provider_cache_identity(std::string_view library) {
    const std::filesystem::path path(std::u8string_view(
        reinterpret_cast<const char8_t*>(library.data()), library.size()));
    // Catalog metadata may differ between the compiler and inference process.
    // Key both processes by the installed DLL, not an optional catalog string.
    const auto size = std::filesystem::file_size(path);
    const auto timestamp = std::filesystem::last_write_time(path).time_since_epoch().count();
    return std::format("{}:{}|{}|{}", library.size(), library, size, timestamp);
}

} // namespace llavon::ime::core::ryzen_ai::detail

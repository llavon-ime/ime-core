#include "backends/ryzen_ai/provider_metadata.hpp"

#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>

using namespace llavon::ime::core::ryzen_ai::detail;

namespace {

struct CatalogReply {
    std::string text;
    size_t capacity = 0;
    size_t used = 0;
    HRESULT size_status = S_OK;
    HRESULT read_status = S_OK;
    int reads = 0;
} reply;

HRESULT get_size(WinMLEpHandle, size_t* size) {
    *size = reply.capacity;
    return reply.size_status;
}

HRESULT read_string(WinMLEpHandle, size_t capacity, char* buffer, size_t* used) {
    ++reply.reads;
    if (capacity < reply.text.size()) return E_INVALIDARG;
    std::memcpy(buffer, reply.text.data(), reply.text.size());
    *used = reply.used;
    return reply.read_status;
}

void require(bool condition, std::string_view message) {
    if (!condition) throw std::runtime_error(std::string(message));
}

template <class Action>
void require_failure(Action action, std::string_view message) {
    try {
        action();
    } catch (const std::runtime_error&) {
        return;
    }
    throw std::runtime_error(std::string(message));
}

std::string read_version() {
    return provider_string(nullptr, get_size, read_string, "version", ProviderString::optional);
}

void test_catalog_strings() {
    reply = {};
    require(read_version().empty() && reply.reads == 0, "Zero-length version should be accepted");
    require_failure([] { provider_string(nullptr, get_size, read_string, "library path"); },
                    "Missing required library path should fail");

    reply = {.text = std::string(1, '\0'), .capacity = 1, .used = 1};
    require(read_version().empty(), "NUL-only version should be accepted");
    require_failure([] { provider_string(nullptr, get_size, read_string, "library path"); },
                    "NUL-only required library path should fail");

    reply = {.text = std::string("1.2.3\0", 6), .capacity = 32, .used = 6};
    require(read_version() == "1.2.3", "Version must exclude trailing NUL and unused capacity");
    reply.used = 0;
    require(read_version() == "1.2.3", "Read should tolerate an unset used count");
    reply.used = 33;
    require_failure(read_version, "Oversized read count should fail");

    reply = {.size_status = E_FAIL};
    require_failure(read_version, "Failed size call must preserve its error");
    reply = {.capacity = 1, .read_status = E_FAIL};
    require_failure(read_version, "Failed read call must preserve its error");
}

struct TemporaryLibrary {
    std::filesystem::path path = std::filesystem::current_path() / u8"provider-測試-";

    TemporaryLibrary() {
        path += std::format("{}-{}.dll", GetCurrentProcessId(), GetTickCount64());
    }

    ~TemporaryLibrary() {
        std::error_code ignored;
        std::filesystem::remove(path, ignored);
    }
};

void test_cache_identity() {
    const TemporaryLibrary library;
    const auto utf8 = library.path.u8string();
    const std::string path(reinterpret_cast<const char*>(utf8.data()), utf8.size());
    require_failure([&] { provider_cache_identity(path); }, "Missing DLL should fail");
    std::ofstream(library.path, std::ios::binary) << "first";
    const auto initial = provider_cache_identity(path);
    require(initial == provider_cache_identity(path), "Unchanged DLL must reuse its identity");

    const auto timestamp = std::filesystem::last_write_time(library.path);
    std::ofstream(library.path, std::ios::binary) << "changed file size";
    std::filesystem::last_write_time(library.path, timestamp);
    const auto resized = provider_cache_identity(path);
    require(initial != resized, "DLL size change must invalidate cache without a version");
    std::filesystem::last_write_time(library.path, timestamp + std::chrono::seconds(5));
    require(resized != provider_cache_identity(path),
            "DLL timestamp change must invalidate cache without a version");
}

} // namespace

int main() {
    try {
        test_catalog_strings();
        test_cache_identity();
        std::cout << "Windows ML optional metadata and EP cache identity tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

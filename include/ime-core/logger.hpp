#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <utility>

namespace llavon::ime::core {

enum class LogInformation : std::uint8_t {
    general = 0,
    context = 1,
};

class Logger {
public:
    // std::function instead of std::move_only_function: libc++ (Apple
    // toolchains) does not implement P0288.
    using MessageFactory = std::function<std::string()>;

    virtual ~Logger() = default;

    // Compatibility entry points for existing injected logger implementations.
    // Implementations must not block the caller and must not throw.
    virtual void log(std::string message) noexcept = 0;

    // Submits a lazily materialized message. Implementations must never invoke
    // the factory on the calling thread. If logging is disabled or the message
    // cannot be accepted, the factory must not be invoked at all. Accepted
    // factories may be invoked at most once on the logger's worker thread.
    virtual void log(MessageFactory make_message) noexcept = 0;

    // Classified overloads. Existing logger implementations remain valid and
    // receive classified messages through their original entry points unless
    // they override these functions to preserve the flag across a transport.
    virtual void log(LogInformation, std::string message) noexcept {
        log(std::move(message));
    }
    virtual void log(LogInformation, MessageFactory make_message) noexcept {
        log(std::move(make_message));
    }
};

}  // namespace llavon::ime::core

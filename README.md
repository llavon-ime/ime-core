# IME Core

Cross-platform static C++ inference library for Llavon IME. This project owns model loading,
tokenization, candidate masking, llama.cpp inference, and per-client inference
sessions. It contains no service IPC, process startup, or operating-system
specific path discovery.

## Build

Pass a vcpkg toolchain from the caller:

```powershell
cmake --preset windows -DCMAKE_TOOLCHAIN_FILE="$env:VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake"
cmake --build --preset windows
cmake --install build/windows --config Release
```

The installed CMake package exports the static target `ime-core::ime-core`.

## Core data

`CoreConfig` requires a GGUF model path and a tables directory. The installed
reference tables are placed under `share/ime-core/tables`.

# ggml overlay

The `ggml` overlay pins the Llavon fork commit used by ime-core. Backend and
vcpkg compatibility changes live as normal commits on the fork's
`feature/configurable-vulkan-pipeline-cache` branch; this overlay intentionally
contains no source patches.

The Vulkan backend exposes optional registry functions that configure and save
a persistent pipeline cache. Caching is cross-platform but opt-in: a host must
pass an explicit UTF-8 directory through `CoreConfig::vulkan_pipeline_cache_dir`
before model loading. An empty path performs no cache reads or writes.

The cache envelope identifies the backend schema, GPU vendor/device, driver,
pointer width and Vulkan pipeline-cache UUID. It validates the Vulkan header and
payload checksum before passing data to a driver, ignores files larger than
128 MiB, and replaces cache files with `std::filesystem::rename` from a unique
temporary file in the same directory. Cache failures never prevent inference.

The core saves immediately after Vulkan pipeline preparation and the backend
saves remaining changes during device destruction. Prediction does not write
cache files.

Manual hardware validation is in
`service/tests/vulkan_pipeline_cache_hardware.ps1`. It runs fresh processes
against `ime-core-latency-check` using an explicit cache directory and checks
reuse, corrupt data, incompatible identities, repair, and an unwritable path.

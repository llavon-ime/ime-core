# ggml and llama.cpp overlays

The `ggml` overlay pins the Llavon fork commit used by ime-core. Backend and
vcpkg compatibility changes live on the fork's
`feature/configurable-vulkan-pipeline-cache` branch. Host-only pipeline
preparation is included in fork commit
`80b8081c627deee05ebcf377f64746b58b4b82c7`, so the overlay uses that source
directly without a local ggml preparation patch.

The `llama-cpp` overlay pins the organization's `llavon-ime/llama.cpp` fork at
`8bc5e47a6e8f69f71d781e985f0327e00e0ff881`, on
`feature/vulkan-pipeline-preparation`. It is based on upstream b9030 and includes
both the graph-construction bridge and the vcpkg packaging adjustments. Neither
overlay applies source patches. Update these two dependency overlays together:
the bridge uses the new ggml scheduler preparation API.

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

Pipeline preparation builds representative single-sequence graphs for batch
sizes 1 through 8, powers of two up to the microbatch capacity, one/all output
tokens, and short/full KV extents. The scheduler allocates and splits these
graphs, but never copies graph inputs or evaluates CPU splits. Vulkan walks its
existing fusion and kernel-selection logic using a temporary recording context.
Dispatches are skipped and recorded commands are discarded. Submission and
synchronization guards reject accidental execution during preparation.

Where `pipelineCreationCacheControl` is available, each new pipeline is first
created with `FAIL_ON_PIPELINE_COMPILE_REQUIRED`. A successful handle is retained;
only misses are compiled. The cache-only API returns `GGML_STATUS_ABORTED` for
missing pipelines, `GGML_STATUS_FAILED` if probing is unsupported, and never
silently falls back to compilation. Normal preparation permits a regular host
pipeline creation on devices without this feature and reports it as `unprobed`.
`cache_hits` means creation without compilation; the driver may use either the
application cache or its own cache. It is not an application-cache-hit feedback
measurement. Coverage is limited to the prepared shapes; other runtime shapes
can still request additional pipelines normally.

`llama_prepare_pipelines` changes neither KV metadata/data nor evaluation
counters. It currently prepares one sequence at a time. The short-KV simulation
supports the standard KV cache; other memory implementations support only the
full-memory form. Builds with Vulkan tensor-execution diagnostics reject the
preparation API because those diagnostic helpers deliberately execute tensors.

The optional `ime-core-vulkan-prepare-tests MODEL CACHE_DIRECTORY` target checks
cache-only misses, repeated probes, empty/populated KV preservation, multiple
sequence slots and logits against a context that did not run preparation. Run
it twice with the same cache directory to check reuse across processes.

Manual hardware validation is in
`service/tests/vulkan_pipeline_cache_hardware.ps1`. It runs fresh processes
against `ime-core-latency-check` using an explicit cache directory and checks
reuse, corrupt data, incompatible identities, repair, and an unwritable path.

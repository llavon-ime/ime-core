# AMD Ryzen AI ONNX backend

The optional backend runs a complete decoder through the Windows ML VitisAI
execution provider. `ime-core` continues to load vocabulary metadata from the
GGUF file, so token ids and the IME candidate pipeline remain shared with the
llama.cpp path.

## Model representation

`tools/export_onnx.py` exports the original Hugging Face Llama weights as an
ONNX Runtime GenAI graph. The graph uses `com.microsoft::MatMulNBits` with:

- 4-bit blockwise weights and block size 32 for the model body;
- 8-bit weights for the output and the attention/MLP matrices selected by the
  same layer pattern used by llama.cpp's mixed K-quant strategy;
- `accuracy_level=3` for quantized matrix multiplication;
- a last-token language-model head for decoder inference.

ONNX cannot store ggml's `Q4_K` and `Q6_K` byte layout directly. The mixed
`MatMulNBits` graph preserves the same design: four-bit base weights and higher
precision for selected matrices. The mixed INT4/INT8 graph still requires
compilation and inference validation on the target VitisAI EP.

## Runtime

`create_ryzen_ai_accelerator` receives only a model directory and a cache
directory:

```cpp
core::RyzenAiConfig runtime{
    .model_directory = L"C:/models/llavon-ime-llama-250m-Q4_K_M-onnx",
    .cache_directory = L"C:/Users/me/AppData/Local/Llavon IME/onnx-npu-cache",
};
```

Preparation performs these operations synchronously:

1. locate and install the certified `VitisAIExecutionProvider` through
   `WinMLEpCatalog`;
2. register its library with ONNX Runtime GenAI;
3. select AMD vendor `0x1022` and hardware type `NPU`;
4. disable CPU execution-provider fallback;
5. load an existing EPContext or compile and persist a new one.

The caller publishes the replacement `Core` only after preparation and the
warm-up decode succeed. Each IME session owns an OGA generator and its KV cache;
the prepared ONNX model and NPU binary are shared.

Hosts can set `RyzenAiConfig::allow_compilation` to `false` to require an existing
EPContext. In that mode, missing or unusable contexts fail without recompiling.
The Windows service owns its compiler process and uses this option when loading
the context into the service; process management is outside `ime-core`.

The `ryzen-ai` vcpkg feature supplies official Windows ML and ONNX Runtime GenAI
NuGet binaries. The target machine downloads the hardware execution provider
through Windows ML. No AMD SDK or copied ONNX headers are part of the source tree.

## Graph validation and the staging failure

`tools/validate_onnx.py` imports missing operator signatures from ONNX Runtime
before running the ONNX checker and strict shape inference. ORT contributes
`SimplifiedLayerNormalization` in the default domain; the upstream ONNX checker
does not recognize it on its own. Custom-operator shape inference is still
limited by the inference functions available in ONNX; runtime checks remain necessary.

The model from the September 29 staging failure had an incorrect shape annotation
for `/model/constants/INT64/[1]`: its Constant contains a vector of length one,
but the value metadata declared a scalar. The exporter now derives Constant
shapes from their tensors. The corrected model passes both checks, retains the
same external weights, and produced bit-identical CPU logits for 12 prompts with
prefill, decode and rewind (36 arrays).

This does not establish the cause of the VitisAI `staging-graph.cpp:721` fatal
assertion. Optional empty output slots in SkipSimplifiedLayerNormalization are
legal ONNX; they must not simply be removed because output positions carry
meaning. AMD's model conversion also applies additional BF16/custom-operator
transformations. Its [Windows ML LLM example](https://github.com/amd/RyzenAI-SW/tree/main/WinML/LLM)
uses `VitisGenerateModelLLM` after quantization. Changing the provider name alone
does not reproduce that flow. Target-machine testing is still needed to determine
whether this EP accepts the existing mixed-precision decoder.

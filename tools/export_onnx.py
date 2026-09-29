#!/usr/bin/env python3
"""Export Llavon's Llama model to an NPU-oriented ONNX Runtime GenAI model."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from validate_onnx import normalize_constant_shapes, validate_model


def export_model(source: str, output: Path, cache: Path) -> None:
    command = [
        sys.executable,
        "-m",
        "onnxruntime_genai.models.builder",
        "--model_name",
        source,
        "--output",
        str(output),
        "--precision",
        "int4",
        "--execution_provider",
        "dml",
        "--cache_dir",
        str(cache),
        "--extra_options",
        "algo_config=k_quant_mixed",
        "block_size=32",
        "accuracy_level=3",
        "is_symmetric=true",
        "prune_lm_head=true",
        "hf_token=false",
    ]
    subprocess.run(command, check=True)


def configure_vitisai(output: Path) -> Path:
    config_path = output / "genai_config.json"
    with config_path.open("r", encoding="utf-8") as source:
        config = json.load(source)
    session = config["model"]["decoder"]["session_options"]
    session["provider_options"] = [{"VitisAI": {}}]
    with config_path.open("w", encoding="utf-8", newline="\n") as destination:
        json.dump(config, destination, ensure_ascii=False, indent=2)
        destination.write("\n")
    return output / config["model"]["decoder"]["filename"]


def validate_quantization(model_path: Path) -> None:
    try:
        import onnx
    except ImportError as error:
        raise RuntimeError("The exporter environment is missing the 'onnx' package") from error

    model = onnx.load(model_path, load_external_data=False)
    changed = normalize_constant_shapes(model)
    if changed:
        # Preserve packed weights and external-data references byte for byte.
        onnx.save_model(model, model_path)
        print(f"Corrected {changed} Constant shape annotation(s)")
    validate_model(model_path)
    widths: dict[int, int] = {}
    for node in model.graph.node:
        if node.domain != "com.microsoft" or node.op_type != "MatMulNBits":
            continue
        attributes = {attribute.name: attribute for attribute in node.attribute}
        bits = attributes.get("bits")
        if bits is not None:
            widths[bits.i] = widths.get(bits.i, 0) + 1
    if widths.get(4, 0) == 0 or widths.get(8, 0) == 0:
        raise RuntimeError(
            "Export did not produce the expected mixed INT4/INT8 MatMulNBits graph "
            f"(found {widths})"
        )
    print(f"Validated MatMulNBits nodes: {widths}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        default="tony65535/llavon-ime-llama-250m",
        help="Hugging Face model id or local Hugging Face model directory",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=Path(".onnx-export-cache"))
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    args.cache.mkdir(parents=True, exist_ok=True)
    export_model(args.source, args.output, args.cache)
    validate_quantization(configure_vitisai(args.output))
    print(f"ONNX Runtime GenAI model written to {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

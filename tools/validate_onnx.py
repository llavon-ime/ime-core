#!/usr/bin/env python3
"""Validate an ORT GenAI graph, including schemas supplied by ONNX Runtime."""

from __future__ import annotations

import argparse
from pathlib import Path

import onnx


def register_runtime_schemas(model: onnx.ModelProto) -> None:
    # ORT still exports SimplifiedLayerNormalization in the default domain.
    # Upstream ONNX has no schema for it. Import the real runtime signatures;
    # do not suppress checker errors or relabel the operator's domain.
    from onnxruntime.capi import _pybind_state

    schema_type = onnx.defs.OpSchema
    versions = {item.domain: item.version for item in model.opset_import}
    runtime_schemas = _pybind_state.get_all_operator_schema()
    for node in model.graph.node:
        version = versions[node.domain]
        if onnx.defs.has(node.op_type, version, node.domain):
            continue
        matches = [s for s in runtime_schemas if s.name == node.op_type
                   and s.domain == node.domain and s.since_version <= version]
        if not matches:
            raise ValueError(f"No ONNX/ORT schema for {node.domain}:{node.op_type}")
        source = max(matches, key=lambda s: s.since_version)

        def parameters(items):
            return [schema_type.FormalParameter(
                item.name, item.typeStr,
                param_option=getattr(schema_type.FormalParameterOption, item.option.name))
                for item in items]

        schema = schema_type(
            source.name, source.domain, source.since_version,
            inputs=parameters(source.inputs), outputs=parameters(source.outputs),
            type_constraints=[(item.type_param_str, item.allowed_type_strs, "")
                              for item in source.type_constraints],
            attributes=[schema_type.Attribute(
                name, getattr(schema_type.AttrType, attribute.type.name),
                required=attribute.required)
                for name, attribute in source.attributes.items()],
        )
        onnx.defs.register_schema(schema)


def normalize_constant_shapes(model: onnx.ModelProto) -> int:
    values = {value.name: value for value in model.graph.value_info}
    values.update((value.name, value) for value in model.graph.output)
    changed = 0
    for node in model.graph.node:
        if node.domain or node.op_type != "Constant":
            continue
        tensor = next((a.t for a in node.attribute if a.name == "value"), None)
        if tensor is None or node.output[0] not in values:
            continue
        value = values[node.output[0]]
        corrected = onnx.helper.make_tensor_value_info(value.name, tensor.data_type, list(tensor.dims))
        if value.type != corrected.type:
            value.type.CopyFrom(corrected.type)
            changed += 1
    return changed


def validate_model(path: Path) -> None:
    model = onnx.load(path, load_external_data=False)
    register_runtime_schemas(model)
    onnx.checker.check_model(str(path), full_check=True, check_custom_domain=True)
    onnx.shape_inference.infer_shapes(model, check_type=True, strict_mode=True, data_prop=True)
    print(f"ONNX checker and strict shape inference passed: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    args = parser.parse_args()
    validate_model(args.model)

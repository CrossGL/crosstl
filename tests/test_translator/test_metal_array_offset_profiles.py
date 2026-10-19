"""Array address arithmetic remains separate from profiled numeric arithmetic."""

import pytest

from crosstl import translate
from crosstl.backend.Metal.MetalAst import BinaryOpNode, VariableNode
from crosstl.backend.Metal.MetalCrossGLCodeGen import MetalToCrossGLConverter

SOURCE = """#include <metal_stdlib>
using namespace metal;
kernel void array_offset(device float* output [[buffer(0)]],
                         uint lane [[thread_position_in_threadgroup]]) {
    threadgroup float scratch[32];
    scratch[lane] = float(lane);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float* selected = scratch + (lane / 4u) * 4u + lane % 4u;
    output[lane + 4u] = selected[0] + 1.0f;
}
"""


@pytest.mark.parametrize("element", ("float", "int", "float2", "Cell"))
@pytest.mark.parametrize("operator,commuted", (("+", False), ("+", True), ("-", False)))
def test_array_offset_infers_pointer_result(element, operator, commuted):
    converter = MetalToCrossGLConverter()
    converter.current_variable_types.update(values=f"{element}[32]", offset="uint")
    left, right = VariableNode("", "values"), VariableNode("", "offset")
    if commuted:
        left, right = right, left
    expression = BinaryOpNode(left, operator, right)
    assert converter.expression_metal_type(expression) == f"{element}*"
    assert (
        converter.expression_metal_type(
            BinaryOpNode(expression, "+", right if not commuted else left)
        )
        == f"{element}*"
    )


@pytest.mark.parametrize("space", ("thread", "threadgroup", "device", "constant"))
@pytest.mark.parametrize("commuted", (False, True))
def test_array_offset_retains_only_array_storage_provenance(space, commuted):
    converter = MetalToCrossGLConverter()
    converter.current_variable_types.update(values="float[32]", offset="uint")
    converter.current_variable_type_qualifiers.update(
        values=(space, "const"), offset=("thread",)
    )
    values, offset = VariableNode("", "values"), VariableNode("", "offset")
    expression = (
        BinaryOpNode(offset, "+", values)
        if commuted
        else BinaryOpNode(values, "+", offset)
    )
    assert converter.expression_metal_type_qualifiers(expression) == (space, "const")


@pytest.mark.parametrize(
    "array_type,offset_type,operator,commuted",
    (
        ("float[32]", "float", "+", False),
        ("float[32]", "float", "+", True),
        ("float[32]", "uint2", "+", False),
        ("float[32]", "uint", "-", True),
        ("float[32]", "uint", "*", False),
        ("float[32]", "uint", "/", False),
        ("float[2][16]", "uint", "+", False),
        ("float*[32]", "uint", "+", False),
        ("array<float,32>", "uint", "+", False),
        ("float[32]", "Unknown", "+", False),
    ),
)
def test_array_offset_does_not_guess_unsupported_pointer_operations(
    array_type, offset_type, operator, commuted
):
    converter = MetalToCrossGLConverter()
    converter.current_variable_types.update(values=array_type, offset=offset_type)
    values, offset = VariableNode("", "values"), VariableNode("", "offset")
    expression = (
        BinaryOpNode(offset, operator, values)
        if commuted
        else BinaryOpNode(values, operator, offset)
    )
    assert converter.metal_pointer_arithmetic_source(expression) is None


@pytest.mark.parametrize("target", ("crossgl", "metal", "directx", "opengl"))
@pytest.mark.parametrize("profile", ("native-precise", "rne-gradual", "rne-flush"))
def test_array_offsets_preserve_profiled_value_addition(tmp_path, target, profile):
    path = tmp_path / "array.metal"
    path.write_text(SOURCE)
    generated = translate(
        str(path),
        backend=target,
        format_output=False,
        source_options={"binary32_additive_profile": profile},
    )
    assert "crossgl_metal_add_float" in generated
    assert "scratch" in generated


def native_request(root, target):
    import json

    from crosstl.project import (
        ProjectConfig,
        build_native_loader_abi_descriptor,
        build_native_loader_dispatch_request,
        build_runtime_artifact_manifest,
        build_runtime_loader_manifest,
        build_runtime_package,
        translate_project,
    )

    (root / "array.metal").write_text(SOURCE)
    report = translate_project(
        ProjectConfig(
            root=root,
            include_patterns=("array.metal",),
            targets=(target,),
            workgroup_size=(32, 1, 1),
            source_options={
                "metal": {
                    "binary32_additive_profile": "native-precise",
                    "preserve_resource_origins": True,
                }
            },
        ),
        format_output=False,
    )
    assert report.to_json()["summary"]["failedCount"] == 0, report.to_json()[
        "diagnostics"
    ]
    report.write_json(root / "report.json")
    manifest = build_runtime_artifact_manifest(root / "report.json")
    assert manifest["success"], manifest
    (root / "artifacts.json").write_text(json.dumps(manifest))
    package = root / "package"
    assert build_runtime_package(root / "artifacts.json", package)["success"]
    loader = build_runtime_loader_manifest(package / "runtime-package.json")
    assert loader["success"], loader
    descriptor = build_native_loader_abi_descriptor(loader)
    (binding,) = descriptor["bindings"]
    values = {
        binding["name"]: {"dtype": "float32", "shape": [40], "values": [-1234.0] * 40}
    }
    expected = {
        binding["name"]: {
            "dtype": "float32",
            "shape": [40],
            "values": [-1234.0] * 4 + [float(n + 1) for n in range(32)] + [-1234.0] * 4,
        }
    }
    request = build_native_loader_dispatch_request(
        descriptor,
        package,
        values,
        expected,
        {"workgroupCount": [1, 1, 1], "workgroupSize": [32, 1, 1]},
    )
    return request, expected

"""Range iteration preserves values, binding conversions and source storage."""

import os
import sys
from pathlib import Path

import pytest

from crosstl import translator
from crosstl.backend.Metal.MetalLexer import MetalLexer
from crosstl.backend.Metal.MetalParser import MetalParser
from crosstl.project import ProjectConfig, translate_project
from crosstl.translator.ast import ReferenceType
from tests.test_backend.test_metal.test_codegen import convert
from tests.test_translator.test_boolean_buffer_runtime import _bound_values, _request
from tests.test_translator.test_loop_updates import _execute
from tests.test_translator.test_software_subgroup_product import _package

REQUIRE_ENV = "CROSTL_REQUIRE_RANGE_BINDINGS"
CASES = {
    "copy": (
        "uint values[2] = {3u, 5u};",
        "for (auto value : values) { value += 1u; }",
        "float(values[0] + values[1])",
        8,
    ),
    "conversion": (
        "uint values[2] = {3u, 5u}; float total = 0.0f;",
        "for (float value : values) { total += value / 2; }",
        "total",
        4,
    ),
    "reference": (
        "uint values[2] = {3u, 5u};",
        "for (thread auto& value : values) { value += 1u; }",
        "float(values[0] + values[1])",
        10,
    ),
    "readonly_reference": (
        "uint values[2] = {3u, 5u}; uint total = 0;",
        "for (const thread uint& value : values) { values[0] += 1u; total += value; }",
        "float(total)",
        9,
    ),
    "live_elements": (
        "uint values[2] = {3u, 5u}; uint total = 0;",
        "for (auto value : values) { values[1] = 9u; total += value; }",
        "float(total)",
        12,
    ),
    "subarray": (
        "uint values[2][3] = {{2u, 3u, 5u}, {11u, 13u, 17u}}; "
        "uint row = 0; uint total = 0;",
        "for (auto value : values[row++]) { total += value; }",
        "float(total + row)",
        11,
    ),
    "shadow": (
        "uint values[2] = {3u, 5u}; uint total = 0;",
        "for (auto values : values) { total += values; }",
        "float(total + values[0])",
        11,
    ),
    "control_flow": (
        "uint values[4] = {3u, 5u, 7u, 11u}; uint total = 0;",
        "for (auto value : values) { if (value == 5u) { continue; } "
        "total += value; if (value == 7u) { break; } }",
        "float(total)",
        10,
    ),
    "selector_identity": (
        "uint values[2][2] = {{3u, 5u}, {11u, 13u}}; uint row = 0; uint total = 0;",
        "for (auto value : values[row]) { row = 1; values[0][1] = 7u; total += value; }",
        "float(total + row)",
        11,
    ),
    "reference_control_flow": (
        "uint values[4] = {3u, 5u, 7u, 11u};",
        "for (thread uint& value : values) { value += 1u; "
        "if (value == 4u) { continue; } if (value == 6u) { break; } }",
        "float(values[0] + values[1] + values[2] + values[3])",
        28,
    ),
    "reference_alias_read": (
        "uint values[2] = {3u, 5u}; uint total = 0;",
        "for (thread uint& value : values) { value += 1u; "
        "values[0] += 2u; total += value; }",
        "float(total + values[0])",
        20,
    ),
    "converted_const_reference": (
        "uint values[2] = {3u, 5u}; float total = 0.0f;",
        "for (const thread float& value : values) { values[0] += 1u; total += value / 2; }",
        "total",
        4,
    ),
    "member_array": (
        "struct Data { uint values[2]; }; Data data; "
        "data.values[0] = 3u; data.values[1] = 5u; uint total = 0;",
        "for (thread uint& value : data.values) { value += 1u; total += value; }",
        "float(total + data.values[0])",
        14,
    ),
    "global_subarray": (
        "uint row = 0; uint total = 0;",
        "for (auto value : rotations[row++]) { total += value; }",
        "float(total + row)",
        61,
    ),
    "nested_references": (
        "uint values[2][2] = {{3u, 5u}, {11u, 13u}};",
        "for (thread auto& row : values) { for (thread auto& value : row) { value += 1u; } }",
        "float(values[0][0] + values[0][1] + values[1][0] + values[1][1])",
        36,
    ),
    "reference_return": (
        "uint values[2] = {3u, 5u};",
        "for (thread uint& value : values) { value += 1u; "
        "results[1] = float(values[0]); return; }",
        "0.0f",
        4,
    ),
    "reference_value_call": (
        "uint values[2] = {3u, 5u}; uint total = 0;",
        "for (thread uint& value : values) { total += increment(value); value += 1u; }",
        "float(total + values[0] + values[1])",
        20,
    ),
    "copy_reference_call": (
        "uint values[2] = {3u, 5u}; uint total = 0;",
        "for (uint value : values) { increment_copy(value); total += value; }",
        "float(total + values[0] + values[1])",
        18,
    ),
}


def _source(case):
    declarations, loop, result, expected = CASES[case]
    source = f"""#include <metal_stdlib>
using namespace metal;
{'constant uint rotations[2][4] = {{13u, 15u, 26u, 6u}, {17u, 29u, 16u, 24u}};' if case == 'global_subarray' else ''}
{'uint increment(uint value) { return value + 1u; }' if case == 'reference_value_call' else ''}
{'void increment_copy(thread uint& value) { value += 1u; }' if case == 'copy_reference_call' else ''}
kernel void range_bindings(device float* results [[buffer(0)]]) {{
    {declarations}
    {loop}
    results[1] = {result};
}}
"""
    return source, expected


def _case(root, target, case):
    source, value = _source(case)
    _, descriptor, package = _package(
        root, target, "uint", (1, 1, 1), source=source, software_subgroups=False
    )
    inputs = {"results": {"dtype": "float32", "shape": [3], "values": [99.0] * 3}}
    outputs = {
        "results": {"dtype": "float32", "shape": [3], "values": [99.0, value, 99.0]}
    }
    return (
        source,
        _request(descriptor, package, inputs, outputs, 1),
        _bound_values(descriptor, outputs),
    )


@pytest.mark.parametrize("case", CASES)
def test_range_binding_contract_survives_import(case):
    source, _ = _source(case)
    parsed = MetalParser(MetalLexer(source).tokenize()).parse()
    entry = next(
        function for function in parsed.functions if function.name == "range_bindings"
    )
    loop = next(node for node in entry.body if hasattr(node, "iterable"))
    crossgl = convert(source)
    shader = translator.parse(crossgl)
    imported = next(
        node for node in shader.walk() if type(node).__name__ == "ForInNode"
    )
    assert imported.binding_type is not None
    assert isinstance(imported.binding_type, ReferenceType) == loop.vtype.endswith("&")
    assert set(loop.qualifiers) <= set(imported.binding_qualifiers)
    if isinstance(imported.binding_type, ReferenceType):
        assert imported.binding_type.is_mutable == ("const" not in loop.qualifiers)


@pytest.mark.parametrize("target", ("metal", "opengl", "directx"))
@pytest.mark.parametrize("case", CASES)
def test_range_bindings_package(tmp_path, target, case):
    _case(tmp_path, target, case)


@pytest.mark.parametrize("case", CASES)
def test_range_bindings_execute_natively(tmp_path, case):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required range binding execution")
    target = {"darwin": "metal", "win32": "directx", "linux": "opengl"}[sys.platform]
    source, request, expected = _case(tmp_path, target, case)
    _execute(
        request,
        expected,
        tmp_path,
        original_source=source,
        original_entry="range_bindings",
    )


@pytest.mark.parametrize("target", ("directx", "opengl"))
@pytest.mark.parametrize("shadow", ("value", "values"))
def test_range_reference_shadowing_is_explicit(tmp_path, target, shadow):
    source = f"""#include <metal_stdlib>
using namespace metal;
kernel void range_shadow(device uint* results [[buffer(0)]]) {{
    uint values[2] = {{3u, 5u}};
    for (thread uint& value : values) {{
        {{ uint {shadow} = 7u; results[0] = {shadow}; }}
        value += 1u;
    }}
    results[1] = values[0];
}}
"""
    (tmp_path / "input.metal").write_text(source)
    report = translate_project(
        ProjectConfig(
            root=tmp_path, include_patterns=("input.metal",), targets=(target,)
        ),
        format_output=False,
    ).to_json()
    assert report["summary"]["failedCount"] == 1
    diagnostic = next(
        item for item in report["diagnostics"] if item["severity"] == "error"
    )
    assert (
        diagnostic["code"] == f"project.translate.{target}-for-in-iterable-unsupported"
    )
    assert "binding or container is shadowed" in diagnostic["message"]
    assert diagnostic["missingCapabilities"] == [
        f"{target}.fixed-array-for-in-lowering"
    ]


def test_range_binding_native_checks_are_required_on_three_platforms():
    workflow = (
        Path(__file__).parents[2] / ".github/workflows/demo-project-testing.yml"
    ).read_text()
    assert workflow.count('CROSTL_REQUIRE_RANGE_BINDINGS: "1"') == 3
    assert workflow.count("tests/test_translator/test_range_bindings.py") == 3


@pytest.mark.parametrize("target", ("directx", "opengl"))
@pytest.mark.parametrize("readonly", (False, True))
def test_range_reference_helper_transport_is_explicit(tmp_path, target, readonly):
    qualifier = "const " if readonly else ""
    (tmp_path / "input.metal").write_text(f"""#include <metal_stdlib>
using namespace metal;
void combine({qualifier}thread uint& first, thread uint& second) {{
    second += 1u;
    second += first;
}}
kernel void reference_call(device uint* results [[buffer(0)]]) {{
    uint values[1] = {{3u}};
    for ({qualifier}thread uint& value : values) {{
        combine(value, values[0]);
    }}
    results[1] = values[0];
}}
""")
    report = translate_project(
        ProjectConfig(
            root=tmp_path, include_patterns=("input.metal",), targets=(target,)
        ),
        format_output=False,
    ).to_json()
    assert report["summary"]["failedCount"] == 1
    assert all(not (tmp_path / item["path"]).exists() for item in report["artifacts"])
    diagnostic = next(
        item for item in report["diagnostics"] if item["severity"] == "error"
    )
    assert (
        diagnostic["code"] == f"project.translate.{target}-for-in-iterable-unsupported"
    )
    assert "parameter copies cannot preserve aliasing" in diagnostic["message"]
    assert diagnostic["missingCapabilities"] == [
        f"{target}.fixed-array-for-in-lowering"
    ]


@pytest.mark.parametrize("target", ("metal", "directx", "opengl"))
def test_typed_range_binding_does_not_become_integer_count(tmp_path, target):
    (tmp_path / "input.metal").write_text(
        "kernel void invalid_range() { for (float value : 4) { } }"
    )
    report = translate_project(
        ProjectConfig(
            root=tmp_path, include_patterns=("input.metal",), targets=(target,)
        ),
        format_output=False,
    ).to_json()
    assert report["summary"]["failedCount"] == 1
    assert all(
        not (tmp_path / artifact["path"]).exists() for artifact in report["artifacts"]
    )
    diagnostic = next(
        item for item in report["diagnostics"] if item["severity"] == "error"
    )
    assert diagnostic["missingCapabilities"] == [
        f"{target}.fixed-array-for-in-lowering"
    ]

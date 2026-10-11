"""Vector list initialization retains zero fill and differs from scalar splats."""

import os
import sys
from pathlib import Path

import pytest

from crosstl.project import ProjectConfig, translate_project
from tests.test_translator.test_boolean_buffer_runtime import _bound_values, _request
from tests.test_translator.test_loop_updates import _execute
from tests.test_translator.test_software_subgroup_product import _package

REQUIRE_ENV = "CROSTL_REQUIRE_VECTOR_LIST_INITIALIZATION"
SHAPES = ((2, 0), (2, 1), (2, 2), (3, 1), (3, 2), (4, 0), (4, 1), (4, 3), (4, 4))
CONTEXTS = {
    "alias": ("using Vec = uint4;", "Vec value = {first};", "value", [3, 0, 0, 0, 3]),
    "identifier": ("", "uint4 value = {first};", "value", [3, 0, 0, 0, 3]),
    "return": (
        "uint4 build(uint first) { return uint4{first, first + 1}; }",
        "uint4 value = build(first);",
        "value",
        [3, 4, 0, 0, 3],
    ),
    "member": (
        "struct Holder { uint4 lanes; };",
        "Holder value = {uint4{first, first + 1}};",
        "value.lanes",
        [3, 4, 0, 0, 3],
    ),
    "assignment": (
        "",
        "uint4 value = uint4(99); value = uint4{first, first + 1};",
        "value",
        [3, 4, 0, 0, 3],
    ),
    "effects": (
        "uint next_value(thread uint& cursor) { uint value = cursor; ++cursor; return value; }",
        "uint4 value = {next_value(first), next_value(first), next_value(first)};",
        "value",
        [3, 4, 5, 0, 6],
    ),
}
WIDE_FORMS = (
    "empty",
    "partial",
    "full",
    "splat",
    "mixed",
    "effects",
    "alias",
    "return",
)


def _case(root, target, scalar, width, count, explicit):
    vector = f"{scalar}{width}"
    expressions = [f"{scalar}(values[{i}])" for i in range(count)]
    initializer = "{" + ", ".join(expressions) + "}"
    if explicit:
        initializer = vector + initializer
    source = f"""#include <metal_stdlib>
using namespace metal;
kernel void vector_lists(const device float* values [[buffer(0)]],
                         device float* results [[buffer(1)]]) {{
    {vector} value = {initializer};
    {vector} splat = {vector}({scalar}(values[0]));
"""
    for i, lane in enumerate("xyzw"[:width]):
        source += f"    results[{i + 1}] = float(value.{lane});\n"
        source += f"    results[{i + width + 1}] = float(splat.{lane});\n"
    source += "}\n"
    _, descriptor, package = _package(
        root, target, "uint", (1, 1, 1), source=source, software_subgroups=False
    )
    values = [1.5, 0.0, 3.5, 7.0] if scalar == "float" else [3.0, 0.0, 5.0, 7.0]
    if scalar == "int":
        values[0] = -3.0
    converted = [float(bool(value)) if scalar == "bool" else value for value in values]
    result = [
        99.0,
        *converted[:count],
        *([0.0] * (width - count)),
        *([converted[0]] * width),
        99.0,
    ]
    inputs = {
        "values": {"dtype": "float32", "shape": [4], "values": values},
        "results": {
            "dtype": "float32",
            "shape": [len(result)],
            "values": [99.0] * len(result),
        },
    }
    outputs = {
        "results": {"dtype": "float32", "shape": [len(result)], "values": result}
    }
    request = _request(descriptor, package, inputs, outputs, 1)
    return source, request, _bound_values(descriptor, outputs)


@pytest.mark.parametrize("target", ("metal", "directx", "opengl"))
@pytest.mark.parametrize("scalar", ("int", "uint", "float", "bool"))
@pytest.mark.parametrize("width,count", SHAPES)
@pytest.mark.parametrize("explicit", (False, True))
def test_vector_lists_translate_through_public_packages(
    tmp_path, target, scalar, width, count, explicit
):
    _case(tmp_path, target, scalar, width, count, explicit)


@pytest.mark.parametrize("scalar", ("int", "uint", "float", "bool"))
@pytest.mark.parametrize("width,count", SHAPES)
@pytest.mark.parametrize("explicit", (False, True))
def test_vector_lists_execute_natively(tmp_path, scalar, width, count, explicit):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required vector list initialization")
    target = {"darwin": "metal", "win32": "directx", "linux": "opengl"}[sys.platform]
    source, request, expected = _case(tmp_path, target, scalar, width, count, explicit)
    _execute(
        request,
        expected,
        tmp_path,
        original_source=source,
        original_entry="vector_lists",
    )


def _context_case(root, target, context):
    helper, declaration, value, result = CONTEXTS[context]
    source = f"""#include <metal_stdlib>
using namespace metal;
{helper}
kernel void list_context(const device uint* values [[buffer(0)]],
                        device uint* results [[buffer(1)]]) {{
    uint first = values[0];
    {declaration}
    results[1] = {value}.x;
    results[2] = {value}.y;
    results[3] = {value}.z;
    results[4] = {value}.w;
    results[5] = first;
}}
"""
    _, descriptor, package = _package(
        root, target, "uint", (1, 1, 1), source=source, software_subgroups=False
    )
    inputs = {
        "values": {"dtype": "uint32", "shape": [1], "values": [3]},
        "results": {"dtype": "uint32", "shape": [7], "values": [99] * 7},
    }
    outputs = {
        "results": {"dtype": "uint32", "shape": [7], "values": [99, *result, 99]}
    }
    return (
        source,
        _request(descriptor, package, inputs, outputs, 1),
        _bound_values(descriptor, outputs),
    )


@pytest.mark.parametrize("target", ("metal", "directx", "opengl"))
@pytest.mark.parametrize("context", CONTEXTS)
def test_vector_list_contexts_translate(tmp_path, target, context):
    _context_case(tmp_path, target, context)


@pytest.mark.parametrize("context", CONTEXTS)
def test_vector_list_contexts_execute_natively(tmp_path, context):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required vector list initialization")
    target = {"darwin": "metal", "win32": "directx", "linux": "opengl"}[sys.platform]
    source, request, expected = _context_case(tmp_path, target, context)
    _execute(
        request,
        expected,
        tmp_path,
        original_source=source,
        original_entry="list_context",
    )


def _wide_case(root, target, signed, width, form):
    scalar = "long" if signed else "ulong"
    dtype = "int64" if signed else "uint64"
    vector = f"{scalar}{width}"
    values = [
        2**54 + 3,
        -(2**53 + 1) if signed else 2**63 + 17,
        -(2**62) + 9 if signed else 2**64 - 1,
        2**40 + 7,
    ]
    count = {"empty": 0, "partial": 1, "return": width - 1}.get(form, width)
    arguments = [f"values[{index}]" for index in range(count)]
    expected = values[:count] + [0] * (width - count)
    helper = ""
    initializer = "{" + ", ".join(arguments) + "}"
    spelling = vector
    calls = 0
    if form == "splat":
        initializer = f"{vector}(values[0])"
        expected = [values[0]] * width
    elif form == "mixed":
        arguments[:2] = [f"{scalar}2(values[0], values[1])"]
        initializer = vector + "(" + ", ".join(arguments) + ")"
    elif form == "effects":
        helper = (
            f"{scalar} next_value(thread uint& cursor, const device {scalar}* values) "
            "{ return values[cursor++]; }"
        )
        initializer = "{" + ", ".join(["next_value(cursor, values)"] * width) + "}"
        calls = width
    elif form == "alias":
        helper = f"using Wide = {vector};"
        spelling = "Wide"
        initializer = "Wide" + initializer
    elif form == "return":
        helper = f"{vector} make_value(const device {scalar}* values) {{ return {vector}{initializer}; }}"
        initializer = "make_value(values)"
    source = f"""#include <metal_stdlib>
using namespace metal;
{helper}
kernel void wide_lists(const device {scalar}* values [[buffer(0)]],
                       device {scalar}* results [[buffer(1)]]) {{
    uint cursor = 0u;
    {spelling} value = {initializer};
"""
    for index, lane in enumerate("xyzw"[:width]):
        source += f"    results[{index + 1}] = value.{lane};\n"
    source += f"    results[{width + 1}] = {scalar}(cursor);\n}}\n"
    _, descriptor, package = _package(
        root, target, "uint", (1, 1, 1), source=source, software_subgroups=False
    )
    result = [97, *expected, calls, 97]
    inputs = {
        "values": {"dtype": dtype, "shape": [4], "values": values},
        "results": {
            "dtype": dtype,
            "shape": [len(result)],
            "values": [97] * len(result),
        },
    }
    outputs = {"results": {"dtype": dtype, "shape": [len(result)], "values": result}}
    return (
        source,
        _request(descriptor, package, inputs, outputs, 1),
        _bound_values(descriptor, outputs),
    )


@pytest.mark.parametrize("target", ("metal", "directx", "opengl"))
@pytest.mark.parametrize("signed", (True, False), ids=("signed", "unsigned"))
@pytest.mark.parametrize("width", (2, 3, 4))
@pytest.mark.parametrize("form", WIDE_FORMS)
def test_wide_vector_lists_translate(tmp_path, target, signed, width, form):
    _, request, _ = _wide_case(tmp_path, target, signed, width, form)
    assert "ConstructorNode(" not in request.artifact_path.read_text()


@pytest.mark.parametrize("signed", (True, False), ids=("signed", "unsigned"))
@pytest.mark.parametrize("width", (2, 3, 4))
@pytest.mark.parametrize("form", WIDE_FORMS)
def test_wide_vector_lists_execute_natively(tmp_path, signed, width, form):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required vector list initialization")
    target = {"darwin": "metal", "win32": "directx", "linux": "opengl"}[sys.platform]
    source, request, expected = _wide_case(tmp_path, target, signed, width, form)
    _execute(
        request, expected, tmp_path, original_source=source, original_entry="wide_lists"
    )


def test_vector_list_native_checks_are_required():
    from tools import ci_coverage

    workflow = (
        Path(__file__).resolve().parents[2]
        / ".github/workflows/demo-project-testing.yml"
    ).read_text()
    for name in (
        "Validate Metal byte and vector storage",
        "Validate indexed OpenGL gather and resource aggregates",
        "Validate indexed DirectX gather and resource aggregates",
    ):
        step = ci_coverage.workflow_step_section(workflow, name)
        assert "test_vector_list_initialization.py" in step
        assert f'{REQUIRE_ENV}: "1"' in step
        assert (
            "--timeout-seconds" in step and "--junitxml" in step and "-n auto" in step
        )
        assert "if:" not in step and "continue-on-error" not in step


@pytest.mark.parametrize("target", ("metal", "directx", "opengl", "vulkan"))
@pytest.mark.parametrize("scalar", ("uint", "long", "ulong"))
def test_excess_vector_list_elements_fail_project_translation(tmp_path, target, scalar):
    (tmp_path / "excess.metal").write_text(f"""#include <metal_stdlib>
using namespace metal;
kernel void excess(device {scalar}* results [[buffer(0)]]) {{
    {scalar}4 value = {{{scalar}(1), {scalar}(2), {scalar}(3), {scalar}(4), {scalar}(5)}};
    results[0] = value.w;
}}
""")
    report = translate_project(
        ProjectConfig(
            root=tmp_path,
            targets=(target,),
            include_patterns=("excess.metal",),
            output_dir="out",
            workgroup_size=(1, 1, 1),
        ),
        format_output=False,
    ).to_json()
    assert report["summary"]["failedCount"] == 1
    assert report["summary"]["translatedCount"] == 0
    codes = {
        "metal": "project.translate.unsupported-feature",
        "vulkan": "project.translate.unsupported-feature",
        "directx": "project.translate.directx-aggregate-initializer-invalid",
        "opengl": "project.translate.opengl-aggregate-initializer-invalid",
    }
    assert any(
        item["code"] == codes[target] and item["severity"] == "error"
        for item in report["diagnostics"]
    ), report

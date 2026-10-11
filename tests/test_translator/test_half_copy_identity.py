"""Logical half copies preserve storage bits after physical GLSL widening."""

import json
import os
import struct
import sys
from pathlib import Path

import pytest

from crosstl import translate
from tests.test_translator.test_boolean_buffer_runtime import _bound_values, _request
from tests.test_translator.test_half_buffer_runtime import WORDS, _validate_half
from tests.test_translator.test_loop_updates import _execute
from tests.test_translator.test_software_subgroup_product import _package

REQUIRE_ENV = "CROSTL_REQUIRE_HALF_BUFFER_RUNTIME"
CASES = {
    "direct": ("", "half", "values[i]"),
    "local": ("", "half", "values[i]", "half value = values[i];", "value"),
    "constructor": ("", "half", "half(values[i])"),
    "helper": (
        "half identity(half value) { return value; }",
        "half",
        "identity(values[i])",
    ),
    "conditional": ("", "half", "i % 2u == 0u ? values[i] : values[i]"),
    "vector2": ("", "half2", "values[i]"),
    "vector4": ("", "half4", "values[i]"),
    "vector-constructor": ("", "half2", "half2(values[i].x, values[i].y)"),
    "vector-swizzle": ("", "half2", "values[i].xy"),
    "vector-index": ("", "half2", "half2(values[i][0], values[i][1])"),
    "vector-helper": (
        "half2 identity(half2 value) { return value; }",
        "half2",
        "identity(values[i])",
    ),
    "splat": ("", "half", "half2(values[i])[1]"),
    "dereference": ("", "half", "*(values + i)"),
    "shared-helper": (
        "half load_shared(threadgroup half* source) { return *source; }",
        "half",
        "values[i]",
        "threadgroup half shared[1]; shared[0] = values[i];",
        "load_shared(shared)",
    ),
    "structure": (
        "struct Cell { half first; half second; };",
        "Cell",
        "values[i]",
    ),
    "structure-members": (
        "struct Cell { half first; half second; };",
        "Cell",
        "values[i]",
        "Cell value; value.first = values[i].first; value.second = values[i].second;",
        "value",
    ),
    "array": (
        "",
        "half",
        "values[i]",
        "half value[2] = {values[i], values[i]};",
        "value[i % 2u]",
    ),
    "pointer": (
        "",
        "half",
        "values[i]",
        "const device half* pointer = values;",
        "pointer[i]",
    ),
    "pointer-helper": (
        "half load(const device half* source, uint i) { return source[i]; }",
        "half",
        "load(values, i)",
    ),
}


def _source(case):
    declaration, dtype, expression, *local = CASES[case]
    body = ""
    if local:
        body, expression = local
    return f"""#include <metal_stdlib>
using namespace metal;
{declaration}
kernel void half_identity(const device {dtype}* values [[buffer(0)]],
                          device {dtype}* results [[buffer(1)]],
                          uint i [[thread_position_in_grid]]) {{
    {body}
    results[i + 1u] = {expression};
}}
"""


def _widen(word):
    sign = (word & 0x8000) << 16
    exponent = (word >> 10) & 31
    fraction = word & 1023
    if exponent == 0:
        if not fraction:
            return sign
        high = fraction.bit_length() - 1
        return sign | ((103 + high) << 23) | ((fraction - (1 << high)) << (23 - high))
    return sign | ((255 if exponent == 31 else exponent + 112) << 23) | (fraction << 13)


def test_half_physical_representation_covers_every_word():
    values = [_widen(word) for word in range(65536)]
    assert len(set(values)) == 65536
    for word, value in enumerate(values):
        if word & 0x7C00 == 0x7C00:
            expected = ((word & 0x8000) << 16) | 0x7F800000 | ((word & 1023) << 13)
        else:
            expected = struct.unpack(
                "<I", struct.pack("<f", struct.unpack("<e", struct.pack("<H", word))[0])
            )[0]
        assert value == expected


@pytest.mark.parametrize("case", CASES)
def test_half_identity_does_not_emit_numeric_narrowing(tmp_path, case):
    source = tmp_path / "copy.metal"
    source.write_text(_source(case), encoding="utf-8")
    generated = translate(str(source), backend="opengl", format_output=False)
    assert "crossgl_round_half" not in generated


@pytest.mark.parametrize("expression", ("half(values[i])", "half2(values[i], 1.1).x"))
def test_half_numeric_constructors_still_require_rounding(tmp_path, expression):
    source = tmp_path / "narrow.metal"
    source.write_text(
        _source("direct")
        .replace("const device half*", "const device float*")
        .replace("= values[i];", f"= {expression};"),
        encoding="utf-8",
    )
    generated = translate(str(source), backend="opengl", format_output=False)
    assert "crossgl_round_half" in generated


def test_half_copy_controls_are_required_on_each_platform():
    from tools import ci_coverage

    workflow = Path(".github/workflows/demo-project-testing.yml").read_text()
    for name in (
        "Validate indexed OpenGL gather and resource aggregates",
        "Validate indexed DirectX gather and resource aggregates",
        "Validate Metal byte and vector storage",
    ):
        step = ci_coverage.workflow_step_section(workflow, name)
        assert f'{REQUIRE_ENV}: "1"' in step
        assert "tests/test_translator/test_half_copy_identity.py" in step
        assert "-n auto" in step
        assert "continue-on-error" not in step and "if:" not in step


def _case(root, target, case, words=WORDS):
    source = _source(case)
    _, descriptor, package = _package(
        root, target, "uint", (1, 1, 1), source=source, software_subgroups=False
    )
    (root / "case.json").write_text(
        json.dumps(
            {
                "case": case,
                "exhaustiveStart": words.start if isinstance(words, range) else None,
            }
        ),
        encoding="utf-8",
    )
    dtype = CASES[case][1]
    width = {"half": 1, "half2": 2, "half4": 4, "Cell": 2}[dtype]
    assert len(words) % width == 0
    values, guard = list(words), 0x3555
    dtype, encoding = "float16", "ieee754-binary16"
    if target == "opengl":
        values, guard = [_widen(word) for word in values], _widen(guard)
        dtype, encoding = "float32", "ieee754-binary32"
    initial = [guard] * (len(values) + 2 * width)

    def payload(words):
        return {
            "dtype": dtype,
            "encoding": encoding,
            "shape": [len(words)],
            "values": words,
        }

    inputs = {"values": payload(values), "results": payload(initial)}
    outputs = {"results": payload([guard] * width + values + [guard] * width)}
    request = _request(descriptor, package, inputs, outputs, len(values) // width)
    return source, request, _bound_values(descriptor, outputs)


@pytest.mark.parametrize("target", ("metal", "directx", "opengl"))
@pytest.mark.parametrize("case", CASES)
def test_half_copy_packages_keep_target_storage_layout(tmp_path, target, case):
    _, request, _ = _case(tmp_path, target, case)
    dtype = "float32" if target == "opengl" else "float16"
    encoding = "ieee754-binary32" if target == "opengl" else "ieee754-binary16"
    assert all(value.dtype == dtype for value in request.fixture.inputs)
    assert all(value.encoding == encoding for value in request.fixture.inputs)


@pytest.mark.parametrize("case", CASES)
def test_half_copy_executes_without_changing_payloads(tmp_path, case):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required half copies")
    target = {"darwin": "metal", "win32": "directx", "linux": "opengl"}[sys.platform]
    source, request, expected = _case(tmp_path, target, case)
    _execute(
        request,
        expected,
        tmp_path,
        original_source=source,
        original_entry="half_identity",
        validate=_validate_half,
    )


@pytest.mark.parametrize("start", (0, 16384, 32768, 49152))
def test_half_copy_preserves_every_binary16_word(tmp_path, start):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required half copies")
    target = {"darwin": "metal", "win32": "directx", "linux": "opengl"}[sys.platform]
    source, request, expected = _case(
        tmp_path, target, "direct", range(start, start + 16384)
    )
    _execute(
        request,
        expected,
        tmp_path,
        original_source=source,
        original_entry="half_identity",
        validate=_validate_half,
    )


MATH_RESULTS = {
    "sqrt": (1.4140625, 1.732421875, 0.70703125, 1.0),
    "sin": (0.9091796875, 0.14111328125, 0.4794921875, 0.84130859375),
    "log2": (1.0, 1.5849609375, -1.0, 0.0),
    "exp2": (4.0, 8.0, 1.4140625, 2.0),
}


def _math_case(root, target, operation, destination):
    source = f"""#include <metal_stdlib>
using namespace metal;
kernel void half_math(const device half* values [[buffer(0)]],
                      device {destination}* results [[buffer(1)]],
                      uint i [[thread_position_in_grid]]) {{
    results[i + 1u] = {destination}({operation}(values[i]));
}}
"""
    _, descriptor, package = _package(
        root, target, "uint", (1, 1, 1), source=source, software_subgroups=False
    )

    def payload(words, half):
        widened = not half or target == "opengl"
        return {
            "dtype": "float32" if widened else "float16",
            "encoding": "ieee754-binary32" if widened else "ieee754-binary16",
            "shape": [len(words)],
            "values": [_widen(word) for word in words] if widened else list(words),
        }

    expected = [
        struct.unpack("<H", struct.pack("<e", value))[0]
        for value in MATH_RESULTS[operation]
    ]
    inputs = {
        "values": payload([0x4000, 0x4200, 0x3800, 0x3C00], True),
        "results": payload([0x3555] * 6, destination == "half"),
    }
    outputs = {"results": payload([0x3555, *expected, 0x3555], destination == "half")}
    request = _request(descriptor, package, inputs, outputs, 4)
    return source, request, _bound_values(descriptor, outputs)


@pytest.mark.parametrize("target", ("metal", "directx", "opengl"))
@pytest.mark.parametrize("destination", ("half", "float"))
@pytest.mark.parametrize("operation", MATH_RESULTS)
def test_half_math_retains_rounding_before_storage(
    tmp_path, target, destination, operation
):
    _, request, _ = _math_case(tmp_path, target, operation, destination)
    if target == "opengl":
        generated = request.artifact_path.read_text(encoding="utf-8")
        assert f"crossgl_round_half1({operation}(values[i]))" in generated


@pytest.mark.parametrize("destination", ("half", "float"))
@pytest.mark.parametrize("operation", MATH_RESULTS)
def test_half_math_executes_with_logical_precision(tmp_path, destination, operation):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required half math")
    target = {"darwin": "metal", "win32": "directx", "linux": "opengl"}[sys.platform]
    source, request, expected = _math_case(tmp_path, target, operation, destination)
    _execute(
        request,
        expected,
        tmp_path,
        original_source=source,
        original_entry="half_math",
        validate=_validate_half,
        compare_with_original=target == "metal" and operation == "sin",
    )
    evidence = json.loads((tmp_path / "evidence.json").read_text())
    for record in evidence["records"].values():
        assert record["outputs"].keys() == expected.keys()
        for name, value in record["outputs"].items():
            reference = expected[name]
            assert {k: v for k, v in value.items() if k != "values"} == {
                k: v for k, v in reference.items() if k != "values"
            }
            assert len(value["values"]) == 6
            assert value["values"][0] == reference["values"][0]
            assert value["values"][-1] == reference["values"][-1]
            if value["dtype"] == "float32":
                for bits in value["values"]:
                    number = struct.unpack("<f", struct.pack("<I", bits))[0]
                    half = struct.unpack("<H", struct.pack("<e", number))[0]
                    assert _widen(half) == bits

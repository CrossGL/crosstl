"""Source byte conversions retain their range across widened arithmetic values."""

import os
import sys
from pathlib import Path

import pytest

from crosstl._crosstl import translate
from crosstl.translator.codegen.directx_codegen import DirectXContextualConversionError
from crosstl.translator.codegen.metal_codegen import UnsupportedMetalFeatureError
from tests.test_translator.test_boolean_buffer_runtime import _bound_values, _request
from tests.test_translator.test_loop_updates import _execute
from tests.test_translator.test_software_subgroup_product import _package

REQUIRE_ENV = "CROSTL_REQUIRE_NARROW_INTEGER_CONVERSIONS"
VALUES = [-257, -256, -129, -128, -1, 0, 1, 127, 128, 255, 256, 257, 65535]
FORMS = (
    "constructor",
    "cast",
    "initializer",
    "alias",
    "assignment",
    "argument",
    "return",
    "array",
    "member",
    "conditional",
    "single_evaluation",
    "vector",
    "vector_braces",
    "array_assignment",
    "buffer_store",
    "buffer_offset",
    "compound",
    "prefix",
    "postfix",
)


def _narrow(value, signed):
    value %= 256
    return value - 256 if signed and value >= 128 else value


def _case(root, target, signed, form):
    byte = "char" if signed else "uchar"
    declarations = ""
    expression = "int(value)"
    body = f"{byte} value = input_value;"
    if form == "constructor":
        body = f"int value = int({byte}(input_value));"
    elif form == "cast":
        body = f"int value = int(({byte})input_value);"
    elif form == "alias":
        declarations = f"using Byte = {byte};"
        body = "Byte value = Byte(input_value);"
    elif form == "assignment":
        body = f"{byte} value = 0; value = input_value;"
    elif form == "argument":
        declarations = f"int promote({byte} value) {{ return int(value); }}"
        body = "int value = promote(input_value);"
    elif form == "return":
        declarations = f"{byte} narrow(int value) {{ return value; }}"
        body = "int value = int(narrow(input_value));"
    elif form == "array":
        body = f"{byte} values[2] = {{{byte}(0), {byte}(input_value)}};"
        expression = "int(values[1])"
    elif form == "member":
        declarations = f"struct Cell {{ {byte} value; }};"
        body = "Cell cell; cell.value = input_value;"
        expression = "int(cell.value)"
    elif form == "conditional":
        body = f"{byte} value = input_value < 0 ? input_value : input_value + 256;"
    elif form == "single_evaluation":
        declarations = "int next_value(thread int& value) { int old = value; value += 1; return old; }"
        body = f"int previous = input_value; int value = int({byte}(next_value(previous)));"
        expression = "value + (previous - input_value - 1) * 1000"
    elif form == "vector":
        body = f"{byte}2 value = {byte}2(int2(input_value, input_value + 2));"
        expression = "int(value.x) + int(value.y) * 1024"
    elif form == "vector_braces":
        body = f"{byte}2 value = {{{byte}(input_value), {byte}(input_value + 2)}};"
        expression = "int(value.x) + int(value.y) * 1024"
    elif form == "array_assignment":
        body = f"{byte} values[2]; values[1] = input_value;"
        expression = "int(values[1])"
    elif form in {"buffer_store", "buffer_offset"}:
        body = ""
        expression = "input_value"
        if form == "buffer_offset":
            body = "results += 1;"
    elif form == "compound":
        body += " value += 3;"
    elif form == "prefix":
        expression = "int(++value)"
    elif form == "postfix":
        body += " int old = int(value++);"
        expression = "old + int(value) * 1024"
    output_type = byte if form in {"buffer_store", "buffer_offset"} else "int"
    result_index = "tid" if form == "buffer_offset" else "tid + 1u"
    source = f"""#include <metal_stdlib>
using namespace metal;
{declarations}
kernel void byte_conversions(const device int* inputs [[buffer(0)]],
                             device {output_type}* results [[buffer(1)]],
                             uint tid [[thread_position_in_grid]]) {{
    int input_value = inputs[tid];
    {body}
    results[{result_index}] = {expression};
}}
"""
    _, descriptor, package = _package(
        root, target, "int", (1, 1, 1), source=source, software_subgroups=False
    )
    values = []
    for value in VALUES:
        narrowed = _narrow(value, signed)
        if form in {"vector", "vector_braces"}:
            narrowed += _narrow(value + 2, signed) * 1024
        elif form == "compound":
            narrowed = _narrow(narrowed + 3, signed)
        elif form == "prefix":
            narrowed = _narrow(narrowed + 1, signed)
        elif form == "postfix":
            narrowed += _narrow(narrowed + 1, signed) * 1024
        values.append(narrowed)
    count = len(VALUES) + 2
    dtype = "int32"
    guard = 987654
    if form in {"buffer_store", "buffer_offset"}:
        dtype = ("int" if signed else "uint") + ("8" if target == "metal" else "32")
        guard = 91
    inputs = {
        "inputs": {"dtype": "int32", "shape": [len(VALUES)], "values": VALUES},
        "results": {"dtype": dtype, "shape": [count], "values": [guard] * count},
    }
    outputs = {
        "results": {
            "dtype": dtype,
            "shape": [count],
            "values": [guard, *values, guard],
        }
    }
    return (
        source,
        _request(descriptor, package, inputs, outputs, len(VALUES)),
        _bound_values(descriptor, outputs),
    )


@pytest.mark.parametrize("target", ("metal", "directx", "opengl"))
@pytest.mark.parametrize("signed", (True, False), ids=("signed", "unsigned"))
@pytest.mark.parametrize("form", FORMS)
def test_narrow_integer_conversion_packages(tmp_path, target, signed, form):
    _, request, _ = _case(tmp_path, target, signed, form)
    if target == "directx" and form in {"buffer_store", "buffer_offset"}:
        stores = [
            line
            for line in request.artifact_path.read_text().splitlines()
            if line.strip().startswith("results[") and " = " in line
        ]
        assert len(stores) == 1
        assert (" >> 24)" if signed else " & 255u)") in stores[0]


@pytest.mark.parametrize("signed", (True, False), ids=("signed", "unsigned"))
@pytest.mark.parametrize("form", FORMS)
def test_narrow_integer_conversions_execute_natively(tmp_path, signed, form):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required native byte conversions")
    target = {"darwin": "metal", "win32": "directx", "linux": "opengl"}[sys.platform]
    source, request, expected = _case(tmp_path, target, signed, form)
    _execute(
        request,
        expected,
        tmp_path,
        original_source=source if target == "metal" else None,
        original_entry="byte_conversions",
    )


@pytest.mark.parametrize("target", ("metal", "directx"))
@pytest.mark.parametrize("update", ("values[index++] += 3", "values[index++]++"))
def test_byte_update_does_not_repeat_an_index_effect(tmp_path, target, update):
    source = tmp_path / "update.metal"
    source.write_text(
        "#include <metal_stdlib>\nusing namespace metal;\n"
        "kernel void update(device int* results [[buffer(0)]]) {\n"
        "uchar values[2] = {uchar(255), uchar(1)}; int index = 0;\n"
        f"{update}; results[0] = int(values[0]) + index;\n}}\n"
    )
    error_type = (
        UnsupportedMetalFeatureError
        if target == "metal"
        else DirectXContextualConversionError
    )
    with pytest.raises(error_type) as error:
        translate(
            str(source), backend=target, source_backend="metal", format_output=False
        )
    assert error.value.reason == "byte-update-lvalue-unsupported"


def test_byte_conversions_and_directx_random_are_required_in_ci():
    workflow = (
        Path(__file__).parents[2] / ".github/workflows/mlx-gather-roundtrip.yml"
    ).read_text()
    assert workflow.count(f'{REQUIRE_ENV}: "1"') == 3
    assert (
        workflow.count("tests/test_translator/test_narrow_integer_conversions.py") == 3
    )
    step = workflow.split("- name: Validate translated DirectX random kernels\n", 1)[
        1
    ].split("      - name:", 1)[0]
    assert "--target directx --output-dir .mlx-gather-directx/random" in step
    assert "set -euo pipefail" in step and "--timeout-seconds 300" in step
    assert "continue-on-error" not in step

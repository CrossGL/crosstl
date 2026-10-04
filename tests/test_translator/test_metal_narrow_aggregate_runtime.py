"""Native aggregate storage must retain narrow member sizes and union overlap."""

import os
import sys

import pytest

from tests.test_translator.test_boolean_buffer_runtime import _bound_values, _request
from tests.test_translator.test_byte_buffer_runtime import REQUIRE_ENV
from tests.test_translator.test_loop_updates import _execute
from tests.test_translator.test_software_subgroup_product import _package

TYPES = ("char", "uchar", "short", "ushort")
FORMS = ("vector", "array", "nested", "generic", "indexed")


def _case(root, scalar, form):
    width = 4 if scalar in {"char", "uchar"} else 2
    vector = f"{scalar}{width}"
    if form == "vector":
        declarations = f"union Bits {{ uint word; {vector} parts; }};"
        initialize = "value.word = 0x817fff00u;"
        expressions = [f"value.parts.{lane}" for lane in "xyzw"[:width]]
        size = 4
    else:
        if form in {"array", "indexed"}:
            declarations = f"union Bits {{ uint2 words; {vector} parts[2]; }};"
            path = "value.parts"
        elif form == "nested":
            declarations = (
                f"struct Inner {{ {vector} parts[2]; }};\n"
                "union Bits { uint2 words; Inner inner; };"
            )
            path = "value.inner.parts"
        else:
            declarations = (
                "template<typename T> struct Inner { T parts[2]; };\n"
                f"union Bits {{ uint2 words; Inner<{vector}> inner; }};"
            )
            path = "value.inner.parts"
        initialize = "value.words = uint2(0x817fff00u, 0x01234567u);"
        expressions = [
            f"{path}[{i}].{lane}" for i in range(2) for lane in "xyzw"[:width]
        ]
        size = 8
    writes = "\n".join(
        f"    results[{i + 1}] = int({expression});"
        for i, expression in enumerate(expressions)
    )
    if form == "indexed":
        writes = f"""    for (uint i = 0; i < 2; ++i) {{
        for (uint lane = 0; lane < {width}; ++lane) {{
            results[1 + i * {width} + lane] = int(value.parts[i][lane]);
        }}
    }}"""
    source = f"""#include <metal_stdlib>
using namespace metal;
{declarations}
kernel void aggregate_storage(device int* results [[buffer(0)]]) {{
    Bits value;
    {initialize}
{writes}
    results[{len(expressions) + 1}] = int(sizeof(Bits));
}}
"""
    _, descriptor, package = _package(root, "metal", "uint", (1, 1, 1), source=source)
    byte_pattern = bytes.fromhex("00ff7f8167452301")[:size]
    step = 4 // width
    signed = scalar in {"char", "short"}
    values = [
        int.from_bytes(byte_pattern[i : i + step], "little", signed=signed)
        for i in range(0, size, step)
    ]
    initial = [123456] * (len(values) + 3)
    inputs = {"results": {"dtype": "int32", "shape": [len(initial)], "values": initial}}
    outputs = {
        "results": {
            "dtype": "int32",
            "shape": [len(initial)],
            "values": [123456, *values, size, 123456],
        }
    }
    request = _request(descriptor, package, inputs, outputs, 1)
    return source, request, _bound_values(descriptor, outputs)


@pytest.mark.parametrize("scalar", TYPES)
@pytest.mark.parametrize("form", FORMS)
def test_narrow_aggregate_storage_keeps_native_declarations(tmp_path, scalar, form):
    _, request, _ = _case(tmp_path, scalar, form)
    translated = request.artifact_path.read_text()
    width = 4 if scalar in {"char", "uchar"} else 2
    assert f"{scalar}{width} parts" in translated
    assert "union Bits" in translated


@pytest.mark.parametrize("scalar", TYPES)
@pytest.mark.parametrize("form", FORMS)
def test_narrow_aggregate_storage_executes_original_and_generated(
    tmp_path, scalar, form
):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required native aggregate storage")
    assert sys.platform == "darwin", "the required aggregate storage gate needs Metal"
    source, request, expected = _case(tmp_path, scalar, form)
    _execute(
        request,
        expected,
        tmp_path,
        original_source=source,
        original_entry="aggregate_storage",
    )

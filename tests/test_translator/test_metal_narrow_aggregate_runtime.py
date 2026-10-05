"""Native aggregate storage must retain narrow member sizes and union overlap."""

import os
import sys

import pytest

from crosstl import translate
from crosstl.translator import parse
from crosstl.translator.codegen.metal_codegen import MetalCodeGen
from tests.test_translator.test_boolean_buffer_runtime import _bound_values, _request
from tests.test_translator.test_byte_buffer_runtime import REQUIRE_ENV
from tests.test_translator.test_loop_updates import _execute
from tests.test_translator.test_metal_builtin_ownership import _compile
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


@pytest.mark.parametrize("scalar", ["char", "uchar"])
@pytest.mark.parametrize("form", ["call", "positional", "named", "default"])
def test_byte_struct_constructor_fields_use_native_storage(scalar, form):
    initializer = {
        "call": "Pair(index, value)",
        "positional": "Pair { index, value }",
        "named": "Pair { value: value, index: index }",
        "default": "Pair { index: index }",
    }[form]
    source = f"""shader Aggregate {{
        struct Pair {{ uint index; {scalar} value; }}
        Pair make(uint index, {scalar} value) {{ return {initializer}; }}
    }}"""
    generated = MetalCodeGen().generate(parse(source))
    assert f"{scalar} value;" in generated
    assert f"return Pair{{index, {scalar}(" in generated


def _initializer_source(scalar, width, form):
    value_type = scalar if width == 1 else f"{scalar}{width}"
    declarations = f"struct Packet {{ uint index; {value_type} value; }};"
    packet_type = "Packet"
    if form == "alias":
        declarations = (
            f"using Payload = {value_type};\n"
            "struct Packet { uint index; Payload value; };"
        )
    elif form == "generic":
        declarations = "template<typename T> struct Packet { uint index; T value; };"
        packet_type = f"Packet<{value_type}>"
    elif form == "nested":
        declarations += "\nstruct Outer { Packet item; };"
    args = ", ".join(f"{scalar}(raw + {lane * 37}u)" for lane in range(width))
    value = args if width == 1 else f"{value_type}({args})"
    initialize = f"{packet_type} item = make_packet(cursor++, {value});"
    if form == "side-effect":
        initialize = (
            f"Packet item = Packet{{cursor, {scalar}(raw + (cursor++ - tid))}};"
        )
    item = "item"
    if form == "nested":
        initialize = f"Outer outer = Outer{{make_packet(cursor++, {value})}};"
        item = "outer.item"
    writes = "\n".join(
        f"    results[offset + {lane}] = int({item}.value"
        + (f".{component}" if width > 1 else "")
        + ");"
        for lane, component in enumerate("xyzw"[:width])
    )
    return f"""#include <metal_stdlib>
using namespace metal;
{declarations}
{packet_type} make_packet(uint index, {value_type} value) {{
    return {packet_type}{{index, value}};
}}
kernel void aggregate_initializers(const device uint* inputs [[buffer(0)]],
                                   device int* results [[buffer(1)]],
                                   uint tid [[thread_position_in_grid]]) {{
    uint raw = inputs[tid];
    uint cursor = tid;
    {initialize}
    {packet_type} zero = {packet_type}{{tid}};
    uint offset = 4u + tid * {width + 4}u;
{writes}
    results[offset + {width}] = int({item}.index);
    results[offset + {width + 1}] = int(cursor);
    results[offset + {width + 2}] = int(zero.value{'.x' if width > 1 else ''});
    results[offset + {width + 3}] = int(sizeof({packet_type}));
}}
"""


def _initializer_case(root, scalar, width, form):
    source = _initializer_source(scalar, width, form)
    _, descriptor, package = _package(root, "metal", "uint", (1, 1, 1), source=source)
    values = list(range(256))
    expected = [-123456] * 4
    for tid, raw in enumerate(values):
        for lane in range(width):
            byte = (raw + lane * 37) & 255
            expected.append(byte - 256 if scalar == "char" and byte >= 128 else byte)
        expected.extend([tid, tid + 1, 0, 8])
    expected.extend([-123456] * 4)
    inputs = {
        "inputs": {"dtype": "uint32", "shape": [len(values)], "values": values},
        "results": {
            "dtype": "int32",
            "shape": [len(expected)],
            "values": [-123456] * len(expected),
        },
    }
    outputs = {
        "results": {"dtype": "int32", "shape": [len(expected)], "values": expected}
    }
    return (
        source,
        _request(descriptor, package, inputs, outputs, len(values)),
        _bound_values(descriptor, outputs),
    )


@pytest.mark.parametrize("scalar", ["char", "uchar"])
@pytest.mark.parametrize("width", [1, 2, 3, 4])
@pytest.mark.parametrize("form", ["plain", "alias", "generic", "nested"])
def test_narrow_initializers_compile_through_public_and_saved_crossgl(
    tmp_path, scalar, width, form
):
    source, request, _ = _initializer_case(tmp_path, scalar, width, form)
    source_path = tmp_path / "original.metal"
    source_path.write_text(source)
    canonical_path = tmp_path / "saved.cgl"
    canonical_path.write_text(
        translate(str(source_path), backend="cgl", format_output=False)
    )
    canonical_output = translate(
        str(canonical_path), backend="metal", format_output=False
    )
    assert canonical_output == request.artifact_path.read_text()
    for name, text in (("original", source), ("generated", canonical_output)):
        directory = tmp_path / name
        directory.mkdir()
        _compile(text, "metal", directory)


@pytest.mark.parametrize("scalar", ["char", "uchar"])
@pytest.mark.parametrize("width", [1, 2, 3, 4])
@pytest.mark.parametrize("form", ["plain", "alias", "generic", "nested"])
def test_narrow_initializers_execute_original_and_generated(
    tmp_path, scalar, width, form
):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required native aggregate initialization")
    assert sys.platform == "darwin", "aggregate initialization requires Metal"
    source, request, expected = _initializer_case(tmp_path, scalar, width, form)
    _execute(
        request,
        expected,
        tmp_path,
        original_source=source,
        original_entry="aggregate_initializers",
    )


def _array_initializer_case(root, scalar, form):
    components = [f"{scalar}(raw + {lane * 37}u)" for lane in range(4)]
    if form == "array":
        field = f"{scalar} values[4]"
        initializer = "{" + ", ".join(components) + "}"
        accesses = [f"values[{lane}]" for lane in range(4)]
    elif form == "matrix":
        field = f"{scalar} values[2][2]"
        initializer = (
            "{"
            + ", ".join(
                "{" + ", ".join(components[start : start + 2]) + "}" for start in (0, 2)
            )
            + "}"
        )
        accesses = [f"values[{lane // 2}][{lane % 2}]" for lane in range(4)]
    else:
        field = f"{scalar}2 values[2]"
        pairs = [", ".join(components[start : start + 2]) for start in (0, 2)]
        initializer = (
            "{"
            + ", ".join(
                "{" + pair + "}" if form == "vector-braces" else f"{scalar}2({pair})"
                for pair in pairs
            )
            + "}"
        )
        accesses = [f"values[{lane // 2}].{'xy'[lane % 2]}" for lane in range(4)]
    writes = "\n".join(
        f"    results[offset + {lane}] = int(item.{access});"
        for lane, access in enumerate(accesses)
    )
    zero_writes = "\n".join(
        f"    results[offset + {lane + 4}] = int(zero.{access});"
        for lane, access in enumerate(accesses)
    )
    source = f"""#include <metal_stdlib>
using namespace metal;
struct Packet {{ uint index; {field}; }};
Packet make_packet(uint index, uint raw) {{
    return Packet{{index, {initializer}}};
}}
kernel void aggregate_array_initializers(const device uint* inputs [[buffer(0)]],
                                         device int* results [[buffer(1)]],
                                         uint tid [[thread_position_in_grid]]) {{
    uint cursor = tid;
    Packet item = make_packet(cursor++, inputs[tid]);
    Packet zero = Packet{{tid}};
    uint offset = 4u + tid * 11u;
{writes}
{zero_writes}
    results[offset + 8] = int(item.index);
    results[offset + 9] = int(cursor);
    results[offset + 10] = int(sizeof(Packet));
}}
"""
    _, descriptor, package = _package(root, "metal", "uint", (1, 1, 1), source=source)
    values = list(range(256))
    expected = [-123456] * 4
    for tid, raw in enumerate(values):
        for lane in range(4):
            byte = (raw + lane * 37) & 255
            expected.append(byte - 256 if scalar == "char" and byte >= 128 else byte)
        expected.extend([0, 0, 0, 0, tid, tid + 1, 8])
    expected.extend([-123456] * 4)
    inputs = {
        "inputs": {"dtype": "uint32", "shape": [len(values)], "values": values},
        "results": {
            "dtype": "int32",
            "shape": [len(expected)],
            "values": [-123456] * len(expected),
        },
    }
    outputs = {
        "results": {"dtype": "int32", "shape": [len(expected)], "values": expected}
    }
    return (
        source,
        _request(descriptor, package, inputs, outputs, len(values)),
        _bound_values(descriptor, outputs),
    )


@pytest.mark.parametrize("scalar", ["char", "uchar"])
@pytest.mark.parametrize("form", ["array", "matrix", "vector-array", "vector-braces"])
def test_byte_array_initializers_execute_original_and_generated(tmp_path, scalar, form):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required native aggregate initialization")
    assert sys.platform == "darwin", "aggregate initialization requires Metal"
    source, request, expected = _array_initializer_case(tmp_path, scalar, form)
    _execute(
        request,
        expected,
        tmp_path,
        original_source=source,
        original_entry="aggregate_array_initializers",
    )


@pytest.mark.parametrize("scalar", ["char", "uchar"])
@pytest.mark.parametrize("form", ["array", "matrix", "vector-array", "vector-braces"])
def test_byte_array_initializers_translate_and_compile(tmp_path, scalar, form):
    source, request, _ = _array_initializer_case(tmp_path, scalar, form)
    for name, text in (
        ("original", source),
        ("generated", request.artifact_path.read_text()),
    ):
        directory = tmp_path / name
        directory.mkdir()
        _compile(text, "metal", directory)


@pytest.mark.parametrize("scalar", ["char", "uchar"])
def test_byte_field_initializer_executes_once(tmp_path, scalar):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required native aggregate initialization")
    assert sys.platform == "darwin", "aggregate initialization requires Metal"
    source, request, expected = _initializer_case(tmp_path, scalar, 1, "side-effect")
    _execute(
        request,
        expected,
        tmp_path,
        original_source=source,
        original_entry="aggregate_initializers",
    )

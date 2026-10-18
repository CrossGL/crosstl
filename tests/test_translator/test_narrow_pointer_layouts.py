"""Keep widened and packed narrow storage distinct through helper pointers.

DirectX stores each byte of a root ``device uchar*`` buffer in its own 32-bit
carrier, while a byte view cast from a ``device uint*`` buffer keeps four bytes
per word. Helper pointer parameters must read the layout their callers pass
(#2145) and reject helpers that receive both layouts.
"""

import os
import shutil
import struct
import sys

import pytest

from crosstl import translate
from crosstl.project import ProjectConfig, translate_project
from crosstl.translator.codegen.directx_codegen import (
    DirectXResourcePointerParameterError,
)
from tests.test_translator.test_boolean_buffer_runtime import _bound_values, _request
from tests.test_translator.test_loop_updates import _execute
from tests.test_translator.test_metal_builtin_ownership import _compile
from tests.test_translator.test_software_subgroup_product import _package

REQUIRE_ENV = "CROSTL_REQUIRE_SCALAR_ALIASES"
COUNT = 16
GUARD = 0xDEADBEEF
BYTE_COUNT = 32

HELPERS = """#include <metal_stdlib>
using namespace metal;
uint read_inner(const device {scalar}* bytes, uint index) {{
    {scalar} value = bytes[index];
    return uint(int(value));
}}
uint read_outer(const device {scalar}* bytes, uint index) {{
    const device {scalar}* shifted = bytes + 1u;
    return read_inner(shifted, index) + 3u * read_inner(bytes, index + 2u);
}}
"""


def widened_source(scalar):
    return (HELPERS + """uint read_rows(const device {scalar}* rows[2], uint index) {{
    return read_inner(rows[0], index) ^ (read_inner(rows[1], index) << 8u);
}}
kernel void read_widened(const device {scalar}* values [[buffer(0)]],
                         device uint* results [[buffer(1)]],
                         uint i [[thread_position_in_grid]]) {{
    const device {scalar}* rows[2];
    rows[0] = values + 5u;
    for (int row = 1; row < 2; ++row) {{
        rows[row] = rows[row - 1] + 7u;
    }}
    results[4u + 3u * i] = read_outer(values + 1u, i);
    results[5u + 3u * i] = read_rows(rows, i);
    results[6u + 3u * i] = uint(int(values[i]));
}}
""").format(scalar=scalar)


def packed_source():
    return (HELPERS + """kernel void read_packed(const device uint* words [[buffer(0)]],
                        device uint* results [[buffer(1)]],
                        uint i [[thread_position_in_grid]]) {{
    const device {scalar}* bytes = (const device {scalar}*)words;
    results[4u + 2u * i] = read_outer(bytes + 1u, i);
    results[5u + 2u * i] = read_inner(bytes, i + 3u);
}}
""").format(scalar="uchar")


MIXED_SOURCES = {
    # The reduced reproduction from #2145.
    "direct": (
        """#include <metal_stdlib>
using namespace metal;
uint read_byte(const device uchar* bytes, uint index) {
    return uint(bytes[index]);
}
kernel void read_storage(const device uchar* values [[buffer(0)]],
                         const device uint* words [[buffer(1)]],
                         device uint* output [[buffer(2)]]) {
    output[0] = read_byte(values + 1, 0);
    output[1] = read_byte((const device uchar*)words + 1, 0);
}
"""
    ),
    # Both layouts reach the same leaf through different forwarding helpers.
    "forwarded": (
        """#include <metal_stdlib>
using namespace metal;
uint read_byte(const device uchar* bytes, uint index) {
    return uint(bytes[index]);
}
uint forward_widened(const device uchar* bytes) { return read_byte(bytes + 1, 0); }
uint forward_packed(const device uchar* bytes) { return read_byte(bytes, 1); }
kernel void read_storage(const device uchar* values [[buffer(0)]],
                         const device uint* words [[buffer(1)]],
                         device uint* output [[buffer(2)]]) {
    output[0] = forward_widened(values);
    output[1] = forward_packed((const device uchar*)words);
}
"""
    ),
}


def _logical_bytes(scalar):
    bits = [(index * 73 + 29) & 255 for index in range(BYTE_COUNT)]
    if scalar == "char":
        return [value - 256 if value >= 128 else value for value in bits]
    return bits


def _widened_expected(scalar):
    values = [value & 0xFFFFFFFF for value in _logical_bytes(scalar)]
    expected = [GUARD] * 4
    for i in range(COUNT):
        expected.append((values[2 + i] + 3 * values[3 + i]) & 0xFFFFFFFF)
        expected.append(values[5 + i] ^ ((values[12 + i] << 8) & 0xFFFFFFFF))
        expected.append(values[i])
    return expected + [GUARD] * 4


def _packed_expected():
    values = _logical_bytes("uchar")
    expected = [GUARD] * 4
    for i in range(COUNT):
        expected.append(values[2 + i] + 3 * values[3 + i])
        expected.append(values[3 + i])
    return expected + [GUARD] * 4


def _translate(tmp_path, source, target="directx"):
    path = tmp_path / "source.metal"
    path.write_text(source, encoding="utf-8")
    return translate(str(path), backend=target, format_output=False)


def _function_body(generated, name):
    marker = f"uint {name}("
    start = 0
    while True:
        start = generated.index(marker, start)
        signature_end = generated.index(")", start)
        if generated.startswith(" {", signature_end + 1):
            break
        start = signature_end
    return generated[signature_end : generated.index("\n}\n", signature_end)]


@pytest.mark.parametrize("scalar", ("uchar", "char"))
def test_widened_helper_reads_one_carrier_per_element(tmp_path, scalar):
    generated = _translate(tmp_path, widened_source(scalar))

    inner = _function_body(generated, "read_inner")
    assert "bytes[uint((bytes_offset + index))]" in inner
    assert "/ 4" not in generated and "% 4) * 8" not in generated
    if scalar == "char":
        assert "StructuredBuffer<int> values" in generated
        assert "<< 24u) >> 24)" in inner
    else:
        assert "StructuredBuffer<uint> values" in generated
        assert "& 255u)" in inner
    assert "read_outer(values, int64_t(1u), i)" in generated
    assert "read_inner(bytes, int64_t(shifted_offset), index)" in generated
    assert "read_rows(values, int64_t(rows_offsets[0]), int64_t(rows_offsets[1])" in (
        generated
    )
    assert "rows_offsets[uint((rows_base + int64_t(1)))]" in generated
    if shutil.which("dxc"):
        _compile(generated, "directx", tmp_path)
    if shutil.which("xcrun"):
        original = tmp_path / "original"
        original.mkdir()
        _compile(widened_source(scalar), "metal", original)


def test_packed_helper_reads_keep_word_lanes(tmp_path):
    generated = _translate(tmp_path, packed_source())

    inner = _function_body(generated, "read_inner")
    assert "bytes[uint(((bytes_offset + index)) / 4)] >> " in inner
    assert "% 4) * 8)) & 255u" in inner
    assert "read_outer(words, int64_t((bytes_offset + 1u)), i)" in generated
    assert "read_inner(words, int64_t(bytes_offset), (i + 3u))" in generated
    assert "read_inner(bytes, int64_t(shifted_offset), index)" in generated
    if shutil.which("dxc"):
        _compile(generated, "directx", tmp_path)


@pytest.mark.parametrize("case", sorted(MIXED_SOURCES))
def test_mixed_narrow_helper_layouts_are_rejected(tmp_path, case):
    with pytest.raises(DirectXResourcePointerParameterError) as error:
        _translate(tmp_path, MIXED_SOURCES[case])

    assert error.value.reason == "mixed-narrow-storage-layout"
    assert error.value.function_name == "read_byte"
    assert error.value.parameter_name == "bytes"
    assert "both widened narrow-element storage and packed words" in str(error.value)


def widened_store_source(scalar):
    return """#include <metal_stdlib>
using namespace metal;
void write_byte(device {scalar}* bytes, uint index, {scalar} value) {{
    bytes[index] = value;
}}
void forward_write(device {scalar}* bytes, uint index) {{
    write_byte(bytes + 1u, index, {scalar}(index + 7u));
}}
kernel void write_widened(device {scalar}* values [[buffer(0)]],
                          uint i [[thread_position_in_grid]]) {{
    forward_write(values, i);
    write_byte(values, i + 20u, {scalar}(3));
}}
""".format(scalar=scalar)


PACKED_STORE_SOURCE = """#include <metal_stdlib>
using namespace metal;
void write_byte(device uchar* bytes, uint index, uchar value) {
    bytes[index] = value;
}
kernel void write_packed(device uint* words [[buffer(0)]],
                         uint i [[thread_position_in_grid]]) {
    write_byte((device uchar*)words, i, uchar(i));
}
"""


@pytest.mark.parametrize("scalar", ("uchar", "char"))
def test_widened_helper_stores_write_one_carrier_per_element(tmp_path, scalar):
    # Helpers precede their callers, so the first pass reaches the store
    # through the default packed view before any call site is recorded; the
    # widened layout is assumed once and proven by the complete pass.
    generated = _translate(tmp_path, widened_store_source(scalar))

    carrier = "int" if scalar == "char" else "uint"
    assert f"RWStructuredBuffer<{carrier}> values" in generated
    assert (
        f"void write_byte(RWStructuredBuffer<{carrier}> bytes, "
        f"int64_t bytes_offset, uint index, {carrier} value) {{"
    ) in generated
    assert "bytes[uint((bytes_offset + index))] = " in generated
    assert "write_byte(bytes, int64_t((bytes_offset + 1u)), index, " in generated
    assert "forward_write(values, int64_t(0), i);" in generated
    assert "ByteAddressBuffer" not in generated
    assert "/ 4" not in generated and "% 4) * 8" not in generated
    if shutil.which("dxc"):
        _compile(generated, "directx", tmp_path)


def test_packed_helper_store_still_requires_byte_address_storage(tmp_path):
    from crosstl.translator.codegen.pointer_reinterpret import (
        PointerReinterpretationError,
    )

    # The hypothesis cannot be proven when the only caller passes packed
    # words, so the write keeps its original diagnostic.
    with pytest.raises(PointerReinterpretationError) as error:
        _translate(tmp_path, PACKED_STORE_SOURCE)

    assert error.value.reason == "write-requires-byte-address-storage"


@pytest.mark.parametrize("case", sorted(MIXED_SOURCES))
def test_mixed_narrow_helper_layout_is_a_project_diagnostic(tmp_path, case):
    (tmp_path / "mixed.metal").write_text(MIXED_SOURCES[case], encoding="utf-8")
    report = translate_project(
        ProjectConfig(
            root=tmp_path,
            include_patterns=("mixed.metal",),
            targets=("directx",),
            output_dir="out",
        ),
        format_output=False,
    ).to_json()

    assert report["summary"]["translatedCount"] == 0
    assert report["summary"]["failedCount"] == 1
    codes = {diagnostic["code"] for diagnostic in report["diagnostics"]}
    assert "project.translate.directx-resource-pointer-parameter-unsupported" in codes
    assert not (tmp_path / "out").exists() or not any(
        (tmp_path / "out").rglob("*.hlsl")
    )


@pytest.mark.parametrize(
    "case",
    ("widened-uchar", "widened-char", "packed-uchar"),
)
def test_narrow_helper_layouts_execute(tmp_path, case):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required narrow-layout execution")
    target = {"darwin": "metal", "win32": "directx", "linux": "opengl"}[sys.platform]
    layout, scalar = case.split("-")
    source = widened_source(scalar) if layout == "widened" else packed_source()
    _, descriptor, package = _package(
        tmp_path, target, scalar, (1, 1, 1), source=source, software_subgroups=False
    )
    input_name = "values" if layout == "widened" else "words"
    member_types = {
        binding["scalarLayout"].get("memberName", binding["name"]): binding[
            "scalarLayout"
        ]["elementType"]
        for binding in descriptor["bindings"]
        if binding.get("kind") == "buffer"
    }
    if layout == "widened":
        values = _logical_bytes(scalar)
        expected = _widened_expected(scalar)
        assert member_types[input_name] in (
            {"int8", "int32"} if scalar == "char" else {"uint8", "uint32"}
        )
    else:
        payload = bytes(_logical_bytes("uchar"))
        values = list(struct.unpack(f"<{len(payload) // 4}I", payload))
        expected = _packed_expected()
        assert member_types[input_name] == "uint32"
    inputs = {
        input_name: {
            "dtype": member_types[input_name],
            "shape": [len(values)],
            "values": values,
        },
        "results": {
            "dtype": "uint32",
            "shape": [len(expected)],
            "values": [GUARD] * len(expected),
        },
    }
    outputs = {"results": {**inputs["results"], "values": expected}}
    request = _request(descriptor, package, inputs, outputs, COUNT)
    _execute(
        request,
        _bound_values(descriptor, outputs),
        tmp_path,
        original_source=source if target == "metal" else None,
        original_entry=f"read_{layout}",
    )


def test_narrow_layout_execution_runs_in_the_scalar_alias_gate():
    from tools import ci_coverage

    workflow = os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        ".github",
        "workflows",
        "demo-project-testing.yml",
    )
    with open(workflow, encoding="utf-8") as stream:
        text = stream.read()
    step = ci_coverage.workflow_step_section(text, "Validate scalar alias resolution")
    assert f'{REQUIRE_ENV}: "1"' in step
    assert "tests/test_translator/test_narrow_pointer_layouts.py" in step
    assert "continue-on-error" not in step and "if:" not in step

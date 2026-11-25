"""8- and 16-bit struct fields passed to ``out``/``inout`` parameters in Metal output.

A struct keeps the declared width of a narrow integer field (``uchar value;``), while Metal output computes with
narrow integers in the widened type and declares a reference parameter as ``thread uint&``. The field cannot bind
that reference, so the call copies it through a widened thread temporary and stores it back, as CrossGL's
copy-in/copy-out parameters specify. The generated Metal, read directly and through the saved CrossGL, compiles
with warnings as errors and computes on the GPU what the source computes, including 8-bit wrap-around.
"""

import re
import shutil
import struct
import subprocess

import pytest

import crosstl

SOURCE = """#include <metal_stdlib>
using namespace metal;
int increment(thread uchar& value) { value += 3; return int(value); }
void lower(thread short& value, thread uchar& low) { value -= 300; low = uchar(value); }
struct Cell { uchar value; short wide; uchar bytes[2]; };
kernel void byte_conversions(const device int* inputs [[buffer(0)]],
                             device int* results [[buffer(1)]],
                             uint tid [[thread_position_in_grid]]) {
    int input_value = inputs[tid];
    Cell cell;
    cell.value = input_value;
    cell.wide = short(input_value);
    cell.bytes[0] = uchar(input_value >> 1);
    cell.bytes[1] = 0;
    increment(cell.value);
    int first = increment(cell.bytes[0]);
    lower(cell.wide, cell.bytes[1]);
    results[4u * tid] = int(cell.value);
    results[4u * tid + 1u] = first + increment(cell.value);
    results[4u * tid + 2u] = int(cell.wide);
    results[4u * tid + 3u] = int(cell.bytes[1]) + int(cell.bytes[0]);
}
"""

CROSSGL = """shader main {
    struct Cell { uint8 value; int16 wide; }
    int increment(inout uint8 value) { value += 3; return int(value); }
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        buffer int* results;
        void main() {
            Cell cell;
            cell.value = 254;
            uint8 local = 2;
            increment(local);
            increment(cell.value);
            results[0] = int(cell.value) + int(local) + increment(cell.value);
        }
    }
}
"""

# Zero, one, values whose low byte wraps at +3, -1, a value beyond 8 bits and the 16-bit extremes.
INPUTS = (0, 1, 250, 253, 255, -1, 1000, 32767, -32768, 77)
RESULTS_PER_INPUT = 4


def _translate(tmp_path, text, name, backend):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return crosstl.translate(str(path), backend=backend)


def _assert_copies_narrow_fields(metal):
    # Every reference parameter has the widened type, and no struct field binds it directly.
    assert re.search(r"thread uint\s*&\s*value", metal), metal
    assert not re.search(r"\b(increment|lower)\(\s*cell\.", metal), metal
    assert re.search(r"\buint (__crossgl_ref\d+) = cell\.value;", metal), metal


def test_metal_output_copies_narrow_fields_through_temporaries(tmp_path):
    _assert_copies_narrow_fields(_translate(tmp_path, SOURCE, "bytes.metal", "metal"))


def test_saved_crossgl_copies_narrow_fields_through_temporaries(tmp_path):
    crossgl = _translate(tmp_path, SOURCE, "bytes.metal", "cgl")
    metal = _translate(tmp_path, crossgl, "bytes.cgl", "metal")
    _assert_copies_narrow_fields(metal)
    assert metal == _translate(tmp_path, SOURCE, "bytes.metal", "metal")


def test_crossgl_narrow_field_argument_is_copied_and_a_local_binds(tmp_path):
    metal = _translate(tmp_path, CROSSGL, "bytes.cgl", "metal")
    # A widened local binds the reference in place; the field goes through a temporary, also inside an expression.
    assert "increment(local);" in metal, metal
    assert not re.search(r"\bincrement\(\s*cell\.", metal), metal
    temporaries = re.findall(r"\buint (__crossgl_ref\d+) = cell\.value;", metal)
    assert len(temporaries) == 2, metal
    for name in temporaries:
        assert f"cell.value = {name};" in metal, metal
    _compile(tmp_path, metal, "crossgl_out")


def _compile(tmp_path, source, stem):
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        pytest.skip("xcrun is not installed")
    artifact = tmp_path / f"{stem}.metal"
    artifact.write_text(source, encoding="utf-8")
    result = subprocess.run(
        [
            xcrun,
            "--sdk",
            "macosx",
            "metal",
            "-Werror",
            "-c",
            str(artifact),
            "-o",
            str(tmp_path / f"{stem}.air"),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 and "unable to find utility" in result.stderr:
        pytest.skip("the Metal compiler is not installed")
    assert result.returncode == 0, result.stdout + result.stderr


def _run(source, entry):
    metal = pytest.importorskip("Metal")
    device = metal.MTLCreateSystemDefaultDevice()
    if device is None:
        pytest.skip("no Metal device")
    library, error = device.newLibraryWithSource_options_error_(source, None, None)
    assert library is not None, error
    pipeline, error = device.newComputePipelineStateWithFunction_error_(
        library.newFunctionWithName_(entry), None
    )
    assert pipeline is not None, error
    words = struct.pack(f"<{len(INPUTS)}i", *INPUTS)
    inputs = device.newBufferWithBytes_length_options_(words, len(words), 0)
    size = 4 * RESULTS_PER_INPUT * len(INPUTS)
    results = device.newBufferWithLength_options_(size, 0)
    commands = device.newCommandQueue().commandBuffer()
    encoder = commands.computeCommandEncoder()
    encoder.setComputePipelineState_(pipeline)
    encoder.setBuffer_offset_atIndex_(inputs, 0, 0)
    encoder.setBuffer_offset_atIndex_(results, 0, 1)
    encoder.dispatchThreads_threadsPerThreadgroup_(
        metal.MTLSizeMake(len(INPUTS), 1, 1), metal.MTLSizeMake(1, 1, 1)
    )
    encoder.endEncoding()
    commands.commit()
    commands.waitUntilCompleted()
    return struct.unpack(f"<{size // 4}i", results.contents().as_buffer(size))


def _entry(metal):
    found = re.search(r"\bkernel\s+void\s+(\w+)\s*\(", metal)
    assert found, metal
    return found.group(1)


def _expected():
    """The source's results, computed with C's 8- and 16-bit conversions."""

    def u8(value):
        return value & 0xFF

    def s16(value):
        value &= 0xFFFF
        return value - 0x10000 if value & 0x8000 else value

    results = []
    for value in INPUTS:
        cell_value = u8(u8(value) + 3)
        byte0 = u8(u8(value >> 1) + 3)
        first = byte0
        wide = s16(s16(value) - 300)
        byte1 = u8(wide)
        second = u8(cell_value + 3)
        results += [cell_value, first + second, wide, byte1 + byte0]
    return tuple(results)


@pytest.mark.parametrize("route", ["direct", "crossgl"])
def test_narrow_field_references_compute_the_source_results_on_the_gpu(tmp_path, route):
    if route == "direct":
        metal = _translate(tmp_path, SOURCE, "bytes.metal", "metal")
    else:
        crossgl = _translate(tmp_path, SOURCE, "bytes.metal", "cgl")
        metal = _translate(tmp_path, crossgl, "bytes.cgl", "metal")
    _compile(tmp_path, SOURCE, "bytes_native")
    _compile(tmp_path, metal, "bytes_out")
    native = _run(SOURCE, "byte_conversions")
    assert native == _expected()
    assert _run(metal, _entry(metal)) == native


# Vector fields: the widened temporary (uint4) and the stored field (uchar4) convert only explicitly in Metal, for a
# reference argument, for the lane write with a side-effecting index the importer lowers to one, and for a call in a
# for update, whose copies are sequenced inside the update expression.
VECTOR_SOURCE = """#include <metal_stdlib>
using namespace metal;
uint next_index(thread uint& count) { uint old = count; count += 1u; return old; }
int bump(thread uchar4& lanes) { lanes += uchar4(1, 2, 3, 250); return int(lanes.w); }
void widen(thread ushort2& halves) { halves.x += 65535; }
struct Lanes { uchar4 bytes[2]; ushort2 halves; };
kernel void vector_fields(const device int* inputs [[buffer(0)]],
                          device int* results [[buffer(1)]],
                          uint tid [[thread_position_in_grid]]) {
    int input_value = inputs[tid];
    Lanes lanes;
    lanes.bytes[0] = uchar4(uchar(input_value), uchar(input_value >> 8), 7, 9);
    lanes.bytes[1] = uchar4(0);
    lanes.halves = ushort2(ushort(input_value), 3);
    uint count = 0u;
    lanes.bytes[0][next_index(count)] = uchar(input_value + 1);
    int first = bump(lanes.bytes[1]);
    widen(lanes.halves);
    results[4u * tid] = int(lanes.bytes[0].x) + int(lanes.bytes[0].y) * 256 + int(count) * 65536;
    results[4u * tid + 1u] = first + bump(lanes.bytes[0]);
    results[4u * tid + 2u] = int(lanes.halves.x) + int(lanes.halves.y);
    int total = 0;
    for (int i = 0; i < 3; i += 1, bump(lanes.bytes[1])) { total += int(lanes.bytes[1].w); }
    results[4u * tid + 3u] = int(lanes.bytes[1].w) + int(lanes.bytes[0].w) + total;
}
"""


def _assert_converts_vector_fields(metal):
    for widened, stored, field in (
        ("uint4", "uchar4", r"lanes\.bytes\[0\]"),
        ("uint4", "uchar4", r"lanes\.bytes\[1\]"),
        ("uint2", "ushort2", r"lanes\.halves"),
    ):
        copies = re.findall(
            rf"\b{widened} (__crossgl_ref\d+) = {widened}\({field}\);", metal
        )
        assert copies, metal
        for name in copies:
            assert re.search(rf"{field} = {stored}\({name}\);", metal), metal
    assert re.search(
        r"(__crossgl_ref\d+) = uint4\(lanes\.bytes\[1\]\), bump\(\1\),\s*lanes\.bytes\[1\] = uchar4\(\1\)",
        metal,
    ), metal
    assert not re.search(r"__crossgl_ref\d+ = lanes\.", metal), metal


@pytest.mark.parametrize("route", ["direct", "crossgl"])
def test_narrow_vector_field_copies_convert_explicitly(tmp_path, route):
    if route == "direct":
        metal = _translate(tmp_path, VECTOR_SOURCE, "lanes.metal", "metal")
    else:
        crossgl = _translate(tmp_path, VECTOR_SOURCE, "lanes.metal", "cgl")
        metal = _translate(tmp_path, crossgl, "lanes.cgl", "metal")
    _assert_converts_vector_fields(metal)
    _compile(tmp_path, VECTOR_SOURCE, "lanes_native")
    _compile(tmp_path, metal, "lanes_out")
    assert _run(metal, _entry(metal)) == _run(VECTOR_SOURCE, "vector_fields")

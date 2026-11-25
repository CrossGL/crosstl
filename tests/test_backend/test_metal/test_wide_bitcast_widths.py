"""64-bit ``as_type`` casts keep their width through Metal input.

``as_type<uint64_t>(long)`` reinterprets all 64 bits. Imported as the 32-bit ``asuint`` alias it would drop the high
word: Metal output would no longer compile and HLSL would silently store the low half. The CrossGL that Metal input
writes keeps the 64-bit result type, every target validates the translation, and the generated Metal computes on the
GPU what the source computes for values whose high word is not zero.
"""

import re
import shutil
import struct
import subprocess

import pytest

import crosstl

SOURCE = """#include <metal_stdlib>
using namespace metal;
kernel void bits(const device long* inputs [[buffer(0)]],
                 device ulong* outputs [[buffer(1)]],
                 uint index [[thread_position_in_grid]]) {
    long value = inputs[index];
    outputs[4u * index] = as_type<uint64_t>(inputs[index]);
    outputs[4u * index + 1u] = as_type<ulong>(as_type<int2>(value).yx);
    outputs[4u * index + 2u] = ulong(as_type<long>(as_type<uint2>(value) ^ uint2(1u, 0u)));
    outputs[4u * index + 3u] = as_type<ulong>(-value) >> 1ul;
}
"""

MASK = (1 << 64) - 1
# -1, the most negative value, a value with only the high word set, an ordinary positive value and zero.
INPUTS = (-1, -(1 << 63), 0x1234567800000000, 0x0123456789ABCDEF, 0)
RESULTS_PER_INPUT = 4


def _expected(value):
    bits = value & MASK
    swapped = ((bits & 0xFFFFFFFF) << 32) | (bits >> 32)
    return (bits, swapped, bits ^ 1, ((-value) & MASK) >> 1)


def _translate(tmp_path, text, name, backend):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return crosstl.translate(str(path), backend=backend)


def test_crossgl_keeps_64_bit_bitcast_results(tmp_path):
    crossgl = _translate(tmp_path, SOURCE, "bits.metal", "cgl")
    stores = [line.strip() for line in crossgl.splitlines() if "buffer_store(" in line]
    assert len(stores) == RESULTS_PER_INPUT, crossgl
    assert "as_type<uint64>(buffer_load(inputs, index))" in stores[0]
    assert "as_type<uint64>(as_type<ivec2>(value).yx)" in stores[1]
    assert "as_type<int64>(as_type<uvec2>(value)" in stores[2]
    assert "as_type<uint64>" in stores[3]
    assert not re.search(r"\bas(uint|int)\(", crossgl), crossgl


_VALIDATORS = {
    "opengl": ("bits.comp", [["glslangValidator", "-S", "comp", "bits.comp"]]),
    "vulkan": (
        "bits.spvasm",
        [
            ["spirv-as", "--target-env", "vulkan1.1", "bits.spvasm", "-o", "bits.spv"],
            ["spirv-val", "--target-env", "vulkan1.1", "bits.spv"],
        ],
    ),
    "directx": ("bits.hlsl", [["dxc", "-T", "cs_6_6", "-E", "{entry}", "bits.hlsl"]]),
}


@pytest.mark.parametrize("backend", sorted(_VALIDATORS))
def test_64_bit_bitcasts_reach_every_target(tmp_path, backend):
    file_name, commands = _VALIDATORS[backend]
    code = _translate(tmp_path, SOURCE, "bits.metal", backend)
    if backend != "vulkan":
        # The entry body never narrows a 64-bit operand to a 32-bit integer.
        body = code[code.rindex("void ") :]
        assert not re.search(r"\b(as)?u?int\((inputs|value|-value)\b", body), body
    entry = re.search(r"\]\s*void\s+(\w+)\s*\(", code)
    commands = [
        [part.format(entry=entry.group(1) if entry else "") for part in command]
        for command in commands
    ]
    missing = [command[0] for command in commands if shutil.which(command[0]) is None]
    if missing:
        pytest.skip(f"{', '.join(missing)} not installed")
    (tmp_path / file_name).write_text(code, encoding="utf-8")
    for command in commands:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=300,
            cwd=tmp_path,
        )
        assert result.returncode == 0, (result.stdout + result.stderr)[-2000:]


def _compile(tmp_path, source):
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        pytest.skip("xcrun is not installed")
    artifact = tmp_path / "bits_out.metal"
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
            str(tmp_path / "bits_out.air"),
        ],
        check=False,
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
    words = struct.pack(f"<{len(INPUTS)}q", *INPUTS)
    inputs = device.newBufferWithBytes_length_options_(words, len(words), 0)
    size = 8 * RESULTS_PER_INPUT * len(INPUTS)
    outputs = device.newBufferWithLength_options_(size, 0)
    commands = device.newCommandQueue().commandBuffer()
    encoder = commands.computeCommandEncoder()
    encoder.setComputePipelineState_(pipeline)
    encoder.setBuffer_offset_atIndex_(inputs, 0, 0)
    encoder.setBuffer_offset_atIndex_(outputs, 0, 1)
    encoder.dispatchThreads_threadsPerThreadgroup_(
        metal.MTLSizeMake(len(INPUTS), 1, 1), metal.MTLSizeMake(1, 1, 1)
    )
    encoder.endEncoding()
    commands.commit()
    commands.waitUntilCompleted()
    return struct.unpack(f"<{size // 8}Q", outputs.contents().as_buffer(size))


def _entry(metal):
    found = re.search(r"\bkernel\s+void\s+(\w+)\s*\(", metal)
    assert found, metal
    return found.group(1)


@pytest.mark.parametrize("through_crossgl", [False, True])
def test_64_bit_bitcast_results_match_the_source_on_metal(tmp_path, through_crossgl):
    if through_crossgl:
        crossgl = _translate(tmp_path, SOURCE, "bits.metal", "cgl")
        metal = _translate(tmp_path, crossgl, "bits.cgl", "metal")
    else:
        metal = _translate(tmp_path, SOURCE, "bits.metal", "metal")
    _compile(tmp_path, metal)
    native = _run(SOURCE, "bits")
    expected = tuple(word for value in INPUTS for word in _expected(value))
    assert native == expected
    assert _run(metal, _entry(metal)) == native

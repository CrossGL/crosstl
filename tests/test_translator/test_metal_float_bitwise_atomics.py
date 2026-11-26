"""HLSL ``InterlockedCompareExchangeFloatBitwise`` and ``InterlockedCompareStoreFloatBitwise`` in Metal.

Shader model 6.6 compares and swaps a ``float`` by its bits, and CrossGL's ``atomicCompSwap`` on a ``float`` keeps
that meaning (``semantics/atomic_forms.py``). Metal output refused a ``float`` destination. It now writes a
compare-exchange helper that exchanges the destination's word through ``atomic_uint`` and compares
``as_type<uint>`` of the operands, so ``-0.0`` does not match a stored ``0.0`` and a NaN matches its own bits. Device
and group-shared destinations go through the pointer helper, and a ``float`` member of a buffer block through the
byte-offset helper.
"""

from __future__ import annotations

import re
import shutil
import struct
import subprocess

import pytest

import crosstl

DEVICE = """RWStructuredBuffer<float> operands : register(u0);
RWStructuredBuffer<float> exchanged : register(u1);
RWStructuredBuffer<float> stored : register(u2);
RWStructuredBuffer<float> previous : register(u3);
[numthreads(1, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint i = id.x;
    float compare = operands[2u * i];
    float value = operands[2u * i + 1u];
    float before;
    InterlockedCompareExchangeFloatBitwise(exchanged[i], compare, value, before);
    previous[i] = before;
    InterlockedCompareStoreFloatBitwise(stored[i], compare, value);
}
"""

GROUP_SHARED = """RWStructuredBuffer<float> outp : register(u0);
groupshared float shared_zero;
groupshared float shared_one;
[numthreads(4, 1, 1)]
void main(uint3 id : SV_DispatchThreadID, uint gi : SV_GroupIndex) {
    if (gi == 0) {
        shared_zero = -0.0;
        shared_one = 1.0;
    }
    GroupMemoryBarrierWithGroupSync();
    float seen_zero;
    float seen_one;
    InterlockedCompareExchangeFloatBitwise(shared_zero, 0.0, 9.0, seen_zero);
    InterlockedCompareExchangeFloatBitwise(shared_one, 1.0, 2.0 + float(gi), seen_one);
    GroupMemoryBarrierWithGroupSync();
    outp[gi] = seen_zero;
    outp[4 + gi] = seen_one;
    if (gi == 0) {
        outp[8] = shared_zero;
        outp[9] = shared_one;
    }
}
"""

# A float member of a buffer block is addressed by its byte offset, through the other helper.
BLOCK_MEMBER = """shader main {
    struct Data {
        int pad;
        float v;
        vec2 w;
    };
    Data data @glsl_buffer_block(std430) @binding(0);
    RWStructuredBuffer<float> outp @binding(1);
    compute {
        layout(local_size_x = 1) in;
        void main() {
            uint i = gl_GlobalInvocationID.x;
            buffer_store(outp, i, atomicCompSwap(data.v, (-0.0), 2.0));
        }
    }
}
"""

POSITIVE_ZERO = 0x00000000
NEGATIVE_ZERO = 0x80000000
QUIET_NAN = 0x7FC00000
NEGATIVE_QUIET_NAN = 0xFFC00000


def _bits(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


# (destination, compare, value): the destination changes only when compare has its bits.
CASES = (
    (POSITIVE_ZERO, NEGATIVE_ZERO, _bits(5.0)),
    (QUIET_NAN, QUIET_NAN, _bits(6.0)),
    (_bits(1.0), _bits(1.0), _bits(7.0)),
    (NEGATIVE_ZERO, POSITIVE_ZERO, _bits(8.0)),
    (QUIET_NAN, NEGATIVE_QUIET_NAN, _bits(3.0)),
)


def _translate(tmp_path, source, suffix="hlsl"):
    path = tmp_path / f"probe.{suffix}"
    path.write_text(source)
    return crosstl.translate(str(path), backend="metal")


def _collapsed(code):
    flat = " ".join(code.split())
    return re.sub(r"\s+\)", ")", re.sub(r"\(\s+", "(", flat))


def _compile(tmp_path, metal):
    if shutil.which("xcrun") is None:
        pytest.skip("xcrun is not installed")
    path = tmp_path / "translated.metal"
    path.write_text(metal)
    result = subprocess.run(
        ["xcrun", "-sdk", "macosx", "metal", "-Werror", "-c", str(path)]
        + ["-o", str(tmp_path / "translated.air")],
        capture_output=True,
        text=True,
        check=False,
    )
    if "unable to find utility" in result.stderr:
        pytest.skip("the Metal compiler is not installed")
    assert result.returncode == 0, result.stderr + metal


def _words(data):
    return struct.unpack(f"<{len(data) // 4}I", data)


def _pack(words):
    return struct.pack(f"<{len(words)}I", *words)


def _run_metal(source, buffers, threads, group):
    Metal = pytest.importorskip("Metal")
    device = Metal.MTLCreateSystemDefaultDevice()
    if device is None:
        pytest.skip("no Metal device")
    options = Metal.MTLCompileOptions.new()
    options.setMathMode_(Metal.MTLMathModeSafe)
    library, error = device.newLibraryWithSource_options_error_(source, options, None)
    assert library is not None, error
    entry = re.search(r"kernel void (\w+)\(", source).group(1)
    pipeline, error = device.newComputePipelineStateWithFunction_error_(
        library.newFunctionWithName_(entry), None
    )
    assert pipeline is not None, error
    indices = {
        name: int(index)
        for name, index in re.findall(r"(\w+)\s*\[\[buffer\((\d+)\)\]\]", source)
    }
    assert set(indices) == set(buffers), source
    shared = Metal.MTLResourceStorageModeShared
    native = {
        name: device.newBufferWithBytes_length_options_(data, len(data), shared)
        for name, data in buffers.items()
    }
    command = device.newCommandQueue().commandBuffer()
    encoder = command.computeCommandEncoder()
    encoder.setComputePipelineState_(pipeline)
    for name, buffer in native.items():
        encoder.setBuffer_offset_atIndex_(buffer, 0, indices[name])
    encoder.dispatchThreads_threadsPerThreadgroup_(
        Metal.MTLSizeMake(threads, 1, 1), Metal.MTLSizeMake(group, 1, 1)
    )
    encoder.endEncoding()
    command.commit()
    command.waitUntilCompleted()
    assert command.error() is None, command.error()
    return {
        name: _words(bytes(buffer.contents().as_buffer(len(buffers[name]))))
        for name, buffer in native.items()
    }


def test_device_destinations_compare_bits_through_uint_words(tmp_path):
    metal = _translate(tmp_path, DEVICE)
    flat = _collapsed(metal)
    assert re.search(
        r"device atomic_uint\s*\*\s*atomicTarget = "
        r"reinterpret_cast<device atomic_uint\s*\*>\(target\);",
        flat,
    ), flat
    assert "uint compareWord = as_type<uint>(compareValue);" in flat
    assert "return as_type<float>(original);" in flat
    assert re.search(
        r"before = __crossgl_device_atomic_compare_exchange_float\(&exchanged\[i\], compare, value\);",
        flat,
    )
    # The compare-store discards the previous value.
    assert re.search(
        r"[;{] __crossgl_device_atomic_compare_exchange_float\(&stored\[i\], compare, value\);",
        flat,
    )
    assert "FloatBitwise" not in metal
    _compile(tmp_path, metal)


def test_device_compare_exchange_matches_bits_on_the_gpu(tmp_path):
    metal = _translate(tmp_path, DEVICE)
    destinations = [case[0] for case in CASES]
    operands = [word for case in CASES for word in case[1:]]
    results = _run_metal(
        metal,
        {
            "operands": _pack(operands),
            "exchanged": _pack(destinations),
            "stored": _pack(destinations),
            "previous": _pack([0] * len(CASES)),
        },
        threads=len(CASES),
        group=1,
    )
    expected = tuple(
        value if destination == compare else destination
        for destination, compare, value in CASES
    )
    assert results["exchanged"] == expected
    assert results["stored"] == expected
    assert results["previous"] == tuple(destinations)


def test_group_shared_destinations_compare_bits_on_the_gpu(tmp_path):
    metal = _translate(tmp_path, GROUP_SHARED)
    assert "__crossgl_threadgroup_atomic_compare_exchange_float(" in metal
    assert re.search(r"threadgroup atomic_uint\s*\*\s*atomicTarget", _collapsed(metal))
    _compile(tmp_path, metal)
    words = _run_metal(metal, {"outp": _pack([0] * 10)}, threads=4, group=4)["outp"]
    # 0.0 never matches the stored -0.0, so no thread exchanges it.
    assert words[0:4] == (NEGATIVE_ZERO,) * 4
    assert words[8] == NEGATIVE_ZERO
    # Exactly one thread finds 1.0 and stores 2.0 + its index; the others see that value.
    seen = words[4:8]
    winners = [index for index, word in enumerate(seen) if word == _bits(1.0)]
    assert len(winners) == 1, seen
    final = _bits(2.0 + winners[0])
    assert words[9] == final
    assert all(word == final for index, word in enumerate(seen) if index != winners[0])


@pytest.mark.parametrize(
    ("stored", "expected"),
    ((POSITIVE_ZERO, POSITIVE_ZERO), (NEGATIVE_ZERO, _bits(2.0))),
)
def test_buffer_block_members_compare_bits_on_the_gpu(tmp_path, stored, expected):
    metal = _translate(tmp_path, BLOCK_MEMBER, "cgl")
    flat = _collapsed(metal)
    assert re.search(
        r"device atomic_uint\s*\*\s*target = "
        r"reinterpret_cast<device atomic_uint\s*\*>\(buffer \+ offset\);",
        flat,
    ), flat
    assert "__crossgl_buffer_atomic_compare_exchange_float(data, 4, -0.0, 2.0)" in flat
    _compile(tmp_path, metal)
    pad = 0x12345678
    block = _pack([pad, stored, _bits(3.0), _bits(4.0)])
    results = _run_metal(metal, {"data": block, "outp": _pack([0])}, threads=1, group=1)
    assert results["data"] == (pad, expected, _bits(3.0), _bits(4.0))
    assert results["outp"] == (stored,)

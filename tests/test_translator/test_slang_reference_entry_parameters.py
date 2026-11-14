"""Slang entry parameters for Metal reference arguments.

A Metal ``constant T& x [[buffer(n)]]`` kernel argument reached Slang as ``float &x``, which slangc rejects as an entry
parameter. It is now a ``ConstantBuffer<T>``, which slangc lowers back to Metal's ``constant T* [[buffer(n)]]``
argument. A reference in another address space has no such form and refuses the entry point. Helper functions keep
their reference parameters.
"""

from __future__ import annotations

import shutil
import struct
import subprocess

import pytest

import crosstl
from crosstl.translator.codegen.slang_refusals import SlangLoweringRefused

METAL_SOURCE = """
#include <metal_stdlib>
using namespace metal;
struct Range { float start; float step; };
kernel void arange(constant float& offset [[buffer(0)]], constant Range& range [[buffer(1)]],
                   device float* out [[buffer(2)]], uint index [[thread_position_in_grid]]) {
  out[index] = offset + range.start + index * range.step;
}
"""


def _translate(tmp_path, source):
    path = tmp_path / "arange.metal"
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend="slang", source_backend="metal")


def test_constant_references_become_constant_buffers(tmp_path):
    slang = _translate(tmp_path, METAL_SOURCE)

    assert "ConstantBuffer<float> offset" in slang
    assert "ConstantBuffer<Range> range" in slang
    assert "out_[index] = offset + range.start + index * range.step;" in slang
    assert "&offset" not in slang and "&range" not in slang


def test_device_references_are_refused(tmp_path):
    source = METAL_SOURCE.replace(
        "constant float& offset [[buffer(0)]]", "device float& offset [[buffer(0)]]"
    )

    with pytest.raises(SlangLoweringRefused) as excinfo:
        _translate(tmp_path, source)

    assert "Slang entry parameter 'offset' has the reference type float&" in str(
        excinfo.value
    )


def _metal():
    try:
        import Metal
    except ImportError:
        return None, None
    return Metal, Metal.MTLCreateSystemDefaultDevice()


def _run(metal, device, source, count=64):
    library, error = device.newLibraryWithSource_options_error_(source, None, None)
    assert library is not None, error
    pipeline, error = device.newComputePipelineStateWithFunction_error_(
        library.newFunctionWithName_("arange"), None
    )
    assert pipeline is not None, error
    buffers = (
        device.newBufferWithBytes_length_options_(struct.pack("f", 0.5), 4, 0),
        device.newBufferWithBytes_length_options_(struct.pack("2f", 1.5, 0.25), 8, 0),
        device.newBufferWithLength_options_(4 * count, 0),
    )
    command = device.newCommandQueue().commandBuffer()
    encoder = command.computeCommandEncoder()
    encoder.setComputePipelineState_(pipeline)
    for index, buffer in enumerate(buffers):
        encoder.setBuffer_offset_atIndex_(buffer, 0, index)
    encoder.dispatchThreads_threadsPerThreadgroup_(
        metal.MTLSizeMake(count, 1, 1), metal.MTLSizeMake(32, 1, 1)
    )
    encoder.endEncoding()
    command.commit()
    command.waitUntilCompleted()
    return struct.unpack(f"{count}f", bytes(buffers[2].contents().as_buffer(4 * count)))


@pytest.mark.skipif(shutil.which("slangc") is None, reason="slangc is not available")
def test_constant_buffers_keep_the_metal_bindings_on_the_gpu(tmp_path):
    metal, device = _metal()
    if device is None:
        pytest.skip("no Metal device")
    slang_path = tmp_path / "arange.slang"
    slang_path.write_text(_translate(tmp_path, METAL_SOURCE), encoding="utf-8")
    metal_path = tmp_path / "arange_from_slang.metal"
    result = subprocess.run(
        [
            shutil.which("slangc"),
            str(slang_path),
            "-target",
            "metal",
            "-o",
            str(metal_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    expected = _run(metal, device, METAL_SOURCE)
    actual = _run(metal, device, metal_path.read_text(encoding="utf-8"))

    assert actual == expected

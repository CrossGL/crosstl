"""Nested unary operators keep their grouping through Metal input and Metal output.

``-(-x)`` negates twice; written without the parentheses it is ``--x``, a decrement that also changes ``x``. The
generated Metal, read directly and through the saved CrossGL, keeps every nested ``+`` and ``-`` grouped, keeps a
genuine prefix or postfix update distinct, and computes on the GPU what the source computes, including for zero,
signed zero, subnormal and ordinary positive and negative operands.
"""

import re
import shutil
import struct
import subprocess

import pytest

import crosstl

SOURCE = """#include <metal_stdlib>
using namespace metal;
kernel void k(device const uint* values [[buffer(0)]],
              device uint* results [[buffer(1)]],
              uint i [[thread_position_in_grid]]) {
    float x = as_type<float>(values[i]);
    int n = int(values[i] & 0xffu);
    int j = n;
    float2 v = float2(x, -x);
    results[8u * i] = as_type<uint>(-(-x));
    results[8u * i + 1u] = as_type<uint>((-(-v)).y);
    results[8u * i + 2u] = uint(-(-n));
    results[8u * i + 3u] = uint(+(+n)) + uint(-(+n)) + uint(+(-n));
    results[8u * i + 4u] = uint(-(--j));
    results[8u * i + 5u] = uint(j);
    results[8u * i + 6u] = uint(-(j--));
    results[8u * i + 7u] = as_type<uint>(-(-(float(j))));
}
"""

# Zero, negative zero, the smallest subnormal, a larger subnormal, 1.0, -1.0, the largest finite float and a small
# integer word.
INPUTS = (
    0x00000000,
    0x80000000,
    0x00000001,
    0x007FFFFF,
    0x3F800000,
    0xBF800000,
    0x7F7FFFFF,
    0x00000005,
)
RESULTS_PER_INPUT = 8


def _translate(tmp_path, text, name, backend):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return crosstl.translate(str(path), backend=backend)


def _statements(metal):
    return [line.strip() for line in metal.splitlines() if "results[" in line]


def _assert_grouped(metal):
    statements = _statements(metal)
    assert len(statements) == RESULTS_PER_INPUT, metal
    joined = "\n".join(statements)
    # The only decrements are the source's own updates of j.
    assert re.findall(r"--|\+\+", joined) == ["--", "--"], joined
    assert "--j" in statements[4] and "-(--j)" in statements[4].replace(" ", "")
    assert re.search(r"-\(?j--", statements[6].replace(" ", "")), statements[6]
    for index in (0, 1, 2, 7):
        assert "-(-" in statements[index].replace(" ", ""), statements[index]
    assert "+(+n)" in statements[3].replace(" ", "")


def test_metal_output_keeps_nested_unary_groups(tmp_path):
    _assert_grouped(_translate(tmp_path, SOURCE, "negate.metal", "metal"))


def test_saved_crossgl_keeps_nested_unary_groups(tmp_path):
    crossgl = _translate(tmp_path, SOURCE, "negate.metal", "cgl")
    metal = _translate(tmp_path, crossgl, "negate.cgl", "metal")
    _assert_grouped(metal)
    assert metal == _translate(tmp_path, SOURCE, "negate.metal", "metal")


def _compile(tmp_path, source):
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        pytest.skip("xcrun is not installed")
    artifact = tmp_path / "negate_out.metal"
    artifact.write_text(source, encoding="utf-8")
    result = subprocess.run(
        [
            xcrun,
            "--sdk",
            "macosx",
            "metal",
            "-Werror",
            "-fno-fast-math",
            "-c",
            str(artifact),
            "-o",
            str(tmp_path / "negate_out.air"),
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
    options = metal.MTLCompileOptions.new()
    options.setMathMode_(metal.MTLMathModeSafe)
    library, error = device.newLibraryWithSource_options_error_(source, options, None)
    assert library is not None, error
    pipeline, error = device.newComputePipelineStateWithFunction_error_(
        library.newFunctionWithName_(entry), None
    )
    assert pipeline is not None, error
    words = struct.pack(f"<{len(INPUTS)}I", *INPUTS)
    values = device.newBufferWithBytes_length_options_(words, len(words), 0)
    size = 4 * RESULTS_PER_INPUT * len(INPUTS)
    results = device.newBufferWithLength_options_(size, 0)
    commands = device.newCommandQueue().commandBuffer()
    encoder = commands.computeCommandEncoder()
    encoder.setComputePipelineState_(pipeline)
    encoder.setBuffer_offset_atIndex_(values, 0, 0)
    encoder.setBuffer_offset_atIndex_(results, 0, 1)
    encoder.dispatchThreads_threadsPerThreadgroup_(
        metal.MTLSizeMake(len(INPUTS), 1, 1), metal.MTLSizeMake(1, 1, 1)
    )
    encoder.endEncoding()
    commands.commit()
    commands.waitUntilCompleted()
    return struct.unpack(f"<{size // 4}I", results.contents().as_buffer(size))


def _entry(metal):
    found = re.search(r"\bkernel\s+void\s+(\w+)\s*\(", metal)
    assert found, metal
    return found.group(1)


@pytest.mark.parametrize("through_crossgl", [False, True])
def test_nested_unary_results_match_the_source_on_metal(tmp_path, through_crossgl):
    if through_crossgl:
        crossgl = _translate(tmp_path, SOURCE, "negate.metal", "cgl")
        metal = _translate(tmp_path, crossgl, "negate.cgl", "metal")
    else:
        metal = _translate(tmp_path, SOURCE, "negate.metal", "metal")
    _compile(tmp_path, metal)
    native = _run(SOURCE, "k")
    translated = _run(metal, _entry(metal))
    # Double negation returns the input word itself, signed zero and subnormals included.
    assert native[0::RESULTS_PER_INPUT] == INPUTS
    assert translated == native

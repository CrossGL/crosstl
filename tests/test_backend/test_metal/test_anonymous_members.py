"""Metal input imports anonymous union and struct members of a class.

The members of an anonymous union are members of the enclosing class and share its storage ([class.union.anon]);
Clang gives an anonymous struct the same meaning as an extension. CrossGL has no anonymous members, so the importer
declares each under a reserved member name and reaches the promoted members through it, in accesses and in the
methods that name them unqualified. Metal input used to refuse such a class (MetalAnonymousMemberRefusal), which
kept, for example, every ggml quantized block type out of CrossGL.
"""

import shutil
import struct
import subprocess

import pytest

import crosstl
from crosstl.backend.Metal.member_aggregates import MetalAnonymousMemberRefusal

QUANTIZED_BLOCK = """
#include <metal_stdlib>
using namespace metal;

typedef struct {
    union {
        struct {
            half d;
            half m;
        };
        half2 dm;
    };
    uint8_t qs[4];
} block_q4_1;

kernel void dequantize(device const block_q4_1 * x [[buffer(0)]],
                       device float * out [[buffer(1)]],
                       uint i [[thread_position_in_grid]]) {
    device const block_q4_1 * b = x + i;
    half2 dm = b->dm;
    out[i] = float(b->d) + float(x[i].m) * 2.0 + float(dm.y) * 10.0 + float(b->qs[1]);
}
"""

UNION_METHODS = """
#include <metal_stdlib>
using namespace metal;

struct Word {
    union { float f; uint u; };
    float c;
    float get() const { return f + c; }
    void twice() { u = u + 0x00800000u; }
};

kernel void methods(device const Word * x [[buffer(0)]],
                    device float * out [[buffer(1)]],
                    uint i [[thread_position_in_grid]]) {
    Word w = x[i];
    w.twice();
    out[i] = w.get();
}
"""

ANONYMOUS_STRUCT = """
#include <metal_stdlib>
using namespace metal;

struct Point {
    struct { float x; float y; };
    float z;
};

kernel void point(device const Point * p [[buffer(0)]],
                  device float * out [[buffer(1)]],
                  uint i [[thread_position_in_grid]]) {
    Point q = p[i];
    q.y = q.x * 2.0;
    out[i] = q.x + q.y * 10.0 + q.z * 100.0;
}
"""

TYPEDEF_NAMED_UNION = """
#include <metal_stdlib>
using namespace metal;

typedef struct {
    union { float a; uint b; } u;
    float c;
} Pair;

kernel void pair(device Pair * x [[buffer(0)]],
                 device float * out [[buffer(1)]],
                 uint i [[thread_position_in_grid]]) {
    x[i].u.b = 0x3f800000u;
    out[i] = x[i].u.a + x[i].c;
}
"""

ANONYMOUS_CLASS = """
#include <metal_stdlib>
using namespace metal;

struct Hidden {
    class { float x; };
    float z;
};

kernel void hidden(device float * out [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    out[i] = 1.0;
}
"""


def _write(tmp_path, source):
    path = tmp_path / "input.metal"
    path.write_text(source)
    return str(path)


def _compile(tmp_path, metal):
    if shutil.which("xcrun") is None:
        return
    source = tmp_path / "out.metal"
    source.write_text(metal)
    result = subprocess.run(
        [
            "xcrun",
            "-sdk",
            "macosx",
            "metal",
            "-c",
            str(source),
            "-o",
            str(tmp_path / "out.air"),
        ],
        capture_output=True,
        text=True,
    )
    if "unable to find utility" in result.stderr:
        return
    assert result.returncode == 0, result.stderr


def _run(source, kernel, data, count):
    """Dispatch ``count`` threads of ``kernel`` with ``data`` at buffer 0 and ``count`` floats at buffer 1."""
    metal = pytest.importorskip("Metal")
    device = metal.MTLCreateSystemDefaultDevice()
    if device is None:
        pytest.skip("no Metal device")
    options = metal.MTLCompileOptions.new()
    options.setMathMode_(metal.MTLMathModeSafe)
    library, error = device.newLibraryWithSource_options_error_(source, options, None)
    assert library is not None, error
    pipeline, error = device.newComputePipelineStateWithFunction_error_(
        library.newFunctionWithName_(kernel), None
    )
    assert pipeline is not None, error
    commands = device.newCommandQueue().commandBuffer()
    encoder = commands.computeCommandEncoder()
    encoder.setComputePipelineState_(pipeline)
    inputs = device.newBufferWithBytes_length_options_(data, len(data), 0)
    outputs = device.newBufferWithLength_options_(4 * count, 0)
    encoder.setBuffer_offset_atIndex_(inputs, 0, 0)
    encoder.setBuffer_offset_atIndex_(outputs, 0, 1)
    encoder.dispatchThreads_threadsPerThreadgroup_(
        metal.MTLSizeMake(count, 1, 1), metal.MTLSizeMake(1, 1, 1)
    )
    encoder.endEncoding()
    commands.commit()
    commands.waitUntilCompleted()
    return list(struct.unpack(f"{count}f", outputs.contents().as_buffer(4 * count)))


def _round_trip_matches_native(tmp_path, source, kernel, data, count, expected):
    metal = crosstl.translate(_write(tmp_path, source), backend="metal")
    _compile(tmp_path, metal)
    assert _run(source, kernel, data, count) == expected
    assert _run(metal, kernel, data, count) == expected


def test_anonymous_union_members_import_through_reserved_member_names(tmp_path):
    crossgl = crosstl.translate(_write(tmp_path, QUANTIZED_BLOCK), backend="cgl")

    assert "__crossgl_unnamed_member_1;" in crossgl
    assert "@union_layout(4, 4, little_endian, metal)" in crossgl
    assert "@union_member_layout(0, 4, 2)" in crossgl
    assert "(*b).__crossgl_unnamed_member_1.__crossgl_unnamed_member_2.d" in crossgl
    assert "(*b).__crossgl_unnamed_member_1.dm" in crossgl


def test_anonymous_union_round_trip_computes_the_native_result(tmp_path):
    blocks = struct.pack("<ee4B", 1.5, 0.25, 1, 2, 3, 4) + struct.pack(
        "<ee4B", -2.0, 4.0, 5, 6, 7, 8
    )
    # d + 2 m + 10 m (dm.y) + qs[1]
    expected = [1.5 + 0.5 + 2.5 + 2.0, -2.0 + 8.0 + 40.0 + 6.0]
    _round_trip_matches_native(
        tmp_path, QUANTIZED_BLOCK, "dequantize", blocks, 2, expected
    )


def test_methods_name_anonymous_union_members_unqualified(tmp_path):
    crossgl = crosstl.translate(_write(tmp_path, UNION_METHODS), backend="cgl")
    assert "return self.__crossgl_unnamed_member_1.f + self.c;" in crossgl

    words = struct.pack("<ff", 1.5, 0.25) + struct.pack("<ff", -3.0, 1.0)
    # Adding one to the exponent doubles the float.
    _round_trip_matches_native(
        tmp_path, UNION_METHODS, "methods", words, 2, [3.25, -5.0]
    )


def test_anonymous_struct_members_keep_their_layout(tmp_path):
    points = struct.pack("<3f", 1.0, 9.0, 2.0) + struct.pack("<3f", 0.5, 9.0, -1.0)
    _round_trip_matches_native(
        tmp_path, ANONYMOUS_STRUCT, "point", points, 2, [221.0, -89.5]
    )


def test_typedef_struct_declares_its_nested_union(tmp_path):
    crossgl = crosstl.translate(_write(tmp_path, TYPEDEF_NAMED_UNION), backend="cgl")
    assert "struct Pair_u3a_u3a__crossgl_unnamed_union_1 {" in crossgl
    assert "Pair_u3a_u3a__crossgl_unnamed_union_1 u;" in crossgl

    pairs = struct.pack("<ff", 0.0, 0.5) + struct.pack("<ff", 0.0, 2.0)
    _round_trip_matches_native(
        tmp_path, TYPEDEF_NAMED_UNION, "pair", pairs, 2, [1.5, 3.0]
    )


def test_anonymous_class_members_are_refused(tmp_path):
    with pytest.raises(MetalAnonymousMemberRefusal, match="anonymous class member"):
        crosstl.translate(_write(tmp_path, ANONYMOUS_CLASS), backend="cgl")

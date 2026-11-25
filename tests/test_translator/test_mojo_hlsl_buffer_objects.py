"""HLSL buffer objects in Mojo output: structured and byte-address buffers become kernel pointer arguments.

Mojo 1.2 has no global variables, so a buffer declared outside a function is passed to the kernel as an
``UnsafePointer`` to its first element (a structured buffer) or to its 32-bit words (a byte-address buffer). Each
program below is HLSL that the HLSL front end imports; when a Mojo compiler is installed, the output must build.
"""

from __future__ import annotations

import subprocess

import pytest

import crosstl
from crosstl.translator.codegen.mojo_refusals import MojoLoweringRefused
from crosstl.translator.refusals import MojoRefusal
from tests.test_translator.test_codegen.test_mojo_codegen import find_mojo_compiler

STRUCTURED_AND_WORDS = """
struct P { float a; uint b; };
StructuredBuffer<P> sp : register(t0);
RWStructuredBuffer<float4> outv : register(u3);
ByteAddressBuffer raw : register(t1);
RWByteAddressBuffer rw : register(u4);
[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint n; uint s;
    sp.GetDimensions(n, s);
    uint w = raw.Load(tid.x * 4);
    uint2 w2 = uint2(raw.Load(8), raw.Load(12));
    rw.Store(tid.x * 4, w + w2.y);
    outv[tid.x] = float4(sp[tid.x].a, (float)sp[tid.x].b, (float)n, 1.0);
}
"""

WORD_VECTORS_AND_HELPERS = """
ByteAddressBuffer raw : register(t1);
RWByteAddressBuffer rw : register(u4);
RWStructuredBuffer<uint> cnt : register(u5);
RWStructuredBuffer<float4> o4 : register(u6);
uint total() { uint n; cnt.GetDimensions(n); return n; }
float4 at(uint i) { return o4[i]; }
[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    float f = raw.Load<float>(0);
    uint4 q = raw.Load4(16);
    uint3 t3 = raw.Load3(32);
    rw.Store3(0, q.xyz + t3);
    rw.Store2(16, int2(-1, 2));
    rw.Store(24, f);
    InterlockedAdd(cnt[0], 1);
    uint o; InterlockedMax(cnt[1], tid.x, o);
    uint bytes; rw.GetDimensions(bytes);
    o4[tid.x] = at(tid.x) + float4(total(), bytes, o, f);
}
"""


def _translate(tmp_path, source):
    path = tmp_path / "kernel.hlsl"
    path.write_text(source)
    return crosstl.translate(str(path), backend="mojo", format_output=False)


def _build(tmp_path, code):
    mojo = find_mojo_compiler()
    source = tmp_path / "kernel.mojo"
    source.write_text(code)
    return subprocess.run(
        [
            mojo,
            "build",
            "--emit",
            "llvm",
            "--disable-warnings",
            str(source),
            "-o",
            str(tmp_path / "kernel.ll"),
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )


def test_structured_and_byte_address_buffers_become_pointer_arguments(tmp_path):
    code = _translate(tmp_path, STRUCTURED_AND_WORDS)

    assert (
        "def compute_main(sp: UnsafePointer[P, MutAnyOrigin], sp_length: UInt32, "
        "outv: UnsafePointer[SIMD[DType.float32, 4], MutAnyOrigin], "
        "raw: UnsafePointer[UInt32, MutAnyOrigin], "
        "rw: UnsafePointer[UInt32, MutAnyOrigin]) -> None:"
    ) in code
    # A byte address names a 32-bit word four bytes wide.
    assert "var w: UInt32 = raw[Int((tid[0] * 4)) >> 2]" in code
    assert "crosstl_buffer_store1(rw, Int((tid[0] * 4)) >> 2, (w + w2[1]))" in code
    assert "outv[Int(tid[0])] = SIMD[DType.float32, 4](sp[Int(tid[0])].a," in code
    # GetDimensions reads the element count passed beside the buffer, and its stride is a constant.
    assert "# CrossGL buffer length: name=sp_length buffer=sp counts=elements" in code
    assert "n = UInt32(sp_length)" in code
    assert "s = 8" in code
    assert "var sp:" not in code
    assert "StructuredBuffer" not in code.split("# CrossGL resource metadata")[0]


def test_word_vectors_and_buffers_read_by_helpers(tmp_path):
    code = _translate(tmp_path, WORD_VECTORS_AND_HELPERS)

    assert (
        "def crosstl_buffer_load4(buffer: UnsafePointer[UInt32, MutAnyOrigin], word: Int) "
        "-> SIMD[DType.uint32, 4]:"
    ) in code
    # A 3-word load fills the fourth lane of its SIMD value with zero.
    assert (
        "return SIMD[DType.uint32, 4](buffer[word], buffer[word + 1], buffer[word + 2], "
        "UInt32(0))"
    ) in code
    assert (
        "def crosstl_buffer_store2(buffer: UnsafePointer[UInt32, MutAnyOrigin], word: Int, "
        "value: SIMD[DType.uint32, 2]):"
    ) in code
    # A helper that reads a buffer receives it, and the size argument beside it, from its caller.
    assert (
        "def total(cnt: UnsafePointer[UInt32, MutAnyOrigin], cnt_length: UInt32) -> UInt32:"
        in code
    )
    assert "rw_length: UInt32" in code
    assert "# CrossGL buffer length: name=rw_length buffer=rw counts=bytes" in code


@pytest.mark.parametrize("source", [STRUCTURED_AND_WORDS, WORD_VECTORS_AND_HELPERS])
def test_buffer_object_kernels_compile_with_mojo(tmp_path, source):
    result = _build(tmp_path, _translate(tmp_path, source))

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    ("source", "message"),
    [
        (
            """
            AppendStructuredBuffer<uint> items : register(u0);
            [numthreads(1, 1, 1)]
            void main() { items.Append(1); }
            """,
            "no append or consume buffer",
        ),
        (
            """
            RWStructuredBuffer<uint> items : register(u0);
            [numthreads(1, 1, 1)]
            void main() { items[0] = items.IncrementCounter(); }
            """,
            "structured-buffer counters are not available",
        ),
        (
            """
            RWByteAddressBuffer raw : register(u0);
            [numthreads(1, 1, 1)]
            void main() { uint o; raw.InterlockedAdd(0, 1, o); }
            """,
            "byte-address atomics are not lowered",
        ),
        (
            """
            RWStructuredBuffer<float> items : register(u0);
            float first(RWStructuredBuffer<float> b) { return b[0]; }
            [numthreads(1, 1, 1)]
            void main() { items[1] = first(items); }
            """,
            "only as a kernel argument declared outside a function",
        ),
        (
            # HLSL packs float3 elements 12 bytes apart; a Mojo SIMD[float32, 3] takes 16.
            """
            StructuredBuffer<float3> items : register(t0);
            RWStructuredBuffer<float> outv : register(u0);
            [numthreads(1, 1, 1)]
            void main() { outv[0] = items[1].y; }
            """,
            "places its elements or members at other offsets",
        ),
        (
            """
            struct M { float4x4 m; };
            StructuredBuffer<M> items : register(t0);
            RWStructuredBuffer<float> outv : register(u0);
            [numthreads(1, 1, 1)]
            void main() { outv[0] = items[0].m[0][0]; }
            """,
            "holding a matrix",
        ),
    ],
)
def test_buffer_objects_mojo_cannot_hold_are_refused(tmp_path, source, message):
    with pytest.raises(MojoRefusal, match=message):
        _translate(tmp_path, source)


def test_hlsl_typed_buffer_is_refused(tmp_path):
    # HLSL Buffer<T> imports as a texel buffer, which reads through the format of its view.
    source = """
    Buffer<float> items : register(t0);
    RWStructuredBuffer<float> outv : register(u0);
    [numthreads(1, 1, 1)]
    void main() { outv[0] = items[0]; }
    """
    with pytest.raises(
        MojoLoweringRefused, match="samplerBuffer' has no Mojo spelling"
    ):
        _translate(tmp_path, source)


METAL_BUFFER_PARAMETERS = """
#include <metal_stdlib>
using namespace metal;
struct Params { uint n; };
kernel void scale(constant Params& params [[buffer(0)]], device const float* src [[buffer(1)]],
                  device float* dst [[buffer(2)]], uint2 id [[thread_position_in_grid]]) {
    if (id.x < params.n) {
        dst[id.x] = src[id.x] * 2.0;
    }
}
"""


def test_entry_buffer_parameters_become_pointer_arguments(tmp_path):
    # The Metal front end writes the device pointers of a kernel as structured-buffer entry parameters.
    path = tmp_path / "kernel.metal"
    path.write_text(METAL_BUFFER_PARAMETERS)
    code = crosstl.translate(str(path), backend="mojo", format_output=False)

    assert (
        "def scale(params: Params, src: UnsafePointer[Float32, MutAnyOrigin], "
        "dst: UnsafePointer[Float32, MutAnyOrigin]) -> None:"
    ) in code
    assert "dst[Int(id[0])] = (src[Int(id[0])] * 2.0)" in code
    assert "StructuredBuffer" not in code


def test_entry_buffer_parameter_kernel_compiles_with_mojo(tmp_path):
    path = tmp_path / "kernel.metal"
    path.write_text(METAL_BUFFER_PARAMETERS)
    code = crosstl.translate(str(path), backend="mojo", format_output=False)

    result = _build(tmp_path, code)

    assert result.returncode == 0, result.stderr


def test_entry_buffer_parameter_size_query_is_refused(tmp_path):
    path = tmp_path / "kernel.cgl"
    path.write_text("""
shader main {
    compute {
        @ stage_entry
        void main(RWStructuredBuffer<uint> b @buffer(0), uvec3 tid @gl_GlobalInvocationID) {
            uint n;
            buffer_dimensions(b, n);
            b[tid.x] = n;
        }
    }
}
""")
    with pytest.raises(MojoRefusal, match="queries the size of this"):
        crosstl.translate(str(path), backend="mojo", format_output=False)

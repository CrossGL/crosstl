"""CUDA unions keep their overlapping storage through CrossGL's union layout contract.

The members of a CUDA union share storage, and CUDA code reads a member other than the one it wrote to reinterpret the
stored bytes, which nvcc, GCC and Clang define. The importer used to write a union as a CrossGL struct, so the read
returned a member that was never written. It now marks each union ``@union_layout(size, alignment, little_endian,
cuda)`` and each member ``@union_member_layout(0, size, alignment)`` with its CUDA layout, as the Metal importer does
for MSL, and the word-based targets lower the overlap. A union whose layout the importer cannot describe is refused.
"""

from __future__ import annotations

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.backend.import_refusals import CudaImportRefused
from crosstl.translator.codegen.metal_codegen import UnsupportedMetalFeatureError

BITS = """
union Bits {
    float f;
    unsigned int u;
};
__global__ void probe(float* inp, unsigned int* outp) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    union Bits b;
    b.f = inp[i];
    outp[i] = b.u;
}
"""

LOCAL = """
__device__ unsigned floatAsUInt(float x) {
    volatile union { float f; unsigned i; } xx;
    xx.f = x;
    return xx.i;
}
__global__ void k(float* a, unsigned* o) { o[threadIdx.x] = floatAsUInt(a[threadIdx.x]); }
"""


def _translate(tmp_path, source, backend):
    path = tmp_path / "kernel.cu"
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend=backend, format_output=False)


def _flat(code):
    return re.sub(r"\s+", " ", code)


@pytest.mark.parametrize("declaration", ["union Bits b;", "Bits b;"])
def test_union_imports_with_its_cuda_layout_contract(tmp_path, declaration):
    code = _flat(
        _translate(tmp_path, BITS.replace("union Bits b;", declaration), "cgl")
    )
    assert "@union_layout(4, 4, little_endian, cuda) struct Bits {" in code
    assert "@union_member_layout(0, 4, 4) f32 f;" in code
    assert "@union_member_layout(0, 4, 4) u32 u;" in code


def test_local_union_keeps_its_layout_contract(tmp_path):
    code = _flat(_translate(tmp_path, LOCAL, "cgl"))
    assert (
        "@union_layout(4, 4, little_endian, cuda) struct anonymous_xx_layout {" in code
    )
    assert "@union_member_layout(0, 4, 4) f32 f;" in code
    assert "@union_member_layout(0, 4, 4) u32 i;" in code


@pytest.mark.parametrize(
    ("declarations", "union", "members"),
    [
        (
            "union U { int i; char c; };",
            "@union_layout(4, 4, little_endian, cuda) struct U",
            [
                "@union_member_layout(0, 4, 4) i32 i;",
                "@union_member_layout(0, 1, 1) i8 c;",
            ],
        ),
        (
            "union U { float4 v; float a[4]; };",
            "@union_layout(16, 16, little_endian, cuda) struct U",
            [
                "@union_member_layout(0, 16, 16) vec4<f32> v;",
                "@union_member_layout(0, 16, 4)",
            ],
        ),
        (
            # A three-component vector is aligned to its scalar, so the union is 16 bytes at double's alignment.
            "union U { double d; float3 v; };",
            "@union_layout(16, 8, little_endian, cuda) struct U",
            [
                "@union_member_layout(0, 8, 8) f64 d;",
                "@union_member_layout(0, 12, 4) vec3<f32> v;",
            ],
        ),
        (
            "union __align__(8) U { float f; unsigned u; };",
            "@union_layout(8, 8, little_endian, cuda) struct U",
            [
                "@union_member_layout(0, 4, 4) f32 f;",
                "@union_member_layout(0, 4, 4) u32 u;",
            ],
        ),
        (
            # P is {float a at 0; float2 b at 8}: 16 bytes at 8-byte alignment.
            "struct P { float a; float2 b; }; union U { P p; float4 v; };",
            "@union_layout(16, 16, little_endian, cuda) struct U",
            [
                "@union_member_layout(0, 16, 8) P p;",
                "@union_member_layout(0, 16, 16) vec4<f32> v;",
            ],
        ),
        (
            # A union member of union type is as large as its largest member, not the sum of them.
            "union Inner { char c; int i; }; union U { Inner inner; short s; };",
            "@union_layout(4, 4, little_endian, cuda) struct U",
            [
                "@union_member_layout(0, 4, 4) Inner inner;",
                "@union_member_layout(0, 2, 2) i16 s;",
            ],
        ),
        (
            "typedef unsigned int word; union U { word w; float f; };",
            "@union_layout(4, 4, little_endian, cuda) struct U",
            ["@union_member_layout(0, 4, 4)", "@union_member_layout(0, 4, 4) f32 f;"],
        ),
    ],
)
def test_union_layout_follows_cuda_rules(tmp_path, declarations, union, members):
    source = declarations + "\n__global__ void k(float* o) { U x; o[0] = 0.0f; }\n"
    code = _flat(_translate(tmp_path, source, "cgl"))
    assert union in code
    for member in members:
        assert member in code
    assert code.count("@union_member_layout(") == len(members) + (
        2 if "union Inner" in declarations else 0
    )


@pytest.mark.parametrize(
    ("declarations", "reason"),
    [
        ("union U { unsigned int a : 3; unsigned int b; };", "bit-field"),
        (
            "union U { struct { unsigned short lo; unsigned short hi; }; unsigned int w; };",
            "anonymous struct",
        ),
        ("union U { alignas(16) float f; int i; };", "alignment"),
        ("union U { float* p; int i; };", "member 'p'"),
        ("union __align__(alignof(float4)) U { float f; int i; };", "__align__"),
        ("union U { double4 d; float f; };", "double4"),
        ("struct U { union { float f; unsigned u; }; };", "anonymous union"),
    ],
)
def test_union_whose_layout_cannot_be_described_is_refused(
    tmp_path, declarations, reason
):
    source = declarations + "\n__global__ void k(float* o) { U x; o[0] = 0.0f; }\n"
    with pytest.raises(CudaImportRefused, match=re.escape(reason)) as refused:
        _translate(tmp_path, source, "cgl")
    assert "U" in str(refused.value)


def test_directx_and_opengl_read_the_bits_that_were_written(tmp_path):
    hlsl = _flat(_translate(tmp_path, BITS, "directx"))
    assert "b.CrossGLUnionStorage = asuint(" in hlsl
    assert "b.u" not in hlsl
    glsl = _flat(_translate(tmp_path, BITS, "opengl"))
    assert "b.CrossGLUnionStorage = floatBitsToUint(" in glsl
    assert "b.u" not in glsl


def test_metal_refuses_to_rebuild_a_cuda_union(tmp_path):
    with pytest.raises(UnsupportedMetalFeatureError, match="source ABI 'cuda'"):
        _translate(tmp_path, BITS, "metal")


# A kernel without launch bounds leaves its workgroup size to the host.
WORKGROUP = [f"-DCROSSGL_WORKGROUP_SIZE_{axis}=1" for axis in "XYZ"]


@pytest.mark.parametrize("source", [BITS, LOCAL])
def test_native_validators_accept_the_translations(tmp_path, source):
    ran = False
    if shutil.which("glslangValidator"):
        shader = tmp_path / "kernel.comp"
        shader.write_text(_translate(tmp_path, source, "opengl"), encoding="utf-8")
        _run(["glslangValidator", "-l", "-S", "comp", *WORKGROUP, str(shader)])
        ran = True
    if shutil.which("dxc"):
        shader = tmp_path / "kernel.hlsl"
        shader.write_text(_translate(tmp_path, source, "directx"), encoding="utf-8")
        _run(["dxc", "-T", "cs_6_0", "-E", "CSMain", *WORKGROUP, str(shader)])
        ran = True
    if not ran:
        pytest.skip("no native shader validator is on PATH")


def _run(command):
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr

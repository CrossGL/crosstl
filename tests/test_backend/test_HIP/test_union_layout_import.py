"""HIP unions keep their overlapping storage through CrossGL's union layout contract.

The members of a HIP union share storage, and HIP code reads a member other than the one it wrote to reinterpret the
stored bytes. The importer used to write a union as a CrossGL struct and refuse every program that read another
member, copied a union, passed one to a function or reached one through a pointer. It now marks each named union
``@union_layout(size, alignment, little_endian, hip)`` and each member ``@union_member_layout(0, size, alignment)``
with the layout hipcc gives it, as the CUDA importer does, so those programs import. A union whose layout the importer
cannot describe is refused, and an anonymous union, whose members the importer writes into the enclosing record, is
still refused when a member other than the one written is read.
"""

from __future__ import annotations

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.backend.HIP.HipCrossGLCodeGen import HipRecordSemanticError
from crosstl.backend.import_refusals import HipImportRefused
from crosstl.translator.codegen.metal_codegen import UnsupportedMetalFeatureError

HEADER = "#include <hip/hip_runtime.h>\n"

BITS = """
union Bits {
    float f;
    unsigned int u;
};
__global__ void probe(const float* inp, float* outp) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    Bits b;
    b.f = inp[i];
    b.u = (b.u >> 16) | 0x3F800000u;
    outp[i] = b.f;
}
"""

LOCAL = """
__device__ unsigned floatAsUInt(float x) {
    union { float f; unsigned i; } xx;
    xx.f = x;
    return xx.i;
}
__global__ void k(float* a, unsigned* o) { o[threadIdx.x] = floatAsUInt(a[threadIdx.x]); }
"""

UNION = "union Bits { float f; unsigned int u; };\n"

# Programs the importer refused while it tracked the active member of every union.
VALUE_FLOWS = {
    "other-member": (
        UNION + "__global__ void k(float* o) { Bits b; b.f = o[0]; o[1] = (float)b.u; }"
    ),
    "copy": (
        UNION
        + "__global__ void k(float* o) { Bits a; a.f = o[0]; Bits b = a; o[1] = (float)b.u; }"
    ),
    "argument": (
        UNION + "__device__ unsigned bits(Bits b) { return b.u; }\n"
        "__global__ void k(float* o) { Bits b; b.f = o[0]; o[1] = (float)bits(b); }"
    ),
    "pointer": (
        UNION
        + "__global__ void k(float* o) { Bits b; Bits* p = &b; p->f = o[0]; o[1] = (float)p->u; }"
    ),
    "struct-member": (
        UNION + "struct S { Bits b; int k; };\n"
        "__global__ void k(float* o) { S s; s.b.f = o[0]; o[1] = (float)s.b.u; }"
    ),
    "array": (
        UNION
        + "__global__ void k(float* o) { Bits b[2]; b[0].f = o[0]; o[1] = (float)b[0].u; }"
    ),
    "initializer": (
        UNION + "__global__ void k(float* o) { Bits b = {o[0]}; o[1] = (float)b.u; }"
    ),
}


def _translate(tmp_path, source, backend):
    path = tmp_path / "kernel.hip"
    path.write_text(HEADER + source, encoding="utf-8")
    return crosstl.translate(str(path), backend=backend, format_output=False)


def _flat(code):
    return re.sub(r"\s+", " ", code)


@pytest.mark.parametrize("declaration", ["union Bits b;", "Bits b;"])
def test_union_imports_with_its_hip_layout_contract(tmp_path, declaration):
    code = _flat(_translate(tmp_path, BITS.replace("Bits b;", declaration), "cgl"))
    assert "@union_layout(4, 4, little_endian, hip) struct Bits {" in code
    assert "@union_member_layout(0, 4, 4) f32 f;" in code
    assert "@union_member_layout(0, 4, 4) u32 u;" in code
    assert "overlapping storage is not modeled" not in code


@pytest.mark.parametrize("name", sorted(VALUE_FLOWS))
def test_union_value_flows_import(tmp_path, name):
    code = _flat(_translate(tmp_path, VALUE_FLOWS[name], "cgl"))
    assert "@union_layout(4, 4, little_endian, hip) struct Bits {" in code


def test_local_union_keeps_its_layout_contract(tmp_path):
    code = _flat(_translate(tmp_path, LOCAL, "cgl"))
    assert re.search(
        r"@union_layout\(4, 4, little_endian, hip\) struct CrossGLFunctionLocal_floatAsUInt_xx_\w+ \{",
        code,
    )
    assert "@union_member_layout(0, 4, 4) u32 i;" in code


@pytest.mark.parametrize(
    ("declarations", "union", "members"),
    [
        (
            "union U { int i; char c; };",
            "@union_layout(4, 4, little_endian, hip) struct U",
            ["@union_member_layout(0, 4, 4) i32 i;", "@union_member_layout(0, 1, 1)"],
        ),
        (
            "union __attribute__((aligned(8))) U { float f; unsigned int u; };",
            "@union_layout(8, 8, little_endian, hip) struct U",
            ["@union_member_layout(0, 4, 4) f32 f;"],
        ),
        (
            "union U { float4 v; float a[4]; };",
            "@union_layout(16, 16, little_endian, hip) struct U",
            [
                "@union_member_layout(0, 16, 16) vec4<f32> v;",
                "@union_member_layout(0, 16, 4) array<f32, 4> a;",
            ],
        ),
        (
            "union U { float3 v; double d; };",
            "@union_layout(16, 8, little_endian, hip) struct U",
            ["@union_member_layout(0, 12, 4) vec3<f32> v;"],
        ),
        (
            "union U { float f; struct { unsigned short lo, hi; } h; };",
            "@union_layout(4, 4, little_endian, hip) struct U",
            ["@union_member_layout(0, 4, 2) CrossGLAnonymous_U_h_struct_"],
        ),
        (
            "union U { float f; union { unsigned int u; int i; }; };",
            "@union_layout(4, 4, little_endian, hip) struct U",
            [
                "@union_member_layout(0, 4, 4) u32 u;",
                "@union_member_layout(0, 4, 4) i32 i;",
            ],
        ),
    ],
)
def test_union_layout_follows_hip_rules(tmp_path, declarations, union, members):
    source = declarations + "\n__global__ void k(float* o) { U x; o[0] = 0.0f; }\n"
    code = _flat(_translate(tmp_path, source, "cgl"))
    assert union in code
    for member in members:
        assert member in code


@pytest.mark.parametrize(
    ("declarations", "reason"),
    [
        ("union U { unsigned int a : 3; unsigned int b; };", "bit-field"),
        (
            "union U { struct { unsigned short lo; unsigned short hi; }; unsigned int w; };",
            "anonymous struct members 'lo', 'hi'",
        ),
        ("union U { alignas(16) float f; int i; };", "alignas(16)"),
        ("union U { float f __attribute__((aligned(16))); int i; };", "aligned(16)"),
        ("union __attribute__((packed)) U { float f; int i; };", "packed"),
        ("union U { float* p; int i; };", "member 'p'"),
        ("union U { double4 d; float f; };", "double4"),
    ],
)
def test_union_whose_layout_cannot_be_described_is_refused(
    tmp_path, declarations, reason
):
    source = declarations + "\n__global__ void k(float* o) { o[0] = 0.0f; }\n"
    with pytest.raises(HipImportRefused, match=re.escape(reason)) as refused:
        _translate(tmp_path, source, "cgl")
    assert "HIP union 'U'" in str(refused.value)


def test_anonymous_union_in_a_struct_still_tracks_its_active_member(tmp_path):
    source = (
        "struct S { union { float f; unsigned int u; }; int k; };\n"
        "__global__ void k(float* o) { S s; s.f = o[0]; o[1] = (float)s.u; }\n"
    )
    with pytest.raises(HipRecordSemanticError, match="overlapping storage access"):
        _translate(tmp_path, source, "cgl")


def test_union_kernel_buffer_is_refused(tmp_path):
    source = UNION + "__global__ void k(Bits* d, unsigned int* o) { o[0] = d[0].u; }\n"
    with pytest.raises(Exception, match="it is a union"):
        _translate(tmp_path, source, "cgl")


def test_directx_and_opengl_read_the_bits_that_were_written(tmp_path):
    hlsl = _flat(_translate(tmp_path, BITS, "directx"))
    assert "b.CrossGLUnionStorage = asuint(" in hlsl
    assert "b.u" not in hlsl
    glsl = _flat(_translate(tmp_path, BITS, "opengl"))
    assert "b.CrossGLUnionStorage = floatBitsToUint(" in glsl
    assert "b.u" not in glsl


def test_metal_refuses_to_rebuild_a_hip_union(tmp_path):
    with pytest.raises(UnsupportedMetalFeatureError, match="source ABI 'hip'"):
        _translate(tmp_path, BITS, "metal")


@pytest.mark.parametrize("backend", ["hip", "cuda"])
def test_cuda_and_hip_output_rebuild_the_union(tmp_path, backend):
    code = _flat(_translate(tmp_path, BITS, backend))
    assert "union Bits {" in code
    assert "static_assert(sizeof(Bits) == 4" in code


# A kernel without launch bounds leaves its workgroup size to the host.
WORKGROUP = [f"-DCROSSGL_WORKGROUP_SIZE_{axis}=1" for axis in "XYZ"]


@pytest.mark.parametrize("source", [BITS, LOCAL, VALUE_FLOWS["argument"]])
def test_native_compilers_accept_the_translations(tmp_path, source):
    ran = False
    if shutil.which("hipcc"):
        kernel = tmp_path / "kernel_out.hip"
        kernel.write_text(_translate(tmp_path, source, "hip"), encoding="utf-8")
        _run(
            ["hipcc", "--offload-arch=gfx1100", "-std=c++17", "-c", str(kernel)]
            + ["-o", str(tmp_path / "hip.o")]
        )
        ran = True
    if shutil.which("nvcc"):
        kernel = tmp_path / "kernel_out.cu"
        kernel.write_text(_translate(tmp_path, source, "cuda"), encoding="utf-8")
        _run(
            ["nvcc", "-arch=sm_80", "-std=c++17", "-c", str(kernel)]
            + ["-o", str(tmp_path / "cuda.o")]
        )
        ran = True
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
        pytest.skip("no native compiler or validator is on PATH")


def _run(command):
    result = subprocess.run(
        command, capture_output=True, text=True, check=False, timeout=600
    )
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]

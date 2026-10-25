"""CUDA and HIP kernel buffers keep their element layout on every target.

CUDA's ``float3`` is 12 bytes with 4-byte alignment and ``float2`` is 8-byte aligned, so ``const float3 *`` has a
12-byte stride and ``struct { float3 a; float2 b; }`` is 24 bytes with ``b`` at 16. CrossGL's ``vec3`` has a
16-byte stride in std430 and MSL, and the HLSL structured-buffer rules put ``b`` at 12. The importers mark each
kernel buffer with the CrossGL buffer layout that reproduces CUDA's (``hlsl_structured`` or ``metal`` with
``@ packed`` members), or refuse it. Execution is covered by the ``u_float3_stride`` and ``u_struct_layout``
conformance cases.
"""

from __future__ import annotations

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.backend.CUDA.buffer_layout_import import (
    HLSL_STRUCTURED,
    METAL,
    KernelBufferLayoutError,
    KernelBufferLayouts,
)
from crosstl.translator.codegen.glsl_generic_vectors import (
    glsl_generic_vector_constructor_name,
)

FLOAT3 = """
__global__ void k(const float3 *v, float *o) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    o[i] = v[i].y;
}
"""

STRUCTS = """
struct P { float a; float3 b; };
struct R { float3 a; float2 b; };
struct Q { float a; float2 b; };
__global__ void k(const P *p, R *r, const Q *q, float4 *f, float *o) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    o[i] = p[i].b.z + r[i].a.y + q[i].b.x + f[i].w;
    r[i].b = make_float2(o[i], 1.0f);
}
"""

SHARED = """
struct Y { float3 a; float b; float2 c; };
struct Z { float a; Y y; };
__global__ void k(const Y *ys, Z *zs, float *o) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    o[i] = zs[i].y.b + ys[i].c.y;
}
"""


def _translate(tmp_path, source, backend, suffix="cu"):
    path = tmp_path / f"kernel.{suffix}"
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend=backend, format_output=False)


def _flat(code):
    return re.sub(r"\s+", " ", code)


def _plan(structs, *elements, aliases=None):
    planner = KernelBufferLayouts(
        {name: (members, ()) for name, members in structs.items()}, aliases or {}
    )
    planner.plan((f"buffer {index}", text) for index, text in enumerate(elements))
    return planner


@pytest.mark.parametrize(
    ("element", "layout"),
    [
        ("f32", None),
        ("vec2<f32>", None),
        ("vec4<f32>", None),
        ("vec4<f64>", None),
        ("vec3<f32>", HLSL_STRUCTURED),
        ("vec3<i32>", HLSL_STRUCTURED),
        ("vec3<f64>", HLSL_STRUCTURED),
        ("bool", None),
    ],
)
def test_vector_elements_take_the_layout_of_their_cuda_stride(element, layout):
    assert _plan({}, element).layout(element) == layout


def test_struct_elements_take_the_layout_that_places_their_members():
    structs = {
        "P": [("f32", "a"), ("vec3<f32>", "b")],
        "R": [("vec3<f32>", "a"), ("vec2<f32>", "b")],
        "Q": [("f32", "a"), ("vec2<f32>", "b")],
        "T": [("array<R,2>", "r"), ("f32", "c")],
    }
    planner = _plan(structs, "P", "R", "Q", "T")
    # P: b at 4, as in HLSL structured buffers. Q: b at 8, as in std430 and MSL but not HLSL (4).
    assert planner.layout("P") == HLSL_STRUCTURED
    assert planner.layout("Q") == METAL
    # R: b at 16 after a 12-byte a, which MSL reproduces with a packed vec3.
    assert planner.layout("R") == METAL and planner.layout("T") == METAL
    assert planner.packed_member("R", "vec3<f32>")
    assert not planner.packed_member("R", "vec2<f32>")
    assert not planner.packed_member("P", "vec3<f32>")


def test_a_struct_shared_with_an_msl_layout_buffer_takes_msl_layout():
    structs = {
        "Y": [("vec3<f32>", "a"), ("f32", "b"), ("vec2<f32>", "c")],
        "Z": [("f32", "a"), ("Y", "y")],
    }
    # Y alone matches the HLSL structured layout, but Z needs Y 8-byte aligned at offset 8, and Metal declares one Y.
    assert _plan(structs, "Y").layout("Y") == HLSL_STRUCTURED
    planner = _plan(structs, "Y", "Z")
    assert planner.layout("Y") == METAL and planner.layout("Z") == METAL
    assert planner.packed_member("Y", "vec3<f32>")


def test_type_aliases_resolve_to_their_types():
    assert _plan({}, "V", aliases={"V": "vec3<f32>"}).layout("V") == HLSL_STRUCTURED


@pytest.mark.parametrize(
    ("structs", "attributes", "message"),
    [
        (
            {"S": [("f32", "a"), ("vec4<f64>", "b")]},
            {},
            "double4_32a",
        ),
        (
            {"S": [("f32", "a")], "W": [("f32", "x"), ("S", "s")]},
            {"S": ("__attribute__((packed))",)},
            "layout attribute",
        ),
    ],
)
def test_elements_no_layout_reproduces_are_refused(structs, attributes, message):
    planner = KernelBufferLayouts(
        {name: (members, attributes.get(name, ())) for name, members in structs.items()}
    )
    element = "W" if "W" in structs else "S"
    with pytest.raises(KernelBufferLayoutError, match=message):
        planner.plan([("buffer", element)])


def test_declared_struct_alignment_is_kept_or_refused():
    aligned = {"A": ([("f32", "x")], ("__align__(16)",)), "B": ([("A", "a")], ())}
    planner = KernelBufferLayouts(aligned)
    with pytest.raises(KernelBufferLayoutError, match="declared alignment"):
        planner.plan([("buffer", "B")])
    # A 16-byte aligned struct of four floats has the same layout everywhere.
    planner = KernelBufferLayouts({"A": ([("vec4<f32>", "x")], ("__align__(16)",))})
    planner.plan([("buffer", "A")])
    assert planner.layout("A") is None


def test_cuda_import_marks_buffers_and_packed_members(tmp_path):
    code = _translate(tmp_path, STRUCTS, "cgl")
    flat = _flat(code)
    assert (
        "@buffer_layout(hlsl_structured) var<storage, read_write> p: array<P>" in flat
    )
    assert "@buffer_layout(metal) var<storage, read_write> r: array<R>" in flat
    assert "@buffer_layout(metal) var<storage, read_write> q: array<Q>" in flat
    assert "@binding(3) var<storage, read_write> f: array<vec4<f32>" in flat
    assert "struct R { @packed vec3<f32> a; vec2<f32> b; }" in flat
    assert "struct P { f32 a; vec3<f32> b; }" in flat


def test_hip_import_marks_buffers_and_packed_members(tmp_path):
    source = "#include <hip/hip_runtime.h>\n" + STRUCTS
    flat = _flat(_translate(tmp_path, source, "cgl", suffix="hip"))
    assert "@buffer_layout(hlsl_structured) var<storage, read> p: array<P>" in flat
    assert "@buffer_layout(metal) var<storage, read_write> r: array<R>" in flat
    assert "struct R { @packed vec3<f32> a; vec2<f32> b; }" in flat


def test_cuda_unions_in_buffers_are_refused(tmp_path):
    source = """
union U { float a; int b; };
__global__ void k(const U *u, float *o) { o[0] = u[0].a; }
"""
    with pytest.raises(KernelBufferLayoutError, match="union"):
        _translate(tmp_path, source, "cgl")


def test_metal_keeps_cuda_strides(tmp_path):
    code = _flat(_translate(tmp_path, FLOAT3, "metal"))
    assert "device packed_float3* v [[buffer(0)]]" in code
    code = _flat(_translate(tmp_path, SHARED, "metal"))
    assert "struct Y { packed_float3 a; float b; float2 c; };" in code


def test_opengl_reads_cuda_buffers_in_words(tmp_path):
    code = _flat(_translate(tmp_path, FLOAT3, "opengl"))
    assert "buffer vBuffer { uint v[]; }" in code
    assert "uintBitsToFloat(v[((uint(i) * 3u) + 1u)])" in code
    code = _flat(_translate(tmp_path, STRUCTS, "opengl"))
    # R is six words with b at word 4; P is four words with b.z at word 3.
    assert "uintBitsToFloat(r[((uint(i) * 6u) + 1u)])" in code
    assert "uintBitsToFloat(p[((uint(i) * 4u) + 3u)])" in code
    assert "vec2<f32>" not in code


def test_hlsl_keeps_structured_elements_and_reads_others_in_words(tmp_path):
    code = _flat(_translate(tmp_path, STRUCTS, "directx"))
    assert "RWStructuredBuffer<P> p : register(u0);" in code
    assert "RWStructuredBuffer<uint> r : register(u1);" in code
    assert "RWStructuredBuffer<uint> q : register(u2);" in code
    assert "RWStructuredBuffer<float4> f : register(u3);" in code
    # R.b is at word 4 of six; Q.b.x at word 2 of four.
    assert "asfloat(q[((uint(i) * 4u) + 2u)])" in code


@pytest.mark.parametrize("source", [FLOAT3, STRUCTS, SHARED])
def test_native_validators_accept_the_translations(tmp_path, source):
    ran = False
    if shutil.which("glslangValidator"):
        shader = tmp_path / "kernel.comp"
        shader.write_text(_translate(tmp_path, source, "opengl"), encoding="utf-8")
        _run(["glslangValidator", "-S", "comp", str(shader)])
        ran = True
    if shutil.which("spirv-as") and shutil.which("spirv-val"):
        assembly = tmp_path / "kernel.spvasm"
        binary = tmp_path / "kernel.spv"
        assembly.write_text(_translate(tmp_path, source, "vulkan"), encoding="utf-8")
        _run(
            ["spirv-as", "--target-env", "vulkan1.1", str(assembly), "-o", str(binary)]
        )
        _run(["spirv-val", "--target-env", "vulkan1.1", str(binary)])
        ran = True
    if shutil.which("xcrun") and _metal_available():
        shader = tmp_path / "kernel.metal"
        shader.write_text(_translate(tmp_path, source, "metal"), encoding="utf-8")
        _run(["xcrun", "-sdk", "macosx", "metal", "-c", str(shader), "-o", "/dev/null"])
        ran = True
    if not ran:
        pytest.skip("no native shader validator is on PATH")


def _metal_available():
    result = subprocess.run(
        ["xcrun", "-sdk", "macosx", "-f", "metal"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0


def _run(command):
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    ("name", "glsl"),
    [
        ("vec3<f32>", "vec3"),
        ("vec2<i32>", "ivec2"),
        ("vec4<u32>", "uvec4"),
        ("vec3<f64>", "dvec3"),
        ("vec2<bool>", "bvec2"),
        ("vec3<f16>", "f16vec3"),
        ("vec2<u16>", "u16vec2"),
        ("vec4<i8>", "i8vec4"),
        ("vec2<i64>", "i64vec2"),
        ("vec2<Pair>", "vec2<Pair>"),
        ("max", "max"),
    ],
)
def test_generic_vector_constructors_take_glsl_names(name, glsl):
    assert glsl_generic_vector_constructor_name(name) == glsl

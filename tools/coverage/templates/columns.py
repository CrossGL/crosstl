"""Uniform-matrix2, uniform-matrix3 and uniform-non-square-matrix probes: constants blocks that hold a 2x2 matrix, a
3x3 matrix followed by a vector, or a two-column, three-row matrix followed by a vector, in every language that has
matrices.

std140 and a cbuffer start every matrix column on a 16-byte row, so a 2x2 matrix's second column is at offset 16 and a
3x3 matrix fills 48 bytes. Metal and WGSL give a ``float2x2`` and a ``mat2x2<f32>`` an 8-byte column stride and
Rust's glam packs ``Mat3`` into 36 bytes, so those probes declare their columns as four-component vectors and build the
matrix from them: the same bytes. A translation has to keep the column stride, the matrix's size and its orientation
for both probes to read the same words. Every language gives a three-row column 16 bytes, so the non-square probes
declare their matrix directly; HLSL names CrossGL's ``mat2x3`` ``float3x2``, indexes it by rows and multiplies with
``mul``, so its probes swap the indices. CUDA, HIP, OpenCL C and Mojo have no matrices, and glam no non-square ones.
As in :mod:`.blocks`, each probe combines the block's words exactly and multiplies its input once.
"""

from __future__ import annotations

from .base import ABSENT, Template

NAMES = ("uniform-matrix2", "uniform-matrix3", "uniform-non-square-matrix")

# {0} is the prefix that names a member of the block; Metal reads the 2x2 matrix from a local.
_C_BODIES = {
    "uniform-matrix2": (
        "$vec2 t = m * $vec2(1.0, -0.5);\n"
        "outp[i] = fin[i] * (t.x + t.y * 2.0 + m[1][0] * 4.0);"
    ),
    "uniform-matrix3": (
        "$vec3 t = {0}m * {0}v.xyz;\n"
        "outp[i] = fin[i] * (t.x + t.y * 2.0 + t.z * 4.0 + {0}m[2][1] * {0}v.w);"
    ),
    "uniform-non-square-matrix": (
        "$vec3 t = {0}m * {0}v.xy;\n$vec2 u = {0}v.xyz * {0}m;\n"
        "outp[i] = fin[i] * (t.x + t.y * 2.0 + t.z * 4.0 + u.y * 8.0 + {0}m[1][2] * {0}v.w);"
    ),
}
_HLSL_BODIES = {
    "uniform-matrix2": (
        "float2 t = mul(m, float2(1.0, -0.5));\n"
        "outp[i] = fin[i] * (t.x + t.y * 2.0 + m[0][1] * 4.0);"
    ),
    "uniform-matrix3": (
        "float3 t = mul(m, v.xyz);\n"
        "outp[i] = fin[i] * (t.x + t.y * 2.0 + t.z * 4.0 + m[1][2] * v.w);"
    ),
    "uniform-non-square-matrix": (
        "float3 t = mul(m, v.xy);\nfloat2 u = mul(v.xyz, m);\n"
        "outp[i] = fin[i] * (t.x + t.y * 2.0 + t.z * 4.0 + u.y * 8.0 + m[2][1] * v.w);"
    ),
}
# Each block's members where every matrix column takes a 16-byte row.
_MEMBERS = {
    "uniform-matrix2": "    $mat2 m;",
    "uniform-matrix3": "    $mat3 m;\n    $vec4 v;",
    "uniform-non-square-matrix": "    $mat2x3 m;\n    $vec4 v;",
}
_HLSL_MEMBERS = {
    "uniform-matrix2": "    float2x2 m;",
    "uniform-matrix3": "    float3x3 m;\n    float4 v;",
    "uniform-non-square-matrix": "    float3x2 m;\n    float4 v;",
}


def _blocks(head: str, tail: str) -> dict[str, Template]:
    """The probes of a language with std140 packing that declares the block ``head`` members ``tail``."""
    return {
        name: Template(body.format(""), f"{head}\n{_MEMBERS[name]}\n{tail}")
        for name, body in _C_BODIES.items()
    }


GLSL = _blocks("layout(std140, binding = 3) uniform Params {", "};")
WEBGL = _blocks("layout(std140) uniform Params {", "};")
FAMILY = {
    name: Template(
        body, f"cbuffer Params : register(b3) {{\n{_HLSL_MEMBERS[name]}\n}};"
    )
    for name, body in _HLSL_BODIES.items()
}
CROSSGL = {
    name: Template(
        body.format(""),
        prelude=f"cbuffer Params @binding(3) {{\n{_MEMBERS[name]}\n}};",
    )
    for name, body in _C_BODIES.items()
}

# Metal declares the 2x2 matrix's columns as vectors and builds the matrix in the kernel.
_METAL_MEMBERS = {**_MEMBERS, "uniform-matrix2": "    $vec4 c[2];"}
_METAL_BUILD = {
    "uniform-matrix2": "float2x2 m = float2x2(params.c[0].xy, params.c[1].xy);\n"
}
METAL = {
    name: Template(
        _METAL_BUILD.get(name, "") + body.format("params."),
        f"struct Params {{\n{_METAL_MEMBERS[name]}\n}};",
        params="constant Params& params [[buffer(3)]]",
    )
    for name, body in _C_BODIES.items()
}

_WGSL_MEMBERS = {
    "uniform-matrix2": "    c: array<vec4<f32>, 2>,",
    "uniform-matrix3": "    m: mat3x3<f32>,\n    v: vec4<f32>,",
    "uniform-non-square-matrix": "    m: mat2x3<f32>,\n    v: vec4<f32>,",
}
# WGSL infers the type of each local.
_WGSL_BODIES = {
    "uniform-matrix2": (
        "let m = mat2x2<f32>(params.c[0].xy, params.c[1].xy);\n"
        "let t = m * vec2<f32>(1.0, -0.5);\n"
        "outp[i] = fin[i] * (t.x + t.y * 2.0 + m[1][0] * 4.0);"
    ),
    "uniform-matrix3": (
        "let t = params.m * params.v.xyz;\n"
        "outp[i] = fin[i] * (t.x + t.y * 2.0 + t.z * 4.0 + params.m[2][1] * params.v.w);"
    ),
    "uniform-non-square-matrix": (
        "let t = params.m * params.v.xy;\nlet u = params.v.xyz * params.m;\n"
        "outp[i] = fin[i] * (t.x + t.y * 2.0 + t.z * 4.0 + u.y * 8.0 + params.m[1][2] * params.v.w);"
    ),
}
WGSL = {
    name: Template(
        body,
        f"struct Params {{\n{_WGSL_MEMBERS[name]}\n}}\n@group(0) @binding(3) var<uniform> params: Params;",
    )
    for name, body in _WGSL_BODIES.items()
}

CUDA = OPENCL = MOJO = dict.fromkeys(NAMES, ABSENT)

_RUST_DERIVE = "#[derive(Clone, Copy, Default)]\n#[repr(C)]\n"
_RUST_UNIFORM = "#[spirv(uniform, descriptor_set = 0, binding = 3)] params: &Params"
# glam builds a matrix from its columns and names them x_axis, y_axis and z_axis.
RUST = {
    "uniform-matrix2": Template(
        "let m = Mat2::from_cols(Vec2::new(params.c[0].x, params.c[0].y), Vec2::new(params.c[1].x, params.c[1].y));\n"
        "let t = m * Vec2::new(1.0, -0.5);\n"
        "outp[i] = fin[i] * (t.x + t.y * 2.0 + m.y_axis.x * 4.0);",
        f"{_RUST_DERIVE}pub struct Params {{\n    c: [Vec4; 2],\n}}",
        params=_RUST_UNIFORM,
    ),
    "uniform-matrix3": Template(
        "let m = Mat3::from_cols(params.c[0].truncate(), params.c[1].truncate(), params.c[2].truncate());\n"
        "let t = m * params.v.truncate();\n"
        "outp[i] = fin[i] * (t.x + t.y * 2.0 + t.z * 4.0 + m.z_axis.y * params.v.w);",
        f"{_RUST_DERIVE}pub struct Params {{\n    c: [Vec4; 3],\n    v: Vec4,\n}}",
        params=_RUST_UNIFORM,
    ),
    "uniform-non-square-matrix": ABSENT,
}

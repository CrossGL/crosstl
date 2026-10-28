"""Uniform-row-major-matrix, uniform-row-major-non-square and uniform-row-major-array probes: constants blocks that
store a 3x3 matrix, a two-column, three-row matrix or an array of two 2x2 matrices by rows, in every language that has
matrices.

A row-major matrix keeps each row on its own 16-byte row of the block, so the element in column ``c`` and row ``r``
is at ``16 * r + 4 * c``, and a two-column, three-row matrix takes 48 bytes where its column-major form takes 32. A
translation has to keep the orientation for both probes to read the same words: read by columns, the products, the
element and the vector after the non-square matrix all change. CrossGL marks such a member ``@ row_major``, GLSL
``layout(row_major)`` and HLSL and Slang ``row_major``. Metal, WGSL and Rust have no row-major matrices, so their
probes declare the rows as four-component vectors and build the matrix as the transpose of the one whose columns they
are. HLSL indexes a matrix by rows and multiplies with ``mul``, so its probes swap the indices. CUDA, HIP, OpenCL C
and Mojo have no matrices, and glam no non-square ones. As in :mod:`.blocks`, each probe combines the block's words
exactly and multiplies its input once.
"""

from __future__ import annotations

from .base import ABSENT, Template

NAMES = (
    "uniform-row-major-matrix",
    "uniform-row-major-non-square",
    "uniform-row-major-array",
)

# {m} prefixes the matrix and {v} the vector; Metal and WGSL read a matrix they build in a local.
_C_BODIES = {
    "uniform-row-major-matrix": (
        "$vec3 t = {m}m * {v}v.xyz;\n$vec3 u = {v}v.xyz * {m}m;\n"
        "outp[i] = fin[i] * (t.x + t.y * 2.0 + t.z * 4.0 + u.x * 8.0 + {m}m[2][1] * {v}v.w);"
    ),
    "uniform-row-major-non-square": (
        "$vec3 t = {m}m * {v}v.xy;\n$vec2 u = {v}v.xyz * {m}m;\n"
        "outp[i] = fin[i] * (t.x + t.y * 2.0 + t.z * 4.0 + u.y * 8.0 + {m}m[1][2] * {v}v.w);"
    ),
    "uniform-row-major-array": (
        "int k = iin[i] & 1;\n$vec2 t = {m}m[k] * $vec2(1.0, -0.5);\n$vec2 u = $vec2(0.5, 2.0) * {m}m[1];\n"
        "outp[i] = fin[i] * (t.x + t.y * 2.0 + u.x * 4.0 - u.y + {m}m[0][1][0] * 8.0);"
    ),
}
_HLSL_BODIES = {
    "uniform-row-major-matrix": (
        "float3 t = mul(m, v.xyz);\nfloat3 u = mul(v.xyz, m);\n"
        "outp[i] = fin[i] * (t.x + t.y * 2.0 + t.z * 4.0 + u.x * 8.0 + m[1][2] * v.w);"
    ),
    "uniform-row-major-non-square": (
        "float3 t = mul(m, v.xy);\nfloat2 u = mul(v.xyz, m);\n"
        "outp[i] = fin[i] * (t.x + t.y * 2.0 + t.z * 4.0 + u.y * 8.0 + m[2][1] * v.w);"
    ),
    "uniform-row-major-array": (
        "int k = iin[i] & 1;\nfloat2 t = mul(m[k], float2(1.0, -0.5));\nfloat2 u = mul(float2(0.5, 2.0), m[1]);\n"
        "outp[i] = fin[i] * (t.x + t.y * 2.0 + u.x * 4.0 - u.y + m[0][0][1] * 8.0);"
    ),
}
_MEMBERS = {
    "uniform-row-major-matrix": ("$mat3 m;", "$vec4 v;"),
    "uniform-row-major-non-square": ("$mat2x3 m;", "$vec4 v;"),
    "uniform-row-major-array": ("$mat2 m[2];",),
}
_HLSL_MEMBERS = {
    "uniform-row-major-matrix": ("float3x3 m;", "float4 v;"),
    "uniform-row-major-non-square": ("float3x2 m;", "float4 v;"),
    "uniform-row-major-array": ("float2x2 m[2];",),
}


def _members(members: tuple[str, ...], qualifier: str) -> str:
    """The block's members, the matrix first and declared with ``qualifier``."""
    return "\n".join(
        f"    {qualifier}{member}" if n == 0 else f"    {member}"
        for n, member in enumerate(members)
    )


def _glsl(head: str) -> dict[str, Template]:
    return {
        name: Template(
            body.format(m="", v=""),
            f"{head}\n{_members(_MEMBERS[name], 'layout(row_major) ')}\n}};",
        )
        for name, body in _C_BODIES.items()
    }


GLSL = _glsl("layout(std140, binding = 3) uniform Params {")
WEBGL = _glsl("layout(std140) uniform Params {")
FAMILY = {
    name: Template(
        body,
        f"cbuffer Params : register(b3) {{\n{_members(_HLSL_MEMBERS[name], 'row_major ')}\n}};",
    )
    for name, body in _HLSL_BODIES.items()
}
CROSSGL = {
    name: Template(
        body.format(m="", v=""),
        prelude=f"cbuffer Params @binding(3) {{\n{_members(_MEMBERS[name], '@ row_major ')}\n}};",
    )
    for name, body in _C_BODIES.items()
}

# The rows each probe declares in place of the matrix, and the vector after them.
_ROWS = {
    "uniform-row-major-matrix": 3,
    "uniform-row-major-non-square": 3,
    "uniform-row-major-array": 4,
}
_METAL_BUILD = {
    "uniform-row-major-matrix": (
        "float3x3 m = transpose(float3x3(params.r[0].xyz, params.r[1].xyz, params.r[2].xyz));\n"
    ),
    "uniform-row-major-non-square": (
        "float2x3 m = transpose(float3x2(params.r[0].xy, params.r[1].xy, params.r[2].xy));\n"
    ),
    "uniform-row-major-array": (
        "float2x2 m[2] = {transpose(float2x2(params.r[0].xy, params.r[1].xy)), "
        "transpose(float2x2(params.r[2].xy, params.r[3].xy))};\n"
    ),
}
METAL = {
    name: Template(
        _METAL_BUILD[name] + body.format(m="", v="params."),
        "struct Params {\n    float4 r[%d];%s\n};"
        % (_ROWS[name], "" if name == "uniform-row-major-array" else "\n    float4 v;"),
        params="constant Params& params [[buffer(3)]]",
    )
    for name, body in _C_BODIES.items()
}

# WGSL infers the type of each local, and indexes the matrix array through a variable.
_WGSL_BODIES = {
    "uniform-row-major-matrix": (
        "let m = transpose(mat3x3<f32>(params.r[0].xyz, params.r[1].xyz, params.r[2].xyz));\n"
        "let t = m * params.v.xyz;\nlet u = params.v.xyz * m;\n"
        "outp[i] = fin[i] * (t.x + t.y * 2.0 + t.z * 4.0 + u.x * 8.0 + m[2][1] * params.v.w);"
    ),
    "uniform-row-major-non-square": (
        "let m = transpose(mat3x2<f32>(params.r[0].xy, params.r[1].xy, params.r[2].xy));\n"
        "let t = m * params.v.xy;\nlet u = params.v.xyz * m;\n"
        "outp[i] = fin[i] * (t.x + t.y * 2.0 + t.z * 4.0 + u.y * 8.0 + m[1][2] * params.v.w);"
    ),
    "uniform-row-major-array": (
        "var m = array<mat2x2<f32>, 2>(transpose(mat2x2<f32>(params.r[0].xy, params.r[1].xy)), "
        "transpose(mat2x2<f32>(params.r[2].xy, params.r[3].xy)));\n"
        "let k = iin[i] & 1;\nlet t = m[k] * vec2<f32>(1.0, -0.5);\nlet u = vec2<f32>(0.5, 2.0) * m[1];\n"
        "outp[i] = fin[i] * (t.x + t.y * 2.0 + u.x * 4.0 - u.y + m[0][1][0] * 8.0);"
    ),
}
WGSL = {
    name: Template(
        body,
        "struct Params {\n    r: array<vec4<f32>, %d>,%s\n}\n@group(0) @binding(3) var<uniform> params: Params;"
        % (
            _ROWS[name],
            "" if name == "uniform-row-major-array" else "\n    v: vec4<f32>,",
        ),
    )
    for name, body in _WGSL_BODIES.items()
}

CUDA = OPENCL = MOJO = dict.fromkeys(NAMES, ABSENT)

_RUST_DERIVE = "#[derive(Clone, Copy, Default)]\n#[repr(C)]\n"
_RUST_UNIFORM = "#[spirv(uniform, descriptor_set = 0, binding = 3)] params: &Params"


def _rust_mat2(first: int) -> str:
    """The 2x2 matrix whose rows are the first two components of ``params.r[first]`` and ``params.r[first + 1]``."""
    columns = ", ".join(
        f"Vec2::new(params.r[{row}].x, params.r[{row}].y)" for row in (first, first + 1)
    )
    return f"Mat2::from_cols({columns}).transpose()"


# glam builds a matrix from its columns, names them x_axis, y_axis and z_axis, and multiplies only by column vectors.
RUST = {
    "uniform-row-major-matrix": Template(
        "let m = Mat3::from_cols(params.r[0].truncate(), params.r[1].truncate(), params.r[2].truncate())"
        ".transpose();\n"
        "let t = m * params.v.truncate();\nlet u = m.transpose() * params.v.truncate();\n"
        "outp[i] = fin[i] * (t.x + t.y * 2.0 + t.z * 4.0 + u.x * 8.0 + m.z_axis.y * params.v.w);",
        f"{_RUST_DERIVE}pub struct Params {{\n    r: [Vec4; 3],\n    v: Vec4,\n}}",
        params=_RUST_UNIFORM,
    ),
    "uniform-row-major-non-square": ABSENT,
    "uniform-row-major-array": Template(
        f"let m = [{_rust_mat2(0)}, {_rust_mat2(2)}];\n"
        "let k = (iin[i] & 1) as usize;\nlet t = m[k] * Vec2::new(1.0, -0.5);\n"
        "let u = m[1].transpose() * Vec2::new(0.5, 2.0);\n"
        "outp[i] = fin[i] * (t.x + t.y * 2.0 + u.x * 4.0 - u.y + m[0].y_axis.x * 8.0);",
        f"{_RUST_DERIVE}pub struct Params {{\n    r: [Vec4; 4],\n}}",
        params=_RUST_UNIFORM,
    ),
}

"""Buffer-row-major-matrix, buffer-row-major-non-square and buffer-row-major-array probes: output buffers whose
elements are structs with a 2x2 matrix, a two-column, three-row matrix or an array of two 2x2 matrices stored by
rows, followed by a two-component vector, in every language that has matrices and storage buffers.

Each probe stores the matrix whole, changes one element in place, and reads the matrix whole and by columns. In
std430 and a structured buffer a row-major matrix keeps each row as a vector of its columns' count, so a row-major
``mat2x3`` takes 24 bytes and puts the vector at offset 24, where its column-major form takes 32 and puts it at 32.
A translation has to keep the orientation for the host to read what the native probe writes. CrossGL marks the member
``@ row_major`` and HLSL and Slang ``row_major``. GLSL allows no layout qualifier on a structure member, so its probe
declares the buffer's array ``layout(row_major)``, which applies to the matrices of its structs. HLSL fills and
indexes a matrix by rows, so its probes write the transposed constructor and read a column through ``transpose``.
Metal, WGSL and Rust have no row-major matrices, so their structs declare the rows as vectors, and their probes store
the rows of a matrix's transpose and rebuild the matrix as the transpose of the one whose columns are the rows.
WebGL 2 has no storage buffers, CUDA, HIP, OpenCL C and Mojo no matrices, and glam no non-square ones.
"""

from __future__ import annotations

from .base import ABSENT, Template

NAMES = (
    "buffer-row-major-matrix",
    "buffer-row-major-non-square",
    "buffer-row-major-array",
)

_C_BODIES = {
    "buffer-row-major-matrix": (
        "outp[i].m = $mat2(fin[i], 1.0, 2.0, fin[i] * 0.5);\noutp[i].m[0][1] += 0.25;\n$mat2 c = outp[i].m;\n"
        "outp[i].w = outp[i].m[1] * 2.0 + c[0] * 4.0;"
    ),
    "buffer-row-major-non-square": (
        "outp[i].m = $mat2x3(fin[i], 1.0, 2.0, 3.0, fin[i] * 0.5, 4.0);\noutp[i].m[1][2] += 0.25;\n"
        "$mat2x3 c = outp[i].m;\noutp[i].w = $vec2(outp[i].m[0][2], c[1][1]) * 2.0 + c[1].xy;"
    ),
    "buffer-row-major-array": (
        "int k = iin[i] & 1;\noutp[i].m[0] = $mat2(fin[i], 1.0, 2.0, 3.0);\n"
        "outp[i].m[1] = $mat2(4.0, fin[i] * 0.5, 5.0, 6.0);\noutp[i].m[k][1][0] += 0.25;\n"
        "$mat2 c = outp[i].m[1 - k];\noutp[i].w = outp[i].m[k][0] * 2.0 + c[1];"
    ),
}
# CrossGL's element in column c and row r is HLSL's m[r][c], and a CrossGL column is a row of HLSL's transpose.
_HLSL_BODIES = {
    "buffer-row-major-matrix": (
        "outp[i].m = float2x2(fin[i], 2.0, 1.0, fin[i] * 0.5);\noutp[i].m[1][0] += 0.25;\nfloat2x2 c = outp[i].m;\n"
        "outp[i].w = transpose(outp[i].m)[1] * 2.0 + transpose(c)[0] * 4.0;"
    ),
    "buffer-row-major-non-square": (
        "outp[i].m = float3x2(fin[i], 3.0, 1.0, fin[i] * 0.5, 2.0, 4.0);\noutp[i].m[2][1] += 0.25;\n"
        "float3x2 c = outp[i].m;\noutp[i].w = float2(outp[i].m[2][0], c[1][1]) * 2.0 + transpose(c)[1].xy;"
    ),
    "buffer-row-major-array": (
        "int k = iin[i] & 1;\noutp[i].m[0] = float2x2(fin[i], 2.0, 1.0, 3.0);\n"
        "outp[i].m[1] = float2x2(4.0, 5.0, fin[i] * 0.5, 6.0);\noutp[i].m[k][0][1] += 0.25;\n"
        "float2x2 c = outp[i].m[1 - k];\noutp[i].w = transpose(outp[i].m[k])[0] * 2.0 + transpose(c)[1];"
    ),
}
_MEMBERS = {
    "buffer-row-major-matrix": "$mat2 m;",
    "buffer-row-major-non-square": "$mat2x3 m;",
    "buffer-row-major-array": "$mat2 m[2];",
}
_HLSL_MEMBERS = {
    "buffer-row-major-matrix": "float2x2 m;",
    "buffer-row-major-non-square": "float3x2 m;",
    "buffer-row-major-array": "float2x2 m[2];",
}


def _rows(member: str, vector: str = "$vec2 w;") -> str:
    return f"struct Rows {{\n    {member}\n    {vector}\n}};"


GLSL = {
    name: Template(body, prelude=_rows(_MEMBERS[name]), output="layout(row_major) Rows")
    for name, body in _C_BODIES.items()
}
CROSSGL = {
    name: Template(body, prelude=_rows(f"@ row_major {_MEMBERS[name]}"), output="Rows")
    for name, body in _C_BODIES.items()
}
FAMILY = {
    name: Template(
        body,
        prelude=_rows(f"row_major {_HLSL_MEMBERS[name]}", "float2 w;"),
        output="Rows",
    )
    for name, body in _HLSL_BODIES.items()
}

# The rows each struct declares in place of the matrix: r[row], or r[matrix][row] for the array.
_METAL_ROWS = {
    "buffer-row-major-matrix": "float2 r[2];",
    "buffer-row-major-non-square": "float2 r[3];",
    "buffer-row-major-array": "float2 r[2][2];",
}
_METAL_BODIES = {
    "buffer-row-major-matrix": (
        "float2x2 t = transpose(float2x2(fin[i], 1.0, 2.0, fin[i] * 0.5));\n"
        "outp[i].r[0] = t[0];\noutp[i].r[1] = t[1];\noutp[i].r[1][0] += 0.25;\n"
        "float2x2 c = transpose(float2x2(outp[i].r[0], outp[i].r[1]));\n"
        "outp[i].w = c[1] * 2.0 + c[0] * 4.0;"
    ),
    "buffer-row-major-non-square": (
        "float3x2 t = transpose(float2x3(fin[i], 1.0, 2.0, 3.0, fin[i] * 0.5, 4.0));\n"
        "outp[i].r[0] = t[0];\noutp[i].r[1] = t[1];\noutp[i].r[2] = t[2];\noutp[i].r[2][1] += 0.25;\n"
        "float2x3 c = transpose(float3x2(outp[i].r[0], outp[i].r[1], outp[i].r[2]));\n"
        "outp[i].w = float2(c[0][2], c[1][1]) * 2.0 + c[1].xy;"
    ),
    "buffer-row-major-array": (
        "int k = iin[i] & 1;\nfloat2x2 t0 = transpose(float2x2(fin[i], 1.0, 2.0, 3.0));\n"
        "float2x2 t1 = transpose(float2x2(4.0, fin[i] * 0.5, 5.0, 6.0));\n"
        "outp[i].r[0][0] = t0[0];\noutp[i].r[0][1] = t0[1];\noutp[i].r[1][0] = t1[0];\noutp[i].r[1][1] = t1[1];\n"
        "outp[i].r[k][0][1] += 0.25;\n"
        "float2x2 a = transpose(float2x2(outp[i].r[k][0], outp[i].r[k][1]));\n"
        "float2x2 c = transpose(float2x2(outp[i].r[1 - k][0], outp[i].r[1 - k][1]));\n"
        "outp[i].w = a[0] * 2.0 + c[1];"
    ),
}
METAL = {
    name: Template(body, prelude=_rows(_METAL_ROWS[name], "float2 w;"), output="Rows")
    for name, body in _METAL_BODIES.items()
}

_WGSL_ROWS = {
    "buffer-row-major-matrix": "array<vec2<f32>, 2>",
    "buffer-row-major-non-square": "array<vec2<f32>, 3>",
    "buffer-row-major-array": "array<array<vec2<f32>, 2>, 2>",
}
# WGSL infers the type of each local.
_WGSL_BODIES = {
    name: (
        body.replace("float2x2 ", "let ")
        .replace("float3x2 t", "let t")
        .replace("float2x3 c", "let c")
        .replace("int k", "let k")
        .replace("float2x2(", "mat2x2<f32>(")
        .replace("float2x3(", "mat2x3<f32>(")
        .replace("float3x2(", "mat3x2<f32>(")
        .replace("float2(", "vec2<f32>(")
    )
    for name, body in _METAL_BODIES.items()
}
WGSL = {
    name: Template(
        body,
        f"struct Rows {{\n    r: {_WGSL_ROWS[name]},\n    w: vec2<f32>,\n}}",
        output="Rows",
    )
    for name, body in _WGSL_BODIES.items()
}

WEBGL = CUDA = OPENCL = MOJO = dict.fromkeys(NAMES, ABSENT)

_RUST_DERIVE = "#[derive(Clone, Copy, Default)]\n#[repr(C)]\n"


def _rust_rows(rows: str) -> str:
    return f"{_RUST_DERIVE}pub struct Rows {{\n    r: {rows},\n    w: Vec2,\n}}"


# glam builds a matrix from its columns and names them x_axis and y_axis.
RUST = {
    "buffer-row-major-matrix": Template(
        "let t = Mat2::from_cols(Vec2::new(fin[i], 1.0), Vec2::new(2.0, fin[i] * 0.5)).transpose();\n"
        "outp[i].r[0] = t.x_axis;\noutp[i].r[1] = t.y_axis;\noutp[i].r[1].x += 0.25;\n"
        "let c = Mat2::from_cols(outp[i].r[0], outp[i].r[1]).transpose();\n"
        "outp[i].w = c.y_axis * 2.0 + c.x_axis * 4.0;",
        _rust_rows("[Vec2; 2]"),
        output="Rows",
    ),
    "buffer-row-major-non-square": ABSENT,
    "buffer-row-major-array": Template(
        "let k = (iin[i] & 1) as usize;\n"
        "let t0 = Mat2::from_cols(Vec2::new(fin[i], 1.0), Vec2::new(2.0, 3.0)).transpose();\n"
        "let t1 = Mat2::from_cols(Vec2::new(4.0, fin[i] * 0.5), Vec2::new(5.0, 6.0)).transpose();\n"
        "outp[i].r[0][0] = t0.x_axis;\noutp[i].r[0][1] = t0.y_axis;\n"
        "outp[i].r[1][0] = t1.x_axis;\noutp[i].r[1][1] = t1.y_axis;\noutp[i].r[k][0].y += 0.25;\n"
        "let a = Mat2::from_cols(outp[i].r[k][0], outp[i].r[k][1]).transpose();\n"
        "let c = Mat2::from_cols(outp[i].r[1 - k][0], outp[i].r[1 - k][1]).transpose();\n"
        "outp[i].w = a.x_axis * 2.0 + c.y_axis;",
        _rust_rows("[[Vec2; 2]; 2]"),
        output="Rows",
    ),
}

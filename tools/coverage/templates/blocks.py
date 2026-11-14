"""Uniform-array, uniform-vector-array and uniform-matrix probes: constants blocks that hold an array of floats, an
array of four-component vectors or a 4x4 matrix, read with a dynamic index or a matrix product, in every language.

The harness binds the same 64 bytes to each probe, so a translation has to keep the block's strides and the matrix's
orientation for both probes to read the same words. std140, a cbuffer and CrossGL's constant layout place each
element of a float array on its own 16-byte row. Metal, C, Rust and Mojo pack a float array, and WGSL refuses one in a
uniform block, so their probes declare an array of four-component vectors and read the first components: the same
bytes. Every language stores ``vec4[2]`` with a 16-byte stride and a 4x4 matrix by columns; HLSL indexes a matrix by
rows and multiplies with ``mul``, so its probe swaps the indices. CUDA, HIP, OpenCL C and Mojo have no matrices.

The block's words have few significant bits, so each probe combines them exactly and multiplies its input once: no
contraction or reordering a compiler applies changes the result.
"""

from __future__ import annotations

import re
from dataclasses import replace

from .base import ABSENT, Template
from .statements import c_text

NAMES = ("uniform-array", "uniform-vector-array", "uniform-matrix")

# {0} is the block's prefix, {1} what reads an element's float: nothing, or its first component.
_ARRAY = (
    "int k = iin[i] & 3;\n"
    "outp[i] = fin[i] * ({0}w[0]{1} + {0}w[1]{1} - {0}w[2]{1} * {0}w[3]{1} + {0}w[k]{1});"
)
_VECTORS = (
    "int k = iin[i] & 1;\n$vec4 a = {0}v[k];\n$vec4 b = {0}v[1 - k];\n"
    "outp[i] = fin[i] * (a.x + a.y * 2.0 + a.z * 4.0 + a.w * 8.0 + b.x - b.w);"
)
_MATRIX = (
    "$vec4 t = {0}m * $vec4(1.0, -0.5, 2.0, 0.25);\n"
    "outp[i] = fin[i] * (t.x + t.y * 2.0 + t.z * 4.0 + t.w * 8.0 + {0}m[1][2]);"
)
_HLSL_MATRIX = (
    "float4 t = mul(m, float4(1.0, -0.5, 2.0, 0.25));\n"
    "outp[i] = fin[i] * (t.x + t.y * 2.0 + t.z * 4.0 + t.w * 8.0 + m[2][1]);"
)

# The members of each block where a float array element takes a 16-byte row, and where it takes four bytes.
_PADDED = {
    "uniform-array": "    float w[4];",
    "uniform-vector-array": "    $vec4 v[2];",
    "uniform-matrix": "    $mat4 m;",
}
_PACKED = {**_PADDED, "uniform-array": "    $vec4 w[4];"}


def _bodies(prefix: str, matrix: str = _MATRIX, element: str = ".x") -> dict[str, str]:
    return {
        "uniform-array": _ARRAY.format(prefix, element),
        "uniform-vector-array": _VECTORS.format(prefix),
        "uniform-matrix": matrix.format(prefix),
    }


def _blocks(head: str, tail: str, matrix: str = _MATRIX) -> dict[str, Template]:
    """The probes of a language with std140 or cbuffer packing that declares the block ``head`` members ``tail``."""
    bodies = _bodies("", matrix, "")
    return {
        name: Template(bodies[name], f"{head}\n{members}\n{tail}")
        for name, members in _PADDED.items()
    }


GLSL = _blocks("layout(std140, binding = 3) uniform Params {", "};")
WEBGL = _blocks("layout(std140) uniform Params {", "};")
FAMILY = _blocks("cbuffer Params : register(b3) {", "};", _HLSL_MATRIX)

# CrossGL declares the constants block at shader scope.
_CROSSGL_BODIES = _bodies("", element="")
CROSSGL = {
    name: Template(
        _CROSSGL_BODIES[name], prelude=f"cbuffer Params @binding(3) {{\n{members}\n}};"
    )
    for name, members in _PADDED.items()
}

_METAL_BODIES = _bodies("params.")
METAL = {
    name: Template(
        _METAL_BODIES[name],
        f"struct Params {{\n{members}\n}};",
        params="constant Params& params [[buffer(3)]]",
    )
    for name, members in _PACKED.items()
}

_WGSL_MEMBERS = {
    "uniform-array": "    w: array<vec4<f32>, 4>,",
    "uniform-vector-array": "    v: array<vec4<f32>, 2>,",
    "uniform-matrix": "    m: mat4x4<f32>,",
}
# WGSL infers the type of each local.
_WGSL_BODIES = {
    name: re.sub(r"(?:int|\$vec4) (\w+) =", r"let \1 =", body)
    for name, body in _bodies("params.").items()
}

WGSL = {
    name: Template(
        _WGSL_BODIES[name],
        f"struct Params {{\n{members}\n}}\n@group(0) @binding(3) var<uniform> params: Params;",
    )
    for name, members in _WGSL_MEMBERS.items()
}


def _c(vector: str, block: str) -> dict[str, Template]:
    """The probes CUDA, HIP and OpenCL C share: ``vector`` spells ``float4`` and ``block`` formats the declaration of
    the struct ``Params`` from its members."""
    bodies = _bodies("params.")
    return {
        "uniform-array": Template(
            c_text(bodies["uniform-array"], "uint"),
            block.format(_PACKED["uniform-array"].replace("$vec4", vector)),
        ),
        "uniform-vector-array": Template(
            c_text(bodies["uniform-vector-array"], "uint"),
            block.format(_PADDED["uniform-vector-array"].replace("$vec4", vector)),
        ),
        "uniform-matrix": ABSENT,
    }


CUDA = _c("float4", "struct Params {{\n{}\n}};\n__constant__ Params params;")

OPENCL = {
    name: (
        replace(template, params="const Params params")
        if template is not ABSENT
        else template
    )
    for name, template in _c("float4", "typedef struct {{\n{}\n}} Params;").items()
}

_RUST_DERIVE = "#[derive(Clone, Copy, Default)]\n#[repr(C)]\n"
_RUST_UNIFORM = "#[spirv(uniform, descriptor_set = 0, binding = 3)] params: &Params"
_RUST_BODIES = {
    "uniform-array": (
        "let k = (iin[i] & 3) as usize;\n"
        + _ARRAY.format("params.", ".x").split("\n", 1)[1]
    ),
    "uniform-vector-array": (
        "let k = (iin[i] & 1) as usize;\nlet a = params.v[k];\nlet b = params.v[1 - k];\n"
        + _VECTORS.split("\n")[-1]
    ),
    "uniform-matrix": (
        "let t = params.m * $vec4::new(1.0, -0.5, 2.0, 0.25);\n"
        "outp[i] = fin[i] * (t.x + t.y * 2.0 + t.z * 4.0 + t.w * 8.0 + params.m.y_axis.z);"
    ),
}
_RUST_MEMBERS = {
    "uniform-array": "    w: [Vec4; 4],",
    "uniform-vector-array": "    v: [Vec4; 2],",
    "uniform-matrix": "    m: Mat4,",
}

RUST = {
    name: Template(
        _RUST_BODIES[name],
        f"{_RUST_DERIVE}pub struct Params {{\n{members}\n}}",
        params=_RUST_UNIFORM,
    )
    for name, members in _RUST_MEMBERS.items()
}

_MOJO_PARAMS = "@fieldwise_init\nstruct Params(Copyable, Movable):\n    var {}: Array[SIMD[DType.float32, 4], {}]"


MOJO = {
    "uniform-array": Template(
        "var k = Int(iin[i] & 3)\noutp[i] = fin[i] * (params.w[0][0] + params.w[1][0] - params.w[2][0] * "
        "params.w[3][0] + params.w[k][0])",
        _MOJO_PARAMS.format("w", 4),
        params="params: Params",
    ),
    "uniform-vector-array": Template(
        "var k = Int(iin[i] & 1)\nvar a = params.v[k]\nvar b = params.v[1 - k]\n"
        "outp[i] = fin[i] * (a[0] + a[1] * 2.0 + a[2] * 4.0 + a[3] * 8.0 + b[0] - b[3])",
        _MOJO_PARAMS.format("v", 2),
        params="params: Params",
    ),
    "uniform-matrix": ABSENT,
}

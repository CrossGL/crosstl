"""Uniform-std140-vector, uniform-cbuffer-vector and uniform-std140-array probes: constants blocks that std140 and a
cbuffer lay out differently, each declared with the rule it follows, in every language.

std140 aligns a three-component vector to 16 bytes and rounds an array of floats up to whole 16-byte rows, while a
cbuffer packs a vector into the row it starts in unless it would cross into the next one, and does not pad an array
after its last element. So ``float a; vec3 b; float c; float d; vec2 g;`` puts ``b``, ``c``, ``d`` and ``g`` at 16, 28,
32 and 40 under std140 and at 4, 16, 20 and 24 in a cbuffer, and ``float e[2]; float d;`` puts ``d`` at 32 under
std140. CrossGL marks the rule with ``@buffer_layout(std140)`` or ``@buffer_layout(hlsl_cbuffer)``, and a translation
has to keep each member at its offset for both probes to read the same words.

Every native probe declares the same offsets. HLSL and Slang name each member's ``packoffset``, since slangc lays out
a plain cbuffer by std140 for SPIR-V. GLSL, WebGL and WGSL have no rule that puts a vector at offset 4, so their
cbuffer probes declare the vector's components as floats; WGSL keeps its own uniform layout for the std140 vector
block, which matches. Metal, C, C++, Rust and Mojo declare std140's padding as float members, Metal stores the vector
as a ``packed_float3``, and OpenCL C and Mojo, whose three-component vectors take 16 bytes or do not exist, declare the
components as floats. An array element that takes a 16-byte row is a four-component vector read through its first
component. As in :mod:`.blocks`, each probe combines the block's words exactly and multiplies its input once.
"""

from __future__ import annotations

from dataclasses import replace

from .base import Template
from .statements import c_text

NAMES = ("uniform-std140-vector", "uniform-cbuffer-vector", "uniform-std140-array")

# {f} prefixes a scalar member and {b} the vector; {e} reads an array element's float.
_VECTOR = (
    "outp[i] = fin[i] * ({f}a + {b}b.x * 2.0 + {b}b.y * 4.0 + {b}b.z * 8.0 + {f}c * 16.0 - {f}d"
    " + {f}g.x * 0.5 + {f}g.y * 0.25);"
)
_ARRAY = "outp[i] = fin[i] * ({f}e[0]{e} + {f}e[1]{e} * 2.0 - {f}d * 4.0);"


def _bodies(f: str = "", b: str | None = None, e: str = "") -> dict[str, str]:
    vector = _VECTOR.format(f=f, b=f if b is None else b)
    return {
        "uniform-std140-vector": vector,
        "uniform-cbuffer-vector": vector,
        "uniform-std140-array": _ARRAY.format(f=f, e=e),
    }


_VECTOR_MEMBERS = "    float a;\n    $vec3 b;\n    float c;\n    float d;\n    $vec2 g;"
_SPLIT_MEMBERS = "    float a;\n    float b0;\n    float b1;\n    float b2;\n    float c;\n    float d;\n    $vec2 g;"
_ARRAY_MEMBERS = "    float e[2];\n    float d;"
# A local that builds the vector from the floats a cbuffer probe declares.
_GLSL_SPLIT = "$vec3 b = $vec3(b0, b1, b2);\n"


def _glsl(head: str) -> dict[str, Template]:
    members = {
        "uniform-std140-vector": _VECTOR_MEMBERS,
        "uniform-cbuffer-vector": _SPLIT_MEMBERS,
        "uniform-std140-array": _ARRAY_MEMBERS,
    }
    bodies = _bodies()
    bodies["uniform-cbuffer-vector"] = _GLSL_SPLIT + bodies["uniform-cbuffer-vector"]
    return {
        name: Template(bodies[name], f"{head}\n{members[name]}\n}};") for name in NAMES
    }


GLSL = _glsl("layout(std140, binding = 3) uniform Params {")
WEBGL = _glsl("layout(std140) uniform Params {")

_PACKOFFSETS = {
    "uniform-std140-vector": (
        "    float a : packoffset(c0);\n    float3 b : packoffset(c1);\n    float c : packoffset(c1.w);\n"
        "    float d : packoffset(c2);\n    float2 g : packoffset(c2.z);"
    ),
    "uniform-cbuffer-vector": (
        "    float a : packoffset(c0);\n    float3 b : packoffset(c0.y);\n    float c : packoffset(c1);\n"
        "    float d : packoffset(c1.y);\n    float2 g : packoffset(c1.z);"
    ),
    "uniform-std140-array": (
        "    float e[2] : packoffset(c0);\n    float d : packoffset(c2);"
    ),
}
FAMILY = {
    name: Template(body, f"cbuffer Params : register(b3) {{\n{_PACKOFFSETS[name]}\n}};")
    for name, body in _bodies().items()
}

_RULES = {
    "uniform-std140-vector": ("std140", _VECTOR_MEMBERS),
    "uniform-cbuffer-vector": ("hlsl_cbuffer", _VECTOR_MEMBERS),
    "uniform-std140-array": ("std140", _ARRAY_MEMBERS),
}
CROSSGL = {
    name: Template(
        _bodies()[name],
        prelude=f"cbuffer Params @binding(3) @buffer_layout({rule}) {{\n{members}\n}};",
    )
    for name, (rule, members) in _RULES.items()
}

# std140's padding as float members, in C-family declaration syntax.
_PAD = "    float pad0;\n    float pad1;\n    float pad2;\n"
_METAL_MEMBERS = {
    "uniform-std140-vector": (
        f"    float a;\n{_PAD}    packed_float3 b;\n    float c;\n    float d;\n    float pad3;\n    float2 g;"
    ),
    "uniform-cbuffer-vector": (
        "    float a;\n    packed_float3 b;\n    float c;\n    float d;\n    float2 g;"
    ),
    "uniform-std140-array": "    float4 e[2];\n    float d;",
}
METAL = {
    name: Template(
        body,
        f"struct Params {{\n{_METAL_MEMBERS[name]}\n}};",
        params="constant Params& params [[buffer(3)]]",
    )
    for name, body in _bodies("params.", e=".x").items()
}

_WGSL_MEMBERS = {
    "uniform-std140-vector": (
        "    a: f32,\n    b: vec3<f32>,\n    c: f32,\n    d: f32,\n    g: vec2<f32>,"
    ),
    "uniform-cbuffer-vector": (
        "    a: f32,\n    b0: f32,\n    b1: f32,\n    b2: f32,\n    c: f32,\n    d: f32,\n    g: vec2<f32>,"
    ),
    "uniform-std140-array": "    e: array<vec4<f32>, 2>,\n    d: f32,",
}
_WGSL_BODIES = _bodies("params.", e=".x")
_WGSL_BODIES["uniform-cbuffer-vector"] = (
    "let b = vec3<f32>(params.b0, params.b1, params.b2);\n"
    + _bodies("params.", "")["uniform-cbuffer-vector"]
)
WGSL = {
    name: Template(
        body,
        f"struct Params {{\n{_WGSL_MEMBERS[name]}\n}}\n@group(0) @binding(3) var<uniform> params: Params;",
    )
    for name, body in _WGSL_BODIES.items()
}

_CUDA_MEMBERS = {
    "uniform-std140-vector": (
        f"    float a;\n{_PAD}    float3 b;\n    float c;\n    float d;\n    float pad3;\n    float2 g;"
    ),
    "uniform-cbuffer-vector": (
        "    float a;\n    float3 b;\n    float c;\n    float d;\n    float2 g;"
    ),
    "uniform-std140-array": "    float4 e[2];\n    float d;",
}
CUDA = {
    name: Template(
        c_text(body, "unsigned int"),
        f"struct Params {{\n{_CUDA_MEMBERS[name]}\n}};\n__constant__ Params params;",
    )
    for name, body in _bodies("params.", e=".x").items()
}

# OpenCL C sizes a float3 as 16 bytes, so its probes declare the vector's components and build it in the kernel.
_OPENCL_SPLIT = "    float b0;\n    float b1;\n    float b2;\n"
_OPENCL_MEMBERS = {
    "uniform-std140-vector": (
        f"    float a;\n{_PAD}{_OPENCL_SPLIT}    float c;\n    float d;\n    float pad3;\n    float2 g;"
    ),
    "uniform-cbuffer-vector": (
        f"    float a;\n{_OPENCL_SPLIT}    float c;\n    float d;\n    float2 g;"
    ),
    "uniform-std140-array": "    float4 e[2];\n    float d;",
}
_OPENCL_BUILD = "float3 b = (float3)(params.b0, params.b1, params.b2);\n"
OPENCL = {
    name: Template(
        c_text(body, "uint"),
        "typedef struct {\n" + _OPENCL_MEMBERS[name] + "\n} Params;",
        params="const Params params",
    )
    for name, body in _bodies("params.", "", ".x").items()
}
for _name in ("uniform-std140-vector", "uniform-cbuffer-vector"):
    OPENCL[_name] = replace(OPENCL[_name], body=_OPENCL_BUILD + OPENCL[_name].body)

_RUST_DERIVE = "#[derive(Clone, Copy, Default)]\n#[repr(C)]\n"
_RUST_UNIFORM = "#[spirv(uniform, descriptor_set = 0, binding = 3)] params: &Params"
_RUST_PAD = "    pad0: f32,\n    pad1: f32,\n    pad2: f32,\n"
_RUST_MEMBERS = {
    "uniform-std140-vector": (
        f"    a: f32,\n{_RUST_PAD}    b: Vec3,\n    c: f32,\n    d: f32,\n    pad3: f32,\n    g: Vec2,"
    ),
    "uniform-cbuffer-vector": (
        "    a: f32,\n    b: Vec3,\n    c: f32,\n    d: f32,\n    g: Vec2,"
    ),
    "uniform-std140-array": "    e: [Vec4; 2],\n    d: f32,",
}
RUST = {
    name: Template(
        body,
        f"{_RUST_DERIVE}pub struct Params {{\n{_RUST_MEMBERS[name]}\n}}",
        params=_RUST_UNIFORM,
    )
    for name, body in _bodies("params.", e=".x").items()
}

# Mojo has no three-component SIMD vectors, so its probes read the vector's components as floats.
_MOJO_VECTOR = (
    "outp[i] = fin[i] * (params.a + params.b0 * 2.0 + params.b1 * 4.0 + params.b2 * 8.0 + params.c * 16.0"
    " - params.d + params.g[0] * 0.5 + params.g[1] * 0.25)"
)
_MOJO_HEAD = "@fieldwise_init\nstruct Params(TrivialRegisterPassable):\n"
_MOJO_SPLIT = "    var b0: Float32\n    var b1: Float32\n    var b2: Float32\n"
_MOJO_PAD = "    var pad0: Float32\n    var pad1: Float32\n    var pad2: Float32\n"
MOJO = {
    "uniform-std140-vector": Template(
        _MOJO_VECTOR,
        _MOJO_HEAD
        + f"    var a: Float32\n{_MOJO_PAD}{_MOJO_SPLIT}    var c: Float32\n    var d: Float32\n"
        "    var pad3: Float32\n    var g: $vec2",
        params="params: Params",
    ),
    "uniform-cbuffer-vector": Template(
        _MOJO_VECTOR,
        _MOJO_HEAD
        + f"    var a: Float32\n{_MOJO_SPLIT}    var c: Float32\n    var d: Float32\n    var g: $vec2",
        params="params: Params",
    ),
    "uniform-std140-array": Template(
        "outp[i] = fin[i] * (params.e[0][0] + params.e[1][0] * 2.0 - params.d * 4.0)",
        "@fieldwise_init\nstruct Params(Copyable, Movable):\n    var e: Array[SIMD[DType.float32, 4], 2]\n"
        "    var d: Float32",
        params="params: Params",
    ),
}

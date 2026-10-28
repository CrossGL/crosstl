"""Uniform-bool and uniform-bool-vector probes: constants blocks whose members are booleans between floats, or a
four-component and a two-component boolean vector followed by a boolean and a float, in every language.

std140, a cbuffer and Slang store each boolean of a block in four bytes, nonzero meaning true, so every member below
starts at the same offset in each of them: the scalar block fills one 16-byte row, and the vector block puts its
two-component vector at 16, its boolean at 24 and its float at 28. Metal, C and C++ store a ``bool`` in one byte, and
WGSL, OpenCL C, Rust and Mojo cannot put a boolean in a block the host writes at all, so those probes declare each
boolean as a 32-bit unsigned integer and test it against zero. The harness fills the block with nonzero float words,
so each boolean the native probes read is true, while a translation that keeps a one-byte ``bool`` reads the low byte
of 2.5 or 3.25, which is zero. CUDA, HIP and OpenCL C have no boolean vectors. As in :mod:`.integers`, each probe
multiplies its input once.
"""

from __future__ import annotations

from .base import ABSENT, Template
from .statements import c_text

NAMES = ("uniform-bool", "uniform-bool-vector")

# {b} is the prefix that names a boolean member of the block, {f} the prefix that names a float member.
_BODIES = {
    "uniform-bool": (
        "outp[i] = fin[i] * ({b}b0 ? {f}x : {f}y) + ({b}b1 && !{b}b0 ? 1.0 : {f}y * 2.0) + ({b}b1 ? 4.0 : 0.0);"
    ),
    "uniform-bool-vector": (
        "$bvec4 c = {b}f;\nc.y = !{b}g.x;\n"
        "outp[i] = fin[i] * (c.x ? {f}s : 1.0) + (c.y ? 16.0 : 2.0) + (c.w && {b}g.y ? 4.0 : 0.0)"
        " + ({b}h ? {f}s * 8.0 : 0.0);"
    ),
}
# Each block's members in C-family declaration syntax.
_MEMBERS = {
    "uniform-bool": "    bool b0;\n    float x;\n    bool b1;\n    float y;",
    "uniform-bool-vector": "    $bvec4 f;\n    $bvec2 g;\n    bool h;\n    float s;",
}


def _blocks(head: str, tail: str) -> dict[str, Template]:
    """The probes of a language that declares the block ``head`` members ``tail`` at program scope."""
    return {
        name: Template(_BODIES[name].format(b="", f=""), f"{head}\n{members}\n{tail}")
        for name, members in _MEMBERS.items()
    }


GLSL = _blocks("layout(std140, binding = 3) uniform Params {", "};")
WEBGL = _blocks("layout(std140) uniform Params {", "};")
FAMILY = _blocks("cbuffer Params : register(b3) {", "};")
CROSSGL = {
    name: Template(
        _BODIES[name].format(b="", f=""),
        prelude=f"cbuffer Params @binding(3) {{\n{members}\n}};",
    )
    for name, members in _MEMBERS.items()
}

# Metal declares each boolean as a uint and converts it in the kernel, before the C-family body.
_METAL_MEMBERS = {
    "uniform-bool": "    uint b0;\n    float x;\n    uint b1;\n    float y;",
    "uniform-bool-vector": "    uint4 f;\n    uint2 g;\n    uint h;\n    float s;",
}
_METAL_LOCALS = {
    "uniform-bool": "bool b0 = bool(params.b0);\nbool b1 = bool(params.b1);\n",
    "uniform-bool-vector": (
        "bool4 f = bool4(params.f);\nbool2 g = bool2(params.g);\nbool h = bool(params.h);\n"
    ),
}
METAL = {
    name: Template(
        _METAL_LOCALS[name] + _BODIES[name].format(b="", f="params."),
        f"struct Params {{\n{_METAL_MEMBERS[name]}\n}};",
        params="constant Params& params [[buffer(3)]]",
    )
    for name in NAMES
}

_WGSL_MEMBERS = {
    "uniform-bool": "    b0: u32,\n    x: f32,\n    b1: u32,\n    y: f32,",
    "uniform-bool-vector": (
        "    f: vec4<u32>,\n    g: vec2<u32>,\n    h: u32,\n    s: f32,"
    ),
}
# WGSL converts a u32 to a bool by testing it against zero and chooses with select(false, true, condition).
_WGSL_BODIES = {
    "uniform-bool": (
        "let b0 = bool(params.b0);\nlet b1 = bool(params.b1);\n"
        "outp[i] = fin[i] * select(params.y, params.x, b0) + select(params.y * 2.0, 1.0, b1 && !b0)"
        " + select(0.0, 4.0, b1);"
    ),
    "uniform-bool-vector": (
        "var c = vec4<bool>(params.f);\nlet g = vec2<bool>(params.g);\nlet h = bool(params.h);\nc.y = !g.x;\n"
        "outp[i] = fin[i] * select(1.0, params.s, c.x) + select(2.0, 16.0, c.y) + select(0.0, 4.0, c.w && g.y)"
        " + select(0.0, params.s * 8.0, h);"
    ),
}
WGSL = {
    name: Template(
        body,
        f"struct Params {{\n{_WGSL_MEMBERS[name]}\n}}\n@group(0) @binding(3) var<uniform> params: Params;",
    )
    for name, body in _WGSL_BODIES.items()
}

# CUDA, HIP and OpenCL C test each uint member against zero; they have no boolean vectors.
_C_BOOL = (
    "bool b0 = params.b0 != 0u;\nbool b1 = params.b1 != 0u;\n"
    "outp[i] = fin[i] * (b0 ? params.x : params.y) + (b1 && !b0 ? 1.0 : params.y * 2.0) + (b1 ? 4.0 : 0.0);"
)
_C_MEMBERS = "    {0} b0;\n    float x;\n    {0} b1;\n    float y;"
CUDA = {
    "uniform-bool": Template(
        c_text(_C_BOOL, "unsigned int"),
        "struct Params {\n"
        + _C_MEMBERS.format("unsigned int")
        + "\n};\n__constant__ Params params;",
    ),
    "uniform-bool-vector": ABSENT,
}
OPENCL = {
    "uniform-bool": Template(
        c_text(_C_BOOL, "uint"),
        "typedef struct {\n" + _C_MEMBERS.format("uint") + "\n} Params;",
        params="const Params params",
    ),
    "uniform-bool-vector": ABSENT,
}

_RUST_DERIVE = "#[derive(Clone, Copy, Default)]\n#[repr(C)]\n"
_RUST_UNIFORM = "#[spirv(uniform, descriptor_set = 0, binding = 3)] params: &Params"
# glam compares a vector against zero component by component and names a BVec4's components x, y, z and w.
RUST = {
    "uniform-bool": Template(
        "let b0 = params.b0 != 0;\nlet b1 = params.b1 != 0;\n"
        "outp[i] = fin[i] * (if b0 { params.x } else { params.y })"
        " + (if b1 && !b0 { 1.0 } else { params.y * 2.0 }) + (if b1 { 4.0 } else { 0.0 });",
        f"{_RUST_DERIVE}pub struct Params {{\n    b0: u32,\n    x: f32,\n    b1: u32,\n    y: f32,\n}}",
        params=_RUST_UNIFORM,
    ),
    "uniform-bool-vector": Template(
        "let mut c = params.f.cmpne(UVec4::ZERO);\nlet g = params.g.cmpne(UVec2::ZERO);\nlet h = params.h != 0;\n"
        "c.y = !g.x;\n"
        "outp[i] = fin[i] * (if c.x { params.s } else { 1.0 }) + (if c.y { 16.0 } else { 2.0 })"
        " + (if c.w && g.y { 4.0 } else { 0.0 }) + (if h { params.s * 8.0 } else { 0.0 });",
        f"{_RUST_DERIVE}pub struct Params {{\n    f: UVec4,\n    g: UVec2,\n    h: u32,\n    s: f32,\n}}",
        params=_RUST_UNIFORM,
    ),
}

# Mojo casts a SIMD vector of integers to booleans lane by lane, nonzero meaning true.
_MOJO_HEAD = "@fieldwise_init\nstruct Params(TrivialRegisterPassable):\n"
MOJO = {
    "uniform-bool": Template(
        "var b0 = params.b0 != 0\nvar b1 = params.b1 != 0\n"
        "var r: Float32 = params.x if b0 else params.y\n"
        "var t: Float32 = 1.0 if b1 and not b0 else params.y * 2.0\n"
        "var u: Float32 = 4.0 if b1 else 0.0\n"
        "outp[i] = fin[i] * r + t + u",
        _MOJO_HEAD
        + "    var b0: UInt32\n    var x: Float32\n    var b1: UInt32\n    var y: Float32",
        params="params: Params",
    ),
    "uniform-bool-vector": Template(
        "var c = params.f.cast[DType.bool]()\nvar g = params.g.cast[DType.bool]()\nvar h = params.h != 0\n"
        "c[1] = not g[0]\n"
        "var r: Float32 = params.s if c[0] else 1.0\nvar t: Float32 = 16.0 if c[1] else 2.0\n"
        "var u: Float32 = 4.0 if c[3] and g[1] else 0.0\nvar v: Float32 = params.s * 8.0 if h else 0.0\n"
        "outp[i] = fin[i] * r + t + u + v",
        _MOJO_HEAD
        + "    var f: $uvec4\n    var g: $uvec2\n    var h: UInt32\n    var s: Float32",
        params="params: Params",
    ),
}

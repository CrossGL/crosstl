"""Uniform-integer, uniform-unsigned and uniform-integer-vector probes: constants blocks whose members are signed or
unsigned integers, as four scalars or as two four-component vectors, in every language.

The harness fills the block with float words, so each member holds a float's bit pattern: the first row's second
word, -1.75, is 0xBFE00000, which is -1075838976 as a signed integer and 3219128320 as an unsigned one. Every probe
shifts such a word right, so a translation that declares a member with the other signedness shifts in zeros or sign
bits where the native probe does not, and one that converts the words to integers instead of reinterpreting them
reads other values. The scalar members share one 16-byte row and the vectors fill one row each, so every language
lays the block out alike. As in :mod:`.records`, each probe multiplies its input once.
"""

from __future__ import annotations

import re

from .base import Template
from .statements import c_text

NAMES = ("uniform-integer", "uniform-unsigned", "uniform-integer-vector")

# {0} is the prefix that names a member of the block.
_BODIES = {
    "uniform-integer": (
        "outp[i] = fin[i] * float(({0}n0 >> 20) - ({0}n2 >> 22)) + float(({0}n1 >> 20) + {0}n3 / 1000000);"
    ),
    "uniform-unsigned": (
        "outp[i] = fin[i] * float(({0}u1 >> 20) - ({0}u0 >> 20)) + float({0}u2 % 1000u + ({0}u3 >> 24));"
    ),
    "uniform-integer-vector": (
        "$ivec4 v = {0}iv;\n$uvec4 u = {0}uv;\n"
        "outp[i] = fin[i] * float((v.x >> 20) + (v.y >> 20) + (v.w >> 24)) + float((u.y >> 24) - (u.z >> 24));"
    ),
}
# Each block's members in C-family declaration syntax.
_MEMBERS = {
    "uniform-integer": "    int n0;\n    int n1;\n    int n2;\n    int n3;",
    "uniform-unsigned": "    uint u0;\n    uint u1;\n    uint u2;\n    uint u3;",
    "uniform-integer-vector": "    $ivec4 iv;\n    $uvec4 uv;",
}


def _bodies(prefix: str) -> dict[str, str]:
    return {name: body.format(prefix) for name, body in _BODIES.items()}


def _blocks(head: str, tail: str) -> dict[str, Template]:
    """The probes of a language that declares the block ``head`` members ``tail`` at program scope."""
    bodies = _bodies("")
    return {
        name: Template(bodies[name], f"{head}\n{members}\n{tail}")
        for name, members in _MEMBERS.items()
    }


GLSL = _blocks("layout(std140, binding = 3) uniform Params {", "};")
WEBGL = _blocks("layout(std140) uniform Params {", "};")
FAMILY = _blocks("cbuffer Params : register(b3) {", "};")

_CROSSGL_BODIES = _bodies("")
CROSSGL = {
    name: Template(
        _CROSSGL_BODIES[name], prelude=f"cbuffer Params @binding(3) {{\n{members}\n}};"
    )
    for name, members in _MEMBERS.items()
}

_METAL_BODIES = _bodies("params.")
METAL = {
    name: Template(
        _METAL_BODIES[name],
        f"struct Params {{\n{members}\n}};",
        params="constant Params& params [[buffer(3)]]",
    )
    for name, members in _MEMBERS.items()
}

_WGSL_MEMBERS = {
    "uniform-integer": "    n0: i32,\n    n1: i32,\n    n2: i32,\n    n3: i32,",
    "uniform-unsigned": "    u0: u32,\n    u1: u32,\n    u2: u32,\n    u3: u32,",
    "uniform-integer-vector": "    iv: vec4<i32>,\n    uv: vec4<u32>,",
}
# WGSL infers the type of each local and converts with f32().
_WGSL_BODIES = {
    name: re.sub(r"\$[iu]vec4 (\w) =", r"let \1 =", body).replace("float(", "f32(")
    for name, body in _bodies("params.").items()
}
WGSL = {
    name: Template(
        _WGSL_BODIES[name],
        f"struct Params {{\n{members}\n}}\n@group(0) @binding(3) var<uniform> params: Params;",
    )
    for name, members in _WGSL_MEMBERS.items()
}


def _c(unsigned: str, block: str) -> dict[str, Template]:
    """The probes CUDA, HIP and OpenCL C share: ``block`` formats the declaration of ``Params`` from its members."""
    bodies = _bodies("params.")
    return {
        name: Template(
            c_text(bodies[name], unsigned),
            block.format(re.sub(r"\buint\b", unsigned, members)),
        )
        for name, members in _MEMBERS.items()
    }


CUDA = _c("unsigned int", "struct Params {{\n{}\n}};\n__constant__ Params params;")
OPENCL = {
    name: Template(template.body, template.declarations, params="const Params params")
    for name, template in _c("uint", "typedef struct {{\n{}\n}} Params;").items()
}

_RUST_DERIVE = "#[derive(Clone, Copy, Default)]\n#[repr(C)]\n"
_RUST_MEMBERS = {
    "uniform-integer": "    n0: i32,\n    n1: i32,\n    n2: i32,\n    n3: i32,",
    "uniform-unsigned": "    u0: u32,\n    u1: u32,\n    u2: u32,\n    u3: u32,",
    "uniform-integer-vector": "    iv: $ivec4,\n    uv: $uvec4,",
}


def _cast(body: str, convert: str) -> str:
    """``body`` with each ``float(...)`` conversion written by ``convert``, which formats the operand."""
    return re.sub(
        r"float\(((?:[^()]|\((?:[^()]|\([^()]*\))*\))*)\)",
        lambda m: convert.format(m.group(1)),
        body,
    )


RUST = {
    name: Template(
        re.sub(r"\$[iu]vec4 (\w) =", r"let \1 =", _cast(body, "({}) as f32")).replace(
            "1000u", "1000"
        ),
        f"{_RUST_DERIVE}pub struct Params {{\n{_RUST_MEMBERS[name]}\n}}",
        params="#[spirv(uniform, descriptor_set = 0, binding = 3)] params: &Params",
    )
    for name, body in _bodies("params.").items()
}

_MOJO_MEMBERS = {
    "uniform-integer": (
        "    var n0: Int32\n    var n1: Int32\n    var n2: Int32\n    var n3: Int32"
    ),
    "uniform-unsigned": (
        "    var u0: UInt32\n    var u1: UInt32\n    var u2: UInt32\n    var u3: UInt32"
    ),
    "uniform-integer-vector": "    var iv: $ivec4\n    var uv: $uvec4",
}


def _mojo(body: str) -> str:
    """Mojo reads a SIMD vector's lanes by index, divides integers with //, and declares each local with var."""
    body = re.sub(r"\.([xyzw])\b", lambda m: f"[{'xyzw'.index(m.group(1))}]", body)
    body = re.sub(r"\$[iu]vec4 (\w) =", r"var \1 =", _cast(body, "Float32({})"))
    return body.replace(" / ", " // ").replace("1000u", "1000").replace(";", "")


MOJO = {
    name: Template(
        _mojo(body),
        "@fieldwise_init\nstruct Params(TrivialRegisterPassable):\n"
        + _MOJO_MEMBERS[name],
        params="params: Params",
    )
    for name, body in _bodies("params.").items()
}

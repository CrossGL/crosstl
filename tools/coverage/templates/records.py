"""Uniform-struct, uniform-struct-array and uniform-struct-copy probes: constants blocks whose members are structs,
read member by member, through a dynamic index, or copied into a local that the probe then changes, in every
language.

Each block holds ``Light`` structs of two four-component vectors next to a four-component vector or one another, so
every member starts a 16-byte row in every language's layout and a translation has to keep the struct's size and the
block's member order to read the words the native probe reads. uniform-struct-copy changes its copy and reads the
block's own member afterwards, so a translation that writes through the block or reads the copy where the block
member is named computes something else. As in :mod:`.blocks`, each probe combines the block's words exactly and
multiplies its input once.
"""

from __future__ import annotations

import re

from .base import Template
from .statements import c_text

NAMES = ("uniform-struct", "uniform-struct-array", "uniform-struct-copy")

# {0} is the prefix that names a member of the block.
_STRUCT = (
    "$vec4 c = {0}light.color;\n"
    "outp[i] = fin[i] * (c.x + c.y * 2.0 + {0}light.dir.z * 4.0 + {0}light.dir.w * 8.0 + {0}tint.x - {0}tint.w);"
)
_ARRAY = (
    "int k = iin[i] & 1;\n"
    "outp[i] = fin[i] * ({0}lights[k].color.x + {0}lights[1 - k].dir.y * 2.0 + {0}lights[k].dir.w * 4.0 + "
    "{0}lights[1].color.z * 8.0);"
)
_COPY = (
    "Light l = {0}light;\nl.color.x += {0}tint.y;\nl.dir.y *= 2.0;\n"
    "outp[i] = fin[i] * (l.color.x * 2.0 + l.dir.y * 4.0 + l.color.w - {0}light.color.x + {0}light.dir.y);"
)
_BODIES = {
    "uniform-struct": _STRUCT,
    "uniform-struct-array": _ARRAY,
    "uniform-struct-copy": _COPY,
}

_LIGHT = "struct Light {\n    $vec4 color;\n    $vec4 dir;\n};"
# Each block's members in C-family declaration syntax.
_MEMBERS = {
    "uniform-struct": "    Light light;\n    $vec4 tint;",
    "uniform-struct-array": "    Light lights[2];",
    "uniform-struct-copy": "    $vec4 tint;\n    Light light;",
}


def _bodies(prefix: str) -> dict[str, str]:
    return {name: body.format(prefix) for name, body in _BODIES.items()}


def _blocks(head: str, tail: str) -> dict[str, Template]:
    """The probes of a language that declares the block ``head`` members ``tail`` at program scope."""
    bodies = _bodies("")
    return {
        name: Template(bodies[name], f"{head}\n{members}\n{tail}", prelude=_LIGHT)
        for name, members in _MEMBERS.items()
    }


GLSL = _blocks("layout(std140, binding = 3) uniform Params {", "};")
WEBGL = _blocks("layout(std140) uniform Params {", "};")
FAMILY = _blocks("cbuffer Params : register(b3) {", "};")

_CROSSGL_BODIES = _bodies("")
CROSSGL = {
    name: Template(
        _CROSSGL_BODIES[name],
        prelude=f"{_LIGHT}\ncbuffer Params @binding(3) {{\n{members}\n}};",
    )
    for name, members in _MEMBERS.items()
}

_METAL_BODIES = _bodies("params.")
METAL = {
    name: Template(
        _METAL_BODIES[name],
        f"{_LIGHT}\nstruct Params {{\n{members}\n}};",
        params="constant Params& params [[buffer(3)]]",
    )
    for name, members in _MEMBERS.items()
}

_WGSL_LIGHT = "struct Light {\n    color: vec4<f32>,\n    dir: vec4<f32>,\n}"
_WGSL_MEMBERS = {
    "uniform-struct": "    light: Light,\n    tint: vec4<f32>,",
    "uniform-struct-array": "    lights: array<Light, 2>,",
    "uniform-struct-copy": "    tint: vec4<f32>,\n    light: Light,",
}
# WGSL infers the type of each local, and the copy the probe changes is a var.
_WGSL_BODIES = {
    name: (
        re.sub(r"(?:int|\$vec4) (\w) =", r"let \1 =", body).replace(
            "Light l =", "var l ="
        )
    )
    for name, body in _bodies("params.").items()
}
WGSL = {
    name: Template(
        _WGSL_BODIES[name],
        f"{_WGSL_LIGHT}\nstruct Params {{\n{members}\n}}\n"
        "@group(0) @binding(3) var<uniform> params: Params;",
    )
    for name, members in _WGSL_MEMBERS.items()
}


def _c(unsigned: str, block: str, light: str = _LIGHT) -> dict[str, Template]:
    """The probes CUDA, HIP and OpenCL C share: ``block`` formats the declaration of ``Params`` from its members."""
    bodies = _bodies("params.")
    return {
        name: Template(
            c_text(bodies[name], unsigned), f"{light}\n{block.format(members)}"
        )
        for name, members in _MEMBERS.items()
    }


CUDA = _c("unsigned int", "struct Params {{\n{}\n}};\n__constant__ Params params;")
OPENCL = {
    name: Template(template.body, template.declarations, params="const Params params")
    for name, template in _c(
        "uint",
        "typedef struct {{\n{}\n}} Params;",
        "typedef struct {\n    $vec4 color;\n    $vec4 dir;\n} Light;",
    ).items()
}

_RUST_DERIVE = "#[derive(Clone, Copy, Default)]\n#[repr(C)]\n"
_RUST_MEMBERS = {
    "uniform-struct": "    light: Light,\n    tint: Vec4,",
    "uniform-struct-array": "    lights: [Light; 2],",
    "uniform-struct-copy": "    tint: Vec4,\n    light: Light,",
}
_RUST_BODIES = {
    name: (
        body.replace("$vec4 c =", "let c =")
        .replace("int k = iin[i] & 1;", "let k = (iin[i] & 1) as usize;")
        .replace("Light l =", "let mut l =")
    )
    for name, body in _bodies("params.").items()
}
RUST = {
    name: Template(
        _RUST_BODIES[name],
        f"{_RUST_DERIVE}pub struct Light {{\n    color: Vec4,\n    dir: Vec4,\n}}\n"
        f"{_RUST_DERIVE}pub struct Params {{\n{members}\n}}",
        params="#[spirv(uniform, descriptor_set = 0, binding = 3)] params: &Params",
    )
    for name, members in _RUST_MEMBERS.items()
}

# Mojo reads a SIMD vector's lanes by index and declares each local with var.
_MOJO_VEC4 = "SIMD[DType.float32, 4]"
_MOJO_LIGHT = (
    f"@fieldwise_init\nstruct Light(TrivialRegisterPassable):\n    var color: {_MOJO_VEC4}\n"
    f"    var dir: {_MOJO_VEC4}"
)
_MOJO_MEMBERS = {
    "uniform-struct": f"    var light: Light\n    var tint: {_MOJO_VEC4}",
    "uniform-struct-array": "    var lights: Array[Light, 2]",
    "uniform-struct-copy": f"    var tint: {_MOJO_VEC4}\n    var light: Light",
}


def _mojo(body: str) -> str:
    body = re.sub(r"\.([xyzw])\b", lambda m: f"[{'xyzw'.index(m.group(1))}]", body)
    body = body.replace("int k = iin[i] & 1;", "var k = Int(iin[i] & 1);")
    body = re.sub(r"(?:\$vec4|Light) (\w) =", r"var \1 =", body)
    return body.replace(";", "")


MOJO = {
    name: Template(
        _mojo(body),
        f"{_MOJO_LIGHT}\n\n@fieldwise_init\nstruct Params(Copyable, Movable):\n{_MOJO_MEMBERS[name]}",
        params="params: Params",
    )
    for name, body in _bodies("params.").items()
}

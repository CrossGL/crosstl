"""Push-constant and descriptor-set probes: constants a command buffer pushes instead of a buffer, one member of them
placed past the start of the range, and a constants block bound in a descriptor set other than the first.

Each push-constant probe scales its input by one constant and adds another. GLSL declares them in a
``layout(push_constant)`` block, which only GLSL for Vulkan has (GL_KHR_vulkan_glsl), so those probes and the
descriptor-set probe are checked for Vulkan. HLSL and Slang mark a ``ConstantBuffer`` ``[[vk::push_constant]]``,
rust-gpu marks an entry-point parameter ``push_constant``, and CrossGL marks a ``cbuffer`` ``@push_constant``
(:mod:`crosstl.translator.codegen.spirv_push_constants`). A push-constant range that several stages share places a
stage's members past its start: ``layout(offset = 16)`` in GLSL, ``[[vk::offset(16)]]`` in HLSL and Slang, and
``@offset(16)`` in CrossGL. The descriptor-set probe binds its constants block in set 1: ``set = 1`` in GLSL,
``space1`` in HLSL and Slang, ``@group(1)`` in WGSL, ``descriptor_set = 1`` in rust-gpu and ``@set(1)`` in CrossGL.

Metal has neither: a kernel reads the bytes an encoder sets with ``setBytes`` through the same constant-address-space
argument as a buffer's, and binds each argument at an index of one table per resource kind (Metal Shading Language
Specification, "Attributes to Locate Buffers, Textures, and Samplers"). WGSL has no push-constant address space
(WGSL, "Address Spaces"), and rust-gpu lays a push-constant struct out with ``#[repr(C)]``, with no attribute that
places a member. GLSL ES 3.00 and the kernel languages have neither push constants nor descriptor sets.
"""

from __future__ import annotations

from .base import ABSENT, Template

NAMES = ("push-constant", "push-constant-offset", "descriptor-set")

# {0} is the prefix that names a member of the block.
_PUSH_BODY = "outp[i] = fin[i] * {0}scale + {0}bias;"
_SET_BODY = "outp[i] = fin[i] * {0}scale;"
_OFFSET = 16

GLSL = {
    "push-constant": Template(
        _PUSH_BODY.format("pc."),
        "layout(push_constant) uniform Constants {\n    float scale;\n    float bias;\n} pc;",
        target="vulkan",
    ),
    "push-constant-offset": Template(
        _PUSH_BODY.format("pc."),
        f"layout(push_constant) uniform Constants {{\n    layout(offset = {_OFFSET}) float scale;\n"
        "    float bias;\n} pc;",
        target="vulkan",
    ),
    "descriptor-set": Template(
        _SET_BODY.format(""),
        "layout(std140, set = 1, binding = 3) uniform Params {\n    float scale;\n};",
        target="vulkan",
    ),
}

CROSSGL = {
    "push-constant": Template(
        _PUSH_BODY.format(""),
        prelude="cbuffer Constants @push_constant {\n    float scale;\n    float bias;\n};",
    ),
    "push-constant-offset": Template(
        _PUSH_BODY.format(""),
        prelude=f"cbuffer Constants @push_constant {{\n    float scale @offset({_OFFSET});\n"
        "    float bias;\n};",
    ),
    "descriptor-set": Template(
        _SET_BODY.format(""),
        prelude="cbuffer Params @set(1) @binding(3) {\n    float scale;\n};",
    ),
}


# HLSL and Slang spell all three alike.
HLSL = {
    "push-constant": Template(
        _PUSH_BODY.format("pc."),
        "struct Constants {\n    float scale;\n    float bias;\n};\n"
        "[[vk::push_constant]] ConstantBuffer<Constants> pc;",
    ),
    "push-constant-offset": Template(
        _PUSH_BODY.format("pc."),
        f"struct Constants {{\n    [[vk::offset({_OFFSET})]] float scale;\n    float bias;\n}};\n"
        "[[vk::push_constant]] ConstantBuffer<Constants> pc;",
    ),
    "descriptor-set": Template(
        _SET_BODY.format(""),
        "cbuffer Params : register(b3, space1) {\n    float scale;\n};",
    ),
}
SLANG = HLSL

METAL = dict.fromkeys(NAMES, ABSENT)
WGSL = {
    **dict.fromkeys(NAMES, ABSENT),
    "descriptor-set": Template(
        _SET_BODY.format("params."),
        "struct Params {\n    scale: f32,\n}\n@group(1) @binding(3) var<uniform> params: Params;",
    ),
}
WEBGL = dict.fromkeys(NAMES, ABSENT)
KERNELS = dict.fromkeys(NAMES, ABSENT)

RUST = {
    "push-constant": Template(
        _PUSH_BODY.format("pc."),
        "#[derive(Clone, Copy)]\n#[repr(C)]\npub struct Constants {\n    scale: f32,\n    bias: f32,\n}",
        params="#[spirv(push_constant)] pc: &Constants",
    ),
    "push-constant-offset": ABSENT,
    "descriptor-set": Template(
        _SET_BODY.format("*"),
        params="#[spirv(uniform, descriptor_set = 1, binding = 3)] scale: &f32",
    ),
}

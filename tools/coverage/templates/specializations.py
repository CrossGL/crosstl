"""Specialization-constant probes: an integer, a boolean and a float constant the pipeline may set, a constant and a
vector computed from one, a workgroup size and a workgroup array length the pipeline sets, and a helper function
that reads one.

Each probe scales or negates its input by a constant that defaults to a value in the source and that the pipeline may
replace when it is created. GLSL declares it with ``layout(constant_id = N)``, which only GLSL for Vulkan has
(GL_KHR_vulkan_glsl), so the GLSL probes are checked for Vulkan. Slang uses ``[vk::constant_id(N)]``, WGSL ``override``
declarations, Rust the ``spec_constant`` attribute on an entry-point parameter and CrossGL the ``@constant_id``
semantic. Metal has function constants, which have no default: the probe gives one with
``is_function_constant_defined``. A specialized
workgroup size is ``local_size_x_id`` in GLSL, an override in WGSL's ``@workgroup_size`` and a constant in Slang's
``[numthreads]``. The array-length probe fills a workgroup array whose length is the constant and reads its last
element, and the helper probe reads the constant in a function the kernel calls; rust-gpu passes the entry-point
parameter to that function.

HLSL compiled for Direct3D has no specialization constants, and neither do GLSL ES 3.00 or the kernel languages. Metal
sets the threadgroup size when it dispatches a kernel, so it has no workgroup size to specialize, and the Metal
compiler takes no function constant as an array length. WGSL overrides are scalars (WGSL, "Override Declarations"),
and slangc evaluates a global vector constant at compile time, so neither has a vector computed from a constant.
rust-gpu's specialization constants are ``u32`` parameters, so Rust has only the integer constant and the helper that
reads one: its other probes need a boolean, float or vector constant, an expression of one or an array length or
workgroup size, which takes literals.
"""

from __future__ import annotations

from .base import ABSENT, Template

NAMES = (
    "specialization-constant",
    "specialization-constant-bool",
    "specialization-constant-expression",
    "specialization-workgroup-size",
    "specialization-constant-float",
    "specialization-constant-vector",
    "specialization-array-size",
    "specialization-constant-in-helper",
)

_SCALE_BODY = "outp[i] = fin[i] * float(scale);"
_NEGATE_BODY = "outp[i] = negate ? -fin[i] : fin[i];"
_DOUBLED_BODY = "outp[i] = fin[i] * float(doubled);"
_GAIN_BODY = "outp[i] = fin[i] * gain;"
_STEPS_BODY = "outp[i] = fin[i] * float(steps.x + steps.y);"
# {sync} is the language's workgroup barrier.
_TILE_BODY = (
    "for (int k = 0; k < count; k++) {{\n    tile[k] = fin[i] * float(k + 1);\n}}\n{sync};\n"
    "outp[i] = tile[count - 1] + float(count);"
)
_SCALED_BODY = "outp[i] = scaled(fin[i]);"
_SCALED = "float scaled(float x) {\n    return x * float(scale) + 1.0;\n}"

_GLSL_SCALE = "layout(constant_id = 0) const int scale = 3;"
_GLSL_WORKGROUP = (
    "#version 460\nlayout(local_size_x_id = 0, local_size_y = 1, local_size_z = 1) in;\n"
    "layout(std430, binding = 0) buffer B0 { float fin[]; };\n"
    "layout(std430, binding = 1) buffer B1 { float outp[]; };\n"
    "void main() {\n    uint i = gl_GlobalInvocationID.x;\n"
    "    outp[i] = fin[i] * float(gl_WorkGroupSize.x);\n}\n"
)
GLSL = {
    "specialization-constant": Template(
        _SCALE_BODY, declarations=_GLSL_SCALE, target="vulkan"
    ),
    "specialization-constant-bool": Template(
        _NEGATE_BODY,
        declarations="layout(constant_id = 0) const bool negate = true;",
        target="vulkan",
    ),
    "specialization-constant-expression": Template(
        _DOUBLED_BODY,
        declarations=f"{_GLSL_SCALE}\nconst int doubled = scale * 2 + 1;",
        target="vulkan",
    ),
    "specialization-workgroup-size": Template(program=_GLSL_WORKGROUP, target="vulkan"),
    "specialization-constant-float": Template(
        _GAIN_BODY,
        declarations="layout(constant_id = 0) const float gain = 1.5;",
        target="vulkan",
    ),
    "specialization-constant-vector": Template(
        _STEPS_BODY,
        declarations=f"{_GLSL_SCALE}\nconst ivec2 steps = ivec2(scale, 1);",
        target="vulkan",
    ),
    "specialization-array-size": Template(
        _TILE_BODY.format(sync="barrier()"),
        declarations="layout(constant_id = 0) const int count = 4;\nshared float tile[count];",
        target="vulkan",
    ),
    "specialization-constant-in-helper": Template(
        _SCALED_BODY, declarations=f"{_GLSL_SCALE}\n\n{_SCALED}", target="vulkan"
    ),
}

_CROSSGL_SCALE = "const int scale @constant_id(0) = 3;"
_CROSSGL_WORKGROUP = (
    "shader Probe {\n    compute {\n"
    "        layout(local_size_x_id = 0, local_size_y = 1, local_size_z = 1) in;\n"
    "        buffer float* fin;\n        buffer float* outp;\n        void main() {\n"
    "            uint i = gl_GlobalInvocationID.x;\n"
    "            outp[i] = fin[i] * float(gl_WorkGroupSize.x);\n        }\n    }\n}\n"
)
CROSSGL = {
    "specialization-constant": Template(_SCALE_BODY, prelude=_CROSSGL_SCALE),
    "specialization-constant-bool": Template(
        _NEGATE_BODY, prelude="const bool negate @constant_id(0) = true;"
    ),
    "specialization-constant-expression": Template(
        _DOUBLED_BODY, prelude=f"{_CROSSGL_SCALE}\nconst int doubled = scale * 2 + 1;"
    ),
    "specialization-workgroup-size": Template(program=_CROSSGL_WORKGROUP),
    "specialization-constant-float": Template(
        _GAIN_BODY, prelude="const float gain @constant_id(0) = 1.5;"
    ),
    "specialization-constant-vector": Template(
        _STEPS_BODY, prelude=f"{_CROSSGL_SCALE}\nconst ivec2 steps = ivec2(scale, 1);"
    ),
    "specialization-array-size": Template(
        _TILE_BODY.format(sync="barrier()"),
        "shared float tile[count];",
        prelude="const int count @constant_id(0) = 4;",
    ),
    "specialization-constant-in-helper": Template(
        _SCALED_BODY, _SCALED, prelude=_CROSSGL_SCALE
    ),
}

HLSL = dict.fromkeys(NAMES, ABSENT)
WEBGL = dict.fromkeys(NAMES, ABSENT)
KERNELS = dict.fromkeys(NAMES, ABSENT)

_SLANG_SCALE = "[vk::constant_id(0)] const int scale = 3;"
_SLANG_WORKGROUP = (
    "[vk::constant_id(0)] const int size = 1;\n"
    "RWStructuredBuffer<float> fin;\nRWStructuredBuffer<float> outp;\n"
    '[shader("compute")]\n[numthreads(size, 1, 1)]\n'
    "void main(uint3 id : SV_DispatchThreadID) {\n    uint i = id.x;\n"
    "    outp[i] = fin[i] * float(size);\n}\n"
)
SLANG = {
    "specialization-constant": Template(_SCALE_BODY, declarations=_SLANG_SCALE),
    "specialization-constant-bool": Template(
        _NEGATE_BODY, declarations="[vk::constant_id(0)] const bool negate = true;"
    ),
    "specialization-constant-expression": Template(
        _DOUBLED_BODY,
        declarations=f"{_SLANG_SCALE}\nstatic const int doubled = scale * 2 + 1;",
    ),
    "specialization-workgroup-size": Template(program=_SLANG_WORKGROUP, entry="main"),
    "specialization-constant-float": Template(
        _GAIN_BODY, declarations="[vk::constant_id(0)] const float gain = 1.5;"
    ),
    "specialization-constant-vector": ABSENT,
    "specialization-array-size": Template(
        _TILE_BODY.format(sync="GroupMemoryBarrierWithGroupSync()"),
        declarations="[vk::constant_id(0)] const int count = 4;\ngroupshared float tile[count];",
    ),
    "specialization-constant-in-helper": Template(
        _SCALED_BODY, declarations=f"{_SLANG_SCALE}\n\n{_SCALED}"
    ),
}


def _metal_constant(kind: str, name: str, default: str) -> str:
    """A Metal function constant ``name`` that is ``default`` when the pipeline gives it no value."""
    return (
        f"constant {kind} {name}_value [[function_constant(0)]];\n"
        f"constant {kind} {name} = is_function_constant_defined({name}_value) ? {name}_value : {default};"
    )


METAL = {
    "specialization-constant": Template(
        _SCALE_BODY, declarations=_metal_constant("int", "scale", "3")
    ),
    "specialization-constant-bool": Template(
        _NEGATE_BODY, declarations=_metal_constant("bool", "negate", "true")
    ),
    "specialization-constant-expression": Template(
        _DOUBLED_BODY,
        declarations=_metal_constant("int", "scale", "3")
        + "\nconstant int doubled = scale * 2 + 1;",
    ),
    "specialization-workgroup-size": ABSENT,
    "specialization-constant-float": Template(
        _GAIN_BODY, declarations=_metal_constant("float", "gain", "1.5")
    ),
    "specialization-constant-vector": Template(
        _STEPS_BODY,
        declarations=_metal_constant("int", "scale", "3")
        + "\nconstant int2 steps = int2(scale, 1);",
    ),
    "specialization-array-size": ABSENT,
    "specialization-constant-in-helper": Template(
        _SCALED_BODY,
        declarations=_metal_constant("int", "scale", "3") + f"\n\n{_SCALED}",
    ),
}

_WGSL_WORKGROUP = (
    "override size: u32 = 1;\n"
    "@group(0) @binding(0) var<storage, read_write> fin: array<f32>;\n"
    "@group(0) @binding(1) var<storage, read_write> outp: array<f32>;\n"
    "@compute @workgroup_size(size, 1, 1)\n"
    "fn main(@builtin(global_invocation_id) id: vec3<u32>) {\n    let i = id.x;\n"
    "    outp[i] = fin[i] * f32(size);\n}\n"
)
WGSL = {
    "specialization-constant": Template(
        "outp[i] = fin[i] * f32(scale);", declarations="override scale: i32 = 3;"
    ),
    "specialization-constant-bool": Template(
        "outp[i] = select(fin[i], -fin[i], negate);",
        declarations="override negate: bool = true;",
    ),
    "specialization-constant-expression": Template(
        "outp[i] = fin[i] * f32(doubled);",
        declarations="override scale: i32 = 3;\noverride doubled: i32 = scale * 2 + 1;",
    ),
    "specialization-workgroup-size": Template(program=_WGSL_WORKGROUP),
    "specialization-constant-float": Template(
        _GAIN_BODY, declarations="override gain: f32 = 1.5;"
    ),
    "specialization-constant-vector": ABSENT,
    "specialization-array-size": Template(
        "for (var k: i32 = 0; k < count; k++) {\n    tile[k] = fin[i] * f32(k + 1);\n}\n"
        "workgroupBarrier();\noutp[i] = tile[count - 1] + f32(count);",
        declarations="override count: i32 = 4;\nvar<workgroup> tile: array<f32, count>;",
    ),
    "specialization-constant-in-helper": Template(
        _SCALED_BODY,
        declarations="override scale: i32 = 3;\n\n"
        "fn scaled(x: f32) -> f32 {\n    return x * f32(scale) + 1.0;\n}",
    ),
}

_RUST_SCALE = "#[spirv(spec_constant(id = 0, default = 3))] scale: u32"
RUST = {
    **dict.fromkeys(NAMES, ABSENT),
    "specialization-constant": Template(
        "outp[i] = fin[i] * scale as f32;", params=_RUST_SCALE
    ),
    "specialization-constant-in-helper": Template(
        "outp[i] = scaled(fin[i], scale);",
        "fn scaled(x: f32, scale: u32) -> f32 {\n    x * scale as f32 + 1.0\n}",
        params=_RUST_SCALE,
    ),
}

"""Specialization-constant probes: an integer and a boolean constant the pipeline may set, a constant computed from
one, and a workgroup size the pipeline sets.

Each probe scales or negates its input by a constant that defaults to a value in the source and that the pipeline may
replace when it is created. GLSL declares it with ``layout(constant_id = N)``, which only GLSL for Vulkan has
(GL_KHR_vulkan_glsl), so the GLSL probes are checked for Vulkan. Slang uses ``[vk::constant_id(N)]``, WGSL ``override``
declarations, Rust the ``spec_constant`` attribute on an entry-point parameter and CrossGL the ``@constant_id``
semantic. Metal has function constants, which have no default: the probe gives one with
``is_function_constant_defined``. A specialized
workgroup size is ``local_size_x_id`` in GLSL, an override in WGSL's ``@workgroup_size`` and a constant in Slang's
``[numthreads]``.

HLSL compiled for Direct3D has no specialization constants, and neither do GLSL ES 3.00 or the kernel languages. Metal
sets the threadgroup size when it dispatches a kernel, so it has no workgroup size to specialize. rust-gpu's
specialization constants are ``u32`` parameters, so Rust has no boolean one, no constant computed from one and no
specialized workgroup size, whose ``threads`` takes literals.
"""

from __future__ import annotations

from .base import ABSENT, Template

NAMES = (
    "specialization-constant",
    "specialization-constant-bool",
    "specialization-constant-expression",
    "specialization-workgroup-size",
)

_SCALE_BODY = "outp[i] = fin[i] * float(scale);"
_NEGATE_BODY = "outp[i] = negate ? -fin[i] : fin[i];"
_DOUBLED_BODY = "outp[i] = fin[i] * float(doubled);"

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
}

RUST = {
    "specialization-constant": Template(
        "outp[i] = fin[i] * scale as f32;",
        params="#[spirv(spec_constant(id = 0, default = 3))] scale: u32",
    ),
    "specialization-constant-bool": ABSENT,
    "specialization-constant-expression": ABSENT,
    "specialization-workgroup-size": ABSENT,
}

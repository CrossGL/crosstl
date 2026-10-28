"""Primitive-id, sample-index, sample-mask-input and sample-mask probes: the built-ins through which the rasteriser
tells a fragment shader which primitive and which sample it shades and which samples it covers, and through which
the shader writes its own coverage.

Each probe is the fragment probe of :mod:`.graphics` with one built-in added: it reads the index of the primitive or
of the sample, or the lowest bit of the input coverage mask, into the colour it writes, or writes a coverage mask
next to the colour. GLSL uses ``gl_PrimitiveID``, ``gl_SampleID``, ``gl_SampleMaskIn`` and ``gl_SampleMask``, HLSL
and Slang the ``SV_PrimitiveID``, ``SV_SampleIndex`` and ``SV_Coverage`` semantics, Metal the ``[[primitive_id]]``,
``[[sample_id]]`` and ``[[sample_mask]]`` attributes, WGSL the ``primitive_index`` built-in, which needs
``enable primitive_index;``, and the ``sample_index`` and ``sample_mask`` built-ins, Rust the ``primitive_id``,
``sample_id`` and ``sample_mask`` attributes and CrossGL the GLSL names as member semantics.

GLSL ES 3.00 has none of these built-ins (GLSL ES 3.20 adds them), so WebGL 2 has no probe. The kernel languages
have no fragment stage.
"""

from __future__ import annotations

from .base import ABSENT, Template

NAMES = ("primitive-id", "sample-index", "sample-mask-input", "sample-mask")

_GLSL_STATEMENTS = {
    "primitive-id": "color = vec4(uv, float(gl_PrimitiveID), 1.0);",
    "sample-index": "color = vec4(uv, float(gl_SampleID), 1.0);",
    "sample-mask-input": "color = vec4(uv, float(gl_SampleMaskIn[0] & 1), 1.0);",
    "sample-mask": "color = vec4(uv, 0.0, 1.0);\n    gl_SampleMask[0] = 1;",
}

GLSL = {
    name: Template(
        program="#version 460\nlayout(location = 0) in vec2 uv;\nlayout(location = 0) out vec4 color;\n"
        f"void main() {{\n    {statement}\n}}\n"
    )
    for name, statement in _GLSL_STATEMENTS.items()
}
WEBGL = dict.fromkeys(NAMES, ABSENT)

_HLSL_SIGNATURE = (
    "float4 main(float2 uv : TEXCOORD0, uint {name} : {semantic}) : SV_Target"
)
_HLSL_BODY = "    return float4(uv, float({value}), 1.0);\n"
# Each program as the declarations ahead of the entry point, its signature and its body.
_HLSL_PARTS = {
    "primitive-id": (
        "",
        _HLSL_SIGNATURE.format(name="pid", semantic="SV_PrimitiveID"),
        _HLSL_BODY.format(value="pid"),
    ),
    "sample-index": (
        "",
        _HLSL_SIGNATURE.format(name="sid", semantic="SV_SampleIndex"),
        _HLSL_BODY.format(value="sid"),
    ),
    "sample-mask-input": (
        "",
        _HLSL_SIGNATURE.format(name="mask", semantic="SV_Coverage"),
        _HLSL_BODY.format(value="mask & 1u"),
    ),
    "sample-mask": (
        "struct FragmentOutput {\n    float4 color : SV_Target;\n    uint mask : SV_Coverage;\n};\n\n",
        "FragmentOutput main(float2 uv : TEXCOORD0)",
        "    FragmentOutput output;\n    output.color = float4(uv, 0.0, 1.0);\n"
        "    output.mask = 1u;\n    return output;\n",
    ),
}


def _hlsl(attribute: str = "") -> dict[str, Template]:
    """The HLSL programs; Slang's name their stage with ``attribute`` on the entry point."""
    return {
        name: Template(
            program=f"{types}{attribute}{signature} {{\n{body}}}\n", entry="main"
        )
        for name, (types, signature, body) in _HLSL_PARTS.items()
    }


HLSL = _hlsl()
SLANG = _hlsl('[shader("fragment")]\n')

_METAL_HEADER = (
    "#include <metal_stdlib>\nusing namespace metal;\n\n"
    "struct FragmentIn {\n    float2 uv;\n};\n\n"
)
_METAL_PROBE = (
    "fragment float4 probe(FragmentIn in [[stage_in]], uint {name} [[{attribute}]]) {{\n"
    "    return float4(in.uv, float({value}), 1.0);\n}}\n"
)
_METAL = {
    "primitive-id": _METAL_PROBE.format(
        name="pid", attribute="primitive_id", value="pid"
    ),
    "sample-index": _METAL_PROBE.format(name="sid", attribute="sample_id", value="sid"),
    "sample-mask-input": _METAL_PROBE.format(
        name="mask", attribute="sample_mask", value="mask & 1u"
    ),
    "sample-mask": (
        "struct FragmentOut {\n    float4 color [[color(0)]];\n    uint mask [[sample_mask]];\n};\n\n"
        "fragment FragmentOut probe(FragmentIn in [[stage_in]]) {\n    FragmentOut out;\n"
        "    out.color = float4(in.uv, 0.0, 1.0);\n    out.mask = 1u;\n    return out;\n}\n"
    ),
}
METAL = {name: Template(program=_METAL_HEADER + text) for name, text in _METAL.items()}

_WGSL_PROBE = (
    "@fragment\nfn main(@location(0) uv: vec2<f32>, @builtin({builtin}) {name}: u32) "
    "-> @location(0) vec4<f32> {{\n    return vec4<f32>(uv, f32({value}), 1.0);\n}}\n"
)
_WGSL = {
    "primitive-id": (
        "enable primitive_index;\n\n"
        + _WGSL_PROBE.format(builtin="primitive_index", name="pid", value="pid")
    ),
    "sample-index": _WGSL_PROBE.format(builtin="sample_index", name="sid", value="sid"),
    "sample-mask-input": _WGSL_PROBE.format(
        builtin="sample_mask", name="mask", value="mask & 1u"
    ),
    "sample-mask": (
        "struct FragmentOutput {\n    @location(0) color: vec4<f32>,\n    @builtin(sample_mask) mask: u32,\n}\n\n"
        "@fragment\nfn main(@location(0) uv: vec2<f32>) -> FragmentOutput {\n"
        "    var output: FragmentOutput;\n    output.color = vec4<f32>(uv, 0.0, 1.0);\n"
        "    output.mask = 1u;\n    return output;\n}\n"
    ),
}
WGSL = {name: Template(program=text) for name, text in _WGSL.items()}

_RUST_HEADER = "#![no_std]\nuse spirv_std::glam::*;\nuse spirv_std::spirv;\n\n"
_RUST_BODY = "    *color = Vec4::new(uv.x, uv.y, {value} as f32, 1.0);\n"
# Each entry point as its parameters and its body. SPIR-V declares the coverage mask an array of 32-bit integers, and
# Vulkan requires every integer input of a fragment shader to be flat, built-ins included, as glslang declares them.
_RUST = {
    "primitive-id": (
        "uv: Vec2, #[spirv(primitive_id, flat)] pid: u32, color: &mut Vec4",
        _RUST_BODY.format(value="pid"),
    ),
    "sample-index": (
        "uv: Vec2, #[spirv(sample_id, flat)] sid: u32, color: &mut Vec4",
        _RUST_BODY.format(value="sid"),
    ),
    "sample-mask-input": (
        "uv: Vec2, #[spirv(sample_mask, flat)] mask: [u32; 1], color: &mut Vec4",
        _RUST_BODY.format(value="(mask[0] & 1)"),
    ),
    "sample-mask": (
        "uv: Vec2, color: &mut Vec4, #[spirv(sample_mask)] mask: &mut [u32; 1]",
        "    *color = Vec4::new(uv.x, uv.y, 0.0, 1.0);\n    mask[0] = 1;\n",
    ),
}
RUST = {
    name: Template(
        program=f"{_RUST_HEADER}#[spirv(fragment)]\npub fn main({params}) {{\n{body}}}\n",
        entry="main",
    )
    for name, (params, body) in _RUST.items()
}

KERNELS = dict.fromkeys(NAMES, ABSENT)

_CROSSGL_MEMBERS = {
    "primitive-id": ("        int pid @ gl_PrimitiveID;\n", ""),
    "sample-index": ("        int sid @ gl_SampleID;\n", ""),
    "sample-mask-input": ("        uint mask @ gl_SampleMaskIn;\n", ""),
    "sample-mask": ("", "        uint mask @ gl_SampleMask;\n"),
}
_CROSSGL_STATEMENTS = {
    "primitive-id": "output.color = vec4(input.uv, float(input.pid), 1.0);",
    "sample-index": "output.color = vec4(input.uv, float(input.sid), 1.0);",
    "sample-mask-input": "output.color = vec4(input.uv, float(input.mask & 1u), 1.0);",
    "sample-mask": (
        "output.color = vec4(input.uv, 0.0, 1.0);\n            output.mask = 1u;"
    ),
}
CROSSGL = {
    name: Template(
        program="shader Probe {\n    struct FragmentInput {\n        vec2 uv;\n"
        f"{inputs}    }}\n    struct FragmentOutput {{\n        vec4 color;\n{outputs}    }}\n"
        "    fragment {\n        FragmentOutput main(FragmentInput input) {\n"
        f"            FragmentOutput output;\n            {_CROSSGL_STATEMENTS[name]}\n"
        "            return output;\n        }\n    }\n}\n"
    )
    for name, (inputs, outputs) in _CROSSGL_MEMBERS.items()
}

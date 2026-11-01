"""Barycentric-coordinate, shading-rate, view-index and stencil-export probes: fragment built-ins that GLSL takes
from extensions.

Each probe is the fragment probe of :mod:`.graphics` with one built-in added: it reads the fragment's barycentric
coordinates, its shading rate or the index of the view it renders into the colour it writes, or writes a stencil
reference next to the colour. GLSL uses ``gl_BaryCoordEXT`` (GL_EXT_fragment_shader_barycentric),
``gl_ShadingRateEXT`` (GL_EXT_fragment_shading_rate), ``gl_ViewIndex`` (GL_EXT_multiview) and
``gl_FragStencilRefARB`` (GL_ARB_shader_stencil_export). HLSL and Slang use the ``SV_Barycentrics``,
``SV_ShadingRate``, ``SV_ViewID`` and ``SV_StencilRef`` semantics. Metal uses the ``[[barycentric_coord]]`` and
``[[stencil]]`` attributes and renders views through vertex amplification, whose ``[[amplification_id]]`` names the
view. Rust uses the ``bary_coord``, ``view_index`` and ``frag_stencil_ref_ext`` attributes, and GLSL ES 3.00 reads
``gl_ViewID_OVR`` (OVR_multiview2). CrossGL uses the GLSL names as member semantics.

Metal has no shading-rate input and rust-gpu no shading-rate built-in. WGSL has none of the four built-ins, and GLSL
ES 3.00 only the view index. The kernel languages have no fragment stage.
"""

from __future__ import annotations

from .base import ABSENT, Template

NAMES = ("barycentric-coordinate", "shading-rate", "view-index", "stencil-export")

_STENCIL = "uv.x * 255.0"

# Each program as its extension and its body.
_GLSL_PARTS = {
    "barycentric-coordinate": (
        "GL_EXT_fragment_shader_barycentric",
        "color = vec4(gl_BaryCoordEXT, uv.x);",
    ),
    "shading-rate": (
        "GL_EXT_fragment_shading_rate",
        "color = vec4(uv, float(gl_ShadingRateEXT), 1.0);",
    ),
    "view-index": (
        "GL_EXT_multiview",
        "color = vec4(uv, float(gl_ViewIndex), 1.0);",
    ),
    "stencil-export": (
        "GL_ARB_shader_stencil_export",
        f"color = vec4(uv, 0.0, 1.0);\n    gl_FragStencilRefARB = int({_STENCIL});",
    ),
}
GLSL = {
    name: Template(
        program=f"#version 460\n#extension {extension} : require\n"
        "layout(location = 0) in vec2 uv;\nlayout(location = 0) out vec4 color;\n"
        f"void main() {{\n    {body}\n}}\n"
    )
    for name, (extension, body) in _GLSL_PARTS.items()
}
WEBGL = {
    **dict.fromkeys(NAMES, ABSENT),
    "view-index": Template(
        program="#version 300 es\n#extension GL_OVR_multiview2 : require\nprecision highp float;\n"
        "in vec2 uv;\nlayout(location = 0) out vec4 color;\n"
        "void main() {\n    color = vec4(uv, float(gl_ViewID_OVR), 1.0);\n}\n"
    ),
}
WGSL = dict.fromkeys(NAMES, ABSENT)
KERNELS = dict.fromkeys(NAMES, ABSENT)

_HLSL_INPUT = (
    "float4 main(float2 uv : TEXCOORD0, {parameter}) : SV_Target {{\n"
    "    return float4({value});\n}}\n"
)
# Each program as the declarations ahead of the entry point and the entry point.
_HLSL = {
    "barycentric-coordinate": (
        "",
        _HLSL_INPUT.format(
            parameter="float3 bary : SV_Barycentrics", value="bary, uv.x"
        ),
    ),
    "shading-rate": (
        "",
        _HLSL_INPUT.format(
            parameter="uint rate : SV_ShadingRate", value="uv, float(rate), 1.0"
        ),
    ),
    "view-index": (
        "",
        _HLSL_INPUT.format(
            parameter="uint view : SV_ViewID", value="uv, float(view), 1.0"
        ),
    ),
    "stencil-export": (
        "struct FragmentOutput {\n    float4 color : SV_Target;\n    uint stencil : SV_StencilRef;\n};\n\n",
        "FragmentOutput main(float2 uv : TEXCOORD0) {\n    FragmentOutput output;\n"
        "    output.color = float4(uv, 0.0, 1.0);\n"
        f"    output.stencil = uint({_STENCIL});\n    return output;\n}}\n",
    ),
}


def _hlsl(stage: str = "") -> dict[str, Template]:
    """The HLSL programs; Slang's name their stage with ``stage`` on the entry point."""
    return {
        name: Template(program=types + stage + entry, entry="main")
        for name, (types, entry) in _HLSL.items()
    }


HLSL = _hlsl()
SLANG = _hlsl('[shader("fragment")]\n')

_METAL_HEADER = (
    "#include <metal_stdlib>\nusing namespace metal;\n\n"
    "struct FragmentIn {\n    float2 uv;\n};\n\n"
)
_METAL_PROBE = (
    "fragment float4 probe(FragmentIn in [[stage_in]], {parameter}) {{\n"
    "    return float4({value});\n}}\n"
)
_METAL = {
    "barycentric-coordinate": _METAL_PROBE.format(
        parameter="float3 bary [[barycentric_coord]]", value="bary, in.uv.x"
    ),
    "view-index": _METAL_PROBE.format(
        parameter="uint view [[amplification_id]]", value="in.uv, float(view), 1.0"
    ),
    "stencil-export": (
        "struct FragmentOut {\n    float4 color [[color(0)]];\n    uint stencil [[stencil]];\n};\n\n"
        "fragment FragmentOut probe(FragmentIn in [[stage_in]]) {\n    FragmentOut out;\n"
        "    out.color = float4(in.uv, 0.0, 1.0);\n"
        f"    out.stencil = uint({_STENCIL.replace('uv', 'in.uv')});\n    return out;\n}}\n"
    ),
}
METAL = {
    name: Template(program=_METAL_HEADER + _METAL[name]) if name in _METAL else ABSENT
    for name in NAMES
}

_RUST_HEADER = "#![no_std]\nuse spirv_std::glam::*;\nuse spirv_std::spirv;\n\n"
# Each entry point as its execution modes, its parameters and its body. Vulkan requires every integer input of a
# fragment shader to be flat, built-ins included, as glslang declares them.
_RUST = {
    "barycentric-coordinate": (
        "",
        "uv: Vec2, #[spirv(bary_coord)] bary: Vec3, color: &mut Vec4",
        "    *color = Vec4::new(bary.x, bary.y, bary.z, uv.x);\n",
    ),
    "view-index": (
        "",
        "uv: Vec2, #[spirv(view_index, flat)] view: u32, color: &mut Vec4",
        "    *color = Vec4::new(uv.x, uv.y, view as f32, 1.0);\n",
    ),
    "stencil-export": (
        "(stencil_ref_replacing_ext)",
        "uv: Vec2, color: &mut Vec4, #[spirv(frag_stencil_ref_ext)] stencil: &mut i32",
        f"    *color = Vec4::new(uv.x, uv.y, 0.0, 1.0);\n    *stencil = ({_STENCIL}) as i32;\n",
    ),
}
RUST = {
    name: (
        Template(
            program=f"{_RUST_HEADER}#[spirv(fragment{_RUST[name][0]})]\n"
            f"pub fn main({_RUST[name][1]}) {{\n{_RUST[name][2]}}}\n",
            entry="main",
        )
        if name in _RUST
        else ABSENT
    )
    for name in NAMES
}

# Each probe as its extra input member, its extra output member and its body.
_CROSSGL_PARTS = {
    "barycentric-coordinate": (
        "        vec3 bary @ gl_BaryCoordEXT;\n",
        "",
        "output.color = vec4(input.bary, input.uv.x);",
    ),
    "shading-rate": (
        "        int rate @ gl_ShadingRateEXT;\n",
        "",
        "output.color = vec4(input.uv, float(input.rate), 1.0);",
    ),
    "view-index": (
        "        int view @ gl_ViewIndex;\n",
        "",
        "output.color = vec4(input.uv, float(input.view), 1.0);",
    ),
    "stencil-export": (
        "",
        "        int stencil @ gl_FragStencilRefARB;\n",
        "output.color = vec4(input.uv, 0.0, 1.0);\n"
        f"            output.stencil = int({_STENCIL.replace('uv', 'input.uv')});",
    ),
}
CROSSGL = {
    name: Template(
        program="shader Probe {\n    struct FragmentInput {\n        vec2 uv;\n"
        f"{inputs}    }}\n    struct FragmentOutput {{\n        vec4 color;\n{outputs}    }}\n"
        "    fragment {\n        FragmentOutput main(FragmentInput input) {\n"
        f"            FragmentOutput output;\n            {body}\n"
        "            return output;\n        }\n    }\n}\n"
    )
    for name, (inputs, outputs, body) in _CROSSGL_PARTS.items()
}

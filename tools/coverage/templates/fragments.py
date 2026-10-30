"""Fragment-coordinate, front-facing and fragment-depth probes: the built-in inputs and output of a fragment shader,
in every language with a fragment stage.

Each probe is the fragment probe of :mod:`.graphics` with one built-in added: it reads the fragment's window
position or which face of the primitive it belongs to into the colour it writes, or writes the fragment's depth next
to the colour. GLSL and WebGL 2 use ``gl_FragCoord``, ``gl_FrontFacing`` and ``gl_FragDepth``, HLSL and Slang the
``SV_Position``, ``SV_IsFrontFace`` and ``SV_Depth`` semantics, Metal the ``[[position]]``, ``[[front_facing]]`` and
``[[depth(any)]]`` attributes, WGSL the ``position``, ``front_facing`` and ``frag_depth`` built-ins, Rust the
``frag_coord``, ``front_facing`` and ``frag_depth`` attributes and CrossGL the GLSL names as member semantics. The
probes leave out the coordinate's w component, which Direct3D and the other APIs define differently. The kernel
languages have no fragment stage.
"""

from __future__ import annotations

from .base import ABSENT, Template

NAMES = ("fragment-coordinate", "front-facing", "fragment-depth")

_DEPTH = "uv.x * 0.5 + 0.25"

_GLSL_STATEMENTS = {
    "fragment-coordinate": (
        "color = vec4(gl_FragCoord.xy * 0.5 + uv, gl_FragCoord.z, 1.0);"
    ),
    "front-facing": (
        "color = gl_FrontFacing ? vec4(uv, 0.0, 1.0) : vec4(uv.y, uv.x, 1.0, 1.0);"
    ),
    "fragment-depth": f"color = vec4(uv, 0.0, 1.0);\n    gl_FragDepth = {_DEPTH};",
}


def _glsl(header: str, varying: str) -> dict[str, Template]:
    return {
        name: Template(
            program=f"{header}{varying} vec2 uv;\nlayout(location = 0) out vec4 color;\nvoid main() {{\n"
            f"    {statement}\n}}\n"
        )
        for name, statement in _GLSL_STATEMENTS.items()
    }


GLSL = _glsl("#version 460\n", "layout(location = 0) in")
WEBGL = _glsl("#version 300 es\nprecision highp float;\n", "in")

_HLSL_OUTPUT = "struct FragmentOutput {\n    float4 color : SV_Target;\n    float depth : SV_Depth;\n};\n\n"
# Each program as the declarations ahead of the entry point, its signature and its body.
_HLSL_PARTS = {
    "fragment-coordinate": (
        "",
        "float4 main(float4 position : SV_Position, float2 uv : TEXCOORD0) : SV_Target",
        "    return float4(position.xy * 0.5 + uv, position.z, 1.0);\n",
    ),
    "front-facing": (
        "",
        "float4 main(float2 uv : TEXCOORD0, bool front : SV_IsFrontFace) : SV_Target",
        "    return front ? float4(uv, 0.0, 1.0) : float4(uv.y, uv.x, 1.0, 1.0);\n",
    ),
    "fragment-depth": (
        _HLSL_OUTPUT,
        "FragmentOutput main(float2 uv : TEXCOORD0)",
        "    FragmentOutput output;\n    output.color = float4(uv, 0.0, 1.0);\n"
        f"    output.depth = {_DEPTH};\n    return output;\n",
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

_METAL_HEADER = "#include <metal_stdlib>\nusing namespace metal;\n\n"
_METAL = {
    "fragment-coordinate": (
        "struct FragmentIn {\n    float4 position [[position]];\n    float2 uv;\n};\n\n"
        "fragment float4 probe(FragmentIn in [[stage_in]]) {\n"
        "    return float4(in.position.xy * 0.5 + in.uv, in.position.z, 1.0);\n}\n"
    ),
    "front-facing": (
        "struct FragmentIn {\n    float2 uv;\n};\n\n"
        "fragment float4 probe(FragmentIn in [[stage_in]], bool front [[front_facing]]) {\n"
        "    return front ? float4(in.uv, 0.0, 1.0) : float4(in.uv.y, in.uv.x, 1.0, 1.0);\n}\n"
    ),
    "fragment-depth": (
        "struct FragmentIn {\n    float2 uv;\n};\n\n"
        "struct FragmentOut {\n    float4 color [[color(0)]];\n    float depth [[depth(any)]];\n};\n\n"
        "fragment FragmentOut probe(FragmentIn in [[stage_in]]) {\n    FragmentOut out;\n"
        "    out.color = float4(in.uv, 0.0, 1.0);\n"
        f"    out.depth = {_DEPTH.replace('uv', 'in.uv')};\n    return out;\n}}\n"
    ),
}
METAL = {name: Template(program=_METAL_HEADER + text) for name, text in _METAL.items()}

_WGSL = {
    "fragment-coordinate": (
        "@fragment\nfn main(@builtin(position) position: vec4<f32>, @location(0) uv: vec2<f32>) "
        "-> @location(0) vec4<f32> {\n"
        "    return vec4<f32>(position.xy * 0.5 + uv, position.z, 1.0);\n}\n"
    ),
    "front-facing": (
        "@fragment\nfn main(@location(0) uv: vec2<f32>, @builtin(front_facing) front: bool) "
        "-> @location(0) vec4<f32> {\n"
        "    return select(vec4<f32>(uv.y, uv.x, 1.0, 1.0), vec4<f32>(uv, 0.0, 1.0), front);\n}\n"
    ),
    "fragment-depth": (
        "struct FragmentOutput {\n    @location(0) color: vec4<f32>,\n    @builtin(frag_depth) depth: f32,\n}\n\n"
        "@fragment\nfn main(@location(0) uv: vec2<f32>) -> FragmentOutput {\n"
        "    var output: FragmentOutput;\n    output.color = vec4<f32>(uv, 0.0, 1.0);\n"
        f"    output.depth = {_DEPTH};\n    return output;\n}}\n"
    ),
}
WGSL = {name: Template(program=text) for name, text in _WGSL.items()}

_RUST_HEADER = "#![no_std]\nuse spirv_std::glam::*;\nuse spirv_std::spirv;\n\n"
# Each entry point as its stage attribute, its parameters and its body. A shader that writes the depth declares that
# it replaces it.
_RUST = {
    "fragment-coordinate": (
        "fragment",
        "#[spirv(frag_coord)] coord: Vec4, uv: Vec2, color: &mut Vec4",
        "    *color = Vec4::new(coord.x * 0.5 + uv.x, coord.y * 0.5 + uv.y, coord.z, 1.0);\n",
    ),
    "front-facing": (
        "fragment",
        "uv: Vec2, #[spirv(front_facing)] front: bool, color: &mut Vec4",
        "    *color = if front {\n        Vec4::new(uv.x, uv.y, 0.0, 1.0)\n    } else {\n"
        "        Vec4::new(uv.y, uv.x, 1.0, 1.0)\n    };\n",
    ),
    "fragment-depth": (
        "fragment(depth_replacing)",
        "uv: Vec2, color: &mut Vec4, #[spirv(frag_depth)] depth: &mut f32",
        f"    *color = Vec4::new(uv.x, uv.y, 0.0, 1.0);\n    *depth = {_DEPTH};\n",
    ),
}
RUST = {
    name: Template(
        program=f"{_RUST_HEADER}#[spirv({stage})]\npub fn main({params}) {{\n{body}}}\n",
        entry="main",
    )
    for name, (stage, params, body) in _RUST.items()
}

KERNELS = dict.fromkeys(NAMES, ABSENT)

_CROSSGL_MEMBERS = {
    "fragment-coordinate": ("        vec4 coord @ gl_FragCoord;\n", ""),
    "front-facing": ("        bool front @ gl_FrontFacing;\n", ""),
    "fragment-depth": ("", "        float depth @ gl_FragDepth;\n"),
}
_CROSSGL_STATEMENTS = {
    "fragment-coordinate": (
        "output.color = vec4(input.coord.xy * 0.5 + input.uv, input.coord.z, 1.0);"
    ),
    "front-facing": (
        "output.color = input.front ? vec4(input.uv, 0.0, 1.0) : vec4(input.uv.y, input.uv.x, 1.0, 1.0);"
    ),
    "fragment-depth": (
        "output.color = vec4(input.uv, 0.0, 1.0);\n"
        f"            output.depth = {_DEPTH.replace('uv', 'input.uv')};"
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

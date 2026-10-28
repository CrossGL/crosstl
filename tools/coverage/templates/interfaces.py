"""Interpolation and multiple-render-target probes: how a fragment shader's input is interpolated across the
primitive, and a fragment shader that writes two colour attachments.

Each probe is the fragment probe of :mod:`.fragments` with its ``uv`` input qualified flat, linear without
perspective correction, at the centroid or per sample, or with a second colour output at location 1. GLSL spells the
qualifiers ``flat``, ``noperspective``, ``centroid`` and ``sample``, HLSL and Slang ``nointerpolation``,
``noperspective``, ``centroid`` and ``sample``, Metal ``[[flat]]``, ``[[center_no_perspective]]``,
``[[centroid_perspective]]`` and ``[[sample_perspective]]``, WGSL ``@interpolate(flat)``, ``@interpolate(linear)``
and ``@interpolate(perspective, centroid)`` or ``(perspective, sample)``, and CrossGL ``@ flat``, ``@ noperspective``,
``@ centroid`` and ``@ sample``. GLSL ES 3.00 has neither ``noperspective`` nor ``sample``, and rust-gpu has an
attribute for flat inputs only. The kernel languages have no fragment stage.
"""

from __future__ import annotations

from .base import ABSENT, Template

INTERPOLATIONS = (
    "flat-interpolation",
    "noperspective-interpolation",
    "centroid-interpolation",
    "sample-interpolation",
)
NAMES = (*INTERPOLATIONS, "multiple-render-targets")

_QUALIFIERS = {
    "glsl": ("flat", "noperspective", "centroid", "sample"),
    "hlsl": ("nointerpolation", "noperspective", "centroid", "sample"),
    "metal": (
        "flat",
        "center_no_perspective",
        "centroid_perspective",
        "sample_perspective",
    ),
    "wgsl": ("flat", "linear", "perspective, centroid", "perspective, sample"),
    "crossgl": ("flat", "noperspective", "centroid", "sample"),
}


def _qualifiers(language: str) -> dict[str, str]:
    return dict(zip(INTERPOLATIONS, _QUALIFIERS[language]))


_GLSL_COLOR = "layout(location = 0) out vec4 color;\n"
_GLSL_EXTRA = "layout(location = 1) out vec4 extra;\n"
_GLSL_BODY = "    color = vec4(uv, 0.0, 1.0);\n"
_GLSL_TARGETS = "    extra = vec4(uv.y, uv.x, 1.0, 1.0);\n"


def _glsl(header: str, location: str, names: tuple[str, ...]) -> dict[str, Template]:
    qualified = _qualifiers("glsl")

    def program(name: str) -> str:
        qualifier = f"{qualified[name]} " if name in qualified else ""
        outputs = _GLSL_COLOR + (
            _GLSL_EXTRA if name == "multiple-render-targets" else ""
        )
        body = _GLSL_BODY + (_GLSL_TARGETS if name == "multiple-render-targets" else "")
        return f"{header}{location}{qualifier}in vec2 uv;\n{outputs}void main() {{\n{body}}}\n"

    return {
        name: Template(program=program(name)) if name in names else ABSENT
        for name in NAMES
    }


GLSL = _glsl("#version 460\n", "layout(location = 0) ", NAMES)
WEBGL = _glsl(
    "#version 300 es\nprecision highp float;\n",
    "",
    ("flat-interpolation", "centroid-interpolation", "multiple-render-targets"),
)

_HLSL_TARGETS = "struct FragmentOutput {\n    float4 color : SV_Target0;\n    float4 extra : SV_Target1;\n};\n\n"


def _hlsl(attribute: str = "") -> dict[str, Template]:
    """The HLSL programs; Slang's name their stage with ``attribute`` on the entry point."""
    programs = {
        name: (
            f"{attribute}float4 main({qualifier} float2 uv : TEXCOORD0) : SV_Target {{\n"
            "    return float4(uv, 0.0, 1.0);\n}\n"
        )
        for name, qualifier in _qualifiers("hlsl").items()
    }
    programs["multiple-render-targets"] = (
        f"{_HLSL_TARGETS}{attribute}FragmentOutput main(float2 uv : TEXCOORD0) {{\n"
        "    FragmentOutput output;\n    output.color = float4(uv, 0.0, 1.0);\n"
        "    output.extra = float4(uv.y, uv.x, 1.0, 1.0);\n    return output;\n}\n"
    )
    return {
        name: Template(program=text, entry="main") for name, text in programs.items()
    }


HLSL = _hlsl()
SLANG = _hlsl('[shader("fragment")]\n')

_METAL_HEADER = "#include <metal_stdlib>\nusing namespace metal;\n\n"
_METAL = {
    name: (
        f"struct FragmentIn {{\n    float2 uv [[{qualifier}]];\n}};\n\n"
        "fragment float4 probe(FragmentIn in [[stage_in]]) {\n    return float4(in.uv, 0.0, 1.0);\n}\n"
    )
    for name, qualifier in _qualifiers("metal").items()
}
_METAL["multiple-render-targets"] = (
    "struct FragmentIn {\n    float2 uv;\n};\n\n"
    "struct FragmentOut {\n    float4 color [[color(0)]];\n    float4 extra [[color(1)]];\n};\n\n"
    "fragment FragmentOut probe(FragmentIn in [[stage_in]]) {\n    FragmentOut out;\n"
    "    out.color = float4(in.uv, 0.0, 1.0);\n    out.extra = float4(in.uv.y, in.uv.x, 1.0, 1.0);\n"
    "    return out;\n}\n"
)
METAL = {name: Template(program=_METAL_HEADER + text) for name, text in _METAL.items()}

_WGSL = {
    name: (
        f"@fragment\nfn main(@location(0) @interpolate({qualifier}) uv: vec2<f32>) -> @location(0) vec4<f32> {{\n"
        "    return vec4<f32>(uv, 0.0, 1.0);\n}\n"
    )
    for name, qualifier in _qualifiers("wgsl").items()
}
_WGSL["multiple-render-targets"] = (
    "struct FragmentOutput {\n    @location(0) color: vec4<f32>,\n    @location(1) extra: vec4<f32>,\n}\n\n"
    "@fragment\nfn main(@location(0) uv: vec2<f32>) -> FragmentOutput {\n    var output: FragmentOutput;\n"
    "    output.color = vec4<f32>(uv, 0.0, 1.0);\n    output.extra = vec4<f32>(uv.y, uv.x, 1.0, 1.0);\n"
    "    return output;\n}\n"
)
WGSL = {name: Template(program=text) for name, text in _WGSL.items()}

_RUST_HEADER = (
    "#![no_std]\nuse spirv_std::glam::*;\nuse spirv_std::spirv;\n\n#[spirv(fragment)]\n"
)
RUST = {
    **dict.fromkeys(INTERPOLATIONS, ABSENT),
    "flat-interpolation": Template(
        program=f"{_RUST_HEADER}pub fn main(#[spirv(flat)] uv: Vec2, color: &mut Vec4) {{\n"
        "    *color = Vec4::new(uv.x, uv.y, 0.0, 1.0);\n}\n",
        entry="main",
    ),
    "multiple-render-targets": Template(
        program=f"{_RUST_HEADER}pub fn main(\n    uv: Vec2,\n    #[spirv(location = 0)] color: &mut Vec4,\n"
        "    #[spirv(location = 1)] extra: &mut Vec4,\n) {\n    *color = Vec4::new(uv.x, uv.y, 0.0, 1.0);\n"
        "    *extra = Vec4::new(uv.y, uv.x, 1.0, 1.0);\n}\n",
        entry="main",
    ),
}

KERNELS = dict.fromkeys(NAMES, ABSENT)

_CROSSGL_EXTRA = "            output.extra = vec4(input.uv.y, input.uv.x, 1.0, 1.0);\n"


def _crossgl(uv: str, outputs: str, extra: str) -> Template:
    return Template(
        program=f"shader Probe {{\n    struct FragmentInput {{\n        vec2 uv{uv};\n    }}\n"
        f"    struct FragmentOutput {{\n{outputs}    }}\n"
        "    fragment {\n        FragmentOutput main(FragmentInput input) {\n"
        "            FragmentOutput output;\n            output.color = vec4(input.uv, 0.0, 1.0);\n"
        f"{extra}            return output;\n        }}\n    }}\n}}\n"
    )


CROSSGL = {
    **{
        name: _crossgl(f" @ {qualifier}", "        vec4 color;\n", "")
        for name, qualifier in _qualifiers("crossgl").items()
    },
    "multiple-render-targets": _crossgl(
        "",
        "        vec4 color @location(0);\n        vec4 extra @location(1);\n",
        _CROSSGL_EXTRA,
    ),
}

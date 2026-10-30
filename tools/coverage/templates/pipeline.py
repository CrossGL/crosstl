"""Helper-invocation, early-fragment-tests, layer, viewport-index, point-coordinate and conservative-depth probes:
how a fragment shader meets the fixed-function stages around it.

Each probe is the fragment probe of :mod:`.graphics` with one change. It reads whether the invocation is a helper,
the layer or viewport it renders to or its position in a point sprite into the colour it writes, asks for the depth
and stencil tests to run before it, or promises that the depth it writes never lies in front of the rasterised
depth. GLSL uses ``gl_HelperInvocation``, ``layout(early_fragment_tests) in``, ``gl_Layer``, ``gl_ViewportIndex``,
``gl_PointCoord`` and ``layout(depth_greater)``. HLSL and Slang use ``IsHelperLane()``, ``[earlydepthstencil]``
and the ``SV_RenderTargetArrayIndex``, ``SV_ViewportArrayIndex`` and ``SV_DepthGreaterEqual`` semantics, and Slang
``SV_PointCoord`` as well. Metal uses ``simd_is_helper_thread()``, ``[[early_fragment_tests]]`` and the
``[[render_target_array_index]]``, ``[[viewport_array_index]]``, ``[[point_coord]]`` and ``[[depth(greater)]]``
attributes. Rust uses the ``helper_invocation``, ``layer``, ``viewport_index``, ``point_coord`` and ``frag_depth``
attributes and the ``early_fragment_tests`` and ``depth_greater`` execution modes. CrossGL uses the GLSL names as
member semantics, ``layout(early_fragment_tests) in`` in its fragment stage and the ``@depth_greater`` output that
CrossTL's GLSL importer writes.

Direct3D has no point sprites, so HLSL has no point-coordinate probe. WGSL has none of these features, and GLSL ES
3.00 has only ``gl_PointCoord`` (GLSL ES 3.10 adds the helper flag and the early tests). The kernel languages have
no fragment stage.
"""

from __future__ import annotations

from .base import ABSENT, Template

NAMES = (
    "helper-invocation",
    "early-fragment-tests",
    "layer",
    "viewport-index",
    "point-coordinate",
    "conservative-depth",
)

_DEPTH = "uv.x * 0.5 + 0.25"

# Each program as the declarations ahead of ``main`` and its body.
_GLSL_PARTS = {
    "helper-invocation": (
        "",
        "color = vec4(uv, gl_HelperInvocation ? 1.0 : 0.0, 1.0);",
    ),
    "early-fragment-tests": (
        "layout(early_fragment_tests) in;\n",
        "color = vec4(uv, 0.0, 1.0);",
    ),
    "layer": ("", "color = vec4(uv, float(gl_Layer), 1.0);"),
    "viewport-index": ("", "color = vec4(uv, float(gl_ViewportIndex), 1.0);"),
    "point-coordinate": ("", "color = vec4(gl_PointCoord, uv.x, 1.0);"),
    "conservative-depth": (
        "layout(depth_greater) out float gl_FragDepth;\n",
        f"color = vec4(uv, 0.0, 1.0);\n    gl_FragDepth = {_DEPTH};",
    ),
}


def _glsl(header: str, varying: str, names) -> dict[str, Template]:
    return {
        name: Template(
            program=f"{header}{varying} vec2 uv;\nlayout(location = 0) out vec4 color;\n"
            f"{_GLSL_PARTS[name][0]}void main() {{\n    {_GLSL_PARTS[name][1]}\n}}\n"
        )
        for name in names
    }


GLSL = _glsl("#version 460\n", "layout(location = 0) in", NAMES)
WEBGL = {
    **dict.fromkeys(NAMES, ABSENT),
    **_glsl("#version 300 es\nprecision highp float;\n", "in", ("point-coordinate",)),
}

_HLSL_SIGNATURE = "float4 main(float2 uv : TEXCOORD0{inputs}) : SV_Target"
_HLSL_BODY = "    return float4(uv, {value}, 1.0);\n"
# Each program as the declarations ahead of the entry point, the attribute on it, its signature and its body.
_HLSL_PARTS = {
    "helper-invocation": (
        "",
        "",
        _HLSL_SIGNATURE.format(inputs=""),
        _HLSL_BODY.format(value="IsHelperLane() ? 1.0 : 0.0"),
    ),
    "early-fragment-tests": (
        "",
        "[earlydepthstencil]\n",
        _HLSL_SIGNATURE.format(inputs=""),
        _HLSL_BODY.format(value="0.0"),
    ),
    "layer": (
        "",
        "",
        _HLSL_SIGNATURE.format(inputs=", uint layer : SV_RenderTargetArrayIndex"),
        _HLSL_BODY.format(value="float(layer)"),
    ),
    "viewport-index": (
        "",
        "",
        _HLSL_SIGNATURE.format(inputs=", uint viewport : SV_ViewportArrayIndex"),
        _HLSL_BODY.format(value="float(viewport)"),
    ),
    "conservative-depth": (
        "struct FragmentOutput {\n    float4 color : SV_Target;\n    float depth : SV_DepthGreaterEqual;\n};\n\n",
        "",
        "FragmentOutput main(float2 uv : TEXCOORD0)",
        "    FragmentOutput output;\n    output.color = float4(uv, 0.0, 1.0);\n"
        f"    output.depth = {_DEPTH};\n    return output;\n",
    ),
}
_SLANG_POINT = (
    "",
    "",
    _HLSL_SIGNATURE.format(inputs=", float2 point : SV_PointCoord"),
    "    return float4(point, uv.x, 1.0);\n",
)


def _hlsl(stage: str = "", point: tuple[str, str, str, str] | None = None):
    """The HLSL programs; Slang's name their stage with ``stage`` on the entry point and read the point coordinate."""
    parts = {**_HLSL_PARTS, "point-coordinate": point} if point else _HLSL_PARTS
    programs = {
        name: Template(
            program=f"{types}{stage}{attribute}{signature} {{\n{body}}}\n",
            entry="main",
        )
        for name, (types, attribute, signature, body) in parts.items()
    }
    return {name: programs.get(name, ABSENT) for name in NAMES}


HLSL = _hlsl()
SLANG = _hlsl('[shader("fragment")]\n', _SLANG_POINT)

_METAL_HEADER = (
    "#include <metal_stdlib>\nusing namespace metal;\n\n"
    "struct FragmentIn {\n    float2 uv;\n};\n\n"
)
_METAL_PROBE = (
    "{attribute}fragment float4 probe(FragmentIn in [[stage_in]]{inputs}) {{\n"
    "    return float4({value}, 1.0);\n}}\n"
)
_METAL = {
    "helper-invocation": _METAL_PROBE.format(
        attribute="",
        inputs="",
        value="in.uv, simd_is_helper_thread() ? 1.0 : 0.0",
    ),
    "early-fragment-tests": _METAL_PROBE.format(
        attribute="[[early_fragment_tests]] ", inputs="", value="in.uv, 0.0"
    ),
    "layer": _METAL_PROBE.format(
        attribute="",
        inputs=", uint layer [[render_target_array_index]]",
        value="in.uv, float(layer)",
    ),
    "viewport-index": _METAL_PROBE.format(
        attribute="",
        inputs=", uint viewport [[viewport_array_index]]",
        value="in.uv, float(viewport)",
    ),
    "point-coordinate": _METAL_PROBE.format(
        attribute="",
        inputs=", float2 point [[point_coord]]",
        value="point, in.uv.x",
    ),
    "conservative-depth": (
        "struct FragmentOut {\n    float4 color [[color(0)]];\n    float depth [[depth(greater)]];\n};\n\n"
        "fragment FragmentOut probe(FragmentIn in [[stage_in]]) {\n    FragmentOut out;\n"
        "    out.color = float4(in.uv, 0.0, 1.0);\n"
        f"    out.depth = {_DEPTH.replace('uv', 'in.uv')};\n    return out;\n}}\n"
    ),
}
METAL = {name: Template(program=_METAL_HEADER + text) for name, text in _METAL.items()}

WGSL = dict.fromkeys(NAMES, ABSENT)

_RUST_HEADER = "#![no_std]\nuse spirv_std::glam::*;\nuse spirv_std::spirv;\n\n"
_RUST_BODY = "    *color = Vec4::new({value}, 1.0);\n"
# Each entry point as its execution modes, its parameters and its body. Vulkan requires every integer input of a
# fragment shader to be flat, built-ins included, as glslang declares them.
_RUST = {
    "helper-invocation": (
        "",
        "uv: Vec2, #[spirv(helper_invocation)] helper: bool, color: &mut Vec4",
        _RUST_BODY.format(value="uv.x, uv.y, if helper { 1.0 } else { 0.0 }"),
    ),
    "early-fragment-tests": (
        "(early_fragment_tests)",
        "uv: Vec2, color: &mut Vec4",
        _RUST_BODY.format(value="uv.x, uv.y, 0.0"),
    ),
    "layer": (
        "",
        "uv: Vec2, #[spirv(layer, flat)] layer: u32, color: &mut Vec4",
        _RUST_BODY.format(value="uv.x, uv.y, layer as f32"),
    ),
    "viewport-index": (
        "",
        "uv: Vec2, #[spirv(viewport_index, flat)] viewport: u32, color: &mut Vec4",
        _RUST_BODY.format(value="uv.x, uv.y, viewport as f32"),
    ),
    "point-coordinate": (
        "",
        "uv: Vec2, #[spirv(point_coord)] point: Vec2, color: &mut Vec4",
        _RUST_BODY.format(value="point.x, point.y, uv.x"),
    ),
    "conservative-depth": (
        "(depth_replacing, depth_greater)",
        "uv: Vec2, color: &mut Vec4, #[spirv(frag_depth)] depth: &mut f32",
        f"    *color = Vec4::new(uv.x, uv.y, 0.0, 1.0);\n    *depth = {_DEPTH};\n",
    ),
}
RUST = {
    name: Template(
        program=f"{_RUST_HEADER}#[spirv(fragment{modes})]\npub fn main({params}) {{\n{body}}}\n",
        entry="main",
    )
    for name, (modes, params, body) in _RUST.items()
}

KERNELS = dict.fromkeys(NAMES, ABSENT)

_CROSSGL_INPUTS = {
    "helper-invocation": "        bool helper @ gl_HelperInvocation;\n",
    "layer": "        int layer @ gl_Layer;\n",
    "viewport-index": "        int viewport @ gl_ViewportIndex;\n",
    "point-coordinate": "        vec2 point @ gl_PointCoord;\n",
}
_CROSSGL_STATEMENTS = {
    "helper-invocation": (
        "output.color = vec4(input.uv, input.helper ? 1.0 : 0.0, 1.0);"
    ),
    "early-fragment-tests": "output.color = vec4(input.uv, 0.0, 1.0);",
    "layer": "output.color = vec4(input.uv, float(input.layer), 1.0);",
    "viewport-index": "output.color = vec4(input.uv, float(input.viewport), 1.0);",
    "point-coordinate": "output.color = vec4(input.point, input.uv.x, 1.0);",
}
_CROSSGL_STAGE = {"early-fragment-tests": "        layout(early_fragment_tests) in;\n"}
_CROSSGL = {
    name: (
        "shader Probe {\n    struct FragmentInput {\n        vec2 uv;\n"
        f"{_CROSSGL_INPUTS.get(name, '')}    }}\n"
        "    struct FragmentOutput {\n        vec4 color;\n    }\n"
        f"    fragment {{\n{_CROSSGL_STAGE.get(name, '')}"
        "        FragmentOutput main(FragmentInput input) {\n"
        f"            FragmentOutput output;\n            {statement}\n"
        "            return output;\n        }\n    }\n}\n"
    )
    for name, statement in _CROSSGL_STATEMENTS.items()
}
# The conservative depth output as CrossTL's GLSL importer writes it: a shader-scope output with the depth layout as
# its semantic.
_CROSSGL["conservative-depth"] = (
    "shader Probe {\n    struct FragmentInput {\n        vec2 uv;\n    }\n"
    "    out vec4 color @location(0);\n    out float gl_FragDepth @depth_greater;\n"
    "    fragment {\n        void main(FragmentInput input) {\n"
    "            color = vec4(input.uv, 0.0, 1.0);\n"
    f"            gl_FragDepth = {_DEPTH.replace('uv', 'input.uv')};\n"
    "        }\n    }\n}\n"
)
CROSSGL = {name: Template(program=_CROSSGL[name]) for name in NAMES}

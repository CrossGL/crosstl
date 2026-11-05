"""Tessellation probes: a patch's domain, spacing, winding and point output, its tessellation levels, its per-patch
values and its control-point count.

Each probe is a tessellation probe of :mod:`.graphics`, a triangle patch of three control points, with one part
changed. An evaluation probe tessellates a quad or an isoline patch, reads the levels or a per-patch input, or spaces,
winds or emits points other than the defaults. A control probe writes a per-patch output, reads the patch's control
point count, or spaces, winds or emits points other than the defaults.

GLSL and rust-gpu state the domain, spacing, winding and point output in the evaluation stage's layout or execution
modes. HLSL and Slang state the domain on both stages, but the partitioning and the output topology only on the hull
shader, so their domain shader has no spelling for spacing, winding or point output, and the ``tessellation-control-``
probes measure them on the hull shader instead, where GLSL has no spelling for them. HLSL reads the levels and the
per-patch values through the patch-constant structure, and has no system value for the control point count, which is
the ``InputPatch`` size. Metal evaluates a triangle or a quad patch in a post-tessellation vertex function that reads
its per-patch values as ``[[stage_in]]`` attributes; the partitioning and the winding are pipeline state, and Metal
has neither isolines, point output, a control stage nor a built-in for the levels. rust-gpu has no ``Patch``
decoration. GLSL ES 3.00, WGSL and the kernel languages have no tessellation stages.

CrossGL states a feature every target spells in the form of the tessellation stage probes of :mod:`..crossgl`, with
the domain and the patch-constant structure on the stage, and a feature only GLSL and rust-gpu spell in the GLSL
layouts its GLSL importer writes.
"""

from __future__ import annotations

from dataclasses import replace

from .base import ABSENT, Template

# Each domain as its GLSL layout, HLSL domain, control point count, tessellation factor counts and coordinate size.
_DOMAINS = {
    "triangles": ("triangles", "tri", 3, (3, 1), 3),
    "quads": ("quads", "quad", 4, (4, 2), 2),
    "isolines": ("isolines", "isoline", 2, (2, 0), 2),
}
# Each evaluation probe as its domain, the GLSL spacing, winding and point layouts, the per-patch input it reads and
# the position it writes in terms of ``position`` and ``coord``.
_EVALUATION = {
    "tessellation-domain-quads": (
        "quads",
        "equal_spacing, ccw",
        False,
        "position[0] * coord.x + position[3] * coord.y",
    ),
    "tessellation-domain-isolines": (
        "isolines",
        "equal_spacing, ccw",
        False,
        "position[0] * coord.x + position[1] * coord.y",
    ),
    "tessellation-level-input": (
        "triangles",
        "equal_spacing, ccw",
        False,
        "position[0] * OUTER + position[1] * INNER + position[2] * coord.z",
    ),
    "tessellation-patch-input": (
        "triangles",
        "equal_spacing, ccw",
        True,
        "position[0] * coord.x + EXTRA * coord.y",
    ),
}
_BARYCENTRIC = "position[0] * coord.x + position[1] * coord.y + position[2] * coord.z"
# The evaluation probes of the layouts only GLSL and rust-gpu state there, as their GLSL and rust-gpu layouts.
_LAYOUTS = {
    "tessellation-spacing-fractional-odd": (
        "fractional_odd_spacing, ccw",
        "spacing_fraction_odd, vertex_order_ccw",
    ),
    "tessellation-spacing-fractional-even": (
        "fractional_even_spacing, ccw",
        "spacing_fraction_even, vertex_order_ccw",
    ),
    "tessellation-winding-cw": ("equal_spacing, cw", "spacing_equal, vertex_order_cw"),
    "tessellation-point-mode": (
        "equal_spacing, ccw, point_mode",
        "spacing_equal, vertex_order_ccw, point_mode",
    ),
}
# The control probes of the attributes only HLSL, Slang and rust-gpu state on the control stage, as their HLSL
# partitioning and output topology and their rust-gpu execution modes.
_HULL = {
    "tessellation-control-spacing-fractional-odd": (
        "fractional_odd",
        "triangle_cw",
        "spacing_fraction_odd, vertex_order_cw",
    ),
    "tessellation-control-spacing-fractional-even": (
        "fractional_even",
        "triangle_cw",
        "spacing_fraction_even, vertex_order_cw",
    ),
    "tessellation-control-winding-ccw": (
        "integer",
        "triangle_ccw",
        "spacing_equal, vertex_order_ccw",
    ),
    "tessellation-control-point-mode": (
        "integer",
        "point",
        "spacing_equal, point_mode",
    ),
}
NAMES = (
    *_EVALUATION,
    *_LAYOUTS,
    *_HULL,
    "tessellation-patch-output",
    "tessellation-patch-vertices",
)


def _spelled(expression: str, coord: str, outer: str, inner: str, extra: str) -> str:
    return (
        expression.replace("coord", coord)
        .replace("OUTER", outer)
        .replace("INNER", inner)
        .replace("EXTRA", extra)
    )


_GLSL_LEVELS = (
    "    gl_TessLevelOuter[0] = 1.0;\n    gl_TessLevelOuter[1] = 1.0;\n    gl_TessLevelOuter[2] = 1.0;\n"
    "    gl_TessLevelInner[0] = 1.0;\n"
)


def _glsl_evaluation(
    domain: str, layout: str, expression: str, patch: bool
) -> Template:
    extra = "layout(location = 1) patch in vec4 extra;\n" if patch else ""
    position = _spelled(
        expression,
        "gl_TessCoord",
        "gl_TessLevelOuter[0]",
        "gl_TessLevelInner[0]",
        "extra",
    )
    return Template(
        program=f"#version 460\nlayout({_DOMAINS[domain][0]}, {layout}) in;\n"
        f"layout(location = 0) in vec4 position[];\n{extra}"
        f"void main() {{\n    gl_Position = {position};\n}}\n"
    )


def _glsl_control(declarations: str, scale: str, body: str = "") -> Template:
    return Template(
        program="#version 460\nlayout(vertices = 3) out;\nlayout(location = 0) in vec4 position[];\n"
        f"layout(location = 0) out vec4 outPosition[];\n{declarations}void main() {{\n"
        f"    outPosition[gl_InvocationID] = position[gl_InvocationID] * {scale};\n"
        f"{body}{_GLSL_LEVELS}}}\n"
    )


GLSL = {
    **{
        name: _glsl_evaluation(domain, layout, expression, patch)
        for name, (domain, layout, patch, expression) in _EVALUATION.items()
    },
    **{
        name: _glsl_evaluation("triangles", layout, _BARYCENTRIC, False)
        for name, (layout, _) in _LAYOUTS.items()
    },
    **dict.fromkeys(_HULL, ABSENT),
    "tessellation-patch-output": _glsl_control(
        "layout(location = 1) patch out vec4 extra;\n",
        "2.0",
        "    extra = position[0];\n",
    ),
    "tessellation-patch-vertices": _glsl_control("", "float(gl_PatchVerticesIn)"),
}
WEBGL = dict.fromkeys(NAMES, ABSENT)
WGSL = dict.fromkeys(NAMES, ABSENT)
KERNELS = dict.fromkeys(NAMES, ABSENT)


def _hlsl_patch(domain: str, extra: bool) -> str:
    """The control point and the patch-constant structure of a ``domain`` patch, with a per-patch value if ``extra``."""
    edges, inside = _DOMAINS[domain][3]
    members = f"    float edges[{edges}] : SV_TessFactor;\n"
    if inside:
        size = f"[{inside}]" if inside > 1 else ""
        members += f"    float inside{size} : SV_InsideTessFactor;\n"
    if extra:
        members += "    float4 extra : EXTRA;\n"
    return (
        "struct ControlPoint {\n    float4 position : POSITION;\n};\n\n"
        f"struct PatchConstants {{\n{members}}};\n\n"
    )


def _hlsl_domain(domain: str, expression: str, extra: bool) -> str:
    _, name, count, _, size = _DOMAINS[domain]
    inner = "constants.inside" if domain == "triangles" else "constants.inside[0]"
    position = _spelled(
        expression.replace("position[", "patch[").replace("]", "].position"),
        "coord",
        "constants.edges[0]",
        inner,
        "constants.extra",
    )
    return _hlsl_patch(domain, extra) + (
        f'[domain("{name}")]\n'
        f"float4 main(PatchConstants constants, float{size} coord : SV_DomainLocation,\n"
        f"            const OutputPatch<ControlPoint, {count}> patch) : SV_Position {{\n"
        f"    return {position};\n}}\n"
    )


def _hlsl_hull(partitioning: str, topology: str, extra: bool = False) -> str:
    write = "    output.extra = patch[0].position;\n" if extra else ""
    return _hlsl_patch("triangles", extra) + (
        "PatchConstants constants(InputPatch<ControlPoint, 3> patch) {\n    PatchConstants output;\n"
        "    output.edges[0] = 1.0;\n    output.edges[1] = 1.0;\n    output.edges[2] = 1.0;\n"
        f"    output.inside = 1.0;\n{write}    return output;\n}}\n\n"
        f'[domain("tri")]\n[partitioning("{partitioning}")]\n[outputtopology("{topology}")]\n'
        '[outputcontrolpoints(3)]\n[patchconstantfunc("constants")]\n'
        "ControlPoint main(InputPatch<ControlPoint, 3> patch, uint id : SV_OutputControlPointID) {\n"
        "    ControlPoint output;\n    output.position = patch[id].position * 2.0;\n    return output;\n}\n"
    )


# Each HLSL program as its stage, for Slang's stage attribute, and its text.
_HLSL = {
    **{
        name: ("domain", _hlsl_domain(domain, expression, patch))
        for name, (domain, _, patch, expression) in _EVALUATION.items()
    },
    **{
        name: ("hull", _hlsl_hull(partitioning, topology))
        for name, (partitioning, topology, _) in _HULL.items()
    },
    "tessellation-patch-output": ("hull", _hlsl_hull("integer", "triangle_cw", True)),
}


def _hlsl(slang: bool) -> dict[str, Template]:
    """The HLSL programs; Slang's name their stage ahead of the entry point's first attribute."""
    programs = {
        name: Template(
            program=(
                text.replace("[domain(", f'[shader("{stage}")]\n[domain(', 1)
                if slang
                else text
            ),
            entry="main",
        )
        for name, (stage, text) in _HLSL.items()
    }
    return {
        **programs,
        **dict.fromkeys(_LAYOUTS, ABSENT),
        "tessellation-patch-vertices": ABSENT,
    }


HLSL = _hlsl(slang=False)
SLANG = _hlsl(slang=True)
# slangc writes the per-patch value and the control point at the same output location when it targets SPIR-V, which
# its own SPIR-V validation rejects, so the Slang probe of a per-patch output is checked as the HLSL slangc writes.
SLANG["tessellation-patch-output"] = replace(
    SLANG["tessellation-patch-output"], target="hlsl"
)


def _metal(domain: str, expression: str, extra: bool) -> Template:
    _, _, count, _, size = _DOMAINS[domain]
    member = "    float4 extra [[attribute(1)]];\n" if extra else ""
    position = _spelled(
        expression.replace("position[", "patch.points[").replace("]", "].position"),
        "coord",
        "",
        "",
        "patch.extra",
    )
    return Template(
        program="#include <metal_stdlib>\nusing namespace metal;\n\n"
        "struct ControlPoint {\n    float4 position [[attribute(0)]];\n};\n\n"
        f"struct Patch {{\n    patch_control_point<ControlPoint> points;\n{member}}};\n\n"
        f"[[patch({'triangle' if domain == 'triangles' else 'quad'}, {count})]]\n"
        f"vertex float4 probe(Patch patch [[stage_in]], float{size} coord [[position_in_patch]]) {{\n"
        f"    return {position};\n}}\n"
    )


METAL = {
    **dict.fromkeys(NAMES, ABSENT),
    "tessellation-domain-quads": _metal(
        "quads", _EVALUATION["tessellation-domain-quads"][3], False
    ),
    "tessellation-patch-input": _metal(
        "triangles", _EVALUATION["tessellation-patch-input"][3], True
    ),
}


def _rust_program(model: str, params: str, body: str) -> Template:
    return Template(
        program="#![no_std]\nuse spirv_std::glam::*;\nuse spirv_std::spirv;\n\n"
        f"#[spirv({model})]\npub fn main({params}) {{\n{body}}}\n",
        entry="main",
    )


def _rust_evaluation(
    domain: str, modes: str, expression: str, levels: bool
) -> Template:
    count = _DOMAINS[domain][2]
    params = f"#[spirv(tess_coord)] coord: Vec3, position: [Vec4; {count}], "
    if levels:
        params += "#[spirv(tess_level_outer)] outer: [f32; 4], #[spirv(tess_level_inner)] inner: [f32; 2], "
    position = _spelled(expression, "coord", "outer[0]", "inner[0]", "")
    return _rust_program(
        f"tessellation_evaluation({domain}, {modes})",
        params + "#[spirv(position)] out_position: &mut Vec4",
        f"    *out_position = {position};\n",
    )


_RUST_CONTROL = (
    "#[spirv(invocation_id)] id: u32, position: [Vec4; 3], out_position: &mut [Vec4; 3], "
    "#[spirv(tess_level_outer)] outer: &mut [f32; 4], #[spirv(tess_level_inner)] inner: &mut [f32; 2]"
)
_RUST_LEVELS = "    outer[0] = 1.0;\n    outer[1] = 1.0;\n    outer[2] = 1.0;\n    inner[0] = 1.0;\n"


def _rust_control(modes: str, params: str = "", scale: str = "2.0") -> Template:
    return _rust_program(
        f"tessellation_control(output_vertices = 3{modes})",
        params + _RUST_CONTROL,
        f"    out_position[id as usize] = position[id as usize] * {scale};\n{_RUST_LEVELS}",
    )


RUST = {
    **{
        name: _rust_evaluation(
            domain, "spacing_equal, vertex_order_ccw", expression, "OUTER" in expression
        )
        for name, (domain, _, _, expression) in _EVALUATION.items()
    },
    **{
        name: _rust_evaluation("triangles", modes, _BARYCENTRIC, False)
        for name, (_, modes) in _LAYOUTS.items()
    },
    **{name: _rust_control(f", {modes}") for name, (_, _, modes) in _HULL.items()},
    "tessellation-patch-input": ABSENT,
    "tessellation-patch-output": ABSENT,
    "tessellation-patch-vertices": _rust_control(
        "", "#[spirv(patch_vertices)] count: u32, ", "count as f32"
    ),
}


def _crossgl_patch(domain: str, extra: bool) -> str:
    edges, inside = _DOMAINS[domain][3]
    members = f"        float edges[{edges}] @ SV_TessFactor;\n"
    if inside:
        size = f"[{inside}]" if inside > 1 else ""
        members += f"        float inside{size} @ SV_InsideTessFactor;\n"
    if extra:
        members += "        vec4 extra @ EXTRA;\n"
    return (
        "shader Probe {\n    struct ControlPoint {\n        vec4 position @ POSITION;\n    }\n"
        f"    struct PatchConstants {{\n{members}    }}\n"
    )


def _crossgl_domain(domain: str, expression: str, extra: bool) -> Template:
    _, name, count, _, size = _DOMAINS[domain]
    inner = "constants.inside" if domain == "triangles" else "constants.inside[0]"
    position = _spelled(
        expression.replace("position[", "patch[").replace("]", "].position"),
        "coord",
        "constants.edges[0]",
        inner,
        "constants.extra",
    )
    return Template(
        program=_crossgl_patch(domain, extra)
        + "    struct DomainOutput {\n        vec4 position @ SV_Position;\n    }\n"
        "    tessellation_evaluation {\n"
        f"        DomainOutput main(PatchConstants constants, vec{size} coord @ SV_DomainLocation,\n"
        f"                          OutputPatch<ControlPoint, {count}> patch) @domain({name}) {{\n"
        f"            DomainOutput output;\n            output.position = {position};\n"
        "            return output;\n        }\n    }\n}\n"
    )


def _crossgl_hull(partitioning: str, topology: str, extra: bool = False) -> Template:
    write = "            output.extra = patch[0].position;\n" if extra else ""
    return Template(
        program=_crossgl_patch("triangles", extra) + "    tessellation_control {\n"
        "        PatchConstants constants(InputPatch<ControlPoint, 3> patch) {\n"
        "            PatchConstants output;\n            output.edges[0] = 1.0;\n"
        "            output.edges[1] = 1.0;\n            output.edges[2] = 1.0;\n"
        f"            output.inside = 1.0;\n{write}            return output;\n        }}\n"
        "        ControlPoint main(InputPatch<ControlPoint, 3> patch, uint id @ SV_OutputControlPointID)\n"
        f"            @domain(tri)\n            @partitioning({partitioning})\n"
        f"            @outputtopology({topology})\n            @outputcontrolpoints(3)\n"
        "            @patchconstantfunc(constants) {\n"
        "            ControlPoint output;\n            output.position = patch[id].position * 2.0;\n"
        "            return output;\n        }\n    }\n}\n"
    )


def _crossgl_layout(layout: str) -> Template:
    position = _spelled(
        _BARYCENTRIC.replace("position", "inPosition"), "gl_TessCoord", "", "", ""
    )
    return Template(
        program="shader Probe {\n    in vec4 inPosition[] @location(0);\n"
        f"    tessellation_evaluation {{\n        layout(triangles, {layout}) in;\n"
        f"        void main() {{\n            gl_Position = {position};\n        }}\n    }}\n}}\n"
    )


CROSSGL = {
    **{
        name: _crossgl_domain(domain, expression, patch)
        for name, (domain, _, patch, expression) in _EVALUATION.items()
    },
    **{name: _crossgl_layout(layout) for name, (layout, _) in _LAYOUTS.items()},
    **{
        name: _crossgl_hull(partitioning, topology)
        for name, (partitioning, topology, _) in _HULL.items()
    },
    "tessellation-patch-output": _crossgl_hull("integer", "triangle_cw", True),
    "tessellation-patch-vertices": Template(
        program="shader Probe {\n    in vec4 inPosition[] @location(0);\n"
        "    out vec4 outPosition[3] @location(0);\n    tessellation_control {\n"
        "        layout(vertices = 3) out;\n        void main() {\n"
        "            outPosition[gl_InvocationID] = inPosition[gl_InvocationID] * float(gl_PatchVerticesIn);\n"
        + _GLSL_LEVELS.replace("    ", "            ")
        + "        }\n    }\n}\n"
    ),
}

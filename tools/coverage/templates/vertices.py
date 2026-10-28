"""Vertex-index, instance-index, point-size, clip-distance and invariant-position probes: the built-in inputs and
outputs of a vertex shader, in every language with a vertex stage.

Each probe is the vertex probe of :mod:`.graphics` with one built-in added: it adds the index of the vertex or of the
instance to the position it writes, writes the size of a point or one clip distance next to the position, or declares
the position invariant. GLSL and WebGL 2 use ``gl_VertexID``, ``gl_InstanceID``, ``gl_PointSize``,
``gl_ClipDistance`` and ``invariant gl_Position``, HLSL and Slang the ``SV_VertexID``, ``SV_InstanceID`` and
``SV_ClipDistance0`` semantics and Slang also ``SV_PointSize``, Metal the ``[[vertex_id]]``, ``[[instance_id]]``,
``[[point_size]]``, ``[[clip_distance]]`` and ``[[invariant]]`` attributes, WGSL the ``vertex_index``,
``instance_index`` and ``clip_distances`` built-ins and the ``@invariant`` attribute, Rust the ``vertex_index``,
``instance_index``, ``point_size``, ``clip_distance`` and ``invariant`` attributes and CrossGL the GLSL names as
member semantics.

Direct3D 10 and later rasterise every point one pixel wide, and WGSL has no point-size built-in either, so neither
writes one. HLSL has no invariance qualifier, and GLSL ES 3.00 no clip distances. The
kernel languages have no vertex stage. glslang declares every member of ``gl_PerVertex`` in each vertex shader it
compiles, so the ``PointSize`` anchor of a GLSL translation shows only that it is a vertex shader.
"""

from __future__ import annotations

from .base import ABSENT, Template

NAMES = (
    "vertex-index",
    "instance-index",
    "point-size",
    "clip-distance",
    "invariant-position",
)

_GLSL_POSITION = "gl_Position = position * 2.0;"
_GLSL_STATEMENTS = {
    "vertex-index": (
        "gl_Position = position * 2.0 + vec4(float(gl_VertexID), 0.0, 0.0, 0.0);"
    ),
    "instance-index": (
        "gl_Position = position * 2.0 + vec4(float(gl_InstanceID), 0.0, 0.0, 0.0);"
    ),
    "point-size": f"{_GLSL_POSITION}\n    gl_PointSize = 2.0;",
    "clip-distance": f"{_GLSL_POSITION}\n    gl_ClipDistance[0] = position.x;",
    "invariant-position": _GLSL_POSITION,
}


def _glsl(header: str, attribute: str, names: tuple[str, ...]) -> dict[str, Template]:
    def program(name: str) -> str:
        invariant = "invariant gl_Position;\n" if name == "invariant-position" else ""
        return (
            f"{header}{attribute}in vec4 position;\n{invariant}void main() {{\n"
            f"    {_GLSL_STATEMENTS[name]}\n}}\n"
        )

    return {
        name: Template(program=program(name)) if name in names else ABSENT
        for name in NAMES
    }


GLSL = _glsl("#version 460\n", "layout(location = 0) ", NAMES)
WEBGL = _glsl(
    "#version 300 es\n",
    "",
    tuple(name for name in NAMES if name != "clip-distance"),
)

# Each program as the declarations ahead of the entry point, its signature and its body.
_HLSL_INDEX_BODY = (
    "    return position * 2.0 + float4(float({index}), 0.0, 0.0, 0.0);\n"
)
_HLSL_PARTS = {
    "vertex-index": (
        "",
        "float4 main(float4 position : POSITION, uint vid : SV_VertexID) : SV_Position",
        _HLSL_INDEX_BODY.format(index="vid"),
    ),
    "instance-index": (
        "",
        "float4 main(float4 position : POSITION, uint iid : SV_InstanceID) : SV_Position",
        _HLSL_INDEX_BODY.format(index="iid"),
    ),
    "clip-distance": (
        "struct VertexOutput {\n    float4 position : SV_Position;\n    float clip : SV_ClipDistance0;\n};\n\n",
        "VertexOutput main(float4 position : POSITION)",
        "    VertexOutput output;\n    output.position = position * 2.0;\n"
        "    output.clip = position.x;\n    return output;\n",
    ),
}
_SLANG_PARTS = {
    **_HLSL_PARTS,
    "point-size": (
        "struct VertexOutput {\n    float4 position : SV_Position;\n    float size : SV_PointSize;\n};\n\n",
        "VertexOutput main(float4 position : POSITION)",
        "    VertexOutput output;\n    output.position = position * 2.0;\n"
        "    output.size = 2.0;\n    return output;\n",
    ),
}


def _hlsl(
    parts: dict[str, tuple[str, str, str]], attribute: str = ""
) -> dict[str, Template]:
    """The HLSL programs; Slang's name their stage with ``attribute`` on the entry point."""
    return {
        name: (
            Template(
                program=f"{parts[name][0]}{attribute}{parts[name][1]} {{\n{parts[name][2]}}}\n",
                entry="main",
            )
            if name in parts
            else ABSENT
        )
        for name in NAMES
    }


HLSL = _hlsl(_HLSL_PARTS)
SLANG = _hlsl(_SLANG_PARTS, '[shader("vertex")]\n')

_METAL_HEADER = "#include <metal_stdlib>\nusing namespace metal;\n\n"
_METAL_IN = "struct VertexIn {\n    float4 position [[attribute(0)]];\n};\n\n"
_METAL_INDEX = (
    "vertex float4 probe(VertexIn in [[stage_in]], uint {name} [[{attribute}]]) {{\n"
    "    return in.position * 2.0 + float4(float({name}), 0.0, 0.0, 0.0);\n}}\n"
)


def _metal_output(member: str, statement: str) -> str:
    """A Metal program whose vertex function returns its position with ``member`` beside it."""
    return (
        f"struct VertexOut {{\n    float4 position [[position]];\n{member}}};\n\n"
        "vertex VertexOut probe(VertexIn in [[stage_in]]) {\n    VertexOut out;\n"
        f"    out.position = in.position * 2.0;\n{statement}    return out;\n}}\n"
    )


_METAL = {
    "vertex-index": _METAL_INDEX.format(name="vid", attribute="vertex_id"),
    "instance-index": _METAL_INDEX.format(name="iid", attribute="instance_id"),
    "point-size": _metal_output(
        "    float size [[point_size]];\n", "    out.size = 2.0;\n"
    ),
    "clip-distance": _metal_output(
        "    float clip [[clip_distance]] [1];\n", "    out.clip[0] = in.position.x;\n"
    ),
    "invariant-position": (
        "struct VertexOut {\n    float4 position [[position, invariant]];\n};\n\n"
        "vertex VertexOut probe(VertexIn in [[stage_in]]) {\n    VertexOut out;\n"
        "    out.position = in.position * 2.0;\n    return out;\n}\n"
    ),
}
METAL = {
    name: Template(program=_METAL_HEADER + _METAL_IN + text)
    for name, text in _METAL.items()
}

_WGSL_INDEX = (
    "@vertex\nfn main(@location(0) position: vec4<f32>, @builtin({builtin}) {name}: u32) "
    "-> @builtin(position) vec4<f32> {{\n"
    "    return position * 2.0 + vec4<f32>(f32({name}), 0.0, 0.0, 0.0);\n}}\n"
)
WGSL = {
    "vertex-index": Template(
        program=_WGSL_INDEX.format(builtin="vertex_index", name="vid")
    ),
    "instance-index": Template(
        program=_WGSL_INDEX.format(builtin="instance_index", name="iid")
    ),
    "point-size": ABSENT,
    "clip-distance": Template(
        program="enable clip_distances;\n\nstruct VertexOutput {\n    @builtin(position) position: vec4<f32>,\n"
        "    @builtin(clip_distances) clip: array<f32, 1>,\n}\n\n"
        "@vertex\nfn main(@location(0) position: vec4<f32>) -> VertexOutput {\n"
        "    var output: VertexOutput;\n    output.position = position * 2.0;\n"
        "    output.clip[0] = position.x;\n    return output;\n}\n"
    ),
    "invariant-position": Template(
        program="@vertex\nfn main(@location(0) position: vec4<f32>) -> @builtin(position) @invariant vec4<f32> {\n"
        "    return position * 2.0;\n}\n"
    ),
}

_RUST_HEADER = "#![no_std]\nuse spirv_std::glam::*;\nuse spirv_std::spirv;\n\n"
_RUST_POSITION = "#[spirv(position)] out_position: &mut Vec4"
_RUST_INDEX_BODY = (
    "    *out_position = position * 2.0 + Vec4::new({name} as f32, 0.0, 0.0, 0.0);\n"
)
# Each entry point as its parameters after the position input and its body.
_RUST = {
    "vertex-index": (
        f"#[spirv(vertex_index)] vid: u32, {_RUST_POSITION}",
        _RUST_INDEX_BODY.format(name="vid"),
    ),
    "instance-index": (
        f"#[spirv(instance_index)] iid: u32, {_RUST_POSITION}",
        _RUST_INDEX_BODY.format(name="iid"),
    ),
    "point-size": (
        f"{_RUST_POSITION}, #[spirv(point_size)] size: &mut f32",
        "    *out_position = position * 2.0;\n    *size = 2.0;\n",
    ),
    "clip-distance": (
        f"{_RUST_POSITION}, #[spirv(clip_distance)] clip: &mut [f32; 1]",
        "    *out_position = position * 2.0;\n    clip[0] = position.x;\n",
    ),
    "invariant-position": (
        "#[spirv(position, invariant)] out_position: &mut Vec4",
        "    *out_position = position * 2.0;\n",
    ),
}
RUST = {
    name: Template(
        program=f"{_RUST_HEADER}#[spirv(vertex)]\npub fn main(position: Vec4, {params}) {{\n{body}}}\n",
        entry="main",
    )
    for name, (params, body) in _RUST.items()
}

KERNELS = dict.fromkeys(NAMES, ABSENT)

_CROSSGL_INDEX = (
    "output.position = input.position * 2.0 + vec4(float(input.{name}), 0.0, 0.0, 0.0);"
)
# Each probe as its extra input member, its output members and its body.
_CROSSGL_PARTS = {
    "vertex-index": (
        "        int vid @ gl_VertexID;\n",
        "        vec4 position @ gl_Position;\n",
        _CROSSGL_INDEX.format(name="vid"),
    ),
    "instance-index": (
        "        int iid @ gl_InstanceID;\n",
        "        vec4 position @ gl_Position;\n",
        _CROSSGL_INDEX.format(name="iid"),
    ),
    "point-size": (
        "",
        "        vec4 position @ gl_Position;\n        float size @ gl_PointSize;\n",
        "output.position = input.position * 2.0;\n            output.size = 2.0;",
    ),
    "clip-distance": (
        "",
        "        vec4 position @ gl_Position;\n        float clip @ gl_ClipDistance;\n",
        "output.position = input.position * 2.0;\n            output.clip = input.position.x;",
    ),
    "invariant-position": (
        "",
        "        vec4 position @ gl_Position @ invariant;\n",
        "output.position = input.position * 2.0;",
    ),
}
CROSSGL = {
    name: Template(
        program="shader Probe {\n    struct VertexInput {\n        vec4 position;\n"
        f"{inputs}    }}\n    struct VertexOutput {{\n{outputs}    }}\n"
        "    vertex {\n        VertexOutput main(VertexInput input) {\n"
        f"            VertexOutput output;\n            {body}\n"
        "            return output;\n        }\n    }\n}\n"
    )
    for name, (inputs, outputs, body) in _CROSSGL_PARTS.items()
}

"""Draw-index, base-vertex, base-instance and cull-distance probes: the draw parameters a vertex shader reads and the
cull distances it writes, in every language that has them.

Each probe is the vertex probe of :mod:`.vertices` with one built-in added: it adds the index of the draw, the base
vertex or the base instance to the position it writes, or writes one cull distance next to the position. GLSL uses
``gl_DrawID``, ``gl_BaseVertex``, ``gl_BaseInstance`` and ``gl_CullDistance``, HLSL and Slang the
``SV_StartVertexLocation``, ``SV_StartInstanceLocation`` and ``SV_CullDistance0`` semantics and Slang also
``SV_DrawIndex``, Metal the ``[[base_vertex]]`` and ``[[base_instance]]`` attributes, Rust the ``draw_index``,
``base_vertex``, ``base_instance`` and ``cull_distance`` attributes and CrossGL the GLSL names as member semantics.

HLSL has no draw index and Metal neither a draw index nor cull distances. GLSL ES 3.00 and WGSL have none of the four,
and the kernel languages have no vertex stage.
"""

from __future__ import annotations

from .base import ABSENT, Template
from .vertices import _METAL_HEADER, _METAL_IN, _RUST_HEADER, _RUST_POSITION

NAMES = ("draw-index", "base-vertex", "base-instance", "cull-distance")
_INDICES = ("draw-index", "base-vertex", "base-instance")

_GLSL_INDEX = "gl_Position = position * 2.0 + vec4(float({name}), 0.0, 0.0, 0.0);"
_GLSL_STATEMENTS = {
    "draw-index": _GLSL_INDEX.format(name="gl_DrawID"),
    "base-vertex": _GLSL_INDEX.format(name="gl_BaseVertex"),
    "base-instance": _GLSL_INDEX.format(name="gl_BaseInstance"),
    "cull-distance": "gl_Position = position * 2.0;\n    gl_CullDistance[0] = position.x;",
}
GLSL = {
    name: Template(
        program="#version 460\nlayout(location = 0) in vec4 position;\nvoid main() {\n"
        f"    {statement}\n}}\n"
    )
    for name, statement in _GLSL_STATEMENTS.items()
}
WEBGL = dict.fromkeys(NAMES, ABSENT)
WGSL = dict.fromkeys(NAMES, ABSENT)
KERNELS = dict.fromkeys(NAMES, ABSENT)

_HLSL_INDEX = (
    "float4 main(float4 position : POSITION, {parameter}) : SV_Position {{\n"
    "    return position * 2.0 + float4(float(base), 0.0, 0.0, 0.0);\n}}\n"
)
# slangc lowers a scalar ``SV_CullDistance0`` to a scalar ``CullDistance`` variable, which Vulkan rejects, so the Slang
# probe writes the one-element array instead.
_HLSL_CULL = (
    "struct VertexOutput {{\n    float4 position : SV_Position;\n    {member};\n}};\n\n"
    "{attribute}VertexOutput main(float4 position : POSITION) {{\n    VertexOutput output;\n"
    "    output.position = position * 2.0;\n    output.{cull} = position.x;\n    return output;\n}}\n"
)
_HLSL_PARAMETERS = {
    "base-vertex": "int base : SV_StartVertexLocation",
    "base-instance": "uint base : SV_StartInstanceLocation",
}


def _hlsl(
    parameters: dict[str, str], cull: tuple[str, str], attribute: str = ""
) -> dict[str, Template]:
    """The HLSL programs; Slang's name their stage with ``attribute`` on the entry point."""
    table = {
        name: (
            Template(
                program=attribute + _HLSL_INDEX.format(parameter=parameters[name]),
                entry="main",
            )
            if name in parameters
            else ABSENT
        )
        for name in _INDICES
    }
    table["cull-distance"] = Template(
        program=_HLSL_CULL.format(member=cull[0], cull=cull[1], attribute=attribute),
        entry="main",
    )
    return table


HLSL = _hlsl(_HLSL_PARAMETERS, ("float cull : SV_CullDistance0", "cull"))
SLANG = _hlsl(
    {**_HLSL_PARAMETERS, "draw-index": "uint base : SV_DrawIndex"},
    ("float cull[1] : SV_CullDistance", "cull[0]"),
    '[shader("vertex")]\n',
)

_METAL_INDEX = (
    "vertex float4 probe(VertexIn in [[stage_in]], uint base [[{attribute}]]) {{\n"
    "    return in.position * 2.0 + float4(float(base), 0.0, 0.0, 0.0);\n}}\n"
)
METAL = {
    "draw-index": ABSENT,
    "base-vertex": Template(
        program=_METAL_HEADER + _METAL_IN + _METAL_INDEX.format(attribute="base_vertex")
    ),
    "base-instance": Template(
        program=_METAL_HEADER
        + _METAL_IN
        + _METAL_INDEX.format(attribute="base_instance")
    ),
    "cull-distance": ABSENT,
}

_RUST_INDEX_BODY = (
    "    *out_position = position * 2.0 + Vec4::new(base as f32, 0.0, 0.0, 0.0);\n"
)
# Each entry point as its parameters after the position input and its body.
_RUST = {
    "draw-index": (
        f"#[spirv(draw_index)] base: u32, {_RUST_POSITION}",
        _RUST_INDEX_BODY,
    ),
    "base-vertex": (
        f"#[spirv(base_vertex)] base: u32, {_RUST_POSITION}",
        _RUST_INDEX_BODY,
    ),
    "base-instance": (
        f"#[spirv(base_instance)] base: u32, {_RUST_POSITION}",
        _RUST_INDEX_BODY,
    ),
    "cull-distance": (
        f"{_RUST_POSITION}, #[spirv(cull_distance)] cull: &mut [f32; 1]",
        "    *out_position = position * 2.0;\n    cull[0] = position.x;\n",
    ),
}
RUST = {
    name: Template(
        program=f"{_RUST_HEADER}#[spirv(vertex)]\npub fn main(position: Vec4, {params}) {{\n{body}}}\n",
        entry="main",
    )
    for name, (params, body) in _RUST.items()
}

_CROSSGL_INDEX = (
    "output.position = input.position * 2.0 + vec4(float(input.base), 0.0, 0.0, 0.0);"
)
# Each probe as its extra input member, its output members and its body.
_CROSSGL_PARTS = {
    "draw-index": (
        "        int base @ gl_DrawID;\n",
        "        vec4 position @ gl_Position;\n",
        _CROSSGL_INDEX,
    ),
    "base-vertex": (
        "        int base @ gl_BaseVertex;\n",
        "        vec4 position @ gl_Position;\n",
        _CROSSGL_INDEX,
    ),
    "base-instance": (
        "        int base @ gl_BaseInstance;\n",
        "        vec4 position @ gl_Position;\n",
        _CROSSGL_INDEX,
    ),
    "cull-distance": (
        "",
        "        vec4 position @ gl_Position;\n        float cull @ gl_CullDistance;\n",
        "output.position = input.position * 2.0;\n            output.cull = input.position.x;",
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

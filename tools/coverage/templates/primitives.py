"""Geometry-stage probes: the primitives a geometry stage reads and emits, its instances and its output streams.

Each probe is the geometry probe of :mod:`.graphics`, a stage that emits a point from a point, with one part changed:
it reads a line, a triangle or either with adjacency, emits a two-vertex line strip or a three-vertex triangle strip,
runs two instances that scale their point by the instance number, or emits a second vertex on stream 1 that carries
an extra output. GLSL states the primitives and the instance count in its ``layout`` qualifiers and the stream in the
output's ``stream`` qualifier. HLSL and Slang state them in the input array's primitive type, the output stream's
type, the ``[instance(n)]`` attribute with ``SV_GSInstanceID``, and a second ``PointStream`` parameter. rust-gpu
states them in the ``geometry`` execution modes and the ``invocation_id`` built-in. CrossGL uses GLSL's layouts and
writes the stream qualifier as an ``@stream`` attribute, as its GLSL importer does.

rust-gpu has no ``Stream`` decoration, so every output it writes belongs to stream 0. Metal, WGSL and GLSL ES 3.00
have no geometry stage, and the kernel languages have no graphics stages.
"""

from __future__ import annotations

from dataclasses import replace

from .base import ABSENT, Template

# Each input primitive as its GLSL layout, HLSL array prefix, rust-gpu execution mode and vertex count.
_INPUTS = {
    "points": ("points", "point", "input_points", 1),
    "lines": ("lines", "line", "input_lines", 2),
    "triangles": ("triangles", "triangle", "triangles", 3),
    "lines-adjacency": ("lines_adjacency", "lineadj", "input_lines_adjacency", 4),
    "triangles-adjacency": (
        "triangles_adjacency",
        "triangleadj",
        "input_triangles_adjacency",
        6,
    ),
}
# Each output primitive as its GLSL layout, HLSL stream type and rust-gpu execution mode.
_OUTPUTS = {
    "points": ("points", "PointStream", "output_points"),
    "line-strip": ("line_strip", "LineStream", "output_line_strip"),
    "triangle-strip": ("triangle_strip", "TriangleStream", "output_triangle_strip"),
}
# Each primitive probe as its input primitive, its output primitive and the positions of the vertices it emits.
_PRIMITIVES = {
    "geometry-input-lines": ("lines", "points", ("position[0] + position[1]",)),
    "geometry-input-triangles": (
        "triangles",
        "points",
        ("position[0] + position[1] + position[2]",),
    ),
    "geometry-input-lines-adjacency": (
        "lines-adjacency",
        "points",
        ("position[1] + position[2] + position[3]",),
    ),
    "geometry-input-triangles-adjacency": (
        "triangles-adjacency",
        "points",
        ("position[0] + position[2] + position[5]",),
    ),
    "geometry-output-line-strip": (
        "points",
        "line-strip",
        ("position[0]", "position[0] * 2.0"),
    ),
    "geometry-output-triangle-strip": (
        "points",
        "triangle-strip",
        ("position[0]", "position[0] * 2.0", "position[0] * 3.0"),
    ),
}
NAMES = (*_PRIMITIVES, "geometry-invocations", "geometry-streams")


def _glsl_program(layouts: str, body: str) -> Template:
    return Template(
        program=f"#version 460\n{layouts}layout(location = 0) in vec4 position[];\n"
        f"void main() {{\n{body}}}\n"
    )


def _glsl(name: str) -> Template:
    source, target, positions = _PRIMITIVES[name]
    body = "".join(
        f"    gl_Position = {position};\n    EmitVertex();\n" for position in positions
    )
    return _glsl_program(
        f"layout({_INPUTS[source][0]}) in;\n"
        f"layout({_OUTPUTS[target][0]}, max_vertices = {len(positions)}) out;\n",
        body + "    EndPrimitive();\n",
    )


GLSL = {
    **{name: _glsl(name) for name in _PRIMITIVES},
    "geometry-invocations": _glsl_program(
        "layout(points, invocations = 2) in;\nlayout(points, max_vertices = 1) out;\n",
        "    gl_Position = position[0] * float(gl_InvocationID + 1);\n"
        "    EmitVertex();\n    EndPrimitive();\n",
    ),
    "geometry-streams": _glsl_program(
        "layout(points) in;\nlayout(points, max_vertices = 2) out;\n"
        "layout(stream = 1, location = 0) out vec4 extra;\n",
        "    gl_Position = position[0];\n    EmitStreamVertex(0);\n    EndStreamPrimitive(0);\n"
        "    extra = position[0] * 2.0;\n    EmitStreamVertex(1);\n    EndStreamPrimitive(1);\n",
    ),
}
WEBGL = dict.fromkeys(NAMES, ABSENT)
WGSL = dict.fromkeys(NAMES, ABSENT)
METAL = dict.fromkeys(NAMES, ABSENT)
KERNELS = dict.fromkeys(NAMES, ABSENT)

_HLSL_OUTPUT = "struct GeometryOut {\n    float4 position : SV_Position;\n};\n\n"


def _hlsl_entry(source: str, target: str, positions: tuple[str, ...]) -> str:
    """The entry point of a primitive probe, after its attributes."""
    _, prefix, _, count = _INPUTS[source]
    emitted = "".join(
        f"    output.position = {position};\n    stream.Append(output);\n"
        for position in positions
    )
    return (
        f"[maxvertexcount({len(positions)})]\n"
        f"void main({prefix} float4 position[{count}] : POSITION, "
        f"inout {_OUTPUTS[target][1]}<GeometryOut> stream) {{\n"
        f"    GeometryOut output;\n{emitted}    stream.RestartStrip();\n}}\n"
    )


# Each program as the declarations ahead of the entry point's attributes, and the attributes and entry point.
_HLSL = {
    **{
        name: (_HLSL_OUTPUT, _hlsl_entry(*parts)) for name, parts in _PRIMITIVES.items()
    },
    "geometry-invocations": (
        _HLSL_OUTPUT,
        "[instance(2)]\n[maxvertexcount(1)]\n"
        "void main(point float4 position[1] : POSITION, uint id : SV_GSInstanceID,\n"
        "          inout PointStream<GeometryOut> stream) {\n"
        "    GeometryOut output;\n    output.position = position[0] * float(id + 1);\n"
        "    stream.Append(output);\n    stream.RestartStrip();\n}\n",
    ),
    "geometry-streams": (
        _HLSL_OUTPUT + "struct ExtraOut {\n    float4 extra : TEXCOORD0;\n};\n\n",
        "[maxvertexcount(2)]\n"
        "void main(point float4 position[1] : POSITION, inout PointStream<GeometryOut> stream,\n"
        "          inout PointStream<ExtraOut> extras) {\n"
        "    GeometryOut output;\n    output.position = position[0];\n"
        "    stream.Append(output);\n    stream.RestartStrip();\n"
        "    ExtraOut extra;\n    extra.extra = position[0] * 2.0;\n"
        "    extras.Append(extra);\n    extras.RestartStrip();\n}\n",
    ),
}


def _hlsl(stage: str = "") -> dict[str, Template]:
    """The HLSL programs; Slang's name their stage with ``stage`` ahead of the entry point's attributes."""
    return {
        name: Template(program=types + stage + entry, entry="main")
        for name, (types, entry) in _HLSL.items()
    }


HLSL = _hlsl()
SLANG = _hlsl('[shader("geometry")]\n')
# slangc aborts with an internal error when it writes a second stream as SPIR-V, so the Slang probe of streams is
# checked as the HLSL slangc writes for it, which DXC compiles.
SLANG["geometry-streams"] = replace(SLANG["geometry-streams"], target="hlsl")

_RUST_EMIT = "    unsafe {\n        spirv_std::arch::emit_vertex();\n    }\n"
_RUST_END = "    unsafe {\n        spirv_std::arch::end_primitive();\n    }\n"
_RUST_OUTPUT = "#[spirv(position)] out_position: &mut Vec4"


def _rust_program(modes: str, params: str, body: str) -> Template:
    return Template(
        program="#![no_std]\nuse spirv_std::glam::*;\nuse spirv_std::spirv;\n\n"
        f"#[spirv(geometry({modes}))]\npub fn main({params}) {{\n{body}}}\n",
        entry="main",
    )


def _rust(name: str) -> Template:
    source, target, positions = _PRIMITIVES[name]
    _, _, mode, count = _INPUTS[source]
    emitted = len(positions)
    return _rust_program(
        f"{mode} = {count}, {_OUTPUTS[target][2]} = {emitted}, output_vertices = {emitted}",
        f"position: [Vec4; {count}], {_RUST_OUTPUT}",
        "".join(
            f"    *out_position = {position};\n{_RUST_EMIT}" for position in positions
        )
        + _RUST_END,
    )


RUST = {
    **{name: _rust(name) for name in _PRIMITIVES},
    "geometry-invocations": _rust_program(
        "input_points = 1, output_points = 1, output_vertices = 1, invocations = 2",
        f"#[spirv(invocation_id)] id: u32, position: [Vec4; 1], {_RUST_OUTPUT}",
        f"    *out_position = position[0] * (id + 1) as f32;\n{_RUST_EMIT}{_RUST_END}",
    ),
    "geometry-streams": ABSENT,
}


def _crossgl_program(
    count: int, layouts: tuple[str, str], body: str, outputs: str = ""
) -> Template:
    return Template(
        program=f"shader Probe {{\n    in vec4 inPosition[{count}] @location(0);\n{outputs}"
        "    geometry {\n"
        + "".join(f"        {layout}\n" for layout in layouts)
        + f"        void main() {{\n{body}        }}\n    }}\n}}\n"
    )


def _crossgl(name: str) -> Template:
    source, target, positions = _PRIMITIVES[name]
    body = "".join(
        f"            gl_Position = {position.replace('position', 'inPosition')};\n"
        "            EmitVertex();\n"
        for position in positions
    )
    return _crossgl_program(
        _INPUTS[source][3],
        (
            f"layout({_INPUTS[source][0]}) in;",
            f"layout({_OUTPUTS[target][0]}, max_vertices = {len(positions)}) out;",
        ),
        body + "            EndPrimitive();\n",
    )


CROSSGL = {
    **{name: _crossgl(name) for name in _PRIMITIVES},
    "geometry-invocations": _crossgl_program(
        1,
        (
            "layout(points, invocations = 2) in;",
            "layout(points, max_vertices = 1) out;",
        ),
        "            gl_Position = inPosition[0] * float(gl_InvocationID + 1);\n"
        "            EmitVertex();\n            EndPrimitive();\n",
    ),
    "geometry-streams": _crossgl_program(
        1,
        ("layout(points) in;", "layout(points, max_vertices = 2) out;"),
        "            gl_Position = inPosition[0];\n            EmitStreamVertex(0);\n"
        "            EndStreamPrimitive(0);\n            extra = inPosition[0] * 2.0;\n"
        "            EmitStreamVertex(1);\n            EndStreamPrimitive(1);\n",
        outputs="    out vec4 extra @location(0) @stream(1);\n",
    ),
}

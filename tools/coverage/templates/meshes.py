"""Mesh-stage probes: the primitives a mesh stage emits, the per-primitive values it writes and the payload it reads.

Each probe is the mesh probe of :mod:`.graphics`, a stage that writes one triangle, with one part changed: it emits a
point or a line instead, writes a per-primitive output, culls its primitive, writes the primitive's ID, layer or
viewport, or reads the payload a task stage hands it into the position of its second vertex. GLSL states the topology
in its ``layout`` qualifier, writes the indices through ``gl_Primitive*IndicesEXT`` and the per-primitive built-ins
through ``gl_MeshPrimitivesEXT``, declares a per-primitive output ``perprimitiveEXT`` and the payload
``taskPayloadSharedEXT``. HLSL and Slang state the topology in ``[outputtopology]`` and the index array's element
type, write the per-primitive values through an ``out primitives`` array of a structure with semantics, and read the
payload through an ``in payload`` parameter. Metal states the topology and the primitive structure in the ``mesh``
type, the built-ins as attributes of the primitive structure, and the payload as an ``object_data`` reference.
rust-gpu states the topology in the ``mesh_ext`` execution modes, and the built-ins, per-primitive outputs and payload
as attributed parameters. CrossGL writes them as its HLSL importer does, with the attributes after the parameters as
in the mesh stage probe of :mod:`..crossgl`.

HLSL has no point topology: DXC accepts only lines and triangles. WGSL, GLSL ES 3.00 and the kernel languages have no
mesh stage.
"""

from __future__ import annotations

from .base import ABSENT, Template

# Each topology as its GLSL layout and index built-in, HLSL topology, index element type and index array name, Metal
# topology, rust-gpu execution mode, index built-in and index element type, and the vertex count. The index values
# are 0, (0, 1) and (0, 1, 2).
_TOPOLOGIES = {
    "points": (
        "points",
        "gl_PrimitivePointIndicesEXT",
        "point",
        ("uint", "uint", "u32"),
        "points",
        "point",
        "output_points",
        "primitive_point_indices_ext",
        1,
    ),
    "lines": (
        "lines",
        "gl_PrimitiveLineIndicesEXT",
        "line",
        ("uint2", "uvec2", "UVec2"),
        "lines",
        "line",
        "output_lines_ext",
        "primitive_line_indices_ext",
        2,
    ),
    "triangles": (
        "triangles",
        "gl_PrimitiveTriangleIndicesEXT",
        "triangle",
        ("uint3", "uvec3", "UVec3"),
        "tris",
        "triangle",
        "output_triangles_ext",
        "primitive_triangle_indices_ext",
        3,
    ),
}
_POSITIONS = ("0.0, 0.0, 0.0, 1.0", "1.0, 0.0, 0.0, 1.0", "0.0, 1.0, 0.0, 1.0")
# The x coordinate of the second vertex in a probe that reads the payload, in each language.
_PAYLOAD_X = {
    "glsl": "float(payload.value)",
    "hlsl": "float(payload.value)",
    "metal": "float(payload.value)",
    "rust": "payload.value as f32",
}
# Each per-primitive built-in probe as its GLSL member, HLSL and Slang semantic, Metal attribute, rust-gpu built-in,
# CrossGL semantic, member name and the value it writes. rust-gpu's SPIR-V validation rejects an unsigned
# per-primitive ID, layer or viewport, so its arrays are of ``i32``.
_BUILTINS = {
    "mesh-cull-primitive": (
        "gl_CullPrimitiveEXT",
        "SV_CullPrimitive",
        "primitive_culled",
        "cull_primitive_ext",
        "SV_CullPrimitive",
        "culled",
        "true",
    ),
    "mesh-primitive-id": (
        "gl_PrimitiveID",
        "SV_PrimitiveID",
        "primitive_id",
        "primitive_id",
        "gl_PrimitiveID",
        "id",
        "7",
    ),
    "mesh-layer": (
        "gl_Layer",
        "SV_RenderTargetArrayIndex",
        "render_target_array_index",
        "layer",
        "gl_Layer",
        "layer",
        "1",
    ),
    "mesh-viewport-index": (
        "gl_ViewportIndex",
        "SV_ViewportArrayIndex",
        "viewport_array_index",
        "viewport_index",
        "gl_ViewportIndex",
        "viewport",
        "1",
    ),
}
NAMES = (
    "mesh-output-points",
    "mesh-output-lines",
    "mesh-per-primitive-output",
    *_BUILTINS,
    "mesh-task-payload",
)
_COLOR = "1.0, 0.0, 0.0, 1.0"


def _index(topology: str, language: str) -> str:
    """The index value of the one primitive of ``topology`` in ``language``."""
    count = _TOPOLOGIES[topology][8]
    if count == 1:
        return "0u" if language in ("glsl", "crossgl") else "0"
    vector = {
        "glsl": f"uvec{count}",
        "crossgl": f"uvec{count}",
        "hlsl": f"uint{count}",
        "rust": f"UVec{count}::new",
    }
    return f"{vector[language]}({', '.join(str(i) for i in range(count))})"


def _glsl(
    topology: str = "triangles",
    declarations: str = "",
    body: str = "",
    second: str = "1.0",
) -> Template:
    layout, indices, *_, count = _TOPOLOGIES[topology]
    vertices = "".join(
        f"    gl_MeshVerticesEXT[{i}].gl_Position = vec4({position});\n"
        for i, position in enumerate(_POSITIONS[:count])
    ).replace("vec4(1.0, ", f"vec4({second}, ")
    return Template(
        program="#version 460\n#extension GL_EXT_mesh_shader : require\n"
        "layout(local_size_x = 1) in;\n"
        f"layout({layout}, max_vertices = {count}, max_primitives = 1) out;\n"
        f"{declarations}void main() {{\n    SetMeshOutputsEXT({count}, 1);\n{vertices}"
        f"    {indices}[0] = {_index(topology, 'glsl')};\n{body}}}\n"
    )


GLSL = {
    "mesh-output-points": _glsl("points"),
    "mesh-output-lines": _glsl("lines"),
    "mesh-per-primitive-output": _glsl(
        declarations="layout(location = 0) perprimitiveEXT out vec4 color[];\n",
        body=f"    color[0] = vec4({_COLOR});\n",
    ),
    **{
        name: _glsl(body=f"    gl_MeshPrimitivesEXT[0].{member} = {value};\n")
        for name, (member, *_, value) in _BUILTINS.items()
    },
    "mesh-task-payload": _glsl(
        declarations="struct Payload {\n    uint value;\n};\n"
        "taskPayloadSharedEXT Payload payload;\n",
        second=_PAYLOAD_X["glsl"],
    ),
}
WEBGL = dict.fromkeys(NAMES, ABSENT)
WGSL = dict.fromkeys(NAMES, ABSENT)
KERNELS = dict.fromkeys(NAMES, ABSENT)


def _hlsl_program(
    stage: str,
    topology: str = "triangles",
    types: str = "",
    params: str = "",
    body: str = "",
    second: str = "1.0",
) -> Template:
    _, _, name, (element, *_), array, *_, count = _TOPOLOGIES[topology]
    vertices = "".join(
        f"    verts[{i}].position = float4({position});\n"
        for i, position in enumerate(_POSITIONS[:count])
    ).replace("float4(1.0, ", f"float4({second}, ")
    return Template(
        program="struct Vertex {\n    float4 position : SV_Position;\n};\n\n"
        f'{types}{stage}[numthreads(1, 1, 1)]\n[outputtopology("{name}")]\n'
        f"void main(out vertices Vertex verts[{count}], out indices {element} {array}[1]{params}) {{\n"
        f"    SetMeshOutputCounts({count}, 1);\n{vertices}"
        f"    {array}[0] = {_index(topology, 'hlsl')};\n{body}}}\n",
        entry="main",
    )


def _hlsl_primitive(stage: str, member: str, field: str, value: str) -> Template:
    """A probe that writes ``value`` to the ``field`` of its one primitive, declared as ``member``."""
    return _hlsl_program(
        stage,
        types=f"struct Primitive {{\n    {member};\n}};\n\n",
        params=", out primitives Primitive prims[1]",
        body=f"    prims[0].{field} = {value};\n",
    )


def _hlsl(stage: str = "") -> dict[str, Template]:
    """The HLSL programs; Slang's name their stage with ``stage`` ahead of the entry point's attributes."""
    return {
        "mesh-output-points": _hlsl_program(stage, "points"),
        "mesh-output-lines": _hlsl_program(stage, "lines"),
        "mesh-per-primitive-output": _hlsl_primitive(
            stage, "float4 color : COLOR0", "color", f"float4({_COLOR})"
        ),
        **{
            name: _hlsl_primitive(
                stage,
                f"{'bool' if field == 'culled' else 'uint'} {field} : {semantic}",
                field,
                value,
            )
            for name, (_, semantic, *_, field, value) in _BUILTINS.items()
        },
        "mesh-task-payload": _hlsl_program(
            stage,
            types="struct Payload {\n    uint value;\n};\n\n",
            params=", in payload Payload payload",
            second=_PAYLOAD_X["hlsl"],
        ),
    }


# DXC accepts only line and triangle topologies; slangc writes a point topology too.
HLSL = {**_hlsl(), "mesh-output-points": ABSENT}
SLANG = _hlsl('[shader("mesh")]\n')


def _metal(
    topology: str = "triangles",
    types: str = "",
    primitive: str = "",
    params: str = "",
    body: str = "",
    second: str = "1.0",
) -> Template:
    *_, name, _, _, count = _TOPOLOGIES[topology]
    if primitive:
        types += f"struct Primitive {{\n    {primitive};\n}};\n\n"
    vertices = "".join(
        f"    output.set_vertex({i}, Vertex{{float4({position})}});\n"
        for i, position in enumerate(_POSITIONS[:count])
    ).replace("float4(1.0, ", f"float4({second}, ")
    indices = "".join(f"    output.set_index({i}, {i});\n" for i in range(count))
    return Template(
        program="#include <metal_stdlib>\nusing namespace metal;\n\n"
        f"struct Vertex {{\n    float4 position [[position]];\n}};\n\n{types}"
        f"[[mesh]] void probe(mesh<Vertex, {'Primitive' if primitive else 'void'}, {count}, 1, "
        f"topology::{name}> output{params}) {{\n"
        f"    output.set_primitive_count(1);\n{vertices}{indices}{body}}}\n"
    )


METAL = {
    "mesh-output-points": _metal("points"),
    "mesh-output-lines": _metal("lines"),
    "mesh-per-primitive-output": _metal(
        primitive="float4 color [[user(locn0)]]",
        body=f"    output.set_primitive(0, Primitive{{float4({_COLOR})}});\n",
    ),
    **{
        name: _metal(
            primitive=f"{'bool' if field == 'culled' else 'uint'} {field} [[{attribute}]]",
            body=f"    output.set_primitive(0, Primitive{{{value}}});\n",
        )
        for name, (_, _, attribute, *_, field, value) in _BUILTINS.items()
    },
    "mesh-task-payload": _metal(
        types="struct Payload {\n    uint value;\n};\n\n",
        params=",\n                    const object_data Payload& payload [[payload]]",
        second=_PAYLOAD_X["metal"],
    ),
}


def _rust(
    topology: str = "triangles",
    types: str = "",
    params: str = "",
    body: str = "",
    second: str = "1.0",
) -> Template:
    *_, mode, indices, count = _TOPOLOGIES[topology]
    element = _TOPOLOGIES[topology][3][2]
    vertices = "".join(
        f"    positions[{i}] = Vec4::new({position});\n"
        for i, position in enumerate(_POSITIONS[:count])
    ).replace("Vec4::new(1.0, ", f"Vec4::new({second}, ")
    return Template(
        program="#![no_std]\nuse spirv_std::glam::*;\nuse spirv_std::spirv;\n\n"
        f"{types}#[spirv(mesh_ext(threads(1), output_vertices = {count}, output_primitives_ext = 1, {mode}))]\n"
        f"pub fn main(#[spirv(position)] positions: &mut [Vec4; {count}], "
        f"#[spirv({indices})] indices: &mut [{element}; 1]{params}) {{\n"
        f"    unsafe {{\n        spirv_std::arch::set_mesh_outputs_ext({count}, 1);\n    }}\n"
        f"{vertices}    indices[0] = {_index(topology, 'rust')};\n{body}}}\n",
        entry="main",
    )


RUST = {
    "mesh-output-points": _rust("points"),
    "mesh-output-lines": _rust("lines"),
    "mesh-per-primitive-output": _rust(
        params=", #[spirv(per_primitive_ext)] color: &mut [Vec4; 1]",
        body=f"    color[0] = Vec4::new({_COLOR});\n",
    ),
    **{
        name: _rust(
            params=f", #[spirv({builtin}, per_primitive_ext)] {field}: "
            f"&mut [{'bool' if field == 'culled' else 'i32'}; 1]",
            body=f"    {field}[0] = {value};\n",
        )
        for name, (_, _, _, builtin, _, field, value) in _BUILTINS.items()
    },
    "mesh-task-payload": _rust(
        types="pub struct Payload {\n    pub value: u32,\n}\n\n",
        params=", #[spirv(task_payload_workgroup_ext)] payload: &Payload",
        second=_PAYLOAD_X["rust"],
    ),
}


def _crossgl(
    topology: str = "triangles",
    types: str = "",
    params: str = "",
    body: str = "",
    second: str = "1.0",
) -> Template:
    _, _, name, (_, element, _), array, *_, count = _TOPOLOGIES[topology]
    vertices = "".join(
        f"            verts[{i}].position = vec4({position});\n"
        for i, position in enumerate(_POSITIONS[:count])
    ).replace("vec4(1.0, ", f"vec4({second}, ")
    return Template(
        program="shader Probe {\n    struct Vertex {\n        vec4 position @ gl_Position;\n    }\n"
        f"{types}    mesh {{\n"
        f"        void main(@ vertices out Vertex verts[{count}], @ indices out {element} {array}[1]{params})\n"
        f"            @numthreads(1, 1, 1)\n            @outputtopology({name}) {{\n"
        f"            SetMeshOutputCounts({count}, 1);\n{vertices}"
        f"            {array}[0] = {_index(topology, 'crossgl')};\n{body}        }}\n    }}\n}}\n"
    )


def _crossgl_primitive(member: str, field: str, value: str) -> Template:
    return _crossgl(
        types=f"    struct Primitive {{\n        {member};\n    }}\n",
        params=", @ primitives out Primitive prims[1]",
        body=f"            prims[0].{field} = {value};\n",
    )


CROSSGL = {
    "mesh-output-points": _crossgl("points"),
    "mesh-output-lines": _crossgl("lines"),
    "mesh-per-primitive-output": _crossgl_primitive(
        "vec4 color @ Color0", "color", f"vec4({_COLOR})"
    ),
    **{
        name: _crossgl_primitive(
            f"{'bool' if field == 'culled' else 'uint'} {field} @ {semantic}",
            field,
            value,
        )
        for name, (*_, semantic, field, value) in _BUILTINS.items()
    },
    "mesh-task-payload": _crossgl(
        types="    struct Payload {\n        uint value;\n    }\n",
        params=", @ mesh_payload in Payload payload",
        second="float(payload.value)",
    ),
}

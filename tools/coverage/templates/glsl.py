"""Probe templates for GLSL."""

from __future__ import annotations

from . import (
    abstractions,
    aggregates,
    bindings,
    blocks,
    booleans,
    columns,
    constructors,
    draws,
    edges,
    elements,
    extensions,
    formats,
    forms,
    fragments,
    helpers,
    identifiers,
    idioms,
    images,
    initializers,
    integers,
    interfaces,
    interlocks,
    layouts,
    lexical,
    meshes,
    orientation,
    packing,
    pipeline,
    portability,
    primitives,
    qualifiers,
    rasterization,
    rays,
    records,
    relations,
    rows,
    samplers,
    specializations,
    storage,
    subgroups,
    tasks,
    tessellation,
    textures,
    values,
    vectors,
    vertices,
    workgroups,
)
from .base import Template, scalar
from .family import C_FAMILY


def _glsl_extension(name: str) -> str:
    return f"#extension GL_EXT_shader_explicit_arithmetic_types_{name} : require"


_GLSL_MESH = "#version 460\n#extension GL_EXT_mesh_shader : require\n"

# Ray-tracing stage programs. Each stage writes a buffer or the ray data it is handed, which is all a stage needs to
# exist; the hit stages also read the triangle's barycentrics.
_GLSL_RAYTRACING = "#version 460\n#extension GL_EXT_ray_tracing : require\n"
_GLSL_WRITE = (
    "layout(std430, binding = 0) buffer Output {\n    float outp[];\n};\n"
    "void main() {\n    outp[0] = 1.0;\n}\n"
)
_GLSL_PAYLOAD = "struct Payload {\n    vec4 color;\n};\nlayout(location = 0) rayPayloadInEXT Payload payload;\n"
_GLSL_HIT = (
    _GLSL_RAYTRACING
    + _GLSL_PAYLOAD
    + "hitAttributeEXT vec2 attributes;\nvoid main() {\n    payload.color = vec4(attributes, 0.0, 1.0);\n}\n"
)
_GLSL_RAYTRACING_STAGES = {
    "ray-generation": Template(program=_GLSL_RAYTRACING + _GLSL_WRITE),
    "intersection": Template(program=_GLSL_RAYTRACING + _GLSL_WRITE),
    "any-hit": Template(program=_GLSL_HIT),
    "closest-hit": Template(program=_GLSL_HIT),
    "miss": Template(
        program=_GLSL_RAYTRACING
        + _GLSL_PAYLOAD
        + "void main() {\n    payload.color = vec4(0.0, 0.0, 0.0, 1.0);\n}\n"
    ),
    "callable": Template(
        program=_GLSL_RAYTRACING
        + "struct CallData {\n    vec4 color;\n};\nlayout(location = 0) callableDataInEXT CallData data;\n"
        "void main() {\n    data.color = vec4(1.0, 0.0, 0.0, 1.0);\n}\n"
    ),
}

GLSL = {
    **C_FAMILY,
    **aggregates.FAMILY,
    **values.FAMILY,
    **edges.FAMILY,
    **lexical.FAMILY,
    **forms.FAMILY,
    **forms.GLSL,
    **idioms.FAMILY,
    **idioms.GLSL,
    **abstractions.FAMILY,
    **abstractions.GLSL,
    **portability.FAMILY,
    **portability.GLSL,
    **initializers.FAMILY,
    **initializers.GLSL,
    **helpers.GLSL,
    **layouts.GLSL,
    **elements.GLSL,
    **blocks.GLSL,
    **workgroups.GLSL,
    **records.GLSL,
    **identifiers.GLSL,
    **integers.GLSL,
    **columns.GLSL,
    **booleans.GLSL,
    **packing.GLSL,
    **orientation.GLSL,
    **rows.GLSL,
    **vectors.GLSL,
    **relations.GLSL,
    **fragments.GLSL,
    **interfaces.GLSL,
    **vertices.GLSL,
    **draws.GLSL,
    **rasterization.GLSL,
    **pipeline.GLSL,
    **extensions.GLSL,
    **primitives.GLSL,
    **tessellation.GLSL,
    **meshes.GLSL,
    **tasks.GLSL,
    **rays.GLSL,
    **bindings.GLSL,
    **interlocks.GLSL,
    **subgroups.GLSL,
    **specializations.GLSL,
    **qualifiers.GLSL,
    **textures.GLSL,
    **images.GLSL,
    **formats.GLSL,
    **storage.GLSL,
    **samplers.GLSL,
    **_GLSL_RAYTRACING_STAGES,
    **constructors.GLSL,
    "global-constant": Template(
        "outp[i] = fin[i] * SCALE;", "const float SCALE = 2.0;"
    ),
    "runtime-array-length": Template("outp[i] = float(outp.length());"),
    "half": scalar("float16_t", "fin", _glsl_extension("float16")),
    "double": scalar("double", "fin"),
    "int8": scalar("int8_t", prelude=_glsl_extension("int8")),
    "uint8": scalar("uint8_t", prelude=_glsl_extension("int8")),
    "int16": scalar("int16_t", prelude=_glsl_extension("int16")),
    "uint16": scalar("uint16_t", prelude=_glsl_extension("int16")),
    "int64": scalar("int64_t", prelude=_glsl_extension("int64")),
    "uint64": scalar("uint64_t", prelude=_glsl_extension("int64")),
    "shared-memory": Template(
        "tile[0] = fin[i];\nbarrier();\noutp[i] = tile[0];", "shared float tile[1];"
    ),
    "local-invocation-id": Template("outp[i] = float(gl_LocalInvocationID.x);"),
    "workgroup-id": Template("outp[i] = float(gl_WorkGroupID.x);"),
    "local-invocation-index": Template("outp[i] = float(gl_LocalInvocationIndex);"),
    "num-workgroups": Template("outp[i] = float(gl_NumWorkGroups.x);"),
    "uniform-buffer": Template(
        "outp[i] = fin[i] * scale;",
        "layout(std140, binding = 3) uniform Params {\n    float scale;\n};",
    ),
    "read-only-buffer": Template(
        "outp[i] = rin[i];",
        "layout(std430, binding = 3) readonly buffer R {\n    float rin[];\n};",
    ),
    "sampled-texture": Template(
        "outp[i] = textureLod(tex, vec2(fin[i], 0.5), 0.0).x;",
        "layout(binding = 3) uniform sampler2D tex;",
    ),
    "storage-image": Template(
        "imageStore(img, ivec2(0, 0), vec4(fin[i]));\noutp[i] = 0.0;",
        "layout(rgba32f, binding = 4) uniform writeonly image2D img;",
    ),
    "vertex": Template(
        program="#version 460\nlayout(location = 0) in vec4 position;\nvoid main() {\n"
        "    gl_Position = position * 2.0;\n}\n"
    ),
    "fragment": Template(
        program="#version 460\nlayout(location = 0) in vec2 uv;\nlayout(location = 0) out vec4 color;\n"
        "void main() {\n    color = vec4(uv, 0.0, 1.0);\n}\n"
    ),
    "geometry": Template(
        program="#version 460\nlayout(points) in;\nlayout(points, max_vertices = 1) out;\n"
        "layout(location = 0) in vec4 position[];\nvoid main() {\n    gl_Position = position[0] * 2.0;\n"
        "    EmitVertex();\n    EndPrimitive();\n}\n"
    ),
    "discard": Template(
        program="#version 460\nlayout(location = 0) in vec2 uv;\nlayout(location = 0) out vec4 color;\n"
        "void main() {\n    if (uv.x < 0.5) {\n        discard;\n    }\n    color = vec4(uv, 0.0, 1.0);\n}\n"
    ),
    # A triangle patch of three control points, tessellated once.
    "tessellation-control": Template(
        program="#version 460\nlayout(vertices = 3) out;\nlayout(location = 0) in vec4 position[];\n"
        "layout(location = 0) out vec4 outPosition[];\nvoid main() {\n"
        "    outPosition[gl_InvocationID] = position[gl_InvocationID] * 2.0;\n"
        "    gl_TessLevelOuter[0] = 1.0;\n    gl_TessLevelOuter[1] = 1.0;\n    gl_TessLevelOuter[2] = 1.0;\n"
        "    gl_TessLevelInner[0] = 1.0;\n}\n"
    ),
    "tessellation-evaluation": Template(
        program="#version 460\nlayout(triangles, equal_spacing, ccw) in;\nlayout(location = 0) in vec4 position[];\n"
        "void main() {\n    gl_Position = position[0] * gl_TessCoord.x + position[1] * gl_TessCoord.y"
        " + position[2] * gl_TessCoord.z;\n}\n"
    ),
    # A mesh shader that writes one triangle, and a task shader that launches one mesh workgroup with a payload.
    "mesh": Template(
        program=_GLSL_MESH
        + "layout(local_size_x = 1) in;\nlayout(triangles, max_vertices = 3, max_primitives = 1) out;\n"
        "void main() {\n    SetMeshOutputsEXT(3, 1);\n"
        "    gl_MeshVerticesEXT[0].gl_Position = vec4(0.0, 0.0, 0.0, 1.0);\n"
        "    gl_MeshVerticesEXT[1].gl_Position = vec4(1.0, 0.0, 0.0, 1.0);\n"
        "    gl_MeshVerticesEXT[2].gl_Position = vec4(0.0, 1.0, 0.0, 1.0);\n"
        "    gl_PrimitiveTriangleIndicesEXT[0] = uvec3(0, 1, 2);\n}\n"
    ),
    "task": Template(
        program=_GLSL_MESH
        + "layout(local_size_x = 1) in;\nstruct Payload {\n    uint value;\n};\n"
        "taskPayloadSharedEXT Payload payload;\nvoid main() {\n    payload.value = 1u;\n"
        "    EmitMeshTasksEXT(1, 1, 1);\n}\n"
    ),
}

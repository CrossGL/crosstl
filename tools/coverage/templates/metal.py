"""Probe templates for the Metal Shading Language."""

from __future__ import annotations

from . import (
    abstractions,
    aggregates,
    assignments,
    bindings,
    blocks,
    booleans,
    calls,
    columns,
    components,
    constructors,
    draws,
    edges,
    elements,
    evaluation,
    extensions,
    formats,
    forms,
    fragments,
    functions,
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
    lvalues,
    meshes,
    orientation,
    packing,
    pipeline,
    pointers,
    portability,
    primitives,
    qualifiers,
    rasterization,
    rays,
    records,
    relations,
    rows,
    samplers,
    scopes,
    specializations,
    statements,
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
from .base import ABSENT, Template, scalar
from .family import C_FAMILY

_METAL_HEADER = "#include <metal_stdlib>\nusing namespace metal;\n\n"

METAL = {
    **C_FAMILY,
    **aggregates.FAMILY,
    **functions.METAL,
    **constructors.METAL,
    **scopes.METAL,
    **calls.METAL,
    **evaluation.METAL,
    **assignments.METAL,
    **statements.METAL,
    **lvalues.METAL,
    **aggregates.METAL,
    **values.FAMILY,
    **values.METAL,
    **edges.FAMILY,
    **edges.METAL,
    **lexical.FAMILY,
    **forms.FAMILY,
    **lexical.METAL,
    **forms.METAL,
    **idioms.FAMILY,
    **idioms.METAL,
    **abstractions.FAMILY,
    **portability.FAMILY,
    **portability.METAL,
    **initializers.FAMILY,
    **initializers.METAL,
    **helpers.FAMILY,
    **helpers.METAL,
    **layouts.METAL,
    **elements.METAL,
    **blocks.METAL,
    **workgroups.METAL,
    **records.METAL,
    **identifiers.METAL,
    **integers.METAL,
    **columns.METAL,
    **booleans.METAL,
    **packing.METAL,
    **orientation.METAL,
    **rows.METAL,
    **vectors.METAL,
    **relations.METAL,
    **fragments.METAL,
    **interfaces.METAL,
    **vertices.METAL,
    **draws.METAL,
    **rasterization.METAL,
    **pipeline.METAL,
    **extensions.METAL,
    **primitives.METAL,
    **tessellation.METAL,
    **meshes.METAL,
    **tasks.METAL,
    **rays.METAL,
    **bindings.METAL,
    **interlocks.METAL,
    **components.ELSEWHERE,
    **pointers.METAL,
    **subgroups.METAL,
    **specializations.METAL,
    **qualifiers.METAL,
    **textures.METAL,
    **images.METAL,
    **formats.METAL,
    **storage.METAL,
    **samplers.METAL,
    "out-parameter": Template(
        C_FAMILY["out-parameter"].body,
        "void split(float x, thread float& a, thread float& b) {\n    a = x;\n    b = x * 2.0;\n}",
    ),
    "inout-parameter": Template(
        C_FAMILY["inout-parameter"].body,
        "void bump(thread float& a) {\n    a = a + 1.0;\n}",
    ),
    "global-constant": Template(
        "outp[i] = fin[i] * SCALE;", "constant float SCALE = 2.0;"
    ),
    # A Metal buffer is a pointer; its length is not part of the language.
    "runtime-array-length": ABSENT,
    "half": scalar("half", "fin"),
    "double": ABSENT,
    "int8": scalar("char"),
    "uint8": scalar("uchar"),
    "int16": scalar("short"),
    "uint16": scalar("ushort"),
    "int64": scalar("long"),
    "uint64": scalar("ulong"),
    "shared-memory": Template(
        "threadgroup float tile[1];\ntile[0] = fin[i];\nthreadgroup_barrier(mem_flags::mem_threadgroup);\n"
        "outp[i] = tile[0];"
    ),
    "local-invocation-id": Template(
        "outp[i] = float(lid.x);", params="uint3 lid [[thread_position_in_threadgroup]]"
    ),
    "workgroup-id": Template(
        "outp[i] = float(gid.x);", params="uint3 gid [[threadgroup_position_in_grid]]"
    ),
    "local-invocation-index": Template(
        "outp[i] = float(lidx);", params="uint lidx [[thread_index_in_threadgroup]]"
    ),
    "num-workgroups": Template(
        "outp[i] = float(groups.x);", params="uint3 groups [[threadgroups_per_grid]]"
    ),
    "uniform-buffer": Template(
        "outp[i] = fin[i] * params.scale;",
        "struct Params {\n    float scale;\n};",
        params="constant Params& params [[buffer(3)]]",
    ),
    "read-only-buffer": Template(
        "outp[i] = rin[i];", params="const device float* rin [[buffer(3)]]"
    ),
    "sampled-texture": Template(
        "outp[i] = tex.sample(smp, float2(fin[i], 0.5), level(0.0)).x;",
        params="texture2d<float> tex [[texture(0)]]\nsampler smp [[sampler(0)]]",
    ),
    "storage-image": Template(
        "img.write(float4(fin[i]), uint2(0, 0));\noutp[i] = 0.0;",
        params="texture2d<float, access::write> img [[texture(1)]]",
    ),
    "vertex": Template(
        program=_METAL_HEADER
        + "struct VertexIn {\n    float4 position [[attribute(0)]];\n};\n\n"
        "vertex float4 probe(VertexIn in [[stage_in]]) {\n    return in.position * 2.0;\n}\n"
    ),
    "fragment": Template(
        program=_METAL_HEADER + "struct FragmentIn {\n    float2 uv;\n};\n\n"
        "fragment float4 probe(FragmentIn in [[stage_in]]) {\n    return float4(in.uv, 0.0, 1.0);\n}\n"
    ),
    "discard": Template(
        program=_METAL_HEADER + "struct FragmentIn {\n    float2 uv;\n};\n\n"
        "fragment float4 probe(FragmentIn in [[stage_in]]) {\n    if (in.uv.x < 0.5) {\n"
        "        discard_fragment();\n    }\n    return float4(in.uv, 0.0, 1.0);\n}\n"
    ),
    # The post-tessellation vertex function of a triangle patch.
    "tessellation-evaluation": Template(
        program=_METAL_HEADER
        + "struct ControlPoint {\n    float4 position [[attribute(0)]];\n};\n\n"
        "struct Patch {\n    patch_control_point<ControlPoint> points;\n};\n\n"
        "[[patch(triangle, 3)]]\nvertex float4 probe(Patch patch [[stage_in]], "
        "float3 coord [[position_in_patch]]) {\n"
        "    return patch.points[0].position * coord.x + patch.points[1].position * coord.y"
        " + patch.points[2].position * coord.z;\n}\n"
    ),
    # A mesh function that writes one triangle, and an object function, Metal's task stage, that launches one mesh
    # threadgroup with a payload.
    "mesh": Template(
        program=_METAL_HEADER
        + "struct Vertex {\n    float4 position [[position]];\n};\n\n"
        "[[mesh]] void probe(mesh<Vertex, void, 3, 1, topology::triangle> output) {\n"
        "    output.set_primitive_count(1);\n"
        "    output.set_vertex(0, Vertex{float4(0.0, 0.0, 0.0, 1.0)});\n"
        "    output.set_vertex(1, Vertex{float4(1.0, 0.0, 0.0, 1.0)});\n"
        "    output.set_vertex(2, Vertex{float4(0.0, 1.0, 0.0, 1.0)});\n"
        "    output.set_index(0, 0);\n    output.set_index(1, 1);\n    output.set_index(2, 2);\n}\n"
    ),
    "task": Template(
        program=_METAL_HEADER + "struct Payload {\n    uint value;\n};\n\n"
        "[[object]] void probe(object_data Payload& payload [[payload]], mesh_grid_properties grid) {\n"
        "    payload.value = 1;\n    grid.set_threadgroups_per_grid(uint3(1, 1, 1));\n}\n"
    ),
    # A bounding-box intersection function, which tests a ray against a custom primitive as an intersection shader
    # does in Direct3D and Vulkan, and accepts the hit at the ray's minimum distance.
    "intersection": Template(
        program=_METAL_HEADER + "using namespace raytracing;\n\n"
        "struct BoxResult {\n    bool accept [[accept_intersection]];\n"
        "    float distance [[distance]];\n};\n\n"
        "[[intersection(bounding_box)]]\n"
        "BoxResult probe(float min_distance [[min_distance]]) {\n"
        "    BoxResult result;\n    result.accept = true;\n"
        "    result.distance = min_distance;\n    return result;\n}\n"
    ),
    # Metal has no geometry stage, runs tessellation factors in a compute kernel, and traces rays from kernels
    # with intersection functions rather than dedicated ray stages.
    **dict.fromkeys(
        (
            "geometry",
            "tessellation-control",
            "ray-generation",
            "any-hit",
            "closest-hit",
            "miss",
            "callable",
        ),
        ABSENT,
    ),
}

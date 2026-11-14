"""Probe templates for HLSL."""

from __future__ import annotations

from . import (
    abstractions,
    aggregates,
    bindings,
    blocks,
    booleans,
    calls,
    columns,
    components,
    composites,
    constructors,
    draws,
    edges,
    elements,
    evaluation,
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
    memory,
    meshes,
    operators,
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
    shapes,
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
from .base import ABSENT, Template, scalar
from .family import C_FAMILY

_HLSL_PATCH = (
    "struct ControlPoint {\n    float4 position : POSITION;\n};\n\nstruct PatchConstants {\n"
    "    float edges[3] : SV_TessFactor;\n    float inside : SV_InsideTessFactor;\n};\n\n"
)

HLSL = {
    **C_FAMILY,
    **aggregates.FAMILY,
    "global-constant": Template(
        "outp[i] = fin[i] * SCALE;", "static const float SCALE = 2.0;"
    ),
    "runtime-array-length": Template(
        "uint count;\nuint stride;\noutp.GetDimensions(count, stride);\noutp[i] = float(count);"
    ),
    # HLSL fills a matrix from scalars row by row, where CrossGL fills it column by column, so the scalars are listed
    # by rows to make the matrix CrossGL's mat2(fin[i], 1.0, 2.0, 3.0).
    "matrix-vector-product": Template(
        "float2x2 m = float2x2(fin[i], 2.0, 1.0, 3.0);\nfloat2 v = mul(m, float2(1.0, fin[i]));\n"
        "outp[i] = v.y;"
    ),
    **operators.HLSL,
    **composites.HLSL,
    **constructors.HLSL,
    **scopes.HLSL,
    **calls.HLSL,
    **shapes.HLSL,
    **evaluation.HLSL,
    **values.FAMILY,
    **values.HLSL,
    **edges.FAMILY,
    **lexical.FAMILY,
    **forms.FAMILY,
    **idioms.FAMILY,
    **abstractions.FAMILY,
    **portability.FAMILY,
    **initializers.FAMILY,
    **helpers.FAMILY,
    **layouts.FAMILY,
    **elements.FAMILY,
    **blocks.FAMILY,
    **workgroups.FAMILY,
    **records.FAMILY,
    **identifiers.FAMILY,
    **integers.FAMILY,
    **columns.FAMILY,
    **booleans.FAMILY,
    **packing.FAMILY,
    **orientation.FAMILY,
    **rows.FAMILY,
    **vectors.HLSL,
    **relations.HLSL,
    **fragments.HLSL,
    **interfaces.HLSL,
    **vertices.HLSL,
    **draws.HLSL,
    **rasterization.HLSL,
    **pipeline.HLSL,
    **extensions.HLSL,
    **primitives.HLSL,
    **tessellation.HLSL,
    **meshes.HLSL,
    **tasks.HLSL,
    **rays.HLSL,
    **bindings.HLSL,
    **interlocks.HLSL,
    **components.ELSEWHERE,
    **pointers.ELSEWHERE,
    **memory.ELSEWHERE,
    **subgroups.HLSL,
    **specializations.HLSL,
    **qualifiers.HLSL,
    **textures.HLSL,
    **images.HLSL,
    **formats.HLSL,
    **storage.HLSL,
    **samplers.HLSL,
    "half": scalar("half", "fin"),
    "double": scalar("double", "fin"),
    "int8": ABSENT,
    "uint8": ABSENT,
    "int16": scalar("int16_t"),
    "uint16": scalar("uint16_t"),
    "int64": scalar("int64_t"),
    "uint64": scalar("uint64_t"),
    "shared-memory": Template(
        "tile[0] = fin[i];\nGroupMemoryBarrierWithGroupSync();\noutp[i] = tile[0];",
        "groupshared float tile[1];",
    ),
    "local-invocation-id": Template(
        "outp[i] = float(lid.x);", params="uint3 lid : SV_GroupThreadID"
    ),
    "workgroup-id": Template(
        "outp[i] = float(gid.x);", params="uint3 gid : SV_GroupID"
    ),
    "local-invocation-index": Template(
        "outp[i] = float(lidx);", params="uint lidx : SV_GroupIndex"
    ),
    # HLSL has no system value for the number of groups; applications pass it in a constant buffer.
    "num-workgroups": ABSENT,
    # Slot 3, as in the other languages: the SPIR-V that DXC and slangc write puts register b0 at the binding of u0.
    "uniform-buffer": Template(
        "outp[i] = fin[i] * scale;",
        "cbuffer Params : register(b3) {\n    float scale;\n};",
    ),
    "read-only-buffer": Template(
        "outp[i] = rin[i];", "StructuredBuffer<float> rin : register(t0);"
    ),
    "sampled-texture": Template(
        "outp[i] = tex.SampleLevel(smp, float2(fin[i], 0.5), 0.0).x;",
        "Texture2D<float4> tex : register(t1);\nSamplerState smp : register(s0);",
    ),
    "storage-image": Template(
        "img[uint2(0, 0)] = float4(fin[i], fin[i], fin[i], fin[i]);\noutp[i] = 0.0;",
        "RWTexture2D<float4> img : register(u3);",
    ),
    "vertex": Template(
        program="float4 main(float4 position : POSITION) : SV_Position {\n    return position * 2.0;\n}\n",
        entry="main",
    ),
    "fragment": Template(
        program="float4 main(float2 uv : TEXCOORD0) : SV_Target {\n    return float4(uv, 0.0, 1.0);\n}\n",
        entry="main",
    ),
    "geometry": Template(
        program="struct GeometryOut {\n    float4 position : SV_Position;\n};\n\n[maxvertexcount(1)]\n"
        "void main(point float4 position[1] : POSITION, inout PointStream<GeometryOut> stream) {\n"
        "    GeometryOut output;\n    output.position = position[0] * 2.0;\n    stream.Append(output);\n"
        "    stream.RestartStrip();\n}\n",
        entry="main",
    ),
    "discard": Template(
        program="float4 main(float2 uv : TEXCOORD0) : SV_Target {\n    if (uv.x < 0.5) {\n        discard;\n"
        "    }\n    return float4(uv, 0.0, 1.0);\n}\n",
        entry="main",
    ),
    "tessellation-control": Template(
        program=_HLSL_PATCH
        + "PatchConstants constants(InputPatch<ControlPoint, 3> patch) {\n    PatchConstants output;\n"
        "    output.edges[0] = 1.0;\n    output.edges[1] = 1.0;\n    output.edges[2] = 1.0;\n"
        "    output.inside = 1.0;\n    return output;\n}\n\n"
        '[domain("tri")]\n[partitioning("integer")]\n[outputtopology("triangle_cw")]\n[outputcontrolpoints(3)]\n'
        '[patchconstantfunc("constants")]\n'
        "ControlPoint main(InputPatch<ControlPoint, 3> patch, uint id : SV_OutputControlPointID) {\n"
        "    ControlPoint output;\n    output.position = patch[id].position * 2.0;\n    return output;\n}\n",
        entry="main",
    ),
    "tessellation-evaluation": Template(
        program=_HLSL_PATCH
        + '[domain("tri")]\nfloat4 main(PatchConstants constants, float3 coord : SV_DomainLocation,\n'
        "            const OutputPatch<ControlPoint, 3> patch) : SV_Position {\n"
        "    return patch[0].position * coord.x + patch[1].position * coord.y + patch[2].position * coord.z;\n}\n",
        entry="main",
    ),
    "mesh": Template(
        program="struct Vertex {\n    float4 position : SV_Position;\n};\n\n"
        '[numthreads(1, 1, 1)]\n[outputtopology("triangle")]\n'
        "void main(out vertices Vertex verts[3], out indices uint3 tris[1]) {\n"
        "    SetMeshOutputCounts(3, 1);\n    verts[0].position = float4(0.0, 0.0, 0.0, 1.0);\n"
        "    verts[1].position = float4(1.0, 0.0, 0.0, 1.0);\n    verts[2].position = float4(0.0, 1.0, 0.0, 1.0);\n"
        "    tris[0] = uint3(0, 1, 2);\n}\n",
        entry="main",
    ),
    "task": Template(
        program="struct Payload {\n    uint value;\n};\n\ngroupshared Payload payload;\n\n"
        "[numthreads(1, 1, 1)]\nvoid main() {\n    payload.value = 1;\n    DispatchMesh(1, 1, 1, payload);\n}\n",
        entry="main",
    ),
}


def raytracing_stages(payload: str) -> dict[str, Template]:
    """HLSL's ray-tracing stage programs, which compile as a library; ``payload`` declares the ray payload."""

    def write(stage: str) -> Template:
        return Template(
            program=f'RWStructuredBuffer<float> outp : register(u0);\n\n[shader("{stage}")]\n'
            "void main() {\n    outp[0] = 1.0;\n}\n"
        )

    def hit(stage: str) -> Template:
        return Template(
            program=f'{payload}\n[shader("{stage}")]\nvoid main(inout Payload payload, '
            "in BuiltInTriangleIntersectionAttributes attributes) {\n"
            "    payload.color = float4(attributes.barycentrics, 0.0, 1.0);\n}\n"
        )

    miss = Template(
        program=f'{payload}\n[shader("miss")]\nvoid main(inout Payload payload) {{\n'
        "    payload.color = float4(0.0, 0.0, 0.0, 1.0);\n}\n"
    )
    callable = Template(
        program='struct CallData {\n    float4 color;\n};\n\n[shader("callable")]\n'
        "void main(inout CallData data) {\n    data.color = float4(1.0, 0.0, 0.0, 1.0);\n}\n"
    )
    return {
        "ray-generation": write("raygeneration"),
        "intersection": write("intersection"),
        "any-hit": hit("anyhit"),
        "closest-hit": hit("closesthit"),
        "miss": miss,
        "callable": callable,
    }


# Shader model 6.7 and later require a payload to say which stages read and write each field.
HLSL.update(
    raytracing_stages(
        "struct [raypayload] Payload {\n"
        "    float4 color : read(caller) : write(anyhit, closesthit, miss);\n};\n"
    )
)

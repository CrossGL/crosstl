"""Probe templates for the shading languages: GLSL, HLSL, Metal, WGSL and Slang."""

from __future__ import annotations

import re
from dataclasses import replace

from . import composites, operators
from .base import ABSENT, C_FAMILY, Template, scalar


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
    **_GLSL_RAYTRACING_STAGES,
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

_HLSL_PATCH = (
    "struct ControlPoint {\n    float4 position : POSITION;\n};\n\nstruct PatchConstants {\n"
    "    float edges[3] : SV_TessFactor;\n    float inside : SV_InsideTessFactor;\n};\n\n"
)

HLSL = {
    **C_FAMILY,
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


def _hlsl_raytracing(payload: str) -> dict[str, Template]:
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
    _hlsl_raytracing(
        "struct [raypayload] Payload {\n"
        "    float4 color : read(caller) : write(anyhit, closesthit, miss);\n};\n"
    )
)

_METAL_HEADER = "#include <metal_stdlib>\nusing namespace metal;\n\n"

METAL = {
    **C_FAMILY,
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


WGSL = {
    "if-else": C_FAMILY["if-else"],
    "switch": Template(
        "switch (iin[i]) {\n    case 0: {\n        outp[i] = 1.0;\n    }\n    case 1: {\n        outp[i] = 3.0;\n"
        "    }\n    default: {\n        outp[i] = 2.0;\n    }\n}"
    ),
    "for-loop": Template(
        "var acc = 0.0;\nfor (var k = 0; k < iin[i]; k++) {\n    acc += fin[i];\n}\noutp[i] = acc;"
    ),
    "while-loop": Template(
        "var acc = 0.0;\nvar k = 0;\nwhile (k < iin[i]) {\n    acc += fin[i];\n    k++;\n}\noutp[i] = acc;"
    ),
    # WGSL's post-tested loop is ``loop`` with a ``break if`` in its continuing block.
    "do-while-loop": Template(
        "var acc = 0.0;\nvar k = 0;\nloop {\n    acc += fin[i];\n    k++;\n    continuing {\n"
        "        break if k >= iin[i];\n    }\n}\noutp[i] = acc;"
    ),
    "break": Template(
        "var acc = 0.0;\nfor (var k = 0; k < 8; k++) {\n    if (k == iin[i]) {\n        break;\n    }\n"
        "    acc += fin[i];\n}\noutp[i] = acc;"
    ),
    "continue": Template(
        "var acc = 0.0;\nfor (var k = 0; k < 8; k++) {\n    if (k == iin[i]) {\n        continue;\n    }\n"
        "    acc += fin[i];\n}\noutp[i] = acc;"
    ),
    "early-return": C_FAMILY["early-return"],
    "conditional-expression": Template("outp[i] = select(0.0, fin[i], fin[i] > 0.0);"),
    "short-circuit-logic": Template(
        "if (iin[i] > 0 && fin[i] / f32(iin[i]) > 1.0) {\n    outp[i] = 1.0;\n}"
        " else if (iin[i] < 0 || fin[i] < 0.0) {\n    outp[i] = 2.0;\n}"
    ),
    "function-call": Template(
        "outp[i] = twice(fin[i]);", "fn twice(x: f32) -> f32 {\n    return x * 2.0;\n}"
    ),
    "out-parameter": Template(
        "var a: f32;\nvar b: f32;\nsplit(fin[i], &a, &b);\noutp[i] = a + b;",
        "fn split(x: f32, a: ptr<function, f32>, b: ptr<function, f32>) {\n    *a = x;\n    *b = x * 2.0;\n}",
    ),
    "inout-parameter": Template(
        "var a = fin[i];\nbump(&a);\noutp[i] = a;",
        "fn bump(a: ptr<function, f32>) {\n    *a = *a + 1.0;\n}",
    ),
    "array": Template(
        "var values: array<f32, 4>;\nfor (var k = 0; k < 4; k++) {\n    values[k] = fin[i] * f32(k);\n}\n"
        "outp[i] = values[iin[i] & 3];"
    ),
    "struct": Template(
        "var p: Pair;\np.a = fin[i];\np.b = 1.0;\noutp[i] = p.a + p.b;",
        "struct Pair {\n    a: f32,\n    b: f32,\n}",
    ),
    "array-of-structs": Template(
        "var ps: array<Pair, 2>;\nps[0].a = fin[i];\nps[1].b = 2.0;\noutp[i] = ps[0].a + ps[1].b;",
        "struct Pair {\n    a: f32,\n    b: f32,\n}",
    ),
    "swizzle": Template(
        "let v = vec4<f32>(fin[i], 1.0, 2.0, 3.0);\noutp[i] = v.wzyx.x + v.xy.y;"
    ),
    "matrix-vector-product": Template(
        "let m = mat2x2<f32>(fin[i], 1.0, 2.0, 3.0);\nlet v = m * vec2<f32>(1.0, fin[i]);\noutp[i] = v.y;"
    ),
    **operators.WGSL,
    **composites.WGSL,
    "global-constant": Template("outp[i] = fin[i] * SCALE;", "const SCALE: f32 = 2.0;"),
    "runtime-array-length": Template("outp[i] = f32(arrayLength(&outp));"),
    "half": Template(
        "let v = f16(fin[i]);\noutp[i] = f32(v * v);", prelude="enable f16;"
    ),
    **dict.fromkeys(
        ("double", "int8", "uint8", "int16", "uint16", "int64", "uint64"), ABSENT
    ),
    "shared-memory": Template(
        "tile[0] = fin[i];\nworkgroupBarrier();\noutp[i] = tile[0];",
        "var<workgroup> tile: array<f32, 1>;",
    ),
    "local-invocation-id": Template(
        "outp[i] = f32(lid.x);", params="@builtin(local_invocation_id) lid: vec3<u32>"
    ),
    "workgroup-id": Template(
        "outp[i] = f32(gid.x);", params="@builtin(workgroup_id) gid: vec3<u32>"
    ),
    "local-invocation-index": Template(
        "outp[i] = f32(lidx);", params="@builtin(local_invocation_index) lidx: u32"
    ),
    "num-workgroups": Template(
        "outp[i] = f32(groups.x);", params="@builtin(num_workgroups) groups: vec3<u32>"
    ),
    "uniform-buffer": Template(
        "outp[i] = fin[i] * params.scale;",
        "struct Params {\n    scale: f32,\n}\n@group(0) @binding(3) var<uniform> params: Params;",
    ),
    "read-only-buffer": Template(
        "outp[i] = rin[i];", "@group(0) @binding(3) var<storage, read> rin: array<f32>;"
    ),
    "sampled-texture": Template(
        "outp[i] = textureSampleLevel(tex, smp, vec2<f32>(fin[i], 0.5), 0.0).x;",
        "@group(0) @binding(4) var tex: texture_2d<f32>;\n@group(0) @binding(5) var smp: sampler;",
    ),
    "storage-image": Template(
        "textureStore(img, vec2<i32>(0, 0), vec4<f32>(fin[i]));\noutp[i] = 0.0;",
        "@group(0) @binding(6) var img: texture_storage_2d<rgba32float, write>;",
    ),
    "compute": C_FAMILY["compute"],
    "vertex": Template(
        program="@vertex\nfn main(@location(0) position: vec4<f32>) -> @builtin(position) vec4<f32> {\n"
        "    return position * 2.0;\n}\n"
    ),
    "fragment": Template(
        program="@fragment\nfn main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {\n"
        "    return vec4<f32>(uv, 0.0, 1.0);\n}\n"
    ),
    "discard": Template(
        program="@fragment\nfn main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {\n"
        "    if (uv.x < 0.5) {\n        discard;\n    }\n    return vec4<f32>(uv, 0.0, 1.0);\n}\n"
    ),
    # WGSL has only vertex, fragment and compute shaders.
    **dict.fromkeys(
        (
            "geometry",
            "tessellation-control",
            "tessellation-evaluation",
            "task",
            "mesh",
            "ray-generation",
            "intersection",
            "any-hit",
            "closest-hit",
            "miss",
            "callable",
        ),
        ABSENT,
    ),
}


def _slang(template: Template, stage: str = "") -> Template:
    """An HLSL template as Slang: slangc assigns bindings itself, and a stage program names its stage."""
    declarations = re.sub(r"\s*:\s*register\(\w+\)", "", template.declarations)
    program = f'[shader("{stage}")]\n{template.program}' if stage else template.program
    return replace(template, declarations=declarations, program=program)


# Slang compiles HLSL, adds 8-bit integers, and reads the workgroup count through WorkgroupCount().
SLANG = {
    **{name: _slang(template) for name, template in HLSL.items()},
    "int8": scalar("int8_t"),
    "uint8": scalar("uint8_t"),
    "num-workgroups": Template("outp[i] = float(WorkgroupCount().x);"),
    "vertex": _slang(HLSL["vertex"], "vertex"),
    "fragment": _slang(HLSL["fragment"], "fragment"),
    # The stage attribute goes on the entry point, after the output struct.
    "geometry": replace(
        HLSL["geometry"],
        program=HLSL["geometry"].program.replace(
            "[maxvertexcount", '[shader("geometry")]\n[maxvertexcount'
        ),
    ),
    "discard": _slang(HLSL["discard"], "fragment"),
    # The stage attribute goes on the entry point, ahead of its other attributes.
    **{
        name: replace(
            HLSL[name],
            program=HLSL[name].program.replace(
                attribute, f'[shader("{stage}")]\n{attribute}', 1
            ),
        )
        for name, stage, attribute in (
            ("tessellation-control", "hull", "[domain("),
            ("tessellation-evaluation", "domain", "[domain("),
            ("mesh", "mesh", "[numthreads("),
            ("task", "amplification", "[numthreads("),
        )
    },
    # slangc takes no payload access qualifiers.
    **{
        name: replace(template, program=template.program.replace(" : register(u0)", ""))
        for name, template in _hlsl_raytracing(
            "struct Payload {\n    float4 color;\n};\n"
        ).items()
    },
}

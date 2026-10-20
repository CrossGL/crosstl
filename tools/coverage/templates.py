"""Probe templates for the language-level features of :mod:`.constructs`, in each target language.

A kernel template gives the statements of a compute kernel (``body``), what precedes them (``declarations``; for
GLSL extensions and WGSL ``enable`` directives, ``prelude``) and extra entry-point parameters (``params``). The kernel
around them is the language's :class:`~tools.coverage.dialect.Dialect` kernel: invocation index ``i``, input buffers
``fin`` (float) and ``iin`` (int), output buffer ``outp`` (float), at bindings 0, 1 and 2. A stage template is a whole
``program``. Templates may write ``$vec4``, ``$mat2`` and similar for the language's spelling of a type.

:data:`ABSENT` marks a feature the language does not have; a feature missing from a language's table has no template
yet.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class Template:
    body: str = ""
    declarations: str = ""
    prelude: str = ""
    params: str = ""
    program: str = ""
    entry: str = ""


ABSENT = Template(entry="<absent>")

# Probes most C-like languages, CrossGL among them, write the same way.
C_FAMILY = {
    "if-else": Template(
        "if (fin[i] > 0.0) {\n    outp[i] = 1.0;\n} else {\n    outp[i] = 2.0;\n}"
    ),
    "switch": Template(
        "switch (iin[i]) {\n    case 0:\n        outp[i] = 1.0;\n        break;\n    case 1:\n"
        "        outp[i] = 3.0;\n        break;\n    default:\n        outp[i] = 2.0;\n        break;\n}"
    ),
    "for-loop": Template(
        "float acc = 0.0;\nfor (int k = 0; k < iin[i]; k++) {\n    acc += fin[i];\n}\noutp[i] = acc;"
    ),
    "while-loop": Template(
        "float acc = 0.0;\nint k = 0;\nwhile (k < iin[i]) {\n    acc += fin[i];\n    k++;\n}\noutp[i] = acc;"
    ),
    "do-while-loop": Template(
        "float acc = 0.0;\nint k = 0;\ndo {\n    acc += fin[i];\n    k++;\n} while (k < iin[i]);\noutp[i] = acc;"
    ),
    "break": Template(
        "float acc = 0.0;\nfor (int k = 0; k < 8; k++) {\n    if (k == iin[i]) {\n        break;\n    }\n"
        "    acc += fin[i];\n}\noutp[i] = acc;"
    ),
    "continue": Template(
        "float acc = 0.0;\nfor (int k = 0; k < 8; k++) {\n    if (k == iin[i]) {\n        continue;\n    }\n"
        "    acc += fin[i];\n}\noutp[i] = acc;"
    ),
    "early-return": Template(
        "if (fin[i] < 0.0) {\n    outp[i] = 0.0;\n    return;\n}\noutp[i] = fin[i];"
    ),
    "conditional-expression": Template("outp[i] = fin[i] > 0.0 ? fin[i] : 0.0;"),
    "short-circuit-logic": Template(
        "if (iin[i] > 0 && fin[i] / float(iin[i]) > 1.0) {\n    outp[i] = 1.0;\n}"
        " else if (iin[i] < 0 || fin[i] < 0.0) {\n    outp[i] = 2.0;\n}"
    ),
    "function-call": Template(
        "outp[i] = twice(fin[i]);",
        "float twice(float x) {\n    return x * 2.0;\n}",
    ),
    "out-parameter": Template(
        "float a;\nfloat b;\nsplit(fin[i], a, b);\noutp[i] = a + b;",
        "void split(float x, out float a, out float b) {\n    a = x;\n    b = x * 2.0;\n}",
    ),
    "inout-parameter": Template(
        "float a = fin[i];\nbump(a);\noutp[i] = a;",
        "void bump(inout float a) {\n    a = a + 1.0;\n}",
    ),
    "array": Template(
        "float values[4];\nfor (int k = 0; k < 4; k++) {\n    values[k] = fin[i] * float(k);\n}\n"
        "outp[i] = values[iin[i] & 3];"
    ),
    "struct": Template(
        "Pair p;\np.a = fin[i];\np.b = 1.0;\noutp[i] = p.a + p.b;",
        "struct Pair {\n    float a;\n    float b;\n};",
    ),
    "array-of-structs": Template(
        "Pair ps[2];\nps[0].a = fin[i];\nps[1].b = 2.0;\noutp[i] = ps[0].a + ps[1].b;",
        "struct Pair {\n    float a;\n    float b;\n};",
    ),
    "swizzle": Template(
        "$vec4 v = $vec4(fin[i], 1.0, 2.0, 3.0);\noutp[i] = v.wzyx.x + v.xy.y;"
    ),
    "matrix-vector-product": Template(
        "$mat2 m = $mat2(fin[i], 1.0, 2.0, 3.0);\n$vec2 v = m * $vec2(1.0, fin[i]);\noutp[i] = v.y;"
    ),
    "compute": Template("outp[i] = fin[i];"),
}


def _scalar(spelling: str, source: str = "iin", prelude: str = "") -> Template:
    return Template(
        f"{spelling} v = {spelling}({source}[i]);\noutp[i] = float(v * v);",
        prelude=prelude,
    )


def _glsl_extension(name: str) -> str:
    return f"#extension GL_EXT_shader_explicit_arithmetic_types_{name} : require"


GLSL = {
    **C_FAMILY,
    "global-constant": Template(
        "outp[i] = fin[i] * SCALE;", "const float SCALE = 2.0;"
    ),
    "runtime-array-length": Template("outp[i] = float(outp.length());"),
    "half": _scalar("float16_t", "fin", _glsl_extension("float16")),
    "double": _scalar("double", "fin"),
    "int8": _scalar("int8_t", prelude=_glsl_extension("int8")),
    "uint8": _scalar("uint8_t", prelude=_glsl_extension("int8")),
    "int16": _scalar("int16_t", prelude=_glsl_extension("int16")),
    "uint16": _scalar("uint16_t", prelude=_glsl_extension("int16")),
    "int64": _scalar("int64_t", prelude=_glsl_extension("int64")),
    "uint64": _scalar("uint64_t", prelude=_glsl_extension("int64")),
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
    "discard": Template(
        program="#version 460\nlayout(location = 0) in vec2 uv;\nlayout(location = 0) out vec4 color;\n"
        "void main() {\n    if (uv.x < 0.5) {\n        discard;\n    }\n    color = vec4(uv, 0.0, 1.0);\n}\n"
    ),
}

HLSL = {
    **C_FAMILY,
    "global-constant": Template(
        "outp[i] = fin[i] * SCALE;", "static const float SCALE = 2.0;"
    ),
    "runtime-array-length": Template(
        "uint count;\nuint stride;\noutp.GetDimensions(count, stride);\noutp[i] = float(count);"
    ),
    "matrix-vector-product": Template(
        "float2x2 m = float2x2(fin[i], 1.0, 2.0, 3.0);\nfloat2 v = mul(m, float2(1.0, fin[i]));\n"
        "outp[i] = v.y;"
    ),
    "half": _scalar("half", "fin"),
    "double": _scalar("double", "fin"),
    "int8": ABSENT,
    "uint8": ABSENT,
    "int16": _scalar("int16_t"),
    "uint16": _scalar("uint16_t"),
    "int64": _scalar("int64_t"),
    "uint64": _scalar("uint64_t"),
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
    "uniform-buffer": Template(
        "outp[i] = fin[i] * scale;",
        "cbuffer Params : register(b0) {\n    float scale;\n};",
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
    "discard": Template(
        program="float4 main(float2 uv : TEXCOORD0) : SV_Target {\n    if (uv.x < 0.5) {\n        discard;\n"
        "    }\n    return float4(uv, 0.0, 1.0);\n}\n",
        entry="main",
    ),
}

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
    "half": _scalar("half", "fin"),
    "double": ABSENT,
    "int8": _scalar("char"),
    "uint8": _scalar("uchar"),
    "int16": _scalar("short"),
    "uint16": _scalar("ushort"),
    "int64": _scalar("long"),
    "uint64": _scalar("ulong"),
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

_CSTDINT = "#include <cstdint>"

CUDA = {
    **C_FAMILY,
    "function-call": Template(
        C_FAMILY["function-call"].body,
        "__device__ float twice(float x) {\n    return x * 2.0f;\n}",
    ),
    "out-parameter": Template(
        C_FAMILY["out-parameter"].body,
        "__device__ void split(float x, float& a, float& b) {\n    a = x;\n    b = x * 2.0f;\n}",
    ),
    "inout-parameter": Template(
        C_FAMILY["inout-parameter"].body,
        "__device__ void bump(float& a) {\n    a = a + 1.0f;\n}",
    ),
    # CUDA vector types are plain structs: no swizzles, and no matrix types at all.
    "swizzle": ABSENT,
    "matrix-vector-product": ABSENT,
    "global-constant": Template(
        "outp[i] = fin[i] * SCALE;", "constexpr float SCALE = 2.0f;"
    ),
    # A CUDA buffer is a pointer; its length is not part of the language.
    "runtime-array-length": ABSENT,
    "half": _scalar("__half", "fin", "#include <cuda_fp16.h>"),
    "double": _scalar("double", "fin"),
    "int8": _scalar("int8_t", prelude=_CSTDINT),
    "uint8": _scalar("uint8_t", prelude=_CSTDINT),
    "int16": _scalar("int16_t", prelude=_CSTDINT),
    "uint16": _scalar("uint16_t", prelude=_CSTDINT),
    "int64": _scalar("int64_t", prelude=_CSTDINT),
    "uint64": _scalar("uint64_t", prelude=_CSTDINT),
    "shared-memory": Template(
        "__shared__ float tile[1];\ntile[0] = fin[i];\n__syncthreads();\noutp[i] = tile[0];"
    ),
    "local-invocation-id": Template("outp[i] = float(threadIdx.x);"),
    "workgroup-id": Template("outp[i] = float(blockIdx.x);"),
    # CUDA has no built-in for the flattened thread index; kernels compute it the way GLSL defines it.
    "local-invocation-index": Template(
        "outp[i] = float(threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z));"
    ),
    "num-workgroups": Template("outp[i] = float(gridDim.x);"),
    "uniform-buffer": Template(
        "outp[i] = fin[i] * scale;", "__constant__ float scale;"
    ),
    "read-only-buffer": Template(
        "outp[i] = rin[i];", params="const float* __restrict__ rin"
    ),
    "sampled-texture": Template(
        "outp[i] = tex2DLod<float4>(tex, fin[i], 0.5f, 0.0f).x;",
        params="cudaTextureObject_t tex",
    ),
    "storage-image": Template(
        "surf2Dwrite(make_float4(fin[i], fin[i], fin[i], fin[i]), img, 0, 0);\noutp[i] = 0.0f;",
        params="cudaSurfaceObject_t img",
    ),
    # CUDA has kernels only; graphics and ray-tracing stages belong to the graphics APIs and OptiX.
    **dict.fromkeys(
        (
            "discard",
            "vertex",
            "fragment",
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

# HIP keeps CUDA's kernel language and renames the runtime types.
HIP = {
    **CUDA,
    "half": _scalar("__half", "fin", "#include <hip/hip_fp16.h>"),
    "sampled-texture": Template(
        CUDA["sampled-texture"].body, params="hipTextureObject_t tex"
    ),
    "storage-image": Template(
        CUDA["storage-image"].body, params="hipSurfaceObject_t img"
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
    "int8": _scalar("int8_t"),
    "uint8": _scalar("uint8_t"),
    "num-workgroups": Template("outp[i] = float(WorkgroupCount().x);"),
    "vertex": _slang(HLSL["vertex"], "vertex"),
    "fragment": _slang(HLSL["fragment"], "fragment"),
    "discard": _slang(HLSL["discard"], "fragment"),
}

TEMPLATES = {
    "cuda": CUDA,
    "hip": HIP,
    "glsl": GLSL,
    "hlsl": HLSL,
    "metal": METAL,
    "slang": SLANG,
    "wgsl": WGSL,
    "spirv": {},
}

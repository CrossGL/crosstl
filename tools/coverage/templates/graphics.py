"""Probe templates for the shading languages: GLSL, HLSL, Metal, WGSL and Slang."""

from __future__ import annotations

import re
from dataclasses import replace

from .base import ABSENT, C_FAMILY, Template, scalar


def _glsl_extension(name: str) -> str:
    return f"#extension GL_EXT_shader_explicit_arithmetic_types_{name} : require"


GLSL = {
    **C_FAMILY,
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
    "discard": _slang(HLSL["discard"], "fragment"),
}

import textwrap

import pytest

from crosstl.backend.DirectX import DirectxCrossGLCodeGen
from crosstl.backend.DirectX.DirectxLexer import HLSLLexer
from crosstl.backend.DirectX.DirectxParser import HLSLParser
from crosstl.translator.ast import ShaderStage
from crosstl.translator.codegen.GLSL_codegen import GLSLCodeGen
from crosstl.translator.codegen.directx_codegen import (
    HLSLCodeGen as TranslatorHLSLCodeGen,
)
from crosstl.translator.codegen.metal_codegen import MetalCodeGen
from crosstl.translator.lexer import Lexer as CrossGLLexer
from crosstl.translator.parser import Parser as CrossGLParser

VERTEX_PIXEL_HLSL = textwrap.dedent("""
    cbuffer CameraBuffer : register(b0) {
        float4x4 viewProj;
        float3 eyePos;
        float padding;
    };

    Texture2D tex0 : register(t0);
    SamplerState samp0 : register(s0);

    struct VSInput {
        float3 position : POSITION;
        float3 normal : NORMAL;
        float2 uv : TEXCOORD0;
    };

    struct VSOutput {
        float4 position : SV_Position;
        float3 normal : TEXCOORD1;
        float2 uv : TEXCOORD0;
    };

    VSOutput VSMain(VSInput input) {
        VSOutput output;
        output.position = mul(viewProj, float4(input.position, 1.0));
        output.normal = input.normal;
        output.uv = input.uv;
        return output;
    }

    struct PSInput {
        float4 position : SV_Position;
        float3 normal : TEXCOORD1;
        float2 uv : TEXCOORD0;
    };

    struct PSOutput {
        float4 color : SV_Target0;
    };

    float4 Lighting(float3 n, float2 uv) {
        float3 lightDir = normalize(float3(0.0, 1.0, 0.0));
        float ndotl = max(dot(n, lightDir), 0.0);
        float4 texColor = tex0.Sample(samp0, uv);
        return texColor * ndotl;
    }

    PSOutput PSMain(PSInput input) {
        PSOutput output;
        output.color = Lighting(input.normal, input.uv);
        return output;
    }
    """).strip()

COMPUTE_HLSL = textwrap.dedent("""
    RWTexture2D<float4> outputTex : register(u0);

    [numthreads(8, 8, 1)]
    void CSMain(uint3 dtid : SV_DispatchThreadID) {
        outputTex[dtid.xy] = float4(1.0, 0.0, 0.0, 1.0);
    }
    """).strip()

CONSTANTS_HLSL = textwrap.dedent("""
    cbuffer Globals : register(b0) {
        float4 baseColor;
        int lightCount;
    };

    static const float PI = 3.14159;

    struct PSInput {
        float4 position : SV_Position;
    };

    struct PSOutput {
        float4 color : SV_Target0;
    };

    PSOutput PSMain(PSInput input) {
        PSOutput output;
        float factor = PI * float(lightCount);
        output.color = baseColor * factor;
        return output;
    }
    """).strip()

CONTROL_FLOW_SHADER_HLSL = textwrap.dedent("""
    int ControlFlow(int a, int b) {
        int sum = 0;
        for (int i = 0; i < 4; ++i) {
            if (i % 2 == 0) {
                continue;
            } else if (i == 3) {
                break;
            }
            sum += i;
        }

        int j = 0;
        while (j < 3) {
            sum += j;
            j++;
        }

        int k = 0;
        do {
            sum += k;
            k++;
        } while (k < 2);

        switch (a) {
            case 0:
                sum += 1;
                break;
            case 1:
                sum += 2;
                // fallthrough
            default:
                sum += 3;
                break;
        }

        int ternaryVal = (a > b) ? a : b;
        return sum + ternaryVal;
    }

    struct PSInput {
        float4 position : SV_Position;
    };

    struct PSOutput {
        float4 color : SV_Target0;
    };

    PSOutput PSMain(PSInput input) {
        PSOutput output;
        int value = ControlFlow(1, 2);
        float fv = float(value);
        output.color = float4(fv, fv, fv, 1.0);
        return output;
    }
    """).strip()

GEOMETRY_HLSL = textwrap.dedent("""
    struct GSInput {
        float4 pos : SV_Position;
    };
    struct GSOutput {
        float4 pos : SV_Position;
    };

    [maxvertexcount(3)]
    void GSMain(triangle GSInput input[3], inout TriangleStream<GSOutput> triStream) {
        GSOutput outVert;
        outVert.pos = input[0].pos;
        triStream.Append(outVert);
    }
    """).strip()

TESSELLATION_HLSL = textwrap.dedent("""
    struct HSInput {
        float4 pos : SV_Position;
    };
    struct HSOutput {
        float4 pos : SV_Position;
    };
    struct HSConstData {
        float edges[3] : SV_TessFactor;
        float inside : SV_InsideTessFactor;
    };

    [domain(\"tri\")]
    [partitioning(\"fractional_even\")]
    [outputtopology(\"triangle_cw\")]
    [outputcontrolpoints(3)]
    [patchconstantfunc(\"HSConst\")]
    HSOutput HSMain(InputPatch<HSInput, 3> patch, uint id : SV_OutputControlPointID) {
        HSOutput output;
        output.pos = patch[id].pos;
        return output;
    }

    [domain(\"tri\")]
    float4 DSMain(HSConstData data, const OutputPatch<HSOutput, 3> patch, float3 uvw : SV_DomainLocation) : SV_Position {
        return patch[0].pos;
    }
    """).strip()

MESH_TASK_HLSL = textwrap.dedent("""
    [shader(\"amplification\")]
    void ASMain() {
        DispatchMesh(1, 1, 1);
    }

    [shader(\"mesh\")]
    void MSMain() {
        SetMeshOutputCounts(1, 1);
    }
    """).strip()

RAYTRACING_HLSL = textwrap.dedent("""
    RaytracingAccelerationStructure accel : register(t0, space1);

    [shader(\"raygeneration\")]
    void RayGen() {
        TraceRay(accel, 0, 0xFF, 0, 1, 0, float3(0.0, 0.0, 0.0), 0.0, float3(0.0, 0.0, 1.0), 100.0, 0);
    }
    """).strip()

TEXTURE_SAMPLE_HLSL = textwrap.dedent("""
    Texture2D tex : register(t0);
    SamplerState samp : register(s0);
    float4 PSMain(float2 uv : TEXCOORD0) : SV_Target0 {
        return tex.Sample(samp, uv);
    }
    """).strip()

REGISTER_BINDINGS_HLSL = textwrap.dedent("""
    cbuffer FrameData : register(b0, space1) {
        float4x4 viewProj : packoffset(c0);
    };
    Texture2D tex0 : register(t0, space2);
    SamplerState samp0 : register(s0, space2);
    float4 main(float2 uv : TEXCOORD0) : SV_Target0 {
        return tex0.Sample(samp0, uv);
    }
    """).strip()

NUMTHREADS_HLSL = textwrap.dedent("""
    [numthreads(8, 8, 1)]
    void CSMain(uint3 dtid : SV_DispatchThreadID) {
    }
    """).strip()

ENUM_TYPEDEF_HLSL = textwrap.dedent("""
    enum BlendMode {
        BlendOpaque = 0,
        BlendAdd = 1,
    };
    typedef float4 Color;
    Color main(Color input) : SV_Target0 {
        return input;
    }
    """).strip()

ATTRIBUTE_HLSL = textwrap.dedent("""
    struct HSInput { float4 pos : SV_Position; };
    struct HSOutput { float4 pos : SV_Position; };
    [domain(\"tri\")]
    [partitioning(\"fractional_even\")]
    [outputtopology(\"triangle_cw\")]
    [outputcontrolpoints(3)]
    [patchconstantfunc(\"HSConst\")]
    HSOutput HSMain(InputPatch<HSInput, 3> patch, uint id : SV_OutputControlPointID) {
        HSOutput output;
        return output;
    }

    [earlydepthstencil]
    float4 PSMain(float4 pos : SV_Position) : SV_Target0 {
        return pos;
    }
    """).strip()

EXTRA_ATTRIBUTES_HLSL = textwrap.dedent("""
    [unroll]
    [loop]
    [branch]
    [flatten]
    [maxtessfactor(8)]
    [instance(2)]
    [fastopt]
    [allow_uav_condition]
    void AttrMain() { }
    """).strip()

INTERLOCKED_HLSL = textwrap.dedent("""
    RWStructuredBuffer<int> buffer : register(u0);
    void main() {
        int original;
        InterlockedAdd(buffer[0], 1, original);
    }
    """).strip()

BUFFER_OPS_HLSL = textwrap.dedent("""
    StructuredBuffer<int> buffer : register(t0);
    void main() {
        int v = buffer.Load(0);
    }
    """).strip()

WAVE_OPS_HLSL = textwrap.dedent("""
    uint main(uint value) : SV_Target0 {
        return WaveActiveSum(value);
    }
    """).strip()

RESOURCE_ARRAYS_HLSL = textwrap.dedent("""
    Texture2D textures[4] : register(t0, space1);
    SamplerState samplers[4] : register(s0, space1);
    float4 main(float2 uv : TEXCOORD0) : SV_Target0 {
        return textures[0].Sample(samplers[0], uv);
    }
    """).strip()

RW_TEXTURE_ARRAY_HLSL = textwrap.dedent("""
    RWTexture1DArray rwTex[2] : register(u0);
    void main() { }
    """).strip()

RAY_STAGES_HLSL = textwrap.dedent("""
    [shader(\"intersection\")]
    void IsMain() { }
    [shader(\"closesthit\")]
    void ChMain() { }
    [shader(\"anyhit\")]
    void AhMain() { }
    [shader(\"miss\")]
    void MsMain() { }
    [shader(\"callable\")]
    void ClMain() { }
    """).strip()

TEXTURE_METHODS_HLSL = textwrap.dedent("""
    Texture2D tex : register(t0);
    SamplerState samp : register(s0);
    float4 main(float2 uv : TEXCOORD0) : SV_Target0 {
        float4 a = tex.SampleLevel(samp, uv, 0.0);
        float4 b = tex.SampleCmp(samp, uv, 0.5);
        float4 c = tex.Gather(samp, uv);
        return a + b + c;
    }
    """).strip()

MS_TEXTURE_HLSL = textwrap.dedent("""
    Texture2DMS<float4> texMs : register(t0);
    float4 main() : SV_Target0 {
        return texMs.Load(int2(0, 0), 0);
    }
    """).strip()

MATH_INTRINSICS_HLSL = textwrap.dedent("""
    float4 main(float4 a : TEXCOORD0, float4 b : TEXCOORD1) : SV_Target0 {
        float d = dot(a.xyz, b.xyz);
        float3 n = normalize(a.xyz);
        float4 m = mul(a, b);
        float4 l = lerp(a, b, 0.5);
        float4 s = saturate(l);
        return float4(d, n.x, m.x, l.x);
    }
    """).strip()

DIMENSIONS_HLSL = textwrap.dedent("""
    Texture2D tex : register(t0);
    StructuredBuffer<int> buf : register(t1);
    void main() {
        uint w;
        uint h;
        tex.GetDimensions(w, h);
        uint len;
        buf.GetDimensions(len);
    }
    """).strip()

PAREN_HLSL = textwrap.dedent("""
    float main(float a : TEXCOORD0, float b : TEXCOORD1, float c : TEXCOORD2) : SV_Target0 {
        return saturate((a - b) / c);
    }
    """).strip()


def generate_crossgl(code: str) -> str:
    tokens = HLSLLexer(code).tokenize()
    ast = HLSLParser(tokens).parse()
    converter = DirectxCrossGLCodeGen.HLSLToCrossGLConverter()
    return converter.generate(ast)


def parse_crossgl(code: str):
    tokens = CrossGLLexer(code).get_tokens()
    parser = CrossGLParser(tokens)
    return parser.parse()


def test_codegen_vertex_fragment_roundtrip():
    output = generate_crossgl(VERTEX_PIXEL_HLSL)
    assert isinstance(output, str)
    assert output.strip()

    lowered = output.lower()
    assert "shader" in lowered
    assert "vertex" in lowered
    assert "fragment" in lowered
    assert "@" in output

    shader_ast = parse_crossgl(output)
    assert ShaderStage.VERTEX in shader_ast.stages
    assert ShaderStage.FRAGMENT in shader_ast.stages
    assert shader_ast.structs

    entry_points = [
        stage.entry_point
        for stage in shader_ast.stages.values()
        if getattr(stage, "entry_point", None)
    ]
    assert entry_points, "Expected entry points for shader stages"


def test_codegen_preserves_constants_and_names():
    output = generate_crossgl(CONSTANTS_HLSL)
    assert "baseColor" in output
    assert "lightCount" in output
    assert "PI" in output

    shader_ast = parse_crossgl(output)
    assert ShaderStage.FRAGMENT in shader_ast.stages


def test_codegen_compute_roundtrip():
    output = generate_crossgl(COMPUTE_HLSL)
    lowered = output.lower()
    assert "compute" in lowered

    shader_ast = parse_crossgl(output)
    assert ShaderStage.COMPUTE in shader_ast.stages


def test_codegen_control_flow_roundtrip():
    output = generate_crossgl(CONTROL_FLOW_SHADER_HLSL)
    shader_ast = parse_crossgl(output)
    assert ShaderStage.FRAGMENT in shader_ast.stages


def test_codegen_geometry_stage():
    output = generate_crossgl(GEOMETRY_HLSL)
    lowered = output.lower()
    assert "geometry" in lowered
    assert "@ maxvertexcount" in lowered


def test_codegen_tessellation_stages():
    output = generate_crossgl(TESSELLATION_HLSL)
    lowered = output.lower()
    assert "tessellation_control" in lowered
    assert "tessellation_evaluation" in lowered


def test_codegen_mesh_task_stages():
    output = generate_crossgl(MESH_TASK_HLSL)
    lowered = output.lower()
    assert "mesh" in lowered
    assert "task" in lowered


def test_codegen_raytracing_stage():
    output = generate_crossgl(RAYTRACING_HLSL)
    assert "ray_generation" in output.lower()


def test_codegen_texture_sample_mapping():
    output = generate_crossgl(TEXTURE_SAMPLE_HLSL)
    assert "texture(tex, samp, uv)" in output
    assert "texture_sample" not in output


def test_codegen_register_bindings_emitted():
    output = generate_crossgl(REGISTER_BINDINGS_HLSL)
    assert "@ register" in output


def test_codegen_numthreads_attribute_emitted():
    output = generate_crossgl(NUMTHREADS_HLSL)
    assert "@ numthreads" in output


def test_codegen_enum_and_typedef():
    output = generate_crossgl(ENUM_TYPEDEF_HLSL)
    assert "enum BlendMode" in output
    assert "typedef" in output


def test_codegen_attributes_emitted():
    output = generate_crossgl(ATTRIBUTE_HLSL)
    lowered = output.lower()
    assert "@ domain" in lowered
    assert "@ partitioning" in lowered
    assert "@ outputtopology" in lowered
    assert "@ outputcontrolpoints" in lowered
    assert "@ patchconstantfunc" in lowered
    assert "@ earlydepthstencil" in lowered


def test_codegen_extra_attributes_emitted():
    output = generate_crossgl(EXTRA_ATTRIBUTES_HLSL)
    lowered = output.lower()
    assert "@ unroll" in lowered
    assert "@ loop" in lowered
    assert "@ branch" in lowered
    assert "@ flatten" in lowered
    assert "@ maxtessfactor" in lowered
    assert "@ instance" in lowered
    assert "@ fastopt" in lowered
    assert "@ allow_uav_condition" in lowered


def test_codegen_interlocked_mapping():
    output = generate_crossgl(INTERLOCKED_HLSL)
    assert "atomicAdd" in output


def test_codegen_buffer_method_mapping():
    output = generate_crossgl(BUFFER_OPS_HLSL)
    assert "buffer_load" in output


def test_codegen_wave_ops_passthrough():
    output = generate_crossgl(WAVE_OPS_HLSL)
    assert "WaveActiveSum" in output


def test_codegen_resource_arrays_and_spaces():
    output = generate_crossgl(RESOURCE_ARRAYS_HLSL)
    assert "textures[4]" in output
    assert "@ register(t0, space1)" in output


def test_codegen_rw_texture_array_mapping():
    output = generate_crossgl(RW_TEXTURE_ARRAY_HLSL)
    assert "image1darray" in output.lower()


def test_codegen_1d_rw_texture_signedness_mapping():
    code = textwrap.dedent("""
        RWTexture1D<float4> line : register(u0);
        RWTexture1DArray<uint> counters : register(u1);
        RWTexture1D<int> signedLine : register(u2);

        [numthreads(1, 1, 1)]
        void CSMain(uint3 tid : SV_DispatchThreadID) {
            float4 c = line[tid.x];
            uint v = counters[uint2(tid.x, 0)];
            int s = signedLine[tid.x];
        }
    """).strip()

    output = generate_crossgl(code)
    assert "image1D line;" in output
    assert "uimage1DArray counters;" in output
    assert "iimage1D signedLine;" in output

    shader_ast = parse_crossgl(output)
    assert ShaderStage.COMPUTE in shader_ast.stages


def test_codegen_rw_texture_indexing_uses_image_operations():
    code = textwrap.dedent("""
        RWTexture1D<float4> line : register(u0);
        RWTexture1DArray<uint> counters : register(u1);
        RWTexture2D<float4> images[2] : register(u2);

        [numthreads(1, 1, 1)]
        void CSMain(uint3 tid : SV_DispatchThreadID) {
            float4 c = line[tid.x];
            line[tid.x] = c;
            uint v = counters[uint2(tid.x, 0)];
            counters[uint2(tid.x, 0)] += 1u;
            float4 d = images[0][tid.xy];
            images[1][tid.xy] = d;
            uint original;
            InterlockedAdd(counters[uint2(tid.x, 0)], 1u, original);
        }
    """).strip()

    output = generate_crossgl(code)
    assert "vec4 c = imageLoad(line, tid.x);" in output
    assert "imageStore(line, tid.x, c);" in output
    assert "uint v = imageLoad(counters, uvec2(tid.x, 0));" in output
    assert (
        "imageStore(counters, uvec2(tid.x, 0), "
        "imageLoad(counters, uvec2(tid.x, 0)) + 1);"
    ) in output
    assert "vec4 d = imageLoad(images[0], tid.xy);" in output
    assert "imageStore(images[1], tid.xy, d);" in output
    assert "original = imageAtomicAdd(counters, uvec2(tid.x, 0), 1u);" in output

    shader_ast = parse_crossgl(output)
    assert ShaderStage.COMPUTE in shader_ast.stages
    regenerated_hlsl = TranslatorHLSLCodeGen().generate(shader_ast)
    assert (
        "uint imageAtomicAdd_uimage1DArray("
        "RWTexture1DArray<uint> image, int2 coord, uint value)" in regenerated_hlsl
    )
    assert (
        "original = imageAtomicAdd_uimage1DArray(counters, uint2(tid.x, 0), 1u);"
        in regenerated_hlsl
    )


def test_codegen_preserves_uav_coherency_and_register_space():
    code = textwrap.dedent("""
        globallycoherent RWTexture2D<uint> counters : register(u4, space2);
        RWTexture2D<float4> outImage : register(u5, space2);
        globallycoherent RWStructuredBuffer<int> values : register(u6, space2);

        [numthreads(1, 1, 1)]
        void CSMain(uint3 tid : SV_DispatchThreadID) {
            uint value = counters[tid.xy];
            values.Store(0, int(value));
            outImage[tid.xy] = float4(value, 0, 0, 1);
        }
    """).strip()

    output = generate_crossgl(code)

    assert "@ globallycoherent" in output
    assert "@ register(u4, space2)" in output
    assert "uimage2D counters;" in output
    assert "@ register(u5, space2)" in output
    assert "image2D outImage;" in output
    assert "@ register(u6, space2)" in output
    assert "RWStructuredBuffer<int> values;" in output
    assert "uint value = imageLoad(counters, tid.xy);" in output
    assert "buffer_store(values, 0, int(value));" in output
    assert "imageStore(outImage, tid.xy, vec4(value, 0, 0, 1));" in output

    shader_ast = parse_crossgl(output)
    regenerated_hlsl = TranslatorHLSLCodeGen().generate(shader_ast)

    assert (
        "globallycoherent RWTexture2D<uint> counters : register(u4, space2);"
        in regenerated_hlsl
    )
    assert "RWTexture2D<float4> outImage : register(u5, space2);" in regenerated_hlsl
    assert (
        "globallycoherent RWStructuredBuffer<int> values : register(u6, space2);"
        in regenerated_hlsl
    )


def test_codegen_ray_shader_stages():
    output = generate_crossgl(RAY_STAGES_HLSL)
    lowered = output.lower()
    assert "ray_intersection" in lowered
    assert "ray_closest_hit" in lowered
    assert "ray_any_hit" in lowered
    assert "ray_miss" in lowered
    assert "ray_callable" in lowered


def test_codegen_texture_method_descriptors():
    converter = DirectxCrossGLCodeGen.HLSLToCrossGLConverter()

    assert converter.texture_method_descriptor("SampleLevel", 3) == {
        "member": "SampleLevel",
        "function": "textureLod",
        "texture_function": "textureLod",
        "buffer_function": None,
        "component": None,
        "usage": "regular",
        "buffer_when_max_args": None,
    }
    assert converter.texture_method_descriptor("SampleCmp", 3)["usage"] == "comparison"
    assert converter.texture_method_descriptor("GatherBlue", 2) == {
        "member": "GatherBlue",
        "function": "textureGather",
        "texture_function": "textureGather",
        "buffer_function": None,
        "component": "2",
        "usage": "regular",
        "buffer_when_max_args": None,
    }
    assert converter.texture_method_descriptor("Load", 1)["function"] == "buffer_load"
    assert converter.texture_method_descriptor("Load", 2)["function"] == "texelFetch"
    assert (
        converter.texture_method_descriptor("GetDimensions", 1)["function"]
        == "buffer_dimensions"
    )
    assert (
        converter.texture_method_descriptor("GetDimensions", 2)["function"]
        == "texture_dimensions"
    )
    assert converter.texture_method_descriptor("Store", 2) is None
    assert converter.resource_method_descriptor("Load", 1)["resource"] == "buffer"
    assert converter.resource_method_descriptor("Load", 2)["resource"] == "texture"
    assert (
        converter.resource_method_descriptor("GetDimensions", 1)["operation"]
        == "dimensions"
    )
    assert converter.resource_method_descriptor("Store", 2) == {
        "member": "Store",
        "function": "buffer_store",
        "texture_function": None,
        "buffer_function": "buffer_store",
        "component": None,
        "usage": None,
        "buffer_when_max_args": None,
        "resource": "buffer",
        "operation": "store",
    }
    assert converter.resource_method_descriptor("Append", 1)["operation"] == "append"
    assert converter.resource_method_descriptor("Consume", 0)["operation"] == "consume"


def test_codegen_texture_method_mappings():
    output = generate_crossgl(TEXTURE_METHODS_HLSL)
    assert "sampler2D tex;" in output
    assert "sampler2DShadow tex;" not in output
    assert "textureLod(tex, samp, uv, 0.0)" in output
    assert "textureCompare(tex, samp, uv, 0.5)" in output
    assert "textureGather(tex, samp, uv)" in output
    assert "texture_sample" not in output
    assert "texture_gather" not in output


def test_codegen_ms_texture_load_mapping():
    output = generate_crossgl(MS_TEXTURE_HLSL)
    assert "texelFetch(texMs, ivec2(0, 0), 0)" in output
    assert "texture_load" not in output


def test_codegen_texture_methods_roundtrip_through_translator_codegen():
    code = textwrap.dedent("""
        Texture2D tex : register(t0);
        SamplerState samp : register(s0);

        float4 main(float2 uv : TEXCOORD0) : SV_Target {
            float4 base = tex.Sample(samp, uv);
            float4 mip = tex.SampleLevel(samp, uv, 1.0);
            float4 grad = tex.SampleGrad(samp, uv, float2(1.0, 0.0), float2(0.0, 1.0));
            float4 gathered = tex.GatherRed(samp, uv);
            return base + mip + grad + gathered;
        }
    """).strip()

    crossgl = generate_crossgl(code)

    assert "sampler2D tex;" in crossgl
    assert "sampler samp;" in crossgl
    assert "texture(tex, samp, uv)" in crossgl
    assert "textureLod(tex, samp, uv, 1.0)" in crossgl
    assert "textureGrad(tex, samp, uv, vec2(1.0, 0.0), vec2(0.0, 1.0))" in crossgl
    assert "textureGather(tex, samp, uv, 0)" in crossgl
    assert "texture_sample" not in crossgl
    assert "texture_gather" not in crossgl

    glsl = GLSLCodeGen().generate(parse_crossgl(crossgl))
    assert "layout(binding = 0) uniform sampler2D tex;" in glsl
    assert "texture(tex, uv)" in glsl
    assert "textureLod(tex, uv, 1.0)" in glsl
    assert "textureGrad(tex, uv, vec2(1.0, 0.0), vec2(0.0, 1.0))" in glsl
    assert "textureGather(tex, uv, 0)" in glsl
    assert "texture_sample" not in glsl


def test_codegen_resource_array_receivers_use_canonical_calls():
    code = textwrap.dedent("""
        Texture2D textures[2] : register(t0);
        SamplerState samp : register(s0);
        RWTexture2D<float4> images[2] : register(u0);
        RWStructuredBuffer<int> buffers[2] : register(u2);

        float4 main(float2 uv : TEXCOORD0, uint index : TEXCOORD1) : SV_Target0 {
            float4 c = textures[index].Sample(samp, uv);
            images[index][uint2(1, 2)] = c;
            buffers[index].Store(0, 7);
            int v = buffers[index].Load(0);
            return c + images[index][uint2(1, 2)] + float4(v);
        }
    """).strip()

    crossgl = generate_crossgl(code)

    assert "sampler2D textures[2];" in crossgl
    assert "image2D images[2];" in crossgl
    assert "RWStructuredBuffer<int> buffers[2];" in crossgl
    assert "texture(textures[index], samp, uv)" in crossgl
    assert "imageStore(images[index], uvec2(1, 2), c);" in crossgl
    assert "buffer_store(buffers[index], 0, 7);" in crossgl
    assert "buffer_load(buffers[index], 0)" in crossgl
    assert ".Sample(" not in crossgl
    assert ".Store(" not in crossgl

    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    hlsl = TranslatorHLSLCodeGen().generate(shader_ast)
    assert "RWStructuredBuffer<int> buffers[2] : register(u2);" in hlsl
    assert "buffers[index].Store(0, 7);" in hlsl
    assert "buffers[index].Load(0)" in hlsl
    assert "buffer_store(" not in hlsl
    assert "buffer_load(" not in hlsl


def test_codegen_resource_array_spaces_roundtrip_for_srv_uav_and_typed_buffers():
    code = textwrap.dedent("""
        Texture2D textures[2] : register(t3, space1);
        SamplerState samplers[2] : register(s5, space1);
        RWTexture2D<float4> images[2] : register(u4, space2);
        RWStructuredBuffer<int> buffers[2] : register(u6, space3);
        RWBuffer<uint> counters[2] : register(u8, space4);

        float4 main(float2 uv : TEXCOORD0, uint index : TEXCOORD1) : SV_Target0 {
            float4 c = textures[index].Sample(samplers[index], uv);
            images[index][uint2(1, 2)] = c;
            buffers[index].Store(0, 7);
            int v = buffers[index].Load(0);
            uint oldValue;
            InterlockedAdd(counters[index][0], 1u, oldValue);
            return c + images[index][uint2(1, 2)] + float4(v + int(oldValue));
        }
    """).strip()

    crossgl = generate_crossgl(code)

    assert "@ register(t3, space1)" in crossgl
    assert "sampler2D textures[2];" in crossgl
    assert "@ register(s5, space1)" in crossgl
    assert "sampler samplers[2];" in crossgl
    assert "@ register(u4, space2)" in crossgl
    assert "image2D images[2];" in crossgl
    assert "@ register(u6, space3)" in crossgl
    assert "RWStructuredBuffer<int> buffers[2];" in crossgl
    assert "@ register(u8, space4)" in crossgl
    assert "RWBuffer<uint> counters[2];" in crossgl
    assert "texture(textures[index], samplers[index], uv)" in crossgl
    assert "imageStore(images[index], uvec2(1, 2), c);" in crossgl
    assert "buffer_store(buffers[index], 0, 7);" in crossgl
    assert "buffer_load(buffers[index], 0)" in crossgl
    assert "oldValue = atomicAdd(counters[index][0], 1u);" in crossgl

    hlsl = TranslatorHLSLCodeGen().generate(parse_crossgl(crossgl))

    assert "Texture2D textures[2] : register(t3, space1);" in hlsl
    assert "SamplerState samplers[2] : register(s5, space1);" in hlsl
    assert "RWTexture2D<float4> images[2] : register(u4, space2);" in hlsl
    assert "RWStructuredBuffer<int> buffers[2] : register(u6, space3);" in hlsl
    assert "RWBuffer<uint> counters[2] : register(u8, space4);" in hlsl
    assert "textures[index].Sample(samplers[index], uv)" in hlsl
    assert "images[index][uint2(1, 2)] = c;" in hlsl
    assert "buffers[index].Store(0, 7);" in hlsl
    assert "buffers[index].Load(0)" in hlsl
    assert "InterlockedAdd(counters[index][0], 1u, oldValue);" in hlsl
    assert "buffer_store(" not in hlsl
    assert "buffer_load(" not in hlsl
    assert "atomicAdd(counters" not in hlsl


def test_codegen_append_consume_structured_buffers_roundtrip():
    code = textwrap.dedent("""
        AppendStructuredBuffer<int> appendValues : register(u1);
        ConsumeStructuredBuffer<int> consumeValues : register(u2);

        void main(uint value : TEXCOORD0) {
            appendValues.Append(int(value));
            int consumed = consumeValues.Consume();
        }
    """).strip()

    crossgl = generate_crossgl(code)

    assert "AppendStructuredBuffer<int> appendValues;" in crossgl
    assert "ConsumeStructuredBuffer<int> consumeValues;" in crossgl
    assert "buffer_append(appendValues, int(value));" in crossgl
    assert "int consumed = buffer_consume(consumeValues);" in crossgl
    assert ".Append(" not in crossgl
    assert ".Consume(" not in crossgl

    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    hlsl = TranslatorHLSLCodeGen().generate(shader_ast)
    assert "AppendStructuredBuffer<int> appendValues : register(u1);" in hlsl
    assert "ConsumeStructuredBuffer<int> consumeValues : register(u2);" in hlsl
    assert "appendValues.Append(int(value));" in hlsl
    assert "int consumed = consumeValues.Consume();" in hlsl
    assert "buffer_append(" not in hlsl
    assert "buffer_consume(" not in hlsl


def test_codegen_sample_cmp_infers_shadow_texture_for_translator_roundtrip():
    code = textwrap.dedent("""
        Texture2D<float> shadowMap : register(t0);
        SamplerComparisonState compareSampler : register(s0);

        float4 main(float2 uv : TEXCOORD0, float depth : TEXCOORD1) : SV_Target {
            float sampled = shadowMap.SampleCmpLevelZero(compareSampler, uv, depth);
            return float4(sampled);
        }
    """).strip()

    crossgl = generate_crossgl(code)

    assert "sampler2DShadow shadowMap;" in crossgl
    assert "sampler compareSampler;" in crossgl
    assert "samplerShadow" not in crossgl
    assert "textureCompare(shadowMap, compareSampler, uv, depth)" in crossgl

    ast = parse_crossgl(crossgl)
    glsl = GLSLCodeGen().generate(ast)
    assert "layout(binding = 0) uniform sampler2DShadow shadowMap;" in glsl
    assert "texture(shadowMap, vec3(uv, depth))" in glsl
    assert "unsupported GLSL texture compare" not in glsl

    hlsl = TranslatorHLSLCodeGen().generate(ast)
    assert "Texture2D shadowMap" in hlsl
    assert "SamplerComparisonState compareSampler" in hlsl
    assert "shadowMap.SampleCmp(compareSampler, uv, depth)" in hlsl


def test_codegen_sample_cmp_infers_shadow_arrays_cubes_and_helper_args():
    code = textwrap.dedent("""
        Texture2DArray<float> shadowArray : register(t0);
        TextureCube<float> shadowCube : register(t1);
        Texture2D<float> shadowMap : register(t2);
        SamplerComparisonState compareSampler : register(s0);

        float helper(Texture2D<float> localShadow, SamplerComparisonState localSampler, float2 uv, float depth) {
            return localShadow.SampleCmpLevelZero(localSampler, uv, depth);
        }

        float4 main(float3 uvLayer : TEXCOORD0, float3 direction : TEXCOORD1, float2 uv : TEXCOORD2, float depth : TEXCOORD3) : SV_Target {
            float a = shadowArray.SampleCmpLevelZero(compareSampler, uvLayer, depth);
            float b = shadowCube.SampleCmpLevelZero(compareSampler, direction, depth);
            float c = helper(shadowMap, compareSampler, uv, depth);
            return float4(a + b + c);
        }
    """).strip()

    crossgl = generate_crossgl(code)

    assert "sampler2DArrayShadow shadowArray;" in crossgl
    assert "samplerCubeShadow shadowCube;" in crossgl
    assert "sampler2DShadow shadowMap;" in crossgl
    assert "float helper(sampler2DShadow localShadow, sampler localSampler" in crossgl
    assert "textureCompare(shadowArray, compareSampler, uvLayer, depth)" in crossgl
    assert "textureCompare(shadowCube, compareSampler, direction, depth)" in crossgl
    assert "helper(shadowMap, compareSampler, uv, depth)" in crossgl

    ast = parse_crossgl(crossgl)
    glsl = GLSLCodeGen().generate(ast)
    assert "layout(binding = 0) uniform sampler2DArrayShadow shadowArray;" in glsl
    assert "layout(binding = 1) uniform samplerCubeShadow shadowCube;" in glsl
    assert "layout(binding = 2) uniform sampler2DShadow shadowMap;" in glsl
    assert "texture(shadowArray, vec4(uvLayer, depth))" in glsl
    assert "texture(shadowCube, vec4(direction, depth))" in glsl
    assert "helper(shadowMap, uv, depth)" in glsl
    assert "unsupported GLSL texture compare" not in glsl

    hlsl = TranslatorHLSLCodeGen().generate(ast)
    assert "Texture2DArray shadowArray" in hlsl
    assert "TextureCube shadowCube" in hlsl
    assert "Texture2D shadowMap" in hlsl
    assert "shadowArray.SampleCmp(compareSampler, uvLayer, depth)" in hlsl
    assert "shadowCube.SampleCmp(compareSampler, direction, depth)" in hlsl
    assert "localShadow.SampleCmp(localSampler, uv, depth)" in hlsl

    metal = MetalCodeGen().generate(ast)
    assert "depth2d_array<float> shadowArray" in metal
    assert "depthcube<float> shadowCube" in metal
    assert "depth2d<float> shadowMap" in metal
    assert (
        "shadowArray.sample_compare(compareSampler, uvLayer.xy, uint(uvLayer.z), depth)"
        in metal
    )
    assert "shadowCube.sample_compare(compareSampler, direction, depth)" in metal
    assert "localShadow.sample_compare(localSampler, uv, depth)" in metal


def test_codegen_helper_propagation_keeps_mixed_texture_non_shadow():
    code = textwrap.dedent("""
        Texture2D<float4> tex : register(t0);
        SamplerState samp : register(s0);
        SamplerComparisonState compareSampler : register(s1);

        float4 readRegular(Texture2D<float4> regularSource, SamplerState localSampler, float2 uv) {
            return regularSource.Sample(localSampler, uv);
        }

        float readShadow(Texture2D<float> shadowSource, SamplerComparisonState localSampler, float2 uv, float depth) {
            return shadowSource.SampleCmpLevelZero(localSampler, uv, depth);
        }

        float4 main(float2 uv : TEXCOORD0, float depth : TEXCOORD1) : SV_Target {
            return readRegular(tex, samp, uv) + float4(readShadow(tex, compareSampler, uv, depth));
        }
    """).strip()

    crossgl = generate_crossgl(code)

    assert "sampler2D tex;" in crossgl
    assert "sampler2DShadow tex;" not in crossgl
    assert "vec4 readRegular(sampler2D regularSource, sampler localSampler" in crossgl
    assert (
        "float readShadow(sampler2DShadow shadowSource, sampler localSampler" in crossgl
    )
    assert "readRegular(tex, samp, uv)" in crossgl
    assert "readShadow(tex, compareSampler, uv, depth)" in crossgl


def test_codegen_shadow_inference_keeps_parameter_names_scoped():
    code = textwrap.dedent("""
        Texture2D<float> shared : register(t0);
        SamplerComparisonState compareSampler : register(s0);

        float4 readRegular(Texture2D<float4> shared, SamplerState localSampler, float2 uv) {
            return shared.Sample(localSampler, uv);
        }

        float4 main(float2 uv : TEXCOORD0, float depth : TEXCOORD1) : SV_Target {
            float sampled = shared.SampleCmpLevelZero(compareSampler, uv, depth);
            return float4(sampled);
        }
    """).strip()

    crossgl = generate_crossgl(code)

    assert "sampler2DShadow shared;" in crossgl
    assert "vec4 readRegular(sampler2D shared, sampler localSampler" in crossgl
    assert "textureCompare(shared, compareSampler, uv, depth)" in crossgl
    assert "texture(shared, localSampler, uv)" in crossgl


def test_codegen_sample_cmp_infers_cube_array_shadow_through_nested_wrappers():
    code = textwrap.dedent("""
        TextureCubeArray<float> cubeShadowArray : register(t0);
        SamplerComparisonState compareSampler : register(s0);

        float leaf(TextureCubeArray<float> leafShadow, SamplerComparisonState leafSampler, float4 cubeLayer, float depth) {
            return leafShadow.SampleCmpLevelZero(leafSampler, cubeLayer, depth);
        }

        float wrapper(TextureCubeArray<float> wrappedShadow, SamplerComparisonState wrappedSampler, float4 cubeLayer, float depth) {
            return leaf(wrappedShadow, wrappedSampler, cubeLayer, depth);
        }

        float4 main(float4 cubeLayer : TEXCOORD0, float depth : TEXCOORD1) : SV_Target {
            float sampled = wrapper(cubeShadowArray, compareSampler, cubeLayer, depth);
            return float4(sampled);
        }
    """).strip()

    crossgl = generate_crossgl(code)

    assert "samplerCubeArrayShadow cubeShadowArray;" in crossgl
    assert (
        "float leaf(samplerCubeArrayShadow leafShadow, sampler leafSampler" in crossgl
    )
    assert (
        "float wrapper(samplerCubeArrayShadow wrappedShadow, sampler wrappedSampler"
        in crossgl
    )
    assert "textureCompare(leafShadow, leafSampler, cubeLayer, depth)" in crossgl
    assert "leaf(wrappedShadow, wrappedSampler, cubeLayer, depth)" in crossgl
    assert "wrapper(cubeShadowArray, compareSampler, cubeLayer, depth)" in crossgl

    ast = parse_crossgl(crossgl)
    glsl = GLSLCodeGen().generate(ast)
    assert "layout(binding = 0) uniform samplerCubeArrayShadow cubeShadowArray;" in glsl
    assert (
        "float leaf(samplerCubeArrayShadow leafShadow, vec4 cubeLayer, float depth)"
        in glsl
    )
    assert "texture(leafShadow, cubeLayer, depth)" in glsl
    assert "leaf(wrappedShadow, cubeLayer, depth)" in glsl
    assert "wrapper(cubeShadowArray, cubeLayer, depth)" in glsl
    assert "unsupported GLSL texture compare" not in glsl

    hlsl = TranslatorHLSLCodeGen().generate(ast)
    assert "TextureCubeArray cubeShadowArray" in hlsl
    assert (
        "float leaf(TextureCubeArray leafShadow, SamplerComparisonState leafSampler"
        in hlsl
    )
    assert "leafShadow.SampleCmp(leafSampler, cubeLayer, depth)" in hlsl
    assert "leaf(wrappedShadow, wrappedSampler, cubeLayer, depth)" in hlsl

    metal = MetalCodeGen().generate(ast)
    assert "depthcube_array<float> cubeShadowArray" in metal
    assert "float leaf(depthcube_array<float> leafShadow, sampler leafSampler" in metal
    assert (
        "leafShadow.sample_compare(leafSampler, cubeLayer.xyz, uint(cubeLayer.w), depth)"
        in metal
    )
    assert "leaf(wrappedShadow, wrappedSampler, cubeLayer, depth)" in metal


def test_codegen_nested_cube_array_helper_mixed_use_stays_non_shadow():
    code = textwrap.dedent("""
        TextureCubeArray<float4> cubeTex : register(t0);
        SamplerState samp : register(s0);
        SamplerComparisonState compareSampler : register(s1);

        float4 regularLeaf(TextureCubeArray<float4> source, SamplerState localSampler, float4 cubeLayer) {
            return source.Sample(localSampler, cubeLayer);
        }

        float4 regularWrapper(TextureCubeArray<float4> source, SamplerState localSampler, float4 cubeLayer) {
            return regularLeaf(source, localSampler, cubeLayer);
        }

        float shadowLeaf(TextureCubeArray<float> source, SamplerComparisonState localSampler, float4 cubeLayer, float depth) {
            return source.SampleCmpLevelZero(localSampler, cubeLayer, depth);
        }

        float shadowWrapper(TextureCubeArray<float> source, SamplerComparisonState localSampler, float4 cubeLayer, float depth) {
            return shadowLeaf(source, localSampler, cubeLayer, depth);
        }

        float4 main(float4 cubeLayer : TEXCOORD0, float depth : TEXCOORD1) : SV_Target {
            return regularWrapper(cubeTex, samp, cubeLayer) + float4(shadowWrapper(cubeTex, compareSampler, cubeLayer, depth));
        }
    """).strip()

    crossgl = generate_crossgl(code)

    assert "samplerCubeArray cubeTex;" in crossgl
    assert "samplerCubeArrayShadow cubeTex;" not in crossgl
    assert "vec4 regularLeaf(samplerCubeArray source, sampler localSampler" in crossgl
    assert (
        "float shadowLeaf(samplerCubeArrayShadow source, sampler localSampler"
        in crossgl
    )
    assert "regularWrapper(cubeTex, samp, cubeLayer)" in crossgl
    assert "shadowWrapper(cubeTex, compareSampler, cubeLayer, depth)" in crossgl


def test_codegen_math_intrinsics_mapping():
    output = generate_crossgl(MATH_INTRINSICS_HLSL)
    assert "dot" in output
    assert "normalize" in output
    assert "mul" in output
    assert "mix" in output
    assert "clamp(" in output


def test_codegen_get_dimensions_mapping():
    output = generate_crossgl(DIMENSIONS_HLSL)
    assert "texture_dimensions" in output
    assert "buffer_dimensions" in output

    regenerated_hlsl = TranslatorHLSLCodeGen().generate(parse_crossgl(output))
    assert "buf.GetDimensions(len);" in regenerated_hlsl
    assert "buffer_dimensions" not in regenerated_hlsl


def test_codegen_preserves_parentheses():
    output = generate_crossgl(PAREN_HLSL)
    assert "clamp((a - b) / c, 0.0, 1.0)" in output


def test_codegen_multiview_and_viewport_semantics():
    code = textwrap.dedent("""
        struct GSInput {
            float4 pos : SV_Position;
        };
        struct GSOutput {
            float4 pos : SV_Position;
            uint view : SV_ViewID;
            uint layer : SV_RenderTargetArrayIndex;
            uint viewport : SV_ViewportArrayIndex;
        };
        [maxvertexcount(1)]
        void GSMain(point GSInput input[1], inout PointStream<GSOutput> stream) {
            GSOutput o;
            o.pos = input[0].pos;
            o.view = 1;
            o.layer = 2;
            o.viewport = 3;
            stream.Append(o);
        }
    """).strip()

    output = generate_crossgl(code)
    for expected in ["gl_ViewID", "gl_Layer", "gl_ViewportIndex"]:
        assert expected in output


def test_codegen_interlocked_compare_exchange_mapping():
    code = textwrap.dedent("""
        RWTexture2D<uint> tex : register(u0);
        RWBuffer<uint> buf : register(u1);

        [numthreads(1, 1, 1)]
        void CSMain(uint3 dtid : SV_DispatchThreadID) {
            uint original;
            InterlockedCompareExchange(tex[dtid.xy], 1u, 0u, original);
            InterlockedCompareExchange(buf[dtid.x], 2u, 1u, original);
        }
    """).strip()

    output = generate_crossgl(code)
    assert "original = imageAtomicCompSwap(tex, dtid.xy, 1u, 0u);" in output
    assert "RWBuffer<uint> buf;" in output
    assert "original = atomicCompareExchange(buf[dtid.x], 2u, 1u);" in output

    regenerated_hlsl = TranslatorHLSLCodeGen().generate(parse_crossgl(output))
    assert "RWBuffer<uint> buf : register(u1);" in regenerated_hlsl
    assert (
        "InterlockedCompareExchange(buf[dtid.x], 2u, 1u, original);" in regenerated_hlsl
    )
    assert "atomicCompareExchange(buf" not in regenerated_hlsl


def test_codegen_interlocked_typed_buffer_resource_array_roundtrip():
    code = textwrap.dedent("""
        RWBuffer<uint> counterBuffers[2] : register(u4);
        RWStructuredBuffer<int> signedValues : register(u6);

        [numthreads(1, 1, 1)]
        void CSMain(uint3 dtid : SV_DispatchThreadID) {
            uint original;
            InterlockedAdd(counterBuffers[1][dtid.x], 1u, original);
            int oldSigned;
            InterlockedMax(signedValues[dtid.x], -1, oldSigned);
        }
    """).strip()

    output = generate_crossgl(code)

    assert "RWBuffer<uint> counterBuffers[2];" in output
    assert "RWStructuredBuffer<int> signedValues;" in output
    assert "original = atomicAdd(counterBuffers[1][dtid.x], 1u);" in output
    assert "oldSigned = atomicMax(signedValues[dtid.x], -1);" in output

    regenerated_hlsl = TranslatorHLSLCodeGen().generate(parse_crossgl(output))
    assert "RWBuffer<uint> counterBuffers[2] : register(u4);" in regenerated_hlsl
    assert "RWStructuredBuffer<int> signedValues : register(u6);" in regenerated_hlsl
    assert (
        "InterlockedAdd(counterBuffers[1][dtid.x], 1u, original);" in regenerated_hlsl
    )
    assert "InterlockedMax(signedValues[dtid.x], -1, oldSigned);" in regenerated_hlsl
    assert "atomicAdd(counterBuffers" not in regenerated_hlsl
    assert "atomicMax(signedValues" not in regenerated_hlsl


def test_codegen_invalid_hlsl_raises():
    code = "float4 main() : SV_Target0 { float x = 1.0 return float4(x, 0, 0, 1); }"
    with pytest.raises(SyntaxError):
        generate_crossgl(code)


if __name__ == "__main__":
    pytest.main()

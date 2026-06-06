import textwrap

import pytest

from crosstl.backend.DirectX import DirectxCrossGLCodeGen
from crosstl.backend.DirectX.DirectxLexer import HLSLLexer
from crosstl.backend.DirectX.DirectxParser import HLSLParser
from crosstl.translator.ast import ArrayLiteralNode, ArrayType, ShaderStage
from crosstl.translator.codegen.directx_codegen import (
    HLSLCodeGen as TranslatorHLSLCodeGen,
)
from crosstl.translator.codegen.GLSL_codegen import GLSLCodeGen
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

RAY_QUERY_HLSL = textwrap.dedent("""
    RaytracingAccelerationStructure accel : register(t0);

    [shader(\"raygeneration\")]
    void RayGen() {
        RayDesc ray;
        ray.Origin = float3(0.0, 0.0, 0.0);
        ray.TMin = 0.001;
        ray.Direction = float3(0.0, 0.0, 1.0);
        ray.TMax = 100.0;

        RayQuery<RAY_FLAG_NONE> rq;
        rq.TraceRayInline(accel, RAY_FLAG_NONE, 0xFF, ray);
        while (rq.Proceed()) {
            if (rq.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE) {
                rq.CommitNonOpaqueTriangleHit();
            }
        }

        uint status = rq.CommittedStatus();
        float t = rq.CommittedRayT();
    }
    """).strip()

TEXTURE_SAMPLE_HLSL = textwrap.dedent("""
    Texture2D tex : register(t0);
    SamplerState samp : register(s0);
    float4 PSMain(float2 uv : TEXCOORD0) : SV_Target0 {
        return tex.Sample(samp, uv);
    }
    """).strip()

SAMPLER_STATE_BLOCK_HLSL = textwrap.dedent("""
    Texture2D tex : register(t0);
    SamplerState MeshTextureSampler
    {
        Filter = MIN_MAG_MIP_LINEAR;
        AddressU = Wrap;
        AddressV = Wrap;
    };

    float4 PSMain(float2 uv : TEXCOORD0) : SV_Target0 {
        return tex.Sample(MeshTextureSampler, uv);
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

BYTE_ADDRESS_VECTOR_OPS_HLSL = textwrap.dedent("""
    ByteAddressBuffer rawInput : register(t3);
    RWByteAddressBuffer rawOutput : register(u4);

    uint4 main(uint offset : TEXCOORD0) : SV_Target0 {
        uint scalar = rawInput.Load<uint>(offset + 48u);
        uint2 pair = rawInput.Load2(offset);
        uint3 triple = rawOutput.Load3(offset + 16u);
        uint4 quad = rawOutput.Load4(offset + 32u);
        rawOutput.Store<uint>(offset + 48u, scalar);
        rawOutput.Store2(offset, pair);
        rawOutput.Store3(offset + 16u, triple);
        rawOutput.Store4(offset + 32u, quad);
        return quad;
    }
    """).strip()

WAVE_OPS_HLSL = textwrap.dedent("""
    uint main(uint value) : SV_Target0 {
        bool predicate = value != 0u;
        uint lane = WaveGetLaneIndex();
        uint laneCount = WaveGetLaneCount();
        bool first = WaveIsFirstLane();
        uint sum = WaveActiveSum(value);
        uint product = WaveActiveProduct(value);
        uint minimum = WaveActiveMin(value);
        uint maximum = WaveActiveMax(value);
        bool allTrue = WaveActiveAllTrue(predicate);
        bool anyTrue = WaveActiveAnyTrue(predicate);
        bool allEqual = WaveActiveAllEqual(value);
        uint4 ballot = WaveActiveBallot(predicate);
        uint countBits = WaveActiveCountBits(predicate);
        uint laneValue = WaveReadLaneAt(value, 0u);
        uint firstValue = WaveReadLaneFirst(value);
        uint prefixSum = WavePrefixSum(value);
        uint prefixProduct = WavePrefixProduct(value);
        uint prefixCount = WavePrefixCountBits(predicate);
        uint4 matchMask = WaveMatch(value);
        uint multiSum = WaveMultiPrefixSum(value, ballot);
        uint multiCount = WaveMultiPrefixCountBits(predicate, ballot);
        uint multiProduct = WaveMultiPrefixProduct(value, ballot);
        uint multiAnd = WaveMultiPrefixBitAnd(value, ballot);
        uint multiOr = WaveMultiPrefixBitOr(value, ballot);
        uint multiXor = WaveMultiPrefixBitXor(value, ballot);
        uint quadX = QuadReadAcrossX(value);
        uint quadY = QuadReadAcrossY(value);
        uint quadDiag = QuadReadAcrossDiagonal(value);
        uint quadLane = QuadReadLaneAt(value, 2u);
        bool quadAny = QuadAny(predicate);
        bool quadAll = QuadAll(predicate);
        return lane + laneCount + sum + product + minimum + maximum + laneValue
            + firstValue + prefixSum + prefixProduct + matchMask.x + multiSum
            + multiCount + multiProduct + multiAnd + multiOr + multiXor
            + quadX + quadY + quadDiag + quadLane + ballot.x + countBits
            + prefixCount + (first ? 1u : 0u) + (allTrue ? 1u : 0u)
            + (anyTrue ? 1u : 0u) + (allEqual ? 1u : 0u)
            + (quadAny ? 1u : 0u) + (quadAll ? 1u : 0u);
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

INTERPOLATION_INTRINSICS_HLSL = textwrap.dedent("""
    float4 main(float4 color : COLOR0, uint sampleIndex : SV_SampleIndex) : SV_Target0 {
        int2 snappedOffset = int2(1, -1);
        float4 atSample = EvaluateAttributeAtSample(color, sampleIndex);
        float4 atOffset = EvaluateAttributeSnapped(color, snappedOffset);
        float4 atCentroid = EvaluateAttributeCentroid(color);
        return atSample + atOffset + atCentroid;
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


def test_hlsl_psize_roundtrips_to_gl_point_size():
    crossgl = generate_crossgl("float pointSize() : PSIZE { return 1.0; }")

    assert "@ gl_PointSize" in crossgl
    assert "@ PointSize" not in crossgl


def test_codegen_shaderlab_legacy_tex2d_imports_canonical_texture_call():
    shaderlab = textwrap.dedent("""
        Shader "Custom/LegacyTex2D"
        {
            SubShader
            {
                Pass
                {
                    CGPROGRAM
                    sampler2D _MainTex;

                    struct v2f {
                        float4 position : SV_POSITION;
                        float2 uv : TEXCOORD0;
                    };

                    float4 frag(v2f i) : SV_Target
                    {
                        return tex2D(_MainTex, i.uv);
                    }
                    ENDCG
                }
            }
        }
        """).strip()

    crossgl = generate_crossgl(shaderlab)

    assert "sampler2D _MainTex;" in crossgl
    assert "return texture(_MainTex, i.uv);" in crossgl
    assert "tex2D(" not in crossgl

    hlsl = TranslatorHLSLCodeGen().generate(parse_crossgl(crossgl))

    assert "Texture2D _MainTex" in hlsl
    assert "SamplerState _MainTexSampler" in hlsl
    assert "_MainTex.Sample(_MainTexSampler, i.uv)" in hlsl


def test_codegen_shaderlab_legacy_tex2d_lod_bias_grad_imports_canonical_calls():
    shaderlab = textwrap.dedent("""
        Shader "Custom/LegacyTex2DVariants"
        {
            SubShader
            {
                Pass
                {
                    CGPROGRAM
                    sampler2D _MainTex;

                    struct v2f {
                        float4 position : SV_POSITION;
                        float2 uv : TEXCOORD0;
                    };

                    float4 frag(v2f i) : SV_Target
                    {
                        float4 lodColor = tex2Dlod(_MainTex, float4(i.uv, 0.0, 2.0));
                        float4 biasColor = tex2Dbias(_MainTex, float4(i.uv.x, i.uv.y, 0.0, 0.5));
                        float4 gradColor = tex2Dgrad(_MainTex, i.uv, float2(1.0, 0.0), float2(0.0, 1.0));
                        return lodColor + biasColor + gradColor;
                    }
                    ENDCG
                }
            }
        }
        """).strip()

    crossgl = generate_crossgl(shaderlab)

    assert "vec4 lodColor = textureLod(_MainTex, i.uv, 2.0);" in crossgl
    assert "vec4 biasColor = texture(_MainTex, vec2(i.uv.x, i.uv.y), 0.5);" in crossgl
    assert (
        "vec4 gradColor = textureGrad(_MainTex, i.uv, vec2(1.0, 0.0), "
        "vec2(0.0, 1.0));" in crossgl
    )
    assert "tex2Dlod(" not in crossgl
    assert "tex2Dbias(" not in crossgl
    assert "tex2Dgrad(" not in crossgl

    ast = parse_crossgl(crossgl)
    hlsl = TranslatorHLSLCodeGen().generate(ast)

    assert "_MainTex.SampleLevel(_MainTexSampler, i.uv, 2.0)" in hlsl
    assert "_MainTex.SampleBias(_MainTexSampler, float2(i.uv.x, i.uv.y), 0.5)" in hlsl
    assert (
        "_MainTex.SampleGrad(_MainTexSampler, i.uv, float2(1.0, 0.0), "
        "float2(0.0, 1.0))" in hlsl
    )


def test_codegen_pragma_once_from_directx_fallback_samples():
    hlsl = textwrap.dedent("""
        #pragma once

        struct RWByteAddressBufferPointer {
            RWByteAddressBuffer buffer;
            uint offsetInBytes;
        };
        """).strip()

    crossgl = generate_crossgl(hlsl)

    assert "#pragma once\n" in crossgl
    assert "#pragma once None" not in crossgl
    assert "#pragma once;" not in crossgl
    assert "struct RWByteAddressBufferPointer" in crossgl
    assert parse_crossgl(crossgl) is not None


def test_codegen_skips_struct_forward_declarations_from_dxc_tests():
    # Sources:
    # microsoft/DirectXShaderCompiler tools/clang/test/SemaHLSL/sizeof-requires-complete-type.hlsl
    # microsoft/DirectXShaderCompiler tools/clang/test/CodeGenDXIL/templates/incomplete-target-in-CanConvert.hlsl
    crossgl = generate_crossgl("""
        struct Payload;
        template<typename T> struct Wrapper;

        struct Payload {
            float value;
        };

        float get(Wrapper<float> o);
        float readPayload(Payload payload) {
            return payload.value;
        }
    """)

    assert crossgl.count("struct Payload") == 1
    assert "struct Wrapper" not in crossgl
    assert "struct Payload {\n        float value;" in crossgl
    parse_crossgl(crossgl)


def test_codegen_preserves_interpolation_modifiers_as_crossgl_metadata():
    crossgl = generate_crossgl("""
        struct PSInput {
            centroid noperspective float2 uv : TEXCOORD0;
            nointerpolation uint id : TEXCOORD1;
            sample float4 color : COLOR0;
        };

        float4 PSMain(noperspective float4 color : COLOR0) : SV_Target0 {
            return color;
        }
    """)

    assert "vec2 uv @ TexCoord0 @ noperspective @ centroid;" in crossgl
    assert "uint id @ TexCoord1 @ flat;" in crossgl
    assert "vec4 color @ Color0 @ sample;" in crossgl
    assert "vec4 color @ Color0 @ noperspective" in crossgl
    parse_crossgl(crossgl)


def test_codegen_preserves_contextual_shared_storage_modifier():
    crossgl = generate_crossgl("""
        shared float cachedWeight;
        Texture2D<float> shared : register(t0);

        float4 PSMain(float2 uv : TEXCOORD0) : SV_Target0 {
            return shared.SampleLevel(samplerState, uv, 0.0);
        }
    """)

    assert "shared float cachedWeight;" in crossgl
    assert "sampler2D shared;" in crossgl
    assert "shared sampler2D shared;" not in crossgl
    parse_crossgl(crossgl)


def test_codegen_clip_intrinsic_imports_to_parseable_crossgl():
    crossgl = generate_crossgl("""
        float4 PSMain(float4 color : COLOR0) : SV_Target {
            clip(color.a < 0.1f ? -1 : 1);
            return color;
        }
    """)

    assert "if ((color.a < 0.1 ? -1 : 1) < 0.0) {" in crossgl
    assert "discard;" in crossgl
    assert "clip(" not in crossgl
    ast = parse_crossgl(crossgl)
    glsl = GLSLCodeGen().generate(ast)
    assert "discard(" not in glsl
    assert "discard;" in glsl


def test_codegen_vector_clip_intrinsic_imports_to_conditional_discard():
    crossgl = generate_crossgl("""
        float4 PSMain(float4 color : COLOR0) : SV_Target {
            clip(color);
            return color;
        }
    """)

    assert "if (any(lessThan(color, vec4(0.0)))) {" in crossgl
    ast = parse_crossgl(crossgl)
    glsl = GLSLCodeGen().generate(ast)
    assert "discard(" not in glsl
    assert "any(lessThan(color, vec4(0.0)))" in glsl


def test_codegen_clip_identifier_from_vkd3d_clip_distance_sample():
    crossgl = generate_crossgl("""
        struct VertexOut {
            float4 position : SV_POSITION;
            float clip : SV_CLIPDISTANCE;
            float cull : SV_CULLDISTANCE1;
        };

        void main(float4 position : POSITION, out VertexOut outputVertex) {
            outputVertex.position = position;
            outputVertex.clip = position.y;
            outputVertex.cull = position.x;
            clip(outputVertex.clip - 0.5f);
        }
    """)

    assert "float clip @ gl_ClipDistance;" in crossgl
    assert "float cull @ gl_CullDistance;" in crossgl
    assert "outputVertex.clip = position.y;" in crossgl
    assert "outputVertex.cull = position.x;" in crossgl
    assert "if ((outputVertex.clip - 0.5) < 0.0) {" in crossgl
    assert "discard;" in crossgl
    parse_crossgl(crossgl)


def test_codegen_mul_intrinsic_imports_to_multiply_expression():
    crossgl = generate_crossgl("""
        float4x4 viewProj;

        float4 VSMain(float3 position : POSITION) : SV_Position {
            return mul(viewProj, float4(position, 1.0));
        }
    """)

    assert "return (viewProj * vec4(position, 1.0));" in crossgl
    assert "*(viewProj" not in crossgl
    assert "mul(" not in crossgl
    parse_crossgl(crossgl)


def test_codegen_mad_intrinsic_from_microsoft_docs_imports_to_arithmetic_expression():
    # Source: Microsoft Learn HLSL mad intrinsic docs.
    # URL: https://learn.microsoft.com/windows/win32/direct3dhlsl/mad
    crossgl = generate_crossgl("""
        float4 main(float4 base : TEXCOORD0, float4 scale : TEXCOORD1, float4 bias : TEXCOORD2) : SV_Target0 {
            float4 adjusted = mad(base + 1.0, scale, bias);
            return adjusted;
        }
    """)

    assert "vec4 adjusted = (((base + 1.0) * scale) + bias);" in crossgl
    assert "mad(" not in crossgl
    ast = parse_crossgl(crossgl)
    glsl = GLSLCodeGen().generate(ast)
    assert "mad(" not in glsl
    assert "((base + 1.0) * scale) + bias" in glsl


def test_codegen_dst_intrinsic_from_microsoft_docs_imports_to_distance_vector_expression():
    # Sources: Microsoft Learn HLSL dst intrinsic and vertex-shader dst instruction.
    # URLs:
    # https://learn.microsoft.com/windows/win32/direct3dhlsl/dst
    # https://learn.microsoft.com/windows/win32/direct3dhlsl/dst---vs
    crossgl = generate_crossgl("""
        float4 main(float4 src0 : TEXCOORD0, float4 src1 : TEXCOORD1) : SV_Target0 {
            float4 distanceVector = dst(src0, src1);
            return distanceVector;
        }
    """)

    assert (
        "vec4 distanceVector = vec4(1.0, src0.y * src1.y, src0.z, src1.w);" in crossgl
    )
    assert "dst(" not in crossgl
    ast = parse_crossgl(crossgl)
    glsl = GLSLCodeGen().generate(ast)
    assert "dst(" not in glsl
    assert "vec4 distanceVector = vec4(1.0, (src0.y * src1.y), src0.z, src1.w);" in glsl


def test_codegen_bit_scan_intrinsics_from_microsoft_docs_reparse():
    # Source: Microsoft Learn HLSL intrinsic functions and firstbitlow docs.
    # URLs:
    # https://learn.microsoft.com/windows/win32/direct3dhlsl/dx-graphics-hlsl-intrinsic-functions
    # https://learn.microsoft.com/windows/win32/direct3dhlsl/firstbitlow
    crossgl = generate_crossgl("""
        uint4 main(uint value : TEXCOORD0) : SV_Target0 {
            uint low = firstbitlow(value);
            uint high = firstbithigh(value);
            uint count = countbits(value);
            uint reversed = reversebits(value);
            return uint4(low, high, count, reversed);
        }
    """)

    assert "uint low = findLSB(value);" in crossgl
    assert "uint high = findMSB(value);" in crossgl
    assert "uint count = bitCount(value);" in crossgl
    assert "uint reversed = reverseBits(value);" in crossgl
    assert "firstbitlow" not in crossgl
    assert "firstbithigh" not in crossgl
    assert "countbits" not in crossgl
    assert "reversebits" not in crossgl
    parse_crossgl(crossgl)


def test_codegen_derivative_intrinsics_from_microsoft_docs_reparse():
    # Source: Microsoft Learn ddx/ddy docs and DirectX-Specs SM 6.6 derivatives.
    # URLs:
    # https://learn.microsoft.com/windows/win32/direct3dhlsl/dx-graphics-hlsl-ddx
    # https://learn.microsoft.com/windows/win32/direct3dhlsl/dx-graphics-hlsl-ddy
    # https://microsoft.github.io/DirectX-Specs/d3d/HLSL_SM_6_6_Derivatives.html
    crossgl = generate_crossgl("""
        float4 main(float2 uv : TEXCOORD0) : SV_Target0 {
            float fineX = ddx_fine(uv.x);
            float coarseX = ddx_coarse(uv.x);
            float coarseY = ddy_coarse(uv.y);
            float fineY = ddy_fine(uv.y);
            return float4(ddx(uv.x) + coarseX, ddy(uv.y) + fineY, fwidth(fineX), coarseY);
        }
    """)

    assert "float fineX = dFdxFine(uv.x);" in crossgl
    assert "float coarseX = dFdxCoarse(uv.x);" in crossgl
    assert "float coarseY = dFdyCoarse(uv.y);" in crossgl
    assert "float fineY = dFdyFine(uv.y);" in crossgl
    assert (
        "return vec4(dFdx(uv.x) + coarseX, dFdy(uv.y) + fineY, "
        "fwidth(fineX), coarseY);"
    ) in crossgl
    assert "ddx" not in crossgl
    assert "ddy" not in crossgl
    parse_crossgl(crossgl)


def test_codegen_sincos_intrinsic_from_microsoft_docs_reparse():
    # Source: Microsoft Learn HLSL sincos intrinsic.
    # URL: https://learn.microsoft.com/windows/win32/direct3dhlsl/dx-graphics-hlsl-sincos
    crossgl = generate_crossgl("""
        float2 main(float angle : TEXCOORD0) : SV_Target0 {
            float s;
            float c;
            sincos(angle, s, c);
            return float2(s, c);
        }
    """)

    assert "s = sin(angle);" in crossgl
    assert "c = cos(angle);" in crossgl
    assert "sincos(" not in crossgl
    parse_crossgl(crossgl)


def test_codegen_bitcast_intrinsics_from_microsoft_docs_reparse():
    # Source: Microsoft Learn asfloat/asint/asuint docs.
    # URLs:
    # https://learn.microsoft.com/windows/win32/direct3dhlsl/dx-graphics-hlsl-asfloat
    # https://learn.microsoft.com/windows/win32/direct3dhlsl/dx-graphics-hlsl-asint
    # https://learn.microsoft.com/windows/win32/direct3dhlsl/dx-graphics-hlsl-asuint
    crossgl = generate_crossgl("""
        float4 main(uint4 packed : TEXCOORD0, float4 value : TEXCOORD1, int4 signedPacked : TEXCOORD2) : SV_Target0 {
            float4 unpacked = asfloat(packed);
            float4 signedUnpacked = asfloat(signedPacked);
            int4 signedBits = asint(value);
            uint4 unsignedBits = asuint(value);
            return unpacked + signedUnpacked + uintBitsToFloat(unsignedBits) + intBitsToFloat(signedBits);
        }
    """)

    assert "vec4 unpacked = uintBitsToFloat(packed);" in crossgl
    assert "vec4 signedUnpacked = intBitsToFloat(signedPacked);" in crossgl
    assert "ivec4 signedBits = floatBitsToInt(value);" in crossgl
    assert "uvec4 unsignedBits = floatBitsToUint(value);" in crossgl
    assert "asfloat" not in crossgl
    assert "asint" not in crossgl
    assert "asuint" not in crossgl
    parse_crossgl(crossgl)


def test_codegen_bitcast_intrinsics_infer_resource_load_value_types():
    # Sources: Microsoft Learn asfloat and ByteAddressBuffer::Load docs.
    # URLs:
    # https://learn.microsoft.com/windows/win32/direct3dhlsl/dx-graphics-hlsl-asfloat
    # https://learn.microsoft.com/windows/win32/direct3dhlsl/sm5-object-byteaddressbuffer-load
    crossgl = generate_crossgl("""
        StructuredBuffer<uint4> packedVectors : register(t0);
        ByteAddressBuffer rawData : register(t1);

        float4 main(uint index : TEXCOORD0, uint offset : TEXCOORD1) : SV_Target0 {
            float3 vectorValue = asfloat(packedVectors[index].xyz);
            float scalarValue = asfloat(rawData.Load(offset));
            return float4(vectorValue, scalarValue);
        }
    """)

    assert "vec3 vectorValue = uintBitsToFloat(packedVectors[index].xyz);" in crossgl
    assert (
        "float scalarValue = uintBitsToFloat(buffer_load(rawData, offset));" in crossgl
    )
    assert "asfloat" not in crossgl
    parse_crossgl(crossgl)


def test_codegen_sampler_state_initializer_block_imports_to_crossgl():
    crossgl = generate_crossgl(SAMPLER_STATE_BLOCK_HLSL)

    assert "sampler MeshTextureSampler" in crossgl
    assert "Filter = MIN_MAG_MIP_LINEAR" not in crossgl
    assert "texture(tex, MeshTextureSampler, uv)" in crossgl
    parse_crossgl(crossgl)


def test_codegen_export_function_specifier_imports_to_crossgl():
    crossgl = generate_crossgl("export void LogTraceRayStart() { }")

    assert "void LogTraceRayStart()" in crossgl
    assert "export" not in crossgl
    parse_crossgl(crossgl)


def test_codegen_namespace_block_scoped_call_imports_to_parseable_crossgl():
    crossgl = generate_crossgl("""
        using namespace dx;

        namespace CrossBilateral {
            float Weight(float value) {
                return value;
            }
        }

        float UseWeight(float value) {
            return CrossBilateral::Weight(value);
        }
    """)

    assert "float Weight(float value)" in crossgl
    assert "CrossBilateral::Weight(value)" in crossgl
    parse_crossgl(crossgl)


def test_codegen_array_template_argument_from_dxc_buffer_tests():
    crossgl = generate_crossgl("""
        StructuredBuffer<float[2]> ArrBuf : register(t3);

        float2 main(uint ix0 : IX0) : SV_Target {
            return (float2)ArrBuf.Load(ix0 + 1);
        }
    """)

    assert "StructuredBuffer<float[2]> ArrBuf;" in crossgl
    assert "StructuredBuffer<float, [, 2, ]>" not in crossgl
    parse_crossgl(crossgl)


def test_codegen_anonymous_nested_struct_member_roundtrip():
    hlsl = textwrap.dedent("""
        struct NRCPathState {
            float energy;
            struct {
                float3 origin;
                float pdf;
            } path;
        };
    """)

    crossgl = generate_crossgl(hlsl)

    assert "struct NRCPathState_path {" in crossgl
    assert "vec3 origin;" in crossgl
    assert "float pdf;" in crossgl
    assert "struct NRCPathState {" in crossgl
    assert "NRCPathState_path path;" in crossgl


def test_brace_initializer_declarations_generate_crossgl():
    output = generate_crossgl(textwrap.dedent("""
            struct MyPayload {
                int val;
            };

            struct RayDesc {
                float3 Origin;
                float TMin;
                float3 Direction;
                float TMax;
            };

            void RayGen() {
                float3 origin = float3(0.0, 0.0, 0.0);
                float3 rayDir = float3(0.0, 0.0, 1.0);
                RayDesc myRay = { origin, 0.0f, rayDir, 10000.0f };
                MyPayload payload = { 0 };
            }

            [shader("miss")]
            void Miss(inout MyPayload payload) {
                payload.val = 1;
            }

            [shader("closesthit")]
            void Hit(
                inout MyPayload payload,
                in BuiltInTriangleIntersectionAttributes attr
            ) {
                payload.val = 2;
            }
            """))

    assert "RayDesc myRay = {origin, 0.0, rayDir, 10000.0};" in output
    assert "MyPayload payload = {0};" in output
    assert "void main(inout MyPayload payload @ payload)" in output
    assert "in BuiltInTriangleIntersectionAttributes attr @ hit_attribute" in output
    parse_crossgl(output)


def test_global_static_const_array_initializer_generates_crossgl():
    output = generate_crossgl("static const float Weights[2] = { 0.25f, 0.75f };")

    assert "static const float Weights[2] = {0.25, 0.75};" in output

    shader_ast = parse_crossgl(output)
    weights = shader_ast.global_variables[0]
    assert weights.name == "Weights"
    assert weights.qualifiers == ["static", "const"]
    assert isinstance(weights.var_type, ArrayType)
    assert getattr(weights.var_type.size, "value", None) == 2
    assert isinstance(weights.initial_value, ArrayLiteralNode)
    assert [
        getattr(element, "value", None) for element in weights.initial_value.elements
    ] == [0.25, 0.75]


def test_codegen_local_static_const_array_from_dxc_bc6hdecode():
    # Source: microsoft/DirectXShaderCompiler@517dd5eb5d8cbb46c15fc1230acac1d2f4779092
    # tools/clang/test/CodeGenHLSL/Samples/DX11/BC6HDecode.hlsl
    output = generate_crossgl("""
        void generate_palette_unquantized8(int i) {
            static const int aWeight3[] = {0, 9, 18, 27, 37, 46, 55, 64};
            int weight = aWeight3[i];
        }
    """)

    assert "static const int aWeight3[] = {0, 9, 18, 27, 37, 46, 55, 64};" in output

    shader_ast = parse_crossgl(output)
    local_weights = shader_ast.functions[0].body.statements[0]
    assert local_weights.name == "aWeight3"
    assert local_weights.qualifiers == ["static", "const"]
    assert isinstance(local_weights.var_type, ArrayType)
    assert local_weights.var_type.size is None
    assert isinstance(local_weights.initial_value, ArrayLiteralNode)


def test_codegen_sizeof_type_operand_from_dxc_implicit_cast():
    # Source URL: https://github.com/microsoft/DirectXShaderCompiler
    # Commit: 517dd5eb5d8cbb46c15fc1230acac1d2f4779092
    # Path: tools/clang/test/HLSLFileCheck/hlsl/types/cast/implicit-cast-almost-identical.hlsl
    output = generate_crossgl("uint main() : OUT { return abs(sizeof(float)); }")

    assert "return abs(4);" in output
    assert "sizeof(" not in output
    parse_crossgl(output)


def test_codegen_sizeof_vector_array_operands_from_dxc_unary_sizeof():
    # Source URL: https://github.com/microsoft/DirectXShaderCompiler
    # Commit: 517dd5eb5d8cbb46c15fc1230acac1d2f4779092
    # Path: tools/clang/test/HLSLFileCheck/hlsl/operators/unary/sizeof.hlsl
    output = generate_crossgl("""
        AppendStructuredBuffer<int> buf;

        void main() {
            buf.Append(sizeof(int3[2]));
            buf.Append(sizeof(half3[2]));
            buf.Append(sizeof(int64_t3[2]));
        }
    """)

    assert "buffer_append(buf, 24);" in output
    assert "buffer_append(buf, 12);" in output
    assert "buffer_append(buf, 48);" in output
    assert "sizeof(" not in output
    parse_crossgl(output)


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


def test_codegen_min_precision_vector_and_matrix_types():
    code = textwrap.dedent("""
        struct MinPrecisionData {
            min16float3 hdr : COLOR0;
            min10float2 uv : TEXCOORD0;
            min16float2x3 colorMatrix;
        };

        min16float3 Shade(
            min16float3 color : COLOR0,
            min12int2 offset : TEXCOORD1,
            min16uint4 mask : TEXCOORD2
        ) : SV_Target0 {
            min10float2 localUv = min10float2(0.0, 1.0);
            min16float3 tint = min16float3(color.x, localUv.x, localUv.y);
            return tint;
        }
    """).strip()

    output = generate_crossgl(code)

    assert "f16vec3 hdr @ Color0;" in output
    assert "f16vec2 uv @ TexCoord0;" in output
    assert "f16mat2x3 colorMatrix;" in output
    assert "f16vec3 Shade(" in output
    assert "f16vec3 color @ Color0" in output
    assert "i16vec2 offset @ TexCoord1" in output
    assert "u16vec4 mask @ TexCoord2" in output
    assert "f16vec2 localUv = f16vec2(0.0, 1.0);" in output
    assert "f16vec3 tint = f16vec3(color.x, localUv.x, localUv.y);" in output
    assert "min16float" not in output
    assert "min10float" not in output

    shader_ast = parse_crossgl(output)
    assert ShaderStage.FRAGMENT in shader_ast.stages


def test_codegen_template_style_vector_matrix_types_and_constructors():
    hlsl = textwrap.dedent("""
        struct TemplateTypes {
            vector<float, 3> normal : NORMAL;
            matrix<float, 3, 3> basis;
        };

        vector<double, 4> MakeTemplateVector(
            vector<float, 3> input : TEXCOORD0
        ) : SV_Target0 {
            matrix<float, 2, 3> localBasis;
            vector<float> defaultWidth = vector<float>(1.0, 2.0, 3.0, 4.0);
            return vector<double, 4>(input.x, input.y, input.z, 1.0);
        }
    """).strip()

    output = generate_crossgl(hlsl)

    assert "vec3 normal @ Normal;" in output
    assert "mat3 basis;" in output
    assert "dvec4 MakeTemplateVector(vec3 input @ TexCoord0) @ gl_FragData[0]" in output
    assert "mat2x3 localBasis;" in output
    assert "vec4 defaultWidth = vec4(1.0, 2.0, 3.0, 4.0);" in output
    assert "return dvec4(input.x, input.y, input.z, 1.0);" in output
    assert "vector<" not in output
    assert "matrix<" not in output

    parse_crossgl(output)


def test_codegen_signedness_prefixed_int1_aliases_from_microsoft_docs_reparse():
    # Sources: Microsoft Learn HLSL vector and scalar type docs.
    # URLs:
    # https://learn.microsoft.com/en-us/windows/win32/direct3dhlsl/dx-graphics-hlsl-vector
    # https://learn.microsoft.com/en-us/windows/win32/direct3dhlsl/dx-graphics-hlsl-scalar
    crossgl = generate_crossgl("""
        uint4 main() : SV_Target0 {
            unsigned int1 unsignedOne = 1;
            signed int1 signedOne = -1;
            return uint4(unsignedOne, uint(signedOne), 0, 1);
        }
    """)

    assert "uint unsignedOne = 1;" in crossgl
    assert "int signedOne = -1;" in crossgl
    assert "unsigned int1" not in crossgl
    assert "signed int1" not in crossgl
    parse_crossgl(crossgl)


def test_codegen_struct_template_methods_semantic_case_from_dxc_spirv():
    # Source: microsoft/DirectXShaderCompiler@517dd5eb5d8cbb46c15fc1230acac1d2f4779092
    # tools/clang/test/CodeGenSPIRV/use.rvalue.for.member-expr.of.array-subscript.hlsl
    hlsl = textwrap.dedent("""
        [[vk::binding(0, 0)]] ByteAddressBuffer babuf[]: register(t0, space0);
        [[vk::binding(0, 0)]] RWByteAddressBuffer rwbuf[]: register(u0, space0);

        struct BufferAccess {
          uint handle;

          template<typename T>
          T load(uint index) {
            return babuf[this.handle].Load<T>(index * sizeof(T));
          }

          template<typename T>
          void store(uint index, T value) {
            return rwbuf[this.handle].Store<T>(index * sizeof(T), value);
          }
        };

        struct Data {
          BufferAccess buf;
          uint a0;
        };

        struct A {
          uint x;
        };

        [[vk::push_constant]] ConstantBuffer<Data> cbuf;

        [numthreads(1, 1, 1)]
        void main(uint tid : SV_DispatchThreadId) {
          A b = cbuf.buf.load<A>(0);
          b.x = 12;
          cbuf.buf.store<A>(0, b);
        }
    """).strip()

    output = generate_crossgl(hlsl)

    assert "@ vk::binding(0, 0)" in output
    assert "ByteAddressBuffer babuf[];" in output
    assert "RWByteAddressBuffer rwbuf[];" in output
    assert "ConstantBuffer<Data> cbuf;" in output
    assert "void main(uint tid @ gl_GlobalInvocationID)" in output
    assert "SV_DispatchThreadId" not in output
    assert "A b = cbuf.buf.load<A>(0);" in output
    assert "cbuf.buf.store<A>(0, b);" in output

    parse_crossgl(output)


def test_codegen_one_row_matrix_aliases_from_dxc_matrix_syntax():
    # Source: microsoft/DirectXShaderCompiler@517dd5eb5d8cbb46c15fc1230acac1d2f4779092
    # tools/clang/test/SemaHLSL/matrix-syntax.hlsl
    hlsl = textwrap.dedent("""
        void abs_on_demand() {
            float1x2 f12;
            float1x2 result = abs(f12);
            matrix<float, 1, 4> templatedRow;
        }
    """).strip()

    output = generate_crossgl(hlsl)

    assert "vec2 f12;" in output
    assert "vec2 result = abs(f12);" in output
    assert "vec4 templatedRow;" in output
    assert "float1x2" not in output
    assert "matrix<" not in output

    parse_crossgl(output)


def test_codegen_function_array_return_from_dxc_longvec_decls():
    # Source: microsoft/DirectXShaderCompiler@517dd5eb5d8cbb46c15fc1230acac1d2f4779092
    # tools/clang/test/CodeGenDXIL/hlsl/types/longvec-decls.hlsl
    hlsl = textwrap.dedent("""
        vector<float, 4> lv_param_arr_passthru(vector<float, 4> vec[10])[10] {
            return vec;
        }
    """).strip()

    output = generate_crossgl(hlsl)

    assert "vec4[10] lv_param_arr_passthru(vec4 vec[10])" in output
    assert "return vec;" in output

    shader_ast = parse_crossgl(output)
    func = shader_ast.functions[0]
    assert isinstance(func.return_type, ArrayType)


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


def test_codegen_geometry_primitive_before_direction_from_dxc_spirv():
    # Source: microsoft/DirectXShaderCompiler@517dd5eb5d8cbb46c15fc1230acac1d2f4779092
    # tools/clang/test/CodeGenSPIRV/primitive.point.gs.hlsl
    output = generate_crossgl("""
        struct S { float4 val : VAL; };

        [maxvertexcount(3)]
        void main(point in uint id[1] : VertexID, inout LineStream<S> outData) {
        }
    """)

    assert "geometry {" in output
    assert "@ maxvertexcount(3)" in output
    assert (
        "void main(in point uint id[1] @ VertexID, inout lineStream outData)" in output
    )
    parse_crossgl(output)


def test_codegen_tessellation_stages():
    output = generate_crossgl(TESSELLATION_HLSL)
    lowered = output.lower()
    assert "tessellation_control" in lowered
    assert "tessellation_evaluation" in lowered
    parse_crossgl(output)


def test_codegen_tessellation_stage_inference_from_dxc_main_attributes():
    hlsl = textwrap.dedent("""
        struct DSFoo {
            float Edges[4] : SV_TessFactor;
            float Inside[2] : SV_InsideTessFactor;
        };

        struct HSFoo {
            float3 pos : POSITION;
        };

        DSFoo PatchFoo(InputPatch<HSFoo, 16> ip, uint patchID : SV_PrimitiveID) {
            DSFoo output;
            output.Edges[0] = ip[patchID].pos.x;
            output.Inside[0] = ip[patchID].pos.y;
            return output;
        }

        [domain("quad")]
        [partitioning("fractional_odd")]
        [outputtopology("triangle_cw")]
        [outputcontrolpoints(16)]
        [patchconstantfunc("PatchFoo")]
        HSFoo main(InputPatch<HSFoo, 16> patch, uint i : SV_OutputControlPointID) {
            HSFoo output;
            output.pos = patch[i].pos;
            return output;
        }

        [domain("quad")]
        float4 DomainMain(
            OutputPatch<HSFoo, 16> patch,
            DSFoo input,
            float2 uv : SV_DomainLocation
        ) : SV_Position {
            return float4(patch[0].pos, 1.0);
        }
        """)

    output = generate_crossgl(hlsl)

    assert "tessellation_control {" in output
    assert "tessellation_evaluation {" in output
    assert '@domain("quad")' in output
    parse_crossgl(output)


def test_codegen_mesh_task_stages():
    output = generate_crossgl(MESH_TASK_HLSL)
    lowered = output.lower()
    assert "mesh" in lowered
    assert "task" in lowered


def test_codegen_pascal_case_mesh_attributes_infer_stage():
    code = textwrap.dedent("""
        struct VertexOut {
            float4 position : SV_Position;
        };

        [NumThreads(128, 1, 1)]
        [OutputTopology("triangle")]
        void main(
            out indices uint3 tris[1],
            out vertices VertexOut verts[1]
        ) {
            SetMeshOutputCounts(1, 1);
        }
    """).strip()

    crossgl = generate_crossgl(code)

    assert "mesh {" in crossgl
    assert "@ NumThreads(128, 1, 1)" in crossgl
    assert '@ OutputTopology("triangle")' in crossgl
    assert "@ indices out uvec3 tris[1]" in crossgl
    assert "@ vertices out VertexOut verts[1]" in crossgl
    assert "SetMeshOutputCounts(1, 1);" in crossgl

    shader_ast = parse_crossgl(crossgl)
    assert ShaderStage.MESH in shader_ast.stages


def test_codegen_mesh_payload_parameters_roundtrip():
    code = textwrap.dedent("""
        struct MeshPayload {
            uint meshlet;
        };

        struct VertexOut {
            float4 position : SV_Position;
        };

        struct PrimitiveOut {
            bool culled : SV_CullPrimitive;
            uint layer : SV_RenderTargetArrayIndex;
        };

        groupshared MeshPayload payload;

        [shader("amplification")]
        [numthreads(1, 1, 1)]
        void ASMain() {
            payload.meshlet = 7u;
            DispatchMesh(1, 1, 1, payload);
        }

        [shader("mesh")]
        [numthreads(32, 1, 1)]
        [outputtopology("triangle")]
        void MSMain(
            in payload MeshPayload payload,
            out vertices VertexOut verts[3],
            out indices uint3 tris[1],
            out primitives PrimitiveOut prims[1]
        ) {
            SetMeshOutputCounts(3, 1);
            verts[0].position = float4(
                float(payload.meshlet), 0.0, 0.0, 1.0
            );
            tris[0] = uint3(0, 1, 2);
            prims[0].culled = false;
            prims[0].layer = 0u;
        }
    """).strip()

    crossgl = generate_crossgl(code)

    assert "task {" in crossgl
    assert "mesh {" in crossgl
    assert "groupshared MeshPayload payload;" in crossgl
    assert "@ mesh_payload in MeshPayload payload" in crossgl
    assert "@ vertices out VertexOut verts[3]" in crossgl
    assert "@ indices out uvec3 tris[1]" in crossgl
    assert "@ primitives out PrimitiveOut prims[1]" in crossgl
    assert "DispatchMesh(1, 1, 1, payload);" in crossgl
    assert "SetMeshOutputCounts(3, 1);" in crossgl

    hlsl = TranslatorHLSLCodeGen().generate(parse_crossgl(crossgl))

    assert "groupshared MeshPayload payload;" in hlsl
    assert "DispatchMesh(1, 1, 1, payload);" in hlsl
    assert "in payload MeshPayload payload" in hlsl
    assert "out vertices VertexOut verts[3]" in hlsl
    assert "out indices uint3 tris[1]" in hlsl
    assert "out primitives PrimitiveOut prims[1]" in hlsl
    assert ": mesh_payload" not in hlsl


def test_codegen_mesh_dispatch_mesh_id_semantic_roundtrip():
    code = textwrap.dedent("""
        struct VertexOut {
            float4 position : SV_Position;
        };

        [shader("mesh")]
        [numthreads(32, 1, 1)]
        [outputtopology("triangle")]
        void MSMain(
            uint3 dispatchMeshId : SV_DispatchMeshID,
            uint3 groupId : SV_GroupID,
            uint3 groupThreadId : SV_GroupThreadID,
            uint groupIndex : SV_GroupIndex,
            out vertices VertexOut verts[3],
            out indices uint3 tris[1]
        ) {
            SetMeshOutputCounts(3, 1);
            verts[0].position = float4(
                float(dispatchMeshId.x + groupId.x + groupThreadId.x + groupIndex),
                0.0,
                0.0,
                1.0
            );
            tris[0] = uint3(0, 1, 2);
        }
    """).strip()

    crossgl = generate_crossgl(code)

    assert "uvec3 dispatchMeshId @ mesh_DispatchMeshID" in crossgl
    assert "uvec3 groupId @ gl_WorkGroupID" in crossgl
    assert "uvec3 groupThreadId @ gl_LocalInvocationID" in crossgl
    assert "uint groupIndex @ gl_LocalInvocationIndex" in crossgl

    hlsl = TranslatorHLSLCodeGen().generate(parse_crossgl(crossgl))

    assert "uint3 dispatchMeshId : SV_DispatchMeshID" in hlsl
    assert "uint3 groupId : SV_GroupID" in hlsl
    assert "uint3 groupThreadId : SV_GroupThreadID" in hlsl
    assert "uint groupIndex : SV_GroupIndex" in hlsl
    assert "mesh_DispatchMeshID" not in hlsl
    assert "gl_WorkGroupID" not in hlsl
    assert "gl_LocalInvocationID" not in hlsl
    assert "gl_LocalInvocationIndex" not in hlsl


def test_codegen_mesh_view_id_semantic_roundtrip():
    code = textwrap.dedent("""
        struct VertexOut {
            float4 position : SV_Position;
        };

        [shader("mesh")]
        [numthreads(32, 1, 1)]
        [outputtopology("triangle")]
        void MSMain(
            uint viewId : SV_ViewID,
            out vertices VertexOut verts[3],
            out indices uint3 tris[1]
        ) {
            SetMeshOutputCounts(3, 1);
            verts[0].position = float4(float(viewId), 0.0, 0.0, 1.0);
            tris[0] = uint3(0, 1, 2);
        }
    """).strip()

    crossgl = generate_crossgl(code)

    assert "uint viewId @ gl_ViewID" in crossgl

    hlsl = TranslatorHLSLCodeGen().generate(parse_crossgl(crossgl))

    assert "uint viewId : SV_ViewID" in hlsl
    assert "gl_ViewID" not in hlsl


def test_codegen_raytracing_stage():
    output = generate_crossgl(RAYTRACING_HLSL)
    assert "ray_generation" in output.lower()


def test_codegen_rayquery_stage_roundtrip_drops_native_shader_attribute():
    output = generate_crossgl(RAY_QUERY_HLSL)

    assert "ray_generation" in output.lower()
    assert "@ shader" not in output
    assert "rayQuery rq;" in output
    assert "rq.TraceRayInline(accel, RAY_FLAG_NONE, 255, ray);" in output
    assert "rq.Proceed()" in output
    assert "rq.CommitNonOpaqueTriangleHit();" in output
    assert "uint status = rq.CommittedStatus();" in output
    assert "float t = rq.CommittedRayT();" in output

    shader_ast = parse_crossgl(output)
    assert ShaderStage.RAY_GENERATION in shader_ast.stages


def test_codegen_texture_sample_mapping():
    output = generate_crossgl(TEXTURE_SAMPLE_HLSL)
    assert "texture(tex, samp, uv)" in output
    assert "texture_sample" not in output


def test_codegen_texture2dms_operator_index_from_microsoft_docs():
    # Source: Microsoft Texture2DMS::Operator docs.
    output = generate_crossgl("""
        Texture2DMS<float4> texMs : register(t0);

        float4 main(uint2 pos : TEXCOORD0) : SV_Target0 {
            return texMs[pos];
        }
    """)

    assert "return texelFetch(texMs, pos, 0);" in output
    assert "texMs[pos]" not in output
    parse_crossgl(output)


def test_codegen_texture2dms_sample_operator_from_microsoft_docs():
    # Source: Microsoft Texture2DMS::sample.Operator docs.
    output = generate_crossgl("""
        Texture2DMS<float4, 8> texMs : register(t0);

        float4 main(uint2 pos : TEXCOORD0) : SV_Target0 {
            return texMs.sample[2][pos];
        }
    """)

    assert "return texelFetch(texMs, pos, 2);" in output
    assert "texMs.sample[2][pos]" not in output
    parse_crossgl(output)


def test_codegen_sv_target_index_roundtrips_to_hlsl_semantic():
    output = generate_crossgl("""
        float4 PSMain(float2 uv : TEXCOORD0) : SV_Target1 {
            return float4(uv, 0.0, 1.0);
        }
    """)

    assert "@ gl_FragData[1]" in output

    regenerated_hlsl = TranslatorHLSLCodeGen().generate(parse_crossgl(output))

    assert "float4 PSMain(float2 uv : TexCoord0): SV_TARGET1" in regenerated_hlsl
    assert "gl_FragData" not in regenerated_hlsl


def test_codegen_register_bindings_emitted():
    output = generate_crossgl(REGISTER_BINDINGS_HLSL)
    assert "@ register" in output


def test_codegen_numthreads_attribute_emitted():
    output = generate_crossgl(NUMTHREADS_HLSL)
    assert "@ numthreads" in output


def test_codegen_enum_and_typedef():
    output = generate_crossgl(ENUM_TYPEDEF_HLSL)
    assert "enum BlendMode" in output
    assert "type Color = vec4;" in output
    parse_crossgl(output)


def test_codegen_using_alias_declarations_from_hlsl_spec():
    hlsl = textwrap.dedent("""
        using Float = float;
        using Color = vector<float, 4>;
        using IndexPair = int[2];

        Color main(Color input) : SV_Target0 {
            Float bias = 0.0;
            IndexPair indices;
            return input + Color(bias, bias, bias, 0.0);
        }
    """)

    output = generate_crossgl(hlsl)

    assert "type Float = float;" in output
    assert "type Color = vec4;" in output
    assert "type IndexPair = int[2];" in output
    assert "Color main(Color input) @ gl_FragData[0]" in output
    parse_crossgl(output)


def test_codegen_block_scope_typedef_from_dxc_linalg_vectors_reparse():
    hlsl = textwrap.dedent("""
        ByteAddressBuffer BAB : register(t0);

        [numthreads(4, 4, 4)]
        void main(uint ID : SV_GroupID) {
          typedef vector<half, 16> half16;
          half16 srcF16 = BAB.Load<half16>(128);
        }
    """)

    output = generate_crossgl(hlsl)

    assert "type half16" not in output
    assert "vec4 srcF16" not in output
    assert "half16 srcF16 = buffer_load(BAB, 128);" in output
    parse_crossgl(output)


def test_codegen_block_scope_class_declarations_from_dxc_spec_reparse():
    # Source: https://github.com/microsoft/DirectXShaderCompiler
    # Commit: 517dd5eb5d8cbb46c15fc1230acac1d2f4779092
    # Path: tools/clang/test/SemaHLSL/spec.hlsl
    hlsl = textwrap.dedent("""
        namespace ns_general {
          void subobjects() {
            class Class { uint field; };
            class SuperClass { Class C; uint field; };

            Class C = { 0 };
            SuperClass SC = { 0, 0 };
          }
        }
    """)

    output = generate_crossgl(hlsl)

    assert "class Class;" not in output
    assert "class SuperClass;" not in output
    assert "Class C = {0};" in output
    assert "SuperClass SC = {0, 0};" in output
    parse_crossgl(output)


def test_codegen_anonymous_struct_array_typedef_from_dxc_codegen_debug_tests():
    hlsl = textwrap.dedent("""
        typedef struct { int a[4]; float2 b[2]; } type[3];

        int main() : OUT {
            type var = (type)0;
            return var[0].a[0];
        }
    """)

    output = generate_crossgl(hlsl)

    assert "type type = AnonymousStruct_type[3];" in output
    assert "struct AnonymousStruct_type" in output
    parse_crossgl(output)


def test_codegen_directx_samples_alias_and_scoped_types_parse_crossgl():
    hlsl = textwrap.dedent("""
        typedef BuiltInTriangleIntersectionAttributes MyAttributes;

        float GetDistance(
            in float3 position,
            in SignedDistancePrimitive::Enum primitiveType
        ) {
            SignedDistancePrimitive::Enum localPrimitive =
                (SignedDistancePrimitive::Enum) primitiveType;
            return position.x;
        }

        void MyClosestHitShader(inout RayPayload payload, in MyAttributes attr) {
        }
    """)

    output = generate_crossgl(hlsl)

    assert "type MyAttributes = BuiltInTriangleIntersectionAttributes;" in output
    assert "SignedDistancePrimitive_Enum primitiveType" in output
    assert "SignedDistancePrimitive_Enum localPrimitive" in output
    assert "SignedDistancePrimitive_Enum(primitiveType)" in output
    parse_crossgl(output)


def test_codegen_attributes_emitted():
    output = generate_crossgl(ATTRIBUTE_HLSL)
    lowered = output.lower()
    assert "@domain" in lowered
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


def test_codegen_cxx11_namespaced_resource_attribute_passthrough():
    hlsl = textwrap.dedent("""
        [[vk::binding(3, 1)]]
        Texture2D<float4> texture2 : register(t0, space0);
        """).strip()

    output = generate_crossgl(hlsl)

    assert "@ vk::binding(3, 1)" in output
    assert "@ register(t0, space0)" in output
    assert "sampler2D texture2;" in output


def test_codegen_cxx11_namespaced_cbuffer_attribute_passthrough():
    hlsl = textwrap.dedent("""
        [[vk::push_constant]]
        cbuffer PushConstants : register(b0, space1) {
            float4 tint : packoffset(c0);
        };
        """).strip()

    output = generate_crossgl(hlsl)

    assert "@ vk::push_constant" in output
    assert "@ register(b0, space1)" in output
    assert "cbuffer PushConstants" in output
    assert "vec4 tint;" in output


def test_codegen_anonymous_old_style_cbuffer_uses_synthetic_name():
    hlsl = textwrap.dedent("""
        cbuffer : register(b1)
        {
            float4 a;
            int2 b;
        };
        """).strip()

    output = generate_crossgl(hlsl)

    assert "@ register(b1)" in output
    assert "cbuffer AnonymousCBuffer_b1" in output
    assert "vec4 a;" in output
    assert "ivec2 b;" in output


def test_codegen_cbuffer_member_layout_metadata_passthrough():
    hlsl = textwrap.dedent("""
        cbuffer Material : register(b0) {
            row_major float4x4 transform : packoffset(c0);
            float roughness : packoffset(c4.x);
        };
        """).strip()

    output = generate_crossgl(hlsl)

    assert "@ row_major" in output
    assert "@ packoffset(c0)" in output
    assert "@ packoffset(c4.x)" in output
    assert "mat4 transform;" in output
    assert "float roughness;" in output

    shader_ast = parse_crossgl(output)
    cbuffer = shader_ast.cbuffers[0]
    transform, roughness = cbuffer.members
    assert [attr.name for attr in transform.attributes] == [
        "row_major",
        "packoffset",
    ]
    assert transform.attributes[1].arguments[0].name == "c0"
    assert [attr.name for attr in roughness.attributes] == ["packoffset"]
    roughness_offset = roughness.attributes[0].arguments[0]
    assert roughness_offset.object.name == "c4"
    assert roughness_offset.member == "x"


def test_codegen_cbuffer_name_matching_struct_is_renamed_for_crossgl():
    # Source: microsoft/DirectX-Graphics-Samples
    # Libraries/D3D12RaytracingFallback/src/GpuBvh2CopyBindings.h
    hlsl = textwrap.dedent("""
        struct DispatchWidthConstant
        {
            uint DispatchWidth;
        };

        cbuffer DispatchWidthConstant : register(b0)
        {
            DispatchWidthConstant Constants;
        };

        float main() : SV_Target0
        {
            return Constants.DispatchWidth;
        }
        """).strip()

    output = generate_crossgl(hlsl)

    assert "struct DispatchWidthConstant" in output
    assert "@ register(b0)" in output
    assert "cbuffer DispatchWidthConstant_1" in output
    assert "DispatchWidthConstant Constants;" in output
    parse_crossgl(output)


def test_codegen_duplicate_cbuffer_names_are_renamed_for_crossgl():
    # Source: microsoft/DirectX-Graphics-Samples
    # Libraries/D3D12RaytracingFallback/src/FallbackLayerUnitTests/ReadRootConstants.hlsl
    hlsl = textwrap.dedent("""
        cbuffer Constants : register(b0)
        {
            float4 color0;
        }

        cbuffer Constants : register(b1)
        {
            float4 color1;
        }

        cbuffer Constants : register(b2)
        {
            float4 color2;
        }

        float4 main() : SV_Target0
        {
            return color0 + color1 + color2;
        }
        """).strip()

    output = generate_crossgl(hlsl)

    assert "cbuffer Constants {" in output
    assert "cbuffer Constants_1 {" in output
    assert "cbuffer Constants_2 {" in output
    assert "@ register(b0)" in output
    assert "@ register(b1)" in output
    assert "@ register(b2)" in output
    parse_crossgl(output)


def test_codegen_hlsl_tbuffer_roundtrips_with_texture_register():
    hlsl = textwrap.dedent("""
        tbuffer LookupData : register(t3, space2) {
            float4 values[4] : packoffset(c0);
            float scale : packoffset(c4.x);
        };
        """).strip()

    output = generate_crossgl(hlsl)

    assert "@ tbuffer" in output
    assert "@ register(t3, space2)" in output
    assert "cbuffer LookupData" in output
    assert "vec4 values[4];" in output
    assert "@ packoffset(c4.x)" in output

    regenerated_hlsl = TranslatorHLSLCodeGen().generate(parse_crossgl(output))

    assert "tbuffer LookupData : register(t3, space2)" in regenerated_hlsl
    assert "float4 values[4];" in regenerated_hlsl
    assert "float scale;" in regenerated_hlsl


def test_codegen_object_style_constant_and_texture_buffers_roundtrip():
    hlsl = textwrap.dedent("""
        struct FrameConstants {
            float4 tint;
        };

        ConstantBuffer<FrameConstants> frame : register(b2, space1);
        TextureBuffer<FrameConstants> lookup : register(t5, space3);

        float4 main() : SV_Target0 {
            return frame.tint + lookup.tint;
        }
        """).strip()

    output = generate_crossgl(hlsl)

    assert "ConstantBuffer<FrameConstants> frame;" in output
    assert "@ register(b2, space1)" in output
    assert "TextureBuffer<FrameConstants> lookup;" in output
    assert "@ register(t5, space3)" in output

    regenerated_hlsl = TranslatorHLSLCodeGen().generate(parse_crossgl(output))

    assert (
        "ConstantBuffer<FrameConstants> frame : register(b2, space1);"
        in regenerated_hlsl
    )
    assert (
        "TextureBuffer<FrameConstants> lookup : register(t5, space3);"
        in regenerated_hlsl
    )
    assert "return (frame.tint + lookup.tint);" in regenerated_hlsl


def test_codegen_resource_binding_aliases_from_microsoft_docs():
    # Source: https://learn.microsoft.com/en-us/windows/win32/direct3d12/resource-binding-in-hlsl
    hlsl = textwrap.dedent("""
        Texture2D terrain[] : register(t8);
        Sampler samp : register(s0);

        struct Data
        {
            UINT index;
            float4 color;
        };

        struct Stuff
        {
            float2 factor;
            UINT drawID;
        };

        ConstantBuffer<Data> myData : register(b0);
        ConstantBuffer<Stuff> myStuff[][3][8] : register(b2, space3);

        float4 main(uint idx : TEXCOORD0, float2 uv : TEXCOORD1) : SV_Target0
        {
            return terrain[idx].Sample(samp, uv) + myData.color;
        }
        """).strip()

    output = generate_crossgl(hlsl)

    assert "sampler samp;" in output
    assert "uint index;" in output
    assert "uint drawID;" in output
    assert "ConstantBuffer<Stuff> myStuff[][3][8];" in output
    assert "Sampler samp" not in output
    assert "UINT index" not in output
    assert "UINT drawID" not in output

    parse_crossgl(output)


def test_codegen_waveops_include_helper_lanes_attribute_passthrough():
    hlsl = textwrap.dedent("""
        [WaveOpsIncludeHelperLanes]
        float4 main(bool predicate : TEXCOORD0) : SV_Target0 {
            return QuadAny(predicate) ? 1.0.xxxx : 0.0.xxxx;
        }
        """).strip()

    output = generate_crossgl(hlsl)

    assert "@ WaveOpsIncludeHelperLanes" in output
    assert "QuadAny(predicate)" in output


def test_codegen_wave_size_attribute_passthrough():
    hlsl = textwrap.dedent("""
        [WaveSize(32)]
        [numthreads(8, 1, 1)]
        void main(uint3 tid : SV_DispatchThreadID) {
        }
        """).strip()

    output = generate_crossgl(hlsl)

    assert "@ WaveSize(32)" in output
    assert "@ numthreads(8, 1, 1)" in output


def test_codegen_interlocked_mapping():
    output = generate_crossgl(INTERLOCKED_HLSL)
    assert "atomicAdd" in output


def test_codegen_buffer_method_mapping():
    output = generate_crossgl(BUFFER_OPS_HLSL)
    assert "buffer_load" in output


def test_codegen_byte_address_vector_method_mapping():
    output = generate_crossgl(BYTE_ADDRESS_VECTOR_OPS_HLSL)
    assert "ByteAddressBuffer rawInput;" in output
    assert "RWByteAddressBuffer rawOutput;" in output
    assert "@ register(t3)" in output
    assert "@ register(u4)" in output
    assert "uint scalar = buffer_load(rawInput, offset + 48);" in output
    assert "uvec2 pair = buffer_load2(rawInput, offset);" in output
    assert "uvec3 triple = buffer_load3(rawOutput, offset + 16);" in output
    assert "uvec4 quad = buffer_load4(rawOutput, offset + 32);" in output
    assert "buffer_store(rawOutput, offset + 48, scalar);" in output
    assert "buffer_store2(rawOutput, offset, pair);" in output
    assert "buffer_store3(rawOutput, offset + 16, triple);" in output
    assert "buffer_store4(rawOutput, offset + 32, quad);" in output
    assert ".Load<uint>(" not in output
    assert ".Store<uint>(" not in output
    assert ".Load2(" not in output
    assert ".Store4(" not in output

    regenerated_hlsl = TranslatorHLSLCodeGen().generate(parse_crossgl(output))
    assert "ByteAddressBuffer rawInput : register(t3);" in regenerated_hlsl
    assert "RWByteAddressBuffer rawOutput : register(u4);" in regenerated_hlsl
    assert "uint scalar = rawInput.Load((offset + 48));" in regenerated_hlsl
    assert "uint2 pair = rawInput.Load2(offset);" in regenerated_hlsl
    assert "uint3 triple = rawOutput.Load3((offset + 16));" in regenerated_hlsl
    assert "uint4 quad = rawOutput.Load4((offset + 32));" in regenerated_hlsl
    assert "rawOutput.Store((offset + 48), scalar);" in regenerated_hlsl
    assert "rawOutput.Store2(offset, pair);" in regenerated_hlsl
    assert "rawOutput.Store3((offset + 16), triple);" in regenerated_hlsl
    assert "rawOutput.Store4((offset + 32), quad);" in regenerated_hlsl
    assert "buffer_load2(" not in regenerated_hlsl
    assert "buffer_store4(" not in regenerated_hlsl


def test_codegen_wave_ops_passthrough():
    output = generate_crossgl(WAVE_OPS_HLSL)
    for intrinsic in [
        "WaveGetLaneIndex()",
        "WaveGetLaneCount()",
        "WaveIsFirstLane()",
        "WaveActiveSum(value)",
        "WaveActiveProduct(value)",
        "WaveActiveMin(value)",
        "WaveActiveMax(value)",
        "WaveActiveAllTrue(predicate)",
        "WaveActiveAnyTrue(predicate)",
        "WaveActiveAllEqual(value)",
        "WaveActiveBallot(predicate)",
        "WaveActiveCountBits(predicate)",
        "WaveReadLaneAt(value, 0)",
        "WaveReadLaneFirst(value)",
        "WavePrefixSum(value)",
        "WavePrefixProduct(value)",
        "WavePrefixCountBits(predicate)",
        "WaveMatch(value)",
        "WaveMultiPrefixSum(value, ballot)",
        "WaveMultiPrefixCountBits(predicate, ballot)",
        "WaveMultiPrefixProduct(value, ballot)",
        "WaveMultiPrefixBitAnd(value, ballot)",
        "WaveMultiPrefixBitOr(value, ballot)",
        "WaveMultiPrefixBitXor(value, ballot)",
        "QuadReadAcrossX(value)",
        "QuadReadAcrossY(value)",
        "QuadReadAcrossDiagonal(value)",
        "QuadReadLaneAt(value, 2)",
        "QuadAny(predicate)",
        "QuadAll(predicate)",
    ]:
        assert intrinsic in output
    assert "uvec4 ballot = WaveActiveBallot(predicate);" in output
    assert "uvec4 matchMask = WaveMatch(value);" in output


def test_codegen_barrier_intrinsics_import_to_crossgl_builtins():
    code = textwrap.dedent("""
        [numthreads(1, 1, 1)]
        void CSMain(uint3 tid : SV_DispatchThreadID) {
            GroupMemoryBarrierWithGroupSync();
            GroupMemoryBarrier();
            DeviceMemoryBarrier();
            AllMemoryBarrier();
            DeviceMemoryBarrierWithGroupSync();
            AllMemoryBarrierWithGroupSync();
        }
    """).strip()

    crossgl = generate_crossgl(code)

    assert crossgl.count("workgroupBarrier();") == 3
    assert crossgl.count("groupMemoryBarrier();") == 1
    assert crossgl.count("deviceMemoryBarrier();") == 2
    assert crossgl.count("allMemoryBarrier();") == 2
    assert "GroupMemoryBarrierWithGroupSync();" not in crossgl
    assert "GroupMemoryBarrier();" not in crossgl
    assert "DeviceMemoryBarrier();" not in crossgl
    assert "AllMemoryBarrier();" not in crossgl
    assert "DeviceMemoryBarrierWithGroupSync();" not in crossgl
    assert "AllMemoryBarrierWithGroupSync();" not in crossgl

    regenerated_hlsl = TranslatorHLSLCodeGen().generate(parse_crossgl(crossgl))
    assert regenerated_hlsl.count("GroupMemoryBarrierWithGroupSync();") == 3
    assert regenerated_hlsl.count("GroupMemoryBarrier();") == 1
    assert regenerated_hlsl.count("DeviceMemoryBarrier();") == 2
    assert regenerated_hlsl.count("AllMemoryBarrier();") == 2
    assert "workgroupBarrier();" not in regenerated_hlsl
    assert "groupMemoryBarrier();" not in regenerated_hlsl
    assert "deviceMemoryBarrier();" not in regenerated_hlsl
    assert "allMemoryBarrier();" not in regenerated_hlsl
    assert "DeviceMemoryBarrierWithGroupSync();" not in regenerated_hlsl
    assert "AllMemoryBarrierWithGroupSync();" not in regenerated_hlsl


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


def test_codegen_preserves_globallycoherent_uav_parameter_from_hlsl_specs():
    # Source: Microsoft HLSL Specifications proposal 0021, vk cooperative matrix.
    # It declares CoherentStore(globallycoherent RWStructuredBuffer<Type> data, ...).
    code = textwrap.dedent("""
        struct Payload {
            uint value;
        };

        void CoherentStore(
            globallycoherent RWStructuredBuffer<Payload> data,
            uint index
        ) {
            data[index].value = 1u;
        }
    """).strip()

    output = generate_crossgl(code)

    assert (
        "void CoherentStore(@ globallycoherent "
        "RWStructuredBuffer<Payload> data, uint index)" in output
    )
    assert "data[index].value = 1;" in output


def test_codegen_preserves_reordercoherent_uav_from_dxc_dxil_69():
    # Source: microsoft/DirectXShaderCompiler@517dd5eb5d8cbb46c15fc1230acac1d2f4779092
    # tools/clang/test/CodeGenDXIL/hlsl/attributes/reordercoherent_uav.hlsl
    code = textwrap.dedent("""
        reordercoherent RWTexture1D<float4> uav1 : register(u3);

        [shader("raygeneration")]
        void main()
        {
          reordercoherent RWTexture1D<float4> uav3 = uav1;
          uav3[0] = float4(5.0, 0.0, 0.0, 1.0);
        }
    """).strip()

    output = generate_crossgl(code)

    assert "@ reordercoherent" in output
    assert "@ register(u3)" in output
    assert "image1D uav1;" in output
    assert "image1D uav3 = uav1;" in output
    assert "imageStore(uav3, 0, vec4(5.0, 0.0, 0.0, 1.0));" in output

    parse_crossgl(output)


def test_codegen_rasterizer_ordered_resources_roundtrip():
    code = textwrap.dedent("""
        RasterizerOrderedTexture2D<uint> pixelCounts : register(u0, space1);
        RasterizerOrderedTexture2DArray<float4> layers : register(u1);
        RasterizerOrderedBuffer<uint> bins : register(u2);
        RasterizerOrderedStructuredBuffer<int> values : register(u3);
        RasterizerOrderedByteAddressBuffer rawBytes : register(u4);

        [earlydepthstencil]
        float4 PSMain(uint2 pixel : TEXCOORD0, uint layer : TEXCOORD1) : SV_Target0 {
            uint oldCount;
            InterlockedAdd(pixelCounts[pixel], 1u, oldCount);
            layers[uint3(pixel, layer)] = float4(oldCount, 0, 0, 1);
            uint oldBin;
            InterlockedAdd(bins[0], oldCount, oldBin);
            int oldValue;
            InterlockedMax(values[0], int(oldBin), oldValue);
            rawBytes.Store(0, oldBin);
            return layers[uint3(pixel, layer)];
        }
    """).strip()

    output = generate_crossgl(code)

    assert output.count("@ rasterizer_ordered") == 5
    assert "@ register(u0, space1)" in output
    assert "uimage2D pixelCounts;" in output
    assert "image2DArray layers;" in output
    assert "RWBuffer<uint> bins;" in output
    assert "RWStructuredBuffer<int> values;" in output
    assert "RWByteAddressBuffer rawBytes;" in output
    assert "oldCount = imageAtomicAdd(pixelCounts, pixel, 1u);" in output
    assert "imageStore(layers, uvec3(pixel, layer), vec4(oldCount, 0, 0, 1));" in output
    assert "oldBin = atomicAdd(bins[0], oldCount);" in output
    assert "oldValue = atomicMax(values[0], int(oldBin));" in output
    assert "buffer_store(rawBytes, 0, oldBin);" in output

    regenerated_hlsl = TranslatorHLSLCodeGen().generate(parse_crossgl(output))
    assert (
        "RasterizerOrderedTexture2D<uint> pixelCounts : register(u0, space1);"
        in regenerated_hlsl
    )
    assert "RasterizerOrderedTexture2DArray<float4> layers : register(u1);" in (
        regenerated_hlsl
    )
    assert "RasterizerOrderedBuffer<uint> bins : register(u2);" in regenerated_hlsl
    assert (
        "RasterizerOrderedStructuredBuffer<int> values : register(u3);"
        in regenerated_hlsl
    )
    assert (
        "RasterizerOrderedByteAddressBuffer rawBytes : register(u4);"
        in regenerated_hlsl
    )
    assert (
        "uint imageAtomicAdd_uimage2D("
        "RasterizerOrderedTexture2D<uint> image, int2 coord, uint value)"
        in regenerated_hlsl
    )
    assert "RWTexture2D<uint> pixelCounts" not in regenerated_hlsl
    assert "RWTexture2D<uint> image" not in regenerated_hlsl
    assert "RWBuffer<uint> bins : register(u2);" not in regenerated_hlsl


def test_codegen_ray_shader_stages():
    output = generate_crossgl(RAY_STAGES_HLSL)
    lowered = output.lower()
    assert "ray_intersection" in lowered
    assert "ray_closest_hit" in lowered
    assert "ray_any_hit" in lowered
    assert "ray_miss" in lowered
    assert "ray_callable" in lowered
    assert "@ shader" not in output


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
    assert converter.texture_method_descriptor("SampleBias", 3)["function"] == "texture"
    assert (
        converter.texture_method_descriptor("SampleBias", 4)["function"]
        == "textureOffset"
    )
    assert converter.texture_method_descriptor("Sample", 5) == {
        "member": "Sample",
        "function": "textureOffset",
        "texture_function": "textureOffset",
        "buffer_function": None,
        "component": None,
        "usage": "regular",
        "buffer_when_max_args": None,
        "drop_trailing_args": 2,
        "dropped_parameters": ["LOD clamp", "status output"],
    }
    assert converter.texture_method_descriptor("SampleBias", 6)[
        "dropped_parameters"
    ] == ["LOD clamp", "status output"]
    assert (
        converter.texture_method_descriptor("SampleLevel", 5)["function"]
        == "textureLodOffset"
    )
    assert converter.texture_method_descriptor("SampleLevel", 5)[
        "dropped_parameters"
    ] == ["status output"]
    assert (
        converter.texture_method_descriptor("SampleGrad", 7)["function"]
        == "textureGradOffset"
    )
    assert converter.texture_method_descriptor("SampleGrad", 7)[
        "dropped_parameters"
    ] == ["LOD clamp", "status output"]
    assert (
        converter.texture_method_descriptor("SampleCmp", 4)["function"]
        == "textureCompareOffset"
    )
    assert converter.texture_method_descriptor("SampleCmp", 6)[
        "dropped_parameters"
    ] == ["LOD clamp", "status output"]
    assert (
        converter.texture_method_descriptor("SampleCmpLevel", 4)["function"]
        == "textureCompareLod"
    )
    assert (
        converter.texture_method_descriptor("SampleCmpLevel", 5)["function"]
        == "textureCompareLodOffset"
    )
    assert converter.texture_method_descriptor("SampleCmpLevel", 6)[
        "dropped_parameters"
    ] == ["status output"]
    assert (
        converter.texture_method_descriptor("SampleCmpGrad", 5)["function"]
        == "textureCompareGrad"
    )
    assert (
        converter.texture_method_descriptor("SampleCmpGrad", 6)["function"]
        == "textureCompareGradOffset"
    )
    assert converter.texture_method_descriptor("SampleCmpGrad", 8)[
        "dropped_parameters"
    ] == ["LOD clamp", "status output"]
    sample_cmp_bias = converter.texture_method_descriptor(
        "SampleCmpBias", 4, "Texture2D<float>"
    )
    assert sample_cmp_bias["function"] == "textureCompare"
    assert sample_cmp_bias["usage"] == "comparison"
    assert sample_cmp_bias["dropped_parameters"] == ["LOD bias"]
    sample_cmp_bias_offset = converter.texture_method_descriptor(
        "SampleCmpBias", 5, "Texture2DArray<float>"
    )
    assert sample_cmp_bias_offset["function"] == "textureCompareOffset"
    assert sample_cmp_bias_offset["dropped_parameters"] == ["LOD bias"]
    sample_cmp_bias_status = converter.texture_method_descriptor(
        "SampleCmpBias", 7, "Texture2D<float>"
    )
    assert sample_cmp_bias_status["function"] == "textureCompareOffset"
    assert sample_cmp_bias_status["dropped_parameters"] == [
        "LOD bias",
        "LOD clamp",
        "status output",
    ]
    sample_cmp_bias_cube = converter.texture_method_descriptor(
        "SampleCmpBias", 6, "TextureCube<float>"
    )
    assert sample_cmp_bias_cube["function"] == "textureCompare"
    assert sample_cmp_bias_cube["dropped_parameters"] == [
        "LOD bias",
        "LOD clamp",
        "status output",
    ]
    assert (
        converter.texture_method_descriptor("SampleCmpLevelZero", 4)["function"]
        == "textureCompareOffset"
    )
    assert converter.texture_method_descriptor("SampleCmpLevelZero", 5)[
        "dropped_parameters"
    ] == ["status output"]
    assert (
        converter.texture_method_descriptor("Gather", 3)["function"]
        == "textureGatherOffset"
    )
    assert converter.texture_method_descriptor("Gather", 4)["dropped_parameters"] == [
        "status output"
    ]
    assert (
        converter.texture_method_descriptor("GatherCmp", 4)["function"]
        == "textureGatherCompareOffset"
    )
    assert converter.texture_method_descriptor("GatherCmp", 5)[
        "dropped_parameters"
    ] == ["status output"]
    assert (
        converter.texture_method_descriptor("GatherCmpRed", 3)["function"]
        == "textureGatherCompare"
    )
    assert (
        converter.texture_method_descriptor("GatherCmpRed", 4)["function"]
        == "textureGatherCompareOffset"
    )
    assert converter.texture_method_descriptor("GatherCmpRed", 5)[
        "dropped_parameters"
    ] == ["status output"]
    gather_cmp_red_offsets = converter.texture_method_descriptor("GatherCmpRed", 7)
    assert gather_cmp_red_offsets["fallback_expression"] == "vec4(0.0)"
    assert "four-offset overload" in gather_cmp_red_offsets["diagnostic_reason"]
    gather_cmp_green = converter.texture_method_descriptor("GatherCmpGreen", 4)
    assert gather_cmp_green["fallback_expression"] == "vec4(0.0)"
    assert "component selector" in gather_cmp_green["diagnostic_reason"]
    assert converter.texture_method_descriptor("GatherBlue", 2) == {
        "member": "GatherBlue",
        "function": "textureGather",
        "texture_function": "textureGather",
        "buffer_function": None,
        "component": "2",
        "usage": "regular",
        "buffer_when_max_args": None,
    }
    assert (
        converter.texture_method_descriptor("GatherBlue", 3)["function"]
        == "textureGatherOffset"
    )
    assert (
        converter.texture_method_descriptor("GatherBlue", 6)["function"]
        == "textureGatherOffsets"
    )
    assert converter.texture_method_descriptor("GatherBlue", 7)[
        "dropped_parameters"
    ] == ["status output"]
    assert converter.texture_method_descriptor("Sample", 3, "TextureCube") == {
        "member": "Sample",
        "function": "texture",
        "texture_function": "texture",
        "buffer_function": None,
        "component": None,
        "usage": "regular",
        "buffer_when_max_args": None,
        "drop_trailing_args": 1,
        "dropped_parameters": ["LOD clamp"],
    }
    assert (
        converter.texture_method_descriptor("SampleLevel", 4, "TextureCube")["function"]
        == "textureLod"
    )
    assert (
        converter.texture_method_descriptor("SampleGrad", 6, "TextureCubeArray")[
            "function"
        ]
        == "textureGrad"
    )
    assert (
        converter.texture_method_descriptor("SampleCmp", 5, "TextureCube")["function"]
        == "textureCompare"
    )
    assert (
        converter.texture_method_descriptor("SampleCmpLevel", 5, "TextureCube")[
            "function"
        ]
        == "textureCompareLod"
    )
    assert converter.texture_method_descriptor("SampleCmpGrad", 7, "TextureCubeArray")[
        "dropped_parameters"
    ] == ["LOD clamp", "status output"]
    assert (
        converter.texture_method_descriptor("Gather", 3, "TextureCubeArray")["function"]
        == "textureGather"
    )
    assert (
        converter.texture_method_descriptor("GatherCmp", 4, "TextureCube")["function"]
        == "textureGatherCompare"
    )
    assert converter.texture_method_descriptor("Load", 1)["function"] == "buffer_load"
    assert converter.texture_method_descriptor("Load", 2)["function"] == "texelFetch"
    assert (
        converter.resource_method_descriptor("Load", 1, "Texture2D")["function"]
        == "texelFetch"
    )
    assert (
        converter.resource_method_descriptor("Load", 2, "Texture2D")["function"]
        == "texelFetchOffset"
    )
    assert converter.resource_method_descriptor("Load", 3, "Texture2D")[
        "dropped_parameters"
    ] == ["status output"]
    assert converter.resource_method_descriptor("Load", 2, "RWTexture2D<float4>") == {
        "member": "Load",
        "function": "imageLoad",
        "texture_function": "texelFetch",
        "buffer_function": "buffer_load",
        "component": None,
        "usage": "regular",
        "buffer_when_max_args": 1,
        "resource_type": "RWTexture2D<float4>",
        "diagnostic_kind": "tiled_resource_status",
        "drop_trailing_args": 1,
        "dropped_parameters": ["status output"],
        "resource": "image",
        "operation": "load",
    }
    assert (
        converter.resource_method_descriptor("Load", 4, "Texture2DMS<float4>")[
            "function"
        ]
        == "texelFetchOffset"
    )
    assert converter.resource_method_descriptor("Load", 4, "Texture2DMS<float4>")[
        "dropped_parameters"
    ] == ["status output"]
    assert converter.resource_method_descriptor("Load", 3, "RWTexture2DMS<float4>")[
        "dropped_parameters"
    ] == ["status output"]
    assert converter.resource_method_descriptor("Load", 3, "RWTexture2DMSArray<uint>")[
        "dropped_parameters"
    ] == ["status output"]
    assert (
        converter.texture_method_descriptor("GetDimensions", 1)["function"]
        == "buffer_dimensions"
    )
    assert (
        converter.texture_method_descriptor("GetDimensions", 2)["function"]
        == "texture_dimensions"
    )
    assert (
        converter.resource_method_descriptor(
            "GetDimensions", 2, "StructuredBuffer<float4>"
        )["function"]
        == "buffer_dimensions"
    )
    assert (
        converter.resource_method_descriptor("GetDimensions", 2, "Texture2D<float4>")[
            "function"
        ]
        == "texture_dimensions"
    )
    assert converter.texture_method_descriptor("Store", 2) is None
    assert converter.resource_method_descriptor("Load", 1)["resource"] == "buffer"
    assert converter.resource_method_descriptor("Load", 2)["resource"] == "texture"
    assert (
        converter.resource_method_descriptor("GetDimensions", 1)["operation"]
        == "dimensions"
    )
    assert converter.texture_method_descriptor("CalculateLevelOfDetail", 2) == {
        "member": "CalculateLevelOfDetail",
        "function": "textureQueryLod",
        "texture_function": "textureQueryLod",
        "buffer_function": None,
        "component": None,
        "usage": "regular",
        "buffer_when_max_args": None,
        "result_component": ".x",
    }
    assert (
        converter.resource_method_descriptor(
            "CalculateLevelOfDetailUnclamped", 2, "Texture2D<float4>"
        )["result_component"]
        == ".y"
    )
    assert (
        converter.resource_method_descriptor(
            "CalculateLevelOfDetail", 2, "Texture2D<float4>"
        )["operation"]
        == "query_lod"
    )
    invalid_lod = converter.resource_method_descriptor(
        "CalculateLevelOfDetail", 2, "Texture2DMS<float4>"
    )
    assert invalid_lod["fallback_expression"] == "0.0"
    assert "unavailable for multisample textures" in invalid_lod["diagnostic_reason"]
    assert converter.texture_method_descriptor("GetSamplePosition", 1) == {
        "member": "GetSamplePosition",
        "function": "textureSamplePosition",
        "texture_function": "textureSamplePosition",
        "buffer_function": None,
        "component": None,
        "usage": "regular",
        "buffer_when_max_args": None,
        "diagnostic_label": "texture sample-position query",
    }
    assert (
        converter.resource_method_descriptor(
            "GetSamplePosition", 1, "Texture2DMSArray<float4>"
        )["operation"]
        == "query_sample_position"
    )
    invalid_sample_position = converter.resource_method_descriptor(
        "GetSamplePosition", 1, "Texture2D<float4>"
    )
    assert invalid_sample_position["fallback_expression"] == "vec2(0.0, 0.0)"
    assert (
        "sampled multisample textures" in invalid_sample_position["diagnostic_reason"]
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


def test_codegen_sample_bias_offset_roundtrip_through_translator_codegen():
    code = textwrap.dedent("""
        Texture2D tex : register(t0);
        SamplerState samp : register(s0);

        float4 main(float2 uv : TEXCOORD0, float bias : TEXCOORD1) : SV_Target {
            float4 biased = tex.SampleBias(samp, uv, bias);
            float4 offsetBiased = tex.SampleBias(samp, uv, bias, int2(1, 0));
            return biased + offsetBiased;
        }
    """).strip()

    crossgl = generate_crossgl(code)

    assert "texture(tex, samp, uv, bias)" in crossgl
    assert "textureOffset(tex, samp, uv, ivec2(1, 0), bias)" in crossgl
    assert "texture(tex, samp, uv, bias, ivec2(1, 0))" not in crossgl

    shader_ast = parse_crossgl(crossgl)
    hlsl = TranslatorHLSLCodeGen().generate(shader_ast)
    assert "tex.SampleBias(samp, uv, bias)" in hlsl
    assert "tex.SampleBias(samp, uv, bias, int2(1, 0))" in hlsl


def test_codegen_texture_sample_offsets_roundtrip_through_translator_codegen():
    code = textwrap.dedent("""
        Texture2D tex : register(t0);
        SamplerState samp : register(s0);

        float4 main(
            float2 uv : TEXCOORD0,
            float lod : TEXCOORD1,
            float2 ddx : TEXCOORD2,
            float2 ddy : TEXCOORD3,
            int2 offset : TEXCOORD4
        ) : SV_Target {
            float4 plain = tex.Sample(samp, uv, offset);
            float4 mip = tex.SampleLevel(samp, uv, lod, offset);
            float4 grad = tex.SampleGrad(samp, uv, ddx, ddy, offset);
            return plain + mip + grad;
        }
    """).strip()

    crossgl = generate_crossgl(code)

    assert "textureOffset(tex, samp, uv, offset)" in crossgl
    assert "textureLodOffset(tex, samp, uv, lod, offset)" in crossgl
    assert "textureGradOffset(tex, samp, uv, ddx, ddy, offset)" in crossgl
    assert "texture(tex, samp, uv, offset)" not in crossgl
    assert "textureLod(tex, samp, uv, lod, offset)" not in crossgl
    assert "textureGrad(tex, samp, uv, ddx, ddy, offset)" not in crossgl
    assert ".Sample(" not in crossgl
    assert ".SampleLevel(" not in crossgl
    assert ".SampleGrad(" not in crossgl

    hlsl = TranslatorHLSLCodeGen().generate(parse_crossgl(crossgl))
    assert "tex.Sample(samp, uv, offset)" in hlsl
    assert "tex.SampleLevel(samp, uv, lod, offset)" in hlsl
    assert "tex.SampleGrad(samp, uv, ddx, ddy, offset)" in hlsl
    assert "textureOffset(" not in hlsl
    assert "textureLodOffset(" not in hlsl
    assert "textureGradOffset(" not in hlsl


def test_codegen_texture_compare_and_gather_offsets_roundtrip_through_translator_codegen():
    code = textwrap.dedent("""
        Texture2D colorMap : register(t0);
        Texture2D<float> shadowMap : register(t1);
        SamplerState linearSampler : register(s0);
        SamplerComparisonState compareSampler : register(s1);

        float4 main(
            float2 uv : TEXCOORD0,
            float depth : TEXCOORD1,
            int2 offset : TEXCOORD2
        ) : SV_Target {
            float cmp = shadowMap.SampleCmp(compareSampler, uv, depth, offset);
            float cmpZero = shadowMap.SampleCmpLevelZero(
                compareSampler, uv, depth, offset
            );
            float4 gather = colorMap.GatherRed(linearSampler, uv, offset);
            float4 gatherAny = colorMap.Gather(linearSampler, uv, offset);
            float4 gatherCmp = shadowMap.GatherCmp(
                compareSampler, uv, depth, offset
            );
            return gather + gatherAny + gatherCmp + float4(cmp + cmpZero);
        }
    """).strip()

    crossgl = generate_crossgl(code)

    assert (
        "textureCompareOffset(shadowMap, compareSampler, uv, depth, offset)" in crossgl
    )
    assert "textureGatherOffset(colorMap, linearSampler, uv, offset, 0)" in crossgl
    assert "textureGatherOffset(colorMap, linearSampler, uv, offset)" in crossgl
    assert (
        "textureGatherCompareOffset(shadowMap, compareSampler, uv, depth, offset)"
        in crossgl
    )
    assert "textureCompare(shadowMap, compareSampler, uv, depth, offset)" not in crossgl
    assert "textureGather(colorMap, linearSampler, uv, offset" not in crossgl
    assert ".GatherCmp(" not in crossgl

    hlsl = TranslatorHLSLCodeGen().generate(parse_crossgl(crossgl))
    assert "shadowMap.SampleCmp(compareSampler, uv, depth, offset)" in hlsl
    assert "colorMap.GatherRed(linearSampler, uv, offset)" in hlsl
    assert "colorMap.Gather(linearSampler, uv, offset)" in hlsl
    assert "shadowMap.GatherCmp(compareSampler, uv, depth, offset)" in hlsl
    assert "textureCompareOffset(" not in hlsl
    assert "textureGatherOffset(" not in hlsl
    assert "textureGatherCompareOffset(" not in hlsl


def test_codegen_texture_gather_compare_red_imports_to_helpers():
    code = textwrap.dedent("""
        Texture2D<float> shadowMap : register(t0);
        Texture2DArray<float> shadowArray : register(t1);
        SamplerComparisonState compareSampler : register(s0);

        float4 main(
            float2 uv : TEXCOORD0,
            float3 uvLayer : TEXCOORD1,
            float depth : TEXCOORD2,
            int2 offset0 : TEXCOORD3,
            int2 offset1 : TEXCOORD4,
            int2 offset2 : TEXCOORD5,
            int2 offset3 : TEXCOORD6,
            uint status : TEXCOORD7
        ) : SV_Target {
            float4 red = shadowMap.GatherCmpRed(compareSampler, uv, depth);
            float4 redOffset = shadowArray.GatherCmpRed(
                compareSampler, uvLayer, depth, offset0
            );
            float4 redStatus = shadowArray.GatherCmpRed(
                compareSampler, uvLayer, depth, offset0, status
            );
            float4 redOffsets = shadowArray.GatherCmpRed(
                compareSampler, uvLayer, depth, offset0, offset1, offset2, offset3
            );
            float4 redOffsetsStatus = shadowArray.GatherCmpRed(
                compareSampler, uvLayer, depth,
                offset0, offset1, offset2, offset3, status
            );
            float4 green = shadowArray.GatherCmpGreen(
                compareSampler, uvLayer, depth, offset0
            );
            return red + redOffset + redStatus + redOffsets + redOffsetsStatus
                + green;
        }
    """).strip()

    crossgl = generate_crossgl(code)

    assert "sampler2DShadow shadowMap;" in crossgl
    assert "sampler2DArrayShadow shadowArray;" in crossgl
    assert "textureGatherCompare(shadowMap, compareSampler, uv, depth)" in crossgl
    assert (
        "textureGatherCompareOffset(shadowArray, compareSampler, uvLayer, depth, "
        "offset0)"
    ) in crossgl
    assert (
        crossgl.count(
            "unsupported DirectX texture overload extras for GatherCmpRed: "
            "dropped status output"
        )
        == 2
    )
    assert (
        "textureGatherCompareOffset(shadowArray, compareSampler, uvLayer, depth, "
        "offset0).x"
    ) in crossgl
    assert (
        "textureGatherCompareOffset(shadowArray, compareSampler, uvLayer, depth, "
        "offset1).y"
    ) in crossgl
    assert (
        "textureGatherCompareOffset(shadowArray, compareSampler, uvLayer, depth, "
        "offset2).z"
    ) in crossgl
    assert (
        "textureGatherCompareOffset(shadowArray, compareSampler, uvLayer, depth, "
        "offset3).w"
    ) in crossgl
    assert (
        "unsupported DirectX texture gather-compare component for Texture2DArray: "
        "GatherCmpGreen requires a compare-gather component selector"
    ) in crossgl
    assert "GatherCmpRed four-offset overload" not in crossgl
    assert "vec4(0.0)" in crossgl
    assert ".GatherCmpRed(" not in crossgl
    assert ".GatherCmpGreen(" not in crossgl

    hlsl = TranslatorHLSLCodeGen().generate(parse_crossgl(crossgl))
    assert "shadowMap.GatherCmp(compareSampler, uv, depth)" in hlsl
    assert "shadowArray.GatherCmp(compareSampler, uvLayer, depth, offset0)" in hlsl
    assert "shadowArray.GatherCmp(compareSampler, uvLayer, depth, offset0).x" in hlsl
    assert "shadowArray.GatherCmp(compareSampler, uvLayer, depth, offset1).y" in hlsl
    assert "shadowArray.GatherCmp(compareSampler, uvLayer, depth, offset2).z" in hlsl
    assert "shadowArray.GatherCmp(compareSampler, uvLayer, depth, offset3).w" in hlsl
    assert "GatherCmpRed" not in hlsl
    assert "GatherCmpGreen" not in hlsl


def test_codegen_texture_compare_lod_grad_member_imports_to_helpers():
    code = textwrap.dedent("""
        Texture2D<float> shadowMap : register(t0);
        TextureCube<float> cubeShadow : register(t1);
        SamplerComparisonState compareSampler : register(s0);

        float4 main(
            float2 uv : TEXCOORD0,
            float depth : TEXCOORD1,
            float lod : TEXCOORD2,
            float2 ddx : TEXCOORD3,
            float2 ddy : TEXCOORD4,
            int2 offset : TEXCOORD5,
            uint status : TEXCOORD6,
            float3 direction : TEXCOORD7,
            float3 ddx3 : TEXCOORD8,
            float3 ddy3 : TEXCOORD9
        ) : SV_Target {
            float lodValue = shadowMap.SampleCmpLevel(
                compareSampler, uv, depth, lod
            );
            float lodOffset = shadowMap.SampleCmpLevel(
                compareSampler, uv, depth, lod, offset
            );
            float lodStatus = shadowMap.SampleCmpLevel(
                compareSampler, uv, depth, lod, offset, status
            );
            float gradValue = shadowMap.SampleCmpGrad(
                compareSampler, uv, depth, ddx, ddy
            );
            float gradOffset = shadowMap.SampleCmpGrad(
                compareSampler, uv, depth, ddx, ddy, offset
            );
            float gradStatus = shadowMap.SampleCmpGrad(
                compareSampler, uv, depth, ddx, ddy, offset, 0.0, status
            );
            float cubeLod = cubeShadow.SampleCmpLevel(
                compareSampler, direction, depth, lod, status
            );
            float cubeGrad = cubeShadow.SampleCmpGrad(
                compareSampler, direction, depth, ddx3, ddy3, 0.0, status
            );
            return float4(
                lodValue + lodOffset + lodStatus + gradValue + gradOffset
                + gradStatus + cubeLod + cubeGrad
            );
        }
    """).strip()

    crossgl = generate_crossgl(code)

    assert "sampler2DShadow shadowMap;" in crossgl
    assert "samplerCubeShadow cubeShadow;" in crossgl
    assert "textureCompareLod(shadowMap, compareSampler, uv, depth, lod)" in crossgl
    assert (
        "textureCompareLodOffset(shadowMap, compareSampler, uv, depth, lod, offset)"
        in crossgl
    )
    assert "textureCompareGrad(shadowMap, compareSampler, uv, depth, ddx, ddy)" in (
        crossgl
    )
    assert (
        "textureCompareGradOffset("
        "shadowMap, compareSampler, uv, depth, ddx, ddy, offset)" in crossgl
    )
    assert (
        "textureCompareLod(cubeShadow, compareSampler, direction, depth, lod)"
        in crossgl
    )
    assert (
        "textureCompareGrad("
        "cubeShadow, compareSampler, direction, depth, ddx3, ddy3)" in crossgl
    )
    assert (
        "unsupported DirectX texture overload extras for SampleCmpLevel: "
        "dropped status output"
    ) in crossgl
    assert (
        "unsupported DirectX texture overload extras for SampleCmpGrad: "
        "dropped LOD clamp, status output"
    ) in crossgl
    assert ".SampleCmpLevel(" not in crossgl
    assert ".SampleCmpGrad(" not in crossgl

    hlsl = TranslatorHLSLCodeGen().generate(parse_crossgl(crossgl))
    assert "shadowMap.SampleCmpLevel(compareSampler, uv, depth, lod)" in hlsl
    assert "shadowMap.SampleCmpLevel(compareSampler, uv, depth, lod, offset)" in hlsl
    assert "shadowMap.SampleCmpGrad(compareSampler, uv, depth, ddx, ddy)" in hlsl
    assert (
        "shadowMap.SampleCmpGrad(compareSampler, uv, depth, ddx, ddy, offset)" in hlsl
    )
    assert "cubeShadow.SampleCmpLevel(compareSampler, direction, depth, lod)" in hlsl
    assert "cubeShadow.SampleCmpGrad(compareSampler, direction, depth, ddx3, ddy3)" in (
        hlsl
    )
    assert "lod, offset, status" not in hlsl
    assert "ddy, offset, 0.0, status" not in hlsl
    assert "lod, status" not in hlsl
    assert "ddy3, 0.0, status" not in hlsl


def test_codegen_texture_compare_bias_member_imports_as_compare_with_diagnostics():
    code = textwrap.dedent("""
        Texture2D<float> shadowMap : register(t0);
        Texture2DArray<float> shadowArray : register(t1);
        TextureCube<float> cubeShadow : register(t2);
        SamplerComparisonState compareSampler : register(s0);

        float4 main(
            float2 uv : TEXCOORD0,
            float3 uvLayer : TEXCOORD1,
            float3 direction : TEXCOORD2,
            float depth : TEXCOORD3,
            float bias : TEXCOORD4,
            int2 offset : TEXCOORD5
        ) : SV_Target {
            float sampled = shadowMap.SampleCmpBias(
                compareSampler, uv, depth, bias
            );
            uint status;
            float offsetSampled = shadowArray.SampleCmpBias(
                compareSampler, uvLayer, depth, bias, offset, 0.0, status
            );
            float cubeSampled = cubeShadow.SampleCmpBias(
                compareSampler, direction, depth, bias, 0.0, status
            );
            return float4(sampled + offsetSampled + cubeSampled);
        }
    """).strip()

    crossgl = generate_crossgl(code)

    assert "sampler2DShadow shadowMap;" in crossgl
    assert "sampler2DArrayShadow shadowArray;" in crossgl
    assert "samplerCubeShadow cubeShadow;" in crossgl
    assert (
        "unsupported DirectX texture overload extras for SampleCmpBias: "
        "dropped LOD bias"
    ) in crossgl
    assert (
        "unsupported DirectX texture overload extras for SampleCmpBias: "
        "dropped LOD bias, LOD clamp, status output"
    ) in crossgl
    assert "textureCompare(shadowMap, compareSampler, uv, depth)" in crossgl
    assert (
        "textureCompareOffset(shadowArray, compareSampler, uvLayer, depth, offset)"
        in crossgl
    )
    assert "textureCompare(cubeShadow, compareSampler, direction, depth)" in crossgl
    assert ".SampleCmpBias(" not in crossgl
    assert "textureCompare(shadowMap, compareSampler, uv, depth, bias)" not in crossgl

    parse_crossgl(crossgl)

    hlsl = TranslatorHLSLCodeGen().generate(parse_crossgl(crossgl))

    assert "shadowMap.SampleCmp(compareSampler, uv, depth)" in hlsl
    assert "shadowArray.SampleCmp(compareSampler, uvLayer, depth, offset)" in hlsl
    assert "cubeShadow.SampleCmp(compareSampler, direction, depth)" in hlsl
    assert ".SampleCmpBias(" not in hlsl
    assert "bias, offset" not in hlsl
    assert "0.0, status" not in hlsl


def test_codegen_texture_status_and_clamp_overloads_import_as_valid_crossgl():
    code = textwrap.dedent("""
        Texture2D colorMap : register(t0);
        Texture2D<float> shadowMap : register(t1);
        SamplerState linearSampler : register(s0);
        SamplerComparisonState compareSampler : register(s1);

        float4 main(
            float2 uv : TEXCOORD0,
            float depth : TEXCOORD1,
            float lod : TEXCOORD2,
            float bias : TEXCOORD3,
            float2 ddx : TEXCOORD4,
            float2 ddy : TEXCOORD5,
            int2 offset : TEXCOORD6,
            uint status : TEXCOORD7
        ) : SV_Target {
            float4 plain = colorMap.Sample(
                linearSampler, uv, offset, 0.0, status
            );
            float4 biased = colorMap.SampleBias(
                linearSampler, uv, bias, offset, 0.0, status
            );
            float4 mip = colorMap.SampleLevel(
                linearSampler, uv, lod, offset, status
            );
            float4 grad = colorMap.SampleGrad(
                linearSampler, uv, ddx, ddy, offset, 0.0, status
            );
            float cmp = shadowMap.SampleCmp(
                compareSampler, uv, depth, offset, 0.0, status
            );
            float cmpZero = shadowMap.SampleCmpLevelZero(
                compareSampler, uv, depth, offset, status
            );
            float4 gather = colorMap.Gather(linearSampler, uv, offset, status);
            float4 gatherRed = colorMap.GatherRed(
                linearSampler, uv, offset, status
            );
            float4 gatherOffsets = colorMap.GatherRed(
                linearSampler, uv, offset, offset, offset, offset, status
            );
            float4 gatherCmp = shadowMap.GatherCmp(
                compareSampler, uv, depth, offset, status
            );
            return (
                plain + biased + mip + grad + gather + gatherRed + gatherOffsets
                + gatherCmp + float4(cmp + cmpZero)
            );
        }
    """).strip()

    crossgl = generate_crossgl(code)

    assert (
        "unsupported DirectX texture overload extras for Sample: "
        "dropped LOD clamp, status output"
    ) in crossgl
    assert (
        "unsupported DirectX texture overload extras for SampleLevel: "
        "dropped status output"
    ) in crossgl
    assert (
        "unsupported DirectX texture overload extras for GatherRed: "
        "dropped status output"
    ) in crossgl
    assert "textureOffset(colorMap, linearSampler, uv, offset)" in crossgl
    assert "textureOffset(colorMap, linearSampler, uv, offset, bias)" in crossgl
    assert "textureLodOffset(colorMap, linearSampler, uv, lod, offset)" in crossgl
    assert "textureGradOffset(colorMap, linearSampler, uv, ddx, ddy, offset)" in crossgl
    assert (
        "textureCompareOffset(shadowMap, compareSampler, uv, depth, offset)" in crossgl
    )
    assert "textureGatherOffset(colorMap, linearSampler, uv, offset)" in crossgl
    assert "textureGatherOffset(colorMap, linearSampler, uv, offset, 0)" in crossgl
    assert (
        "textureGatherOffsets(colorMap, linearSampler, uv, offset, offset, "
        "offset, offset, 0)"
    ) in crossgl
    assert (
        "textureGatherCompareOffset(shadowMap, compareSampler, uv, depth, offset)"
        in crossgl
    )
    assert ".Sample(" not in crossgl
    assert ".Gather" not in crossgl

    hlsl = TranslatorHLSLCodeGen().generate(parse_crossgl(crossgl))
    assert "colorMap.Sample(linearSampler, uv, offset)" in hlsl
    assert "colorMap.SampleBias(linearSampler, uv, bias, offset)" in hlsl
    assert "colorMap.SampleLevel(linearSampler, uv, lod, offset)" in hlsl
    assert "colorMap.SampleGrad(linearSampler, uv, ddx, ddy, offset)" in hlsl
    assert "shadowMap.SampleCmp(compareSampler, uv, depth, offset)" in hlsl
    assert "colorMap.Gather(linearSampler, uv, offset)" in hlsl
    assert "colorMap.GatherRed(linearSampler, uv, offset)" in hlsl
    assert (
        "colorMap.GatherRed(linearSampler, uv, offset, offset, offset, offset)" in hlsl
    )
    assert "shadowMap.GatherCmp(compareSampler, uv, depth, offset)" in hlsl
    assert "0.0, status" not in hlsl
    assert "offset, status" not in hlsl


def test_codegen_cube_texture_status_overloads_do_not_become_offset_calls():
    code = textwrap.dedent("""
        TextureCube cubeMap : register(t0);
        TextureCube<float> cubeShadow : register(t1);
        SamplerState linearSampler : register(s0);
        SamplerComparisonState compareSampler : register(s1);

        float4 main(
            float3 direction : TEXCOORD0,
            float depth : TEXCOORD1,
            float lod : TEXCOORD2,
            float bias : TEXCOORD3,
            float3 ddx : TEXCOORD4,
            float3 ddy : TEXCOORD5,
            uint status : TEXCOORD6
        ) : SV_Target {
            float4 plain = cubeMap.Sample(linearSampler, direction, 0.0, status);
            float4 biased = cubeMap.SampleBias(
                linearSampler, direction, bias, 0.0, status
            );
            float4 mip = cubeMap.SampleLevel(linearSampler, direction, lod, status);
            float4 grad = cubeMap.SampleGrad(
                linearSampler, direction, ddx, ddy, 0.0, status
            );
            float cmp = cubeShadow.SampleCmp(
                compareSampler, direction, depth, 0.0, status
            );
            float cmpZero = cubeShadow.SampleCmpLevelZero(
                compareSampler, direction, depth, status
            );
            float4 gather = cubeMap.Gather(linearSampler, direction, status);
            float4 gatherRed = cubeMap.GatherRed(linearSampler, direction, status);
            float4 gatherCmp = cubeShadow.GatherCmp(
                compareSampler, direction, depth, status
            );
            return (
                plain + biased + mip + grad + gather + gatherRed + gatherCmp
                + float4(cmp + cmpZero)
            );
        }
    """).strip()

    crossgl = generate_crossgl(code)

    assert (
        "unsupported DirectX texture overload extras for Sample: "
        "dropped LOD clamp, status output"
    ) in crossgl
    assert (
        "unsupported DirectX texture overload extras for Gather: "
        "dropped status output"
    ) in crossgl
    assert "texture(cubeMap, linearSampler, direction)" in crossgl
    assert "texture(cubeMap, linearSampler, direction, bias)" in crossgl
    assert "textureLod(cubeMap, linearSampler, direction, lod)" in crossgl
    assert "textureGrad(cubeMap, linearSampler, direction, ddx, ddy)" in crossgl
    assert "textureCompare(cubeShadow, compareSampler, direction, depth)" in crossgl
    assert "textureGather(cubeMap, linearSampler, direction)" in crossgl
    assert "textureGather(cubeMap, linearSampler, direction, 0)" in crossgl
    assert (
        "textureGatherCompare(cubeShadow, compareSampler, direction, depth)" in crossgl
    )
    assert "textureOffset(" not in crossgl
    assert "textureLodOffset(" not in crossgl
    assert "textureGradOffset(" not in crossgl
    assert "textureCompareOffset(" not in crossgl
    assert "textureGatherOffset(" not in crossgl
    assert "textureGatherCompareOffset(" not in crossgl

    hlsl = TranslatorHLSLCodeGen().generate(parse_crossgl(crossgl))
    assert "cubeMap.Sample(linearSampler, direction)" in hlsl
    assert "cubeMap.SampleBias(linearSampler, direction, bias)" in hlsl
    assert "cubeMap.SampleLevel(linearSampler, direction, lod)" in hlsl
    assert "cubeMap.SampleGrad(linearSampler, direction, ddx, ddy)" in hlsl
    assert "cubeShadow.SampleCmp(compareSampler, direction, depth)" in hlsl
    assert "cubeMap.Gather(linearSampler, direction)" in hlsl
    assert "cubeMap.GatherRed(linearSampler, direction)" in hlsl
    assert "cubeShadow.GatherCmp(compareSampler, direction, depth)" in hlsl
    assert "0.0, status" not in hlsl
    assert "direction, status" not in hlsl


def test_codegen_tiled_resource_status_loads_import_as_diagnostics():
    code = textwrap.dedent("""
        Texture2D colorMap : register(t0);
        Texture2DMS<float4> msMap : register(t1);
        RWTexture2D<float4> outputImage : register(u0);

        float4 main(
            int2 pixel : TEXCOORD0,
            int sampleIndex : TEXCOORD1,
            int2 offset : TEXCOORD2
        ) : SV_Target0 {
            uint status = 0;
            float4 fetched = colorMap.Load(int3(pixel, 0), offset, status);
            float4 stored = outputImage.Load(pixel, status);
            float4 ms = msMap.Load(pixel, sampleIndex, offset, status);
            bool mapped = CheckAccessFullyMapped(status);
            return mapped ? fetched + stored + ms : float4(0.0, 0.0, 0.0, 0.0);
        }
    """).strip()

    crossgl = generate_crossgl(code)

    assert (
        crossgl.count(
            "unsupported DirectX tiled-resource status for Load: "
            "dropped status output"
        )
        == 3
    )
    assert (
        "unsupported DirectX tiled-resource status check: "
        "CheckAccessFullyMapped assumed fully mapped"
    ) in crossgl
    assert "texelFetchOffset(colorMap, pixel, 0, offset)" in crossgl
    assert "imageLoad(outputImage, pixel)" in crossgl
    assert "texelFetchOffset(msMap, pixel, sampleIndex, offset)" in crossgl
    assert "bool mapped = " in crossgl
    assert "true;" in crossgl
    assert "CheckAccessFullyMapped(" not in crossgl
    assert ".Load(" not in crossgl
    assert ", status" not in crossgl

    parse_crossgl(crossgl)


def test_codegen_tiled_resource_status_only_loads_roundtrip_without_offsets():
    code = textwrap.dedent("""
        Texture1D ramp : register(t0);
        Texture2D colorMap : register(t1);
        Texture2DArray layers : register(t2);
        Texture3D volume : register(t3);
        Texture2DMS<float4> msMap : register(t4);
        Texture2DMSArray<float4> msLayers : register(t5);

        float4 main(
            int x : TEXCOORD0,
            int2 pixel : TEXCOORD1,
            int3 pixelLayer : TEXCOORD2,
            int3 voxel : TEXCOORD3,
            int lod : TEXCOORD4,
            int sampleIndex : TEXCOORD5
        ) : SV_Target0 {
            uint status = 0;
            float4 line = ramp.Load(int2(x, lod), status);
            float4 color = colorMap.Load(int3(pixel, lod), status);
            float4 layer = layers.Load(int4(pixelLayer, lod), status);
            float4 volumeColor = volume.Load(int4(voxel, lod), status);
            float4 ms = msMap.Load(pixel, sampleIndex, status);
            float4 msLayer = msLayers.Load(pixelLayer, sampleIndex, status);
            bool mapped = CheckAccessFullyMapped(status);
            return mapped ? line + color + layer + volumeColor + ms + msLayer
                : float4(0.0, 0.0, 0.0, 0.0);
        }
    """).strip()

    crossgl = generate_crossgl(code)

    assert (
        crossgl.count(
            "unsupported DirectX tiled-resource status for Load: "
            "dropped status output"
        )
        == 6
    )
    assert "texelFetch(ramp, x, lod)" in crossgl
    assert "texelFetch(colorMap, pixel, lod)" in crossgl
    assert "texelFetch(layers, pixelLayer, lod)" in crossgl
    assert "texelFetch(volume, voxel, lod)" in crossgl
    assert "texelFetch(msMap, pixel, sampleIndex)" in crossgl
    assert "texelFetch(msLayers, pixelLayer, sampleIndex)" in crossgl
    assert "texelFetchOffset(" not in crossgl
    assert ".Load(" not in crossgl
    assert ", status" not in crossgl

    hlsl = TranslatorHLSLCodeGen().generate(parse_crossgl(crossgl))

    assert "float4 line = ramp.Load(int2(x, lod));" in hlsl
    assert "float4 color = colorMap.Load(int3(pixel, lod));" in hlsl
    assert "float4 layer = layers.Load(int4(pixelLayer, lod));" in hlsl
    assert "float4 volumeColor = volume.Load(int4(voxel, lod));" in hlsl
    assert "float4 ms = msMap.Load(pixel, sampleIndex);" in hlsl
    assert "float4 msLayer = msLayers.Load(pixelLayer, sampleIndex);" in hlsl
    assert "bool mapped = true;" in hlsl
    assert "CheckAccessFullyMapped(" not in hlsl
    assert "texelFetch(" not in hlsl
    assert "texelFetchOffset(" not in hlsl
    assert ", status" not in hlsl


def test_codegen_multisample_storage_image_status_loads_roundtrip_to_srv_reads():
    code = textwrap.dedent("""
        RWTexture2DMS<float4> msImage : register(u0);
        RWTexture2DMSArray<uint> counters : register(u1);

        float4 main(
            int2 pixel : TEXCOORD0,
            int3 pixelLayer : TEXCOORD1,
            int sampleIndex : TEXCOORD2
        ) : SV_Target0 {
            uint status = 0;
            float4 color = msImage.Load(pixel, sampleIndex, status);
            uint count = counters.Load(pixelLayer, sampleIndex, status);
            bool mapped = CheckAccessFullyMapped(status);
            return mapped ? color + float4(count, count, count, count) : float4(0.0);
        }
    """).strip()

    crossgl = generate_crossgl(code)

    assert (
        crossgl.count(
            "unsupported DirectX tiled-resource status for Load: "
            "dropped status output"
        )
        == 2
    )
    assert (
        "unsupported DirectX tiled-resource status check: "
        "CheckAccessFullyMapped assumed fully mapped"
    ) in crossgl
    assert "imageLoad(msImage, pixel, sampleIndex)" in crossgl
    assert "imageLoad(counters, pixelLayer, sampleIndex)" in crossgl
    assert ".Load(" not in crossgl
    assert ", status)" not in crossgl

    hlsl = TranslatorHLSLCodeGen().generate(parse_crossgl(crossgl))

    assert "Texture2DMS<float4> msImage : register(t0);" in hlsl
    assert "Texture2DMSArray<uint> counters : register(t1);" in hlsl
    assert "float4 color = msImage.Load(pixel, sampleIndex);" in hlsl
    assert "uint count = counters.Load(pixelLayer, sampleIndex);" in hlsl
    assert "bool mapped = true;" in hlsl
    assert "RWTexture2DMS" not in hlsl
    assert "RWTexture2DMSArray" not in hlsl
    assert "CheckAccessFullyMapped(" not in hlsl
    assert ", status)" not in hlsl


@pytest.mark.parametrize(
    ("resource_decl", "main_signature", "load_expr", "type_name"),
    [
        (
            "RWTexture2DMS<float4> msImage : register(u0);",
            "float4 main(int2 pixel : TEXCOORD0, float sampleIndex : TEXCOORD1) : SV_Target0",
            "msImage.Load(pixel, sampleIndex)",
            "float",
        ),
        (
            "RWTexture2DMSArray<uint> counters : register(u0);",
            "float4 main(int3 pixelLayer : TEXCOORD0, int2 sampleIndex : TEXCOORD1) : SV_Target0",
            "float4(counters.Load(pixelLayer, sampleIndex), 0.0, 0.0, 1.0)",
            "int2",
        ),
        (
            "RWTexture2DMS<float4> msImage : register(u0);",
            "float4 main(int2 pixel : TEXCOORD0) : SV_Target0",
            "msImage.Load(pixel, msImage)",
            "RWTexture2DMS<float4>",
        ),
    ],
)
def test_codegen_multisample_storage_image_load_rejects_bad_sample_index_roundtrip(
    resource_decl, main_signature, load_expr, type_name
):
    code = textwrap.dedent(f"""
        {resource_decl}

        {main_signature} {{
            return {load_expr};
        }}
    """).strip()

    crossgl = generate_crossgl(code)
    assert "imageLoad(" in crossgl
    assert ".Load(" not in crossgl

    with pytest.raises(
        ValueError,
        match=(
            "DirectX multisample image operation 'imageLoad' requires a scalar "
            f"integer sample index argument: .* has type {type_name}"
        ),
    ):
        TranslatorHLSLCodeGen().generate(parse_crossgl(crossgl))


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


def test_codegen_nonuniform_resource_index_descriptor_arrays_roundtrip():
    code = textwrap.dedent("""
        Texture2D<float4> textures[4] : register(t0, space1);
        SamplerState samplers[4] : register(s0, space1);

        float4 main(
            float2 uv : TEXCOORD0,
            uint materialIndex : TEXCOORD1,
            uint samplerIndex : TEXCOORD2
        ) : SV_Target0 {
            uint textureIndex = NonUniformResourceIndex(materialIndex);
            uint samplerSlot = NonUniformResourceIndex(samplerIndex);
            return textures[textureIndex].Sample(samplers[samplerSlot], uv);
        }
    """).strip()

    crossgl = generate_crossgl(code)

    assert "sampler2D textures[4];" in crossgl
    assert "sampler samplers[4];" in crossgl
    assert "uint textureIndex = NonUniformResourceIndex(materialIndex);" in crossgl
    assert "uint samplerSlot = NonUniformResourceIndex(samplerIndex);" in crossgl
    assert "texture(textures[textureIndex], samplers[samplerSlot], uv)" in crossgl
    assert ".Sample(" not in crossgl

    hlsl = TranslatorHLSLCodeGen().generate(parse_crossgl(crossgl))

    assert "Texture2D textures[4] : register(t0, space1);" in hlsl
    assert "SamplerState samplers[4] : register(s0, space1);" in hlsl
    assert "uint textureIndex = NonUniformResourceIndex(materialIndex);" in hlsl
    assert "uint samplerSlot = NonUniformResourceIndex(samplerIndex);" in hlsl
    assert "textures[textureIndex].Sample(samplers[samplerSlot], uv)" in hlsl
    assert "float textureIndex" not in hlsl
    assert "float samplerSlot" not in hlsl


def test_codegen_nonuniform_resource_index_multidimensional_arrays_roundtrip():
    code = textwrap.dedent("""
        Texture2D<float4> textures[][4] : register(t0, space2);
        SamplerState samplers[] : register(s0, space2);

        float4 main(
            float2 uv : TEXCOORD0,
            uint materialIndex : TEXCOORD1,
            uint tileIndex : TEXCOORD2
        ) : SV_Target0 {
            uint nonUniformMaterial = NonUniformResourceIndex(materialIndex);
            uint nonUniformTile = NonUniformResourceIndex(tileIndex);
            return textures[nonUniformMaterial][nonUniformTile].Sample(
                samplers[nonUniformMaterial],
                uv
            );
        }
    """).strip()

    crossgl = generate_crossgl(code)

    assert "sampler2D textures[][4];" in crossgl
    assert "sampler samplers[];" in crossgl
    assert (
        "uint nonUniformMaterial = NonUniformResourceIndex(materialIndex);" in crossgl
    )
    assert "uint nonUniformTile = NonUniformResourceIndex(tileIndex);" in crossgl
    assert (
        "texture(textures[nonUniformMaterial][nonUniformTile], "
        "samplers[nonUniformMaterial], uv)" in crossgl
    )

    hlsl = TranslatorHLSLCodeGen().generate(parse_crossgl(crossgl))

    assert "Texture2D textures[][4] : register(t0, space2);" in hlsl
    assert "SamplerState samplers[] : register(s0, space2);" in hlsl
    assert "uint nonUniformMaterial = NonUniformResourceIndex(materialIndex);" in hlsl
    assert "uint nonUniformTile = NonUniformResourceIndex(tileIndex);" in hlsl
    assert (
        "textures[nonUniformMaterial][nonUniformTile].Sample("
        "samplers[nonUniformMaterial], uv)" in hlsl
    )
    assert "float nonUniformMaterial" not in hlsl
    assert "float nonUniformTile" not in hlsl


def test_codegen_feedback_texture_arrays_preserve_feedback_kind_and_uav_registers():
    code = textwrap.dedent("""
        Texture2D<float4> pairedTextures[2] : register(t0, space9);
        SamplerState linearSampler : register(s0, space9);
        FeedbackTexture2D<SAMPLER_FEEDBACK_MIN_MIP> feedbackMin[2] : register(u0, space9);
        FeedbackTexture2DArray<SAMPLER_FEEDBACK_MIP_REGION_USED> feedbackUsed[] : register(u2, space9);

        void main(float2 uv : TEXCOORD0, uint materialIndex : TEXCOORD1) {
            uint slot = NonUniformResourceIndex(materialIndex);
        }
    """).strip()

    crossgl = generate_crossgl(code)

    assert "@ register(t0, space9)" in crossgl
    assert "sampler2D pairedTextures[2];" in crossgl
    assert "@ register(u0, space9)" in crossgl
    assert "feedbackTexture2D<SAMPLER_FEEDBACK_MIN_MIP> feedbackMin[2];" in crossgl
    assert "@ register(u2, space9)" in crossgl
    assert (
        "feedbackTexture2DArray<SAMPLER_FEEDBACK_MIP_REGION_USED> feedbackUsed[];"
        in crossgl
    )
    assert "uint slot = NonUniformResourceIndex(materialIndex);" in crossgl

    hlsl = TranslatorHLSLCodeGen().generate(parse_crossgl(crossgl))

    assert "Texture2D pairedTextures[2] : register(t0, space9);" in hlsl
    assert "SamplerState linearSampler : register(s0, space9);" in hlsl
    assert (
        "FeedbackTexture2D<SAMPLER_FEEDBACK_MIN_MIP> feedbackMin[2] : "
        "register(u0, space9);" in hlsl
    )
    assert (
        "FeedbackTexture2DArray<SAMPLER_FEEDBACK_MIP_REGION_USED> feedbackUsed[] : "
        "register(u2, space9);" in hlsl
    )
    assert "uint slot = NonUniformResourceIndex(materialIndex);" in hlsl
    assert "feedbackTexture2D" not in hlsl
    assert "feedbackTexture2DArray" not in hlsl


def test_codegen_feedback_texture_write_methods_roundtrip_to_canonical_helpers():
    code = textwrap.dedent("""
        Texture2D<float4> pairedTexture : register(t0, space10);
        Texture2DArray<float4> pairedLayers : register(t1, space10);
        SamplerState pairedSampler : register(s0, space10);
        FeedbackTexture2D<SAMPLER_FEEDBACK_MIN_MIP> feedbackMin[2] : register(u0, space10);
        FeedbackTexture2DArray<SAMPLER_FEEDBACK_MIP_REGION_USED> feedbackUsed[] : register(u2, space10);

        float4 main(float2 uv : TEXCOORD0, float layer : TEXCOORD1, uint feedbackIndex : TEXCOORD2) : SV_Target0 {
            uint slot = NonUniformResourceIndex(feedbackIndex);
            float2 ddxValue = float2(0.25, 0.0);
            float2 ddyValue = float2(0.0, 0.25);
            float3 uvLayer = float3(uv, layer);
            feedbackMin[slot].WriteSamplerFeedback(pairedTexture, pairedSampler, uv);
            feedbackMin[slot].WriteSamplerFeedbackBias(pairedTexture, pairedSampler, uv, 0.5);
            feedbackMin[slot].WriteSamplerFeedbackGrad(pairedTexture, pairedSampler, uv, ddxValue, ddyValue);
            feedbackMin[slot].WriteSamplerFeedbackLevel(pairedTexture, pairedSampler, uv, 2.0);
            feedbackUsed[slot].WriteSamplerFeedbackLevel(pairedLayers, pairedSampler, uvLayer, 1.0);
            return pairedTexture.Sample(pairedSampler, uv);
        }
    """).strip()

    crossgl = generate_crossgl(code)

    assert (
        "write_sampler_feedback(feedbackMin[slot], pairedTexture, pairedSampler, uv);"
        in crossgl
    )
    assert (
        "write_sampler_feedback_bias("
        "feedbackMin[slot], pairedTexture, pairedSampler, uv, 0.5);" in crossgl
    )
    assert (
        "write_sampler_feedback_grad("
        "feedbackMin[slot], pairedTexture, pairedSampler, uv, ddxValue, ddyValue);"
        in crossgl
    )
    assert (
        "write_sampler_feedback_level("
        "feedbackMin[slot], pairedTexture, pairedSampler, uv, 2.0);" in crossgl
    )
    assert (
        "write_sampler_feedback_level("
        "feedbackUsed[slot], pairedLayers, pairedSampler, uvLayer, 1.0);" in crossgl
    )
    assert ".WriteSamplerFeedback" not in crossgl

    hlsl = TranslatorHLSLCodeGen().generate(parse_crossgl(crossgl))

    assert (
        "FeedbackTexture2D<SAMPLER_FEEDBACK_MIN_MIP> feedbackMin[2] : "
        "register(u0, space10);" in hlsl
    )
    assert (
        "FeedbackTexture2DArray<SAMPLER_FEEDBACK_MIP_REGION_USED> feedbackUsed[] : "
        "register(u2, space10);" in hlsl
    )
    assert (
        "feedbackMin[slot].WriteSamplerFeedback(pairedTexture, pairedSampler, uv);"
        in hlsl
    )
    assert (
        "feedbackMin[slot].WriteSamplerFeedbackBias("
        "pairedTexture, pairedSampler, uv, 0.5);" in hlsl
    )
    assert (
        "feedbackMin[slot].WriteSamplerFeedbackGrad("
        "pairedTexture, pairedSampler, uv, ddxValue, ddyValue);" in hlsl
    )
    assert (
        "feedbackMin[slot].WriteSamplerFeedbackLevel("
        "pairedTexture, pairedSampler, uv, 2.0);" in hlsl
    )
    assert (
        "feedbackUsed[slot].WriteSamplerFeedbackLevel("
        "pairedLayers, pairedSampler, uvLayer, 1.0);" in hlsl
    )
    assert "write_sampler_feedback" not in hlsl


def test_codegen_nonuniform_resource_index_typed_and_raw_buffer_arrays_roundtrip():
    code = textwrap.dedent("""
        StructuredBuffer<float4> positions[4] : register(t0, space3);
        RWStructuredBuffer<float4> outPositions[4] : register(u0, space3);
        ByteAddressBuffer rawBuffers[4] : register(t4, space3);
        RWByteAddressBuffer rwRawBuffers[4] : register(u4, space3);

        float4 main(
            uint materialIndex : TEXCOORD0,
            uint outputIndex : TEXCOORD1
        ) : SV_Target0 {
            uint src = NonUniformResourceIndex(materialIndex);
            uint dst = NonUniformResourceIndex(outputIndex);
            float4 value = positions[src][0];
            outPositions[dst][0] = value;
            uint raw = rawBuffers[src].Load(0);
            rwRawBuffers[dst].Store(0, raw);
            return value + float4(raw & 1u, 0, 0, 0);
        }
    """).strip()

    crossgl = generate_crossgl(code)

    assert "StructuredBuffer<float4> positions[4];" in crossgl
    assert "RWStructuredBuffer<float4> outPositions[4];" in crossgl
    assert "ByteAddressBuffer rawBuffers[4];" in crossgl
    assert "RWByteAddressBuffer rwRawBuffers[4];" in crossgl
    assert "uint src = NonUniformResourceIndex(materialIndex);" in crossgl
    assert "uint dst = NonUniformResourceIndex(outputIndex);" in crossgl
    assert "vec4 value = positions[src][0];" in crossgl
    assert "outPositions[dst][0] = value;" in crossgl
    assert "uint raw = buffer_load(rawBuffers[src], 0);" in crossgl
    assert "buffer_store(rwRawBuffers[dst], 0, raw);" in crossgl

    hlsl = TranslatorHLSLCodeGen().generate(parse_crossgl(crossgl))

    assert "StructuredBuffer<float4> positions[4] : register(t0, space3);" in hlsl
    assert "RWStructuredBuffer<float4> outPositions[4] : register(u0, space3);" in hlsl
    assert "ByteAddressBuffer rawBuffers[4] : register(t4, space3);" in hlsl
    assert "RWByteAddressBuffer rwRawBuffers[4] : register(u4, space3);" in hlsl
    assert "uint src = NonUniformResourceIndex(materialIndex);" in hlsl
    assert "uint dst = NonUniformResourceIndex(outputIndex);" in hlsl
    assert "float4 value = positions[src][0];" in hlsl
    assert "outPositions[dst][0] = value;" in hlsl
    assert "uint raw = rawBuffers[src].Load(0);" in hlsl
    assert "rwRawBuffers[dst].Store(0, raw);" in hlsl
    assert "float src" not in hlsl
    assert "float dst" not in hlsl
    assert "buffer_load(" not in hlsl
    assert "buffer_store(" not in hlsl


def test_codegen_nonuniform_resource_index_multidimensional_typed_buffer_arrays_roundtrip():
    code = textwrap.dedent("""
        Buffer<float4> coefficients[][2] : register(t0, space4);
        RWBuffer<uint> counters[] : register(u0, space4);

        float4 main(
            uint materialIndex : TEXCOORD0,
            uint tileIndex : TEXCOORD1
        ) : SV_Target0 {
            uint nonUniformMaterial = NonUniformResourceIndex(materialIndex);
            uint nonUniformTile = NonUniformResourceIndex(tileIndex);
            float4 value = coefficients[nonUniformMaterial][nonUniformTile][0];
            uint oldValue;
            InterlockedAdd(counters[nonUniformMaterial][0], 1u, oldValue);
            return value + float4(oldValue, 0, 0, 0);
        }
    """).strip()

    crossgl = generate_crossgl(code)

    assert "Buffer<float4> coefficients[][2];" in crossgl
    assert "RWBuffer<uint> counters[];" in crossgl
    assert (
        "uint nonUniformMaterial = NonUniformResourceIndex(materialIndex);" in crossgl
    )
    assert "uint nonUniformTile = NonUniformResourceIndex(tileIndex);" in crossgl
    assert (
        "vec4 value = coefficients[nonUniformMaterial][nonUniformTile][0];" in crossgl
    )
    assert "oldValue = atomicAdd(counters[nonUniformMaterial][0], 1u);" in crossgl

    hlsl = TranslatorHLSLCodeGen().generate(parse_crossgl(crossgl))

    assert "Buffer<float4> coefficients[][2] : register(t0, space4);" in hlsl
    assert "RWBuffer<uint> counters[] : register(u0, space4);" in hlsl
    assert "uint nonUniformMaterial = NonUniformResourceIndex(materialIndex);" in hlsl
    assert "uint nonUniformTile = NonUniformResourceIndex(tileIndex);" in hlsl
    assert "float4 value = coefficients[nonUniformMaterial][nonUniformTile][0];" in (
        hlsl
    )
    assert "InterlockedAdd(counters[nonUniformMaterial][0], 1u, oldValue);" in hlsl
    assert "float nonUniformMaterial" not in hlsl
    assert "float nonUniformTile" not in hlsl
    assert "atomicAdd(" not in hlsl


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


def test_codegen_append_consume_structured_buffer_arrays_roundtrip():
    code = textwrap.dedent("""
        AppendStructuredBuffer<int> appendValues[4] : register(u0, space5);
        ConsumeStructuredBuffer<int> consumeValues[4] : register(u4, space5);

        void main(uint queueIndex : TEXCOORD0, uint value : TEXCOORD1) {
            uint slot = NonUniformResourceIndex(queueIndex);
            appendValues[slot].Append(int(value));
            int consumed = consumeValues[slot].Consume();
            appendValues[slot].Append(consumed + 1);
        }
    """).strip()

    crossgl = generate_crossgl(code)

    assert "@ register(u0, space5)" in crossgl
    assert "AppendStructuredBuffer<int> appendValues[4];" in crossgl
    assert "@ register(u4, space5)" in crossgl
    assert "ConsumeStructuredBuffer<int> consumeValues[4];" in crossgl
    assert "uint slot = NonUniformResourceIndex(queueIndex);" in crossgl
    assert "buffer_append(appendValues[slot], int(value));" in crossgl
    assert "int consumed = buffer_consume(consumeValues[slot]);" in crossgl
    assert "buffer_append(appendValues[slot], consumed + 1);" in crossgl
    assert ".Append(" not in crossgl
    assert ".Consume(" not in crossgl

    hlsl = TranslatorHLSLCodeGen().generate(parse_crossgl(crossgl))

    assert "AppendStructuredBuffer<int> appendValues[4] : register(u0, space5);" in hlsl
    assert (
        "ConsumeStructuredBuffer<int> consumeValues[4] : register(u4, space5);" in hlsl
    )
    assert "uint slot = NonUniformResourceIndex(queueIndex);" in hlsl
    assert "appendValues[slot].Append(int(value));" in hlsl
    assert "int consumed = consumeValues[slot].Consume();" in hlsl
    assert "appendValues[slot].Append((consumed + 1));" in hlsl
    assert "float slot" not in hlsl
    assert "buffer_append(" not in hlsl
    assert "buffer_consume(" not in hlsl


def test_codegen_append_consume_multidimensional_buffer_arrays_roundtrip():
    code = textwrap.dedent("""
        AppendStructuredBuffer<uint> appendQueues[][2] : register(u0, space6);
        ConsumeStructuredBuffer<uint> consumeQueues[] : register(u8, space6);

        uint main(uint queueIndex : TEXCOORD0, uint laneIndex : TEXCOORD1) : SV_Target0 {
            uint queue = NonUniformResourceIndex(queueIndex);
            uint lane = NonUniformResourceIndex(laneIndex);
            uint consumed = consumeQueues[queue].Consume();
            appendQueues[queue][lane].Append(consumed + lane);
            return consumed;
        }
    """).strip()

    crossgl = generate_crossgl(code)

    assert "@ register(u0, space6)" in crossgl
    assert "AppendStructuredBuffer<uint> appendQueues[][2];" in crossgl
    assert "@ register(u8, space6)" in crossgl
    assert "ConsumeStructuredBuffer<uint> consumeQueues[];" in crossgl
    assert "uint queue = NonUniformResourceIndex(queueIndex);" in crossgl
    assert "uint lane = NonUniformResourceIndex(laneIndex);" in crossgl
    assert "uint consumed = buffer_consume(consumeQueues[queue]);" in crossgl
    assert "buffer_append(appendQueues[queue][lane], consumed + lane);" in crossgl

    hlsl = TranslatorHLSLCodeGen().generate(parse_crossgl(crossgl))

    assert (
        "AppendStructuredBuffer<uint> appendQueues[][2] : register(u0, space6);" in hlsl
    )
    assert "ConsumeStructuredBuffer<uint> consumeQueues[] : register(u8, space6);" in (
        hlsl
    )
    assert "uint queue = NonUniformResourceIndex(queueIndex);" in hlsl
    assert "uint lane = NonUniformResourceIndex(laneIndex);" in hlsl
    assert "uint consumed = consumeQueues[queue].Consume();" in hlsl
    assert "appendQueues[queue][lane].Append((consumed + lane));" in hlsl
    assert "float queue" not in hlsl
    assert "float lane" not in hlsl
    assert "buffer_append(" not in hlsl
    assert "buffer_consume(" not in hlsl


def test_codegen_rwstructured_buffer_counter_methods_roundtrip():
    code = textwrap.dedent("""
        RWStructuredBuffer<uint> counters[2] : register(u0, space7);

        uint main(uint queueIndex : TEXCOORD0) : SV_Target0 {
            uint slot = NonUniformResourceIndex(queueIndex);
            uint nextIndex = counters[slot].IncrementCounter();
            uint oldIndex = counters[slot].DecrementCounter();
            counters[slot][nextIndex] = oldIndex;
            return nextIndex + oldIndex;
        }
    """).strip()

    crossgl = generate_crossgl(code)

    assert "@ register(u0, space7)" in crossgl
    assert "RWStructuredBuffer<uint> counters[2];" in crossgl
    assert "uint slot = NonUniformResourceIndex(queueIndex);" in crossgl
    assert "uint nextIndex = buffer_increment_counter(counters[slot]);" in crossgl
    assert "uint oldIndex = buffer_decrement_counter(counters[slot]);" in crossgl
    assert "counters[slot][nextIndex] = oldIndex;" in crossgl
    assert ".IncrementCounter(" not in crossgl
    assert ".DecrementCounter(" not in crossgl

    hlsl = TranslatorHLSLCodeGen().generate(parse_crossgl(crossgl))

    assert "RWStructuredBuffer<uint> counters[2] : register(u0, space7);" in hlsl
    assert "uint slot = NonUniformResourceIndex(queueIndex);" in hlsl
    assert "uint nextIndex = counters[slot].IncrementCounter();" in hlsl
    assert "uint oldIndex = counters[slot].DecrementCounter();" in hlsl
    assert "counters[slot][nextIndex] = oldIndex;" in hlsl
    assert "float nextIndex" not in hlsl
    assert "float oldIndex" not in hlsl
    assert "buffer_increment_counter(" not in hlsl
    assert "buffer_decrement_counter(" not in hlsl


def test_codegen_buffer_get_dimensions_array_receivers_roundtrip():
    code = textwrap.dedent("""
        StructuredBuffer<float4> sourceBuffers[3] : register(t0, space8);
        RWStructuredBuffer<uint> outputBuffers[3] : register(u0, space8);
        ByteAddressBuffer rawInputs[3] : register(t3, space8);
        RWByteAddressBuffer rawOutputs[3] : register(u3, space8);

        uint main(uint bufferIndex : TEXCOORD0) : SV_Target0 {
            uint slot = NonUniformResourceIndex(bufferIndex);
            uint sourceCount;
            uint sourceStride;
            uint outputCount;
            uint outputStride;
            uint rawInputBytes;
            uint rawOutputBytes;
            sourceBuffers[slot].GetDimensions(sourceCount, sourceStride);
            outputBuffers[slot].GetDimensions(outputCount, outputStride);
            rawInputs[slot].GetDimensions(rawInputBytes);
            rawOutputs[slot].GetDimensions(rawOutputBytes);
            return sourceCount + sourceStride + outputCount + outputStride
                + rawInputBytes + rawOutputBytes;
        }
    """).strip()

    crossgl = generate_crossgl(code)

    assert "StructuredBuffer<float4> sourceBuffers[3];" in crossgl
    assert "RWStructuredBuffer<uint> outputBuffers[3];" in crossgl
    assert "ByteAddressBuffer rawInputs[3];" in crossgl
    assert "RWByteAddressBuffer rawOutputs[3];" in crossgl
    assert "uint slot = NonUniformResourceIndex(bufferIndex);" in crossgl
    assert (
        "buffer_dimensions(sourceBuffers[slot], sourceCount, sourceStride);" in crossgl
    )
    assert (
        "buffer_dimensions(outputBuffers[slot], outputCount, outputStride);" in crossgl
    )
    assert "buffer_dimensions(rawInputs[slot], rawInputBytes);" in crossgl
    assert "buffer_dimensions(rawOutputs[slot], rawOutputBytes);" in crossgl
    assert ".GetDimensions(" not in crossgl

    hlsl = TranslatorHLSLCodeGen().generate(parse_crossgl(crossgl))

    assert "StructuredBuffer<float4> sourceBuffers[3] : register(t0, space8);" in hlsl
    assert "RWStructuredBuffer<uint> outputBuffers[3] : register(u0, space8);" in hlsl
    assert "ByteAddressBuffer rawInputs[3] : register(t3, space8);" in hlsl
    assert "RWByteAddressBuffer rawOutputs[3] : register(u3, space8);" in hlsl
    assert "uint slot = NonUniformResourceIndex(bufferIndex);" in hlsl
    assert "sourceBuffers[slot].GetDimensions(sourceCount, sourceStride);" in hlsl
    assert "outputBuffers[slot].GetDimensions(outputCount, outputStride);" in hlsl
    assert "rawInputs[slot].GetDimensions(rawInputBytes);" in hlsl
    assert "rawOutputs[slot].GetDimensions(rawOutputBytes);" in hlsl
    assert "float slot" not in hlsl
    assert "buffer_dimensions(" not in hlsl


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


def test_codegen_sample_cmp_infers_shadow_struct_members():
    code = textwrap.dedent("""
        struct ResourceBundle {
            Texture2D<float> shadow;
            Texture2D<float4> color;
        };

        ResourceBundle resources;
        SamplerComparisonState compareSampler : register(s0);
        SamplerState linearSampler : register(s1);

        float4 main(float2 uv : TEXCOORD0, float depth : TEXCOORD1) : SV_Target0 {
            float sampled = resources.shadow.SampleCmpLevelZero(
                compareSampler, uv, depth
            );
            float4 color = resources.color.Sample(linearSampler, uv);
            return color + float4(sampled);
        }
    """).strip()

    crossgl = generate_crossgl(code)

    assert "sampler2DShadow shadow;" in crossgl
    assert "sampler2D color;" in crossgl
    assert "sampler2DShadow color;" not in crossgl
    assert "textureCompare(resources.shadow, compareSampler, uv, depth)" in crossgl
    assert "texture(resources.color, linearSampler, uv)" in crossgl

    ast = parse_crossgl(crossgl)
    glsl = GLSLCodeGen().generate(ast)
    assert "sampler2DShadow shadow;" in glsl
    assert "sampler2D color;" in glsl
    assert "texture(resources.shadow, vec3(uv, depth))" in glsl
    assert "texture(resources.color, uv)" in glsl
    assert "unsupported GLSL texture compare" not in glsl

    hlsl = TranslatorHLSLCodeGen().generate(ast)
    assert "Texture2D shadow" in hlsl
    assert "Texture2D color" in hlsl
    assert "resources.shadow.SampleCmp(compareSampler, uv, depth)" in hlsl
    assert "resources.color.Sample(linearSampler, uv)" in hlsl
    assert "textureCompare(" not in hlsl


def test_codegen_shadow_struct_member_inference_is_scoped_by_struct_type():
    code = textwrap.dedent("""
        struct ShadowBundle {
            Texture2D<float> texture;
        };

        struct ColorBundle {
            Texture2D<float4> texture;
        };

        ShadowBundle shadowResources;
        ColorBundle colorResources;
        SamplerComparisonState compareSampler : register(s0);
        SamplerState linearSampler : register(s1);

        float4 main(float2 uv : TEXCOORD0, float depth : TEXCOORD1) : SV_Target0 {
            float sampled = shadowResources.texture.SampleCmpLevelZero(
                compareSampler, uv, depth
            );
            float4 color = colorResources.texture.Sample(linearSampler, uv);
            return color + float4(sampled);
        }
    """).strip()

    crossgl = generate_crossgl(code)

    assert "struct ShadowBundle {\n        sampler2DShadow texture;" in crossgl
    assert "struct ColorBundle {\n        sampler2D texture;" in crossgl
    assert "struct ColorBundle {\n        sampler2DShadow texture;" not in crossgl
    assert "textureCompare(shadowResources.texture, compareSampler, uv, depth)" in (
        crossgl
    )
    assert "texture(colorResources.texture, linearSampler, uv)" in crossgl


def test_codegen_sample_cmp_infers_shadow_struct_members_through_helpers():
    code = textwrap.dedent("""
        struct ResourceBundle {
            Texture2D<float> shadow;
            Texture2D<float4> color;
        };

        ResourceBundle resources;
        SamplerComparisonState compareSampler : register(s0);
        SamplerState linearSampler : register(s1);

        float readShadow(
            Texture2D<float> source,
            SamplerComparisonState cmpSampler,
            float2 uv,
            float depth
        ) {
            return source.SampleCmpLevelZero(cmpSampler, uv, depth);
        }

        float readShadowWrapper(
            Texture2D<float> source,
            SamplerComparisonState cmpSampler,
            float2 uv,
            float depth
        ) {
            return readShadow(source, cmpSampler, uv, depth);
        }

        float4 readColor(
            Texture2D<float4> source,
            SamplerState texSampler,
            float2 uv
        ) {
            return source.Sample(texSampler, uv);
        }

        float4 main(float2 uv : TEXCOORD0, float depth : TEXCOORD1) : SV_Target0 {
            float sampled = readShadowWrapper(
                resources.shadow, compareSampler, uv, depth
            );
            float4 color = readColor(resources.color, linearSampler, uv);
            return color + float4(sampled);
        }
    """).strip()

    crossgl = generate_crossgl(code)

    assert "sampler2DShadow shadow;" in crossgl
    assert "sampler2D color;" in crossgl
    assert "sampler2DShadow color;" not in crossgl
    assert "float readShadow(sampler2DShadow source" in crossgl
    assert "float readShadowWrapper(sampler2DShadow source" in crossgl
    assert "vec4 readColor(sampler2D source" in crossgl
    assert "readShadowWrapper(resources.shadow, compareSampler, uv, depth)" in crossgl
    assert "readColor(resources.color, linearSampler, uv)" in crossgl

    glsl = GLSLCodeGen().generate(parse_crossgl(crossgl))
    assert "sampler2DShadow shadow;" in glsl
    assert "sampler2D color;" in glsl
    assert "readShadowWrapper(resources.shadow, uv, depth)" in glsl
    assert "texture(source, vec3(uv, depth))" in glsl
    assert "unsupported GLSL texture compare" not in glsl


def test_codegen_sample_cmp_infers_shadow_struct_member_arrays_through_helpers():
    code = textwrap.dedent("""
        struct ResourceBundle {
            Texture2D<float> shadows[2];
            Texture2D<float4> colors[2];
        };

        ResourceBundle resources;
        SamplerComparisonState compareSampler : register(s0);
        SamplerState linearSampler : register(s1);

        float readShadow(
            Texture2D<float> source,
            SamplerComparisonState cmpSampler,
            float2 uv,
            float depth
        ) {
            return source.SampleCmpLevelZero(cmpSampler, uv, depth);
        }

        float4 readColor(
            Texture2D<float4> source,
            SamplerState texSampler,
            float2 uv
        ) {
            return source.Sample(texSampler, uv);
        }

        float4 main(float2 uv : TEXCOORD0, float depth : TEXCOORD1, uint index : TEXCOORD2) : SV_Target0 {
            float sampled = readShadow(
                resources.shadows[index], compareSampler, uv, depth
            );
            float4 color = readColor(resources.colors[index], linearSampler, uv);
            return color + float4(sampled);
        }
    """).strip()

    crossgl = generate_crossgl(code)

    assert "sampler2DShadow shadows[2];" in crossgl
    assert "sampler2D colors[2];" in crossgl
    assert "sampler2DShadow colors[2];" not in crossgl
    assert "float readShadow(sampler2DShadow source" in crossgl
    assert "vec4 readColor(sampler2D source" in crossgl
    assert "readShadow(resources.shadows[index], compareSampler, uv, depth)" in crossgl
    assert "readColor(resources.colors[index], linearSampler, uv)" in crossgl

    glsl = GLSLCodeGen().generate(parse_crossgl(crossgl))
    assert "sampler2DShadow shadows[2];" in glsl
    assert "sampler2D colors[2];" in glsl
    assert "readShadow(resources.shadows[index], uv, depth)" in glsl
    assert "texture(source, vec3(uv, depth))" in glsl
    assert "unsupported GLSL texture compare" not in glsl


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
    assert "vec4 readRegular(sampler2D shared_, sampler localSampler" in crossgl
    assert "textureCompare(shared, compareSampler, uv, depth)" in crossgl
    assert "texture(shared_, localSampler, uv)" in crossgl
    assert parse_crossgl(crossgl) is not None


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
    assert "vec4 m = (a * b);" in output
    assert "mix" in output
    assert "clamp(" in output


def test_codegen_frac_intrinsic_from_microsoft_docs_roundtrips():
    hlsl = textwrap.dedent("""
        float4 main(float4 uvPhase : TEXCOORD0) : SV_Target0 {
            float4 wrapped = frac(uvPhase);
            return wrapped;
        }
    """).strip()

    crossgl = generate_crossgl(hlsl)

    assert "vec4 wrapped = fract(uvPhase);" in crossgl
    assert "frac(" not in crossgl

    regenerated_hlsl = TranslatorHLSLCodeGen().generate(parse_crossgl(crossgl))
    assert "float4 wrapped = frac(uvPhase);" in regenerated_hlsl
    assert "fract(" not in regenerated_hlsl


def test_codegen_rcp_intrinsic_from_microsoft_docs_imports_to_reciprocal_expression():
    # Source: Microsoft Learn rcp intrinsic docs.
    # URL: https://learn.microsoft.com/windows/win32/direct3dhlsl/rcp
    crossgl = generate_crossgl("""
        float4 main(float4 lightFalloff : TEXCOORD0) : SV_Target0 {
            float4 invFalloff = rcp(lightFalloff + 1.0);
            return invFalloff;
        }
    """)

    assert "vec4 invFalloff = (1.0 / (lightFalloff + 1.0));" in crossgl
    assert "rcp(" not in crossgl
    parse_crossgl(crossgl)


def test_codegen_fmod_and_atan2_intrinsics_from_microsoft_docs_import_to_crossgl_names():
    # Sources: Microsoft Learn HLSL intrinsic docs for fmod and atan2.
    # URLs:
    # https://learn.microsoft.com/windows/win32/direct3dhlsl/dx-graphics-hlsl-fmod
    # https://learn.microsoft.com/windows/win32/direct3dhlsl/dx-graphics-hlsl-atan2
    crossgl = generate_crossgl("""
        float4 main(float2 p : TEXCOORD0) : SV_Target0 {
            float wrapped = fmod(p.x, 1.0);
            float angle = atan2(p.y, p.x);
            return float4(wrapped, angle, 0.0, 1.0);
        }
    """)

    assert "float wrapped = mod(p.x, 1.0);" in crossgl
    assert "float angle = atan(p.y, p.x);" in crossgl
    assert "fmod(" not in crossgl
    assert "atan2(" not in crossgl
    parse_crossgl(crossgl)


def test_codegen_interpolation_intrinsics_roundtrip():
    crossgl = generate_crossgl(INTERPOLATION_INTRINSICS_HLSL)

    assert "interpolateAtSample(color, sampleIndex)" in crossgl
    assert "interpolateAtOffset(color, snappedOffset)" in crossgl
    assert "interpolateAtCentroid(color)" in crossgl
    assert "EvaluateAttribute" not in crossgl

    regenerated_hlsl = TranslatorHLSLCodeGen().generate(parse_crossgl(crossgl))
    assert "EvaluateAttributeAtSample(color, sampleIndex)" in regenerated_hlsl
    assert "EvaluateAttributeSnapped(color, snappedOffset)" in regenerated_hlsl
    assert "EvaluateAttributeCentroid(color)" in regenerated_hlsl
    assert "interpolateAtSample(" not in regenerated_hlsl
    assert "interpolateAtOffset(" not in regenerated_hlsl
    assert "interpolateAtCentroid(" not in regenerated_hlsl


def test_codegen_get_dimensions_mapping():
    output = generate_crossgl(DIMENSIONS_HLSL)
    assert "w = uint(textureSize(tex, 0).x);" in output
    assert "h = uint(textureSize(tex, 0).y);" in output
    assert "texture_dimensions" not in output
    assert "buffer_dimensions" in output

    regenerated_hlsl = TranslatorHLSLCodeGen().generate(parse_crossgl(output))
    assert "buf.GetDimensions(len);" in regenerated_hlsl
    assert "texture_dimensions" not in regenerated_hlsl
    assert "buffer_dimensions" not in regenerated_hlsl


def test_codegen_get_dimensions_out_parameters_import_to_queries():
    code = textwrap.dedent("""
        Texture2D<float4> colorMap : register(t0);
        Texture2DArray<float4> layerMap : register(t1);
        Texture2DMS<float4> msMap : register(t2);
        RWTexture3D<float4> volume : register(u0);
        StructuredBuffer<float4> structs : register(t3);

        void main(uint lod : TEXCOORD0) {
            uint baseWidth;
            uint baseHeight;
            uint baseLevels;
            uint arrayWidth;
            uint arrayHeight;
            uint arrayElements;
            uint arrayLevels;
            uint msWidth;
            uint msHeight;
            uint msSamples;
            uint volumeWidth;
            uint volumeHeight;
            uint volumeDepth;
            uint structCount;
            uint stride;
            colorMap.GetDimensions(baseWidth, baseHeight, baseLevels);
            layerMap.GetDimensions(lod, arrayWidth, arrayHeight, arrayElements, arrayLevels);
            msMap.GetDimensions(msWidth, msHeight, msSamples);
            volume.GetDimensions(volumeWidth, volumeHeight, volumeDepth);
            structs.GetDimensions(structCount, stride);
        }
    """).strip()

    crossgl = generate_crossgl(code)

    assert ".GetDimensions(" not in crossgl
    assert "texture_dimensions(" not in crossgl
    assert "baseWidth = uint(textureSize(colorMap, 0).x);" in crossgl
    assert "baseHeight = uint(textureSize(colorMap, 0).y);" in crossgl
    assert "baseLevels = uint(textureQueryLevels(colorMap));" in crossgl
    assert "arrayWidth = uint(textureSize(layerMap, lod).x);" in crossgl
    assert "arrayHeight = uint(textureSize(layerMap, lod).y);" in crossgl
    assert "arrayElements = uint(textureSize(layerMap, lod).z);" in crossgl
    assert "arrayLevels = uint(textureQueryLevels(layerMap));" in crossgl
    assert "msWidth = uint(textureSize(msMap).x);" in crossgl
    assert "msHeight = uint(textureSize(msMap).y);" in crossgl
    assert "msSamples = uint(textureSamples(msMap));" in crossgl
    assert "volumeWidth = uint(imageSize(volume).x);" in crossgl
    assert "volumeHeight = uint(imageSize(volume).y);" in crossgl
    assert "volumeDepth = uint(imageSize(volume).z);" in crossgl
    assert "buffer_dimensions(structs, structCount, stride);" in crossgl

    regenerated_hlsl = TranslatorHLSLCodeGen().generate(parse_crossgl(crossgl))
    assert "texture_dimensions(" not in regenerated_hlsl
    assert "buffer_dimensions(" not in regenerated_hlsl
    assert "structs.GetDimensions(structCount, stride);" in regenerated_hlsl
    assert "int textureQueryLevels(Texture2D tex)" in regenerated_hlsl
    assert "int3 textureSize(Texture2DArray tex, int lod)" in regenerated_hlsl
    assert "int2 textureSize(Texture2DMS<float4> tex)" in regenerated_hlsl
    assert "int3 imageSize(RWTexture3D<float4> image)" in regenerated_hlsl


def test_codegen_get_dimensions_edge_overloads_import_to_queries():
    code = textwrap.dedent("""
        Texture1D<float4> lineMap : register(t0);
        Texture1DArray<float4> lineArray : register(t1);
        TextureCube<float4> cubeMap : register(t2);
        TextureCubeArray<float4> cubeArray : register(t3);
        RWTexture1DArray<float4> imageArray : register(u0);
        Texture2DMSArray<float4> msArray : register(t4);
        RWTexture2DMSArray<float4> msImage : register(u1);

        void main(uint lod : TEXCOORD0) {
            uint lineWidth;
            uint lineLevels;
            uint arrayWidth;
            uint arrayElements;
            uint arrayLevels;
            uint cubeWidth;
            uint cubeHeight;
            uint cubeLevels;
            uint cubeArrayWidth;
            uint cubeArrayHeight;
            uint cubeArrayElements;
            uint cubeArrayLevels;
            uint imageWidth;
            uint imageElements;
            uint msWidth;
            uint msHeight;
            uint msElements;
            uint msSamples;
            uint msImageWidth;
            uint msImageHeight;
            uint msImageElements;
            uint msImageSamples;
            lineMap.GetDimensions(lineWidth, lineLevels);
            lineArray.GetDimensions(lod, arrayWidth, arrayElements, arrayLevels);
            cubeMap.GetDimensions(cubeWidth, cubeHeight, cubeLevels);
            cubeArray.GetDimensions(
                lod, cubeArrayWidth, cubeArrayHeight, cubeArrayElements,
                cubeArrayLevels
            );
            imageArray.GetDimensions(imageWidth, imageElements);
            msArray.GetDimensions(msWidth, msHeight, msElements, msSamples);
            msImage.GetDimensions(
                msImageWidth, msImageHeight, msImageElements, msImageSamples
            );
        }
    """).strip()

    crossgl = generate_crossgl(code)

    assert ".GetDimensions(" not in crossgl
    assert "texture_dimensions(" not in crossgl
    assert "lineWidth = uint(textureSize(lineMap, 0));" in crossgl
    assert "lineLevels = uint(textureQueryLevels(lineMap));" in crossgl
    assert "arrayWidth = uint(textureSize(lineArray, lod).x);" in crossgl
    assert "arrayElements = uint(textureSize(lineArray, lod).y);" in crossgl
    assert "arrayLevels = uint(textureQueryLevels(lineArray));" in crossgl
    assert "cubeWidth = uint(textureSize(cubeMap, 0).x);" in crossgl
    assert "cubeHeight = uint(textureSize(cubeMap, 0).y);" in crossgl
    assert "cubeLevels = uint(textureQueryLevels(cubeMap));" in crossgl
    assert "cubeArrayWidth = uint(textureSize(cubeArray, lod).x);" in crossgl
    assert "cubeArrayHeight = uint(textureSize(cubeArray, lod).y);" in crossgl
    assert "cubeArrayElements = uint(textureSize(cubeArray, lod).z);" in crossgl
    assert "cubeArrayLevels = uint(textureQueryLevels(cubeArray));" in crossgl
    assert "imageWidth = uint(imageSize(imageArray).x);" in crossgl
    assert "imageElements = uint(imageSize(imageArray).y);" in crossgl
    assert "msWidth = uint(textureSize(msArray).x);" in crossgl
    assert "msHeight = uint(textureSize(msArray).y);" in crossgl
    assert "msElements = uint(textureSize(msArray).z);" in crossgl
    assert "msSamples = uint(textureSamples(msArray));" in crossgl
    assert "msImageWidth = uint(imageSize(msImage).x);" in crossgl
    assert "msImageHeight = uint(imageSize(msImage).y);" in crossgl
    assert "msImageElements = uint(imageSize(msImage).z);" in crossgl
    assert "msImageSamples = uint(textureSamples(msImage));" in crossgl

    regenerated_hlsl = TranslatorHLSLCodeGen().generate(parse_crossgl(crossgl))
    assert "int textureSize(Texture1D tex, int lod)" in regenerated_hlsl
    assert "int2 textureSize(Texture1DArray tex, int lod)" in regenerated_hlsl
    assert "int2 textureSize(TextureCube tex, int lod)" in regenerated_hlsl
    assert "int3 textureSize(TextureCubeArray tex, int lod)" in regenerated_hlsl
    assert "int2 imageSize(RWTexture1DArray<float4> image)" in regenerated_hlsl
    assert "int3 textureSize(Texture2DMSArray<float4> tex)" in regenerated_hlsl
    assert "int3 imageSize(Texture2DMSArray<float4> image)" in regenerated_hlsl
    assert (
        regenerated_hlsl.count("int textureSamples(Texture2DMSArray<float4> tex)") == 1
    )


def test_codegen_get_dimensions_uint_vector_component_outputs_keep_casts():
    code = textwrap.dedent("""
        Texture2D<float4> colorMap : register(t0);
        Texture2DArray<float4> layerMap : register(t1);
        Texture2DMS<float4> msMap : register(t2);
        RWTexture2D<uint> counters : register(u0);

        struct DimensionBundle {
            uint4 color;
            uint4 layers;
            uint4 ms;
            uint2 image;
        };

        void main(uint lod : TEXCOORD0) {
            uint4 colorDims;
            uint4 layerDims;
            uint4 msDims;
            uint2 imageDims;
            DimensionBundle nestedDims;
            colorMap.GetDimensions(colorDims.x, colorDims.y, colorDims.z);
            layerMap.GetDimensions(
                lod, layerDims.x, layerDims.y, layerDims.z, layerDims.w
            );
            msMap.GetDimensions(msDims.x, msDims.y, msDims.z);
            counters.GetDimensions(imageDims.x, imageDims.y);
            colorMap.GetDimensions(
                nestedDims.color.x, nestedDims.color.y, nestedDims.color.z
            );
            layerMap.GetDimensions(
                lod,
                nestedDims.layers.x,
                nestedDims.layers.y,
                nestedDims.layers.z,
                nestedDims.layers.w
            );
            msMap.GetDimensions(
                nestedDims.ms.x, nestedDims.ms.y, nestedDims.ms.z
            );
            counters.GetDimensions(nestedDims.image.x, nestedDims.image.y);
        }
    """).strip()

    crossgl = generate_crossgl(code)

    assert "colorDims.x = uint(textureSize(colorMap, 0).x);" in crossgl
    assert "colorDims.y = uint(textureSize(colorMap, 0).y);" in crossgl
    assert "colorDims.z = uint(textureQueryLevels(colorMap));" in crossgl
    assert "layerDims.x = uint(textureSize(layerMap, lod).x);" in crossgl
    assert "layerDims.y = uint(textureSize(layerMap, lod).y);" in crossgl
    assert "layerDims.z = uint(textureSize(layerMap, lod).z);" in crossgl
    assert "layerDims.w = uint(textureQueryLevels(layerMap));" in crossgl
    assert "msDims.x = uint(textureSize(msMap).x);" in crossgl
    assert "msDims.y = uint(textureSize(msMap).y);" in crossgl
    assert "msDims.z = uint(textureSamples(msMap));" in crossgl
    assert "imageDims.x = uint(imageSize(counters).x);" in crossgl
    assert "imageDims.y = uint(imageSize(counters).y);" in crossgl
    assert "nestedDims.color.x = uint(textureSize(colorMap, 0).x);" in crossgl
    assert "nestedDims.color.y = uint(textureSize(colorMap, 0).y);" in crossgl
    assert "nestedDims.color.z = uint(textureQueryLevels(colorMap));" in crossgl
    assert "nestedDims.layers.x = uint(textureSize(layerMap, lod).x);" in crossgl
    assert "nestedDims.layers.y = uint(textureSize(layerMap, lod).y);" in crossgl
    assert "nestedDims.layers.z = uint(textureSize(layerMap, lod).z);" in crossgl
    assert "nestedDims.layers.w = uint(textureQueryLevels(layerMap));" in crossgl
    assert "nestedDims.ms.x = uint(textureSize(msMap).x);" in crossgl
    assert "nestedDims.ms.y = uint(textureSize(msMap).y);" in crossgl
    assert "nestedDims.ms.z = uint(textureSamples(msMap));" in crossgl
    assert "nestedDims.image.x = uint(imageSize(counters).x);" in crossgl
    assert "nestedDims.image.y = uint(imageSize(counters).y);" in crossgl


def test_codegen_get_dimensions_unsupported_overload_imports_diagnostic():
    code = textwrap.dedent("""
        Texture2D<float4> colorMap : register(t0);

        void main() {
            uint width;
            uint height;
            uint levels;
            uint extra;
            uint overflow;
            colorMap.GetDimensions(width, height, levels, extra, overflow);
        }
    """).strip()

    crossgl = generate_crossgl(code)

    assert ".GetDimensions(" not in crossgl
    assert (
        "/* unsupported DirectX GetDimensions overload for Texture2D: "
        "preserved dimension helper call */ "
        "texture_dimensions(colorMap, width, height, levels, extra, overflow);"
    ) in crossgl


def test_codegen_texture_lod_query_member_imports_to_texture_query_lod():
    code = textwrap.dedent("""
        Texture2D<float4> colorMap : register(t0);
        SamplerState linearSampler : register(s0);

        float4 main(float2 uv : TEXCOORD0) : SV_Target0 {
            float clamped = colorMap.CalculateLevelOfDetail(linearSampler, uv);
            float unclamped = colorMap.CalculateLevelOfDetailUnclamped(
                linearSampler, uv
            );
            return float4(clamped, unclamped, 0.0, 1.0);
        }
    """).strip()

    crossgl = generate_crossgl(code)

    assert ".CalculateLevelOfDetail(" not in crossgl
    assert ".CalculateLevelOfDetailUnclamped(" not in crossgl
    assert "clamped = textureQueryLod(colorMap, linearSampler, uv).x;" in crossgl
    assert "unclamped = textureQueryLod(colorMap, linearSampler, uv).y;" in crossgl

    regenerated_hlsl = TranslatorHLSLCodeGen().generate(parse_crossgl(crossgl))
    assert (
        "float clamped = float2(colorMap.CalculateLevelOfDetail("
        "linearSampler, uv), colorMap.CalculateLevelOfDetailUnclamped(linearSampler, uv)).x;"
        in regenerated_hlsl
    )
    assert (
        "float unclamped = float2(colorMap.CalculateLevelOfDetail("
        "linearSampler, uv), colorMap.CalculateLevelOfDetailUnclamped(linearSampler, uv)).y;"
        in regenerated_hlsl
    )
    assert "textureQueryLod(" not in regenerated_hlsl


def test_codegen_texture_lod_query_invalid_imports_diagnostic():
    code = textwrap.dedent("""
        Texture2D<float4> colorMap : register(t0);
        Texture2DMS<float4> msMap : register(t1);
        SamplerState linearSampler : register(s0);

        float4 main(float2 uv : TEXCOORD0) : SV_Target0 {
            float missingCoord = colorMap.CalculateLevelOfDetail(linearSampler);
            float msLod = msMap.CalculateLevelOfDetail(linearSampler, uv);
            return float4(missingCoord + msLod);
        }
    """).strip()

    crossgl = generate_crossgl(code)

    assert ".CalculateLevelOfDetail(" not in crossgl
    assert (
        "/* unsupported DirectX texture LOD query for Texture2D: "
        "expected sampler and coordinate arguments */ 0.0"
    ) in crossgl
    assert (
        "/* unsupported DirectX texture LOD query for Texture2DMS: "
        "CalculateLevelOfDetail is unavailable for multisample textures */ 0.0"
    ) in crossgl
    assert "textureQueryLod(msMap" not in crossgl


def test_codegen_texture_sample_position_member_imports_to_helper():
    code = textwrap.dedent("""
        Texture2DMS<float4> msMap : register(t0);
        Texture2DMSArray<float4> msArray : register(t1);

        float4 main(uint sampleIndex : SV_SampleIndex) : SV_Target0 {
            float2 pos = msMap.GetSamplePosition(sampleIndex);
            float2 arrayPos = msArray.GetSamplePosition(sampleIndex);
            return float4(pos + arrayPos, 0.0, 1.0);
        }
    """).strip()

    crossgl = generate_crossgl(code)

    assert ".GetSamplePosition(" not in crossgl
    assert "pos = textureSamplePosition(msMap, sampleIndex);" in crossgl
    assert "arrayPos = textureSamplePosition(msArray, sampleIndex);" in crossgl

    regenerated_hlsl = TranslatorHLSLCodeGen().generate(parse_crossgl(crossgl))
    assert "float2 pos = msMap.GetSamplePosition(sampleIndex);" in regenerated_hlsl
    assert (
        "float2 arrayPos = msArray.GetSamplePosition(sampleIndex);" in regenerated_hlsl
    )
    assert "textureSamplePosition(" not in regenerated_hlsl


def test_codegen_texture_sample_position_rejects_non_integer_sample_index_roundtrip():
    code = textwrap.dedent("""
        Texture2DMS<float4> msMap : register(t0);

        float4 main(float sampleIndex : TEXCOORD0) : SV_Target0 {
            float2 pos = msMap.GetSamplePosition(sampleIndex);
            return float4(pos, 0.0, 1.0);
        }
    """).strip()

    crossgl = generate_crossgl(code)

    assert "textureSamplePosition(msMap, sampleIndex)" in crossgl
    with pytest.raises(
        ValueError,
        match=(
            "DirectX texture sample-position query operation "
            "'textureSamplePosition' requires a scalar integer sample index "
            "argument: sampleIndex has type float"
        ),
    ):
        TranslatorHLSLCodeGen().generate(parse_crossgl(crossgl))


def test_codegen_struct_member_resource_methods_roundtrip_to_hlsl_members():
    code = textwrap.dedent("""
        struct ResourceBundle {
            Texture2D<float4> color;
            Texture2DMS<float4> ms;
            RWTexture2D<float4> image;
        };

        SamplerState linearSampler : register(s0);
        ResourceBundle resources;

        float4 main(float2 uv : TEXCOORD0, int2 pixel : TEXCOORD1, uint sampleIndex : SV_SampleIndex) : SV_Target0 {
            uint width;
            uint height;
            uint levels;
            resources.color.GetDimensions(width, height, levels);
            float lod = resources.color.CalculateLevelOfDetail(linearSampler, uv);
            float2 pos = resources.ms.GetSamplePosition(sampleIndex);
            float4 loaded = resources.image.Load(pixel);
            return resources.color.Sample(linearSampler, uv) + loaded + float4(lod + pos.x + width + height + levels);
        }
    """).strip()

    crossgl = generate_crossgl(code)

    assert ".GetDimensions(" not in crossgl
    assert ".CalculateLevelOfDetail(" not in crossgl
    assert ".GetSamplePosition(" not in crossgl
    assert ".Load(" not in crossgl
    assert "width = uint(textureSize(resources.color, 0).x);" in crossgl
    assert "height = uint(textureSize(resources.color, 0).y);" in crossgl
    assert "levels = uint(textureQueryLevels(resources.color));" in crossgl
    assert "lod = textureQueryLod(resources.color, linearSampler, uv).x;" in crossgl
    assert "pos = textureSamplePosition(resources.ms, sampleIndex);" in crossgl
    assert "loaded = imageLoad(resources.image, pixel);" in crossgl
    assert "texture(resources.color, linearSampler, uv)" in crossgl

    regenerated_hlsl = TranslatorHLSLCodeGen().generate(parse_crossgl(crossgl))
    assert "int2 textureSize(Texture2D tex, int lod)" in regenerated_hlsl
    assert "int textureQueryLevels(Texture2D tex)" in regenerated_hlsl
    assert "width = uint(textureSize(resources.color, 0).x);" in regenerated_hlsl
    assert "height = uint(textureSize(resources.color, 0).y);" in regenerated_hlsl
    assert "levels = uint(textureQueryLevels(resources.color));" in regenerated_hlsl
    assert (
        "float lod = float2(resources.color.CalculateLevelOfDetail("
        "linearSampler, uv), resources.color.CalculateLevelOfDetailUnclamped(linearSampler, uv)).x;"
        in regenerated_hlsl
    )
    assert (
        "float2 pos = resources.ms.GetSamplePosition(sampleIndex);" in regenerated_hlsl
    )
    assert "float4 loaded = resources.image[pixel];" in regenerated_hlsl
    assert "resources.color.Sample(linearSampler, uv)" in regenerated_hlsl
    assert "textureSamplePosition(" not in regenerated_hlsl
    assert "unsupported DirectX texture sample-position query" not in regenerated_hlsl


def test_codegen_indexed_struct_member_resource_methods_roundtrip_to_hlsl_members():
    code = textwrap.dedent("""
        struct ResourceBundle {
            Texture2D<float4> color;
            Texture2DMS<float4> ms;
            RWTexture2D<float4> image;
        };

        SamplerState linearSampler : register(s0);
        ResourceBundle resources[2];

        float4 main(float2 uv : TEXCOORD0, int2 pixel : TEXCOORD1, uint layer : TEXCOORD2, uint sampleIndex : SV_SampleIndex) : SV_Target0 {
            uint width;
            uint height;
            uint levels;
            resources[layer].color.GetDimensions(width, height, levels);
            float lod = resources[layer].color.CalculateLevelOfDetail(linearSampler, uv);
            float2 pos = resources[layer].ms.GetSamplePosition(sampleIndex);
            float4 loaded = resources[layer].image.Load(pixel);
            return resources[layer].color.Sample(linearSampler, uv) + loaded + float4(lod + pos.x + width + height + levels);
        }
    """).strip()

    crossgl = generate_crossgl(code)

    assert ".GetDimensions(" not in crossgl
    assert ".CalculateLevelOfDetail(" not in crossgl
    assert ".GetSamplePosition(" not in crossgl
    assert ".Load(" not in crossgl
    assert "width = uint(textureSize(resources[layer].color, 0).x);" in crossgl
    assert "height = uint(textureSize(resources[layer].color, 0).y);" in crossgl
    assert "levels = uint(textureQueryLevels(resources[layer].color));" in crossgl
    assert (
        "lod = textureQueryLod(resources[layer].color, linearSampler, uv).x;" in crossgl
    )
    assert "pos = textureSamplePosition(resources[layer].ms, sampleIndex);" in crossgl
    assert "loaded = imageLoad(resources[layer].image, pixel);" in crossgl
    assert "texture(resources[layer].color, linearSampler, uv)" in crossgl

    regenerated_hlsl = TranslatorHLSLCodeGen().generate(parse_crossgl(crossgl))
    assert "ResourceBundle resources[2];" in regenerated_hlsl
    assert "int2 textureSize(Texture2D tex, int lod)" in regenerated_hlsl
    assert "int textureQueryLevels(Texture2D tex)" in regenerated_hlsl
    assert "width = uint(textureSize(resources[layer].color, 0).x);" in regenerated_hlsl
    assert (
        "height = uint(textureSize(resources[layer].color, 0).y);" in regenerated_hlsl
    )
    assert (
        "levels = uint(textureQueryLevels(resources[layer].color));" in regenerated_hlsl
    )
    assert (
        "float lod = float2(resources[layer].color.CalculateLevelOfDetail("
        "linearSampler, uv), resources[layer].color.CalculateLevelOfDetailUnclamped(linearSampler, uv)).x;"
        in regenerated_hlsl
    )
    assert (
        "float2 pos = resources[layer].ms.GetSamplePosition(sampleIndex);"
        in regenerated_hlsl
    )
    assert "float4 loaded = resources[layer].image[pixel];" in regenerated_hlsl
    assert "resources[layer].color.Sample(linearSampler, uv)" in regenerated_hlsl
    assert "textureSamplePosition(" not in regenerated_hlsl
    assert "unsupported DirectX texture sample-position query" not in regenerated_hlsl


def test_codegen_texture_sample_position_invalid_imports_diagnostic():
    code = textwrap.dedent("""
        Texture2D<float4> colorMap : register(t0);
        Texture2DMS<float4> msMap : register(t1);
        RWTexture2DMS<float4> msImage : register(u0);

        float4 main(uint sampleIndex : SV_SampleIndex) : SV_Target0 {
            float2 missingIndex = msMap.GetSamplePosition();
            float2 nonMultisample = colorMap.GetSamplePosition(sampleIndex);
            float2 storageImage = msImage.GetSamplePosition(sampleIndex);
            return float4(missingIndex + nonMultisample + storageImage, 0.0, 1.0);
        }
    """).strip()

    crossgl = generate_crossgl(code)

    assert ".GetSamplePosition(" not in crossgl
    assert (
        "/* unsupported DirectX texture sample-position query for Texture2DMS: "
        "expected sample-index argument */ vec2(0.0, 0.0)"
    ) in crossgl
    assert (
        "/* unsupported DirectX texture sample-position query for Texture2D: "
        "GetSamplePosition is only available on sampled multisample textures */ "
        "vec2(0.0, 0.0)"
    ) in crossgl
    assert (
        "/* unsupported DirectX texture sample-position query for RWTexture2DMS: "
        "GetSamplePosition is only available on sampled multisample textures */ "
        "vec2(0.0, 0.0)"
    ) in crossgl
    assert "textureSamplePosition(colorMap" not in crossgl
    assert "textureSamplePosition(msImage" not in crossgl


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


def test_codegen_byte_address_interlocked_compare_store_roundtrip():
    code = textwrap.dedent("""
        RWByteAddressBuffer rawBytes : register(u4);

        [numthreads(1, 1, 1)]
        void CSMain(uint3 dtid : SV_DispatchThreadID) {
            uint offset = dtid.x * 4u;
            uint compare = 3u;
            rawBytes.InterlockedCompareStore(offset, compare, 7u);
        }
    """).strip()

    crossgl = generate_crossgl(code)

    assert "RWByteAddressBuffer rawBytes;" in crossgl
    assert "@ register(u4)" in crossgl
    assert "rawBytes.InterlockedCompareStore(offset, compare, 7);" in crossgl
    assert "atomicCompareExchange(rawBytes" not in crossgl

    regenerated_hlsl = TranslatorHLSLCodeGen().generate(parse_crossgl(crossgl))
    assert "RWByteAddressBuffer rawBytes : register(u4);" in regenerated_hlsl
    assert "rawBytes.InterlockedCompareStore(offset, compare, 7);" in regenerated_hlsl
    assert "InterlockedCompareExchange(rawBytes" not in regenerated_hlsl


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


def test_codegen_template_specialized_struct_declarations_reparse_crossgl():
    # Source: microsoft/DirectXShaderCompiler@517dd5eb5d8cbb46c15fc1230acac1d2f4779092
    # tools/clang/test/SemaHLSL/template-implicit-this-sfinae.hlsl
    code = textwrap.dedent("""
        template <typename T>
        struct TemplateValue {
            T value;
        };

        template <> struct TemplateValue<float> {
            float value;
        };

        TemplateValue<float> selected;

        float main() : SV_Target0 {
            return selected.value;
        }
    """).strip()

    output = generate_crossgl(code)

    assert output.count("struct TemplateValue") == 2
    assert "TemplateValue selected;" in output
    assert "TemplateValue<float>" not in output
    parse_crossgl(output)


def test_codegen_upstream_declarations_defaults_and_for_update_sequences():
    code = textwrap.dedent("""
        cbuffer CSConstants : register(b0) {
            uint ViewportWidth, ViewportHeight;
        };

        float4 ToRGBM(float3 rgb, float PeakValue = 255.0 / 16.0) {
            return float4(rgb, PeakValue);
        }

        float main(uint count) : SV_Target0 {
            uint tileLightLoadOffset = 0;
            float sum = 0.0;
            [unroll]
            for (uint n = 0; n < count; n++, tileLightLoadOffset += 4) {
                sum += n;
            }
            return sum;
        }
    """).strip()

    output = generate_crossgl(code)

    assert "uint ViewportWidth;" in output
    assert "uint ViewportHeight;" in output
    assert "vec4 ToRGBM(vec3 rgb, float PeakValue)" in output
    assert "255.0 / 16.0" not in output
    assert "for (uint n = 0; n < count; n++, tileLightLoadOffset += 4)" in output


def test_codegen_unnamed_hlsl_parameters_get_synthetic_crossgl_names():
    code = textwrap.dedent("""
        float4 Rand4(inout uint4 ctx, uint) {
            return float4(ctx);
        }
    """).strip()

    output = generate_crossgl(code)

    assert "uint _param1" in output
    parse_crossgl(output)


def test_codegen_linalg_post_type_attributes_from_dxc_reparse():
    # Source: microsoft/DirectXShaderCompiler@517dd5eb5d8cbb46c15fc1230acac1d2f4779092
    # tools/clang/test/CodeGenDXIL/hlsl/linalg/attr-matrix-type.hlsl
    code = textwrap.dedent("""
        typedef __builtin_LinAlgMatrix [[__LinAlgMatrix_Attributes(ComponentType::F32, 10, 20, MatrixUse::A, MatrixScope::Thread)]] Mat10by20;

        void f2(__builtin_LinAlgMatrix [[__LinAlgMatrix_Attributes(ComponentType::I32, 4, 5, MatrixUse::B, MatrixScope::ThreadGroup)]] mat2) {
            __builtin_LinAlgMatrix [[__LinAlgMatrix_Attributes(ComponentType::I16, 2, 3, MatrixUse::Accumulator, MatrixScope::ThreadGroup)]] mat1;
        }

        Mat10by20 f3() {
            Mat10by20 mat3;
            return mat3;
        }
        """).strip()

    output = generate_crossgl(code)

    assert "type Mat10by20 = __builtin_LinAlgMatrix;" in output
    assert "void f2(__builtin_LinAlgMatrix mat2_)" in output
    assert "__builtin_LinAlgMatrix mat1;" in output
    assert "Mat10by20 f3()" in output
    parse_crossgl(output)


def test_codegen_invalid_hlsl_raises():
    code = "float4 main() : SV_Target0 { float x = 1.0 return float4(x, 0, 0, 1); }"
    with pytest.raises(SyntaxError):
        generate_crossgl(code)


if __name__ == "__main__":
    pytest.main()

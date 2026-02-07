import textwrap

import pytest

from crosstl.backend.DirectX import DirectxCrossGLCodeGen
from crosstl.backend.DirectX.DirectxLexer import HLSLLexer
from crosstl.backend.DirectX.DirectxParser import HLSLParser
from crosstl.translator.ast import ShaderStage
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
    assert "texture_sample" in output


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


def test_codegen_ray_shader_stages():
    output = generate_crossgl(RAY_STAGES_HLSL)
    lowered = output.lower()
    assert "ray_intersection" in lowered
    assert "ray_closest_hit" in lowered
    assert "ray_any_hit" in lowered
    assert "ray_miss" in lowered
    assert "ray_callable" in lowered


def test_codegen_texture_method_mappings():
    output = generate_crossgl(TEXTURE_METHODS_HLSL)
    assert "texture_sample_level" in output
    assert "texture_sample_cmp" in output
    assert "texture_gather" in output


def test_codegen_ms_texture_load_mapping():
    output = generate_crossgl(MS_TEXTURE_HLSL)
    assert "texture_load" in output


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
    assert "atomicCompareExchange" in output


def test_codegen_invalid_hlsl_raises():
    code = "float4 main() : SV_Target0 { float x = 1.0 return float4(x, 0, 0, 1); }"
    with pytest.raises(SyntaxError):
        generate_crossgl(code)


if __name__ == "__main__":
    pytest.main()

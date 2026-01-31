import textwrap

import pytest

from crosstl.backend.DirectX.DirectxLexer import HLSLLexer
from crosstl.backend.DirectX.DirectxParser import HLSLParser


VERTEX_PIXEL_HLSL = textwrap.dedent(
    """
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
    """
).strip()

CONTROL_FLOW_HLSL = textwrap.dedent(
    """
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
    """
).strip()

ARRAYS_HLSL = textwrap.dedent(
    """
    float4 colors[4];
    float3 grid[2][3];

    float sumArray(float4 values[4]) {
        float total = 0.0;
        for (int i = 0; i < 4; i++) {
            total += values[i].x;
        }
        return total;
    }

    float useGrid() {
        return grid[1][2].x;
    }
    """
).strip()

RESOURCES_HLSL = textwrap.dedent(
    """
    Texture2D<float4> tex1 : register(t1);
    SamplerComparisonState sampComp : register(s1);
    RWTexture2D<float4> outputTex : register(u0);
    StructuredBuffer<float4> data : register(t2);
    RWStructuredBuffer<float4> outData : register(u1);

    float4 SampleTex(float2 uv) : SV_Target0 {
        float4 value = tex1.SampleCmpLevelZero(sampComp, uv, 0.5);
        outData[0] = value;
        return value;
    }
    """
).strip()

OVERLOADS_HLSL = textwrap.dedent(
    """
    float4 Blend(float4 a, float4 b) {
        return a + b;
    }

    float4 Blend(float4 a, float4 b, float t) {
        return lerp(a, b, t);
    }

    float4 UseBlend(float4 a, float4 b) {
        return Blend(a, b, 0.5);
    }
    """
).strip()

COMPUTE_HLSL = textwrap.dedent(
    """
    RWTexture2D<float4> outputTex : register(u0);

    [numthreads(8, 8, 1)]
    void CSMain(uint3 dtid : SV_DispatchThreadID) {
        outputTex[dtid.xy] = float4(1.0, 0.0, 0.0, 1.0);
    }
    """
).strip()

PREPROCESSOR_HLSL = textwrap.dedent(
    """
    #define USE_LIGHTING 1
    #if USE_LIGHTING
    float3 Lighting(float3 n) { return n; }
    #endif

    float4 main() : SV_Target0 {
        return float4(1.0, 1.0, 1.0, 1.0);
    }
    """
).strip()


def tokenize_code(code: str):
    lexer = HLSLLexer(code)
    return lexer.tokenize()


def parse_code(code: str):
    tokens = tokenize_code(code)
    parser = HLSLParser(tokens)
    return parser.parse()


def assert_parses(code: str):
    try:
        parse_code(code)
    except SyntaxError as exc:
        pytest.fail(f"Expected code to parse, but got SyntaxError: {exc}")


def assert_parse_error(code: str):
    with pytest.raises(SyntaxError):
        parse_code(code)


def test_parse_vertex_pixel_shader():
    assert_parses(VERTEX_PIXEL_HLSL)


def test_parse_control_flow_and_operators():
    assert_parses(CONTROL_FLOW_HLSL)


def test_parse_arrays_and_indexing():
    assert_parses(ARRAYS_HLSL)


def test_parse_resources_and_bindings():
    assert_parses(RESOURCES_HLSL)


def test_parse_function_overloads_and_calls():
    assert_parses(OVERLOADS_HLSL)


def test_parse_compute_attributes_and_semantics():
    assert_parses(COMPUTE_HLSL)


def test_parse_preprocessor_directives():
    assert_parses(PREPROCESSOR_HLSL)


def test_preprocessor_evaluates_conditionals():
    code = """
    #define ENABLE_BAD 0
    #if ENABLE_BAD
    int broken = ;
    #endif

    int ok() { return 1; }
    """
    assert_parses(code)


def test_parse_enum_and_typedef():
    code = """
    enum BlendMode {
        BlendOpaque = 0,
        BlendAdd = 1,
    };
    typedef float4 Color;
    Color main(Color input) : SV_Target0 {
        return input;
    }
    """
    assert_parses(code)


def test_parse_resource_arrays_and_register_space():
    code = """
    Texture2D textures[4] : register(t0, space1);
    RWTexture1DArray rwTexArray[2] : register(u1, space2);
    SamplerState samplers[4] : register(s0);
    float4 main(float2 uv : TEXCOORD0) : SV_Target0 {
        return textures[0].Sample(samplers[0], uv);
    }
    """
    assert_parses(code)


def test_parse_geometry_shader():
    code = """
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
        triStream.RestartStrip();
    }
    """
    assert_parses(code)


def test_parse_tessellation_shaders():
    code = """
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
    """
    assert_parses(code)


def test_parse_mesh_task_shaders():
    code = """
    [shader(\"amplification\")]
    void ASMain() {
        DispatchMesh(1, 1, 1);
    }

    [shader(\"mesh\")]
    void MSMain() {
        SetMeshOutputCounts(1, 1);
    }
    """
    assert_parses(code)


def test_parse_raytracing_shader():
    code = """
    RaytracingAccelerationStructure accel : register(t0, space1);

    [shader(\"raygeneration\")]
    void RayGen() {
        TraceRay(accel, 0, 0xFF, 0, 1, 0, float3(0.0, 0.0, 0.0), 0.0, float3(0.0, 0.0, 1.0), 100.0, 0);
    }
    """
    assert_parses(code)


def test_parse_raytracing_shader_stages():
    code = """
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
    """
    assert_parses(code)


def test_parse_additional_attributes():
    code = """
    [earlydepthstencil]
    float4 PSMain(float4 pos : SV_Position) : SV_Target0 { return pos; }

    [unroll]
    void CSMain() { }

    [branch]
    float4 BRMain(float4 pos : SV_Position) : SV_Target0 { return pos; }

    [loop]
    void LoopMain() { }

    [flatten]
    float4 FLMain(float4 pos : SV_Position) : SV_Target0 { return pos; }

    [maxtessfactor(16)]
    [instance(2)]
    [fastopt]
    [allow_uav_condition]
    void AttrMain() { }
    """
    assert_parses(code)


def test_parse_wave_intrinsics():
    code = """
    uint WaveMain(uint value) {
        uint sum = WaveActiveSum(value);
        uint lane = WaveGetLaneIndex();
        return sum + lane;
    }
    """
    assert_parses(code)


def test_parse_texture_methods():
    code = """
    Texture2D tex : register(t0);
    SamplerState samp : register(s0);
    float4 main(float2 uv : TEXCOORD0) : SV_Target0 {
        float4 a = tex.Sample(samp, uv);
        float4 b = tex.SampleLevel(samp, uv, 0.0);
        float4 c = tex.SampleGrad(samp, uv, float2(1.0, 0.0), float2(0.0, 1.0));
        float4 d = tex.SampleBias(samp, uv, 0.5);
        float4 e = tex.SampleCmpLevelZero(samp, uv, 0.5);
        float4 f = tex.GatherRed(samp, uv);
        return a + b + c + d + e + f;
    }
    """
    assert_parses(code)


@pytest.mark.parametrize(
    "code",
    [
        "float4 main() : SV_Target0 { float x = 1.0 return float4(x, 0, 0, 1); }",
        "struct Foo { float4 a; ",
        "void main() { for (int i = 0; i < 4 i++) { } }",
    ],
)
def test_parse_invalid_syntax(code):
    assert_parse_error(code)


if __name__ == "__main__":
    pytest.main()

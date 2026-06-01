import textwrap

import pytest

from crosstl.backend.common_ast import (
    FunctionCallNode,
    MemberAccessNode,
    TextureSampleNode,
)
from crosstl.backend.DirectX.DirectxLexer import HLSLLexer
from crosstl.backend.DirectX.DirectxParser import HLSLParser

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

CONTROL_FLOW_HLSL = textwrap.dedent("""
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
    """).strip()

ARRAYS_HLSL = textwrap.dedent("""
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
    """).strip()

RESOURCES_HLSL = textwrap.dedent("""
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
    """).strip()

OVERLOADS_HLSL = textwrap.dedent("""
    float4 Blend(float4 a, float4 b) {
        return a + b;
    }

    float4 Blend(float4 a, float4 b, float t) {
        return lerp(a, b, t);
    }

    float4 UseBlend(float4 a, float4 b) {
        return Blend(a, b, 0.5);
    }
    """).strip()

COMPUTE_HLSL = textwrap.dedent("""
    RWTexture2D<float4> outputTex : register(u0);

    [numthreads(8, 8, 1)]
    void CSMain(uint3 dtid : SV_DispatchThreadID) {
        outputTex[dtid.xy] = float4(1.0, 0.0, 0.0, 1.0);
    }
    """).strip()

PREPROCESSOR_HLSL = textwrap.dedent("""
    #define USE_LIGHTING 1
    #if USE_LIGHTING
    float3 Lighting(float3 n) { return n; }
    #endif

    float4 main() : SV_Target0 {
        return float4(1.0, 1.0, 1.0, 1.0);
    }
    """).strip()


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


def iter_ast_nodes(node):
    if node is None or isinstance(node, (str, int, float, bool)):
        return
    if isinstance(node, dict):
        for value in node.values():
            yield from iter_ast_nodes(value)
        return
    if isinstance(node, (list, tuple, set)):
        for value in node:
            yield from iter_ast_nodes(value)
        return
    yield node
    for value in getattr(node, "__dict__", {}).values():
        yield from iter_ast_nodes(value)


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


def test_parse_rootsignature_macro_adjacent_string_literals():
    code = r"""
    #define RootSig \
        "RootFlags(ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT|ALLOW_STREAM_OUTPUT)," \
        "DescriptorTable(SRV(t0, numDescriptors=1))," \
        "DescriptorTable(UAV(u0, numDescriptors=2))," \
        "DescriptorTable(Sampler(s0, numDescriptors=2))"

    [RootSignature(RootSig)]
    float4 RootSignaturePS(float4 pos : SV_POSITION) : SV_TARGET {
        return pos;
    }
    """

    ast = parse_code(code)
    attributes = ast.functions[0].attributes

    assert attributes[0].name == "RootSignature"
    assert attributes[0].args == [
        '"RootFlags(ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT|ALLOW_STREAM_OUTPUT),'
        "DescriptorTable(SRV(t0, numDescriptors=1)),"
        "DescriptorTable(UAV(u0, numDescriptors=2)),"
        'DescriptorTable(Sampler(s0, numDescriptors=2))"'
    ]


def test_parse_interpolation_intrinsics_keep_free_function_calls():
    code = """
    float4 main(float4 color : COLOR0, uint sampleIndex : SV_SampleIndex) : SV_Target0 {
        int2 snappedOffset = int2(1, -1);
        float4 atSample = EvaluateAttributeAtSample(color, sampleIndex);
        float4 atOffset = EvaluateAttributeSnapped(color, snappedOffset);
        float4 atCentroid = EvaluateAttributeCentroid(color);
        return atSample + atOffset + atCentroid;
    }
    """

    ast = parse_code(code)
    free_calls = [
        node.name
        for node in iter_ast_nodes(ast)
        if isinstance(node, FunctionCallNode) and isinstance(node.name, str)
    ]

    assert "EvaluateAttributeAtSample" in free_calls
    assert "EvaluateAttributeSnapped" in free_calls
    assert "EvaluateAttributeCentroid" in free_calls


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


def test_parse_rasterizer_ordered_resources_and_register_space():
    code = """
    RasterizerOrderedTexture2D<uint> counters : register(u0, space1);
    RasterizerOrderedTexture2DArray<float4> layers[2] : register(u1, space2);
    RasterizerOrderedBuffer<uint> bins : register(u3);
    RasterizerOrderedStructuredBuffer<int> values : register(u4);
    RasterizerOrderedByteAddressBuffer bytes : register(u5, space3);

    float4 PSMain(uint2 pixel : TEXCOORD0, uint layer : TEXCOORD1) : SV_Target0 {
        uint oldCount;
        InterlockedAdd(counters[pixel], 1u, oldCount);
        values[0] = int(oldCount);
        bytes.Store(0, oldCount);
        return layers[0][uint3(pixel, layer)];
    }
    """

    ast = parse_code(code)
    globals_by_name = {node.name: node for node in ast.global_variables}

    assert globals_by_name["counters"].vtype == "RasterizerOrderedTexture2D<uint>"
    assert globals_by_name["counters"].register == "u0, space1"
    assert globals_by_name["layers"].vtype == "RasterizerOrderedTexture2DArray<float4>"
    assert globals_by_name["layers"].array_sizes == [2]
    assert globals_by_name["layers"].register == "u1, space2"
    assert globals_by_name["bins"].vtype == "RasterizerOrderedBuffer<uint>"
    assert globals_by_name["values"].vtype == "RasterizerOrderedStructuredBuffer<int>"
    assert globals_by_name["bytes"].vtype == "RasterizerOrderedByteAddressBuffer"
    assert globals_by_name["bytes"].register == "u5, space3"


def test_parse_min_precision_vector_and_matrix_types():
    code = """
    struct MinPrecisionData {
        min16float3 hdr : COLOR0;
        min10float2 uv : TEXCOORD0;
        min16float2x3 colorMatrix;
    };

    min16float3 Shade(
        min16float3 color,
        min12int2 offset,
        min16uint4 mask
    ) {
        min10float2 localUv = min10float2(0.0, 1.0);
        return min16float3(color.x, localUv.x, localUv.y);
    }
    """

    ast = parse_code(code)
    struct = ast.structs[0]
    func = ast.functions[0]

    assert [member.vtype for member in struct.members] == [
        "min16float3",
        "min10float2",
        "min16float2x3",
    ]
    assert func.return_type == "min16float3"
    assert [param.vtype for param in func.params] == [
        "min16float3",
        "min12int2",
        "min16uint4",
    ]
    local_uv = next(
        node
        for node in iter_ast_nodes(func)
        if getattr(node, "name", None) == "localUv"
    )
    assert local_uv.vtype == "min10float2"


def test_parse_cbuffer_preserves_buffer_and_member_bindings():
    code = """
    cbuffer FrameData : register(b0, space1) {
        row_major float4x4 viewProj : packoffset(c0);
        float4 tint : packoffset(c4);
    };
    """

    ast = parse_code(code)
    cbuffer = ast.cbuffers[0]

    assert cbuffer.register == "b0, space1"
    assert cbuffer.packoffset is None
    assert cbuffer.members[0].name == "viewProj"
    assert cbuffer.members[0].packoffset == "c0"
    assert cbuffer.members[0].register is None
    assert cbuffer.members[1].name == "tint"
    assert cbuffer.members[1].packoffset == "c4"
    assert cbuffer.members[1].register is None


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
        bool predicate = value != 0u;
        uint lane = WaveGetLaneIndex();
        uint laneCount = WaveGetLaneCount();
        bool first = WaveIsFirstLane();
        uint sum = WaveActiveSum(value);
        uint product = WaveActiveProduct(value);
        uint andValue = WaveActiveBitAnd(value);
        uint orValue = WaveActiveBitOr(value);
        uint xorValue = WaveActiveBitXor(value);
        bool allTrue = WaveActiveAllTrue(predicate);
        bool anyTrue = WaveActiveAnyTrue(predicate);
        uint4 ballot = WaveActiveBallot(predicate);
        uint laneValue = WaveReadLaneAt(value, 0u);
        uint firstValue = WaveReadLaneFirst(value);
        uint prefixSum = WavePrefixSum(value);
        uint prefixProduct = WavePrefixProduct(value);
        uint4 matchMask = WaveMatch(value);
        uint multiSum = WaveMultiPrefixSum(value, ballot);
        uint quadX = QuadReadAcrossX(value);
        uint quadY = QuadReadAcrossY(value);
        uint quadDiagonal = QuadReadAcrossDiagonal(value);
        uint quadLane = QuadReadLaneAt(value, 2u);
        return lane + laneCount + sum + product + andValue + orValue + xorValue
            + laneValue + firstValue + prefixSum + prefixProduct + matchMask.x
            + multiSum + quadX + quadY + quadDiagonal + quadLane + ballot.x
            + (first ? 1u : 0u) + (allTrue ? 1u : 0u) + (anyTrue ? 1u : 0u);
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


def test_parse_texture_sample_offset_methods_keep_member_calls():
    code = """
    Texture2D tex : register(t0);
    SamplerState samp : register(s0);

    float4 main(
        float2 uv : TEXCOORD0,
        float lod : TEXCOORD1,
        float2 ddx : TEXCOORD2,
        float2 ddy : TEXCOORD3,
        int2 offset : TEXCOORD4
    ) : SV_Target0 {
        float4 plain = tex.Sample(samp, uv, offset);
        float4 mip = tex.SampleLevel(samp, uv, lod, offset);
        float4 grad = tex.SampleGrad(samp, uv, ddx, ddy, offset);
        return plain + mip + grad;
    }
    """

    ast = parse_code(code)
    nodes = list(iter_ast_nodes(ast))

    assert not [node for node in nodes if isinstance(node, TextureSampleNode)]
    members = [
        node.name.member
        for node in nodes
        if isinstance(node, FunctionCallNode)
        and isinstance(node.name, MemberAccessNode)
    ]
    assert {"Sample", "SampleLevel", "SampleGrad"}.issubset(set(members))


def test_parse_texture_compare_and_gather_offset_methods_keep_member_calls():
    code = """
    Texture2D colorMap : register(t0);
    Texture2D<float> shadowMap : register(t1);
    SamplerState linearSampler : register(s0);
    SamplerComparisonState compareSampler : register(s1);

    float4 main(
        float2 uv : TEXCOORD0,
        float depth : TEXCOORD1,
        int2 offset : TEXCOORD2
    ) : SV_Target0 {
        float cmp = shadowMap.SampleCmp(compareSampler, uv, depth, offset);
        float cmpZero = shadowMap.SampleCmpLevelZero(
            compareSampler, uv, depth, offset
        );
        float4 gather = colorMap.GatherRed(linearSampler, uv, offset);
        float4 gatherAny = colorMap.Gather(linearSampler, uv, offset);
        float4 gatherCmp = shadowMap.GatherCmp(
            compareSampler, uv, depth, offset
        );
        float4 gatherCmpRed = shadowMap.GatherCmpRed(
            compareSampler, uv, depth, offset
        );
        return gather + gatherAny + gatherCmp + gatherCmpRed + float4(cmp + cmpZero);
    }
    """

    ast = parse_code(code)
    nodes = list(iter_ast_nodes(ast))

    members = [
        node.name.member
        for node in nodes
        if isinstance(node, FunctionCallNode)
        and isinstance(node.name, MemberAccessNode)
    ]
    assert {
        "SampleCmp",
        "SampleCmpLevelZero",
        "GatherRed",
        "Gather",
        "GatherCmp",
        "GatherCmpRed",
    }.issubset(set(members))


def test_parse_texture_status_and_clamp_overloads_keep_member_calls():
    code = """
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
    ) : SV_Target0 {
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
        float4 gatherRed = colorMap.GatherRed(linearSampler, uv, offset, status);
        float4 gatherOffsets = colorMap.GatherRed(
            linearSampler, uv, offset, offset, offset, offset, status
        );
        float4 gatherCmp = shadowMap.GatherCmp(
            compareSampler, uv, depth, offset, status
        );
        float4 gatherCmpGreen = shadowMap.GatherCmpGreen(
            compareSampler, uv, depth, offset
        );
        return (
            plain + biased + mip + grad + gather + gatherRed + gatherOffsets
            + gatherCmp + gatherCmpGreen + float4(cmp + cmpZero)
        );
    }
    """

    ast = parse_code(code)
    nodes = list(iter_ast_nodes(ast))

    assert not [node for node in nodes if isinstance(node, TextureSampleNode)]
    members = [
        node.name.member
        for node in nodes
        if isinstance(node, FunctionCallNode)
        and isinstance(node.name, MemberAccessNode)
    ]
    assert {
        "Sample",
        "SampleBias",
        "SampleLevel",
        "SampleGrad",
        "SampleCmp",
        "SampleCmpLevelZero",
        "Gather",
        "GatherRed",
        "GatherCmp",
        "GatherCmpGreen",
    }.issubset(set(members))


def test_parse_tiled_resource_status_loads_and_checks_keep_calls():
    code = """
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
    """

    ast = parse_code(code)
    nodes = list(iter_ast_nodes(ast))

    member_calls = [
        node.name.member
        for node in nodes
        if isinstance(node, FunctionCallNode)
        and isinstance(node.name, MemberAccessNode)
    ]
    free_calls = [
        node.name
        for node in nodes
        if isinstance(node, FunctionCallNode) and isinstance(node.name, str)
    ]
    assert member_calls.count("Load") == 3
    assert "CheckAccessFullyMapped" in free_calls


def test_parse_get_dimensions_overloads_keep_member_calls():
    code = """
    Texture2D<float4> colorMap : register(t0);
    Texture2DArray<float4> layerMap : register(t1);
    Texture2DMS<float4> msMap : register(t2);
    RWTexture3D<float4> volume : register(u0);
    StructuredBuffer<float4> structs : register(t3);

    void main(uint lod : TEXCOORD0) {
        uint width;
        uint height;
        uint depth;
        uint elements;
        uint levels;
        uint samples;
        uint count;
        uint stride;
        colorMap.GetDimensions(width, height, levels);
        layerMap.GetDimensions(lod, width, height, elements, levels);
        msMap.GetDimensions(width, height, samples);
        volume.GetDimensions(width, height, depth);
        structs.GetDimensions(count, stride);
    }
    """

    ast = parse_code(code)
    nodes = list(iter_ast_nodes(ast))

    member_calls = [
        node.name.member
        for node in nodes
        if isinstance(node, FunctionCallNode)
        and isinstance(node.name, MemberAccessNode)
    ]
    assert member_calls.count("GetDimensions") == 5


def test_parse_get_dimensions_edge_overloads_keep_member_calls():
    code = """
    Texture1D<float4> lineMap : register(t0);
    Texture1DArray<float4> lineArray : register(t1);
    TextureCube<float4> cubeMap : register(t2);
    TextureCubeArray<float4> cubeArray : register(t3);
    RWTexture1DArray<float4> imageArray : register(u0);
    Texture2DMSArray<float4> msArray : register(t4);
    RWTexture2DMSArray<float4> msImage : register(u1);

    void main(uint lod : TEXCOORD0) {
        uint width;
        uint height;
        uint elements;
        uint levels;
        uint samples;
        lineMap.GetDimensions(width, levels);
        lineArray.GetDimensions(lod, width, elements, levels);
        cubeMap.GetDimensions(width, height, levels);
        cubeArray.GetDimensions(lod, width, height, elements, levels);
        imageArray.GetDimensions(width, elements);
        msArray.GetDimensions(width, height, elements, samples);
        msImage.GetDimensions(width, height, elements, samples);
    }
    """

    ast = parse_code(code)
    nodes = list(iter_ast_nodes(ast))

    member_calls = [
        node.name.member
        for node in nodes
        if isinstance(node, FunctionCallNode)
        and isinstance(node.name, MemberAccessNode)
    ]
    assert member_calls.count("GetDimensions") == 7


def test_parse_texture_lod_query_methods_keep_member_calls():
    code = """
    Texture2D<float4> colorMap : register(t0);
    Texture2DMS<float4> msMap : register(t1);
    SamplerState linearSampler : register(s0);

    float4 main(float2 uv : TEXCOORD0) : SV_Target0 {
        float clamped = colorMap.CalculateLevelOfDetail(linearSampler, uv);
        float unclamped = colorMap.CalculateLevelOfDetailUnclamped(
            linearSampler, uv
        );
        float msLod = msMap.CalculateLevelOfDetail(linearSampler, uv);
        return float4(clamped + unclamped + msLod);
    }
    """

    ast = parse_code(code)
    nodes = list(iter_ast_nodes(ast))

    member_calls = [
        node.name.member
        for node in nodes
        if isinstance(node, FunctionCallNode)
        and isinstance(node.name, MemberAccessNode)
    ]
    assert member_calls.count("CalculateLevelOfDetail") == 2
    assert member_calls.count("CalculateLevelOfDetailUnclamped") == 1


def test_parse_texture_sample_position_query_keeps_member_calls():
    code = """
    Texture2DMS<float4> msMap : register(t0);
    Texture2DMSArray<float4> msArray : register(t1);
    Texture2D<float4> colorMap : register(t2);

    float4 main(uint sampleIndex : SV_SampleIndex) : SV_Target0 {
        float2 pos = msMap.GetSamplePosition(sampleIndex);
        float2 arrayPos = msArray.GetSamplePosition(sampleIndex);
        float2 invalid = colorMap.GetSamplePosition(sampleIndex);
        return float4(pos + arrayPos + invalid, 0.0, 1.0);
    }
    """

    ast = parse_code(code)
    nodes = list(iter_ast_nodes(ast))

    member_calls = [
        node.name.member
        for node in nodes
        if isinstance(node, FunctionCallNode)
        and isinstance(node.name, MemberAccessNode)
    ]
    assert member_calls.count("GetSamplePosition") == 3


def test_parse_resource_method_ast_shapes():
    code = """
    Texture2D tex : register(t0);
    SamplerState samp : register(s0);
    RWStructuredBuffer<int> buffer : register(u0);
    AppendStructuredBuffer<int> appendBuf : register(u1);
    ConsumeStructuredBuffer<int> consumeBuf : register(u2);

    float4 main(float2 uv : TEXCOORD0) : SV_Target0 {
        float4 base = tex.Sample(
            samp,
            uv
        );
        float4 mip = tex.SampleLevel(
            samp,
            uv,
            1.0
        );
        float4 grad = tex.SampleGrad(
            samp,
            uv,
            float2(1.0, 0.0),
            float2(0.0, 1.0)
        );
        float cmp = tex.SampleCmpLevelZero(samp, uv, 0.5);
        float cmpLod = tex.SampleCmpLevel(samp, uv, 0.5, 1.0);
        float cmpGrad = tex.SampleCmpGrad(
            samp,
            uv,
            0.5,
            float2(1.0, 0.0),
            float2(0.0, 1.0)
        );
        float cmpBias = tex.SampleCmpBias(samp, uv, 0.5, 0.25);
        int loaded = buffer.Load(0);
        buffer.Store(
            1,
            loaded
        );
        appendBuf.Append(
            loaded
        );
        int consumed = consumeBuf.Consume();
        return base + mip + grad + float4(cmp + loaded + consumed);
    }
    """
    ast = parse_code(code)
    nodes = list(iter_ast_nodes(ast))

    samples = [node for node in nodes if isinstance(node, TextureSampleNode)]
    assert len(samples) == 2
    assert samples[0].lod is None
    assert samples[1].lod is not None

    members = [
        node.name.member
        for node in nodes
        if isinstance(node, FunctionCallNode)
        and isinstance(node.name, MemberAccessNode)
    ]
    assert "Sample" not in members
    assert "SampleLevel" not in members
    assert {
        "SampleGrad",
        "SampleCmpLevelZero",
        "SampleCmpLevel",
        "SampleCmpGrad",
        "SampleCmpBias",
        "Load",
        "Store",
        "Append",
        "Consume",
    }.issubset(set(members))


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

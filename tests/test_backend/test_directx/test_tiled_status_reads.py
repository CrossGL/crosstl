"""HLSL tiled-resource Status operands and ``CheckAccessFullyMapped`` import as CrossGL texture residency queries."""

import re
import shutil
import subprocess
import textwrap

import pytest

import crosstl
from crosstl.backend.DirectX.DirectxCrossGLCodeGen import HLSLToCrossGLConverter
from crosstl.backend.DirectX.DirectxLexer import HLSLLexer
from crosstl.backend.DirectX.DirectxParser import HLSLParser
from crosstl.backend.DirectX.tiled_status_reads import status_operand_count
from crosstl.backend.import_refusals import DirectXImportRefused

COMPUTE = textwrap.dedent("""
    Texture2D<float> depth : register(t0);
    Texture2D<uint2> ids : register(t1);
    Texture2D colorMap : register(t2);
    TextureCube<float4> sky : register(t3);
    RWTexture2D<float4> image : register(u1);
    Buffer<int4> texels : register(t4);
    Texture1D ramp : register(t5);
    SamplerState linearSampler : register(s0);
    SamplerComparisonState shadowSampler : register(s1);
    RWStructuredBuffer<float4> results : register(u0);
    RWStructuredBuffer<uint> flags : register(u2);

    [numthreads(4, 1, 1)]
    void main(uint3 id : SV_DispatchThreadID) {
        float2 uv = float2(id.xy) / 8.0;
        uint status = 0;
        uint mapped = 0;
        float d = depth.SampleLevel(linearSampler, uv, 0, int2(1, 0), status);
        mapped |= CheckAccessFullyMapped(status) ? 1u : 0u;
        uint2 k = ids.Load(int3(id.xy, 0), int2(0, 1), status);
        mapped |= CheckAccessFullyMapped(status) ? 2u : 0u;
        float4 g = colorMap.GatherGreen(linearSampler, uv, int2(0, 0), status);
        float c = depth.SampleCmpLevelZero(shadowSampler, uv, 0.5, int2(0, 0), status);
        float4 s = sky.SampleLevel(linearSampler, float3(uv, 1.0), 0.0, status);
        float4 i = id.x > 1 ? image.Load(int2(id.xy), status) : float4(0, 0, 0, 0);
        int4 t = texels.Load(id.x, status);
        float x = colorMap.SampleGrad(
            linearSampler, uv, float2(0.1, 0), float2(0, 0.1), int2(0, 0), 0.5, status).y;
        float4 r = ramp.Load(int2(id.x, 0), 0, status);
        results[id.x] = float4(d, k.x, c, x) + g + s + i + float4(t) + r;
        flags[id.x] = mapped + status;
    }
""").strip()


def hlsl_to_crossgl(code, converter=None):
    ast = HLSLParser(HLSLLexer(code).tokenize()).parse()
    return (converter or HLSLToCrossGLConverter()).generate(ast)


def translate(tmp_path, code, backend, suffix=".hlsl"):
    path = tmp_path / f"shader{suffix}"
    path.write_text(code, encoding="utf-8")
    return crosstl.translate(str(path), backend=backend)


def dxil(tmp_path, code, entry, profile="cs_6_0"):
    dxc = shutil.which("dxc")
    if dxc is None:
        pytest.skip("dxc is not available")
    path = tmp_path / f"{entry}.hlsl"
    path.write_text(code, encoding="utf-8")
    result = subprocess.run(
        [dxc, "-T", profile, "-E", entry, str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


def test_status_reads_become_residency_queries_around_the_plain_read():
    crossgl = hlsl_to_crossgl(COMPUTE)

    for declaration, read, value in (
        (
            "float d",
            "textureLodOffset(depth, linearSampler, uv, 0, ivec2(1, 0))",
            "crosstl_status_texel_0.x",
        ),
        (
            "uvec2 k",
            "texelFetchOffset(ids, id.xy, 0, ivec2(0, 1))",
            "crosstl_status_texel_1.xy",
        ),
        (
            "float c",
            "textureCompareLodOffset(depth, shadowSampler, uv, 0.5, 0.0, ivec2(0, 0))",
            "crosstl_status_texel_3",
        ),
        (
            "ivec4 t",
            "texelFetch(texels, id.x)",
            "crosstl_status_texel_6",
        ),
        (
            "vec4 r",
            "texelFetchOffset(ramp, id.x, 0, 0)",
            "crosstl_status_texel_8",
        ),
    ):
        texel = re.sub(r"\.[xyzw]+$", "", value)
        assert (
            f"{declaration} = (status = uint(textureResidency({read}, {texel})), {value});"
        ) in crossgl
    # The texel local takes the type of the whole read: a four-vector of the texture's kind, a float for a compare.
    for declaration in (
        "vec4 crosstl_status_texel_0;",
        "uvec4 crosstl_status_texel_1;",
        "vec4 crosstl_status_texel_2;",
        "float crosstl_status_texel_3;",
        "ivec4 crosstl_status_texel_6;",
    ):
        assert declaration in crossgl
    assert "mapped |= textureResident(int(status)) ? 1u : 0u;" in crossgl
    assert "CheckAccessFullyMapped" not in crossgl


def test_a_status_read_in_a_ternary_arm_stays_conditional():
    crossgl = hlsl_to_crossgl(COMPUTE)

    assert (
        "vec4 i = id.x > 1 ? (status = uint(textureResidency(imageLoad(image, ivec2(id.xy)), "
        "crosstl_status_texel_5)), crosstl_status_texel_5) : vec4(0, 0, 0, 0);"
    ) in crossgl


def test_a_status_read_in_a_loop_condition_declares_its_texel_before_the_loop():
    code = textwrap.dedent("""
        Texture2D<float> heights : register(t0);
        RWStructuredBuffer<uint> steps : register(u0);

        [numthreads(1, 1, 1)]
        void main(uint3 id : SV_DispatchThreadID) {
            uint status = 0;
            uint n = 0;
            while (heights.Load(int3(n, id.y, 0), int2(0, 0), status) > 0.5 && n < 8) {
                n += 1;
            }
            steps[id.x] = CheckAccessFullyMapped(status) ? n : 0;
        }
    """).strip()

    crossgl = hlsl_to_crossgl(code)
    loop = crossgl.index("while (")

    assert crossgl.index("vec4 crosstl_status_texel_0;") < loop
    assert (
        "while (((status = uint(textureResidency(texelFetchOffset(heights, ivec2(n, id.y), 0, ivec2(0, 0)), "
        "crosstl_status_texel_0)), crosstl_status_texel_0.x) > 0.5) && (n < 8))"
    ) in crossgl


def test_texel_locals_number_from_zero_in_each_function_and_on_reuse():
    code = textwrap.dedent("""
        Texture2D colorMap : register(t0);
        RWStructuredBuffer<float4> results : register(u0);

        float4 fetch(int2 pixel, out uint status) {
            return colorMap.Load(int3(pixel, 0), int2(0, 0), status);
        }

        [numthreads(1, 1, 1)]
        void main(uint3 id : SV_DispatchThreadID) {
            uint status;
            float4 a = fetch(int2(id.xy), status);
            float4 b = colorMap.Load(int3(id.xy, 0), int2(1, 0), status);
            results[id.x] = CheckAccessFullyMapped(status) ? a + b : a;
        }
    """).strip()
    converter = HLSLToCrossGLConverter()

    first = hlsl_to_crossgl(code, converter)
    second = hlsl_to_crossgl(code, converter)

    assert first == second
    assert first.count("vec4 crosstl_status_texel_0;") == 2
    assert "crosstl_status_texel_1" not in first


def test_check_access_fully_mapped_of_any_status_round_trips(tmp_path):
    code = textwrap.dedent("""
        RWStructuredBuffer<uint> inputs : register(u0);
        RWStructuredBuffer<int> outputs : register(u1);

        [numthreads(1, 1, 1)]
        void main(uint3 id : SV_DispatchThreadID) {
            bool mapped = CheckAccessFullyMapped(inputs[id.x]);
            outputs[id.x] = mapped ? 1 : 0;
        }
    """).strip()

    crossgl = hlsl_to_crossgl(code)
    hlsl = translate(tmp_path, code, "directx")

    assert "bool mapped = textureResident(int(inputs[id.x]));" in crossgl
    # Status is the code the HLSL read wrote, so the HLSL generator tests it with CheckAccessFullyMapped again.
    assert "CheckAccessFullyMapped(uint(int(inputs[id.x])))" in hlsl
    assert "sparseTexelsResidentARB(int(inputs[" in translate(tmp_path, code, "opengl")
    assert "OpImageSparseTexelsResident" in translate(tmp_path, code, "vulkan")


def test_round_trip_keeps_every_status_read_and_check(tmp_path):
    native = dxil(tmp_path, COMPUTE, "main")
    translated = dxil(tmp_path, translate(tmp_path, COMPUTE, "directx"), "CSMain")

    def operations(listing):
        found = re.findall(
            r"call [^@]*@dx\.op\.(checkAccessFullyMapped|sampleLevel|sampleGrad|sampleCmpLevelZero|"
            r"textureLoad|textureGather|bufferLoad)\.",
            listing,
        )
        return sorted(found)

    assert operations(translated) == operations(native)
    assert "checkAccessFullyMapped" in operations(native)


def test_status_reads_lower_to_glsl_and_spirv_sparse_reads(tmp_path):
    code = textwrap.dedent("""
        Texture2D colorMap : register(t0);
        RWTexture2D<float4> image : register(u1);
        SamplerState linearSampler : register(s0);
        RWStructuredBuffer<float4> results : register(u0);

        [numthreads(1, 1, 1)]
        void main(uint3 id : SV_DispatchThreadID) {
            uint status;
            float4 a = colorMap.SampleLevel(linearSampler, float2(id.xy) / 4.0, 0.0, int2(0, 0), status);
            bool first = CheckAccessFullyMapped(status);
            float4 b = image.Load(int2(id.xy), status);
            results[id.x] = first && CheckAccessFullyMapped(status) ? a + b : a;
        }
    """).strip()

    glsl = translate(tmp_path, code, "opengl")
    spirv = translate(tmp_path, code, "vulkan")

    assert "#extension GL_ARB_sparse_texture2 : require" in glsl
    assert "sparseTextureLodOffsetARB(" in glsl
    assert "sparseImageLoadARB(" in glsl
    assert "sparseTexelsResidentARB(" in glsl
    assert "OpImageSparseSampleExplicitLod" in spirv
    assert "OpImageSparseRead" in spirv


def test_buffer_status_loads_stay_refused():
    code = textwrap.dedent("""
        ByteAddressBuffer words : register(t0);
        StructuredBuffer<float4> items : register(t1);
        RWStructuredBuffer<float4> results : register(u0);

        [numthreads(1, 1, 1)]
        void main(uint3 id : SV_DispatchThreadID) {
            uint status;
            uint word = words.Load(id.x * 4u, status);
            float4 item = items.Load(word, status);
            results[id.x] = CheckAccessFullyMapped(status) ? item : float4(0, 0, 0, 0);
        }
    """).strip()

    with pytest.raises(DirectXImportRefused) as error:
        hlsl_to_crossgl(code)

    assert (
        "DirectX tiled-resource status for Load: CrossGL has no operand for status output"
        in str(error.value)
    )


def test_four_offset_compare_gathers_with_status_stay_refused():
    code = textwrap.dedent("""
        Texture2D<float> shadowMap : register(t0);
        SamplerComparisonState shadowSampler : register(s0);

        float4 main(float2 uv : TEXCOORD0, float depth : TEXCOORD1) : SV_Target {
            uint status;
            float4 gathered = shadowMap.GatherCmpRed(
                shadowSampler, uv, depth, int2(0, 0), int2(1, 0), int2(0, 1), int2(1, 1), status);
            return CheckAccessFullyMapped(status) ? gathered : float4(0, 0, 0, 0);
        }
    """).strip()

    with pytest.raises(DirectXImportRefused) as error:
        hlsl_to_crossgl(code)

    assert (
        "tiled-resource status for Texture2D.GatherCmpRed: the read imports as 'vec4("
        in (str(error.value))
    )


@pytest.mark.parametrize(
    ("member", "resource", "counts"),
    [
        ("Sample", "Texture2D", {5}),
        ("Sample", "TextureCube", {4}),
        ("SampleCmpGrad", "Texture2DArray", {8}),
        ("SampleCmpLevelZero", "TextureCubeArray", {4}),
        ("GatherRed", "Texture2D", {4, 7}),
        ("GatherCmpAlpha", "Texture2DArray", {5, 8}),
        ("GatherRed", "TextureCube", {3}),
        ("GatherRaw", "TextureCube", set()),
        ("Gather", "Texture3D", set()),
        ("Gather", "Texture1D", set()),
        ("Load", "Texture2D", {3}),
        ("Load", "Texture2DMS", {4}),
        ("Load", "RWTexture2DMSArray", {3}),
        ("Load", "RWTexture3D", {2}),
        ("Load", "Buffer", {2}),
        ("Load", "ByteAddressBuffer", set()),
        ("Load", "StructuredBuffer", set()),
    ],
)
def test_status_overload_operand_counts_follow_dxc(member, resource, counts):
    assert status_operand_count(member, resource) == frozenset(counts)

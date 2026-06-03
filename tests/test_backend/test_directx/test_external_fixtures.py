from __future__ import annotations

import textwrap
from dataclasses import dataclass

import pytest

from crosstl.backend.DirectX.DirectxCrossGLCodeGen import HLSLToCrossGLConverter
from crosstl.backend.DirectX.DirectxLexer import HLSLLexer
from crosstl.backend.DirectX.DirectxParser import HLSLParser
from crosstl.translator.lexer import Lexer as CrossGLLexer
from crosstl.translator.parser import Parser as CrossGLParser


@dataclass(frozen=True)
class ExternalFixture:
    name: str
    repo: str
    commit: str
    path: str
    code: str
    contains: tuple[str, ...]

    @property
    def source_url(self):
        return f"{self.repo}/blob/{self.commit}/{self.path}"


DIRECTX_GRAPHICS_SAMPLES_REPO = "https://github.com/microsoft/DirectX-Graphics-Samples"
DIRECTX_GRAPHICS_SAMPLES_COMMIT = "31ae3c91160d8634264004cdaf4e41a99c41243e"
DIRECTX_SHADER_COMPILER_REPO = "https://github.com/microsoft/DirectXShaderCompiler"
DIRECTX_SHADER_COMPILER_COMMIT = "517dd5eb5d8cbb46c15fc1230acac1d2f4779092"
FIDELITYFX_FSR_REPO = "https://github.com/GPUOpen-Effects/FidelityFX-FSR"
FIDELITYFX_FSR_COMMIT = "a21ffb8f6c13233ba336352bdff293894c706575"


EXTERNAL_FIXTURES = [
    ExternalFixture(
        name="directx_graphics_samples_hello_triangle",
        repo=DIRECTX_GRAPHICS_SAMPLES_REPO,
        commit=DIRECTX_GRAPHICS_SAMPLES_COMMIT,
        path="Samples/Desktop/D3D12HelloWorld/src/HelloTriangle/shaders.hlsl",
        code=textwrap.dedent("""
            struct PSInput
            {
                float4 position : SV_POSITION;
                float4 color : COLOR;
            };

            PSInput VSMain(float4 position : POSITION, float4 color : COLOR)
            {
                PSInput result;

                result.position = position;
                result.color = color;

                return result;
            }

            float4 PSMain(PSInput input) : SV_TARGET
            {
                return input.color;
            }
        """).strip(),
        contains=(
            "vec4 position @ gl_Position",
            "vec4 PSMain(PSInput input) @ gl_FragColor",
        ),
    ),
    ExternalFixture(
        name="directx_graphics_samples_miniengine_present_sdr",
        repo=DIRECTX_GRAPHICS_SAMPLES_REPO,
        commit=DIRECTX_GRAPHICS_SAMPLES_COMMIT,
        path="MiniEngine/Core/Shaders/PresentSDRPS.hlsl",
        code=textwrap.dedent("""
            #include "ShaderUtility.hlsli"
            #include "PresentRS.hlsli"

            Texture2D<float3> ColorTex : register(t0);

            [RootSignature(Present_RootSig)]
            float3 main(float4 position : SV_Position) : SV_Target0
            {
                float3 LinearRGB = ColorTex[(int2)position.xy];
                return ApplyDisplayProfile(LinearRGB, DISPLAY_PLANE_FORMAT);
            }
        """).strip(),
        contains=(
            "@ RootSignature(Present_RootSig)",
            "sampler2D ColorTex;",
            "vec3 LinearRGB = ColorTex[ivec2(position.xy)];",
        ),
    ),
    ExternalFixture(
        name="directx_shader_compiler_groupshared_splat",
        repo=DIRECTX_SHADER_COMPILER_REPO,
        commit=DIRECTX_SHADER_COMPILER_COMMIT,
        path="tools/clang/test/CodeGenHLSL/SplatGroupSharedScalar.hlsl",
        code=textwrap.dedent("""
            groupshared int a;
            [numthreads(64, 1, 1)]
            void main() {
              a = 123;
              int4 x = (a).xxxx;
            }
        """).strip(),
        contains=(
            "groupshared int a;",
            "@ numthreads(64, 1, 1)",
            "ivec4 x = a.xxxx;",
        ),
    ),
    ExternalFixture(
        name="directx_shader_compiler_texture_cube_dimensions_lod",
        repo=DIRECTX_SHADER_COMPILER_REPO,
        commit=DIRECTX_SHADER_COMPILER_COMMIT,
        path="tools/clang/test/CodeGenHLSL/GetDimCalcLOD.hlsl",
        code=textwrap.dedent("""
            TextureCube<float4> cube;

            SamplerState    g_sam;

            float4 main(float2 uv : UV) : SV_TARGET
            {
                uint w;
                uint h;

                cube.GetDimensions(w,h);
                float lod = cube.CalculateLevelOfDetail(g_sam, float3(uv,1));
                return float4(w, h, lod, 1.0);
            }
        """).strip(),
        contains=(
            "samplerCube cube;",
            "w = uint(textureSize(cube, 0).x);",
            "float lod = textureQueryLod(cube, g_sam, vec3(uv, 1)).x;",
        ),
    ),
    ExternalFixture(
        name="directx_graphics_samples_meshlet_render_pixel",
        repo=DIRECTX_GRAPHICS_SAMPLES_REPO,
        commit=DIRECTX_GRAPHICS_SAMPLES_COMMIT,
        path="Samples/Desktop/D3D12MeshShaders/src/MeshletRender/MeshletPS.hlsl",
        code=textwrap.dedent("""
            struct Constants
            {
                float4x4 World;
                float4x4 WorldView;
                float4x4 WorldViewProj;
                uint     DrawMeshlets;
            };

            struct VertexOut
            {
                float4 PositionHS   : SV_Position;
                float3 PositionVS   : POSITION0;
                float3 Normal       : NORMAL0;
                uint   MeshletIndex : COLOR0;
            };

            ConstantBuffer<Constants> Globals : register(b0);

            float4 main(VertexOut input) : SV_TARGET
            {
                float ambientIntensity = 0.1;
                float3 lightColor = float3(1, 1, 1);
                float3 lightDir = -normalize(float3(1, -1, 1));

                float3 diffuseColor;
                float shininess;
                if (Globals.DrawMeshlets)
                {
                    uint meshletIndex = input.MeshletIndex;
                    diffuseColor = float3(
                        float(meshletIndex & 1),
                        float(meshletIndex & 3) / 4,
                        float(meshletIndex & 7) / 8);
                    shininess = 16.0;
                }
                else
                {
                    diffuseColor = 0.8;
                    shininess = 64.0;
                }

                float3 normal = normalize(input.Normal);

                float cosAngle = saturate(dot(normal, lightDir));
                float3 viewDir = -normalize(input.PositionVS);
                float3 halfAngle = normalize(lightDir + viewDir);

                float blinnTerm = saturate(dot(normal, halfAngle));
                blinnTerm = cosAngle != 0.0 ? blinnTerm : 0.0;
                blinnTerm = pow(blinnTerm, shininess);

                float3 finalColor = (cosAngle + blinnTerm + ambientIntensity) * diffuseColor;

                return float4(finalColor, 1);
            }
        """).strip(),
        contains=(
            "ConstantBuffer<Constants> Globals;",
            "float cosAngle = clamp(dot(normal, lightDir), 0.0, 1.0);",
            "return vec4(finalColor, 1);",
        ),
    ),
    ExternalFixture(
        name="fidelityfx_fsr_dx12_pass_dispatch_filter",
        repo=FIDELITYFX_FSR_REPO,
        commit=FIDELITYFX_FSR_COMMIT,
        path="sample/src/DX12/FSR_Pass.hlsl",
        code=textwrap.dedent("""
            cbuffer cb : register(b0)
            {
                uint4 Const0;
                uint4 Sample;
            };

            SamplerState samLinearClamp : register(s0);

            Texture2D InputTexture : register(t0);
            RWTexture2D<float4> OutputTexture : register(u0);

            void CurrFilter(int2 pos)
            {
                float2 pp = (float2(pos) * float2(Const0.xy));
                OutputTexture[pos] = InputTexture.SampleLevel(samLinearClamp, pp, 0.0);
                float3 c;
                if (Sample.x == 1)
                    c *= c;
                OutputTexture[pos] = float4(c, 1);
            }

            [numthreads(8, 8, 1)]
            void mainCS(uint3 LocalThreadId : SV_GroupThreadID, uint3 WorkGroupId : SV_GroupID, uint3 Dtid : SV_DispatchThreadID)
            {
                uint2 gxy = LocalThreadId.xy + uint2(WorkGroupId.x << 4u, WorkGroupId.y << 4u);
                CurrFilter(gxy);
                gxy.x += 8u;
                CurrFilter(gxy);
            }
        """).strip(),
        contains=(
            "@ register(u0)",
            "imageStore(OutputTexture, pos, textureLod(InputTexture, samLinearClamp, pp, 0.0));",
            "uvec2 gxy = LocalThreadId.xy + uvec2(WorkGroupId.x << 4, WorkGroupId.y << 4);",
        ),
    ),
]


def parse_hlsl(code):
    tokens = HLSLLexer(code).tokenize()
    return HLSLParser(tokens).parse()


def generate_crossgl(code):
    ast = parse_hlsl(code)
    return HLSLToCrossGLConverter().generate(ast)


def parse_crossgl(code):
    tokens = CrossGLLexer(code).get_tokens()
    return CrossGLParser(tokens).parse()


def test_external_fixture_metadata_records_repositories_and_commits():
    assert all(
        fixture.repo.startswith("https://github.com/") for fixture in EXTERNAL_FIXTURES
    )
    assert all(len(fixture.commit) == 40 for fixture in EXTERNAL_FIXTURES)
    assert all(fixture.path.endswith(".hlsl") for fixture in EXTERNAL_FIXTURES)


@pytest.mark.parametrize("fixture", EXTERNAL_FIXTURES, ids=lambda fixture: fixture.name)
def test_parse_external_directx_fixture(fixture):
    ast = parse_hlsl(fixture.code)

    assert ast is not None
    assert ast.functions
    assert fixture.source_url.startswith(fixture.repo)


@pytest.mark.parametrize("fixture", EXTERNAL_FIXTURES, ids=lambda fixture: fixture.name)
def test_codegen_external_directx_fixture_to_parseable_crossgl(fixture):
    crossgl = generate_crossgl(fixture.code)

    for expected in fixture.contains:
        assert expected in crossgl
    assert parse_crossgl(crossgl) is not None

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


DIRECTX_GRAPHICS_SAMPLES_REPO = (
    "https://github.com/microsoft/DirectX-Graphics-Samples"
)
DIRECTX_GRAPHICS_SAMPLES_COMMIT = "31ae3c91160d8634264004cdaf4e41a99c41243e"


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
    assert all(fixture.repo.startswith("https://github.com/") for fixture in EXTERNAL_FIXTURES)
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

import pytest

import crosstl.translator as cgl_translator
from crosstl.backend.slang import SlangCrossGLCodeGen, SlangLexer, SlangParser


EXTERNAL_REPOS = {
    "shader-slang/slang": {
        "url": "https://github.com/shader-slang/slang",
        "commit": "adc996670ec281aa8a4ee131f30b324648cbbe60",
    },
    "NVIDIAGameWorks/Falcor": {
        "url": "https://github.com/NVIDIAGameWorks/Falcor",
        "commit": "eb540f6748774680ce0039aaf3ac9279266ec521",
    },
    "NVIDIAGameWorks/RTXGI": {
        "url": "https://github.com/NVIDIAGameWorks/RTXGI",
        "commit": "10b5770b8eaddfc1faab82b65f799ac6f47dcc44",
        "note": "searched; current tree has no .slang/.slangh files",
    },
}


EXTERNAL_FIXTURES = [
    {
        "id": "slang_default_parameter",
        "repo": "shader-slang/slang",
        "path": "tests/compute/default-parameter.slang",
        "source": """
            RWStructuredBuffer<int> outputBuffer;

            int helper(int val, int a = 16)
            {
                return val + a;
            }

            int test(int val)
            {
                return helper(val) + helper(val, 256);
            }

            [numthreads(4, 1, 1)]
            void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
            {
                outputBuffer[dispatchThreadID.x] = test((int)dispatchThreadID.x);
            }
        """,
        "crossgl": True,
        "contains": [
            "int helper(int val, int a)",
            "return helper(val) + helper(val, 256);",
        ],
        "not_contains": ["int a, = , 16", "int a = 16"],
    },
    {
        "id": "slang_tbuffer",
        "repo": "shader-slang/slang",
        "path": "tests/hlsl/tbuffer.slang",
        "source": """
            tbuffer tbuf : register(t0)
            {
                float4 tb_val1;
            }

            tbuffer tbuf2 : register(t1)
            {
                Texture2D<float4> texture2D;
                float4 tb_val2;
            }

            RWStructuredBuffer<float4> outputBuffer;

            [numthreads(1, 1, 1)]
            void computeMain()
            {
                outputBuffer[0] = tb_val1 + texture2D[0] + tb_val2;
            }
        """,
        "crossgl": True,
        "contains": [
            "cbuffer tbuf @register(t0)",
            "cbuffer tbuf2 @register(t1)",
            "sampler2D texture2D;",
        ],
    },
    {
        "id": "falcor_texture_load",
        "repo": "NVIDIAGameWorks/Falcor",
        "path": "Source/Tools/FalcorTest/Tests/Core/TextureLoadTests.cs.slang",
        "source": """
            Texture2D<float4> gTex;
            RWStructuredBuffer<float4> result;

            [numthreads(1, 1, 1)]
            void main(uint3 dispatchThreadID : SV_DispatchThreadID)
            {
                result[dispatchThreadID.x] = gTex.Load(int3(0, 0, 0));
            }
        """,
        "crossgl": True,
        "contains": [
            "sampler2D gTex;",
            "texelFetch(gTex, ivec2(0, 0), 0)",
        ],
        "not_contains": ["gTex.Load"],
    },
]


def parse_slang(source):
    tokens = SlangLexer(source).tokenize()
    return SlangParser(tokens).parse()


def generate_crossgl(ast):
    return SlangCrossGLCodeGen.SlangToCrossGLConverter().generate(ast)


@pytest.mark.parametrize("fixture", EXTERNAL_FIXTURES, ids=lambda item: item["id"])
def test_external_fixture_codegen_crossgl_reparse(fixture):
    ast = parse_slang(fixture["source"])
    generated = generate_crossgl(ast)

    for expected in fixture.get("contains", []):
        assert expected in generated
    for rejected in fixture.get("not_contains", []):
        assert rejected not in generated

    if fixture.get("crossgl", False):
        cgl_translator.parse(generated)


def test_falcor_exported_import_and_default_interface_parameter_parse():
    source = """
        __exported import Rendering.Materials.IMaterialInstance;
        __exported import Scene.Material.TextureSampler;
        import Rendering.Volumes.PhaseFunction;

        [anyValueSize(128)]
        interface IMaterial
        {
            associatedtype MaterialInstance : IMaterialInstance;

            MaterialInstance setupMaterialInstance(
                const MaterialSystem ms,
                const ShadingData sd,
                const ITextureSampler lod,
                const uint hints = (uint)MaterialInstanceHints::None);
        }
    """

    ast = parse_slang(source)
    method = ast.interfaces[0].methods[0]
    hints = method.params[-1]

    assert [node.module_name for node in ast.imports] == [
        "Rendering.Materials.IMaterialInstance",
        "Scene.Material.TextureSampler",
        "Rendering.Volumes.PhaseFunction",
    ]
    assert ast.imports[0].qualifiers == ["__exported"]
    assert ast.imports[1].qualifiers == ["__exported"]
    assert hints.vtype == "uint"
    assert hints.name == "hints"
    assert hints.value.target_type == "uint"
    assert hints.value.expression.name == "MaterialInstanceHints::None"

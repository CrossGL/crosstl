import re
import textwrap

import pytest

from crosstl._crosstl import translate

HLSL_INOUT_VERTEX = textwrap.dedent("""
    cbuffer Constants
    {
        float4x4 g_WorldViewProj;
    };

    struct VSInput
    {
        float3 Pos   : ATTRIB0;
        float4 Color : ATTRIB1;
    };

    struct PSInput
    {
        float4 Pos   : SV_POSITION;
        float4 Color : COLOR0;
    };

    void main(in  VSInput VSIn,
              out PSInput PSIn)
    {
        PSIn.Pos   = mul(float4(VSIn.Pos, 1.0), g_WorldViewProj);
        PSIn.Color = VSIn.Color;
    }
    """).strip()


HLSL_INOUT_FRAGMENT = textwrap.dedent("""
    struct PSInput
    {
        float4 Pos   : SV_POSITION;
        float4 Color : COLOR0;
    };

    struct PSOutput
    {
        float4 Color : SV_TARGET;
    };

    void main(in  PSInput  PSIn,
              out PSOutput PSOut)
    {
        float4 Color = PSIn.Color;
        PSOut.Color = Color;
    }
    """).strip()


def translate_hlsl(tmp_path, source, backend):
    shader_path = tmp_path / f"inout-stage-entry.{backend}.hlsl"
    shader_path.write_text(source, encoding="utf-8")
    return translate(
        str(shader_path),
        backend=backend,
        source_backend="directx",
        format_output=False,
    )


def assert_no_hlsl_entry_parameters(generated):
    assert not re.search(
        r"void\s+main\s*\([^)]*\b(?:VSInput|PSInput|PSOutput)\b",
        generated,
    )
    assert "out PSInput" not in generated
    assert "out PSOutput" not in generated


@pytest.mark.parametrize(
    ("backend", "expected_snippets"),
    [
        (
            "opengl",
            (
                "// Vertex Shader",
                "void main()",
                "in vec3 VSIn_Pos;",
                "in vec4 VSIn_Color;",
                "out vec4 PSIn_Color;",
                "gl_Position =",
            ),
        ),
        (
            "metal",
            (
                "vertex vertex_main_Return vertex_main(",
                "vertex_main_Input _crossglInput [[stage_in]]",
                "float3 VSIn_Pos [[attribute(0)]];",
                "float4 PSIn_Color [[user(Color0)]];",
            ),
        ),
        (
            "vulkan",
            (
                "OpEntryPoint Vertex",
                "OpDecorate",
                '"CrossGL_vertex_input_VSIn_Pos"',
                '"CrossGL_vertex_output_PSIn_Color"',
            ),
        ),
    ],
)
def test_hlsl_struct_inout_vertex_entry_lowers_to_native_stage_entry(
    tmp_path, backend, expected_snippets
):
    generated = translate_hlsl(tmp_path, HLSL_INOUT_VERTEX, backend)

    assert_no_hlsl_entry_parameters(generated)
    for snippet in expected_snippets:
        assert snippet in generated
    if backend == "vulkan":
        assert "OpEntryPoint Fragment" not in generated


@pytest.mark.parametrize(
    ("backend", "expected_snippets"),
    [
        (
            "opengl",
            (
                "// Fragment Shader",
                "void main()",
                "in vec4 PSIn_Color;",
                "layout(location = 0) out vec4 fragColor;",
                "fragColor = Color;",
            ),
        ),
        (
            "metal",
            (
                "fragment fragment_main_Return fragment_main(",
                "fragment_main_Input _crossglInput [[stage_in]]",
                "float4 PSOut_Color [[color(0)]];",
            ),
        ),
        (
            "vulkan",
            (
                "OpEntryPoint Fragment",
                "OpExecutionMode",
                '"CrossGL_fragment_input_PSIn_Color"',
                '"CrossGL_fragment_output_PSOut_Color"',
            ),
        ),
    ],
)
def test_hlsl_struct_inout_fragment_entry_lowers_to_native_stage_entry(
    tmp_path, backend, expected_snippets
):
    generated = translate_hlsl(tmp_path, HLSL_INOUT_FRAGMENT, backend)

    assert_no_hlsl_entry_parameters(generated)
    for snippet in expected_snippets:
        assert snippet in generated
    if backend == "vulkan":
        assert "OpEntryPoint Vertex" not in generated

"""Importers refuse source constructs CrossGL cannot represent instead of emitting placeholders."""

from __future__ import annotations

import textwrap

import pytest

import crosstl
from crosstl.backend.DirectX import DirectxCrossGLCodeGen
from crosstl.backend.DirectX.DirectxLexer import HLSLLexer
from crosstl.backend.DirectX.DirectxParser import HLSLParser
from crosstl.backend.import_refusals import (
    DirectXImportRefused,
    ImportRefused,
    MetalImportRefused,
    SlangImportRefused,
)
from crosstl.backend.Metal.MetalCrossGLCodeGen import MetalToCrossGLConverter
from crosstl.backend.Metal.MetalLexer import MetalLexer
from crosstl.backend.Metal.MetalParser import MetalParser
from crosstl.translator.diagnostics import TranslationDiagnostic
from tests.lowering_refusal_helpers import (
    DIRECTX_IMPORT_REFUSAL_CODE,
    METAL_IMPORT_REFUSAL_CODE,
    refused_hlsl_import,
    refused_metal_import,
)

TILED_STATUS_HLSL = textwrap.dedent("""
    Texture2D<float4> colorMap : register(t0);

    float4 main(int2 pixel : TEXCOORD0) : SV_Target0 {
        uint status;
        float4 color = colorMap.Load(int3(pixel, 0), int2(0, 0), status);
        return CheckAccessFullyMapped(status) ? color : float4(1.0, 0.0, 1.0, 1.0);
    }
""").strip()

CLEAN_HLSL = textwrap.dedent("""
    Texture2D<float4> colorMap : register(t0);

    float4 main(int2 pixel : TEXCOORD0) : SV_Target0 {
        return colorMap.Load(int3(pixel, 0));
    }
""").strip()

SAMPLED_WRITE_METAL = textwrap.dedent("""
    #include <metal_stdlib>
    using namespace metal;

    fragment void fragment_main(texture2d<float> tex [[texture(0)]]) {
        tex.write(float4(1.0), uint2(0, 0));
    }
""").strip()

CLEAN_METAL = textwrap.dedent("""
    #include <metal_stdlib>
    using namespace metal;

    fragment float4 fragment_main(texture2d<float> tex [[texture(0)]],
                                  sampler samp [[sampler(0)]]) {
        return tex.sample(samp, float2(0.5, 0.5));
    }
""").strip()


def hlsl_to_crossgl(code, converter=None):
    ast = HLSLParser(HLSLLexer(code).tokenize()).parse()
    converter = converter or DirectxCrossGLCodeGen.HLSLToCrossGLConverter()
    return converter.generate(ast)


def metal_to_crossgl(code, converter=None):
    ast = MetalParser(MetalLexer(code).tokenize()).parse()
    converter = converter or MetalToCrossGLConverter()
    return converter.generate(ast)


def test_import_refusals_are_typed_translation_diagnostics():
    for error_type, code, capability in (
        (DirectXImportRefused, DIRECTX_IMPORT_REFUSAL_CODE, "directx.import"),
        (MetalImportRefused, METAL_IMPORT_REFUSAL_CODE, "metal.import"),
        (
            SlangImportRefused,
            "project.translate.slang-import-unsupported",
            "slang.import",
        ),
    ):
        error = error_type(["first reason", "second reason"])
        assert isinstance(error, ImportRefused)
        assert isinstance(error, TranslationDiagnostic)
        assert error.code == code
        assert error.target == "cgl"
        assert error.reasons == ("first reason", "second reason")
        assert error.missing_capabilities == (capability,)
        assert "2 " in str(error) and "constructs: first reason; second reason" in (
            str(error)
        )


def test_hlsl_import_reports_every_refused_construct_once():
    message = refused_hlsl_import(hlsl_to_crossgl, TILED_STATUS_HLSL)

    assert message.startswith("CrossGL cannot represent 2 HLSL constructs: ")
    assert (
        "DirectX tiled-resource status for Load: CrossGL has no operand for "
        "status output"
    ) in message
    assert (
        "DirectX tiled-resource status check CheckAccessFullyMapped: CrossGL has "
        "no residency status, so the mapped result is unknown"
    ) in message
    assert "/*" not in message


def test_hlsl_importer_reuse_does_not_leak_refusals_between_programs():
    converter = DirectxCrossGLCodeGen.HLSLToCrossGLConverter()
    with pytest.raises(DirectXImportRefused):
        hlsl_to_crossgl(TILED_STATUS_HLSL, converter)

    crossgl = hlsl_to_crossgl(CLEAN_HLSL, converter)

    assert "texelFetch(colorMap" in crossgl
    assert "unsupported" not in crossgl


def test_metal_import_refuses_sampled_texture_write_and_reuse_is_clean():
    converter = MetalToCrossGLConverter()
    message = refused_metal_import(metal_to_crossgl, SAMPLED_WRITE_METAL, converter)

    assert message.startswith("CrossGL cannot represent 1 Metal construct: ")
    assert "Metal sampled texture write write on tex" in message

    crossgl = metal_to_crossgl(CLEAN_METAL, converter)
    assert "texture(tex, samp" in crossgl
    assert "unsupported" not in crossgl


@pytest.mark.parametrize(
    "suffix, source, error_type",
    [
        (".hlsl", TILED_STATUS_HLSL, DirectXImportRefused),
        (".metal", SAMPLED_WRITE_METAL, MetalImportRefused),
    ],
)
@pytest.mark.parametrize("backend", ["cgl", "opengl", "metal", "directx", "vulkan"])
def test_public_translate_returns_no_program_for_refused_imports(
    tmp_path, suffix, source, error_type, backend
):
    shader = tmp_path / f"shader{suffix}"
    shader.write_text(source)
    output = tmp_path / "translated.out"

    with pytest.raises(error_type):
        crosstl.translate(str(shader), backend=backend, save_shader=str(output))

    assert not output.exists()

from pathlib import Path

import pytest

import crosstl
import crosstl.translator
import crosstl.translator.codegen as codegen


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_ROOT = ROOT / "examples"

FULL_BACKEND_EXAMPLES = (
    "advanced/ArrayTest.cgl",
    "compute/ParticleSimulation.cgl",
    "gpu_computing/MatrixMultiplication.cgl",
    "graphics/PerlinNoise.cgl",
    "graphics/SimpleShader.cgl",
)

KNOWN_PRIMARY_GRAPHICS_GAPS = (
    pytest.param(
        "advanced/GenericPatternMatching.cgl",
        "directx",
        marks=pytest.mark.xfail(
            strict=True,
            reason="guarded match arms are not lowerable to HLSL switch yet",
        ),
    ),
    pytest.param(
        "advanced/GenericPatternMatching.cgl",
        "metal",
        marks=pytest.mark.xfail(
            strict=True,
            reason="guarded match arms are not lowerable to Metal switch yet",
        ),
    ),
    pytest.param(
        "advanced/GenericPatternMatching.cgl",
        "opengl",
        marks=pytest.mark.xfail(
            strict=True,
            reason="guarded match arms are not lowerable to GLSL switch yet",
        ),
    ),
    pytest.param(
        "cross_platform/UniversalPBRShader.cgl",
        "directx",
        marks=pytest.mark.xfail(
            strict=True,
            reason="PBR example still uses texture identifiers missing resource declarations",
        ),
    ),
    pytest.param(
        "cross_platform/UniversalPBRShader.cgl",
        "metal",
        marks=pytest.mark.xfail(
            strict=True,
            reason="PBR example still uses texture identifiers missing resource declarations",
        ),
    ),
    pytest.param(
        "cross_platform/UniversalPBRShader.cgl",
        "opengl",
        marks=pytest.mark.xfail(
            strict=True,
            reason="PBR example still uses texture identifiers missing resource declarations",
        ),
    ),
    pytest.param(
        "graphics/ComplexShader.cgl",
        "directx",
        marks=pytest.mark.xfail(
            strict=True,
            reason="complex graphics example still references undeclared texture resources",
        ),
    ),
    pytest.param(
        "graphics/ComplexShader.cgl",
        "metal",
        marks=pytest.mark.xfail(
            strict=True,
            reason="complex graphics example still references undeclared texture resources",
        ),
    ),
    pytest.param(
        "graphics/ComplexShader.cgl",
        "opengl",
        marks=pytest.mark.xfail(
            strict=True,
            reason="complex graphics example currently has overlapping fragment outputs",
        ),
    ),
)


def _example_path(relative_path):
    return EXAMPLES_ROOT / relative_path


def _assert_generated_output_is_usable(generated):
    assert isinstance(generated, str)
    assert generated.strip()
    assert "Traceback" not in generated
    assert "NotImplemented" not in generated
    assert "<crosstl." not in generated


@pytest.mark.parametrize("example_path", sorted(EXAMPLES_ROOT.rglob("*.cgl")))
def test_checked_in_examples_parse_as_crossgl(example_path):
    crosstl.translator.parse(example_path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("relative_path", FULL_BACKEND_EXAMPLES)
@pytest.mark.parametrize("backend", codegen.backend_names())
def test_portable_examples_translate_to_all_registered_backends(relative_path, backend):
    generated = crosstl.translate(
        str(_example_path(relative_path)), backend=backend, format_output=False
    )

    _assert_generated_output_is_usable(generated)


@pytest.mark.parametrize("relative_path,backend", KNOWN_PRIMARY_GRAPHICS_GAPS)
def test_known_primary_graphics_example_gaps_are_tracked(relative_path, backend):
    generated = crosstl.translate(
        str(_example_path(relative_path)), backend=backend, format_output=False
    )

    _assert_generated_output_is_usable(generated)

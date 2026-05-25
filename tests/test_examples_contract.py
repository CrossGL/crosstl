import re
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

KNOWN_PRIMARY_GRAPHICS_GAPS = ()

PRIMARY_GRAPHICS_FIXED_CASES = (
    ("advanced/GenericPatternMatching.cgl", "directx"),
    ("advanced/GenericPatternMatching.cgl", "metal"),
    ("advanced/GenericPatternMatching.cgl", "opengl"),
    ("cross_platform/UniversalPBRShader.cgl", "directx"),
    ("cross_platform/UniversalPBRShader.cgl", "metal"),
    ("cross_platform/UniversalPBRShader.cgl", "opengl"),
    ("graphics/ComplexShader.cgl", "directx"),
    ("graphics/ComplexShader.cgl", "metal"),
    ("graphics/ComplexShader.cgl", "opengl"),
)

KNOWN_PRIMARY_GRAPHICS_DIAGNOSTICS = ()


def _example_path(relative_path):
    return EXAMPLES_ROOT / relative_path


def _assert_generated_output_is_usable(generated):
    assert isinstance(generated, str)
    assert generated.strip()
    assert "Traceback" not in generated
    assert "NotImplemented" not in generated
    assert "<crosstl." not in generated
    assert "MatchNode(" not in generated
    assert "ConstructorNode(" not in generated


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


@pytest.mark.parametrize("relative_path,backend", PRIMARY_GRAPHICS_FIXED_CASES)
def test_primary_graphics_examples_with_stage_local_resources_translate(
    relative_path, backend
):
    generated = crosstl.translate(
        str(_example_path(relative_path)), backend=backend, format_output=False
    )

    _assert_generated_output_is_usable(generated)


@pytest.mark.parametrize(
    "relative_path,backend,exc_type,message", KNOWN_PRIMARY_GRAPHICS_DIAGNOSTICS
)
def test_known_primary_graphics_example_gaps_report_actionable_diagnostics(
    relative_path, backend, exc_type, message
):
    with pytest.raises(exc_type, match=re.escape(message)):
        crosstl.translate(
            str(_example_path(relative_path)), backend=backend, format_output=False
        )

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

WEBGL_COMPUTE_EXAMPLE_DIAGNOSTICS = (
    ("compute/ParticleSimulation.cgl", "webgl"),
    ("gpu_computing/MatrixMultiplication.cgl", "webgl"),
)
WEBGL_COMPUTE_EXAMPLE_DIAGNOSTIC_CASES = set(WEBGL_COMPUTE_EXAMPLE_DIAGNOSTICS)
FULL_BACKEND_EXAMPLE_CASES = tuple(
    (relative_path, backend)
    for relative_path in FULL_BACKEND_EXAMPLES
    for backend in codegen.backend_names()
    if (relative_path, backend) not in WEBGL_COMPUTE_EXAMPLE_DIAGNOSTIC_CASES
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

ADDITIONAL_FIXED_CASES = (
    ("cross_platform/UniversalPBRShader.cgl", "mojo"),
    ("cross_platform/UniversalPBRShader.cgl", "slang"),
)

GENERIC_FUNCTION_UNSUPPORTED_BACKEND_CASES = (
    ("advanced/GenericPatternMatching.cgl", "vulkan", "SPIR-V"),
    ("advanced/GenericPatternMatching.cgl", "cuda", "CUDA"),
    ("advanced/GenericPatternMatching.cgl", "hip", "HIP"),
    ("advanced/GenericPatternMatching.cgl", "mojo", "Mojo"),
    ("advanced/GenericPatternMatching.cgl", "slang", "Slang"),
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
    assert "NamedType(" not in generated
    assert "Result_T" not in generated
    assert "Vec3_T" not in generated
    assert "T::zero" not in generated
    assert "T::one" not in generated
    assert "unsupported CUDA match" not in generated
    assert "unsupported HIP match" not in generated
    assert "unsupported Slang match" not in generated


@pytest.mark.parametrize("example_path", sorted(EXAMPLES_ROOT.rglob("*.cgl")))
def test_checked_in_examples_parse_as_crossgl(example_path):
    crosstl.translator.parse(example_path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("relative_path,backend", FULL_BACKEND_EXAMPLE_CASES)
def test_portable_examples_translate_to_all_registered_backends(relative_path, backend):
    generated = crosstl.translate(
        str(_example_path(relative_path)), backend=backend, format_output=False
    )

    _assert_generated_output_is_usable(generated)


@pytest.mark.parametrize("relative_path,backend", WEBGL_COMPUTE_EXAMPLE_DIAGNOSTICS)
def test_webgl_compute_examples_report_actionable_diagnostics(relative_path, backend):
    with pytest.raises(
        ValueError,
        match="WebGL target does not support shader stage\\(s\\): compute",
    ):
        crosstl.translate(
            str(_example_path(relative_path)), backend=backend, format_output=False
        )


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
    if relative_path == "advanced/GenericPatternMatching.cgl":
        for marker in (
            "Option_Self",
            "Vec3_T",
            "Result_Vec3_T_MathError",
            "Result_T_MathError",
            "None(",
            ".add(",
            ".mul(",
            "T::one",
            "T::zero",
            "safe_divide(",
        ):
            assert marker not in generated
        assert re.search(r"\bT\b", generated) is None
        assert re.search(r"\bstr\b", generated) is None


@pytest.mark.parametrize("relative_path,backend", ADDITIONAL_FIXED_CASES)
def test_additional_fixed_examples_translate(relative_path, backend):
    generated = crosstl.translate(
        str(_example_path(relative_path)), backend=backend, format_output=False
    )

    _assert_generated_output_is_usable(generated)
    if backend == "slang":
        assert "unsupported Slang match" not in generated


@pytest.mark.parametrize(
    "relative_path,backend,backend_label", GENERIC_FUNCTION_UNSUPPORTED_BACKEND_CASES
)
def test_generic_function_examples_report_backend_diagnostics(
    relative_path, backend, backend_label
):
    with pytest.raises(
        ValueError,
        match=rf"{backend_label} codegen does not support generic functions",
    ):
        crosstl.translate(
            str(_example_path(relative_path)), backend=backend, format_output=False
        )


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

import re
import shutil
import subprocess
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

MOJO_PLACEHOLDER_MARKERS = (
    "# CrossGL builtin placeholders",
    "# CrossGL GPU builtin placeholders",
    "# CrossGL resource placeholders",
    "# CrossGL geometry stream placeholders",
    "# CrossGL tessellation patch placeholders",
    "# CrossGL mesh/task placeholders",
    "# CrossGL synchronization placeholders",
    "# CrossGL wave/subgroup placeholders",
    "# CrossGL ray tracing placeholders",
)
MOJO_PLACEHOLDER_EXAMPLE_CASES = (
    (
        "compute/ParticleSimulation.cgl",
        "mojo",
        ("# CrossGL synchronization placeholders",),
    ),
    (
        "gpu_computing/MatrixMultiplication.cgl",
        "mojo",
        (
            "# CrossGL synchronization placeholders",
            "# CrossGL wave/subgroup placeholders",
        ),
    ),
    (
        "cross_platform/UniversalPBRShader.cgl",
        "mojo",
        ("# CrossGL resource placeholders",),
    ),
    (
        "graphics/ComplexShader.cgl",
        "mojo",
        ("# CrossGL resource placeholders",),
    ),
)
MOJO_PLACEHOLDER_EXAMPLE_CASE_KEYS = {
    (relative_path, backend)
    for relative_path, backend, _markers in MOJO_PLACEHOLDER_EXAMPLE_CASES
}
FULL_BACKEND_EXAMPLE_CASES = tuple(
    (relative_path, backend)
    for relative_path in FULL_BACKEND_EXAMPLES
    for backend in codegen.backend_names()
    if (relative_path, backend) not in WEBGL_COMPUTE_EXAMPLE_DIAGNOSTIC_CASES
    and (relative_path, backend) not in MOJO_PLACEHOLDER_EXAMPLE_CASE_KEYS
)

KNOWN_PRIMARY_GRAPHICS_GAPS = ()

PRIMARY_GRAPHICS_FIXED_CASES = (
    ("advanced/GenericPatternMatching.cgl", "directx"),
    ("advanced/GenericPatternMatching.cgl", "metal"),
    ("advanced/GenericPatternMatching.cgl", "opengl"),
    ("advanced/GenericPatternMatching.cgl", "webgl"),
    ("cross_platform/UniversalPBRShader.cgl", "directx"),
    ("cross_platform/UniversalPBRShader.cgl", "metal"),
    ("cross_platform/UniversalPBRShader.cgl", "opengl"),
    ("cross_platform/UniversalPBRShader.cgl", "webgl"),
    ("graphics/ComplexShader.cgl", "directx"),
    ("graphics/ComplexShader.cgl", "metal"),
    ("graphics/ComplexShader.cgl", "opengl"),
    ("graphics/ComplexShader.cgl", "webgl"),
    ("graphics/ComplexShader.cgl", "wgsl"),
)

ADDITIONAL_FIXED_CASES = (("cross_platform/UniversalPBRShader.cgl", "slang"),)

GENERIC_FUNCTION_UNSUPPORTED_BACKEND_CASES = (
    (
        "advanced/GenericPatternMatching.cgl",
        "vulkan",
        "SPIR-V codegen does not support generic functions: unspecialized "
        "generic helper 'vector_operation' with generic parameters (T); "
        "specialize the function before SPIR-V generation",
    ),
    (
        "advanced/GenericPatternMatching.cgl",
        "cuda",
        "CUDA codegen does not support generic functions",
    ),
    (
        "advanced/GenericPatternMatching.cgl",
        "hip",
        "HIP codegen does not support generic functions",
    ),
    (
        "advanced/GenericPatternMatching.cgl",
        "mojo",
        "Mojo codegen does not support generic functions",
    ),
    (
        "advanced/GenericPatternMatching.cgl",
        "slang",
        "Slang codegen does not support generic functions",
    ),
)

KNOWN_PRIMARY_GRAPHICS_DIAGNOSTICS = (
    (
        "advanced/GenericPatternMatching.cgl",
        "wgsl",
        ValueError,
        "WGSL target does not support generic structs yet",
    ),
    (
        "cross_platform/UniversalPBRShader.cgl",
        "wgsl",
        ValueError,
        "WGSL target does not support resource arrays of sampler2D; "
        "WebGPU/WGSL requires texture, sampler, image, and storage-buffer "
        "resources to be declared as individual module-scope bindings",
    ),
)


def _example_path(relative_path):
    return EXAMPLES_ROOT / relative_path


def _assert_generated_output_has_no_internal_failures(generated):
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


def _assert_generated_output_is_usable(generated):
    _assert_generated_output_has_no_internal_failures(generated)
    for marker in MOJO_PLACEHOLDER_MARKERS:
        assert marker not in generated


@pytest.mark.parametrize("example_path", sorted(EXAMPLES_ROOT.rglob("*.cgl")))
def test_checked_in_examples_parse_as_crossgl(example_path):
    crosstl.translator.parse(example_path.read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    "spvasm_path",
    sorted((EXAMPLES_ROOT / "output" / "vulkan").rglob("*.spvasm")),
)
def test_checked_in_vulkan_example_outputs_validate_with_spirv_tools(
    tmp_path, spvasm_path
):
    spirv_as = shutil.which("spirv-as")
    spirv_val = shutil.which("spirv-val")
    if spirv_as is None or spirv_val is None:
        pytest.skip("spirv-as and spirv-val are not installed")

    binary_path = tmp_path / f"{spvasm_path.stem}.spv"
    assemble = subprocess.run(
        [spirv_as, str(spvasm_path), "-o", str(binary_path)],
        capture_output=True,
        text=True,
    )
    assert assemble.returncode == 0, assemble.stderr

    validate = subprocess.run(
        [spirv_val, str(binary_path)],
        capture_output=True,
        text=True,
    )
    assert validate.returncode == 0, validate.stderr


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


@pytest.mark.parametrize(
    "relative_path,backend,expected_markers", MOJO_PLACEHOLDER_EXAMPLE_CASES
)
def test_mojo_placeholder_examples_are_tracked(
    relative_path, backend, expected_markers
):
    generated = crosstl.translate(
        str(_example_path(relative_path)), backend=backend, format_output=False
    )

    _assert_generated_output_has_no_internal_failures(generated)
    for marker in expected_markers:
        assert marker in generated


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
    "relative_path,backend,message", GENERIC_FUNCTION_UNSUPPORTED_BACKEND_CASES
)
def test_generic_function_examples_report_backend_diagnostics(
    relative_path, backend, message
):
    with pytest.raises(ValueError, match=re.escape(message)):
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

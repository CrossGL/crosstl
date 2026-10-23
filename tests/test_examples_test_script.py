import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

import crosstl.translator.codegen as codegen

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "examples" / "test.py"


def load_examples_test_module():
    spec = importlib.util.spec_from_file_location("examples_test_script", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_example_catalog_covers_all_committed_source_examples():
    module = load_examples_test_module()
    catalog_paths = {
        module.EXAMPLES_ROOT / category / example
        for category, examples in module.EXAMPLES_BY_CATEGORY.items()
        for example in examples
    }
    source_paths = {
        path
        for path in module.EXAMPLES_ROOT.glob("*/*.cgl")
        if path.parent.name != "output"
    }

    assert catalog_paths == source_paths


def test_backend_compatibility_and_rejections_reference_scheduled_examples():
    module = load_examples_test_module()

    assert set(module.BACKEND_COMPATIBILITY) == set(module.EXAMPLES_BY_CATEGORY)
    for category, backends in module.BACKEND_COMPATIBILITY.items():
        assert set(backends) <= set(module.BACKENDS), category
    assert set(module.EXPECTED_REJECTIONS) <= set(module.scheduled_cases())


def test_examples_runner_schedules_all_registered_backend_targets():
    module = load_examples_test_module()

    assert set(module.BACKENDS) == set(codegen.backend_names())


def test_validate_generated_output_rejects_internal_ast_repr(tmp_path):
    module = load_examples_test_module()
    output = tmp_path / "shader.mojo"
    output.write_text(
        "// generated\n"
        + ("x" * 120)
        + "\nvar tile: float[IdentifierNode(name=TILE_SIZE)]\n",
        encoding="utf-8",
    )

    assert (
        module.validate_generated_output(output)
        == "Output contains internal AST representation: IdentifierNode("
    )


def test_generic_pattern_matching_cli_reports_documented_generic_gaps():
    source = ROOT / "examples" / "advanced" / "GenericPatternMatching.cgl"
    expected_diagnostics = {
        "cuda": "CUDA codegen cannot emit unresolved generic parameter 'T'",
        "mojo": "generic payload enum specializations must be concrete",
        "slang": "Slang codegen cannot emit unresolved generic parameter 'T'",
        # SPIR-V has no generics: the unspecialised generic helper is refused
        # before any of its `T::zero()` calls could be lowered to a default value.
        "vulkan": (
            "SPIR-V codegen does not support generic functions: unspecialized "
            "generic helper 'vector_operation' with generic parameters (T)"
        ),
    }

    for backend, diagnostic in expected_diagnostics.items():
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "crosstl._crosstl",
                "translate",
                str(source),
                "--backend",
                backend,
                "--no-format",
                "--output",
                "-",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        combined_output = result.stdout + result.stderr

        assert result.returncode != 0
        assert diagnostic in combined_output

    for backend in ("hip",):
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "crosstl._crosstl",
                "translate",
                str(source),
                "--backend",
                backend,
                "--no-format",
                "--output",
                "-",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, (result.stdout + result.stderr)[-4000:]


@pytest.fixture
def one_example(monkeypatch, tmp_path):
    """Point the runner at a single graphics example and the Metal backend."""
    module = load_examples_test_module()
    examples_root = tmp_path / "examples"
    source = examples_root / "graphics" / "SimpleShader.cgl"
    source.parent.mkdir(parents=True)
    source.write_text("shader SimpleShader {}", encoding="utf-8")

    monkeypatch.setattr(module, "EXAMPLES_ROOT", examples_root)
    monkeypatch.setattr(
        module, "EXAMPLES_BY_CATEGORY", {"graphics": ["SimpleShader.cgl"]}
    )
    monkeypatch.setattr(module, "BACKENDS", {"metal": ".metal"})
    monkeypatch.setattr(module, "BACKEND_COMPATIBILITY", {"graphics": ["metal"]})
    monkeypatch.setattr(module, "EXPECTED_REJECTIONS", {})
    return module


def write_valid_output(input_path, backend, save_shader=None):
    assert Path(input_path).is_absolute()
    assert backend == "metal"
    Path(save_shader).write_text("//" + ("x" * 120), encoding="utf-8")
    return "x" * 120


def refuse(*args, **kwargs):
    raise ValueError("forced rejection")


def test_main_is_independent_of_current_working_directory(
    one_example, monkeypatch, tmp_path
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(one_example.crosstl, "translate", write_valid_output)

    assert one_example.main([]) == 0
    assert (
        one_example.EXAMPLES_ROOT
        / "output"
        / "metal"
        / "graphics"
        / "SimpleShader.metal"
    ).exists()


def test_main_can_write_outputs_outside_committed_examples_tree(
    one_example, monkeypatch, tmp_path
):
    output_root = tmp_path / "generated"
    monkeypatch.setattr(one_example.crosstl, "translate", write_valid_output)

    assert one_example.main(["--output-root", str(output_root)]) == 0
    assert (output_root / "metal" / "graphics" / "SimpleShader.metal").exists()
    assert not (one_example.EXAMPLES_ROOT / "output").exists()


def test_main_fails_on_any_unexpected_rejection(one_example, monkeypatch, capsys):
    monkeypatch.setattr(one_example.crosstl, "translate", refuse)

    assert one_example.main([]) == 1
    captured = capsys.readouterr()
    assert "0 passed, 0 rejected as expected, 1 failed" in captured.out
    assert (
        "graphics/SimpleShader -> metal: ValueError: forced rejection" in captured.err
    )


def test_main_fails_on_output_that_does_not_validate(one_example, monkeypatch, capsys):
    def write_small_output(input_path, backend, save_shader=None):
        Path(save_shader).write_text("//", encoding="utf-8")

    monkeypatch.setattr(one_example.crosstl, "translate", write_small_output)

    assert one_example.main([]) == 1
    assert "Output file too small" in capsys.readouterr().err


def test_main_accepts_a_listed_rejection(one_example, monkeypatch, capsys):
    monkeypatch.setattr(
        one_example,
        "EXPECTED_REJECTIONS",
        {("graphics", "SimpleShader", "metal"): "the target cannot express it"},
    )
    monkeypatch.setattr(one_example.crosstl, "translate", refuse)

    assert one_example.main([]) == 0
    assert "0 passed, 1 rejected as expected, 0 failed" in capsys.readouterr().out


def test_main_fails_when_a_listed_rejection_translates(
    one_example, monkeypatch, capsys
):
    monkeypatch.setattr(
        one_example,
        "EXPECTED_REJECTIONS",
        {("graphics", "SimpleShader", "metal"): "the target cannot express it"},
    )
    monkeypatch.setattr(one_example.crosstl, "translate", write_valid_output)

    assert one_example.main([]) == 1
    assert "delete its EXPECTED_REJECTIONS entry" in capsys.readouterr().err

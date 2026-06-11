import importlib.util
import json
import sys
from pathlib import Path

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


def test_backend_compatibility_and_skips_reference_scheduled_examples():
    module = load_examples_test_module()
    scheduled = {
        (category, Path(example).stem, backend)
        for category, examples in module.EXAMPLES_BY_CATEGORY.items()
        for example in examples
        for backend in module.BACKEND_COMPATIBILITY.get(category, module.BACKENDS)
    }

    assert set(module.BACKEND_COMPATIBILITY) == set(module.EXAMPLES_BY_CATEGORY)
    for category, backends in module.BACKEND_COMPATIBILITY.items():
        assert set(backends) <= set(module.BACKENDS), category
    assert set(module.EXAMPLE_BACKEND_SKIPS) <= scheduled


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


def test_main_can_write_outputs_outside_committed_examples_tree(monkeypatch, tmp_path):
    module = load_examples_test_module()
    examples_root = tmp_path / "examples"
    source = examples_root / "graphics" / "SimpleShader.cgl"
    source.parent.mkdir(parents=True)
    source.write_text("shader SimpleShader {}", encoding="utf-8")
    output_root = tmp_path / "generated"
    summary_path = tmp_path / "summary.json"

    monkeypatch.setattr(module, "EXAMPLES_ROOT", examples_root)
    monkeypatch.setattr(
        module, "EXAMPLES_BY_CATEGORY", {"graphics": ["SimpleShader.cgl"]}
    )
    monkeypatch.setattr(module, "BACKENDS", {"metal": ".metal"})
    monkeypatch.setattr(module, "BACKEND_COMPATIBILITY", {"graphics": ["metal"]})
    monkeypatch.setattr(module, "EXAMPLE_BACKEND_SKIPS", {})
    monkeypatch.setattr(
        module,
        "test_cross_backend_consistency",
        lambda: {"successful": 0, "total": 0},
    )

    def translate(input_path, backend, save_shader=None):
        assert Path(input_path).is_absolute()
        assert backend == "metal"
        if save_shader:
            Path(save_shader).write_text("//" + ("x" * 120), encoding="utf-8")
        return "x" * 120

    monkeypatch.setattr(module.crosstl, "translate", translate)

    assert (
        module.main(
            [
                "--output-root",
                str(output_root),
                "--summary-json",
                str(summary_path),
            ]
        )
        == 0
    )

    assert (output_root / "metal" / "graphics" / "SimpleShader.metal").exists()
    assert not (examples_root / "output").exists()
    assert json.loads(summary_path.read_text(encoding="utf-8"))["successful"] == 1

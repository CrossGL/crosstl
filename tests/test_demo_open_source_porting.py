import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEMO_ROOT = ROOT / "demos" / "open-source-porting"
CASE_ROOT = DEMO_ROOT / "cases"


def _load_demo_runner():
    spec = importlib.util.spec_from_file_location(
        "open_source_demo", DEMO_ROOT / "run_demo.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_open_source_demo_cases_have_pinned_manifests_and_references():
    runner = _load_demo_runner()
    case_dirs = sorted(path for path in CASE_ROOT.iterdir() if path.is_dir())

    assert {path.name for path in case_dirs} == {
        "apple-modern-rendering-mesh-viewdir",
        "directx-graphics-samples-hello-triangle",
        "lonelydevil-vulkan-tutorial-triangle",
        "metal-performance-testing-matmul",
        "opencl-sdk-saxpy",
        "raylib-base-fragment",
        "raylib-base-vertex",
        "sascha-willems-vulkan-conservative-triangle",
        "slang-hello-world-compute",
        "spirv-tools-basic-src",
        "vulkan-samples-dynamic-line-grid",
    }

    for case_dir in case_dirs:
        config_targets = set(runner._case_targets(case_dir))
        manifest = json.loads((case_dir / "corpus.json").read_text(encoding="utf-8"))

        assert manifest["schemaVersion"] == 1
        assert manifest["entries"], case_dir.name
        assert not (case_dir / "crosstl-out" / "portability-report.json").exists()

        for entry in manifest["entries"]:
            assert (case_dir / entry["path"]).is_file()
            assert entry["sourceUrl"].startswith(entry["repository"])
            assert re.fullmatch(r"[0-9a-f]{40}", entry["commit"])
            assert set(entry["targets"]).issubset(config_targets)

        for target in config_targets:
            target_dir = case_dir / "crosstl-out" / target
            assert target_dir.is_dir(), f"{case_dir.name} missing {target} references"
            assert any(path.is_file() for path in target_dir.rglob("*"))


def test_open_source_demo_workflow_runs_platform_toolchain_smokes():
    workflow = (ROOT / ".github" / "workflows" / "demo.yml").read_text(encoding="utf-8")

    assert "os: [ubuntu-latest, macOS-latest, windows-latest]" in workflow
    assert "glslang-tools spirv-tools" in workflow
    assert "brew install glslang spirv-tools" in workflow
    assert "DirectXShaderCompiler/releases/download/v1.9.2602.24" in workflow
    assert "--run-toolchains" in workflow
    assert "--require-toolchain-runs" in workflow
    assert "--target opengl" in workflow
    assert "--target vulkan" in workflow
    assert "--target metal" in workflow
    assert "--target directx" in workflow
    assert "macOS Metal compile references" in workflow
    assert "spirv-tools-basic-src/crosstl-out/metal/basic_src.metal" in workflow
    assert "demo-reports-${{ matrix.os }}" in workflow


def _workflow_step_block(workflow: str, step_name: str) -> str:
    marker = f"      - name: {step_name}"
    start = workflow.index(marker)
    next_step = workflow.find("\n      - name:", start + len(marker))
    return workflow[start:] if next_step == -1 else workflow[start:next_step]


def _workflow_step_cases(workflow: str, step_name: str) -> set[str]:
    return set(
        re.findall(r"--case ([a-z0-9-]+)", _workflow_step_block(workflow, step_name))
    )


def test_open_source_demo_workflow_case_smoke_lists_match_checked_targets():
    workflow = (ROOT / ".github" / "workflows" / "demo.yml").read_text(encoding="utf-8")

    assert _workflow_step_cases(workflow, "Linux OpenGL and Vulkan smoke checks") == {
        "apple-modern-rendering-mesh-viewdir",
        "directx-graphics-samples-hello-triangle",
        "lonelydevil-vulkan-tutorial-triangle",
        "metal-performance-testing-matmul",
        "opencl-sdk-saxpy",
        "raylib-base-fragment",
        "raylib-base-vertex",
        "sascha-willems-vulkan-conservative-triangle",
        "slang-hello-world-compute",
        "spirv-tools-basic-src",
        "vulkan-samples-dynamic-line-grid",
    }
    assert _workflow_step_cases(workflow, "macOS Metal smoke checks") == {
        "apple-modern-rendering-mesh-viewdir",
        "directx-graphics-samples-hello-triangle",
        "metal-performance-testing-matmul",
        "opencl-sdk-saxpy",
        "raylib-base-fragment",
        "raylib-base-vertex",
        "sascha-willems-vulkan-conservative-triangle",
        "slang-hello-world-compute",
        "spirv-tools-basic-src",
        "vulkan-samples-dynamic-line-grid",
    }
    assert _workflow_step_cases(workflow, "Windows DirectX smoke checks") == {
        "apple-modern-rendering-mesh-viewdir",
        "directx-graphics-samples-hello-triangle",
        "slang-hello-world-compute",
        "lonelydevil-vulkan-tutorial-triangle",
        "vulkan-samples-dynamic-line-grid",
    }


def test_open_source_demo_artifact_comparison_normalizes_platform_text(tmp_path):
    runner = _load_demo_runner()
    lf_text = tmp_path / "shader.glsl"
    crlf_text = tmp_path / "shader-windows.glsl"
    lf_source_map = tmp_path / "shader.source-remap.json"
    windows_source_map = tmp_path / "shader-windows.source-remap.json"

    lf_text.write_bytes(b"void main() {\n}\n")
    crlf_text.write_bytes(b"void main() {\r\n}\r\n\r\n")
    lf_source_map.write_text(
        json.dumps(
            {
                "generatedFile": "crosstl-out/cgl/shader.cgl",
                "mappings": [{"generated": {"line": 1, "offset": 0}}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    windows_source_map.write_text(
        json.dumps(
            {
                "generatedFile": r"crosstl-out\cgl\shader.cgl",
                "mappings": [{"generated": {"line": 1, "offset": 3}}],
            }
        )
        + "\r\n",
        encoding="utf-8",
    )

    assert runner._comparison_bytes(lf_text) == runner._comparison_bytes(crlf_text)
    assert runner._comparison_bytes(lf_source_map) == runner._comparison_bytes(
        windows_source_map
    )


def test_open_source_demo_runner_verifies_fast_reference_subset():
    result = subprocess.run(
        [
            sys.executable,
            "demos/open-source-porting/run_demo.py",
            "--check",
            "--case",
            "directx-graphics-samples-hello-triangle",
            "--target",
            "cgl",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "directx-graphics-samples-hello-triangle: verified cgl" in result.stdout

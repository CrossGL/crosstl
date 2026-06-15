import ast
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEMO_ROOT = ROOT / "demos" / "open-source-porting"
CASE_ROOT = DEMO_ROOT / "cases"
DEMO_CI_METADATA_PATH = ROOT / "support" / "demo-ci-metadata.json"
DEMO_CI_TOOL_PATH = ROOT / "tools" / "demo_ci_metadata.py"
DEMO_WORKFLOW_PATH = ROOT / ".github" / "workflows" / "demo.yml"


def _load_demo_runner():
    spec = importlib.util.spec_from_file_location(
        "open_source_demo", DEMO_ROOT / "run_demo.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_demo_ci_tool():
    spec = importlib.util.spec_from_file_location("demo_ci_metadata", DEMO_CI_TOOL_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _test_function_names(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("test_")
    }


def _upstream_source_path(entry):
    source_prefix = f"{entry['repository']}/blob/{entry['commit']}/"
    if not entry["sourceUrl"].startswith(source_prefix):
        raise AssertionError(f"sourceUrl does not match repository/commit: {entry}")
    return entry["sourceUrl"][len(source_prefix) :]


def test_open_source_demo_third_party_notices_cover_manifest_sources():
    notices = (DEMO_ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
    notice_blocks = [
        block for block in re.split(r"(?m)^## ", notices) if block.strip()
    ]
    missing = []

    for case_dir in sorted(path for path in CASE_ROOT.iterdir() if path.is_dir()):
        manifest = json.loads((case_dir / "corpus.json").read_text(encoding="utf-8"))

        for entry in manifest["entries"]:
            source_path = _upstream_source_path(entry)
            covered = any(
                entry["repository"] in block
                and entry["commit"] in block
                and (source_path in block or entry["sourceUrl"] in block)
                for block in notice_blocks
            )
            if not covered:
                missing.append(
                    f"{case_dir.name}:{entry['id']} missing {source_path}"
                )

    assert not missing, (
        "Missing third-party notice coverage for manifest sources:\n"
        + "\n".join(missing)
    )


def test_open_source_demo_case_targets_use_toml_project_config(tmp_path):
    runner = _load_demo_runner()
    (tmp_path / "crosstl.toml").write_text(
        """
        [project]
        output_dir = "crosstl-out"
        targets = [
            "cgl", # checked source-form target
            "opengl",
            # keep this trailing entry on a separate line
            "vulkan",
        ]

        [project.sources]
        "shaders/*.hlsl" = "directx"
        """,
        encoding="utf-8",
    )

    assert runner._case_targets(tmp_path) == ["cgl", "opengl", "vulkan"]

def test_open_source_demo_cases_have_pinned_manifests_and_references():
    runner = _load_demo_runner()
    case_dirs = sorted(path for path in CASE_ROOT.iterdir() if path.is_dir())

    assert {path.name for path in case_dirs} == {
        "angle-simple-texture-2d",
        "apple-modern-rendering-mesh-viewdir",
        "arm-opengl-es-sdk-cube",
        "glslang-push-constant-vertex",
        "glslang-spec-constant-vertex",
        "directx-graphics-samples-hello-const-buffers",
        "directx-graphics-samples-hello-triangle",
        "directx-graphics-samples-hello-texture",
        "directx-shader-compiler-groupshared-splat",
        "directx-shader-compiler-neg1",
        "directx-sdk-samples-tutorial02",
        "diligent-samples-tutorial02-cube",
        "diligent-samples-vrs-cube",
        "glfw-opengl-triangle",
        "godot-betsy-alpha-stitch",
        "libgdx-batch-shader",
        "lonelydevil-vulkan-tutorial-triangle",
        "monogame-sprite-effect",
        "metal-performance-testing-matmul",
        "nvidia-cuda-samples-vector-add",
        "nvpro-vk-mini-samples-rectangle",
        "ogl-samples-flat-color",
        "opencl-sdk-reduce",
        "opencl-sdk-saxpy",
        "openframeworks-noise-shader",
        "rocm-examples-add-kernel",
        "rocm-examples-bit-extract",
        "raylib-base-fragment",
        "raylib-base-vertex",
        "raylib-lighting-shader-pair",
        "renderdoc-vktext-fragment",
        "rust-gpu-compute-collatz",
        "rust-gpu-graphics-stage-inputs",
        "rust-gpu-vulkan-examples-triangle-overlay",
        "sascha-willems-vulkan-conservative-triangle",
        "sascha-willems-vulkan-headless-compute",
        "slang-default-parameter-compute",
        "slang-hello-world-compute",
        "spirv-cross-round-fragment",
        "spirv-tools-basic-src",
        "vulkan-tools-cube",
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

        assert {
            target for entry in manifest["entries"] for target in entry["targets"]
        } == config_targets

        output_targets = {
            target_dir.name
            for target_dir in (case_dir / "crosstl-out").iterdir()
            if target_dir.is_dir()
            and any(path.is_file() for path in target_dir.rglob("*"))
        }
        assert output_targets == config_targets

        for target in config_targets:
            target_dir = case_dir / "crosstl-out" / target
            assert target_dir.is_dir(), f"{case_dir.name} missing {target} references"
            assert any(path.is_file() for path in target_dir.rglob("*"))


def test_open_source_demo_workflow_runs_platform_toolchain_smokes():
    workflow = DEMO_WORKFLOW_PATH.read_text(encoding="utf-8")

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
    assert "Windows DirectX compile references" in workflow
    assert 'dxc -T "$profile" -E "$entry"' in workflow
    assert "--emit-case-args opengl" in workflow
    assert "--emit-case-args vulkan" in workflow
    assert "--emit-case-args metal" in workflow
    assert "--emit-case-args directx" in workflow
    assert "--emit-artifact-paths metal" in workflow
    assert "--emit-directx-compile-jobs" in workflow
    assert "profile=\"${profile%$'\\r'}\"" in workflow
    assert "open-source-porting-demo-reports-${{ matrix.os }}" in workflow


def _workflow_step_block(workflow: str, step_name: str) -> str:
    marker = f"      - name: {step_name}"
    start = workflow.index(marker)
    next_step = workflow.find("\n      - name:", start + len(marker))
    return workflow[start:] if next_step == -1 else workflow[start:next_step]


def _cases_for_target(runner, target: str) -> set[str]:
    return {
        case_dir.name
        for case_dir in CASE_ROOT.iterdir()
        if case_dir.is_dir() and target in runner._case_targets(case_dir)
    }


def _assert_workflow_step_uses_case_generator(
    workflow: str, step_name: str, target: str
) -> None:
    block = _workflow_step_block(workflow, step_name)
    assert f"--emit-case-args {target}" in block
    assert "--case " not in block


def test_open_source_demo_runner_case_arg_generation_matches_checked_targets():
    runner = _load_demo_runner()

    for target in ("opengl", "vulkan", "metal", "directx"):
        args = runner._case_args_for_target(target)
        assert args[::2] == ["--case"] * (len(args) // 2)
        assert set(args[1::2]) == _cases_for_target(runner, target)


def test_open_source_demo_workflow_uses_generated_case_smoke_lists():
    workflow = DEMO_WORKFLOW_PATH.read_text(encoding="utf-8")

    _assert_workflow_step_uses_case_generator(
        workflow, "Linux OpenGL smoke checks", "opengl"
    )
    _assert_workflow_step_uses_case_generator(
        workflow, "Linux Vulkan smoke checks", "vulkan"
    )
    _assert_workflow_step_uses_case_generator(
        workflow, "macOS Metal smoke checks", "metal"
    )
    _assert_workflow_step_uses_case_generator(
        workflow, "Windows DirectX smoke checks", "directx"
    )


def test_open_source_demo_workflow_compile_reference_paths_exist():
    runner = _load_demo_runner()
    workflow = DEMO_WORKFLOW_PATH.read_text(encoding="utf-8")

    metal_block = _workflow_step_block(workflow, "macOS Metal compile references")
    assert "--emit-artifact-paths metal" in metal_block
    assert "--artifact-suffix .metal" in metal_block
    metal_paths = set(runner._artifact_paths_for_target("metal", ".metal"))
    assert {
        str(path.relative_to(ROOT))
        for path in CASE_ROOT.glob("*/crosstl-out/metal/*.metal")
    } == metal_paths
    assert all((ROOT / path).is_file() for path in metal_paths)

    directx_block = _workflow_step_block(workflow, "Windows DirectX compile references")
    assert "--emit-directx-compile-jobs" in directx_block
    directx_jobs = runner._directx_compile_jobs()
    directx_paths = {path for path, _entry, _profile in directx_jobs}
    assert {
        str(path.relative_to(ROOT))
        for path in CASE_ROOT.glob("*/crosstl-out/directx/*.hlsl")
    } == directx_paths
    assert all((ROOT / path).is_file() for path in directx_paths)
    assert (
        "demos/open-source-porting/cases/diligent-samples-vrs-cube/"
        "crosstl-out/directx/CubeFDM_fs.hlsl",
        "PSMain",
        "ps_6_4",
    ) in directx_jobs
    sprite_path = (
        "demos/open-source-porting/cases/monogame-sprite-effect/"
        "crosstl-out/directx/SpriteEffect.hlsl"
    )
    assert (sprite_path, "VSMain", "vs_6_0") in directx_jobs
    assert (sprite_path, "PSMain", "ps_6_0") in directx_jobs


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


def test_open_source_demo_runner_requires_toolchain_runs_per_selected_target(tmp_path):
    runner = _load_demo_runner()
    original_run = runner.subprocess.run

    class Completed:
        returncode = 0
        stdout = json.dumps(
            {
                "success": True,
                "diagnosticCounts": {},
                "toolchainRunStatusCounts": {"ok": 1, "failed": 0},
                "toolchainRunStatusByTarget": {
                    "opengl": {"okCount": 1, "failedCount": 0, "runCount": 1}
                },
            }
        )
        stderr = ""

    def fake_run(*args, **kwargs):
        return Completed()

    runner.subprocess.run = fake_run
    try:
        try:
            runner._validate_report(
                tmp_path / "report.json",
                run_toolchains=True,
                require_toolchain_runs=True,
                selected_targets=["opengl", "vulkan"],
                reports_dir=None,
                case_name="example",
            )
        except SystemExit as exc:
            assert "vulkan: ok=0, failed=0" in str(exc)
        else:
            raise AssertionError("Expected missing target toolchain run to fail")
    finally:
        runner.subprocess.run = original_run


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


def test_open_source_demo_runner_ignores_trailing_text_artifact_whitespace(tmp_path):
    runner = _load_demo_runner()
    expected = tmp_path / "expected.spvasm"
    actual = tmp_path / "actual.spvasm"
    expected.write_text("OpFunctionCall %8 %20\nOpReturn\n", encoding="utf-8")
    actual.write_text("OpFunctionCall %8 %20 \r\nOpReturn\t\r\n", encoding="utf-8")

    assert runner._comparison_bytes(expected) == runner._comparison_bytes(actual)


def test_demo_ci_metadata_matches_checked_in_pytest_cases():
    tool = _load_demo_ci_tool()
    metadata = tool.load_metadata(DEMO_CI_METADATA_PATH)
    test_file = ROOT / metadata["pytest"]["test_file"]
    test_names = _test_function_names(test_file)

    assert tool.pytest_files(metadata) == [metadata["pytest"]["test_file"]]
    assert metadata["pytest"]["cases"]

    for case in metadata["pytest"]["cases"]:
        selector = case["selector"]
        matched_tests = sorted(name for name in test_names if selector in name)

        assert matched_tests, selector
        assert sorted(case["tests"]) == matched_tests
        assert case["targets"]
        assert all(target == target.lower() for target in case["targets"])


def test_demo_ci_metadata_generates_workflow_pytest_inputs():
    tool = _load_demo_ci_tool()
    metadata = tool.load_metadata(DEMO_CI_METADATA_PATH)
    selectors = [case["selector"] for case in metadata["pytest"]["cases"]]

    assert tool.pytest_selectors(metadata) == selectors
    assert tool.pytest_selector_expression(metadata) == " or ".join(selectors)


def test_demo_workflow_consumes_generated_demo_ci_inputs():
    metadata = json.loads(DEMO_CI_METADATA_PATH.read_text(encoding="utf-8"))
    workflow = DEMO_WORKFLOW_PATH.read_text(encoding="utf-8")

    assert "tools/demo_ci_metadata.py emit-pytest-files" in workflow
    assert "tools/demo_ci_metadata.py emit-pytest-selector" in workflow
    assert '"${demo_test_files[@]}"' in workflow
    assert '-k "$demo_selector"' in workflow

    for case in metadata["pytest"]["cases"]:
        assert case["selector"] not in workflow


def test_demo_ci_metadata_cli_check_and_emitters():
    check = subprocess.run(
        [sys.executable, "tools/demo_ci_metadata.py", "check"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert check.returncode == 0, check.stdout + check.stderr
    assert "demo CI metadata is valid" in check.stdout

    files = subprocess.run(
        [sys.executable, "tools/demo_ci_metadata.py", "emit-pytest-files"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=True,
    )
    selector = subprocess.run(
        [sys.executable, "tools/demo_ci_metadata.py", "emit-pytest-selector"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=True,
    )
    metadata = json.loads(DEMO_CI_METADATA_PATH.read_text(encoding="utf-8"))

    assert files.stdout.splitlines() == [metadata["pytest"]["test_file"]]
    assert selector.stdout.strip() == " or ".join(
        case["selector"] for case in metadata["pytest"]["cases"]
    )

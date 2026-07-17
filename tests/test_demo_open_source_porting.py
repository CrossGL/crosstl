import ast
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

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
    notice_blocks = [block for block in re.split(r"(?m)^## ", notices) if block.strip()]
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
                missing.append(f"{case_dir.name}:{entry['id']} missing {source_path}")

    assert (
        not missing
    ), "Missing third-party notice coverage for manifest sources:\n" + "\n".join(
        missing
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
    assert "--emit-case-args opengl" in workflow
    assert "--emit-case-args vulkan" in workflow
    assert "--emit-case-args metal" in workflow
    assert "--emit-case-args directx" in workflow
    assert "--emit-artifact-paths metal" in workflow
    assert "--compile-directx-references" in workflow
    assert '--compiler-output-dir "$out_dir"' in workflow
    assert "--emit-directx-compile-jobs" not in workflow
    assert "open-source-porting-demo-reports-${{ matrix.os }}" in workflow
    assert "write_demo_failure_summary" in workflow
    assert "--demo-step" in workflow
    assert "open-source-porting-demo-linux-opengl" in workflow
    assert "open-source-porting-demo-linux-vulkan" in workflow
    assert "open-source-porting-demo-macos-metal-compile" in workflow
    assert "open-source-porting-demo-windows-directx-compile" in workflow
    assert "support/generated/demo-reports/**/*-failure-summary.json" in workflow


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
    assert '--case="$case_name"' in block
    assert "write_demo_failure_summary \\" in block
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
    assert "--compile-directx-references" in directx_block
    assert '--compiler-output-dir "$out_dir"' in directx_block
    assert 'if ! "${command[@]}"; then' in directx_block
    assert "write_demo_failure_summary \\" in directx_block
    assert "exit 1" in directx_block
    assert "--emit-directx-compile-jobs" not in directx_block
    assert "while read -r shader entry profile" not in directx_block
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
    mesh_path = (
        "demos/open-source-porting/cases/apple-modern-rendering-mesh-viewdir/"
        "crosstl-out/directx/AAPLMeshRenderer.hlsl"
    )
    assert (mesh_path, "VSMain", "vs_6_2") in directx_jobs


def test_open_source_demo_directx_compile_command_uses_source_requirements(tmp_path):
    runner = _load_demo_runner()
    mesh_path = (
        "demos/open-source-porting/cases/apple-modern-rendering-mesh-viewdir/"
        "crosstl-out/directx/AAPLMeshRenderer.hlsl"
    )

    command = runner._directx_compile_command(
        mesh_path,
        "VSMain",
        "vs_6_0",
        output_dir=tmp_path,
    )

    assert command[:4] == ["dxc", "-T", "vs_6_2", "-enable-16bit-types"]
    assert command[4:7] == ["-E", "VSMain", mesh_path]
    assert command[7] == "-Fo"
    assert command[8] == str(
        tmp_path
        / "demos_open-source-porting_cases_apple-modern-rendering-mesh-viewdir_"
        "crosstl-out_directx_AAPLMeshRenderer_hlsl_VSMain_vs_6_2.dxil"
    )


def test_open_source_demo_directx_compile_references_require_dxc(tmp_path, monkeypatch):
    runner = _load_demo_runner()
    monkeypatch.setattr(runner.shutil, "which", lambda executable: None)

    with pytest.raises(SystemExit, match="dxc is required"):
        runner._compile_directx_references(tmp_path)


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
                "mappings": [{"generated": {"line": 1, "offset": 0}}],
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


def test_open_source_demo_artifact_comparison_detects_source_map_offset_drift(
    tmp_path,
):
    runner = _load_demo_runner()
    baseline = tmp_path / "baseline.source-remap.json"
    drifted = tmp_path / "drifted.source-remap.json"
    baseline.write_text(
        json.dumps(
            {
                "generatedFile": "crosstl-out/cgl/shader.cgl",
                "mappings": [
                    {
                        "generated": {
                            "line": 1,
                            "offset": 0,
                            "endOffset": 24,
                            "length": 24,
                        }
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    drifted.write_text(
        json.dumps(
            {
                "generatedFile": "crosstl-out/cgl/shader.cgl",
                "mappings": [
                    {
                        "generated": {
                            "line": 1,
                            "offset": 5,
                            "endOffset": 29,
                            "length": 24,
                        }
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert runner._comparison_bytes(baseline) != runner._comparison_bytes(drifted)


def test_open_source_demo_compare_artifacts_fails_on_source_map_offset_drift(
    tmp_path,
):
    runner = _load_demo_runner()
    case_dir = tmp_path / "case"
    work_dir = tmp_path / "work"
    for root, offset in ((case_dir, 0), (work_dir, 7)):
        target_dir = root / runner.OUTPUT_DIR_NAME / "cgl"
        target_dir.mkdir(parents=True)
        (target_dir / "shader.source-remap.json").write_text(
            json.dumps(
                {
                    "generatedFile": "crosstl-out/cgl/shader.cgl",
                    "mappings": [
                        {
                            "generated": {
                                "line": 1,
                                "offset": offset,
                                "endOffset": offset + 10,
                                "length": 10,
                            }
                        }
                    ],
                }
            )
            + "\n",
            encoding="utf-8",
        )

    try:
        runner._compare_artifacts(case_dir, work_dir, ["cgl"])
    except SystemExit as exc:
        assert "shader.source-remap.json" in str(exc)
    else:
        raise AssertionError("Expected source-map offset drift to fail verification")


def test_open_source_demo_corpus_entries_classify_translation_units():
    runner = _load_demo_runner()
    support_files = set()
    for case_dir in sorted(
        path for path in CASE_ROOT.iterdir() if (path / "crosstl.toml").is_file()
    ):
        assert runner._corpus_manifest_problems(case_dir) == [], case_dir.name
        translation_units = runner._translation_unit_paths(case_dir)
        manifest = json.loads((case_dir / "corpus.json").read_text(encoding="utf-8"))
        for entry in manifest["entries"]:
            role = entry.get("role", "translation_unit")
            assert role in {"translation_unit", "support_file"}, case_dir.name
            translated = entry["path"].replace("\\", "/") in translation_units
            if role == "translation_unit":
                assert translated, f"{case_dir.name}:{entry['path']}"
            else:
                assert not translated, f"{case_dir.name}:{entry['path']}"
                support_files.add(f"{case_dir.name}:{entry['path']}")

    assert support_files == {
        "metal-performance-testing-matmul:ShaderParams.h",
        "monogame-sprite-effect:Macros.fxh",
    }


def test_open_source_demo_corpus_manifest_validation_flags_role_mismatch(tmp_path):
    runner = _load_demo_runner()
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    (case_dir / "crosstl.toml").write_text(
        '[project]\ninclude = ["shader.glsl"]\n'
        'targets = ["cgl"]\noutput_dir = "crosstl-out"\n',
        encoding="utf-8",
    )
    (case_dir / "corpus.json").write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "entries": [
                    {"id": "a", "path": "missing.glsl"},
                    {"id": "b", "role": "support_file", "path": "shader.glsl"},
                ],
            }
        ),
        encoding="utf-8",
    )

    problems = runner._corpus_manifest_problems(case_dir)
    assert any("missing.glsl" in p and "translation_unit" in p for p in problems)
    assert any("shader.glsl" in p and "support_file" in p for p in problems)
    try:
        runner._validate_corpus_manifest(case_dir)
    except SystemExit as exc:
        assert "invalid corpus.json" in str(exc)
    else:
        raise AssertionError("Expected invalid corpus manifest to fail")


def test_open_source_demo_corpus_records_source_adjustments():
    runner = _load_demo_runner()
    adjusted = {}
    for case_dir in sorted(
        path for path in CASE_ROOT.iterdir() if (path / "crosstl.toml").is_file()
    ):
        manifest = json.loads((case_dir / "corpus.json").read_text(encoding="utf-8"))
        if "adjustments" not in manifest and "outOfScope" not in manifest:
            continue
        assert runner._corpus_manifest_problems(case_dir) == [], case_dir.name
        adjustments = manifest.get("adjustments", [])
        assert adjustments, case_dir.name
        for adjustment in adjustments:
            assert adjustment["kind"].strip()
            assert adjustment["summary"].strip()
        adjusted[case_dir.name] = {a["kind"] for a in adjustments}

    assert "godot-betsy-alpha-stitch" in adjusted
    assert "rocm-examples-add-kernel" in adjusted
    assert "rocm-examples-bit-extract" in adjusted
    assert "define-selection" in adjusted["rocm-examples-bit-extract"]


def test_open_source_demo_corpus_manifest_validation_flags_bad_adjustments(tmp_path):
    runner = _load_demo_runner()
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    (case_dir / "crosstl.toml").write_text(
        '[project]\ninclude = ["shader.glsl"]\n'
        'targets = ["cgl"]\noutput_dir = "crosstl-out"\n',
        encoding="utf-8",
    )
    (case_dir / "corpus.json").write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "adjustments": [{"kind": "", "summary": "  "}],
                "outOfScope": [""],
                "entries": [{"id": "a", "path": "shader.glsl"}],
            }
        ),
        encoding="utf-8",
    )

    problems = runner._corpus_manifest_problems(case_dir)
    assert any("adjustments[0] kind" in p for p in problems)
    assert any("adjustments[0] summary" in p for p in problems)
    assert any("outOfScope" in p for p in problems)


def test_open_source_demo_runner_requires_toolchain_runs_per_artifact(tmp_path):
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
                    "opengl": {"runCount": 2, "okCount": 1, "failedCount": 0}
                },
                "validation": {
                    "toolchainRuns": [
                        {
                            "target": "opengl",
                            "path": "crosstl-out/opengl/a.glsl",
                            "checkKind": "artifact",
                            "status": "ok",
                        },
                        {
                            "target": "opengl",
                            "path": "crosstl-out/opengl/b.glsl",
                            "checkKind": "artifact",
                            "status": "skipped",
                        },
                    ]
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
                selected_targets=["opengl"],
                reports_dir=None,
                case_name="example",
            )
        except SystemExit as exc:
            assert "crosstl-out/opengl/b.glsl" in str(exc)
            assert "crosstl-out/opengl/a.glsl" not in str(exc)
        else:
            raise AssertionError(
                "Expected an unvalidated translated artifact to fail the gate"
            )
    finally:
        runner.subprocess.run = original_run

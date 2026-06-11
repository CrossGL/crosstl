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
        "angle-simple-texture-2d",
        "apple-modern-rendering-mesh-viewdir",
        "arm-opengl-es-sdk-cube",
        "glslang-push-constant-vertex",
        "glslang-spec-constant-vertex",
        "directx-graphics-samples-hello-triangle",
        "directx-graphics-samples-hello-texture",
        "directx-shader-compiler-neg1",
        "directx-sdk-samples-tutorial02",
        "diligent-samples-vrs-cube-vertex",
        "godot-betsy-alpha-stitch",
        "libgdx-batch-shader",
        "lonelydevil-vulkan-tutorial-triangle",
        "monogame-sprite-effect",
        "metal-performance-testing-matmul",
        "nvidia-cuda-samples-vector-add",
        "nvpro-vk-mini-samples-rectangle",
        "ogl-samples-flat-color",
        "opencl-sdk-saxpy",
        "openframeworks-noise-shader",
        "rocm-examples-add-kernel",
        "raylib-base-fragment",
        "raylib-base-vertex",
        "raylib-lighting-shader-pair",
        "renderdoc-vktext-fragment",
        "rust-gpu-compute-collatz",
        "rust-gpu-graphics-stage-inputs",
        "rust-gpu-vulkan-examples-triangle-overlay",
        "sascha-willems-vulkan-conservative-triangle",
        "sascha-willems-vulkan-headless-compute",
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
    assert "Windows DirectX compile references" in workflow
    assert 'dxc -T "$profile" -E "$entry"' in workflow
    assert "spirv-tools-basic-src/crosstl-out/metal/basic_src.metal" in workflow
    assert "vulkan-tools-cube/crosstl-out/metal/cube.vert.metal" in workflow
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


def _workflow_run_demo_invocations(workflow: str, step_name: str):
    block = _workflow_step_block(workflow, step_name)
    normalized = re.sub(r"\\\n\s*", " ", block)
    invocations = []
    for line in normalized.splitlines():
        if "python demos/open-source-porting/run_demo.py" not in line:
            continue
        invocations.append(
            {
                "cases": set(re.findall(r"--case ([a-z0-9-]+)", line)),
                "targets": set(re.findall(r"--target ([a-z0-9-]+)", line)),
            }
        )
    return invocations


def _smoke_covers_target(invocations, case_name: str, target: str) -> bool:
    return any(
        case_name in invocation["cases"] and target in invocation["targets"]
        for invocation in invocations
    )


def test_open_source_demo_workflow_case_smoke_lists_match_checked_targets():
    workflow = (ROOT / ".github" / "workflows" / "demo.yml").read_text(encoding="utf-8")

    assert _workflow_step_cases(workflow, "Linux OpenGL and Vulkan smoke checks") == {
        "angle-simple-texture-2d",
        "apple-modern-rendering-mesh-viewdir",
        "arm-opengl-es-sdk-cube",
        "directx-graphics-samples-hello-triangle",
        "directx-graphics-samples-hello-texture",
        "directx-shader-compiler-neg1",
        "directx-sdk-samples-tutorial02",
        "diligent-samples-vrs-cube-vertex",
        "glslang-push-constant-vertex",
        "glslang-spec-constant-vertex",
        "godot-betsy-alpha-stitch",
        "libgdx-batch-shader",
        "lonelydevil-vulkan-tutorial-triangle",
        "monogame-sprite-effect",
        "metal-performance-testing-matmul",
        "nvidia-cuda-samples-vector-add",
        "nvpro-vk-mini-samples-rectangle",
        "ogl-samples-flat-color",
        "opencl-sdk-saxpy",
        "openframeworks-noise-shader",
        "rocm-examples-add-kernel",
        "raylib-base-fragment",
        "raylib-base-vertex",
        "raylib-lighting-shader-pair",
        "renderdoc-vktext-fragment",
        "rust-gpu-compute-collatz",
        "rust-gpu-graphics-stage-inputs",
        "rust-gpu-vulkan-examples-triangle-overlay",
        "sascha-willems-vulkan-conservative-triangle",
        "sascha-willems-vulkan-headless-compute",
        "slang-hello-world-compute",
        "spirv-cross-round-fragment",
        "spirv-tools-basic-src",
        "vulkan-tools-cube",
        "vulkan-samples-dynamic-line-grid",
    }
    assert _workflow_step_cases(workflow, "macOS Metal smoke checks") == {
        "angle-simple-texture-2d",
        "apple-modern-rendering-mesh-viewdir",
        "arm-opengl-es-sdk-cube",
        "directx-graphics-samples-hello-triangle",
        "directx-graphics-samples-hello-texture",
        "directx-shader-compiler-neg1",
        "directx-sdk-samples-tutorial02",
        "glslang-push-constant-vertex",
        "glslang-spec-constant-vertex",
        "godot-betsy-alpha-stitch",
        "libgdx-batch-shader",
        "lonelydevil-vulkan-tutorial-triangle",
        "metal-performance-testing-matmul",
        "monogame-sprite-effect",
        "nvidia-cuda-samples-vector-add",
        "nvpro-vk-mini-samples-rectangle",
        "ogl-samples-flat-color",
        "opencl-sdk-saxpy",
        "openframeworks-noise-shader",
        "raylib-base-fragment",
        "raylib-base-vertex",
        "raylib-lighting-shader-pair",
        "rocm-examples-add-kernel",
        "rust-gpu-compute-collatz",
        "rust-gpu-graphics-stage-inputs",
        "rust-gpu-vulkan-examples-triangle-overlay",
        "sascha-willems-vulkan-conservative-triangle",
        "sascha-willems-vulkan-headless-compute",
        "slang-hello-world-compute",
        "spirv-tools-basic-src",
        "vulkan-samples-dynamic-line-grid",
        "vulkan-tools-cube",
    }
    assert _workflow_step_cases(workflow, "Windows DirectX smoke checks") == {
        "angle-simple-texture-2d",
        "apple-modern-rendering-mesh-viewdir",
        "arm-opengl-es-sdk-cube",
        "directx-graphics-samples-hello-triangle",
        "directx-graphics-samples-hello-texture",
        "directx-shader-compiler-neg1",
        "directx-sdk-samples-tutorial02",
        "glslang-push-constant-vertex",
        "godot-betsy-alpha-stitch",
        "libgdx-batch-shader",
        "slang-hello-world-compute",
        "lonelydevil-vulkan-tutorial-triangle",
        "monogame-sprite-effect",
        "openframeworks-noise-shader",
        "sascha-willems-vulkan-headless-compute",
        "vulkan-tools-cube",
        "vulkan-samples-dynamic-line-grid",
    }


def test_open_source_demo_workflow_covers_platform_targets():
    runner = _load_demo_runner()
    workflow = (ROOT / ".github" / "workflows" / "demo.yml").read_text(encoding="utf-8")
    linux_invocations = _workflow_run_demo_invocations(
        workflow, "Linux OpenGL and Vulkan smoke checks"
    )
    macos_invocations = _workflow_run_demo_invocations(
        workflow, "macOS Metal smoke checks"
    )
    windows_invocations = _workflow_run_demo_invocations(
        workflow, "Windows DirectX smoke checks"
    )

    for case_dir in sorted(path for path in CASE_ROOT.iterdir() if path.is_dir()):
        targets = set(runner._case_targets(case_dir))
        for target in {"opengl", "vulkan"} & targets:
            assert _smoke_covers_target(linux_invocations, case_dir.name, target)
        if "metal" in targets:
            assert _smoke_covers_target(macos_invocations, case_dir.name, "metal")
        if "directx" in targets:
            assert _smoke_covers_target(windows_invocations, case_dir.name, "directx")


def test_open_source_demo_workflow_compile_reference_paths_exist():
    workflow = (ROOT / ".github" / "workflows" / "demo.yml").read_text(encoding="utf-8")

    metal_block = _workflow_step_block(workflow, "macOS Metal compile references")
    metal_paths = set(
        re.findall(r"(demos/open-source-porting/cases/[^\s]+?\.metal)", metal_block)
    )
    assert {
        str(path.relative_to(ROOT))
        for path in CASE_ROOT.glob("*/crosstl-out/metal/*.metal")
        if path.parts[-4]
        not in {
            "directx-graphics-samples-hello-triangle",
            "sascha-willems-vulkan-conservative-triangle",
        }
    } == metal_paths
    assert all((ROOT / path).is_file() for path in metal_paths)

    directx_block = _workflow_step_block(workflow, "Windows DirectX compile references")
    directx_paths = set(
        re.findall(
            r"compile_hlsl (demos/open-source-porting/cases/[^\s]+?\.hlsl)",
            directx_block,
        )
    )
    assert {
        "demos/open-source-porting/cases/angle-simple-texture-2d/crosstl-out/directx/SimpleTexture2D.vert.hlsl",
        "demos/open-source-porting/cases/angle-simple-texture-2d/crosstl-out/directx/SimpleTexture2D.frag.hlsl",
        "demos/open-source-porting/cases/apple-modern-rendering-mesh-viewdir/crosstl-out/directx/AAPLMeshRenderer.hlsl",
        "demos/open-source-porting/cases/arm-opengl-es-sdk-cube/crosstl-out/directx/Cube_cube.vert.hlsl",
        "demos/open-source-porting/cases/arm-opengl-es-sdk-cube/crosstl-out/directx/Cube_cube.frag.hlsl",
        "demos/open-source-porting/cases/directx-graphics-samples-hello-triangle/crosstl-out/directx/shaders.hlsl",
        "demos/open-source-porting/cases/directx-graphics-samples-hello-texture/crosstl-out/directx/shaders.hlsl",
        "demos/open-source-porting/cases/directx-shader-compiler-neg1/crosstl-out/directx/neg1.hlsl",
        "demos/open-source-porting/cases/directx-sdk-samples-tutorial02/crosstl-out/directx/Tutorial02.hlsl",
        "demos/open-source-porting/cases/godot-betsy-alpha-stitch/crosstl-out/directx/alpha_stitch.hlsl",
        "demos/open-source-porting/cases/libgdx-batch-shader/crosstl-out/directx/batch.vert.hlsl",
        "demos/open-source-porting/cases/libgdx-batch-shader/crosstl-out/directx/batch.frag.hlsl",
        "demos/open-source-porting/cases/monogame-sprite-effect/crosstl-out/directx/SpriteEffect.hlsl",
        "demos/open-source-porting/cases/openframeworks-noise-shader/crosstl-out/directx/noise.vert.hlsl",
        "demos/open-source-porting/cases/openframeworks-noise-shader/crosstl-out/directx/noise.frag.hlsl",
        "demos/open-source-porting/cases/sascha-willems-vulkan-headless-compute/crosstl-out/directx/headless.hlsl",
        "demos/open-source-porting/cases/slang-hello-world-compute/crosstl-out/directx/hello-world.hlsl",
        "demos/open-source-porting/cases/vulkan-tools-cube/crosstl-out/directx/cube.vert.hlsl",
        "demos/open-source-porting/cases/vulkan-tools-cube/crosstl-out/directx/cube.frag.hlsl",
        "demos/open-source-porting/cases/vulkan-samples-dynamic-line-grid/crosstl-out/directx/grid.hlsl",
    } == directx_paths
    assert all((ROOT / path).is_file() for path in directx_paths)


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

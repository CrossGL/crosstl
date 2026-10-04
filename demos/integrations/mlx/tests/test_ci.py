import ast
import re
import textwrap
from fnmatch import fnmatchcase
from pathlib import Path

import pytest

from tests.test_ci_workflows import (
    RUNNER_OSES,
    _load_ci_coverage_module,
    _matrix_values,
    _workflow_event_paths,
    _workflow_job_section,
    _workflow_texts,
)
from tests.test_translator.test_atomic_store_runtime import REQUIRE_ENV as STORE_ENV

ROOT = Path(__file__).resolve().parents[4]
WORKFLOW_DIR = ROOT / ".github/workflows"


def _assert_workflow_triggers(workflow, path):
    assert (ROOT / path).is_file(), path
    for event in ("pull_request", "push"):
        patterns = _workflow_event_paths(workflow, event)
        assert any(fnmatchcase(path, pattern) for pattern in patterns), (event, path)


def test_project_demo_has_one_workflow_and_local_test_ownership():
    assert not list(WORKFLOW_DIR.glob("mlx-*.yml"))
    assert not list((ROOT / "tests").rglob("test_mlx_*.py"))
    assert not list((ROOT / "tests/fixtures").rglob("mlx_*"))
    assert (WORKFLOW_DIR / "demo-project-testing.yml").is_file()
    assert (ROOT / "demos/integrations/mlx/tests/host").is_dir()
    assert (ROOT / "demos/integrations/mlx/tests/kernels").is_dir()


def test_project_demo_triggers_cover_code_without_root_documentation():
    workflow = _workflow_texts()["demo-project-testing.yml"]
    paths = _workflow_event_paths(workflow, "pull_request")
    assert paths == _workflow_event_paths(workflow, "push")
    assert len(paths) == len(set(paths))
    for path in (
        "crosstl/project/pipeline.py",
        "demos/integrations/mlx/contracts/copy.directx.json",
        "tests/runtime_helpers.py",
        "tools/run_bounded_command.py",
        "support/features.json",
        ".github/actions/install-linux-dxc/action.yml",
        ".github/workflows/demo-project-testing.yml",
        ".pre-commit-config.yaml",
        "conftest.py",
        "pyproject.toml",
        "requirements.txt",
        "setup.py",
        "setup.cfg",
    ):
        assert any(fnmatchcase(path, pattern) for pattern in paths), path
    for path in ("README.md", "docs/source/index.rst", ".github/TESTING.md"):
        assert not any(fnmatchcase(path, pattern) for pattern in paths), path


def test_project_demo_queues_each_job_and_matrix_leg_independently():
    workflow = _workflow_texts()["demo-project-testing.yml"]
    coverage = _load_ci_coverage_module()
    assert not coverage.nested_yaml_section(workflow, "concurrency", 0)
    jobs = coverage.workflow_job_names(workflow)
    assert len(jobs) == 32
    groups = set()
    for name in jobs:
        job = _workflow_job_section(workflow, name)
        section = coverage.nested_yaml_section(job, "concurrency", 4)
        expected = "${{ github.workflow }}-${{ github.ref }}-" + name
        if coverage.nested_yaml_section(job, "matrix", 6):
            expected += "-${{ strategy.job-index }}"
        assert section == [
            "      group: " + expected,
            "      cancel-in-progress: false",
        ]
        assert expected not in groups
        groups.add(expected)


def test_metal_execution_budget_preserves_all_storage_cases():
    job = _workflow_job_section(_workflow_texts()["demo-project-testing.yml"], "metal")
    step = _load_ci_coverage_module().workflow_step_section(
        job, "Validate Metal byte and vector storage"
    )
    assert '--timeout-seconds 1800 --label "Native Metal storage"' in step
    assert "--dist loadgroup" in step
    assert 'PYTEST_XDIST_AUTO_NUM_WORKERS: "2"' in step
    assert "-k " not in step
    assert "continue-on-error" not in step
    for test in (
        "test_half_buffer_runtime",
        "test_bfloat_buffer_runtime",
        "test_opengl_bfloat_conversion",
        "test_bfloat_constants",
        "test_half_copy_identity",
        "test_metal_narrow_aggregate_runtime",
    ):
        assert f"tests/test_translator/{test}.py" in step
    seconds = sum(int(value) for value in re.findall(r"--timeout-seconds (\d+)", job))
    minutes = int(re.search(r"timeout-minutes: (\d+)", job).group(1))
    assert seconds + 300 <= minutes * 60


@pytest.mark.parametrize("family", ("unary", "binary", "copy", "reduce"))
def test_directx_corpus_retains_failure_evidence_without_additional_jobs(family):
    workflow = _workflow_texts()["demo-project-testing.yml"]
    job = _workflow_job_section(workflow, f"mlx-{family}-complete-directx-translation")
    coverage = _load_ci_coverage_module()
    proof = coverage.workflow_step_section(
        job, f"Prove current MLX complete {family} family DirectX translation"
    )
    assert "runs-on: ubuntu-24.04" in job
    assert 'CROSTL_KEEP_CORPUS_EVIDENCE: "1"' in proof
    assert "python -m pytest -q -n auto" in proof
    assert f"--junitxml={family}-directx-junit.xml" in proof
    assert "continue-on-error" not in job
    upload = coverage.workflow_step_section(
        job, f"Upload {family} DirectX corpus evidence"
    )
    assert "if: always()" in upload
    assert f"name: mlx-{family}-directx-corpus-${{{{ matrix.shard_index }}}}" in upload
    assert "mlx-current-upstream/.crosstl-corpus-evidence" in upload
    assert f"{family}-directx-junit.xml" in upload
    assert "include-hidden-files: true" in upload
    assert "if-no-files-found: error" in upload
    assert "retention-days: 14" in upload


def test_mlx_gather_checkout_preserves_pinned_source_bytes():
    workflow = (WORKFLOW_DIR / "demo-project-testing.yml").read_text(encoding="utf-8")
    ci_coverage = _load_ci_coverage_module()
    for target in ("metal", "metal-float", "directx", "opengl"):
        job = _workflow_job_section(workflow, target)
        step = ci_coverage.workflow_step_section(job, "Checkout pinned upstream MLX")
        initialization = step.index("git init mlx-upstream")
        configuration = step.index("git -C mlx-upstream config core.autocrlf false")
        checkout = step.index("git -C mlx-upstream checkout --detach FETCH_HEAD")
        assert initialization < configuration < checkout


def test_mlx_float_atomic_reference_uses_fixed_metal_toolchain():
    workflow = (WORKFLOW_DIR / "demo-project-testing.yml").read_text(encoding="utf-8")
    job = _workflow_job_section(workflow, "metal-float")
    assert "runs-on: xcode-27" in job
    assert "DEVELOPER_DIR: /Applications/Xcode_27.1.app/Contents/Developer" in job
    assert "xcodebuild -version | tee .mlx-float-metal/xcode-version.txt" in job
    assert "metal --version | tee .mlx-float-metal/metal-version.txt" in job
    assert "if-no-files-found: error" in job
    assert "include-hidden-files: true" in job
    assert "if: always()" in job
    assert "continue-on-error" not in job
    previous = _workflow_job_section(workflow, "metal")
    assert "runs-on: macos-26" in previous
    assert 'CROSTL_REQUIRE_FLOAT_ATOMIC_MEMORY: "1"' in previous


def test_mlx_project_porting_workflow_runs_tracked_porting_harness():
    workflows = _workflow_texts()
    mlx_porting = workflows.get("demo-project-testing.yml", "")
    harness = (ROOT / "demos" / "integrations" / "mlx" / "run_porting.py").read_text(
        encoding="utf-8"
    )
    mlx_reference_commit = "4367c73b60541ddd5a266ce4644fd93d20223b6e"
    mlx_corpus_commit = "846d176227a0ac13d2667e58d2bb68b322109ab0"
    mlx_current_tree_commit = "9c3d35571ac450a8ecf5c17b4d0e3fac52c08bc8"

    assert mlx_porting, "demo-project-testing.yml must exist"
    for event_name in ("push", "pull_request"):
        trigger_paths = set(_workflow_event_paths(mlx_porting, event_name))
        for path in (
            "tools/run_bounded_command.py",
            "tests/test_run_bounded_command.py",
            "tools/run_directx_diagnostics.py",
            "tests/test_run_directx_diagnostics.py",
            "tests/test_translator/test_directx_reduction_primitives.py",
        ):
            assert any(fnmatchcase(path, pattern) for pattern in trigger_paths), path
    assert "demos/integrations/mlx/run_porting.py" in mlx_porting
    assert f'MLX_COMMIT: "{mlx_reference_commit}"' in mlx_porting
    assert f'MLX_CORPUS_COMMIT: "{mlx_corpus_commit}"' in mlx_porting
    assert f'MLX_CURRENT_TREE_COMMIT: "{mlx_current_tree_commit}"' in mlx_porting
    assert 'git -C mlx-upstream checkout "$MLX_COMMIT"' in mlx_porting
    assert 'git -C mlx-upstream checkout "$MLX_CORPUS_COMMIT"' in mlx_porting
    current_runtime_checkout = _load_ci_coverage_module().workflow_step_section(
        mlx_porting,
        "Checkout current MLX runtime proof corpus",
    )
    assert "git clone --filter=blob:none --no-checkout" in current_runtime_checkout
    assert "mlx-current-upstream" in current_runtime_checkout
    assert (
        'git -C mlx-current-upstream checkout --detach "$MLX_CORPUS_COMMIT"'
        in current_runtime_checkout
    )
    assert (
        'test "$(git -C mlx-current-upstream rev-parse HEAD)" = '
        '"$MLX_CORPUS_COMMIT"' in current_runtime_checkout
    )
    current_tree_checkout = _load_ci_coverage_module().workflow_step_section(
        mlx_porting,
        "Checkout current MLX kernel tree",
    )
    assert "mlx-current-tree-upstream" in current_tree_checkout
    assert (
        "git -C mlx-current-tree-upstream checkout --detach "
        '"$MLX_CURRENT_TREE_COMMIT"' in current_tree_checkout
    )
    assert (
        'test "$(git -C mlx-current-tree-upstream rev-parse HEAD)" = '
        '"$MLX_CURRENT_TREE_COMMIT"' in current_tree_checkout
    )
    current_census = _load_ci_coverage_module().workflow_step_section(
        mlx_porting,
        "Audit current MLX kernel census",
    )
    assert "if: runner.os == 'Linux'" in current_census
    assert '--expected-commit "$MLX_CURRENT_TREE_COMMIT"' in current_census
    assert "--expected-unit-count 49" in current_census
    assert "--expected-entry-count 17832" in current_census
    math_checks = _load_ci_coverage_module().workflow_step_section(
        mlx_porting, "Validate Direct3D Metal math semantics"
    )
    _assert_workflow_triggers(
        mlx_porting, "tests/test_translator/test_directx_metal_math.py"
    )
    assert "if: runner.os == 'Windows'" in math_checks
    assert 'CROSTL_REQUIRE_DIRECTX_METAL_MATH: "1"' in math_checks
    assert "--timeout-seconds 120 --" in math_checks
    assert "test_directx_metal_math.py::test_directx_metal_math_executes" in math_checks
    assert "mkdir -p directx-math-results" in math_checks
    math_upload = _load_ci_coverage_module().workflow_step_section(
        mlx_porting, "Upload Direct3D Metal math evidence"
    )
    assert "if: always() && runner.os == 'Windows'" in math_upload
    assert "path: directx-math-results" in math_upload
    assert "if-no-files-found: error" in math_upload
    atan2_checks = _load_ci_coverage_module().workflow_step_section(
        mlx_porting, "Validate atan2 signed-zero semantics"
    )
    _assert_workflow_triggers(
        mlx_porting, "tests/test_translator/test_directx_atan2.py"
    )
    _assert_workflow_triggers(
        mlx_porting, "tests/fixtures/runtime_verification/metal_uint32_buffers.swift"
    )
    assert "if: runner.os == 'Windows' || runner.os == 'macOS'" in atan2_checks
    assert (
        "CROSTL_REQUIRE_DIRECTX_ATAN2: ${{ runner.os == 'Windows' && '1' || '0' }}"
        in atan2_checks
    )
    assert (
        "CROSTL_REQUIRE_METAL_ATAN2: ${{ runner.os == 'macOS' && '1' || '0' }}"
        in atan2_checks
    )
    assert "--timeout-seconds 120 --" in atan2_checks
    assert "pytest -q -n auto" in atan2_checks
    assert "tests/test_translator/test_directx_atan2.py" in atan2_checks
    atan2_upload = _load_ci_coverage_module().workflow_step_section(
        mlx_porting, "Upload atan2 signed-zero evidence"
    )
    assert (
        "if: always() && (runner.os == 'Windows' || runner.os == 'macOS')"
        in atan2_upload
    )
    assert "path: atan2-results" in atan2_upload
    assert "if-no-files-found: error" in atan2_upload
    opengl_math_checks = _load_ci_coverage_module().workflow_step_section(
        mlx_porting, "Validate OpenGL Metal math semantics"
    )
    _assert_workflow_triggers(
        mlx_porting, "tests/test_translator/test_opengl_metal_math.py"
    )
    assert "if: runner.os == 'Linux'" in opengl_math_checks
    assert 'CROSTL_REQUIRE_OPENGL_METAL_MATH: "1"' in opengl_math_checks
    assert "--timeout-seconds 120 --" in opengl_math_checks
    assert "pytest -q -n auto" in opengl_math_checks
    assert "tests/test_translator/test_opengl_metal_math.py" in opengl_math_checks
    opengl_math_upload = _load_ci_coverage_module().workflow_step_section(
        mlx_porting, "Upload OpenGL Metal math evidence"
    )
    assert "if: always() && runner.os == 'Linux'" in opengl_math_upload
    assert "path: opengl-math-results" in opengl_math_upload
    assert "if-no-files-found: error" in opengl_math_upload
    metal_package = _load_ci_coverage_module().workflow_step_section(
        mlx_porting, "Validate native Metal package execution"
    )
    assert 'CROSTL_REQUIRE_METAL_PACKAGE_RUNTIME: "1"' in metal_package
    assert 'CROSTL_REQUIRE_METAL_HELPER_LINKAGE: "1"' in metal_package
    assert "tests/test_translator/test_metal_helper_linkage.py" in metal_package
    _assert_workflow_triggers(
        mlx_porting, "tests/test_translator/test_metal_helper_linkage.py"
    )
    assert "if: runner.os == 'macOS'" in metal_package
    assert "--timeout-seconds 300 --" in metal_package
    assert "pytest -q -n auto" in metal_package
    assert "tests/test_translator/test_metal_native_runtime.py" in metal_package
    assert "tests/test_translator/test_exact_thread_grid_runtime.py" in metal_package
    _assert_workflow_triggers(
        mlx_porting, "tests/test_translator/test_exact_thread_grid_runtime.py"
    )
    _assert_workflow_triggers(
        mlx_porting, "tests/test_translator/test_metal_native_runtime.py"
    )
    metal_package_upload = _load_ci_coverage_module().workflow_step_section(
        mlx_porting, "Upload native Metal package evidence"
    )
    assert "if: always() && runner.os == 'macOS'" in metal_package_upload
    assert "path: metal-package-results" in metal_package_upload
    assert "if-no-files-found: error" in metal_package_upload
    ownership = _load_ci_coverage_module().workflow_step_section(
        mlx_porting, "Validate Metal builtin ownership"
    )
    _assert_workflow_triggers(
        mlx_porting, "tests/test_translator/test_metal_builtin_ownership.py"
    )
    assert 'CROSTL_REQUIRE_METAL_BUILTIN_OWNERSHIP: "1"' in ownership
    assert "--timeout-seconds 120 --" in ownership
    assert "pytest -q -n auto" in ownership
    assert "if: runner.os" not in ownership
    ownership_upload = _load_ci_coverage_module().workflow_step_section(
        mlx_porting, "Upload Metal builtin ownership evidence"
    )
    assert "if: always()" in ownership_upload
    assert "name: metal-builtin-ownership-${{ runner.os }}" in ownership_upload
    assert "if-no-files-found: error" in ownership_upload
    struct_checks = _load_ci_coverage_module().workflow_step_section(
        mlx_porting, "Prove current MLX complex-power native dispatch"
    )
    assert 'CROSTL_REQUIRE_MLX_CURRENT_COMPLEX_POWER: "1"' in struct_checks
    assert 'CROSTL_REQUIRE_STRUCT_BUFFER_RUNTIME: "1"' in struct_checks
    assert "--timeout-seconds 900 --" in struct_checks
    assert "pytest -q -n auto" in struct_checks
    assert "if: runner.os" not in struct_checks
    for filename in (
        "tests/test_translator/test_struct_buffer_layouts.py",
        "tests/test_translator/test_buffer_requirements.py",
        "demos/integrations/mlx/tests/kernels/test_current_complex_power.py",
    ):
        _assert_workflow_triggers(mlx_porting, filename)
        assert filename in struct_checks
    struct_upload = _load_ci_coverage_module().workflow_step_section(
        mlx_porting, "Upload current MLX complex-power evidence"
    )
    assert "if: always()" in struct_upload
    assert "name: mlx-complex-power-${{ runner.os }}" in struct_upload
    assert "if-no-files-found: error" in struct_upload
    binary_shapes = _load_ci_coverage_module().workflow_step_section(
        mlx_porting, "Prove current MLX binary shape parity"
    )
    assert 'CROSTL_REQUIRE_MLX_CURRENT_BINARY_SHAPES: "1"' in binary_shapes
    assert "--timeout-seconds 900 --" in binary_shapes
    assert "pytest -q -n auto" in binary_shapes
    assert "if: runner.os" not in binary_shapes
    assert (
        "demos/integrations/mlx/tests/kernels/test_current_binary_shapes.py"
        in binary_shapes
    )
    _assert_workflow_triggers(
        mlx_porting,
        "demos/integrations/mlx/tests/kernels/test_current_binary_shapes.py",
    )
    binary_upload = _load_ci_coverage_module().workflow_step_section(
        mlx_porting, "Upload current MLX binary shape evidence"
    )
    assert "if: always()" in binary_upload
    assert "name: mlx-binary-shapes-${{ runner.os }}" in binary_upload
    assert "mlx-current-tree-upstream/.current-binary-shapes-*" in binary_upload
    assert "include-hidden-files: true" in binary_upload
    assert "if-no-files-found: error" in binary_upload
    primitive_checks = _load_ci_coverage_module().workflow_step_section(
        mlx_porting, "Validate Direct3D reduction primitives"
    )
    assert "if: runner.os == 'Windows'" in primitive_checks
    assert 'CROSTL_REQUIRE_DIRECTX_REDUCTION_PRIMITIVES: "1"' in primitive_checks
    assert (
        "for case in metadata division shuffle combined pair-reduction "
        "special-values two-phase private-array; do" in primitive_checks
    )
    assert "--timeout-seconds 120 --" in primitive_checks
    assert "test_directx_reduction_primitives_execute[$case]" in primitive_checks
    assert "|| failed=1" in primitive_checks
    assert 'exit "$failed"' in primitive_checks
    primitive_upload = _load_ci_coverage_module().workflow_step_section(
        mlx_porting, "Upload Direct3D reduction primitive diagnostics"
    )
    assert "if: always() && runner.os == 'Windows'" in primitive_upload
    assert "name: directx-reduction-primitives" in primitive_upload
    assert "path: mlx-current-results" in primitive_upload
    assert mlx_porting.index(
        "Upload Direct3D reduction primitive diagnostics"
    ) < mlx_porting.index("Prove current MLX arg-reduce native validation")
    assert mlx_porting.index(
        "Validate Direct3D reduction primitives"
    ) < mlx_porting.index("Prove current MLX arg-reduce native validation")
    current_arg_reduce = _load_ci_coverage_module().workflow_step_section(
        mlx_porting,
        "Prove current MLX arg-reduce native validation",
    )
    assert (
        "CROSTL_MLX_CURRENT_ROOT: "
        "${{ github.workspace }}/mlx-current-tree-upstream" in current_arg_reduce
    )
    assert 'CROSTL_REQUIRE_MLX_CURRENT_ARG_REDUCE: "1"' in current_arg_reduce
    assert 'CROSTL_REQUIRE_MLX_CURRENT_ARG_REDUCE_RUNTIME: "1"' in current_arg_reduce
    assert "Linux) export CROSTL_MLX_CURRENT_TARGET=opengl" in current_arg_reduce
    assert "Windows) export CROSTL_MLX_CURRENT_TARGET=directx" in current_arg_reduce
    assert "macOS) export CROSTL_MLX_CURRENT_TARGET=metal" in current_arg_reduce
    assert (
        "python -m pytest -q "
        "demos/integrations/mlx/tests/kernels/test_current_arg_reduce.py"
        in current_arg_reduce
    )
    _assert_workflow_triggers(
        mlx_porting, "demos/integrations/mlx/tests/kernels/test_current_arg_reduce.py"
    )
    assert '--expected-commit "$MLX_COMMIT"' in mlx_porting
    assert "--expected-unit-count 40" in mlx_porting
    assert "--expected-entry-count 16446" in mlx_porting
    assert "Enumerate current MLX Metal entry points" in mlx_porting
    assert _matrix_values(mlx_porting, "os") == RUNNER_OSES
    matrix_job = _workflow_job_section(mlx_porting, "mlx-metal-porting")
    assert "timeout-minutes: 360" in matrix_job
    assert 'if [[ "$RUNNER_OS" == Windows ]]' in current_arg_reduce
    assert 'export PYTEST_ADDOPTS="-n auto"' not in current_arg_reduce
    assert "python -m pytest -q -n auto \\" in current_arg_reduce
    assert (
        "demos/integrations/mlx/tests/kernels/test_current_arg_reduce.py \\"
        in current_arg_reduce
    )
    assert '-k "not argmin_float32 and not argmax_float32"' in current_arg_reduce
    assert 'PYTHONUNBUFFERED: "1"' in current_arg_reduce
    assert "for entry in argmin_float32 argmax_float32; do" in current_arg_reduce
    assert "python tools/run_bounded_command.py \\" in current_arg_reduce
    assert '--label "current MLX $entry WARP runtime" \\' in current_arg_reduce
    assert "--timeout-seconds 900 \\" in current_arg_reduce
    assert "python tools/run_directx_diagnostics.py \\" in current_arg_reduce
    assert '--output "mlx-current-results/$entry-directx.jsonl"' in current_arg_reduce
    assert "--module pytest -- \\" in current_arg_reduce
    assert "-vv -s --tb=long -o faulthandler_timeout=120 \\" in current_arg_reduce
    assert '--basetemp="mlx-current-results/$entry"' in current_arg_reduce
    assert current_arg_reduce.index(
        "mkdir -p mlx-current-results"
    ) < current_arg_reduce.index('--basetemp="mlx-current-results/$entry"')
    assert '--junitxml="mlx-current-results/$entry.xml"' in current_arg_reduce
    runtime_upload = _load_ci_coverage_module().workflow_step_section(
        mlx_porting, "Upload current MLX runtime diagnostics"
    )
    assert "if: always() && runner.os == 'Windows'" in runtime_upload
    assert "path: mlx-current-results" in runtime_upload
    assert "actions/upload-artifact@v4" in runtime_upload
    assert '-k "$entry"' in current_arg_reduce
    assert '-k "argmin_float32 or argmax_float32"' not in current_arg_reduce
    assert "timeout-minutes: 60" in mlx_porting
    assert re.search(r"\bschedule\s*:", mlx_porting)
    assert 'cron: "31 4 * * 1"' in mlx_porting
    assert "github.event_name != 'schedule'" in mlx_porting
    assert "--mode reduced-frontier" in mlx_porting
    assert "--require-metal-toolchain" in mlx_porting
    assert "Install macOS Metal Toolchain" in mlx_porting
    assert "if: runner.os == 'macOS'" in mlx_porting
    assert "xcodebuild -downloadComponent MetalToolchain" in mlx_porting
    for metal_job_name in (
        "mlx-metal-porting",
        "mlx-unary-metal-roundtrip",
        "mlx-binary-complete-metal-roundtrip",
        "mlx-copy-complete-metal-roundtrip",
        "mlx-reduce-complete-metal-roundtrip",
        "mlx-quantized-complete-metal-roundtrip",
    ):
        metal_job = _workflow_job_section(mlx_porting, metal_job_name)
        assert "xcrun --sdk macosx metal --version" in metal_job
    assert "--require-directx-toolchain" in mlx_porting
    assert "--require-directx-gemv-compiler-frontier" in mlx_porting
    assert "--require-opengl-frontier-toolchain" in mlx_porting
    assert "--require-opengl-gemv-toolchain" in mlx_porting
    assert "--require-opengl-gemv-frontier" not in mlx_porting
    assert "--require-vulkan-gemv-toolchain" in mlx_porting
    assert "--require-vulkan-native-runtime" in mlx_porting
    assert "--require-opengl-native-runtime" in mlx_porting
    assert "Install Windows DirectX Shader Compiler" in mlx_porting
    assert "DirectXShaderCompiler/releases/download/v1.9.2602.24" in mlx_porting
    assert "dxc --version" in mlx_porting
    assert "glslang-tools" in mlx_porting
    assert "glslangValidator --version" in mlx_porting
    assert "moderngl==5.12.0" in mlx_porting
    assert "libegl1" in mlx_porting
    assert "libgl1" in mlx_porting
    assert "libgl1-mesa-dri" in mlx_porting
    assert "libopengl0" in mlx_porting
    assert "Validate OpenGL lowering contracts" in mlx_porting
    assert "opengl_lowers_expected_scalar_and_vector_conversions" in mlx_porting
    assert "opengl_preserves_metal_arithmetic_conversion_order" in mlx_porting
    assert "glsl_atomic_thread_fence" in mlx_porting
    assert "translate_project_reports_unrepresentable_atomic_fence_contract" in (
        mlx_porting
    )
    assert (
        "python -m pytest -q -n auto "
        "demos/integrations/mlx/tests/test_dispatch_contract_fixture.py "
        "demos/integrations/mlx/tests/test_logsumexp_dispatch_contract_fixture.py"
        in mlx_porting
    )
    _assert_workflow_triggers(
        mlx_porting, "demos/integrations/mlx/tests/test_porting_harness.py"
    )
    _assert_workflow_triggers(
        mlx_porting,
        "demos/integrations/mlx/tests/test_logsumexp_dispatch_contract_fixture.py",
    )
    _assert_workflow_triggers(
        mlx_porting,
        "demos/integrations/mlx/tests/test_rms_norm_dispatch_contract_fixture.py",
    )
    _assert_workflow_triggers(
        mlx_porting,
        "demos/integrations/mlx/tests/kernels/test_softmax_native_loader.py",
    )
    assert "Prove pinned MLX Softmax Direct3D native-loader execution" in mlx_porting
    assert "CROSTL_REQUIRE_MLX_SOFTMAX_DIRECTX_NATIVE_LOADER" in mlx_porting
    assert (
        "test_pinned_mlx_softmax_executes_through_directx_native_loader" in mlx_porting
    )
    assert "Prove pinned MLX Softmax OpenGL native-loader execution" in mlx_porting
    assert "CROSTL_REQUIRE_MLX_SOFTMAX_OPENGL_TOOLCHAIN" in mlx_porting
    assert "CROSTL_REQUIRE_MLX_SOFTMAX_OPENGL_NATIVE_LOADER" in mlx_porting
    assert (
        "test_pinned_mlx_softmax_executes_through_opengl_native_loader" in mlx_porting
    )
    _assert_workflow_triggers(
        mlx_porting,
        "demos/integrations/mlx/tests/kernels/test_rms_norm_native_loader.py",
    )
    assert "Prove pinned MLX RMSNorm Direct3D native-loader execution" in mlx_porting
    assert "CROSTL_REQUIRE_MLX_RMS_NORM_DIRECTX_NATIVE_LOADER" in mlx_porting
    assert (
        "test_pinned_mlx_rms_norm_executes_through_directx_native_loader" in mlx_porting
    )
    assert "Prove pinned MLX RMSNorm OpenGL native-loader execution" in mlx_porting
    assert "CROSTL_REQUIRE_MLX_RMS_NORM_OPENGL_NATIVE_LOADER" in mlx_porting
    assert (
        "test_pinned_mlx_rms_norm_executes_through_opengl_native_loader" in mlx_porting
    )
    _assert_workflow_triggers(
        mlx_porting,
        "demos/integrations/mlx/tests/kernels/test_layer_norm_native_loader.py",
    )
    _assert_workflow_triggers(
        mlx_porting,
        "demos/integrations/mlx/tests/kernels/test_layer_norm_vjp_native_loader.py",
    )
    assert "Prove pinned MLX LayerNorm Direct3D native-loader execution" in mlx_porting
    assert "CROSTL_REQUIRE_MLX_LAYER_NORM_DIRECTX_NATIVE_LOADER" in mlx_porting
    assert (
        "test_pinned_mlx_layer_norm_executes_through_directx_native_loader"
        in mlx_porting
    )
    assert "Prove pinned MLX LayerNorm OpenGL native-loader execution" in mlx_porting
    assert "CROSTL_REQUIRE_MLX_LAYER_NORM_OPENGL_NATIVE_LOADER" in mlx_porting
    assert (
        "test_pinned_mlx_layer_norm_executes_through_opengl_native_loader"
        in mlx_porting
    )
    _assert_workflow_triggers(
        mlx_porting, "demos/integrations/mlx/tests/test_quantized_directx_proof.py"
    )
    _assert_workflow_triggers(
        mlx_porting, "demos/integrations/mlx/tests/test_quantized_opengl_proof.py"
    )
    _assert_workflow_triggers(
        mlx_porting, "tests/test_translator/test_codegen/test_SPIRV_codegen.py"
    )
    _assert_workflow_triggers(
        mlx_porting, "tests/test_translator/test_codegen/test_directx_codegen.py"
    )
    _assert_workflow_triggers(
        mlx_porting, "tests/test_translator/test_private_pointer_partition_runtime.py"
    )
    _assert_workflow_triggers(
        mlx_porting, "tests/test_translator/test_project_translation.py"
    )
    assert "Verify MLX frontier accounting" in mlx_porting
    assert "expected the exact 11-source non-fence MLX frontier" in mlx_porting
    assert 'scope["nonFenceFrontierSources"]' in mlx_porting
    assert "cleanFrontierSources" not in mlx_porting
    assert "MLX summary identity or status is incorrect" in mlx_porting
    assert "MLX summary check names must be unique" in mlx_porting
    assert "MLX checkout proof does not match the pinned revision" in mlx_porting
    assert "fence contract accounting must be 3 failed, 0 emitted" in mlx_porting
    assert "MLX_DIRECTX_TOOLCHAIN_FRONTIER_SOURCES" in mlx_porting
    assert "MLX_DIRECTX_TOOLCHAIN_ARTIFACT_COUNT" in mlx_porting
    assert "MLX_DIRECTX_TOOLCHAIN_ENTRY_POINT_COUNTS" in mlx_porting
    assert "MLX_DIRECTX_BFLOAT16_LOWERING_EVIDENCE" in mlx_porting
    assert "MLX_DYNAMIC_WORKGROUP_DISPATCH_EVIDENCE" in mlx_porting
    assert "MLX_DIRECTX_DYNAMIC_WORKGROUP_ENTRY_POINT_COUNTS" in mlx_porting
    assert "MLX_DIRECTX_DYNAMIC_WORKGROUP_FRONTIER_SOURCES" in mlx_porting
    assert "MLX_HOST_DISPATCH_IMPORT_RESOLVED_ISSUE" in mlx_porting
    assert "MLX_LAYER_NORM_DISPATCH_VARIANTS" in mlx_porting
    assert "MLX_LOGSUMEXP_DISPATCH_VARIANTS" in mlx_porting
    assert "MLX_RMS_NORM_DISPATCH_VARIANTS" in mlx_porting
    assert 'checks["directx-frontier"]' in mlx_porting
    assert 'checks["vulkan-frontier"]' in mlx_porting
    assert 'directx["directxToolchainArtifactCount"]' in mlx_porting
    assert 'directx["directxToolchainValidatedArtifactCount"]' in mlx_porting
    assert 'directx["directxToolchainValidatedEntryPointCounts"]' in mlx_porting
    assert 'directx["directxToolchainValidatedEntryPointCount"]' in mlx_porting
    assert 'directx["toolchainRuns"] != directx_entry_point_count' in mlx_porting
    assert "DirectX frontier accounting is incomplete" in mlx_porting
    assert 'directx["bfloat16LoweringEvidence"]' in mlx_porting
    assert "DirectX bfloat16 lowering evidence is incomplete" in mlx_porting
    assert "DirectX workgroup blocker evidence changed" in mlx_porting
    assert "expected 76 fail-closed DirectX compute entries" in mlx_porting
    assert "LayerNorm dispatch frontier evidence is incomplete" in mlx_porting
    assert "LogSumExp dispatch frontier evidence is incomplete" in mlx_porting
    assert "RMSNorm dispatch frontier evidence is incomplete" in mlx_porting
    assert "matched-materialized-host-names" in mlx_porting
    assert "DirectX frontier toolchain must validate every configured" in mlx_porting
    assert "source artifact and compute entry" in mlx_porting
    assert "Vulkan frontier accounting is incomplete" in mlx_porting
    assert "Vulkan frontier toolchain validation is incomplete" in mlx_porting
    assert 'checks["gemv-directx-compiler-frontier"]' in mlx_porting
    assert "GEMV_DIRECTX_EXPECTED_ENTRY_POINTS" in mlx_porting
    assert "GEMV_WORKGROUP_SIZE_RULE" in mlx_porting
    assert "GEMV_REPORT_WORKGROUP_SIZE_RULE" in mlx_porting
    assert "GEMV_EXPECTED_RESOLVED_WORKGROUP_SIZES" in mlx_porting
    assert 'gemv_directx["entryProfile"] != "cs_6_2"' in mlx_porting
    assert 'gemv_directx["compilerArguments"]' in mlx_porting
    assert 'gemv_directx["minimumShaderModel"] != "6.2"' in mlx_porting
    assert 'run["profile"] != "cs_6_2"' in mlx_porting
    assert 'run["compilerArguments"] != ["-enable-16bit-types"]' in mlx_porting
    assert 'run["minimumShaderModel"] != "6.2"' in mlx_porting
    assert 'gemv_directx["artifactPackaging"]' in mlx_porting
    assert 'gemv_directx["hostNamedMaterializationCount"] != 224' in mlx_porting
    assert 'gemv_directx["reportExecutionEntryCount"] != 224' in mlx_porting
    assert 'gemv_directx["executionIdentityJoinCount"] != 224' in mlx_porting
    assert 'gemv_directx["generatedTargetEntryIdentityCount"] != 224' in mlx_porting
    assert 'gemv_directx["generatedNumthreadsContractCount"] != 224' in mlx_porting
    assert 'gemv_directx["bareValueDiscardCount"] != 0' in mlx_porting
    assert 'gemv_directx["entryProfileDiagnosticCount"] != 0' in mlx_porting
    assert 'gemv_directx["entryProfileUnusedValueWarningCount"] != 0' in mlx_porting
    assert 'library_run["unusedValueWarningCount"] != 0' in mlx_porting
    assert 'gemv_directx["libraryProfile"] != "lib_6_6"' in mlx_porting
    assert 'library_run["profile"] != "lib_6_6"' in mlx_porting
    assert 'library_run["compilerArguments"]' in mlx_porting
    assert 'library_run["minimumShaderModel"] != "6.2"' in mlx_porting
    assert 'gemv_directx["libraryExportCount"] != 224' in mlx_porting
    assert 'gemv_directx["compilerCoveredEntryPointCount"] != 224' in mlx_porting
    assert "DirectX GEMV resolved workgroup-size evidence is incomplete" in (
        mlx_porting
    )
    assert "library-profile-numthreads-ignored" in mlx_porting
    assert "DirectX GEMV compiler warning classification changed" in mlx_porting
    assert "DirectX GEMV compiler warning evidence changed" in mlx_porting
    assert "DirectX GEMV execution non-claims changed" in mlx_porting
    assert "GEMV_OPENGL_WORKGROUP_SIZE_ISSUE" not in mlx_porting
    assert 'gemv_directx["numthreadsContractEstablished"] is not True' in mlx_porting
    assert 'gemv_directx["exactWorkgroupSizeEstablished"] is not True' in mlx_porting
    assert 'checks["gemv-opengl-toolchain"]' in mlx_porting
    assert 'scope["openglGemvToolchainRequired"]' in mlx_porting
    assert 'gemv_opengl["workgroupSizeRuleConfigured"] is not True' in mlx_porting
    assert 'gemv_opengl["runnableArtifactClaimed"] is not False' in mlx_porting
    assert 'gemv_opengl["toolchainValidatedArtifactCount"]' in mlx_porting
    assert 'gemv_opengl["compilerValidatedArtifactClaimed"] is not True' in (
        mlx_porting
    )
    assert "OpenGL GEMV toolchain evidence is incomplete" in mlx_porting
    assert 'checks["reference-accessor-lvalue-identity"]' in mlx_porting
    assert "reference accessor proof accounting is incomplete" in mlx_porting
    assert "reference accessor {target} storage evidence is incomplete" in mlx_porting
    assert (
        "reference accessor {target} const-read evidence is incomplete" in mlx_porting
    )
    assert "reference accessor {target} native validation must be" in mlx_porting
    assert "MLX_OPENGL_TOOLCHAIN_FRONTIER_SOURCES" in mlx_porting
    assert "MLX_OPENGL_INDEX_RANGE_ASSERTIONS" in mlx_porting
    assert "MLX_OPENGL_INDEX_RANGE_ASSERTION_EXPRESSIONS" in mlx_porting
    assert "MLX_OPENGL_INDEX_RANGE_ASSERTION_MINIMUM" in mlx_porting
    assert "MLX_OPENGL_INDEX_RANGE_ASSERTION_MAXIMUM" in mlx_porting
    assert "MLX_OPENGL_DYNAMIC_WORKGROUP_FRONTIER_SOURCES" in mlx_porting
    assert "expected 8 OpenGL attempted frontier sources" in mlx_porting
    assert "expected 3 OpenGL toolchain frontier sources" in mlx_porting
    assert "OpenGL target-split frontier accounting is incomplete" in mlx_porting
    assert "OpenGL workgroup blocker evidence changed" in mlx_porting
    assert "OpenGL frontier toolchain must validate every emitted source" in (
        mlx_porting
    )
    assert 'opengl["indexRangeAssertionEvidence"]' in mlx_porting
    assert "OpenGL index-range portability evidence is incomplete" in mlx_porting
    assert 'opengl["runtimeIntegrationIncluded"] is not False' in mlx_porting
    assert "mlx/backend/metal/kernels/binary_two.metal" in mlx_porting
    assert "mesa-vulkan-drivers" in mlx_porting
    assert "vulkan==1.3.275.1" in mlx_porting
    assert "vulkaninfo --summary" in mlx_porting
    assert "mlx-full-corpus-scout:" in mlx_porting
    assert "MLX full-corpus artifact scout" in mlx_porting
    assert (
        "github.event_name == 'schedule' || github.event_name == 'workflow_dispatch'"
        in mlx_porting
    )
    assert "--mode full-corpus" in mlx_porting
    assert "full-corpus-summary.json" in mlx_porting
    assert "out-full-corpus" in mlx_porting
    assert "name: mlx-full-corpus-scout" in mlx_porting
    assert "include-hidden-files: true" in mlx_porting
    assert "retention-days: 30" in mlx_porting
    assert 'command_name="validate-directx-frontier-toolchain"' in harness
    assert 'command_name="validate-vulkan-frontier-toolchain"' in harness
    assert "require_directx_toolchain" in harness
    assert '"--run-toolchains"' in harness
    assert '"--validate"' in harness
    assert "FULL_CORPUS_EXPECTED_ARTIFACT_COUNT" in harness
    assert "FULL_CORPUS_EXPECTED_TRANSLATED_ARTIFACT_COUNT" in harness
    assert "FULL_CORPUS_EXPECTED_FENCE_FAILURE_COUNT" in harness
    assert "FULL_CORPUS_MAX_TEMPLATE_SPECIALIZATIONS = 4096" in harness
    assert "FULL_CORPUS_MAX_TEMPLATE_MATERIALIZATION_WORK = 131072" in harness
    assert "FULL_CORPUS_TRANSLATION_TIMEOUT_SECONDS = 900" in harness
    assert "blocked-by-tracked-issues" in harness
    assert "without tracked issue references" in harness
    assert "runtime-readiness" in harness
    assert "runtime-test-manifest" in harness
    assert "VulkanComputeRuntime" in harness
    assert "require_vulkan_native_runtime" in harness
    for tracked_issue_number in (
        1312,
        1376,
        1388,
        1392,
        1394,
        1471,
    ):
        assert f"https://github.com/CrossGL/crosstl/issues/{tracked_issue_number}" in (
            harness
        )
    for resolved_issue_number in (
        1661,
        1184,
        1203,
        1204,
        1206,
        1205,
        1207,
        1218,
        1222,
        1238,
        1239,
        1240,
        1246,
        1248,
        1249,
        1250,
        1259,
        1260,
        1261,
        1274,
        1287,
        1329,
        1338,
        1340,
        1346,
        1355,
        1354,
        1362,
        1396,
        1452,
        1453,
        1454,
        1300,
        1317,
    ):
        assert (
            f"https://github.com/CrossGL/crosstl/issues/{resolved_issue_number}"
            not in mlx_porting
        )
        assert (
            f"https://github.com/CrossGL/crosstl/issues/{resolved_issue_number}"
            in harness
        )
    for tracked_issue_number in (1317,):
        assert (
            f"https://github.com/CrossGL/crosstl/issues/{tracked_issue_number}"
            not in mlx_porting
        )
        assert (
            f"https://github.com/CrossGL/crosstl/issues/{tracked_issue_number}"
            in harness
        )
    assert "MLX_DIRECTX_VULKAN_FRONTIER_SOURCES" in harness
    assert "MLX_DIRECTX_TOOLCHAIN_FRONTIER_SOURCES" in harness
    assert "MLX_DIRECTX_TOOLCHAIN_ENTRY_POINT_COUNTS" in harness
    assert "MLX_DIRECTX_TOOLCHAIN_ENTRY_POINT_COUNT" in harness
    assert "MLX_DYNAMIC_WORKGROUP_FRONTIER_SOURCES" in harness
    assert "MLX_DYNAMIC_WORKGROUP_DIAGNOSTIC_CODE" in harness
    assert "MLX_DYNAMIC_WORKGROUP_DISPATCH_EVIDENCE" in harness
    assert '"specializationCount": 39' in harness
    assert '"sourceEntryPointIdentityStatus"' in harness
    assert "MLX_BLOCKED_REDUCED_FRONTIER_SOURCES" in harness
    assert "_check_atomic_fence_contract" in harness
    assert "project.translate.directx-atomic-fence-unsupported" in harness
    assert "project.translate.opengl-atomic-fence-unsupported" in harness
    assert "project.translate.vulkan-atomic-fence-unsupported" in harness
    assert "directx.atomic-thread-fence-contract-lowering" in harness
    assert "opengl.atomic-thread-fence-contract-lowering" in harness
    assert "spirv.atomic-thread-fence-contract-lowering" in harness
    assert "mlx/backend/metal/kernels/binary_two.metal" in harness
    assert "mlx/backend/metal/kernels/fence.metal" in harness
    assert "mlx/backend/metal/kernels/random.metal" in harness
    assert "mlx/backend/metal/kernels/ternary.metal" in harness
    assert "arange-opengl" in harness
    assert "metalIncludesFiltered" in harness


def test_mlx_platform_runtime_workflow_preserves_pinned_checkout():
    workflow = _workflow_texts().get("demo-project-testing.yml", "")
    mlx_commit = "4367c73b60541ddd5a266ce4644fd93d20223b6e"

    assert workflow, "demo-project-testing.yml must exist"
    assert f'MLX_COMMIT: "{mlx_commit}"' in workflow
    assert any(
        fnmatchcase("tests/test_ci_workflows.py", pattern)
        for pattern in _workflow_event_paths(workflow, "pull_request")
    )
    assert '"tests/test_tools/test_ci_workflows.py"' not in workflow
    workflow = "\n".join(
        _workflow_job_section(workflow, name)
        for name in ("directx-dispatch-sequence", "opengl-dispatch-sequence")
    )
    assert workflow.count("checkout --detach FETCH_HEAD") == 2
    assert workflow.count('rev-parse HEAD)" = "$MLX_COMMIT"') == 2
    assert workflow.count("sparse-checkout set mlx/backend/metal/kernels") == 2
    assert workflow.count("CROSTL_MLX_SOURCE_ROOT: mlx-upstream") == 2
    assert "continue-on-error" not in workflow


def test_mlx_frontier_accounting_workflow_imports_available_harness_symbols():
    workflow = _workflow_texts()["demo-project-testing.yml"]
    step_start = workflow.index("- name: Verify MLX frontier accounting")
    command_marker = "          python - <<'PY'\n"
    script_start = workflow.index(command_marker, step_start) + len(command_marker)
    script_end = workflow.index("\n          PY", script_start)
    script = textwrap.dedent(workflow[script_start:script_end])
    tree = ast.parse(script)
    harness_import = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module == "demos.integrations.mlx.run_porting"
    )
    harness = __import__(
        "demos.integrations.mlx.run_porting",
        fromlist=["run_porting"],
    )

    missing = sorted(
        alias.name for alias in harness_import.names if not hasattr(harness, alias.name)
    )
    assert missing == []


def test_mlx_platform_runtime_wires_directx_graph_device_proof():
    workflow = _workflow_texts().get("demo-project-testing.yml", "")
    directx = _workflow_job_section(workflow, "directx-dispatch-sequence")
    node_id = (
        "tests/test_translator/test_runtime_graph_device.py::"
        "test_directx_dispatch_sequence_executes_shared_temporary_on_device"
    )

    assert "runs-on: windows-latest" in directx
    assert "Install DirectX Shader Compiler" in directx
    assert "Get-FileHash -Path $archive -Algorithm SHA256" in directx
    assert "cf658aacf070d3045e31b8f1f8a696c2945f37c1095019481ef7c513368db3b4" in (
        directx
    )
    assert 'python -m pip install -e ".[directx-runtime]" pytest-xdist' in directx
    assert 'CROSTL_RUN_DIRECTX_DISPATCH_SEQUENCE_DEVICE_TEST: "1"' in directx
    assert node_id in directx
    assert "python -m pytest -q -n auto" in directx
    assert "-k" not in directx


def test_mlx_platform_runtime_wires_opengl_graph_device_proof():
    workflow = _workflow_texts().get("demo-project-testing.yml", "")
    opengl = _workflow_job_section(workflow, "opengl-dispatch-sequence")
    node_id = (
        "tests/test_translator/test_runtime_graph_device.py::"
        "test_opengl_dispatch_sequence_executes_shared_temporary_on_device"
    )

    assert "runs-on: ubuntu-latest" in opengl
    assert "moderngl==5.12.0" in opengl
    assert "glslangValidator --version" in opengl
    assert 'CROSTL_RUN_OPENGL_DISPATCH_SEQUENCE_DEVICE_TEST: "1"' in opengl
    assert "EGL_PLATFORM: surfaceless" in opengl
    assert 'LIBGL_ALWAYS_SOFTWARE: "1"' in opengl
    assert "PYOPENGL_PLATFORM: egl" in opengl
    assert node_id in opengl
    assert "python -m pytest -q -n auto" in opengl
    assert "-k" not in opengl


def test_mlx_project_porting_workflow_runs_quantized_directx_proof_on_windows():
    mlx_porting = _workflow_texts().get("demo-project-testing.yml", "")

    assert "name: Prove pinned MLX quantized DirectX lowering" in mlx_porting
    assert "name: Prove pinned MLX quantized gather DirectX lowering" in mlx_porting
    assert "if: runner.os == 'Windows'" in mlx_porting
    assert "python demos/integrations/mlx/prove_quantized_directx.py" in mlx_porting
    assert "--work-dir .crosstl-mlx-porting/quantized-directx" in mlx_porting
    assert "--work-dir .crosstl-mlx-porting/quantized-gather-directx" in mlx_porting
    assert "--entry-point affine_gather_qmv_fast_float_gs_32_b_2" in mlx_porting
    assert "--require-directx-toolchain" in mlx_porting


def test_mlx_project_porting_workflow_runs_quantized_opengl_proof_on_linux():
    mlx_porting = _workflow_texts().get("demo-project-testing.yml", "")

    assert "name: Prove pinned MLX quantized OpenGL lowering" in mlx_porting
    assert "name: Prove pinned MLX quantized gather OpenGL lowering" in mlx_porting
    assert "if: runner.os == 'Linux'" in mlx_porting
    assert "python demos/integrations/mlx/prove_quantized_opengl.py" in mlx_porting
    assert "--work-dir .crosstl-mlx-porting/quantized-opengl" in mlx_porting
    assert "--work-dir .crosstl-mlx-porting/quantized-gather-opengl" in mlx_porting
    assert "--entry-point affine_gather_qmv_fast_float_gs_32_b_2" in mlx_porting
    assert "--require-opengl-toolchain" in mlx_porting


def test_mlx_project_porting_workflow_runs_backend_runtime_contracts():
    mlx_porting = _workflow_texts().get("demo-project-testing.yml", "")
    ci_coverage = _load_ci_coverage_module()

    for name, environment, test_name in (
        (
            "Validate Direct3D wave shuffle runtime contract",
            "CROSTL_RUN_DIRECTX_BOUNDED_WAVE_SHUFFLE_DEVICE_TEST",
            "directx_compute_runtime_executes_bounded_wave_shuffle_and_fill_up_on_device",
        ),
        (
            "Validate Direct3D copysign runtime contract",
            "CROSTL_RUN_DIRECTX_COPYSIGN_DEVICE_TEST",
            "directx_compute_runtime_executes_copysign_bit_patterns_on_device",
        ),
        (
            "Validate Direct3D inverse-hyperbolic runtime contract",
            "CROSTL_RUN_DIRECTX_INVERSE_HYPERBOLIC_DEVICE_TEST",
            "directx_compute_runtime_executes_inverse_hyperbolic_numerics_on_device",
        ),
    ):
        step = ci_coverage.workflow_step_section(mlx_porting, name)
        assert "if: runner.os == 'Windows'" in step
        assert f'{environment}: "1"' in step
        assert test_name in step
        assert "-n auto" in step
        assert "mlx-upstream" not in step

    directx_private_pointer_step = ci_coverage.workflow_step_section(
        mlx_porting,
        "Prove Direct3D local-struct byte-view native readback",
    )
    assert "if: runner.os == 'Windows'" in directx_private_pointer_step
    assert (
        "CROSTL_REQUIRE_PRIVATE_POINTER_RUNTIME: directx"
        in directx_private_pointer_step
    )
    assert (
        "test_private_pointer_native_readback[local-struct-byte-view-directx]"
        in directx_private_pointer_step
    )
    assert "-k" not in directx_private_pointer_step
    assert "-n auto" in directx_private_pointer_step

    opengl_step = ci_coverage.workflow_step_section(
        mlx_porting,
        "Validate OpenGL copysign runtime contract",
    )
    assert "if: runner.os == 'Linux'" in opengl_step
    assert 'CROSTL_RUN_OPENGL_COPYSIGN_DEVICE_TEST: "1"' in opengl_step
    assert "opengl_compute_runtime_executes_copysign_bit_patterns_on_device" in (
        opengl_step
    )
    assert "EGL_PLATFORM: surfaceless" in opengl_step
    assert 'LIBGL_ALWAYS_SOFTWARE: "1"' in opengl_step
    assert "PYOPENGL_PLATFORM: egl" in opengl_step
    assert "-n auto" in opengl_step
    assert "mlx-upstream" not in opengl_step

    opengl_private_pointer_step = ci_coverage.workflow_step_section(
        mlx_porting,
        "Prove OpenGL local-struct byte-view native readback",
    )
    assert "if: runner.os == 'Linux'" in opengl_private_pointer_step
    assert (
        "CROSTL_REQUIRE_PRIVATE_POINTER_RUNTIME: opengl" in opengl_private_pointer_step
    )
    assert (
        "test_private_pointer_native_readback[local-struct-byte-view-opengl]"
        in opengl_private_pointer_step
    )
    assert "-k" not in opengl_private_pointer_step
    assert "EGL_PLATFORM: surfaceless" in opengl_private_pointer_step
    assert 'LIBGL_ALWAYS_SOFTWARE: "1"' in opengl_private_pointer_step
    assert "PYOPENGL_PLATFORM: egl" in opengl_private_pointer_step
    assert "-n auto" in opengl_private_pointer_step

    vulkan_step = ci_coverage.workflow_step_section(
        mlx_porting,
        "Validate generated OpenGL subgroup contract through Vulkan",
    )
    assert "if: runner.os == 'Linux'" in vulkan_step
    assert 'CROSTL_RUN_OPENGL_GLSL_VULKAN_DEVICE_TEST: "1"' in vulkan_step
    assert "opengl_glsl_wave_shuffle_executes_via_vulkan_on_device" in vulkan_step
    assert "VK_DRIVER_FILES" in vulkan_step
    assert "VK_ICD_FILENAMES" in vulkan_step
    assert "lvp_icd*.json" in vulkan_step
    assert "vulkaninfo --summary" in vulkan_step
    assert "-n auto" in vulkan_step
    assert "mlx-upstream" not in vulkan_step


def test_mlx_project_porting_workflow_requires_native_artifact_refresh():
    workflow = _workflow_texts().get("demo-project-testing.yml", "")
    coverage = _load_ci_coverage_module()
    step = coverage.workflow_step_section(
        workflow, "Prove native artifact contract refresh"
    )
    assert 'CROSTL_REQUIRE_ARTIFACT_REFRESH_COMPILER: "1"' in step
    assert "tests/test_artifact_contract_refresh.py" in step
    assert "tests/test_artifact_contract_refresh.py::" not in step
    assert "-n auto" in step
    assert "if:" not in step
    assert "continue-on-error" not in step
    assert "--basetemp support/generated/artifact-refresh-native" in step
    assert "--junitxml support/generated/artifact-refresh-native.xml" in step
    assert workflow.index("Install Linux SPIR-V tools") < workflow.index(
        "Prove native artifact contract refresh"
    )
    assert workflow.index("Install Windows DirectX Shader Compiler") < workflow.index(
        "Prove native artifact contract refresh"
    )
    assert workflow.index("Install macOS Metal Toolchain") < workflow.index(
        "Prove native artifact contract refresh"
    )
    for path in (
        "tools/refresh_artifact_contract.py",
        "tests/test_artifact_contract_refresh.py",
    ):
        _assert_workflow_triggers(workflow, path)
    upload = coverage.workflow_step_section(
        workflow, "Upload native artifact refresh evidence"
    )
    assert "if: always()" in upload
    assert "artifact-refresh-native-${{ runner.os }}" in upload
    assert "support/generated/artifact-refresh-native" in upload


def test_mlx_project_porting_workflow_runs_native_loader_dispatch_bridge():
    mlx_porting = _workflow_texts().get("demo-project-testing.yml", "")
    ci_coverage = _load_ci_coverage_module()
    integration_test = (
        "tests/test_translator/test_native_loader_dispatch_integration.py"
    )
    limits_step = ci_coverage.workflow_step_section(
        mlx_porting, "Validate native dispatch limits"
    )
    assert "tests/test_translator/test_native_dispatch_limits.py" in limits_step
    assert "-n auto" in limits_step
    assert "if:" not in limits_step

    _assert_workflow_triggers(mlx_porting, integration_test)
    _assert_workflow_triggers(
        mlx_porting, "tests/test_translator/test_native_dispatch_limits.py"
    )

    directx_step = ci_coverage.workflow_step_section(
        mlx_porting,
        "Prove native loader Direct3D dispatch bridge",
    )
    assert "if: runner.os == 'Windows'" in directx_step
    assert 'CROSTL_RUN_NATIVE_LOADER_DIRECTX_DEVICE_TEST: "1"' in directx_step
    assert (
        f"{integration_test}::"
        "test_native_loader_descriptor_executes_directx_on_device" in directx_step
    )
    assert "-n auto" in directx_step
    assert "-k" not in directx_step
    assert "mlx-upstream" not in directx_step
    assert f"{integration_test}::test_native_directx_workgroup_limits" in directx_step

    opengl_step = ci_coverage.workflow_step_section(
        mlx_porting,
        "Prove native loader OpenGL dispatch bridge",
    )
    assert "if: runner.os == 'Linux'" in opengl_step
    assert 'CROSTL_RUN_NATIVE_LOADER_OPENGL_DEVICE_TEST: "1"' in opengl_step
    assert (
        f"{integration_test}::"
        "test_native_loader_descriptor_executes_opengl_on_device" in opengl_step
    )
    assert "EGL_PLATFORM: surfaceless" in opengl_step
    assert 'LIBGL_ALWAYS_SOFTWARE: "1"' in opengl_step
    assert "PYOPENGL_PLATFORM: egl" in opengl_step
    assert "-n auto" in opengl_step
    assert "-k" not in opengl_step
    assert "mlx-upstream" not in opengl_step
    assert f"{integration_test}::test_native_opengl_workgroup_limits" in opengl_step
    assert (
        f"{integration_test}::test_native_opengl_submission_error_is_not_success"
        in opengl_step
    )


def test_mlx_project_porting_workflow_proves_initialized_read_write_execution():
    mlx_porting = _workflow_texts().get("demo-project-testing.yml", "")
    ci_coverage = _load_ci_coverage_module()
    test_path = "tests/test_translator/test_initialized_read_write_runtime.py"

    _assert_workflow_triggers(mlx_porting, test_path)

    directx_step = ci_coverage.workflow_step_section(
        mlx_porting,
        "Prove initialized read-write Direct3D native execution",
    )
    assert "if: runner.os == 'Windows'" in directx_step
    assert 'CROSTL_RUN_INITIALIZED_READ_WRITE_DIRECTX_DEVICE_TEST: "1"' in directx_step
    assert (
        f"{test_path}::"
        "test_directx_initialized_read_write_resource_executes_on_device"
        in directx_step
    )
    assert "-n auto" in directx_step
    assert "-k" not in directx_step

    opengl_step = ci_coverage.workflow_step_section(
        mlx_porting,
        "Prove initialized read-write OpenGL native execution",
    )
    assert "if: runner.os == 'Linux'" in opengl_step
    assert 'CROSTL_RUN_INITIALIZED_READ_WRITE_OPENGL_DEVICE_TEST: "1"' in opengl_step
    assert "EGL_PLATFORM: surfaceless" in opengl_step
    assert 'LIBGL_ALWAYS_SOFTWARE: "1"' in opengl_step
    assert "PYOPENGL_PLATFORM: egl" in opengl_step
    assert (
        f"{test_path}::"
        "test_opengl_initialized_read_write_resource_executes_on_device" in opengl_step
    )
    assert "-n auto" in opengl_step
    assert "-k" not in opengl_step


def test_mlx_project_porting_workflow_proves_shared_native_allocation_execution():
    mlx_porting = _workflow_texts().get("demo-project-testing.yml", "")
    ci_coverage = _load_ci_coverage_module()
    test_path = "tests/test_translator/test_shared_native_allocations_runtime.py"

    _assert_workflow_triggers(mlx_porting, test_path)

    directx_step = ci_coverage.workflow_step_section(
        mlx_porting,
        "Prove shared allocation Direct3D native execution",
    )
    assert ci_coverage.workflow_step_after(
        mlx_porting,
        "Prove shared allocation Direct3D native execution",
        "Install Windows DirectX Shader Compiler",
    )
    assert ci_coverage.workflow_step_after(
        mlx_porting,
        "Prove shared allocation Direct3D native execution",
        "Install Windows Direct3D runtime dependencies",
    )
    assert "if: runner.os == 'Windows'" in directx_step
    assert (
        'CROSTL_RUN_SHARED_NATIVE_ALLOCATION_DIRECTX_DEVICE_TEST: "1"' in directx_step
    )
    assert (
        f"{test_path}::test_directx_shared_native_allocation_executes_on_device"
        in directx_step
    )
    assert "-n auto" in directx_step
    assert "-k" not in directx_step
    assert "mlx-upstream" not in directx_step

    opengl_step = ci_coverage.workflow_step_section(
        mlx_porting,
        "Prove shared allocation OpenGL native execution",
    )
    assert ci_coverage.workflow_step_after(
        mlx_porting,
        "Prove shared allocation OpenGL native execution",
        "Install Linux SPIR-V tools",
    )
    assert ci_coverage.workflow_step_after(
        mlx_porting,
        "Prove shared allocation OpenGL native execution",
        "Install Linux runtime dependencies",
    )
    assert "if: runner.os == 'Linux'" in opengl_step
    assert 'CROSTL_RUN_SHARED_NATIVE_ALLOCATION_OPENGL_DEVICE_TEST: "1"' in opengl_step
    assert "EGL_PLATFORM: surfaceless" in opengl_step
    assert 'LIBGL_ALWAYS_SOFTWARE: "1"' in opengl_step
    assert "PYOPENGL_PLATFORM: egl" in opengl_step
    assert (
        f"{test_path}::test_opengl_shared_native_allocation_executes_on_device"
        in opengl_step
    )
    assert "-n auto" in opengl_step
    assert "-k" not in opengl_step
    assert "mlx-upstream" not in opengl_step


def test_mlx_project_porting_workflow_runs_pinned_arange_native_loader_proof():
    mlx_porting = _workflow_texts().get("demo-project-testing.yml", "")
    ci_coverage = _load_ci_coverage_module()
    test_path = "demos/integrations/mlx/tests/kernels/test_arange_native_loader.py"

    _assert_workflow_triggers(mlx_porting, test_path)
    step = ci_coverage.workflow_step_section(
        mlx_porting,
        "Prove pinned MLX arange OpenGL native-loader execution",
    )
    assert "if: runner.os == 'Linux'" in step
    assert "CROSTL_MLX_ROOT: ${{ github.workspace }}/mlx-current-upstream" in step
    assert 'CROSTL_REQUIRE_MLX_ARANGE_OPENGL_NATIVE_LOADER: "1"' in step
    assert "EGL_PLATFORM: surfaceless" in step
    assert 'LIBGL_ALWAYS_SOFTWARE: "1"' in step
    assert "PYOPENGL_PLATFORM: egl" in step
    assert (
        f"{test_path}::"
        "test_pinned_mlx_arange_executes_through_opengl_native_loader" in step
    )
    assert "-n auto" in step
    assert "-k" not in step

    directx_step = ci_coverage.workflow_step_section(
        mlx_porting,
        "Prove pinned MLX arange Direct3D native-loader execution",
    )
    assert "if: runner.os == 'Windows'" in directx_step
    assert (
        "CROSTL_MLX_ROOT: ${{ github.workspace }}/mlx-current-upstream" in directx_step
    )
    assert 'CROSTL_REQUIRE_MLX_ARANGE_DIRECTX_NATIVE_LOADER: "1"' in directx_step
    assert (
        f"{test_path}::"
        "test_pinned_mlx_arange_executes_through_directx_native_loader" in directx_step
    )
    assert "-n auto" in directx_step
    assert "-k" not in directx_step


def test_mlx_project_porting_workflow_runs_pinned_binary_native_loader_proof():
    mlx_porting = _workflow_texts().get("demo-project-testing.yml", "")
    ci_coverage = _load_ci_coverage_module()
    test_path = "demos/integrations/mlx/tests/kernels/test_binary_native_loader.py"

    _assert_workflow_triggers(mlx_porting, test_path)
    opengl_step = ci_coverage.workflow_step_section(
        mlx_porting,
        "Prove pinned MLX binary add OpenGL native-loader execution",
    )
    assert "if: runner.os == 'Linux'" in opengl_step
    assert (
        "CROSTL_MLX_ROOT: ${{ github.workspace }}/mlx-current-upstream" in opengl_step
    )
    assert 'CROSTL_REQUIRE_MLX_BINARY_OPENGL_NATIVE_LOADER: "1"' in opengl_step
    assert "EGL_PLATFORM: surfaceless" in opengl_step
    assert 'LIBGL_ALWAYS_SOFTWARE: "1"' in opengl_step
    assert "PYOPENGL_PLATFORM: egl" in opengl_step
    assert (
        f"{test_path}::"
        "test_pinned_mlx_binary_add_executes_through_opengl_native_loader"
        in opengl_step
    )
    assert "-n auto" in opengl_step
    assert "-k" not in opengl_step

    directx_step = ci_coverage.workflow_step_section(
        mlx_porting,
        "Prove pinned MLX binary add Direct3D native-loader execution",
    )
    assert "if: runner.os == 'Windows'" in directx_step
    assert (
        "CROSTL_MLX_ROOT: ${{ github.workspace }}/mlx-current-upstream" in directx_step
    )
    assert 'CROSTL_REQUIRE_MLX_BINARY_DIRECTX_NATIVE_LOADER: "1"' in directx_step
    assert (
        f"{test_path}::"
        "test_pinned_mlx_binary_add_executes_through_directx_native_loader"
        in directx_step
    )
    assert "-n auto" in directx_step
    assert "-k" not in directx_step


def test_mlx_project_porting_workflow_runs_pinned_copy_native_loader_proof():
    mlx_porting = _workflow_texts().get("demo-project-testing.yml", "")
    ci_coverage = _load_ci_coverage_module()
    test_path = "demos/integrations/mlx/tests/kernels/test_copy_native_loader.py"

    _assert_workflow_triggers(mlx_porting, test_path)
    opengl_step = ci_coverage.workflow_step_section(
        mlx_porting,
        "Prove pinned MLX copy OpenGL native-loader execution",
    )
    assert "if: runner.os == 'Linux'" in opengl_step
    assert (
        "CROSTL_MLX_ROOT: ${{ github.workspace }}/mlx-current-upstream" in opengl_step
    )
    assert 'CROSTL_REQUIRE_MLX_COPY_OPENGL_NATIVE_LOADER: "1"' in opengl_step
    assert "EGL_PLATFORM: surfaceless" in opengl_step
    assert 'LIBGL_ALWAYS_SOFTWARE: "1"' in opengl_step
    assert "PYOPENGL_PLATFORM: egl" in opengl_step
    assert (
        f"{test_path}::"
        "test_pinned_mlx_copy_executes_through_opengl_native_loader" in opengl_step
    )
    assert "-n auto" in opengl_step
    assert "-k" not in opengl_step

    directx_step = ci_coverage.workflow_step_section(
        mlx_porting,
        "Prove pinned MLX copy Direct3D native-loader execution",
    )
    assert "if: runner.os == 'Windows'" in directx_step
    assert (
        "CROSTL_MLX_ROOT: ${{ github.workspace }}/mlx-current-upstream" in directx_step
    )
    assert 'CROSTL_REQUIRE_MLX_COPY_DIRECTX_NATIVE_LOADER: "1"' in directx_step
    assert (
        f"{test_path}::"
        "test_pinned_mlx_copy_executes_through_directx_native_loader" in directx_step
    )
    assert "-n auto" in directx_step
    assert "-k" not in directx_step


def test_mlx_project_porting_workflow_runs_pinned_logsumexp_native_loader_proof():
    mlx_porting = _workflow_texts().get("demo-project-testing.yml", "")
    ci_coverage = _load_ci_coverage_module()
    test_path = "demos/integrations/mlx/tests/kernels/test_logsumexp_native_loader.py"

    _assert_workflow_triggers(mlx_porting, test_path)
    step = ci_coverage.workflow_step_section(
        mlx_porting,
        "Prove pinned MLX LogSumExp Direct3D native-loader execution",
    )
    assert "if: runner.os == 'Windows'" in step
    assert "CROSTL_MLX_ROOT: ${{ github.workspace }}/mlx-current-upstream" in step
    assert 'CROSTL_REQUIRE_MLX_LOGSUMEXP_DIRECTX_NATIVE_LOADER: "1"' in step
    assert (
        f"{test_path}::"
        "test_pinned_mlx_logsumexp_executes_through_directx_native_loader" in step
    )
    assert "-n auto" in step
    assert "-k" not in step
    assert ci_coverage.workflow_step_after(
        mlx_porting,
        "Prove pinned MLX LogSumExp Direct3D native-loader execution",
        "Checkout current MLX runtime proof corpus",
    )
    assert mlx_porting.index(
        "- name: Prove pinned MLX LogSumExp Direct3D native-loader execution"
    ) < mlx_porting.index("- name: Run MLX project-porting checks")

    runtime_step = ci_coverage.workflow_step_section(
        mlx_porting,
        "Prove pinned MLX LogSumExp OpenGL native-loader execution",
    )
    assert "if: runner.os == 'Linux'" in runtime_step
    assert (
        "CROSTL_MLX_ROOT: ${{ github.workspace }}/mlx-current-upstream" in runtime_step
    )
    assert 'CROSTL_REQUIRE_MLX_LOGSUMEXP_OPENGL_NATIVE_LOADER: "1"' in runtime_step
    assert "EGL_PLATFORM: surfaceless" in runtime_step
    assert 'LIBGL_ALWAYS_SOFTWARE: "1"' in runtime_step
    assert "PYOPENGL_PLATFORM: egl" in runtime_step
    assert (
        f"{test_path}::"
        "test_pinned_mlx_logsumexp_executes_through_opengl_native_loader"
        in runtime_step
    )
    assert "-n auto" in runtime_step
    assert "-k" not in runtime_step
    assert ci_coverage.workflow_step_after(
        mlx_porting,
        "Prove pinned MLX LogSumExp OpenGL native-loader execution",
        "Install Linux runtime dependencies",
    )
    assert ci_coverage.workflow_step_after(
        mlx_porting,
        "Prove pinned MLX LogSumExp OpenGL native-loader execution",
        "Checkout current MLX runtime proof corpus",
    )

    opengl_step = ci_coverage.workflow_step_section(
        mlx_porting,
        "Validate pinned MLX LogSumExp OpenGL dispatch artifacts",
    )
    assert "if: runner.os == 'Linux'" in opengl_step
    assert (
        "CROSTL_MLX_ROOT: ${{ github.workspace }}/mlx-current-upstream" in opengl_step
    )
    assert 'CROSTL_REQUIRE_MLX_LOGSUMEXP_OPENGL_TOOLCHAIN: "1"' in opengl_step
    assert (
        f"{test_path}::"
        "test_pinned_mlx_logsumexp_translates_to_guarded_opengl_artifacts"
        in opengl_step
    )
    assert "-n auto" in opengl_step
    assert "-k" not in opengl_step
    assert ci_coverage.workflow_step_after(
        mlx_porting,
        "Validate pinned MLX LogSumExp OpenGL dispatch artifacts",
        "Checkout current MLX runtime proof corpus",
    )


def test_mlx_project_porting_workflow_runs_pinned_rms_norm_native_loader_proof():
    mlx_porting = _workflow_texts().get("demo-project-testing.yml", "")
    ci_coverage = _load_ci_coverage_module()
    test_path = "demos/integrations/mlx/tests/kernels/test_rms_norm_native_loader.py"

    _assert_workflow_triggers(mlx_porting, test_path)
    directx_step = ci_coverage.workflow_step_section(
        mlx_porting,
        "Prove pinned MLX RMSNorm Direct3D native-loader execution",
    )
    assert "if: runner.os == 'Windows'" in directx_step
    assert (
        "CROSTL_MLX_ROOT: ${{ github.workspace }}/mlx-current-upstream" in directx_step
    )
    assert 'CROSTL_REQUIRE_MLX_RMS_NORM_DIRECTX_NATIVE_LOADER: "1"' in (directx_step)
    assert (
        f"{test_path}::"
        "test_pinned_mlx_rms_norm_executes_through_directx_native_loader"
        in directx_step
    )
    assert "-n auto" in directx_step
    assert "-k" not in directx_step
    assert ci_coverage.workflow_step_after(
        mlx_porting,
        "Prove pinned MLX RMSNorm Direct3D native-loader execution",
        "Checkout current MLX runtime proof corpus",
    )

    opengl_step = ci_coverage.workflow_step_section(
        mlx_porting,
        "Prove pinned MLX RMSNorm OpenGL native-loader execution",
    )
    assert "if: runner.os == 'Linux'" in opengl_step
    assert (
        "CROSTL_MLX_ROOT: ${{ github.workspace }}/mlx-current-upstream" in opengl_step
    )
    assert 'CROSTL_REQUIRE_MLX_RMS_NORM_OPENGL_NATIVE_LOADER: "1"' in (opengl_step)
    assert "EGL_PLATFORM: surfaceless" in opengl_step
    assert 'LIBGL_ALWAYS_SOFTWARE: "1"' in opengl_step
    assert "PYOPENGL_PLATFORM: egl" in opengl_step
    assert (
        f"{test_path}::"
        "test_pinned_mlx_rms_norm_executes_through_opengl_native_loader" in opengl_step
    )
    assert "-n auto" in opengl_step
    assert "-k" not in opengl_step
    assert ci_coverage.workflow_step_after(
        mlx_porting,
        "Prove pinned MLX RMSNorm OpenGL native-loader execution",
        "Install Linux runtime dependencies",
    )
    assert ci_coverage.workflow_step_after(
        mlx_porting,
        "Prove pinned MLX RMSNorm OpenGL native-loader execution",
        "Checkout current MLX runtime proof corpus",
    )


def test_mlx_project_porting_workflow_runs_pinned_rms_norm_vjp_native_loader_proof():
    mlx_porting = _workflow_texts().get("demo-project-testing.yml", "")
    ci_coverage = _load_ci_coverage_module()
    test_path = (
        "demos/integrations/mlx/tests/kernels/test_rms_norm_vjp_native_loader.py"
    )

    _assert_workflow_triggers(mlx_porting, test_path)
    directx_step = ci_coverage.workflow_step_section(
        mlx_porting,
        "Prove pinned MLX RMSNorm VJP Direct3D native-loader execution",
    )
    assert "if: runner.os == 'Windows'" in directx_step
    assert (
        "CROSTL_MLX_ROOT: ${{ github.workspace }}/mlx-current-upstream" in directx_step
    )
    assert 'CROSTL_REQUIRE_MLX_RMS_NORM_VJP_DIRECTX_NATIVE_LOADER: "1"' in directx_step
    assert (
        f"{test_path}::"
        "test_pinned_mlx_rms_norm_vjp_executes_through_directx_native_loader"
        in directx_step
    )
    assert "-n auto" in directx_step
    assert "-k" not in directx_step
    assert ci_coverage.workflow_step_after(
        mlx_porting,
        "Prove pinned MLX RMSNorm VJP Direct3D native-loader execution",
        "Checkout current MLX runtime proof corpus",
    )

    opengl_step = ci_coverage.workflow_step_section(
        mlx_porting,
        "Prove pinned MLX RMSNorm VJP OpenGL native-loader execution",
    )
    assert "if: runner.os == 'Linux'" in opengl_step
    assert (
        "CROSTL_MLX_ROOT: ${{ github.workspace }}/mlx-current-upstream" in opengl_step
    )
    assert 'CROSTL_REQUIRE_MLX_RMS_NORM_VJP_OPENGL_NATIVE_LOADER: "1"' in opengl_step
    assert "EGL_PLATFORM: surfaceless" in opengl_step
    assert 'LIBGL_ALWAYS_SOFTWARE: "1"' in opengl_step
    assert "PYOPENGL_PLATFORM: egl" in opengl_step
    assert (
        f"{test_path}::"
        "test_pinned_mlx_rms_norm_vjp_executes_through_opengl_native_loader"
        in opengl_step
    )
    assert "-n auto" in opengl_step
    assert "-k" not in opengl_step
    assert ci_coverage.workflow_step_after(
        mlx_porting,
        "Prove pinned MLX RMSNorm VJP OpenGL native-loader execution",
        "Install Linux runtime dependencies",
    )
    assert ci_coverage.workflow_step_after(
        mlx_porting,
        "Prove pinned MLX RMSNorm VJP OpenGL native-loader execution",
        "Checkout current MLX runtime proof corpus",
    )


def test_mlx_project_porting_workflow_runs_pinned_layer_norm_native_loader_proof():
    mlx_porting = _workflow_texts().get("demo-project-testing.yml", "")
    ci_coverage = _load_ci_coverage_module()
    test_path = "demos/integrations/mlx/tests/kernels/test_layer_norm_native_loader.py"

    _assert_workflow_triggers(mlx_porting, test_path)
    directx_step = ci_coverage.workflow_step_section(
        mlx_porting,
        "Prove pinned MLX LayerNorm Direct3D native-loader execution",
    )
    assert "if: runner.os == 'Windows'" in directx_step
    assert (
        "CROSTL_MLX_ROOT: ${{ github.workspace }}/mlx-current-upstream" in directx_step
    )
    assert 'CROSTL_REQUIRE_MLX_LAYER_NORM_DIRECTX_NATIVE_LOADER: "1"' in directx_step
    assert (
        f"{test_path}::"
        "test_pinned_mlx_layer_norm_executes_through_directx_native_loader"
        in directx_step
    )
    assert "-n auto" in directx_step
    assert "-k" not in directx_step
    assert ci_coverage.workflow_step_after(
        mlx_porting,
        "Prove pinned MLX LayerNorm Direct3D native-loader execution",
        "Checkout current MLX runtime proof corpus",
    )

    opengl_step = ci_coverage.workflow_step_section(
        mlx_porting,
        "Prove pinned MLX LayerNorm OpenGL native-loader execution",
    )
    assert "if: runner.os == 'Linux'" in opengl_step
    assert (
        "CROSTL_MLX_ROOT: ${{ github.workspace }}/mlx-current-upstream" in opengl_step
    )
    assert 'CROSTL_REQUIRE_MLX_LAYER_NORM_OPENGL_NATIVE_LOADER: "1"' in opengl_step
    assert "EGL_PLATFORM: surfaceless" in opengl_step
    assert 'LIBGL_ALWAYS_SOFTWARE: "1"' in opengl_step
    assert "PYOPENGL_PLATFORM: egl" in opengl_step
    assert (
        f"{test_path}::"
        "test_pinned_mlx_layer_norm_executes_through_opengl_native_loader"
        in opengl_step
    )
    assert "-n auto" in opengl_step
    assert "-k" not in opengl_step
    assert ci_coverage.workflow_step_after(
        mlx_porting,
        "Prove pinned MLX LayerNorm OpenGL native-loader execution",
        "Install Linux runtime dependencies",
    )
    assert ci_coverage.workflow_step_after(
        mlx_porting,
        "Prove pinned MLX LayerNorm OpenGL native-loader execution",
        "Checkout current MLX runtime proof corpus",
    )


def test_mlx_project_porting_workflow_runs_pinned_layer_norm_vjp_native_loader_proof():
    mlx_porting = _workflow_texts().get("demo-project-testing.yml", "")
    ci_coverage = _load_ci_coverage_module()
    test_path = (
        "demos/integrations/mlx/tests/kernels/test_layer_norm_vjp_native_loader.py"
    )

    _assert_workflow_triggers(mlx_porting, test_path)
    directx_step = ci_coverage.workflow_step_section(
        mlx_porting,
        "Prove pinned MLX LayerNorm VJP Direct3D native-loader execution",
    )
    assert "if: runner.os == 'Windows'" in directx_step
    assert (
        "CROSTL_MLX_ROOT: ${{ github.workspace }}/mlx-current-upstream" in directx_step
    )
    assert (
        'CROSTL_REQUIRE_MLX_LAYER_NORM_VJP_DIRECTX_NATIVE_LOADER: "1"' in directx_step
    )
    assert (
        f"{test_path}::"
        "test_pinned_mlx_layer_norm_vjp_executes_through_directx_native_loader"
        in directx_step
    )
    assert "-n auto" in directx_step
    assert "-k" not in directx_step
    assert ci_coverage.workflow_step_after(
        mlx_porting,
        "Prove pinned MLX LayerNorm VJP Direct3D native-loader execution",
        "Checkout current MLX runtime proof corpus",
    )

    opengl_step = ci_coverage.workflow_step_section(
        mlx_porting,
        "Prove pinned MLX LayerNorm VJP OpenGL native-loader execution",
    )
    assert "if: runner.os == 'Linux'" in opengl_step
    assert (
        "CROSTL_MLX_ROOT: ${{ github.workspace }}/mlx-current-upstream" in opengl_step
    )
    assert 'CROSTL_REQUIRE_MLX_LAYER_NORM_VJP_OPENGL_NATIVE_LOADER: "1"' in opengl_step
    assert "EGL_PLATFORM: surfaceless" in opengl_step
    assert 'LIBGL_ALWAYS_SOFTWARE: "1"' in opengl_step
    assert "PYOPENGL_PLATFORM: egl" in opengl_step
    assert (
        f"{test_path}::"
        "test_pinned_mlx_layer_norm_vjp_executes_through_opengl_native_loader"
        in opengl_step
    )
    assert "-n auto" in opengl_step
    assert "-k" not in opengl_step
    assert ci_coverage.workflow_step_after(
        mlx_porting,
        "Prove pinned MLX LayerNorm VJP OpenGL native-loader execution",
        "Install Linux runtime dependencies",
    )
    assert ci_coverage.workflow_step_after(
        mlx_porting,
        "Prove pinned MLX LayerNorm VJP OpenGL native-loader execution",
        "Checkout current MLX runtime proof corpus",
    )


def test_mlx_project_porting_workflow_runs_pinned_arg_reduce_native_loader_proof():
    mlx_porting = _workflow_texts().get("demo-project-testing.yml", "")
    ci_coverage = _load_ci_coverage_module()
    test_path = "demos/integrations/mlx/tests/kernels/test_arg_reduce_native_loader.py"

    _assert_workflow_triggers(mlx_porting, test_path)
    directx_step_name = "Prove pinned MLX arg-reduce Direct3D native-loader execution"
    directx_step = ci_coverage.workflow_step_section(mlx_porting, directx_step_name)
    assert "if: runner.os == 'Windows'" in directx_step
    assert (
        "CROSTL_MLX_ROOT: ${{ github.workspace }}/mlx-current-upstream" in directx_step
    )
    assert 'CROSTL_REQUIRE_MLX_ARG_REDUCE_DIRECTX_NATIVE_LOADER: "1"' in directx_step
    assert (
        f"{test_path}::"
        "test_pinned_mlx_arg_reduce_executes_through_directx_native_loader"
        in directx_step
    )
    assert "-n auto" in directx_step
    assert "-k" not in directx_step
    assert ci_coverage.workflow_step_after(
        mlx_porting,
        directx_step_name,
        "Checkout current MLX runtime proof corpus",
    )

    opengl_step_name = "Prove pinned MLX arg-reduce OpenGL native-loader execution"
    opengl_step = ci_coverage.workflow_step_section(mlx_porting, opengl_step_name)
    assert "if: runner.os == 'Linux'" in opengl_step
    assert (
        "CROSTL_MLX_ROOT: ${{ github.workspace }}/mlx-current-upstream" in opengl_step
    )
    assert 'CROSTL_REQUIRE_MLX_ARG_REDUCE_OPENGL_TOOLCHAIN: "1"' in opengl_step
    assert 'CROSTL_REQUIRE_MLX_ARG_REDUCE_OPENGL_NATIVE_LOADER: "1"' in opengl_step
    assert "EGL_PLATFORM: surfaceless" in opengl_step
    assert 'LIBGL_ALWAYS_SOFTWARE: "1"' in opengl_step
    assert "MESA_LOADER_DRIVER_OVERRIDE: llvmpipe" in opengl_step
    assert "PYOPENGL_PLATFORM: egl" in opengl_step
    assert (
        f"{test_path}::"
        "test_pinned_mlx_arg_reduce_executes_through_opengl_native_loader"
        in opengl_step
    )
    assert "-n auto" in opengl_step
    assert "-k" not in opengl_step
    assert ci_coverage.workflow_step_after(
        mlx_porting,
        opengl_step_name,
        "Install Linux runtime dependencies",
    )
    assert ci_coverage.workflow_step_after(
        mlx_porting,
        opengl_step_name,
        "Checkout current MLX runtime proof corpus",
    )


def test_mlx_project_porting_workflow_runs_pinned_attention_native_loader_proof():
    mlx_porting = _workflow_texts().get("demo-project-testing.yml", "")
    ci_coverage = _load_ci_coverage_module()
    test_path = "demos/integrations/mlx/tests/kernels/test_scaled_dot_product_attention_native_loader.py"

    _assert_workflow_triggers(mlx_porting, test_path)
    directx_step_name = (
        "Prove pinned MLX scaled-attention Direct3D native-loader execution"
    )
    directx_step = ci_coverage.workflow_step_section(mlx_porting, directx_step_name)
    assert "if: runner.os == 'Windows'" in directx_step
    assert (
        "CROSTL_MLX_ROOT: ${{ github.workspace }}/mlx-current-upstream" in directx_step
    )
    assert 'CROSTL_REQUIRE_MLX_ATTENTION_DIRECTX_NATIVE_LOADER: "1"' in directx_step
    assert (
        f"{test_path}::"
        "test_pinned_mlx_attention_executes_through_directx_native_loader"
        in directx_step
    )
    assert "-n auto" in directx_step
    assert "-k" not in directx_step
    assert ci_coverage.workflow_step_after(
        mlx_porting,
        directx_step_name,
        "Checkout current MLX runtime proof corpus",
    )

    opengl_step_name = (
        "Prove pinned MLX scaled-attention OpenGL native-loader execution"
    )
    opengl_step = ci_coverage.workflow_step_section(mlx_porting, opengl_step_name)
    assert "if: runner.os == 'Linux'" in opengl_step
    assert (
        "CROSTL_MLX_ROOT: ${{ github.workspace }}/mlx-current-upstream" in opengl_step
    )
    assert 'CROSTL_REQUIRE_MLX_ATTENTION_OPENGL_NATIVE_LOADER: "1"' in opengl_step
    assert "EGL_PLATFORM: surfaceless" in opengl_step
    assert 'LIBGL_ALWAYS_SOFTWARE: "1"' in opengl_step
    assert "MESA_LOADER_DRIVER_OVERRIDE: llvmpipe" in opengl_step
    assert "PYOPENGL_PLATFORM: egl" in opengl_step
    assert (
        f"{test_path}::"
        "test_pinned_mlx_attention_executes_through_opengl_native_loader" in opengl_step
    )
    assert "-n auto" in opengl_step
    assert "-k" not in opengl_step
    assert ci_coverage.workflow_step_after(
        mlx_porting,
        opengl_step_name,
        "Install Linux runtime dependencies",
    )
    assert ci_coverage.workflow_step_after(
        mlx_porting,
        opengl_step_name,
        "Checkout current MLX runtime proof corpus",
    )


def test_mlx_project_porting_workflow_runs_pinned_dot_proofs():
    mlx_porting = _workflow_texts().get("demo-project-testing.yml", "")
    ci_coverage = _load_ci_coverage_module()
    test_path = "demos/integrations/mlx/tests/kernels/test_dot_native_loader.py"

    _assert_workflow_triggers(mlx_porting, test_path)
    directx_step = ci_coverage.workflow_step_section(
        mlx_porting,
        "Prove pinned MLX dot Direct3D native-loader execution",
    )
    assert "if: runner.os == 'Windows'" in directx_step
    assert (
        "CROSTL_MLX_ROOT: ${{ github.workspace }}/mlx-current-upstream" in directx_step
    )
    assert 'CROSTL_REQUIRE_MLX_DOT_DIRECTX_NATIVE_LOADER: "1"' in directx_step
    assert (
        f"{test_path}::test_pinned_mlx_dot_executes_through_directx_native_loader"
        in directx_step
    )
    assert "-n auto" in directx_step
    assert "-k" not in directx_step
    assert ci_coverage.workflow_step_after(
        mlx_porting,
        "Prove pinned MLX dot Direct3D native-loader execution",
        "Checkout current MLX runtime proof corpus",
    )

    metal_step = ci_coverage.workflow_step_section(
        mlx_porting,
        "Record pinned MLX dot Metal round-trip boundary",
    )
    assert "if: runner.os == 'macOS'" in metal_step
    assert "CROSTL_MLX_ROOT: ${{ github.workspace }}/mlx-current-upstream" in metal_step
    assert (
        f"{test_path}::test_pinned_mlx_dot_records_metal_roundtrip_boundary"
        in metal_step
    )
    assert "-n auto" in metal_step
    assert "-k" not in metal_step
    assert ci_coverage.workflow_step_after(
        mlx_porting,
        "Record pinned MLX dot Metal round-trip boundary",
        "Checkout current MLX runtime proof corpus",
    )

    opengl_step = ci_coverage.workflow_step_section(
        mlx_porting,
        "Validate pinned MLX dot OpenGL artifact",
    )
    assert "if: runner.os == 'Linux'" in opengl_step
    assert (
        "CROSTL_MLX_ROOT: ${{ github.workspace }}/mlx-current-upstream" in opengl_step
    )
    assert 'CROSTL_REQUIRE_MLX_DOT_OPENGL_TOOLCHAIN: "1"' in opengl_step
    assert (
        f"{test_path}::test_pinned_mlx_dot_translates_to_guarded_opengl_artifact"
        in opengl_step
    )
    assert "-n auto" in opengl_step
    assert "-k" not in opengl_step
    assert ci_coverage.workflow_step_after(
        mlx_porting,
        "Validate pinned MLX dot OpenGL artifact",
        "Checkout current MLX runtime proof corpus",
    )

    software_step_name = "Prove pinned MLX dot OpenGL software-subgroup execution"
    software_step = ci_coverage.workflow_step_section(
        mlx_porting,
        software_step_name,
    )
    assert "if: runner.os == 'Linux'" in software_step
    assert (
        "CROSTL_MLX_ROOT: ${{ github.workspace }}/mlx-current-upstream" in software_step
    )
    assert 'CROSTL_REQUIRE_MLX_DOT_OPENGL_NATIVE_LOADER: "1"' in software_step
    assert "EGL_PLATFORM: surfaceless" in software_step
    assert 'LIBGL_ALWAYS_SOFTWARE: "1"' in software_step
    assert "MESA_LOADER_DRIVER_OVERRIDE: llvmpipe" in software_step
    assert "PYOPENGL_PLATFORM: egl" in software_step
    assert (
        f"{test_path}::test_pinned_mlx_dot_executes_with_opengl_software_subgroups"
        in software_step
    )
    assert "-n auto" in software_step
    assert "-k" not in software_step
    assert ci_coverage.workflow_step_after(
        mlx_porting,
        software_step_name,
        "Install Linux runtime dependencies",
    )
    assert ci_coverage.workflow_step_after(
        mlx_porting,
        software_step_name,
        "Checkout current MLX runtime proof corpus",
    )


def test_mlx_project_porting_workflow_runs_affine_quantize_opengl_proof():
    mlx_porting = _workflow_texts().get("demo-project-testing.yml", "")
    ci_coverage = _load_ci_coverage_module()
    test_path = "demos/integrations/mlx/tests/kernels/test_quantized_native_loader.py"
    step_name = "Prove pinned MLX affine quantize OpenGL native-loader execution"

    _assert_workflow_triggers(mlx_porting, test_path)
    step = ci_coverage.workflow_step_section(mlx_porting, step_name)
    assert "if: runner.os == 'Linux'" in step
    assert "CROSTL_MLX_ROOT: ${{ github.workspace }}/mlx-current-upstream" in step
    assert 'CROSTL_REQUIRE_MLX_QUANTIZED_OPENGL_NATIVE_LOADER: "1"' in step
    assert "EGL_PLATFORM: surfaceless" in step
    assert 'LIBGL_ALWAYS_SOFTWARE: "1"' in step
    assert "PYOPENGL_PLATFORM: egl" in step
    assert (
        f"{test_path}::"
        "test_pinned_mlx_quantized_affine_executes_through_opengl_native_loader" in step
    )
    assert "-n auto" in step
    assert "-k" not in step
    assert ci_coverage.workflow_step_after(
        mlx_porting,
        step_name,
        "Checkout current MLX runtime proof corpus",
    )


def test_mlx_project_porting_workflow_runs_pinned_unary_proofs():
    mlx_porting = _workflow_texts().get("demo-project-testing.yml", "")
    ci_coverage = _load_ci_coverage_module()
    test_path = "demos/integrations/mlx/tests/kernels/test_unary_native_loader.py"

    _assert_workflow_triggers(mlx_porting, test_path)
    translation_step = ci_coverage.workflow_step_section(
        mlx_porting,
        "Translate pinned MLX unary Square entry",
    )
    assert "if: runner.os" not in translation_step
    assert (
        "CROSTL_MLX_ROOT: ${{ github.workspace }}/mlx-current-upstream"
        in translation_step
    )
    assert (
        f"{test_path}::test_pinned_mlx_unary_square_translates_to_selected_target"
        in translation_step
    )
    assert "-n auto" in translation_step
    assert "-k" not in translation_step
    assert ci_coverage.workflow_step_after(
        mlx_porting,
        "Translate pinned MLX unary Square entry",
        "Checkout current MLX runtime proof corpus",
    )

    directx_step = ci_coverage.workflow_step_section(
        mlx_porting,
        "Prove pinned MLX unary Square Direct3D native-loader execution",
    )
    assert "if: runner.os == 'Windows'" in directx_step
    assert (
        "CROSTL_MLX_ROOT: ${{ github.workspace }}/mlx-current-upstream" in directx_step
    )
    assert 'CROSTL_REQUIRE_MLX_UNARY_DIRECTX_NATIVE_LOADER: "1"' in directx_step
    assert (
        f"{test_path}::"
        "test_pinned_mlx_unary_square_executes_through_directx_native_loader"
        in directx_step
    )
    assert "-n auto" in directx_step
    assert "-k" not in directx_step

    metal_step = ci_coverage.workflow_step_section(
        mlx_porting,
        "Prove pinned MLX unary Square Metal round-trip",
    )
    assert "if: runner.os == 'macOS'" in metal_step
    assert "CROSTL_MLX_ROOT: ${{ github.workspace }}/mlx-current-upstream" in metal_step
    assert 'CROSTL_REQUIRE_MLX_UNARY_METAL_ROUNDTRIP: "1"' in metal_step
    assert (
        f"{test_path}::test_pinned_mlx_unary_square_roundtrips_through_metal"
        in metal_step
    )
    assert "-n auto" in metal_step
    assert "-k" not in metal_step
    assert ci_coverage.workflow_step_after(
        mlx_porting,
        "Prove pinned MLX unary Square Metal round-trip",
        "Checkout current MLX runtime proof corpus",
    )

    arccos_metal_step = ci_coverage.workflow_step_section(
        mlx_porting,
        "Prove pinned MLX unary ArcCos Metal round-trip",
    )
    assert "if: runner.os == 'macOS'" in arccos_metal_step
    assert (
        "CROSTL_MLX_ROOT: ${{ github.workspace }}/mlx-current-upstream"
        in arccos_metal_step
    )
    assert 'CROSTL_REQUIRE_MLX_UNARY_METAL_ROUNDTRIP: "1"' in arccos_metal_step
    assert (
        f"{test_path}::test_pinned_mlx_unary_arccos_roundtrips_through_metal"
        in arccos_metal_step
    )
    assert "-n auto" in arccos_metal_step
    assert "-k" not in arccos_metal_step
    assert ci_coverage.workflow_step_after(
        mlx_porting,
        "Prove pinned MLX unary ArcCos Metal round-trip",
        "Checkout current MLX runtime proof corpus",
    )

    family_metal_job = _workflow_job_section(
        mlx_porting,
        "mlx-unary-metal-roundtrip",
    )
    assert (
        "name: MLX complete unary Metal round-trip "
        "(shard ${{ matrix.shard_index }} of 5)" in family_metal_job
    )
    assert "if: github.event_name != 'schedule'" in family_metal_job
    assert "runs-on: macOS-latest" in family_metal_job
    assert "timeout-minutes: 75" in family_metal_job
    assert "fail-fast: false" in family_metal_job
    assert _matrix_values(family_metal_job, "shard_index") == {
        "0",
        "1",
        "2",
        "3",
        "4",
    }
    assert 'python-version: "3.12"' in family_metal_job
    assert "python -m pip install -e . pytest-xdist" in family_metal_job
    assert "xcrun --sdk macosx metal --version" in family_metal_job
    assert "Checkout current MLX unary corpus" in family_metal_job
    assert 'checkout --detach "$MLX_CORPUS_COMMIT"' in family_metal_job

    family_metal_step = ci_coverage.workflow_step_section(
        family_metal_job,
        "Prove current MLX complete unary family Metal round-trips",
    )
    assert "if: runner.os" not in family_metal_step
    assert (
        "CROSTL_MLX_ROOT: ${{ github.workspace }}/mlx-current-upstream"
        in family_metal_step
    )
    assert 'CROSTL_REQUIRE_MLX_UNARY_METAL_ROUNDTRIP: "1"' in family_metal_step
    assert (
        "CROSTL_MLX_UNARY_METAL_SHARD_INDEX: ${{ matrix.shard_index }}"
        in family_metal_step
    )
    assert 'CROSTL_MLX_UNARY_METAL_SHARD_COUNT: "5"' in family_metal_step
    assert (
        f"{test_path}::"
        "test_current_mlx_unary_family_roundtrips_through_metal" in family_metal_step
    )
    assert "-n auto" in family_metal_step
    assert "-k" not in family_metal_step
    matrix_job = _workflow_job_section(mlx_porting, "mlx-metal-porting")
    assert "Prove current MLX complete unary family Metal round-trips" not in matrix_job

    opengl_step = ci_coverage.workflow_step_section(
        mlx_porting,
        "Prove pinned MLX unary Square OpenGL native-loader execution",
    )
    assert "if: runner.os == 'Linux'" in opengl_step
    assert (
        "CROSTL_MLX_ROOT: ${{ github.workspace }}/mlx-current-upstream" in opengl_step
    )
    assert 'CROSTL_REQUIRE_MLX_UNARY_OPENGL_NATIVE_LOADER: "1"' in opengl_step
    assert "EGL_PLATFORM: surfaceless" in opengl_step
    assert 'LIBGL_ALWAYS_SOFTWARE: "1"' in opengl_step
    assert "PYOPENGL_PLATFORM: egl" in opengl_step
    assert (
        f"{test_path}::"
        "test_pinned_mlx_unary_square_executes_through_opengl_native_loader"
        in opengl_step
    )
    assert "-n auto" in opengl_step
    assert "-k" not in opengl_step

    arccos_translation_step = ci_coverage.workflow_step_section(
        mlx_porting,
        "Translate pinned MLX unary ArcCos entry",
    )
    assert "if: runner.os" not in arccos_translation_step
    assert (
        "CROSTL_MLX_ROOT: ${{ github.workspace }}/mlx-current-upstream"
        in arccos_translation_step
    )
    assert (
        f"{test_path}::test_pinned_mlx_unary_arccos_translates_to_selected_target"
        in arccos_translation_step
    )
    assert "-n auto" in arccos_translation_step
    assert "-k" not in arccos_translation_step

    arccos_directx_step = ci_coverage.workflow_step_section(
        mlx_porting,
        "Prove pinned MLX unary ArcCos Direct3D native-loader execution",
    )
    assert "if: runner.os == 'Windows'" in arccos_directx_step
    assert (
        "CROSTL_MLX_ROOT: ${{ github.workspace }}/mlx-current-upstream"
        in arccos_directx_step
    )
    assert 'CROSTL_REQUIRE_MLX_UNARY_DIRECTX_NATIVE_LOADER: "1"' in (
        arccos_directx_step
    )
    assert (
        f"{test_path}::"
        "test_pinned_mlx_unary_arccos_executes_through_directx_native_loader"
        in arccos_directx_step
    )
    assert "-n auto" in arccos_directx_step
    assert "-k" not in arccos_directx_step

    arccos_opengl_step = ci_coverage.workflow_step_section(
        mlx_porting,
        "Prove pinned MLX unary ArcCos OpenGL native-loader execution",
    )
    assert "if: runner.os == 'Linux'" in arccos_opengl_step
    assert (
        "CROSTL_MLX_ROOT: ${{ github.workspace }}/mlx-current-upstream"
        in arccos_opengl_step
    )
    assert 'CROSTL_REQUIRE_MLX_UNARY_OPENGL_NATIVE_LOADER: "1"' in (arccos_opengl_step)
    assert "EGL_PLATFORM: surfaceless" in arccos_opengl_step
    assert 'LIBGL_ALWAYS_SOFTWARE: "1"' in arccos_opengl_step
    assert "PYOPENGL_PLATFORM: egl" in arccos_opengl_step
    assert (
        f"{test_path}::"
        "test_pinned_mlx_unary_arccos_executes_through_opengl_native_loader"
        in arccos_opengl_step
    )
    assert "-n auto" in arccos_opengl_step
    assert "-k" not in arccos_opengl_step


def test_mlx_project_porting_workflow_runs_unary_complete_opengl_proof():
    mlx_porting = _workflow_texts().get("demo-project-testing.yml", "")
    ci_coverage = _load_ci_coverage_module()
    test_path = "demos/integrations/mlx/tests/kernels/test_unary_complete_opengl.py"

    opengl_job = _workflow_job_section(
        mlx_porting,
        "mlx-unary-complete-opengl-translation",
    )
    assert (
        "name: MLX complete unary OpenGL translation "
        "(shard ${{ matrix.shard_index }} of 5)" in opengl_job
    )
    assert "if: github.event_name != 'schedule'" in opengl_job
    assert "runs-on: ubuntu-latest" in opengl_job
    assert "timeout-minutes: 120" in opengl_job
    assert "fail-fast: false" in opengl_job
    assert _matrix_values(opengl_job, "shard_index") == {
        "0",
        "1",
        "2",
        "3",
        "4",
    }
    assert 'python-version: "3.12"' in opengl_job
    assert "sudo apt-get install -y glslang-tools spirv-tools" in opengl_job
    assert "python -m pip install -e . pytest-xdist" in opengl_job
    assert "glslangValidator --version" in opengl_job
    assert "spirv-val --version" in opengl_job
    assert "Checkout current MLX unary corpus" in opengl_job
    assert 'checkout --detach "$MLX_CORPUS_COMMIT"' in opengl_job

    opengl_step = ci_coverage.workflow_step_section(
        opengl_job,
        "Prove current MLX complete unary family OpenGL translation",
    )
    assert "if: runner.os" not in opengl_step
    assert (
        "CROSTL_MLX_ROOT: ${{ github.workspace }}/mlx-current-upstream" in opengl_step
    )
    assert 'CROSTL_REQUIRE_MLX_UNARY_OPENGL_TRANSLATION: "1"' in opengl_step
    assert (
        "CROSTL_MLX_UNARY_OPENGL_SHARD_INDEX: ${{ matrix.shard_index }}" in opengl_step
    )
    assert 'CROSTL_MLX_UNARY_OPENGL_SHARD_COUNT: "5"' in opengl_step
    assert (
        f"{test_path}::test_current_mlx_unary_family_translates_to_opengl"
        in opengl_step
    )
    assert "-n auto" in opengl_step
    assert "-k" not in opengl_step
    matrix_job = _workflow_job_section(mlx_porting, "mlx-metal-porting")
    assert "Prove current MLX complete unary family OpenGL translation" not in (
        matrix_job
    )


def test_mlx_project_porting_workflow_runs_unary_complete_directx_proof():
    mlx_porting = _workflow_texts().get("demo-project-testing.yml", "")
    ci_coverage = _load_ci_coverage_module()
    test_path = "demos/integrations/mlx/tests/kernels/test_unary_complete_directx.py"

    directx_job = _workflow_job_section(
        mlx_porting,
        "mlx-unary-complete-directx-translation",
    )
    assert (
        "name: MLX complete unary DirectX translation "
        "(shard ${{ matrix.shard_index }} of 5)" in directx_job
    )
    assert "if: github.event_name != 'schedule'" in directx_job
    assert "runs-on: ubuntu-24.04" in directx_job
    assert "timeout-minutes: 180" in directx_job
    assert "fail-fast: false" in directx_job
    assert _matrix_values(directx_job, "shard_index") == {
        "0",
        "1",
        "2",
        "3",
        "4",
    }
    assert 'python-version: "3.12"' in directx_job
    assert "python -m pip install -e . pytest-xdist" in directx_job
    assert "uses: ./.github/actions/install-linux-dxc" in directx_job
    assert 'pinned: "true"' in directx_job
    assert "shell: pwsh" not in directx_job
    assert "Checkout current MLX unary corpus" in directx_job
    assert 'checkout --detach "$MLX_CORPUS_COMMIT"' in directx_job

    directx_step = ci_coverage.workflow_step_section(
        directx_job,
        "Prove current MLX complete unary family DirectX translation",
    )
    assert "if: runner.os" not in directx_step
    assert (
        "CROSTL_MLX_ROOT: ${{ github.workspace }}/mlx-current-upstream" in directx_step
    )
    assert 'CROSTL_REQUIRE_MLX_UNARY_DIRECTX_TRANSLATION: "1"' in directx_step
    assert (
        "CROSTL_MLX_UNARY_DIRECTX_SHARD_INDEX: ${{ matrix.shard_index }}"
        in directx_step
    )
    assert 'CROSTL_MLX_UNARY_DIRECTX_SHARD_COUNT: "5"' in directx_step
    assert (
        f"{test_path}::test_current_mlx_unary_family_translates_to_directx"
        in directx_step
    )
    assert "-n auto" in directx_step
    assert "-k" not in directx_step
    _assert_workflow_triggers(mlx_porting, test_path)
    matrix_job = _workflow_job_section(mlx_porting, "mlx-metal-porting")
    assert "Prove current MLX complete unary family DirectX translation" not in (
        matrix_job
    )


def test_mlx_project_porting_workflow_runs_binary_complete_metal_proof():
    mlx_porting = _workflow_texts().get("demo-project-testing.yml", "")
    ci_coverage = _load_ci_coverage_module()
    test_path = (
        "demos/integrations/mlx/tests/kernels/test_binary_complete_metal_roundtrip.py"
    )

    binary_metal_job = _workflow_job_section(
        mlx_porting,
        "mlx-binary-complete-metal-roundtrip",
    )
    assert (
        "name: MLX complete binary Metal round-trip "
        "(shard ${{ matrix.shard_index }} of 24)" in binary_metal_job
    )
    assert "if: github.event_name != 'schedule'" in binary_metal_job
    assert "runs-on: macOS-latest" in binary_metal_job
    assert "timeout-minutes: 180" in binary_metal_job
    assert "fail-fast: false" in binary_metal_job
    assert _matrix_values(binary_metal_job, "shard_index") == {
        str(index) for index in range(24)
    }
    assert 'python-version: "3.12"' in binary_metal_job
    assert "python -m pip install -e . pytest-xdist" in binary_metal_job
    assert "xcrun --sdk macosx metal --version" in binary_metal_job
    assert "Checkout current MLX binary corpus" in binary_metal_job
    assert 'checkout --detach "$MLX_CORPUS_COMMIT"' in binary_metal_job

    binary_metal_step = ci_coverage.workflow_step_section(
        binary_metal_job,
        "Prove current MLX complete binary family Metal round-trips",
    )
    assert "if: runner.os" not in binary_metal_step
    assert (
        "CROSTL_MLX_ROOT: ${{ github.workspace }}/mlx-current-upstream"
        in binary_metal_step
    )
    assert 'CROSTL_REQUIRE_MLX_BINARY_METAL_ROUNDTRIP: "1"' in binary_metal_step
    assert (
        "CROSTL_MLX_BINARY_METAL_SHARD_INDEX: ${{ matrix.shard_index }}"
        in binary_metal_step
    )
    assert 'CROSTL_MLX_BINARY_METAL_SHARD_COUNT: "24"' in binary_metal_step
    assert (
        f"{test_path}::test_current_mlx_binary_family_roundtrips_through_metal"
        in binary_metal_step
    )
    assert "-n auto" in binary_metal_step
    assert "-k" not in binary_metal_step
    matrix_job = _workflow_job_section(mlx_porting, "mlx-metal-porting")
    assert "Prove current MLX complete binary family Metal round-trips" not in (
        matrix_job
    )


def test_mlx_project_porting_workflow_runs_binary_complete_opengl_proof():
    mlx_porting = _workflow_texts().get("demo-project-testing.yml", "")
    ci_coverage = _load_ci_coverage_module()
    test_path = "demos/integrations/mlx/tests/kernels/test_binary_complete_opengl.py"

    binary_opengl_job = _workflow_job_section(
        mlx_porting,
        "mlx-binary-complete-opengl-translation",
    )
    assert (
        "name: MLX complete binary OpenGL translation "
        "(shard ${{ matrix.shard_index }} of 24)" in binary_opengl_job
    )
    assert "if: github.event_name != 'schedule'" in binary_opengl_job
    assert "runs-on: ubuntu-latest" in binary_opengl_job
    assert "timeout-minutes: 240" in binary_opengl_job
    assert "fail-fast: false" in binary_opengl_job
    assert _matrix_values(binary_opengl_job, "shard_index") == {
        str(index) for index in range(24)
    }
    assert 'python-version: "3.12"' in binary_opengl_job
    assert "sudo apt-get install -y glslang-tools spirv-tools" in binary_opengl_job
    assert "python -m pip install -e . pytest-xdist" in binary_opengl_job
    assert "glslangValidator --version" in binary_opengl_job
    assert "spirv-val --version" in binary_opengl_job
    assert "Checkout current MLX binary corpus" in binary_opengl_job
    assert 'checkout --detach "$MLX_CORPUS_COMMIT"' in binary_opengl_job

    binary_opengl_step = ci_coverage.workflow_step_section(
        binary_opengl_job,
        "Prove current MLX complete binary family OpenGL translation",
    )
    assert "if: runner.os" not in binary_opengl_step
    assert (
        "CROSTL_MLX_ROOT: ${{ github.workspace }}/mlx-current-upstream"
        in binary_opengl_step
    )
    assert 'CROSTL_REQUIRE_MLX_BINARY_OPENGL_TRANSLATION: "1"' in binary_opengl_step
    assert (
        "CROSTL_MLX_BINARY_OPENGL_SHARD_INDEX: ${{ matrix.shard_index }}"
        in binary_opengl_step
    )
    assert 'CROSTL_MLX_BINARY_OPENGL_SHARD_COUNT: "24"' in binary_opengl_step
    assert (
        f"{test_path}::test_current_mlx_binary_family_translates_to_opengl"
        in binary_opengl_step
    )
    assert "-n auto" in binary_opengl_step
    assert "-k" not in binary_opengl_step
    _assert_workflow_triggers(mlx_porting, test_path)
    matrix_job = _workflow_job_section(mlx_porting, "mlx-metal-porting")
    assert "Prove current MLX complete binary family OpenGL translation" not in (
        matrix_job
    )


def test_mlx_project_porting_workflow_runs_binary_complete_directx_proof():
    mlx_porting = _workflow_texts().get("demo-project-testing.yml", "")
    ci_coverage = _load_ci_coverage_module()
    test_path = "demos/integrations/mlx/tests/kernels/test_binary_complete_directx.py"

    directx_job = _workflow_job_section(
        mlx_porting,
        "mlx-binary-complete-directx-translation",
    )
    assert (
        "name: MLX complete binary DirectX translation "
        "(shard ${{ matrix.shard_index }} of 24)" in directx_job
    )
    assert "if: github.event_name != 'schedule'" in directx_job
    assert "runs-on: ubuntu-24.04" in directx_job
    assert "timeout-minutes: 240" in directx_job
    assert "fail-fast: false" in directx_job
    assert _matrix_values(directx_job, "shard_index") == {
        str(index) for index in range(24)
    }
    assert 'python-version: "3.12"' in directx_job
    assert "python -m pip install -e . pytest-xdist" in directx_job
    assert "persist-credentials: false" in directx_job
    assert "continue-on-error" not in directx_job
    assert "uses: ./.github/actions/install-linux-dxc" in directx_job
    assert 'pinned: "true"' in directx_job
    assert "shell: pwsh" not in directx_job
    assert "Checkout current MLX binary corpus" in directx_job
    assert "config core.autocrlf false" in directx_job
    assert "sparse-checkout init --cone" in directx_job
    assert "sparse-checkout set mlx/backend/metal/kernels" in directx_job
    assert 'checkout --detach "$MLX_CORPUS_COMMIT"' in directx_job

    directx_step = ci_coverage.workflow_step_section(
        directx_job,
        "Prove current MLX complete binary family DirectX translation",
    )
    assert "if: runner.os" not in directx_step
    assert (
        "CROSTL_MLX_ROOT: ${{ github.workspace }}/mlx-current-upstream" in directx_step
    )
    assert 'CROSTL_REQUIRE_MLX_BINARY_DIRECTX_TRANSLATION: "1"' in directx_step
    assert (
        "CROSTL_MLX_BINARY_DIRECTX_SHARD_INDEX: ${{ matrix.shard_index }}"
        in directx_step
    )
    assert 'CROSTL_MLX_BINARY_DIRECTX_SHARD_COUNT: "24"' in directx_step
    assert (
        f"{test_path}::test_current_mlx_binary_family_translates_to_directx"
        in directx_step
    )
    assert "-n auto" in directx_step
    assert "-k" not in directx_step
    _assert_workflow_triggers(mlx_porting, test_path)
    matrix_job = _workflow_job_section(mlx_porting, "mlx-metal-porting")
    assert "Prove current MLX complete binary family DirectX translation" not in (
        matrix_job
    )


def test_mlx_project_porting_workflow_runs_current_fft_runtime_proofs():
    mlx_porting = _workflow_texts().get("demo-project-testing.yml", "")
    ci_coverage = _load_ci_coverage_module()
    step_name = "Prove current MLX FFT Direct3D native-loader execution"
    step = ci_coverage.workflow_step_section(mlx_porting, step_name)

    assert "if: runner.os == 'Windows'" in step
    assert (
        "CROSTL_MLX_CURRENT_ROOT: ${{ github.workspace }}/mlx-current-upstream" in step
    )
    assert 'CROSTL_REQUIRE_MLX_FFT_DIRECTX_NATIVE_LOADER: "1"' in step
    assert "python -m pytest -q -n auto" in step
    assert (
        "demos/integrations/mlx/tests/kernels/test_fft_native_loader.py::"
        "test_current_mlx_fft_executes_through_directx_native_loader" in step
    )
    assert "-k" not in step
    assert ci_coverage.workflow_step_after(
        mlx_porting,
        step_name,
        "Checkout current MLX runtime proof corpus",
    )

    opengl_step_name = "Prove current MLX FFT OpenGL native-loader execution"
    opengl_step = ci_coverage.workflow_step_section(mlx_porting, opengl_step_name)
    assert "if: runner.os == 'Linux'" in opengl_step
    assert (
        "CROSTL_MLX_CURRENT_ROOT: ${{ github.workspace }}/mlx-current-upstream"
        in opengl_step
    )
    assert 'CROSTL_REQUIRE_MLX_FFT_OPENGL_NATIVE_LOADER: "1"' in opengl_step
    assert "EGL_PLATFORM: surfaceless" in opengl_step
    assert 'LIBGL_ALWAYS_SOFTWARE: "1"' in opengl_step
    assert "MESA_LOADER_DRIVER_OVERRIDE: llvmpipe" in opengl_step
    assert "PYOPENGL_PLATFORM: egl" in opengl_step
    assert "python -m pytest -q -n auto" in opengl_step
    assert (
        "demos/integrations/mlx/tests/kernels/test_fft_native_loader.py::"
        "test_current_mlx_fft_executes_through_opengl_native_loader" in opengl_step
    )
    assert "-k" not in opengl_step
    assert ci_coverage.workflow_step_after(
        mlx_porting,
        opengl_step_name,
        "Install Linux runtime dependencies",
    )
    assert ci_coverage.workflow_step_after(
        mlx_porting,
        opengl_step_name,
        "Checkout current MLX runtime proof corpus",
    )


def test_mlx_project_porting_workflow_runs_current_gemv_runtime_proofs():
    mlx_porting = _workflow_texts().get("demo-project-testing.yml", "")
    ci_coverage = _load_ci_coverage_module()
    test_path = "demos/integrations/mlx/tests/kernels/test_gemv_native_loader.py"

    _assert_workflow_triggers(mlx_porting, test_path)
    directx_step_name = "Prove current MLX GEMV Direct3D native-loader execution"
    directx_step = ci_coverage.workflow_step_section(
        mlx_porting,
        directx_step_name,
    )
    assert "if: runner.os == 'Windows'" in directx_step
    assert (
        "CROSTL_MLX_ROOT: ${{ github.workspace }}/mlx-current-upstream" in directx_step
    )
    assert 'CROSTL_REQUIRE_MLX_GEMV_DIRECTX_NATIVE_LOADER: "1"' in directx_step
    assert "python -m pytest -q -n auto" in directx_step
    assert (
        f"{test_path}::"
        "test_current_mlx_gemv_executes_through_directx_native_loader" in directx_step
    )
    assert "-k" not in directx_step
    assert ci_coverage.workflow_step_after(
        mlx_porting,
        directx_step_name,
        "Checkout current MLX runtime proof corpus",
    )

    software_step_name = "Validate Direct3D software subgroup shuffle execution"
    software_step = ci_coverage.workflow_step_section(
        mlx_porting,
        software_step_name,
    )
    assert "if: runner.os == 'Windows'" in software_step
    assert 'CROSTL_RUN_DIRECTX_SOFTWARE_SUBGROUP_DEVICE_TEST: "1"' in software_step
    assert "python -m pytest -q -n auto" in software_step
    assert (
        "tests/test_translator/test_native_runtime_drivers.py::"
        "test_directx_compute_runtime_executes_software_subgroup_shuffle_on_device"
        in software_step
    )
    assert "-k" not in software_step
    assert ci_coverage.workflow_step_after(
        mlx_porting,
        software_step_name,
        "Checkout current MLX runtime proof corpus",
    )
    assert ci_coverage.workflow_step_after(
        mlx_porting,
        directx_step_name,
        software_step_name,
    )

    opengl_step_name = "Prove current MLX GEMV OpenGL native-loader execution"
    opengl_step = ci_coverage.workflow_step_section(
        mlx_porting,
        opengl_step_name,
    )
    assert "if: runner.os == 'Linux'" in opengl_step
    assert (
        "CROSTL_MLX_ROOT: ${{ github.workspace }}/mlx-current-upstream" in opengl_step
    )
    assert 'CROSTL_REQUIRE_MLX_GEMV_OPENGL_NATIVE_LOADER: "1"' in opengl_step
    assert "EGL_PLATFORM: surfaceless" in opengl_step
    assert 'LIBGL_ALWAYS_SOFTWARE: "1"' in opengl_step
    assert "MESA_LOADER_DRIVER_OVERRIDE: llvmpipe" in opengl_step
    assert "PYOPENGL_PLATFORM: egl" in opengl_step
    assert "python -m pytest -q -n auto" in opengl_step
    assert (
        f"{test_path}::"
        "test_current_mlx_gemv_executes_with_opengl_software_subgroups" in opengl_step
    )
    assert "-k" not in opengl_step
    assert ci_coverage.workflow_step_after(
        mlx_porting,
        opengl_step_name,
        "Install Linux runtime dependencies",
    )
    assert ci_coverage.workflow_step_after(
        mlx_porting,
        opengl_step_name,
        "Checkout current MLX runtime proof corpus",
    )


def test_mlx_project_porting_workflow_runs_current_mxfp4_runtime_proofs():
    mlx_porting = _workflow_texts().get("demo-project-testing.yml", "")
    ci_coverage = _load_ci_coverage_module()
    test_path = (
        "demos/integrations/mlx/tests/kernels/test_fp_quantized_native_loader.py"
    )

    _assert_workflow_triggers(mlx_porting, test_path)
    directx_step_name = "Prove current MLX MXFP4 Direct3D native-loader execution"
    directx_step = ci_coverage.workflow_step_section(
        mlx_porting,
        directx_step_name,
    )
    assert "if: runner.os == 'Windows'" in directx_step
    assert (
        "CROSTL_MLX_ROOT: ${{ github.workspace }}/mlx-current-upstream" in directx_step
    )
    assert 'CROSTL_REQUIRE_MLX_MXFP4_DIRECTX_NATIVE_LOADER: "1"' in directx_step
    assert "python -m pytest -q -n auto" in directx_step
    assert (
        f"{test_path}::"
        "test_current_mlx_mxfp4_executes_through_directx_native_loader" in directx_step
    )
    assert "-k" not in directx_step
    assert ci_coverage.workflow_step_after(
        mlx_porting,
        directx_step_name,
        "Checkout current MLX runtime proof corpus",
    )

    opengl_step_name = "Prove current MLX MXFP4 OpenGL native-loader execution"
    opengl_step = ci_coverage.workflow_step_section(
        mlx_porting,
        opengl_step_name,
    )
    assert "if: runner.os == 'Linux'" in opengl_step
    assert (
        "CROSTL_MLX_ROOT: ${{ github.workspace }}/mlx-current-upstream" in opengl_step
    )
    assert 'CROSTL_REQUIRE_MLX_MXFP4_OPENGL_NATIVE_LOADER: "1"' in opengl_step
    assert "EGL_PLATFORM: surfaceless" in opengl_step
    assert 'LIBGL_ALWAYS_SOFTWARE: "1"' in opengl_step
    assert "MESA_LOADER_DRIVER_OVERRIDE: llvmpipe" in opengl_step
    assert "PYOPENGL_PLATFORM: egl" in opengl_step
    assert "python -m pytest -q -n auto" in opengl_step
    assert (
        f"{test_path}::"
        "test_current_mlx_mxfp4_executes_with_opengl_software_subgroups" in opengl_step
    )
    assert "-k" not in opengl_step
    assert ci_coverage.workflow_step_after(
        mlx_porting,
        opengl_step_name,
        "Install Linux runtime dependencies",
    )
    assert ci_coverage.workflow_step_after(
        mlx_porting,
        opengl_step_name,
        "Checkout current MLX runtime proof corpus",
    )


def test_mlx_project_porting_workflow_installs_pinned_warp_runtime():
    mlx_porting = _workflow_texts().get("demo-project-testing.yml", "")
    ci_coverage = _load_ci_coverage_module()
    step_name = "Install pinned Windows WARP runtime"
    step = ci_coverage.workflow_step_section(mlx_porting, step_name)

    assert ci_coverage.workflow_step_after(
        mlx_porting,
        step_name,
        "Install Windows Direct3D runtime dependencies",
    )
    assert "if: runner.os == 'Windows'" in step
    assert 'warpVersion = "1.0.21"' in step
    assert "ea44b77eb30eec14427193e20f40cdb2e4c31ed11ab39cb880538fdf4bac2681" in step
    assert "api.nuget.org/v3-flatcontainer/microsoft.direct3d.warp" in step
    assert "Get-FileHash -Path $archive -Algorithm SHA256" in step
    assert '"build\\native\\bin\\x64\\d3d10warp.dll"' in step
    assert 'Join-Path $env:pythonLocation "d3d10warp.dll"' in step
    assert "Get-FileHash -Path $pythonWarp -Algorithm SHA256" in step


def test_mlx_project_porting_workflow_runs_copy_complete_opengl_proof():
    mlx_porting = _workflow_texts().get("demo-project-testing.yml", "")
    ci_coverage = _load_ci_coverage_module()
    test_path = "demos/integrations/mlx/tests/kernels/test_copy_complete_opengl.py"

    copy_opengl_job = _workflow_job_section(
        mlx_porting,
        "mlx-copy-complete-opengl-translation",
    )
    assert (
        "name: MLX complete copy OpenGL translation "
        "(shard ${{ matrix.shard_index }} of 24)" in copy_opengl_job
    )
    assert "if: github.event_name != 'schedule'" in copy_opengl_job
    assert "runs-on: ubuntu-latest" in copy_opengl_job
    assert "timeout-minutes: 180" in copy_opengl_job
    assert "fail-fast: false" in copy_opengl_job
    assert _matrix_values(copy_opengl_job, "shard_index") == {
        str(index) for index in range(24)
    }
    assert 'python-version: "3.12"' in copy_opengl_job
    assert "sudo apt-get install -y glslang-tools spirv-tools" in copy_opengl_job
    assert "python -m pip install -e . pytest-xdist" in copy_opengl_job
    assert "glslangValidator --version" in copy_opengl_job
    assert "spirv-val --version" in copy_opengl_job
    assert "Checkout current MLX copy corpus" in copy_opengl_job
    assert 'checkout --detach "$MLX_CORPUS_COMMIT"' in copy_opengl_job

    copy_opengl_step = ci_coverage.workflow_step_section(
        copy_opengl_job,
        "Prove current MLX complete copy family OpenGL translation",
    )
    assert "if: runner.os" not in copy_opengl_step
    assert (
        "CROSTL_MLX_ROOT: ${{ github.workspace }}/mlx-current-upstream"
        in copy_opengl_step
    )
    assert 'CROSTL_REQUIRE_MLX_COPY_OPENGL_TRANSLATION: "1"' in copy_opengl_step
    assert (
        "CROSTL_MLX_COPY_OPENGL_SHARD_INDEX: ${{ matrix.shard_index }}"
        in copy_opengl_step
    )
    assert 'CROSTL_MLX_COPY_OPENGL_SHARD_COUNT: "24"' in copy_opengl_step
    assert (
        f"{test_path}::test_current_mlx_copy_family_translates_to_opengl"
        in copy_opengl_step
    )
    assert "-n auto" in copy_opengl_step
    assert "-k" not in copy_opengl_step
    _assert_workflow_triggers(mlx_porting, test_path)
    matrix_job = _workflow_job_section(mlx_porting, "mlx-metal-porting")
    assert "Prove current MLX complete copy family OpenGL translation" not in (
        matrix_job
    )


def test_mlx_project_porting_workflow_runs_copy_complete_directx_proof():
    mlx_porting = _workflow_texts().get("demo-project-testing.yml", "")
    ci_coverage = _load_ci_coverage_module()
    test_path = "demos/integrations/mlx/tests/kernels/test_copy_complete_directx.py"

    copy_directx_job = _workflow_job_section(
        mlx_porting,
        "mlx-copy-complete-directx-translation",
    )
    assert (
        "name: MLX complete copy DirectX translation "
        "(shard ${{ matrix.shard_index }} of 24)" in copy_directx_job
    )
    assert "if: github.event_name != 'schedule'" in copy_directx_job
    assert "runs-on: ubuntu-24.04" in copy_directx_job
    assert "timeout-minutes: 180" in copy_directx_job
    assert "fail-fast: false" in copy_directx_job
    assert _matrix_values(copy_directx_job, "shard_index") == {
        str(index) for index in range(24)
    }
    assert 'python-version: "3.12"' in copy_directx_job
    assert "python -m pip install -e . pytest-xdist" in copy_directx_job
    assert "uses: ./.github/actions/install-linux-dxc" in copy_directx_job
    assert 'pinned: "true"' in copy_directx_job
    assert "shell: pwsh" not in copy_directx_job
    assert "Checkout current MLX copy corpus" in copy_directx_job
    assert 'checkout --detach "$MLX_CORPUS_COMMIT"' in copy_directx_job

    copy_directx_step = ci_coverage.workflow_step_section(
        copy_directx_job,
        "Prove current MLX complete copy family DirectX translation",
    )
    assert "if: runner.os" not in copy_directx_step
    assert (
        "CROSTL_MLX_ROOT: ${{ github.workspace }}/mlx-current-upstream"
        in copy_directx_step
    )
    assert 'CROSTL_REQUIRE_MLX_COPY_DIRECTX_TRANSLATION: "1"' in (copy_directx_step)
    assert (
        "CROSTL_MLX_COPY_DIRECTX_SHARD_INDEX: ${{ matrix.shard_index }}"
        in copy_directx_step
    )
    assert 'CROSTL_MLX_COPY_DIRECTX_SHARD_COUNT: "24"' in copy_directx_step
    assert (
        f"{test_path}::test_current_mlx_copy_family_translates_to_directx"
        in copy_directx_step
    )
    assert "-n auto" in copy_directx_step
    assert "-k" not in copy_directx_step
    _assert_workflow_triggers(mlx_porting, test_path)
    matrix_job = _workflow_job_section(mlx_porting, "mlx-metal-porting")
    assert "Prove current MLX complete copy family DirectX translation" not in (
        matrix_job
    )


def test_mlx_project_porting_workflow_runs_copy_complete_metal_proof():
    mlx_porting = _workflow_texts().get("demo-project-testing.yml", "")
    ci_coverage = _load_ci_coverage_module()
    test_path = (
        "demos/integrations/mlx/tests/kernels/test_copy_complete_metal_roundtrip.py"
    )

    copy_metal_job = _workflow_job_section(
        mlx_porting,
        "mlx-copy-complete-metal-roundtrip",
    )
    assert (
        "name: MLX complete copy Metal round-trip "
        "(shard ${{ matrix.shard_index }} of 24)" in copy_metal_job
    )
    assert "if: github.event_name != 'schedule'" in copy_metal_job
    assert "runs-on: macOS-latest" in copy_metal_job
    assert "timeout-minutes: 180" in copy_metal_job
    assert "fail-fast: false" in copy_metal_job
    assert _matrix_values(copy_metal_job, "shard_index") == {
        str(index) for index in range(24)
    }
    assert 'python-version: "3.12"' in copy_metal_job
    assert "python -m pip install -e . pytest-xdist" in copy_metal_job
    assert "xcrun --sdk macosx metal --version" in copy_metal_job
    assert "Checkout current MLX copy corpus" in copy_metal_job
    assert 'checkout --detach "$MLX_CORPUS_COMMIT"' in copy_metal_job

    copy_metal_step = ci_coverage.workflow_step_section(
        copy_metal_job,
        "Prove current MLX complete copy family Metal round-trips",
    )
    assert "if: runner.os" not in copy_metal_step
    assert (
        "CROSTL_MLX_ROOT: ${{ github.workspace }}/mlx-current-upstream"
        in copy_metal_step
    )
    assert 'CROSTL_REQUIRE_MLX_COPY_METAL_ROUNDTRIP: "1"' in copy_metal_step
    assert (
        "CROSTL_MLX_COPY_METAL_SHARD_INDEX: ${{ matrix.shard_index }}"
        in copy_metal_step
    )
    assert 'CROSTL_MLX_COPY_METAL_SHARD_COUNT: "24"' in copy_metal_step
    assert (
        f"{test_path}::test_current_mlx_copy_family_roundtrips_through_metal"
        in copy_metal_step
    )
    assert "-n auto" in copy_metal_step
    assert "-k" not in copy_metal_step
    _assert_workflow_triggers(mlx_porting, test_path)
    matrix_job = _workflow_job_section(mlx_porting, "mlx-metal-porting")
    assert "Prove current MLX complete copy family Metal round-trips" not in (
        matrix_job
    )


def test_mlx_project_porting_workflow_runs_reduce_complete_metal_proof():
    mlx_porting = _workflow_texts().get("demo-project-testing.yml", "")
    ci_coverage = _load_ci_coverage_module()
    test_path = (
        "demos/integrations/mlx/tests/kernels/test_reduce_complete_metal_roundtrip.py"
    )

    reduce_metal_job = _workflow_job_section(
        mlx_porting,
        "mlx-reduce-complete-metal-roundtrip",
    )
    assert (
        "name: MLX complete reduce Metal round-trip "
        "(shard ${{ matrix.shard_index }} of 24)" in reduce_metal_job
    )
    assert "if: github.event_name != 'schedule'" in reduce_metal_job
    assert "runs-on: macOS-latest" in reduce_metal_job
    assert "timeout-minutes: 180" in reduce_metal_job
    assert "fail-fast: false" in reduce_metal_job
    assert _matrix_values(reduce_metal_job, "shard_index") == {
        str(index) for index in range(24)
    }
    assert 'python-version: "3.12"' in reduce_metal_job
    assert "python -m pip install -e . pytest-xdist" in reduce_metal_job
    assert "xcrun --sdk macosx metal --version" in reduce_metal_job
    assert "Checkout current MLX reduce corpus" in reduce_metal_job
    assert 'checkout --detach "$MLX_CORPUS_COMMIT"' in reduce_metal_job

    reduce_metal_step = ci_coverage.workflow_step_section(
        reduce_metal_job,
        "Prove current MLX complete reduce family Metal round-trips",
    )
    assert "if: runner.os" not in reduce_metal_step
    assert (
        "CROSTL_MLX_ROOT: ${{ github.workspace }}/mlx-current-upstream"
        in reduce_metal_step
    )
    assert 'CROSTL_REQUIRE_MLX_REDUCE_METAL_ROUNDTRIP: "1"' in reduce_metal_step
    assert (
        "CROSTL_MLX_REDUCE_METAL_SHARD_INDEX: ${{ matrix.shard_index }}"
        in reduce_metal_step
    )
    assert 'CROSTL_MLX_REDUCE_METAL_SHARD_COUNT: "24"' in reduce_metal_step
    assert (
        f"{test_path}::test_current_mlx_reduce_family_roundtrips_through_metal"
        in reduce_metal_step
    )
    assert "-n auto" in reduce_metal_step
    assert "-k" not in reduce_metal_step
    _assert_workflow_triggers(mlx_porting, test_path)
    matrix_job = _workflow_job_section(mlx_porting, "mlx-metal-porting")
    assert "Prove current MLX complete reduce family Metal round-trips" not in (
        matrix_job
    )


def test_mlx_project_porting_workflow_runs_quantized_complete_metal_proof():
    mlx_porting = _workflow_texts().get("demo-project-testing.yml", "")
    ci_coverage = _load_ci_coverage_module()
    test_path = "demos/integrations/mlx/tests/kernels/test_quantized_complete_metal_roundtrip.py"

    quantized_metal_job = _workflow_job_section(
        mlx_porting,
        "mlx-quantized-complete-metal-roundtrip",
    )
    assert (
        "name: MLX complete quantized Metal round-trip "
        "(shard ${{ matrix.shard_index }} of 24)" in quantized_metal_job
    )
    assert "if: github.event_name != 'schedule'" in quantized_metal_job
    assert "runs-on: macOS-latest" in quantized_metal_job
    assert "timeout-minutes: 180" in quantized_metal_job
    assert "fail-fast: false" in quantized_metal_job
    assert _matrix_values(quantized_metal_job, "shard_index") == {
        str(index) for index in range(24)
    }
    assert 'python-version: "3.12"' in quantized_metal_job
    assert "python -m pip install -e . pytest-xdist" in quantized_metal_job
    assert "persist-credentials: false" in quantized_metal_job
    assert "continue-on-error" not in quantized_metal_job
    assert "xcrun --sdk macosx metal --version" in quantized_metal_job
    assert "xcodebuild -downloadComponent MetalToolchain" in quantized_metal_job
    assert "Checkout current MLX quantized corpus" in quantized_metal_job
    assert "config core.autocrlf false" in quantized_metal_job
    assert "sparse-checkout init --cone" in quantized_metal_job
    assert "sparse-checkout set mlx/backend/metal/kernels" in quantized_metal_job
    assert 'checkout --detach "$MLX_CORPUS_COMMIT"' in quantized_metal_job

    quantized_metal_step = ci_coverage.workflow_step_section(
        quantized_metal_job,
        "Prove current MLX complete quantized family Metal round-trips",
    )
    assert "if: runner.os" not in quantized_metal_step
    assert (
        "CROSTL_MLX_ROOT: ${{ github.workspace }}/mlx-current-upstream"
        in quantized_metal_step
    )
    assert 'CROSTL_REQUIRE_MLX_QUANTIZED_METAL_ROUNDTRIP: "1"' in quantized_metal_step
    assert (
        "CROSTL_MLX_QUANTIZED_METAL_SHARD_INDEX: ${{ matrix.shard_index }}"
        in quantized_metal_step
    )
    assert 'CROSTL_MLX_QUANTIZED_METAL_SHARD_COUNT: "24"' in quantized_metal_step
    assert (
        f"{test_path}::test_current_mlx_quantized_metal_discovery_matches_contract"
        in quantized_metal_step
    )
    assert (
        f"{test_path}::test_current_mlx_quantized_family_roundtrips_through_metal"
        in quantized_metal_step
    )
    assert "-n auto" in quantized_metal_step
    assert "-k" not in quantized_metal_step
    _assert_workflow_triggers(mlx_porting, test_path)
    matrix_job = _workflow_job_section(mlx_porting, "mlx-metal-porting")
    assert "Prove current MLX complete quantized family Metal round-trips" not in (
        matrix_job
    )


def test_mlx_project_porting_workflow_requires_quantized_wide_opengl_compilation():
    workflow = _workflow_texts()["demo-project-testing.yml"]
    job = _workflow_job_section(workflow, "mlx-quantized-wide-opengl")
    test_path = "demos/integrations/mlx/tests/kernels/test_quantized_wide_opengl.py"

    assert "runs-on: ubuntu-latest" in job
    assert "timeout-minutes: 60" in job
    assert "persist-credentials: false" in job
    assert "continue-on-error" not in job
    assert "sudo apt-get install -y glslang-tools spirv-tools" in job
    assert 'checkout --detach "$MLX_CORPUS_COMMIT"' in job
    assert 'CROSTL_REQUIRE_MLX_QUANTIZED_WIDE_OPENGL: "1"' in job
    assert "CROSTL_MLX_ROOT: ${{ github.workspace }}/mlx-current-upstream" in job
    assert "python -m pytest -q -n auto" in job
    assert test_path in job
    assert "--junitxml=quantized-wide-junit.xml" in job
    assert "--basetemp=quantized-wide-results" in job
    assert "if: always()" in job
    assert "if-no-files-found: error" in job
    _assert_workflow_triggers(workflow, test_path)


def test_mlx_project_porting_workflow_runs_reduce_complete_opengl_proof():
    mlx_porting = _workflow_texts().get("demo-project-testing.yml", "")
    ci_coverage = _load_ci_coverage_module()
    test_path = "demos/integrations/mlx/tests/kernels/test_reduce_complete_opengl.py"

    reduce_opengl_job = _workflow_job_section(
        mlx_porting,
        "mlx-reduce-complete-opengl-translation",
    )
    assert (
        "name: MLX complete reduce OpenGL translation "
        "(shard ${{ matrix.shard_index }} of 24)" in reduce_opengl_job
    )
    assert "if: github.event_name != 'schedule'" in reduce_opengl_job
    assert "runs-on: ubuntu-latest" in reduce_opengl_job
    assert "timeout-minutes: 180" in reduce_opengl_job
    assert "fail-fast: false" in reduce_opengl_job
    assert _matrix_values(reduce_opengl_job, "shard_index") == {
        str(index) for index in range(24)
    }
    assert 'python-version: "3.12"' in reduce_opengl_job
    assert "sudo apt-get install -y glslang-tools spirv-tools" in reduce_opengl_job
    assert "python -m pip install -e . pytest-xdist" in reduce_opengl_job
    assert "glslangValidator --version" in reduce_opengl_job
    assert "spirv-val --version" in reduce_opengl_job
    assert "Checkout current MLX reduce corpus" in reduce_opengl_job
    assert 'checkout --detach "$MLX_CORPUS_COMMIT"' in reduce_opengl_job

    reduce_opengl_step = ci_coverage.workflow_step_section(
        reduce_opengl_job,
        "Prove current MLX complete reduce family OpenGL translation",
    )
    assert "if: runner.os" not in reduce_opengl_step
    assert (
        "CROSTL_MLX_ROOT: ${{ github.workspace }}/mlx-current-upstream"
        in reduce_opengl_step
    )
    assert 'CROSTL_REQUIRE_MLX_REDUCE_OPENGL_TRANSLATION: "1"' in reduce_opengl_step
    assert (
        "CROSTL_MLX_REDUCE_OPENGL_SHARD_INDEX: ${{ matrix.shard_index }}"
        in reduce_opengl_step
    )
    assert 'CROSTL_MLX_REDUCE_OPENGL_SHARD_COUNT: "24"' in reduce_opengl_step
    assert (
        f"{test_path}::test_current_mlx_reduce_family_translates_to_opengl"
        in reduce_opengl_step
    )
    assert "-n auto" in reduce_opengl_step
    assert "-k" not in reduce_opengl_step
    _assert_workflow_triggers(mlx_porting, test_path)
    matrix_job = _workflow_job_section(mlx_porting, "mlx-metal-porting")
    assert "Prove current MLX complete reduce family OpenGL translation" not in (
        matrix_job
    )


def test_mlx_project_porting_workflow_runs_reduce_complete_directx_proof():
    mlx_porting = _workflow_texts().get("demo-project-testing.yml", "")
    ci_coverage = _load_ci_coverage_module()
    test_path = "demos/integrations/mlx/tests/kernels/test_reduce_complete_directx.py"

    reduce_directx_job = _workflow_job_section(
        mlx_porting,
        "mlx-reduce-complete-directx-translation",
    )
    assert (
        "name: MLX complete reduce DirectX translation "
        "(shard ${{ matrix.shard_index }} of 24)" in reduce_directx_job
    )
    assert "if: github.event_name != 'schedule'" in reduce_directx_job
    assert "runs-on: ubuntu-24.04" in reduce_directx_job
    assert "timeout-minutes: 240" in reduce_directx_job
    assert "fail-fast: false" in reduce_directx_job
    assert _matrix_values(reduce_directx_job, "shard_index") == {
        str(index) for index in range(24)
    }
    assert 'python-version: "3.12"' in reduce_directx_job
    assert "python -m pip install -e . pytest-xdist" in reduce_directx_job
    assert "uses: ./.github/actions/install-linux-dxc" in reduce_directx_job
    assert 'pinned: "true"' in reduce_directx_job
    assert "shell: pwsh" not in reduce_directx_job
    assert "Checkout current MLX reduce corpus" in reduce_directx_job
    assert 'checkout --detach "$MLX_CORPUS_COMMIT"' in reduce_directx_job

    reduce_directx_step = ci_coverage.workflow_step_section(
        reduce_directx_job,
        "Prove current MLX complete reduce family DirectX translation",
    )
    assert "if: runner.os" not in reduce_directx_step
    assert (
        "CROSTL_MLX_ROOT: ${{ github.workspace }}/mlx-current-upstream"
        in reduce_directx_step
    )
    assert 'CROSTL_REQUIRE_MLX_REDUCE_DIRECTX_TRANSLATION: "1"' in (reduce_directx_step)
    assert (
        "CROSTL_MLX_REDUCE_DIRECTX_SHARD_INDEX: ${{ matrix.shard_index }}"
        in reduce_directx_step
    )
    assert 'CROSTL_MLX_REDUCE_DIRECTX_SHARD_COUNT: "24"' in reduce_directx_step
    assert (
        f"{test_path}::test_current_mlx_reduce_family_translates_to_directx"
        in reduce_directx_step
    )
    assert "-n auto" in reduce_directx_step
    assert "-k" not in reduce_directx_step
    _assert_workflow_triggers(mlx_porting, test_path)
    matrix_job = _workflow_job_section(mlx_porting, "mlx-metal-porting")
    assert "Prove current MLX complete reduce family DirectX translation" not in (
        matrix_job
    )


def test_atomic_stores_and_pinned_scatter_are_required_on_each_native_target():
    from demos.integrations.mlx.tests.kernels.test_scatter_axis_runtime import (
        REQUIRE_ENV as SCATTER_ENV,
    )
    from tools import ci_coverage

    workflow = (
        Path(__file__).resolve().parents[4]
        / ".github/workflows/demo-project-testing.yml"
    ).read_text()
    for name in (
        "Validate general gather and empty arrays",
        "Validate indexed DirectX gather and resource aggregates",
        "Validate indexed OpenGL gather and resource aggregates",
    ):
        step = ci_coverage.workflow_step_section(workflow, name)
        assert f'{STORE_ENV}: "1"' in step
        assert f'{SCATTER_ENV}: "1"' in step
        assert "test_atomic_store_runtime.py" in step
        assert "test_scatter_axis_runtime.py" in step
        timeout = 1800 if name == "Validate general gather and empty arrays" else 1200
        assert f"--timeout-seconds {timeout}" in step
        assert "if:" not in step and "continue-on-error" not in workflow

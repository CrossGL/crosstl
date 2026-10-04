"""Execute unchanged MLX integer compare-exchange and multiplication helpers."""

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

from crosstl.project import (
    ProjectConfig,
    build_native_loader_dispatch_request,
    translate_project,
)
from demos.integrations.mlx.portable_host.prepare import COMMIT
from demos.integrations.mlx.tests.kernels.test_atomic_load_runtime import HEADER
from demos.integrations.mlx.tests.kernels.test_current_binary_shapes import (
    _prepare_native_package,
)
from demos.integrations.mlx.tests.kernels.test_current_gather import _validate
from demos.integrations.mlx.tests.kernels.test_general_scatter_runtime import (
    _verify_source,
)
from tests.test_translator.test_atomic_compare_exchange_runtime import REQUIRE_ENV
from tests.test_translator.test_atomic_load_runtime import GUARD
from tests.test_translator.test_boolean_buffer_runtime import _bound_values
from tests.test_translator.test_loop_updates import _execute

CASES = ("compare", "multiply", "contention")


def test_compare_exchange_and_dependencies_are_required_on_each_native_target():
    from tests.test_translator.test_metal_constexpr_branches import (
        REQUIRE_ENV as constexpr_env,
    )
    from tests.test_translator.test_metal_constrained_calls import (
        REQUIRE_ENV as constrained_env,
    )
    from tools import ci_coverage

    workflow = (
        Path(__file__).resolve().parents[5]
        / ".github/workflows/demo-project-testing.yml"
    ).read_text()
    for name in (
        "Validate general gather and empty arrays",
        "Validate indexed DirectX gather and resource aggregates",
        "Validate indexed OpenGL gather and resource aggregates",
    ):
        step = ci_coverage.workflow_step_section(workflow, name)
        for variable in (REQUIRE_ENV, constexpr_env, constrained_env):
            assert f'{variable}: "1"' in step
        for module in (
            "test_atomic_compare_exchange_runtime.py",
            "test_atomic_compare_exchange_runtime.py",
            "test_metal_constexpr_branches.py",
            "test_metal_constrained_calls.py",
        ):
            assert module in step
        timeout = 1800 if name == "Validate general gather and empty arrays" else 1200
        assert f"--timeout-seconds {timeout}" in step and "-n auto" in step
        assert "--junitxml=" in step and "--basetemp=" in step
        assert "if:" not in step and "continue-on-error" not in workflow


def _workload(kind, case):
    values = [-7, 13, -19] if kind == "int" else [7, 13, 19]
    if case == "contention":
        values = [3, 13, 19]
    dtype = f"{kind}32"
    result = [GUARD] * (11 if case == "contention" else 9)
    if case == "compare":
        for index, value in enumerate(values):
            result[index * 2 : index * 2 + 2] = [0, value]
        final = [17, 17, 17]
    elif case == "multiply":
        final = [value * 3 for value in values]
        result[:3] = final
    else:
        final = [3 * 2**8, *values[1:]]
        result[:8] = [final[0]] * 8
    inputs = {
        "values": {"dtype": dtype, "shape": [5, 1], "values": [GUARD, *values, GUARD]},
        "results": {
            "dtype": dtype,
            "shape": [len(result)],
            "values": [GUARD] * len(result),
        },
    }
    outputs = {
        "values": {**inputs["values"], "values": [GUARD, *final, GUARD]},
        "results": {**inputs["results"], "values": result},
    }
    return inputs, outputs


def _request(root, target, work, kind, case):
    body = f"""{kind} expected = {kind}(0);
    bool first = mlx_atomic_compare_exchange_weak_explicit(values, &expected, {kind}(11), size_t(tid + 1u));
    results[tid * 2u] = {kind}(first);
    results[tid * 2u + 1u] = expected;
    while (!mlx_atomic_compare_exchange_weak_explicit(values, &expected, {kind}(17), size_t(tid + 1u))) {{}}"""
    if case == "multiply":
        body = f"""mlx_atomic_fetch_mul_explicit(values, {kind}(3), size_t(tid + 1u));
        results[tid] = mlx_atomic_load_explicit(values, size_t(tid + 1u));"""
    elif case == "contention":
        body = f"""mlx_atomic_fetch_mul_explicit(values, {kind}(2), size_t(1));
        threadgroup_barrier(mem_flags::mem_device);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        results[tid] = mlx_atomic_load_explicit(values, size_t(1));"""
    source = f"""#include "{HEADER}"
kernel void exchange_values(device mlx_atomic<{kind}>* values [[buffer(0)]],
    device {kind}* results [[buffer(1)]], uint tid [[thread_position_in_grid]]) {{
    {body}
}}
"""
    width = 8 if case == "contention" else 1
    (work / "source.metal").write_text(source, encoding="utf-8")
    with tempfile.TemporaryDirectory(
        prefix=".atomic-compare-proof-", dir=root
    ) as temporary:
        staging = Path(temporary)
        wrapper = staging / "compare.metal"
        wrapper.write_text(source, encoding="utf-8")
        relative = wrapper.relative_to(root).as_posix()
        assertions = (
            (
                {
                    "source": relative,
                    "expression": "offset",
                    "minimum": 1,
                    "maximum": 1 if case == "contention" else 3,
                },
            )
            if target == "opengl"
            else ()
        )
        report = translate_project(
            ProjectConfig(
                root=root,
                source_roots=(staging.name,),
                include_patterns=(relative,),
                include_dirs=(".",),
                targets=(target,),
                output_dir=f"{staging.name}/out",
                entry_points={relative: ("exchange_values",)},
                workgroup_size=(width, 1, 1),
                index_range_assertions=assertions,
            ),
            format_output=False,
        )
        descriptor, package = _prepare_native_package(report, work)
    inputs, outputs = _workload(kind, case)
    expected = _bound_values(descriptor, outputs)
    request = build_native_loader_dispatch_request(
        descriptor,
        package,
        _bound_values(descriptor, inputs),
        expected,
        {
            "workgroupCount": [1 if case == "contention" else 3, 1, 1],
            "workgroupSize": [width, 1, 1],
        },
        expected_target=target,
    )
    assert not request.execution_plan.diagnostics
    return source, request, expected


@pytest.mark.parametrize("kind", ("int", "uint"))
@pytest.mark.parametrize("case", CASES)
def test_pinned_atomic_compare_exchange_executes_natively(tmp_path, kind, case):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required pinned atomic execution")
    root = Path(os.environ["CROSTL_MLX_CURRENT_ROOT"]).resolve()
    target = {"darwin": "metal", "win32": "directx", "linux": "opengl"}[sys.platform]
    hashes = _verify_source(root, (HEADER,))
    try:
        source, request, outputs = _request(root, target, tmp_path, kind, case)
        (tmp_path / "workload.json").write_text(
            json.dumps(
                {
                    "commit": COMMIT,
                    "headers": hashes,
                    "kind": kind,
                    "case": case,
                    "inputs": _workload(kind, case)[0],
                    "outputs": outputs,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        _execute(
            request,
            outputs,
            tmp_path,
            original_source=source,
            original_entry="exchange_values",
            metal_compile_flags=("-I", str(root)),
            validate=_validate,
        )
    finally:
        assert _verify_source(root, (HEADER,)) == hashes


@pytest.mark.parametrize("kind", ("int", "uint"))
@pytest.mark.parametrize("case", CASES)
def test_atomic_compare_workload_preserves_guards(kind, case):
    inputs, outputs = _workload(kind, case)
    assert inputs["values"]["values"][0] == outputs["values"]["values"][0] == GUARD
    assert inputs["values"]["values"][-1] == outputs["values"]["values"][-1] == GUARD
    assert outputs["results"]["values"][-3:] == [GUARD] * 3

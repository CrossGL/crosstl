"""Execute unchanged MLX floating compare-exchange and reduction helpers."""

import json
import os
import struct
import sys
import tempfile
from pathlib import Path

import pytest

from crosstl.project import (
    ProjectConfig,
    build_native_loader_dispatch_request,
    translate_project,
)
from crosstl.project.runtime_value_encoding import FLOAT32_BITS
from demos.integrations.mlx.portable_host.prepare import COMMIT
from demos.integrations.mlx.tests.kernels.test_atomic_load_runtime import HEADER
from demos.integrations.mlx.tests.kernels.test_current_gather import _validate
from demos.integrations.mlx.tests.kernels.test_general_scatter_runtime import (
    _verify_source,
)
from tests.runtime_helpers import _prepare_native_package
from tests.test_translator.test_atomic_load_runtime import GUARD
from tests.test_translator.test_boolean_buffer_runtime import _bound_values
from tests.test_translator.test_float_atomic_compare_exchange import REQUIRE_ENV
from tests.test_translator.test_float_storage_encoding import WORDS
from tests.test_translator.test_loop_updates import _execute

CASES = ("compare", "multiply", "minimum", "maximum", "contention")


def test_float_compare_exchange_requires_native_windows_and_metal_execution():
    from tools import ci_coverage

    workflow = (
        Path(__file__).resolve().parents[5]
        / ".github/workflows/demo-project-testing.yml"
    ).read_text()
    for name in (
        "Validate float compare-exchange and scatter",
        "Validate indexed DirectX gather and resource aggregates",
    ):
        step = ci_coverage.workflow_step_section(workflow, name)
        assert f'{REQUIRE_ENV}: "1"' in step
        assert "test_float_atomic_compare_exchange.py" in step
        assert "test_float_atomic_compare_exchange.py" in step
        assert "--timeout-seconds 1200" in step and "-n auto" in step
        assert "--junitxml=" in step and "--basetemp=" in step
        assert "if:" not in step
    opengl = ci_coverage.workflow_step_section(
        workflow, "Validate indexed OpenGL gather and resource aggregates"
    )
    assert "test_float_atomic_compare_exchange.py" in opengl
    assert REQUIRE_ENV not in opengl
    assert "continue-on-error" not in workflow


def _word(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def _workload(case):
    initial = list(WORDS)
    if case == "multiply":
        initial = [
            _word(value)
            for value in (
                0.0,
                -0.0,
                1.5,
                -2.5,
                16.0,
                -64.0,
                float("inf"),
                -float("inf"),
            )
        ]
    elif case == "contention":
        initial = [_word(1.5)]
    final = initial.copy()
    count = 8 if case == "contention" else len(initial)
    result = [GUARD] * (count * 4 + 3)
    for index, bits in enumerate(initial):
        value = struct.unpack("<f", struct.pack("<I", bits))[0]
        if case == "compare":
            final[index] = WORDS[len(WORDS) - index - 1]
            result[index * 4 : index * 4 + 4] = [0, bits, 1, final[index]]
        else:
            if case == "multiply":
                final[index] = _word(value * 3)
            elif case == "minimum" and -1.5 < value:
                final[index] = _word(-1.5)
            elif case == "maximum" and 1.5 > value:
                final[index] = _word(1.5)
            elif case == "contention":
                final[index] = _word(1.5 * 2**8)
            result[index * 4] = final[index]
    if case == "contention":
        for index in range(count):
            result[index * 4] = final[0]

    def typed(values, floating=False):
        return {
            "dtype": "float32" if floating else "uint32",
            "values": values,
            "shape": [len(values), 1] if floating else [len(values)],
            **({"encoding": FLOAT32_BITS} if floating else {}),
        }

    incoming = [
        item
        for bits, desired in zip(initial, final)
        for item in (bits ^ 0x80000000, desired)
    ]
    inputs = {
        "values": typed([GUARD, *initial, GUARD], True),
        "incoming": typed(incoming),
        "results": typed([GUARD] * len(result)),
    }
    outputs = {
        "values": typed([GUARD, *final, GUARD], True),
        "incoming": inputs["incoming"],
        "results": typed(result),
    }
    return count, inputs, outputs


def _request(root, target, work, case):
    count, inputs, outputs = _workload(case)
    body = """float expected = as_type<float>(incoming[tid * 2u]);
    float desired = as_type<float>(incoming[tid * 2u + 1u]);
    bool first = mlx_atomic_compare_exchange_weak_explicit(values, &expected, desired, offset);
    results[tid * 4u] = uint(first);
    results[tid * 4u + 1u] = as_type<uint>(expected);
    bool success = false;
    for (uint attempt = 0u; attempt < 8u && !success; ++attempt) {
        success = mlx_atomic_compare_exchange_weak_explicit(values, &expected, desired, offset);
    }
    results[tid * 4u + 2u] = uint(success);
    results[tid * 4u + 3u] = as_type<uint>(mlx_atomic_load_explicit(values, offset));"""
    if case != "compare":
        operation = {
            "multiply": "mul",
            "minimum": "min",
            "maximum": "max",
            "contention": "mul",
        }[case]
        value = {
            "multiply": "3.0f",
            "minimum": "-1.5f",
            "maximum": "1.5f",
            "contention": "2.0f",
        }[case]
        body = f"mlx_atomic_fetch_{operation}_explicit(values, {value}, offset);"
        if case == "contention":
            body += "threadgroup_barrier(mem_flags::mem_device); threadgroup_barrier(mem_flags::mem_threadgroup);"
        body += "results[tid * 4u] = as_type<uint>(mlx_atomic_load_explicit(values, offset));"
    source = f"""#include "{HEADER}"
kernel void atomic_compare(device mlx_atomic<float>* values [[buffer(0)]],
                           device uint* incoming [[buffer(1)]],
                           device uint* results [[buffer(2)]],
                           uint tid [[thread_position_in_grid]]) {{
    size_t offset = size_t({'1u' if case == 'contention' else 'tid + 1u'});
    {body}
}}
"""
    (work / "source.metal").write_text(source, encoding="utf-8")
    width = 8 if case == "contention" else 1
    with tempfile.TemporaryDirectory(
        prefix=".float-atomic-compare-", dir=root
    ) as temporary:
        staging = Path(temporary)
        wrapper = staging / "atomic.metal"
        wrapper.write_text(source, encoding="utf-8")
        relative = wrapper.relative_to(root).as_posix()
        report = translate_project(
            ProjectConfig(
                root=root,
                source_roots=(staging.name,),
                include_patterns=(relative,),
                include_dirs=(".",),
                targets=(target,),
                output_dir=f"{staging.name}/out",
                entry_points={relative: ("atomic_compare",)},
                workgroup_size=(width, 1, 1),
                index_range_assertions=(
                    (
                        {
                            "source": relative,
                            "expression": "offset",
                            "minimum": 1,
                            "maximum": 1 if case == "contention" else count,
                        },
                    )
                    if target == "opengl"
                    else ()
                ),
            ),
            format_output=False,
        )
        descriptor, package = _prepare_native_package(report, work)
    if target == "opengl":
        for values in (inputs, outputs):
            values["values"]["dtype"] = "uint32"
            del values["values"]["encoding"]
    expected = _bound_values(descriptor, outputs)
    request = build_native_loader_dispatch_request(
        descriptor,
        package,
        _bound_values(descriptor, inputs),
        expected,
        {
            "workgroupCount": [1 if case == "contention" else count, 1, 1],
            "workgroupSize": [width, 1, 1],
        },
        expected_target=target,
    )
    assert not request.execution_plan.diagnostics
    return source, request, expected


@pytest.mark.parametrize("case", CASES)
def test_pinned_float_atomic_compare_executes_natively(tmp_path, case):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for native pinned floating compare-exchange")
    assert sys.platform in {"darwin", "win32"}
    root = Path(os.environ["CROSTL_MLX_CURRENT_ROOT"]).resolve()
    target = "metal" if sys.platform == "darwin" else "directx"
    hashes = _verify_source(root, (HEADER,))
    try:
        source, request, expected = _request(root, target, tmp_path, case)
        (tmp_path / "workload.json").write_text(
            json.dumps(
                {
                    "commit": COMMIT,
                    "headers": hashes,
                    "case": case,
                    "inputs": _workload(case)[1],
                    "outputs": expected,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        _execute(
            request,
            expected,
            tmp_path,
            original_source=source,
            original_entry="atomic_compare",
            metal_compile_flags=("-I", str(root)),
            validate=_validate,
        )
    finally:
        assert _verify_source(root, (HEADER,)) == hashes


@pytest.mark.parametrize("case", CASES)
def test_pinned_float_atomic_compare_workload_preserves_guards(case):
    _, inputs, outputs = _workload(case)
    assert inputs["values"]["values"][0] == outputs["values"]["values"][0] == GUARD
    assert inputs["values"]["values"][-1] == outputs["values"]["values"][-1] == GUARD
    assert outputs["results"]["values"][-3:] == [GUARD] * 3

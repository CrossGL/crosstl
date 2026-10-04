"""Preserve binary32 payloads through unchanged MLX atomic memory helpers."""

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
from crosstl.project.runtime_value_encoding import FLOAT32_BITS
from demos.integrations.mlx.portable_host.prepare import COMMIT
from demos.integrations.mlx.tests.kernels.test_atomic_load_runtime import HEADER
from demos.integrations.mlx.tests.kernels.test_current_binary_shapes import (
    _prepare_native_package,
)
from demos.integrations.mlx.tests.kernels.test_current_gather import _validate
from demos.integrations.mlx.tests.kernels.test_general_scatter_runtime import (
    _verify_source,
)
from tests.test_translator.test_atomic_load_runtime import GUARD
from tests.test_translator.test_boolean_buffer_runtime import _bound_values
from tests.test_translator.test_float_atomic_memory import REQUIRE_ENV
from tests.test_translator.test_float_storage_encoding import WORDS
from tests.test_translator.test_loop_updates import _execute


def _workload(operation):
    count = len(WORDS)
    incoming = list(reversed(WORDS))
    final = WORDS if operation == "load" else incoming

    def typed(values, floating=False):
        return {
            "dtype": "float32" if floating else "uint32",
            "shape": [len(values), 1] if floating else [len(values)],
            "values": values,
            **({"encoding": FLOAT32_BITS} if floating else {}),
        }

    inputs = {
        "values": typed([GUARD, *WORDS, GUARD], True),
        "incoming": typed(incoming),
        "results": typed([GUARD] * (count + 3)),
        "counts": typed([GUARD] * (2 * count + 3)),
    }
    counts = [
        value for i in range(count) for value in (i + 2, i + (operation == "store"))
    ]
    outputs = {
        "values": typed([GUARD, *final, GUARD], True),
        "incoming": inputs["incoming"],
        "results": typed([*final, GUARD, GUARD, GUARD]),
        "counts": typed([*counts, GUARD, GUARD, GUARD]),
    }
    return inputs, outputs


def _request(root, target, work, operation):
    body = "results[tid] = as_type<uint>(mlx_atomic_load_explicit(values, offset++));"
    if operation == "store":
        body = """mlx_atomic_store_explicit(values, as_type<float>(incoming[position++]), offset++);
            results[tid] = as_type<uint>(mlx_atomic_load_explicit(values, size_t(tid + 1u)));"""
    source = f"""#include "{HEADER}"
kernel void atomic_memory(device mlx_atomic<float>* values [[buffer(0)]],
                          device uint* incoming [[buffer(1)]],
                          device uint* results [[buffer(2)]],
                          device uint* counts [[buffer(3)]],
                          uint tid [[thread_position_in_grid]]) {{
    size_t offset = size_t(tid + 1u);
    uint position = tid;
    {body}
    counts[tid * 2u] = uint(offset);
    counts[tid * 2u + 1u] = position;
}}
"""
    (work / "source.metal").write_text(source, encoding="utf-8")
    with tempfile.TemporaryDirectory(
        prefix=".float-atomic-memory-", dir=root
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
                entry_points={relative: ("atomic_memory",)},
                workgroup_size=(1, 1, 1),
                index_range_assertions=(
                    (
                        {
                            "source": relative,
                            "expression": "offset",
                            "minimum": 1,
                            "maximum": len(WORDS),
                        },
                    )
                    if target == "opengl"
                    else ()
                ),
            ),
            format_output=False,
        )
        descriptor, package = _prepare_native_package(report, work)
    inputs, outputs = _workload(operation)
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
        {"workgroupCount": [len(WORDS), 1, 1], "workgroupSize": [1, 1, 1]},
        expected_target=target,
    )
    assert not request.execution_plan.diagnostics
    return source, request, expected


@pytest.mark.parametrize("operation", ("load", "store"))
def test_pinned_float_atomic_memory_executes_natively(tmp_path, operation):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required pinned float atomic execution")
    assert sys.platform in {"darwin", "win32"}
    root = Path(os.environ["CROSTL_MLX_CURRENT_ROOT"]).resolve()
    target = "metal" if sys.platform == "darwin" else "directx"
    hashes = _verify_source(root, (HEADER,))
    try:
        source, request, expected = _request(root, target, tmp_path, operation)
        (tmp_path / "workload.json").write_text(
            json.dumps(
                {
                    "commit": COMMIT,
                    "headers": hashes,
                    "operation": operation,
                    "inputs": _workload(operation)[0],
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
            original_entry="atomic_memory",
            metal_compile_flags=("-I", str(root)),
            validate=_validate,
        )
    finally:
        assert _verify_source(root, (HEADER,)) == hashes


@pytest.mark.parametrize("operation", ("load", "store"))
def test_pinned_float_atomic_memory_workload_preserves_raw_words(operation):
    inputs, outputs = _workload(operation)
    assert inputs["values"]["encoding"] == outputs["values"]["encoding"] == FLOAT32_BITS
    assert inputs["values"]["values"][1:-1] == WORDS
    assert outputs["results"]["values"][:-3] == (
        WORDS if operation == "load" else WORDS[::-1]
    )
    assert outputs["values"]["values"][1:-1] == outputs["results"]["values"][:-3]
    assert outputs["values"]["values"][:: len(WORDS) + 1] == [GUARD, GUARD]
    assert (
        outputs["results"]["values"][-3:]
        == outputs["counts"]["values"][-3:]
        == [GUARD] * 3
    )

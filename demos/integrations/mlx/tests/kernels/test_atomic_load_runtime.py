"""Execute the pinned MLX atomic-load helper without changing its source."""

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
from demos.integrations.mlx.tests.kernels.test_current_binary_shapes import (
    _prepare_native_package,
)
from demos.integrations.mlx.tests.kernels.test_current_gather import _validate
from demos.integrations.mlx.tests.kernels.test_general_scatter_runtime import (
    _verify_source,
)
from tests.test_translator.test_atomic_load_runtime import REQUIRE_ENV
from tests.test_translator.test_boolean_buffer_runtime import _bound_values
from tests.test_translator.test_loop_updates import _execute

HEADER = "mlx/backend/metal/kernels/atomic.h"
GUARD = 123456789


def _workload(kind):
    values = (
        [-2147483648, -7, 2147483647] if kind == "int" else [0, 2147483648, 4294967295]
    )
    dtype = f"{kind}32"
    source = {"dtype": dtype, "shape": [5, 1], "values": [GUARD, *values, GUARD]}
    inputs = {
        "values": source,
        "results": {"dtype": dtype, "shape": [6], "values": [GUARD] * 6},
    }
    outputs = {
        "values": source,
        "results": {
            "dtype": dtype,
            "shape": [6],
            "values": [*values, GUARD, GUARD, GUARD],
        },
    }
    return inputs, outputs


def _request(root, target, work, kind):
    source = f"""#include "{HEADER}"
kernel void load_values(device mlx_atomic<{kind}>* values [[buffer(0)]],
                        device {kind}* results [[buffer(1)]],
                        uint tid [[thread_position_in_grid]]) {{
    results[tid] = mlx_atomic_load_explicit(values, size_t(tid + 1u));
}}
"""
    (work / "source.metal").write_text(source, encoding="utf-8")
    with tempfile.TemporaryDirectory(
        prefix=".atomic-load-proof-", dir=root
    ) as temporary:
        staging = Path(temporary)
        wrapper = staging / "load.metal"
        wrapper.write_text(source, encoding="utf-8")
        relative = wrapper.relative_to(root).as_posix()
        # Three invocations address elements 1..3 of the five-element allocation.
        assertions = (
            ({"source": relative, "expression": "offset", "minimum": 1, "maximum": 3},)
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
                entry_points={relative: ("load_values",)},
                workgroup_size=(1, 1, 1),
                index_range_assertions=assertions,
            ),
            format_output=False,
        )
        descriptor, package = _prepare_native_package(report, work)
    inputs, outputs = _workload(kind)
    expected = _bound_values(descriptor, outputs)
    request = build_native_loader_dispatch_request(
        descriptor,
        package,
        _bound_values(descriptor, inputs),
        expected,
        {"workgroupCount": [3, 1, 1], "workgroupSize": [1, 1, 1]},
        expected_target=target,
    )
    assert not request.execution_plan.diagnostics
    return source, request, expected


@pytest.mark.parametrize("kind", ("int", "uint"))
def test_pinned_atomic_load_executes_natively(tmp_path, kind):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required pinned atomic-load execution")
    root = Path(os.environ["CROSTL_MLX_CURRENT_ROOT"]).resolve()
    target = {"darwin": "metal", "win32": "directx", "linux": "opengl"}[sys.platform]
    hashes = _verify_source(root, (HEADER,))
    try:
        source, request, outputs = _request(root, target, tmp_path, kind)
        (tmp_path / "workload.json").write_text(
            json.dumps(
                {
                    "commit": COMMIT,
                    "headers": hashes,
                    "kind": kind,
                    "inputs": _workload(kind)[0],
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
            original_entry="load_values",
            metal_compile_flags=("-I", str(root)),
            validate=_validate,
        )
    finally:
        assert _verify_source(root, (HEADER,)) == hashes


@pytest.mark.parametrize("kind", ("int", "uint"))
def test_atomic_load_workload_preserves_input_and_guards(kind):
    inputs, outputs = _workload(kind)
    assert outputs["values"] == inputs["values"]
    assert outputs["results"]["values"][:3] == inputs["values"]["values"][1:4]
    assert outputs["results"]["values"][3:] == [GUARD] * 3
    assert inputs["results"]["values"] == [GUARD] * 6

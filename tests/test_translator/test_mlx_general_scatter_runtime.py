"""Pinned general scatter preserves indexed atomic updates and partial chunks."""

import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from crosstl.project import (
    ProjectConfig,
    build_native_loader_dispatch_request,
    translate_project,
)
from demos.integrations.mlx.portable_host.prepare import COMMIT, require_revision
from tests.test_translator.test_loop_updates import _execute
from tests.test_translator.test_mlx_current_binary_shapes import _prepare_native_package
from tests.test_translator.test_mlx_current_gather import _validate

REQUIRE_ENV = "CROSTL_REQUIRE_MLX_GENERAL_SCATTER"
HEADERS = (
    "mlx/backend/metal/kernels/utils.h",
    "mlx/backend/metal/kernels/reduce_utils.h",
    "mlx/backend/metal/kernels/indexing/scatter.h",
)
JIT = "mlx/backend/metal/jit/indexing.h"


def _verify_source(root, headers):
    require_revision(root)
    subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "diff",
            "--exit-code",
            "HEAD",
            "--",
            "mlx/backend/metal/kernels",
            JIT,
        ],
        check=True,
        capture_output=True,
        timeout=30,
    )
    hashes = {}
    for header in headers:
        content = (root / header).read_bytes()
        original = subprocess.check_output(
            ["git", "-C", str(root), "show", f"{COMMIT}:{header}"]
        )
        assert content == original
        hashes[header] = hashlib.sha256(content).hexdigest()
    return hashes


def _source(root, count, operation="sum"):
    contiguous, nwork = (False, 1) if count == 1 else (True, 4)
    template = re.search(
        r'scatter_kernels = R"\((.*?)\)";', (root / JIT).read_text(), re.S
    ).group(1)
    entry = f"scatterint32int64_{operation}_{count}_updc_{str(contiguous).lower()}_nwork{nwork}_int"
    wrapper = template.format(
        f"int32int64_{operation}",
        "int",
        "int64_t",
        {"sum": "Sum<int>", "prod": "Prod<int>"}[operation],
        count,
        "\n".join(
            f"const device int64_t *idx{i} [[buffer({20 + i})]]," for i in range(count)
        ),
        ",".join(f"idx{i}" for i in range(count)),
        str(contiguous).lower(),
        nwork,
        "int",
    )
    return entry, "".join(f'#include "{header}"\n' for header in HEADERS) + wrapper


def _workload(count, target="metal", operation="sum"):
    updates = [-7, 3, 8, 2, -5, 4, 9, 6][: 8 if count == 1 else 7]
    if operation == "prod":
        updates = [-1, 2, 1, -2, -1, 2, 1, 0][: 8 if count == 1 else 7]
    columns = [2, 2, 4, 3, 2, 4, -1, -1][: len(updates)]
    rows = [0, 0, 2, 1, 0, 2, -1][: len(updates)]
    shape = [5] if count == 1 else [3, 5]
    strides = [1] if count == 1 else [5, 1]
    size = 5 if count == 1 else 15
    initial = [100 + i * 10 for i in range(size)] + [-12345] * 32
    expected = list(initial)
    for i, value in enumerate(updates):
        column = columns[i] % 5
        row = 0 if count == 1 else rows[i] % 3
        if operation == "prod":
            expected[row * 5 + column] *= value
        else:
            assert operation == "sum"
            expected[row * 5 + column] += value

    def typed(dtype, values, shape=None):
        return {"dtype": dtype, "shape": shape or [len(values)], "values": values}

    inputs = {
        "updates": typed(
            "int32",
            [v for value in updates for v in (value, -999)] if count == 1 else updates,
        ),
        "out": typed("int32", initial, [size + 32, 1]),
        "upd_shape": typed("int32", [len(updates)] + [1] * count),
        "upd_strides": typed("int64", [2, 0] if count == 1 else [1, 1, 1]),
        "upd_ndim": typed("uint64", [count + 1]),
        "upd_size": typed("uint64", [1]),
        "out_shape": typed("int32", shape),
        "out_strides": typed("int64", strides),
        "out_ndim": typed("uint64", [count]),
        "axes": typed("int32", list(range(count))),
        "idx_shapes": typed("int32", [len(updates)] * count),
        "idx_strides": typed("int64", [2] if count == 1 else [1, 1]),
        "idx_contigs": typed("bool", [count != 1] * count),
        "idx_ndim": typed("int32", [1]),
        "idx_size": typed("uint64", [len(updates)]),
        "idx0": typed(
            "int64",
            [v for index in columns for v in (index, -999)] if count == 1 else rows,
        ),
    }
    if count == 2:
        inputs["idx1"] = typed("int64", columns)
    if target != "metal":
        inputs["idx_contigs"] = typed("uint32", [int(count != 1)] * count)
    grid = [1, len(updates) if count == 1 else (len(updates) + 3) // 4, 1]
    return inputs, {**inputs["out"], "values": expected}, grid


def _request(root, target, work, count, operation="sum"):
    entry, source = _source(root, count, operation)
    supplied, expected, grid = _workload(count, target, operation)
    return _prepare_request(root, target, work, entry, source, supplied, expected, grid)


def _prepare_request(root, target, work, entry, source, supplied, expected, grid):
    (work / "source.metal").write_text(source, encoding="utf-8")
    with tempfile.TemporaryDirectory(
        prefix=".general-scatter-proof-", dir=root
    ) as temporary:
        staging = Path(temporary)
        wrapper = staging / "scatter.metal"
        wrapper.write_text(source, encoding="utf-8")
        relative = wrapper.relative_to(root).as_posix()
        assertions = (
            (
                {
                    "source": relative,
                    "expression": "reference.offset + index",
                    "minimum": 0,
                    "maximum": 15,
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
                entry_points={relative: (entry,)},
                workgroup_size=(1, 1, 1),
                index_range_assertions=assertions,
            ),
            format_output=False,
        )
        descriptor, package = _prepare_native_package(report, work)
    inputs, outputs, matched = {}, {}, set()
    for binding in descriptor["bindings"]:
        if "executionInput" in binding.get("provenance", {}):
            continue
        layout = binding["scalarLayout"]
        member = layout.get("memberName", binding["name"]).removeprefix(
            entry.rstrip("_") + "_"
        )
        name = "out" if member == "out_" else member
        assert name not in matched
        assert layout["elementType"] == supplied[name]["dtype"]
        inputs[binding["name"]] = supplied[name]
        matched.add(name)
        if name == "out":
            outputs[binding["name"]] = expected
    assert matched == set(supplied)
    request = build_native_loader_dispatch_request(
        descriptor,
        package,
        inputs,
        outputs,
        {"workgroupCount": grid, "workgroupSize": [1, 1, 1]},
        expected_target=target,
    )
    assert not request.execution_plan.diagnostics
    return entry, source, request, outputs


@pytest.mark.parametrize("count", (1, 2))
def test_general_scatter_workload_preserves_guards_and_duplicate_updates(count):
    supplied, expected, grid = _workload(count)
    size = 5 if count == 1 else 15
    assert expected["values"][size:] == [-12345] * 32
    assert expected["values"][:2] == [100, 110]
    assert sum(expected["values"][:size]) - sum(
        supplied["out"]["values"][:size]
    ) == sum([-7, 3, 8, 2, -5, 4, 9, 6][: 8 if count == 1 else 7])
    assert grid == ([1, 8, 1] if count == 1 else [1, 2, 1])


@pytest.mark.parametrize("count", (1, 2))
def test_general_product_scatter_has_bounded_signed_zero_and_duplicate_updates(count):
    supplied, expected, grid = _workload(count, operation="prod")
    size = 5 if count == 1 else 15
    assert expected["values"][size:] == [-12345] * 32
    if count == 1:
        assert expected["values"][:size] == [100, 110, 240, -260, 0]
    else:
        assert expected["values"][2] == 240
        assert expected["values"][8] == -360
        assert expected["values"][14] == 480
    assert max(abs(value) for value in expected["values"][:size]) < 1024
    assert len(supplied["updates"]["values"]) == (16 if count == 1 else 7)
    assert grid == ([1, 8, 1] if count == 1 else [1, 2, 1])


@pytest.mark.parametrize("count", (1, 2))
@pytest.mark.parametrize("operation", ("sum", "prod"))
def test_pinned_general_scatter_executes_natively(tmp_path, count, operation):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(
            f"set {REQUIRE_ENV}=1 for required pinned general-scatter execution"
        )
    root = Path(os.environ["CROSTL_MLX_CURRENT_ROOT"]).resolve()
    target = {"darwin": "metal", "win32": "directx", "linux": "opengl"}[sys.platform]
    hashes = _verify_source(root, (*HEADERS, JIT))
    try:
        entry, source, request, outputs = _request(
            root, target, tmp_path, count, operation
        )
        (tmp_path / "workload.json").write_text(
            json.dumps(
                {
                    "commit": COMMIT,
                    "headers": hashes,
                    "entry": entry,
                    "indexCount": count,
                    "operation": operation,
                    "inputs": _workload(count, target, operation)[0],
                    "outputs": outputs,
                },
                indent=2,
            )
        )
        _execute(
            request,
            outputs,
            tmp_path,
            original_source=source,
            original_entry=entry,
            metal_compile_flags=("-I", str(root)),
            validate=_validate,
        )
    finally:
        assert _verify_source(root, (*HEADERS, JIT)) == hashes

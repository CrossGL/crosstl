"""Execute unchanged pinned replacement and additive axis-scatter kernels."""

import hashlib
import json
import os
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
from demos.integrations.mlx.tests.kernels.test_current_binary_shapes import (
    _prepare_native_package,
)
from demos.integrations.mlx.tests.kernels.test_current_gather import _validate
from tests.test_translator.test_loop_updates import _execute

REQUIRE_ENV = "CROSTL_REQUIRE_MLX_SCATTER_AXIS"
HEADERS = (
    "mlx/backend/metal/kernels/utils.h",
    "mlx/backend/metal/kernels/reduce_utils.h",
    "mlx/backend/metal/kernels/indexing/scatter_axis.h",
)


def _verify_source(root):
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
        ],
        check=True,
        capture_output=True,
        timeout=30,
    )
    hashes = {}
    for header in HEADERS:
        content = (root / header).read_bytes()
        original = subprocess.check_output(
            ["git", "-C", str(root), "show", f"{COMMIT}:{header}"]
        )
        assert content == original
        hashes[header] = hashlib.sha256(content).hexdigest()
    return hashes


def _source(operation):
    op = "None" if operation == "replace" else "Sum<int>"
    label = "none" if operation == "replace" else "sum"
    entry = f"scatter_axisint32int32_{label}_intncnc"
    arguments = f"int, int, int, {op}, false, false"
    source = "".join(f'#include "{header}"\n' for header in HEADERS)
    source += (
        f'template [[host_name("{entry}")]] [[kernel]]\n'
        f"decltype(scatter_axis<{arguments}>) scatter_axis<{arguments}>;\n"
    )
    return entry, source


def _workload(operation):
    # Replacement duplicates have identical values, so every legal interleaving
    # has the same result. Addition uses distinct updates to detect lost writes.
    updates = (
        [
            -2147483648,
            -2147483648,
            2147483647,
            -7,
            -2147483648,
            2147483647,
            -7,
            2147483647,
        ]
        if operation == "replace"
        else [-7, 3, 8, 2, -5, 4, 9, 6]
    )
    indices = [2, 2, 4, 3, 2, 4, 3, -1]
    initial = [-12345, -12345, 100, 200, 300] + [-12345] * 32
    expected = list(initial)
    for index, value in zip(indices, updates):
        position = index if index >= 0 else index + 5
        assert 0 <= position <= 4
        if operation == "replace":
            expected[position] = value
        else:
            expected[position] += value

    def typed(dtype, values, shape=None):
        return {"dtype": dtype, "shape": shape or [len(values)], "values": values}

    inputs = {
        "upd": typed("int32", updates),
        "indices": typed("int32", indices),
        "out": typed("int32", initial, [37, 1]),
        "shape": typed("int32", [1]),
        "upd_strides": typed("int64", [0]),
        "idx_strides": typed("int64", [0]),
        "ndim": typed("uint64", [0]),
        "axis": typed("int32", [0]),
        "out_axis_size": typed("int32", [5]),
        "upd_ax_stride": typed("uint64", [1]),
        "idx_ax_stride": typed("uint64", [1]),
    }
    return inputs, {**inputs["out"], "values": expected}


def _request(root, target, work, operation):
    entry, source = _source(operation)
    (work / "source.metal").write_text(source, encoding="utf-8")
    with tempfile.TemporaryDirectory(
        prefix=".scatter-axis-proof-", dir=root
    ) as temporary:
        staging = Path(temporary)
        wrapper = staging / "axis.metal"
        wrapper.write_text(source, encoding="utf-8")
        relative = wrapper.relative_to(root).as_posix()
        assertions = (
            (
                {
                    "source": relative,
                    "expression": "offset",
                    "minimum": 0,
                    "maximum": 4,
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
    supplied, expected = _workload(operation)
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
        {"workgroupCount": [1, 8, 1], "workgroupSize": [1, 1, 1]},
        expected_target=target,
    )
    assert not request.execution_plan.diagnostics
    return entry, source, request, outputs


@pytest.mark.parametrize("operation", ("replace", "add"))
def test_scatter_axis_workload_preserves_guards_and_signed_indices(operation):
    supplied, expected = _workload(operation)
    assert supplied["indices"]["values"][-1] == -1
    assert expected["values"][:2] == [-12345] * 2
    assert expected["values"][5:] == [-12345] * 32
    assert expected["values"][2:5] == (
        [-2147483648, -7, 2147483647] if operation == "replace" else [91, 211, 318]
    )


@pytest.mark.parametrize("operation", ("replace", "add"))
def test_pinned_scatter_axis_executes_natively(tmp_path, operation):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required pinned scatter-axis execution")
    root = Path(os.environ["CROSTL_MLX_CURRENT_ROOT"]).resolve()
    target = {"darwin": "metal", "win32": "directx", "linux": "opengl"}[sys.platform]
    hashes = _verify_source(root)
    entry, source, request, outputs = _request(root, target, tmp_path, operation)
    (tmp_path / "workload.json").write_text(
        json.dumps(
            {
                "commit": COMMIT,
                "headers": hashes,
                "entry": entry,
                "operation": operation,
                "offsetRange": [0, 4],
                "inputs": _workload(operation)[0],
                "outputs": outputs,
            },
            indent=2,
        )
    )
    try:
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
        assert _verify_source(root) == hashes

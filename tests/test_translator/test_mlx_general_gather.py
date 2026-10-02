"""Execute the pinned general-gather JIT wrapper without rewriting its body."""

import hashlib
import itertools
import json
import math
import os
import re
import subprocess
import tempfile
from pathlib import Path

import pytest

from crosstl.project import (
    ProjectConfig,
    build_native_loader_dispatch_request,
    translate_project,
)
from crosstl.project.runtime_value_encoding import FLOAT32_BITS
from demos.integrations.mlx.portable_host.prepare import COMMIT, require_revision
from tests.test_translator.test_boolean_buffer_runtime import _bound_values
from tests.test_translator.test_float_storage_encoding import WORDS
from tests.test_translator.test_loop_updates import _execute
from tests.test_translator.test_mlx_current_binary_shapes import _prepare_native_package
from tests.test_translator.test_mlx_current_gather import _validate

REQUIRE_ENV = "CROSTL_REQUIRE_MLX_GENERAL_GATHER"
HEADER = "mlx/backend/metal/kernels/indexing/gather.h"
JIT_HEADER = "mlx/backend/metal/jit/indexing.h"
SPECIALIZATIONS = ((0, 0), (1, 0), (1, 1), (1, 2), (2, 2), (2, 3))
LAYOUTS = ("dense", "transposed", "strided", "broadcast")
GUARD = 0x42F60000


def _source(root, count, ndim):
    template = re.search(
        r'gather_kernels = R"\((.*?)\)";', (root / JIT_HEADER).read_text(), re.S
    )
    assert template is not None, "the pinned gather JIT definition is missing"
    kind = "int" if count else "bool"
    suffix = "float32int32" if count else "float32"
    arguments = "\n".join(
        f"const device {kind}* idx{i} [[buffer({20 + i})]]," for i in range(count)
    )
    indices = ", ".join(f"idx{i}" for i in range(count))
    wrapper = template.group(1).format(
        suffix, "float", kind, count, arguments, indices, ndim, "int"
    )
    return (
        f"gather{suffix}_{count}_{ndim}_int",
        f'#include "mlx/backend/metal/kernels/utils.h"\n#include "{HEADER}"\n{wrapper}',
    )


def _coordinates(shape):
    return itertools.product(*(range(extent) for extent in shape))


def _dense_strides(shape):
    return tuple(math.prod(shape[axis + 1 :]) for axis in range(len(shape)))


def _workload(count, ndim, layout):
    shape = (4, 3, 2)
    strides = {
        "dense": (6, 2, 1),
        "transposed": (1, 8, 4),
        "strided": (13, 4, 2),
        "broadcast": (6, 2, 0),
    }[layout]
    axes = tuple(range(count))
    slices = tuple(1 if axis in axes else extent for axis, extent in enumerate(shape))
    if not count:
        slices = (2, 2, 2)
    index_shape = {0: (), 1: (6,), 2: (2, 3), 3: (2, 2, 3)}[ndim]
    storage_size = 1 + sum(
        (extent - 1) * stride for extent, stride in zip(shape, strides)
    )
    src = [WORDS[i] if i < len(WORDS) else 0x3F000000 + i for i in range(storage_size)]
    buffers, index_strides, contiguous = [], [], []
    for axis in axes:
        steps = _dense_strides(index_shape)
        if layout == "strided":
            steps = tuple(step * 2 for step in steps)
        elif layout == "transposed" and axis == 1:
            steps = tuple(math.prod(index_shape[:i]) for i in range(ndim))
        elif layout == "broadcast" and ndim:
            steps = (0, *steps[1:])
        size = 1 + sum(
            (extent - 1) * stride for extent, stride in zip(index_shape, steps)
        )
        buffers.append(
            [
                (i + axis) % shape[axis] - (shape[axis] if i % 2 else 0)
                for i in range(size)
            ]
        )
        index_strides.append(steps)
        contiguous.append(steps == _dense_strides(index_shape))

    expected = []
    for coordinate in _coordinates(index_shape):
        origins = [0] * len(shape)
        for axis, buffer, steps in zip(axes, buffers, index_strides):
            value = buffer[
                sum(index * stride for index, stride in zip(coordinate, steps))
            ]
            origins[axis] = value + shape[axis] if value < 0 else value
        for offset in _coordinates(slices):
            location = sum(
                (base + delta) * stride
                for base, delta, stride in zip(origins, offset, strides)
            )
            assert 0 <= location < len(src)
            expected.append(src[location])

    def typed(dtype, values):
        return {
            "dtype": dtype,
            "shape": [len(values)],
            "values": list(values),
            **({"encoding": FLOAT32_BITS} if dtype == "float32" else {}),
        }

    inputs = {
        "src": typed("float32", src + [GUARD] * 16),
        "out_": typed("float32", [GUARD] * (len(expected) + 16)),
        "src_shape": typed("int32", shape),
        "src_strides": typed("int64", strides),
        "src_ndim": typed("uint64", [len(shape)]),
        "slice_sizes": typed("int32", slices),
        "axes": typed("int32", axes or (0,)),
        "idx_shapes": typed("int32", index_shape * count or (1,)),
        "idx_strides": typed(
            "int64", tuple(itertools.chain.from_iterable(index_strides)) or (0,)
        ),
        "idx_contigs": typed("bool", contiguous or (True,)),
        "idx_ndim": typed("int32", [ndim]),
        **{f"idx{i}": typed("int32", buffer) for i, buffer in enumerate(buffers)},
    }
    return {
        "count": count,
        "ndim": ndim,
        "layout": layout,
        "indexShape": list(index_shape),
        "inputs": inputs,
        "outputs": {"out_": typed("float32", expected + [GUARD] * 16)},
        "grid": {
            "workgroupCount": [
                index_shape[0] if ndim else 1,
                math.prod(index_shape[1:]),
                math.prod(slices),
            ],
            "workgroupSize": [1, 1, 1],
        },
    }


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
            JIT_HEADER,
        ],
        check=True,
        capture_output=True,
        timeout=30,
    )
    return {
        path: hashlib.sha256((root / path).read_bytes()).hexdigest()
        for path in (HEADER, JIT_HEADER)
    }


@pytest.fixture(scope="module")
def gather_root():
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required pinned general-gather execution")
    root = Path(os.environ["CROSTL_MLX_CURRENT_ROOT"]).resolve()
    before = _verify_source(root)
    yield root, before
    assert _verify_source(root) == before


@pytest.fixture(
    scope="module",
    params=[
        pytest.param(
            spec,
            id=f"indices{spec[0]}-ndim{spec[1]}",
            marks=pytest.mark.xdist_group(f"gather-{spec[0]}-{spec[1]}"),
        )
        for spec in SPECIALIZATIONS
    ],
)
def gather_package(request, tmp_path_factory, gather_root):
    root, hashes = gather_root
    count, ndim = request.param
    retained = tmp_path_factory.mktemp(f"gather-{count}-{ndim}")
    entry, source, descriptor, package = _package(root, count, ndim, retained, "metal")
    return root, hashes, count, ndim, entry, source, descriptor, package


def _package(root, count, ndim, retained, target, *, index_range_assertions=()):
    entry, source = _source(root, count, ndim)
    (retained / "source.metal").write_text(source)
    with tempfile.TemporaryDirectory(prefix=".general-gather-", dir=root) as directory:
        work = Path(directory)
        wrapper = work / "gather.metal"
        wrapper.write_text(source)
        relative = wrapper.relative_to(root).as_posix()
        report = translate_project(
            ProjectConfig(
                root=root,
                source_roots=(work.name,),
                include_patterns=(relative,),
                include_dirs=(".",),
                targets=(target,),
                output_dir=f"{work.name}/out",
                entry_points={relative: (entry,)},
                workgroup_size=(1, 1, 1),
                index_range_assertions=index_range_assertions,
            ),
            format_output=False,
        )
        report.write_json(retained / "report.json")
        assert report.to_json()["summary"]["failedCount"] == 0, report.to_json()
        descriptor, package = _prepare_native_package(report, retained)
    return entry, source, descriptor, package


@pytest.mark.parametrize("layout", LAYOUTS)
def test_pinned_general_gather_executes_original_and_generated(
    gather_package, tmp_path, layout
):
    root, hashes, count, ndim, entry, source, descriptor, package = gather_package
    workload = _workload(count, ndim, layout)
    (tmp_path / "workload.json").write_text(
        json.dumps(
            {"commit": COMMIT, "headers": hashes, "entry": entry, **workload}, indent=2
        )
    )
    (tmp_path / "source.metal").write_text(source)
    expected = _bound_values(descriptor, workload["outputs"])
    request = build_native_loader_dispatch_request(
        descriptor,
        package,
        _bound_values(descriptor, workload["inputs"]),
        expected,
        workload["grid"],
        expected_target="metal",
    )
    assert not request.execution_plan.diagnostics
    _execute(
        request,
        expected,
        tmp_path,
        original_source=source,
        original_entry=entry,
        metal_compile_flags=("-I", str(root)),
        validate=_validate,
    )


@pytest.mark.parametrize(("count", "ndim"), SPECIALIZATIONS)
@pytest.mark.parametrize("layout", LAYOUTS)
def test_general_gather_workload_is_bounded(count, ndim, layout):
    workload = _workload(count, ndim, layout)
    output = workload["outputs"]["out_"]
    assert output["values"][-16:] == [GUARD] * 16
    assert math.prod(workload["grid"]["workgroupCount"]) == len(output["values"]) - 16
    assert workload["inputs"]["idx_ndim"]["values"] == [ndim]


def test_general_gather_gate_requires_native_evidence():
    from tools import ci_coverage

    root = Path(__file__).resolve().parents[2]
    workflow = (root / ".github/workflows/mlx-gather-roundtrip.yml").read_text()
    step = ci_coverage.workflow_step_section(
        workflow, "Validate general gather and empty arrays"
    )
    assert f'{REQUIRE_ENV}: "1"' in step
    assert 'CROSTL_REQUIRE_METAL_EMPTY_ARRAYS: "1"' in step
    assert "test_mlx_general_gather.py" in step and "test_metal_empty_arrays.py" in step
    assert "--timeout-seconds 1200" in step and "--junitxml=" in step
    assert "--basetemp=" in step and "-n auto --dist loadgroup" in step
    assert "continue-on-error" not in workflow and "if:" not in step
    assert f'MLX_COMMIT: "{COMMIT}"' in workflow
    assert "runs-on: macos-26" in workflow and "timeout-minutes: 35" in workflow
    assert "if: always()" in workflow and "include-hidden-files: true" in workflow
    for event in ("pull_request", "push"):
        assert "tests/test_translator/**" in ci_coverage.workflow_event_path_filters(
            workflow, event
        )

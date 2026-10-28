"""Long-column reductions follow strided_reduce_longcolumn's plan and launch."""

import ctypes
import re
from pathlib import Path

import pytest

from demos.integrations.mlx.portable_host import (
    column_long_layout,
    column_long_packages,
    column_workloads,
    lazy_package_evidence,
    runtime,
)
from demos.integrations.mlx.portable_host.reduction_packages import (
    ENTRIES,
    HALF_FLOAT_ENTRIES,
)

HOST = Path(column_long_layout.__file__).parent


def _buffers(rows=1024, columns=3, outer=2, dimension=1, *, partials=None):
    """One dense long-column launch: ``outer`` column spans of ``columns``."""
    reductions = [2] * (dimension - 1) + [rows]
    steps, extent = [], columns
    for size in reversed(reductions):
        steps.insert(0, extent)
        extent *= size
    # The output spans lie outside the reduced axes, as MLX plans them.
    span, extent = extent, extent * outer
    total = rows * 2 ** (dimension - 1)
    blocks = column_long_layout.outer_blocks(total) if partials is None else partials
    data = {
        "in": ("float32", [1.0] * extent),
        "out": ("float32", [0.0] * (outer * columns * blocks)),
        "reduction_size": ("uint64", [rows]),
        "reduction_stride": ("int64", [columns]),
        "shape": ("int32", [outer]),
        "strides": ("int64", [span]),
        "ndim": ("int32", [1]),
        "reduce_shape": ("int32", reductions),
        "reduce_strides": ("int64", steps),
        "reduce_ndim": ("int32", [dimension]),
        "non_col_reductions": ("uint64", [2 ** (dimension - 1)]),
        "out_size": ("uint64", [outer * columns]),
    }
    memory = {
        name: (runtime.TYPES[dtype] * len(values))(*values)
        for name, (dtype, values) in data.items()
    }
    buffers = {
        name: runtime.Buffer(
            name.encode(),
            dtype.encode(),
            ctypes.addressof(memory[name]),
            len(values),
            int(name == "out"),
        )
        for name, (dtype, values) in data.items()
    }
    template = 1 if dimension == 1 else 2 if dimension == 2 else 5
    entry = f"col_reduce_longcolumn_{template}_reduce_sumfloat32"
    execution = {
        "workgroupCount": [outer, 1, blocks],
        "workgroupSize": [*column_long_layout.geometry(columns, total), 1],
    }
    return entry, buffers, total * outer * columns, execution, memory


def test_long_entries_are_the_pinned_instantiations_the_column_passes_run():
    source = (
        Path(__file__).resolve().parents[5]
        / "demos/integrations/mlx/portable_host/column_long_layout.py"
    )
    assert source.exists()
    operations = {
        name.removeprefix("all_reduce_") for name in {**ENTRIES, **HALF_FLOAT_ENTRIES}
    }
    assert len(column_long_layout.LONG_COLUMN_ENTRIES) == 3 * len(operations) == 66
    for entry, dtype in column_long_layout.LONG_COLUMN_ENTRIES.items():
        match = re.fullmatch(r"col_reduce_longcolumn_([125])_reduce_(\w+)", entry)
        assert match and match.group(2) in operations
        assert match.group(2).endswith(dtype)
    assert set(column_workloads.LONG_COLUMN_ENTRIES) == {
        entry
        for entry in column_long_layout.LONG_COLUMN_ENTRIES
        if not entry.endswith(("float16", "bfloat16"))
    }


@pytest.mark.parametrize(
    "total,blocks,rows", [(1024, 32, 32), (32767, 32, 32), (32768, 128, 32)]
)
def test_long_geometry_follows_strided_reduce_longcolumn(total, blocks, rows):
    assert column_long_layout.outer_blocks(total) == blocks
    assert column_long_layout.geometry(7, total) == (7, rows)


def test_long_variants_round_trip_and_bound_the_threadgroup():
    entry = "col_reduce_longcolumn_2_reduce_maxint32"
    key = column_long_layout.variant(entry, (31, 32, 1))
    assert key == f"{entry}@31x32"
    assert column_long_layout.split_variant(key) == (entry, (31, 32, 1))
    for workgroup in ((32, 32, 1), (0, 32, 1), (3, 33, 1), (3, 32, 2), (3.0, 32, 1)):
        with pytest.raises(ValueError, match="threadgroup"):
            column_long_layout.variant(entry, workgroup)
    for key in (f"{entry}@3", f"{entry}@3x032", "col_reduce_small_1_reduce_sumfloat32@3x32"):
        with pytest.raises(ValueError):
            column_long_layout.split_variant(key)


@pytest.mark.parametrize("dimension", [1, 2, 3])
@pytest.mark.parametrize("rows", [1024, 32768])
def test_valid_long_column_launch(dimension, rows):
    if dimension > 1:
        rows //= 2 ** (dimension - 1)
    entry, buffers, count, execution, _ = _buffers(rows=rows, dimension=dimension)
    assert column_long_layout.validate(entry, buffers, count, execution)


@pytest.mark.parametrize(
    "fault",
    ["short", "wide-stride", "blocks", "out-size", "grid", "rows", "template", "entry"],
)
def test_invalid_long_column_launch_is_rejected(fault):
    kwargs = {}
    if fault == "short":
        kwargs["rows"] = 1023
    elif fault == "wide-stride":
        kwargs["columns"] = 32
    elif fault == "blocks":
        kwargs["partials"] = 128
    entry, buffers, count, execution, memory = _buffers(**kwargs)
    if fault == "wide-stride":
        execution["workgroupSize"][0] = 31
    if fault == "out-size":
        memory["out_size"][0] = 2
    elif fault == "grid":
        execution["workgroupCount"] = [1, 2, 32]
    elif fault == "rows":
        execution["workgroupSize"][1] = 16
    elif fault == "template":
        entry = "col_reduce_longcolumn_2_reduce_sumfloat32"
    elif fault == "entry":
        entry = "col_reduce_small_1_reduce_sumfloat32"
    with pytest.raises(ValueError):
        column_long_layout.validate(entry, buffers, count, execution)


def test_long_partials_reduce_to_the_result_in_every_case():
    import numpy as np

    cases = [
        case
        for case in column_workloads.cases()
        if case["entry"] in column_workloads.LONG_COLUMN_ENTRIES
    ]
    assert len(cases) == 76
    seen = set()
    for case in cases:
        _, logical, expected, partials = column_workloads.reference(np, case)
        total = logical.size // expected.size
        assert total >= 1024 and column_workloads.column_stride(case) < 32
        blocks = column_long_layout.outer_blocks(total)
        assert partials.shape == (blocks, *expected.shape)
        combined = column_workloads.reduce(np, partials, case["operation"], 0)
        np.testing.assert_array_equal(combined, expected)
        seen.add(blocks)
    assert seen == {32, 128}


def test_long_variant_cache_and_lazy_family():
    keys = column_long_packages.verification_entries()
    assert len(keys) == 44 and keys == sorted(keys)
    assert all(column_long_layout.split_variant(key) for key in keys)
    entry = "col_reduce_longcolumn_1_reduce_sumfloat32"
    assert lazy_package_evidence.family(entry) == "long-column"
    assert "long-column" in lazy_package_evidence.FAMILIES
    config = column_long_packages.project_config(
        HOST, HOST / "out", "opengl", f"{entry}@7x32"
    )
    rules = config.entry_workgroup_size_rules[column_long_packages.SOURCE]
    assert tuple(map(int, rules[entry])) == (7, 32, 1)


def test_backend_runs_the_long_column_plan_and_its_second_pass():
    source = (HOST / "backend.cpp").read_text()
    assert "long-column reduction plans are not implemented" not in source
    assert '"col_reduce_longcolumn_"' in source
    assert "total >= 32768 ? 128 : 32" in source
    assert "ColumnPass::LongColumn" in source

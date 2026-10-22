"""Small column reductions: plan geometry, launch validation and packaging."""

import ctypes
from pathlib import Path

import pytest

from demos.integrations.mlx.portable_host import (
    column_small_layout,
    column_small_packages,
    column_workloads,
    runtime,
)

ENTRY = "col_reduce_small_1_reduce_sumfloat32"


def _buffers(rows=5, columns=130, prefix=(), outer=1):
    """One rank-``len(prefix) + 1`` launch over ``outer`` dense output rows."""
    reductions = [*prefix, rows]
    steps, extent = [], columns
    for size in reversed(reductions):
        steps.insert(0, extent)
        extent *= size
    data = {
        "in": ("float32", [1.0] * (extent * outer)),
        "out": ("float32", [0.0] * (columns * outer)),
        "reduction_size": ("uint64", [rows]),
        "reduction_stride": ("int64", [columns]),
        "shape": ("int32", [outer] if outer > 1 else [0]),
        "strides": ("int64", [extent] if outer > 1 else [0]),
        "ndim": ("int32", [int(outer > 1)]),
        "reduce_shape": ("int32", reductions),
        "reduce_strides": ("int64", steps),
        "reduce_ndim": ("int32", [len(reductions)]),
        "non_col_reductions": ("uint64", [max(1, len(prefix) and 2 ** len(prefix))]),
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
    return buffers, extent * outer, memory


def _launch(columns, total, outer=1):
    lanes, rows = column_small_layout.geometry(columns, total)
    blocks = -(-columns // 4)
    return {
        "workgroupCount": [-(-blocks // lanes), outer, 1],
        "workgroupSize": [lanes, rows, 1],
    }


def test_entries_cover_every_template_rank_operation_and_type():
    entries = column_small_layout.SMALL_COLUMN_ENTRIES
    assert len(entries) == 3 * 14
    assert {entry.split("_")[3] for entry in entries} == {"1", "2", "5"}
    assert entries[ENTRY] == "float32"
    assert entries["col_reduce_small_5_reduce_orbool_"] == "bool_"
    assert not any("_large_" in entry for entry in entries)


@pytest.mark.parametrize(
    "stride,total,expected",
    [
        (3, 2, (1, 2)),
        (16, 3, (4, 3)),
        (32, 31, (8, 8)),
        (128, 8, (32, 8)),
        (130, 5, (32, 5)),
        (2048, 8, (32, 8)),
        (1, 1, (1, 1)),
    ],
)
def test_geometry_follows_strided_reduce_small(stride, total, expected):
    assert column_small_layout.geometry(stride, total) == expected


def test_variants_name_one_entry_at_one_threadgroup():
    key = column_small_layout.variant(ENTRY, (32, 8, 1))
    assert key == ENTRY + "@32x8"
    assert column_small_layout.split_variant(key) == (ENTRY, (32, 8, 1))
    for workgroup in ((33, 1, 1), (32, 9, 1), (0, 1, 1), (4, 2, 2), (4.0, 2, 1)):
        with pytest.raises(ValueError):
            column_small_layout.variant(ENTRY, workgroup)
    for key in (
        ENTRY,
        ENTRY + "@32x",
        ENTRY + "@032x8",
        "col_reduce_small_3_reduce_sumfloat32@32x8",
        None,
    ):
        with pytest.raises(ValueError):
            column_small_layout.split_variant(key)


@pytest.mark.parametrize(
    "rows,columns,prefix,outer",
    [
        (5, 130, (), 1),
        (31, 32, (), 1),
        (2, 3, (), 1),
        (3, 33, (2,), 2),
        (3, 32, (2, 2), 1),
    ],
)
def test_validate_accepts_the_upstream_launch(rows, columns, prefix, outer):
    buffers, count, _ = _buffers(rows, columns, prefix, outer)
    dimension = {0: 1, 1: 2, 2: 5}[len(prefix)]
    entry = f"col_reduce_small_{dimension}_reduce_sumfloat32"
    total = rows * 2 ** len(prefix)
    metadata = column_small_layout.validate(
        entry, buffers, count, _launch(columns, total, outer)
    )
    assert metadata["reduction_size"] == [rows]
    assert metadata["reduce_shape"] == [*prefix, rows]


@pytest.mark.parametrize(
    "fault", ["looped", "lanes", "rows", "groups", "outer", "rank", "entry", "output"]
)
def test_validate_rejects_launches_upstream_would_not_make(fault):
    rows = 32 if fault == "looped" else 5
    buffers, count, memory = _buffers(rows, 130, (), 2 if fault == "outer" else 1)
    launch = _launch(130, rows, 2 if fault == "outer" else 1)
    entry = ENTRY
    if fault == "lanes":
        launch["workgroupSize"][0] = 16
    elif fault == "rows":
        launch["workgroupSize"][1] = 4
    elif fault == "groups":
        launch["workgroupCount"][0] = 3
    elif fault == "outer":
        launch["workgroupCount"][1] = 1
    elif fault == "rank":
        entry = "col_reduce_small_2_reduce_sumfloat32"
    elif fault == "entry":
        entry = "col_reduce_looped_1_32_32_reduce_sumfloat32"
    elif fault == "output":
        memory["reduction_stride"][0] = 65
    with pytest.raises(ValueError):
        column_small_layout.validate(entry, buffers, count, launch)


def test_package_config_translates_one_entry_at_its_threadgroup(tmp_path):
    key = column_small_layout.variant(ENTRY, (4, 3, 1))
    config = column_small_packages.project_config(tmp_path, "out", "opengl", key)
    assert config.entry_points == {column_small_packages.SOURCE: (ENTRY,)}
    (rule,) = config.entry_workgroup_size_rules[column_small_packages.SOURCE].items()
    assert rule[0] == ENTRY and tuple(int(value) for value in rule[1]) == (4, 3, 1)
    cache = column_small_packages.SmallColumnPackageCache(tmp_path, tmp_path, "opengl")
    assert cache.source_entry(key) == ENTRY
    assert cache.workgroup_size(key) == (4, 3, 1)
    with pytest.raises(ValueError):
        cache.signature(ENTRY)


def test_workload_variants_match_their_launches():
    variants = column_workloads.small_variants()
    assert len(variants) == len(column_small_layout.SMALL_COLUMN_ENTRIES) + 4
    assert set(column_small_packages.verification_entries()) >= variants
    for key in (
        ENTRY + "@1x2",
        ENTRY + "@4x3",
        ENTRY + "@8x8",
        ENTRY + "@32x8",
        "col_reduce_small_5_reduce_maxint32@8x8",
        "col_reduce_small_2_reduce_prodfloat32@9x6",
    ):
        assert key in variants
    for case in column_workloads._small_cases():
        total, columns, _ = column_workloads.small_plan(case)
        assert total < 32 and 1 <= columns <= 2048


def test_backend_routes_small_plans_before_the_looped_passes():
    source = Path(column_small_layout.__file__).with_name("backend.cpp").read_text()
    start = source.index("void dispatch_column_reduce(")
    body = source[start : source.index("\n}\n", start)]
    assert body.index("if (total < 32)") < body.index("const bool two_pass")
    assert "ColumnPass::Small" in body
    assert '"col_reduce_small_"' in source
    assert "small-column" not in source
    assert "std::min<uint64_t>({8, 1024 / lanes, total})" in source


def test_runtime_routes_small_columns_through_the_variant_cache():
    source = Path(runtime.__file__).read_text()
    assert "column_small_layout.variant(" in source
    assert "self.small_columns.get(" in source

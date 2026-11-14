"""Small column reductions: plan geometry, launch validation and packaging."""

import ctypes
import math
from pathlib import Path

import pytest

from demos.integrations.mlx.portable_host import (
    column_reduction_layout,
    column_small_layout,
    column_small_packages,
    column_workloads,
    runtime,
)

ENTRY = "col_reduce_small_1_reduce_sumfloat32"


def _buffers(rows=5, columns=130, prefix=(), outer=1, dtype="float32"):
    """One rank-``len(prefix) + 1`` launch over ``outer`` dense output rows."""
    reductions = [*prefix, rows]
    steps, extent = [], columns
    for size in reversed(reductions):
        steps.insert(0, extent)
        extent *= size
    one = {"float32": 1.0, "float16": 0x3C00, "bfloat16": 0x3F80}[dtype]
    data = {
        "in": (dtype, [one] * (extent * outer)),
        "out": (dtype, [0 * one] * (columns * outer)),
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


@pytest.mark.parametrize("dtype", ("float16", "bfloat16"))
def test_partition_sums_accept_their_16_bit_launches(dtype):
    entry = f"col_reduce_small_1_reduce_sum{dtype}"
    assert column_small_layout.PARTITION_SUM_ENTRIES[entry] == dtype
    # The column workloads reduce MLX arrays, whose 16-bit sums stay refused.
    assert entry not in column_small_layout.SMALL_COLUMN_ENTRIES
    # test_qmm's 8 partitions of 64 by 256 outputs: 131,072 partial products.
    buffers, count, _ = _buffers(8, 16384, dtype=dtype)
    assert count == 131072
    metadata = column_small_layout.validate(entry, buffers, count, _launch(16384, 8))
    assert metadata["reduce_strides"] == [16384]
    key = column_small_layout.variant(entry, (32, 8, 1))
    assert column_small_layout.split_variant(key) == (entry, (32, 8, 1))
    # A float32 view under a 16-bit entry is another kernel's layout.
    buffers, count, _ = _buffers(8, 16384)
    with pytest.raises(ValueError, match="buffer layout"):
        column_small_layout.validate(entry, buffers, count, _launch(16384, 8))
    # Only the partition axis's rank-one sum is translated.
    for other in (
        f"col_reduce_small_2_reduce_sum{dtype}",
        f"col_reduce_small_1_reduce_max{dtype}",
        f"col_reduce_small_1_reduce_prod{dtype}",
    ):
        with pytest.raises(ValueError):
            column_small_layout.signature(other)


def _view(shape, strides, rows, columns):
    """A float32 launch over an explicit outer view of ``rows`` by ``columns``."""
    extent = columns + sum((size - 1) * step for size, step in zip(shape, strides))
    extent += (rows - 1) * columns
    data = {
        "in": ("float32", [1.0] * extent),
        "out": ("float32", [0.0] * (columns * math.prod(shape))),
        "reduction_size": ("uint64", [rows]),
        "reduction_stride": ("int64", [columns]),
        "shape": ("int32", list(shape)),
        "strides": ("int64", list(strides)),
        "ndim": ("int32", [len(shape)]),
        "reduce_shape": ("int32", [rows]),
        "reduce_strides": ("int64", [columns]),
        "reduce_ndim": ("int32", [1]),
        "non_col_reductions": ("uint64", [1]),
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
    return buffers, rows * columns * math.prod(shape), memory


def test_validate_bounds_elements_and_launch_axes(monkeypatch):
    # 90,000 outputs exceed the 65,535 workgroups of the launch's second axis.
    buffers, count, _ = _view((300, 300), (600, 2), 2, 1)
    assert count == 180000 == buffers["in"].count
    with pytest.raises(ValueError, match="65535 workgroups"):
        column_small_layout.validate(ENTRY, buffers, count, _launch(1, 2, 90000))
    # 60,000 outputs fit.
    buffers, count, _ = _view((200, 300), (600, 2), 2, 1)
    column_small_layout.validate(ENTRY, buffers, count, _launch(1, 2, 60000))
    # The element bound covers the logical size and both views.
    monkeypatch.setattr(column_reduction_layout, "MAX_ELEMENTS", count - 1)
    with pytest.raises(ValueError, match="metadata does not match"):
        column_small_layout.validate(ENTRY, buffers, count, _launch(1, 2, 60000))


def test_validate_accepts_reduction_strides_past_one_axis_bound():
    # test_qmm_vjp's 8 partitions of 128 by 512 outputs: a 65,536-column
    # stride, which strided_reduce_small covers with 512 workgroups.
    buffers, count, _ = _buffers(8, 65536)
    assert count == 524288
    launch = _launch(65536, 8)
    assert launch["workgroupCount"] == [512, 1, 1]
    metadata = column_small_layout.validate(ENTRY, buffers, count, launch)
    assert metadata["reduce_strides"] == [65536]


def test_validate_bounds_the_column_axis_of_the_launch(monkeypatch):
    # 2,097,152 columns need 16,384 workgroups of 32 lanes of four columns;
    # with MAX_GROUPS lowered below that, the column axis bounds the launch.
    buffers, count, _ = _buffers(1, 2097152)
    launch = _launch(2097152, 1)
    assert launch["workgroupCount"] == [16384, 1, 1]
    column_small_layout.validate(ENTRY, buffers, count, launch)
    monkeypatch.setattr(column_small_layout, "MAX_GROUPS", 16383)
    with pytest.raises(ValueError, match="65535 workgroups"):
        column_small_layout.validate(ENTRY, buffers, count, launch)


def test_backend_bounds_column_passes_by_elements_and_workgroups():
    source = Path(column_small_layout.__file__).with_name("backend.cpp").read_text()
    assert "constexpr uint64_t max_column_elements = (uint64_t(1) << 28) - 1;" in source
    start = source.index("void dispatch_column_pass(")
    body = source[start : source.index("\n}\n", start)]
    for text in (
        "in.size() > max_column_elements || out.size() > max_column_elements",
        "outer * (two_pass ? 32 : 1) > 65535",
        "if (span > max_column_elements || in.offset() < 0 ||",
        # The stride is bounded by the elements and the launch's column axis.
        "uint64_t(reduction_stride) > max_column_elements",
        "uint64_t(steps[axis]) > max_column_elements",
        "if (column_groups > 65535) {",
    ):
        assert text in body, text
    assert "65535 ||\n      in.size() > 65535" not in body
    assert "reduction_stride > 65535" not in body
    # The launch bound is checked before the native dispatch.
    assert body.index("if (column_groups > 65535) {") < body.index(
        "dispatch_callback.load()"
    )
    assert column_reduction_layout.MAX_ELEMENTS == 2**28 - 1
    # MLX's reductions refuse row plans over 65,535 elements before a general
    # plan's dense copy, and leave column plans to their passes' bounds.
    start = source.index("void Reduce::eval_gpu(")
    body = source[start : source.index("\n}\n", start)]
    refusal = body.index("row reductions support at most 65535 elements")
    assert body.index("const bool column_plan =") < refusal
    assert refusal < body.index("in = dense_input(in);")
    assert "row and column reductions support" not in source


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
        assert total < 32 and 1 <= columns <= 65536


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

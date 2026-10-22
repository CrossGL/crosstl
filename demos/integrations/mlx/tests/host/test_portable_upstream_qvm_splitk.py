"""The unchanged upstream split-K test_qvm runner: routing, trace and package checks."""

import json
import math
from collections import Counter

import pytest

from demos.integrations.mlx.portable_host import (
    column_small_layout,
    gemm_layout,
    matmul_layout,
)
from demos.integrations.mlx.portable_host import (
    verify_upstream_quantization as quantization,
)
from demos.integrations.mlx.portable_host import verify_upstream_qvm_splitk as upstream

LOOPED = "col_reduce_looped_1_32_32_reduce_sumfloat32"
SMALL = "col_reduce_small_1_reduce_sumfloat32"
SPLITK = (
    "steel_gemm_splitk_nn_float32_float32_bm16_bn32_bk16_wm2_wn2_MN_naligned_K_taligned"
)
ACCUM = "steel_gemm_splitk_accum_float32_float32"


def trace(target="opengl"):
    items = [
        {"entry": entry, "target": target}
        for entry, count in sorted(upstream.expected_dispatches().items())
        for _ in range(count)
    ]
    return items + [
        {"entry": "all_reduce_maxfloat32", "target": target, "threads": 64},
        {"entry": "rbitsc", "target": target},
    ]


def test_products_follow_both_upstream_tests():
    rows = list(upstream.products())
    # test_qvm_splitk: 3 groups * 3 bits * 2 B, then its 1-D vector;
    # test_qvm_splitk_multi_row: 2 groups * 2 bits * 2 K * 2 M.
    assert len(rows) == 18 + 1 + 16
    batched, (vector,), multi_row = rows[:18], rows[18:19], rows[19:]
    assert all(
        row[2:6] == (1, 128, 16384, row[5])
        and row[6] == (row[5], 1, 16384)
        and row[7] == (row[5], 16384, 128)
        for row in batched
    )
    assert {row[5] for row in batched} == {1, 3}
    assert vector == (32, 8, 1, 2048, 2048, 1, (2048,), (2048, 2048))
    assert {(row[2], row[4]) for row in multi_row} == {
        (m, k) for m in (2, 3) for k in (2048, 4096)
    }
    assert all(
        row[6] == (row[2], row[4]) and row[7] == (row[4], 128) for row in multi_row
    )


def test_reference_operands_expand_a_vector_to_one_row():
    assert upstream.reference_operands((2048,), (2048, 2048)) == (
        (1, 2048),
        [2048, 1],
        (2048, 2048),
        [2048, 1],
    )
    assert upstream.reference_operands((3, 1, 64), (3, 64, 32)) == (
        (3, 1, 64),
        [64, 64, 1],
        (3, 64, 32),
        [64 * 32, 32, 1],
    )


def test_products_split_k_and_sum_their_partitions():
    routes = list(upstream.routes())
    assert Counter(route[4] for route in routes) == {"gemv": 19, "splitk": 16}
    for (group, bits, rows, n, k, *_), route in zip(upstream.products(), routes):
        quantized, (column, variant), family, matmul = route[2:]
        split = matmul_layout.signature(quantized)["splitK"]
        # quantized.cpp: 32 partitions above K of 8192, else 8.
        assert split == (32 if k > 8192 else 8)
        assert quantized == f"affine_qvm_split_k_float_gs_{group}_b_{bits}_spk_{split}"
        if split == 32:
            assert (column, variant) == (LOOPED, None)
        else:
            # strided_reduce_small: 32 lanes of four columns, eight rows.
            assert column == SMALL
            assert variant == column_small_layout.variant(SMALL, (32, 8, 1))
        assert (family == "splitk") == (rows > 1)
        if family == "splitk":
            assert matmul["entry"] == SPLITK and matmul["accum"] == ACCUM


def test_expected_dispatches_follow_every_upstream_product():
    counts = upstream.expected_dispatches()
    entries = upstream.package_entries()
    # backend.cpp quantizes and dequantizes w in launches of at most 65,535
    # groups; the 128 by 16,384 weights at group 32 take two each.
    launches = sum(
        -(-math.prod(w_shape) // group // 65535)
        for group, *_, w_shape in upstream.products()
    )
    assert launches == 52
    for operation in ("quantize", "dequantize"):
        assert (
            sum(n for e, n in counts.items() if e.startswith(f"affine_{operation}_"))
            == launches
        )
    # Product, partition sum and reference per product, plus 16 split-K
    # accumulations.
    assert sum(counts.values()) == 2 * launches + 35 * 3 + 16
    assert counts[LOOPED] == 18 and counts[SMALL] == 17
    assert counts[SPLITK] == counts[ACCUM] == 16
    assert entries["small"] == [column_small_layout.variant(SMALL, (32, 8, 1))]
    assert entries["gemm"] == []
    assert entries["splitk"] == sorted([SPLITK, ACCUM])
    assert len(entries["gemv"]) == 2
    # The 1-D vector's 8-partition kernel is also a multi-row one.
    assert len(entries["quantized"]) == 9 + 4
    assert "affine_qvm_split_k_float_gs_32_b_8_spk_8" in entries["quantized"]
    assert upstream.COLUMN_SUMS == (LOOPED,)
    for group, bits, rows, n, k, batches, x_shape, w_shape in upstream.products():
        family, _ = gemm_layout.matmul_route(
            "float32", *upstream.reference_operands(x_shape, w_shape)
        )
        assert family == ("splitk" if rows > 1 else "gemv")


@pytest.mark.parametrize("target", ["metal", "opengl", "directx"])
@pytest.mark.parametrize(
    "fault",
    [None, "missing", "extra", "target", "column", "partitions", "maximum"],
)
def test_trace_requires_every_translated_product_and_sum(target, fault):
    items = trace(target)
    count = len(items)

    def first(entry):
        return next(i for i, item in enumerate(items) if item["entry"] == entry)

    if fault == "missing":
        items.pop(first(LOOPED))
        count -= 1
    elif fault == "extra":
        items.append(dict(items[first(SMALL)]))
        count += 1
    elif fault == "target":
        items[0]["target"] = "other"
    elif fault == "column":
        # A two-pass sum in place of the looped one.
        items[first(LOOPED)]["entry"] = LOOPED.replace("_looped_", "_2pass_")
    elif fault == "partitions":
        index = next(i for i, item in enumerate(items) if "_spk_32" in item["entry"])
        items[index]["entry"] = items[index]["entry"].replace("_spk_32", "_spk_8")
    elif fault == "maximum":
        items = [item for item in items if item["entry"] != "all_reduce_maxfloat32"]
        count = len(items)
    if fault:
        with pytest.raises(RuntimeError, match="test_qvm_splitk"):
            upstream.validate_trace(items, target, count)
    else:
        upstream.validate_trace(items, target, count)


def test_prepare_translates_every_family_and_both_reduction_kinds(
    tmp_path, monkeypatch
):
    from demos.integrations.mlx.portable_host import (
        column_small_packages,
        gemm_packages,
        gemm_splitk_packages,
        gemv_packages,
        matmul_packages,
        reduction_packages,
    )

    packages = tmp_path / "packages"
    packages.mkdir()
    (packages / "index.json").write_text(json.dumps({"target": "directx"}))
    calls = []

    def build(name):
        def run(root, directory, target, entries, jobs):
            calls.append((name, target, list(entries), jobs))
            return [{"entry": entry} for entry in entries]

        return run

    for name, module in (
        ("quantized", matmul_packages),
        ("gemv", gemv_packages),
        ("gemm", gemm_packages),
        ("splitk", gemm_splitk_packages),
        ("small", column_small_packages),
    ):
        monkeypatch.setattr(module, "build_entries", build(name))
    reductions = []
    monkeypatch.setattr(
        reduction_packages,
        "build_packages",
        lambda root, output, target, **options: reductions.append(
            (output, target, options)
        ),
    )
    upstream.prepare_packages(tmp_path, packages, tmp_path / "reductions", 3)
    entries = upstream.package_entries()
    # The reference products never take the steel GEMM route, so it is skipped.
    assert calls == [
        (name, "directx", entries[name], 3)
        for name in upstream.FAMILIES
        if name != "gemm"
    ]
    assert reductions == [
        (
            tmp_path / "reductions",
            "directx",
            {"family": "all", "entries": ("all_reduce_maxfloat32",), "on_demand": True},
        ),
        (
            tmp_path / "reductions-column",
            "directx",
            {"family": "column", "entries": (LOOPED,), "jobs": 3},
        ),
    ]


def test_results_require_both_upstream_tests_to_run():
    record = {
        "mode": "native",
        "tests": list(upstream.UPSTREAM_TESTS),
        "testsRun": 2,
        "success": True,
        "skipped": [],
        "errors": [],
        "failures": [],
        "nativeDispatches": 191,
    }
    assert len(upstream.UPSTREAM_TESTS) == 2
    quantization.validate_result(record, "native", upstream.UPSTREAM_TESTS)
    with pytest.raises(RuntimeError, match="did not pass"):
        quantization.validate_result(
            {**record, "testsRun": 1}, "native", upstream.UPSTREAM_TESTS
        )

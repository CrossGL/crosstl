"""The unchanged upstream quantized-shape tests runner: routing, trace and packages."""

import json
from collections import Counter

import pytest

from demos.integrations.mlx.portable_host import (
    column_small_layout,
    matmul_layout,
    qmm_layout,
)
from demos.integrations.mlx.portable_host import verify_upstream_qmm_shapes as upstream

SMALL = "col_reduce_small_1_reduce_sumfloat32"
SPLIT = "affine_qmm_t_splitk_float_gs_64_b_4_alN_false"
QMM_N = "affine_qmm_n_float_gs_64_b_4_batch_0"
QMM_N_BATCHED = "affine_qmm_n_float_gs_64_b_4_batch_1"
GEMV = "gemv_float32_bm1_bn8_sm1_sn32_tm4_tn4_nc0_axpby0"
ACCUM = "steel_gemm_splitk_accum_float32_float32"
QMV_FAST = "affine_qmv_fast_float_gs_64_b_4_batch_0"


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


def test_products_follow_the_checked_upstream_tests():
    rows = list(upstream.products())
    # test_qmm_shapes 4, test_small_matrix 3 weights by 5, test_non_multiples 9.
    assert len(rows) == 4 + 15 + 9 == 28
    assert len(upstream.WEIGHTS) == 8
    assert Counter(transpose for *_, transpose in rows) == {True: 16, False: 12}
    # Every product contracts x's last axis with the weights' quantized axis
    # or their other one.
    for w_shape, x_shape, transpose in rows:
        assert x_shape[-1] == (w_shape[-1] if transpose else w_shape[-2])


def test_quantized_operands_fold_or_broadcast_like_quantized_matmul():
    # A row-contiguous x against 2-D weights folds every row into M.
    assert upstream.quantized_operands((32, 256), (2, 1, 7, 256), True) == (
        14,
        32,
        256,
        1,
    )
    # Batched weights keep x's matrices; a single batch broadcasts.
    assert upstream.quantized_operands((1, 8, 256), (3, 10, 8), False) == (
        10,
        256,
        8,
        3,
    )
    assert upstream.quantized_operands((3, 8, 256), (3, 4, 256), True) == (4, 8, 256, 3)
    with pytest.raises(ValueError, match="shapes differ"):
        upstream.quantized_operands((8, 256), (3, 8), True)


def test_products_take_every_quantized_route():
    routes = [quantized for quantized, _, _ in upstream.routes()]
    kinds = Counter((family, entry.split("_float")[0]) for family, entry, *_ in routes)
    assert kinds == {
        ("quantized", "affine_qmv"): 14,
        ("quantized", "affine_qvm"): 6,
        ("qmm", "affine_qmm_n"): 6,
        ("qmm", "affine_qmm_t_splitk"): 2,
    }
    for (w_shape, x_shape, transpose), (family, entry, column, variant) in zip(
        upstream.products(), routes
    ):
        m, n, k, batches = upstream.quantized_operands(w_shape, x_shape, transpose)
        limit = matmul_layout.vector_limit(k, n, transpose)
        assert (family == "qmm") == (m >= limit)
        if family == "qmm":
            assert qmm_layout.transposed(entry) == transpose
            assert entry == qmm_layout.select(
                "float32", 64, 4, m, n, k, batches, transpose
            )
        if entry == SPLIT:
            # 30 by 8 and 129 by 99 outputs: four partitions of K = 256, so a
            # small column sum of 32 lanes over four rows.
            assert (m, n, k) in {(30, 8, 256), (129, 99, 256)}
            assert qmm_layout.split_k(m, n, k, 64) == 4
            assert column == SMALL
            assert variant == column_small_layout.variant(SMALL, (32, 4, 1))
        else:
            assert column is None and variant is None


def test_unchecked_product_takes_the_fast_vector_route():
    # test_throw: 10 rows of K = 512 against 32 transposed outputs stay below
    # the vector limit, and K and N meet qmv_fast's alignment.
    assert upstream.UNCHECKED == (((32, 512), (10, 512), True),)
    assert upstream.quantized_operands((32, 512), (10, 512), True) == (10, 32, 512, 1)
    assert matmul_layout.vector_limit(512, 32, True) > 10
    assert list(upstream.unchecked_routes()) == [("quantized", QMV_FAST, None, None)]
    assert QMV_FAST not in {entry for (_, entry, *_), _, _ in upstream.routes()}


def test_reference_operands_are_matmuls_views_of_dequantized_weights():
    # w_hat.T: a transposed view, x folded into its rows.
    assert upstream.reference_operands((32, 256), (2, 1, 7, 256), True) == (
        (14, 256),
        [256, 1],
        (256, 32),
        [1, 256],
    )
    # A single weight batch broadcasts with a zero stride.
    assert upstream.reference_operands((1, 8, 256), (3, 1, 256), True) == (
        (3, 1, 256),
        [256, 256, 1],
        (3, 256, 8),
        [0, 1, 256],
    )
    assert upstream.reference_operands((3, 8, 256), (3, 10, 8), False) == (
        (3, 10, 8),
        [80, 8, 1],
        (3, 8, 256),
        [2048, 256, 1],
    )


def test_expected_dispatches_follow_every_upstream_product():
    counts = upstream.expected_dispatches()
    entries = upstream.package_entries()
    # Eight weights quantized and dequantized once each, in one launch; each
    # product, its reference, two partition sums and the split-K accumulations;
    # test_throw's weights quantized once and its one unchecked product.
    quantize = "affine_quantize_float_gs_64_b_4"
    assert counts[quantize] == 8 + 1
    assert counts["affine_dequantize_float_gs_64_b_4"] == 8
    accumulations = counts[ACCUM]
    assert sum(counts.values()) == 2 * 8 + 2 * 28 + 2 + accumulations + 1 + 1
    assert accumulations == 13
    assert counts[QMV_FAST] == 1 and QMV_FAST in entries["quantized"]
    assert counts[SPLIT] == counts[SMALL] == 2
    assert counts[QMM_N] == 4 and counts[QMM_N_BATCHED] == 2
    assert entries["qmm"] == sorted([SPLIT, QMM_N, QMM_N_BATCHED])
    assert entries["small"] == [column_small_layout.variant(SMALL, (32, 4, 1))]
    assert ACCUM in entries["splitk"] and GEMV in entries["gemv"]
    assert upstream.COLUMN_SUMS == ()


@pytest.mark.parametrize("target", ["metal", "opengl", "directx"])
@pytest.mark.parametrize(
    "fault",
    [None, "missing", "extra", "target", "route", "sum", "unchecked", "maximum"],
)
def test_trace_requires_every_translated_product(target, fault):
    items = trace(target)
    count = len(items)

    def first(entry):
        return next(i for i, item in enumerate(items) if item["entry"] == entry)

    if fault == "missing":
        items.pop()
    elif fault == "extra":
        items.append(items[0])
    elif fault == "target":
        items[0]["target"] = "other"
    elif fault == "route":
        # The unsplit kernel in place of the split-K one.
        items[first(SPLIT)]["entry"] = "affine_qmm_t_float_gs_64_b_4_alN_false_batch_0"
    elif fault == "sum":
        items.pop(first(SMALL))
        count -= 1
    elif fault == "unchecked":
        items.pop(first(QMV_FAST))
        count -= 1
    elif fault == "maximum":
        items = [item for item in items if item["entry"] != "all_reduce_maxfloat32"]
        count = len(items)
    if fault:
        with pytest.raises(RuntimeError, match="test_qmm_shapes"):
            upstream.validate_trace(items, target, count)
    else:
        upstream.validate_trace(items, target, count)


def test_prepare_translates_every_family_and_indexes_the_maximum(tmp_path, monkeypatch):
    from demos.integrations.mlx.portable_host import (
        column_small_packages,
        gemm_packages,
        gemm_splitk_packages,
        gemv_packages,
        matmul_packages,
        qmm_packages,
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
        ("qmm", qmm_packages),
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
        lambda root, output, target, **options: reductions.append((target, options)),
    )
    upstream.prepare_packages(tmp_path, packages, tmp_path / "reductions", 3)
    entries = upstream.package_entries()
    assert calls == [(name, "directx", entries[name], 3) for name in upstream.FAMILIES]
    # The maximum alone: no product here sums its partitions in a looped pass.
    assert reductions == [
        (
            "directx",
            {"family": "all", "entries": ("all_reduce_maxfloat32",), "on_demand": True},
        )
    ]

"""The unchanged upstream ``test_qmm`` runner: float16 routing, trace and packages."""

import json
from collections import Counter

import pytest

from demos.integrations.mlx.portable_host import (
    column_small_layout,
    matmul_layout,
    qmm_layout,
    quantization_layout,
)
from demos.integrations.mlx.portable_host import verify_upstream_qmm as upstream
from demos.integrations.mlx.portable_host import verify_upstream_qmm_shapes as shapes

SMALL = "col_reduce_small_1_reduce_sumfloat16"
SPLIT = "affine_qmm_t_splitk_float16_t_gs_64_b_4_alN_true"
UNSPLIT = "affine_qmm_t_float16_t_gs_128_b_4_alN_true_batch_0"
MAXIMUM = "all_reduce_maxfloat16"
ACCUM = "steel_gemm_splitk_accum_float16_float32"


def trace(target="opengl"):
    items = [
        {"entry": entry, "target": target}
        for entry, count in sorted(upstream.expected_dispatches().items())
        for _ in range(count)
    ]
    return items + [
        {"entry": MAXIMUM, "target": target, "threads": 64},
        {"entry": "v_copyfloat32float16", "target": target},
        {"entry": "rbitsc", "target": target},
    ]


def test_products_follow_the_upstream_test():
    rows = list(upstream.products())
    # 3 group sizes, 3 widths, 4 row counts, 2 by 2 shapes, both layouts.
    assert len(rows) == 3 * 3 * 4 * 2 * 2 * 2 == 288
    assert rows[0] == (128, 2, 8, 128, 128, True)
    assert rows[-1] == (32, 8, 64, 256, 256, False)
    assert upstream.weight_shape(128, 256, True) == (128, 256)
    assert upstream.weight_shape(128, 256, False) == (256, 128)
    assert upstream.UPSTREAM_TESTS == ("test_quantized.TestQuantized.test_qmm",)


def test_products_take_every_float16_route():
    kinds, families = Counter(), Counter()
    for product, (stem, quantized, family, route) in zip(
        upstream.products(), upstream.routes()
    ):
        group, bits, m, n, k, transpose = product
        kind, entry, column, variant = quantized
        assert stem == (f"float16_t_gs_{group}_b_{bits}", n * k, group)
        assert f"_float16_t_gs_{group}_b_{bits}" in entry
        assert (kind == "qmm") == (m >= matmul_layout.vector_limit(k, n, transpose))
        if kind == "qmm":
            assert qmm_layout.transposed(entry) == transpose
        kinds[(kind, entry.split("_float16")[0])] += 1
        families[family] += 1
        if "splitk" in entry:
            # Fewer than 32 partitions: the element-type small pass.
            split = qmm_layout.split_k(m, n, k, group)
            assert 1 < split < 32
            assert column == SMALL
            workgroup = (*column_small_layout.geometry(m * n, split), 1)
            assert variant == column_small_layout.variant(SMALL, workgroup)
        else:
            assert column is None and variant is None
        # The reference multiplies the dequantized float16 weights.
        prefix = "steel_gemm_splitk_" if family == "splitk" else "steel_gemm_fused_"
        assert route["entry"].startswith(prefix + ("nt_" if transpose else "nn_"))
        assert "_float16_" in route["entry"]
    assert kinds == {
        ("qmm", "affine_qmm_n"): 144,
        ("qmm", "affine_qmm_t"): 18,
        ("qmm", "affine_qmm_t_splitk"): 90,
        ("quantized", "affine_qmv"): 12,
        ("quantized", "affine_qmv_fast"): 6,
        ("quantized", "affine_qmv_quad"): 18,
    }
    assert families == {"gemm": 72, "splitk": 216}


def test_16_bit_split_k_products_refuse_looped_partition_sums():
    # 128 partitions of K = 8192: float32 sums them in the looped pass.
    m, n, k = 32, 32, 8192
    assert qmm_layout.partition_sum(m, n, qmm_layout.split_k(m, n, k, 64)) == "looped"
    assert shapes.quantized_route(m, n, k, 1, True)[2] == (
        "col_reduce_looped_1_32_32_reduce_sumfloat32"
    )
    with pytest.raises(ValueError, match="no looped partition sum"):
        shapes.quantized_route(m, n, k, 1, True, dtype="float16")


def test_expected_dispatches_follow_every_upstream_product():
    counts = upstream.expected_dispatches()
    entries = upstream.package_entries()
    # Every product's weights fit one quantization launch.
    for group in upstream.GROUPS:
        for bits in upstream.BITS:
            stem = f"float16_t_gs_{group}_b_{bits}"
            assert counts[f"affine_quantize_{stem}"] == 32
            assert counts[f"affine_dequantize_{stem}"] == 32
            assert all(
                quantization_layout.launches(n * k, group) == 1
                for n in upstream.COLUMNS
                for k in upstream.DEPTHS
            )
    splits = sum(count for entry, count in counts.items() if "qmm_t_splitk" in entry)
    accumulations = counts[ACCUM]
    assert splits == counts[SMALL] == 90
    assert accumulations == 216
    assert sum(counts.values()) == 2 * 288 + 2 * 288 + splits + accumulations
    # A group of 128 leaves K = 128 one partition, so it stays unsplit.
    assert counts[SPLIT] == 12 and counts[UNSPLIT] == 6
    assert ACCUM in entries["splitk"]
    assert all(SMALL + column_small_layout.SEPARATOR in key for key in entries["small"])
    assert set(entries) == set(upstream.FAMILIES)
    assert not hasattr(upstream, "COLUMN_SUMS")
    for family, names in entries.items():
        assert names == sorted(set(names)) and names, family


@pytest.mark.parametrize("target", ["metal", "opengl", "directx"])
@pytest.mark.parametrize(
    "fault",
    [None, "missing", "extra", "target", "route", "sum", "maximum", "float32"],
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
        items[first(SPLIT)]["entry"] = SPLIT.replace("_splitk", "") + "_batch_0"
    elif fault == "sum":
        items.pop(first(SMALL))
        count -= 1
    elif fault == "maximum":
        items = [item for item in items if item["entry"] != MAXIMUM]
        count = len(items)
    elif fault == "float32":
        # A float32 error bound is not the float16 one the GPU device computes.
        items[first(MAXIMUM)]["entry"] = "all_reduce_maxfloat32"
    if fault:
        with pytest.raises(RuntimeError, match="test_qmm"):
            upstream.validate_trace(items, target, count)
    else:
        upstream.validate_trace(items, target, count)


def test_prepare_translates_every_family_the_maximum_and_operators(
    tmp_path, monkeypatch
):
    from demos.integrations.mlx.portable_host import (
        column_small_packages,
        gemm_packages,
        gemm_splitk_packages,
        matmul_packages,
    )
    from demos.integrations.mlx.portable_host import packages as operator_packages
    from demos.integrations.mlx.portable_host import qmm_packages, reduction_packages

    packages = tmp_path / "packages"
    packages.mkdir()
    (packages / "index.json").write_text(json.dumps({"target": "metal"}))
    calls = []

    def build(name):
        def run(root, directory, target, entries, jobs):
            calls.append((name, target, list(entries), jobs))
            return [{"entry": entry} for entry in entries]

        return run

    for name, module in (
        ("quantized", matmul_packages),
        ("qmm", qmm_packages),
        ("gemm", gemm_packages),
        ("splitk", gemm_splitk_packages),
        ("small", column_small_packages),
    ):
        monkeypatch.setattr(module, "build_entries", build(name))
    reductions, operators = [], []
    monkeypatch.setattr(
        reduction_packages,
        "build_packages",
        lambda root, output, target, **options: reductions.append(
            (output, target, options)
        ),
    )
    monkeypatch.setattr(
        operator_packages,
        "build_packages",
        lambda root, output, target, **options: operators.append(
            (output, target, options["family"])
        ),
    )
    upstream.prepare_packages(tmp_path, packages, tmp_path / "reductions", 3)
    entries = upstream.package_entries()
    assert calls == [(name, "metal", entries[name], 3) for name in upstream.FAMILIES]
    assert reductions == [
        (
            tmp_path / "reductions-half",
            "metal",
            {"family": "half", "entries": (MAXIMUM,), "on_demand": True},
        )
    ]
    assert operators == [
        (tmp_path / "half-packages", "metal", "half"),
        (tmp_path / "half-arithmetic-packages", "metal", "half-arithmetic"),
    ]

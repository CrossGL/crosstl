"""The unchanged upstream test_qmv_wide runner: routes, trace, packages."""

import json
from collections import Counter

import pytest

from demos.integrations.mlx.portable_host import (
    fp_matmul_layout,
    gemm_layout,
    matmul_layout,
)
from demos.integrations.mlx.portable_host import verify_upstream_qmv_wide as upstream
from demos.integrations.mlx.portable_host.verify_upstream_qmv import (
    reference_operands,
)

# quantized.cpp qmv_wide's tiling: the fewest tiles of at most five vectors,
# then the smallest tile that fills them.
TILE_VECTORS = {2: 2, 3: 3, 4: 4, 5: 5, 6: 3, 7: 4, 9: 5}


def trace(target="opengl"):
    items = [
        {"entry": entry, "target": target}
        for entry, count in sorted(upstream.expected_dispatches().items())
        for _ in range(count)
    ]
    return items + [
        {"entry": "vv_Subtractfloat32", "target": target},
        {"entry": "all_reduce_maxfloat32", "target": target, "threads": 64},
        {"entry": "rbitsc", "target": target},
    ]


def test_products_follow_the_upstream_sweeps_in_order():
    products = list(upstream.products())
    # 18 affine settings and 6 block-scaled ones, each over 7 x 2 x 2 shapes,
    # then 4 tiny shapes in 3 settings.
    assert len(products) == 18 * 2 * 28 + 3 * 2 * 28 + 12
    assert products[:3] == [
        ("affine", 32, 2, 2, 256, 128, 1),
        ("affine", 32, 2, 2, 256, 128, 3),
        ("affine", 32, 2, 2, 67, 128, 1),
    ]
    assert products[28] == ("affine", 32, 2, 2, 256, 512, 1)
    assert products[1008] == ("mxfp4", 32, 4, 2, 256, 128, 1)
    assert products[1176:1179] == [
        ("affine", 32, 4, 2, 10, 32, 1),
        ("affine", 32, 6, 2, 10, 32, 1),
        ("nvfp4", 16, 4, 2, 10, 32, 1),
    ]
    assert {product[0] for product in products[1008:1176]} == {
        "mxfp4",
        "nvfp4",
        "mxfp8",
    }
    # Every product stays below the transposed vector limit.
    assert all(
        m < matmul_layout.vector_limit(k, n, True) for *_, m, n, k, _ in products
    )


def test_affine_products_keep_off_qmv_wide_without_a_gpu_generation():
    seen = Counter()
    for (mode, group, bits, m, n, k, b), (family, quantized, *_) in zip(
        upstream.products(), upstream.routes()
    ):
        if mode != "affine":
            continue
        assert family == "affine" and "_qmv_wide_" not in quantized
        info = matmul_layout.signature(quantized)
        assert (info["bits"], info["groupSize"], info["batched"]) == (
            bits,
            group,
            b > 1,
        )
        if k in (64, 128) and bits in (2, 4, 8):
            kernel = "qmv_quad"
        elif n % 8 == 0 and k % matmul_layout.fast_alignment(bits) == 0:
            kernel = "qmv_fast"
        else:
            kernel = "qmv"
        assert info["kernel"] == kernel
        seen[kernel] += 1
    assert set(seen) == {"qmv_quad", "qmv_fast", "qmv"}


def test_block_scaled_products_reach_qmv_quad_or_qmv_wide():
    for (mode, _, _, m, n, k, b), (family, quantized, *_) in zip(
        upstream.products(), upstream.routes()
    ):
        if mode == "affine":
            continue
        assert family == "fp"
        info = fp_matmul_layout.signature(quantized)
        assert info["batched"] == (b > 1)
        if k in (64, 128):
            assert info["kernel"] == "qmv_quad"
        else:
            # Without a global scale every row count of two or more streams
            # through one tile of vectors per threadgroup row.
            assert info["kernel"] == "qmv_wide"
            assert info["vectors"] == TILE_VECTORS[m]


def test_references_take_steel_gemm_or_split_k():
    families = Counter()
    for (_, _, _, m, n, k, b), (_, _, family, route) in zip(
        upstream.products(), upstream.routes()
    ):
        x_shape = (m, k) if b == 1 else (b, m, k)
        assert (family, route) == gemm_layout.matmul_route(
            "float32", *reference_operands(n, k, 0 if b == 1 else b, x_shape)
        )
        # Two or more rows never take gemv; split-K needs a single matrix.
        assert family in ("gemm", "splitk")
        assert family == "gemm" or b == 1
        families[family] += 1
    assert families == {"gemm": 747, "splitk": 441}


def test_expected_dispatches_quantize_and_dequantize_every_product():
    counts = upstream.expected_dispatches()
    # Each weight matrix holds at most 3 x 256 x 512 values: one launch each.
    for group in (32, 64, 128):
        for bits in (2, 3, 5, 8):
            for operation in ("quantize", "dequantize"):
                assert counts[f"affine_{operation}_float_gs_{group}_b_{bits}"] == 56
    # The tiny sweep adds four products for each affine group-32 setting.
    for bits in (4, 6):
        assert counts[f"affine_quantize_float_gs_32_b_{bits}"] == 60
        assert counts[f"affine_quantize_float_gs_64_b_{bits}"] == 56
    assert counts["nvfp4_quantize_float_gs_16_b_4_hgs_false"] == 60
    assert counts["nvfp4_dequantize_bfloat16_t_gs_16_b_4_hgs_false"] == 60
    for mode, (group, bits) in (("mxfp4", (32, 4)), ("mxfp8", (32, 8))):
        assert counts[f"{mode}_quantize_float_gs_{group}_b_{bits}_hgs_false"] == 56
        assert (
            counts[f"{mode}_dequantize_bfloat16_t_gs_{group}_b_{bits}_hgs_false"] == 56
        )
    products = len(list(upstream.products()))
    # Quantize, dequantize, the product, its reference and 441 accumulations.
    assert sum(counts.values()) == 4 * products + 441
    assert all(entry.startswith(upstream.PREFIXES) for entry in counts)


def test_package_entries_split_affine_and_block_scaled_products():
    entries = upstream.package_entries()
    assert set(entries) == set(upstream.FAMILIES)
    assert entries["gemv"] == []
    assert len(entries["affine"]) == 91 and len(entries["fp"]) == 31
    assert all(name.startswith("affine_qmv") for name in entries["affine"])
    assert {fp_matmul_layout.signature(name)["kernel"] for name in entries["fp"]} == {
        "qmv_quad",
        "qmv_wide",
    }
    assert entries["splitk"] == sorted(
        [
            "steel_gemm_splitk_accum_float32_float32",
            "steel_gemm_splitk_nt_float32_float32_bm16_bn32_bk16_wm2_wn2"
            "_MN_naligned_K_taligned",
        ]
    )
    assert len(entries["gemm"]) == 2
    assert all(name.startswith("steel_gemm_fused_nt_") for name in entries["gemm"])


def test_settings_reject_a_block_scaled_mode_off_its_pinned_parameters():
    with pytest.raises(ValueError, match="pinned group size and bits"):
        upstream.quantized_entry("nvfp4", 32, 4, 2, 10, 32, 1)


@pytest.mark.parametrize("target", ["metal", "opengl", "directx"])
@pytest.mark.parametrize(
    "fault",
    [None, "missing", "extra", "target", "wide", "affine-wide", "slab", "reduction"],
)
def test_trace_requires_every_translated_product(target, fault):
    items = trace(target)
    count = len(items)

    def first(text):
        return next(i for i, item in enumerate(items) if text in item["entry"])

    if fault == "missing":
        items.pop(0)
        count -= 1
    elif fault == "extra":
        items.append(dict(items[0]))
        count += 1
    elif fault == "target":
        items[0]["target"] = "other"
    elif fault == "wide":
        # A wide product served by a tile of another size.
        index = first("_qmv_wide_float_gs_16_b_4_nv_3_")
        items[index]["entry"] = items[index]["entry"].replace("_nv_3_", "_nv_5_")
    elif fault == "affine-wide":
        # An affine product routed as on a generation-15 device.
        index = first("affine_qmv_float_gs_32_b_3_batch_0")
        items[index]["entry"] = "affine_qmv_wide_float_gs_32_b_3_nv_2_kl_8_batch_0"
    elif fault == "slab":
        index = first("_batch_1")
        items[index]["entry"] = items[index]["entry"].replace("_batch_1", "_batch_0")
    elif fault == "reduction":
        items = [item for item in items if item["entry"] != "all_reduce_maxfloat32"]
        count = len(items)
    if fault:
        with pytest.raises(RuntimeError, match="test_qmv_wide"):
            upstream.validate_trace(items, target, count)
    else:
        upstream.validate_trace(items, target, count)


def test_prepare_translates_both_quantized_families_and_references(
    tmp_path, monkeypatch
):
    from demos.integrations.mlx.portable_host import (
        fp_matmul_packages,
        gemm_packages,
        gemm_splitk_packages,
        gemv_packages,
        matmul_packages,
    )
    from demos.integrations.mlx.portable_host import packages as operator_packages
    from demos.integrations.mlx.portable_host import reduction_packages

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
        ("affine", matmul_packages),
        ("fp", fp_matmul_packages),
        ("gemv", gemv_packages),
        ("gemm", gemm_packages),
        ("splitk", gemm_splitk_packages),
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
    operators = []
    monkeypatch.setattr(
        operator_packages,
        "build_packages",
        lambda root, output, target, **options: operators.append(
            (output, target, options["family"])
        ),
    )
    upstream.prepare_packages(tmp_path, packages, tmp_path / "reductions", 3)
    entries = upstream.package_entries()
    # The empty gemv family translates nothing.
    assert calls == [
        (family, "metal", entries[family], 3)
        for family in ("affine", "fp", "gemm", "splitk")
    ]
    assert reductions == [
        (
            tmp_path / "reductions",
            "metal",
            {"family": "all", "entries": ("all_reduce_maxfloat32",), "on_demand": True},
        )
    ]
    assert operators == [(tmp_path / "bfloat-packages", "metal", "bfloat")]


def test_package_verification_entries_include_the_runners_block_scaled_products():
    from demos.integrations.mlx.portable_host import fp_matmul_packages

    entries = fp_matmul_packages.verification_entries()
    assert entries == sorted(set(entries))
    assert set(upstream.package_entries()["fp"]) <= set(entries)

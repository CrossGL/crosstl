"""The unchanged upstream test_qmv_small_non_multiples runner: routes, trace, packages."""

import json

import pytest

from demos.integrations.mlx.portable_host import (
    fp_matmul_layout,
    gemm_layout,
    matmul_layout,
)
from demos.integrations.mlx.portable_host import (
    verify_upstream_qmv_small_non_multiples as upstream,
)

GEMV = "gemv_float32_bm1_bn1_sm8_sn4_tm{}_tn4_nc0_axpby0"
GEMM = (
    "steel_gemm_fused_nt_float32_float32_bm64_bn64_bk16_wm2_wn2_has_batch_n"
    "_use_out_source_n_do_axpby_n_align_M_n_align_N_n_align_K_t"
)
WIDE = "nvfp4_qmv_wide_float_gs_16_b_4_nv_{}_kl_16_batch_0"


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


def test_products_follow_the_upstream_cases_and_settings():
    products = list(upstream.products())
    assert len(products) == 12
    assert products[:3] == [
        (1, 32, 3, 4, 32, "affine"),
        (1, 32, 3, 6, 32, "affine"),
        (1, 32, 3, 4, 16, "nvfp4"),
    ]
    assert [product[:3] for product in products[::3]] == [
        (1, 32, 3),
        (2, 32, 10),
        (1, 32, 5),
        (4, 32, 7),
    ]
    # The nvfp4 setting is the mode's pinned group size and bits.
    assert fp_matmul_layout.MODES["nvfp4"] == (16, 4)


def test_quantized_products_take_the_general_vector_kernels():
    for (m, k, n, bits, group, mode), (family, quantized, *_) in zip(
        upstream.products(), upstream.routes()
    ):
        # No output width is a multiple of 8 and K is neither 64 nor 128.
        assert n % 8 and k not in (64, 128)
        if mode == "affine":
            assert family == "affine"
            # Without an Apple GPU generation affine qmv_wide stays unselected.
            assert quantized == f"affine_qmv_float_gs_32_b_{bits}_batch_0"
            info = matmul_layout.signature(quantized)
            assert (info["kernel"], info["bits"], info["groupSize"]) == (
                "qmv",
                bits,
                32,
            )
        elif m == 1:
            assert (family, quantized) == ("fp", "nvfp4_qmv_float_gs_16_b_4_batch_0")
        else:
            # One threadgroup streams all of the two or four vectors.
            assert (family, quantized) == ("fp", WIDE.format(m))
            info = fp_matmul_layout.signature(quantized)
            assert info["kernel"] == "qmv_wide" and info["vectors"] == m


def test_references_take_gemv_for_one_row_and_steel_gemm_otherwise():
    for (m, k, n, *_), (_, _, family, route) in zip(
        upstream.products(), upstream.routes()
    ):
        assert (family, route) == gemm_layout.matmul_route(
            "float32", *upstream.reference_operands(n, k, 0, (m, k))
        )
        if m == 1:
            assert family == "gemv"
            assert route["entry"] == GEMV.format(1 if n < 4 else 4)
        else:
            assert family == "gemm" and route["entry"] == GEMM


def test_expected_dispatches_quantize_and_dequantize_every_product():
    counts = upstream.expected_dispatches()
    # Every weight matrix holds at most ten groups: one launch each.
    for bits in (4, 6):
        for operation in ("quantize", "dequantize"):
            assert counts[f"affine_{operation}_float_gs_32_b_{bits}"] == 4
        assert counts[f"affine_qmv_float_gs_32_b_{bits}_batch_0"] == 4
    assert counts["nvfp4_quantize_float_gs_16_b_4_hgs_false"] == 4
    assert counts["nvfp4_dequantize_bfloat16_t_gs_16_b_4_hgs_false"] == 4
    assert counts["nvfp4_qmv_float_gs_16_b_4_batch_0"] == 2
    assert counts[WIDE.format(2)] == counts[WIDE.format(4)] == 1
    assert counts[GEMV.format(1)] == counts[GEMV.format(4)] == 3
    assert counts[GEMM] == 6
    assert sum(counts.values()) == 3 * 4 * 2 + 12 + 12
    assert all(entry.startswith(upstream.PREFIXES) for entry in counts)


def test_package_entries_split_affine_and_block_scaled_products():
    entries = upstream.package_entries()
    assert entries == {
        "affine": [f"affine_qmv_float_gs_32_b_{bits}_batch_0" for bits in (4, 6)],
        "fp": sorted(
            [
                "nvfp4_qmv_float_gs_16_b_4_batch_0",
                WIDE.format(2),
                WIDE.format(4),
            ]
        ),
        "gemv": [GEMV.format(1), GEMV.format(4)],
        "gemm": [GEMM],
        "splitk": [],
    }


def test_settings_reject_a_block_scaled_mode_off_its_pinned_parameters():
    with pytest.raises(ValueError, match="pinned group size and bits"):
        upstream.quantized_entry(1, 32, 3, 4, 32, "nvfp4")


@pytest.mark.parametrize("target", ["metal", "opengl", "directx"])
@pytest.mark.parametrize(
    "fault",
    [None, "missing", "extra", "target", "wide", "bits", "dequantize", "reduction"],
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
        # A wide product served by the general vector kernel instead.
        index = first("_qmv_wide_")
        items[index]["entry"] = "nvfp4_qmv_float_gs_16_b_4_batch_0"
    elif fault == "bits":
        index = first("affine_qmv_float_gs_32_b_6")
        items[index]["entry"] = items[index]["entry"].replace("_b_6_", "_b_4_")
    elif fault == "dequantize":
        index = first("nvfp4_dequantize_bfloat16_t")
        items[index]["entry"] = items[index]["entry"].replace("bfloat16_t", "float")
    elif fault == "reduction":
        items = [item for item in items if item["entry"] != "all_reduce_maxfloat32"]
        count = len(items)
    if fault:
        with pytest.raises(RuntimeError, match="test_qmv_small_non_multiples"):
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
    (packages / "index.json").write_text(json.dumps({"target": "directx"}))
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
    upstream.prepare_packages(tmp_path, packages, tmp_path / "reductions", 2)
    entries = upstream.package_entries()
    # The empty split-K family translates nothing.
    assert calls == [
        (family, "directx", entries[family], 2)
        for family in ("affine", "fp", "gemv", "gemm")
    ]
    assert reductions == [
        (
            tmp_path / "reductions",
            "directx",
            {"family": "all", "entries": ("all_reduce_maxfloat32",), "on_demand": True},
        )
    ]
    assert operators == [(tmp_path / "bfloat-packages", "directx", "bfloat")]


def test_package_verification_entries_include_the_runners_block_scaled_products():
    from demos.integrations.mlx.portable_host import fp_matmul_packages

    entries = fp_matmul_packages.verification_entries()
    assert entries == sorted(set(entries))
    assert set(upstream.package_entries()["fp"]) <= set(entries)

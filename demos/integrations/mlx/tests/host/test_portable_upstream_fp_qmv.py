"""The unchanged upstream test_fp_qmv runner: routing, trace and package checks."""

import json
from collections import Counter

import pytest

from demos.integrations.mlx.portable_host import fp_matmul_layout, gemm_layout
from demos.integrations.mlx.portable_host import verify_upstream_fp_qmv as upstream

SPLITK_STEM = "steel_gemm_splitk_nt_float32_float32_bm16_bn{}_bk16_wm2_wn2"
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


def test_products_follow_the_upstream_grid_and_its_closing_case():
    rows = list(upstream.products())
    # Five M, two N and four B for each of three modes, then the nvfp4 case.
    assert len(rows) == 5 * 2 * 4 * 3 + 1 == 121
    assert rows[-1] == ("nvfp4", 128, 48, 0, (1, 48), (128, 48))
    assert [row[0] for row in rows[:3]] == ["mxfp4", "nvfp4", "mxfp8"]
    for mode, m, n, b, x_shape, w_shape in rows[:-1]:
        assert x_shape == ((3, 1, n) if b == 0 else (b, 1, n))
        assert w_shape == ((m, n) if b == 0 else (b, m, n))


def test_quantized_routes_follow_dispatch_qmv():
    kernels = Counter()
    for (mode, m, n, b, x_shape, _), (_, quantized, _, _) in zip(
        upstream.products(), upstream.routes()
    ):
        info = fp_matmul_layout.signature(quantized)
        assert info["mode"] == mode and info["batched"] == (b > 1)
        kernels[info["kernel"]] += 1
        if n == 64:
            # qmv_quad takes K of 64 before qmv_wide takes the three rows.
            assert info["kernel"] == "qmv_quad" and info["D"] == 64
        elif x_shape[0] == 3 and b == 0:
            assert info["kernel"] == "qmv_wide" and info["vectors"] == 3
        elif mode == "mxfp8" and m % 8 == 0 and n == 256:
            assert info["kernel"] == "qmv_fast"
        else:
            # 4-bit qmv_fast needs K in multiples of 512.
            assert info["kernel"] == "qmv"
    assert kernels == {"qmv_quad": 60, "qmv_wide": 15, "qmv_fast": 6, "qmv": 40}


def test_references_take_matmuls_routes_without_copies():
    families = Counter(family for _, _, family, _ in upstream.routes())
    assert families == {"gemv": 91, "gemm": 18, "splitk": 12}
    for (mode, m, n, b, x_shape, _), (_, _, family, route) in zip(
        upstream.products(), upstream.routes()
    ):
        assert (family, route) == gemm_layout.matmul_route(
            "float32", *upstream.reference_operands(m, n, b, x_shape)
        )
        if family == "splitk":
            assert b == 0 and n == 256 and m in (256, 67, 5, 7)
            assert route["accum"] == ACCUM
            assert route["entry"].startswith(SPLITK_STEM.format(16 if m < 16 else 32))


def test_expected_dispatches_quantize_dequantize_and_multiply_each_product():
    counts = upstream.expected_dispatches()
    entries = upstream.package_entries()
    # nvfp4's largest weights, 8 x 512 x 256 values, hold 65,536 groups of 16:
    # one more than a launch covers, so each quantization launches twice.
    assert sum(counts.values()) == 121 * 4 + 2 + 12
    assert len(entries["quantized"]) == 17
    assert counts[ACCUM] == 12
    # Float32 weights quantize; dequantize returns its bfloat16 default.
    assert upstream.quantization_entries("nvfp4", "quantize") == (
        "nvfp4_quantize_float_gs_16_b_4_hgs_false"
    )
    assert upstream.quantization_entries("mxfp8", "dequantize") == (
        "mxfp8_dequantize_bfloat16_t_gs_32_b_8_hgs_false"
    )
    assert upstream.OPERATORS == ("bfloat",)
    for mode in upstream.MODES:
        for operation in ("quantize", "dequantize"):
            entry = upstream.quantization_entries(mode, operation)
            assert entry.endswith("_hgs_false")
            assert counts[entry] == (42 if mode == "nvfp4" else 40)
    quantized = sum(
        count for entry, count in counts.items() if entry in entries["quantized"]
    )
    assert quantized == 121
    assert all(entry.startswith(upstream.PREFIXES) for entry in counts)


@pytest.mark.parametrize("target", ["metal", "opengl", "directx"])
@pytest.mark.parametrize(
    "fault",
    [None, "missing", "extra", "target", "specialization", "quantize", "maximum"],
)
def test_trace_requires_every_translated_product(target, fault):
    items = trace(target)
    count = len(items)
    if fault == "missing":
        items.pop(0)
        count -= 1
    elif fault == "extra":
        items.append(dict(items[0]))
        count += 1
    elif fault == "target":
        items[0]["target"] = "other"
    elif fault == "specialization":
        index = next(i for i, item in enumerate(items) if "_qmv_wide_" in item["entry"])
        items[index]["entry"] = items[index]["entry"].replace("_nv_3_", "_nv_2_")
    elif fault == "quantize":
        index = next(i for i, item in enumerate(items) if "_quantize_" in item["entry"])
        items[index]["entry"] = items[index]["entry"].replace("hgs_false", "hgs_true")
    elif fault == "maximum":
        items = [item for item in items if item["entry"] != "all_reduce_maxfloat32"]
        count = len(items)
    if fault:
        with pytest.raises(RuntimeError, match="test_fp_qmv"):
            upstream.validate_trace(items, target, count)
    else:
        upstream.validate_trace(items, target, count)


def test_prepare_translates_every_family_and_indexes_the_maximum(tmp_path, monkeypatch):
    from demos.integrations.mlx.portable_host import (
        fp_matmul_packages,
        gemm_packages,
        gemm_splitk_packages,
        gemv_packages,
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
        ("quantized", fp_matmul_packages),
        ("gemv", gemv_packages),
        ("gemm", gemm_packages),
        ("splitk", gemm_splitk_packages),
    ):
        monkeypatch.setattr(module, "build_entries", build(name))
    reductions = {}
    monkeypatch.setattr(
        reduction_packages,
        "build_packages",
        lambda root, output, target, **options: reductions.update(
            target=target, options=options
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
    assert calls == [(name, "directx", entries[name], 3) for name in upstream.FAMILIES]
    assert reductions == {
        "target": "directx",
        "options": {
            "family": "all",
            "entries": ("all_reduce_maxfloat32",),
            "on_demand": True,
        },
    }
    # The bfloat16 conversions the reference's AsType launches.
    assert operators == [(tmp_path / "bfloat-packages", "directx", "bfloat")]


def test_package_verification_entries_include_the_runners_products():
    from demos.integrations.mlx.portable_host import fp_matmul_packages

    entries = fp_matmul_packages.verification_entries()
    assert entries == sorted(set(entries))
    assert set(upstream.package_entries()["quantized"]) <= set(entries)

"""The unchanged upstream test_fp_qvm runner: routing, trace and package checks."""

import json
from collections import Counter

import pytest

from demos.integrations.mlx.portable_host import fp_matmul_layout, gemm_layout
from demos.integrations.mlx.portable_host import verify_upstream_fp_qvm as upstream

GEMV = "gemv_t_float32_bm1_bn2_sm8_sn4_tm4_tn4_nc0_axpby0"
LOOPED_SUM = "col_reduce_looped_1_32_32_reduce_sumfloat32"


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


def test_products_follow_the_upstream_grid_and_its_split_case():
    rows = list(upstream.products())
    # Three M, three N and four B for each of three modes, then the split case.
    assert len(rows) == 3 * 3 * 4 * 3 + 3 == 111
    assert [row[0] for row in rows[:3]] == ["mxfp4", "nvfp4", "mxfp8"]
    assert rows[-3:] == [
        (mode, 128, 16384, 0, (1, 16384), (16384, 128)) for mode in upstream.MODES
    ]
    for mode, m, n, b, x_shape, w_shape in rows:
        assert x_shape == ((1, n) if b == 0 else (b, 1, n))
        assert w_shape == ((n, m) if b == 0 else (b, n, m))


def test_quantized_routes_follow_quantized_matmul():
    kernels = Counter()
    for (mode, m, n, b, _, _), (_, quantized, column, _, _) in zip(
        upstream.products(), upstream.routes()
    ):
        info = fp_matmul_layout.signature(quantized)
        assert info["mode"] == mode and info["dtype"] == "float32"
        kernels[info["kernel"]] += 1
        if n >= fp_matmul_layout.QVM_SPLIT_K:
            # 16,384 values split into 32 partitions, summed by the looped
            # column reduction.
            assert info["kernel"] == "qvm_split_k" and info["splitK"] == 32
            assert column == (LOOPED_SUM, None)
        else:
            assert info["kernel"] == "qvm" and info["batched"] == (b > 1)
            assert column is None
    assert kernels == {"qvm": 108, "qvm_split_k": 3}


def test_references_take_matmuls_routes_without_copies():
    families = Counter(family for *_, family, _ in upstream.routes())
    assert families == {"gemv": 111}
    for (_, _, _, _, x_shape, w_shape), (*_, family, route) in zip(
        upstream.products(), upstream.routes()
    ):
        assert (family, route) == gemm_layout.matmul_route(
            "float32", *upstream.reference_operands(x_shape, w_shape)
        )
        assert route["entry"] == GEMV


def test_expected_dispatches_quantize_dequantize_multiply_and_sum_each_product():
    counts = upstream.expected_dispatches()
    entries = upstream.package_entries()
    # The split case's 16,384 x 128 weights hold 65,536 groups of 32 and
    # 131,072 of 16: one and two more launches than 65,535 groups each.
    assert sum(counts.values()) == 111 * 4 + 3 + 2 * (1 + 2 + 1)
    assert counts[LOOPED_SUM] == 3 and counts[GEMV] == 111
    assert len(entries["quantized"]) == 9
    assert {family for family, names in entries.items() if names} == {
        "quantized",
        "gemv",
    }
    assert upstream.OPERATORS == ("bfloat",)
    for mode in upstream.MODES:
        for operation in ("quantize", "dequantize"):
            entry = upstream.quantization_entries(mode, operation)
            assert counts[entry] == (39 if mode == "nvfp4" else 38)
    quantized = sum(
        count for entry, count in counts.items() if entry in entries["quantized"]
    )
    assert quantized == 111
    assert all(entry.startswith(upstream.PREFIXES) for entry in counts)


@pytest.mark.parametrize("target", ["metal", "opengl", "directx"])
@pytest.mark.parametrize(
    "fault",
    [None, "missing", "extra", "target", "partitions", "sum", "maximum"],
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
    elif fault == "partitions":
        index = next(i for i, item in enumerate(items) if "_split_k_" in item["entry"])
        items[index]["entry"] = items[index]["entry"].replace("_spk_32", "_spk_8")
    elif fault == "sum":
        index = next(i for i, item in enumerate(items) if item["entry"] == LOOPED_SUM)
        items[index]["entry"] = "col_reduce_small_1_reduce_sumfloat32"
    elif fault == "maximum":
        items = [item for item in items if item["entry"] != "all_reduce_maxfloat32"]
        count = len(items)
    if fault:
        with pytest.raises(RuntimeError, match="test_fp_qvm"):
            upstream.validate_trace(items, target, count)
    else:
        upstream.validate_trace(items, target, count)


def test_prepare_translates_the_products_columns_and_conversions(tmp_path, monkeypatch):
    from demos.integrations.mlx.portable_host import (
        fp_matmul_packages,
        gemv_packages,
    )
    from demos.integrations.mlx.portable_host import packages as operator_packages
    from demos.integrations.mlx.portable_host import reduction_packages, upstream_runner

    packages = tmp_path / "packages"
    packages.mkdir()
    (packages / "index.json").write_text(json.dumps({"target": "opengl"}))
    calls = []

    def build(name):
        def run(root, directory, target, entries, jobs):
            calls.append((name, target, list(entries), jobs))
            return [{"entry": entry} for entry in entries]

        return run

    monkeypatch.setattr(fp_matmul_packages, "build_entries", build("quantized"))
    monkeypatch.setattr(gemv_packages, "build_entries", build("gemv"))
    reductions = []
    monkeypatch.setattr(
        reduction_packages,
        "build_packages",
        lambda root, output, target, **options: reductions.append(
            (target, options["family"], tuple(options["entries"]))
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
    assert calls == [
        ("quantized", "opengl", entries["quantized"], 3),
        ("gemv", "opengl", entries["gemv"], 3),
    ]
    assert ("opengl", "all", upstream.REDUCTIONS) in reductions
    assert any(LOOPED_SUM in names for _, _, names in reductions)
    assert operators == [
        (upstream_runner.operator_directory(tmp_path / "reductions", "bfloat"),)
        + ("opengl", "bfloat")
    ]


def test_package_verification_entries_include_the_split_products():
    from demos.integrations.mlx.portable_host import (
        fp_matmul_packages,
        verify_upstream_fp_qmv,
        verify_upstream_fp_qmv_large_output,
        verify_upstream_qmv_small_non_multiples,
    )

    assert set(fp_matmul_packages.verification_entries()) == {
        *upstream.package_entries()["quantized"],
        *verify_upstream_fp_qmv.package_entries()["quantized"],
        *verify_upstream_fp_qmv_large_output.package_entries()["quantized"],
        *verify_upstream_qmv_small_non_multiples.package_entries()["fp"],
    }
    assert any(
        "_qvm_split_k_" in name for name in fp_matmul_packages.verification_entries()
    )

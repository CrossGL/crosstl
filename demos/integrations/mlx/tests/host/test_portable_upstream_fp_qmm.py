"""The unchanged upstream test_fp_qmm_non_multiple_of_32 runner: routes, trace, packages."""

import json
from collections import Counter

import pytest

from demos.integrations.mlx.portable_host import fp_qmm_layout, gemm_layout
from demos.integrations.mlx.portable_host import verify_upstream_fp_qmm as upstream
from demos.integrations.mlx.portable_host.verify_upstream_qmm_shapes import (
    reference_operands,
)


def trace(target="opengl"):
    items = [
        {"entry": entry, "target": target}
        for entry, count in sorted(upstream.expected_dispatches().items())
        for _ in range(count)
    ]
    return items + [
        {"entry": "all_reduce_maxfloat32", "target": target, "threads": 64},
        {"entry": "all_reduce_andbool_", "target": target, "threads": 64},
        {"entry": "rbitsc", "target": target},
    ]


def test_products_follow_the_upstream_checks_in_every_gpu_type():
    rows = list(upstream.products())
    # Five nvfp4 tails and two controls for each of two modes, per type.
    assert len(rows) == 3 * (5 + 2 * 2) == 27
    assert [row[0] for row in rows[::9]] == ["float32", "float16", "bfloat16"]
    assert rows[:5] == [("float32", "nvfp4", *shape) for shape in upstream.NVFP4_SHAPES]
    assert rows[5:9] == [
        ("float32", mode, *shape)
        for mode in ("mxfp4", "mxfp8")
        for shape in upstream.CONTROL_SHAPES
    ]


def test_quantized_routes_follow_quantized_matmul():
    kernels = Counter()
    for (dtype, mode, m, k, n, transpose), (_, quantized, column, _, _) in zip(
        upstream.products(), upstream.routes()
    ):
        info = fp_qmm_layout.signature(quantized)
        assert (info["mode"], info["dtype"]) == (mode, dtype)
        assert info["batched"] is False
        kernels[info["kernel"]] += 1
        if not transpose:
            assert info["kernel"] == "qmm_n" and column is None
        elif mode == "nvfp4":
            # K % 32 == 16 (or K == 16) admits no split of 32-wide partitions.
            assert info["kernel"] == "qmm_t" and column is None
        else:
            # 512 / 8 tiles capped at K / 32 = 33 partitions: a looped sum.
            assert info["kernel"] == "qmm_t_splitk"
            assert column == f"col_reduce_looped_1_32_32_reduce_sum{dtype}"
    assert kernels == {"qmm_t": 9, "qmm_n": 12, "qmm_t_splitk": 6}
    assert upstream.COLUMN_SUMS == tuple(
        sorted(
            f"col_reduce_looped_1_32_32_reduce_sum{dtype}" for dtype in upstream.DTYPES
        )
    )


def test_references_multiply_in_the_promoted_type_without_copies():
    families = Counter()
    for (dtype, _, m, k, n, transpose), (*_, family, route) in zip(
        upstream.products(), upstream.routes()
    ):
        reference = upstream.reference_type(dtype)
        assert reference == ("bfloat16" if dtype == "bfloat16" else "float32")
        w_shape = upstream.weight_shape(k, n, transpose)
        assert (family, route) == gemm_layout.matmul_route(
            reference, *reference_operands(w_shape, (m, k), transpose)
        )
        families[family] += 1
    assert set(families) <= {"gemm", "splitk"} and sum(families.values()) == 27


def test_expected_dispatches_quantize_multiply_twice_and_sum_each_check():
    counts = upstream.expected_dispatches()
    entries = upstream.package_entries()
    # The 1040 x 1040 nvfp4 weights hold 67,600 groups: two launches each way.
    quantizations = 2 * 3 * (5 + 1 + 4)
    products = sum(count for entry, count in counts.items() if entry in entries["qmm"])
    sums = sum(count for entry, count in counts.items() if entry.startswith("col_"))
    assert products == 2 * 27 and sums == 2 * 6
    accumulations = sum(
        count for entry, count in counts.items() if "_splitk_accum_" in entry
    )
    assert sum(counts.values()) == quantizations + products + sums + 27 + accumulations
    assert len(entries["qmm"]) == 18
    for dtype in upstream.DTYPES:
        entry = upstream.quantization_entries("nvfp4", dtype, "quantize")
        assert counts[entry] == 6
    # Every type dequantizes to bfloat16: one entry per mode.
    dequantize = upstream.quantization_entries("mxfp8", "float16", "dequantize")
    assert dequantize == upstream.quantization_entries("mxfp8", "float32", "dequantize")
    assert counts[dequantize] == 2 * 3
    assert all(
        "_dequantize_bfloat16_t_" in entry
        for entry in counts
        if "_dequantize_" in entry
    )
    assert all(entry.startswith(upstream.PREFIXES) for entry in counts)


@pytest.mark.parametrize("target", ["metal", "opengl", "directx"])
@pytest.mark.parametrize(
    "fault", [None, "missing", "extra", "target", "split", "sum", "reductions"]
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
    elif fault == "split":
        index = next(
            i for i, item in enumerate(items) if "_splitk_float" in item["entry"]
        )
        items[index]["entry"] = items[index]["entry"].replace(
            "_qmm_t_splitk_", "_qmm_t_"
        )
    elif fault == "sum":
        index = next(i for i, item in enumerate(items) if "col_reduce" in item["entry"])
        items[index]["entry"] = "col_reduce_small_1_reduce_sumfloat32"
    elif fault == "reductions":
        items = [item for item in items if item["entry"] not in upstream.REDUCTIONS]
        count = len(items)
    if fault:
        with pytest.raises(RuntimeError, match="test_fp_qmm_non_multiple_of_32"):
            upstream.validate_trace(items, target, count)
    else:
        upstream.validate_trace(items, target, count)


def test_prepare_translates_the_products_columns_and_conversions(tmp_path, monkeypatch):
    from demos.integrations.mlx.portable_host import (
        fp_qmm_packages,
        gemm_packages,
        gemm_splitk_packages,
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

    monkeypatch.setattr(fp_qmm_packages, "build_entries", build("qmm"))
    monkeypatch.setattr(gemm_packages, "build_entries", build("gemm"))
    monkeypatch.setattr(gemm_splitk_packages, "build_entries", build("splitk"))
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
        lambda root, output, target, **options: operators.append(options["family"]),
    )
    upstream.prepare_packages(tmp_path, packages, tmp_path / "reductions", 3)
    entries = upstream.package_entries()
    assert calls == [
        (family, "directx", entries[family], 3)
        for family in upstream.FAMILIES
        if entries[family]
    ]
    assert ("directx", "all", ("all_reduce_maxfloat32", "all_reduce_andbool_")) in (
        reductions
    )
    assert ("directx", "bfloat", ("all_reduce_maxbfloat16",)) in reductions
    assert ("directx", "column", upstream.COLUMN_SUMS) in reductions
    assert operators == list(upstream.OPERATORS)


def test_package_verification_entries_are_the_routed_products():
    from demos.integrations.mlx.portable_host import fp_qmm_packages

    assert fp_qmm_packages.verification_entries() == upstream.package_entries()["qmm"]

"""The unchanged upstream test_fp_qmv_large_output runner: routes, trace, packages."""

import json

import pytest

from demos.integrations.mlx.portable_host import (
    fp_matmul_layout,
    gemm_layout,
    quantization_layout,
)
from demos.integrations.mlx.portable_host import (
    verify_upstream_fp_qmv_large_output as upstream,
)

GEMV = "gemv_{}_bm8_bn1_sm1_sn32_tm4_tn4_nc0_axpby0"
SOURCES = {"float16": "float16_t", "bfloat16": "bfloat16_t", "float32": "float"}


def trace(target="opengl"):
    items = [
        {"entry": entry, "target": target}
        for entry, count in sorted(upstream.expected_dispatches().items())
        for _ in range(count)
    ]
    return items + [
        {"entry": "vv_Subtractfloat16", "target": target},
        {"entry": "all_reduce_andbool_", "target": target, "threads": 64},
        {"entry": "rbitsc", "target": target},
    ]


def test_products_follow_the_upstream_batches_and_dtypes():
    assert list(upstream.products()) == [
        ("float16", 1, (1, 512), (4096, 512)),
        ("bfloat16", 1, (1, 512), (4096, 512)),
        ("float32", 1, (1, 512), (4096, 512)),
        ("float16", 2, (2, 1, 512), (2, 4096, 512)),
        ("bfloat16", 2, (2, 1, 512), (2, 4096, 512)),
        ("float32", 2, (2, 1, 512), (2, 4096, 512)),
    ]


def test_every_product_takes_the_fast_vector_kernel_of_its_type():
    for (dtype, batches, *_), (quantized, family, route) in zip(
        upstream.products(), upstream.routes()
    ):
        info = fp_matmul_layout.signature(quantized)
        # K = 512 is a multiple of the 4-bit fast alignment and N of 8; one
        # row stays below every vector limit.
        assert info["kernel"] == "qmv_fast"
        assert info["mode"] == "nvfp4" and info["dtype"] == dtype
        assert info["batched"] == (batches > 1)
        assert quantized == (
            f"nvfp4_qmv_fast_{SOURCES[dtype]}_gs_16_b_4_batch_{int(batches > 1)}"
        )
        # The references multiply in the element type through Matmul's gemv.
        assert family == "gemv" and route["entry"] == GEMV.format(dtype)
        stacked = 0 if batches == 1 else batches
        assert (family, route) == gemm_layout.matmul_route(
            dtype,
            *upstream.reference_operands(
                4096, 512, stacked, upstream.shapes(batches)[0]
            ),
        )


def test_expected_dispatches_quantize_once_per_shape_and_dequantize_per_type():
    counts = upstream.expected_dispatches()
    # 4,096 x 512 values hold 131,072 groups of 16: three launches of at most
    # 65,535; the stacked pair needs five.
    assert quantization_layout.launches(4096 * 512, 16) == 3
    assert quantization_layout.launches(2 * 4096 * 512, 16) == 5
    assert counts["nvfp4_quantize_float_gs_16_b_4_hgs_false"] == 8
    for dtype, source in SOURCES.items():
        assert counts[f"nvfp4_dequantize_{source}_gs_16_b_4_hgs_false"] == 8
        assert counts[GEMV.format(dtype)] == 2
        for batched in (0, 1):
            assert counts[f"nvfp4_qmv_fast_{source}_gs_16_b_4_batch_{batched}"] == 1
    assert sum(counts.values()) == 8 + 3 * 8 + 3 * 2 + 6
    assert all(entry.startswith(upstream.PREFIXES) for entry in counts)
    assert upstream.quantization_entries("float32", "quantize") == (
        "nvfp4_quantize_float_gs_16_b_4_hgs_false"
    )


def test_package_entries_hold_six_products_and_three_references():
    entries = upstream.package_entries()
    assert entries["quantized"] == sorted(
        f"nvfp4_qmv_fast_{source}_gs_16_b_4_batch_{batched}"
        for source in SOURCES.values()
        for batched in (0, 1)
    )
    assert entries["gemv"] == sorted(GEMV.format(dtype) for dtype in SOURCES)
    assert entries["gemm"] == entries["splitk"] == []


@pytest.mark.parametrize("target", ["metal", "opengl", "directx"])
@pytest.mark.parametrize(
    "fault",
    [None, "missing", "extra", "target", "dtype", "batch", "dequantize", "reduction"],
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
    elif fault == "dtype":
        index = next(
            i for i, item in enumerate(items) if "qmv_fast_float16_t" in item["entry"]
        )
        items[index]["entry"] = items[index]["entry"].replace("float16_t", "float")
    elif fault == "batch":
        index = next(
            i for i, item in enumerate(items) if item["entry"].endswith("_batch_1")
        )
        items[index]["entry"] = items[index]["entry"].replace("_batch_1", "_batch_0")
    elif fault == "dequantize":
        index = next(
            i
            for i, item in enumerate(items)
            if "_dequantize_bfloat16_t" in item["entry"]
        )
        items[index]["entry"] = items[index]["entry"].replace("bfloat16_t", "float")
    elif fault == "reduction":
        items = [item for item in items if item["entry"] != "all_reduce_andbool_"]
        count = len(items)
    if fault:
        with pytest.raises(RuntimeError, match="test_fp_qmv_large_output"):
            upstream.validate_trace(items, target, count)
    else:
        upstream.validate_trace(items, target, count)


def test_prepare_translates_products_references_and_16_bit_operators(
    tmp_path, monkeypatch
):
    from demos.integrations.mlx.portable_host import (
        fp_matmul_packages,
        gemv_packages,
    )
    from demos.integrations.mlx.portable_host import packages as operator_packages
    from demos.integrations.mlx.portable_host import reduction_packages

    packages = tmp_path / "packages"
    packages.mkdir()
    (packages / "index.json").write_text(json.dumps({"target": "opengl"}))
    calls = []

    def build(name):
        def run(root, directory, target, entries, jobs):
            calls.append((name, target, list(entries), jobs))
            return [{"entry": entry} for entry in entries]

        return run

    for name, module in (("quantized", fp_matmul_packages), ("gemv", gemv_packages)):
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
    # The empty steel families translate nothing.
    assert calls == [
        ("quantized", "opengl", entries["quantized"], 2),
        ("gemv", "opengl", entries["gemv"], 2),
    ]
    assert reductions == [
        (
            tmp_path / "reductions",
            "opengl",
            {"family": "all", "entries": ("all_reduce_andbool_",), "on_demand": True},
        )
    ]
    assert operators == [
        (tmp_path / f"{family}-packages", "opengl", family)
        for family in ("half", "half-arithmetic", "bfloat")
    ]


def test_package_verification_entries_include_the_runners_products():
    from demos.integrations.mlx.portable_host import fp_matmul_packages

    entries = fp_matmul_packages.verification_entries()
    assert entries == sorted(set(entries))
    assert set(upstream.package_entries()["quantized"]) <= set(entries)

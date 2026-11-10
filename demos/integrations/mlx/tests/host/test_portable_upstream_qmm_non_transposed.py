"""The unchanged upstream test_qmm_non_transposed runner: routes, trace, packages."""

import json
from collections import Counter

import pytest

from demos.integrations.mlx.portable_host import (
    fp_matmul_layout,
    fp_qmm_layout,
    gemm_layout,
    matmul_layout,
    qmm_layout,
)
from demos.integrations.mlx.portable_host import (
    verify_upstream_qmm_non_transposed as upstream,
)
from demos.integrations.mlx.portable_host.verify_upstream_qmm_shapes import (
    reference_operands,
)

# The fp32-promoted maximum, the bfloat16 one and the float16 one.
MAXIMA = {
    ("float16", "affine"): "all_reduce_maxfloat16",
    ("float16", "fp"): "all_reduce_maxfloat32",
    ("bfloat16", "affine"): "all_reduce_maxbfloat16",
    ("bfloat16", "fp"): "all_reduce_maxbfloat16",
}


def trace(target="opengl"):
    items = [
        {"entry": entry, "target": target}
        for entry, count in sorted(upstream.expected_dispatches().items())
        for _ in range(count)
    ]
    return items + [
        {"entry": "vv_Subtractfloat16", "target": target},
        *(
            {"entry": entry, "target": target, "threads": 64}
            for entry in upstream.REDUCTIONS
        ),
        {"entry": "rbitsc", "target": target},
    ]


def test_products_follow_the_upstream_sweeps_in_order():
    products = list(upstream.products())
    # Per type: 14 sweep rows x 6 settings, 4 transformer shapes x (2 + 3),
    # 2 batches x (2 + 3), the 2 large checks and 3 x 3 x (3 + 3) depths.
    assert len(products) == 2 * (84 + 20 + 10 + 2 + 54)
    assert products[:4] == [
        ("float16", "affine", 64, 4, 1, 512, 1024, ()),
        ("float16", "affine", 128, 4, 1, 512, 1024, ()),
        ("float16", "affine", 64, 8, 1, 512, 1024, ()),
        ("float16", "mxfp4", 32, 4, 1, 512, 1024, ()),
    ]
    assert products[5] == ("float16", "mxfp8", 32, 8, 1, 512, 1024, ())
    # The block-scaled checks of the transformer loop reuse K = 512, N = 1024
    # and the loop's last M.
    assert products[84:89] == [
        ("float16", "affine", 64, 4, 100, 2048, 2048, ()),
        ("float16", "affine", 64, 4, 256, 2048, 2048, ()),
        ("float16", "mxfp4", 32, 4, 256, 512, 1024, ()),
        ("float16", "nvfp4", 16, 4, 256, 512, 1024, ()),
        ("float16", "mxfp8", 32, 8, 256, 512, 1024, ()),
    ]
    assert products[104:109] == [
        ("float16", "affine", 64, 4, 33, 512, 1024, (2,)),
        ("float16", "affine", 64, 4, 250, 512, 1024, (2,)),
        ("float16", "mxfp4", 32, 4, 250, 512, 1024, (2,)),
        ("float16", "nvfp4", 16, 4, 250, 512, 1024, (2,)),
        ("float16", "mxfp8", 32, 8, 250, 512, 1024, (2,)),
    ]
    # The large block-scaled check keeps the batched loop's last mode.
    assert products[114:116] == [
        ("float16", "affine", 64, 4, 33000, 128, 64, ()),
        ("float16", "mxfp8", 32, 8, 33000, 128, 64, ()),
    ]
    assert products[116] == ("float16", "affine", 32, 2, 8, 64, 128, ())
    assert products[169] == ("float16", "mxfp8", 32, 8, 65, 256, 128, ())
    assert products[170] == ("bfloat16", "affine", 64, 4, 1, 512, 1024, ())
    assert products[170:] == [("bfloat16", *product[1:]) for product in products[:170]]


def test_products_route_below_and_above_the_untransposed_vector_limit():
    seen = Counter()
    for (dtype, mode, group, bits, m, k, n, batch), (family, entry, *_) in zip(
        upstream.products(), upstream.routes()
    ):
        rows = m * (batch[0] * (batch[1] if len(batch) > 1 else 1) if batch else 1)
        # Batched activations fold into the rows of the 2-D weights.
        if mode == "affine":
            info = (
                matmul_layout.signature(entry)
                if rows < 4
                else qmm_layout.signature(entry)
            )
            assert info["groupSize"] == group and info["bits"] == bits
        else:
            info = (
                fp_matmul_layout.signature(entry)
                if rows < 4
                else fp_qmm_layout.signature(entry)
            )
            assert info["mode"] == mode
        assert info["batched"] is False
        assert info["dtype"] == dtype
        kernel = "qvm" if rows < matmul_layout.vector_limit(k, n, False) else "qmm_n"
        assert info["kernel"] == kernel
        assert (
            family
            == {
                ("affine", "qvm"): "quantized",
                ("affine", "qmm_n"): "qmm",
                ("fp", "qvm"): "fp",
                ("fp", "qmm_n"): "fpqmm",
            }["affine" if mode == "affine" else "fp", kernel]
        )
        seen[family] += 1
    # Rows 1 and 2 of the sweep take qvm in both types for all six settings.
    assert seen == {"quantized": 12, "fp": 12, "qmm": 152, "fpqmm": 164}


def test_references_promote_block_scaled_float16_products_to_float32():
    families = Counter()
    for (dtype, mode, _, _, m, k, n, batch), (_, _, family, route) in zip(
        upstream.products(), upstream.routes()
    ):
        reference = upstream.reference_type(dtype, mode)
        assert reference == (
            "float32" if (dtype, mode != "affine") == ("float16", True) else dtype
        )
        assert (family, route) == gemm_layout.matmul_route(
            reference, *reference_operands((k, n), (*batch, m, k), False)
        )
        assert reference in route["entry"]
        families[family] += 1
    # A single row takes gemv, short deep products split K.
    assert families == {"gemv": 12, "gemm": 248, "splitk": 80}
    assert upstream.reference_type("float16", "affine") == "float16"
    assert upstream.reference_type("bfloat16", "nvfp4") == "bfloat16"


def test_expected_dispatches_quantize_and_dequantize_every_check():
    counts = upstream.expected_dispatches()

    def launches(k, n, group):
        return -(-k * n // group // 65535)

    # 2048 x 2048 group-64 weights hold 65,536 groups, one past a launch, and
    # 11008 x 2048 ones take six launches.
    assert launches(2048, 2048, 64) == 2 and launches(11008, 2048, 64) == 6
    transformer = sum(launches(k, n, 64) for k, n in upstream.TRANSFORMER_SHAPES)
    assert transformer == 10
    for dtype in ("float16_t", "bfloat16_t"):
        # The sweep's 14 rows, the transformer's two rows per shape, 4 batched
        # checks and the large one.
        assert counts[f"affine_quantize_{dtype}_gs_64_b_4"] == 14 + 2 * transformer + 5
        assert counts[f"affine_quantize_{dtype}_gs_128_b_4"] == 14
        assert counts[f"affine_dequantize_{dtype}_gs_64_b_8"] == 14
        # Group-32 weights of 64 to 256 rows by 128 columns: one launch each.
        for bits in (2, 8):
            assert counts[f"affine_quantize_{dtype}_gs_32_b_{bits}"] == 9
        assert counts[f"affine_quantize_{dtype}_gs_32_b_4"] == 9
        # Block-scaled weights quantize from the test's type and dequantize
        # to bfloat16: 14 + 4 + 2 + 9 checks per mode, and the large mxfp8.
        assert counts[f"mxfp4_quantize_{dtype}_gs_32_b_4_hgs_false"] == 29
        assert counts[f"nvfp4_quantize_{dtype}_gs_16_b_4_hgs_false"] == 29
        assert counts[f"mxfp8_quantize_{dtype}_gs_32_b_8_hgs_false"] == 30
    assert counts["mxfp4_dequantize_bfloat16_t_gs_32_b_4_hgs_false"] == 58
    assert counts["mxfp8_dequantize_bfloat16_t_gs_32_b_8_hgs_false"] == 60
    products = list(upstream.products())
    quantizations = sum(
        2 * launches(k, n, group) for _, _, group, _, _, k, n, _ in products
    )
    # Quantize, dequantize, the product, its reference and 80 accumulations.
    assert sum(counts.values()) == quantizations + 2 * len(products) + 80
    assert all(entry.startswith(upstream.PREFIXES) for entry in counts)
    assert not any("_qmm_t" in entry or "_qmv" in entry for entry in counts)


def test_package_entries_cover_both_types_of_every_route():
    entries = upstream.package_entries()
    assert set(entries) == set(upstream.FAMILIES)
    assert all(names == sorted(set(names)) for names in entries.values())
    # Six affine settings, three modes, two types, unbatched.
    assert len(entries["qmm"]) == 12 and len(entries["fpqmm"]) == 6
    assert len(entries["quantized"]) == 6 and len(entries["fp"]) == 6
    assert all("_batch_0" in name for name in entries["qmm"] + entries["fpqmm"])
    assert all(name.startswith("gemv_t_") for name in entries["gemv"])
    assert all(name.startswith("steel_gemm_fused_nn_") for name in entries["gemm"])
    assert {"steel_gemm_splitk_accum_float16_float32"} <= set(entries["splitk"])


@pytest.mark.parametrize("target", ["metal", "opengl", "directx"])
@pytest.mark.parametrize(
    "fault", [None, "missing", "extra", "target", "transposed", "batched", "reduction"]
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
    elif fault == "transposed":
        # An untransposed product served by the transposed kernel.
        index = first("affine_qmm_n_float16_t_gs_64_b_4_batch_0")
        items[index]["entry"] = "affine_qmm_t_float16_t_gs_64_b_4_alN_true_batch_0"
    elif fault == "batched":
        # Folded activations launched as a batch.
        index = first("mxfp8_qmm_n_bfloat16_t_gs_32_b_8_batch_0")
        items[index]["entry"] = items[index]["entry"].replace("_batch_0", "_batch_1")
    elif fault == "reduction":
        items = [item for item in items if item["entry"] != "all_reduce_maxbfloat16"]
        count = len(items)
    if fault:
        with pytest.raises(RuntimeError, match="test_qmm_non_transposed"):
            upstream.validate_trace(items, target, count)
    else:
        upstream.validate_trace(items, target, count)


def test_prepare_translates_every_route_and_the_16_bit_operators(tmp_path, monkeypatch):
    from demos.integrations.mlx.portable_host import (
        fp_matmul_packages,
        fp_qmm_packages,
        gemm_packages,
        gemm_splitk_packages,
        gemv_packages,
        matmul_packages,
    )
    from demos.integrations.mlx.portable_host import packages as operator_packages
    from demos.integrations.mlx.portable_host import qmm_packages, reduction_packages

    packages = tmp_path / "packages"
    packages.mkdir()
    (packages / "index.json").write_text(json.dumps({"target": "opengl"}))
    calls = []

    def build(name):
        def run(root, directory, target, entries, jobs):
            calls.append((name, target, list(entries), jobs))
            return [{"entry": entry} for entry in entries]

        return run

    for name, module in (
        ("quantized", matmul_packages),
        ("fp", fp_matmul_packages),
        ("qmm", qmm_packages),
        ("fpqmm", fp_qmm_packages),
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
            (output.name, options["family"], options["entries"])
        ),
    )
    operators = []
    monkeypatch.setattr(
        operator_packages,
        "build_packages",
        lambda root, output, target, **options: operators.append(
            (output.name, options["family"])
        ),
    )
    upstream.prepare_packages(tmp_path, packages, tmp_path / "reductions", 3)
    entries = upstream.package_entries()
    assert calls == [
        (family, "opengl", entries[family], 3) for family in upstream.FAMILIES
    ]
    assert reductions == [
        ("reductions", "all", ("all_reduce_maxfloat32",)),
        ("reductions-half", "half", ("all_reduce_maxfloat16",)),
        ("reductions-bfloat", "bfloat", ("all_reduce_maxbfloat16",)),
    ]
    assert operators == [
        ("half-packages", "half"),
        ("half-arithmetic-packages", "half-arithmetic"),
        ("bfloat-packages", "bfloat"),
    ]
    assert set(MAXIMA.values()) == set(upstream.REDUCTIONS)

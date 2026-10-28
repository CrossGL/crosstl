"""The unchanged upstream test_qmv runner: routing, trace and package checks."""

import json
from collections import Counter

import pytest

from demos.integrations.mlx.portable_host import gemm_layout, matmul_layout
from demos.integrations.mlx.portable_host import verify_upstream_qmv as upstream

GEMV = "gemv_float32_bm4_bn1_sm1_sn32_tm4_tn4_nc0_axpby0"
GEMV_SHORT = "gemv_float32_bm1_bn1_sm8_sn4_tm4_tn4_nc0_axpby0"
GEMM_STEM = (
    "steel_gemm_fused_nt_float32_float32_bm64_bn64_bk16_wm2_wn2_has_batch_n"
    "_use_out_source_n_do_axpby_n_align_M_n"
)
GEMM_ALIGNED = GEMM_STEM + "_align_N_t_align_K_t"
GEMM_TAIL = GEMM_STEM + "_align_N_n_align_K_t"
SPLITK = (
    "steel_gemm_splitk_nt_float32_float32_bm16_bn32_bk16_wm2_wn2_MN_naligned_K_taligned"
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


def test_products_follow_the_upstream_grid():
    rows = list(upstream.products())
    # Two group sizes fit N of 64 and three fit 256; each meets 6 bits, 3 M, 4 B.
    assert len(rows) == (2 + 3) * 6 * 3 * 4 == 360
    unbatched = [row for row in rows if row[4] == 0]
    assert {row[5] for row in unbatched} == {(3, 1, 64), (3, 1, 256)}
    assert all(row[6] == row[2:4] for row in unbatched)
    batched = [row for row in rows if row[4]]
    assert all(row[5] == (row[4], 1, row[3]) for row in batched)
    assert all(row[6] == (row[4], row[2], row[3]) for row in batched)


def test_reference_operands_are_matmuls_view_of_swapped_weights():
    assert upstream.reference_operands(67, 256, 0, (3, 1, 256)) == (
        (3, 256),
        [256, 1],
        (256, 67),
        [1, 256],
    )
    assert upstream.reference_operands(512, 64, 8, (8, 1, 64)) == (
        (8, 1, 64),
        [64, 64, 1],
        (8, 64, 512),
        [64 * 512, 1, 64],
    )


def test_references_take_every_matmul_route():
    families = Counter(
        (b == 0, family)
        for (_, _, _, _, b, _, _), (_, _, _, family, _) in zip(
            upstream.products(), upstream.routes()
        )
    )
    # Batched vectors take gemv. The unbatched three rows split K when K = 256
    # covers M (256 or 67), across all three groups: 3 * 6 * 2 = 36.
    assert families == {
        (False, "gemv"): 270,
        (True, "gemm"): 54,
        (True, "splitk"): 36,
    }
    for group, bits, m, n, b, x_shape, _ in upstream.products():
        family, route = gemm_layout.matmul_route(
            "float32", *upstream.reference_operands(m, n, b, x_shape)
        )
        if family == "splitk":
            assert (n, m) in ((256, 256), (256, 67)) and b == 0
            assert route["entry"] == SPLITK and route["accum"] == ACCUM
        elif family == "gemm":
            assert route["entry"] == (GEMM_TAIL if m == 67 else GEMM_ALIGNED)


def test_expected_dispatches_follow_every_upstream_product():
    counts = upstream.expected_dispatches()
    entries = upstream.package_entries()
    assert sum(counts.values()) == 360 * 4 + 36
    assert entries["gemv"] == sorted([GEMV, GEMV_SHORT])
    assert entries["gemm"] == sorted([GEMM_ALIGNED, GEMM_TAIL])
    assert entries["splitk"] == sorted([SPLITK, ACCUM])
    assert counts[SPLITK] == counts[ACCUM] == 36
    assert counts[GEMM_ALIGNED] + counts[GEMM_TAIL] == 54
    assert counts[GEMV] + counts[GEMV_SHORT] == 270
    quantized = entries["quantized"]
    assert len(quantized) == 60
    assert Counter(entry.split("_float")[0] for entry in quantized) == {
        "affine_qmv": 36,
        "affine_qmv_quad": 12,
        "affine_qmv_fast": 12,
    }
    # Every quantized entry is one quantized.cpp would pick for its product.
    for group, bits, m, n, b, x_shape, _ in upstream.products():
        rows = 3 if b == 0 else 1
        entry = matmul_layout.select(
            "float32", group, bits, rows, m, n, max(b, 1), transpose=True
        )
        assert entry in quantized
        assert entry.endswith("_batch_1" if b > 1 else "_batch_0")
        assert ("_qmv_quad_" in entry) == (n == 64 and bits in (2, 4, 8))
    for group, m_count in ((128, 3), (64, 6), (32, 6)):
        for bits in (2, 3, 4, 5, 6, 8):
            for operation in ("quantize", "dequantize"):
                entry = f"affine_{operation}_float_gs_{group}_b_{bits}"
                assert counts[entry] == m_count * 4


@pytest.mark.parametrize("target", ["metal", "opengl", "directx"])
@pytest.mark.parametrize(
    "fault",
    [
        None,
        "missing",
        "extra",
        "target",
        "specialization",
        "reference",
        "accumulation",
        "maximum",
    ],
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
    elif fault == "specialization":
        index = next(i for i, item in enumerate(items) if "_qmv_fast_" in item["entry"])
        items[index]["entry"] = items[index]["entry"].replace("_qmv_fast_", "_qmv_")
    elif fault == "reference":
        items[first(GEMV)][
            "entry"
        ] = "gemv_t_float32_bm1_bn2_sm8_sn4_tm4_tn4_nc0_axpby0"
    elif fault == "accumulation":
        items.pop(first(ACCUM))
        count -= 1
    elif fault == "maximum":
        items = [item for item in items if item["entry"] != "all_reduce_maxfloat32"]
        count = len(items)
    if fault:
        with pytest.raises(RuntimeError, match="test_qmv"):
            upstream.validate_trace(items, target, count)
    else:
        upstream.validate_trace(items, target, count)


def test_prepare_translates_every_family_and_indexes_the_maximum(tmp_path, monkeypatch):
    from demos.integrations.mlx.portable_host import (
        gemm_packages,
        gemm_splitk_packages,
        gemv_packages,
        matmul_packages,
        reduction_packages,
    )

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
    upstream.prepare_packages(tmp_path, packages, tmp_path / "reductions", 3)
    entries = upstream.package_entries()
    assert calls == [(name, "opengl", entries[name], 3) for name in upstream.FAMILIES]
    assert reductions == {
        "target": "opengl",
        "options": {
            "family": "all",
            "entries": ("all_reduce_maxfloat32",),
            "on_demand": True,
        },
    }

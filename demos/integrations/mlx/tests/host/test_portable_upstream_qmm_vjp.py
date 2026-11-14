"""The unchanged upstream test_qmm_vjp and test_qmm_jvp runner."""

import json

import pytest

from demos.integrations.mlx.portable_host import qmm_layout
from demos.integrations.mlx.portable_host import verify_upstream_qmm_vjp as upstream

QMM_N = "affine_qmm_n_float_gs_64_b_8_batch_0"
SPLITK = "affine_qmm_t_splitk_float_gs_64_b_8_alN_true"
SUM = "col_reduce_small_1_reduce_sumfloat32"


def trace(target="opengl"):
    items = [
        {"entry": entry, "target": target}
        for entry, count in sorted(upstream.expected_dispatches().items())
        for _ in range(count)
    ]
    return items + [
        {"entry": "vv_Subtractfloat32", "target": target},
        {"entry": "all_reduce_andbool_", "target": target, "threads": 64},
        {"entry": "rbitsc", "target": target},
    ]


def test_products_differentiate_with_respect_to_the_input():
    assert list(upstream.products()) == [
        # The cotangent multiplies the weights with the transposition flipped.
        ("vjp", (1024, 512), (2, 64, 1024), False),
        ("vjp", (512, 1024), (2, 64, 1024), True),
        # The tangent multiplies them with the transposition kept.
        ("jvp", (128, 128), (2, 64, 128), True),
        ("jvp", (128, 128), (2, 64, 128), False),
    ]


def test_folded_rows_reach_the_matrix_kernels():
    routes = [route for _, route in upstream.routes()]
    assert [route[1] for route in routes] == [QMM_N, SPLITK, SPLITK, QMM_N]
    assert [route[2] for route in routes] == [None, SUM, SUM, None]
    # Both split products sum their partitions in the small pass, over
    # quantized.cpp's partition count for each depth.
    assert [route[3] for route in routes] == [
        None,
        f"{SUM}@32x8",
        f"{SUM}@32x2",
        None,
    ]
    assert qmm_layout.split_k(128, 512, 1024, 64) == 8
    assert qmm_layout.split_k(128, 128, 128, 64) == 2


def test_expected_dispatches_quantize_once_and_multiply_twice():
    assert upstream.expected_dispatches() == {
        "affine_quantize_float_gs_64_b_8": 4,
        QMM_N: 4,
        SPLITK: 4,
        SUM: 4,
    }


def test_package_entries_translate_both_kernels_and_both_sum_widths():
    assert upstream.package_entries() == {
        "qmm": [QMM_N, SPLITK],
        "small": [f"{SUM}@32x2", f"{SUM}@32x8"],
    }


@pytest.mark.parametrize("target", ["metal", "directx", "opengl"])
@pytest.mark.parametrize(
    "fault", [None, "missing", "extra", "target", "primal", "sum", "reduction"]
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
    elif fault == "primal":
        # An evaluated primal output launches a fifth product.
        items.append({"entry": QMM_N, "target": target})
        count += 1
    elif fault == "sum":
        # A split product without its partition sum.
        items.remove({"entry": SUM, "target": target})
        count -= 1
    elif fault == "reduction":
        items = [item for item in items if item["entry"] != "all_reduce_andbool_"]
        count = len(items)
    if fault:
        with pytest.raises(RuntimeError, match="test_qmm_vjp"):
            upstream.validate_trace(items, target, count)
    else:
        upstream.validate_trace(items, target, count)


def test_prepare_translates_the_products_sums_and_conjunction(tmp_path, monkeypatch):
    from demos.integrations.mlx.portable_host import column_small_packages
    from demos.integrations.mlx.portable_host import packages as operator_packages
    from demos.integrations.mlx.portable_host import qmm_packages, reduction_packages

    packages = tmp_path / "packages"
    packages.mkdir()
    (packages / "index.json").write_text(json.dumps({"target": "directx"}))
    calls = []

    def build(name):
        def run(root, directory, target, entries, jobs):
            calls.append((name, target, list(entries), jobs))
            return [{"entry": entry} for entry in entries]

        return run

    monkeypatch.setattr(qmm_packages, "build_entries", build("qmm"))
    monkeypatch.setattr(column_small_packages, "build_entries", build("small"))
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
        lambda root, output, target, **options: operators.append(options["family"]),
    )
    upstream.prepare_packages(tmp_path, packages, tmp_path / "reductions", 3)
    entries = upstream.package_entries()
    assert calls == [(name, "directx", entries[name], 3) for name in upstream.FAMILIES]
    assert reductions == [
        (
            tmp_path / "reductions",
            "directx",
            {"family": "all", "entries": ("all_reduce_andbool_",), "on_demand": True},
        )
    ]
    # Every comparison is in float32, so no 16-bit operator family loads.
    assert operators == []

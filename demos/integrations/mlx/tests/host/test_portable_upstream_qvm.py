"""The unchanged upstream test_qvm runner: routing, trace and package checks."""

import json

import pytest

from demos.integrations.mlx.portable_host import (
    verify_upstream_quantization as quantization,
)
from demos.integrations.mlx.portable_host import verify_upstream_qvm as upstream

GEMV = "gemv_t_float32_bm1_bn2_sm8_sn4_tm4_tn4_nc0_axpby0"


def record(mode, dispatches=None):
    return {
        "mode": mode,
        "tests": list(upstream.UPSTREAM_TESTS),
        "testsRun": 1,
        "success": True,
        "skipped": [],
        "errors": [],
        "failures": [],
        "nativeDispatches": (dispatches or len(trace())) if mode == "native" else 0,
    }


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


def test_expected_dispatches_follow_every_upstream_product():
    counts = upstream.expected_dispatches()
    qvm, gemv = upstream.package_entries()
    # Seven (group, M) pairs keep M >= group; each meets 6 bits, 3 N and 4 B.
    assert sum(1 for _ in upstream.products()) == 7 * 6 * 3 * 4 == 504
    assert gemv == [GEMV] and counts[GEMV] == 504
    assert qvm == sorted(
        f"affine_qvm_float_gs_{group}_b_{bits}_batch_{batch}"
        for group in (32, 64, 128)
        for bits in (2, 3, 4, 5, 6, 8)
        for batch in (0, 1)
    )
    for group, m_count in ((32, 3), (64, 2), (128, 2)):
        for bits in (2, 3, 4, 5, 6, 8):
            # B of 0 and 1 run unbatched, 3 and 8 batched.
            for batch in (0, 1):
                entry = f"affine_qvm_float_gs_{group}_b_{bits}_batch_{batch}"
                assert counts[entry] == m_count * 3 * 2
            for operation in ("quantize", "dequantize"):
                entry = f"affine_{operation}_float_gs_{group}_b_{bits}"
                assert counts[entry] == m_count * 3 * 4
    assert sum(counts.values()) == 504 * 4


@pytest.mark.parametrize("target", ["metal", "opengl", "directx"])
@pytest.mark.parametrize(
    "fault",
    [None, "missing", "extra", "target", "specialization", "reference", "maximum"],
)
def test_trace_requires_every_translated_product(target, fault):
    items = trace(target)
    count = len(items)
    if fault == "missing":
        items.pop()
    elif fault == "extra":
        items.append(items[0])
    elif fault == "target":
        items[0]["target"] = "other"
    elif fault == "specialization":
        items[0]["entry"] = "affine_qmv_float_gs_32_b_4_batch_0"
    elif fault == "reference":
        index = next(i for i, item in enumerate(items) if item["entry"] == GEMV)
        items[index]["entry"] = "gemv_float32_bm4_bn1_sm1_sn32_tm4_tn4_nc0_axpby0"
    elif fault == "maximum":
        items = [item for item in items if item["entry"] != "all_reduce_maxfloat32"]
        count = len(items)
    if fault:
        with pytest.raises(RuntimeError):
            upstream.validate_trace(items, target, count)
    else:
        upstream.validate_trace(items, target, count)


def test_results_name_test_qvm():
    quantization.validate_result(record("cpu"), "cpu", upstream.UPSTREAM_TESTS)
    with pytest.raises(RuntimeError, match="test_qvm did not pass"):
        quantization.validate_result(
            {**record("native"), "failures": [["subtest", "1e-3"]]},
            "native",
            upstream.UPSTREAM_TESTS,
        )
    with pytest.raises(RuntimeError, match="did not pass"):
        quantization.validate_result(record("cpu"), "cpu")


def test_prepare_translates_routes_and_indexes_the_maximum(tmp_path, monkeypatch):
    from demos.integrations.mlx.portable_host import (
        gemv_packages,
        matmul_packages,
        reduction_packages,
    )

    packages = tmp_path / "packages"
    packages.mkdir()
    (packages / "index.json").write_text(json.dumps({"target": "metal"}))
    calls = {}

    def build(name):
        def run(root, directory, target, entries, jobs):
            calls[name] = (target, list(entries), jobs)
            return [{"entry": entry} for entry in entries]

        return run

    monkeypatch.setattr(matmul_packages, "build_entries", build("qvm"))
    monkeypatch.setattr(gemv_packages, "build_entries", build("gemv"))
    monkeypatch.setattr(
        reduction_packages,
        "build_packages",
        lambda root, output, target, **options: calls.update(
            reductions=(target, options)
        ),
    )
    upstream.prepare_packages(tmp_path, packages, tmp_path / "reductions", 3)
    qvm, gemv = upstream.package_entries()
    assert calls["qvm"] == ("metal", qvm, 3) and len(qvm) == 36
    assert calls["gemv"] == ("metal", [GEMV], 3)
    assert calls["reductions"] == (
        "metal",
        {"family": "all", "entries": ("all_reduce_maxfloat32",), "on_demand": True},
    )

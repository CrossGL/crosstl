"""The SMT check of the rule table: every entry is sound, and the checker rejects unsound entries."""

import pytest

z3 = pytest.importorskip("z3")

from crosstl.translator.semantics import helpers, rule_table  # noqa: E402
from crosstl.translator.semantics.lower import template  # noqa: E402
from crosstl.translator.semantics.rule_table import TargetEntry  # noqa: E402

from . import rule_table_smt  # noqa: E402


def test_every_rule_table_entry_is_sound():
    records, failures = rule_table_smt.check_table()
    assert failures == []
    checked = {(r["intrinsic"], r["target"]) for r in records if r["result"] == "sound"}
    assert ("__cgl_wrap_add", "metal") in checked
    assert ("__cgl_wrap_mul", "vulkan") in checked


def test_native_signed_arithmetic_is_unsound_where_overflow_is_undefined():
    for target in ("metal", "directx", "vulkan", "cuda"):
        record = rule_table_smt.check_entry(
            "__cgl_wrap_add", target, TargetEntry("native", "mutation")
        )
        assert record["result"] == "unsound", target
        x0 = int(record["counterexample"]["x0"], 16)
        x1 = int(record["counterexample"]["x1"], 16)
        signed = [v - (1 << 32) if v >= 1 << 31 else v for v in (x0, x1)]
        assert not -(1 << 31) <= sum(signed) < 1 << 31


def test_rust_native_entry_needs_its_recorded_configuration():
    unrecorded = TargetEntry("native", "mutation")
    assert rule_table_smt.check_entry("__cgl_wrap_mul", "rust", unrecorded)[
        "result"
    ] == ("unsound")
    recorded = rule_table.TARGET_ENTRIES[(rule_table.WRAP, "rust")]
    assert rule_table_smt.check_entry("__cgl_wrap_mul", "rust", recorded)["result"] == (
        "sound"
    )


@pytest.mark.parametrize(
    "polyfill, result",
    [
        ("x0 + x1", "unsound"),
        ("asint(asuint(x0) + asuint(x0))", "malformed"),
        ("asint(asuint(x0) + asuint(x1) + asuint(x1))", "malformed"),
        ("asint(asuint(x0) << asuint(x1))", "unmodelled"),
        ("asint(asuint(x0) / asuint(x1))", "unsound"),
        ("asint(asuint(x0) + asuint(x1) + 1u)", "unsound"),
    ],
)
def test_checker_rejects_defective_polyfills(monkeypatch, polyfill, result):
    monkeypatch.setitem(rule_table.POLYFILLS, "__cgl_wrap_add", polyfill)
    template.cache_clear()
    try:
        record = rule_table_smt.check_entry(
            "__cgl_wrap_add", "metal", TargetEntry("polyfill", "mutation")
        )
    finally:
        monkeypatch.undo()
        template.cache_clear()
    assert record["result"] == result


def test_native_unsigned_division_misses_the_direct3d_zero_divisor_rule():
    for target in ("metal", "directx", "vulkan", "opengl", "wgsl", "rust"):
        for intrinsic in ("__cgl_udiv_ones", "__cgl_urem_ones"):
            record = rule_table_smt.check_entry(
                intrinsic, target, TargetEntry("native", "mutation")
            )
            assert record["result"] == "unsound", (intrinsic, target)
            assert int(record["counterexample"]["x1"], 16) == 0


@pytest.mark.parametrize(
    "body, target, result",
    [
        # Selecting all ones does not guard the division: a target may evaluate both arms.
        ("x1 == 0u ? 0xFFFFFFFFu : x0 / x1", "metal", "unsound"),
        ("x1 == 0u ? 0xFFFFFFFFu : x0 / x1", "vulkan", "unsound"),
        # WGSL defines e1 / 0 as e1, so the unguarded division is sound there.
        ("x1 == 0u ? 0xFFFFFFFFu : x0 / x1", "wgsl", "sound"),
        ("x1 == 0u ? 0u : x0 / (x1 == 0u ? 1u : x1)", "metal", "unsound"),
        ("x1 != 0u ? x0 / (x1 == 0u ? 1u : x1) : 0xFFFFFFFFu", "metal", "sound"),
        ("x1 == 0u ? 0xFFFFFFFFu : x0 % (x1 == 0u ? 1u : x1)", "metal", "unsound"),
        ("x1 == 0u ? 0xFFFFFFFFu : x0 / (x1 + 1u)", "directx", "unsound"),
        ("x1 == 0u ? 0xFFFFFFFFu : uint(asint(x0) / asint(x1))", "metal", "unmodelled"),
    ],
)
def test_checker_rejects_defective_udiv_helpers(monkeypatch, body, target, result):
    original = rule_table.HELPERS["__cgl_udiv_ones"]
    monkeypatch.setitem(
        rule_table.HELPERS,
        "__cgl_udiv_ones",
        rule_table.Helper(original.name, original.scalar, body),
    )
    helpers._parsed.cache_clear()
    try:
        record = rule_table_smt.check_entry(
            "__cgl_udiv_ones", target, TargetEntry("polyfill", "mutation")
        )
    finally:
        monkeypatch.undo()
        helpers._parsed.cache_clear()
    assert record["result"] == result


def test_a_backend_without_an_entry_fails_the_check():
    records, failures = rule_table_smt.check_table(backends=["metal", "newbackend"])
    assert {"family": "wrap", "target": "newbackend", "result": "missing"} in failures
    assert all(
        r.get("target") != "newbackend" or r["result"] == "missing" for r in records
    )


def test_cli_exits_zero_on_the_committed_table(capsys):
    assert rule_table_smt.main([]) == 0
    assert "0 failures" in capsys.readouterr().out


def test_every_saturated_conversion_entry_is_checked_and_sound():
    records, _ = rule_table_smt.check_table()
    checked = {(r["intrinsic"], r["target"]): r["result"] for r in records}
    for intrinsic in rule_table.CONVERT_SAT_INTRINSICS:
        for target in ("metal", "directx", "vulkan", "opengl", "wgsl"):
            assert checked[(intrinsic, target)] == "sound", (intrinsic, target)


_MAGNITUDE = "(asuint(x0) & 0x7FFFFFFFu)"


@pytest.mark.parametrize(
    "intrinsic, body, result",
    [
        # The unguarded conversion is undefined for NaN and out-of-range operands.
        ("__cgl_sat_ftoi", "int(x0)", "unsound"),
        # Converting x0 itself in the in-range arm is undefined where both arms are evaluated.
        (
            "__cgl_sat_ftoi",
            f"{_MAGNITUDE} < 0x4F000000u ? int(x0) : ({_MAGNITUDE} > 0x7F800000u ? 0 : "
            "(asuint(x0) >= 0x80000000u ? asint(0x80000000u) : 2147483647))",
            "unsound",
        ),
        # Without the NaN test, NaN saturates instead of giving 0.
        (
            "__cgl_sat_ftoi",
            f"{_MAGNITUDE} < 0x4F000000u ? int(asfloat({_MAGNITUDE} < 0x4F000000u ? asuint(x0) : 0u)) : "
            "(asuint(x0) >= 0x80000000u ? asint(0x80000000u) : 2147483647)",
            "unsound",
        ),
        # 2^31 itself is not representable as int.
        (
            "__cgl_sat_ftoi",
            f"{_MAGNITUDE} <= 0x4F000000u ? int(asfloat({_MAGNITUDE} <= 0x4F000000u ? asuint(x0) : 0u)) : "
            f"({_MAGNITUDE} > 0x7F800000u ? 0 : (asuint(x0) >= 0x80000000u ? asint(0x80000000u) : 2147483647))",
            "unsound",
        ),
        # Negative values saturate to 0, so testing the magnitude alone is wrong for uint.
        (
            "__cgl_sat_ftou",
            f"{_MAGNITUDE} > 0x7F800000u ? 0u : ({_MAGNITUDE} < 0x4F800000u ? "
            f"uint(asfloat({_MAGNITUDE} < 0x4F800000u ? {_MAGNITUDE} : 0u)) : 0xFFFFFFFFu)",
            "unsound",
        ),
        ("__cgl_sat_itou", "uint(x0)", "unsound"),
        ("__cgl_sat_itou", "x0 <= 0 ? 0u : uint(x0)", "sound"),
        ("__cgl_sat_utoi", "x0 >= 0x7FFFFFFFu ? 2147483647 : int(x0)", "sound"),
        ("__cgl_sat_utoi", "x0 > 0x80000000u ? 2147483647 : int(x0)", "unsound"),
        # Float comparisons are not modelled; the helpers decide on bits.
        ("__cgl_sat_ftoi", "x0 == x0 ? int(x0) : 0", "unmodelled"),
    ],
)
def test_checker_rejects_defective_conversion_helpers(
    monkeypatch, intrinsic, body, result
):
    original = rule_table.HELPERS[intrinsic]
    monkeypatch.setitem(
        rule_table.HELPERS,
        intrinsic,
        rule_table.Helper(
            original.name, original.scalar, body, original.parameters, original.result
        ),
    )
    helpers._parsed.cache_clear()
    try:
        records = [
            rule_table_smt.check_entry(
                intrinsic, target, TargetEntry("polyfill", "mutation")
            )
            for target in ("metal", "opengl")
        ]
    finally:
        monkeypatch.undo()
        helpers._parsed.cache_clear()
    assert [record["result"] for record in records] == [result, result]


def test_native_conversion_is_unsound_on_every_target():
    for target in ("metal", "opengl", "wgsl", "vulkan", "directx"):
        record = rule_table_smt.check_entry(
            "__cgl_sat_ftoi", target, TargetEntry("native", "mutation")
        )
        assert record["result"] == "unsound", target

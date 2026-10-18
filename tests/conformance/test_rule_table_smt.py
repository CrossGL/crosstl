"""The SMT check of the rule table: every entry is sound, and the checker rejects unsound entries."""

import pytest

z3 = pytest.importorskip("z3")

from crosstl.translator.semantics import rule_table  # noqa: E402
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
        ("asint(asuint(x0) / asuint(x1))", "unmodelled"),
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


def test_a_backend_without_an_entry_fails_the_check():
    records, failures = rule_table_smt.check_table(backends=["metal", "newbackend"])
    assert {"family": "wrap", "target": "newbackend", "result": "missing"} in failures
    assert all(
        r.get("target") != "newbackend" or r["result"] == "missing" for r in records
    )


def test_cli_exits_zero_on_the_committed_table(capsys):
    assert rule_table_smt.main([]) == 0
    assert "0 failures" in capsys.readouterr().out

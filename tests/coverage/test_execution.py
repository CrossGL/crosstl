import numpy as np
import pytest

from tests.conformance import runner as harness
from tests.conformance import runtimes
from tools.coverage import (
    execution,
    issues,
    probes,
    publish,
    report,
    runner,
    validators,
)
from tools.coverage.model import Feature, Param, Result, Signature, Type

SIN = Feature(
    "metal",
    "function",
    "sin",
    Signature(Type("float"), (Param(Type("float")),)),
    crossgl="sin",
)
NATIVE = probes.build(SIN, "metal").source


def supported(translated=NATIVE, direction="output"):
    return Result(SIN, direction, "supported", sources=(NATIVE, translated))


def metal_runs():
    try:
        return runtimes.provision(runtimes.RUNTIMES["metal-apple"]).available
    except Exception:  # noqa: BLE001 - an unprovisionable runtime is an unavailable one
        return False


def test_only_supported_probes_that_read_one_input_buffer_run():
    assert execution.executable(supported())
    assert not execution.executable(Result(SIN, "output", "supported"))
    assert not execution.executable(
        Result(SIN, "output", "wrong", sources=(NATIVE, NATIVE))
    )
    ldexp = Feature(
        "metal",
        "function",
        "ldexp",
        Signature(Type("float"), (Param(Type("float")), Param(Type("int")))),
    )
    two_buffers = probes.build(ldexp, "metal").source
    assert not execution.executable(supported(two_buffers))
    assert not execution.executable(
        Result(SIN, "output", "supported", sources=(two_buffers, two_buffers))
    )


def test_every_run_gets_the_same_inputs_and_small_integers():
    assert np.array_equal(execution.inputs("fin"), execution.inputs("fin"))
    for buffer, dtype in (("iin", np.int32), ("uin", np.uint32)):
        data = execution.inputs(buffer)
        assert data.dtype == dtype
        assert data.min() >= 0 and data.max() < 16
    assert len(execution.inputs("fin")) == execution.THREADS + execution.SPAN


def test_results_match_word_for_word_and_nan_matches_nan():
    words = np.array([0x3F800000, 0x7FC00000, 0xA5A5A5A5], dtype=np.uint32)
    other_nan = np.array([0x3F800000, 0x7FC00001, 0xA5A5A5A5], dtype=np.uint32)
    assert execution.difference(words, other_nan) == ""
    changed = np.array([0x3F800001, 0x7FC00000, 0], dtype=np.uint32)
    found = execution.difference(words, changed)
    assert found.startswith(execution.DIVERGED + "\n")
    assert "2 output words differ. The first, word 0, is 0x3f800001" in found
    assert "the native probe writes 0x3f800000" in found


def _fake_harness(monkeypatch, translated_outcome, written, runtime="metal-apple"):
    """Make run_suite report the native probe as run and the translation as ``translated_outcome``."""

    class Provided:
        available = True

    runtime = runtimes.RUNTIMES[runtime]
    monkeypatch.setattr(
        harness,
        "select_runtimes",
        lambda spec, targets: {t: [(runtime, Provided())] for t in targets},
    )

    def run_suite(cases, targets, configs, selection, root, *, reference=False):
        assert reference and targets == [runtime.target] and configs == ["strict"]
        records = []
        for case in cases:
            assert "#version 460" not in case.src.read_text()
            role = case.id.split("-")[1]
            outcome = "preserved" if role == "native" else translated_outcome
            words = np.zeros(case.out_count, dtype=np.uint32)
            if role == "translated":
                words[:] = written
            if outcome in ("preserved", "memory-corruption"):
                case.check(case.inputs, words)
            records.append({"case": case.id, "runtime": runtime.id, "outcome": outcome})
        return records

    monkeypatch.setattr(harness, "run_suite", run_suite)


@pytest.mark.parametrize(
    ("outcome", "written", "expected", "detail"),
    [
        ("preserved", 0, "supported", ""),
        (
            "preserved",
            1,
            "wrong",
            f"on metal-apple, the translation {execution.DIVERGED}",
        ),
        (
            "memory-corruption",
            0,
            "wrong",
            "on metal-apple, the translation writes outside its output buffer",
        ),
        ("driver-rejected", 0, "wrong", "on metal-apple, the translation is rejected"),
        # No evidence either way: the validation result stands.
        ("timeout", 0, "supported", ""),
        ("interface-changed", 0, "supported", ""),
    ],
)
def test_a_translation_that_computes_otherwise_is_wrong(
    monkeypatch, outcome, written, expected, detail
):
    _fake_harness(monkeypatch, outcome, written)
    untouched = Result(SIN, "input", "refused", "a diagnostic")
    [result, other], used = execution.run([supported(), untouched], workers=2)
    assert used == ["metal-apple"]
    assert other == untouched
    assert result.outcome == expected
    assert result.detail.startswith(detail)
    ran = outcome in ("preserved", "memory-corruption", "driver-rejected")
    assert result.runtime == ("metal-apple" if ran else "")
    assert execution.found(result.detail) == (expected == "wrong")


def test_glsl_runs_as_glsl_450_because_llvmpipe_implements_opengl_45():
    assert execution.runnable("glsl", "#version 460\nvoid main() {}\n") == (
        "#version 450 core\nvoid main() {}\n"
    )
    assert execution.runnable("glsl", "#version 450 core\n") == "#version 450 core\n"
    assert execution.runnable("metal", NATIVE) == NATIVE


@pytest.mark.parametrize(
    ("declared", "expected"),
    [("#version 460", "supported"), ("#version 450 core", "wrong")],
)
def test_a_rejected_glsl_translation_is_wrong_unless_it_declares_460(
    monkeypatch, declared, expected
):
    _fake_harness(monkeypatch, "driver-rejected", 0, "opengl-llvmpipe")
    feature = Feature("glsl", "function", "sin", SIN.signature, crossgl="sin")
    native = probes.build(feature, "glsl").source
    assert native.startswith("#version 460\n")
    translated = native.replace("#version 460", declared, 1)
    probe = Result(feature, "input", "supported", sources=(native, translated))
    [result], used = execution.run([probe])
    assert used == ["opengl-llvmpipe"]
    assert result.outcome == expected


def test_nothing_runs_without_a_runtime(monkeypatch):
    monkeypatch.setattr(
        harness, "select_runtimes", lambda spec, targets: {t: [] for t in targets}
    )
    results = [supported()]
    assert execution.run(results) == (results, [])
    assert execution.idle(results, []) == ["metal"]
    assert execution.idle(results, ["metal-apple"]) == []
    assert execution.idle([Result(SIN, "output", "refused")], []) == []


def test_the_harness_takes_coverage_probes_as_cases(tmp_path):
    outputs = {}
    path = tmp_path / "probe.metal"
    path.write_text(NATIVE)
    case = execution._case(
        "00000-native", path, "metal", execution.inputs("fin"), outputs
    )
    [record] = harness.run_suite(
        [case], ["metal"], ["strict"], {"metal": []}, tmp_path / "run", reference=True
    )
    assert record["case"] == "00000-native"
    assert record["runtime"].startswith("compile-only@")


@pytest.mark.skipif(not metal_runs(), reason="no Metal runtime")
def test_metal_runs_the_native_probe_against_the_translation():
    same = supported()
    other = supported(NATIVE.replace("sin(a0)", "cos(a0)"))
    outside = supported(NATIVE.replace("outp[i] = r;", "outp[i + 1024u] = r;"))
    (same, other, outside), used = execution.run([same, other, outside])
    assert used == ["metal-apple"]
    assert (same.outcome, same.runtime) == ("supported", "metal-apple")
    assert other.outcome == "wrong" and execution.DIVERGED in other.detail
    assert outside.outcome == "wrong" and "outside its output buffer" in outside.detail


def test_supported_metal_compute_results_keep_both_sources(monkeypatch):
    monkeypatch.setattr(validators, "available", lambda language, target="": True)
    monkeypatch.setattr(runner, "_native_rejection", lambda feature, language: "")
    monkeypatch.setattr(runner, "translate", lambda *args: "translated metal")
    monkeypatch.setattr(
        runner, "_validated", lambda *args, **kwargs: validators.Verdict(True)
    )
    assert runner.run_output(SIN, "metal").sources == (NATIVE, "translated metal")
    assert runner.run_input(SIN, "metal").sources == (NATIVE, "translated metal")
    fragment = Feature(
        "metal", "function", "sin", SIN.signature, crossgl="sin", stage="fragment"
    )
    assert runner.run_output(fragment, "metal").sources == ()


def test_the_report_names_the_runtime_and_coverage_md_says_what_ran():
    ran = Result(SIN, "output", "supported", runtime="metal-apple")
    data = report.build([ran], ["metal"], ["output"])
    [row] = [b for b in data["backends"] if b["language"] == "metal"]
    assert (row["runtime"], row["executed"]) == ("metal-apple", 1)
    assert "Executed: Metal output, 1 probes on metal-apple." in publish.markdown(data)
    plain = report.build([Result(SIN, "output", "supported")], ["metal"], ["output"])
    assert "Executed:" not in publish.markdown(plain)


def test_an_execution_bug_is_reproduced_with_execute():
    detail = (
        f"on metal-apple, the translation {execution.DIVERGED}\n1 output words differ."
    )
    feature = {
        "name": "sin",
        "category": "function",
        "outcome": "wrong",
        "detail": detail,
    }
    row = {
        "language": "metal",
        "direction": "output",
        "offered": True,
        "score": 0.0,
        "counts": {"supported": 0, "refused": 0, "wrong": 1, "untested": 0},
        "features": [feature],
    }
    body = issues.bug_body(row, "cause", [feature], "CrossGL/crosstl")
    assert "--feature sin --execute" in body
    assert "but run, it does not compute what the native Metal probe computes" in body
    assert not execution.found("probe.metal:3:1: error: no matching function")

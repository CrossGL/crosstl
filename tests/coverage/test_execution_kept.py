"""Execution verdicts kept between coverage runs (tools/coverage/executed.py)."""

import os
from dataclasses import replace

import pytest

from tests.conformance import runner as harness
from tools.coverage import executed, execution

from .execution_cases import fake_harness, hlsl_result


def _count_runs(monkeypatch):
    """Count the cases each ``run_suite`` call gets, on top of the fake harness."""
    calls = []
    original = harness.run_suite

    def run_suite(cases, *args, **kwargs):
        calls.append(len(cases))
        return original(cases, *args, **kwargs)

    monkeypatch.setattr(harness, "run_suite", run_suite)
    return calls


@pytest.fixture
def kept(monkeypatch, tmp_path):
    folder = tmp_path / "executions"
    monkeypatch.setenv(executed.EXECUTIONS, str(folder))
    return folder


def _verdict(result):
    return tuple(getattr(result, name) for name in executed.FIELDS)


def test_a_run_without_the_variable_keeps_nothing(monkeypatch, tmp_path):
    monkeypatch.delenv(executed.EXECUTIONS, raising=False)
    fake_harness(monkeypatch, "preserved", 0, "d3d12-warp")
    calls = _count_runs(monkeypatch)
    execution.run([hlsl_result()], "d3d12-warp")
    execution.run([hlsl_result()], "d3d12-warp")
    assert calls == [2, 2]
    assert not any(tmp_path.iterdir())


def test_a_later_run_takes_the_kept_verdict_instead_of_running_again(monkeypatch, kept):
    fake_harness(monkeypatch, "preserved", 0, "d3d12-warp")
    calls = _count_runs(monkeypatch)
    [first], used = execution.run([hlsl_result()], "d3d12-warp")
    assert first.runtime == "d3d12-warp" and used == ["d3d12-warp"]
    assert len(list(kept.iterdir())) == 1
    [again], used_again = execution.run([hlsl_result()], "d3d12-warp")
    assert calls == [2]
    assert _verdict(again) == _verdict(first) and used_again == used


@pytest.mark.parametrize(
    "change",
    [
        "translation",
        "runtime identity",
        "compiler version",
        "code",
    ],
)
def test_anything_the_verdict_depends_on_runs_the_probes_again(
    monkeypatch, kept, change
):
    fake_harness(monkeypatch, "preserved", 0, "d3d12-warp")
    calls = _count_runs(monkeypatch)
    execution.run([hlsl_result()], "d3d12-warp")
    result = hlsl_result()
    if change == "translation":
        native, translated = result.sources
        result = replace(result, sources=(native, translated + "\n"))
    elif change == "runtime identity":
        select = harness.select_runtimes

        def other_device(spec, targets):
            chosen = select(spec, targets)
            return {
                target: [
                    (runtime, replace_identity(provision))
                    for runtime, provision in pairs
                ]
                for target, pairs in chosen.items()
            }

        monkeypatch.setattr(harness, "select_runtimes", other_device)
    elif change == "compiler version":
        executed._compilers.cache_clear()
        monkeypatch.setattr(executed.validators, "version", lambda tool: "next")
    else:
        executed.code_digest.cache_clear()
        monkeypatch.setattr(executed, "distributions", lambda: "numpy==99")
    try:
        execution.run([result], "d3d12-warp")
    finally:
        executed._compilers.cache_clear()
        executed.code_digest.cache_clear()
    assert calls == [2, 2]


def replace_identity(provision):
    class Other:
        available = provision.available
        identity = f"{provision.identity} (another driver)"

    return Other()


def test_only_the_verdicts_the_latest_run_used_survive(monkeypatch, kept):
    fake_harness(monkeypatch, "preserved", 0, "d3d12-warp")
    execution.run([hlsl_result()], "d3d12-warp")
    [old] = list(kept.iterdir())
    native, translated = hlsl_result().sources
    changed = replace(hlsl_result(), sources=(native, translated + "\n"))
    # The kept file predates this run by more than the one second of slack forget_unused allows.
    stamp = old.stat().st_mtime - 5
    os.utime(old, (stamp, stamp))
    execution.run([changed], "d3d12-warp")
    [new] = list(kept.iterdir())
    assert new != old


def test_a_verdict_the_runtime_did_not_give_is_not_kept(monkeypatch, kept):
    fake_harness(
        monkeypatch, "skipped", 0, "d3d12-warp", message="needs 64-bit atomics"
    )
    [result], _ = execution.run([hlsl_result()], "d3d12-warp")
    assert result.runtime == "" and result.unrun
    assert not any(kept.iterdir())


def test_the_code_digest_follows_the_installed_distributions(monkeypatch):
    assert "numpy==" in executed.distributions()
    executed.code_digest.cache_clear()
    try:
        before = executed.code_digest()
        executed.code_digest.cache_clear()
        monkeypatch.setattr(executed, "distributions", lambda: "numpy==99")
        assert executed.code_digest() != before
    finally:
        executed.code_digest.cache_clear()


def test_the_linux_coverage_leg_carries_its_executions_like_its_verdicts():
    import yaml

    workflow = yaml.safe_load(
        (executed.ROOT / ".github/workflows/coverage.yml").read_text()
    )
    steps = {step.get("name"): step for step in workflow["jobs"]["linux"]["steps"]}
    restored = steps["Restore verdicts"]["with"]["path"].splitlines()
    saved = steps["Keep verdicts"]["with"]["path"].splitlines()
    for name, folder in (
        ("Measure coverage", "coverage-executions"),
        ("Measure the translation matrix", "pairs-executions"),
    ):
        env = steps[name]["env"]
        assert env[executed.EXECUTIONS] == "${{ runner.temp }}/" + folder
        assert (
            env[executed.EXECUTIONS] in restored and env[executed.EXECUTIONS] in saved
        )

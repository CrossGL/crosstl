"""The smoke-tier canary covers every complete MLX family and cannot pass vacuously."""

import importlib
import os
import subprocess
import sys
import xml.etree.ElementTree as ElementTree
from pathlib import Path
from types import SimpleNamespace

import pytest

from demos.integrations.mlx import smoke_canary

ROOT = Path(__file__).resolve().parents[4]
KERNELS = ROOT / smoke_canary.KERNEL_TESTS


def _modules():
    return {node.split("::")[0] for node in smoke_canary.node_ids()}


def _without_corpus():
    environment = {
        name: value
        for name, value in os.environ.items()
        if name != smoke_canary.ROOT_ENV
        and not name.startswith(("CROSTL_REQUIRE_MLX_", "PYTEST_XDIST"))
    }
    environment["PYTHONPATH"] = str(ROOT)
    return environment


def test_canary_runs_every_complete_family_proof():
    complete = {
        f"{smoke_canary.KERNEL_TESTS}/{path.name}"
        for path in KERNELS.glob("test_*_complete_*.py")
    }
    unary_metal = f"{smoke_canary.KERNEL_TESTS}/test_unary_native_loader.py"
    assert complete
    assert _modules() == complete | {unary_metal}
    for family in smoke_canary.FAMILIES:
        assert family.workloads, family.name
        expected = ("metal",) if family.name == "quantized" else smoke_canary.TARGETS
        assert family.targets == expected, family.name


def test_canary_workloads_exist_in_every_target_proof():
    expected = smoke_canary.node_ids()
    assert len(set(expected)) == len(expected)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            "-p",
            "no:cacheprovider",
        ]
        + list(expected),
        cwd=ROOT,
        env=_without_corpus(),
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
    collected = {line for line in result.stdout.splitlines() if "::" in line}
    assert collected == set(expected)


def test_canary_fails_rather_than_skips_without_a_corpus(tmp_path):
    report = tmp_path / "canary.xml"
    environment = _without_corpus() | smoke_canary.environment()
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            "-p",
            "no:xdist",
            f"--junitxml={report}",
            *smoke_canary.node_ids(),
        ],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 1, result.stdout[-2000:]
    cases = ElementTree.parse(report).getroot().iter("testcase")
    outcomes = {
        f"{case.get('classname')}.{case.get('name')}": [child.tag for child in case]
        for case in cases
    }
    assert len(outcomes) == len(smoke_canary.node_ids())
    assert all(tags == ["failure"] for tags in outcomes.values()), outcomes


def test_canary_main_requires_a_corpus(monkeypatch, capsys):
    monkeypatch.delenv(smoke_canary.ROOT_ENV, raising=False)
    assert smoke_canary.main([]) == 2
    assert smoke_canary.CORPUS_COMMIT in capsys.readouterr().err


def _report(nodeid, when, outcome):
    return SimpleNamespace(
        nodeid=nodeid,
        when=when,
        outcome=outcome,
        passed=outcome == "passed",
        failed=outcome == "failed",
        skipped=outcome == "skipped",
    )


@pytest.mark.parametrize("skipped", [False, True])
def test_canary_main_rejects_a_pass_that_skipped_a_test(
    monkeypatch, capsys, tmp_path, skipped
):
    """Pytest exits 0 when a test skips; the canary must not."""
    expected = smoke_canary.node_ids()

    def fake_main(arguments, plugins):
        assert arguments[-len(expected) :] == list(expected)
        assert "--junitxml=out.xml" in arguments
        (outcomes,) = plugins
        for index, node in enumerate(expected):
            if skipped and index == 0:
                outcomes.pytest_runtest_logreport(_report(node, "setup", "skipped"))
                continue
            for when in ("setup", "call", "teardown"):
                outcomes.pytest_runtest_logreport(_report(node, when, "passed"))
        return 0

    monkeypatch.setenv(smoke_canary.ROOT_ENV, str(tmp_path))
    # Record every variable main sets, so monkeypatch restores the environment.
    for name in smoke_canary.environment():
        monkeypatch.setenv(name, "")
    monkeypatch.setattr(smoke_canary.pytest, "main", fake_main)
    assert smoke_canary.main(["--junitxml=out.xml"]) == int(skipped)
    error = capsys.readouterr().err
    assert (f"{expected[0]}: skipped in setup" in error) is skipped
    assert os.environ[smoke_canary.PIN_MODE_ENV] == "report"


def test_canary_reports_pins_and_requires_every_target_proof():
    environment = smoke_canary.environment()
    assert environment.pop(smoke_canary.PIN_MODE_ENV) == "report"
    sources = "".join(
        (ROOT / module).read_text(encoding="utf-8") for module in _modules()
    )
    assert environment
    for name, value in environment.items():
        assert value == "1"
        assert f'"{name}"' in sources, name


def test_canary_corpus_matches_every_family_proof():
    for module in sorted(_modules()):
        name = module.removesuffix(".py").replace("/", ".")
        assert importlib.import_module(name).MLX_COMMIT == smoke_canary.CORPUS_COMMIT


def test_canary_problems_reject_skips_failures_and_missing_tests():
    expected = ("a", "b", "c")
    assert smoke_canary.problems(expected, {"a", "b", "c"}, {}) == []
    assert smoke_canary.problems(
        expected, {"a"}, {"b": "skipped in setup", "x": "failed in call"}
    ) == [
        "b: skipped in setup",
        "x: failed in call",
        "c: did not run",
    ]
    assert smoke_canary.problems(expected, {"a", "b", "c", "d"}, {}) == [
        "d: not a canary test"
    ]

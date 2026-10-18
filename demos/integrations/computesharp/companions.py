#!/usr/bin/env python3
"""Build, export, check and run the companion upstream test suites.

``ComputeSharp.Tests`` has its own workflow steps. Every other ported project
of ``test_projects.json`` is a companion suite, which this module runs in the
same job and writes under ``<output>/<project>``:

``baseline`` (Windows) builds the original project as the upstream
``dotnet.yml`` step does (``-c Release -p:Platform=x64``, with the audit
exceptions of ``baseline-audit.props``), exports its cases and generated HLSL
with the discovery tool, requires the frozen lock, runs it (``dotnet test``, or
``program_runner.py`` for a program, which the upstream step starts with
``dotnet run``), requires the assembly unchanged and accounts every case with
``baseline.py`` into ``baseline.json``.

``translated`` (macOS, Linux) builds the adapted project against the portable
runtime, exports and checks it the same way and runs ``run_tests.py --suite``
with the negative controls into ``run/``.

Every suite runs even when an earlier one fails, so the evidence of all of
them is uploaded; ``companions.json`` records each step and the command exits
non-zero when any suite failed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import baseline  # noqa: E402
import coverage_contract  # noqa: E402
import program_runner  # noqa: E402
import suites  # noqa: E402

DISCOVERY_TOOL = HERE / "discovery" / "Discovery.csproj"
AUDIT_PROPS = HERE / "baseline-audit.props"
PORTABLE_TARGETS = HERE / "runtime" / "Portable.targets"
DIRECTX_PROFILE = "directx-warp"
BUILD_DEADLINE_SECONDS = 20 * 60
TEST_DEADLINE_SECONDS = 20 * 60
LOG_TAIL_LINES = 30


class StepError(RuntimeError):
    """One step of a companion suite failed."""


def companion_suites(inventory: dict | None = None) -> list[suites.Suite]:
    """Every ported project except the main suite, in name order."""

    return [
        item
        for name, item in suites.ported_suites(inventory).items()
        if name != suites.MAIN_SUITE
    ]


def _tail(text: str) -> str:
    return "\n".join(text.strip().splitlines()[-LOG_TAIL_LINES:])


def run_command(
    command: list[str], log: Path, *, deadline: float, check: bool = True
) -> subprocess.CompletedProcess:
    """Run ``command``, keep its output in ``log`` and raise when it fails."""

    log.parent.mkdir(parents=True, exist_ok=True)
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=deadline,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        log.write_text("$ " + " ".join(command) + "\n(deadline exceeded)\n")
        raise StepError(f"{command[0]} {command[1]} exceeded {deadline:.0f}s") from exc
    output = (completed.stdout or "") + (completed.stderr or "")
    log.write_text("$ " + " ".join(command) + "\n" + output, encoding="utf-8")
    if check and completed.returncode != 0:
        raise StepError(
            f"{' '.join(command[:3])} exited with {completed.returncode}:\n{_tail(output)}"
        )
    return completed


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _discover(assembly: Path, directory: Path, logs: Path, runner: Callable) -> None:
    runner(
        [
            "dotnet",
            "run",
            "--project",
            str(DISCOVERY_TOOL),
            "-c",
            "Release",
            "--",
            str(assembly),
            str(directory / "discovery"),
        ],
        logs / "discovery.log",
        deadline=BUILD_DEADLINE_SECONDS,
    )


def _check_lock(item: suites.Suite, directory: Path) -> dict[str, Any]:
    checked = coverage_contract.check(directory / "discovery", item.lock, item.contract)
    return {"caseCount": checked["caseCount"], "totals": checked["totals"]}


def baseline_suite(
    item: suites.Suite,
    source: Path,
    directory: Path,
    *,
    runner: Callable = run_command,
    program: Callable = program_runner.run,
) -> dict[str, Any]:
    """Run one companion suite against the original Direct3D 12 runtime."""

    directory.mkdir(parents=True, exist_ok=True)
    logs = directory / "logs"
    project = source / item.project
    properties = ["-c", "Release", "-p:Platform=x64"]
    runner(
        [
            "dotnet",
            "build",
            str(project),
            *properties,
            "-p:CrossTLTrustedCorpusBaseline=true",
            f"-p:CustomAfterMicrosoftCommonTargets={AUDIT_PROPS}",
        ],
        logs / "build.log",
        deadline=BUILD_DEADLINE_SECONDS,
    )
    completed = runner(
        [
            "dotnet",
            "msbuild",
            str(project),
            "-getProperty:TargetPath",
            "-p:Configuration=Release",
            "-p:Platform=x64",
        ],
        logs / "target-path.log",
        deadline=BUILD_DEADLINE_SECONDS,
    )
    assembly = Path(completed.stdout.strip())
    if not assembly.is_file():
        raise StepError(f"The {item.name} assembly {assembly} is missing")
    _discover(assembly, directory, logs, runner)
    report: dict[str, Any] = {"coverage": _check_lock(item, directory)}
    trx_dir = directory / "trx"
    if item.kind == "program":
        try:
            program(
                assembly,
                directory / "discovery",
                trx_dir,
                log=logs / "test-run.log",
                deadline=TEST_DEADLINE_SECONDS,
                trx_name="baseline.trx",
            )
        except program_runner.ProgramError as exc:
            raise StepError(str(exc)) from exc
    else:
        # A failing case is accounted below, so the exit status is not checked.
        runner(
            [
                "dotnet",
                "test",
                str(project),
                *properties,
                "--no-build",
                "--logger",
                "trx;LogFileName=baseline.trx",
                "--results-directory",
                str(trx_dir),
            ],
            logs / "test-run.log",
            deadline=TEST_DEADLINE_SECONDS,
            check=False,
        )
    discovered = baseline.load_discovery(directory / "discovery")
    if _sha256(assembly) != discovered["assembly"]["sha256"]:
        raise StepError(f"The {item.name} assembly changed during execution")
    accounting = baseline.verify_results(
        directory / "discovery",
        trx_dir / "baseline.trx",
        DIRECTX_PROFILE,
        item.contract,
    )
    (directory / "baseline.json").write_text(
        json.dumps(accounting, indent=2) + "\n", encoding="utf-8"
    )
    report["accounting"] = {
        key: accounting[key]
        for key in ("expected", "requiredCasesPassed", "requiredGpuCasesPassed")
    }
    return report


def translated_suite(
    item: suites.Suite,
    adapted: Path,
    directory: Path,
    *,
    profile: str,
    runner: Callable = run_command,
) -> dict[str, Any]:
    """Run one companion suite on a translated target with its controls."""

    logs = directory / "logs"
    runner(
        [
            "dotnet",
            "build",
            str(adapted / item.project),
            "-c",
            "Release",
            "-p:CrossTLTrustedCorpusBaseline=true",
            f"-p:CustomAfterMicrosoftCommonTargets={PORTABLE_TARGETS}",
        ],
        logs / "build.log",
        deadline=BUILD_DEADLINE_SECONDS,
    )
    _discover(adapted / item.assembly, directory, logs, runner)
    report: dict[str, Any] = {"coverage": _check_lock(item, directory)}
    runner(
        [
            sys.executable,
            str(HERE / "run_tests.py"),
            "--suite",
            item.name,
            "--adapted",
            str(adapted),
            "--discovery",
            str(directory / "discovery"),
            "--profile",
            profile,
            "--work",
            str(directory / "run"),
            "--prebuilt-tests",
            "--negative-controls",
        ],
        logs / "run-tests.log",
        deadline=2 * TEST_DEADLINE_SECONDS,
    )
    results = json.loads((directory / "run" / "results.json").read_text("utf-8"))
    accounting = results.get("accounting")
    report["run"] = {
        "status": results.get("status"),
        "translation": results.get("translation"),
        "requiredCasesPassed": (
            accounting.get("requiredCasesPassed")
            if isinstance(accounting, dict)
            else None
        ),
    }
    return report


def run_all(
    selected: list[suites.Suite], output: Path, run_one: Callable
) -> dict[str, Any]:
    """Run every suite, record each outcome and write ``companions.json``."""

    output.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"schemaVersion": 1, "suites": {}}
    for item in selected:
        directory = output / item.name
        directory.mkdir(parents=True, exist_ok=True)
        try:
            entry = {"status": "passed", **run_one(item, directory)}
        except (StepError, OSError, ValueError, KeyError) as exc:
            entry = {"status": "failed", "error": str(exc)}
        report["suites"][item.name] = entry
    report["status"] = (
        "passed"
        if selected and all(e["status"] == "passed" for e in report["suites"].values())
        else "failed"
    )
    (output / "companions.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    original = sub.add_parser("baseline", help="original Direct3D 12 runs (Windows)")
    original.add_argument("--source", required=True, type=Path)
    translated = sub.add_parser("translated", help="translated runs with controls")
    translated.add_argument("--adapted", required=True, type=Path)
    translated.add_argument("--profile", required=True)
    for command in (original, translated):
        command.add_argument("--output", required=True, type=Path)
        command.add_argument("--suite", action="append", help="limit to one suite")
    args = parser.parse_args(argv)
    selected = companion_suites()
    if args.suite:
        unknown = set(args.suite) - {item.name for item in selected}
        if unknown:
            parser.error(f"not companion suites: {', '.join(sorted(unknown))}")
        selected = [item for item in selected if item.name in args.suite]
    if args.command == "baseline":
        source = args.source.resolve()

        def run_one(item, directory):
            return baseline_suite(item, source, directory)

    else:
        adapted = args.adapted.resolve()

        def run_one(item, directory):
            return translated_suite(item, adapted, directory, profile=args.profile)

    report = run_all(selected, args.output.resolve(), run_one)
    for name, entry in report["suites"].items():
        print(f"{name}: {entry['status']}")
        if entry.get("error"):
            print("  " + entry["error"].replace("\n", "\n  "))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

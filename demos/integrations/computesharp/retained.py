#!/usr/bin/env python3
"""Run the retained Windows-specific and CPU-only upstream test projects.

These projects test what the port does not translate: Direct3D 12 device
lifetime (``ComputeSharp.Tests.DeviceLost``) and the Roslyn source generators
(``ComputeSharp.Tests.SourceGenerators``). They never run on a translated
target and never count as translated GPU execution; their original results
are retained separately, on the ``directx-warp`` profile of their own frozen
contract and lock.

``run`` (Windows) builds each pristine project in the configuration of its
upstream steps (with the audit exceptions of ``baseline-audit.props``),
exports its cases with the discovery tool (``--host-only`` for a project
without generated shaders), requires the frozen lock and runs every recorded
``dotnet test`` command of ``test_projects.json``:

- a run that reproduces an upstream step verbatim but is not accounted must
  exit successfully without a failing result; the number of cases it selected
  is recorded (the upstream ``TestCategory`` filters of DeviceLost select none);
- the accounted runs together must report every discovered case exactly once.
  Their TRX files are merged into ``trx/baseline.trx``, which ``baseline.py``
  accounts into ``baseline.json``.

Every project runs even when an earlier one fails; ``retained.json`` records
each outcome and the command exits non-zero when any project failed.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Callable

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import baseline  # noqa: E402
import companions  # noqa: E402
import suites  # noqa: E402

TRX_NAMESPACE = "http://microsoft.com/schemas/VisualStudio/TeamTest/2010"
TRX = {"t": TRX_NAMESPACE}
NO_MATCH = "No test matches the given testcase filter"
COMPLETED = {"Passed", "Completed", "Failed"}
MERGED_SECTIONS = ("Results", "TestDefinitions", "TestEntries")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _results(root: ET.Element) -> list[ET.Element]:
    return root.findall("t:Results/t:UnitTestResult", TRX)


def merge_trx(paths: list[Path], output: Path) -> None:
    """Write the results of several MSTest runs as one TRX file.

    Definitions, entries and results are concatenated; counters are summed;
    the run outcome is the first incomplete outcome, else ``Failed`` when any
    run failed, else ``Completed``. ``baseline.py`` rejects a test defined by
    two runs, so overlapping runs cannot hide a missing case.
    """

    if not paths:
        raise ValueError("No TRX files to merge")
    roots = [ET.parse(path).getroot() for path in paths]
    merged = copy.deepcopy(roots[0])
    for section in MERGED_SECTIONS:
        target = merged.find(f"t:{section}", TRX)
        if target is None:
            target = ET.SubElement(merged, f"{{{TRX_NAMESPACE}}}{section}")
        for root in roots[1:]:
            source = root.find(f"t:{section}", TRX)
            for child in [] if source is None else list(source):
                target.append(copy.deepcopy(child))
    outcomes = []
    totals: dict[str, int] = {}
    for root in roots:
        summary = root.find("t:ResultSummary", TRX)
        outcomes.append(summary.get("outcome") if summary is not None else None)
        counters = None if summary is None else summary.find("t:Counters", TRX)
        for key, value in ({} if counters is None else counters.attrib).items():
            if value.isdigit():
                totals[key] = totals.get(key, 0) + int(value)
    incomplete = [outcome for outcome in outcomes if outcome not in COMPLETED]
    if incomplete:
        outcome = str(incomplete[0])
    else:
        outcome = "Failed" if "Failed" in outcomes else "Completed"
    summary = merged.find("t:ResultSummary", TRX)
    if summary is None:
        summary = ET.SubElement(merged, f"{{{TRX_NAMESPACE}}}ResultSummary")
    summary.set("outcome", outcome)
    counters = summary.find("t:Counters", TRX)
    if counters is not None:
        for key, value in totals.items():
            counters.set(key, str(value))
    ET.register_namespace("", TRX_NAMESPACE)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Bytes, not text: the qualification job on Linux merges the same runs
    # again and compares bytes, so Windows must not translate line endings.
    output.write_bytes(ET.tostring(merged, encoding="utf-8", xml_declaration=True))


def _test_command(
    project: Path, configuration: str, run: dict, directory: Path
) -> list[str]:
    command = ["dotnet", "test", str(project), "-c", configuration, "--no-build"]
    if run.get("filter"):
        command += ["--filter", run["filter"]]
    return command + [
        "--logger",
        f"trx;LogFileName={run['name']}.trx",
        "--results-directory",
        str(directory),
    ]


def _execute_run(
    project: Path, configuration: str, run: dict, directory: Path, runner: Callable
) -> dict[str, Any]:
    """Run one recorded command; an unaccounted run must succeed by itself."""

    runs = directory / "trx" / "runs"
    completed = runner(
        _test_command(project, configuration, run, runs),
        directory / "logs" / f"test-{run['name']}.log",
        deadline=companions.TEST_DEADLINE_SECONDS,
        check=False,
    )
    trx = runs / f"{run['name']}.trx"
    output = (completed.stdout or "") + (completed.stderr or "")
    entry: dict[str, Any] = {
        "name": run["name"],
        "upstreamStep": run.get("upstreamStep"),
        "filter": run.get("filter"),
        "accounted": run["accounted"],
        "exitCode": completed.returncode,
        "trx": trx.name if trx.is_file() else None,
    }
    if trx.is_file():
        results = _results(ET.parse(trx).getroot())
        entry["selectedCases"] = len(results)
        entry["failedCases"] = sum(r.get("outcome") == "Failed" for r in results)
        entry["trxSHA256"] = _sha256(trx)
    elif run["accounted"]:
        raise companions.StepError(f"Run {run['name']} wrote no TRX results")
    elif NO_MATCH in output:
        entry["selectedCases"] = 0
        entry["failedCases"] = 0
    else:
        raise companions.StepError(f"Run {run['name']} wrote no TRX results")
    if not run["accounted"] and (completed.returncode != 0 or entry["failedCases"]):
        raise companions.StepError(
            f"Upstream step run {run['name']} failed ({completed.returncode}):\n"
            + companions._tail(output)
        )
    return entry


def run_suite(
    item: suites.Suite,
    source: Path,
    directory: Path,
    *,
    runner: Callable = companions.run_command,
) -> dict[str, Any]:
    """Build, export, check, run and account one retained project."""

    retention = item.retention or {}
    configuration = retention["configuration"]
    directory.mkdir(parents=True, exist_ok=True)
    logs = directory / "logs"
    project = source / item.project
    runner(
        [
            "dotnet",
            "build",
            str(project),
            "-c",
            configuration,
            "-p:CrossTLTrustedCorpusBaseline=true",
            f"-p:CustomAfterMicrosoftCommonTargets={companions.AUDIT_PROPS}",
        ],
        logs / "build.log",
        deadline=companions.BUILD_DEADLINE_SECONDS,
    )
    completed = runner(
        [
            "dotnet",
            "msbuild",
            str(project),
            "-getProperty:TargetPath",
            f"-p:Configuration={configuration}",
        ],
        logs / "target-path.log",
        deadline=companions.BUILD_DEADLINE_SECONDS,
    )
    assembly = Path(completed.stdout.strip())
    if not assembly.is_file():
        raise companions.StepError(f"The {item.name} assembly {assembly} is missing")
    companions._discover(
        assembly,
        directory,
        logs,
        runner,
        host_only=retention["hostOnly"],
    )
    report: dict[str, Any] = {
        "scope": item.scope,
        "configuration": configuration,
        "coverage": companions._check_lock(item, directory),
    }
    report["runs"] = [
        _execute_run(project, configuration, run, directory, runner)
        for run in retention["runs"]
    ]
    accounted = [
        directory / "trx" / "runs" / entry["trx"]
        for entry in report["runs"]
        if entry["accounted"]
    ]
    merged = directory / "trx" / "baseline.trx"
    merge_trx(accounted, merged)
    discovered = baseline.load_discovery(directory / "discovery")
    if _sha256(assembly) != discovered["assembly"]["sha256"]:
        raise companions.StepError(f"The {item.name} assembly changed during execution")
    accounting = baseline.verify_results(
        directory / "discovery", merged, companions.DIRECTX_PROFILE, item.contract
    )
    (directory / "baseline.json").write_text(
        json.dumps(accounting, indent=2) + "\n", encoding="utf-8"
    )
    report["accounting"] = {
        key: accounting[key]
        for key in ("expected", "requiredCasesPassed", "requiredGpuCasesPassed")
    }
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    command = sub.add_parser("run", help="original runs on Windows")
    command.add_argument("--source", required=True, type=Path)
    command.add_argument("--output", required=True, type=Path)
    command.add_argument("--suite", action="append", help="limit to one project")
    args = parser.parse_args(argv)
    available = suites.retained_suites()
    unknown = sorted(set(args.suite or []) - set(available))
    if unknown:
        parser.error("not a retained project: " + ", ".join(unknown))
    selected = [
        item for name, item in available.items() if not args.suite or name in args.suite
    ]
    report = companions.run_all(
        selected,
        args.output,
        lambda item, directory: run_suite(item, args.source, directory),
        report_name="retained.json",
    )
    for name, entry in report["suites"].items():
        print(
            f"{name}: {entry['status']}"
            + (f" ({entry['error']})" if "error" in entry else "")
        )
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

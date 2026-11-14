#!/usr/bin/env python3
"""Run an upstream test project that is a program and record its outcome as TRX.

The upstream workflow runs ``ComputeSharp.Tests.GlobalStatements`` with
``dotnet run``. The program dispatches a compute shader, evaluates the same
function on the CPU and throws when the two differ, so its exit status is the
upstream assertion. Discovery exports the program's entry point as the
project's one case (``projectKind: program``). This runner executes exactly the
discovered assembly, requires a zero exit status for a pass and writes the
outcome as a one-case TRX file, so the case is accounted, attributed in the
dispatch evidence and checked by negative controls exactly as MSTest cases are.
The TRX names this runner as its adapter; it is never produced by MSTest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import uuid
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import baseline  # noqa: E402

TRX_NAMESPACE = baseline.TRX["t"]
ADAPTER = "executor://crosstl-computesharp/program"
# MSTest's unit test type, so TRX readers treat the case like any other test.
UNIT_TEST_TYPE = "13cdc9d9-ddb5-4fa4-a97d-d965ccfc6d4b"
PROGRAM_ENVIRONMENT = "CROSSTL_COMPUTESHARP_PROGRAM"
DEFAULT_DEADLINE_SECONDS = 5 * 60
MESSAGE_LINES = 40


class ProgramError(RuntimeError):
    """The program could not be run as its discovered case."""


def entry_case(discovery_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return the discovery report and the one entry-point case of a program."""

    report = baseline.load_discovery(discovery_dir)
    if report.get("projectKind") != "program":
        raise ProgramError(f"{discovery_dir} is not the discovery of a program")
    if len(report["tests"]) != 1:
        raise ProgramError("A program has exactly one case, its entry point")
    return report, report["tests"][0]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stamp(moment: datetime) -> str:
    return moment.isoformat(timespec="milliseconds")


def write_trx(
    path: Path,
    case: dict[str, Any],
    *,
    assembly: Path,
    outcome: str,
    message: str,
    output: str,
    started: datetime,
    finished: datetime,
) -> Path:
    """Write the result of the program's one case as a TRX file."""

    if outcome not in {"Passed", "Failed"}:
        raise ValueError(f"Unsupported program outcome {outcome}")
    ET.register_namespace("", TRX_NAMESPACE)

    def element(parent: ET.Element | None, tag: str, **attributes: str) -> ET.Element:
        name = f"{{{TRX_NAMESPACE}}}{tag}"
        if parent is None:
            return ET.Element(name, attributes)
        return ET.SubElement(parent, name, attributes)

    owner, name = case["method"].rsplit(".", 1)
    test_id = str(uuid.uuid5(uuid.NAMESPACE_URL, "computesharp-case:" + case["id"]))
    execution_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{test_id}:{_stamp(started)}"))
    run = element(
        None,
        "TestRun",
        id=str(uuid.uuid5(uuid.NAMESPACE_URL, execution_id)),
        name=f"{ADAPTER} {assembly.name}",
    )
    element(
        run,
        "Times",
        creation=_stamp(started),
        start=_stamp(started),
        finish=_stamp(finished),
    )
    results = element(run, "Results")
    result = element(
        results,
        "UnitTestResult",
        executionId=execution_id,
        testId=test_id,
        testName=case["displayName"],
        computerName=platform.node(),
        duration=str(finished - started),
        startTime=_stamp(started),
        endTime=_stamp(finished),
        testType=UNIT_TEST_TYPE,
        outcome=outcome,
    )
    stream = element(result, "Output")
    element(stream, "StdOut").text = output
    if outcome == "Failed":
        error = element(stream, "ErrorInfo")
        element(error, "Message").text = message
        element(error, "StackTrace").text = ""
    definitions = element(run, "TestDefinitions")
    unit = element(
        definitions,
        "UnitTest",
        name=case["displayName"],
        storage=str(assembly),
        id=test_id,
    )
    element(unit, "Execution", id=execution_id)
    element(
        unit,
        "TestMethod",
        codeBase=str(assembly),
        adapterTypeName=ADAPTER,
        className=owner,
        name=name,
    )
    summary = element(
        run, "ResultSummary", outcome="Completed" if outcome == "Passed" else "Failed"
    )
    element(
        summary,
        "Counters",
        total="1",
        executed="1",
        passed="1" if outcome == "Passed" else "0",
        failed="0" if outcome == "Passed" else "1",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    # Bytes, so the TRX is the same on every platform.
    path.write_bytes(ET.tostring(run, encoding="utf-8", xml_declaration=True))
    return path


def run(
    assembly: Path,
    discovery_dir: Path,
    results_dir: Path,
    *,
    log: Path,
    env: dict[str, str] | None = None,
    deadline: float = DEFAULT_DEADLINE_SECONDS,
    trx_name: str = "results.trx",
) -> Path:
    """Run the discovered program once and write its TRX result."""

    report, case = entry_case(discovery_dir)
    if not assembly.is_file():
        raise ProgramError(f"The program assembly {assembly} does not exist")
    if _sha256(assembly) != report["assembly"]["sha256"]:
        raise ProgramError(f"{assembly} is not the assembly that discovery exported")
    environment = dict(os.environ if env is None else env)
    environment[PROGRAM_ENVIRONMENT] = report["assembly"]["name"]
    command = ["dotnet", str(assembly)]
    started = datetime.now(timezone.utc)
    try:
        completed = subprocess.run(
            command,
            env=environment,
            capture_output=True,
            text=True,
            timeout=deadline,
            check=False,
        )
        returncode = completed.returncode
        output = (completed.stdout or "") + (completed.stderr or "")
    except subprocess.TimeoutExpired as exc:
        returncode = None
        partial = exc.stdout or b""
        output = (
            partial.decode("utf-8", "replace")
            if isinstance(partial, bytes)
            else partial
        )
    finished = datetime.now(timezone.utc)
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text("$ " + " ".join(command) + "\n" + output, encoding="utf-8")
    if _sha256(assembly) != report["assembly"]["sha256"]:
        raise ProgramError(f"{assembly} changed while the program ran")
    if returncode == 0:
        outcome, message = "Passed", ""
    elif returncode is None:
        outcome = "Failed"
        message = (
            f"The program exceeded its deadline of {deadline:.0f}s and was stopped."
        )
    else:
        tail = "\n".join(output.strip().splitlines()[-MESSAGE_LINES:])
        outcome = "Failed"
        message = f"The program exited with status {returncode}.\n{tail}"
    results_dir.mkdir(parents=True, exist_ok=True)
    return write_trx(
        results_dir / trx_name,
        case,
        assembly=assembly,
        outcome=outcome,
        message=message,
        output=output,
        started=started,
        finished=finished,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assembly", required=True, type=Path)
    parser.add_argument("--discovery", required=True, type=Path)
    parser.add_argument("--results-directory", required=True, type=Path)
    parser.add_argument("--trx-name", default="results.trx")
    parser.add_argument("--log", type=Path)
    parser.add_argument("--deadline", type=float, default=DEFAULT_DEADLINE_SECONDS)
    args = parser.parse_args(argv)
    try:
        trx = run(
            args.assembly.resolve(),
            args.discovery,
            args.results_directory,
            log=args.log or args.results_directory / "program.log",
            deadline=args.deadline,
            trx_name=args.trx_name,
        )
    except (OSError, ValueError, ProgramError) as error:
        parser.exit(1, str(error) + "\n")
    outcome = ET.parse(trx).getroot().find("t:Results/t:UnitTestResult", baseline.TRX)
    sys.stdout.write(
        json.dumps({"trx": str(trx), "outcome": outcome.get("outcome")}) + "\n"
    )
    # Like dotnet test: the TRX records the failure and the exit status reports it.
    return 0 if outcome.get("outcome") == "Passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

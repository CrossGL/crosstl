#!/usr/bin/env python3
"""Run the upstream ComputeSharp tests against a translated backend and account them.

The runner builds the native backend library, builds the adapted upstream test
project against the portable runtime, translates the exported HLSL with CrossTL,
checks with ``exactness.py`` that the target compiler keeps the translated
arithmetic as written, runs ``dotnet test`` without the Windows-specific methods
of the coverage contract and accounts the TRX results with ``baseline.py`` for
the requested profile.

Inputs are an adapted upstream checkout produced by ``prepare_upstream.py`` and a
discovery export of the test assembly built from that checkout. Every step writes
its log under ``--work``; the final report is ``results.json`` and the process
exits non-zero unless accounting matches the frozen contract and every artifact
is exact.
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import json
import os
import platform
import shutil
import signal
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))

import baseline  # noqa: E402
import collect_outputs  # noqa: E402
import evidence as evidence_module  # noqa: E402
import exactness  # noqa: E402

TEST_PROJECT = Path("tests/ComputeSharp.Tests/ComputeSharp.Tests.csproj")
TEST_ASSEMBLY = Path(
    "tests/ComputeSharp.Tests/bin/Release/net10.0/ComputeSharp.Tests.dll"
)
NATIVE_SOURCES = {
    "metal": ("csbackend_metal.m", "libcsbackend_metal.dylib"),
    "opengl": ("csbackend_gl.c", "libcsbackend_gl.so"),
}
FAULTS = (
    "omit-dispatch",
    "swap-bindings",
    "perturb-constants",
    "perturb-unorm-scales",
    "corrupt-readback",
    "corrupt-guard",
)
CONTROLS = HERE / "negative_controls.json"
IMAGE_CHECKS = HERE / "image_checks.json"
# Deadlines. A hung dispatch or test host fails the run instead of holding the
# job until its timeout: MSTest stops the test host when one case runs longer
# than TEST_CASE_DEADLINE and records the aborted run, and the process group of
# a ``dotnet test`` run that exceeds its run deadline is stopped. The slowest
# case takes about 10 s and a complete run about 3 to 9 minutes.
TEST_CASE_DEADLINE = "5m"
RUN_DEADLINE_SECONDS = 25 * 60
CONTROL_RUN_DEADLINE_SECONDS = 4 * 60
DEADLINES = {
    "testCase": TEST_CASE_DEADLINE,
    "runSeconds": RUN_DEADLINE_SECONDS,
    "controlRunSeconds": CONTROL_RUN_DEADLINE_SECONDS,
}


class DeadlineExceeded(RuntimeError):
    """A command did not finish within its deadline and was stopped."""


def stop_process_group(process: subprocess.Popen) -> None:
    """Stop a command started in its own session together with its children."""

    for stop_signal in (signal.SIGTERM, signal.SIGKILL):
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(process.pid, stop_signal)
        try:
            process.wait(timeout=10)
            return
        except subprocess.TimeoutExpired:
            continue


def run_logged(
    command: list[str], log: Path, *, deadline: float | None = None, **kwargs: Any
) -> int:
    """Run a command, tee its output into a log file and return the exit status.

    With a ``deadline`` in seconds the command runs in its own session, and the
    whole process group is stopped when it exceeds the deadline.
    """

    log.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with log.open("w", encoding="utf-8") as stream:
        stream.write("$ " + " ".join(command) + "\n")
        stream.flush()
        process = subprocess.Popen(
            command,
            stdout=stream,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=deadline is not None,
            **kwargs,
        )
        try:
            returncode = process.wait(timeout=deadline)
        except subprocess.TimeoutExpired:
            stop_process_group(process)
            stream.write(
                f"\n[deadline of {deadline:.0f}s exceeded; the command was stopped]\n"
            )
            raise DeadlineExceeded(
                f"{Path(command[0]).name} exceeded its deadline of {deadline:.0f}s;"
                f" see {log}"
            ) from None
        except BaseException:
            if deadline is not None:
                stop_process_group(process)
            else:
                process.kill()
                process.wait()
            raise
        stream.write(f"\n[exit {returncode} after {time.monotonic() - started:.1f}s]\n")
    return returncode


def build_native(target: str, output_dir: Path, log: Path) -> Path:
    source_name, library_name = NATIVE_SOURCES[target]
    source = HERE / "runtime" / "native" / source_name
    output_dir.mkdir(parents=True, exist_ok=True)
    library = output_dir / library_name
    if target == "metal":
        command = [
            "clang",
            "-fobjc-arc",
            "-O2",
            "-Wall",
            "-Werror",
            "-shared",
            "-framework",
            "Metal",
            "-framework",
            "Foundation",
            "-framework",
            "IOKit",
            "-o",
            str(library),
            str(source),
        ]
    else:
        command = [
            os.environ.get("CC", "cc"),
            "-O2",
            "-Wall",
            "-Werror",
            "-fPIC",
            "-shared",
            "-std=c11",
            "-o",
            str(library),
            str(source),
            "-lEGL",
        ]
    if run_logged(command, log) != 0:
        raise SystemExit(f"Native backend build failed; see {log}")
    return library


def excluded_methods(contract_path: Path, discovery: Path, profile: str) -> list[str]:
    """Return methods whose every case must not run on the profile."""

    from coverage_contract import classify, expected_outcome, load_contract

    contract = load_contract(contract_path)
    settings = contract["profiles"][profile]
    records = classify(contract, baseline.load_discovery(discovery))
    outcomes: dict[str, set[str]] = {}
    for record in records:
        outcomes.setdefault(record["method"], set()).add(
            expected_outcome(record, settings)
        )
    mixed = sorted(
        method
        for method, values in outcomes.items()
        if "excluded" in values and len(values) > 1
    )
    if mixed:
        raise SystemExit("Methods mix excluded and executed cases: " + ", ".join(mixed))
    return sorted(
        method for method, values in outcomes.items() if values == {"excluded"}
    )


def test_filter(methods: list[str]) -> str:
    return "&".join(f"FullyQualifiedName!={name}" for name in methods)


class _DeviceInfo(ctypes.Structure):
    """``cs_device_info`` from runtime/native/csbackend.h."""

    _fields_ = [
        ("abi_version", ctypes.c_uint32),
        ("is_hardware", ctypes.c_uint32),
        ("supports_fp64", ctypes.c_uint32),
        ("unified_memory", ctypes.c_uint32),
        ("dedicated_memory", ctypes.c_uint64),
        ("shared_memory", ctypes.c_uint64),
        ("compute_units", ctypes.c_uint32),
        ("wavefront_size", ctypes.c_uint32),
        ("max_threads_per_group", ctypes.c_uint32),
        ("max_group_size_x", ctypes.c_uint32),
        ("max_group_size_y", ctypes.c_uint32),
        ("max_group_size_z", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("name", ctypes.c_char * 256),
        ("api", ctypes.c_char * 64),
        ("driver", ctypes.c_char * 512),
    ]


def preflight(native_library: Path, settings: dict[str, Any]) -> dict[str, Any]:
    """Create the native device once and check it against the frozen profile.

    A missing device is an infrastructure blocker, and a device whose execution
    kind or optional capabilities differ from the profile would change the
    expected outcomes, so both stop the run before any test executes.
    """

    errors: list[str] = []
    report: dict[str, Any] = {"library": str(native_library)}
    try:
        library = ctypes.CDLL(str(native_library))
    except OSError as exc:
        return {**report, "status": "blocked", "errors": [f"cannot load: {exc}"]}
    library.cs_abi_version.restype = ctypes.c_uint32
    library.cs_device_create.restype = ctypes.c_int
    library.cs_device_create.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(_DeviceInfo),
        ctypes.c_char_p,
        ctypes.c_int32,
    ]
    library.cs_device_destroy.restype = None
    library.cs_device_destroy.argtypes = [ctypes.c_void_p]
    report["abiVersion"] = library.cs_abi_version()
    device = ctypes.c_void_p()
    info = _DeviceInfo()
    message = ctypes.create_string_buffer(1024)
    status = library.cs_device_create(
        ctypes.byref(device), ctypes.byref(info), message, len(message)
    )
    if status != 0:
        text = message.value.decode("utf-8", "replace")
        return {
            **report,
            "status": "blocked",
            "errors": [f"no {settings['target']} device ({status}): {text}"],
        }
    library.cs_device_destroy(device)
    report["device"] = {
        "name": info.name.decode("utf-8", "replace"),
        "api": info.api.decode("utf-8", "replace"),
        "driver": info.driver.decode("utf-8", "replace"),
        "hardware": bool(info.is_hardware),
        "fp64": bool(info.supports_fp64),
        "maxThreadsPerGroup": info.max_threads_per_group,
    }
    hardware = settings.get("execution") == "hardware"
    if report["device"]["hardware"] is not hardware:
        errors.append(
            f"the device executes in {'hardware' if info.is_hardware else 'software'}, "
            f"but the profile requires {settings.get('execution')} execution"
        )
    fp64 = settings["capabilities"].get("fp64")
    if report["device"]["fp64"] is not fp64:
        errors.append(
            f"the device reports fp64={bool(info.supports_fp64)}, "
            f"but the profile freezes fp64={fp64}"
        )
    if info.abi_version != report["abiVersion"]:
        errors.append("the device reports a different ABI version than the library")
    report["status"] = "failed" if errors else "passed"
    report["errors"] = errors
    return report


def combine_filters(*parts: str | None) -> str | None:
    present = [part for part in parts if part]
    if not present:
        return None
    if len(present) == 1:
        return present[0]
    return "&".join(f"({part})" for part in present)


def run_tests(
    *,
    adapted: Path,
    artifacts: Path,
    target: str,
    native_library: Path,
    excluded: list[str],
    results_dir: Path,
    evidence: Path | None,
    fault: str | None,
    extra_filter: str | None,
    log: Path,
    capture: Path | None = None,
    capture_cases: list[str] | None = None,
    deadline: float = RUN_DEADLINE_SECONDS,
) -> Path:
    if results_dir.exists():
        shutil.rmtree(results_dir)
    results_dir.mkdir(parents=True)
    trx_name = "results.trx"
    filter_text = combine_filters(test_filter(excluded), extra_filter)
    environment = dict(os.environ)
    environment.update(
        {
            "CROSSTL_COMPUTESHARP_BACKEND": target,
            "CROSSTL_COMPUTESHARP_ARTIFACTS": str(artifacts),
            "CROSSTL_COMPUTESHARP_NATIVE_LIBRARY": str(native_library),
        }
    )
    for name in (
        "CROSSTL_COMPUTESHARP_EVIDENCE",
        "CROSSTL_COMPUTESHARP_FAULT",
        "CROSSTL_COMPUTESHARP_CAPTURE",
        "CROSSTL_COMPUTESHARP_CAPTURE_CASES",
    ):
        environment.pop(name, None)
    if capture is not None and capture_cases:
        if capture.exists():
            shutil.rmtree(capture)
        environment["CROSSTL_COMPUTESHARP_CAPTURE"] = str(capture)
        environment["CROSSTL_COMPUTESHARP_CAPTURE_CASES"] = ";".join(capture_cases)
    if evidence is not None:
        environment["CROSSTL_COMPUTESHARP_EVIDENCE"] = str(evidence)
        if evidence.exists():
            evidence.unlink()
    if fault is not None:
        environment["CROSSTL_COMPUTESHARP_FAULT"] = fault
    command = [
        "dotnet",
        "test",
        str(adapted / TEST_ASSEMBLY),
        "--logger",
        f"trx;LogFileName={trx_name}",
        "--results-directory",
        str(results_dir),
        "--blame-hang-timeout",
        TEST_CASE_DEADLINE,
        "--blame-hang-dump-type",
        "none",
    ]
    if filter_text:
        command[3:3] = ["--filter", filter_text]
    run_logged(command, log, env=environment, deadline=deadline)
    trx = results_dir / trx_name
    if not trx.is_file():
        raise SystemExit(f"dotnet test did not write {trx}; see {log}")
    return trx


def build_tests(adapted: Path, log: Path) -> None:
    command = [
        "dotnet",
        "build",
        str(adapted / TEST_PROJECT),
        "-c",
        "Release",
        "-p:CrossTLTrustedCorpusBaseline=true",
        f"-p:CustomAfterMicrosoftCommonTargets={HERE / 'runtime' / 'Portable.targets'}",
    ]
    if run_logged(command, log) != 0:
        raise SystemExit(f"Test project build failed; see {log}")


def translate(discovery: Path, artifacts: Path, target: str, log: Path) -> None:
    command = [
        sys.executable,
        str(HERE / "translate.py"),
        "--discovery",
        str(discovery),
        "--output",
        str(artifacts),
        "--target",
        target,
    ]
    if target == "metal":
        command.append("--validate")
    if run_logged(command, log) != 0:
        raise SystemExit(f"Translation failed; see {log}")


def summarize_failures(trx: Path, limit: int = 60) -> list[dict[str, str]]:
    root = ET.parse(trx).getroot()
    failures = []
    for result in root.findall("t:Results/t:UnitTestResult", baseline.TRX):
        if result.get("outcome") != "Failed":
            continue
        message, _trace = baseline._trx_message(result)
        failures.append({"name": result.get("testName", ""), "message": message[:400]})
    return failures[:limit]


def run_negative_controls(
    *,
    adapted: Path,
    artifacts: Path,
    target: str,
    native_library: Path,
    discovery: Path,
    profile: str,
    contract: Path,
    work: Path,
    spec_path: Path,
    passed: Counter[str] | None,
) -> dict[str, Any]:
    """Run every frozen fault and require each of its control cases to fail."""

    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    manifest_path = artifacts / target / "manifest.json"
    results = []
    for control in spec["controls"]:
        fault = control["fault"]
        if fault not in FAULTS:
            results.append(
                {
                    "fault": fault,
                    "status": "failed",
                    "errors": [f"unknown fault {fault}"],
                }
            )
            continue
        directory = work / "controls" / fault
        selection = "|".join(
            f"FullyQualifiedName={method}" for method in control["methods"]
        )
        try:
            trx = run_tests(
                adapted=adapted,
                artifacts=artifacts,
                target=target,
                native_library=native_library,
                excluded=[],
                results_dir=directory / "trx",
                evidence=directory / "evidence.jsonl",
                fault=fault,
                extra_filter=selection,
                log=directory / "test-run.log",
                deadline=CONTROL_RUN_DEADLINE_SECONDS,
            )
        except DeadlineExceeded as exc:
            results.append({"fault": fault, "status": "failed", "errors": [str(exc)]})
            continue
        results.append(
            evidence_module.verify_control(
                control,
                trx=trx,
                evidence=directory / "evidence.jsonl",
                manifest_path=manifest_path,
                discovery_dir=discovery,
                profile=profile,
                contract_path=contract,
                passed=passed,
            )
        )
    status = (
        "passed" if all(item["status"] == "passed" for item in results) else "failed"
    )
    return {"status": status, "controls": results}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapted", required=True, type=Path)
    parser.add_argument("--discovery", required=True, type=Path)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--work", required=True, type=Path)
    parser.add_argument("--contract", type=Path, default=HERE / "coverage.json")
    parser.add_argument("--controls", type=Path, default=CONTROLS)
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument(
        "--prebuilt-tests",
        action="store_true",
        help="build only the native backend; the test project is already built",
    )
    parser.add_argument("--skip-translate", action="store_true")
    parser.add_argument("--filter", help="additional MSTest filter, for diagnosis only")
    parser.add_argument(
        "--no-evidence", action="store_true", help="do not write dispatch evidence"
    )
    parser.add_argument(
        "--negative-controls",
        action="store_true",
        help="after the complete run, run every frozen fault and require its cases to fail",
    )
    parser.add_argument(
        "--controls-only",
        action="store_true",
        help="run only the negative controls, for diagnosis",
    )
    args = parser.parse_args(argv)

    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    profile = contract["profiles"].get(args.profile)
    if profile is None or profile.get("role") != "translated-target":
        raise SystemExit(f"{args.profile} is not a translated-target profile")
    target = profile["target"]
    work = args.work.resolve()
    work.mkdir(parents=True, exist_ok=True)
    logs = work / "logs"
    artifacts = work / "artifacts"
    native_library = work / "native" / NATIVE_SOURCES[target][1]

    if not args.skip_build:
        build_native(target, native_library.parent, logs / "native-build.log")
        if not args.prebuilt_tests:
            build_tests(args.adapted, logs / "test-build.log")
    device_check = preflight(native_library, profile)
    print(json.dumps({"preflight": device_check}, indent=2))
    if device_check["status"] != "passed":
        blocked = {
            "schemaVersion": 2,
            "kind": "computesharp-translated-run",
            "profile": args.profile,
            "target": target,
            "execution": profile.get("execution"),
            "host": {"system": platform.system(), "machine": platform.machine()},
            "preflight": device_check,
            "status": device_check["status"],
            "nativeTargetExecutionQualified": False,
        }
        (work / "results.json").write_text(
            json.dumps(blocked, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        kind = (
            "infrastructure blocker"
            if device_check["status"] == "blocked"
            else "capability mismatch"
        )
        print(f"No test was run ({kind}); see {work / 'results.json'}")
        return 1
    if not args.skip_translate:
        translate(args.discovery, artifacts, target, logs / "translate.log")
    manifest_path = artifacts / target / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    report: dict[str, Any] = {
        "schemaVersion": 2,
        "kind": "computesharp-translated-run",
        "profile": args.profile,
        "target": target,
        "execution": profile.get("execution"),
        "host": {"system": platform.system(), "machine": platform.machine()},
        "translator": manifest["translator"],
        "translation": manifest["counts"],
        "preflight": device_check,
        "deadlines": DEADLINES,
    }
    status = 0
    exactness_report = exactness.analyze(
        target, artifacts, library=native_library, device=device_check.get("device")
    )
    (work / "exactness.json").write_text(
        json.dumps(exactness_report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report["arithmeticExactness"] = {
        "status": exactness_report["status"],
        "tool": exactness_report["tool"],
        "totals": exactness_report["totals"],
        "errors": exactness_report.get("errors", [])[:20],
    }
    if exactness_report["status"] != "passed":
        status = 1
    passed: Counter[str] | None = None
    if not args.controls_only:
        evidence = None if args.no_evidence else work / "evidence.jsonl"
        excluded = excluded_methods(args.contract, args.discovery, args.profile)
        image_checks = json.loads(IMAGE_CHECKS.read_text(encoding="utf-8"))
        try:
            trx = run_tests(
                adapted=args.adapted,
                artifacts=artifacts,
                target=target,
                native_library=native_library,
                excluded=excluded,
                results_dir=work / "trx",
                evidence=evidence,
                fault=None,
                extra_filter=args.filter,
                log=logs / "test-run.log",
                capture=None if evidence is None else work / "captures",
                capture_cases=image_checks["captureCases"],
            )
        except DeadlineExceeded as exc:
            report.update(
                status="failed",
                deadlineExceeded=str(exc),
                nativeTargetExecutionQualified=False,
            )
            (work / "results.json").write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            print(f"The run did not qualify: {exc}")
            return 1
        report["trx"] = str(trx)
        if args.filter:
            report["accounting"] = "skipped: a diagnostic filter was supplied"
            status = 1
        else:
            try:
                report["accounting"] = baseline.verify_results(
                    args.discovery, trx, args.profile, args.contract
                )
            except baseline.AccountingError as exc:
                report["accounting"] = {"status": "failed", "errors": exc.errors}
                status = 1
        if evidence is None:
            report["evidence"] = "skipped: evidence was disabled"
            status = 1
        else:
            try:
                report["evidence"] = evidence_module.verify(
                    evidence=evidence,
                    manifest_path=manifest_path,
                    discovery_dir=args.discovery,
                    trx=trx,
                    profile=args.profile,
                    contract_path=args.contract,
                )
            except evidence_module.EvidenceError as exc:
                report["evidence"] = {"status": "failed", "errors": exc.errors[:200]}
                status = 1
        report["failures"] = summarize_failures(trx)
        outputs = collect_outputs.collect(
            (args.adapted / TEST_ASSEMBLY).resolve().parent, work / "outputs"
        )
        report["outputs"] = {
            "directory": str(work / "outputs"),
            "outputs": len(outputs["outputs"]),
            "references": len(outputs["references"]),
        }
        passed = Counter(
            case["name"]
            for case in evidence_module.trx_cases(trx)
            if case["outcome"] == "Passed"
        )
    if args.negative_controls or args.controls_only:
        report["negativeControls"] = run_negative_controls(
            adapted=args.adapted,
            artifacts=artifacts,
            target=target,
            native_library=native_library,
            discovery=args.discovery,
            profile=args.profile,
            contract=args.contract,
            work=work,
            spec_path=args.controls,
            passed=passed,
        )
        if report["negativeControls"]["status"] != "passed":
            status = 1
    if args.controls_only:
        status = 1
        report["accounting"] = "skipped: only negative controls were run"
    report["status"] = "passed" if status == 0 else "failed"
    report["nativeTargetExecutionQualified"] = bool(
        status == 0 and args.negative_controls and not args.filter
    )
    (work / "results.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    summary = {
        key: report.get(key)
        for key in (
            "profile",
            "translation",
            "status",
            "nativeTargetExecutionQualified",
        )
    }
    accounting = report.get("accounting")
    if isinstance(accounting, dict):
        summary["accounting"] = accounting.get("status") or {
            "passed": accounting.get("requiredCasesPassed"),
            "capabilityGated": len(accounting.get("capabilityInconclusive", [])),
            "expected": accounting.get("expected"),
        }
    if isinstance(report.get("evidence"), dict):
        summary["evidence"] = report["evidence"].get("status")
    summary["arithmeticExactness"] = report["arithmeticExactness"]["status"]
    if isinstance(report.get("negativeControls"), dict):
        summary["negativeControls"] = {
            item["fault"]: item["status"]
            for item in report["negativeControls"]["controls"]
        }
    print(json.dumps(summary, indent=2))
    if status:
        print(f"The run did not qualify; see {work / 'results.json'}")
    return status


if __name__ == "__main__":
    raise SystemExit(main())

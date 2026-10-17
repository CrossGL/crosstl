#!/usr/bin/env python3
"""Run the upstream ComputeSharp tests against a translated backend and account them.

The runner builds the native backend library, builds the adapted upstream test
project against the portable runtime, translates the exported HLSL with CrossTL,
runs ``dotnet test`` without the Windows-specific methods of the coverage contract
and accounts the TRX results with ``baseline.py`` for the requested profile.

Inputs are an adapted upstream checkout produced by ``prepare_upstream.py`` and a
discovery export of the test assembly built from that checkout. Every step writes
its log under ``--work``; the final report is ``results.json`` and the process
exits non-zero unless accounting matches the frozen contract.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))

import baseline  # noqa: E402

TEST_PROJECT = Path("tests/ComputeSharp.Tests/ComputeSharp.Tests.csproj")
TEST_ASSEMBLY = Path(
    "tests/ComputeSharp.Tests/bin/Release/net10.0/ComputeSharp.Tests.dll"
)
NATIVE_SOURCES = {
    "metal": ("csbackend_metal.m", "libcsbackend_metal.dylib"),
    "opengl": ("csbackend_gl.c", "libcsbackend_gl.so"),
}
FAULTS = ("omit-dispatch", "swap-bindings", "perturb-constants", "corrupt-readback")


def run_logged(command: list[str], log: Path, **kwargs: Any) -> int:
    """Run a command, tee its output into a log file and return the exit status."""

    log.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with log.open("w", encoding="utf-8") as stream:
        stream.write("$ " + " ".join(command) + "\n")
        stream.flush()
        completed = subprocess.run(
            command, stdout=stream, stderr=subprocess.STDOUT, text=True, **kwargs
        )
        stream.write(
            f"\n[exit {completed.returncode} after {time.monotonic() - started:.1f}s]\n"
        )
    return completed.returncode


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
) -> Path:
    if results_dir.exists():
        shutil.rmtree(results_dir)
    results_dir.mkdir(parents=True)
    trx_name = "results.trx"
    filter_text = test_filter(excluded)
    if extra_filter:
        filter_text = f"({filter_text})&({extra_filter})"
    environment = dict(os.environ)
    environment.update(
        {
            "CROSSTL_COMPUTESHARP_BACKEND": target,
            "CROSSTL_COMPUTESHARP_ARTIFACTS": str(artifacts),
            "CROSSTL_COMPUTESHARP_NATIVE_LIBRARY": str(native_library),
        }
    )
    environment.pop("CROSSTL_COMPUTESHARP_EVIDENCE", None)
    environment.pop("CROSSTL_COMPUTESHARP_FAULT", None)
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
        "--filter",
        filter_text,
        "--logger",
        f"trx;LogFileName={trx_name}",
        "--results-directory",
        str(results_dir),
    ]
    run_logged(command, log, env=environment)
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapted", required=True, type=Path)
    parser.add_argument("--discovery", required=True, type=Path)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--work", required=True, type=Path)
    parser.add_argument("--contract", type=Path, default=HERE / "coverage.json")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--skip-translate", action="store_true")
    parser.add_argument("--filter", help="additional MSTest filter, for diagnosis only")
    parser.add_argument(
        "--no-evidence", action="store_true", help="do not write dispatch evidence"
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
        build_tests(args.adapted, logs / "test-build.log")
    if not args.skip_translate:
        translate(args.discovery, artifacts, target, logs / "translate.log")
    manifest = json.loads(
        (artifacts / target / "manifest.json").read_text(encoding="utf-8")
    )

    evidence = None if args.no_evidence else work / "evidence.jsonl"
    excluded = excluded_methods(args.contract, args.discovery, args.profile)
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
    )

    report: dict[str, Any] = {
        "schemaVersion": 1,
        "kind": "computesharp-translated-run",
        "profile": args.profile,
        "target": target,
        "host": {"system": platform.system(), "machine": platform.machine()},
        "translator": manifest["translator"],
        "translation": manifest["counts"],
        "trx": str(trx),
    }
    status = 0
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
    report["failures"] = summarize_failures(trx)
    (work / "results.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: report[k] for k in ("profile", "translation")}, indent=2))
    if status:
        print(f"Accounting did not match the contract; see {work / 'results.json'}")
    return status


if __name__ == "__main__":
    raise SystemExit(main())

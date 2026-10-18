#!/usr/bin/env python3
"""Verify the dispatch evidence of a translated ComputeSharp run.

The portable runtime writes one JSON object per line for the device, every pipeline
it compiles, every dispatch and every readback, each attributed to the executing
test case. This module checks that the evidence is complete and consistent with the
translated artifacts and the frozen inventory:

- the device record names the selected backend, states hardware or software
  execution as the profile requires and references the manifest that was used;
- every pipeline loads a translated artifact listed in the manifest, whose file
  content hash and thread-group size match the manifest;
- every dispatch uses a compiled pipeline, binds every descriptor range of the
  generated shader to the target slot that CrossTL reported, and launches
  ``ceil(dispatch / threads)`` thread groups;
- every record belongs to an executed test case, and dispatches only come from
  required GPU cases;
- no fault was injected, unless a negative control expects one.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import baseline  # noqa: E402

RECORD_KINDS = ("device", "pipeline", "dispatch", "readback")
RANGE_PREFIXES = {0: "t", 1: "u", 2: "b"}
MAX_ERRORS = 200


class EvidenceError(ValueError):
    """Raised when evidence is missing or inconsistent; lists every problem."""

    def __init__(self, errors: list[str]):
        self.errors = list(errors)
        shown = self.errors[:MAX_ERRORS]
        text = f"{len(self.errors)} evidence problem(s):\n" + "\n".join(shown)
        if len(self.errors) > len(shown):
            text += f"\n... {len(self.errors) - len(shown)} more"
        super().__init__(text)


def trx_cases(trx: Path) -> list[dict[str, str]]:
    """Return executed results with fully qualified case names, outcomes and messages."""

    root = ET.parse(trx).getroot()
    classes = {}
    for test in root.findall("t:TestDefinitions/t:UnitTest", baseline.TRX):
        method = test.find("t:TestMethod", baseline.TRX)
        if method is not None:
            classes[test.get("id")] = method.get("className", "")
    results = []
    for result in root.findall("t:Results/t:UnitTestResult", baseline.TRX):
        message, _trace = baseline._trx_message(result)
        owner = classes.get(result.get("testId"), "")
        name = result.get("testName", "")
        results.append(
            {
                "name": f"{owner}.{name}" if owner else name,
                "outcome": result.get("outcome", ""),
                "message": message,
            }
        )
    return results


def read_records(path: Path) -> Iterable[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise EvidenceError([f"line {number}: invalid JSON ({exc})"]) from exc
            if not isinstance(record, dict):
                raise EvidenceError([f"line {number}: record is not an object"])
            record["_line"] = number
            yield record


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _groups(dispatch: list[int], threads: list[int]) -> list[int]:
    return [-(-size // count) for size, count in zip(dispatch, threads)]


def case_categories(contract_path: Path, discovery: dict) -> dict[str, str]:
    """Map every fully qualified case display name to its contract category."""

    from coverage_contract import classify, load_contract

    contract = load_contract(contract_path)
    categories: dict[str, str] = {}
    for record in classify(contract, discovery):
        owner = record["method"].rsplit(".", 1)[0]
        name = f"{owner}.{record['displayName']}"
        previous = categories.setdefault(name, record["category"])
        if previous != record["category"]:
            raise EvidenceError([f"{name}: cases with one name have two categories"])
    return categories


def verify(
    *,
    evidence: Path,
    manifest_path: Path,
    discovery_dir: Path,
    trx: Path,
    profile: str,
    contract_path: Path,
    fault: str | None = None,
) -> dict[str, Any]:
    """Check one evidence file and return a summary; raise EvidenceError on problems."""

    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    settings = contract["profiles"][profile]
    manifest_sha256 = _file_sha256(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    discovery = baseline.load_discovery(discovery_dir)
    categories = case_categories(contract_path, discovery)
    executed = Counter(case["name"] for case in trx_cases(trx))
    shaders = manifest["shaders"]
    artifact_root = manifest_path.parent.parent
    discovery_types = defaultdict(set)
    for shader in discovery["shaders"]:
        discovery_types[shader["sha256"]].add(shader["type"])

    errors: list[str] = []
    counts: Counter[str] = Counter()
    pipelines: dict[str, dict[str, Any]] = {}
    verified_artifacts: dict[str, str] = {}
    devices: list[dict[str, Any]] = []
    cases_with_records: set[str] = set()
    cases_with_dispatch: Counter[str] = Counter()
    dispatched_shaders: Counter[str] = Counter()

    def problem(record: dict[str, Any], text: str) -> None:
        errors.append(f"line {record['_line']} ({record.get('record')}): {text}")

    for record in read_records(evidence):
        kind = record.get("record")
        if kind not in RECORD_KINDS:
            problem(record, "unknown record kind")
            continue
        counts[kind] += 1
        if record.get("backend") != settings["target"]:
            problem(
                record, f"backend {record.get('backend')!r} is not {settings['target']}"
            )
        if record.get("fault") != fault:
            problem(
                record, f"fault {record.get('fault')!r} where {fault!r} was expected"
            )
        case = record.get("case")
        if not isinstance(case, str) or not case:
            problem(record, "record is not attributed to a test case")
        elif case not in executed:
            problem(record, f"case {case!r} is not an executed result")
        else:
            cases_with_records.add(case)

        if kind == "device":
            devices.append(record)
            expected_hardware = settings.get("execution") == "hardware"
            if record.get("hardware") is not expected_hardware:
                problem(
                    record,
                    f"hardware={record.get('hardware')!r} but the profile requires {settings.get('execution')} execution",
                )
            if record.get("manifestSha256") != manifest_sha256:
                problem(record, "device loaded a different translation manifest")
            if record.get("translatorCommit") != manifest["translator"]["commit"]:
                problem(record, "translator commit differs from the manifest")
            if (
                not record.get("name")
                or not record.get("api")
                or not record.get("driver")
            ):
                problem(record, "device name, API or driver is missing")
            if (
                not isinstance(record.get("bufferGuardBytes"), int)
                or record["bufferGuardBytes"] <= 0
            ):
                problem(record, "buffer guards are not enabled")
        elif kind == "pipeline":
            hlsl = record.get("hlslSha256")
            entry = shaders.get(hlsl)
            if entry is None or entry.get("status") != "translated":
                problem(record, f"HLSL {hlsl} has no translated artifact")
                continue
            if record.get("artifactSha256") != entry["artifactSha256"]:
                problem(record, "artifact hash differs from the manifest")
            artifact = entry["artifact"]
            if artifact not in verified_artifacts:
                path = artifact_root / artifact
                verified_artifacts[artifact] = (
                    _file_sha256(path) if path.is_file() else ""
                )
            if verified_artifacts[artifact] != entry["artifactSha256"]:
                problem(
                    record, f"artifact file {artifact} does not match its manifest hash"
                )
            if record.get("threads") != entry["threads"]:
                problem(record, "thread-group size differs from the generated shader")
            if record.get("shader") not in discovery_types.get(hlsl, set()):
                problem(record, f"{record.get('shader')} does not generate HLSL {hlsl}")
            pipelines[str(record.get("shader"))] = entry
        elif kind == "dispatch":
            entry = pipelines.get(str(record.get("shader")))
            if entry is None:
                problem(
                    record,
                    f"{record.get('shader')} was dispatched without a compiled pipeline",
                )
                continue
            if (
                record.get("hlslSha256") != entry["hlslSha256"]
                or record.get("artifactSha256") != entry["artifactSha256"]
            ):
                problem(record, "dispatch source or artifact differs from its pipeline")
            size = record.get("dispatch")
            groups = record.get("groups")
            if (
                not isinstance(size, list)
                or len(size) != 3
                or any(type(value) is not int or value < 1 for value in size)
                or groups != _groups(size, entry["threads"])
            ):
                problem(record, f"thread groups {groups} do not cover dispatch {size}")
            constants = record.get("constants")
            if (
                not isinstance(constants, str)
                or len(constants) < 32
                or len(constants) % 32
            ):
                problem(record, "constants are not recorded in 16-byte registers")
            expected = [
                RANGE_PREFIXES[item["type"]] + str(item["register"])
                for item in entry["resourceDescriptorRanges"]
            ]
            bindings = record.get("bindings")
            if (
                not isinstance(bindings, list)
                or [item.get("register") for item in bindings] != expected
            ):
                problem(
                    record,
                    f"bound registers differ from the descriptor ranges {expected}",
                )
            else:
                targets = {
                    item.get("register"): item
                    for item in entry["bindings"]
                    if item.get("kind") != "sampler"
                }
                for item in bindings:
                    target = targets.get(item["register"])
                    if (
                        target is None
                        or item.get("slot") != target.get("binding")
                        or item.get("namespace") != target.get("namespace")
                    ):
                        problem(
                            record,
                            f"register {item['register']} is not bound to the slot CrossTL reported",
                        )
                    if type(item.get("resourceId")) is not int:
                        problem(record, f"register {item['register']} has no resource")
            if isinstance(case, str):
                cases_with_dispatch[case] += 1
                if categories.get(case) != "required-gpu":
                    problem(record, f"dispatch from a {categories.get(case)} case")
            dispatched_shaders[entry["hlslSha256"]] += 1
        elif kind == "readback":
            if type(record.get("bytes")) is not int or record["bytes"] <= 0:
                problem(record, "readback size is missing")
            digest = record.get("sha256")
            if not isinstance(digest, str) or len(digest) != 64:
                problem(record, "readback hash is missing")
            if "width" in record:
                geometry = [
                    record.get(key) for key in ("width", "height", "depth", "rowPitch")
                ]
                if any(type(value) is not int or value < 1 for value in geometry):
                    problem(record, f"texture readback region {geometry} is invalid")
            capture = record.get("capture")
            if capture is not None and capture != f"{digest}.png":
                problem(record, "captured pixels are not named by the readback hash")

    if not devices:
        errors.append("no device record: the translated backend never initialized")
    if counts["dispatch"] == 0:
        errors.append("no dispatch was recorded")
    if errors:
        raise EvidenceError(errors)

    required_executed = sorted(
        name for name in executed if categories.get(name) == "required-gpu"
    )
    translated = {
        key for key, value in shaders.items() if value.get("status") == "translated"
    }
    never = sorted(
        sorted(discovery_types[key])[0] for key in translated - set(dispatched_shaders)
    )
    device = {
        key: value for key, value in devices[0].items() if not key.startswith("_")
    }
    return {
        "status": "passed",
        "device": {
            key: device.get(key)
            for key in ("name", "api", "driver", "hardware", "fp64", "bufferGuardBytes")
        },
        "records": dict(sorted(counts.items())),
        "pipelines": len(pipelines),
        "verifiedArtifacts": len(verified_artifacts),
        "dispatchedShaders": len(dispatched_shaders),
        "translatedShaders": len(translated),
        "translatedShadersNotDispatched": never,
        "casesWithEvidence": len(cases_with_records),
        "casesWithDispatch": len(cases_with_dispatch),
        "requiredCaseNamesExecuted": len(required_executed),
        "requiredCaseNamesWithoutEvidence": sum(
            1 for name in required_executed if name not in cases_with_records
        ),
    }


def case_names_by_method(discovery_dir: Path) -> dict[str, Counter[str]]:
    """Fully qualified case names of every discovered method, as multisets."""

    discovery = baseline.load_discovery(discovery_dir)
    names: dict[str, Counter[str]] = defaultdict(Counter)
    for case in discovery["tests"]:
        owner = case["method"].rsplit(".", 1)[0]
        names[case["method"]][f"{owner}.{case['displayName']}"] += 1
    return names


def verify_control(
    control: dict[str, Any],
    *,
    trx: Path,
    evidence: Path,
    manifest_path: Path,
    discovery_dir: Path,
    profile: str,
    contract_path: Path,
    passed: Counter[str] | None = None,
) -> dict[str, Any]:
    """Check one negative-control run: every case of its methods ran and failed.

    ``passed`` holds the cases that passed in the run without faults; each
    control case must be among them, so a control cannot be satisfied by a
    case that fails anyway. The control evidence must carry the fault.
    """

    fault = control["fault"]
    names = case_names_by_method(discovery_dir)
    errors: list[str] = []
    expected: Counter[str] = Counter()
    for method in control["methods"]:
        if method not in names:
            errors.append(f"{method} is not a discovered test method")
        expected.update(names.get(method, Counter()))
    cases = trx_cases(trx) if trx.is_file() else []
    if not trx.is_file():
        errors.append(f"the control run wrote no results ({trx})")
    actual = Counter(case["name"] for case in cases)
    missing, unexpected = expected - actual, actual - expected
    if missing or unexpected:
        errors.append(
            f"executed cases differ from the discovery: {sum(missing.values())} missing, "
            f"{sum(unexpected.values())} unexpected"
        )
    pattern = control.get("messagePattern")
    for case in cases:
        if case["outcome"] != "Failed":
            errors.append(f"{case['name']}: {case['outcome']} under {fault}")
        elif pattern and not re.search(pattern, case["message"]):
            errors.append(f"{case['name']}: the failure does not report {pattern!r}")
    if passed is not None:
        for name, count in expected.items():
            if passed[name] < count:
                errors.append(f"{name}: did not pass in the run without faults")
    if trx.is_file():
        try:
            verify(
                evidence=evidence,
                manifest_path=manifest_path,
                discovery_dir=discovery_dir,
                trx=trx,
                profile=profile,
                contract_path=contract_path,
                fault=fault,
            )
        except (EvidenceError, OSError) as exc:
            errors.extend(
                exc.errors[:20] if isinstance(exc, EvidenceError) else [str(exc)]
            )
    return {
        "fault": fault,
        "methods": control["methods"],
        "cases": sum(expected.values()),
        "failed": sum(1 for case in cases if case["outcome"] == "Failed"),
        "status": "passed" if not errors else "failed",
        "errors": errors[:50],
        "sampleFailures": [
            {"name": case["name"], "message": case["message"][:300]}
            for case in cases[:2]
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--discovery", required=True, type=Path)
    parser.add_argument("--trx", required=True, type=Path)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--contract", type=Path, default=HERE / "coverage.json")
    parser.add_argument("--fault")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        summary = verify(
            evidence=args.evidence,
            manifest_path=args.manifest,
            discovery_dir=args.discovery,
            trx=args.trx,
            profile=args.profile,
            contract_path=args.contract,
            fault=args.fault,
        )
    except EvidenceError as exc:
        print(exc, file=sys.stderr)
        return 1
    text = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

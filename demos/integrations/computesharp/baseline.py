"""Check original MSTest results against the complete compiled discovery export."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

TRX = {"t": "http://microsoft.com/schemas/VisualStudio/TeamTest/2010"}
DEVICE_TYPE = "ComputeSharp.Tests.Extensions.Device"
MAX_REPORTED_ERRORS = 500


class AccountingError(ValueError):
    """Raised when results do not match the contract; lists every mismatch."""

    def __init__(self, errors: list[str]):
        self.errors = list(errors)
        shown = self.errors[:MAX_REPORTED_ERRORS]
        more = len(self.errors) - len(shown)
        text = f"{len(self.errors)} accounting mismatch(es):\n" + "\n".join(shown)
        if more:
            text += f"\n... {more} more"
        super().__init__(text)


def device_of(case: dict) -> str | None:
    """Return the upstream ``Device`` argument of a case, if it has one."""
    names = [
        argument.get("name")
        for argument in case["arguments"]
        if argument.get("kind") == "enum" and argument.get("type") == DEVICE_TYPE
    ]
    if len(names) > 1:
        raise ValueError("Case has more than one device argument: " + case["id"])
    return names[0] if names else None


def load_discovery(directory: Path) -> dict:
    report = json.loads((directory / "discovery.json").read_text(encoding="utf-8"))
    if (
        report.get("kind") != "computesharp.discovery"
        or report.get("schemaVersion") != 1
    ):
        raise ValueError("Unsupported discovery report")
    tests, shaders = report.get("tests"), report.get("shaders")
    if (
        not isinstance(tests, list)
        or not tests
        or report.get("caseCount") != len(tests)
    ):
        raise ValueError("Missing or inconsistent test inventory")
    methods = report.get("methods")
    if not isinstance(methods, list) or report.get("testMethodCount") != len(methods):
        raise ValueError("Missing or inconsistent test method inventory")
    method_names = set()
    for entry in methods:
        source = entry.get("source") if isinstance(entry, dict) else None
        if (
            not isinstance(entry.get("method"), str)
            or not isinstance(entry.get("signature"), str)
            or not entry["signature"].startswith(entry["method"] + "(")
            or not isinstance(source, dict)
            or not isinstance(source.get("path"), str)
            or not source["path"]
            or Path(source["path"]).is_absolute()
            or "\\" in source["path"]
            or ".." in Path(source["path"]).parts
            or type(source.get("line")) is not int
            or source["line"] < 1
        ):
            raise ValueError("Invalid test method source location")
        if entry["signature"] in method_names:
            raise ValueError("Duplicate test method: " + entry["signature"])
        method_names.add(entry["signature"])
    if {case.get("signature") for case in tests} != method_names:
        raise ValueError("Cases and test methods differ")
    if (
        not isinstance(shaders, list)
        or not shaders
        or report.get("shaderCount") != len(shaders)
    ):
        raise ValueError("Missing or inconsistent shader inventory")
    identities = set()
    occurrences = defaultdict(list)
    for case in tests:
        identity = case.get("id")
        if not isinstance(identity, str) or not identity or identity in identities:
            raise ValueError("Missing or duplicate case identity")
        key = (case.get("method"), case.get("displayName"))
        if not all(isinstance(value, str) and value for value in key):
            raise ValueError("Missing MSTest case name")
        if (
            not isinstance(case.get("arguments"), list)
            or type(case.get("ignored")) is not bool
        ):
            raise ValueError("Invalid case arguments or ignore status")
        occurrence = case.get("occurrence")
        if type(occurrence) is not int or occurrence < 1:
            raise ValueError("Invalid case occurrence")
        identities.add(identity)
        occurrences[
            (case.get("signature"), json.dumps(case["arguments"], sort_keys=True))
        ].append(occurrence)
    for values in occurrences.values():
        if sorted(values) != list(range(1, len(values) + 1)):
            raise ValueError("Repeated data rows must be numbered consecutively")
    hlsl = directory / "hlsl"
    if hlsl.is_symlink():
        raise ValueError("Shader directory must not be a symbolic link")
    paths = set()
    for shader in shaders:
        relative = shader.get("path")
        if (
            not isinstance(relative, str)
            or "\\" in relative
            or Path(relative).is_absolute()
            or Path(relative).parts != ("hlsl", Path(relative).name)
        ):
            raise ValueError("Invalid shader path")
        path = directory / relative
        if path.is_symlink() or path.resolve().parent != hlsl.resolve():
            raise ValueError("Shader path escapes the HLSL directory")
        if relative in paths:
            raise ValueError("Duplicate shader path")
        paths.add(relative)
        if hashlib.sha256(path.read_bytes()).hexdigest() != shader.get("sha256"):
            raise ValueError("Generated HLSL content changed: " + relative)
    return report


def _trx_message(result: ET.Element) -> tuple[str, str]:
    message = result.find("t:Output/t:ErrorInfo/t:Message", TRX)
    trace = result.find("t:Output/t:ErrorInfo/t:StackTrace", TRX)
    return (
        (message.text or "") if message is not None else "",
        (trace.text or "") if trace is not None else "",
    )


def verify_results(
    directory: Path,
    trx_path: Path,
    profile: str = "directx-warp",
    contract_path: Path | None = None,
) -> dict:
    """Account every discovered case of one profile against MSTest TRX results.

    The coverage contract decides the expected outcome of each case: ``pass``
    cases must pass, capability-gated ``inconclusive`` cases must report the
    upstream capability check (not a missing device), ``optional`` hardware rows
    may pass or report a missing device, and ``excluded`` cases must not run.
    """

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from coverage_contract import CONTRACT, classify, expected_outcome, load_contract

    contract_path = contract_path or CONTRACT
    contract = load_contract(contract_path)
    if profile not in contract["profiles"]:
        raise ValueError("Unknown coverage profile: " + profile)
    settings = contract["profiles"][profile]
    discovery = load_discovery(directory)
    records = {record["id"]: record for record in classify(contract, discovery)}
    expected = defaultdict(list)
    for case in discovery["tests"]:
        expected[(case["method"], case["displayName"])].append(case)
    root = ET.parse(trx_path).getroot()
    summary = root.find("t:ResultSummary", TRX)
    run_outcome = summary.get("outcome") if summary is not None else None
    if run_outcome not in {"Passed", "Completed", "Failed"}:
        raise ValueError("MSTest run did not complete: " + str(run_outcome))
    errors: list[str] = []
    definitions = {}
    for unit in root.findall("t:TestDefinitions/t:UnitTest", TRX):
        method = unit.find("t:TestMethod", TRX)
        if method is None or not unit.get("id") or unit.get("id") in definitions:
            raise ValueError("Invalid or duplicate TRX test definition")
        owner, name = method.get("className", "").split(",", 1)[0], method.get("name")
        if not owner or not name:
            raise ValueError("Missing TRX method identity")
        definitions[unit.get("id")] = owner + "." + name
    observed = defaultdict(list)
    for result in root.findall("t:Results/t:UnitTestResult", TRX):
        method = definitions.get(result.get("testId"))
        key = (method, result.get("testName"))
        if key not in expected:
            errors.append("Unexpected executed case: " + str(key))
            continue
        observed[key].append((result.get("outcome"), *_trx_message(result)))
    counts = defaultdict(int)
    optional_skipped, gated, passed_ids = [], [], []
    for key, cases in expected.items():
        outcomes = sorted(observed.get(key, []))
        wanted = {expected_outcome(records[case["id"]], settings) for case in cases}
        if len(wanted) != 1:
            raise ValueError(
                "Cases sharing a name have different expectations: " + str(key)
            )
        wanted = wanted.pop()
        counts["expected-" + wanted] += len(cases)
        if wanted == "excluded":
            if outcomes:
                errors.append("Excluded case was executed: " + str(key))
            continue
        if len(outcomes) != len(cases):
            errors.append(
                f"Case {key} expected {len(cases)} results but found {len(outcomes)}"
            )
            continue
        for case, (outcome, message, trace) in zip(cases, outcomes):
            inconclusive = outcome == "NotExecuted" and message.startswith(
                "Assert.Inconclusive failed."
            )
            missing_device = inconclusive and "DeviceExtensions.Get" in trace
            if outcome == "Passed" and wanted in {"pass", "optional"}:
                passed_ids.append(case["id"])
                counts["passed"] += 1
                counts["warpPassed"] += device_of(case) == "Warp"
                counts["requiredGpuPassed"] += (
                    records[case["id"]]["category"] == "required-gpu"
                )
            elif wanted == "optional" and inconclusive:
                optional_skipped.append(case["id"])
            elif wanted == "inconclusive" and inconclusive and not missing_device:
                gated.append(case["id"])
            else:
                errors.append(
                    f"Case {key} expected {wanted} but reported {outcome}: {message.strip()[:200]}"
                )
    if counts["passed"] == 0:
        errors.append("No cases passed")
    if settings["deviceRows"]["Warp"] == "execute" and counts["warpPassed"] == 0:
        errors.append("No WARP-row cases passed")
    if run_outcome == "Failed" and not errors:
        errors.append("MSTest reported a failed run without a failing case")
    if errors:
        raise AccountingError(errors)
    return {
        "schemaVersion": 1,
        "kind": "computesharp.test-results",
        "profile": profile,
        "role": settings["role"],
        "execution": settings["execution"],
        "assembly": discovery["assembly"],
        "caseCount": len(discovery["tests"]),
        "expected": {
            outcome: counts["expected-" + outcome]
            for outcome in ("pass", "inconclusive", "optional", "excluded")
        },
        "requiredCasesPassed": counts["passed"],
        "requiredGpuCasesPassed": counts["requiredGpuPassed"],
        "warpCasesPassed": counts["warpPassed"],
        "capabilityInconclusive": sorted(gated),
        "optionalHardwareSkipped": sorted(optional_skipped),
        "passedCasesSHA256": (
            hashlib.sha256("\n".join(sorted(passed_ids)).encode("utf-8")).hexdigest()
        ),
        "contractSHA256": hashlib.sha256(contract_path.read_bytes()).hexdigest(),
        "discoverySHA256": (
            hashlib.sha256((directory / "discovery.json").read_bytes()).hexdigest()
        ),
        "trxSHA256": hashlib.sha256(trx_path.read_bytes()).hexdigest(),
        "coverageFrozen": True,
        "nativeTargetExecutionQualified": False,
    }


def verify_baseline(
    directory: Path, trx_path: Path, contract_path: Path | None = None
) -> dict:
    """Account the original Windows WARP results."""

    return verify_results(directory, trx_path, "directx-warp", contract_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--discovery", required=True, type=Path)
    parser.add_argument("--trx", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--profile", default="directx-warp")
    parser.add_argument("--contract", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists")
    try:
        report = verify_results(args.discovery, args.trx, args.profile, args.contract)
    except (OSError, ValueError, TypeError, KeyError, ET.ParseError) as error:
        parser.exit(1, str(error) + "\n")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

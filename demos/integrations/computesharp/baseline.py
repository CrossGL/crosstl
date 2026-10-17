"""Check original MSTest results against the complete compiled discovery export."""

from __future__ import annotations

import argparse
import hashlib
import json
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

TRX = {"t": "http://microsoft.com/schemas/VisualStudio/TeamTest/2010"}
DEVICE_TYPE = "ComputeSharp.Tests.Extensions.Device"


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


def verify_baseline(directory: Path, trx_path: Path) -> dict:
    discovery = load_discovery(directory)
    expected = defaultdict(list)
    for case in discovery["tests"]:
        expected[(case["method"], case["displayName"])].append(case)
    root = ET.parse(trx_path).getroot()
    summary = root.find("t:ResultSummary", TRX)
    if summary is None or summary.get("outcome") not in {"Passed", "Completed"}:
        raise ValueError("MSTest run did not complete successfully")
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
            raise ValueError("Unexpected executed case: " + str(key))
        observed[key].append(result.get("outcome"))
    skipped, required, warp = [], 0, 0
    for key, cases in expected.items():
        outcomes = observed.get(key, [])
        if len(outcomes) != len(cases):
            raise ValueError(
                f"Case {key} expected {len(cases)} results but found {len(outcomes)}"
            )
        devices = {device_of(case) for case in cases}
        if len(devices) != 1:
            raise ValueError("Cases sharing a name have different devices: " + str(key))
        device = devices.pop()
        for case, outcome in zip(cases, sorted(outcomes)):
            if case["ignored"]:
                raise ValueError(
                    "Ignored case cannot satisfy the baseline: " + str(key)
                )
            if device == "Discrete" and outcome == "NotExecuted":
                skipped.append(case["id"])
            elif outcome != "Passed":
                raise ValueError("Required baseline case did not pass: " + str(key))
            elif device != "Discrete":
                required += 1
                warp += device == "Warp"
    if required == 0 or warp == 0:
        raise ValueError("No required WARP baseline cases executed")
    return {
        "schemaVersion": 1,
        "kind": "computesharp.original-baseline",
        "assembly": discovery["assembly"],
        "caseCount": len(discovery["tests"]),
        "requiredCasesPassed": required,
        "warpCasesPassed": warp,
        "optionalHardwareSkipped": sorted(skipped),
        "discoverySHA256": (
            hashlib.sha256((directory / "discovery.json").read_bytes()).hexdigest()
        ),
        "trxSHA256": hashlib.sha256(trx_path.read_bytes()).hexdigest(),
        "coverageFrozen": False,
        "nativeTargetExecutionQualified": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--discovery", required=True, type=Path)
    parser.add_argument("--trx", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists")
    try:
        report = verify_baseline(args.discovery, args.trx)
    except (OSError, ValueError, TypeError, KeyError, ET.ParseError) as error:
        parser.exit(1, str(error) + "\n")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Verify the dispatch evidence of a translated ComputeSharp run.

The portable runtime writes one JSON object per line for the device, every
capability query made through the public API, every resource allocation and
release, every pipeline it compiles, every dispatch and every readback, each
attributed to the executing test case. This module checks that the evidence is
complete and consistent with the translated artifacts and the frozen inventory:

- the device record names the selected backend, states hardware or software
  execution as the profile requires and references the manifest that was used;
- every pipeline loads a translated artifact listed in the manifest, whose file
  content hash and thread-group size match the manifest;
- every dispatch uses a compiled pipeline, binds every descriptor range of the
  generated shader to the target slot that CrossTL reported, binds only device
  resources that were allocated and not yet released, binds the channel scales
  of the view format to every UNORM store scale buffer of the artifact, and
  launches ``ceil(dispatch / threads)`` thread groups;
- readbacks read allocated, live device resources; every resource is released
  at most once;
- every capability query that answers a contract capability returns the value
  the profile declares, and every case of a capability-gated method shows the
  query that decided it;
- every record belongs to an executed test case (releases excepted, because
  finalizers release resources while other cases run), dispatches only come
  from required GPU cases, and every executed required GPU case left evidence
  unless the evidence policy lists its method as host-only with a reason;
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

RECORD_KINDS = (
    "device",
    "capability",
    "resource",
    "release",
    "pipeline",
    "dispatch",
    "readback",
)
MEMORY_KINDS = ("device", "host")
POLICY = HERE / "evidence_policy.json"
POLICY_KIND = "computesharp.evidence-policy"
RANGE_PREFIXES = {0: "t", 1: "u", 2: "b"}
# Channel scales (2^n - 1) of the UNORM DXGI formats of the supported texture
# element types; absent channels are 0. The runtime binds them to the
# ``crosstl_unorm_store_<register>`` buffers of CrossTL's UNORM store rounding.
UNORM_STORE_SCALE_PREFIX = "crosstl_unorm_store_"
UNORM_CHANNEL_SCALES = {
    28: [255, 255, 255, 255],  # R8G8B8A8_UNORM
    87: [255, 255, 255, 255],  # B8G8R8A8_UNORM
    11: [65535, 65535, 65535, 65535],  # R16G16B16A16_UNORM
    49: [255, 255, 0, 0],  # R8G8_UNORM
    35: [65535, 65535, 0, 0],  # R16G16_UNORM
    61: [255, 0, 0, 0],  # R8_UNORM
    56: [65535, 0, 0, 0],  # R16_UNORM
}
MAX_ERRORS = 200


def expected_unorm_scales(format_code: Any, fault: str | None) -> list[int] | None:
    """The channel scales a dispatch must bind for a view of ``format_code``.

    The ``perturb-unorm-scales`` control binds one level per channel instead,
    so its evidence must show exactly that perturbation.
    """

    scales = UNORM_CHANNEL_SCALES.get(format_code)
    if scales is not None and fault == "perturb-unorm-scales":
        return [min(scale, 1) for scale in scales]
    return scales


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

    return {
        name: info["category"]
        for name, info in case_details(contract_path, discovery).items()
    }


def case_details(contract_path: Path, discovery: dict) -> dict[str, dict[str, Any]]:
    """Map every fully qualified case display name to its method, category and gate."""

    from coverage_contract import classify, load_contract

    contract = load_contract(contract_path)
    details: dict[str, dict[str, Any]] = {}
    for record in classify(contract, discovery):
        owner = record["method"].rsplit(".", 1)[0]
        name = f"{owner}.{record['displayName']}"
        info = {
            "method": record["method"],
            "category": record["category"],
            "gate": record["gate"],
        }
        previous = details.setdefault(name, info)
        if previous != info:
            raise EvidenceError(
                [f"{name}: cases with one name have different categories or gates"]
            )
    return details


def load_policy(
    path: Path, contract: dict[str, Any], details: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """Load the evidence policy and check it against the contract and the inventory.

    ``capabilityQueries`` states, for every contract capability, the fields of
    the runtime capability record that answers it; a list value matches any of
    its items. ``hostOnlyMethods`` names required GPU methods whose cases do
    no device work on the translated runtime, each with the reason.
    """

    try:
        policy = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvidenceError([f"evidence policy {path}: {exc}"]) from exc
    errors = []
    if policy.get("kind") != POLICY_KIND or policy.get("schemaVersion") != 1:
        errors.append(f"evidence policy {path} is not a {POLICY_KIND} version 1")
    queries = policy.get("capabilityQueries")
    host_only = policy.get("hostOnlyMethods")
    if not isinstance(queries, dict) or not isinstance(host_only, dict):
        errors.append("evidence policy needs capabilityQueries and hostOnlyMethods")
        raise EvidenceError(errors)
    for capability in contract.get("capabilities", {}):
        query = queries.get(capability)
        if not isinstance(query, dict) or not isinstance(query.get("capability"), str):
            errors.append(f"no runtime capability query answers {capability}")
    for capability in sorted(set(queries) - set(contract.get("capabilities", {}))):
        errors.append(f"capability query for unknown capability {capability}")
    categories = {info["method"]: info["category"] for info in details.values()}
    for method, reason in sorted(host_only.items()):
        if categories.get(method) != "required-gpu":
            errors.append(f"host-only method {method} is not a required GPU method")
        if not isinstance(reason, str) or len(reason.strip()) < 20:
            errors.append(f"host-only method {method} needs a specific reason")
    if errors:
        raise EvidenceError(errors)
    return policy


def _answers(record: dict[str, Any], query: dict[str, Any]) -> bool:
    for key, value in query.items():
        actual = record.get(key)
        if isinstance(value, list):
            if actual not in value:
                return False
        elif actual != value:
            return False
    return True


def verify(
    *,
    evidence: Path,
    manifest_path: Path,
    discovery_dir: Path,
    trx: Path,
    profile: str,
    contract_path: Path,
    fault: str | None = None,
    policy_path: Path = POLICY,
) -> dict[str, Any]:
    """Check one evidence file and return a summary; raise EvidenceError on problems."""

    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    settings = contract["profiles"][profile]
    manifest_sha256 = _file_sha256(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    discovery = baseline.load_discovery(discovery_dir)
    details = case_details(contract_path, discovery)
    categories = {name: info["category"] for name, info in details.items()}
    policy = load_policy(policy_path, contract, details)
    queries = policy["capabilityQueries"]
    host_only = policy["hostOnlyMethods"]
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
    resources: dict[int, dict[str, Any]] = {}
    released: set[int] = set()
    memory_counts: Counter[str] = Counter()
    capability_answers: dict[str, list[dict[str, Any]]] = defaultdict(list)
    fp64_answers: set[bool] = set()

    def problem(record: dict[str, Any], text: str) -> None:
        errors.append(f"line {record['_line']} ({record.get('record')}): {text}")

    def live_device_resource(record: dict[str, Any], identifier: Any, use: str) -> None:
        resource = resources.get(identifier) if type(identifier) is int else None
        if resource is None:
            problem(
                record, f"{use} uses resource {identifier!r}, which was never allocated"
            )
        elif resource["memory"] != "device":
            problem(
                record, f"{use} uses {resource['type']} in {resource['memory']} memory"
            )
        elif identifier in released:
            problem(record, f"{use} uses resource {identifier} after its release")

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
        if kind == "release" and case is None:
            pass
        elif not isinstance(case, str) or not case:
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
        elif kind == "capability":
            available = record.get("available")
            if (
                not isinstance(record.get("capability"), str)
                or type(available) is not bool
            ):
                problem(record, "capability query has no name or answer")
                continue
            for capability, query in queries.items():
                if not _answers(record, query):
                    continue
                declared = settings["capabilities"].get(capability)
                if available is not declared:
                    problem(
                        record,
                        f"the device answers {capability}={available} but the profile declares {declared}",
                    )
                if isinstance(case, str):
                    capability_answers[case].append(record | {"answers": capability})
            if record["capability"] == "fp64":
                fp64_answers.add(available)
        elif kind == "resource":
            identifier = record.get("resourceId")
            if type(identifier) is not int or identifier < 1:
                problem(record, f"resource id {identifier!r} is invalid")
                continue
            if identifier in resources:
                problem(record, f"resource {identifier} was allocated twice")
                continue
            memory = record.get("memory")
            size = record.get("bytes")
            if memory not in MEMORY_KINDS:
                problem(
                    record, f"resource memory {memory!r} is not one of {MEMORY_KINDS}"
                )
            if type(size) is not int or size < 1:
                problem(record, f"resource size {size!r} is invalid")
            if not isinstance(record.get("type"), str) or not record["type"]:
                problem(record, "resource type is missing")
            resources[identifier] = {
                "memory": memory,
                "type": record.get("type"),
            }
            memory_counts[str(memory)] += 1
        elif kind == "release":
            identifier = record.get("resourceId")
            if type(identifier) is not int or identifier not in resources:
                problem(record, f"released resource {identifier!r} was never allocated")
            elif identifier in released:
                problem(record, f"resource {identifier} was released twice")
            else:
                released.add(identifier)
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
                unorm_scales = {
                    item["name"][len(UNORM_STORE_SCALE_PREFIX) :]: item
                    for item in entry["bindings"]
                    if item.get("kind") == "constant-buffer"
                    and str(item.get("name") or "").startswith(UNORM_STORE_SCALE_PREFIX)
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
                    scale_target = unorm_scales.get(item["register"])
                    if scale_target is None:
                        if "unormScales" in item or "unormScaleSlot" in item:
                            problem(
                                record,
                                f"register {item['register']} has UNORM store scales the artifact does not declare",
                            )
                    elif item.get("unormScaleSlot") != scale_target.get(
                        "binding"
                    ) or item.get("unormScales") != expected_unorm_scales(
                        item.get("format"), fault
                    ):
                        problem(
                            record,
                            f"register {item['register']} UNORM store scales {item.get('unormScales')} "
                            f"at slot {item.get('unormScaleSlot')} do not match format {item.get('format')} "
                            "and the slot CrossTL reported",
                        )
                    if type(item.get("resourceId")) is not int:
                        problem(record, f"register {item['register']} has no resource")
                    else:
                        live_device_resource(
                            record, item["resourceId"], f"register {item['register']}"
                        )
            if isinstance(case, str):
                cases_with_dispatch[case] += 1
                if categories.get(case) != "required-gpu":
                    problem(record, f"dispatch from a {categories.get(case)} case")
            dispatched_shaders[entry["hlslSha256"]] += 1
        elif kind == "readback":
            if type(record.get("bytes")) is not int or record["bytes"] <= 0:
                problem(record, "readback size is missing")
            live_device_resource(record, record.get("resourceId"), "readback")
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
    device_fp64 = {record.get("fp64") for record in devices}
    if fp64_answers and fp64_answers != device_fp64:
        errors.append(
            f"fp64 queries answered {sorted(fp64_answers)} but the device reports {sorted(device_fp64, key=str)}"
        )

    required_executed = sorted(
        name for name in executed if categories.get(name) == "required-gpu"
    )
    host_only_cases = []
    gated_cases = 0
    for name in required_executed:
        info = details[name]
        if name not in cases_with_records:
            if info["method"] in host_only:
                host_only_cases.append(name)
            else:
                errors.append(f"{name}: the required GPU case left no evidence")
        gate = info["gate"]
        if gate is None:
            continue
        capability = gate["capability"]
        declared = settings["capabilities"].get(capability)
        if not any(
            answer["answers"] == capability and answer["available"] is declared
            for answer in capability_answers.get(name, [])
        ):
            errors.append(
                f"{name}: no {capability} capability query decided the gated case"
            )
        gated_cases += 1
    if errors:
        raise EvidenceError(errors)

    # A host-only method whose executed cases all left evidence no longer needs its entry.
    without = {details[name]["method"] for name in host_only_cases}
    ran = {details[name]["method"] for name in required_executed}
    stale_host_only = sorted(set(host_only) & ran - without)
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
        "policySha256": _file_sha256(policy_path),
        "resources": dict(sorted(memory_counts.items())),
        "releasedResources": len(released),
        "casesWithEvidence": len(cases_with_records),
        "casesWithDispatch": len(cases_with_dispatch),
        "requiredCaseNamesExecuted": len(required_executed),
        "requiredCaseNamesWithEvidence": len(required_executed) - len(host_only_cases),
        "hostOnlyCaseNames": len(host_only_cases),
        "hostOnlyMethodsWithEvidence": stale_host_only,
        "gatedCaseNamesWithCapabilityQuery": gated_cases,
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
    policy_path: Path = POLICY,
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
                policy_path=policy_path,
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
    parser.add_argument("--policy", type=Path, default=POLICY)
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
            policy_path=args.policy,
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

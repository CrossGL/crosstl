"""Apply the reviewed coverage contract to a discovery export.

``coverage.json`` classifies every upstream test method and records the target
profiles and optional device capabilities. ``freeze`` combines it with a
discovery export into ``coverage.lock.json``: per-method categories, source
locations, case counts and case-identity digests, plus the expected outcome
counts of every profile. ``check`` requires a new export to reproduce the lock
exactly, so cases cannot disappear or change category without an explicit,
reviewed contract change.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from baseline import device_of, load_discovery  # noqa: E402
from source import PIN  # noqa: E402

CONTRACT = HERE / "coverage.json"
LOCK = HERE / "coverage.lock.json"
CATEGORIES = ("required-gpu", "cpu-only", "windows-specific")
DEVICE_ROWS = ("Warp", "Discrete")
OUTCOMES = ("pass", "inconclusive", "optional", "excluded")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_contract(path: Path = CONTRACT) -> dict:
    contract = json.loads(path.read_text(encoding="utf-8"))
    if (
        contract.get("kind") != "computesharp.coverage-contract"
        or contract.get("schemaVersion") != 1
    ):
        raise ValueError("Unsupported coverage contract")
    if contract.get("commit") != PIN:
        raise ValueError("Coverage contract is for a different upstream revision")
    if set(contract.get("categories", {})) != set(CATEGORIES):
        raise ValueError("Coverage contract must define exactly the known categories")
    profiles = contract.get("profiles")
    if not isinstance(profiles, dict) or not profiles:
        raise ValueError("Coverage contract has no target profiles")
    capabilities = contract.get("capabilities", {})
    for name, profile in profiles.items():
        if set(profile.get("deviceRows", {})) != set(DEVICE_ROWS):
            raise ValueError(f"Profile {name} must map every upstream device row")
        if set(profile.get("capabilities", {})) != set(capabilities):
            raise ValueError(f"Profile {name} must state every capability")
        if not all(type(v) is bool for v in profile["capabilities"].values()):
            raise ValueError(f"Profile {name} capabilities must be booleans")
        if not set(profile["deviceRows"].values()) <= {"execute", "optional-hardware"}:
            raise ValueError(f"Profile {name} has an unknown device row policy")
        if profile.get("execution") not in {"hardware", "software"}:
            raise ValueError(
                f"Profile {name} must state hardware or software execution"
            )
    for entries in (contract.get("classes", {}), contract.get("methods", {})):
        for name, entry in entries.items():
            if entry.get("category") not in CATEGORIES or not entry.get("reason"):
                raise ValueError(f"Invalid classification for {name}")
    return contract


def _owner(method: str) -> str:
    return method.rsplit(".", 1)[0]


def _type_arguments(case: dict) -> set[str]:
    return {
        argument["name"]
        for argument in case["arguments"]
        if argument.get("kind") == "type"
    }


def classify(contract: dict, discovery: dict) -> list[dict]:
    """Return one classified record per discovered case."""

    classes, methods = contract.get("classes", {}), contract.get("methods", {})
    discovered = {entry["method"] for entry in discovery["methods"]}
    owners = {_owner(name) for name in discovered}
    stale = sorted((set(methods) - discovered) | (set(classes) - owners))
    if stale:
        raise ValueError(
            "Contract entries without discovered tests: " + ", ".join(stale)
        )
    gates = defaultdict(list)
    for capability, entry in contract.get("capabilities", {}).items():
        for gate in entry.get("gates", []):
            if gate.get("method") not in discovered:
                raise ValueError("Capability gate for unknown method: " + str(gate))
            if type(gate.get("runsWhenAvailable")) is not bool:
                raise ValueError("Capability gate must state runsWhenAvailable")
            gates[gate["method"]].append((capability, gate))
    records, matched_gates = [], Counter()
    for case in discovery["tests"]:
        method = case["method"]
        entry = methods.get(method) or classes.get(_owner(method))
        if entry is None:
            raise ValueError("Unclassified test method: " + method)
        if case["ignored"]:
            raise ValueError("Ignored upstream case cannot be accounted: " + case["id"])
        selected = []
        for capability, gate in gates.get(method, []):
            wanted = gate.get("typeArguments")
            if wanted is None or _type_arguments(case) & set(wanted):
                selected.append(
                    {
                        "capability": capability,
                        "runsWhenAvailable": gate["runsWhenAvailable"],
                    }
                )
                matched_gates[(method, capability)] += 1
        if len(selected) > 1:
            raise ValueError(
                "Case matches more than one capability gate: " + case["id"]
            )
        records.append(
            {
                "id": case["id"],
                "method": method,
                "signature": case["signature"],
                "displayName": case["displayName"],
                "occurrence": case["occurrence"],
                "device": device_of(case),
                "category": entry["category"],
                "gate": selected[0] if selected else None,
            }
        )
    for method, entries in gates.items():
        for capability, gate in entries:
            if matched_gates[(method, capability)] == 0:
                raise ValueError(
                    f"Capability gate {capability} matches no case of {method}"
                )
    by_method = {record["method"]: record["category"] for record in records}
    for name, adaptation in contract.get("adaptations", {}).items():
        for method in adaptation.get("methods", []):
            if by_method.get(method) != "required-gpu":
                raise ValueError(
                    f"Adaptation {name} lists a non-required method {method}"
                )
        for owner in adaptation.get("classes", []):
            if owner not in owners:
                raise ValueError(f"Adaptation {name} lists an unknown class {owner}")
    return records


def expected_outcome(record: dict, profile: dict) -> str:
    """Expected outcome of one classified case on one profile."""

    translated = profile["role"] == "translated-target"
    if record["category"] == "windows-specific" and translated:
        return "excluded"
    device = record["device"]
    if device is not None and profile["deviceRows"][device] == "optional-hardware":
        return "optional"
    gate = record["gate"]
    if gate is not None:
        available = profile["capabilities"][gate["capability"]]
        return "pass" if available == gate["runsWhenAvailable"] else "inconclusive"
    return "pass"


def build_lock(contract: dict, discovery: dict, contract_bytes: bytes) -> dict:
    records = classify(contract, discovery)
    sources = {entry["signature"]: entry["source"] for entry in discovery["methods"]}
    grouped = defaultdict(list)
    for record in records:
        grouped[record["signature"]].append(record)
    methods = {}
    for signature in sorted(grouped):
        cases = grouped[signature]
        categories = {case["category"] for case in cases}
        gates = Counter(
            f"{case['gate']['capability']}:{'available' if case['gate']['runsWhenAvailable'] else 'unavailable'}"
            for case in cases
            if case["gate"]
        )
        methods[signature] = {
            "category": categories.pop(),
            "source": sources[signature],
            "cases": len(cases),
            "devices": dict(
                sorted(Counter(case["device"] or "none" for case in cases).items())
            ),
            "caseDigest": _sha256(
                "\n".join(sorted(case["id"] for case in cases)).encode()
            ),
        }
        if gates:
            methods[signature]["capabilityGates"] = dict(sorted(gates.items()))
    totals = {category: {"methods": 0, "cases": 0} for category in CATEGORIES}
    for entry in methods.values():
        totals[entry["category"]]["methods"] += 1
        totals[entry["category"]]["cases"] += entry["cases"]
    profiles = {}
    for name, profile in sorted(contract["profiles"].items()):
        counts = Counter(expected_outcome(record, profile) for record in records)
        required = Counter(
            expected_outcome(record, profile)
            for record in records
            if record["category"] == "required-gpu"
        )
        profiles[name] = {
            "role": profile["role"],
            "execution": profile["execution"],
            "expected": {outcome: counts.get(outcome, 0) for outcome in OUTCOMES},
            "requiredGpuExpected": {
                outcome: required.get(outcome, 0) for outcome in OUTCOMES
            },
        }
    return {
        "schemaVersion": 1,
        "kind": "computesharp.coverage-lock",
        "commit": PIN,
        "contractSHA256": _sha256(contract_bytes),
        "testMethodCount": discovery["testMethodCount"],
        "caseCount": discovery["caseCount"],
        "allCasesDigest": _sha256("\n".join(sorted(r["id"] for r in records)).encode()),
        "totals": totals,
        "profiles": profiles,
        "methods": methods,
    }


def compare_locks(expected: dict, actual: dict) -> list[str]:
    problems = []
    for key in sorted(set(expected) | set(actual)):
        if key == "methods":
            continue
        if expected.get(key) != actual.get(key):
            problems.append(f"{key} differs")
    left, right = expected.get("methods", {}), actual.get("methods", {})
    for method in sorted(set(left) - set(right)):
        problems.append(f"missing method {method}")
    for method in sorted(set(right) - set(left)):
        problems.append(f"unexpected method {method}")
    for method in sorted(set(left) & set(right)):
        if left[method] != right[method]:
            fields = sorted(
                k
                for k in set(left[method]) | set(right[method])
                if left[method].get(k) != right[method].get(k)
            )
            problems.append(f"{method} differs in {', '.join(fields)}")
    return problems


def check(
    discovery_dir: Path, lock_path: Path = LOCK, contract_path: Path = CONTRACT
) -> dict:
    contract_bytes = contract_path.read_bytes()
    actual = build_lock(
        load_contract(contract_path), load_discovery(discovery_dir), contract_bytes
    )
    expected = json.loads(lock_path.read_text(encoding="utf-8"))
    problems = compare_locks(expected, actual)
    if problems:
        raise ValueError("Coverage lock mismatch:\n" + "\n".join(problems[:50]))
    return actual


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("freeze", "check"):
        command = sub.add_parser(name)
        command.add_argument("--discovery", required=True, type=Path)
        command.add_argument("--lock", type=Path, default=LOCK)
    args = parser.parse_args()
    try:
        if args.command == "freeze":
            contract_bytes = CONTRACT.read_bytes()
            lock = build_lock(
                load_contract(), load_discovery(args.discovery), contract_bytes
            )
            args.lock.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
        else:
            lock = check(args.discovery, args.lock)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, str(error) + "\n")
    summary = {name: entry["expected"] for name, entry in lock["profiles"].items()}
    sys.stdout.write(
        json.dumps({"totals": lock["totals"], "profiles": summary}, indent=2) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

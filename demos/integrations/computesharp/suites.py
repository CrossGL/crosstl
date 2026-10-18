#!/usr/bin/env python3
"""Inventory of every upstream test project and the suites the port executes.

``test_projects.json`` assigns every project under the pinned ``tests/``
directory a scope (``ported``, ``windows-specific``, ``cpu-only`` or
``out-of-scope``) with a reason, and lists the steps of the upstream
``dotnet.yml`` workflow that build or run it. ``check`` requires the inventory
to match the pinned checkout exactly: every project directory is listed, no
listed project is missing, and the recorded steps are exactly the workflow
steps that reference the project. Every ported project names its coverage
contract, lock, evidence policy and negative controls; a ported program (an
executable without MSTest methods) is one case, its entry point, which
``program_runner.py`` runs. Every contract must
share the target profiles, capabilities and categories of the main contract,
so one profile cannot mean different devices in different suites.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from source import PIN  # noqa: E402

INVENTORY = HERE / "test_projects.json"
KIND = "computesharp.test-projects"
SCOPES = ("ported", "windows-specific", "cpu-only", "out-of-scope")
PROJECT_KINDS = ("mstest", "program")
SUITE_FILES = ("contract", "lock", "evidencePolicy", "negativeControls")
MAIN_SUITE = "ComputeSharp.Tests"
SHARED_CONTRACT_KEYS = ("categories", "profiles", "deviceRowPolicy", "capabilities")
_PROJECT_REFERENCE = re.compile(r"tests[\\/]([A-Za-z0-9_.]+)[\\/]")


@dataclass(frozen=True)
class Suite:
    """One ported upstream test project and the files that freeze its coverage."""

    name: str
    kind: str
    contract: Path | None = None
    lock: Path | None = None
    evidence_policy: Path | None = None
    negative_controls: Path | None = None
    image_checks: Path | None = None
    # Why no translated artifact of the suite does float arithmetic, if so.
    no_float_arithmetic_reason: str | None = None

    @property
    def project(self) -> Path:
        return Path("tests") / self.name / f"{self.name}.csproj"

    @property
    def assembly(self) -> Path:
        return (
            Path("tests")
            / self.name
            / "bin"
            / "Release"
            / "net10.0"
            / f"{self.name}.dll"
        )


def load_inventory(path: Path = INVENTORY) -> dict:
    inventory = json.loads(path.read_text(encoding="utf-8"))
    if inventory.get("kind") != KIND or inventory.get("schemaVersion") != 1:
        raise ValueError(f"{path} is not a {KIND} version 1")
    if inventory.get("commit") != PIN:
        raise ValueError("Test project inventory is for a different upstream revision")
    if set(inventory.get("scopes", {})) != set(SCOPES):
        raise ValueError("Test project inventory must define exactly the known scopes")
    projects = inventory.get("projects")
    if not isinstance(projects, dict) or MAIN_SUITE not in projects:
        raise ValueError("Test project inventory does not list the main test project")
    for name, entry in projects.items():
        if entry.get("scope") not in SCOPES:
            raise ValueError(f"{name} has an unknown scope")
        if entry.get("kind") not in PROJECT_KINDS:
            raise ValueError(f"{name} has an unknown project kind")
        reason = entry.get("reason")
        if not isinstance(reason, str) or len(reason.strip()) < 30:
            raise ValueError(f"{name} needs a specific reason")
        steps = entry.get("upstreamSteps")
        if not isinstance(steps, list) or not all(isinstance(s, str) for s in steps):
            raise ValueError(f"{name} must list its upstream workflow steps")
        for step in entry.get("variants", {}):
            if step not in steps:
                raise ValueError(
                    f"{name} describes a variant of an unknown step {step}"
                )
        files = entry.get("suite")
        if entry["scope"] == "ported":
            if not isinstance(files, dict) or not set(SUITE_FILES) <= set(files):
                raise ValueError(f"{name} must name its {', '.join(SUITE_FILES)}")
            reason = files.get("noFloatArithmeticReason")
            if reason is not None and (
                not isinstance(reason, str) or len(reason.strip()) < 30
            ):
                raise ValueError(f"{name} needs a specific noFloatArithmeticReason")
        elif files is not None:
            raise ValueError(f"{name} is not a ported project but names suite files")
    return inventory


def _suite(name: str, entry: dict) -> Suite:
    files = entry.get("suite") or {}

    def resolve(key: str) -> Path | None:
        value = files.get(key)
        return None if value is None else HERE / value

    return Suite(
        name=name,
        kind=entry["kind"],
        contract=resolve("contract"),
        lock=resolve("lock"),
        evidence_policy=resolve("evidencePolicy"),
        negative_controls=resolve("negativeControls"),
        image_checks=resolve("imageChecks"),
        no_float_arithmetic_reason=files.get("noFloatArithmeticReason"),
    )


def ported_suites(inventory: dict | None = None) -> dict[str, Suite]:
    """Return every ported project in name order."""

    inventory = inventory or load_inventory()
    return {
        name: _suite(name, entry)
        for name, entry in sorted(inventory["projects"].items())
        if entry["scope"] == "ported"
    }


def suite(name: str, inventory: dict | None = None) -> Suite:
    suites = ported_suites(inventory)
    if name not in suites:
        raise ValueError(f"{name} is not a ported upstream test project")
    return suites[name]


def workflow_steps(workflow: dict) -> dict[str, list[str]]:
    """Map each test project to the ``job: step`` names that reference it."""

    references: dict[str, set[str]] = {}
    for job_name, job in (workflow.get("jobs") or {}).items():
        for step in job.get("steps") or []:
            command = step.get("run")
            if not isinstance(command, str):
                continue
            label = f"{job_name}: {step.get('name', '')}"
            for project in _PROJECT_REFERENCE.findall(command):
                references.setdefault(project, set()).add(label)
    return {name: sorted(labels) for name, labels in sorted(references.items())}


def _load_workflow(path: Path) -> dict:
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _shared(contract: dict, key: str):
    """The part of a contract every suite must share; capability gates name
    methods of their own suite."""

    value = contract.get(key)
    if key == "capabilities" and isinstance(value, dict):
        return {
            name: {field: v for field, v in entry.items() if field != "gates"}
            for name, entry in value.items()
        }
    return value


def check_contracts(inventory: dict) -> list[str]:
    """Every ported suite has its files, shares the profiles of the main
    contract and has a lock frozen from its current contract."""

    problems = []
    suites = ported_suites(inventory)
    main = json.loads(suites[MAIN_SUITE].contract.read_text(encoding="utf-8"))
    for name, item in suites.items():
        for path in (
            item.contract,
            item.lock,
            item.evidence_policy,
            item.negative_controls,
        ):
            if not path.is_file():
                problems.append(f"{name}: missing {path.relative_to(HERE)}")
        if not item.contract.is_file():
            continue
        contract = json.loads(item.contract.read_text(encoding="utf-8"))
        for key in SHARED_CONTRACT_KEYS:
            if _shared(contract, key) != _shared(main, key):
                problems.append(f"{name}: contract {key} differs from {MAIN_SUITE}")
        if item.lock.is_file():
            lock = json.loads(item.lock.read_text(encoding="utf-8"))
            digest = hashlib.sha256(
                item.contract.read_bytes().replace(b"\r\n", b"\n")
            ).hexdigest()
            if lock.get("contractSHA256") != digest:
                problems.append(
                    f"{name}: the lock was not frozen from the current contract"
                )
    return problems


def check(source: Path, inventory_path: Path = INVENTORY) -> dict:
    """Compare the inventory with a pinned upstream checkout."""

    inventory = load_inventory(inventory_path)
    projects = inventory["projects"]
    problems = []
    tests = source / "tests"
    on_disk = sorted(
        path.name
        for path in tests.iterdir()
        if path.is_dir() and any(path.glob("*.csproj"))
    )
    for name in sorted(set(on_disk) - set(projects)):
        problems.append(f"unlisted upstream test project {name}")
    for name in sorted(set(projects) - set(on_disk)):
        problems.append(f"listed project {name} is not in the upstream checkout")
    for name in sorted(set(projects) & set(on_disk)):
        if not (tests / name / f"{name}.csproj").is_file():
            problems.append(f"{name} has no {name}.csproj")
    steps = workflow_steps(_load_workflow(source / inventory["workflow"]))
    for name in sorted(set(steps) - set(projects)):
        problems.append(f"workflow references unlisted project {name}")
    for name, entry in sorted(projects.items()):
        if sorted(entry["upstreamSteps"]) != steps.get(name, []):
            problems.append(
                f"{name}: recorded steps {sorted(entry['upstreamSteps'])} "
                f"differ from the workflow {steps.get(name, [])}"
            )
    problems.extend(check_contracts(inventory))
    if problems:
        raise ValueError("Test project inventory mismatch:\n" + "\n".join(problems))
    return summary(inventory)


def summary(inventory: dict) -> dict:
    scopes: dict[str, list[str]] = {scope: [] for scope in SCOPES}
    for name, entry in sorted(inventory["projects"].items()):
        scopes[entry["scope"]].append(name)
    return {"projects": len(inventory["projects"]), "scopes": scopes}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    command = sub.add_parser("check", help="compare with a pinned upstream checkout")
    command.add_argument("--source", required=True, type=Path)
    sub.add_parser("list", help="print the ported suites")
    args = parser.parse_args(argv)
    try:
        if args.command == "check":
            report = check(args.source)
        else:
            report = {
                name: {"kind": item.kind, "project": item.project.as_posix()}
                for name, item in ported_suites().items()
            }
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, str(error) + "\n")
    sys.stdout.write(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

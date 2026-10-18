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

A Windows-specific or CPU-only project is either retained, with a contract and
lock that classify all of its cases in that category for the original
``directx-warp`` profile alone and the ``dotnet test`` runs that reproduce its
upstream steps (``retained.py`` executes them), or names the reason it is not.
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
# The category of every case of a retained project of each scope.
RETAINED_SCOPES = {"windows-specific": "windows-specific", "cpu-only": "cpu-only"}
RETAINED_PROFILE = "directx-warp"
CONFIGURATIONS = ("Debug", "Release")
_RUN_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")
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
    scope: str = "ported"
    # How a retained project builds and runs (``retained`` of its inventory entry).
    retention: dict | None = None

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
        _check_retention(name, entry)
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


def _reason(value) -> bool:
    return isinstance(value, str) and len(value.strip()) >= 30


def _check_retention(name: str, entry: dict) -> None:
    """A retained project reproduces every upstream step it does not explain."""

    retained = entry.get("retained")
    if entry["scope"] not in RETAINED_SCOPES:
        if retained is not None or "notRetainedReason" in entry:
            raise ValueError(f"{name} is not Windows-specific or CPU-only")
        return
    if retained is None:
        if not _reason(entry.get("notRetainedReason")):
            raise ValueError(f"{name} must be retained or say why it is not")
        return
    if "notRetainedReason" in entry or not isinstance(retained, dict):
        raise ValueError(f"{name} is retained and must not say why it is not")
    if not all(isinstance(retained.get(key), str) for key in ("contract", "lock")):
        raise ValueError(f"{name} must name its retained contract and lock")
    if retained.get("configuration") not in CONFIGURATIONS:
        raise ValueError(f"{name} must name the configuration its steps build")
    if type(retained.get("hostOnly")) is not bool:
        raise ValueError(f"{name} must state whether it is host-only")
    runs = retained.get("runs")
    if not isinstance(runs, list) or not runs:
        raise ValueError(f"{name} must list its retained runs")
    names, reproduced = set(), set()
    for run in runs:
        label = run.get("name") if isinstance(run, dict) else None
        if not isinstance(label, str) or not _RUN_NAME.match(label) or label in names:
            raise ValueError(f"{name} has an invalid or repeated run name")
        names.add(label)
        if run.get("filter") is not None and not isinstance(run["filter"], str):
            raise ValueError(f"{name} run {label} has an invalid filter")
        if type(run.get("accounted")) is not bool:
            raise ValueError(f"{name} run {label} must state whether it is accounted")
        step = run.get("upstreamStep")
        if step is not None:
            if step not in entry["upstreamSteps"]:
                raise ValueError(f"{name} run {label} reproduces an unknown step")
            reproduced.add(step)
        elif not run["accounted"] or not _reason(run.get("reason")):
            raise ValueError(f"{name} run {label} must reproduce a step or say why")
    if not any(run["accounted"] for run in runs):
        raise ValueError(f"{name} has no accounted run")
    unexplained = (
        set(entry["upstreamSteps"]) - reproduced - set(entry.get("variants", {}))
    )
    if unexplained:
        raise ValueError(
            f"{name} neither reproduces nor explains {', '.join(sorted(unexplained))}"
        )


def _suite(name: str, entry: dict) -> Suite:
    files = entry.get("suite") or entry.get("retained") or {}

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
        scope=entry["scope"],
        retention=entry.get("retained"),
    )


def ported_suites(inventory: dict | None = None) -> dict[str, Suite]:
    """Return every ported project in name order."""

    inventory = inventory or load_inventory()
    return {
        name: _suite(name, entry)
        for name, entry in sorted(inventory["projects"].items())
        if entry["scope"] == "ported"
    }


def retained_suites(inventory: dict | None = None) -> dict[str, Suite]:
    """Return every retained Windows-specific or CPU-only project in name order."""

    inventory = inventory or load_inventory()
    return {
        name: _suite(name, entry)
        for name, entry in sorted(inventory["projects"].items())
        if entry.get("retained") is not None
    }


def suite(name: str, inventory: dict | None = None, *, retained: bool = False) -> Suite:
    """The ported project ``name``, or also a retained one if ``retained``."""

    suites = ported_suites(inventory)
    if retained:
        suites.update(retained_suites(inventory))
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
        problems.extend(_check_lock_frozen(name, item))
    for name, item in retained_suites(inventory).items():
        problems.extend(_check_retained_contract(name, item, main))
    return problems


def _check_lock_frozen(name: str, item: Suite) -> list[str]:
    if not item.lock.is_file():
        return []
    lock = json.loads(item.lock.read_text(encoding="utf-8"))
    digest = hashlib.sha256(
        item.contract.read_bytes().replace(b"\r\n", b"\n")
    ).hexdigest()
    if lock.get("contractSHA256") != digest:
        return [f"{name}: the lock was not frozen from the current contract"]
    host_only = bool(item.retention and item.retention.get("hostOnly"))
    if lock.get("hostOnly", False) is not host_only:
        return [f"{name}: the lock and the inventory disagree on host-only export"]
    return []


def _check_retained_contract(name: str, item: Suite, main: dict) -> list[str]:
    """A retained contract has the original profile alone and the scope's category."""

    problems = [
        f"{name}: missing {path.relative_to(HERE)}"
        for path in (item.contract, item.lock)
        if not path.is_file()
    ]
    if not item.contract.is_file():
        return problems
    contract = json.loads(item.contract.read_text(encoding="utf-8"))
    for key in SHARED_CONTRACT_KEYS:
        wanted = _shared(main, key)
        if key == "profiles":
            wanted = {RETAINED_PROFILE: main["profiles"][RETAINED_PROFILE]}
        if _shared(contract, key) != wanted:
            problems.append(f"{name}: contract {key} differs from {MAIN_SUITE}")
    category = RETAINED_SCOPES[item.scope]
    entries = {**contract.get("classes", {}), **contract.get("methods", {})}
    if not entries or any(e.get("category") != category for e in entries.values()):
        problems.append(f"{name}: every case of the project must be {category}")
    if any(entry.get("gates") for entry in contract.get("capabilities", {}).values()):
        problems.append(f"{name}: a retained project has no capability gates")
    return problems + _check_lock_frozen(name, item)


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
                name: {
                    "kind": item.kind,
                    "scope": item.scope,
                    "project": item.project.as_posix(),
                }
                for name, item in {**ported_suites(), **retained_suites()}.items()
            }
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, str(error) + "\n")
    sys.stdout.write(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

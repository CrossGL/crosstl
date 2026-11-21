"""Tests for the heavy MLX leg selection in ``ci_legs``."""

import json
from pathlib import Path

import pytest
import yaml

from demos.integrations.mlx import ci_legs
from demos.integrations.mlx.ci_legs import (
    EVERY_LEG,
    HEAVY,
    Closure,
    PathIndex,
    affected_legs,
    changed_legs,
    leg_entries,
    workflow_legs,
)

ROOT = Path(__file__).resolve().parents[4]
WORKFLOW = ROOT / ".github/workflows/demo-project-testing.yml"

# A small MLX demo: one host test, two kernel groups and the files they read.
TREE = {
    "__init__.py": "",
    "portable_host/__init__.py": "",
    "portable_host/runtime.py": "def run():\n    return 1\n",
    "tests/host/test_host.py": (
        "from demos.integrations.mlx.portable_host import runtime\n"
    ),
    "tests/kernels/conftest.py": "",
    "tests/kernels/test_sort.py": (
        '"""Reads contracts/scan.current-tree.translation.json in prose only."""\n'
        "from pathlib import Path\n"
        'CONTRACT = Path(__file__).parents[2] / "contracts" / '
        '"sort.current-tree.translation.json"\n'
    ),
    "tests/kernels/test_scan.py": (
        'KIND = "current-tree.translation"\nPATH = f"contracts/scan.{KIND}.json"\n'
    ),
    "tests/kernels/test_unused.py": "",
    "contracts/sort.current-tree.translation.json": '{"helper": "sort_helper.py"}',
    "contracts/scan.current-tree.translation.json": "{}",
    "sort_helper.py": "",
    "orphan.py": "",
    "README.md": "See orphan.py.\n",
}
MLX = "demos/integrations/mlx"
DEMO_WORKFLOW = f"""
name: Demo
on:
  push:
    branches: [main]
jobs:
  ci-plan:
    runs-on: ubuntu-24.04
    steps:
      - run: echo plan
  host:
    needs: ci-plan
    if: {HEAVY}
    runs-on: ubuntu-24.04
    steps:
      - run: python -m pytest {MLX}/tests/host/test_host.py
  parity:
    needs: ci-plan
    if: {HEAVY}
    runs-on: ubuntu-24.04
    strategy:
      matrix:
        os: [ubuntu-latest, windows-latest]
        group: [sort, scan]
    steps:
      - run: echo shared
      - if: matrix.group == 'sort'
        run: python -m pytest {MLX}/tests/kernels/test_sort.py
      - if: matrix.group == 'scan'
        run: python -m pytest {MLX}/tests/kernels/test_scan.py
  translator:
    needs: ci-plan
    if: {HEAVY}
    runs-on: ubuntu-24.04
    steps:
      - run: python -m pytest tests/test_translator
"""


def _legs(workflow=DEMO_WORKFLOW):
    return workflow_legs(yaml.safe_load(workflow))


def _closure():
    return Closure(PathIndex(TREE), TREE.get)


def test_heavy_jobs_and_matrix_groups_are_legs():
    legs = _legs()
    assert sorted(legs) == ["host", "parity[scan]", "parity[sort]", "translator"]
    sort = legs["parity[sort]"]
    assert [step["run"] for step in sort["steps"]] == [
        "echo shared",
        f"python -m pytest {MLX}/tests/kernels/test_sort.py",
    ]
    # The other matrix axes stay part of every group's definition.
    assert sort["strategy"]["matrix"] == {"os": ["ubuntu-latest", "windows-latest"]}
    assert sort["matrix-entry"] == {"group": "sort"}


def test_a_leg_reads_imports_loaded_modules_path_strings_and_data_files():
    legs, closure = _legs(), _closure()

    def files(leg):
        return closure.of(leg_entries(legs[leg], closure.index))

    assert files("host") == {
        "__init__.py",
        "portable_host/__init__.py",
        "portable_host/runtime.py",
        "tests/host/test_host.py",
    }
    # The contract names the helper; the docstring's path names nothing.
    assert files("parity[sort]") == {
        "__init__.py",
        "tests/kernels/conftest.py",
        "tests/kernels/test_sort.py",
        "contracts/sort.current-tree.translation.json",
        "sort_helper.py",
    }
    # A formatted path names the files that contain its fixed pieces.
    assert "contracts/scan.current-tree.translation.json" in files("parity[scan]")
    assert "contracts/sort.current-tree.translation.json" not in files("parity[scan]")
    assert files("translator") == set()


@pytest.mark.parametrize(
    "changed,selected",
    [
        (["sort_helper.py"], {"parity[sort]"}),
        (["portable_host/runtime.py"], {"host"}),
        (["tests/kernels/conftest.py"], {"parity[sort]", "parity[scan]"}),
        # A file no leg reaches may be read in a way the scan cannot see.
        (["orphan.py"], {"host", "parity[sort]", "parity[scan]", "translator"}),
        # The complete test suite runs a test module that no heavy leg runs.
        (["tests/kernels/test_unused.py"], set()),
        ([], set()),
    ],
)
def test_changed_files_select_the_legs_that_read_them(changed, selected):
    assert affected_legs(_legs(), changed, _closure()) == selected


def test_a_json_document_names_files_in_keys_and_values_but_not_in_prose():
    document = {
        "helper": "sort_helper.py",
        "orphan.py": {
            "reason": "tests/kernels/test_unused.py executes the excluded entries",
            "steps": ["echo sort_helper.py", "README.md"],
        },
    }
    tree = {**TREE, "contracts/gemv.json": json.dumps(document)}
    closure = Closure(PathIndex(tree), tree.get)
    assert closure.references("contracts/gemv.json") == {
        "sort_helper.py",
        "orphan.py",
        "README.md",
    }
    # A data file that is not JSON keeps naming every file it spells.
    tree["contracts/gemv.yaml"] = "reason: tests/kernels/test_unused.py runs them\n"
    closure = Closure(PathIndex(tree), tree.get)
    assert closure.references("contracts/gemv.yaml") == {"tests/kernels/test_unused.py"}


def test_a_deleted_file_that_no_leg_reaches_selects_every_leg():
    closure = Closure(PathIndex({*TREE, "removed.py"}), TREE.get)
    assert affected_legs(_legs(), ["removed.py"], closure) == set(_legs())


def test_directory_listings_read_every_file_below_them():
    tree = {
        "tests/test_listing.py": (
            "from pathlib import Path\n"
            'ROOT = Path("demos/integrations/mlx/contracts")\n'
            'FILES = sorted(ROOT.glob("*.json"))\n'
        ),
        "contracts/a.json": "{}",
        "contracts/b.json": "{}",
        "other/c.json": "{}",
    }
    closure = Closure(PathIndex(tree), tree.get)
    assert closure.of({"tests/test_listing.py"}) == {
        "tests/test_listing.py",
        "contracts/a.json",
        "contracts/b.json",
    }


def test_a_formatted_family_path_names_every_family_file():
    tree = {
        "tests/test_family.py": (
            "def contract(family):\n"
            '    return f"contracts/{family}.current-tree.translation.json"\n'
        ),
        "contracts/sort.current-tree.translation.json": "{}",
        "contracts/scan.current-tree.translation.json": "{}",
        "contracts/sort.dispatch.json": "{}",
    }
    closure = Closure(PathIndex(tree), tree.get)
    assert closure.of({"tests/test_family.py"}) == {
        "tests/test_family.py",
        "contracts/sort.current-tree.translation.json",
        "contracts/scan.current-tree.translation.json",
    }


def test_workflow_changes_select_the_legs_they_alter():
    before = yaml.safe_load(DEMO_WORKFLOW)
    after = yaml.safe_load(DEMO_WORKFLOW.replace("echo shared", "echo changed"))
    assert changed_legs(before, after) == {"parity[sort]", "parity[scan]"}
    after = yaml.safe_load(
        DEMO_WORKFLOW.replace("tests/kernels/test_scan.py", "tests/kernels/x.py")
    )
    assert changed_legs(before, after) == {"parity[scan]"}
    after = yaml.safe_load(DEMO_WORKFLOW.replace("[main]", "[release]"))
    assert changed_legs(before, after) is EVERY_LEG
    assert changed_legs(None, before) is EVERY_LEG
    added = DEMO_WORKFLOW.replace("group: [sort, scan]", "group: [sort, scan, fft]")
    assert changed_legs(before, yaml.safe_load(added)) == {"parity[fft]"}


def test_every_real_heavy_job_is_planned_and_keeps_its_steps():
    workflow = yaml.safe_load(WORKFLOW.read_text())
    legs = workflow_legs(workflow)
    for name, job in workflow["jobs"].items():
        planned = [leg for leg in legs if ci_legs.job_of(leg) == name]
        if str(job.get("if", "")).strip() != HEAVY:
            assert not planned, name
            continue
        assert planned, name
        matrix = (job.get("strategy") or {}).get("matrix") or {}
        if "group" not in matrix:
            continue
        values = {("group", group) for group in matrix["group"]} | {
            (key, str(value))
            for entry in matrix.get("include", [])
            for key, value in entry.items()
        }
        kept = {
            json.dumps(step, sort_keys=True)
            for leg in planned
            for step in legs[leg]["steps"]
        }
        for step in job["steps"]:
            tests = ci_legs._MATRIX_TEST.findall(str(step.get("if", "")))
            if not tests or any(test in values for test in tests):
                assert json.dumps(step, sort_keys=True) in kept, (name, step)
        for leg in planned:
            # Each group runs a step of its own, so it is not the whole job.
            assert any("matrix." in str(s.get("if", "")) for s in legs[leg]["steps"])

"""The defect ledger: every S, V and X defect of the 2026-10-16 audit has a status backed by evidence.

``defects.json`` records, for each defect, whether it is fixed, refused, still open or not yet re-verified, and the
evidence: conformance cases (whose cells in ``scoreboards/`` must then show no divergence), translation probes
under ``defect_probes/`` that run here, and regression tests elsewhere in the suite. A probe of an open defect is a
strict expected failure, so fixing the defect fails this test until its status is updated.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from crosstl import translate

from .cases import BY_ID

ROOT = Path(__file__).resolve().parent
REPOSITORY = ROOT.parent.parent
LEDGER = json.loads((ROOT / "defects.json").read_text(encoding="utf-8"))
DEFECTS = {entry["id"]: entry for entry in LEDGER["defects"]}
PROBES = ROOT / "defect_probes"
SCOREBOARDS = sorted((ROOT / "scoreboards").glob("*.json"))

EXPECTED_IDS = (
    [f"S{n}" for n in range(1, 32)]
    + [f"V{n}" for n in range(1, 13)]
    + [f"X{n}" for n in range(1, 4)]
)
CLASSES = {"S": "silent", "V": "invalid", "X": "crash"}
STATUSES = set(LEDGER["statuses"])
CLOSED = {"fixed", "refused"}
# Outcomes that mean a translation is not usable on that runtime.
UNUSABLE = {"diverged", "invalid", "interface-changed", "driver-rejected"}


def _scoreboard_cells():
    cells = {}
    for path in SCOREBOARDS:
        for identity, cell in json.loads(path.read_text(encoding="utf-8"))[
            "cells"
        ].items():
            cells[(path.stem, identity)] = cell["outcome"]
    return cells


CELLS = _scoreboard_cells()


def _probe_parameters():
    for entry in LEDGER["defects"]:
        for index, probe in enumerate(entry.get("probes", [])):
            marks = []
            if entry["status"] not in CLOSED:
                marks.append(
                    pytest.mark.xfail(
                        strict=True, reason=f"{entry['id']} is {entry['status']}"
                    )
                )
            yield pytest.param(
                entry["id"], probe, id=f"{entry['id']}-{index}", marks=marks
            )


def test_the_ledger_lists_every_audited_defect_once():
    identifiers = [entry["id"] for entry in LEDGER["defects"]]
    assert identifiers == EXPECTED_IDS


@pytest.mark.parametrize("identifier", EXPECTED_IDS)
def test_each_entry_has_a_status_and_matching_evidence(identifier):
    entry = DEFECTS[identifier]
    assert entry["class"] == CLASSES[identifier[0]]
    assert entry["status"] in STATUSES
    assert entry["component"] and entry["summary"]
    if entry["status"] in CLOSED:
        assert (
            entry.get("cases") or entry.get("probes") or entry.get("tests")
        ), f"{identifier} is {entry['status']} without evidence"
        if entry["status"] == "refused":
            assert entry.get("probes") or entry.get(
                "tests"
            ), f"{identifier}: a refusal needs a probe or a test that observes it"
    else:
        assert entry.get("note"), f"{identifier} is {entry['status']} without a note"


@pytest.mark.parametrize("identifier", EXPECTED_IDS)
def test_cited_tests_exist(identifier):
    for reference in DEFECTS[identifier].get("tests", []):
        path, _, name = reference.partition("::")
        source = REPOSITORY / path
        assert source.is_file(), reference
        if name:
            assert re.search(
                rf"^\s*def {re.escape(name)}\(",
                source.read_text(encoding="utf-8"),
                re.M,
            ), reference


@pytest.mark.parametrize("identifier", EXPECTED_IDS)
def test_cited_cases_are_measured_without_divergence(identifier):
    """Every recorded cell of a closed defect's cases is usable, except the cells the entry explains."""
    entry = DEFECTS[identifier]
    targets = set(entry.get("targets", []))
    excepted = {item["cell"]: item["reason"] for item in entry.get("exceptions", [])}
    assert all(excepted.values()), f"{identifier}: every exception needs a reason"
    seen_exceptions = set()
    for case in entry.get("cases", []):
        assert case in BY_ID, f"{identifier} cites unknown case {case}"
        cells = {
            key: outcome
            for key, outcome in CELLS.items()
            if key[1].split("|")[0] == case
            and (not targets or key[1].split("|")[2] in targets)
        }
        assert cells, f"{identifier}: case {case} has no recorded cell"
        if entry["status"] not in CLOSED:
            continue
        for (profile, identity), outcome in sorted(cells.items()):
            if outcome not in UNUSABLE:
                continue
            assert (
                identity in excepted
            ), f"{identifier}: {profile} {identity} is {outcome}"
            seen_exceptions.add(identity)
    assert (
        seen_exceptions == set(excepted) or entry["status"] not in CLOSED
    ), f"{identifier}: stale exceptions {sorted(set(excepted) - seen_exceptions)}"


def _validate(validator, output, tmp_path):
    if validator == "naga":
        if shutil.which("naga") is None:
            return
        path = tmp_path / "probe.wgsl"
        path.write_text(output, encoding="utf-8")
        subprocess.run(["naga", str(path)], check=True, capture_output=True)
    else:  # pragma: no cover - guarded by the ledger schema test
        raise AssertionError(f"unknown validator {validator}")


@pytest.mark.parametrize(("identifier", "probe"), list(_probe_parameters()))
def test_probe(identifier, probe, tmp_path):
    source = PROBES / probe["source"]
    assert source.is_file(), probe["source"]
    if probe["expect"] == "refused":
        with pytest.raises(Exception) as raised:
            translate(str(source), backend=probe["target"], format_output=False)
        assert getattr(
            raised.value, "project_diagnostic_code", None
        ), f"{identifier}: {type(raised.value).__name__} is not a typed diagnostic"
        return
    assert probe["expect"] == "translated"
    output = translate(str(source), backend=probe["target"], format_output=False)
    for pattern in probe.get("present", []):
        assert re.search(pattern, output), f"{identifier}: missing {pattern!r}"
    for pattern in probe.get("absent", []):
        assert not re.search(pattern, output), f"{identifier}: found {pattern!r}"
    if probe.get("validator"):
        _validate(probe["validator"], output, tmp_path)


def test_probe_sources_are_all_used():
    used = {
        probe["source"]
        for entry in LEDGER["defects"]
        for probe in entry.get("probes", [])
    }
    assert {path.name for path in PROBES.iterdir()} == used


def test_readme_showcase_glsl_validates_per_stage(tmp_path):
    """V4: the README PBR shader routes its position to gl_Position and each stage compiles."""
    glslang = shutil.which("glslangValidator")
    output = translate(
        str(REPOSITORY / "demos" / "readme" / "showcase_pbr.cgl"),
        backend="opengl",
        format_output=False,
    )
    assert "gl_Position" in output
    if glslang is None:
        pytest.skip("glslangValidator is not installed")
    path = tmp_path / "showcase.glsl"
    path.write_text(output, encoding="utf-8")
    for stage in ("vert", "frag"):
        subprocess.run(
            [glslang, "-S", stage, str(path)], check=True, capture_output=True
        )

"""Refreshing MLX generated-artifact pins from report-mode drift records."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from demos.integrations.mlx import refresh_pins

SCOPE = Path("demos/integrations/mlx")
OLD_A, NEW_A = "a" * 64, "1" * 64
OLD_B, NEW_B = "b" * 64, "2" * 64
OLD_C, NEW_C = "c" * 64, "3" * 64
OLD_D, NEW_D = "d" * 64, "4" * 64
UNCHANGED = "e" * 64


def _record(subject, field, expected, actual, test="t.py::case"):
    return {
        "subject": subject,
        "field": field,
        "expected": expected,
        "actual": actual,
        "test": test,
    }


def _drift(subject, old_sha, new_sha, old_size=None, new_size=None, test="t.py::x"):
    records = [_record(subject, "generatedHash", old_sha, new_sha, test)]
    if old_size is not None:
        records.append(_record(subject, "generatedSizeBytes", old_size, new_size, test))
    return records


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_drift_lines_are_read_from_job_logs_and_drift_files():
    record = _record("out/k.hlsl", "generatedHash", OLD_A, NEW_A)
    log = "\n".join(
        [
            "2026-10-29T11:14:10.7725127Z ##[group]Run pytest",
            "2026-10-29T11:14:10.7725127Z MLX-PIN-DRIFT " + json.dumps(record),
            '2026-10-29T11:14:10.7725127Z {"field": "x", "expected": 1, "actual": 2}',
        ]
    )
    assert refresh_pins.parse_drift_text(log, prefixed_only=True) == [record]
    assert refresh_pins.parse_drift_text(json.dumps(record) + "\nnot json\n") == [
        record
    ]
    quoted = '    print("MLX-PIN-DRIFT " + json.dumps(record), file=sys.stderr)'
    assert refresh_pins.parse_drift_text(quoted, prefixed_only=True) == []
    with pytest.raises(refresh_pins.RefreshError, match="Unreadable drift record"):
        refresh_pins.parse_drift_text("MLX-PIN-DRIFT {broken")


def test_hash_and_size_drift_pair_per_subject_and_merge_across_tests():
    records = [
        *_drift("out/a.glsl", OLD_A, NEW_A, 100, 120, test="one.py::a"),
        *_drift("out/a.glsl", OLD_A, NEW_A, test="two.py::a"),
        _record("out/b.glsl", "generatedSizeBytes", 5, 6),
    ]
    pins, size_only = refresh_pins.build_pins(records)
    assert pins == {OLD_A: refresh_pins.Pin(OLD_A, NEW_A, 100, 120)}
    assert size_only == [
        {"subject": "out/b.glsl", "test": "t.py::case", "size": (5, 6)}
    ]


@pytest.mark.parametrize(
    "records, message",
    [
        (
            [
                *_drift("x", OLD_A, NEW_A, test="one"),
                *_drift("y", OLD_A, NEW_B, test="two"),
            ],
            "drifted to both",
        ),
        (
            [
                *_drift("x", OLD_A, NEW_A, 1, 2, "one"),
                *_drift("y", OLD_A, NEW_A, 1, 3, "two"),
            ],
            "different sizes",
        ),
        (_drift("x", "not-a-digest", NEW_A), "is not a SHA-256"),
        ([_record("x", "generatedThing", 1, 2)], "Unknown drift field"),
    ],
)
def test_contradictory_drift_is_refused(records, message):
    with pytest.raises(refresh_pins.RefreshError, match=message):
        refresh_pins.build_pins(records)


def _contract():
    entries = [
        {
            "entryPoint": "k_small",
            "sha256": OLD_A,
            "sizeBytes": 100,
            "dxilSha256": UNCHANGED,
            "dxilSizeBytes": 100,
        },
        {"entryPoint": "k_large", "sha256": OLD_B, "sizeBytes": 300},
        {"entryPoint": "k_middle", "sha256": UNCHANGED, "sizeBytes": 200},
    ]
    return {
        "entries": entries,
        "artifactContract": {
            "generatedSizeBytesTotal": 600,
            "generatedSizeRange": {
                "minimum": {"entryPoint": "k_small", "sizeBytes": 100},
                "maximum": {"entryPoint": "k_large", "sizeBytes": 300},
            },
        },
        "proof": {"generatedSizeBytesTotal": 400},
    }


def _tree(root: Path) -> dict[str, Path]:
    base = root / SCOPE
    (base / "contracts").mkdir(parents=True)
    (base / "tests").mkdir()
    contract = base / "contracts" / "family.translation.json"
    contract.write_text(json.dumps(_contract(), indent=2) + "\n", encoding="utf-8")
    compact = base / "contracts" / "compact.json"
    compact.write_text(
        '{\n  "rows": {"k": {"metal": "'
        + OLD_C
        + ':4096", "opengl": "'
        + UNCHANGED
        + ':77"}}\n}\n',
        encoding="utf-8",
    )
    digest, size = _sha(contract), contract.stat().st_size
    test = base / "tests" / "test_family.py"
    test.write_text(
        f"""CONTRACT = "contracts/family.translation.json"
FAMILY_CONTRACT_SHA256 = (
    "{digest}"
)
FAMILY_CONTRACT_SIZE_BYTES = {size}
FAMILY_GENERATED_SIZE_BYTES_TOTAL = 600
FAMILY_GENERATED_SIZE_MINIMUM = ("k_small", 100)
SUMMARY = {{
    "maximum": {{
        "entryPoint": (
            "k_lar"
            "ge"
        ),
        "sizeBytes": 300,
    }},
}}
# the reviewed loader artifact isn't in the contract
LOADER_GENERATED_SHA256 = (
    "{OLD_D}"
)
LOADER_GENERATED_SIZE_BYTES = 7129


def test_status(status):
    assert sum(entry["sizeBytes"] for entry in status) == 600
    assert max((e["sizeBytes"], e["entryPoint"]) for e in status) == (300, "k_large")
    assert status["artifacts"]["metal"]["sha256"] == (
        "{OLD_C}"
    )
    assert status["artifacts"]["metal"]["size_bytes"] == 4096
    check(artifact, sha256="{OLD_A}", size_bytes=100)
    assert "are 7,129 and 9,000 bytes" in README
""",
        encoding="utf-8",
    )
    readme = base / "README.md"
    readme.write_text(
        "# Demo\n\nThe loader artifacts are 7,129 and 9,000 bytes with SHA-256\n"
        f"`{OLD_D}`\nand `{UNCHANGED}`.\n\n"
        "`contracts/family.translation.json` pins 600 generated bytes.\n",
        encoding="utf-8",
    )
    return {"contract": contract, "compact": compact, "test": test, "readme": readme}


def _pins():
    records = [
        *_drift("out/k_small.hlsl", OLD_A, NEW_A, 100, 1500),
        *_drift("out/k_large.hlsl", OLD_B, NEW_B, 300, 290),
        *_drift("out/k.metal", OLD_C, NEW_C, 4096, 4100),
        *_drift("out/loader.hlsl", OLD_D, NEW_D, 7129, 7200),
    ]
    pins, size_only = refresh_pins.build_pins(records)
    assert size_only == []
    return pins


def test_refresh_rewrites_leaves_summaries_mirrors_and_file_pins(tmp_path):
    paths = _tree(tmp_path)
    before = {name: path.read_text() for name, path in paths.items()}

    dry = refresh_pins.refresh(_pins(), tmp_path, (SCOPE,))
    assert dry.complete, dry.to_json()
    assert {name: path.read_text() for name, path in paths.items()} == before

    report = refresh_pins.refresh(_pins(), tmp_path, (SCOPE,), write=True)
    assert report.complete, report.to_json()
    assert report.to_json()["files"] == dry.to_json()["files"]

    contract = json.loads(paths["contract"].read_text())
    assert [(e["sha256"], e["sizeBytes"]) for e in contract["entries"]] == [
        (NEW_A, 1500),
        (NEW_B, 290),
        (UNCHANGED, 200),
    ]
    # The recorded DXIL identity is not recomputed and its size is not a pin.
    assert contract["entries"][0]["dxilSizeBytes"] == 100
    assert report.native_identities == {
        "demos/integrations/mlx/contracts/family.translation.json:dxilSha256": 1
    }
    assert contract["artifactContract"] == {
        "generatedSizeBytesTotal": 1990,
        "generatedSizeRange": {
            "minimum": {"entryPoint": "k_middle", "sizeBytes": 200},
            "maximum": {"entryPoint": "k_small", "sizeBytes": 1500},
        },
    }
    assert contract["proof"] == {"generatedSizeBytesTotal": 400}
    assert paths["contract"].read_text() == json.dumps(contract, indent=2) + "\n"
    assert NEW_C + ":4100" in paths["compact"].read_text()
    assert UNCHANGED + ":77" in paths["compact"].read_text()

    test = paths["test"].read_text()
    assert f'    "{_sha(paths["contract"])}"\n' in test
    assert f"FAMILY_CONTRACT_SIZE_BYTES = {paths['contract'].stat().st_size}\n" in test
    assert "FAMILY_GENERATED_SIZE_BYTES_TOTAL = 1990\n" in test
    assert 'FAMILY_GENERATED_SIZE_MINIMUM = ("k_middle", 200)\n' in test
    assert '(\n            "k_small"\n        ),\n        "sizeBytes": 1500,' in test
    assert "== 1990\n" in test
    assert '== (1500, "k_small")' in test
    assert f'    "{NEW_D}"\n)\nLOADER_GENERATED_SIZE_BYTES = 7200\n' in test
    assert (
        f'"{NEW_C}"\n    )\n    assert status["artifacts"]["metal"]["size_bytes"] == 4100'
        in test
    )
    assert f'sha256="{NEW_A}", size_bytes=1500' in test
    assert '"are 7,200 and 9,000 bytes"' in test

    readme = paths["readme"].read_text()
    assert f"are 7,200 and 9,000 bytes with SHA-256\n`{NEW_D}`" in readme
    assert "pins 1,990 generated bytes" in readme
    assert dict(report.applied) == {
        "sha256": 7,
        "size": 7,
        "totals": 1,
        "ranges": 1,
        "mirroredTotals": 3,
        "mirroredRanges": 3,
        "mirroredProse": 1,
        "fileSha256": 1,
        "fileSize": 1,
    }


def test_ambiguous_sizes_are_reported_and_left_alone(tmp_path):
    readme = tmp_path / SCOPE / "README.md"
    readme.parent.mkdir(parents=True)
    readme.write_text(
        f"Two 7,129-byte artifacts: `{OLD_D}` and a copy, also 7,129 bytes.\n",
        encoding="utf-8",
    )
    pins = _pins()
    report = refresh_pins.refresh(pins, tmp_path, (SCOPE,), write=True)
    assert not report.complete
    assert report.unresolved == [
        {
            "path": "demos/integrations/mlx/README.md",
            "line": 1,
            "sha256": OLD_D,
            "reason": "found 2 size fields equal to 7129",
        }
    ]
    assert readme.read_text().count("7,129") == 2
    assert NEW_D in readme.read_text()
    assert sorted(report.unmatched) == sorted({OLD_A, OLD_B, OLD_C})


def test_artifact_size_bytes_pairs_with_the_artifact_hash(tmp_path):
    # Gap evidence pins a corrected artifact as artifact_sha256 with
    # artifact_size_bytes, in JSON and in the Python mirror of the same record.
    gaps = tmp_path / SCOPE / "expected-gaps.json"
    harness = tmp_path / SCOPE / "tests" / "test_harness.py"
    harness.parent.mkdir(parents=True)
    gaps.write_text(
        json.dumps(
            {
                "corrected": {
                    "artifact_sha256": OLD_A,
                    "artifact_size_bytes": 100,
                    "output_element_count": 100,
                }
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    harness.write_text(
        "CORRECTED = {\n"
        f'    "artifact_sha256": (\n        "{OLD_A}"\n    ),\n'
        '    "artifact_size_bytes": 100,\n'
        "}\n",
        encoding="utf-8",
    )
    pins, _ = refresh_pins.build_pins(_drift("out/k.hlsl", OLD_A, NEW_A, 100, 104))

    report = refresh_pins.refresh(pins, tmp_path, (SCOPE,), write=True)

    assert report.unresolved == []
    corrected = json.loads(gaps.read_text())["corrected"]
    assert corrected == {
        "artifact_sha256": NEW_A,
        "artifact_size_bytes": 104,
        "output_element_count": 100,
    }
    assert f'"{NEW_A}"' in harness.read_text()
    assert '"artifact_size_bytes": 104,' in harness.read_text()


def test_command_line_reads_drift_and_writes_a_summary(tmp_path, capsys):
    paths = _tree(tmp_path)
    drift = tmp_path / "drift"
    drift.mkdir()
    log = "".join(
        "2026-10-29T11:14:10Z MLX-PIN-DRIFT " + json.dumps(record) + "\n"
        for pin in _pins().values()
        for record in _drift(
            "s-" + pin.expected_sha256[:4],
            pin.expected_sha256,
            pin.actual_sha256,
            pin.expected_size,
            pin.actual_size,
        )
    )
    (drift / "job.log").write_text(log + log, encoding="utf-8")
    summary = tmp_path / "summary.json"
    saved = tmp_path / "records.jsonl"
    status = refresh_pins.main(
        [
            str(drift),
            "--root",
            str(tmp_path),
            "--summary",
            str(summary),
            "--save-records",
            str(saved),
            "--write",
        ]
    )
    assert status == 0, capsys.readouterr()
    payload = json.loads(summary.read_text())
    assert payload["records"] == 8
    assert payload["pins"] == 4
    assert payload["unresolved"] == []
    assert len(saved.read_text().splitlines()) == 8
    assert NEW_A in paths["contract"].read_text()
    printed = json.loads(capsys.readouterr().out)
    assert printed["written"] is True
    assert printed["changedFiles"] == 4


def test_the_tool_and_its_tests_are_never_rewritten(tmp_path):
    # The tool's own fixtures quote README phrases such as "7,129 bytes"; a
    # real README drift of that phrase must not rewrite them.
    assert refresh_pins.SELF_FILES == {
        Path(refresh_pins.__file__).resolve(),
        Path(__file__).resolve(),
    }
    scope = tmp_path / SCOPE
    (scope / "tests").mkdir(parents=True)
    (scope / "README.md").write_text("A 7,129-byte artifact.\n", encoding="utf-8")
    kept = scope / "tests" / "test_kept.py"
    fixture = scope / "tests" / "test_fixture.py"
    for path in (kept, fixture):
        path.write_text('assert "7,129-byte" in README\n', encoding="utf-8")
    texts = refresh_pins._scan(
        tmp_path, [Path(SCOPE)], excluded=frozenset({fixture.resolve()})
    )
    assert sorted(texts) == [
        f"{SCOPE}/README.md",
        f"{SCOPE}/tests/test_kept.py",
    ]


OLD_M, NEW_M = "5" * 64, "6" * 64
OLD_N, NEW_N = "7" * 64, "8" * 64
SHARED_M = "9" * 64


def _materialization_drift(subject, old_sha, new_sha, old_count, new_count):
    return [
        _record(subject, "materializationSha256", old_sha, new_sha),
        _record(subject, "specializationCount", old_count, new_count),
    ]


def test_materialization_digests_pair_with_specialization_counts():
    records = [
        *_drift("out/k.metal", OLD_A, NEW_A, 100, 120),
        *_materialization_drift("out/k.metal", OLD_M, NEW_M, 5, 6),
        _record("out/j.metal", "specializationCount", 2, 3),
    ]
    pins, count_only = refresh_pins.build_pins(records)
    assert pins == {
        OLD_A: refresh_pins.Pin(OLD_A, NEW_A, 100, 120),
        OLD_M: refresh_pins.Pin(OLD_M, NEW_M, 5, 6, size_keys=refresh_pins.COUNT_KEYS),
    }
    assert count_only == [
        {"subject": "out/j.metal", "test": "t.py::case", "size": (2, 3)}
    ]
    with pytest.raises(refresh_pins.RefreshError, match="two kinds of value"):
        refresh_pins.build_pins(
            [
                *_drift("out/a.metal", OLD_A, NEW_A),
                _record("out/b.metal", "materializationSha256", OLD_A, NEW_A),
            ]
        )


def _counted_contract(materializations):
    entries = [
        {
            "entryPoint": f"k_{index}",
            "shape": shape,
            "sha256": UNCHANGED if index else OLD_A,
            "sizeBytes": 100 + index,
            "specializationCount": count,
            "materializationSha256": digest,
        }
        for index, (shape, count, digest) in enumerate(materializations)
    ]
    return {
        "entries": entries,
        "artifactContract": {
            "specializationCount": sum(e[1] for e in materializations),
            "specializationCountsByShape": {
                "row": sum(e[1] for e in materializations if e[0] == "row"),
                "col": sum(e[1] for e in materializations if e[0] == "col"),
            },
            "unsupportedSpecializationCount": 0,
        },
    }


def _counted_tree(root: Path):
    base = root / SCOPE
    (base / "contracts").mkdir(parents=True)
    (base / "tests").mkdir()
    first = base / "contracts" / "family.metal.json"
    other = base / "contracts" / "family.opengl.json"
    first.write_text(
        json.dumps(
            _counted_contract([("row", 5, OLD_M), ("col", 3, SHARED_M)]), indent=2
        )
        + "\n",
        encoding="utf-8",
    )
    other.write_text(
        json.dumps(
            _counted_contract([("row", 5, OLD_N), ("col", 3, SHARED_M)]), indent=2
        )
        + "\n",
        encoding="utf-8",
    )
    test = base / "tests" / "test_family.py"
    test.write_text(
        f"""CONTRACTS = ("contracts/family.metal.json", "contracts/family.opengl.json")
FAMILY_METAL_CONTRACT_SHA256 = (
    "{_sha(first)}"
)
FAMILY_SPECIALIZATION_COUNT = 8
""",
        encoding="utf-8",
    )
    gaps = base / "gaps.json"
    gaps.write_text(
        '{\n  "path": "contracts/family.opengl.json",\n'
        '  "specialization_count": 8,\n  "entries": 2\n}\n',
        encoding="utf-8",
    )
    readme = base / "README.md"
    readme.write_text(
        "`contracts/family.metal.json` pins 8 exact materializations.\n",
        encoding="utf-8",
    )
    return {
        "first": first,
        "other": other,
        "test": test,
        "gaps": gaps,
        "readme": readme,
    }


def test_refresh_rewrites_materialization_pins_and_their_totals(tmp_path):
    paths = _counted_tree(tmp_path)
    pins, count_only = refresh_pins.build_pins(
        [
            *_materialization_drift("out/k_0.metal", OLD_M, NEW_M, 5, 6),
            *_materialization_drift("out/k_0.glsl", OLD_N, NEW_N, 5, 6),
            *_drift("out/k_0.metal", OLD_A, NEW_A),
        ]
    )
    assert count_only == []

    report = refresh_pins.refresh(pins, tmp_path, (SCOPE,), write=True)
    assert report.complete, report.to_json()

    for name, digest in (("first", NEW_M), ("other", NEW_N)):
        contract = json.loads(paths[name].read_text())
        assert paths[name].read_text() == json.dumps(contract, indent=2) + "\n"
        assert [
            (e["materializationSha256"], e["specializationCount"], e["sha256"])
            for e in contract["entries"]
        ] == [(digest, 6, NEW_A), (SHARED_M, 3, UNCHANGED)]
        assert contract["artifactContract"] == {
            "specializationCount": 9,
            "specializationCountsByShape": {"row": 6, "col": 3},
            "unsupportedSpecializationCount": 0,
        }
    test = paths["test"].read_text()
    assert "FAMILY_SPECIALIZATION_COUNT = 9\n" in test
    assert f'    "{_sha(paths["first"])}"\n' in test
    assert '"specialization_count": 9,\n  "entries": 2\n' in paths["gaps"].read_text()
    assert "pins 9 exact materializations" in paths["readme"].read_text()
    assert dict(report.applied) == {
        "sha256": 2,
        "materializationSha256": 2,
        "specializationCount": 2,
        "specializationTotals": 2,
        "specializationShapes": 2,
        "mirroredSpecializationTotals": 3,
        "fileSha256": 1,
    }


def test_conflicting_specialization_totals_are_reported_and_left_alone(tmp_path):
    paths = _counted_tree(tmp_path)
    pins, _ = refresh_pins.build_pins(
        [
            *_materialization_drift("out/k_0.metal", OLD_M, NEW_M, 5, 6),
            *_materialization_drift("out/k_0.glsl", OLD_N, NEW_N, 5, 7),
        ]
    )

    report = refresh_pins.refresh(pins, tmp_path, (SCOPE,), write=True)

    assert report.unresolved == [{"reason": "specialization total 8 moved to [9, 10]"}]
    assert (
        json.loads(paths["first"].read_text())["artifactContract"][
            "specializationCount"
        ]
        == 9
    )
    assert (
        json.loads(paths["other"].read_text())["artifactContract"][
            "specializationCount"
        ]
        == 10
    )
    assert "FAMILY_SPECIALIZATION_COUNT = 8\n" in paths["test"].read_text()
    assert '"specialization_count": 8,' in paths["gaps"].read_text()


def test_a_specialization_count_is_located_only_beside_its_digest(tmp_path):
    readme = tmp_path / SCOPE / "README.md"
    readme.parent.mkdir(parents=True)
    readme.write_text(f"A 5-specialization record `{OLD_M}`.\n", encoding="utf-8")
    pins, _ = refresh_pins.build_pins(
        _materialization_drift("out/k.metal", OLD_M, NEW_M, 5, 6)
    )

    report = refresh_pins.refresh(pins, tmp_path, (SCOPE,), write=True)

    assert report.unresolved == [
        {
            "path": "demos/integrations/mlx/README.md",
            "line": 1,
            "sha256": OLD_M,
            "reason": "found 0 specializationCount fields equal to 5",
        }
    ]
    assert readme.read_text() == f"A 5-specialization record `{NEW_M}`.\n"

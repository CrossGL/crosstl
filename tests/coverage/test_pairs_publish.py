"""Publishing the translation matrix: reconciling cells with the rows, the coverage branch files and the workflow."""

import json
from pathlib import Path

import pytest
import yaml

from tools.coverage import __main__ as cli
from tools.coverage import pairs, publish, report
from tools.coverage.model import Feature, Param, Signature, Type

REPO = Path(__file__).resolve().parents[2]
FMA = {"semantic": "GLSL.std.450:Fma"}
SQRT = {"semantic": "GLSL.std.450:Sqrt"}


def _row(language, direction, *features):
    listed = [
        {"name": name, "category": "function", "outcome": outcome, **extra}
        for name, outcome, extra in features
    ]
    return {
        "language": language,
        "direction": direction,
        "offered": True,
        "validator": "tool",
        "score": None,
        "counts": dict.fromkeys(("supported", "refused", "wrong", "untested"), 0),
        "features": listed,
    }


def _rows(sqrt_input="supported"):
    """Rows that put HLSL's mad and sqrt in the HLSL to GLSL cell."""
    return [
        _row("glsl", "output", ("fma", "supported", FMA), ("sqrt", "supported", SQRT)),
        _row("hlsl", "output", ("mad", "supported", FMA), ("sqrt", "supported", SQRT)),
        _row(
            "hlsl",
            "input",
            ("mad", "supported", {}),
            ("sqrt", sqrt_input, {"detail": "dxc rejects the probe"}),
        ),
    ]


def _cell(source, target, *features):
    return {
        "source": source,
        "target": target,
        "validator": "glslangValidator",
        "executes": True,
        "score": None,
        "counts": {},
        "features": [
            {"name": name, "category": "function", "outcome": outcome, **extra}
            for name, outcome, extra in features
        ],
    }


def _measured(sqrt_input="supported"):
    return {
        "commit": "c" * 40,
        "date": "2026-11-21T01:53:00Z",
        "sources": {},
        "validators": {},
        "backends": _rows(sqrt_input),
        "pairs": [
            _cell(
                "hlsl",
                "glsl",
                ("mad", "supported", {}),
                ("sqrt", "refused", {"detail": "E1: no sqrt"}),
                # Listed by the rows the cells were planned from, but no longer in the cell.
                ("gone", "wrong", {"detail": "E2"}),
            ),
            # A cell the rows no longer define: CrossTL has no CUDA input row here.
            _cell("cuda", "glsl", ("fma", "supported", {})),
        ],
    }


def test_reconcile_keeps_the_features_the_rows_put_in_each_cell_and_scores_again():
    data = pairs.reconcile(_measured())
    (cell,) = data["pairs"]
    assert (cell["source"], cell["target"]) == ("hlsl", "glsl")
    assert [f["name"] for f in cell["features"]] == ["mad", "sqrt"]
    assert cell["counts"] == {"supported": 1, "refused": 1, "wrong": 0, "untested": 0}
    assert cell["score"] == 50.0
    assert data["matrix"] == {"commit": "c" * 40, "date": "2026-11-21T01:53:00Z"}


def test_reconcile_leaves_untested_what_the_source_row_now_leaves_untested():
    (cell,) = pairs.reconcile(_measured(sqrt_input="untested"))["pairs"]
    sqrt = cell["features"][1]
    assert sqrt["outcome"] == "untested"
    assert sqrt["detail"] == "untested in the hlsl input row: dxc rejects the probe"
    assert cell["score"] == 100.0


def test_a_report_without_cells_keeps_the_previous_matrix_and_its_stamp():
    measured = pairs.reconcile(_measured())
    later = {
        "commit": "d" * 40,
        "date": "2026-11-21T09:00:00Z",
        "backends": _rows(sqrt_input="untested"),
    }
    carried = pairs.reconcile(later, measured)
    assert carried["matrix"] == measured["matrix"]
    assert carried["commit"] == "d" * 40
    (cell,) = carried["pairs"]
    assert [f["outcome"] for f in cell["features"]] == ["supported", "untested"]
    # Without cells of its own or a previous matrix, a report stays as it is.
    assert pairs.reconcile(later) == later
    assert pairs.reconcile(later, {"backends": []}) == later


def test_a_cell_with_a_type_the_target_has_none_of_is_outside_it():
    double = Type("double")
    fma = Feature("cuda", "function", "fma", Signature(double, (Param(double),)))
    fmaf = Feature("cuda", "function", "fmaf", Signature(Type("float"), ()))
    assert pairs.outside(fma, "metal") == "Metal has no double type"
    assert pairs.outside(fma, "wgsl") == "WGSL has no double type"
    assert pairs.outside(fma, "hlsl") == ""
    assert pairs.outside(fmaf, "metal") == ""
    imat = Type("int", (2, 2))
    transpose = Feature("glsl", "function", "t", Signature(imat, (Param(imat),)))
    assert pairs.outside(transpose, "metal") == "Metal has no int matrices"


def test_cells_are_published_in_files_of_their_own_and_load_back():
    data = pairs.reconcile(_measured())
    files = report.site(data)
    stored = json.loads(files["coverage.json"])
    (cell,) = stored["pairs"]
    assert cell["file"] == "features/pairs/hlsl-to-glsl.json"
    assert "features" not in cell
    assert stored["matrix"] == data["matrix"]
    assert json.loads(files["features/pairs/hlsl-to-glsl.json"])["source"] == "hlsl"


def test_loading_a_published_report_reads_its_cell_files(tmp_path):
    data = pairs.reconcile(_measured())
    for name, text in report.site(data).items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    assert report.load(tmp_path / "coverage.json")["pairs"] == data["pairs"]


@pytest.mark.parametrize(
    "name",
    [
        "../coverage.json",
        "features/pairs/../../x-to-y.json",
        "features/hlsl-input.json",
    ],
)
def test_a_cell_file_must_be_named_as_a_cell(tmp_path, name):
    (tmp_path / "coverage.json").write_text(
        json.dumps(
            {
                "backends": [],
                "pairs": [{"source": "hlsl", "target": "glsl", "file": name}],
            }
        )
    )
    with pytest.raises(ValueError):
        report.load(tmp_path / "coverage.json")


def test_a_cell_file_must_hold_its_own_cell(tmp_path):
    files = report.site(pairs.reconcile(_measured()))
    files["features/pairs/hlsl-to-glsl.json"] = files[
        "features/pairs/hlsl-to-glsl.json"
    ].replace('"hlsl"', '"metal"')
    for name, text in files.items():
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text(text)
    with pytest.raises(ValueError, match="holds the features of metal glsl"):
        report.load(tmp_path / "coverage.json")


def _published():
    rows = [
        {**row, "score": 75.0, "counts": {**row["counts"], "supported": 3, "wrong": 1}}
        for row in _rows()
    ]
    rows.append(
        {
            **_row("opencl", "input", ("fma", "supported", {})),
            "score": 50.0,
            "counts": {"supported": 1, "refused": 1, "wrong": 0, "untested": 0},
        }
    )
    data = pairs.reconcile({**_measured(), "backends": rows})
    empty = {**_cell("opencl", "glsl"), "score": None}
    return {**data, "pairs": [*data["pairs"], empty]}


def test_coverage_md_has_the_translation_matrix():
    text = publish.markdown(_published())
    section = text.split("## Translation matrix", 1)[1].split("## Trend", 1)[0]
    assert "measured on commit [`ccccccc`]" in section
    table = [line for line in section.splitlines() if line.startswith("|")]
    assert table[0] == "| Source \\ Target | GLSL | HLSL |"
    # The HLSL row: a measured cell links to its page, and HLSL to HLSL is the input row, in italics.
    assert table[2] == "| HLSL | [50.0 %](pairs/hlsl-to-glsl.md) | *75.0 %* |"
    # OpenCL C shares no feature with GLSL here, and its input row is its translation into HLSL.
    assert table[3] == "| OpenCL C | none | *50.0 %* |"


def test_each_cell_has_a_page_with_its_gaps_and_no_bug_link():
    data = _published()
    text = publish.cell_markdown(data["pairs"][0], data)
    assert text.startswith("# CrossTL translation matrix: HLSL to GLSL\n")
    assert "Score 50.0 %: 1 supported, 1 refused, 0 wrong, 0 untested." in text
    assert "`E1: no sqrt`: `sqrt`" in text
    assert "open bug issues" not in text
    assert "](../COVERAGE.md#translation-matrix)" in text
    clean = {
        **data["pairs"][0],
        "features": data["pairs"][0]["features"][:1],
        "counts": {"supported": 1, "refused": 0, "wrong": 0, "untested": 0},
        "score": 100.0,
    }
    assert publish.cell_markdown(clean, data).rstrip().endswith("No gaps.")


def test_a_page_name_holds_only_language_ids():
    assert publish.page({"source": "hlsl", "target": "glsl"}) == "pairs/hlsl-to-glsl.md"
    with pytest.raises(ValueError):
        publish.page({"source": "../x", "target": "glsl"})


def _write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    for name, text in report.site(data).items():
        (path.parent / name).parent.mkdir(parents=True, exist_ok=True)
        (path.parent / name).write_text(text)


def test_publish_keeps_the_previous_matrix_for_a_report_without_one(tmp_path):
    previous = _published()
    _write(tmp_path / "previous" / "coverage.json", previous)
    current = {
        **{k: v for k, v in previous.items() if k not in ("pairs", "matrix")},
        "commit": "e" * 40,
    }
    _write(tmp_path / "current" / "coverage.json", current)
    site = tmp_path / "site"
    arguments = ["publish", str(tmp_path / "current" / "coverage.json")]
    arguments += ["--output-dir", str(site)]
    arguments += ["--previous", str(tmp_path / "previous" / "coverage.json")]
    assert cli.main(arguments) == 0
    published = report.load(site / "coverage.json")
    assert published["commit"] == "e" * 40
    assert published["matrix"] == previous["matrix"]
    assert (site / "pairs" / "hlsl-to-glsl.md").is_file()
    assert (site / "features" / "pairs" / "hlsl-to-glsl.json").is_file()
    # A missing previous report is a first publication: the report is published without a matrix.
    assert cli.main([*arguments[:4], "--previous", str(tmp_path / "none.json")]) == 0
    assert "pairs" not in report.load(site / "coverage.json")


def test_publish_puts_a_measured_matrix_in_place_of_the_one_newer_rows_carry(
    tmp_path,
):
    # A push published rows for commit e, newer than the nightly run's commit c, with a matrix from commit a.
    newer = {**_published(), "commit": "e" * 40, "date": "2026-11-21T03:00:00Z"}
    newer["matrix"] = {"commit": "a" * 40, "date": "2026-11-20T01:00:00Z"}
    _write(tmp_path / "previous" / "coverage.json", newer)
    (tmp_path / "nightly.json").write_text(report.dumps(_measured()))
    site = tmp_path / "site"
    arguments = ["publish", str(tmp_path / "previous" / "coverage.json")]
    arguments += ["--matrix", str(tmp_path / "nightly.json"), "--output-dir", str(site)]
    assert cli.main(arguments) == 0
    published = report.load(site / "coverage.json")
    # The rows and their commit are the newer ones; the matrix is the nightly run's, scored against those rows.
    assert (published["commit"], published["date"]) == (newer["commit"], newer["date"])
    assert (
        published["backends"]
        == report.load(tmp_path / "previous" / "coverage.json")["backends"]
    )
    assert published["matrix"] == {"commit": "c" * 40, "date": "2026-11-21T01:53:00Z"}
    rows = {k: v for k, v in newer.items() if k not in ("pairs", "matrix")}
    assert published["pairs"] == pairs.reconcile(rows, _measured())["pairs"]
    # The newer publication carried an opencl-to-glsl cell the nightly run did not measure; it is gone.
    assert [(c["source"], c["target"]) for c in published["pairs"]] == [
        ("hlsl", "glsl")
    ]
    assert (site / "pairs" / "hlsl-to-glsl.md").is_file()


def test_publish_refuses_a_matrix_report_without_a_matrix(tmp_path, capsys):
    rows = {k: v for k, v in _measured().items() if k != "pairs"}
    (tmp_path / "rows.json").write_text(report.dumps(rows))
    arguments = ["publish", str(tmp_path / "rows.json"), "--output-dir", str(tmp_path)]
    assert cli.main([*arguments, "--matrix", str(tmp_path / "rows.json")]) == 2
    assert "has no translation matrix" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        cli.main([*arguments, "--matrix", "a.json", "--previous", "b.json"])


def test_merge_reconciles_the_cells_with_the_rows(tmp_path):
    rows = {k: v for k, v in _measured().items() if k != "pairs"}
    cells = {**_measured(), "backends": []}
    for name, data in (("rows", rows), ("cells", cells)):
        (tmp_path / f"{name}.json").write_text(report.dumps(data))
    output = tmp_path / "coverage.json"
    arguments = ["merge", str(tmp_path / "rows.json"), str(tmp_path / "cells.json")]
    assert cli.main([*arguments, "--output", str(output)]) == 0
    merged = report.load(output)
    assert [(c["source"], c["target"]) for c in merged["pairs"]] == [("hlsl", "glsl")]
    assert merged["matrix"]["commit"] == "c" * 40


def _workflow():
    return yaml.safe_load(
        (REPO / ".github/workflows/coverage.yml").read_text(encoding="utf-8")
    )


LEGS = {"linux": "linux", "compute": "$BACKEND", "rust": "rust", "metal": "metal"}


@pytest.mark.parametrize("leg", sorted(LEGS))
def test_every_leg_measures_the_matrix_only_in_the_nightly_and_manual_runs(leg):
    steps = _workflow()["jobs"][leg]["steps"]
    (measure,) = (s for s in steps if s.get("name") == "Measure the translation matrix")
    assert measure["if"] == (
        "github.event_name == 'schedule' || github.event_name == 'workflow_dispatch'"
    )
    assert 'run --pairs "$published/coverage.json" --strict' in measure["run"]
    assert "git fetch --quiet --depth=1 origin coverage" in measure["run"]
    name = LEGS[leg] if leg != "compute" else "${{ matrix.backend }}"
    (upload,) = (s for s in steps if "upload-artifact" in str(s.get("uses", "")))
    assert f"coverage-pairs-{name}.json" in upload["with"]["path"].splitlines()
    rows = next(s for s in steps if s.get("name") == "Measure coverage")
    assert 0 < rows["timeout-minutes"] < _workflow()["jobs"][leg]["timeout-minutes"]


@pytest.mark.parametrize("leg", ["linux", "compute"])
def test_cell_verdicts_are_kept_apart_from_the_row_verdicts(leg):
    steps = _workflow()["jobs"][leg]["steps"]
    (measure,) = (s for s in steps if s.get("name") == "Measure the translation matrix")
    assert measure["env"]["CROSSTL_COVERAGE_VERDICTS"].endswith("/pairs-verdicts")
    caches = [s for s in steps if "Restore verdicts" == s.get("name")]
    caches += [s for s in steps if "Keep verdicts" == s.get("name")]
    assert len(caches) == 2
    for cache in caches:
        assert "/pairs-verdicts" in cache["with"]["path"]
    # A nightly run of a commit a push already measured saves its own cell verdicts.
    assert caches[0]["with"]["key"].endswith("-${{ github.run_id }}")


def test_publishing_keeps_the_previous_matrix_and_adds_the_cell_pages():
    steps = _workflow()["jobs"]["publish"]["steps"]
    run = "\n".join(str(s.get("run", "")) for s in steps)
    assert '--previous "$RUNNER_TEMP/previous/coverage.json"' in run
    assert 'git archive FETCH_HEAD | tar -x -C "$RUNNER_TEMP/previous"' in run
    assert "if [ -d pairs ]; then git add pairs; fi" in run


def test_a_late_nightly_run_publishes_its_matrix_with_the_newer_rows():
    jobs = _workflow()["jobs"]
    steps = {s.get("name"): s for s in jobs["publish"]["steps"]}
    read = steps["Read the published coverage"]["run"]
    # Behind a newer publication, a run that measured the matrix publishes it if the carried matrix is older.
    assert "publish=matrix" in read and "publish=false" in read
    assert '"pairs" in json.load(open("coverage.json"))' in read
    assert "compare/$carried...$GITHUB_SHA" in read
    build = steps["Build the published files"]
    assert build["if"] == "steps.published.outputs.publish != 'false'"
    assert (
        'publish "$RUNNER_TEMP/previous/coverage.json" --matrix coverage.json' in build[
            "run"
        ]
    )
    assert steps["Replace the coverage branch"]["if"] == build["if"]
    kept = steps["Keep the published report"]
    assert kept["if"] == build["if"]
    assert kept["with"]["name"] == "coverage-published"
    assert kept["with"]["retention-days"] <= 14
    # The issues reconcile what was published, so pair issues follow the matrix the coverage branch carries.
    assert "needs.publish.outputs.published != 'false'" in jobs["issues"]["if"]
    issues = jobs["issues"]["steps"]
    (download,) = (s for s in issues if "download-artifact" in str(s.get("uses", "")))
    assert download["with"] == {"name": "coverage-published", "path": "published"}
    assert any(
        "tools.coverage issues published/coverage.json" in str(s.get("run", ""))
        for s in issues
    )

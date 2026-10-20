import json
import re
import subprocess
from fnmatch import fnmatch
from pathlib import Path

import pytest
import yaml

from tools import labels
from tools.labels import Change, GitHub, Label, load, plan

ROOT = Path(__file__).resolve().parents[1]
TAXONOMY = {
    "bug",
    "enhancement",
    "documentation",
    "question",
    "good first issue",
    "help wanted",
    *(
        f"backend:{name}"
        for name in (
            "glsl",
            "hlsl",
            "metal",
            "spirv",
            "wgsl",
            "cuda",
            "hip",
            "slang",
            "rust",
            "mojo",
            "crossgl",
        )
    ),
    "direction:input",
    "direction:output",
    *(f"area:{name}" for name in ("frontend", "codegen", "runtime", "ci", "docs")),
    "coverage",
    "stale",
}


def _write(tmp_path, entries):
    path = tmp_path / "labels.yml"
    path.write_text(yaml.safe_dump(entries), encoding="utf-8")
    return path


def _many(count=labels.MINIMUM):
    return [{"name": f"label-{index}", "color": "ededed"} for index in range(count)]


def test_the_label_file_defines_the_taxonomy_and_little_else():
    names = {label.name for label in load()}
    assert TAXONOMY <= names
    assert len(names) <= 30


def test_labels_that_workflows_and_dependabot_apply_are_defined():
    names = {label.name for label in load()}
    used = set()
    for path in (ROOT / ".github" / "workflows").glob("*.yml"):
        text = path.read_text(encoding="utf-8")
        used |= set(re.findall(r"labels\.\*\.name, '([^']+)'", text))
        used |= set(
            re.findall(r"^\s*stale-(?:pr|issue)-label:\s*(\S+)", text, re.MULTILINE)
        )
    for update in yaml.safe_load(
        (ROOT / ".github" / "dependabot.yml").read_text(encoding="utf-8")
    )["updates"]:
        used |= set(update["labels"])
    assert {"ci:full", "stale", "dependencies"} <= used
    assert used <= names


def test_every_dependency_manifest_has_a_dependabot_entry():
    # Dependabot labels a pull request for an unlisted manifest with its own defaults,
    # such as ".NET", which the label sync then deletes again.
    ecosystems = {"pyproject.toml": "pip", ".csproj": "nuget", "Cargo.toml": "cargo"}
    ecosystems |= {"package.json": "npm", "go.mod": "gomod"}
    entries = yaml.safe_load(
        (ROOT / ".github" / "dependabot.yml").read_text(encoding="utf-8")
    )["updates"]
    tracked = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.splitlines()
    for path in map(Path, tracked):
        ecosystem = ecosystems.get(path.name) or ecosystems.get(path.suffix)
        if ecosystem is None:
            continue
        directory = "/" + path.parent.as_posix() if path.parent.parts else "/"
        patterns = [
            pattern.rstrip("/") or "/"
            for entry in entries
            if entry["package-ecosystem"] == ecosystem
            for pattern in entry.get("directories", [entry.get("directory")])
        ]
        assert any(fnmatch(directory, pattern) for pattern in patterns), path


def test_load_rejects_a_bad_color_a_duplicate_name_and_a_short_file(tmp_path):
    with pytest.raises(ValueError, match="six hex digits"):
        load(_write(tmp_path, [*_many(), {"name": "x", "color": "#fff"}]))
    with pytest.raises(ValueError, match="appears twice"):
        load(
            _write(
                tmp_path,
                [
                    *_many(),
                    {"name": "X", "color": "ffffff", "renamed-from": ["label-1"]},
                ],
            )
        )
    with pytest.raises(ValueError, match="at least"):
        load(_write(tmp_path, _many(3)))


def test_plan_creates_updates_and_deletes_in_a_stable_order():
    wanted = [Label("bug", "d73a4a", "Wrong output"), Label("coverage", "0e8a16")]
    existing = [
        Label("zeta", "000000"),
        Label("Bug", "d73a4a", "Wrong output"),
        Label("alpha", "000000"),
    ]
    assert [change.describe() for change in plan(wanted, existing)] == [
        "rename 'Bug' to 'bug'",
        "create 'coverage'",
        "delete 'alpha'",
        "delete 'zeta'",
    ]


def test_plan_renames_an_old_label_so_its_issues_keep_it():
    wanted = [Label("backend:metal", "c5def5", "Metal", ("Metal backend",))]
    changes = plan(wanted, [Label("Metal backend", "f9d0c4")])
    assert changes == [Change("update", "Metal backend", wanted[0])]


def test_plan_merges_an_old_label_into_one_that_already_exists():
    wanted = [Label("area:frontend", "1d76db", "Parsers", ("Translator Frontend",))]
    existing = [
        Label("area:frontend", "1d76db", "Parsers"),
        Label("Translator Frontend", "ededed"),
    ]
    assert [change.describe() for change in plan(wanted, existing)] == [
        "merge 'Translator Frontend' into 'area:frontend'"
    ]


def test_plan_is_empty_once_github_matches_the_file():
    wanted = load()
    assert (
        plan(
            wanted,
            [Label(label.name, label.color, label.description) for label in wanted],
        )
        == []
    )


class _Recorder:
    def __init__(self, responses):
        self.calls = []
        self._responses = responses

    def __call__(self, command, input=None, **_):
        path = command[4].split("repos/CrossGL/crosstl/", 1)[1]
        self.calls.append(
            (command[3], path, None if input is None else json.loads(input))
        )
        body = self._responses.get((command[3], path), "")
        return subprocess.CompletedProcess(
            command, 0, json.dumps(body) if body != "" else "", ""
        )


def test_merge_adds_the_new_label_to_every_issue_before_deleting_the_old_one():
    issues = [{"number": 7}, {"number": 9}]
    recorder = _Recorder(
        {
            (
                "GET",
                "issues?state=all&labels=Translator%20Frontend&per_page=100&page=1",
            ): issues
        }
    )
    target = Label("area:frontend", "1d76db", "Parsers")
    GitHub("CrossGL/crosstl", recorder).apply(
        Change("merge", "Translator Frontend", target)
    )
    assert [call[:2] for call in recorder.calls[1:]] == [
        ("POST", "issues/7/labels"),
        ("POST", "issues/9/labels"),
        ("DELETE", "labels/Translator%20Frontend"),
    ]
    assert recorder.calls[1][2] == {"labels": ["area:frontend"]}


def test_update_renames_through_the_old_name():
    recorder = _Recorder({})
    label = Label("backend:glsl", "c5def5", "GLSL")
    GitHub("CrossGL/crosstl", recorder).apply(Change("update", "opengl backend", label))
    assert recorder.calls == [
        (
            "PATCH",
            "labels/opengl%20backend",
            {"new_name": "backend:glsl", "color": "c5def5", "description": "GLSL"},
        )
    ]


def test_the_command_line_only_prints_without_apply(monkeypatch, capsys):
    applied = []
    monkeypatch.setattr(GitHub, "labels", lambda self: [Label("old", "000000")])
    monkeypatch.setattr(GitHub, "apply", lambda self, change: applied.append(change))
    assert labels.main([]) == 0
    output = capsys.readouterr().out
    assert "would delete 'old'" in output
    assert applied == []

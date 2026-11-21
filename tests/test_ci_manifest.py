import json

import pytest

from tools import ci_manifest
from tools.ci_manifest import Manifest, compare, parse

A, B, C = ("a" * 64), ("b" * 64), ("c" * 64)


def test_compare_sorts_keys_by_how_their_hash_moved():
    baseline = Manifest({"leg-2": A, "leg-1": A, "gone": B})
    current = Manifest({"leg-2": B, "leg-1": A, "new": C})
    result = compare(baseline, current)
    assert result.added == ("new",)
    assert result.changed == ("leg-2",)
    assert result.unchanged == ("leg-1",)
    assert result.removed == ("gone",)
    assert result.selected == ("leg-2", "new")
    assert result.reason is None


def test_a_missing_baseline_selects_every_current_key():
    result = compare(None, Manifest({"b": A, "a": B}))
    assert result.selected == ("a", "b")
    assert result.added == ("a", "b")
    assert result.reason == "no baseline manifest"


@pytest.mark.parametrize(
    ("before", "after", "reason"),
    [
        ({"schema": 1}, {"schema": 2}, "the manifest schema differs"),
        (
            {"toolchain": {"dxc": "1.8"}},
            {"toolchain": {"dxc": "1.9"}},
            "the toolchain differs",
        ),
        ({}, {"toolchain": "macos-15"}, "the toolchain differs"),
    ],
)
def test_a_different_schema_or_toolchain_selects_every_key_even_with_equal_hashes(
    before, after, reason
):
    baseline = Manifest({"same": A, "old": B}, **before)
    current = Manifest({"same": A, "new": C}, **after)
    result = compare(baseline, current)
    assert result.selected == ("new", "same")
    assert result.changed == ("same",)
    assert result.added == ("new",)
    assert result.removed == ("old",)
    assert result.unchanged == ()
    assert result.reason == reason


def test_equal_structured_manifests_select_nothing():
    data = {"schema": 1, "toolchain": {"xcode": "16.4"}, "entries": {"x": A}}
    result = compare(parse(data), parse(json.loads(json.dumps(data))))
    assert result.selected == ()
    assert result.unchanged == ("x",)


def test_parse_reads_both_manifest_forms():
    assert parse({"x": A}) == Manifest({"x": A})
    assert parse({"schema": 1, "entries": {"x": A}}) == Manifest({"x": A}, schema=1)


@pytest.mark.parametrize(
    ("data", "message"),
    [
        ([], "a manifest is a JSON object"),
        ({"entries": []}, "'entries' is a JSON object"),
        ({"entries": {}, "toolchian": "x"}, "unknown manifest fields: toolchian"),
        ({"x": A.upper()}, "is not a lowercase hexadecimal SHA-256"),
        ({"x": A[:63]}, "is not a lowercase hexadecimal SHA-256"),
        ({"x": 1}, "is not a lowercase hexadecimal SHA-256"),
        ({"": A}, "is empty"),
        ({" x": A}, "surrounding space"),
        ({"x\ny": A}, "control character"),
    ],
)
def test_parse_rejects_malformed_manifests(data, message):
    with pytest.raises(ValueError, match=message):
        parse(data)


def _write(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")
    return str(path)


def test_cli_prints_the_comparison_as_json(tmp_path, capsys):
    baseline = _write(tmp_path / "old.json", {"entries": {"a": A, "b": A}})
    current = _write(tmp_path / "new.json", {"entries": {"a": A, "b": B}})
    assert ci_manifest.main([baseline, current]) == 0
    assert json.loads(capsys.readouterr().out) == {
        "selected": ["b"],
        "added": [],
        "changed": ["b"],
        "removed": [],
        "unchanged": 1,
        "reason": None,
    }


def test_cli_lines_lists_one_selected_key_per_line(tmp_path, capsys):
    baseline = _write(tmp_path / "old.json", {"a": A, "b": A, "c": A})
    current = _write(tmp_path / "new.json", {"a": B, "b": A, "c": B})
    assert ci_manifest.main([baseline, current, "--format", "lines"]) == 0
    assert capsys.readouterr().out == "a\nc\n"


def test_cli_fails_safe_on_a_missing_or_unreadable_baseline(tmp_path, capsys):
    current = _write(tmp_path / "new.json", {"a": A})
    assert ci_manifest.main([str(tmp_path / "absent.json"), current]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["selected"] == ["a"] and out["reason"] == "no baseline manifest"

    (tmp_path / "broken.json").write_text("{", encoding="utf-8")
    assert (
        ci_manifest.main([str(tmp_path / "broken.json"), current, "--format", "lines"])
        == 0
    )
    captured = capsys.readouterr()
    assert captured.out == "a\n"
    assert "ignoring the baseline" in captured.err


def test_cli_rejects_a_malformed_current_manifest(tmp_path, capsys):
    baseline = _write(tmp_path / "old.json", {"a": A})
    current = _write(tmp_path / "new.json", {"a": "not-a-digest"})
    assert ci_manifest.main([baseline, current]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert (
        "new.json: a: 'not-a-digest' is not a lowercase hexadecimal SHA-256"
        in captured.err
    )

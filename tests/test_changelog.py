from pathlib import Path

import pytest

from tools import changelog
from tools.check_release import release_notes

ROOT = Path(__file__).resolve().parents[1]

CHANGELOG = """# Changelog

Notable changes.

## [Unreleased]

### Fixed

- An older fix.

---

## [1.2.3] - 2026-10-07

### Fixed

- A released fix.
"""


def _tree(tmp_path, text=CHANGELOG, **fragments):
    (tmp_path / "CHANGELOG.md").write_text(text, encoding="utf-8")
    directory = tmp_path / "changelog.d"
    directory.mkdir()
    for name, body in fragments.items():
        (directory / name.replace("__", ".")).write_text(body, encoding="utf-8")
    return tmp_path


def test_repository_changelog_entries_are_valid():
    changelog.entries(ROOT)
    assert changelog.main(["check"]) == 0


def test_entries_join_wrapped_lines_and_split_bullets(tmp_path):
    root = _tree(
        tmp_path,
        a_hip__fixed__md="HIP input reports a `SyntaxError`\nfor an unterminated template header.\n",
        b_two__added__md="- WGSL output writes `pack2x16float`.\n- Metal output\n  writes `simd_sum`.\n",
    )
    assert changelog.entries(root) == {
        "breaking": [],
        "added": [
            "WGSL output writes `pack2x16float`.",
            "Metal output writes `simd_sum`.",
        ],
        "improved": [],
        "fixed": [
            "HIP input reports a `SyntaxError` for an unterminated template header."
        ],
        "removed": [],
    }


def test_missing_directory_has_no_entries(tmp_path):
    assert not any(changelog.entries(tmp_path).values())
    assert changelog.fold(tmp_path) == 0


@pytest.mark.parametrize(
    ("name", "body", "message"),
    [
        ("hip.fix.md", "A fix.", "<kind> is one of"),
        ("HIP.fixed.md", "A fix.", "lower case"),
        ("hip.fixed.txt", "A fix.", "<name>.<kind>.md"),
        ("hip.fixed.md", "\n\n", "has no entry"),
        ("hip.fixed.md", "## Fixed\n\nA fix.", "without headings"),
    ],
)
def test_entries_reject_files_they_cannot_place(tmp_path, name, body, message):
    directory = tmp_path / "changelog.d"
    directory.mkdir()
    (directory / name).write_text(body, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        changelog.entries(tmp_path)


def test_fold_puts_new_entries_first_and_headings_in_release_order(tmp_path):
    root = _tree(
        tmp_path,
        b__fixed__md="Second new fix.",
        a__fixed__md="First new fix.",
        x__breaking__md="A breaking change.",
        y__improved__md="An improvement.",
    )
    assert changelog.fold(root) == 4
    assert not (root / "changelog.d").exists()
    text = (root / "CHANGELOG.md").read_text(encoding="utf-8")
    assert text.split("---")[0].endswith(
        "## [Unreleased]\n\n"
        "### Breaking Changes\n\n- A breaking change.\n\n"
        "### Improved\n\n- An improvement.\n\n"
        "### Fixed\n\n- First new fix.\n- Second new fix.\n- An older fix.\n\n"
    )
    assert text.endswith("## [1.2.3] - 2026-10-07\n\n### Fixed\n\n- A released fix.\n")
    assert changelog.fold(root) == 0


def test_fold_creates_the_unreleased_section_after_a_release(tmp_path):
    released = CHANGELOG.replace(
        "## [Unreleased]\n\n### Fixed\n\n- An older fix.\n\n---\n\n", ""
    )
    root = _tree(tmp_path, released, a__added__md="A new feature.")
    changelog.fold(root)
    text = (root / "CHANGELOG.md").read_text(encoding="utf-8")
    assert text == released.replace(
        "## [1.2.3]",
        "## [Unreleased]\n\n### Added\n\n- A new feature.\n\n---\n\n## [1.2.3]",
    )
    # The folded section becomes the release notes once it is renamed.
    renamed = text.replace("## [Unreleased]", "## [1.2.4] - 2026-10-08")
    (root / "CHANGELOG.md").write_text(renamed, encoding="utf-8")
    assert release_notes(root, "1.2.4") == "### Added\n\n- A new feature.\n"


def test_fold_leaves_files_in_place_when_an_entry_is_invalid(tmp_path):
    root = _tree(tmp_path, a__fixed__md="A fix.", b__unknown__md="Something.")
    with pytest.raises(ValueError):
        changelog.fold(root)
    assert (root / "CHANGELOG.md").read_text(encoding="utf-8") == CHANGELOG
    assert len(list((root / "changelog.d").iterdir())) == 2


def test_check_folded_fails_while_an_entry_is_left(tmp_path, capsys):
    root = _tree(tmp_path, a__fixed__md="A fix.")
    assert changelog.main(["check"], root) == 0
    assert changelog.main(["check", "--folded"], root) == 1
    assert "run python tools/changelog.py fold" in capsys.readouterr().err
    assert changelog.main(["fold"], root) == 0
    assert changelog.main(["check", "--folded"], root) == 0
    (root / "changelog.d").mkdir()
    (root / "changelog.d" / "a.fixd.md").write_text("A fix.", encoding="utf-8")
    assert changelog.main(["check"], root) == 1
    assert "a.fixd.md" in capsys.readouterr().err

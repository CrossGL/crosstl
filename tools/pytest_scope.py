"""Deselect the tests a change cannot affect.

``python -m pytest -p tools.pytest_scope --changed-files changed.txt tests`` reads
the paths a pull request or merge group changes, one per line, relative to the
rootdir (the repository root, where ``pyproject.toml`` configures pytest). A test that
carries one of the markers in :data:`SCOPES` reads only the files under that
marker's inputs, so when no changed path lies under them the test would pass or
fail exactly as it did on the commit the change builds on, and it is deselected.

The scoped tests still run on every change that reaches their inputs, and the
Complete Test Suite runs every test, with no list of changed files, on pushes to
``main`` the merge queue did not test, nightly and on dispatch.

The plugin fails towards running more tests:

- a change to any path in :data:`EVERYTHING`, which can change any test's
  outcome or selection, runs every test;
- an empty list runs every test, since an empty diff more likely means the list
  went wrong than that the change touches no file;
- a test that carries a marker but whose file lies outside the marker's inputs
  runs, because its file already reads code outside them.
"""

from __future__ import annotations

from pathlib import Path

import pytest

# Marker -> the paths (a directory ends with "/") that decide a marked test's outcome.
SCOPES = {
    # Pure-Python checks of the MLX port's models, pinned contracts and data. They
    # read nothing outside the demo; demos/integrations/mlx/tests/conftest.py
    # registers the marker.
    "mlx_model": ("demos/integrations/mlx/",),
}

# Paths whose change can change the outcome or the selection of any test.
EVERYTHING = (
    ".github/workflows/full-tests.yml",
    "conftest.py",
    "pyproject.toml",
    "tools/pytest_scope.py",
)


def _under(path: str, inputs: tuple[str, ...]) -> bool:
    return any(
        path.startswith(entry) if entry.endswith("/") else path == entry
        for entry in inputs
    )


def read_changed(path: Path) -> list[str]:
    """The changed paths listed in ``path``, one per line, without blank lines."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise pytest.UsageError(f"--changed-files: cannot read {path}: {error}")
    return [line.strip() for line in text.splitlines() if line.strip()]


def unaffected(changed: list[str]) -> list[str]:
    """The markers of :data:`SCOPES` whose tests ``changed`` cannot affect."""
    if not changed or any(_under(path, EVERYTHING) for path in changed):
        return []
    return [
        marker
        for marker, inputs in sorted(SCOPES.items())
        if not any(_under(path, inputs) for path in changed)
    ]


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--changed-files",
        type=Path,
        default=None,
        metavar="FILE",
        help="deselect scoped tests that no path listed in FILE can affect",
    )


def pytest_configure(config: pytest.Config) -> None:
    listed = config.getoption("changed_files")
    config._crosstl_unaffected = unaffected(read_changed(listed)) if listed else []


def _file(item: pytest.Item, root: Path) -> str:
    try:
        return Path(item.path).resolve().relative_to(root).as_posix()
    except ValueError:
        return ""


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    markers = config._crosstl_unaffected
    if not markers:
        return
    root = Path(config.rootpath).resolve()
    keep, drop = [], []
    for item in items:
        skipped = any(
            item.get_closest_marker(marker) is not None
            and _under(_file(item, root), SCOPES[marker])
            for marker in markers
        )
        (drop if skipped else keep).append(item)
    if drop:
        config.hook.pytest_deselected(items=drop)
        items[:] = keep


def pytest_report_header(config: pytest.Config) -> str | None:
    markers = config._crosstl_unaffected
    if not markers:
        return None
    return "pytest_scope: deselecting tests marked " + ", ".join(
        f"{marker} (no changed path under {' '.join(SCOPES[marker])})"
        for marker in markers
    )

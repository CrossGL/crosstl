"""Fill pytest's assertion-rewrite cache in parallel before a pytest-xdist run.

pytest rewrites the asserts of each test module it imports and caches the result
in ``__pycache__``. On a fresh checkout every xdist worker finds that cache empty
and rewrites every module of the session itself, so four workers repeat the same
work four times before the first test runs.

``python -m tools.pytest_warm tests demos/integrations`` collects those paths in
one process per CPU. Each process imports a disjoint slice of the test modules,
balanced by file size, so each module is rewritten once and the xdist run that
follows reads the cache. ``--ignore PATH`` leaves out a subtree, as it does for
pytest. Collection errors are left for that run to report.
"""

import argparse
import os
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path

import pytest

from tools.pytest_shard import parse_shard

SLICE = pytest.StashKey[frozenset[Path]]()


def is_test_module(path: Path) -> bool:
    """Match pytest's default ``python_files``: test_*.py and *_test.py."""
    name = path.name
    return name.endswith(".py") and (
        name.startswith("test_") or name.endswith("_test.py")
    )


def find_test_modules(paths: Iterable[str], ignore: Iterable[str] = ()) -> list[Path]:
    """Return the test modules under ``paths``, leaving out those under ``ignore``."""
    skipped = [Path(path).resolve() for path in ignore]
    found: set[Path] = set()
    for path in map(Path, paths):
        candidates = [path] if path.is_file() else path.rglob("*.py")
        found.update(p.resolve() for p in candidates if is_test_module(p))
    return sorted(
        module
        for module in found
        if not any(module == skip or skip in module.parents for skip in skipped)
    )


def split_by_size(modules: Iterable[Path], count: int) -> list[frozenset[Path]]:
    """Deal the modules, largest first, to whichever slice is smallest so far."""
    sizes = {module: module.stat().st_size for module in modules}
    totals = [0] * count
    slices: list[set[Path]] = [set() for _ in range(count)]
    for module in sorted(sizes, key=lambda m: (-sizes[m], str(m))):
        smallest = totals.index(min(totals))
        totals[smallest] += sizes[module]
        slices[smallest].add(module)
    return [frozenset(s) for s in slices]


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--warm-slice",
        type=parse_shard,
        default=None,
        metavar="K/N",
        help="collect only the test modules in slice K of N, balanced by file size",
    )


def pytest_configure(config: pytest.Config) -> None:
    selected = config.getoption("warm_slice")
    if selected is not None:
        index, count = selected
        modules = find_test_modules(config.args, config.getoption("ignore") or ())
        config.stash[SLICE] = split_by_size(modules, count)[index - 1]


def pytest_ignore_collect(collection_path: Path, config: pytest.Config) -> bool | None:
    keep = config.stash.get(SLICE, None)
    if keep is None or not is_test_module(collection_path):
        return None
    return True if collection_path.resolve() not in keep else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="+", help="test files or directories")
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=os.cpu_count() or 1,
        help="parallel processes (default: CPU count)",
    )
    parser.add_argument(
        "--ignore",
        action="append",
        default=[],
        metavar="PATH",
        help="leave out the tests under PATH, as pytest's --ignore does",
    )
    args = parser.parse_args(argv)
    jobs = max(1, args.jobs)
    command = [sys.executable, "-m", "pytest", "-p", "tools.pytest_warm"]
    command += ["-p", "no:cacheprovider", "--collect-only", "-q", *args.paths]
    command += [f"--ignore={path}" for path in args.ignore]
    workers = [
        subprocess.Popen(
            [*command, "--warm-slice", f"{index}/{jobs}"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        for index in range(1, jobs + 1)
    ]
    for worker in workers:
        worker.wait()
    return 0


if __name__ == "__main__":
    sys.exit(main())

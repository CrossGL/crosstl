"""Run one shard of a test session.

``python -m pytest -p tools.pytest_shard --shard 2/3 tests`` runs the test files
that fall into shard 2 of 3. A file's shard is a CRC-32 of its path relative to
the rootdir, so a module's tests and fixtures stay on one runner, every run picks
the same files, and the shards together run each file exactly once.
"""

import argparse
import zlib

import pytest


def parse_shard(value: str) -> tuple[int, int]:
    index, _, total = value.partition("/")
    try:
        shard, count = int(index), int(total)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected K/N, got {value!r}") from None
    if not 1 <= shard <= count:
        raise argparse.ArgumentTypeError(f"shard {shard} is outside 1..{count}")
    return shard, count


def shard_of(path: str, count: int) -> int:
    return zlib.crc32(path.encode("utf-8")) % count + 1


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--shard",
        type=parse_shard,
        default=None,
        metavar="K/N",
        help="run only the test files in shard K of N",
    )


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    selected = config.getoption("shard")
    if selected is None:
        return
    shard, count = selected
    keep, drop = [], []
    for item in items:
        path = item.nodeid.split("::", 1)[0]
        (keep if shard_of(path, count) == shard else drop).append(item)
    if drop:
        config.hook.pytest_deselected(items=drop)
        items[:] = keep

"""Run one shard of a test session.

``python -m pytest -p tools.pytest_shard --shard 2/3 tests`` runs the tests that
fall into shard 2 of 3. A test's shard is a CRC-32 of its node id, so every run
picks the same tests and the shards together run each test exactly once.

Hashing node ids rather than files keeps the shards even: a few files hold most
of the slow parametrized cases, and splitting by file left one shard with twice
the work of another. pytest-xdist already spreads a file's tests over workers,
so no test may rely on running in the same process as the rest of its file.
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


def shard_of(nodeid: str, count: int) -> int:
    return zlib.crc32(nodeid.encode("utf-8")) % count + 1


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--shard",
        type=parse_shard,
        default=None,
        metavar="K/N",
        help="run only the tests in shard K of N",
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
        (keep if shard_of(item.nodeid, count) == shard else drop).append(item)
    if drop:
        config.hook.pytest_deselected(items=drop)
        items[:] = keep

"""Run one shard of a test session.

``python -m pytest -p tools.pytest_shard --shard 2/3 tests`` runs the tests that
fall into shard 2 of 3. A test's shard is a CRC-32 of its node id, so every run
picks the same tests and the shards together run each test exactly once.

Hashing node ids rather than files keeps the shards even: a few files hold most
of the slow parametrized cases, and splitting by file left one shard with twice
the work of another. pytest-xdist already spreads a file's tests over workers,
so no test may rely on running in the same process as the rest of its file.

A test that uses a class- or module-scoped fixture is hashed by its file
instead, so that the file's shared setup, such as building a native helper,
runs on one runner rather than on every shard.
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


FILE_SCOPES = frozenset({"class", "module"})


def shard_of(nodeid: str, count: int) -> int:
    return zlib.crc32(nodeid.encode("utf-8")) % count + 1


def shard_key(item: pytest.Item) -> str:
    """Return the node id, or the file part of it if the test shares setup."""
    info = getattr(item, "_fixtureinfo", None)
    if info is not None:
        for definitions in info.name2fixturedefs.values():
            if definitions[-1].scope in FILE_SCOPES:
                return item.nodeid.split("::", 1)[0]
    return item.nodeid


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
        (keep if shard_of(shard_key(item), count) == shard else drop).append(item)
    if drop:
        config.hook.pytest_deselected(items=drop)
        items[:] = keep

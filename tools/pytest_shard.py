"""Run one shard of a test session.

``python -m pytest -p tools.pytest_shard --shard 2/3 tests`` runs the tests that
fall into shard 2 of 3. Every shard collects the same tests and makes the same
choice, so the shards together run each test exactly once.

Without recorded durations a test's shard is a CRC-32 of its node id. Hashing
node ids rather than files keeps the shards even in test count: a few files hold
most of the slow parametrized cases, and splitting by file left one shard with
twice the work of another. pytest-xdist already spreads a file's tests over
workers, so no test may rely on running in the same process as the rest of its
file.

A test that uses a class- or module-scoped fixture is placed by its file
instead, so that the file's shared setup, such as building a native helper,
runs on one runner rather than on every shard.

``--shard-durations DIR`` places tests by the durations that earlier runs wrote
with ``--shard-durations-out FILE``. A few tests take minutes while most take
milliseconds, so even hashed shards differ in work by a factor of two. The
groups are handed out longest first, each to the shard with the least work so
far, and each shard starts its long tests first so that none of them is left
running alone at the end.
"""

import argparse
import heapq
import json
import zlib
from collections import defaultdict
from collections.abc import Callable, Sequence
from pathlib import Path

import pytest

FILE_SCOPES = frozenset({"class", "module"})
#: Tests faster than this are recorded only in a count and a total.
TIMED_SECONDS = 0.2
#: Each shard starts the tests it expects to take this long first.
LONG_SECONDS = 1.0

Estimate = Callable[[str], float]


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


def shard_key(item: pytest.Item) -> str:
    """Return the node id, or the file part of it if the test shares setup."""
    info = getattr(item, "_fixtureinfo", None)
    if info is not None:
        for definitions in info.name2fixturedefs.values():
            if definitions[-1].scope in FILE_SCOPES:
                return item.nodeid.split("::", 1)[0]
    return item.nodeid


def load_durations(directory: Path) -> Estimate:
    """Read every recording in DIR; an unlisted test costs the mean short test."""
    files = sorted(directory.glob("*.json"))
    if not files:
        raise pytest.UsageError(f"--shard-durations: no *.json files in {directory}")
    timed: dict[str, float] = {}
    count, seconds = 0, 0.0
    for path in files:
        data = json.loads(path.read_text(encoding="utf-8"))
        timed.update(data["tests"])
        count += data["short"]["count"]
        seconds += data["short"]["seconds"]
    # A positive estimate keeps new tests from all landing on one shard.
    other = seconds / count if count and seconds > 0 else TIMED_SECONDS / 2
    return lambda nodeid: timed.get(nodeid, other)


def balance(
    keys: Sequence[str], estimates: Sequence[float], count: int
) -> dict[str, int]:
    """Map each key to a shard, longest group first onto the least-loaded shard."""
    work: dict[str, float] = defaultdict(float)
    for key, seconds in zip(keys, estimates):
        work[key] += seconds
    loads = [(0.0, shard) for shard in range(1, count + 1)]
    placed = {}
    for key in sorted(work, key=lambda key: (-work[key], key)):
        load, shard = heapq.heappop(loads)
        placed[key] = shard
        heapq.heappush(loads, (load + work[key], shard))
    return placed


def write_durations(path: Path, seconds: dict[str, float]) -> None:
    timed = {
        nodeid: round(value, 3)
        for nodeid, value in sorted(seconds.items())
        if value >= TIMED_SECONDS
    }
    short = [value for value in seconds.values() if value < TIMED_SECONDS]
    data = {
        "tests": timed,
        "short": {"count": len(short), "seconds": round(sum(short), 3)},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=0) + "\n", encoding="utf-8")


class DurationRecorder:
    """Sum the setup, call and teardown time of each test the session ran."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.seconds: dict[str, float] = defaultdict(float)

    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        self.seconds[report.nodeid] += report.duration

    def pytest_sessionfinish(self) -> None:
        write_durations(self.path, self.seconds)


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--shard",
        type=parse_shard,
        default=None,
        metavar="K/N",
        help="run only the tests in shard K of N",
    )
    parser.addoption(
        "--shard-durations",
        type=Path,
        default=None,
        metavar="DIR",
        help="balance the shards by the durations recorded in DIR",
    )
    parser.addoption(
        "--shard-durations-out",
        type=Path,
        default=None,
        metavar="FILE",
        help="record the duration of each test that runs in FILE",
    )


def pytest_configure(config: pytest.Config) -> None:
    path = config.getoption("shard_durations_out")
    # Under pytest-xdist the controller receives every worker's reports.
    if path is not None and not hasattr(config, "workerinput"):
        config.pluginmanager.register(DurationRecorder(path), "shard-durations-out")


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    selected = config.getoption("shard")
    if selected is None:
        return
    shard, count = selected
    keys = [shard_key(item) for item in items]
    directory = config.getoption("shard_durations")
    if directory is None:
        keep = [item for item, key in zip(items, keys) if shard_of(key, count) == shard]
    else:
        estimate = load_durations(directory)
        placed = balance(keys, [estimate(item.nodeid) for item in items], count)
        keep = [item for item, key in zip(items, keys) if placed[key] == shard]
        keep = long_first(keep, estimate)
    kept = {id(item) for item in keep}
    drop = [item for item in items if id(item) not in kept]
    if drop:
        config.hook.pytest_deselected(items=drop)
    items[:] = keep


def long_first(items: list[pytest.Item], estimate: Estimate) -> list[pytest.Item]:
    """Move long tests that share no setup to the front, longest first."""

    def is_long(item: pytest.Item) -> bool:
        return estimate(item.nodeid) >= LONG_SECONDS and shard_key(item) == item.nodeid

    long = sorted(filter(is_long, items), key=lambda item: -estimate(item.nodeid))
    return long + [item for item in items if not is_long(item)]

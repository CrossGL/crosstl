"""Split one upstream test's subtest loop across verify_upstream_ops groups.

Some unchanged upstream methods run one ``subTest`` for each element of an
``itertools.product`` over their cases and take longer than one native worker
may run. Shard ``k`` of ``n`` of such a method, written ``name[k/n]``, runs the
method as it is, with the ``product`` its module imported replaced by one that
yields every ``n``-th element from the ``k``-th. The ``n`` shards together run
each subtest once, in the upstream order, and each shard records how many of
the loop's elements it ran out of how many the loop has.
"""

import contextlib
import re

SHARD = re.compile(
    r"^(?P<test>[A-Za-z_][\w.]*)\[(?P<shard>[1-9]\d*)/(?P<count>[1-9]\d*)\]$"
)
# The module global that each sharded method takes its subtest cases from.
CASES = "product"


def shard_names(test, count):
    """The names of the ``count`` shards of upstream test ``test``."""
    if count < 2:
        raise ValueError("A sharded test needs at least two shards")
    return tuple(f"{test}[{shard}/{count}]" for shard in range(1, count + 1))


def parse(name):
    """``(test, shard, count)`` for a shard name and ``(name, None, None)`` otherwise."""
    match = SHARD.match(name)
    if match is None:
        if "[" in name or "]" in name:
            raise ValueError(f"Malformed upstream test shard: {name}")
        return name, None, None
    shard, count = int(match["shard"]), int(match["count"])
    if shard > count or count < 2:
        raise ValueError(f"Malformed upstream test shard: {name}")
    return match["test"], shard, count


def expected_selection(shard, count, total):
    """How many of a loop's ``total`` elements shard ``shard`` of ``count`` runs."""
    return len(range(shard - 1, total, count))


@contextlib.contextmanager
def selecting(module, shard, count):
    """Run only shard ``shard`` of ``count`` of ``module``'s subtest loops.

    Yields the tally the shard record keeps: its position, the number of
    elements of every loop it was asked for (``total``) and the number it ran
    (``selected``). The module's own ``product`` is restored afterwards.
    """
    cases = getattr(module, CASES, None)
    if not callable(cases):
        raise ValueError(
            f"{module.__name__} has no {CASES} to take its subtest cases from"
        )
    tally = {"shard": shard, "count": count, "total": 0, "selected": 0}

    def sharded(*iterables, **options):
        elements = list(cases(*iterables, **options))
        chosen = elements[shard - 1 :: count]
        tally["total"] += len(elements)
        tally["selected"] += len(chosen)
        return iter(chosen)

    setattr(module, CASES, sharded)
    try:
        yield tally
    finally:
        setattr(module, CASES, cases)


def validate(name, tally, *, ran):
    """Refuse a shard record that does not cover exactly its share of the loop.

    ``ran`` is false when the method was skipped as a whole, so it must not
    have drawn a single case.
    """
    _, shard, count = parse(name)
    if shard is None:
        if tally is not None:
            raise ValueError(f"An unsharded upstream test records subtests: {name}")
        return
    if not isinstance(tally, dict) or set(tally) != {
        "shard",
        "count",
        "total",
        "selected",
    }:
        raise ValueError(f"An upstream test shard has no subtest record: {name}")
    total = tally["total"]
    if (
        tally["shard"] != shard
        or tally["count"] != count
        or type(total) is not int
        or (total < count if ran else total != 0)
        or tally["selected"] != expected_selection(shard, count, total)
    ):
        raise ValueError(f"An upstream test shard ran the wrong subtests: {name}")

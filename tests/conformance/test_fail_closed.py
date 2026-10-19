import json

from . import fail_closed
from .fail_closed import compare, scan, scan_source, totals

FAIL_OPEN = '''
"""Module docstring mentioning a placeholder is documentation, not output."""

MAP = {}


def emit(name, out):
    """Docstrings that say ; WARNING are skipped too."""
    out.append("; WARNING: lowered to 0")
    out.append(f"{name} /* unsupported construct */")
    return MAP.get(name, name)


def parse(self):
    try:
        return self.parse_block()
    except SyntaxError:
        return BlockNode([])


def refuse(name):
    raise ValueError(f"; WARNING text inside a raised diagnostic is an error message: {name}")
'''


def test_each_fail_open_shape_is_found_and_scoped():
    sites = scan_source(FAIL_OPEN)
    assert sites == {
        ("emit", "fallback-text"): 2,
        ("emit", "identity-fallback"): 1,
        ("parse", "empty-block-recovery"): 1,
    }


def test_closed_shapes_are_not_sites():
    closed = """
def lookup(name, table):
    if name not in table:
        raise KeyError(name)
    return table.get(name, None)


def recover(self):
    try:
        return self.parse_block()
    except SyntaxError:
        return BlockNode([self.error_statement()])
"""
    assert scan_source(closed) == {}


def test_compare_fails_on_growth_new_keys_and_unrecorded_removals():
    baseline = {"a::f::fallback-text": 2, "b::g::identity-fallback": 1}
    assert compare(baseline, dict(baseline)) == []
    grown = compare(
        baseline,
        {**baseline, "a::f::fallback-text": 3, "c::h::empty-block-recovery": 1},
    )
    assert [line.split(":")[0] for line in grown] == [
        "New fail-open site a",
        "New fail-open site c",
    ]
    removed = compare(baseline, {"a::f::fallback-text": 2})
    assert len(removed) == 1 and removed[0].startswith(
        "Removed fail-open site b::g::identity-fallback"
    )
    assert totals(baseline) == {
        "fallback-text": 2,
        "identity-fallback": 1,
        "empty-block-recovery": 0,
    }


def test_translator_has_no_unrecorded_fail_open_sites():
    baseline = json.loads(fail_closed.BASELINE.read_text())
    current = scan()
    assert compare(baseline["sites"], current) == []
    assert baseline["totals"] == totals(current)

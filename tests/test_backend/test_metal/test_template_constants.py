"""The one-pass constant-reference scan agrees with per-name static-constant substitution, and scales linearly."""

import time

import pytest

from crosstl.backend.Metal.preprocessor import MetalPreprocessor
from crosstl.backend.Metal.template_constants import (
    static_constant_references,
    unresolved_constant_references,
)

hypothesis = pytest.importorskip("hypothesis")
st = hypothesis.strategies

NAMES = ("N", "kTile", "_k", "x1", "e5", "value")
FRAGMENTS = NAMES + (
    " ",
    "\t",
    "(",
    ")",
    "+",
    "*",
    "<",
    ">",
    ",",
    ".",
    ":",
    "::",
    ": :",
    "0",
    "1e5",
    "2u",
    '"N"',
    "'N'",
    '"a\\"N"',
    "// N",
    "/* N */",
    "/* N",
    "S::",
    "s.",
    "f(",
    "f (",
    "\n",
)


def substitution_changes(preprocessor, argument, name):
    """The check the scan replaces: does substituting ``name`` with ``0`` change ``argument``?"""
    return (
        preprocessor._substitute_template_argument_static_constants(
            argument, {name: "0"}
        )
        != argument
    )


@hypothesis.settings(max_examples=600, deadline=None, derandomize=True)
@hypothesis.given(
    st.lists(
        st.lists(st.sampled_from(FRAGMENTS), max_size=12).map("".join), max_size=3
    ),
    st.lists(st.sampled_from(NAMES), unique=True),
)
def test_scan_matches_per_name_substitution(arguments, names):
    preprocessor = MetalPreprocessor()
    expected = sorted(
        name
        for name in names
        if any(
            substitution_changes(preprocessor, argument, name) for argument in arguments
        )
    )
    assert unresolved_constant_references(arguments, names) == expected


@pytest.mark.parametrize(
    ("argument", "expected"),
    [
        ("N * 2", {"N"}),
        ("S::N + s.N", {"S", "s"}),
        ("f(N) + g (1)", {"N"}),
        ('"N" + M', {"M"}),
        ("M /* N */ + K // N", {"M", "K"}),
        ("1e5", {"e5"}),
        ("A : : B", {"A"}),
    ],
)
def test_references_skip_members_calls_literals_and_comments(argument, expected):
    assert static_constant_references(argument) == expected


def test_surrounding_whitespace_counts_as_a_change_for_every_name():
    assert unresolved_constant_references([" 4"], ["Z", "A"]) == ["A", "Z"]
    assert unresolved_constant_references(["4"], ["Z", "A"]) == []


def test_many_constants_and_arguments_stay_linear():
    names = [f"local{i}" for i in range(20000)]
    arguments = [f"local{i} + {i}" for i in range(0, 20000, 7)]
    started = time.process_time()
    found = unresolved_constant_references(arguments, names)
    assert found == sorted(f"local{i}" for i in range(0, 20000, 7))
    assert time.process_time() - started < 5

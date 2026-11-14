"""A call argument can bind a variable by reference only if it designates it."""

import re

import pytest

from crosstl.backend.Metal.reference_alias_arguments import call_argument_designates


def _access_end(text, start, limit):
    match = re.compile(r"(?:\s*(?:\.\s*\w+|\[[^\[\]]*\]))*\s*").match(
        text, start, limit
    )
    return match.end()


@pytest.mark.parametrize(
    ("call", "designates"),
    [
        ("f(i)", True),
        ("f(k, i)", True),
        ("f((i))", True),
        ("f(i.x)", True),
        ("f(i[0])", True),
        ("f(c ? i : j)", True),
        ("f((k, i))", True),
        ("f(i + 1)", False),
        ("f(a[(i) * 4 + j])", False),
        ("f(g(k), 2 * i)", False),
    ],
)
def test_only_designating_arguments_can_bind_the_name(call, designates):
    open_paren = call.index("(")
    close_paren = len(call) - 1
    name_start = call.index("i", open_paren)
    assert (
        call_argument_designates(
            call,
            (open_paren, close_paren),
            name_start,
            name_start + 1,
            lambda start, limit: _access_end(call, start, limit),
        )
        is designates
    )


def test_a_name_outside_the_call_is_not_an_argument():
    call = "i + f(k)"
    assert (
        call_argument_designates(call, (5, 7), 0, 1, lambda start, limit: start) is None
    )

"""Joining ``for`` initializers that hold several items."""

import pytest

from crosstl.translator.codegen.for_initializers import (
    ForInitializerError,
    join_for_initializer,
)


def _join(items):
    return join_for_initializer(
        items,
        lambda item: item.startswith("decl:"),
        lambda item: item[len("decl:") :],
        lambda item: item,
    )


@pytest.mark.parametrize(
    "items,expected",
    [
        ([], ""),
        (["decl:int i = 0"], "int i = 0"),
        (["decl:int i = 0", "decl:int j = i + 1"], "int i = 0, j = i + 1"),
        (["decl:const int i = 0", "decl:const int n"], "const int i = 0, n"),
        (
            ["decl:int i = 0", "decl:int values[2] = {1, 2}"],
            "int i = 0, values[2] = {1, 2}",
        ),
        (
            ["decl:device int* p = a", "decl:device int* q = b"],
            "device int* p = a, q = b",
        ),
        (["i = 0", "j = 1;"], "i = 0, j = 1"),
    ],
)
def test_join_for_initializer_emits_one_declaration_or_comma_list(items, expected):
    assert _join(items) == expected


@pytest.mark.parametrize(
    "items,reason",
    [
        (["decl:int i = 0", "j = 1"], "declarations mixed with expressions"),
        (
            ["decl:int i = 0", "decl:uint j = 0"],
            "the declarations do not share one type",
        ),
        (
            ["decl:int i = 0", "decl:float j = 0"],
            "the declarations do not share one type",
        ),
        (
            ["decl:const int i = 0", "decl:const uint j = 0"],
            "the declarations do not share one type",
        ),
        (
            ["decl:int i = 0", "decl:int j;\nj = 1"],
            "a declaration needs more than one statement",
        ),
        (["decl:int i = 0", "decl:"], "a declaration needs more than one statement"),
    ],
)
def test_join_for_initializer_rejects_items_one_declaration_cannot_hold(items, reason):
    with pytest.raises(ForInitializerError) as error:
        _join(items)
    assert error.value.reason == reason

"""Memoized template-argument spellings and the template-id index of materialized Metal structs."""

import gc
import weakref

import pytest

from crosstl.backend.Metal import static_member_constants, template_argument_text
from crosstl.backend.Metal.preprocessor import MetalPreprocessor
from crosstl.backend.Metal.static_member_constants import (
    materialized_struct_for_template_id,
)


@pytest.fixture(autouse=True)
def fresh_memo():
    template_argument_text._memo.clear()
    yield
    template_argument_text._memo.clear()


@pytest.mark.parametrize(
    ("spelling", "normalized"),
    [
        ("  Foo < int ,  2 >  ", "Foo<int,2>"),
        ("device  float  *", "device float*"),
        ("((4))", "4"),
        ("( 0x10u )", "0x10u"),
        ("(true)", "true"),
        # Grouping stays when the parenthesized argument is not a literal.
        ("(2 > 2)", "(2>2)"),
        ("(a) + (b)", "(a)+(b)"),
        ("()", "()"),
        ("T /* element */ , 4 // count", "T,4"),
        ("", ""),
    ],
)
def test_spellings_normalize_as_before(spelling, normalized):
    preprocessor = MetalPreprocessor()
    assert preprocessor._normalize_template_argument_text(spelling) == normalized
    # The memoized answer is the computed one, for any preprocessor.
    assert MetalPreprocessor()._normalize_template_argument_text(spelling) == normalized


def test_memo_starts_over_when_full(monkeypatch):
    monkeypatch.setattr(template_argument_text, "TEMPLATE_ARGUMENT_TEXT_MEMO_LIMIT", 2)
    preprocessor = MetalPreprocessor()
    for spelling in ("A <int>", "B <int>", "C <int>"):
        preprocessor._normalize_template_argument_text(spelling)
    assert list(template_argument_text._memo) == ["C <int>"]
    assert preprocessor._normalize_template_argument_text("A <int>") == "A<int>"


def _with_specializations(entries):
    preprocessor = MetalPreprocessor()
    preprocessor._materialized_struct_specializations.update(entries)
    return preprocessor


def test_template_id_names_its_unique_materialized_struct():
    preprocessor = _with_specializations(
        {
            "Frag_float_1": ("Frag", ("float", "1")),
            "Frag_half_1": ("::Frag", ("half", " 1 ")),
        }
    )
    assert materialized_struct_for_template_id(preprocessor, "Frag<float, 1>") == (
        "Frag_float_1"
    )
    assert materialized_struct_for_template_id(preprocessor, "Frag<half,(1)>") == (
        "Frag_half_1"
    )
    assert materialized_struct_for_template_id(preprocessor, "Frag<int,1>") is None
    assert materialized_struct_for_template_id(preprocessor, "float") is None


def test_template_id_of_two_materialized_structs_names_neither():
    preprocessor = _with_specializations(
        {
            "Frag_float_1": ("Frag", ("float", "1")),
            "Frag_float_1_again": ("Frag", ("float", "(1)")),
        }
    )
    assert materialized_struct_for_template_id(preprocessor, "Frag<float,1>") is None


def test_index_follows_added_replaced_and_restored_specializations():
    preprocessor = _with_specializations({"Frag_float_1": ("Frag", ("float", "1"))})
    assert materialized_struct_for_template_id(preprocessor, "Frag<half,1>") is None

    preprocessor._materialized_struct_specializations["Frag_half_1"] = (
        "Frag",
        ("half", "1"),
    )
    assert materialized_struct_for_template_id(preprocessor, "Frag<half,1>") == (
        "Frag_half_1"
    )

    # Replacing an entry in place keeps the mapping's size and order.
    preprocessor._materialized_struct_specializations["Frag_float_1"] = (
        "Frag",
        ("int", "1"),
    )
    assert materialized_struct_for_template_id(preprocessor, "Frag<float,1>") is None
    assert materialized_struct_for_template_id(preprocessor, "Frag<int,1>") == (
        "Frag_float_1"
    )

    # A retry restores an earlier snapshot as a new mapping.
    preprocessor._materialized_struct_specializations = {
        "Frag_float_1": ("Frag", ("float", "1"))
    }
    assert materialized_struct_for_template_id(preprocessor, "Frag<half,1>") is None
    assert materialized_struct_for_template_id(preprocessor, "Frag<float,1>") == (
        "Frag_float_1"
    )


def test_indexes_belong_to_their_preprocessor():
    first = _with_specializations({"Frag_float_1": ("Frag", ("float", "1"))})
    second = _with_specializations({"Frag_float_one": ("Frag", ("float", "1"))})
    assert materialized_struct_for_template_id(first, "Frag<float,1>") == (
        "Frag_float_1"
    )
    assert materialized_struct_for_template_id(second, "Frag<float,1>") == (
        "Frag_float_one"
    )
    reference = weakref.ref(first)
    del first
    gc.collect()
    # The index does not keep a finished preprocessor alive.
    assert reference() is None
    assert second in static_member_constants._SPECIALIZATION_INDEXES

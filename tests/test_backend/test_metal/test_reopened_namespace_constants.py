"""Integral constants of reopened namespaces and multi-word unsigned integer spellings in Metal static assertions."""

import pytest

from crosstl.backend.Metal.preprocessor import (
    MetalPreprocessor,
    MetalStaticAssertionError,
)
from crosstl.backend.Metal.reopened_namespaces import reopened_namespace_scopes
from crosstl.backend.Metal.signed_type_spellings import fold_integer_type_spellings


def _evaluate(code: str) -> str:
    return MetalPreprocessor()._evaluate_static_assertions(code)


@pytest.mark.parametrize(
    ("tokens", "folded"),
    [
        (["unsigned"], ["uint"]),
        (["unsigned", "int"], ["uint"]),
        (["unsigned", "short", "int"], ["ushort"]),
        (["unsigned", "long", "int"], ["ulong"]),
        (["short", "int"], ["short"]),
        (["long", "int"], ["long"]),
        (["signed", "char"], ["char"]),
        (["int"], ["int"]),
        (["uint"], ["uint"]),
    ],
)
def test_integer_spellings_fold_to_one_metal_scalar(tokens, folded):
    assert fold_integer_type_spellings(tokens) == folded


@pytest.mark.parametrize(
    "spelling", ["unsigned", "unsigned int", "unsigned short int", "long int"]
)
def test_multi_word_integer_constant_resolves_static_assertion(spelling):
    code = f"""
    constant constexpr {spelling} LANES = 4;
    static_assert(LANES == 4, "lanes");
    kernel void k(device int* out [[buffer(0)]]) {{ out[0] = 1; }}
    """
    assert "static_assert" not in _evaluate(code)


def test_multi_word_unsigned_static_member_resolves_static_assertion():
    code = """
    struct Tile {
        static constexpr constant unsigned int kRows = 8;
    };
    static_assert(Tile::kRows == 8, "rows");
    kernel void k(device int* out [[buffer(0)]]) { out[0] = 1; }
    """
    assert "static_assert" not in _evaluate(code)


def test_member_of_reopened_nested_namespace_is_visible_in_later_block():
    code = """
    namespace outer { namespace inner {
    constant constexpr unsigned PER_THREAD = 4;
    } }
    namespace outer { namespace inner {
    static_assert(PER_THREAD == 4, "per thread");
    } }
    kernel void k(device int* out [[buffer(0)]]) { out[0] = 1; }
    """
    assert "static_assert" not in _evaluate(code)


def test_reopened_namespace_member_hides_global_declared_between_blocks():
    code = """
    namespace lanes { constant constexpr int COUNT = 2; }
    constant constexpr int COUNT = 5;
    namespace lanes { static_assert(COUNT == 2, "namespace member"); }
    static_assert(COUNT == 5, "global");
    kernel void k(device int* out [[buffer(0)]]) { out[0] = 1; }
    """
    assert "static_assert" not in _evaluate(code)


def test_reopened_namespace_member_still_checks_false_assertion():
    code = """
    namespace lanes { constant constexpr int COUNT = 2; }
    namespace lanes { static_assert(COUNT == 3, "wrong count"); }
    kernel void k(device int* out [[buffer(0)]]) { out[0] = 1; }
    """
    with pytest.raises(MetalStaticAssertionError) as exc_info:
        _evaluate(code)
    assert exc_info.value.reason == "assertion-failed"


def test_member_of_other_namespace_stays_invisible():
    code = """
    namespace first { constant constexpr int COUNT = 2; }
    namespace second { static_assert(COUNT == 2, "not visible"); }
    kernel void k(device int* out [[buffer(0)]]) { out[0] = 1; }
    """
    with pytest.raises(MetalStaticAssertionError) as exc_info:
        _evaluate(code)
    assert exc_info.value.unresolved_dependencies == ("COUNT",)


def test_reopened_scopes_skip_blocks_nested_in_returned_block():
    spans = [
        (10, 20, "a"),
        (30, 60, "a"),
        (40, 50, "a"),
        (70, 80, "b"),
        (90, 95, "a"),
    ]
    assert reopened_namespace_scopes(spans, (10, 20), 12) == [(30, 60), (90, 95)]
    assert reopened_namespace_scopes(spans, (0, 100), 12) == []

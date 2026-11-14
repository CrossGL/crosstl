"""``using metal::<name>;`` declarations are found in one scan of a unit.

The constrained call pass asked :func:`library_using_declaration_present` once per constrained call, and each call
compiled a pattern for its name and searched the whole unit. On PyTorch's ``ReduceOps.metal`` that took 166 of 376
profiled seconds, and the corpus cell timed out at 300 seconds on Linux.
"""

from crosstl.backend.Metal import library_fallback_lookup as lookup

SOURCE = """
using metal::max;
using ::metal :: isnan ;
namespace ns { using metal::min; }
xusing metal::clamp;
using metal::maximum;
using other::fma;
"""


def test_declarations_are_matched_by_whole_name():
    present = {
        name
        for name in ("max", "isnan", "min", "clamp", "maximum", "fma", "abs")
        if lookup.library_using_declaration_present(SOURCE, name)
    }

    assert present == {"max", "isnan", "min", "maximum"}


def test_a_unit_is_scanned_once_for_every_name():
    code = SOURCE + "\n// " + "x" * 10_000
    lookup._library_using_declarations.cache_clear()

    for name in ("max", "min", "abs", "fma") * 50:
        lookup.library_using_declaration_present(code, name)

    info = lookup._library_using_declarations.cache_info()
    assert info.misses == 1 and info.hits == 199


def test_a_unit_without_using_declarations_has_none():
    assert not lookup.library_using_declaration_present("kernel void k() {}", "max")

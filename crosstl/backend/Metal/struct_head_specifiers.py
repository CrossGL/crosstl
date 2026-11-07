"""Specifiers between ``struct`` and the struct's name.

C++ puts a struct's attribute-specifier-seq, including ``alignas``, between the class-key and the class-head-name
([class.pre]): ``struct alignas(1) float8_e4m3fn { uchar x; ... };`` in PyTorch's ``c10/metal/float8.h``. The
preprocessor's struct scanner read the identifier after ``struct``, so it saw a struct named ``alignas`` and skipped
the definition. Its template constructor was then materialized as a free function that kept its
mem-initializer-list (``inline float8_e4m3fn_float(float value) : x(...) {}``), which the parser rejects.

:func:`skip_struct_head_specifiers` steps over those specifiers. ``alignas(N)`` changes the struct's layout when
``N`` exceeds its natural alignment, and an attribute such as ``packed`` may change it too, so
:func:`alignment_is_natural` reports whether the specifiers leave the layout unchanged. That holds for ``alignas(1)``, ``alignas(0)`` and no specifier: an alignment weaker than the
natural one is ill-formed ([dcl.align]/5), so a well-formed ``alignas(1)`` is the natural alignment.
"""

from __future__ import annotations

import re

__all__ = ["alignment_is_natural", "skip_struct_head_specifiers"]

_SPECIFIER_START = re.compile(r"\s*(?P<kind>alignas\s*\(|__attribute__\s*\(\(|\[\[)")
_NATURAL_ALIGNMENT = re.compile(r"alignas\s*\(\s*[01]\s*[uU]?\s*\)")


def _balanced_end(code: str, start: int, opening: str, closing: str) -> int:
    depth = 0
    for index in range(start, len(code)):
        if code[index] == opening:
            depth += 1
        elif code[index] == closing:
            depth -= 1
            if depth == 0:
                return index + 1
    return -1


def skip_struct_head_specifiers(code: str, index: int) -> tuple[int, list[str]]:
    """The position after the specifiers that start at ``index``, and the specifiers' text.

    An unbalanced specifier stops the scan where it starts, so the caller reads it as the name and skips the
    declaration as before.
    """
    alignments: list[str] = []
    while True:
        match = _SPECIFIER_START.match(code, index)
        if match is None:
            return index, alignments
        kind = match.group("kind")
        if kind == "[[":
            end = _balanced_end(code, match.start("kind"), "[", "]")
        else:
            end = _balanced_end(code, match.start("kind") + len(kind) - 1, "(", ")")
        if end < 0:
            return index, alignments
        if kind.startswith("__attribute__"):
            end = code.find(")", end)
            end = -1 if end < 0 else end + 1
            if end < 0:
                return index, alignments
        alignments.append(code[match.start("kind") : end])
        index = end
        while index < len(code) and code[index].isspace():
            index += 1


def alignment_is_natural(alignments: list[str]) -> bool:
    """Whether the specifiers leave a well-formed struct at its natural layout.

    Any attribute other than a natural ``alignas`` (``packed``, for one) may change the layout, so it does not.
    """
    return all(
        _NATURAL_ALIGNMENT.fullmatch(" ".join(text.split())) for text in alignments
    )

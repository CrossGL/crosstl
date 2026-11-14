"""Type names that Metal type traits can compare without further resolution.

``is_same<A, B>`` folds to a constant once both operands are spelled as canonical concrete types. Two kinds of
concrete type were not recognised and left the trait unevaluated:

- Metal's matrix types. ``half4x4`` and ``matrix<half, 4, 4>`` are the same type, so both spell the canonical
  ``half4x4``. MSL defines ``float`` and ``half`` matrices; ``bfloat`` matrices are accepted by the compiler
  through the template spelling, and are canonicalised the same way.
- Structures declared through a typedef, ``typedef struct { ... } block_q4_0;``, which is how C-style headers
  shared with host code declare their blocks.
"""

import re
from typing import Callable, Iterator, Optional, Set

_MATRIX_SCALARS = ("float", "half", "bfloat")
_MATRIX_NAME = re.compile(
    r"(?P<scalar>float|half|bfloat)(?P<columns>[234])x(?P<rows>[234])"
)
_MATRIX_TEMPLATE = re.compile(
    r"(?:::)?(?:metal::)?matrix\s*<\s*(?P<scalar>float|half|bfloat)\s*,\s*(?P<columns>[234])\s*,"
    r"\s*(?P<rows>[234])\s*>"
)
_TYPEDEF_AGGREGATE = re.compile(
    r"\btypedef\s+(?:struct|union)\b(?:\s+[A-Za-z_]\w*)?\s*\{"
)
_DECLARATOR_NAME = re.compile(r"[A-Za-z_]\w*")

METAL_MATRIX_TYPE_NAMES = frozenset(
    f"{scalar}{columns}x{rows}"
    for scalar in _MATRIX_SCALARS
    for columns in "234"
    for rows in "234"
)


def canonical_matrix_type(type_text: str) -> Optional[str]:
    """The ``half4x4`` spelling of a matrix type, or ``None`` when ``type_text`` is not one."""
    text = type_text.strip()
    if _MATRIX_NAME.fullmatch(text):
        return text
    match = _MATRIX_TEMPLATE.fullmatch(text)
    if match is None:
        return None
    return f"{match.group('scalar')}{match.group('columns')}x{match.group('rows')}"


def typedef_aggregate_names(
    code: str,
    find_matching_brace: Callable[[str, int], Optional[int]],
    ignored: Callable[[int], bool] = lambda _position: False,
) -> Set[str]:
    """Names that ``typedef struct { ... } name;`` and ``typedef union`` declarations give their type.

    Pointer and array declarators in the same typedef name other types and are skipped.
    """
    return set(_typedef_aggregate_names(code, find_matching_brace, ignored))


def _typedef_aggregate_names(code, find_matching_brace, ignored) -> Iterator[str]:
    for match in _TYPEDEF_AGGREGATE.finditer(code):
        if ignored(match.start()):
            continue
        body_end = find_matching_brace(code, match.end() - 1)
        if body_end is None:
            continue
        declarators_end = code.find(";", body_end)
        if declarators_end == -1:
            continue
        for declarator in code[body_end + 1 : declarators_end].split(","):
            declarator = declarator.strip()
            if _DECLARATOR_NAME.fullmatch(declarator):
                yield declarator

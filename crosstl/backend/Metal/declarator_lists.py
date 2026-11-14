"""Give each declarator of a concrete struct type its own declaration.

``S a, b;`` declares ``a`` and ``b`` exactly as ``S a; S b;`` does: each
init-declarator is analyzed as if it were in a declaration by itself
([dcl.decl]/3), and each name's point of declaration follows its own complete
declarator ([basic.scope.pdecl]/1). Member-call lowering resolves a receiver
from the declaration that introduces it, so a struct-typed name after the
first declarator of a list would otherwise have no declaration of its own.

Only simple-declarations that start a statement are split, and only when every
declarator is a plain (optionally pointer, reference or array) declarator with
an optional initializer. ``for`` init-statements, parameter lists and any list
that cannot be split exactly stay unchanged.
"""

import re
from bisect import bisect_right
from typing import Iterable, List, Optional, Sequence, Tuple

_SPECIFIER = (
    r"(?:const|volatile|thread|threadgroup|threadgroup_imageblock|device|"
    r"constant|static|constexpr)"
)
_TRAILING_SPECIFIERS = re.compile(rf"(?:\s+{_SPECIFIER}\b)*")
_DECLARATOR = re.compile(
    r"\s*(?:[*&]\s*(?:(?:const|volatile)\s+)*)*"
    r"[A-Za-z_]\w*\s*"
    r"(?:\[[^\[\]]*\]\s*)*"
    r"(?:(?:=|\{|\().*)?",
    re.DOTALL,
)
_OPENERS = {"(": ")", "[": "]", "{": "}"}
_CLOSERS = {")", "]", "}"}


def _previous_significant_character(
    code: str, position: int, spans: Sequence[Tuple[int, int]]
) -> Optional[str]:
    """Return the nearest non-space character before ``position``.

    Comments and literals in ``spans`` (sorted, disjoint) are skipped, so a
    comment line above a declaration does not hide the preceding ``;``.
    """
    starts = [start for start, _end in spans]
    index = position - 1
    while index >= 0:
        containing = bisect_right(starts, index) - 1
        if containing >= 0 and index < spans[containing][1]:
            index = spans[containing][0] - 1
        elif code[index].isspace():
            index -= 1
        else:
            return code[index]
    return None


def _statement_end(code: str, start: int) -> Optional[int]:
    """Return the offset of the top-level ``;`` that ends the statement."""
    stack: List[str] = []
    for index in range(start, len(code)):
        character = code[index]
        if character in _OPENERS:
            stack.append(_OPENERS[character])
        elif character in _CLOSERS:
            if not stack or stack.pop() != character:
                return None
        elif character == ";" and not stack:
            return index
    return None


def _top_level_declarators(text: str) -> List[str]:
    pieces: List[str] = []
    depth = 0
    piece_start = 0
    for index, character in enumerate(text):
        if character in _OPENERS:
            depth += 1
        elif character in _CLOSERS:
            depth -= 1
        elif character == "," and depth == 0:
            pieces.append(text[piece_start:index])
            piece_start = index + 1
    pieces.append(text[piece_start:])
    return pieces


def _overlaps(start: int, end: int, spans: Sequence[Tuple[int, int]]) -> bool:
    return any(span_start < end and start < span_end for span_start, span_end in spans)


def split_struct_declarator_lists(
    code: str,
    struct_names: Iterable[str],
    ignored_spans: Sequence[Tuple[int, int]] = (),
) -> str:
    """Split ``S a, b;`` into ``S a; S b;`` for each concrete struct ``S``."""
    names = sorted(set(struct_names), key=lambda name: (-len(name), name))
    if not names or "," not in code:
        return code
    pattern = re.compile(
        rf"(?<![\w:.>])(?:{_SPECIFIER}\s+)*"
        rf"(?:{'|'.join(re.escape(name) for name in names)})\b"
    )
    spans = sorted(ignored_spans)
    replacements: List[Tuple[int, int, str]] = []
    last_end = 0
    for match in pattern.finditer(code):
        start = match.start()
        if start < last_end:
            continue
        if _previous_significant_character(code, start, spans) not in (
            None,
            ";",
            "{",
            "}",
        ):
            continue
        specifier_end = _TRAILING_SPECIFIERS.match(code, match.end()).end()
        end = _statement_end(code, specifier_end)
        if end is None:
            continue
        declarators = _top_level_declarators(code[specifier_end:end])
        if len(declarators) < 2:
            continue
        if not declarators[0][:1].isspace() and declarators[0][:1] not in "*&":
            continue
        if not all(_DECLARATOR.fullmatch(piece) for piece in declarators):
            continue
        if _overlaps(start, end, spans):
            continue
        specifiers = code[start:specifier_end]
        replacement = "; ".join(
            f"{specifiers} {declarator.strip()}" for declarator in declarators
        )
        replacements.append((start, end, replacement))
        last_end = end
    if not replacements:
        return code
    pieces: List[str] = []
    cursor = 0
    for start, end, replacement in replacements:
        pieces.append(code[cursor:start])
        pieces.append(replacement)
        cursor = end
    pieces.append(code[cursor:])
    return "".join(pieces)

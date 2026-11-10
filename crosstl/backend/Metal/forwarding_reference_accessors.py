"""Reference accessors that forward to another accessor of the same struct.

``thread T& at_t(short i, short j) thread { return at(j, i); }`` returns the
reference ``at`` returns, so it names the same storage element as ``at(j, i)``.
Its direct storage expression is ``at``'s, with ``at``'s parameters bound to
the forwarded arguments (``self.frags[(j) * 2 + (i)]``); the caller then binds
``at_t``'s own parameters as for any direct accessor.

The callee is selected with the direct-accessor rules (constness of the
forwarding method, argument count, no competing overload) and its arguments
must be side-effect free and used once, so the proof is the one the
direct-accessor rewrite already applies. Forwarding chains are followed up to
a small depth; a cycle or anything else is left to the caller, which refuses
the call.
"""

from __future__ import annotations

import re
from typing import Any

_MAX_FORWARDING_DEPTH = 8
_CALL_RE = re.compile(
    r"(?:(?:self|this)\s*(?:\.|->)\s*)?(?P<name>[A-Za-z_]\w*)\s*\(",
)


def forwarded_storage_expression(
    preprocessor: Any,
    struct: Any,
    method: Any,
    expression: str,
    depth: int = 0,
) -> str | None:
    """The storage ``expression`` (``return <expression>;`` of ``method``) names, or ``None``."""

    from .preprocessor import MetalStructMethodError

    if depth >= _MAX_FORWARDING_DEPTH:
        return None
    match = _CALL_RE.match(expression)
    if match is None:
        return None
    arg_open = match.end() - 1
    arg_close = preprocessor._find_matching_delimiter(expression, arg_open, "(", ")")
    if arg_close != len(expression) - 1:
        return None
    candidates = [
        candidate
        for candidate in struct.methods
        if candidate.name == match.group("name")
        and candidate is not method
        and preprocessor._method_returns_lvalue_reference(candidate)
    ]
    if not candidates:
        return None
    try:
        callee = preprocessor._select_direct_reference_method(
            expression,
            struct,
            candidates,
            arg_open,
            receiver_is_const=method.is_const,
        )
        _end, storage = preprocessor._direct_reference_accessor_rewrite(
            expression,
            struct,
            callee,
            receiver="self",
            arg_open=arg_open,
            depth=depth + 1,
        )
    except MetalStructMethodError:
        return None
    return storage


_IF_CONSTEXPR_HEAD_RE = re.compile(r"if\s+constexpr\s*\(")
_MAX_BRANCH_FOLDS = 16


def selected_constexpr_body(preprocessor: Any, body: str) -> str:
    """``body`` reduced through enclosing braces and ``if constexpr`` statements whose condition is a constant.

    Only a body that consists of exactly one such statement is reduced, so the
    result is the single statement the instantiation executes. Anything else is
    returned unchanged.
    """

    for _ in range(_MAX_BRANCH_FOLDS):
        text = body.strip()
        if (
            text.startswith("{")
            and preprocessor._find_matching_delimiter(text, 0, "{", "}")
            == len(text) - 1
        ):
            body = text[1:-1]
            continue
        head = _IF_CONSTEXPR_HEAD_RE.match(text)
        if head is None:
            return body
        open_paren = head.end() - 1
        close = preprocessor._find_matching_delimiter(text, open_paren, "(", ")")
        if close is None:
            return body
        ok, value = preprocessor._evaluate_static_integral_expression(
            text[open_paren + 1 : close]
        )
        if not ok or value is None:
            return body
        then_start = close + 1
        while then_start < len(text) and text[then_start].isspace():
            then_start += 1
        if then_start >= len(text) or text[then_start] != "{":
            return body
        then_end = preprocessor._find_matching_delimiter(text, then_start, "{", "}")
        if then_end is None:
            return body
        rest = text[then_end + 1 :].strip()
        if rest and not re.match(r"else\b", rest):
            return body
        if value:
            body = text[then_start : then_end + 1]
        else:
            body = rest[4:] if rest else ""
    return body

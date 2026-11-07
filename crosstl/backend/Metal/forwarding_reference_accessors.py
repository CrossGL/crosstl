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

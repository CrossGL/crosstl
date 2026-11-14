"""Inline member-template accessors that return a reference into storage.

``template <int i, int j> thread T& frag_at() thread { return frags[i * C + j]; }``
called as ``frag_at<0, 1>()`` from another member names the storage element
``self.frags[0 * C + 1]``. Lowering the call to a by-value free function would
lose that identity (a write through the result, or passing it to a
``thread T&`` parameter, would update a copy), so the call is replaced by the
storage expression itself. The proof is the one concrete reference accessors
use: the body is a single ``return`` of a non-static data member with
side-effect-free indices, every template parameter appears exactly once, and
the call's result is neither bound to a reference nor returned by reference.
Anything else is refused by the caller.
"""

from __future__ import annotations

import re
from dataclasses import replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from .preprocessor import MetalPreprocessor

# Template arguments must already be integer constants: literal arithmetic
# such as ``0`` or ``(1 + 2) * 4``. A runtime operand (``index.value`` on a
# local) is not a valid template argument and stays refused.
_INTEGER_CONSTANT_RE = re.compile(r"[\s()+\-*/%]*\d[\d\s()+\-*/%uUlL]*")


def inline_template_reference_accessor(
    preprocessor: MetalPreprocessor,
    struct,
    enclosing_method,
    selected,
    body: str,
    call_start: int,
    arg_open: int,
    explicit_arguments: list[str],
) -> tuple[int, str] | None:
    """Return ``(call_end, storage_expression)`` or raise the usual refusal.

    Returns ``None`` when ``selected`` does not return an lvalue reference or a
    template argument is not an integer constant, so the caller keeps its
    by-value lowering or its refusal.
    """
    if not preprocessor._method_returns_lvalue_reference(selected):
        return None
    if len(explicit_arguments) != len(selected.template_parameters) or not all(
        _INTEGER_CONSTANT_RE.fullmatch(argument.strip())
        for argument in explicit_arguments
    ):
        return None
    accessor = replace(
        selected,
        parameters=", ".join(
            f"{selected.template_parameter_types[name]} {name}"
            for name in selected.template_parameters
        ),
        parameter_names=list(selected.template_parameters),
    )
    if preprocessor._reference_call_binds_or_escapes(
        body,
        call_start,
        enclosing_return_type=enclosing_method.return_type,
    ):
        arg_close = preprocessor._find_matching_delimiter(body, arg_open, "(", ")")
        preprocessor._reject_reference_returning_method_call(
            body, struct, accessor, arg_open, arg_close
        )
    return preprocessor._direct_reference_accessor_rewrite(
        body,
        struct,
        accessor,
        receiver="self",
        arg_open=arg_open,
        argument_values=[argument.strip() for argument in explicit_arguments],
    )


class _TemplateReferenceAccessor(Exception):
    """Raised by member-template overload selection when the selected overload returns a reference.

    It carries the concrete (instantiated) method, so the caller can inline the
    call as storage with the direct-accessor proof instead of emitting a
    by-value function. Only callers that ask for it receive it. It is a signal the preprocessor raises and
    catches, never a refusal, so it is private and carries no diagnostic code.
    """

    def __init__(self, method) -> None:
        super().__init__(method.name)
        self.method = method

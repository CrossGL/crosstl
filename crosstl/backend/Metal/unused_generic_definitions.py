"""Generic definitions that CrossGL cannot spell but no emitted code uses.

A function template body can name a type through an expression of its parameters::

    template <typename T>
    vec<T, 2> twiddle(int k, int p) {
        using phase_T = decltype(M_PI_F * T(0));   // [dcl.type.decltype], dependent on T
        ...
    }

Each instantiation spells ``phase_T``, because ``T`` is bound and the ``decltype`` folds to a built-in type, but the
generic CrossGL function cannot: CrossGL declares a type, not an expression. The importer still renders the generic
definition of every template it keeps, so a template whose every call reaches a concrete instantiation (a call from a
member of a class template instance, for example) was refused for a definition nothing calls
(:class:`.generic_local_aliases.MetalLocalTypeAliasResolutionError`).

The refusal is deferred instead. If the rendered program names the generic definition anywhere, the deferred error is
raised, so a generic call never loses its callee. Only a definition no emitted code names is left out; an unused
definition has no effect on the program ([basic.def.odr]: a function that is not odr-used needs no definition).

Only free functions with an ordinary identifier as their name are deferred. An operator or a member function can be
used without being named (``a * b``, a constructor, a conversion), so its refusal is never deferred.
"""

from __future__ import annotations

import re
from typing import Any, Callable

from .generic_local_aliases import MetalLocalTypeAliasResolutionError

__all__ = [
    "raise_referenced_deferred_definitions",
    "render_generic_definition",
]

_PLAIN_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_STAGE_QUALIFIERS = frozenset(
    {"vertex", "fragment", "kernel", "mesh", "object", "visible", "intersection"}
)


def _type_parameter_names(converter: Any, function: Any) -> list[str]:
    value_names = set(converter.value_template_parameter_names(function))
    return [
        name
        for name in converter.unbound_template_parameter_names(function)
        if name not in value_names
    ]


def _deferrable_name(converter: Any, function: Any) -> str | None:
    """The CrossGL name of ``function`` when its refusal may wait for a use, else ``None``."""

    name = getattr(function, "name", None)
    if not isinstance(name, str) or not _PLAIN_NAME.fullmatch(name):
        return None
    if name.startswith("operator") or getattr(function, "method_qualifiers", None):
        return None
    if getattr(function, "qualifier", None) in _STAGE_QUALIFIERS:
        return None
    if not _type_parameter_names(converter, function):
        return None
    output = converter.sanitize_identifier(converter.function_output_name(function))
    if not isinstance(output, str) or not _PLAIN_NAME.fullmatch(output):
        return None
    return output


def render_generic_definition(
    converter: Any, function: Any, render: Callable[[], str]
) -> str:
    """Render the generic definition of ``function``, deferring an unspellable body-local alias.

    When ``render`` raises :class:`MetalLocalTypeAliasResolutionError` for a deferrable free function template, the
    error is recorded on ``converter.deferred_generic_definition_errors`` and nothing is emitted.
    """

    try:
        return render()
    except MetalLocalTypeAliasResolutionError as error:
        name = _deferrable_name(converter, function)
        if name is None:
            raise
        deferred = converter.__dict__.setdefault(
            "deferred_generic_definition_errors", []
        )
        deferred.append((name, error))
        return ""


def raise_referenced_deferred_definitions(converter: Any, code: str) -> None:
    """Raise the first deferred error whose generic definition ``code`` names."""

    for name, error in getattr(converter, "deferred_generic_definition_errors", ()):
        if re.search(rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])", code):
            raise error

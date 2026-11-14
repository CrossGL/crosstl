"""Function templates whose result type a CrossGL generic function cannot carry.

A function template can name its result through an expression of its parameters::

    template <typename T, typename U>
    decltype(T(0) + U(0)) add(T x, U y);           // [dcl.type.decltype]

    template <typename U, typename V>
    using common_dtype = decltype(U(0) + V(0));
    template <typename T, typename U>
    common_dtype<T, U> mul(T x, U y);              // the same decltype, through [temp.alias]/2

    template <typename T>
    auto twice(T x) -> decltype(x + x);            // a trailing return type, [dcl.fct]/2

A CrossGL generic function declares a type as its result, not an expression, so the generic path would print the
``decltype`` with its parameters unsubstituted. The constrained call pass therefore materializes such a template as
a carrier whose every constraint holds (constrained_call_carriers.py). Each reachable call gets a concrete
instantiation, whose result is a ``decltype`` of built-in types that alias_template_types.py folds. A call the pass
cannot deduce stays with the generic path, which refuses the dependent result (MetalCrossGLCodeGen).
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Mapping, Sequence

from .alias_template_types import find_alias_templates

__all__ = ["dependent_result_template_names", "has_dependent_decltype"]

_DECLTYPE_RE = re.compile(r"(?<![\w:.])decltype\s*\(")
_IDENTIFIER_RE = re.compile(r"[A-Za-z_]\w*")


def has_dependent_decltype(text: str, parameters: Iterable[str]) -> bool:
    """Whether ``text`` contains a ``decltype`` and names one of ``parameters``."""

    if not text or _DECLTYPE_RE.search(text) is None:
        return False
    return bool(set(_IDENTIFIER_RE.findall(text)) & set(parameters))


def _result_type_text(
    preprocessor: Any, header: str, name: str
) -> tuple[str, Sequence[str]] | None:
    """The result type ``header`` declares for ``name`` (the leading type, plus a trailing return type) and the
    names of its function parameters, which a trailing ``decltype`` can name."""

    match = re.search(rf"(?<![\w.]){re.escape(name)}\s*(?:<[^(){{}};]*>)?\s*\(", header)
    if match is None:
        return None
    leading = header[: match.start()]
    open_paren = match.end() - 1
    close_paren = preprocessor._find_matching_delimiter(header, open_paren, "(", ")")
    if close_paren is None or close_paren < 0:
        return leading, ()
    parameters = preprocessor._parameter_identifier_names(
        header[open_paren + 1 : close_paren]
    )
    trailing = header[close_paren + 1 :]
    arrow = trailing.find("->")
    if arrow < 0:
        return leading, parameters
    return f"{leading} {trailing[arrow + 2 :]}", parameters


def _names_decltype_alias(
    text: str, alias_names: set[str], parameters: Sequence[str]
) -> bool:
    words = set(_IDENTIFIER_RE.findall(text))
    return bool(words & alias_names) and bool(words & set(parameters))


def dependent_result_template_names(
    preprocessor: Any, code: str, templates: Sequence[Any]
) -> set[str]:
    """Names of the templates in ``templates`` whose result type is a ``decltype`` of their parameters."""

    if "decltype" not in code:
        return set()
    aliases: Mapping[str, Sequence[Any]] = find_alias_templates(preprocessor, code)
    alias_names = {
        name
        for name, declarations in aliases.items()
        if any(_DECLTYPE_RE.search(alias.target) for alias in declarations)
    }
    names: set[str] = set()
    for template in templates:
        declaration_start = template.span[1] - len(template.source)
        header = code[declaration_start : template.body_start]
        declared = _result_type_text(preprocessor, header, template.name)
        if declared is None:
            continue
        result, function_parameters = declared
        parameters = template.template_parameters
        if has_dependent_decltype(
            result, [*parameters, *function_parameters]
        ) or _names_decltype_alias(result, alias_names, parameters):
            names.add(template.name)
    return names

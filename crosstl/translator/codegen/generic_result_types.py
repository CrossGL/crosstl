"""Refuse a generic specialization whose result type is still an expression of its parameters.

An importer can carry a source function template's result type into CrossGL as written, for instance Metal's
``decltype(T(0) + U(0))`` or a trailing ``decltype(x + x)``. CrossGL has no ``decltype``: specialization substitutes
types for the generic's type parameters, so the result stays an expression, now of concrete operand types, and no
target can declare a function with it. The Metal importer materializes such templates at every call it can deduce
(``crosstl/backend/Metal/dependent_result_templates.py``); a call that reaches the generic path anyway is refused
here rather than emitting the ``decltype`` text as a type.
"""

from __future__ import annotations

import re

__all__ = ["GenericResultTypeError", "reject_expression_result_type"]

_DECLTYPE_RE = re.compile(r"(?<![\w:.])decltype\s*\(")


class GenericResultTypeError(ValueError):
    """A generic specialization's result type is a ``decltype`` expression, which CrossGL cannot declare."""

    project_diagnostic_code = "project.translate.generic-result-type-unresolved"
    missing_capabilities = ("generic.expression-result-type",)

    def __init__(self, function_name, type_arguments, result_type):
        arguments = ", ".join(str(argument) for argument in type_arguments)
        super().__init__(
            f"generic function '{function_name}' specialized for ({arguments}) declares its result "
            f"as '{result_type}', an expression type that CrossGL cannot declare"
        )
        self.function_name = function_name
        self.type_arguments = tuple(type_arguments)
        self.result_type = result_type
        self.suggested_action = (
            "declare the result type of the function template as a type, or call it with "
            "arguments whose types the importer can deduce"
        )


def _type_text(type_value) -> str:
    if type_value is None:
        return ""
    if isinstance(type_value, str):
        return type_value
    name = getattr(type_value, "name", None)
    return str(name if isinstance(name, str) else type_value)


def reject_expression_result_type(func, key, result_type):
    """Raise :class:`GenericResultTypeError` when ``result_type`` of the specialization ``key`` is a ``decltype``."""

    text = _type_text(result_type)
    if _DECLTYPE_RE.search(text):
        raise GenericResultTypeError(getattr(func, "name", key[0]), key[1], text)

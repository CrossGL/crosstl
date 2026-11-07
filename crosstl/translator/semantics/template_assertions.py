"""Dependent ``static_assert`` declarations of generic structures: checked in specializations, never in the template.

A static assertion declared in a generic structure whose condition names one of the structure's type parameters is a
dependent assertion. C++ checks it when the template is instantiated, once for each specialization; a template that is
never instantiated is not checked at all (C++17 [temp.res]/8, MSL 4.1 section 1.4). The parser records the template
and its parameters on every assertion declared in a generic structure's body (``StaticAssertNode.template`` and
``StaticAssertNode.generic_params``). An importer materializes each specialization the program names as a structure of
its own, with the substituted condition, and keeps the template beside them; the Metal, HLSL and GLSL generators emit
a generic structure only as the instantiations the program names.

:func:`resolve_template_assertions` removes the dependent assertions of a generic structure that no live code names,
the same templates :mod:`.declared_types` leaves unread: none of them is ever instantiated. A dependent assertion of a
template that live code still names would have to be checked once per instantiation, which the translator does not
do, so it raises :class:`~.static_assertions.StaticAssertionUnresolvedError`. A structure-scope assertion that names
none of its template's parameters stays, and :mod:`.static_assertions` checks it like any other.
"""

from __future__ import annotations

import re

from .. import ast as A
from .declared_types import _dead_functions, _nodes, _uninstantiated_templates
from .static_assertions import StaticAssertionUnresolvedError, _condition_text

__all__ = ["resolve_template_assertions"]

_NAME = re.compile(r"[A-Za-z_]\w*")
# An importer spells a source expression it does not model as one identifier, each character no identifier may hold
# written as ``_u`` and its code point in hex (``metal::is_same_v<T, U>`` becomes ``metal_u3a_u3ais_same_v_u3cT...``).
# Every ASCII punctuation character has two hex digits.
_ESCAPED_CHARACTER = re.compile(r"_u[0-9a-f]{2}")


def _condition_names(condition):
    """Every identifier spelled in ``condition``: value names, callees, names in type and escaped source spellings."""
    names = set()
    for node in _nodes(condition):
        if isinstance(node, A.LiteralNode):
            continue
        for field, value in vars(node).items():
            if field in ("parent", "source_location", "op") or not isinstance(
                value, str
            ):
                continue
            for name in _NAME.findall(value):
                names.add(name)
                names.update(part for part in _ESCAPED_CHARACTER.split(name) if part)
    return names


def _is_dependent(assertion):
    parameters = set(getattr(assertion, "generic_params", None) or ())
    if not getattr(assertion, "template", None) or not parameters:
        return False
    return bool(_condition_names(assertion.condition) & parameters)


def _instantiated_template_names(program, skip):
    """The names of the generic structures that live code names, directly or through another named template."""
    unread = _uninstantiated_templates(program, skip)
    return {
        node.name
        for node in _nodes(program, skip)
        if isinstance(node, A.StructNode)
        and getattr(node, "generic_params", None)
        and isinstance(node.name, str)
        and id(node) not in unread
    }


def resolve_template_assertions(program, target) -> int:
    """Remove the dependent assertions of generic structures no live code names; returns how many.

    Raises :class:`~.static_assertions.StaticAssertionUnresolvedError` for the first dependent assertion of a generic
    structure that live code still names.
    """
    if not isinstance(program, A.ShaderNode):
        return 0
    declared = list(getattr(program, "static_assertions", None) or [])
    dependent = [assertion for assertion in declared if _is_dependent(assertion)]
    if not dependent:
        return 0
    # The dependent assertions are not code that instantiates anything: a template named only by them is unused.
    skip = _dead_functions(program) | {id(assertion) for assertion in dependent}
    instantiated = _instantiated_template_names(program, skip)
    for assertion in dependent:
        if assertion.template in instantiated:
            parameters = ", ".join(assertion.generic_params)
            raise StaticAssertionUnresolvedError(
                str(target or ""),
                _condition_text(assertion.condition),
                f"it depends on the type parameters ({parameters}) of the generic structure "
                f"{assertion.template}, which live code instantiates, and the translator does not check an "
                "assertion once for each instantiation",
            )
    removed = {id(assertion) for assertion in dependent}
    program.static_assertions = [
        assertion for assertion in declared if id(assertion) not in removed
    ]
    return len(removed)

"""Declaration-only primary function templates that explicit specializations complete.

``template <typename T> T f(T);`` followed by ``template <> float f<float>(float) { ... }`` declares a primary
template that is never defined: every call ``f(x)`` deduces ``T`` and is only well-formed (and linkable) when an
explicit specialization exists for the deduced arguments ([temp.expl.spec]/1, [temp.over]/1). The specialization
is not a function of its own, so a call never names it directly.

The Metal parser keeps such a primary as a body-less ``FunctionNode`` (``is_template_prototype``), so the importer
can emit a CrossGL generic prototype and mark each specialization with ``@explicit_specialization``. Generators then
deduce the template arguments of each call and select the specialization, and refuse a call no specialization
covers, which a C++ linker would reject as an undefined symbol.

A prototype is retained only where it is needed: when no definition of a primary of that name exists and at least
one explicit specialization names it. Redeclarations with the same parameter types collapse onto the first one.
"""

from __future__ import annotations

import re
from typing import Any, Iterable

__all__ = ["retain_declared_template_primaries"]

_SPECIALIZATION_NAME = re.compile(r"^(?P<base>[A-Za-z_]\w*)\s*<.*>$", re.DOTALL)


def _scope(function: Any) -> str:
    return str(getattr(function, "namespace", "") or "")


def _is_template(function: Any) -> bool:
    return bool(
        getattr(function, "generics", None)
        or getattr(function, "template_parameters", None)
    )


def _specialized_primaries(functions: Iterable[Any]) -> set[tuple[str, str]]:
    """``(namespace, primary name)`` of each explicit specialization ``name<args>``."""
    found = set()
    for function in functions:
        if _is_template(function) or getattr(function, "body", None) is None:
            continue
        match = _SPECIALIZATION_NAME.match(str(getattr(function, "name", "") or ""))
        if match is not None:
            found.add((_scope(function), match.group("base")))
    return found


_TYPE_PARAMETER_KINDS = frozenset({"typename", "class"})


def _has_non_type_parameters(function: Any) -> bool:
    """Whether the template takes value or pack parameters, which CrossGL generics cannot.

    The importer instantiates such templates per call itself and binds each
    call to its explicit specialization, so a retained primary would only
    offer it a body-less clone to call instead.
    """
    return any(
        not isinstance(parameter, (tuple, list))
        or not parameter
        or parameter[0] not in _TYPE_PARAMETER_KINDS
        for parameter in getattr(function, "template_parameters", None) or ()
    )


def _signature(function: Any) -> tuple[str, ...]:
    return tuple(
        str(getattr(parameter, "vtype", None))
        for parameter in getattr(function, "params", None) or ()
    )


def retain_declared_template_primaries(functions: list[Any]) -> list[Any]:
    """Return ``functions`` keeping only the template prototypes explicit specializations need."""
    if not any(
        getattr(function, "is_template_prototype", False) for function in functions
    ):
        return functions
    defined = {
        (_scope(function), str(function.name))
        for function in functions
        if _is_template(function)
        and getattr(function, "body", None) is not None
        and not getattr(function, "is_template_prototype", False)
    }
    specialized = _specialized_primaries(functions)
    retained = []
    declared: set[tuple[str, str, tuple[str, ...]]] = set()
    for function in functions:
        if not getattr(function, "is_template_prototype", False):
            retained.append(function)
            continue
        key = (_scope(function), str(function.name))
        signature = (*key, _signature(function))
        if (
            key in defined
            or key not in specialized
            or signature in declared
            or _has_non_type_parameters(function)
        ):
            continue
        declared.add(signature)
        retained.append(function)
    return retained

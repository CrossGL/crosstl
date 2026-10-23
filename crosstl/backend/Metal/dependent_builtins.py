"""Result types of Metal math builtins whose arguments depend on an unbound template type parameter.

A function template that no call instantiates, such as::

    template <typename T> inline T erf_approx(T x) { ... return sign(x) * exp(-x * x); }

is imported as a CrossGL generic function (``generic<T> T erf_approx(T x)``) whose body keeps ``T``. Metal resolves
``exp(-x * x)`` only when the template is instantiated (C++ [temp.res] dependent names), and every instantiation
calls the overload of ``exp`` for its own ``T``. The importer used to look for that overload with ``T`` itself as the
argument type, found none and refused the whole program, although no instantiation exists, so library headers with
such helpers blocked every kernel that included them.

:func:`dependent_builtin_result_type` gives the dependent result instead: ``T`` for builtins whose result has their
argument's type (``exp``, ``clamp``, ``normalize``), ``bool`` for reductions (``all``, ``any``), and no type where
the result type is derived from ``T`` in a way CrossGL cannot name (an element type, a bool vector of ``T``'s
shape). Instantiated templates bind ``T`` and resolve their builtins exactly as before.
"""

from __future__ import annotations

import re

__all__ = [
    "NOT_DEPENDENT",
    "dependent_builtin_result_type",
    "unbound_template_type_names",
]

# Returned when the call does not depend on an unbound type parameter.
NOT_DEPENDENT = object()
_NAME = re.compile(r"[A-Za-z_]\w*")
_TYPE_PARAMETER_KINDS = frozenset({"typename", "class"})


def unbound_template_type_names(function, bound_names) -> frozenset:
    """The type parameters of ``function`` that no active binding gives a type."""
    parameters = getattr(function, "template_parameters", None) or []
    return frozenset(
        name
        for kind, name in parameters
        if name and kind in _TYPE_PARAMETER_KINDS and name not in bound_names
    )


def _depends_on(type_name, names) -> bool:
    return bool(names) and bool(set(_NAME.findall(str(type_name or ""))) & names)


def dependent_builtin_result_type(result_kind, argument_types, unbound_names):
    """The result type of a builtin call with dependent arguments, ``None`` when it cannot be named, or
    :data:`NOT_DEPENDENT` when no argument depends on ``unbound_names``."""
    dependent = [t for t in argument_types if _depends_on(t, unbound_names)]
    if not dependent:
        return NOT_DEPENDENT
    if result_kind == "same":
        return dependent[0]
    if result_kind == "bool_reduction":
        return "bool"
    return None

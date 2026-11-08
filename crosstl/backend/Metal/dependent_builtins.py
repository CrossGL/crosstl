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

A member or element of a dependent value, such as ``y.x`` for a parameter ``const U y``, is dependent too
([temp.dep.expr]): its type is only known per instantiation, so the importer infers none. A builtin applied to it
(``::metal::abs(y.x)``) has a dependent result as well, which CrossGL cannot name unless the result is a ``bool``.
:func:`dependent_access` recognises such arguments so the call is not refused as if its type were simply unknown.
"""

from __future__ import annotations

import re

__all__ = [
    "NOT_DEPENDENT",
    "dependent_access",
    "dependent_builtin_result_type",
    "dependent_subscript_selection",
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


_OPERAND_ATTRIBUTES = {
    "BinaryOpNode": ("left", "right"),
    "UnaryOpNode": ("operand",),
    "PostfixOpNode": ("operand",),
    # The condition does not take part in the type of a conditional expression.
    "TernaryOpNode": ("true_expr", "false_expr"),
    "FunctionCallNode": ("args",),
}


def dependent_access(expression, type_of, unbound_names) -> bool:
    """Whether ``expression`` reads a member or element of a value whose type depends on ``unbound_names``,
    directly or through the operands and call arguments it is built from. ``type_of`` gives the inferred type of a
    subexpression, or ``None``. A cast has the type it names, so its operand is not examined.
    """
    if not unbound_names:
        return False
    pending = [expression]
    while pending:
        node = pending.pop()
        kind = type(node).__name__
        if kind in ("MemberAccessNode", "ArrayAccessNode"):
            base = getattr(
                node, "object" if kind == "MemberAccessNode" else "array", None
            )
            if base is None:
                continue
            if _depends_on(type_of(base), unbound_names):
                return True
            pending.append(base)
        elif kind == "CastNode":
            if _depends_on(getattr(node, "target_type", None), unbound_names):
                return True
        else:
            for attribute in _OPERAND_ATTRIBUTES.get(kind, ()):
                value = getattr(node, attribute, None)
                pending.extend(value if isinstance(value, list) else [value])
    return False


def dependent_builtin_result_type(
    result_kind, argument_types, unbound_names, dependent_unknown=False
):
    """The result type of a builtin call with dependent arguments, ``None`` when it cannot be named, or
    :data:`NOT_DEPENDENT` when no argument depends on ``unbound_names``.

    ``dependent_unknown`` says that an argument whose type was not inferred is a :func:`dependent_access`.
    """
    dependent = [t for t in argument_types if _depends_on(t, unbound_names)]
    if not dependent and not dependent_unknown:
        return NOT_DEPENDENT
    if not dependent:
        return "bool" if result_kind == "bool_reduction" else None
    if result_kind == "same":
        return dependent[0]
    if result_kind == "bool_reduction":
        return "bool"
    return None


def dependent_subscript_selection(
    base, base_type, type_of, unbound_names, aggregates=()
):
    """The indexed-type selection for subscripting the dependent value ``base``, or ``None``.

    ``reg[i]`` for a parameter ``thread T& reg`` is type-dependent ([temp.dep.expr]/1): ``T = float4`` selects a vector
    lane, ``T = half4`` converts what is assigned to ``half``, and ``T = float4x4`` selects a column that ``reg[i][j]``
    subscripts again. Overload resolution for the subscript happens per instantiation, so the selection is
    ``dependent`` and selects no type, like a member of a dependent value. The importer emits the subscript unchanged,
    and every target generator types it, and refuses it if the instantiation has no such subscript, once it has
    substituted ``T``.

    ``base_type`` is the value type of ``base`` with references and cv-qualifiers removed, or ``None`` when the
    importer inferred none, which a member or element of a dependent value has (:func:`dependent_access`).

    When one of the program's ``aggregates`` declares ``operator[]``, an instantiation may select it ([over.sub]).
    The importer does not lower user subscript operators, and the generic body would reach a target that cannot
    subscript the aggregate, so the subscript is not reported as dependent and is refused as before.
    """
    if not unbound_names:
        return None
    if any(
        getattr(aggregate, "declares_subscript_operator", False)
        for aggregate in aggregates
    ):
        return None
    if base_type is None:
        if not dependent_access(base, type_of, unbound_names):
            return None
    elif str(base_type).strip() not in unbound_names:
        return None
    return {"kind": "dependent", "base_type": base_type, "selected_type": None}

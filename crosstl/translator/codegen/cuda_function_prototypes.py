"""Prototypes for the functions that CUDA and HIP output call before their definitions.

A CrossGL function may call a function that the program defines after it, as GLSL and HLSL source written with a
prototype first and a CUDA or HIP kernel imported without its prototypes are. C++ looks a called name up among the
declarations before the call ([basic.lookup.unqual]), so the CUDA and HIP generators, which write functions in program
order, wrote a call to a name nvcc and hipcc had not seen yet and that they rejected as undefined.

:func:`late_called_functions` names the definitions that a function written before them calls, and the generators
write a prototype of each one, with the qualifiers and parameters of its definition, before the first definition.
A function that calls only itself needs none, because a function's name is declared from its declarator on
([basic.scope.pdecl]). The generators write each specialization of a generic function, ``twice_float`` for
``generic<T> T twice(T x)``, where the generic is defined, and a call spells it by the generic's name, with type
arguments (``twice<float>(x)``) or without them, so a specialization is matched by the generic's name.
"""

from __future__ import annotations

import copy

from .. import ast as A
from ..semantics.hub_types import call_name
from ..semantics.rewrite import child_slots, get_child

__all__ = ["emission_order", "late_called_functions", "prototype_of"]


def _nodes(root):
    pending = list(root) if isinstance(root, (list, tuple)) else [root]
    while pending:
        node = pending.pop()
        if isinstance(node, (list, tuple)):
            pending.extend(node)
            continue
        if not isinstance(node, A.ASTNode):
            continue
        yield node
        pending.extend(get_child(node, field, key) for field, key in child_slots(node))


def _called_names(function):
    """The names ``function`` calls, without explicit type arguments: ``twice<float>(x)`` calls ``twice``."""
    return {
        name.split("<", 1)[0].strip()
        for node in _nodes(getattr(function, "body", None))
        if isinstance(node, A.FunctionCallNode)
        for name in (call_name(node),)
        if name
    }


def late_called_functions(functions, call_names=None):
    """The definitions in ``functions``, a list in the order the generator writes it, that an earlier one calls.

    Every overload of a called name that is defined later is named, since the call may resolve to any of them; a
    second declaration of a function is valid C++ ([dcl.fct.default] keeps it so only for default arguments, which
    these generators never write). ``call_names`` maps ``id`` of a definition to the name calls spell it by when
    that is not its own, as for the specializations of a generic function; every specialization of a called generic
    defined later is named.
    """
    functions = [function for function in functions or [] if function is not None]
    call_names = call_names or {}
    definitions = {}
    for index, function in enumerate(functions):
        if getattr(function, "body", None) is not None:
            names = {getattr(function, "name", None), call_names.get(id(function))}
            for name in names - {None}:
                definitions.setdefault(name, []).append(index)
    late = set()
    for index, function in enumerate(functions):
        if getattr(function, "body", None) is None:
            continue
        for called in _called_names(function):
            late.update(
                target for target in definitions.get(called, ()) if target > index
            )
    return [functions[index] for index in sorted(late)]


def prototype_of(function):
    """A body-less copy of ``function`` for the generator to write as its prototype."""
    prototype = copy.copy(function)
    prototype.body = None
    return prototype


def emission_order(functions, is_generic, specializations):
    """The definitions a generator writes for ``functions``, in order, and the call name of each specialization.

    ``specializations(function)`` lists the specializations written in place of a generic ``function``; the pair
    returned is the argument list for :func:`late_called_functions`.
    """
    order, call_names = [], {}
    for function in functions or []:
        if function is None:
            continue
        if not is_generic(function):
            order.append(function)
            continue
        for specialization in specializations(function):
            order.append(specialization)
            call_names[id(specialization)] = getattr(function, "name", None)
    return order, call_names

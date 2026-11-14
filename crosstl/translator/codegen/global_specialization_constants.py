"""Module-scope variables that are specialization constants, or follow one, are planned as constants.

CrossGL spells a specialization constant two ways. ``const T name @constant_id(N) = value;`` is a
:class:`~crosstl.translator.ast.ConstantNode` in ``ast.constants``; ``constant T name @function_constant(N);``, which
is how the Metal importer writes an MSL function constant, is a module-scope
:class:`~crosstl.translator.ast.VariableNode` in ``ast.global_variables``. So is a program-scope MSL constant computed
from one, such as ``constant int span = tile * stride + 1;``. The GLSL, HLSL, Metal and SPIR-V generators read both
spellings. Generators that plan specialization constants from ``ast.constants`` alone printed the variables as
ordinary globals, which dropped the id: Slang emitted ``static const bool name;`` (rejected by slangc, "static const
variable must have an initializer"), WGSL a ``var<private>`` that every invocation read as zero, and a constant
computed from either froze or failed to compile.

:func:`split_global_specialization_constants` gives such a generator a ``ConstantNode`` view of each of those
variables together with the module-scope variables that remain. A view keeps the variable's name, type, initializer
(``None`` when the host must supply the value), attributes and source location. The specialization-id views come
first, then the program's constants, then the views of the ``const``/``constant`` variables whose initializer reads a
specialization constant, directly or through another constant, so every declaration follows what it reads.
"""

from __future__ import annotations

from ..ast import ConstantNode
from .specialization_constants import declares_specialization_id, identifier_names

__all__ = [
    "specialization_constant_declarations",
    "split_global_specialization_constants",
]

_CONSTANT_QUALIFIERS = frozenset({"const", "constant"})


def _named(node) -> bool:
    return isinstance(getattr(node, "name", None), str)


def _is_constant_variable(node) -> bool:
    qualifiers = {
        str(value).lower() for value in getattr(node, "qualifiers", None) or []
    }
    return bool(qualifiers & _CONSTANT_QUALIFIERS) and (
        getattr(node, "initial_value", None) is not None
    )


def _view(node) -> ConstantNode:
    view = ConstantNode(
        node.name,
        getattr(node, "var_type", getattr(node, "vtype", None)),
        getattr(node, "initial_value", None),
        attributes=list(getattr(node, "attributes", None) or []),
    )
    view.source_location = getattr(node, "source_location", None)
    return view


def _planned_globals(ast) -> tuple[list, list, set[int]]:
    """The specialization-id globals, the dependent constant globals, and the ids of both groups' nodes."""
    global_variables = [
        node for node in getattr(ast, "global_variables", None) or [] if _named(node)
    ]
    specialized = [
        node for node in global_variables if declares_specialization_id(node)
    ]
    dependent_names = {node.name for node in specialized}
    for node in getattr(ast, "constants", None) or []:
        if _named(node) and (
            declares_specialization_id(node)
            or identifier_names(getattr(node, "value", None)) & dependent_names
        ):
            dependent_names.add(node.name)
    dependent = []
    for node in global_variables:
        if declares_specialization_id(node) or not _is_constant_variable(node):
            continue
        if identifier_names(node.initial_value) & dependent_names:
            dependent.append(node)
            dependent_names.add(node.name)
    return specialized, dependent, {id(node) for node in (*specialized, *dependent)}


def specialization_constant_declarations(ast) -> list:
    """Every constant of ``ast``, with the views of its planned module-scope variables, in declaration order."""
    specialized, dependent, _ = _planned_globals(ast)
    return [
        *(_view(node) for node in specialized),
        *(getattr(ast, "constants", None) or []),
        *(_view(node) for node in dependent),
    ]


def split_global_specialization_constants(ast) -> tuple[list, list]:
    """``(constants, global_variables)`` of ``ast`` with the planned module-scope variables moved to the constants."""
    specialized, dependent, planned = _planned_globals(ast)
    constants = [
        *(_view(node) for node in specialized),
        *(getattr(ast, "constants", None) or []),
        *(_view(node) for node in dependent),
    ]
    global_variables = [
        node
        for node in getattr(ast, "global_variables", None) or []
        if id(node) not in planned
    ]
    return constants, global_variables

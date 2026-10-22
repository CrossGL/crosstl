"""Interface attributes that name an integer constant or enumerator, folded to the integer they name.

A Metal shader names its buffer, texture and vertex-attribute slots with the enumerators of the header it shares
with its host (``[[buffer(AAPLVertexInputIndexVertices)]]``, ``[[attribute(AAPLVertexAttributePosition)]]``, where
``typedef enum { AAPLVertexInputIndexVertices = 0, ... }``), and the CrossGL import keeps the names
(``@ buffer(AAPLVertexInputIndexVertices)``). HLSL and CrossGL sources can name a ``const`` integer the same way. Each
generator needs the slot's integer: GLSL's ``layout(binding = N)`` and SPIR-V's ``Binding`` and ``Location``
decorations take one, and the GLSL and SPIR-V generators read only literal digits, refusing every named slot
("must resolve to a concrete integer binding", "SPIR-V buffer must be an integer").

:func:`fold_binding_constants` replaces each such name in a slot attribute (``binding``, ``set``, ``buffer``,
``texture``, ``sampler``, ``location``, ``attribute``, ``color``, ``index``, ``id``, ``input_attachment_index``)
with the integer literal it names. The name may be a program constant that folds to an integer, or an enumerator
of a plain enum, which counts on from the previous enumerator when it has no value of its own, as in C. A scoped
enumerator may be written ``Enum::Name``. Names that fold to no integer are left for the generators to report.
"""

from __future__ import annotations

from .. import ast as A
from ..codegen.array_utils import evaluate_literal_int_expression

__all__ = ["fold_binding_constants"]

_SLOT_ATTRIBUTES = frozenset(
    {
        "binding",
        "set",
        "buffer",
        "texture",
        "sampler",
        "location",
        "attribute",
        "color",
        "index",
        "id",
        "input_attachment_index",
    }
)


def _stages(program):
    stages = getattr(program, "stages", None) or {}
    return list(stages.values() if isinstance(stages, dict) else stages)


def _integer_names(program):
    """Program constants and enumerators that fold to an integer, by name (and ``Enum::Name``)."""
    values = {}
    for constant in getattr(program, "constants", None) or []:
        value = getattr(constant, "value", None)
        if value is None:
            value = getattr(constant, "initial_value", None)
        folded = (
            evaluate_literal_int_expression(value, values)
            if value is not None
            else None
        )
        if isinstance(folded, int) and not isinstance(folded, bool):
            values[constant.name] = folded
    for enum in getattr(program, "structs", None) or []:
        if not isinstance(enum, A.EnumNode):
            continue
        next_value = 0
        for variant in getattr(enum, "variants", None) or []:
            if getattr(variant, "fields", None) or getattr(variant, "payload", None):
                break
            value = getattr(variant, "value", None)
            if value is not None:
                next_value = evaluate_literal_int_expression(value, values)
            if next_value is None:
                break
            values.setdefault(variant.name, next_value)
            values[f"{enum.name}::{variant.name}"] = next_value
            next_value += 1
    return values


def _attributed_nodes(program):
    yield from getattr(program, "global_variables", None) or []
    yield from getattr(program, "cbuffers", None) or []
    for struct in getattr(program, "structs", None) or []:
        yield from getattr(struct, "members", None) or []
    functions = list(getattr(program, "functions", None) or [])
    for stage in _stages(program):
        yield from getattr(stage, "local_variables", None) or []
        yield from getattr(stage, "local_cbuffers", None) or []
        functions.extend(getattr(stage, "local_functions", None) or [])
        if getattr(stage, "entry_point", None) is not None:
            functions.append(stage.entry_point)
    for function in functions:
        yield function
        yield from getattr(function, "parameters", None) or []


def _name(argument):
    if isinstance(argument, A.IdentifierNode):
        return argument.name
    if isinstance(argument, str):
        return argument
    return None


def fold_binding_constants(program) -> int:
    """Replace named integers in ``program``'s slot attributes with literals, in place; returns how many."""
    if not isinstance(program, A.ShaderNode):
        return 0
    values = None
    folded = 0
    for node in _attributed_nodes(program):
        for attribute in getattr(node, "attributes", None) or []:
            if str(getattr(attribute, "name", "")).lower() not in _SLOT_ATTRIBUTES:
                continue
            arguments = getattr(attribute, "arguments", None)
            if not arguments:
                continue
            for position, argument in enumerate(arguments):
                name = _name(argument)
                if name is None:
                    continue
                if values is None:
                    values = _integer_names(program)
                value = values.get(name)
                if value is None:
                    continue
                arguments[position] = A.LiteralNode(value, A.PrimitiveType("int"))
                folded += 1
    return folded

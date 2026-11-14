"""Structural traversal, replacement and cloning of CrossGL AST nodes for the rule passes.

AST nodes keep legacy alias attributes (``AssignmentNode.left`` is ``target``, ``IfNode.if_body`` is
``then_branch``, ``FunctionCallNode.args`` is ``arguments``). A child is therefore visited once per distinct object,
and a replacement updates every attribute that referred to the old object, so emitters that read either spelling
see the same tree.
"""

from __future__ import annotations

import copy
from typing import NamedTuple

from .. import ast as A
from .hub_types import call_name, is_pure_callee


class StageSlot(NamedTuple):
    """The slot of one stage in a :class:`~crosstl.translator.ast.StageMap`.

    A CrossGL module may declare several blocks of one stage kind (two ``vertex`` blocks, each with its own entry
    point), and the stage map keeps every one under the same key. The key alone names only the first, so a slot of
    a stage map carries the block's position among the blocks of its kind too.
    """

    stage: object
    index: int


# Attributes that hold types, metadata or back links rather than structural children.
SKIPPED_FIELDS = frozenset(
    {
        "parent",
        "annotations",
        "source_location",
        "semantic",
        "name",
        "expression_type",
        "vtype",
        "var_type",
        "param_type",
        "return_type",
        "const_type",
        "member_type",
        "literal_type",
        "target_type",
        "constructor_type",
        "binding_type",
        "generic_args",
        "generic_params",
        "attributes",
        "qualifiers",
        "resource_qualifiers",
        "layout_qualifiers",
        "execution_config",
    }
)

# The field kinds that hold replaceable children: tuples, sets and plain values hold none.
_SLOT_KINDS = frozenset({A.FIELD_NODE, A.FIELD_LIST, A.FIELD_STAGES, A.FIELD_DICT})


def child_slots(node):
    """Yield ``(field, key)`` once per distinct structural child; ``key`` is ``None`` for a direct attribute."""
    seen = set()
    kinds = A.FIELD_KINDS
    node_type = A.ASTNode
    for field, value in list(vars(node).items()):
        if field in SKIPPED_FIELDS:
            continue
        kind = kinds.get(type(value))
        if kind is None:
            kind = A.field_kind(value)
        if kind not in _SLOT_KINDS:
            continue
        value_id = id(value)
        if value_id in seen:
            continue
        seen.add(value_id)
        if kind == A.FIELD_NODE:
            yield field, None
        elif kind == A.FIELD_LIST:
            for index, item in enumerate(value):
                if isinstance(item, node_type):
                    yield field, index
        elif kind == A.FIELD_STAGES:
            for stage in list(value.keys()):
                for index, item in enumerate(value.all_for_stage(stage)):
                    if isinstance(item, node_type):
                        yield field, StageSlot(stage, index)
        else:
            for key, item in list(value.items()):
                if isinstance(item, node_type):
                    yield field, key


def get_child(node, field, key):
    value = getattr(node, field)
    if isinstance(key, StageSlot):
        return value.all_for_stage(key.stage)[key.index]
    return value if key is None else value[key]


def field_slots(node, field):
    """``(field, key)`` for each node held in ``node.field``: one slot for a node, one per item for a list.

    A ``for`` header's update is a list when it has several comma-separated expressions (``++i, ++j``).
    """
    value = getattr(node, field, None)
    if isinstance(value, list):
        return [(field, index) for index, item in enumerate(value) if item is not None]
    return [] if value is None else [(field, None)]


def set_child(node, field, key, new):
    """Replace a child, updating every alias attribute that referred to the same object."""
    if isinstance(key, StageSlot):
        getattr(node, field).replace_stage(key.stage, key.index, new)
        return
    if key is not None:
        getattr(node, field)[key] = new
        return
    old = getattr(node, field)
    for name, value in list(vars(node).items()):
        if value is old and name not in ("parent", "annotations", "source_location"):
            setattr(node, name, new)


def clone(node):
    """Copy a structural subtree; types and metadata are shared, back links are cleared."""
    duplicate = copy.copy(node)
    duplicate.annotations = dict(getattr(node, "annotations", None) or {})
    duplicate.parent = None
    copies = {}
    for field, value in list(vars(node).items()):
        if field in SKIPPED_FIELDS:
            continue
        if id(value) in copies:
            setattr(duplicate, field, copies[id(value)])
        elif isinstance(value, A.ASTNode):
            copies[id(value)] = clone(value)
            setattr(duplicate, field, copies[id(value)])
        elif isinstance(value, list) and any(isinstance(v, A.ASTNode) for v in value):
            copies[id(value)] = [
                clone(v) if isinstance(v, A.ASTNode) else v for v in value
            ]
            setattr(duplicate, field, copies[id(value)])
    if isinstance(duplicate, A.FunctionCallNode):
        duplicate.name = duplicate.function
        duplicate.args = duplicate.arguments
    return duplicate


def call(name, arguments):
    node = A.FunctionCallNode(A.IdentifierNode(name), list(arguments))
    return node


def is_pure(expr) -> bool:
    """True when evaluating ``expr`` twice is equivalent to evaluating it once."""
    if isinstance(expr, (A.IdentifierNode, A.LiteralNode)):
        return True
    if isinstance(expr, A.MemberAccessNode):
        return is_pure(expr.object_expr)
    if isinstance(expr, A.SwizzleNode):
        return is_pure(expr.vector_expr)
    if isinstance(expr, A.ArrayAccessNode):
        return is_pure(expr.array_expr) and is_pure(expr.index_expr)
    if isinstance(expr, A.BinaryOpNode):
        return is_pure(expr.left) and is_pure(expr.right)
    if isinstance(expr, A.UnaryOpNode):
        return expr.op not in ("++", "--") and is_pure(expr.operand)
    if isinstance(expr, A.CastNode):
        return is_pure(expr.expression)
    if isinstance(expr, A.FunctionCallNode):
        return is_pure_callee(call_name(expr)) and all(
            is_pure(argument) for argument in expr.arguments or []
        )
    return False

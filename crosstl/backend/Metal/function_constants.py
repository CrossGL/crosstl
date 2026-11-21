"""Metal function constants with a default value, imported as CrossGL specialization constants.

A program-scope ``constant T name [[function_constant(index)]];`` takes its value from the function constants the
host sets when it builds the pipeline, and ``is_function_constant_defined(name)`` says whether the host set one
(Metal Shading Language Specification, section 5.8.1, "Specifying a Function Constant"). A Metal function constant
has no default value of its own, so a shader that needs one declares a second constant:

    constant int scale_value [[function_constant(0)]];
    constant int scale = is_function_constant_defined(scale_value) ? scale_value : 3;

That pair is CrossGL's ``const int scale @constant_id(0) = 3;`` (GLSL ``layout(constant_id = 0)``, SPIR-V
``OpSpecConstant``): the host's value for constant 0 when it sets one, else 3. It is also what Metal output writes
for that CrossGL declaration.

:func:`import_function_constant_defaults` rewrites each such pair in place: the defaulted constant becomes a
``const`` declaration with a ``constant_id`` attribute and the default as its initializer, and the raw function
constant is dropped. It rewrites a pair only when the raw constant is read nowhere else, since CrossGL has no value
for a specialization constant the host left unset. Every other use of ``is_function_constant_defined`` is left for
the undeclared-function check to refuse.
"""

from __future__ import annotations

import re

from ..common_ast import (
    AssignmentNode,
    AttributeNode,
    FunctionCallNode,
    TernaryOpNode,
    VariableNode,
)

__all__ = ["import_function_constant_defaults"]

_DEFINED_QUERY = "is_function_constant_defined"
_SKIPPED_FIELDS = frozenset({"parent", "annotations", "source_location"})
_INDEX = re.compile(r"[0-9]+")


def _walk(root):
    """Yield every node reachable from ``root`` once."""
    seen = set()
    stack = [root]
    while stack:
        node = stack.pop()
        if isinstance(node, (list, tuple)):
            stack.extend(reversed(node))
            continue
        if not hasattr(node, "__dict__") or isinstance(node, type) or id(node) in seen:
            continue
        seen.add(id(node))
        yield node
        stack.extend(
            value
            for key, value in reversed(list(vars(node).items()))
            if key not in _SKIPPED_FIELDS
        )


def _qualifiers(node):
    return [str(qualifier).strip("'\"") for qualifier in node.qualifiers or []]


def _function_constant_index(variable):
    """The index of ``variable``'s ``[[function_constant(index)]]``, or None."""
    if not isinstance(variable, VariableNode) or getattr(variable, "namespace", ""):
        return None
    if _qualifiers(variable) != ["constant"]:
        return None
    attributes = list(variable.attributes or [])
    if len(attributes) != 1 or not isinstance(attributes[0], AttributeNode):
        return None
    attribute = attributes[0]
    arguments = [str(argument).strip("'\"") for argument in attribute.args or []]
    if attribute.name != "function_constant" or len(arguments) != 1:
        return None
    return arguments[0] if _INDEX.fullmatch(arguments[0]) else None


def _reads(node, name):
    return isinstance(node, VariableNode) and node.name == name and not node.vtype


def _defaulted(declaration, raw):
    """The default of ``declaration`` when it is ``constant T x = is_function_constant_defined(raw) ? raw : D``."""
    if not isinstance(declaration, AssignmentNode) or declaration.operator != "=":
        return None
    target, value = declaration.left, declaration.right
    if not isinstance(target, VariableNode) or not isinstance(value, TernaryOpNode):
        return None
    if _qualifiers(target) != ["constant"] or target.attributes:
        return None
    if getattr(target, "namespace", "") or str(target.vtype) != str(raw.vtype):
        return None
    condition = value.condition
    if not (
        isinstance(condition, FunctionCallNode)
        and condition.name == _DEFINED_QUERY
        and len(condition.args or []) == 1
        and _reads(condition.args[0], raw.name)
        and _reads(value.true_expr, raw.name)
    ):
        return None
    return value.false_expr


def import_function_constant_defaults(ast):
    """Rewrite each defaulted Metal function constant of ``ast`` as a CrossGL specialization constant, in place."""
    globals_ = getattr(ast, "global_variables", None)
    if not globals_:
        return ast
    raw_constants = {}
    for declaration in globals_:
        index = _function_constant_index(declaration)
        if index is not None:
            raw_constants.setdefault(declaration.name, (declaration, index))
    if not raw_constants:
        return ast
    pairs = []
    for declaration in globals_:
        for raw, index in raw_constants.values():
            default = _defaulted(declaration, raw)
            if default is not None:
                pairs.append((raw, index, declaration, default))
    if not pairs:
        return ast
    read_counts = {name: 0 for name in raw_constants}
    for node in _walk(ast):
        if isinstance(node, VariableNode) and node.name in read_counts:
            if not any(node is raw for raw, _index in raw_constants.values()):
                read_counts[node.name] += 1
    removed = set()
    for raw, index, declaration, default in pairs:
        # The query and the conditional's true branch are the two reads the pair itself makes.
        if read_counts[raw.name] != 2 or id(raw) in removed:
            continue
        target = declaration.left
        target.qualifiers = []
        target.is_const = True
        target.attributes = [AttributeNode("constant_id", [index])]
        declaration.right = default
        removed.add(id(raw))
    if removed:
        globals_[:] = [
            declaration for declaration in globals_ if id(declaration) not in removed
        ]
    return ast

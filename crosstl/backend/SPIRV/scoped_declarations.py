"""Hoist imported locals whose uses lie outside the block that declares them.

The SPIR-V importer declares each materialized value where its defining instruction sits, which after structured
control flow reconstruction can be inside a loop body, an ``if`` arm or a switch case. SSA dominance lets the value
be used anywhere its definition dominates, for example after the loop whose header computed it, or in the
conditional expression a selection phi becomes at the merge. :func:`hoist_out_of_scope_declarations` moves the
declaration of every such local to the top of the function, leaving an assignment in its place; a declaration
whose uses all follow it in its own block, or in blocks nested there, stays where it is.
"""

from __future__ import annotations

from .VulkanAst import (
    AssignmentNode,
    CaseNode,
    DoWhileNode,
    ForNode,
    IfNode,
    SwitchNode,
    VariableNode,
    WhileNode,
)

__all__ = ["hoist_out_of_scope_declarations"]


def _base_name(name):
    return str(name).split("[", 1)[0]


def _declaration(statement):
    """The ``VariableNode`` a statement declares, if it is a declaration."""
    if isinstance(statement, VariableNode) and statement.vtype:
        return statement
    if (
        isinstance(statement, AssignmentNode)
        and isinstance(statement.left, VariableNode)
        and statement.left.vtype
        and getattr(statement, "operator", "=") == "="
    ):
        return statement.left
    return None


def _scopes(statement):
    """The statement lists nested directly in a compound statement, each a scope of its own."""
    if isinstance(statement, IfNode):
        lists = [statement.if_body, statement.else_body]
        lists.extend(
            body for _condition, body in getattr(statement, "else_if_chain", []) or []
        )
        return [body for body in lists if isinstance(body, list)]
    if isinstance(statement, (WhileNode, DoWhileNode, ForNode)):
        return [statement.body] if isinstance(statement.body, list) else []
    if isinstance(statement, SwitchNode):
        bodies = [case.body for case in statement.cases if isinstance(case, CaseNode)]
        default = getattr(statement, "default_case", None)
        if isinstance(default, list):
            bodies.append(default)
        return [body for body in bodies if isinstance(body, list)]
    return []


def _expression_parts(statement):
    """The parts of a statement that are evaluated in the scope enclosing it."""
    if isinstance(statement, IfNode):
        parts = [statement.condition]
        parts.extend(
            condition
            for condition, _body in getattr(statement, "else_if_chain", []) or []
        )
        return parts
    if isinstance(statement, (WhileNode, DoWhileNode)):
        return [statement.condition]
    if isinstance(statement, ForNode):
        return [statement.init, statement.condition, statement.update]
    if isinstance(statement, SwitchNode):
        return [statement.expression]
    declared = _declaration(statement)
    if declared is not None and isinstance(statement, AssignmentNode):
        return [statement.right]
    if declared is not None:
        return []
    return [statement]


def _references(node, found):
    stack = [node]
    seen = set()
    while stack:
        current = stack.pop()
        if current is None or isinstance(current, (str, int, float, bool)):
            continue
        if isinstance(current, (list, tuple)):
            stack.extend(current)
            continue
        if id(current) in seen:
            continue
        seen.add(id(current))
        if isinstance(current, VariableNode) and not current.vtype:
            found.add(_base_name(current.name))
        attributes = getattr(current, "__dict__", None) or {}
        for value in attributes.values():
            if isinstance(value, (list, tuple)) or hasattr(value, "__dict__"):
                stack.append(value)


class _Index:
    def __init__(self):
        self.declarations = {}
        self.references = {}

    def walk(self, statements, path):
        for index, statement in enumerate(statements):
            here = (*path, (id(statements), index))
            declared = _declaration(statement)
            if declared is not None:
                self.declarations.setdefault(_base_name(declared.name), []).append(
                    (here, declared)
                )
            names = set()
            for part in _expression_parts(statement):
                _references(part, names)
            for name in names:
                self.references.setdefault(name, []).append(here)
            for scope in _scopes(statement):
                self.walk(scope, here)


def _in_scope(declaration_path, reference_path):
    depth = len(declaration_path) - 1
    if len(reference_path) <= depth:
        return False
    if reference_path[:depth] != declaration_path[:depth]:
        return False
    scope, index = declaration_path[depth]
    reference_scope, reference_index = reference_path[depth]
    return reference_scope == scope and reference_index > index


def _hoisted_names(index):
    hoisted = {}
    for name, declarations in index.declarations.items():
        nested = [entry for entry in declarations if len(entry[0]) > 1]
        if not nested or len(nested) != len(declarations):
            # A function-level declaration of the same name is a different local; leave both alone.
            continue
        for reference in index.references.get(name, []):
            if not any(_in_scope(path, reference) for path, _declared in declarations):
                hoisted[name] = nested[0][1]
                break
    return hoisted


def _rewrite(statements, hoisted, top_level):
    rewritten = []
    for statement in statements:
        declared = _declaration(statement)
        if (
            declared is not None
            and not top_level
            and _base_name(declared.name) in hoisted
        ):
            if isinstance(statement, AssignmentNode):
                reference = VariableNode(
                    "",
                    _base_name(declared.name),
                    spirv_id=getattr(declared, "spirv_id", None),
                )
                rewritten.append(AssignmentNode(reference, statement.right))
            continue
        for scope in _scopes(statement):
            scope[:] = _rewrite(scope, hoisted, False)
        rewritten.append(statement)
    return rewritten


def hoist_out_of_scope_declarations(body):
    """``body`` with every nested declaration that is used outside its scope declared at the top instead."""
    index = _Index()
    index.walk(body, ())
    hoisted = _hoisted_names(index)
    if not hoisted:
        return body
    declarations = []
    for name in sorted(hoisted, key=lambda key: str(key)):
        declared = hoisted[name]
        declarations.append(
            VariableNode(
                declared.vtype,
                declared.name,
                spirv_id=getattr(declared, "spirv_id", None),
                spirv_type_id=getattr(declared, "spirv_type_id", None),
            )
        )
    return [*declarations, *_rewrite(list(body), hoisted, True)]

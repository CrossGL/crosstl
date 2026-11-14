"""Resolve the identifiers of a function body to the local declarations they name, following C block scopes.

A name-keyed analysis conflates distinct locals that share a name, such as the ``i`` of two sibling ``for`` loops,
so a fact about one (it differs between invocations) is wrongly applied to the other. :func:`local_bindings` maps
each identifier that names a local ``VariableNode`` to that declaration. Scopes follow GLSL (GLSL 4.60 section
4.2.2), whose block scoping CrossGL shares: a compound statement, a ``for`` statement together with its init
declaration, a ``switch`` body, and each substatement of ``if``, ``while``, ``do`` and ``loop`` statements open one,
and a declared name's scope starts after its initializer. Identifiers that name no local (parameters, globals,
built-ins) are absent from the map.
"""

from __future__ import annotations

from .. import ast as A
from .rewrite import child_slots, get_child

__all__ = ["local_bindings"]

# Statements whose children share one scope, which the statement opens.
_SCOPE_STATEMENTS = (A.BlockNode, A.ForNode, A.SwitchNode)
# Statements each of whose children (condition, bodies) is a scope of its own.
_SUBSTATEMENT_SCOPES = (A.IfNode, A.WhileNode, A.DoWhileNode, A.LoopNode)


def local_bindings(body):
    """Map ``id(identifier)`` to ``(identifier, declaration)`` for each identifier of ``body`` naming a local."""
    bindings = {}
    scopes = [{}]

    def lookup(name):
        for scope in reversed(scopes):
            if name in scope:
                return scope[name]
        return None

    def visit_children(node, each_scoped):
        for field, key in child_slots(node):
            if each_scoped:
                scopes.append({})
            visit(get_child(node, field, key))
            if each_scoped:
                scopes.pop()

    def visit(node):
        if isinstance(node, list):
            for item in node:
                visit(item)
            return
        if not isinstance(node, A.ASTNode):
            return
        if isinstance(node, A.IdentifierNode):
            declaration = lookup(node.name)
            if declaration is not None:
                bindings[id(node)] = (node, declaration)
            return
        if isinstance(node, A.VariableNode):
            visit_children(node, False)
            scopes[-1][node.name] = node
            return
        if isinstance(node, _SCOPE_STATEMENTS):
            scopes.append({})
            visit_children(node, False)
            scopes.pop()
            return
        visit_children(node, isinstance(node, _SUBSTATEMENT_SCOPES))

    visit(body)
    return bindings

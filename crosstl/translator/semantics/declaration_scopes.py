"""GLSL declaration scopes on targets with C++ scoping.

The scope of a GLSL local starts after its initializer (GLSL 4.60 section 4.2.2, "Scoping": it "begins immediately
after the initializer if present"). In ``float scale = scale * 3.0;`` inside a nested block, the initializer
therefore reads the outer ``scale``. CrossGL keeps that rule. In C++, and so in HLSL, MSL, CUDA, HIP and Slang, the
scope starts at the declarator (C++17 [basic.scope.pdecl]). The same text then reads the new, uninitialized
variable. DXC rejects it in the best case ("Instructions should not read uninitialized value"), and the Metal compiler
accepts it silently.

For those targets, :func:`separate_initializer_scopes` renames each local whose initializer names the variable it
declares to a fresh name, together with every later use of the declaration in its scope. Programs imported from C++
sources never take this path in practice; the pass applies only to GLSL and CrossGL sources.
"""

from __future__ import annotations

from .. import ast as A
from .hub_types import iter_functions

__all__ = ["CXX_SCOPED_TARGETS", "separate_initializer_scopes"]

CXX_SCOPED_TARGETS = frozenset(
    {
        "directx",
        "hlsl",
        "dx11",
        "dx12",
        "d3d11",
        "d3d12",
        "metal",
        "cuda",
        "hip",
        "slang",
    }
)
_GLSL_SCOPED_SOURCES = frozenset({"opengl", "glsl", "cgl", "crossgl", ""})
_SKIPPED = frozenset({"parent", "annotations", "source_location", "expression_type"})
_CALLEE_FIELDS = frozenset({"function", "name"})


def _children(node):
    for field, value in vars(node).items():
        if field in _SKIPPED:
            continue
        if isinstance(node, A.FunctionCallNode) and field in _CALLEE_FIELDS:
            continue
        yield value


def _names(node):
    """Identifier names read under ``node``."""
    found = set()
    stack = [node]
    while stack:
        item = stack.pop()
        if isinstance(item, list):
            stack.extend(item)
        elif isinstance(item, A.StageMap):
            stack.extend(item.values())
        elif isinstance(item, A.IdentifierNode) and isinstance(item.name, str):
            found.add(item.name)
        elif isinstance(item, A.ASTNode):
            stack.extend(_children(item))
    return found


def _declared(program):
    names = set()
    stack = [program]
    while stack:
        item = stack.pop()
        if isinstance(item, list):
            stack.extend(item)
        elif isinstance(item, A.StageMap):
            stack.extend(item.values())
        elif isinstance(item, A.ASTNode):
            if isinstance(item, (A.VariableNode, A.ParameterNode)) and isinstance(
                item.name, str
            ):
                names.add(item.name)
            stack.extend(_children(item))
    return names


class _Renamer:
    def __init__(self, taken):
        self.taken = taken
        self.count = 0

    def fresh(self, name):
        index = 1
        while f"{name}_{index}" in self.taken:
            index += 1
        fresh = f"{name}_{index}"
        self.taken.add(fresh)
        return fresh

    def statements(self, items, renames):
        """Walk ``items`` in order, in a scope of their own; ``renames`` maps visible names to new names."""
        renames = dict(renames)
        for item in items:
            if isinstance(item, A.VariableNode):
                self.declaration(item, renames)
            else:
                self.visit(item, renames)

    def declaration(self, node, renames):
        """``node`` declares a name in the scope ``renames`` describes, which it updates."""
        value = node.initial_value
        hazard = value is not None and node.name in _names(value)
        self.visit(value, renames)
        self.visit(getattr(node, "var_type", None), renames)
        if hazard:
            renames[node.name] = node.name = self.fresh(node.name)
            self.count += 1
        else:
            renames.pop(node.name, None)

    def visit(self, node, renames):
        if isinstance(node, list):
            for item in node:
                self.visit(item, renames)
            return
        if not isinstance(node, A.ASTNode):
            return
        if isinstance(node, A.IdentifierNode):
            if node.name in renames:
                node.name = renames[node.name]
            return
        if isinstance(node, A.BlockNode):
            self.statements(node.statements or [], renames)
            return
        if isinstance(node, A.VariableNode):
            # A declaration in a position of its own (a for-init, a condition) scopes to that statement.
            self.declaration(node, dict(renames))
            return
        if isinstance(node, A.ForNode):
            scope = dict(renames)
            if isinstance(node.init, A.VariableNode):
                self.declaration(node.init, scope)
            else:
                self.visit(node.init, scope)
            for field in ("condition", "update", "body"):
                self.visit(getattr(node, field, None), scope)
            return
        for child in _children(node):
            if isinstance(child, list) and any(
                isinstance(c, A.VariableNode) for c in child
            ):
                self.statements(child, renames)
            else:
                self.visit(child, renames)


def separate_initializer_scopes(program, source, target) -> int:
    """Rename locals whose initializer reads an outer variable of the same name; returns how many."""
    if str(target or "").lower() not in CXX_SCOPED_TARGETS:
        return 0
    if str(source or "").lower() not in _GLSL_SCOPED_SOURCES:
        return 0
    if not isinstance(program, A.ShaderNode):
        return 0
    renamer = _Renamer(_names(program) | _declared(program))
    for function in iter_functions(program):
        body = getattr(function, "body", None)
        if isinstance(body, A.BlockNode):
            renamer.statements(body.statements or [], {})
        elif isinstance(body, list):
            renamer.statements(body, {})
    return renamer.count

"""Members of anonymous buffer blocks as named-instance member accesses.

A GLSL buffer block declared without an instance name puts its members in the enclosing scope (GLSL 4.60 section
4.3.9): after ``buffer Data { uint count; float data[]; };`` the program reads ``count`` and ``data[i]``. CrossGL
gives every block an instance, so the parser names one and this pass rewrites each free reference to a member
``m`` as ``instance.m``. That is the form the GLSL importer produces for the same program, so every emitter sees
one representation.

A reference is free when no enclosing declaration in the function shadows it. Scopes follow GLSL: parameters and
locals are visible from the end of their declarator to the end of their block (section 4.2.2), so
``float data = data[0];`` reads the block member in its initializer.
"""

from __future__ import annotations

from .ast import (
    AST_CHILD_FIELD_EXCLUSIONS,
    ASTNode,
    AttributeNode,
    FunctionCallNode,
    FunctionNode,
    IdentifierNode,
    MemberAccessNode,
    ParameterNode,
    StatementNode,
    StructNode,
    VariableNode,
)

__all__ = ["AnonymousBufferBlock", "bind_anonymous_buffer_block_members"]


class AnonymousBufferBlock:
    """A parsed anonymous block: its instance variable and its member names."""

    def __init__(self, variable: VariableNode, member_names):
        self.variable = variable
        self.member_names = tuple(member_names)


def bind_anonymous_buffer_block_members(shader, blocks):
    """Name each anonymous block's instance uniquely and rewrite free member references in ``shader``."""
    if not blocks:
        return shader
    instances = {id(block.variable) for block in blocks}
    used_names = _declared_names(shader, instances)
    owners = {}
    for block in blocks:
        for member in block.member_names:
            owners[member] = block
    for block in blocks:
        block.variable.name = _unique_name(block.variable.name, used_names, owners)
        used_names.add(block.variable.name)
    _Binder({member: block.variable.name for member, block in owners.items()}).visit(
        shader, []
    )
    return shader


def _declared_names(shader, excluded):
    names = set()
    for node in shader.walk():
        if id(node) in excluded:
            continue
        if isinstance(node, (FunctionNode, StructNode, VariableNode)) and isinstance(
            node.name, str
        ):
            names.add(node.name)
    return names


def _unique_name(base, used_names, members):
    candidate = base
    while candidate in used_names or candidate in members:
        candidate = f"{candidate}_"
    return candidate


class _Binder:
    def __init__(self, members):
        self.members = members
        self.visited_lists = set()

    def visit(self, node, scopes):
        if isinstance(node, AttributeNode):
            return
        if isinstance(node, FunctionNode):
            scopes = [*scopes, set()]
            for parameter in node.parameters or []:
                if isinstance(parameter, ParameterNode):
                    self.visit(parameter, scopes)
                    scopes[-1].add(parameter.name)
            self.visit_field(node, "body", node.body, scopes)
            return
        if isinstance(node, VariableNode):
            self.visit_field(node, "initial_value", node.initial_value, scopes)
            if scopes:
                scopes[-1].add(node.name)
            return
        if isinstance(node, StructNode):
            return
        if isinstance(node, StatementNode):
            scopes = [*scopes, set()]
        for field, value in list(vars(node).items()):
            if field in AST_CHILD_FIELD_EXCLUSIONS:
                continue
            if isinstance(node, FunctionCallNode) and field in {"function", "name"}:
                if not isinstance(value, IdentifierNode):
                    self.visit_field(node, field, value, scopes)
                continue
            self.visit_field(node, field, value, scopes)

    def visit_field(self, node, field, value, scopes):
        if isinstance(value, IdentifierNode):
            bound = self.bound(value, scopes)
            if bound is not None:
                for alias, current in list(vars(node).items()):
                    if current is value:
                        setattr(node, alias, bound)
            return
        if isinstance(value, ASTNode):
            self.visit(value, scopes)
            return
        if isinstance(value, dict):
            for item in list(value.values()):
                if isinstance(item, (ASTNode, list, dict)):
                    self.visit_field(node, field, item, scopes)
            return
        if isinstance(value, list):
            if id(value) in self.visited_lists:
                return
            self.visited_lists.add(id(value))
            for index, item in enumerate(value):
                if isinstance(item, IdentifierNode):
                    bound = self.bound(item, scopes)
                    if bound is not None:
                        value[index] = bound
                elif isinstance(item, ASTNode):
                    self.visit(item, scopes)

    def bound(self, identifier, scopes):
        name = identifier.name
        instance = self.members.get(name)
        if instance is None or any(name in scope for scope in scopes):
            return None
        access = MemberAccessNode(IdentifierNode(instance), name)
        access.source_location = getattr(identifier, "source_location", None)
        return access

"""Lower HLSL struct member functions to CrossGL free functions.

HLSL structs may declare member functions that read or modify their receiver,
either through ``this`` or by naming a field directly, and static member
functions that are called as ``Type::Name(...)``. CrossGL has no member
functions, so every non-template member function becomes a free function named
after its owner. Instance functions receive the object through an explicit
leading parameter: ``in`` when the body never modifies the receiver and
``inout`` when it does. CrossGL ``inout`` is copy-in/copy-out, which matches the
reference semantics of an HLSL receiver within one invocation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Iterator

from ..common_ast import (
    ArrayAccessNode,
    AssignmentNode,
    FunctionCallNode,
    FunctionNode,
    MemberAccessNode,
    StructNode,
    UnaryOpNode,
    VariableNode,
)

RECEIVER_PARAMETER = "__this"
_MUTATING_UNARY_OPERATORS = {"++", "--"}


@dataclass
class StructMethod:
    """One HLSL struct member function and its lowering decisions."""

    owner: str
    name: str
    static: bool = False
    definition: FunctionNode | None = None
    inline: bool = False
    mutates_receiver: bool = False

    @property
    def qualified_name(self) -> str:
        return f"{self.owner}::{self.name}"


StructMethodTable = Dict[str, Dict[str, StructMethod]]


def _is_static(function: FunctionNode) -> bool:
    return any(
        str(qualifier).lower() == "static"
        for qualifier in getattr(function, "qualifiers", []) or []
    )


def collect_struct_methods(structs, functions) -> StructMethodTable:
    """Index the non-template member functions of every struct by owner name.

    Member functions declared in a struct body are matched with out-of-line
    definitions named ``Owner::Name``; a definition inside the struct body is
    used directly. Template member functions are left to the existing
    pass-through handling because their instantiation is not lowered.
    """

    table: StructMethodTable = {}
    for struct in structs or []:
        if not isinstance(struct, StructNode) or not isinstance(
            getattr(struct, "name", None), str
        ):
            continue
        for method in getattr(struct, "methods", []) or []:
            if not isinstance(method, FunctionNode) or not isinstance(method.name, str):
                continue
            if getattr(method, "is_template", False) or "<" in method.name:
                continue
            if method.name.startswith("operator"):
                # Operator overloads are invoked through expression syntax, which
                # this lowering does not rewrite.
                continue
            if method.name in {
                struct.name,
                f"~{struct.name}",
            } or method.name.startswith("~"):
                # Constructors and destructors keep their existing import path.
                continue
            entry = table.setdefault(struct.name, {}).setdefault(
                method.name, StructMethod(owner=struct.name, name=method.name)
            )
            entry.static = entry.static or _is_static(method)
            if not getattr(method, "is_prototype", False):
                entry.definition = method
                entry.inline = True
    for function in functions or []:
        name = getattr(function, "name", None)
        if not isinstance(name, str) or "::" not in name:
            continue
        owner, member = name.rsplit("::", 1)
        entry = table.get(owner, {}).get(member)
        if entry is None:
            continue
        entry.static = entry.static or _is_static(function)
        if not getattr(function, "is_prototype", False):
            entry.definition = function
            entry.inline = False
    return table


def lvalue_root(expr):
    """Return the identifier an lvalue expression is rooted at, if any."""
    while isinstance(expr, (ArrayAccessNode, MemberAccessNode)):
        expr = expr.array if isinstance(expr, ArrayAccessNode) else expr.object
    if isinstance(expr, VariableNode):
        return expr.name
    return expr if isinstance(expr, str) else None


def walk(node, children: Callable[[object], Iterable[object]]) -> Iterator[object]:
    """Yield a node and all of its descendants."""
    stack = [node]
    while stack:
        current = stack.pop()
        if current is None or isinstance(current, (str, int, float, bool)):
            continue
        if isinstance(current, (list, tuple)):
            stack.extend(reversed(current))
            continue
        yield current
        stack.extend(reversed(list(children(current))))


def method_local_names(function: FunctionNode, children) -> set:
    """Names of the parameters and local declarations of a function."""
    names = {
        getattr(param, "name", None) for param in getattr(function, "params", []) or []
    }
    for node in walk(getattr(function, "body", None), children):
        if isinstance(node, VariableNode) and isinstance(node.name, str):
            names.add(node.name)
    names.discard(None)
    return names


def analyze_receiver_mutations(
    table: StructMethodTable,
    struct_fields: dict[str, set],
    children,
    writes_argument: Callable[[FunctionCallNode, int], bool],
) -> None:
    """Mark the instance member functions that modify their receiver.

    A body modifies the receiver when it assigns to, increments or passes as a
    writable argument an lvalue rooted at ``this`` or at an unshadowed field,
    or calls a modifying member function on such an lvalue. The analysis is
    iterated to a fixed point so modifications propagate through calls.
    """

    methods = [
        method
        for owner_methods in table.values()
        for method in owner_methods.values()
        if not method.static and method.definition is not None
    ]
    locals_by_method = {
        id(method): method_local_names(method.definition, children)
        for method in methods
    }
    changed = True
    while changed:
        changed = False
        modifying_names = {method.name for method in methods if method.mutates_receiver}
        for method in methods:
            if method.mutates_receiver:
                continue
            fields = struct_fields.get(method.owner, set())
            local_names = locals_by_method[id(method)]

            def rooted_at_receiver(expr, fields=fields, local_names=local_names):
                root = lvalue_root(expr)
                return root == "this" or (
                    isinstance(root, str) and root in fields and root not in local_names
                )

            if _body_modifies_receiver(
                method,
                table,
                rooted_at_receiver,
                modifying_names,
                local_names,
                children,
                writes_argument,
            ):
                method.mutates_receiver = True
                changed = True


def _body_modifies_receiver(
    method,
    table,
    rooted_at_receiver,
    modifying_names,
    local_names,
    children,
    writes_argument,
) -> bool:
    owner_methods = table.get(method.owner, {})
    for node in walk(method.definition.body, children):
        if isinstance(node, AssignmentNode) and rooted_at_receiver(node.left):
            return True
        if (
            isinstance(node, UnaryOpNode)
            and str(getattr(node, "op", "")) in _MUTATING_UNARY_OPERATORS
            and rooted_at_receiver(node.operand)
        ):
            return True
        if not isinstance(node, FunctionCallNode):
            continue
        name = node.name
        if isinstance(name, str) and name not in local_names:
            callee = owner_methods.get(name)
            if callee is not None and not callee.static and callee.mutates_receiver:
                return True
        if (
            isinstance(name, MemberAccessNode)
            and rooted_at_receiver(name.object)
            and str(name.member) in modifying_names
        ):
            return True
        for index, argument in enumerate(getattr(node, "args", []) or []):
            if rooted_at_receiver(argument) and writes_argument(node, index):
                return True
    return False

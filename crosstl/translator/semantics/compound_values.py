"""Values of rewritten compound assignments that can write the assignment's target.

Lowering rewrites a wrapping compound assignment ``t op= v`` as ``t = f(t, v)`` (:mod:`.lower`), which reads ``t``
as the left operand of ``f``. glslang, Mesa and DXC read the target of a compound assignment after its value (DXC
follows C++17 [expr.ass]/1: the right operand is sequenced before the left), so when ``v`` writes ``t`` (a call
with ``t`` as an ``inout`` argument, a function that writes the global ``t``, an assignment in ``v``), the source
combines the new value of ``t``. The targets evaluate the left operand of ``f`` first, which reads the old one.

Such a value is evaluated first, into a local of its converted ``int`` type, and the assignment reads the local:
``{ int cgl_compound_value_0 = v; t = f(t, cgl_compound_value_0); }``.
"""

from __future__ import annotations

from .. import ast as A
from .hub_types import call_name, is_pure_callee, iter_functions
from .rewrite import child_slots, get_child

__all__ = ["CompoundValueHoister"]

_PREFIX = "cgl_compound_value"


def _root_name(expr):
    """The variable an lvalue is rooted at, or ``None``."""
    while True:
        if isinstance(expr, A.IdentifierNode):
            return expr.name
        if isinstance(expr, A.ArrayAccessNode):
            expr = expr.array_expr
        elif isinstance(expr, A.MemberAccessNode):
            expr = expr.object_expr
        elif isinstance(expr, A.SwizzleNode):
            expr = expr.vector_expr
        elif isinstance(expr, A.UnaryOpNode) and expr.op == "*":
            expr = expr.operand
        else:
            return None


def _nodes(node):
    stack, seen = [node], set()
    while stack:
        current = stack.pop()
        if id(current) in seen or not isinstance(current, A.ASTNode):
            continue
        seen.add(id(current))
        yield current
        stack.extend(
            get_child(current, field, key) for field, key in child_slots(current)
        )


def _names(node, root):
    return any(
        isinstance(current, A.IdentifierNode) and current.name == root
        for current in _nodes(node)
    )


def _plain_locals(function):
    """Names declared in ``function``'s body that only the function itself can write.

    A callee writes a caller's local only through an argument that names it; a local whose address is taken,
    and a pointer or reference local, may also be written through an alias, so they are left out.
    """
    declared, aliased = set(), set()
    for node in _nodes(getattr(function, "body", None)):
        if isinstance(node, A.VariableNode) and isinstance(node.name, str):
            var_type = getattr(node, "var_type", None)
            if isinstance(var_type, (A.PointerType, A.ReferenceType)):
                aliased.add(node.name)
            else:
                declared.add(node.name)
        elif isinstance(node, A.UnaryOpNode) and node.op == "&":
            aliased.add(_root_name(node.operand))
    return declared - aliased


class CompoundValueHoister:
    def __init__(self, program):
        self.program = program
        self._functions = None
        self._declared = None
        self._locals = {}

    def functions(self):
        if self._functions is None:
            self._functions = {
                getattr(function, "name", None)
                for function in iter_functions(self.program)
            }
        return self._functions

    def plain_locals(self, function):
        if function is None:
            return set()
        if id(function) not in self._locals:
            self._locals[id(function)] = _plain_locals(function)
        return self._locals[id(function)]

    def may_write(self, value, target, function):
        """Whether evaluating ``value`` can change ``target``, conservatively.

        Any assignment or step can. A call can when it is passed the target's variable, and a call of a function
        defined in the program also when the target is not a plain local of the enclosing ``function`` (globals,
        shared and buffer storage, parameters).
        """
        root = _root_name(target)
        local = root is not None and root in self.plain_locals(function)
        for node in _nodes(value):
            if isinstance(node, A.AssignmentNode):
                return True
            if isinstance(node, A.UnaryOpNode) and node.op in ("++", "--"):
                return True
            if not isinstance(node, A.FunctionCallNode):
                continue
            name = call_name(node)
            if name and is_pure_callee(name):
                continue
            if not local and (not name or name in self.functions()):
                return True
            if root is None or _names(node, root):
                return True
        return False

    def fresh_name(self):
        if self._declared is None:
            self._declared = {
                current.name
                for current in _nodes(self.program)
                if isinstance(getattr(current, "name", None), str)
            }
        index = 0
        while f"{_PREFIX}_{index}" in self._declared:
            index += 1
        name = f"{_PREFIX}_{index}"
        self._declared.add(name)
        return name

    def declare(self, value, lanes):
        """A declaration of a fresh ``int``/``ivecN`` local initialized from ``value``; returns ``(node, name)``."""
        from ..lexer import Lexer
        from ..parser import Parser

        spelling = "int" if lanes == 1 else f"ivec{lanes}"
        type_node = Parser(Lexer(f"{spelling} x;").tokens).parse_type()
        name = self.fresh_name()
        declaration = A.VariableNode(name=name, var_type=type_node, initial_value=value)
        declaration.source_location = getattr(value, "source_location", None)
        return declaration, name

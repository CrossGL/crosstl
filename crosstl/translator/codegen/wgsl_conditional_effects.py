"""Conditionals whose branches have side effects, for WGSL.

WGSL output writes ``c ? a : b`` as ``select(b, a, c)``. ``select`` is a built-in function, so both of its value
arguments are evaluated before it runs (WGSL section 17.3), while the C-family conditional evaluates only the
branch its condition picks (GLSL 4.60 section 5.9, C++ [expr.cond]). A branch that calls a function with a side
effect, such as ``int x = c ? bump() : 7;``, therefore ran its call unconditionally, and the program wrote its
globals, buffers or images, or discarded the fragment, on the path the source never takes.

:meth:`WGSLConditionalEffectsMixin.lower_effectful_ternary` rewrites the statements whose value is such a
conditional into an ``if``/``else`` that evaluates the chosen branch only::

    int x = c ? bump() : 7;      var x: i32;  if (c) { x = bump(); } else { x = 7; }
    y += c ? bump() : 0;         if (c) { y += bump(); } else { y += 0; }
    return c ? bump() : 7;       if (c) { return bump(); } else { return 7; }
    c ? bump() : reset();        if (c) { bump(); } else { reset(); }

A conditional in a branch is lowered again as that branch's statement. Anywhere else, such as ``1 + (c ? bump() :
0)``, :meth:`~WGSLConditionalEffectsMixin.require_unconditional_branches` refuses the conditional with
``ValueError`` rather than evaluating both branches.

A call has a side effect when it is a built-in that writes memory, synchronizes, emits or discards (atomics, stores,
barriers, ``discard``), or a function the shader declares whose body writes anything other than its own locals,
writes an ``out`` or ``inout`` parameter, or makes such a call. Reads are not side effects: moving a texture sample
into the ``if``/``else`` would only add a uniformity requirement the source does not have.
"""

from __future__ import annotations

import copy

from ..ast import (
    ArrayAccessNode,
    AssignmentNode,
    BlockNode,
    ExpressionStatementNode,
    FunctionCallNode,
    IdentifierNode,
    IfNode,
    LambdaNode,
    MemberAccessNode,
    ReturnNode,
    SwizzleNode,
    TernaryOpNode,
    UnaryOpNode,
    VariableNode,
)
from ..refusals import WGSLRefusal

__all__ = ["WGSLConditionalEffectsMixin", "is_effectful_builtin"]

_EFFECT_PREFIXES = ("atomic", "interlocked", "imageatomic", "buffer_append")
_EFFECT_SUBSTRINGS = ("store", "barrier")
_EFFECT_NAMES = frozenset(
    {
        "buffer_consume",
        "debugprintf",
        "discard",
        "emitmeshtasks",
        "emitstreamvertex",
        "emitvertex",
        "endprimitive",
        "endstreamprimitive",
        "executecallable",
        "ignoreintersection",
        "printf",
        "reportintersection",
        "setmeshoutputs",
        "terminateray",
        "traceray",
    }
)
_WRITE_QUALIFIERS = frozenset({"out", "inout"})


def is_effectful_builtin(name: str) -> bool:
    """Whether a call to the built-in ``name`` writes memory, synchronizes, emits or discards."""
    lowered = name.lower()
    return (
        lowered in _EFFECT_NAMES
        or lowered.startswith(_EFFECT_PREFIXES)
        or any(part in lowered for part in _EFFECT_SUBSTRINGS)
    )


def _callee_name(call):
    function = getattr(call, "function", None)
    if isinstance(function, str):
        return function
    if isinstance(function, IdentifierNode):
        return function.name
    if isinstance(function, MemberAccessNode):
        return function.member
    return None


def _nodes(root):
    """``root`` and its descendants, each once, without entering lambdas."""
    if root is None:
        return
    stack, seen = [root], set()
    while stack:
        node = stack.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        yield node
        if not isinstance(node, LambdaNode) and hasattr(node, "child_nodes"):
            stack.extend(node.child_nodes())


def _root_name(target):
    """The variable an l-value writes, or None when it writes through something else."""
    while isinstance(target, (MemberAccessNode, ArrayAccessNode, SwizzleNode)):
        target = (
            target.object_expr
            if isinstance(target, MemberAccessNode)
            else (
                target.array_expr
                if isinstance(target, ArrayAccessNode)
                else target.vector_expr
            )
        )
    return target.name if isinstance(target, IdentifierNode) else None


def _parameter_writes_caller(parameter):
    qualifiers = {str(q).lower() for q in getattr(parameter, "qualifiers", []) or []}
    return bool(qualifiers & _WRITE_QUALIFIERS)


class WGSLConditionalEffectsMixin:
    """Statement lowering for conditionals with side-effecting branches; see the module docstring."""

    def function_has_side_effect(self, name, visiting=None):
        """Whether calling the shader's function ``name`` (any overload) can have a side effect."""
        cache = self.__dict__.setdefault("_wgsl_function_effects", {})
        if name in cache:
            return cache[name]
        overloads = (getattr(self, "_function_overloads_by_name", None) or {}).get(name)
        if not overloads:
            return is_effectful_builtin(name)
        visiting = set() if visiting is None else visiting
        if name in visiting:
            return True
        visiting.add(name)
        result = any(
            self._function_body_has_side_effect(function, visiting)
            for function in overloads
        )
        visiting.discard(name)
        cache[name] = result
        return result

    def _function_body_has_side_effect(self, function, visiting):
        parameters = list(getattr(function, "parameters", None) or [])
        if any(_parameter_writes_caller(parameter) for parameter in parameters):
            return True
        module_names = set(getattr(self, "_module_identifier_names", None) or {})
        locals_ = {getattr(parameter, "name", None) for parameter in parameters}
        body = getattr(function, "body", None)
        nodes = list(_nodes(body if not isinstance(body, list) else BlockNode(body)))
        locals_.update(node.name for node in nodes if isinstance(node, VariableNode))
        # A local that shares a module-scope name may be out of scope where it is written.
        locals_ -= module_names
        for node in nodes:
            if self._node_has_side_effect(node, locals_, visiting):
                return True
        return False

    def _node_has_side_effect(self, node, locals_, visiting):
        if isinstance(node, AssignmentNode):
            return _root_name(node.target) not in locals_
        if isinstance(node, UnaryOpNode) and node.operator in ("++", "--"):
            return _root_name(node.operand) not in locals_
        if isinstance(node, IdentifierNode):
            return node.name == "discard"
        if isinstance(node, FunctionCallNode):
            name = _callee_name(node)
            return name is None or self.function_has_side_effect(name, visiting)
        return False

    def expression_has_side_effect(self, expr):
        """Whether evaluating ``expr`` can call something with a side effect, or itself assigns or updates."""
        for node in _nodes(expr):
            if isinstance(node, AssignmentNode):
                return True
            if isinstance(node, UnaryOpNode) and node.operator in ("++", "--"):
                return True
            if isinstance(node, IdentifierNode) and node.name == "discard":
                return True
            if isinstance(node, FunctionCallNode):
                name = _callee_name(node)
                if name is None or self.function_has_side_effect(name):
                    return True
        return False

    def effectful_branches(self, expr):
        return isinstance(expr, TernaryOpNode) and (
            self.expression_has_side_effect(expr.true_expr)
            or self.expression_has_side_effect(expr.false_expr)
        )

    def lower_effectful_ternary(self, stmt):
        """The statements that evaluate only the chosen branch of a side-effecting conditional ``stmt`` value."""
        if isinstance(stmt, ExpressionStatementNode):
            expression = stmt.expression
            if isinstance(expression, AssignmentNode):
                return self.lower_effectful_ternary(expression)
            if self.effectful_branches(expression):
                return [
                    self._branching(
                        expression,
                        lambda branch: ExpressionStatementNode(branch),
                    )
                ]
            return None
        if isinstance(stmt, VariableNode):
            value = stmt.initial_value
            if not self.effectful_branches(value):
                return None
            if any(
                isinstance(node, IdentifierNode) and node.name == stmt.name
                for node in _nodes(value)
            ):
                raise WGSLRefusal(
                    f"WGSL target cannot lower the conditional that initializes {stmt.name}: "
                    f"it reads a variable named {stmt.name}, which would name the new variable "
                    "once its declaration moves above the if/else; rename one of them"
                )
            var_type = self.local_variable_declaration_type(stmt)
            declaration = copy.copy(stmt)
            declaration.var_type = var_type
            declaration.vtype = var_type
            declaration.initial_value = None
            declaration.is_mutable = True
            assignment = AssignmentNode(IdentifierNode(stmt.name), value)
            return [declaration, *self.lower_effectful_ternary(assignment)]
        if isinstance(stmt, AssignmentNode):
            if not self.effectful_branches(stmt.value):
                return None
            target, operator = stmt.target, stmt.operator
            return [
                self._branching(
                    stmt.value,
                    lambda branch: AssignmentNode(target, branch, operator),
                )
            ]
        if isinstance(stmt, ReturnNode):
            if not self.effectful_branches(stmt.value):
                return None
            return [self._branching(stmt.value, ReturnNode)]
        return None

    @staticmethod
    def _branching(conditional, make):
        return IfNode(
            conditional.condition,
            BlockNode([make(conditional.true_expr)]),
            BlockNode([make(conditional.false_expr)]),
        )

    def require_unconditional_branches(self, expr):
        """Refuse the conditional ``expr`` when ``select`` would run a side effect its condition skips."""
        if not self.effectful_branches(expr):
            return
        raise WGSLRefusal(
            "WGSL target cannot lower this conditional expression: a branch calls a function with a "
            "side effect, and WGSL select evaluates both branches; such a conditional is lowered only "
            "as a whole initializer, assignment value, return value or expression statement"
        )

"""HLSL-style atomic calls in CrossGL, rewritten to the value-returning forms every generator carries.

HLSL's interlocked functions return the previous value through a trailing ``out`` argument:
``InterlockedAdd(dest, value, original)``, ``InterlockedCompareExchange(dest, compare, value, original)``, and
``InterlockedCompareStore(dest, compare, value)`` returns nothing (HLSL intrinsic function reference). CrossGL's
atomic built-ins return the previous value instead, as GLSL's do (GLSL 4.60 section 8.11: ``atomicAdd(mem, data)``,
``atomicCompSwap(mem, compare, data)``).

The HLSL importer writes the value-returning form when the destination is a structured-buffer element, but keeps
HLSL's argument list for any other destination (a group-shared variable, a member or swizzle of an element), passes
``InterlockedCompareStore`` through unchanged, and spells compare-exchange ``atomicCompareExchange``. The OpenGL
generator printed those calls as they were, which glslang rejects, and the Metal and SPIR-V generators refused them.

:func:`canonicalize_atomic_calls` rewrites, for the OpenGL, SPIR-V and Metal targets:

* the statement ``atomicOp(dest, value, original)`` into ``original = atomicOp(dest, value)`` for ``atomicAdd``,
  ``atomicMin``, ``atomicMax``, ``atomicAnd``, ``atomicOr``, ``atomicXor`` and ``atomicExchange``;
* the statement ``atomicCompSwap(dest, compare, value, original)``, or ``atomicCompareExchange`` with the same
  arguments, into ``original = atomicCompSwap(dest, compare, value)``;
* the statement ``InterlockedCompareStore(dest, compare, value)`` into ``atomicCompSwap(dest, compare, value)``;
* any remaining three-argument ``atomicCompareExchange`` call into ``atomicCompSwap``.

HLSL assigns ``original`` after the operation, as the rewritten assignment does. A call whose name a function of the
program defines is left alone.

HLSL output keeps the interlocked argument lists, which are its native form. For HLSL targets,
:func:`hlsl_out_argument_atomics` does the opposite rewrite where the destination is not a buffer, because the
DirectX generator writes value-returning atomics on buffer elements only and printed ``uint o = InterlockedAdd(g,
1u);`` for a group-shared ``g``, which DXC rejects: ``T o = atomicOp(dest, value);`` becomes ``T o;
atomicOp(dest, value, o);`` and ``o = atomicOp(dest, value);`` becomes ``atomicOp(dest, value, o);``, and the same
for ``atomicCompSwap(dest, compare, value)``.
"""

from __future__ import annotations

from .. import ast as A
from .hub_types import call_name, iter_functions
from .rewrite import call, child_slots, get_child, set_child
from .scoped_walk import ScopedWalker

__all__ = [
    "HLSL_TARGETS",
    "TARGETS",
    "canonicalize_atomic_calls",
    "hlsl_out_argument_atomics",
]

_RETURNING_OUT_ORIGINAL = frozenset(
    {
        "atomicAdd",
        "atomicMin",
        "atomicMax",
        "atomicAnd",
        "atomicOr",
        "atomicXor",
        "atomicExchange",
    }
)
_COMPARE_SWAP = "atomicCompSwap"
_COMPARE_EXCHANGE = "atomicCompareExchange"
_COMPARE_STORE = "InterlockedCompareStore"
TARGETS = frozenset({"opengl", "glsl", "vulkan", "spirv", "metal"})
HLSL_TARGETS = frozenset({"directx", "hlsl", "dx11", "dx12", "d3d11", "d3d12"})


class _AtomicWalker(ScopedWalker):
    def __init__(self, program, defined):
        super().__init__(program)
        self.defined = defined
        self.rewritten = 0

    def visit(self, node):
        if isinstance(node, A.ExpressionStatementNode):
            replacement = self.statement_form(node.expression)
            if replacement is not None:
                set_child(node, "expression", None, replacement)
                self.rewritten += 1
            return
        if (
            isinstance(node, A.FunctionCallNode)
            and call_name(node) == _COMPARE_EXCHANGE
            and _COMPARE_EXCHANGE not in self.defined
            and len(node.arguments or []) == 3
        ):
            _rename(node, _COMPARE_SWAP)
            self.rewritten += 1

    def statement_form(self, expression):
        if not isinstance(expression, A.FunctionCallNode):
            return None
        name = call_name(expression)
        if name in self.defined:
            return None
        arguments = list(expression.arguments or [])
        if name in _RETURNING_OUT_ORIGINAL and len(arguments) == 3:
            return A.AssignmentNode(arguments[2], call(name, arguments[:2]))
        if name in (_COMPARE_SWAP, _COMPARE_EXCHANGE) and len(arguments) == 4:
            return A.AssignmentNode(arguments[3], call(_COMPARE_SWAP, arguments[:3]))
        if name == _COMPARE_STORE and len(arguments) == 3:
            return call(_COMPARE_SWAP, arguments)
        return None


def _rename(node, name):
    function = A.IdentifierNode(name)
    node.function = function
    node.name = function


def canonicalize_atomic_calls(program, target) -> int:
    """Rewrite HLSL-style atomic calls in place for ``target``; returns how many were rewritten."""
    if not isinstance(program, A.ShaderNode) or str(target or "") not in TARGETS:
        return 0
    defined = {
        function.name
        for function in iter_functions(program)
        if isinstance(getattr(function, "name", None), str)
    }
    walker = _AtomicWalker(program, defined)
    walker.run_program()
    return walker.rewritten


def _root_name(expression):
    while True:
        if isinstance(expression, A.ArrayAccessNode):
            expression = expression.array_expr
        elif isinstance(expression, A.MemberAccessNode):
            expression = expression.object_expr
        elif isinstance(expression, A.SwizzleNode):
            expression = expression.vector_expr
        else:
            break
    return expression.name if isinstance(expression, A.IdentifierNode) else None


def _shared_names(program):
    """Names declared ``shared`` or ``groupshared`` at program or stage scope, or in a function body."""
    declarations = list(getattr(program, "global_variables", None) or [])
    stages = getattr(program, "stages", None) or {}
    for stage in stages.values() if isinstance(stages, dict) else stages:
        declarations.extend(getattr(stage, "local_variables", None) or [])
    for function in iter_functions(program):
        declarations.extend(
            node
            for node in _walk(getattr(function, "body", None))
            if isinstance(node, A.VariableNode)
        )
    return {
        declaration.name
        for declaration in declarations
        if {"shared", "groupshared"}
        & set(getattr(declaration, "qualifiers", None) or [])
    }


def _walk(node):
    if node is None:
        return
    pending = [node]
    while pending:
        item = pending.pop()
        yield item
        pending.extend(get_child(item, field, key) for field, key in child_slots(item))


_HLSL_INTERLOCKED = {
    "atomicAdd": "InterlockedAdd",
    "atomicMin": "InterlockedMin",
    "atomicMax": "InterlockedMax",
    "atomicAnd": "InterlockedAnd",
    "atomicOr": "InterlockedOr",
    "atomicXor": "InterlockedXor",
    "atomicExchange": "InterlockedExchange",
    _COMPARE_SWAP: "InterlockedCompareExchange",
    _COMPARE_EXCHANGE: "InterlockedCompareExchange",
}


class _OutArgumentWalker(ScopedWalker):
    def __init__(self, program, defined, shared):
        super().__init__(program)
        self.defined = defined
        self.shared = shared
        self.rewritten = 0

    def atomic_call(self, expression):
        """``(name, arguments, value arity)`` of an atomic on group-shared memory, or ``None``."""
        if not isinstance(expression, A.FunctionCallNode):
            return None
        name = call_name(expression)
        arguments = list(expression.arguments or [])
        if name in self.defined or name not in _HLSL_INTERLOCKED or not arguments:
            return None
        if _root_name(arguments[0]) not in self.shared:
            return None
        return (
            name,
            arguments,
            (
                3
                if _HLSL_INTERLOCKED[name].endswith("Exchange")
                and name != "atomicExchange"
                else 2
            ),
        )

    def interlocked(self, name, arguments, original=None):
        operands = [*arguments] if original is None else [*arguments, original]
        return A.ExpressionStatementNode(call(_HLSL_INTERLOCKED[name], operands))

    def statement_forms(self, statement):
        """The statements that replace ``statement``, or ``None`` to keep it."""
        if isinstance(statement, A.VariableNode):
            found = self.atomic_call(statement.initial_value)
            if found is None or len(found[1]) != found[2]:
                return None
            name, arguments, _arity = found
            statement.initial_value = None
            return [
                statement,
                self.interlocked(name, arguments, A.IdentifierNode(statement.name)),
            ]
        if not isinstance(statement, A.ExpressionStatementNode):
            return None
        expression = statement.expression
        if isinstance(expression, A.AssignmentNode) and expression.operator == "=":
            found = self.atomic_call(expression.value)
            if found is None or len(found[1]) != found[2]:
                return None
            name, arguments, _arity = found
            return [self.interlocked(name, arguments, expression.target)]
        found = self.atomic_call(expression)
        if found is None:
            return None
        name, arguments, arity = found
        if len(arguments) == arity + 1:
            return [self.interlocked(name, arguments)]
        if len(arguments) == arity and name in (_COMPARE_SWAP, _COMPARE_EXCHANGE):
            return [A.ExpressionStatementNode(call(_COMPARE_STORE, arguments))]
        if len(arguments) == arity and name != "atomicExchange":
            # HLSL's interlocked operations take the original-value argument optionally,
            # except InterlockedExchange.
            return [self.interlocked(name, arguments)]
        return None

    def visit(self, node):
        if not isinstance(node, A.BlockNode) or not isinstance(node.statements, list):
            return
        statements = []
        for statement in node.statements:
            replacement = self.statement_forms(statement)
            if replacement is None:
                statements.append(statement)
            else:
                statements.extend(replacement)
                self.rewritten += 1
        node.statements[:] = statements


def hlsl_out_argument_atomics(program, target) -> int:
    """Write atomics on group-shared memory as HLSL interlocked calls, for HLSL targets."""
    if not isinstance(program, A.ShaderNode) or str(target or "") not in HLSL_TARGETS:
        return 0
    defined = {
        function.name
        for function in iter_functions(program)
        if isinstance(getattr(function, "name", None), str)
    }
    walker = _OutArgumentWalker(program, defined, _shared_names(program))
    walker.run_program()
    return walker.rewritten

"""Metal ``device const`` buffer parameters that a function writes after casting the ``const`` away.

C++ lets a kernel declare ``device const char* dst [[buffer(n)]]`` and then store through
``(device float*)((device char*)dst + offset)``. Metal binds the same buffer either way, so the kernel writes into
it. The importer chooses a CrossGL ``StructuredBuffer`` (read-only) or ``RWStructuredBuffer`` (read-write) from the
declared qualifiers alone, so such a parameter used to become read-only and every target then refused, or
miscompiled, the store.

:func:`mark_const_cast_written_buffers` runs once over the parsed unit. For each function it follows the pointers
derived from every ``device`` parameter whose pointee is ``const``: casts, pointer arithmetic, address-of
subscripts and members, conditional selections and local pointer variables initialised or assigned from those. A
parameter is marked written (``metal_storage_written``) when

- an assignment, compound assignment, increment or decrement stores through one of its derived pointers, or
- a call receives one of its derived pointers whose ``const`` was cast away, so the callee may store through it.

The analysis is flow-insensitive within a function. It never loses a write it can see; a store it cannot see, such
as one a helper makes through a parameter that was never cast, leaves the parameter read-only, and the target then
refuses the store with a typed diagnostic.
"""

from __future__ import annotations

from .MetalAst import (
    ArrayAccessNode,
    AssignmentNode,
    BinaryOpNode,
    CallNode,
    CastNode,
    FunctionCallNode,
    FunctionNode,
    MemberAccessNode,
    MethodCallNode,
    PostfixOpNode,
    TernaryOpNode,
    UnaryOpNode,
    VariableNode,
)

__all__ = ["mark_const_cast_written_buffers"]

_STEP_OPERATORS = frozenset({"++", "--"})
_POINTER_ARITHMETIC = frozenset({"+", "-"})


def _nodes(root):
    """Every AST node below ``root`` in a deterministic depth-first order."""
    seen = set()
    stack = [root]
    while stack:
        node = stack.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        if isinstance(node, (list, tuple)):
            stack.extend(reversed(node))
            continue
        if isinstance(node, dict):
            stack.extend(reversed(list(node.values())))
            continue
        if not hasattr(node, "__dict__") or isinstance(node, type):
            continue
        yield node
        children = []
        for field, value in vars(node).items():
            if field in ("parent", "source_location"):
                continue
            if isinstance(value, (list, tuple, dict)) or hasattr(value, "__dict__"):
                children.append(value)
        stack.extend(reversed(children))


def _words(text):
    return set(str(text or "").replace("*", " * ").replace("&", " & ").split())


def _is_pointer_type(type_text):
    return "*" in str(type_text or "")


def _const_pointee_device_parameter(param):
    qualifiers = {
        str(qualifier) for qualifier in getattr(param, "qualifiers", []) or []
    }
    vtype = str(getattr(param, "vtype", "") or "")
    if not _is_pointer_type(vtype) or "&" in vtype:
        return False
    words = qualifiers | _words(vtype.split("*", 1)[0])
    return "device" in words and "const" in words and "constant" not in words


def _mutable_device_cast(node):
    if not isinstance(node, CastNode) or not _is_pointer_type(node.target_type):
        return False
    words = {str(qualifier) for qualifier in node.qualifiers} | _words(
        str(node.target_type).split("*", 1)[0]
    )
    return "const" not in words and "constant" not in words


def _declared_local(node):
    """``(variable, initializer)`` when ``node`` declares a local pointer, else ``None``."""
    if isinstance(node, AssignmentNode) and getattr(node, "operator", "=") == "=":
        target = node.left
        if isinstance(target, VariableNode) and str(target.vtype or "").strip():
            return target, node.right
        return None
    if isinstance(node, VariableNode) and str(node.vtype or "").strip():
        return node, node.value
    return None


class _FunctionWrites:
    """Pointer provenance from ``device const`` parameters inside one function."""

    def __init__(self, candidates):
        self.candidates = candidates
        # Local pointer name -> (origin parameter names, whether its pointee is mutable).
        self.aliases = {}

    def origins(self, node, mutable_only=False):
        if isinstance(node, VariableNode) and not str(node.vtype or "").strip():
            if node.name in self.aliases:
                origins, mutable = self.aliases[node.name]
                return set(origins) if mutable or not mutable_only else set()
            if node.name in self.candidates and not mutable_only:
                return {node.name}
            return set()
        if isinstance(node, CastNode):
            inner = self.origins(node.expression)
            if not mutable_only or _mutable_device_cast(node):
                return inner
            return self.origins(node.expression, mutable_only=True)
        if isinstance(node, (list, tuple)) and len(node) == 1:
            return self.origins(node[0], mutable_only)
        if isinstance(node, UnaryOpNode):
            if node.op == "&":
                return self.lvalue_origins(node.operand, mutable_only)
            return set()
        if isinstance(node, BinaryOpNode):
            if node.op in _POINTER_ARITHMETIC:
                return self.origins(node.left, mutable_only) | self.origins(
                    node.right, mutable_only
                )
            return set()
        if isinstance(node, TernaryOpNode):
            return self.origins(node.true_expr, mutable_only) | self.origins(
                node.false_expr, mutable_only
            )
        return set()

    def lvalue_origins(self, node, mutable_only=False):
        """Parameters whose storage an lvalue designates."""
        if isinstance(node, ArrayAccessNode):
            if isinstance(node.array, (ArrayAccessNode, MemberAccessNode)):
                nested = self.lvalue_origins(node.array, mutable_only)
                if nested:
                    return nested
            return self.origins(node.array, mutable_only)
        if isinstance(node, UnaryOpNode) and node.op == "*":
            return self.origins(node.operand, mutable_only)
        if isinstance(node, MemberAccessNode):
            if getattr(node, "is_pointer", False):
                return self.origins(node.object, mutable_only)
            return self.lvalue_origins(node.object, mutable_only)
        return set()

    def collect_aliases(self, nodes):
        changed = True
        while changed:
            changed = False
            for node in nodes:
                declared = _declared_local(node)
                if declared is not None:
                    variable, initializer = declared
                    vtype = str(variable.vtype)
                    if not (_is_pointer_type(vtype) or "auto" in _words(vtype)):
                        continue
                    origins = (
                        self.origins(initializer) if initializer is not None else set()
                    )
                    if "auto" in _words(vtype) and not _is_pointer_type(vtype):
                        mutable = bool(self.origins(initializer, mutable_only=True))
                    else:
                        qualifiers = {str(q) for q in variable.qualifiers or []}
                        pointee = qualifiers | _words(vtype.split("*", 1)[0])
                        mutable = "const" not in pointee
                    changed |= self._merge(variable.name, origins, mutable)
                elif (
                    isinstance(node, AssignmentNode)
                    and getattr(node, "operator", "=") == "="
                    and isinstance(node.left, VariableNode)
                    and node.left.name in self.aliases
                ):
                    origins = self.origins(node.right)
                    _, mutable = self.aliases[node.left.name]
                    changed |= self._merge(node.left.name, origins, mutable)

    def _merge(self, name, origins, mutable):
        previous, previous_mutable = self.aliases.get(name, (frozenset(), False))
        merged = frozenset(previous | origins)
        mutable = previous_mutable or mutable
        if name in self.aliases and merged == previous and mutable == previous_mutable:
            return False
        self.aliases[name] = (merged, mutable)
        return True

    def written(self, nodes):
        written = set()
        for node in nodes:
            if isinstance(node, AssignmentNode) and _declared_local(node) is None:
                written |= self.lvalue_origins(node.left)
            elif isinstance(node, (UnaryOpNode, PostfixOpNode)):
                if node.op in _STEP_OPERATORS:
                    written |= self.lvalue_origins(node.operand)
            elif isinstance(node, (FunctionCallNode, MethodCallNode, CallNode)):
                for argument in getattr(node, "args", None) or []:
                    written |= self.origins(argument, mutable_only=True)
        return written


def mark_const_cast_written_buffers(ast):
    """Set ``metal_storage_written`` on each ``device const`` pointer parameter a function stores through."""
    for function in _nodes(ast):
        if not isinstance(function, FunctionNode):
            continue
        params = [
            param
            for param in getattr(function, "params", None) or []
            if _const_pointee_device_parameter(param) and getattr(param, "name", None)
        ]
        if not params or getattr(function, "body", None) is None:
            continue
        analysis = _FunctionWrites({param.name for param in params})
        body = list(_nodes(function.body))
        analysis.collect_aliases(body)
        written = analysis.written(body)
        for param in params:
            if param.name in written:
                param.metal_storage_written = True

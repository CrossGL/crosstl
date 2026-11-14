"""Swizzles of scalars (``s.xxx``, ``dot(a, b).rr``, ``s.x``) written as vector constructors.

HLSL lets a scalar be swizzled with its only component, ``x`` or ``r``: ``s.xxx`` is ``float3(s, s, s)`` and ``s.x``
is ``s`` (HLSL scalar swizzles, "Per-Component Math Operations"). GLSL accepts the form from version 4.20, but MSL,
WGSL and GLSL ES do not, and the Metal compiler rejected the HLSL importer's ``dot(c, w).xxx`` with "member reference
base type 'float' is not a structure or union". :func:`expand_scalar_swizzles` rewrites every swizzle of a value
whose type is known to be a scalar into ``vecN(value)`` (``ivecN``, ``uvecN``, ``bvecN``, ``f16vecN`` and so on for
the other scalar types), or into the value itself for one component. The value is evaluated once either way.

A swizzle that names every component of a vector in order (``v.xy`` of a ``vec2``) is the vector itself and is
dropped too: generators that lower buffer members to loads and stores (Metal's byte-addressed storage) wrote
``float2(load0, load1).xy = value``, which is not assignable.
"""

from __future__ import annotations

import re

from .. import ast as A
from .rewrite import call, child_slots, get_child, set_child
from .scoped_walk import ScopedWalker

__all__ = ["expand_scalar_swizzles"]

_SCALAR_SWIZZLE = re.compile(r"x{1,4}|r{1,4}")
_IDENTITY_SWIZZLES = frozenset({"xy", "xyz", "xyzw", "rg", "rgb", "rgba"})
_VECTOR_PREFIXES = {
    ("f", 32): "vec",
    ("f", 64): "dvec",
    ("f", 16): "f16vec",
    ("i", 32): "ivec",
    ("u", 32): "uvec",
    ("b", 1): "bvec",
    ("i", 8): "i8vec",
    ("u", 8): "u8vec",
    ("i", 16): "i16vec",
    ("u", 16): "u16vec",
    ("i", 64): "i64vec",
    ("u", 64): "u64vec",
}


def expand_scalar_swizzles(program) -> int:
    """Rewrite the swizzles of scalar values in place; returns how many were rewritten."""
    if not isinstance(program, A.ShaderNode):
        return 0
    walker = _ScalarSwizzles(program)
    walker.run_program()
    return walker.rewritten


class _ScalarSwizzles(ScopedWalker):
    def __init__(self, program):
        super().__init__(program)
        self.rewritten = 0

    def _walk(self, node):
        if isinstance(node, A.VariableNode):
            # The walker declares a variable instead of visiting it; its initializer is rewritten first.
            for field, key in list(child_slots(node)):
                self._walk(get_child(node, field, key))
            self.visit(node)
            self.scope.declare(node.name, node.var_type)
            return
        super()._walk(node)

    def visit(self, node):
        # Children are visited first, so a swizzle is replaced from its parent with the scope it was read in.
        for field, key in list(child_slots(node)):
            child = get_child(node, field, key)
            replacement = self._replacement(child)
            if replacement is not None:
                set_child(node, field, key, replacement)
                self.rewritten += 1

    def _replacement(self, node):
        if isinstance(node, A.SwizzleNode):
            value, components = node.vector_expr, node.components
        elif isinstance(node, A.MemberAccessNode):
            value, components = node.object_expr, node.member
        else:
            return None
        if not isinstance(components, str):
            return None
        if components in _IDENTITY_SWIZZLES:
            # ``v.xy`` of a two-component ``v`` is ``v`` itself, as a value and as a store target.
            value_type = self.scope.infer(value)
            if value_type is not None and value_type.lanes == len(components) > 1:
                return value
            return None
        if not _SCALAR_SWIZZLE.fullmatch(components):
            return None
        value_type = self.scope.infer(value)
        if value_type is None or value_type.lanes != 1:
            return None
        if len(components) == 1:
            return value
        prefix = _VECTOR_PREFIXES.get((value_type.kind, value_type.bits))
        if prefix is None:
            return None
        return call(f"{prefix}{len(components)}", [value])

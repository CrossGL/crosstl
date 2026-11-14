"""GLSL texture-coordinate swizzles (``.stpq``) spelled with the position set (``.xyzw``).

GLSL names vector components with three equivalent sets, ``xyzw``, ``rgba`` and ``stpq`` (GLSL 4.60 section
5.5, "Vector and Scalar Components and Length"). HLSL, Metal, WGSL and Slang accept only the first two, so a
``coord.st`` carried through CrossGL failed to compile there. :func:`canonicalize_texture_swizzles` rewrites every
``stpq`` swizzle of a vector or scalar to the same components in ``xyzw``, which every target accepts with the same
meaning. A member access is a swizzle when no structure in the program has a member of that name, or when the
accessed value is known to be a vector or scalar rather than a structure.
"""

from __future__ import annotations

import re

from .. import ast as A
from .hub_types import iter_structs
from .scoped_walk import ScopedWalker

__all__ = ["canonicalize_texture_swizzles"]

_TEXTURE_SWIZZLE = re.compile(r"[stpq]{1,4}")
_POSITION_OF = str.maketrans("stpq", "xyzw")


def canonicalize_texture_swizzles(program) -> int:
    """Rewrite ``stpq`` swizzles to ``xyzw`` in place; returns how many were rewritten."""
    if not isinstance(program, A.ShaderNode):
        return 0
    walker = _SwizzleWalker(program)
    walker.run_program()
    return walker.rewritten


class _SwizzleWalker(ScopedWalker):
    def __init__(self, program):
        super().__init__(program)
        self.member_names = {
            getattr(member, "name", None)
            for struct in iter_structs(program)
            for member in getattr(struct, "members", None) or []
        }
        self.rewritten = 0

    def visit(self, node):
        if isinstance(node, A.SwizzleNode):
            components = node.components
            if isinstance(components, str) and _TEXTURE_SWIZZLE.fullmatch(components):
                node.components = components.translate(_POSITION_OF)
                self.rewritten += 1
        elif isinstance(node, A.MemberAccessNode):
            member = node.member
            if (
                isinstance(member, str)
                and _TEXTURE_SWIZZLE.fullmatch(member)
                and self._is_swizzle(node.object_expr, member)
            ):
                node.member = member.translate(_POSITION_OF)
                self.rewritten += 1

    def _is_swizzle(self, object_expr, member):
        if member not in self.member_names:
            return True
        declared = self.scope.node_type(object_expr)
        if declared is not None and getattr(declared, "name", None) in (
            self.scope.structs
        ):
            return False
        return self.scope.infer(object_expr) is not None

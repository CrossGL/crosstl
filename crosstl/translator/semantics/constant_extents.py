"""Array extents named by integer constants.

C-family sources size arrays with integral constant expressions of named constants: an MSL header shared with the
host declares ``static constant constexpr size_t N = X * Y * Z;`` and then ``Data items[N];`` (MSL 4.1 section 2.12
and C++17 [dcl.array]: the bound is a converted constant expression). HLSL does the same with ``static const uint``
and GLSL with ``const int`` (GLSL 4.60 section 4.1.9: the size is a constant integral expression). The CrossGL
generators print an extent as written and declare structures before constants. So a structure member whose extent
names a constant reached HLSL, GLSL and MSL before the constant's declaration, which DXC, glslang and the Metal
compiler reject ("use of undeclared identifier"). SPIR-V, which needs the value, refused it.

:func:`fold_constant_extents` replaces each extent that is an integer expression of the program's integer constants
and enumerators (combined with C's integer operators, as :mod:`.binding_constants` evaluates binding indices) by its
value, for every target. An extent that names anything else is left to the generators: a specialization constant,
which GLSL and SPIR-V keep symbolic and the others refuse, a name the program does not declare, or a name that a
parameter or local declaration could shadow.
"""

from __future__ import annotations

from .. import ast as A
from .binding_constants import _ConstantTable, _refers_to_names, _Unresolved
from .declared_types import _nodes
from .storage_blocks import _declared_names, _walk

__all__ = ["fold_constant_extents"]

_CARRYING = frozenset({"cgl", "crossgl"})


def fold_constant_extents(program, target) -> int:
    """Replace the array extents of ``program`` that name integer constants by their values; returns how many."""
    if str(target or "").lower() in _CARRYING or not isinstance(program, A.ShaderNode):
        return 0
    table = shadowed = None
    changed = 0
    # Type nodes included: the extents sit in declared types, which the rewriting walk skips.
    for node in list(_nodes(program)):
        if not isinstance(node, A.ArrayType):
            continue
        size = getattr(node, "size", None)
        if not isinstance(size, A.ASTNode) or not _refers_to_names(size):
            continue
        if table is None:
            table, shadowed = _ConstantTable(program), _declared_names(program)
        names = {
            item.name for item in _walk(size) if isinstance(item, A.IdentifierNode)
        }
        if names & shadowed:
            continue
        try:
            value = table.evaluate(size)
        except _Unresolved:
            continue
        if value <= 0:
            continue
        node.size = A.LiteralNode(value, A.PrimitiveType("int"))
        changed += 1
    return changed

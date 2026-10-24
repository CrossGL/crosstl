"""Metal stores into components of a vector held in a GLSL buffer block lowered to byte-addressed storage.

The Metal generator lowers a GLSL ``buffer`` block whose layout MSL cannot reproduce to a ``device uchar*`` and each
read of a vector member to scalar loads, ``float4(*(device float*)(buf + o), ...)``. A whole-member write stores each
component. A write to some components (``p.color.x = v``, ``p.velocity.yz += v``, ``p.color[i] = v``) had no form of
its own: the generator wrote the load as the target, ``float4(...).x = v``, which the Metal compiler rejects
("expression is not assignable"; sascha-vulkan ``computeparticles/particle.comp``).

:func:`metal_buffer_component_store` stores each written component on its own, as
:mod:`.hlsl_buffer_component_stores` does for HLSL: a 32-bit component ``k`` of a vector at ``offset`` lies at
``offset + 4 * k`` in every block layout (std140, std430 and scalar keep a vector's components contiguous). A
compound assignment reads the component, combines and stores it, and a swizzle that writes several components
evaluates the right-hand side once into a temporary. Components that are not 32 bits wide, and booleans, are left to
the existing paths.
"""

from __future__ import annotations

from ..ast import ArrayAccessNode, MemberAccessNode
from .buffer_block_diagnostics import BufferBlockUnsupportedError
from .glsl_buffer_layout import (
    byte_offset_add,
    byte_offset_expression,
    glsl_buffer_compound_binary_operator,
)

__all__ = ["metal_buffer_component_store"]

_COMPONENTS = {
    **{name: index for index, name in enumerate("xyzw")},
    **{name: index for index, name in enumerate("rgba")},
    **{name: index for index, name in enumerate("stpq")},
}
_WORD_TYPES = frozenset({"float", "int", "uint"})


def _swizzle(member):
    if not isinstance(member, str) or not 1 <= len(member) <= 4:
        return None
    indices = [_COMPONENTS.get(char) for char in member]
    if None in indices or len(set(indices)) != len(indices):
        return None
    return indices


def _vector_access(codegen, base):
    access = codegen.glsl_buffer_block_array_access(base)
    if access is not None:
        return access, access["offset_expr"]
    access = codegen.glsl_buffer_block_member_access(base)
    if access is None or access.get("is_array"):
        return None, None
    return access, access["offset"]


def _component_offsets(codegen, target):
    """``(access, [component offsets])`` for a component write to a buffer vector, or ``(None, None)``."""
    if isinstance(target, MemberAccessNode):
        indices = _swizzle(getattr(target, "member", None))
        if indices is None:
            return None, None
        access, offset = _vector_access(codegen, target.object_expr)
        if access is None:
            return None, None
        if any(index >= access.get("components", 0) for index in indices):
            return None, None
        return access, [byte_offset_add(offset, 4 * index) for index in indices]
    if isinstance(target, ArrayAccessNode):
        base = getattr(target, "array_expr", getattr(target, "array", None))
        access, offset = _vector_access(codegen, base)
        if access is None:
            return None, None
        index = codegen.generate_expression(
            getattr(target, "index_expr", getattr(target, "index", None))
        )
        return access, [byte_offset_expression(offset, index, 4)]
    return None, None


def metal_buffer_component_store(codegen, target, rhs, op):
    """Metal statements storing the components ``target`` writes from the rendered ``rhs``, or ``None``."""
    access, offsets = _component_offsets(codegen, target)
    if access is None:
        return None
    if (
        access.get("members")
        or access.get("matrix_columns")
        or access.get("is_array")
        or access.get("components", 1) < 2
        or access.get("component_type") not in _WORD_TYPES
        or access.get("layout_type") != access.get("type")
    ):
        return None
    if access.get("readonly"):
        raise BufferBlockUnsupportedError(
            "metal",
            "store",
            f"{access['buffer']} is a readonly block, and a readonly device buffer cannot be written",
        )
    component_type = access["component_type"]
    binary_op = None
    if op != "=":
        binary_op = glsl_buffer_compound_binary_operator(op, component_type)
        if binary_op is None:
            return codegen.metal_buffer_compound_store_diagnostic(op, access)
    buffer_name = access["buffer"]
    lines = []
    if len(offsets) == 1:
        sources = [f"({rhs})"]
    else:
        temp = codegen.next_metal_temp_variable("component_store")
        lines.append(f"{component_type}{len(offsets)} {temp} = {rhs}")
        sources = [f"{temp}.{'xyzw'[position]}" for position in range(len(offsets))]
    for offset, source in zip(offsets, sources):
        if binary_op is not None:
            current = codegen.metal_scalar_load(component_type, buffer_name, offset)
            source = f"({current} {binary_op} {source})"
        lines.append(
            codegen.metal_scalar_store(component_type, buffer_name, offset, source)
        )
    return "\n".join(lines)

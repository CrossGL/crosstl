"""Metal stores into components of a vector held in a GLSL buffer block lowered to byte-addressed storage.

The Metal generator lowers a GLSL ``buffer`` block whose layout MSL cannot reproduce to a ``device uchar*`` and each
read of a vector member to scalar loads, ``float4(*(device float*)(buf + o), ...)``. A whole-member write stores each
component. A write to some components (``p.color.x = v``, ``p.velocity.yz += v``, ``p.color[i] = v``) had no form of
its own: the generator wrote the load as the target, ``float4(...).x = v``, which the Metal compiler rejects
("expression is not assignable"; sascha-vulkan ``computeparticles/particle.comp``).

:func:`metal_buffer_component_store` stores each written component on its own, as
:mod:`.hlsl_buffer_component_stores` does for HLSL: a component ``k`` of a vector at ``offset`` lies at
``offset + size * k`` in every block layout (std140, std430 and scalar keep a vector's components contiguous), where
``size`` is the component's width in bytes. A compound assignment reads the component, combines and stores it, and
a swizzle that writes several components evaluates the right-hand side once into a temporary. Booleans are left to
the existing paths.
"""

from __future__ import annotations

import re

from ..ast import ArrayAccessNode, MemberAccessNode
from .buffer_block_diagnostics import BufferBlockUnsupportedError
from .glsl_buffer_layout import (
    byte_offset_add,
    byte_offset_expression,
    glsl_buffer_component_size_refusal,
    glsl_buffer_compound_binary_operator,
)

__all__ = [
    "metal_buffer_component_store",
    "metal_buffer_component_type",
    "metal_glsl_buffer_component_refusal",
]

_COMPONENTS = {
    **{name: index for index, name in enumerate("xyzw")},
    **{name: index for index, name in enumerate("rgba")},
    **{name: index for index, name in enumerate("stpq")},
}
# CrossGL component type of a buffer block member -> its MSL scalar (MSL section 2.1); MSL vector names add
# the component count to the same spelling.
_METAL_COMPONENT_TYPES = {
    "float16_t": "half",
    "int8_t": "char",
    "uint8_t": "uchar",
    "int16_t": "short",
    "uint16_t": "ushort",
    "int64_t": "long",
    "uint64_t": "ulong",
}
_METAL_NATIVE_COMPONENT_TYPES = frozenset({"float", "int", "uint", "bool"})
_WORD_TYPES = frozenset({"float", "int", "uint", *_METAL_COMPONENT_TYPES})


def metal_buffer_component_type(component_type):
    """The MSL scalar a buffer block member's ``component_type`` is read and written as.

    MSL has no 64-bit floating-point type (MSL section 2.1), so a ``double`` component, like any other type
    without an MSL scalar, is refused.
    """
    mapped = _METAL_COMPONENT_TYPES.get(component_type)
    if mapped is not None:
        return mapped
    if component_type in _METAL_NATIVE_COMPONENT_TYPES:
        return component_type
    raise BufferBlockUnsupportedError(
        "metal",
        "component type",
        f"MSL has no buffer scalar for a {component_type} component",
    )


def metal_glsl_buffer_component_refusal(member_info):
    """Why a byte-addressed Metal buffer cannot hold ``member_info``, or ``None``.

    MSL addresses device memory in bytes and has 8-, 16-, 32- and 64-bit scalars, so every
    component width a buffer block lays out reads and writes at its own width.
    """
    return glsl_buffer_component_size_refusal(member_info, frozenset({1, 2, 4, 8}))


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
        size = access.get("component_size", 4)
        return access, [byte_offset_add(offset, size * index) for index in indices]
    if isinstance(target, ArrayAccessNode):
        base = getattr(target, "array_expr", getattr(target, "array", None))
        access, offset = _vector_access(codegen, base)
        if access is None:
            return None, None
        index = codegen.generate_expression(
            getattr(target, "index_expr", getattr(target, "index", None))
        )
        return access, [
            byte_offset_expression(offset, index, access.get("component_size", 4))
        ]
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
        # The temporary holds the value in the vector the generator computes the member's type in (narrow
        # integers widen to int or uint), and each component narrows as it is stored.
        carrier = re.sub(r"[234]$", "", access["metal_type"])
        lines.append(f"{carrier}{len(offsets)} {temp} = {rhs}")
        sources = [f"{temp}.{'xyzw'[position]}" for position in range(len(offsets))]
    for offset, source in zip(offsets, sources):
        if binary_op is not None:
            current = codegen.metal_scalar_load(component_type, buffer_name, offset)
            source = f"({current} {binary_op} {source})"
        lines.append(
            codegen.metal_scalar_store(component_type, buffer_name, offset, source)
        )
    return "\n".join(lines)

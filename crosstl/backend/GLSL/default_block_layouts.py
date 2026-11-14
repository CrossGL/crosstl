"""Default block layouts: ``layout(std140, row_major) uniform;`` and ``layout(...) buffer;``.

A layout declaration with no block or variable sets the packing and the matrix order of the uniform or shader storage
blocks declared after it (GLSL 4.60 section 4.4.5). Each qualifier it lists overrides the one of its group that the
previous default held; a block's own qualifiers then override the default, and a member's override the block's. Only
the packing qualifiers (``shared``, ``packed``, ``std140``, ``std430`` and ``scalar`` of GL_EXT_scalar_block_layout)
and the matrix orders (``row_major``, ``column_major``) may be defaults; any other qualifier there is ill-formed.

The importer resolves the defaults in the front end: every block receives the layout it actually has, as if its
declaration spelled it, so the defaults bind the bytes the host writes instead of reaching CrossGL as a statement no
consumer reads. A block's matrix order also applies to each matrix member, to every depth of nesting (GLSL 4.60
section 4.4.5.1); :func:`member_layouts_with_block_matrix_order` places it on the members that do not name their own.
"""

from __future__ import annotations

import re

from crosstl.translator.refusals import OpenGLSyntaxError

__all__ = [
    "BLOCK_INTERFACES",
    "DefaultBlockLayouts",
    "member_layouts_with_block_matrix_order",
]

BLOCK_INTERFACES = ("uniform", "buffer")
PACKING_QUALIFIERS = ("shared", "packed", "std140", "std430", "scalar")
MATRIX_ORDER_QUALIFIERS = ("row_major", "column_major")
_QUALIFIER_GROUPS = (PACKING_QUALIFIERS, MATRIX_ORDER_QUALIFIERS)
_MATRIX_TYPE = re.compile(r"^d?mat[234](x[234])?$")


def _group_of(name):
    for group in _QUALIFIER_GROUPS:
        if name in group:
            return group
    return None


def _override(layout, new_layout):
    """``layout`` with the qualifiers of ``new_layout``, each replacing the member of its group ``layout`` held."""
    merged = dict(layout or {})
    for key, value in (new_layout or {}).items():
        group = _group_of(str(key).lower())
        if group is not None:
            for existing in list(merged):
                if str(existing).lower() in group:
                    del merged[existing]
        merged[key] = value
    return merged


class DefaultBlockLayouts:
    """The default layouts of the uniform and shader storage blocks at the current point of a translation unit."""

    def __init__(self):
        self.defaults = {interface: {} for interface in BLOCK_INTERFACES}

    @staticmethod
    def default_interface(qualifiers, layout):
        """``uniform`` or ``buffer`` when ``layout`` and ``qualifiers`` form a default block layout declaration."""
        if not layout:
            return None
        names = [str(qualifier).lower() for qualifier in qualifiers or []]
        if len(names) == 1 and names[0] in BLOCK_INTERFACES:
            return names[0]
        return None

    def declare(self, interface, layout):
        """Record ``layout(...) <interface>;``, refusing qualifiers no default may hold."""
        invalid = [str(key) for key in layout if _group_of(str(key).lower()) is None]
        if invalid:
            raise OpenGLSyntaxError(
                f"layout({', '.join(invalid)}) cannot be a default {interface} block "
                "layout: only shared, packed, std140, std430, scalar, row_major and "
                "column_major can (GLSL 4.60 section 4.4.5)"
            )
        self.defaults[interface] = _override(self.defaults[interface], layout)

    def block_layout(self, qualifiers, layout):
        """The layout a block of ``qualifiers`` declared with ``layout`` has under the current defaults."""
        names = {str(qualifier).lower() for qualifier in qualifiers or []}
        interfaces = [interface for interface in BLOCK_INTERFACES if interface in names]
        if len(interfaces) != 1:
            return layout
        default = self.defaults[interfaces[0]]
        if not default:
            return layout
        return _override(default, layout)


def _matrix_order(layout):
    for key in layout or {}:
        if str(key).lower() in MATRIX_ORDER_QUALIFIERS:
            return key
    return None


def _holds_matrix(type_name, struct_member_types, seen=()):
    if _MATRIX_TYPE.match(type_name):
        return True
    if type_name in seen:
        return False
    return any(
        _holds_matrix(member_type, struct_member_types, seen + (type_name,))
        for member_type in struct_member_types.get(type_name, ())
    )


def member_layouts_with_block_matrix_order(
    block_layout, members, member_layouts, struct_member_types
):
    """The layouts of ``members`` with the block's ``row_major`` order on matrices that do not name an order.

    ``column_major`` is the order a member without one already has, so only ``row_major`` is placed. Members of a
    struct type that holds a matrix, directly or through a nested struct, receive it too, since it reaches those
    matrices; ``struct_member_types`` maps each struct name to its member type names. Scalar, vector and opaque
    members, and structs of them, are left unchanged because a matrix order does not affect them.
    """
    order = _matrix_order(block_layout)
    if order is None or str(order).lower() != "row_major":
        return list(member_layouts)
    result = []
    for member, member_layout in zip(members, member_layouts):
        member_type = str(getattr(member, "vtype", "") or "")
        affected = _holds_matrix(member_type, struct_member_types)
        if affected and _matrix_order(member_layout) is None:
            member_layout = dict(member_layout or {})
            member_layout[order] = None
        result.append(member_layout)
    return result

"""Refuse a call to an undeclared static member of a concrete struct.

``Owner::name(args)`` whose owner names a struct calls that struct's static member function or constructs
its member type ``name`` ([class.qual], [expr.call]).  Member lowering rewrites every static member function
it materializes to a free helper, and the converter binds those helpers before it maps a call name.  A
qualified call that is still unbound therefore names a member that the struct does not declare, or one that
was never materialized.  Mapping such a name as a namespace-qualified function would drop ``Owner::`` and
emit a call to an undeclared ``name``, so the converter raises instead.
"""

from __future__ import annotations

from typing import Any

from .scoped_member_types import resolve_concrete_scoped_member_type


def unresolved_static_member_call_owner(converter: Any, name: Any) -> tuple | None:
    """Return ``(owner, member)`` when ``name`` calls an unbound member of a struct.

    ``None`` means that ``name`` is unqualified, that its owner is not a struct the converter emits (a
    namespace such as ``metal::``) or has several declarations, or that ``Owner::name`` names a member
    type, which is a constructor.
    """
    text = str(name or "").lstrip(":")
    if "::" not in text or "<" in text:
        return None
    resolved = converter.static_struct_member_owner(text)
    if resolved is None:
        return None
    owner, member = resolved
    if (
        owner not in converter.struct_name_map
        or owner in getattr(converter, "ambiguous_struct_names", ())
        or text in converter.struct_name_map
        or text in getattr(converter, "type_aliases", {})
        or resolve_concrete_scoped_member_type(converter, text) is not None
    ):
        return None
    declaration = getattr(converter, "struct_declarations", {}).get(owner)
    if declaration is None:
        return None
    # A member type, alias or nested struct makes the call a construction.
    if any(
        getattr(alias, "name", None) == member
        for alias in getattr(declaration, "type_aliases", None) or ()
    ):
        return None
    return owner, member

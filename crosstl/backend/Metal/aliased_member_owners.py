"""Owners of member types named through a type alias.

``using kern = Kern<half, 32>;`` followed by ``typename kern::acc_type`` names the member alias ``acc_type`` of the
struct ``kern`` aliases ([dcl.typedef]/2: a typedef-name is a synonym for its type, so ``kern::`` looks the member
up in ``Kern<half, 32>``). The dependent member resolution matched owners by struct name only, so a member named
through a global or block-scope alias reached CrossGL as the bare, undeclared ``acc_type``.
"""

from __future__ import annotations

import re

__all__ = ["aliased_member_owner"]

_PLAIN_NAME = re.compile(r"[A-Za-z_]\w*")


def aliased_member_owner(codegen, owner):
    """The type ``owner`` names when it is a plain type alias, else ``owner`` unchanged."""
    text = re.sub(r"^typename\s+", "", str(owner or "").strip())
    if not _PLAIN_NAME.fullmatch(text):
        return owner
    if text not in codegen.type_aliases and text not in codegen.local_type_alias_names:
        return owner
    aliased = str(codegen.resolve_type_alias(text) or "").strip()
    if not aliased or aliased == text or aliased.endswith(("*", "&")):
        return owner
    return aliased


def resolve_member_alias_type(codegen, owner, member, required):
    """Resolve ``owner::member``, also when ``owner`` names a struct declared after the use.

    Template materialization appends the concrete struct ``Kern_half_32`` it creates for ``Kern<half, 32>`` after
    the functions that use it, while member lookup only sees declarations before the use. A valid source cannot use
    a struct before declaring it, so when no earlier declaration resolves the member, the later one is the
    instantiation the use names ([temp.inst]/1).
    """
    owner = aliased_member_owner(codegen, owner)
    resolved = codegen.resolve_dependent_alias_type(owner, member, False)
    if resolved is not None or codegen.metal_later_declarations_visible:
        return resolved
    codegen.metal_later_declarations_visible = True
    try:
        return codegen.resolve_dependent_alias_type(owner, member, required)
    finally:
        codegen.metal_later_declarations_visible = False

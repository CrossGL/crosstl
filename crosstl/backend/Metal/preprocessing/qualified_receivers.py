"""Receivers declared with a namespace-qualified struct type, as in ``ops::ReduceOp op; op();``.

Struct-method lowering knows each concrete struct by its unqualified name, because the importer emits every
namespace member at program scope. A receiver declared as ``ops::ReduceOp``, ``::ops::ReduceOp`` or, inside
``namespace outer``, as ``inner::ReduceOp`` names the same struct as an unqualified ``ReduceOp`` does when exactly one
concrete struct of that name exists and the qualification names its namespace: a qualified name is looked up in
the namespace the qualifier nominates ([basic.lookup.qual]), so ``ops::ReduceOp`` cannot name ``alt::ReduceOp``.

When several namespaces declare a struct of the same name, the unqualified name no longer identifies one struct,
so the receiver stays unresolved and method lowering fails closed with its typed diagnostic instead of binding the
call to either struct.
"""

from __future__ import annotations

import re
from typing import Iterable, Optional, Sequence, Tuple

__all__ = ["qualified_struct_receiver"]

_QUALIFIED = re.compile(r"(?:::)?(?:[A-Za-z_]\w*::)+[A-Za-z_]\w*")


def qualified_struct_receiver(
    declared_type: str,
    struct_names: Iterable[str],
    definitions: Sequence[Tuple[str, str]],
    lexical_namespace: str = "",
) -> Optional[str]:
    """The unqualified struct name a namespace-qualified ``declared_type`` names, or ``None``.

    ``definitions`` holds the ``(name, qualified_name)`` of every concrete struct in the unit, and
    ``lexical_namespace`` the namespace that encloses the declaration (``"a::b"``, or ``""`` at global scope).
    """
    spelled = re.sub(r"\s+", "", declared_type or "")
    if _QUALIFIED.fullmatch(spelled) is None:
        return None
    name = spelled.rsplit("::", 1)[1]
    if name not in set(struct_names):
        return None
    owners = [qualified for short, qualified in definitions if short == name]
    if len(owners) != 1:
        return None
    owner = owners[0].lstrip(":")
    if spelled.startswith("::"):
        return name if owner == spelled[2:] else None
    # An unanchored qualifier is looked up from the enclosing namespace outwards.
    scopes = lexical_namespace.split("::") if lexical_namespace else []
    for depth in range(len(scopes), -1, -1):
        prefix = "::".join(scopes[:depth])
        if owner == (f"{prefix}::{spelled}" if prefix else spelled):
            return name
    return None

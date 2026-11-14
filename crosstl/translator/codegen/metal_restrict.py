"""Restricted buffer pointers in Metal output.

CrossGL marks a buffer whose memory no other binding aliases with ``restrict``: the GLSL qualifier (GLSL 4.60
section 4.10) on a ``buffer`` pointer or block, or ``@restrict`` on a structured buffer, as the SPIR-V and GLSL
importers write ``Restrict`` (SPIR-V 1.6 section 3.20). MSL, being based on C++, spells the promise ``__restrict``
after the ``*`` of the pointer, and the Metal compiler marks such a kernel argument ``noalias`` in AIR. The Metal
generator dropped it, so the output lost the aliasing guarantee.

:func:`metal_restrict_suffix` gives the suffix for the pointer that declares ``node``. An array of buffer pointers
takes none: ``array<T*, N>`` carries no qualifier on its elements.
"""

from __future__ import annotations

__all__ = ["metal_restrict_suffix"]


def _named_restrict(values):
    for value in values or []:
        name = getattr(value, "kind", None) or getattr(value, "name", None) or value
        if str(name).lower() == "restrict":
            return True
    return False


def metal_restrict_suffix(node, raw_type=None):
    """`` __restrict`` when ``node`` (a variable or parameter) or its pointer type ``raw_type`` is restricted."""
    if node is None:
        return ""
    restricted = (
        _named_restrict(getattr(node, "qualifiers", None))
        or _named_restrict(getattr(node, "attributes", None))
        or _named_restrict(getattr(raw_type, "resource_qualifiers", None))
    )
    return " __restrict" if restricted else ""

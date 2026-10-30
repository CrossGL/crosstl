"""Declaration qualifiers that decorate a stage interface value without naming it.

CrossGL writes a stage value's semantic and its qualifiers as attributes in any order: ``vec4 p @ gl_Position
@ invariant`` and ``vec4 p @ invariant @ gl_Position`` declare the same output, and the GLSL importer writes the
second form. A generator that takes the first attribute it does not recognise as the semantic must skip these,
or the qualifier is taken for a user semantic and the built-in is lost.
"""

from __future__ import annotations

__all__ = [
    "INTERFACE_QUALIFIER_ATTRIBUTES",
    "has_invariant_attribute",
    "is_interface_qualifier_attribute",
]

INTERFACE_QUALIFIER_ATTRIBUTES = frozenset(
    {
        # Invariance (GLSL 4.60 section 4.8.1) and precision.
        "invariant",
        "precise",
        "lowp",
        "mediump",
        "highp",
        # Interpolation and auxiliary storage (GLSL 4.60 sections 4.3.6 and 4.5).
        "flat",
        "smooth",
        "noperspective",
        "centroid",
        "sample",
    }
)


def is_interface_qualifier_attribute(attr) -> bool:
    """Whether ``attr`` (an attribute node or a name) is a qualifier rather than a semantic."""
    return str(getattr(attr, "name", attr)).lower() in INTERFACE_QUALIFIER_ATTRIBUTES


def has_invariant_attribute(node) -> bool:
    """Whether ``node`` carries the ``invariant`` qualifier as an attribute or a qualifier."""
    names = {str(q).lower() for q in getattr(node, "qualifiers", None) or ()}
    names.update(
        str(getattr(attr, "name", "")).lower()
        for attr in getattr(node, "attributes", None) or ()
    )
    return "invariant" in names

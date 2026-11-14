"""Resource memory qualifiers that HLSL cannot express, refused instead of dropped.

GLSL's ``volatile`` (GLSL 4.60 section 4.10), SPIR-V's ``Volatile`` decoration and MSL's ``volatile`` buffer pointers
require every access to reach memory, since another agent may change it at any time. HLSL has no such qualifier on a
resource: DXC rejects ``volatile`` on a global ("'volatile' is not a valid modifier for a global variable"), and
``globallycoherent`` makes accesses visible across the device without stopping the compiler from merging or removing
repeated ones, so a spin-wait on a flag could become an infinite loop. Coherence wider than the device (MSL's
``coherent(system)``) has no HLSL qualifier either: ``globallycoherent`` covers the device. The DirectX generator
dropped both without a diagnostic.

:func:`hlsl_unrepresentable_memory_qualifier` gives the refusal reason for a resource declaration that carries one,
for the generator to record through its lowering refusals.
"""

from __future__ import annotations

from .memory_qualifier_names import memory_qualifier_name

__all__ = ["hlsl_unrepresentable_memory_qualifier"]


def _declared_qualifiers(node):
    values = list(getattr(node, "qualifiers", None) or [])
    values.extend(getattr(node, "resource_qualifiers", None) or [])
    declared_type = getattr(node, "var_type", None) or getattr(node, "param_type", None)
    while getattr(declared_type, "element_type", None) is not None:
        declared_type = declared_type.element_type
    values.extend(getattr(declared_type, "resource_qualifiers", None) or [])
    names = {memory_qualifier_name(value) for value in values}
    names.update(
        str(getattr(attribute, "name", "")).lower()
        for attribute in getattr(node, "attributes", None) or []
    )
    return names


def hlsl_unrepresentable_memory_qualifier(node):
    """The reason HLSL cannot declare resource ``node``'s memory qualifiers, or ``None``."""
    names = _declared_qualifiers(node)
    resource = getattr(node, "name", None) or "<unnamed>"
    if "volatile" in names:
        return (
            f"resource '{resource}' is volatile: HLSL has no volatile resource qualifier "
            "(DXC rejects 'volatile' on a global), and globallycoherent does not keep the "
            "compiler from merging or removing repeated accesses"
        )
    if {"coherent(system)", "coherent(host)"} & names:
        return (
            f"resource '{resource}' is coherent at system scope: HLSL's globallycoherent "
            "makes a UAV coherent across the device only"
        )
    return None

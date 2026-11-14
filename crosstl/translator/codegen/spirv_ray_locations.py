"""Ray payload and callable data named by location, resolved to their SPIR-V variables.

GL_EXT_ray_tracing names the payload of ``traceRayEXT`` and the callable data of ``executeCallableEXT`` by the
``location`` of a ``rayPayloadEXT`` or ``callableDataEXT`` variable, and the
GLSL and SPIR-V importers write CrossGL's ``TraceRay`` and ``CallShader`` the same way: ``TraceRay(..., 0)`` with
``vec4 payload @location(0) @rayPayloadEXT``. SPIR-V's ``OpTraceRayKHR`` and ``OpExecuteCallableKHR`` take the variable
itself (SPV_KHR_ray_tracing), and the SPIR-V emitter refused the location as "not an addressable value", so no GLSL or
SPIR-V ray generation, closest-hit or miss shader that traces a ray or calls a callable shader reached SPIR-V.

:func:`ray_location_variable` gives the identifier of the module's variable at the location an integer literal names,
in the storage class the instruction takes; the emitter then passes that variable as it passes an HLSL-style payload.
"""

from __future__ import annotations

from .. import ast as A

__all__ = ["ray_location_variable", "ray_variable_location"]


def _integer(value):
    text = str(value).strip().lower().rstrip("u")
    try:
        return int(text, 0)
    except ValueError:
        return None


def ray_variable_location(node):
    """The ``@location(n)`` of a declaration, or ``None``."""
    for attribute in getattr(node, "attributes", None) or []:
        if str(getattr(attribute, "name", "")).lower() != "location":
            continue
        arguments = getattr(attribute, "arguments", None) or []
        if arguments:
            value = getattr(arguments[0], "value", arguments[0])
            return _integer(value)
    return None


def ray_location_variable(expr, locations, storage_class):
    """An identifier for the ``storage_class`` variable at the location ``expr`` names, or ``None``.

    ``locations`` maps ``(storage class, location)`` to the names of the variables declared there. ``expr`` names a
    location when it is an integer literal; any other expression is left to the caller, as is a location no variable
    or more than one variable of the program occupies (the caller refuses both).
    """
    if not isinstance(expr, A.LiteralNode):
        return None
    location = _integer(expr.value)
    if location is None:
        return None
    names = locations.get((storage_class, location)) or []
    return A.IdentifierNode(names[0]) if len(names) == 1 else None

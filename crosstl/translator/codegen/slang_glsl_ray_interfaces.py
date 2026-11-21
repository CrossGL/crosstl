"""GLSL-style ray tracing interfaces in Slang output, rewritten into the DXR form Slang declares.

The GLSL and SPIR-V importers write a ray tracing stage's payload, hit attributes and callable data as module-scope
variables marked ``@rayPayloadEXT``, ``@rayPayloadInEXT``, ``@hitAttributeEXT``, ``@callableDataEXT`` and
``@callableDataInEXT``, and ``ReportHit(t, kind)`` without the attributes (GL_EXT_ray_tracing,
:mod:`crosstl.translator.semantics.hlsl_ray_interfaces`). Slang takes the HLSL forms: a hit stage receives the payload
and the attributes as entry parameters, ``TraceRay`` takes a ``RayDesc`` and the payload, ``CallShader`` the callable
data and ``ReportHit`` the attributes. The Slang generator copied the GLSL form, so slangc rejected a write to the
interface variable (a non-``static`` global is a uniform, error E30011) and ``ReportHit`` without its attribute
argument (error E30439).

:func:`lower_slang_glsl_ray_interfaces` runs the HLSL generator's rewrite,
:func:`~crosstl.translator.semantics.hlsl_ray_interfaces.lower_glsl_ray_interfaces`, on a program bound for Slang. What
that rewrite refuses (the object and world transform matrices, ``gl_CullMaskEXT``, interface variables shared by
several stages) is refused for Slang with the same reason.
"""

from __future__ import annotations

from ..refusals import SlangRefusal
from ..semantics.hlsl_ray_interfaces import (
    RayInterfaceUnsupportedError,
    lower_glsl_ray_interfaces,
)

__all__ = ["lower_slang_glsl_ray_interfaces"]


def lower_slang_glsl_ray_interfaces(program):
    """Rewrite the GLSL-style ray tracing interfaces of ``program`` into their Slang (DXR) form, in place."""
    try:
        lower_glsl_ray_interfaces(program)
    except RayInterfaceUnsupportedError as error:
        detail = error.reason.removeprefix(f"DirectX {error.stage} stage: ")
        raise SlangRefusal(f"Slang {error.stage} stage: {detail}") from error
    return program

"""Source-path values of GLSL specialization constants that have no default.

A specialization constant without a default (an MSL ``[[function_constant]]`` declared without a value) must be
given its value when the pipeline is created. GLSL's ``layout(constant_id)`` exists only when the shader is
compiled to SPIR-V (GLSL 4.60 section 4.4.5, GL_ARB_gl_spirv). A driver that compiles the source has no
specialization step, so the generated source reads the value from a macro the host defines after ``#version``,
the way OpenGL programs select shader variants. Without the macro the source stops at an ``#error`` that names it.
"""

from __future__ import annotations

__all__ = ["glsl_specialization_macro_name"]

_PREFIX = "CROSSGL_SPECIALIZE_"


def glsl_specialization_macro_name(name: str) -> str:
    """The macro that supplies the source-path value of the specialization constant ``name``."""
    return _PREFIX + str(name)

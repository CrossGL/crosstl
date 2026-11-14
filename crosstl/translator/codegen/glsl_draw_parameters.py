"""The draw parameters ``gl_BaseVertex``, ``gl_BaseInstance`` and ``gl_DrawID`` in OpenGL GLSL.

CrossGL spells the draw parameters as GLSL 4.60 does (section 7.1.1, vertex shader inputs). They became core in
GLSL 4.60; before it they exist only as ``gl_BaseVertexARB``, ``gl_BaseInstanceARB`` and ``gl_DrawIDARB`` of
GL_ARB_shader_draw_parameters ("New Built-in Variables"), with the same values. The GLSL generator emits
``#version 450 core`` by default and copied the 4.60 names into it, which glslang rejects as undeclared.

:func:`vertex_builtin_spelling` gives the name a vertex shader at the emitted version declares: the core name from
4.60 on, the suffixed name before it, whose extension :mod:`.glsl_builtin_extensions` then requires. Other stages
use :func:`draw_parameter_spelling`: mesh and task shaders read ``gl_DrawID`` too (Vulkan ``DrawIndex``), and
before 4.60 glslang declares it there only as ``gl_DrawIDARB``. GLSL ES has no draw parameters, so the name stays
as written there and the GLSL ES emitters refuse it.
"""

from __future__ import annotations

from .glsl_instance_index import (
    INSTANCE_INDEX,
    desktop_glsl_version,
    instance_index_expression,
)

__all__ = ["DRAW_PARAMETERS", "draw_parameter_spelling", "vertex_builtin_spelling"]

DRAW_PARAMETERS = ("gl_BaseVertex", "gl_BaseInstance", "gl_DrawID")

# The first desktop GLSL version that declares the draw parameters without the extension suffix.
_CORE_DRAW_PARAMETERS_VERSION = 460


def draw_parameter_spelling(codegen, name):
    """The OpenGL spelling of the draw parameter ``name`` for the version ``codegen`` emits."""
    if name not in DRAW_PARAMETERS:
        return name
    version = desktop_glsl_version(codegen)
    if version is None or version >= _CORE_DRAW_PARAMETERS_VERSION:
        return name
    return f"{name}ARB"


def vertex_builtin_spelling(codegen, name):
    """The OpenGL expression for the CrossGL vertex input built-in ``name``, or ``name`` when it needs none."""
    if name == INSTANCE_INDEX:
        return instance_index_expression(codegen)
    return draw_parameter_spelling(codegen, name)

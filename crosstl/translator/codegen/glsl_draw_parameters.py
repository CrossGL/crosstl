"""``gl_BaseVertex``, ``gl_BaseInstance`` and ``gl_DrawID`` for OpenGL GLSL.

GLSL 4.60 section 7.1.1 declares these draw-parameter built-ins by name. Before 4.60 OpenGL exposes them
only through GL_ARB_shader_draw_parameters, whose spelling carries an ``ARB`` suffix (``gl_BaseVertexARB``,
``gl_BaseInstanceARB``, ``gl_DrawIDARB``). Emitting the unsuffixed name for a pre-4.60 profile yields a shader
glslang rejects as an undeclared identifier, so :func:`draw_parameter_expression` spells each one for the version
``codegen`` emits and :mod:`.glsl_builtin_extensions` requires the extension the suffixed spelling needs. GLSL ES
has none of them; :mod:`.glsl_target_builtins` refuses them for the WebGL target instead.
"""

from __future__ import annotations

from .glsl_instance_index import desktop_glsl_version

__all__ = ["DRAW_PARAMETER_BUILTINS", "draw_parameter_expression"]

DRAW_PARAMETER_BUILTINS = ("gl_BaseVertex", "gl_BaseInstance", "gl_DrawID")


def draw_parameter_expression(codegen, name):
    """The OpenGL spelling of a draw-parameter built-in for the version ``codegen`` emits."""
    version = desktop_glsl_version(codegen)
    if version is None or version >= 460:
        return name
    return f"{name}ARB"

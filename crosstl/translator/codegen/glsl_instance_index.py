"""``gl_InstanceIndex`` for OpenGL GLSL.

``gl_InstanceIndex`` is the Vulkan GLSL instance index: it starts at the draw's first instance (GL_KHR_vulkan_glsl,
"gl_InstanceIndex"), as MSL's ``[[instance_id]]`` does. OpenGL GLSL has no such variable. Its ``gl_InstanceID``
starts at 0 for every draw, and the base instance of ``glDrawArraysInstancedBaseInstance`` and its siblings is
``gl_BaseInstance`` (GLSL 4.60 section 7.1.1, or ``gl_BaseInstanceARB`` from GL_ARB_shader_draw_parameters before
4.60). The GLSL generator copied ``gl_InstanceIndex`` into OpenGL output, which glslang rejects as undeclared.

For desktop GLSL, :func:`instance_index_expression` spells it ``(gl_InstanceID + gl_BaseInstance)``. Before 4.60 it
uses ``gl_BaseInstanceARB``, whose extension :mod:`.glsl_builtin_extensions` requires. GLSL ES has no base
instance: OpenGL ES 3.2 draws cannot set one, and EXT_base_instance leaves ``gl_InstanceID`` unchanged without
exposing the base to shaders. There ``gl_InstanceIndex`` is ``gl_InstanceID``, which equals it for every draw that the
core API can issue.
"""

from __future__ import annotations

__all__ = ["INSTANCE_INDEX", "instance_index_expression"]

INSTANCE_INDEX = "gl_InstanceIndex"


def _desktop_version(codegen):
    version_line = getattr(codegen, "current_glsl_version_line", None)
    if codegen.glsl_version_is_es(version_line):
        return None
    return codegen.glsl_version_number(version_line) or 450


def instance_index_expression(codegen):
    """The OpenGL spelling of ``gl_InstanceIndex`` for the version ``codegen`` emits."""
    version = _desktop_version(codegen)
    if version is None:
        return "gl_InstanceID"
    base = "gl_BaseInstance" if version >= 460 else "gl_BaseInstanceARB"
    return f"(gl_InstanceID + {base})"

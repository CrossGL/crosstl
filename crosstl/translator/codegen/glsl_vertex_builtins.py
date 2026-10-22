"""Vulkan GLSL's vertex-stage instance index in OpenGL GLSL and GLSL ES.

Vulkan GLSL (GL_KHR_vulkan_glsl) replaces ``gl_InstanceID`` with ``gl_InstanceIndex``, which counts from the draw's
first instance. OpenGL's ``gl_InstanceID`` counts from 0 whatever the draw's base instance is, and the base instance
is ``gl_BaseInstanceARB`` (GL_ARB_shader_draw_parameters, core in OpenGL 4.6 as ``gl_BaseInstance``). So
desktop GLSL reads ``gl_InstanceIndex`` as ``(gl_InstanceID + gl_BaseInstanceARB)``, and
:mod:`glsl_builtin_extensions` enables the extension. OpenGL ES 3.x and WebGL 2 have no draw call that sets a base
instance, except through EXT_base_instance and its WebGL counterpart, so there ``gl_InstanceIndex`` is
``gl_InstanceID``.

``gl_VertexIndex`` needs no such sum: OpenGL's ``gl_VertexID`` already includes the draw's first vertex and base
vertex.
"""

from __future__ import annotations

__all__ = ["glsl_instance_index"]


def glsl_instance_index(codegen) -> str:
    """The GLSL expression for Vulkan's ``gl_InstanceIndex`` in the program ``codegen`` is emitting."""
    if codegen.glsl_version_is_es(getattr(codegen, "current_glsl_version_line", None)):
        return "gl_InstanceID"
    return "(gl_InstanceID + gl_BaseInstanceARB)"

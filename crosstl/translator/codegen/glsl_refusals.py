"""Typed refusals for constructs the OpenGL and WebGL emitters cannot lower.

The GLSL emitter used to write ``/* unsupported GLSL ...: reason */`` or ``/* GLSL wave intrinsic diagnostic */``
next to a zero value, an empty statement or a dropped call and return the program. Every such site now records its
reason through :meth:`~.lowering_refusals.LoweringRefusalMixin.refuse_lowering`, the shared image-access helpers
record through :func:`~.lowering_refusals.refuse_unsupported`, and generation raises the emitter's typed refusal
listing all of them. WebGL output is GLSL ES rendered by a subclass of the GLSL emitter, so it refuses with its
own code.
"""

from __future__ import annotations

from ..diagnostics import CODE_PREFIX
from .lowering_refusals import LoweringRefused

__all__ = ["OpenGLLoweringRefused", "WebGLLoweringRefused"]


class OpenGLLoweringRefused(LoweringRefused):
    """One or more constructs in a program have no faithful OpenGL GLSL lowering."""

    target_name = "opengl"
    label = "OpenGL"
    aliases = ("GLSL",)
    code = f"{CODE_PREFIX}opengl-lowering-unsupported"


class WebGLLoweringRefused(OpenGLLoweringRefused):
    """One or more constructs in a program have no faithful WebGL 2 GLSL ES lowering."""

    target_name = "webgl"
    label = "WebGL"
    aliases = ()
    code = f"{CODE_PREFIX}webgl-lowering-unsupported"

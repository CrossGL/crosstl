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

__all__ = [
    "OpenGLLoweringRefused",
    "input_attachment_refusal_reason",
    "WebGLLoweringRefused",
    "patch_parameter_refusal_reason",
    "stageless_entry_refusal_reason",
]


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


def stageless_entry_refusal_reason(func, return_type_name):
    """Return why a stage-less ``main`` has no GLSL form, or ``None`` when it is a valid ``void main()``.

    GLSL 4.60 section 6.1 requires ``main`` to take no arguments and return ``void``. A source entry point keeps
    its parameters and result only while its stage is unknown, because only a stage says which of them become
    ``in``/``out`` variables or built-ins; without one the interface cannot be lowered.
    """
    if getattr(func, "name", None) != "main":
        return None
    parameters = list(
        getattr(func, "parameters", None) or getattr(func, "params", None) or []
    )
    returns_value = (return_type_name or "void") != "void"
    if not parameters and not returns_value:
        return None
    signature = []
    if parameters:
        signature.append(f"{len(parameters)} parameter(s)")
    if returns_value:
        signature.append(f"a '{return_type_name}' result")
    return (
        "entry point 'main' has no shader stage, so its "
        + " and ".join(signature)
        + " cannot be lowered to the stage interface GLSL requires (GLSL 4.60 section 6.1: main takes no "
        "arguments and returns void)"
    )


PATCH_PARAMETER_TYPES = frozenset({"inputpatch", "outputpatch"})


def patch_parameter_refusal_reason(parameter_name, type_name):
    """Return why a tessellation patch parameter has no GLSL form, or ``None`` for any other parameter.

    GLSL tessellation stages read control points through ``gl_in[]`` and per-vertex input arrays (GLSL 4.60
    sections 7.1.3 and 7.1.4); a patch is not a type that a stage variable can be declared with.
    """
    base = (type_name or "").split("<", 1)[0].strip().lower()
    if base not in PATCH_PARAMETER_TYPES:
        return None
    return (
        f"tessellation patch parameter '{parameter_name}' of type '{type_name}' has no GLSL stage variable "
        "form: GLSL reads control points through gl_in[] and per-vertex input arrays"
    )


def input_attachment_refusal_reason(node):
    """Return why an input attachment has no OpenGL form, or ``None`` for other globals.

    ``subpassInput`` and ``subpassLoad`` exist only in Vulkan GLSL (GL_KHR_vulkan_glsl section 4.1.7); OpenGL and
    OpenGL ES have no render-pass input attachments. Emitting the binding as a sampler compiles to a call with no
    matching overload, so the construct is refused instead; SPIR-V output keeps it as an input attachment.
    """
    for attribute in getattr(node, "attributes", None) or []:
        if str(getattr(attribute, "name", "")).lower() == "input_attachment_index":
            name = getattr(node, "name", None) or "<unnamed>"
            return (
                f"input attachment '{name}': OpenGL has no subpass inputs (subpassInput exists only in Vulkan "
                "GLSL, GL_KHR_vulkan_glsl); translate to SPIR-V for Vulkan"
            )
    return None

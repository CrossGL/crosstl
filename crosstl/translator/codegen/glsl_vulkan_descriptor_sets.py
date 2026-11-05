"""Vulkan descriptor sets in GLSL output.

GLSL for Vulkan (GL_KHR_vulkan_glsl, "Descriptor Sets") places a resource in descriptor set ``S`` with
``layout(set = S, binding = B)``. OpenGL GLSL has no ``set`` qualifier and one flat array of binding points per
resource class, so the generator gives each resource of another set an OpenGL binding point
(:mod:`.glsl_descriptor_sets`). GLSL output wrote only that binding point, so a Vulkan compilation of it (glslang
``-V`` defines ``VULKAN``) bound every resource in set 0.

A resource whose set is not 0 now takes its binding from a function-like macro that the output defines after its
``#version`` and ``#extension`` lines, so one source serves both APIs:

.. code-block:: glsl

    #ifdef VULKAN
    #define CROSSGL_BINDING(s, vulkan, opengl) set = s, binding = vulkan
    #else
    #define CROSSGL_BINDING(s, vulkan, opengl) binding = opengl
    #endif
    layout(std140, CROSSGL_BINDING(1, 3, 3)) uniform Params { float scale; };

Resources of set 0 keep ``binding = B``, which both compilations read alike. The definitions stay short enough
for one line each: GLSL before 4.20 and GLSL ES 3.00 have no line continuation.
"""

from __future__ import annotations

import re

__all__ = [
    "DESCRIPTOR_BINDING_MACRO",
    "descriptor_binding_macro_definition",
    "descriptor_binding_qualifier",
    "opengl_binding_pattern",
    "opengl_binding_value",
]

DESCRIPTOR_BINDING_MACRO = "CROSSGL_BINDING"
_PARAMETERS = "s, vulkan, opengl"


def descriptor_binding_qualifier(
    descriptor_set: int, vulkan_binding: int, opengl_binding: int
) -> str:
    """The layout qualifier of a resource in ``descriptor_set``, for a Vulkan and an OpenGL compilation."""
    return (
        f"{DESCRIPTOR_BINDING_MACRO}({descriptor_set}, {vulkan_binding}, "
        f"{opengl_binding})"
    )


def descriptor_binding_macro_definition() -> str:
    """The ``#ifdef VULKAN`` definitions of :data:`DESCRIPTOR_BINDING_MACRO`."""
    macro = f"{DESCRIPTOR_BINDING_MACRO}({_PARAMETERS})"
    return (
        "#ifdef VULKAN\n"
        f"#define {macro} set = s, binding = vulkan\n"
        "#else\n"
        f"#define {macro} binding = opengl\n"
        "#endif\n"
    )


def opengl_binding_pattern() -> str:
    """A regular expression for the OpenGL binding point of a layout, ``binding = N`` or the macro's last argument."""
    return (
        rf"(?:\bbinding\s*=\s*(\d+)|\b{DESCRIPTOR_BINDING_MACRO}\(\s*\d+\s*,\s*\d+\s*,"
        r"\s*(\d+)\s*\))"
    )


def opengl_binding_value(match: re.Match) -> int:
    """The binding point a match of :func:`opengl_binding_pattern` names."""
    return int(match.group(1) or match.group(2))

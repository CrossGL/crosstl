"""Metal attributes for CrossGL's conservative depth layouts.

CrossGL spells GLSL's conservative depth layout (GL_ARB_conservative_depth, GLSL 4.20 section 4.4.2.3) as an
attribute next to the fragment depth output: ``out float gl_FragDepth @depth_greater;`` at stage level, or
``float depth @gl_FragDepth @depth_greater;`` in an output structure. MSL states the same promise with the
argument of the depth attribute on a fragment function output (Metal Shading Language Specification 3.2, section
5.2.3.5): ``[[depth(greater)]]``, ``[[depth(less)]]`` or ``[[depth(any)]]``.

- ``depth_greater`` is ``depth(greater)`` and ``depth_less`` is ``depth(less)``;
- ``depth_any`` makes no promise, which is ``depth(any)``;
- ``depth_unchanged`` promises the depth the rasterizer computed. MSL has no argument for that promise, so the
  output is ``depth(any)``: the shader writes the same value, and only the early depth test the promise allows is
  lost. HLSL output does the same with ``SV_Depth``.

The generator gave a stage-level depth output with a layout the user attribute ``[[attribute(n)]]``, and Metal
rejects a fragment function that returns that structure; an output structure member kept ``depth(any)`` and dropped
the promise. A depth layout on anything other than the fragment depth output has no Metal meaning and is refused.
"""

from __future__ import annotations

from ..refusals import MetalRefusal

__all__ = ["metal_conservative_depth_semantic"]

_DEPTH_ARGUMENTS = {
    "depth_any": "any",
    "depth_unchanged": "any",
    "depth_greater": "greater",
    "depth_less": "less",
}
_DEPTH_NAMES = frozenset({"gl_fragdepth", "sv_depth"})


def _attribute_name(attribute):
    return str(getattr(attribute, "name", "") or "")


def metal_conservative_depth_semantic(node):
    """``depth(any|greater|less)`` for ``node``'s depth layout attribute, or ``None`` when it carries none."""
    attributes = list(getattr(node, "attributes", None) or [])
    layouts = sorted(
        {
            _attribute_name(attribute).lower()
            for attribute in attributes
            if _attribute_name(attribute).lower() in _DEPTH_ARGUMENTS
        }
    )
    if not layouts:
        return None
    name = str(getattr(node, "name", "") or "")
    if len(layouts) > 1:
        raise MetalRefusal(
            f"Metal cannot give '{name}' the conservative depth layouts {', '.join(layouts)} at once: "
            "the depth attribute of a Metal fragment output states one"
        )
    semantic = str(getattr(node, "semantic", "") or "")
    named = {name.lower(), semantic.lower()} | {
        _attribute_name(attribute).lower() for attribute in attributes
    }
    if not named & _DEPTH_NAMES:
        raise MetalRefusal(
            f"Metal cannot apply the conservative depth layout '{layouts[0]}' to '{name}': "
            "MSL states it only through the depth attribute of the fragment depth output"
        )
    return f"depth({_DEPTH_ARGUMENTS[layouts[0]]})"

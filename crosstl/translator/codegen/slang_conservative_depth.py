"""Slang semantics for CrossGL's conservative depth layouts.

CrossGL spells GLSL's conservative depth layout (GL_ARB_conservative_depth, GLSL 4.20 section 4.4.2.3) as an
attribute next to the fragment depth output: ``out float gl_FragDepth @depth_greater;`` at stage level, or
``float depth @gl_FragDepth @depth_greater;`` in an output structure. Slang states the promise as HLSL does, with the
depth semantic itself: ``SV_DepthGreaterEqual`` and ``SV_DepthLessEqual``, which slangc writes as the
``DepthGreater`` and ``DepthLess`` execution modes for SPIR-V. ``depth_any`` makes no promise, and
``depth_unchanged`` promises the depth the rasterizer computed, which Slang has no semantic for; both are
``SV_Depth``, which writes the same value without the promise.

The generator used to take the layout attribute for the semantic (``float gl_FragDepth : depth_greater``), which
slangc reads as a user varying, so the fragment depth was never written. A depth layout on anything other than the
fragment depth output, or two layouts on one output, are refused.
"""

from __future__ import annotations

from ..refusals import SlangRefusal

__all__ = ["SLANG_DEPTH_SEMANTICS", "slang_conservative_depth_semantic"]

_DEPTH_LAYOUT_SEMANTICS = {
    "depth_any": "SV_Depth",
    "depth_unchanged": "SV_Depth",
    "depth_greater": "SV_DepthGreaterEqual",
    "depth_less": "SV_DepthLessEqual",
}
# Every Slang semantic of the fragment depth output, upper-cased.
SLANG_DEPTH_SEMANTICS = frozenset(
    {"SV_DEPTH", "SV_DEPTHGREATEREQUAL", "SV_DEPTHLESSEQUAL"}
)
_DEPTH_NAMES = frozenset({"gl_fragdepth", "sv_depth"})


def _attribute_name(attribute):
    return str(getattr(attribute, "name", "") or "")


def slang_conservative_depth_semantic(node):
    """The Slang depth semantic ``node``'s depth layout attribute names, or ``None`` when it carries none."""
    attributes = list(getattr(node, "attributes", None) or [])
    layouts = sorted(
        {
            _attribute_name(attribute).lower()
            for attribute in attributes
            if _attribute_name(attribute).lower() in _DEPTH_LAYOUT_SEMANTICS
        }
    )
    if not layouts:
        return None
    name = str(getattr(node, "name", "") or "")
    if len(layouts) > 1:
        raise SlangRefusal(
            f"Slang cannot give '{name}' the conservative depth layouts {', '.join(layouts)} at once: "
            "the depth semantic states one"
        )
    semantic = str(getattr(node, "semantic", "") or "")
    named = {name.lower(), semantic.lower()} | {
        _attribute_name(attribute).lower() for attribute in attributes
    }
    if not named & _DEPTH_NAMES:
        raise SlangRefusal(
            f"Slang cannot apply the conservative depth layout '{layouts[0]}' to '{name}': "
            "Slang states it only through the fragment depth output's semantic"
        )
    return _DEPTH_LAYOUT_SEMANTICS[layouts[0]]

"""HLSL semantics for CrossGL's conservative depth layouts.

CrossGL spells GLSL's conservative depth layout (GL_ARB_conservative_depth, GLSL 4.20 section 4.4.2.3) as an
attribute next to the fragment depth output: ``out float gl_FragDepth @depth_greater;`` at stage level, or
``float depth @gl_FragDepth @depth_greater;`` in an output structure. HLSL states the same promise with the depth
semantic itself (Shader Model 5, "Conservative Depth Output"):

- ``depth_greater`` is ``SV_DepthGreaterEqual`` and ``depth_less`` is ``SV_DepthLessEqual``; DXC lowers them to
  the ``DepthGreater`` and ``DepthLess`` execution modes in SPIR-V;
- ``depth_any`` makes no promise and ``depth_unchanged`` promises the depth the rasterizer computed, which HLSL has
  no semantic for; both are ``SV_Depth``, which writes the same value without the promise.

A depth layout on anything other than the fragment depth output has no HLSL meaning and is refused.
"""

from __future__ import annotations

from ..refusals import DirectXRefusal

__all__ = ["hlsl_conservative_depth_semantic"]

_DEPTH_SEMANTICS = {
    "depth_any": "SV_Depth",
    "depth_unchanged": "SV_Depth",
    "depth_greater": "SV_DepthGreaterEqual",
    "depth_less": "SV_DepthLessEqual",
}
_DEPTH_NAMES = {"gl_fragdepth", "sv_depth"}


def _attribute_name(attribute):
    return str(getattr(attribute, "name", "") or "")


def hlsl_conservative_depth_semantic(node):
    """The HLSL depth semantic ``node``'s depth layout attribute names, or ``None`` when it carries none."""
    attributes = list(getattr(node, "attributes", None) or [])
    layouts = [
        _attribute_name(attribute).lower()
        for attribute in attributes
        if _attribute_name(attribute).lower() in _DEPTH_SEMANTICS
    ]
    if not layouts:
        return None
    name = str(getattr(node, "name", "") or "")
    if len(set(layouts)) > 1:
        raise DirectXRefusal(
            f"DirectX cannot give '{name}' the conservative depth layouts "
            f"{', '.join(sorted(set(layouts)))} at once: HLSL's depth semantic states one"
        )
    semantic = str(getattr(node, "semantic", "") or "")
    named = {name.lower(), semantic.lower()} | {
        _attribute_name(attribute).lower() for attribute in attributes
    }
    if not named & _DEPTH_NAMES:
        raise DirectXRefusal(
            f"DirectX cannot apply the conservative depth layout '{layouts[0]}' to '{name}': "
            "HLSL states it only through the fragment depth output's semantic"
        )
    return _DEPTH_SEMANTICS[layouts[0]]

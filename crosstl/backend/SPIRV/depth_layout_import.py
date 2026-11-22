"""Conservative depth execution modes in the SPIR-V importer.

A fragment entry point that writes ``BuiltIn FragDepth`` may promise how the depth moves with ``DepthGreater``,
``DepthLess`` or ``DepthUnchanged`` (SPIR-V 1.6 section 3.6). CrossGL spells the promise as an attribute next to the
fragment depth output, as GLSL's ``layout(depth_greater) out float gl_FragDepth;`` does, so the importer writes
``float gl_FragDepth @output @gl_FragDepth @depth_greater;``.
"""

from __future__ import annotations

__all__ = ["fragment_depth_layout_attribute"]

_DEPTH_LAYOUTS = {
    "DepthGreater": "@depth_greater",
    "DepthLess": "@depth_less",
    "DepthUnchanged": "@depth_unchanged",
}


def fragment_depth_layout_attribute(function):
    """The CrossGL depth layout attribute the execution modes of entry point ``function`` declare, or ``None``."""
    for mode in getattr(function, "spirv_execution_modes", None) or ():
        attribute = _DEPTH_LAYOUTS.get(mode.get("mode"))
        if attribute is not None:
            return attribute
    return None

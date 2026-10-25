"""Colour attachments named by ``gl_FragData[N]`` on CrossGL fragment outputs.

HLSL names colour attachment ``N`` with ``SV_Target<N>``, and the HLSL importer spells it ``@ gl_FragData[N]`` on
the fragment output (a member of the returned structure or an ``out`` parameter). Each target places that output at
attachment ``N``: GLSL ``layout(location = N)`` (GLSL 4.60 section 4.4.2), WGSL ``@location(N)`` (WGSL section
13.3.1.4), MSL ``[[color(N)]]`` and HLSL or Slang ``SV_Target<N>``. Placing outputs in declaration order instead
sends ``SV_Target2`` declared first to attachment 0.
"""

from __future__ import annotations

__all__ = ["frag_data_attachment_index", "frag_data_attribute_index"]


def frag_data_attribute_index(attribute):
    """``N`` of a ``gl_FragData[N]`` ``attribute``, or ``None`` for any other attribute."""
    if str(getattr(attribute, "name", "")) != "gl_FragData":
        return None
    arguments = getattr(attribute, "arguments", None) or []
    if len(arguments) != 1:
        return None
    value = getattr(arguments[0], "value", arguments[0])
    if isinstance(value, bool):
        return None
    try:
        index = int(str(value), 0)
    except ValueError:
        return None
    return index if index >= 0 else None


def frag_data_attachment_index(node):
    """``N`` of the ``@ gl_FragData[N]`` attribute on ``node``, or ``None`` when it has none."""
    for attribute in getattr(node, "attributes", None) or []:
        if str(getattr(attribute, "name", "")) == "gl_FragData":
            return frag_data_attribute_index(attribute)
    return None

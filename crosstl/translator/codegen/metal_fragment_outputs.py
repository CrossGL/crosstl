"""Colour attachments of Metal fragment output structures.

A fragment shader writes colour attachment ``N`` through an output at ``layout(location = N)`` (GLSL, SPIR-V) or
``SV_Target<N>`` (HLSL); CrossGL keeps the location attribute or spells the attachment ``gl_FragData[N]``. MSL
names attachment ``N`` with ``[[color(N)]]`` on a member of the structure a fragment function returns (MSL
specification section 5.2.3.5, "Fragment Function Output Attributes"). The emitter gave such members the vertex
input attribute ``[[attribute(N)]]`` instead, and Metal rejects a fragment function returning that structure.
:func:`metal_fragment_output_color_semantic` gives the member its ``color(N)``.
"""

from __future__ import annotations

import re

__all__ = ["metal_fragment_output_color_semantic"]

_FRAG_DATA = re.compile(r"^gl_FragData\[(\d+)\]$")


def metal_fragment_output_color_semantic(codegen, member):
    """``color(N)`` for a fragment output member placed by location or ``gl_FragData[N]``, else ``None``."""
    location = codegen.explicit_location_attribute_index(member)
    if location is None:
        location = _frag_data_index(codegen, member)
    if location is None:
        return None
    return f"color({location})"


def _frag_data_index(codegen, member):
    for attribute in getattr(member, "attributes", None) or []:
        if str(getattr(attribute, "name", "")) != "gl_FragData":
            continue
        arguments = getattr(attribute, "arguments", None) or []
        if len(arguments) == 1:
            return codegen.binding_index_value(arguments[0])
    match = _FRAG_DATA.match(
        str(codegen.semantic_from_node(member) or "").replace(" ", "")
    )
    return int(match.group(1)) if match else None

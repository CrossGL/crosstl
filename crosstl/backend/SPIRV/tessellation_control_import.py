"""CrossGL attributes for the primitive-generator modes of a SPIR-V tessellation control entry point.

SPIR-V lets either tessellation stage declare the domain (``Triangles``, ``Quads``, ``Isolines``), the spacing,
the vertex order and ``PointMode`` (SPIR-V specification, Execution Mode). DXC writes a hull shader's modes on
the ``TessellationControl`` entry point. CrossGL spells them on a ``tessellation_control`` stage as the hull
attributes HLSL uses (``@domain``, ``@partitioning``, ``@outputtopology``), which the HLSL and SPIR-V
generators read from that stage.
"""

from __future__ import annotations

from typing import Iterable, Mapping

TESSELLATION_CONTROL_DOMAINS = {
    "Triangles": "tri",
    "Quads": "quad",
    "Isolines": "isoline",
}

TESSELLATION_CONTROL_PARTITIONING = {
    "SpacingEqual": "integer",
    "SpacingFractionalEven": "fractional_even",
    "SpacingFractionalOdd": "fractional_odd",
}

TESSELLATION_CONTROL_WINDINGS = {
    "VertexOrderCw": "triangle_cw",
    "VertexOrderCcw": "triangle_ccw",
}


def _first_mode(mode_names: list[str], table: Mapping[str, str]) -> str | None:
    for mode_name in mode_names:
        if mode_name in table:
            return table[mode_name]
    return None


def tessellation_control_output_topology(mode_names: list[str]) -> str | None:
    """Return the HLSL ``outputtopology`` that the control entry point's modes state, if any.

    The vertex order orders the vertices of generated triangles only, so it has no effect on points
    (``PointMode``) or on isolines (Vulkan specification, Tessellation chapter). HLSL names one topology
    for those cases, ``point`` or ``line``, and a winding stated next to them is not carried.
    """
    if "PointMode" in mode_names:
        return "point"
    winding = _first_mode(mode_names, TESSELLATION_CONTROL_WINDINGS)
    if winding is None:
        return None
    if "Isolines" in mode_names:
        return "line"
    return winding


def tessellation_control_mode_attributes(
    execution_modes: Iterable[Mapping],
) -> list[str]:
    """Return the CrossGL attributes for the domain, spacing, winding and point mode of a control entry point.

    Only the modes the entry point declares are spelled; the evaluation stage supplies the others.
    """
    mode_names = [str(mode.get("mode")) for mode in execution_modes or []]
    attributes = []
    domain = _first_mode(mode_names, TESSELLATION_CONTROL_DOMAINS)
    if domain is not None:
        attributes.append(f'@domain("{domain}")')
    partitioning = _first_mode(mode_names, TESSELLATION_CONTROL_PARTITIONING)
    if partitioning is not None:
        attributes.append(f'@partitioning("{partitioning}")')
    topology = tessellation_control_output_topology(mode_names)
    if topology is not None:
        attributes.append(f'@outputtopology("{topology}")')
    return attributes

"""CrossGL spelling of the primitive-generator modes of a SPIR-V tessellation evaluation entry point.

SPIR-V lets either tessellation stage declare the domain, the spacing, the vertex order and ``PointMode``, and a
mode one stage leaves out is the one the other stage declares (SPIR-V specification, Execution Mode; Vulkan
specification, Tessellation chapter). DXC writes a domain shader's ``[domain]`` as the evaluation entry's only mode
and leaves the rest to the hull's control entry.

CrossGL has two spellings for the evaluation stage's modes. A GLSL input layout such as ``layout(triangles) in;``
means what GLSL means: a spacing or vertex order it leaves out is ``equal_spacing`` or ``ccw`` (GLSL 4.60 section
4.4.1.2). The HLSL-style entry attributes ``@domain``, ``@partitioning`` and ``@outputtopology`` state only what
they name and leave the rest to the control stage, as HLSL does. An entry point that states its domain, spacing
and vertex order keeps the layout; one that leaves the spacing or the vertex order out takes the attributes, so no
output invents the modes the module did not declare.
"""

from __future__ import annotations

from typing import Iterable, Mapping

from .tessellation_control_import import (
    TESSELLATION_CONTROL_DOMAINS,
    TESSELLATION_CONTROL_PARTITIONING,
    TESSELLATION_CONTROL_WINDINGS,
    tessellation_control_mode_attributes,
)


def _mode_names(execution_modes: Iterable[Mapping] | None) -> list[str]:
    return [str(mode.get("mode")) for mode in execution_modes or []]


def evaluation_leaves_modes_to_control_stage(
    execution_modes: Iterable[Mapping] | None,
) -> bool:
    """Whether a SPIR-V evaluation entry states its domain but leaves its spacing or vertex order out."""
    names = set(_mode_names(execution_modes))
    if not names & TESSELLATION_CONTROL_DOMAINS.keys():
        return False
    return not (
        names & TESSELLATION_CONTROL_PARTITIONING.keys()
        and names & TESSELLATION_CONTROL_WINDINGS.keys()
    )


def tessellation_evaluation_mode_attributes(
    execution_modes: Iterable[Mapping] | None,
) -> list[str]:
    """Return the HLSL-style attributes of an evaluation entry that leaves modes to its control stage.

    An entry that states every mode GLSL would otherwise default keeps its input layout and gets no attributes.
    """
    if not evaluation_leaves_modes_to_control_stage(execution_modes):
        return []
    return tessellation_control_mode_attributes(execution_modes)

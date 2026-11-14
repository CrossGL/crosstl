"""HLSL system-value semantics: the ones Direct3D defines, the semantic indices each takes, and the GLSL built-in each names.

CrossGL accepts an HLSL system-value semantic as a built-in value attribute (``@SV_RenderTargetArrayIndex``).
:mod:`.builtin_values` gave the GLSL, SPIR-V and Metal generators the GLSL name of fourteen of them, written without
a semantic index. Every other ``SV_`` spelling reached the generators, which declared a value they did not list as a
user varying or vertex attribute: GLSL wrote ``out float extra;`` for ``@SV_ClipDistance0`` and
``flat out uint layer;`` for ``@SV_RenderTargetArrayIndex``, SPIR-V gave both a ``Location``, GLSL and Metal read a
vertex input ``@SV_ViewID`` from vertex attribute 1, and Metal declared a fragment output ``@SV_StencilRef`` as
``[[attribute(0)]]``. HLSL kept each spelling, so only the HLSL output wrote the system value. A spelling Direct3D does
not define (``@SV_Fancy``, ``@SV_Position1``, ``@SV_Target8``), which DXC rejects, became a user varying in the same way.

:data:`SYSTEM_VALUES` lists DXIL's semantic kinds (DXC ``DxilConstants.h``, ``SemanticKind``; HLSL "Semantics",
System-Value Semantics). Each entry records:

* the highest semantic index Direct3D accepts: 7 for ``SV_Target`` (eight render targets), 1 for
  ``SV_Barycentrics`` (one per interpolation mode), any for the clip and cull distances, which pack into two
  four-component registers, and 0 for every other value. DXC 1.9 rejects ``SV_Position1``, ``SV_Depth1``,
  ``SV_StencilRef1``, ``SV_RenderTargetArrayIndex1``, ``SV_ViewportArrayIndex1``, ``SV_ShadingRate1``,
  ``SV_Barycentrics2`` and ``SV_Target8``. It ignores the index of a few inputs it keeps out of the signature, such as
  ``SV_ViewID1``; those spellings are refused too, because no specification gives them a meaning;
* the GLSL built-in the value names for the GLSL-name targets (GLSL 4.60 section 7.1, and the extensions listed in
  :mod:`.builtin_names`), or ``None`` for a value whose ``SV_`` spelling later passes and the generators read: the
  tessellation and geometry values and the mesh culling flag.

A value's GLSL name can depend on the declaration:

* ``SV_Target`` names ``gl_FragColor`` and ``SV_TargetN`` colour attachment ``N``, ``gl_FragData[N]``, as the HLSL
  importer spells them. The GLSL generator gave an ``SV_TargetN`` output the next free location, so
  ``@SV_Target3`` wrote attachment 0;
* ``SV_ClipDistanceN`` and ``SV_CullDistanceN`` name CrossGL's indexed distance members, ``gl_ClipDistance`` for
  index 0 and ``gl_ClipDistanceN`` otherwise (:mod:`..codegen.stage_distance_layout`);
* ``SV_Barycentrics`` names ``gl_BaryCoordNoPerspEXT`` on a ``noperspective`` value and ``gl_BaryCoordEXT``
  otherwise, as DXC's SPIR-V back end chooses ``BaryCoordNoPerspKHR`` or ``BaryCoordKHR``;
* ``SV_ShadingRate`` is the rate a fragment was shaded at as a fragment input, ``gl_ShadingRateEXT``, and the rate
  a primitive requests as a vertex, geometry or mesh output, ``gl_PrimitiveShadingRateEXT``
  (GL_EXT_fragment_shading_rate). On a structure used both ways only the SPIR-V generator, which resolves the
  semantic per stage, keeps it.

``SV_DepthGreaterEqual`` and ``SV_DepthLessEqual`` name ``gl_FragDepth``: they promise the depth moves only one way,
which lets the rasterizer keep early depth testing, and writing the same depth without that promise gives the same
result. ``SV_InnerCoverage`` names ``gl_FullyCoveredEXT`` (SPV_EXT_fragment_fully_covered), which DXC's SPIR-V back
end reads for it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..refusals import TranslationRefusal

__all__ = [
    "SYSTEM_VALUES",
    "SystemValue",
    "SystemValueError",
    "glsl_system_value_name",
    "lookup_system_value",
]


class SystemValueError(TranslationRefusal):
    """An ``SV_`` spelling Direct3D does not define, or with a semantic index the value does not take."""

    code = "project.translate.directx-system-value-invalid"


@dataclass(frozen=True)
class SystemValue:
    name: str
    # The highest semantic index Direct3D accepts, or None for any.
    max_index: int | None
    # The GLSL built-in for GLSL-name targets, or None to keep the SV_ spelling.
    glsl: str | None


_ANY = None
_KEEP = None

SYSTEM_VALUES = {
    value.name.lower(): value
    for value in (
        SystemValue("SV_VertexID", 0, "gl_VertexID"),
        SystemValue("SV_InstanceID", 0, "gl_InstanceID"),
        SystemValue("SV_Position", 0, "gl_Position"),
        SystemValue("SV_RenderTargetArrayIndex", 0, "gl_Layer"),
        SystemValue("SV_ViewportArrayIndex", 0, "gl_ViewportIndex"),
        SystemValue("SV_ClipDistance", _ANY, "gl_ClipDistance"),
        SystemValue("SV_CullDistance", _ANY, "gl_CullDistance"),
        SystemValue("SV_OutputControlPointID", 0, _KEEP),
        SystemValue("SV_DomainLocation", 0, _KEEP),
        SystemValue("SV_PrimitiveID", 0, "gl_PrimitiveID"),
        SystemValue("SV_GSInstanceID", 0, _KEEP),
        SystemValue("SV_SampleIndex", 0, "gl_SampleID"),
        SystemValue("SV_IsFrontFace", 0, "gl_FrontFacing"),
        SystemValue("SV_Coverage", 0, "gl_SampleMask"),
        SystemValue("SV_InnerCoverage", 0, "gl_FullyCoveredEXT"),
        SystemValue("SV_Target", 7, "gl_FragColor"),
        SystemValue("SV_Depth", 0, "gl_FragDepth"),
        SystemValue("SV_DepthLessEqual", 0, "gl_FragDepth"),
        SystemValue("SV_DepthGreaterEqual", 0, "gl_FragDepth"),
        SystemValue("SV_StencilRef", 0, "gl_FragStencilRefEXT"),
        SystemValue("SV_DispatchThreadID", 0, "gl_GlobalInvocationID"),
        SystemValue("SV_GroupID", 0, "gl_WorkGroupID"),
        SystemValue("SV_GroupIndex", 0, "gl_LocalInvocationIndex"),
        SystemValue("SV_GroupThreadID", 0, "gl_LocalInvocationID"),
        SystemValue("SV_TessFactor", 0, _KEEP),
        SystemValue("SV_InsideTessFactor", 0, _KEEP),
        SystemValue("SV_ViewID", 0, "gl_ViewIndex"),
        SystemValue("SV_Barycentrics", 1, "gl_BaryCoordEXT"),
        SystemValue("SV_ShadingRate", 0, "gl_ShadingRateEXT"),
        SystemValue("SV_CullPrimitive", 0, _KEEP),
        SystemValue("SV_StartVertexLocation", 0, "gl_BaseVertex"),
        SystemValue("SV_StartInstanceLocation", 0, "gl_BaseInstance"),
    )
}

# A trailing index belongs to the semantic, not its name (HLSL "Semantics": "SV_Target5"); no Direct3D name ends in
# a digit. HLSL semantics are case-insensitive, and the generators also read CrossGL's snake-case spellings
# (``@sv_group_thread_id``), so case and underscores after the prefix are ignored.
_SPELLING = re.compile(r"sv_([a-z_]*[a-z])([0-9]*)", re.IGNORECASE)


def lookup_system_value(spelling) -> tuple[SystemValue, int | None] | None:
    """``(value, semantic index)`` for an ``SV_`` spelling, or ``None`` when ``spelling`` is not one.

    The index is ``None`` when the spelling writes none. Raises :class:`SystemValueError` for a name Direct3D does
    not define and for an index the value does not take.
    """
    text = str(spelling or "").strip()
    if not text.lower().startswith("sv_"):
        return None
    match = _SPELLING.fullmatch(text)
    value = (
        SYSTEM_VALUES.get("sv_" + match.group(1).replace("_", "").lower())
        if match
        else None
    )
    if value is None:
        raise SystemValueError(
            f"@{text} is not a Direct3D system value; the HLSL system values are "
            f"{', '.join(sorted(entry.name for entry in SYSTEM_VALUES.values()))}"
        )
    digits = match.group(2)
    index = int(digits) if digits else None
    if index is not None and value.max_index is not None and index > value.max_index:
        accepted = (
            "takes no semantic index"
            if value.max_index == 0
            else f"takes semantic indices 0 to {value.max_index}"
        )
        raise SystemValueError(
            f"@{text}: the Direct3D system value {value.name} {accepted}"
        )
    return value, index


def glsl_system_value_name(
    value, index, *, direction, noperspective, keep_shared_shading_rate
):
    """The GLSL built-in ``value`` names on a declaration, or ``None`` to keep its ``SV_`` spelling.

    ``direction`` is ``"input"``, ``"output"`` or anything else for a declaration used both ways or neither.
    Raises :class:`SystemValueError` for a shading rate on such a declaration unless ``keep_shared_shading_rate``.
    """
    if value.glsl is None:
        return None
    if value.glsl == "gl_FragColor":
        return value.glsl if index is None else f"gl_FragData[{index}]"
    if value.glsl in ("gl_ClipDistance", "gl_CullDistance"):
        return value.glsl if not index else f"{value.glsl}{index}"
    if value.glsl == "gl_BaryCoordEXT":
        return "gl_BaryCoordNoPerspEXT" if noperspective else value.glsl
    if value.glsl == "gl_ShadingRateEXT":
        if direction == "input":
            return "gl_ShadingRateEXT"
        if direction == "output":
            return "gl_PrimitiveShadingRateEXT"
        if keep_shared_shading_rate:
            return None
        raise SystemValueError(
            f"{value.name} is the fragment's shading rate as a fragment input (gl_ShadingRateEXT) and the "
            "primitive's requested rate as an output (gl_PrimitiveShadingRateEXT), and its structure is used as "
            f"{direction or 'neither'}"
        )
    return value.glsl

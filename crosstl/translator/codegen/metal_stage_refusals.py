"""Typed refusals for shader stages the Metal generator cannot emit as Metal entry points.

Metal has no geometry stage: an MSL entry point is a ``vertex``, ``fragment``, ``kernel``, ``visible``,
``intersection``, ``object`` or ``mesh`` function (Metal Shading Language Specification, section 5.1, "Function
qualifiers"). A geometry shader can only be emulated by a different pipeline the host builds, for example an
object and mesh pipeline, so the generator refuses the stage instead of writing a function no pipeline can call.

Metal tessellation is structured differently from HLSL's and GLSL's (MSL section 5.1.5, "Post-tessellation vertex
functions"): tessellation factors and per-patch data are written to buffers by a compute kernel, and the
post-tessellation ``[[patch(...)]] vertex`` function reads them with the control points through ``[[stage_in]]``.
A tessellation evaluation (domain) stage becomes that function (:mod:`.metal_post_tessellation`); until the hull
stage is lowered to the compute kernel it is refused too.
"""

from __future__ import annotations

from ..diagnostics import CODE_PREFIX, TranslationDiagnostic

_REASONS = {
    "geometry": (
        "Metal has no geometry stage: MSL entry points are vertex, fragment, kernel, visible, intersection, "
        "object and mesh functions (Metal Shading Language Specification 5.1); a geometry shader needs a "
        "host-built emulation pipeline"
    ),
    "tessellation_control": (
        "Metal computes tessellation factors in a compute kernel that writes them to a buffer (Metal Shading "
        "Language Specification 5.1.5); lowering a tessellation control (hull) stage to that kernel is not "
        "implemented"
    ),
}

METAL_REFUSED_STAGES = frozenset(_REASONS)

# A stage Metal writes in one form and refuses in another (:mod:`.metal_post_tessellation`).
_FORM_REASONS = {
    "tessellation_evaluation": (
        "Metal evaluates tessellated patches in a [[patch]] post-tessellation vertex function that reads its "
        "control points and per-patch data through [[stage_in]] (Metal Shading Language Specification 5.1.5); "
        "a tessellation evaluation stage that reads gl_in, gl_TessCoord and per-vertex input arrays, as GLSL "
        "writes it, is not lowered to that function yet, only one that takes an OutputPatch and @domain"
    ),
}


class MetalStageUnsupportedError(TranslationDiagnostic):
    """A shader stage the Metal generator cannot write as a Metal entry point."""

    code = CODE_PREFIX + "metal-stage-unsupported"

    def __init__(self, stage: str):
        self.stage = stage
        reason = _REASONS.get(stage) or _FORM_REASONS[stage]
        super().__init__(f"{reason} (stage {stage})", target="metal")


def refuse_metal_stages(stage_names) -> None:
    """Raise :class:`MetalStageUnsupportedError` for the first refused stage in ``stage_names``, sorted."""
    refused = sorted(set(stage_names) & METAL_REFUSED_STAGES)
    if refused:
        raise MetalStageUnsupportedError(refused[0])

"""Primitive-generator modes of a SPIR-V tessellation evaluation entry that does not state its domain.

SPIR-V lets either tessellation stage declare the domain, the spacing, the vertex order and ``PointMode``; a mode
one stage leaves out is the one the other stage declares (SPIR-V specification, Execution Mode, ``Triangles``:
"Only valid with one of the tessellation Execution Models"; Vulkan specification, Tessellation chapter). A
CrossGL evaluation stage that names no domain, neither as a GLSL input layout nor as ``@domain`` and with no
``tessellation_control`` stage in the program that names one, therefore leaves its domain to a control stage
outside the program. The generator used to write ``Triangles`` for it, plus GLSL's ``SpacingEqual`` and
``VertexOrderCcw`` defaults, so a module imported from SPIR-V came back with modes it never had.

GLSL's defaults (GLSL 4.60 section 4.4.1.2) belong to a GLSL evaluation shader, which states its domain in its
input layout. They apply only when the evaluation stage does that. A program that contains both tessellation stages
and states no domain on either describes a pipeline that Vulkan rejects, so it is refused.
"""

from __future__ import annotations

GLSL_DOMAIN_LAYOUT_FLAGS = frozenset({"triangles", "quads", "isolines"})


def evaluation_states_glsl_domain(codegen, stage) -> bool:
    """Whether the evaluation ``stage`` states its domain in a GLSL input layout such as ``layout(quads) in;``."""
    return any(
        flag in GLSL_DOMAIN_LAYOUT_FLAGS
        for flag in codegen.stage_layout_flag_names(stage, "in")
    )


def evaluation_takes_glsl_defaults(codegen, stage) -> bool:
    """Whether an evaluation stage's unstated spacing and vertex order mean ``SpacingEqual`` and ``VertexOrderCcw``.

    Only a GLSL-style stage, one whose input layout states the domain, takes GLSL's defaults. A stage that states
    its domain as ``@domain``, or not at all, leaves what it does not state to the control stage.
    """
    if codegen.tessellation_modes_follow_hlsl(stage):
        return False
    return evaluation_states_glsl_domain(codegen, stage)


def refuse_program_without_domain(codegen, stage, domain_mode) -> None:
    """Refuse a program whose control and evaluation stages both leave the tessellation domain unstated."""
    if domain_mode is not None:
        return
    control = getattr(codegen, "tessellation_control_stage", None)
    if control is None or control is stage:
        return
    codegen.refuse_lowering(
        "SPIR-V tessellation pipeline states no domain: neither the tessellation_evaluation stage nor the "
        "tessellation_control stage declares triangles, quads or isolines (as a GLSL input layout or "
        "@domain), and Vulkan requires one of the two to declare it"
    )

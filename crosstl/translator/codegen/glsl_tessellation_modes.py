"""Tessellation modes a GLSL evaluation shader takes from the program's control stage.

GLSL states the domain, spacing, vertex order and point mode only in the tessellation evaluation shader's input
layout (GLSL 4.60 section 4.4.1.2); a control shader's output layout accepts only ``vertices``. A CrossGL program
may state them on its ``tessellation_control`` entry instead, as HLSL does on the hull shader and as SPIR-V allows
on either stage. Without them, the evaluation shader would silently use GLSL's defaults, ``equal_spacing`` and
``ccw``. This module finds the control entries and resolves the modes they state, so the evaluation layout can fill
in every mode it does not state itself. A mode both stages state must agree, and an evaluation stage that leaves a
mode to a control stage the program does not have is refused.
"""

from __future__ import annotations

from ..refusals import OpenGLRefusal
from .stage_utils import collect_stage_entry_records

_CONTROL_STAGE = "tessellation_control"
_SPACING_NAMES = frozenset(
    {"equal_spacing", "fractional_even_spacing", "fractional_odd_spacing"}
)
_DOMAIN_NAMES = frozenset({"triangles", "quads", "isolines"})


def tessellation_control_mode_sources(ast):
    """Return ``(entry, layout_qualifiers)`` for each tessellation_control entry point of ``ast``.

    The whole program is searched, also when only the evaluation stage is generated, because the modes belong to
    the pipeline rather than to either stage.
    """
    stage_qualifiers = {}
    for stage in (getattr(ast, "stages", None) or {}).values():
        entry = getattr(stage, "entry_point", None)
        if entry is not None:
            stage_qualifiers[id(entry)] = getattr(stage, "layout_qualifiers", None)
    return [
        (func, stage_qualifiers.get(func_id))
        for func_id, stage_name, func in collect_stage_entry_records(
            ast, None, {_CONTROL_STAGE}
        )
        if stage_name == _CONTROL_STAGE
    ]


def remember_scoped_control_sources(codegen, ast, scoped_ast):
    """Keep the control entries of ``ast`` for generating ``scoped_ast``, which entry scoping pruned them from."""
    codegen.glsl_entry_scoped_tessellation_sources = (
        scoped_ast,
        tessellation_control_mode_sources(ast),
    )


def program_control_sources(codegen, ast):
    """Return the control entries for generating ``ast``, including those entry scoping pruned from it."""
    scoped = getattr(codegen, "glsl_entry_scoped_tessellation_sources", None)
    if scoped is not None and scoped[0] is ast:
        return scoped[1]
    return tessellation_control_mode_sources(ast)


def _agreed(values, mode_name):
    """Return the one value the control entries state for ``mode_name``, or ``None`` when none states it."""
    stated = [value for value in values if value is not None]
    for value in stated[1:]:
        if value != stated[0]:
            raise OpenGLRefusal(
                f"Conflicting GLSL tessellation {mode_name} {stated[0]} with {value} "
                "across tessellation_control entry points"
            )
    return stated[0] if stated else None


def _merged(own, inherited, mode_name):
    """Combine a mode the evaluation stage states with the one the control stage states."""
    if own is None:
        return inherited
    if inherited is not None and inherited != own:
        raise OpenGLRefusal(
            f"Conflicting GLSL tessellation {mode_name}: the tessellation_evaluation stage "
            f"states {own} and the tessellation_control stage states {inherited}"
        )
    return own


def control_stage_domain(codegen, sources):
    """Return the GLSL domain layout name the control entries state, or ``None``."""
    return _agreed(
        (
            codegen.glsl_stage_consistent_layout_value(
                "tessellation",
                "domain",
                func,
                explicit_attribute_name="domain",
                bare_names=_DOMAIN_NAMES,
                value_mapper=codegen.glsl_tessellation_domain,
                stage_layout_qualifiers=qualifiers,
                direction="in",
            )
            for func, qualifiers in sources
        ),
        "domain",
    )


def control_stage_spacing(codegen, sources):
    """Return the GLSL spacing layout name the control entries state, or ``None``."""
    return _agreed(
        (
            codegen.glsl_stage_consistent_layout_value(
                "tessellation",
                "partitioning",
                func,
                explicit_attribute_name="partitioning",
                bare_names=_SPACING_NAMES,
                value_mapper=codegen.glsl_tessellation_partitioning,
                stage_layout_qualifiers=qualifiers,
                direction="in",
            )
            for func, qualifiers in sources
        ),
        "partitioning",
    )


def control_stage_order_and_point_mode(codegen, sources, domain):
    """Return the vertex order (``cw``/``ccw`` or ``None``) and point mode the control entries state."""
    orders = []
    point_modes = []
    for func, qualifiers in sources:
        order = codegen.glsl_stage_consistent_layout_value(
            "tessellation",
            "winding",
            func,
            bare_names={"cw", "ccw"},
            stage_layout_qualifiers=qualifiers,
            direction="in",
        )
        topology = codegen.glsl_single_stage_attribute_argument(
            func, "outputtopology", qualifiers
        )
        topology_order, topology_point_mode = (
            codegen.glsl_tessellation_output_topology_layout(topology, domain)
        )
        orders.append(_merged(order, topology_order, "winding"))
        point_mode = topology_point_mode or bool(
            codegen.glsl_stage_bare_attribute(func, {"point_mode"}, qualifiers, "in")
        )
        point_modes.append(True if point_mode else None)
    return _agreed(orders, "winding"), bool(_agreed(point_modes, "point_mode"))


def _states_hlsl_domain(func):
    """Whether the evaluation entry ``func`` states its domain as the HLSL-style ``@domain`` attribute."""
    return any(
        str(getattr(attr, "name", "")).lower() == "domain"
        for attr in getattr(func, "attributes", None) or []
    )


def refuse_unstated_modes(codegen, func, domain, spacing, order, point_mode):
    """Refuse an HLSL-style evaluation entry whose spacing or vertex order no stage of the program states.

    An evaluation stage that states its domain as ``@domain`` leaves the other modes to the control stage, as an
    HLSL domain shader leaves them to its hull and as a SPIR-V evaluation entry may. GLSL would replace a missing
    mode with its default, so the shader would tessellate differently from the pipeline it came from. The vertex
    order is not needed for isolines or points, which it does not affect.
    """
    if not _states_hlsl_domain(func):
        return
    missing = []
    if spacing is None:
        missing.append("spacing (@partitioning)")
    if order is None and not point_mode and domain != "isolines":
        missing.append("vertex order (@outputtopology)")
    if missing:
        codegen.refuse_lowering(
            f"GLSL tessellation_evaluation entry '{getattr(func, 'name', '<unnamed>')}' states its domain "
            f"with @domain and leaves its {' and '.join(missing)} to the tessellation_control stage, but no "
            "tessellation_control stage in the program states it; GLSL would use its default instead; "
            "translate the tessellation stages together"
        )


def refuse_missing_domain(codegen, func, domain):
    """Refuse an evaluation entry for which no stage of the program states the primitive mode.

    A GLSL tessellation evaluation shader must declare ``triangles``, ``quads`` or ``isolines`` in its input
    layout (GLSL 4.60 section 4.4.1.2: at least one evaluation compilation unit must declare a primitive mode), and
    glslang refuses to link one that does not. A CrossGL evaluation stage may leave the domain to a control stage
    outside the program, as a SPIR-V evaluation entry may, but GLSL output has nowhere to take it from.
    """
    if domain is not None:
        return
    codegen.refuse_lowering(
        f"GLSL tessellation_evaluation entry '{getattr(func, 'name', '<unnamed>')}' states no domain "
        "(triangles, quads or isolines) and no tessellation_control stage in the program states one; a GLSL "
        "evaluation shader must declare its primitive mode in its input layout; translate the tessellation "
        "stages together or state the domain"
    )


def merge_control_stage_modes(codegen, func, domain, spacing, order, point_mode):
    """Fill the evaluation stage's unstated tessellation modes from the program's control stage.

    Returns the ``(domain, spacing, order, point_mode)`` the evaluation shader's layout must state.
    """
    sources = getattr(codegen, "glsl_tessellation_control_sources", None) or []
    if sources:
        domain = _merged(domain, control_stage_domain(codegen, sources), "domain")
        spacing = _merged(
            spacing, control_stage_spacing(codegen, sources), "partitioning"
        )
        control_order, control_point_mode = control_stage_order_and_point_mode(
            codegen, sources, domain
        )
        order = _merged(order, control_order, "winding")
        point_mode = point_mode or control_point_mode
    refuse_missing_domain(codegen, func, domain)
    refuse_unstated_modes(codegen, func, domain, spacing, order, point_mode)
    return domain, spacing, order, point_mode

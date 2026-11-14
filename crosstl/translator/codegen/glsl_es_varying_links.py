"""GLSL ES 3.00 varyings linked by name where the source links them by location.

GLSL ES 3.00 accepts ``layout(location)`` only on vertex shader inputs and fragment shader outputs (GLSL ES 3.00
sections 4.3.8.1 and 4.3.8.2); a vertex output reaches the fragment input with the same name. Vulkan GLSL and
SPIR-V link the two stages by location instead, so a vertex output and the fragment input it feeds often have
different names. The WebGL emitter used to write the location anyway, which GLSL ES 3.00 rejects.

:func:`es300_linked_varying_name` names a vertex output or fragment input that has an explicit location after that
location, ``cgl_location_<N>``. Stages translated separately then link exactly as the source's locations did. A
varying with a ``component`` qualifier shares its location with others, and GLSL ES 3.00 cannot name part of a
location, so it is refused, as is a program that already uses the location-derived name for something else.
"""

from __future__ import annotations

from .stage_utils import normalize_stage_name

__all__ = [
    "LINKED_VARYING_PREFIX",
    "es300_linked_varying_name",
    "program_identifier_names",
]

LINKED_VARYING_PREFIX = "cgl_location_"
_NAME_LINKED = frozenset({("vertex", "out"), ("fragment", "in")})


def es300_linked_varying_name(
    codegen, node, direction, stage_name, taken_names, source_name
):
    """The location-derived name of a varying GLSL ES 3.00 links by name, or ``None`` to keep the source name.

    ``node`` is a stage-level ``in``/``out`` variable or a member of a stage input or output structure; only an
    explicit ``location`` counts, since names derived from semantics already agree between the stages.
    """
    if (normalize_stage_name(stage_name), direction) not in _NAME_LINKED:
        return None
    layout = codegen.glsl_variable_layout_prefix(node).strip()
    location = codegen.stage_io_layout_location(layout)
    if location is None:
        return None
    if codegen.stage_io_layout_component(layout) is not None:
        codegen.refuse_lowering(
            f"varying '{source_name}': layout(location = {location}, component) shares its location with other "
            "varyings; GLSL ES 3.00 links varyings by name and cannot name part of a location"
        )
        return None
    name = f"{LINKED_VARYING_PREFIX}{location}"
    if name != source_name and name in taken_names:
        codegen.refuse_lowering(
            f"varying '{source_name}': GLSL ES 3.00 links it by the name '{name}' of its location, which the "
            "program already declares"
        )
        return None
    return name


def program_identifier_names(program):
    """Every name a declaration in ``program`` carries."""
    from ..semantics.rewrite import child_slots, get_child

    names = set()
    stack, seen = [program], set()
    while stack:
        current = stack.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        name = getattr(current, "name", None)
        if isinstance(name, str):
            names.add(name)
        stack.extend(
            get_child(current, field, key) for field, key in child_slots(current)
        )
    return names

"""SPIR-V reads of the tessellation levels into narrower stage values.

Vulkan declares ``BuiltIn TessLevelOuter`` as ``float[4]`` and ``BuiltIn TessLevelInner`` as ``float[2]`` in every
tessellation domain (Vulkan 1.3 section 15.9, VUID-TessLevelOuter-TessLevelOuter-04393 and
VUID-TessLevelInner-TessLevelInner-04397). A tessellation evaluation stage value carries only the levels its domain
uses: HLSL's ``float edges[3] : SV_TessFactor`` and ``float inside : SV_InsideTessFactor`` for a triangle domain,
``float edges[2]`` for isolines. As DXC does, element ``i`` of such a value is element ``i`` of the built-in, so a
narrower array takes the built-in's leading elements and a scalar takes element 0.
"""

from __future__ import annotations

__all__ = ["load_tessellation_level_value"]

_LEVELS = frozenset({"TessLevelOuter", "TessLevelInner"})


def load_tessellation_level_value(codegen, variable, result_type) -> object | None:
    """``variable``, a tessellation level built-in, loaded as ``result_type``.

    ``None`` when ``variable`` is not a tessellation level, or when ``result_type`` is the built-in's own type or
    neither a narrower array of its float elements nor one float, so the caller loads it as before.
    """
    if codegen.builtin_names_by_variable_id.get(variable.id) not in _LEVELS:
        return None
    variable_type = codegen.pointer_pointee_type(variable)
    levels = codegen.array_type_info_from_type(variable_type)
    if levels is None:
        return None
    element_type, size = levels
    count = codegen.array_length_default_value(size, fallback=0)
    wanted = codegen.array_type_info_from_type(result_type)
    if wanted is None:
        if result_type.type.base_type != element_type.type.base_type:
            return None
        loaded = codegen.load_from_variable(variable, variable_type)
        return codegen.composite_extract(loaded, element_type, 0)
    wanted_element, wanted_size = wanted
    wanted_count = codegen.array_length_default_value(wanted_size, fallback=0)
    if (
        wanted_element.type.base_type != element_type.type.base_type
        or not 0 < wanted_count < count
    ):
        return None
    loaded = codegen.load_from_variable(variable, variable_type)
    elements = [
        codegen.composite_extract(loaded, element_type, index)
        for index in range(wanted_count)
    ]
    return codegen.composite_construct(result_type, elements)

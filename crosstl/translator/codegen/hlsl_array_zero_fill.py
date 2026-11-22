"""Statements that zero a fixed-size array an HLSL structure member holds.

HLSL has no array expression, so ``s.w = {float(0), float(0)};`` is not HLSL: DXC reads the brace
list only as a declaration initializer. A structure that a brace list such as ``S s = {};`` leaves
partly or wholly zero gets its omitted array members as element assignments instead, written
element by element for short arrays and as a loop for longer ones. Elements that are structures,
vectors or matrices take the HLSL value-initialized form ``(T)0``.
"""

from __future__ import annotations

__all__ = ("hlsl_zero_array_assignments",)

# Arrays up to this length are zeroed with one assignment per element, longer ones with a loop.
_UNROLLED_EXTENT = 4


def hlsl_zero_array_assignments(codegen, target, array_type, node=None, indent=0):
    """HLSL statements that set every element of the array ``target`` of type ``array_type`` to zero."""
    _mapped_type, extent, element_type = codegen.hlsl_fixed_array_aggregate_contract(
        node, array_type
    )
    indent_str = "    " * indent
    if extent > _UNROLLED_EXTENT:
        index = codegen.next_hlsl_temp_variable("zero_index")
        body = _element_assignments(
            codegen, f"{target}[{index}]", element_type, node, indent + 1
        )
        return (
            f"{indent_str}for (uint {index} = 0u; {index} < {extent}u; ++{index}) {{\n"
            f"{body}{indent_str}}}\n"
        )
    return "".join(
        _element_assignments(
            codegen, f"{target}[{position}]", element_type, node, indent
        )
        for position in range(extent)
    )


def _element_assignments(codegen, target, element_type, node, indent):
    if codegen.hlsl_outer_array_type(element_type) is not None:
        return hlsl_zero_array_assignments(codegen, target, element_type, node, indent)
    zero = codegen.hlsl_value_initialized_expression(node, element_type)
    return f"{'    ' * indent}{target} = {zero};\n"

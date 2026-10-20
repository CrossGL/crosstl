"""Slang lowering of CrossGL projected depth compares (``textureCompareProj*``).

Slang has no projective compare, so the coordinate and the reference are divided by the coordinate's last
component ``q`` and sampled with ``SampleCmp``/``SampleCmpLevel``/``SampleCmpGrad``, as the HLSL and Metal
emitters do.
"""

from .image_access_contracts import projected_texture_compare_reference

# The Slang method and the operand roles each projected compare takes after the coordinate and reference.
SLANG_PROJECTED_COMPARE_FORMS = {
    "textureCompareProj": ("SampleCmp", ()),
    "textureCompareProjOffset": ("SampleCmp", ("offset",)),
    "textureCompareProjLod": ("SampleCmpLevel", ("lod",)),
    "textureCompareProjLodOffset": ("SampleCmpLevel", ("lod", "offset")),
    "textureCompareProjGrad": ("SampleCmpGrad", ("gradient", "gradient")),
    "textureCompareProjGradOffset": (
        "SampleCmpGrad",
        ("gradient", "gradient", "offset"),
    ),
}

_VEC3 = {"vec3", "float3"}
_VEC4 = {"vec4", "float4"}


def is_slang_projected_compare(func_name):
    """Whether ``func_name`` is a projected compare this module lowers."""
    return func_name in SLANG_PROJECTED_COMPARE_FORMS


def vector_component(expression, component):
    """``expression.component``, parenthesizing a compound expression."""
    if all(char.isalnum() or char in "_.[]" for char in expression):
        return f"{expression}.{component}"
    return f"({expression}).{component}"


def slang_projected_compare_coordinate(texture_type, coord_type, coord):
    """The projected coordinate and ``q`` for a shadow resource, or ``None`` for an unsupported shape."""
    if texture_type == "sampler2DShadow" and coord_type in _VEC3 | _VEC4:
        divisor = vector_component(coord, "z" if coord_type in _VEC3 else "w")
        return f"{vector_component(coord, 'xy')} / {divisor}", divisor
    if coord_type not in _VEC4:
        return None
    divisor = vector_component(coord, "w")
    if texture_type == "samplerCubeShadow":
        return f"{vector_component(coord, 'xyz')} / {divisor}", divisor
    if texture_type == "sampler2DArrayShadow":
        layer = vector_component(coord, "z")
        return f"float3({vector_component(coord, 'xy')} / {divisor}, {layer})", divisor
    return None


def _operand_reason(codegen, role, texture_node, operand):
    if role == "offset":
        return codegen.shadow_compare_offset_rank_unsupported_reason(
            texture_node, operand
        )
    if role == "gradient":
        return codegen.shadow_compare_gradient_rank_unsupported_reason(
            texture_node, operand
        )
    return codegen.scalar_numeric_texture_argument_unsupported_reason(
        operand, "lod argument"
    )


def generate_slang_projected_compare(codegen, func_name, args):
    """Emit a Slang projected compare, or a typed refusal marker."""
    refuse = codegen.unsupported_texture_compare_call
    compare_args = codegen.texture_compare_args(func_name, args)
    if compare_args is None:
        return refuse(func_name, "requires texture, coordinate, and compare arguments")
    texture_name, coord, compare, extra_args = compare_args
    if not codegen.is_shadow_compare_resource(args[0]):
        return refuse(func_name, "requires a shadow sampler resource")

    coord_index = codegen.sampled_texture_coord_index(args)
    texture_type = codegen.resource_base_type(codegen.get_expression_type(args[0]))
    coord_type = codegen.resource_base_type(
        codegen.expression_result_type(args[coord_index])
    )
    projected = slang_projected_compare_coordinate(texture_type, coord_type, coord)
    if projected is None:
        return refuse(
            func_name,
            "requires sampler2DShadow vec3/vec4, samplerCubeShadow vec4 or "
            "sampler2DArrayShadow vec4 projection coordinates",
        )
    compare_reason = codegen.compare_reference_unsupported_reason(args[coord_index + 1])
    if compare_reason:
        return refuse(func_name, compare_reason)
    expected_reason = codegen.resource_query_expected_type_unsupported_reason(
        func_name, "float"
    )
    if expected_reason:
        return refuse(
            func_name,
            expected_reason,
            codegen.zero_value_for_type(codegen.current_expression_expected_type),
        )

    method, roles = SLANG_PROJECTED_COMPARE_FORMS[func_name]
    if len(extra_args) != len(roles):
        return refuse(
            func_name, f"requires {len(roles)} argument(s) after the reference"
        )
    for role, operand in zip(roles, extra_args):
        reason = _operand_reason(codegen, role, args[0], operand)
        if reason:
            return refuse(func_name, reason)

    projected_coord, divisor = projected
    reference = projected_texture_compare_reference(compare, divisor)
    operands = [codegen.generate_expression(operand) for operand in extra_args]
    return codegen.slang_texture_method_call(
        texture_name, method, args, projected_coord, reference, *operands
    )

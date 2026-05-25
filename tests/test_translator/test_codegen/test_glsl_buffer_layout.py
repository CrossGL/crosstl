from types import SimpleNamespace

from crosstl.translator.ast import (
    ArrayType,
    AttributeNode,
    MatrixType,
    ParameterNode,
    PrimitiveType,
    StructMemberNode,
    StructNode,
    VariableNode,
)
from crosstl.translator.codegen.glsl_buffer_layout import (
    byte_offset_add,
    byte_offset_expression,
    collect_lowered_glsl_buffer_blocks,
    glsl_buffer_compound_binary_operator,
    glsl_buffer_block_node_type,
    matrix_column_offsets,
    std430_array_stride,
    std430_layout_type_name,
    std430_value_type_info,
    vector_component_offsets,
)


def convert_type_node_to_string(type_node):
    if hasattr(type_node, "name"):
        return type_node.name
    if hasattr(type_node, "rows") and hasattr(type_node, "cols"):
        if type_node.rows == type_node.cols:
            return f"float{type_node.rows}x{type_node.rows}"
        return f"float{type_node.cols}x{type_node.rows}"
    if hasattr(type_node, "element_type") and hasattr(type_node, "size"):
        element_type = convert_type_node_to_string(type_node.element_type)
        return (
            f"{element_type}[]"
            if type_node.size is None
            else f"{element_type}[{type_node.size}]"
        )
    return str(type_node)


def primitive(name):
    return PrimitiveType(name)


def member(name, type_node):
    return StructMemberNode(name, type_node)


def block_struct(*members):
    return StructNode("Block", list(members))


def block_var(*, layout="std430", qualifiers=None, var_type=None):
    return VariableNode(
        "block",
        var_type or primitive("Block"),
        qualifiers=qualifiers or [],
        attributes=[
            AttributeNode("glsl_buffer_block", [SimpleNamespace(value=layout)]),
        ],
    )


def collect_for(struct, var=None, *, literal_int_value=None):
    var = var or block_var()
    return collect_lowered_glsl_buffer_blocks(
        [var],
        structs_by_name={"Block": struct},
        is_glsl_buffer_block_variable=lambda node: True,
        resource_base_type=lambda vtype: getattr(vtype, "name", str(vtype)),
        glsl_buffer_block_layout=lambda node: node.attributes[0].arguments[0].value,
        convert_type_node_to_string=convert_type_node_to_string,
        literal_int_value=literal_int_value
        or (lambda value: value if isinstance(value, int) else None),
        map_type=lambda type_name: f"mapped_{type_name}",
        target_type_key="target_type",
        unsupported_type_message="type is not supported by test lowering",
    )


def test_std430_type_info_covers_vec3_and_non_square_matrix_strides():
    vec3 = std430_value_type_info("vec3")
    assert vec3["size"] == 12
    assert vec3["align"] == 16
    assert std430_array_stride(vec3) == 16

    mat2x3 = std430_value_type_info("mat2x3")
    assert mat2x3["matrix_columns"] == 2
    assert mat2x3["matrix_rows"] == 3
    assert mat2x3["column_stride"] == 16
    assert mat2x3["size"] == 32

    float3x2_alias = std430_value_type_info("float3x2")
    assert float3x2_alias == mat2x3


def test_std430_type_info_normalizes_fixed_width_scalar_aliases():
    assert std430_layout_type_name("int8_t") == "int"
    assert std430_layout_type_name("uint16_t") == "uint"
    assert std430_layout_type_name("size_t") == "uint"
    assert std430_value_type_info("int64_t") == std430_value_type_info("int")
    assert std430_value_type_info("uint64_t") == std430_value_type_info("uint")


def test_byte_offset_expression_formats_literal_and_dynamic_indices():
    assert byte_offset_expression(16, "2", 16) == "48"
    assert byte_offset_expression(0, "i", 64) == "(i * 64)"
    assert byte_offset_expression(16, "i", 64) == "(16 + i * 64)"
    assert byte_offset_expression(16, "vertex_id", 16) == "(16 + vertex_id * 16)"


def test_byte_offset_expression_parenthesizes_complex_indices():
    assert byte_offset_expression(0, "i + 1", 16) == "((i + 1) * 16)"
    assert byte_offset_expression(8, "i + 1", 16) == "(8 + (i + 1) * 16)"
    assert byte_offset_expression(8, "(i + 1)", 16) == "(8 + (i + 1) * 16)"
    assert byte_offset_expression(8, "min(i, 3)", 16) == "(8 + (min(i, 3)) * 16)"


def test_byte_offset_add_flattens_dynamic_offset_expressions():
    assert byte_offset_add("16", 48) == "64"
    assert byte_offset_add("(i * 64)", 48) == "(i * 64 + 48)"
    assert byte_offset_add("(16 + i * 64)", 48) == "(16 + i * 64 + 48)"
    assert byte_offset_add("(16 + i * 64 + 48)", 12) == "(16 + i * 64 + 48 + 12)"
    assert byte_offset_add("((i + 1) * 16)", 4) == "((i + 1) * 16 + 4)"


def test_byte_offset_sequence_helpers_return_indexed_offsets():
    assert vector_component_offsets("(16 + i * 16)", 3) == [
        (0, "(16 + i * 16)"),
        (1, "(16 + i * 16 + 4)"),
        (2, "(16 + i * 16 + 8)"),
    ]
    assert matrix_column_offsets("(i * 64)", 4, 16) == [
        (0, "(i * 64)"),
        (1, "(i * 64 + 16)"),
        (2, "(i * 64 + 32)"),
        (3, "(i * 64 + 48)"),
    ]


def test_glsl_buffer_compound_binary_operator_is_type_aware():
    assert glsl_buffer_compound_binary_operator("+=", "float") == "+"
    assert glsl_buffer_compound_binary_operator("/=", "float") == "/"
    assert glsl_buffer_compound_binary_operator("%=", "float") is None
    assert glsl_buffer_compound_binary_operator("&=", "float") is None
    assert glsl_buffer_compound_binary_operator("%=", "int") == "%"
    assert glsl_buffer_compound_binary_operator("<<=", "uint") == "<<"
    assert glsl_buffer_compound_binary_operator("??=", "uint") is None


def test_glsl_buffer_block_node_type_handles_variable_parameter_and_legacy_nodes():
    variable = VariableNode("block", primitive("Block"))
    parameter = ParameterNode("block", primitive("ParamBlock"))
    legacy = SimpleNamespace(vtype=primitive("LegacyBlock"))

    assert convert_type_node_to_string(glsl_buffer_block_node_type(variable)) == "Block"
    assert (
        convert_type_node_to_string(glsl_buffer_block_node_type(parameter))
        == "ParamBlock"
    )
    assert (
        convert_type_node_to_string(glsl_buffer_block_node_type(legacy))
        == "LegacyBlock"
    )


def test_collect_lowered_glsl_buffer_blocks_lays_out_metadata_fixed_and_runtime_arrays():
    struct = block_struct(
        member("count", primitive("uint")),
        member("normal", primitive("ivec3")),
        member(
            "transforms",
            ArrayType(MatrixType(primitive("float"), rows=2, cols=3), size=2),
        ),
        member("data", ArrayType(primitive("float"), size=None)),
    )

    blocks, var_failures, struct_failures = collect_for(struct)

    assert not var_failures
    assert not struct_failures
    block = blocks["block"]
    assert block["readonly"] is False
    assert block["runtime_array"] == "data"

    assert block["members"]["count"]["offset"] == 0
    assert block["members"]["normal"]["offset"] == 16
    assert block["members"]["normal"]["size"] == 12
    assert block["members"]["normal"]["align"] == 16
    assert block["members"]["normal"]["target_type"] == "mapped_ivec3"

    transforms = block["members"]["transforms"]
    assert transforms["offset"] == 32
    assert transforms["stride"] == 32
    assert transforms["array_count"] == 2
    assert transforms["matrix_columns"] == 2
    assert transforms["matrix_rows"] == 3

    data = block["members"]["data"]
    assert data["offset"] == 96
    assert data["stride"] == 4
    assert data["runtime_array"] is True


def test_collect_lowered_glsl_buffer_blocks_accepts_fixed_width_alias_members():
    struct = block_struct(
        member("count", primitive("uint16_t")),
        member("offsets", ArrayType(primitive("size_t"), size=None)),
    )

    blocks, var_failures, struct_failures = collect_for(struct)

    assert not var_failures
    assert not struct_failures
    block = blocks["block"]
    count = block["members"]["count"]
    assert count["type"] == "uint16_t"
    assert count["layout_type"] == "uint"
    assert count["size"] == 4
    assert count["align"] == 4
    assert count["target_type"] == "mapped_uint16_t"

    offsets = block["members"]["offsets"]
    assert offsets["type"] == "size_t"
    assert offsets["layout_type"] == "uint"
    assert offsets["offset"] == 4
    assert offsets["stride"] == 4
    assert offsets["runtime_array"] is True
    assert offsets["target_type"] == "mapped_size_t"


def test_collect_lowered_glsl_buffer_blocks_tracks_readonly_qualifier():
    struct = block_struct(
        member("count", primitive("uint")),
        member("data", ArrayType(primitive("float"), size=None)),
    )
    var = block_var(qualifiers=["readonly"])

    blocks, _, _ = collect_for(struct, var)

    assert blocks["block"]["readonly"] is True


def test_collect_lowered_glsl_buffer_blocks_records_unsupported_type_failure():
    struct = block_struct(
        member("flag", primitive("bool")),
        member("data", ArrayType(primitive("float"), size=None)),
    )

    blocks, var_failures, struct_failures = collect_for(struct)

    assert blocks == {}
    assert var_failures["block"] == (
        "unsupported member flag: type is not supported by test lowering"
    )
    assert struct_failures["Block"] == var_failures["block"]


def test_collect_lowered_glsl_buffer_blocks_rejects_non_final_runtime_array():
    struct = block_struct(
        member("data", ArrayType(primitive("float"), size=None)),
        member("tail", primitive("uint")),
    )

    blocks, var_failures, struct_failures = collect_for(struct)

    assert blocks == {}
    assert var_failures["block"] == (
        "unsupported member data: runtime arrays must be the final buffer block member"
    )
    assert struct_failures["Block"] == var_failures["block"]


def test_collect_lowered_glsl_buffer_blocks_rejects_non_literal_fixed_array_size():
    symbolic_size = SimpleNamespace(name="COUNT")
    struct = block_struct(
        member("weights", ArrayType(primitive("float"), size=symbolic_size)),
        member("data", ArrayType(primitive("float"), size=None)),
    )

    blocks, var_failures, struct_failures = collect_for(struct)

    assert blocks == {}
    assert var_failures["block"] == (
        "unsupported member weights: fixed array size must be a literal integer"
    )
    assert struct_failures["Block"] == var_failures["block"]


def test_collect_lowered_glsl_buffer_blocks_ignores_non_std430_layout():
    struct = block_struct(
        member("count", primitive("uint")),
        member("data", ArrayType(primitive("float"), size=None)),
    )
    var = block_var(layout="std140")

    blocks, var_failures, struct_failures = collect_for(struct, var)

    assert blocks == {}
    assert var_failures == {}
    assert struct_failures == {}

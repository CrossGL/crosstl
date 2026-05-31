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
    glsl_buffer_array_stride,
    glsl_buffer_block_node_type,
    glsl_buffer_compound_binary_operator,
    matrix_column_offsets,
    std140_value_type_info,
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


def collect_for(struct, var=None, *, literal_int_value=None, structs_by_name=None):
    var = var or block_var()
    structs_by_name = {"Block": struct, **(structs_by_name or {})}
    return collect_lowered_glsl_buffer_blocks(
        [var],
        structs_by_name=structs_by_name,
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
    bool_info = std430_value_type_info("bool")
    assert bool_info["size"] == 4
    assert bool_info["align"] == 4
    assert bool_info["components"] == 1
    assert bool_info["component_type"] == "bool"
    assert std430_array_stride(bool_info) == 4

    bvec3 = std430_value_type_info("bvec3")
    assert bvec3["size"] == 12
    assert bvec3["align"] == 16
    assert bvec3["components"] == 3
    assert bvec3["component_type"] == "bool"
    assert std430_array_stride(bvec3) == 16

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


def test_std140_type_info_uses_sixteen_byte_matrix_column_stride():
    mat2 = std140_value_type_info("mat2")
    assert mat2["matrix_columns"] == 2
    assert mat2["matrix_rows"] == 2
    assert mat2["align"] == 16
    assert mat2["column_stride"] == 16
    assert mat2["size"] == 32

    scalar = std140_value_type_info("float")
    vec2 = std140_value_type_info("vec2")
    assert glsl_buffer_array_stride(scalar, "std140") == 16
    assert glsl_buffer_array_stride(vec2, "std140") == 16


def test_std430_type_info_normalizes_fixed_width_scalar_aliases():
    assert std430_layout_type_name("int8_t") == "int"
    assert std430_layout_type_name("uint16_t") == "uint"
    assert std430_layout_type_name("size_t") == "uint"
    assert std430_value_type_info("int64_t") == std430_value_type_info("int")
    assert std430_value_type_info("uint64_t") == std430_value_type_info("uint")


def test_byte_offset_expression_formats_literal_and_dynamic_indices():
    assert byte_offset_expression(16, "2", 16) == "48"
    assert byte_offset_expression("(16 + i * 48)", "1", 4) == "(16 + i * 48 + 4)"
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


def test_collect_lowered_glsl_buffer_blocks_lays_out_bool_members():
    struct = block_struct(
        member("enabled", primitive("bool")),
        member("flags", ArrayType(primitive("bool"), size=2)),
        member("data", ArrayType(primitive("bool"), size=None)),
    )

    blocks, var_failures, struct_failures = collect_for(struct)

    assert not var_failures
    assert not struct_failures
    block = blocks["block"]
    assert block["runtime_array"] == "data"

    enabled = block["members"]["enabled"]
    assert enabled["offset"] == 0
    assert enabled["size"] == 4
    assert enabled["align"] == 4
    assert enabled["component_type"] == "bool"
    assert enabled["target_type"] == "mapped_bool"

    flags = block["members"]["flags"]
    assert flags["offset"] == 4
    assert flags["stride"] == 4
    assert flags["array_count"] == 2
    assert flags["target_type"] == "mapped_bool"

    data = block["members"]["data"]
    assert data["offset"] == 12
    assert data["stride"] == 4
    assert data["runtime_array"] is True


def test_collect_lowered_glsl_buffer_blocks_lays_out_bool_vector_members():
    struct = block_struct(
        member("mask", primitive("bvec3")),
        member("pairs", ArrayType(primitive("bvec2"), size=2)),
        member("data", ArrayType(primitive("bvec4"), size=None)),
    )

    blocks, var_failures, struct_failures = collect_for(struct)

    assert not var_failures
    assert not struct_failures
    block = blocks["block"]
    assert block["runtime_array"] == "data"

    mask = block["members"]["mask"]
    assert mask["offset"] == 0
    assert mask["size"] == 12
    assert mask["align"] == 16
    assert mask["components"] == 3
    assert mask["component_type"] == "bool"
    assert mask["target_type"] == "mapped_bvec3"

    pairs = block["members"]["pairs"]
    assert pairs["offset"] == 16
    assert pairs["stride"] == 8
    assert pairs["array_count"] == 2
    assert pairs["components"] == 2
    assert pairs["target_type"] == "mapped_bvec2"

    data = block["members"]["data"]
    assert data["offset"] == 32
    assert data["stride"] == 16
    assert data["runtime_array"] is True
    assert data["components"] == 4
    assert data["target_type"] == "mapped_bvec4"


def test_collect_lowered_glsl_buffer_blocks_lays_out_nested_struct_members():
    inner = StructNode(
        "Inner",
        [
            member("scale", primitive("float")),
            member("mask", primitive("bvec3")),
            member("flags", ArrayType(primitive("bool"), size=2)),
        ],
    )
    struct = block_struct(
        member("count", primitive("uint")),
        member("inner", primitive("Inner")),
        member("data", ArrayType(primitive("float"), size=None)),
    )

    blocks, var_failures, struct_failures = collect_for(
        struct,
        structs_by_name={"Inner": inner},
    )

    assert not var_failures
    assert not struct_failures
    block = blocks["block"]
    assert block["runtime_array"] == "data"

    count = block["members"]["count"]
    assert count["offset"] == 0
    assert count["size"] == 4

    inner_info = block["members"]["inner"]
    assert inner_info["offset"] == 16
    assert inner_info["size"] == 48
    assert inner_info["align"] == 16
    assert inner_info["target_type"] == "mapped_Inner"
    assert set(inner_info["members"]) == {"scale", "mask", "flags"}

    assert inner_info["members"]["scale"]["offset"] == 0
    assert inner_info["members"]["scale"]["component_type"] == "float"
    assert inner_info["members"]["mask"]["offset"] == 16
    assert inner_info["members"]["mask"]["components"] == 3

    flags = inner_info["members"]["flags"]
    assert flags["offset"] == 28
    assert flags["stride"] == 4
    assert flags["array_count"] == 2
    assert flags["component_type"] == "bool"

    data = block["members"]["data"]
    assert data["offset"] == 64
    assert data["stride"] == 4
    assert data["runtime_array"] is True


def test_collect_lowered_glsl_buffer_blocks_lays_out_nested_struct_arrays():
    item = StructNode(
        "Item",
        [
            member("id", primitive("uint")),
            member("normal", primitive("vec3")),
            member("flags", primitive("bvec2")),
        ],
    )
    struct = block_struct(
        member("fixedItems", ArrayType(primitive("Item"), size=2)),
        member("count", primitive("uint")),
        member("items", ArrayType(primitive("Item"), size=None)),
    )

    blocks, var_failures, struct_failures = collect_for(
        struct,
        structs_by_name={"Item": item},
    )

    assert not var_failures
    assert not struct_failures
    block = blocks["block"]
    assert block["runtime_array"] == "items"

    fixed_items = block["members"]["fixedItems"]
    assert fixed_items["offset"] == 0
    assert fixed_items["stride"] == 48
    assert fixed_items["array_count"] == 2
    assert fixed_items["align"] == 16
    assert fixed_items["size"] == 48
    assert fixed_items["members"]["id"]["offset"] == 0
    assert fixed_items["members"]["normal"]["offset"] == 16
    assert fixed_items["members"]["flags"]["offset"] == 32

    count = block["members"]["count"]
    assert count["offset"] == 96

    items = block["members"]["items"]
    assert items["offset"] == 112
    assert items["stride"] == 48
    assert items["runtime_array"] is True
    assert items["members"]["id"]["offset"] == 0
    assert items["members"]["normal"]["offset"] == 16
    assert items["members"]["flags"]["offset"] == 32


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
        member("flag", primitive("double")),
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


def test_collect_lowered_glsl_buffer_blocks_lays_out_std140_arrays():
    struct = block_struct(
        member("count", primitive("uint")),
        member("basis", primitive("mat2")),
        member("weights", ArrayType(primitive("float"), size=3)),
        member("data", ArrayType(primitive("float"), size=None)),
    )
    var = block_var(layout="std140")

    blocks, var_failures, struct_failures = collect_for(struct, var)

    assert not var_failures
    assert not struct_failures
    block = blocks["block"]
    assert block["layout"] == "std140"
    assert block["members"]["count"]["offset"] == 0
    assert block["members"]["basis"]["offset"] == 16
    assert block["members"]["basis"]["column_stride"] == 16
    assert block["members"]["basis"]["size"] == 32

    weights = block["members"]["weights"]
    assert weights["offset"] == 48
    assert weights["stride"] == 16
    assert weights["array_count"] == 3

    data = block["members"]["data"]
    assert data["offset"] == 96
    assert data["stride"] == 16
    assert data["runtime_array"] is True


def test_collect_lowered_glsl_buffer_blocks_ignores_unknown_layout():
    struct = block_struct(
        member("count", primitive("uint")),
        member("data", ArrayType(primitive("float"), size=None)),
    )
    var = block_var(layout="scalar")

    blocks, var_failures, struct_failures = collect_for(struct, var)

    assert blocks == {}
    assert var_failures == {}
    assert struct_failures == {}

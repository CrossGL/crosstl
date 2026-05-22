"""Shared GLSL std430 buffer block layout helpers."""

from ..ast import ArrayNode


def align_to(value, alignment):
    return ((value + alignment - 1) // alignment) * alignment


def byte_offset_expression(base_offset, index, stride):
    if str(index).isdigit():
        return str(base_offset + int(index) * stride)
    indexed_offset = byte_offset_index_expression(index, stride)
    if base_offset == 0:
        return f"({indexed_offset})"
    return f"({base_offset} + {indexed_offset})"


def byte_offset_index_expression(index, stride):
    index = str(index)
    if byte_offset_is_simple_expression(index):
        return f"{index} * {stride}"
    return f"({index}) * {stride}"


def byte_offset_is_simple_expression(expression):
    expression = str(expression)
    if expression.isidentifier():
        return True
    if expression.replace("_", "").isalnum():
        return True
    return expression.startswith("(") and expression.endswith(")")


def byte_offset_strip_outer_parentheses(expression):
    expression = str(expression)
    if not (expression.startswith("(") and expression.endswith(")")):
        return expression
    depth = 0
    for index, char in enumerate(expression):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0 and index != len(expression) - 1:
                return expression
    return expression[1:-1]


def byte_offset_add(base_offset, delta):
    if delta == 0:
        return str(base_offset)
    if str(base_offset).isdigit():
        return str(int(base_offset) + delta)
    return f"({byte_offset_strip_outer_parentheses(base_offset)} + {delta})"


def byte_offset_sequence(base_offset, count, stride):
    return [
        (index, byte_offset_add(base_offset, index * stride)) for index in range(count)
    ]


def vector_component_offsets(base_offset, components, component_size=4):
    return byte_offset_sequence(base_offset, components, component_size)


def matrix_column_offsets(base_offset, columns, column_stride):
    return byte_offset_sequence(base_offset, columns, column_stride)


GLSL_BUFFER_COMPOUND_BINARY_OPERATORS = {
    "+=": "+",
    "-=": "-",
    "*=": "*",
    "/=": "/",
    "%=": "%",
    "&=": "&",
    "|=": "|",
    "^=": "^",
    "<<=": "<<",
    ">>=": ">>",
}

GLSL_BUFFER_FLOAT_COMPOUND_OPERATORS = {"+=", "-=", "*=", "/="}


def glsl_buffer_compound_binary_operator(operator, component_type):
    binary_operator = GLSL_BUFFER_COMPOUND_BINARY_OPERATORS.get(operator)
    if binary_operator is None:
        return None
    if (
        component_type == "float"
        and operator not in GLSL_BUFFER_FLOAT_COMPOUND_OPERATORS
    ):
        return None
    if component_type in {"float", "int", "uint"}:
        return binary_operator
    return None


def std430_scalar_type_info(component_type):
    return {
        "size": 4,
        "align": 4,
        "components": 1,
        "component_type": component_type,
    }


def std430_vector_type_info(component_type, components):
    info = {
        "size": components * 4,
        "align": 8 if components == 2 else 16,
        "components": components,
        "component_type": component_type,
    }
    if components == 3:
        info["array_stride"] = 16
    return info


def std430_matrix_type_info(columns, rows):
    column_size = rows * 4
    column_align = 8 if rows == 2 else 16
    column_stride = align_to(column_size, column_align)
    return {
        "size": columns * column_stride,
        "align": column_align,
        "matrix_columns": columns,
        "matrix_rows": rows,
        "column_stride": column_stride,
        "component_type": "float",
    }


def std430_scalar_vector_type_entries():
    entries = {
        "float": std430_scalar_type_info("float"),
        "int": std430_scalar_type_info("int"),
        "uint": std430_scalar_type_info("uint"),
    }
    vector_prefixes = {
        "float": ("vec", "float"),
        "int": ("ivec", "int"),
        "uint": ("uvec", "uint"),
    }
    for component_type, prefixes in vector_prefixes.items():
        for components in range(2, 5):
            info = std430_vector_type_info(component_type, components)
            for prefix in prefixes:
                entries[f"{prefix}{components}"] = info
    return entries


def std430_matrix_type_entries():
    entries = {}
    for columns in range(2, 5):
        for rows in range(2, 5):
            info = std430_matrix_type_info(columns, rows)
            names = {f"mat{columns}x{rows}", f"float{rows}x{columns}"}
            if columns == rows:
                names.add(f"mat{columns}")
            for name in names:
                entries[name] = info
    return entries


def std430_value_type_info(type_name):
    type_info = std430_scalar_vector_type_entries()
    type_info.update(std430_matrix_type_entries())
    info = type_info.get(type_name)
    return None if info is None else dict(info)


def std430_array_stride(member_info):
    return member_info.get(
        "array_stride", align_to(member_info["size"], member_info["align"])
    )


def glsl_buffer_block_is_readonly(node):
    qualifiers = {str(q).lower() for q in getattr(node, "qualifiers", []) or []}
    attributes = {
        str(getattr(attr, "name", "")).lower()
        for attr in getattr(node, "attributes", []) or []
    }
    return "readonly" in qualifiers or "readonly" in attributes


def variable_is_array(node):
    var_type = getattr(node, "var_type", None)
    return (
        hasattr(var_type, "element_type")
        and str(type(var_type)).find("ArrayType") != -1
    )


def glsl_buffer_block_member_type(
    member, convert_type_node_to_string, map_type, target_type_key
):
    member_type = getattr(member, "member_type", None)
    is_array = False
    array_size = None
    if member_type is not None:
        if str(type(member_type)).find("ArrayType") != -1:
            is_array = True
            array_size = member_type.size
            member_type = member_type.element_type
        type_name = convert_type_node_to_string(member_type)
    elif isinstance(member, ArrayNode):
        is_array = True
        array_size = member.size
        type_name = getattr(member, "element_type", getattr(member, "vtype", None))
    elif hasattr(member, "vtype"):
        type_name = member.vtype
    else:
        return None

    type_name = str(type_name)
    type_info = std430_value_type_info(type_name)
    if type_info is None:
        return None
    return {
        "type": type_name,
        **type_info,
        target_type_key: map_type(type_name),
        "is_array": is_array,
        "array_size": array_size,
    }


def record_glsl_buffer_block_lowering_failure(
    var_failures, struct_failures, var_name, type_name, reason
):
    if var_name:
        var_failures[var_name] = reason
    if type_name:
        struct_failures.setdefault(str(type_name), reason)


def collect_lowered_glsl_buffer_blocks(
    global_vars,
    *,
    structs_by_name,
    is_glsl_buffer_block_variable,
    resource_base_type,
    glsl_buffer_block_layout,
    convert_type_node_to_string,
    literal_int_value,
    map_type,
    target_type_key,
    unsupported_type_message,
):
    blocks = {}
    var_failures = {}
    struct_failures = {}
    for node in global_vars:
        if variable_is_array(node):
            continue
        var_name = getattr(node, "name", getattr(node, "variable_name", None))
        if not var_name or not is_glsl_buffer_block_variable(node):
            continue
        layout = glsl_buffer_block_layout(node)
        if str(layout).lower() != "std430":
            continue

        type_name = str(resource_base_type(getattr(node, "var_type", None)))
        struct = structs_by_name.get(type_name)
        if struct is None:
            continue

        readonly = glsl_buffer_block_is_readonly(node)
        offset = 0
        members = {}
        runtime_array_name = None
        struct_members = getattr(struct, "members", []) or []
        failure_reason = None
        for index, member in enumerate(struct_members):
            member_info = glsl_buffer_block_member_type(
                member, convert_type_node_to_string, map_type, target_type_key
            )
            member_name = getattr(member, "name", None)
            if member_info is None:
                failure_reason = (
                    f"unsupported member {member_name or '<unnamed>'}: "
                    f"{unsupported_type_message}"
                )
                members = {}
                break
            offset = align_to(offset, member_info["align"])
            if not member_name:
                failure_reason = "unsupported unnamed buffer block member"
                members = {}
                break
            if member_info["is_array"]:
                if member_info["array_size"] is None:
                    if index != len(struct_members) - 1:
                        failure_reason = (
                            f"unsupported member {member_name}: runtime arrays "
                            "must be the final buffer block member"
                        )
                        members = {}
                        break
                    runtime_array_name = member_name
                    members[member_name] = {
                        **member_info,
                        "offset": offset,
                        "stride": std430_array_stride(member_info),
                        "runtime_array": True,
                    }
                    continue

                array_count = literal_int_value(member_info["array_size"])
                if array_count is None:
                    failure_reason = (
                        f"unsupported member {member_name}: fixed array size "
                        "must be a literal integer"
                    )
                    members = {}
                    break
                members[member_name] = {
                    **member_info,
                    "offset": offset,
                    "stride": std430_array_stride(member_info),
                    "array_count": array_count,
                    "runtime_array": False,
                }
                offset += members[member_name]["stride"] * array_count
                continue
            members[member_name] = {
                **member_info,
                "offset": offset,
                "runtime_array": False,
            }
            offset += member_info["size"]

        if runtime_array_name is None or not members:
            if failure_reason:
                record_glsl_buffer_block_lowering_failure(
                    var_failures, struct_failures, var_name, type_name, failure_reason
                )
            continue
        blocks[var_name] = {
            "type_name": type_name,
            "layout": layout,
            "readonly": readonly,
            "members": members,
            "runtime_array": runtime_array_name,
        }
    return blocks, var_failures, struct_failures

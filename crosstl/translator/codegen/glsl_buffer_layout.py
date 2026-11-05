"""Shared GLSL buffer block layout helpers."""

from inspect import Parameter, signature

from ..ast import ArrayNode
from ..matrix_shape import matrix_type_spelling


def align_to(value, alignment):
    return ((value + alignment - 1) // alignment) * alignment


def byte_offset_expression(base_offset, index, stride):
    if str(index).isdigit():
        return byte_offset_add(base_offset, int(index) * stride)
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


GLSL_BUFFER_FLOAT_COMPONENT_TYPES = frozenset({"float", "float16_t"})
GLSL_BUFFER_INTEGER_COMPONENT_TYPES = frozenset(
    {"int", "uint", "int16_t", "uint16_t", "int64_t", "uint64_t", "int8_t", "uint8_t"}
)


def glsl_buffer_compound_binary_operator(operator, component_type):
    binary_operator = GLSL_BUFFER_COMPOUND_BINARY_OPERATORS.get(operator)
    if binary_operator is None:
        return None
    if component_type in GLSL_BUFFER_FLOAT_COMPONENT_TYPES:
        if operator not in GLSL_BUFFER_FLOAT_COMPOUND_OPERATORS:
            return None
        return binary_operator
    if component_type in GLSL_BUFFER_INTEGER_COMPONENT_TYPES:
        return binary_operator
    return None


def std430_scalar_type_info(component_type, component_size=4):
    return {
        "size": component_size,
        "align": component_size,
        "components": 1,
        "component_type": component_type,
        "component_size": component_size,
    }


def std430_vector_type_info(component_type, components, component_size=4):
    info = {
        "size": components * component_size,
        "align": component_size * (2 if components == 2 else 4),
        "components": components,
        "component_type": component_type,
        "component_size": component_size,
    }
    if components == 3:
        info["array_stride"] = 4 * component_size
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
        "component_size": 4,
    }


def std140_matrix_type_info(columns, rows):
    column_stride = 16
    return {
        "size": columns * column_stride,
        "align": 16,
        "matrix_columns": columns,
        "matrix_rows": rows,
        "column_stride": column_stride,
        "component_type": "float",
        "component_size": 4,
    }


def std430_scalar_vector_type_entries():
    entries = {
        "bool": std430_scalar_type_info("bool"),
        "float": std430_scalar_type_info("float"),
        "int": std430_scalar_type_info("int"),
        "uint": std430_scalar_type_info("uint"),
        "int64_t": std430_scalar_type_info("int64_t", 8),
        "uint64_t": std430_scalar_type_info("uint64_t", 8),
        # GL_EXT_shader_16bit_storage and GL_EXT_shader_8bit_storage: the base
        # alignment of a scalar is its size (GLSL 4.60 section 7.6.2.2 rule 1).
        "int16_t": std430_scalar_type_info("int16_t", 2),
        "uint16_t": std430_scalar_type_info("uint16_t", 2),
        "float16_t": std430_scalar_type_info("float16_t", 2),
        "int8_t": std430_scalar_type_info("int8_t", 1),
        "uint8_t": std430_scalar_type_info("uint8_t", 1),
    }
    vector_prefixes = {
        "bool": (("bvec", "bool"), 4),
        "float": (("vec", "float"), 4),
        "int": (("ivec", "int"), 4),
        "uint": (("uvec", "uint"), 4),
        "int64_t": (("i64vec", "int64_t"), 8),
        "uint64_t": (("u64vec", "uint64_t"), 8),
        "int16_t": (("i16vec",), 2),
        "uint16_t": (("u16vec",), 2),
        "float16_t": (("f16vec",), 2),
        "int8_t": (("i8vec",), 1),
        "uint8_t": (("u8vec",), 1),
    }
    for component_type, (prefixes, component_size) in vector_prefixes.items():
        for components in range(2, 5):
            info = std430_vector_type_info(
                component_type, components, component_size=component_size
            )
            for prefix in prefixes:
                entries[f"{prefix}{components}"] = info
    return entries


def std140_scalar_vector_type_entries():
    return std430_scalar_vector_type_entries()


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


def std140_matrix_type_entries():
    entries = {}
    for columns in range(2, 5):
        for rows in range(2, 5):
            info = std140_matrix_type_info(columns, rows)
            names = {f"mat{columns}x{rows}", f"float{rows}x{columns}"}
            if columns == rows:
                names.add(f"mat{columns}")
            for name in names:
                entries[name] = info
    return entries


STD430_SCALAR_TYPE_ALIASES = {
    "int8": "int8_t",
    "int16": "int16_t",
    "short": "int16_t",
    "int32_t": "int",
    "i64": "int64_t",
    "int64": "int64_t",
    "int64_t": "int64_t",
    "long": "int",
    "signed long": "int",
    "ptrdiff_t": "int",
    "uint8": "uint8_t",
    "uint16": "uint16_t",
    "ushort": "uint16_t",
    "float16": "float16_t",
    "half": "float16_t",
    "uint32_t": "uint",
    "u64": "uint64_t",
    "uint64": "uint64_t",
    "uint64_t": "uint64_t",
    "ulong": "uint",
    "unsigned long": "uint",
    "size_t": "uint",
}


def std430_layout_type_name(type_name):
    return STD430_SCALAR_TYPE_ALIASES.get(str(type_name), str(type_name))


def std430_value_type_info(type_name):
    type_info = std430_scalar_vector_type_entries()
    type_info.update(std430_matrix_type_entries())
    info = type_info.get(std430_layout_type_name(type_name))
    return None if info is None else dict(info)


def std140_value_type_info(type_name):
    type_info = std140_scalar_vector_type_entries()
    type_info.update(std140_matrix_type_entries())
    info = type_info.get(std430_layout_type_name(type_name))
    return None if info is None else dict(info)


def glsl_buffer_layout_value_type_info(type_name, layout):
    if str(layout).lower() == "std140":
        return std140_value_type_info(type_name)
    return std430_value_type_info(type_name)


def std430_array_stride(member_info):
    return member_info.get(
        "array_stride", align_to(member_info["size"], member_info["align"])
    )


def glsl_buffer_array_align(member_info, layout):
    if str(layout).lower() == "std140":
        return max(16, member_info["align"])
    return member_info["align"]


def glsl_buffer_array_stride(member_info, layout):
    if str(layout).lower() == "std140":
        return max(16, align_to(member_info["size"], member_info["align"]))
    return std430_array_stride(member_info)


def glsl_buffer_block_is_readonly(node):
    qualifiers = {str(q).lower() for q in getattr(node, "qualifiers", []) or []}
    attributes = {
        str(getattr(attr, "name", "")).lower()
        for attr in getattr(node, "attributes", []) or []
    }
    return "readonly" in qualifiers or "readonly" in attributes


def glsl_buffer_block_node_type(node):
    return getattr(
        node,
        "var_type",
        getattr(node, "param_type", getattr(node, "vtype", None)),
    )


def glsl_buffer_block_predicate_matches(predicate, node, node_type):
    try:
        parameters = signature(predicate).parameters
    except (TypeError, ValueError):
        return predicate(node, node_type)

    accepts_two_args = any(
        param.kind == Parameter.VAR_POSITIONAL for param in parameters.values()
    ) or (
        sum(
            1
            for param in parameters.values()
            if param.kind in (
                Parameter.POSITIONAL_ONLY,
                Parameter.POSITIONAL_OR_KEYWORD,
            )
        )
        >= 2
    )
    if accepts_two_args:
        return predicate(node, node_type)
    return predicate(node)


def glsl_buffer_struct_type_info(
    type_name,
    structs_by_name,
    convert_type_node_to_string,
    literal_int_value,
    map_type,
    target_type_key,
    layout,
    type_stack,
):
    type_name = str(type_name)
    if type_name in type_stack:
        return None
    struct = structs_by_name.get(type_name)
    if struct is None:
        return None

    offset = 0
    max_align = 0
    members = {}
    for member in getattr(struct, "members", []) or []:
        member_info = glsl_buffer_block_member_type(
            member,
            convert_type_node_to_string,
            map_type,
            target_type_key,
            layout,
            structs_by_name=structs_by_name,
            literal_int_value=literal_int_value,
            type_stack=(*type_stack, type_name),
        )
        member_name = getattr(member, "name", None)
        if member_info is None or not member_name:
            return None

        if member_info["is_array"]:
            if member_info["array_size"] is None:
                return None
            member_align = glsl_buffer_array_align(member_info, layout)
            offset = align_to(offset, member_align)
            array_count = literal_int_value(member_info["array_size"])
            if array_count is None:
                return None
            members[member_name] = {
                **member_info,
                "offset": offset,
                "stride": glsl_buffer_array_stride(member_info, layout),
                "array_count": array_count,
                "runtime_array": False,
            }
            offset += members[member_name]["stride"] * array_count
            max_align = max(max_align, member_align)
            continue

        offset = align_to(offset, member_info["align"])
        members[member_name] = {
            **member_info,
            "offset": offset,
            "runtime_array": False,
        }
        offset += member_info["size"]
        max_align = max(max_align, member_info["align"])

    if not members or max_align == 0:
        return None

    struct_align = max_align
    if str(layout).lower() == "std140":
        struct_align = max(16, struct_align)
    return {
        "size": align_to(offset, struct_align),
        "align": struct_align,
        "members": members,
        "is_struct": True,
    }


def glsl_buffer_block_member_type(
    member,
    convert_type_node_to_string,
    map_type,
    target_type_key,
    layout,
    *,
    structs_by_name=None,
    literal_int_value=None,
    type_stack=(),
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
        if type(member_type).__name__ == "MatrixType" and getattr(
            getattr(member_type, "element_type", None), "name", "float"
        ) in {"float", "double"}:
            # Lay matrices out by their CrossGL spelling: target spellings disagree on
            # which of the two numbers counts the columns.
            type_name = matrix_type_spelling(
                member_type, member_type.element_type.name, short_square=False
            )
    elif isinstance(member, ArrayNode):
        is_array = True
        array_size = member.size
        type_name = getattr(member, "element_type", getattr(member, "vtype", None))
    elif hasattr(member, "vtype"):
        type_name = member.vtype
    else:
        return None

    type_name = str(type_name)
    layout_type_name = std430_layout_type_name(type_name)
    type_info = glsl_buffer_layout_value_type_info(layout_type_name, layout)
    if type_info is None and structs_by_name is not None:
        type_info = glsl_buffer_struct_type_info(
            layout_type_name,
            structs_by_name,
            convert_type_node_to_string,
            literal_int_value or (lambda value: None),
            map_type,
            target_type_key,
            layout,
            type_stack,
        )
    if type_info is None:
        return None
    return {
        "type": type_name,
        "layout_type": layout_type_name,
        **type_info,
        target_type_key: map_type(type_name),
        "is_array": is_array,
        "array_size": array_size,
    }


# Component widths every byte-offset lowering reads and writes with its own types.
DEFAULT_GLSL_BUFFER_COMPONENT_SIZES = frozenset({4, 8})


def glsl_buffer_member_leaves(member_info):
    """The scalar, vector and matrix layouts inside ``member_info``, in declaration order."""
    nested = member_info.get("members")
    if not nested:
        yield member_info
        return
    for child in nested.values():
        yield from glsl_buffer_member_leaves(child)


def glsl_buffer_component_size_refusal(member_info, component_sizes):
    """Why a lowering limited to ``component_sizes`` cannot hold ``member_info``, or ``None``."""
    for leaf in glsl_buffer_member_leaves(member_info):
        size = leaf.get("component_size", 4)
        if size not in component_sizes:
            return (
                f"{size * 8}-bit {leaf.get('component_type')} components have no "
                "lowering for this target"
            )
    return None


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
    component_refusal=None,
):
    """Lay out each std140 or std430 buffer block in ``global_vars`` for byte-offset access.

    ``component_refusal(member_info)`` names why the target cannot read or write a
    member, or returns ``None``; by default only 32- and 64-bit components lower.
    """
    if component_refusal is None:

        def component_refusal(member_info):
            return glsl_buffer_component_size_refusal(
                member_info, DEFAULT_GLSL_BUFFER_COMPONENT_SIZES
            )

    blocks = {}
    var_failures = {}
    struct_failures = {}
    for node in global_vars:
        var_name = getattr(node, "name", getattr(node, "variable_name", None))
        node_type = glsl_buffer_block_node_type(node)
        if not var_name or not glsl_buffer_block_predicate_matches(
            is_glsl_buffer_block_variable, node, node_type
        ):
            continue
        layout = glsl_buffer_block_layout(node)
        layout_key = str(layout).lower()
        if layout_key not in {"std140", "std430"}:
            continue

        type_name = str(resource_base_type(node_type))
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
                member,
                convert_type_node_to_string,
                map_type,
                target_type_key,
                layout,
                structs_by_name=structs_by_name,
                literal_int_value=literal_int_value,
            )
            member_name = getattr(member, "name", None)
            if member_info is None:
                failure_reason = (
                    f"unsupported member {member_name or '<unnamed>'}: "
                    f"{unsupported_type_message}"
                )
                members = {}
                break
            if not member_name:
                failure_reason = "unsupported unnamed buffer block member"
                members = {}
                break
            refusal = component_refusal(member_info)
            if refusal is not None:
                failure_reason = f"unsupported member {member_name}: {refusal}"
                members = {}
                break
            if member_info["is_array"]:
                offset = align_to(offset, glsl_buffer_array_align(member_info, layout))
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
                        "stride": glsl_buffer_array_stride(member_info, layout),
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
                    "stride": glsl_buffer_array_stride(member_info, layout),
                    "array_count": array_count,
                    "runtime_array": False,
                }
                offset += members[member_name]["stride"] * array_count
                continue
            offset = align_to(offset, member_info["align"])
            members[member_name] = {
                **member_info,
                "offset": offset,
                "runtime_array": False,
            }
            offset += member_info["size"]

        if not members:
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

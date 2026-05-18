"""Utility functions for array handling in code generators.

This module provides common functions for handling array types, array access,
and type detection across different code generators.
"""

from typing import Optional, Tuple, Dict, Any


def parse_array_type(type_name: str) -> Tuple[str, Optional[int]]:
    """Parse an array type string into base type and size.

    Args:
        type_name: The array type string (e.g., "float[4]", "vec3[]")

    Returns:
        Tuple of (base_type, size) where size is None for dynamic arrays
    """
    if not type_name or "[" not in type_name:
        return type_name, None

    if type_name.endswith("]"):
        open_bracket = type_name.find("[")
        base_type = type_name[:open_bracket]
        size_str = type_name[open_bracket + 1 : -1]

        if not size_str:
            return base_type, None

        try:
            return base_type, int(size_str)
        except ValueError:
            return base_type, None

    return type_name, None


def format_array_type(
    base_type: str, size: Optional[int], lang_style: str = "glsl"
) -> str:
    """Format an array type according to the target language style.

    Args:
        base_type: The base type of the array (e.g., "float", "vec3")
        size: The size of the array, or None for dynamic arrays
        lang_style: The language style ('glsl', 'hlsl', 'metal', 'spirv')

    Returns:
        The formatted array type string for the target language
    """
    if lang_style == "hlsl":
        if size is None:
            return f"{base_type}[1024]"
        else:
            return f"{base_type}[{size}]"
    elif lang_style == "metal":
        if size is None:
            return f"array<{base_type}, 1024>"
        else:
            return f"array<{base_type}, {size}>"
    elif lang_style == "spirv":
        if size is None:
            return f"array_{base_type}_dynamic"
        else:
            return f"array_{base_type}_{size}"
    else:  # glsl and default
        if size is None:
            return f"{base_type}[]"
        else:
            return f"{base_type}[{size}]"


def format_c_style_array_declaration(type_name: str, variable_name: str) -> str:
    """Format a declaration when the type string includes C-style array suffixes.

    Converts "float[4]", "vec3[]", or "float[3][4]" into
    "float name[4]", "vec3 name[]", or "float name[3][4]".
    Non-array type strings are returned as "type name".
    """
    if not type_name or "[" not in type_name:
        return f"{type_name} {variable_name}"

    open_bracket = type_name.find("[")
    base_type = type_name[:open_bracket]
    array_suffix = type_name[open_bracket:]
    return f"{base_type} {variable_name}{array_suffix}"


def split_array_type_suffix(type_name: str) -> Tuple[str, str]:
    """Split an array type string while preserving the full array suffix.

    Unlike parse_array_type(), this keeps non-literal sizes such as
    "float[(2 + 1) * 2]" intact instead of treating them as unsized arrays.
    """
    if not type_name or "[" not in type_name:
        return type_name, ""

    open_bracket = type_name.find("[")
    return type_name[:open_bracket], type_name[open_bracket:]


def collect_struct_member_types(structs, type_name_string) -> Dict[str, Dict[str, str]]:
    """Collect raw struct member types for expression type inference."""
    member_types = {}
    for struct in structs or []:
        struct_name = getattr(struct, "name", None)
        members = getattr(struct, "members", None)
        if not struct_name or members is None:
            continue
        member_types[struct_name] = {
            member.name: _struct_member_type_name(member, type_name_string)
            for member in members
            if getattr(member, "name", None)
        }
    return member_types


def _struct_member_type_name(member, type_name_string) -> str:
    if member.__class__.__name__ == "ArrayNode":
        element_type = getattr(
            member, "element_type", getattr(member, "vtype", "float")
        )
        base_type = type_name_string(element_type)
        size = getattr(member, "size", None)
        return f"{base_type}[{size}]" if size else f"{base_type}[]"
    if hasattr(member, "member_type"):
        return type_name_string(member.member_type)
    return type_name_string(getattr(member, "vtype", "float"))


def detect_array_element_type(
    array_type: str, type_mapping: Dict[str, Any] = None
) -> str:
    """Detect the element type of an array based on its type string.

    Args:
        array_type: The array type string
        type_mapping: Optional mapping of types to use for lookups

    Returns:
        The detected element type string
    """
    base_type, _ = parse_array_type(array_type)

    if type_mapping and base_type in type_mapping:
        return type_mapping[base_type]

    return base_type


def get_array_size_from_node(node) -> Optional[int]:
    """Extract array size from an AST ArrayNode.

    Args:
        node: The ArrayNode to extract size from

    Returns:
        The array size as an integer, or None for dynamic arrays
    """
    if not hasattr(node, "size"):
        return None

    if node.size is None:
        return None

    try:
        return int(node.size)
    except (ValueError, TypeError):
        return None


def evaluate_literal_int_expression(
    expr, constants: Optional[Dict[str, int]] = None
) -> Optional[int]:
    """Evaluate a narrow integer-only AST expression.

    This is used for declaration sizing hints where accepting dynamic values would
    be unsafe. It intentionally supports only literals, unary signs, and basic
    arithmetic over integer literals. Named constants are resolved only from the
    explicit constants map supplied by the caller.
    """
    constants = constants or {}

    if expr is None:
        return None

    if isinstance(expr, bool):
        return None

    if isinstance(expr, int):
        return expr

    if isinstance(expr, str):
        parsed = _parse_decimal_int_literal(expr)
        if parsed is not None:
            return parsed
        return constants.get(expr)

    if hasattr(expr, "value"):
        value = getattr(expr, "value")
        parsed = evaluate_literal_int_expression(value, constants)
        if parsed is not None:
            return parsed

    name = getattr(expr, "name", None)
    if isinstance(name, str) and name in constants:
        return constants[name]

    class_name = expr.__class__.__name__
    if "UnaryOp" in class_name:
        operand = evaluate_literal_int_expression(
            getattr(expr, "operand", None), constants
        )
        if operand is None:
            return None
        operator = getattr(expr, "operator", getattr(expr, "op", None))
        if operator == "+":
            return operand
        if operator == "-":
            return -operand
        return None

    if "BinaryOp" in class_name:
        left = evaluate_literal_int_expression(getattr(expr, "left", None), constants)
        right = evaluate_literal_int_expression(getattr(expr, "right", None), constants)
        if left is None or right is None:
            return None
        operator = getattr(expr, "operator", getattr(expr, "op", None))
        if operator == "+":
            return left + right
        if operator == "-":
            return left - right
        if operator == "*":
            return left * right

    return None


def collect_literal_int_constants(constants) -> Dict[str, int]:
    """Collect resolvable integer compile-time constants by name."""
    resolved = {}
    pending = list(constants or [])

    changed = True
    while changed:
        changed = False
        remaining = []
        for const in pending:
            name = getattr(const, "name", None)
            if not name or name in resolved:
                continue

            value = evaluate_literal_int_expression(
                getattr(const, "value", None), resolved
            )
            if value is None:
                remaining.append(const)
                continue

            resolved[name] = value
            changed = True
        pending = remaining

    return resolved


def _parse_decimal_int_literal(value: str) -> Optional[int]:
    value = value.strip()
    if not value:
        return None

    sign = ""
    digits = value
    if value[0] in "+-":
        sign = value[0]
        digits = value[1:]

    if not digits.isdigit():
        return None

    return int(f"{sign}{digits}")

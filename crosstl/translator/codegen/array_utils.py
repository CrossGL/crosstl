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

    # Handle array types like "float[4]"
    if type_name.endswith("]"):
        open_bracket = type_name.find("[")
        base_type = type_name[:open_bracket]
        size_str = type_name[open_bracket + 1 : -1]

        # Check if it's a dynamic array
        if not size_str:
            return base_type, None

        try:
            return base_type, int(size_str)
        except ValueError:
            # Handle case where size is a constant or expression
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
            # In HLSL, RWStructuredBuffer can be used for dynamic arrays
            # But for simplicity, we'll use a large size
            return f"{base_type}[1024]"
        else:
            return f"{base_type}[{size}]"
    elif lang_style == "metal":
        if size is None:
            # In Metal, device arrays must have a size, so we use a pointer
            # But for simplicity within our framework, we'll use a large array
            return f"array<{base_type}, 1024>"
        else:
            return f"array<{base_type}, {size}>"
    elif lang_style == "spirv":
        # For SPIR-V code generation, we return a type ID format that
        # can be parsed by the Vulkan code generator
        if size is None:
            return f"array_{base_type}_dynamic"
        else:
            return f"array_{base_type}_{size}"
    else:  # glsl and default
        if size is None:
            return f"{base_type}[]"
        else:
            return f"{base_type}[{size}]"


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

    # If we have a type mapping, try to use it
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
        # If size is not a simple integer (e.g., it's an expression)
        # we can't determine it statically
        return None

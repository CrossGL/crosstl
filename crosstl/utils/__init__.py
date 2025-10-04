"""
Utility functions for CrossTL.
"""

from .type_mappings import (
    get_type_mapping,
    map_type,
    get_function_mapping,
    map_function,
    UNIVERSAL_TYPE_MAPPINGS,
    BACKEND_TYPE_MAPPINGS,
)

from .ast_utils import (
    safe_get_attribute,
    get_node_type_string,
    get_function_qualifier,
    get_function_parameters,
    get_function_body,
    get_body_statements,
    get_semantic_from_attributes,
    get_node_semantic,
    is_array_type,
    parse_array_type,
    format_array_type,
    convert_type_node_to_string,
    expression_to_string,
    get_array_size_from_node,
    is_legacy_ast_node,
    normalize_operator,
    safe_visit_node,
    extract_includes_from_ast,
    extract_structs_from_ast,
    extract_functions_from_ast,
    extract_global_variables_from_ast,
    extract_cbuffers_from_ast,
)

__all__ = [
    # Type mappings
    "get_type_mapping",
    "map_type",
    "get_function_mapping",
    "map_function",
    "UNIVERSAL_TYPE_MAPPINGS",
    "BACKEND_TYPE_MAPPINGS",
    # AST utilities
    "safe_get_attribute",
    "get_node_type_string",
    "get_function_qualifier",
    "get_function_parameters",
    "get_function_body",
    "get_body_statements",
    "get_semantic_from_attributes",
    "get_node_semantic",
    "is_array_type",
    "parse_array_type",
    "format_array_type",
    "convert_type_node_to_string",
    "expression_to_string",
    "get_array_size_from_node",
    "is_legacy_ast_node",
    "normalize_operator",
    "safe_visit_node",
    "extract_includes_from_ast",
    "extract_structs_from_ast",
    "extract_functions_from_ast",
    "extract_global_variables_from_ast",
    "extract_cbuffers_from_ast",
]

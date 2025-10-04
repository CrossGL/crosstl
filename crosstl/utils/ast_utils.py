"""
AST Utilities for CrossTL.
Common functions for working with AST nodes across all backends.
"""

from typing import Optional, List, Union


def safe_get_attribute(node, attr_name: str, default=None):
    """Safely get attribute from node, handling both old and new AST structures."""
    return getattr(node, attr_name, default)


def get_node_type_string(node) -> str:
    """Get type string from various node structures."""
    # New AST structure
    if hasattr(node, "var_type"):
        if hasattr(node.var_type, "name"):
            return node.var_type.name
        else:
            return str(node.var_type)
    # Old AST structure
    elif hasattr(node, "vtype"):
        return node.vtype
    # Member type
    elif hasattr(node, "member_type"):
        if hasattr(node.member_type, "name"):
            return node.member_type.name
        else:
            return str(node.member_type)
    # Parameter type
    elif hasattr(node, "param_type"):
        if hasattr(node.param_type, "name"):
            return node.param_type.name
        else:
            return str(node.param_type)
    # Return type
    elif hasattr(node, "return_type"):
        if hasattr(node.return_type, "name"):
            return node.return_type.name
        else:
            return str(node.return_type)
    else:
        return "float"  # Default fallback


def get_function_qualifier(func_node) -> Optional[str]:
    """Get function qualifier from various AST structures."""
    # New AST structure
    if hasattr(func_node, "qualifiers") and func_node.qualifiers:
        return func_node.qualifiers[0]
    # Old AST structure
    elif hasattr(func_node, "qualifier"):
        return func_node.qualifier
    else:
        return None


def get_function_parameters(func_node) -> List:
    """Get function parameters from various AST structures."""
    # New AST structure
    if hasattr(func_node, "parameters"):
        return func_node.parameters
    # Old AST structure
    elif hasattr(func_node, "params"):
        return func_node.params
    else:
        return []


def get_function_body(func_node):
    """Get function body from various AST structures."""
    if hasattr(func_node, "body"):
        return func_node.body
    else:
        return []


def get_body_statements(body) -> List:
    """Get statements from function body, handling both old and new AST."""
    if body is None:
        return []
    elif hasattr(body, "statements"):
        # New AST BlockNode structure
        return body.statements
    elif isinstance(body, list):
        # Old AST structure - list of statements
        return body
    else:
        # Single statement
        return [body]


def get_semantic_from_attributes(attributes) -> Optional[str]:
    """Extract semantic information from attribute list."""
    if not attributes:
        return None

    semantic_attrs = [
        "position",
        "color",
        "texcoord",
        "normal",
        "tangent",
        "binormal",
        "POSITION",
        "COLOR",
        "TEXCOORD",
        "NORMAL",
        "TANGENT",
        "BINORMAL",
        "TEXCOORD0",
        "TEXCOORD1",
        "TEXCOORD2",
        "TEXCOORD3",
        "COLOR0",
        "COLOR1",
    ]

    for attr in attributes:
        if hasattr(attr, "name") and attr.name in semantic_attrs:
            return attr.name

    return None


def get_node_semantic(node) -> Optional[str]:
    """Get semantic from node, handling both old and new AST."""
    # Direct semantic attribute (old AST)
    if hasattr(node, "semantic"):
        return node.semantic
    # New AST structure - extract from attributes
    elif hasattr(node, "attributes"):
        return get_semantic_from_attributes(node.attributes)
    else:
        return None


def is_array_type(type_str: str) -> bool:
    """Check if type string represents an array."""
    return "[" in type_str and "]" in type_str


def parse_array_type(type_str: str) -> tuple:
    """Parse array type string into base type and size."""
    if not is_array_type(type_str):
        return type_str, None

    bracket_start = type_str.index("[")
    bracket_end = type_str.rindex("]")
    base_type = type_str[:bracket_start].strip()
    size_str = type_str[bracket_start + 1 : bracket_end].strip()

    if not size_str:
        return base_type, None

    try:
        return base_type, int(size_str)
    except ValueError:
        return base_type, size_str  # Dynamic size


def format_array_type(base_type: str, size: Union[int, str, None], backend: str) -> str:
    """Format array type for specific backend."""
    if size is None:
        if backend == "rust":
            return f"Vec<{base_type}>"
        elif backend == "mojo":
            return f"DynamicVector[{base_type}]"
        else:
            return f"{base_type}[]"
    else:
        if backend == "rust":
            return f"[{base_type}; {size}]"
        elif backend == "mojo":
            return f"StaticTuple[{base_type}, {size}]"
        else:
            return f"{base_type}[{size}]"


def convert_type_node_to_string(type_node) -> str:
    """Convert new AST TypeNode to string representation."""
    if hasattr(type_node, "name"):
        # PrimitiveType or NamedType
        return type_node.name
    elif hasattr(type_node, "element_type") and hasattr(type_node, "size"):
        # VectorType or ArrayType
        if hasattr(type_node, "rows") and hasattr(type_node, "cols"):
            # MatrixType
            element_type = convert_type_node_to_string(type_node.element_type)
            if type_node.rows == type_node.cols:
                return f"mat{type_node.rows}"
            else:
                return f"mat{type_node.cols}x{type_node.rows}"
        elif str(type(type_node)).find("ArrayType") != -1:
            # ArrayType
            element_type = convert_type_node_to_string(type_node.element_type)
            if type_node.size is not None:
                if isinstance(type_node.size, int):
                    return f"{element_type}[{type_node.size}]"
                else:
                    # Size is an expression node
                    size_str = expression_to_string(type_node.size)
                    return f"{element_type}[{size_str}]"
            else:
                return f"{element_type}[]"
        else:
            # VectorType
            element_type = convert_type_node_to_string(type_node.element_type)
            size = type_node.size

            # Map to proper vector types based on element type
            if element_type == "float":
                return f"vec{size}"
            elif element_type == "int":
                return f"ivec{size}"
            elif element_type == "uint":
                return f"uvec{size}"
            elif element_type == "bool":
                return f"bvec{size}"
            else:
                return f"{element_type}{size}"
    else:
        # Fallback
        return str(type_node)


def expression_to_string(expr) -> str:
    """Convert an expression node to a string representation."""
    if hasattr(expr, "value"):
        return str(expr.value)
    elif hasattr(expr, "name"):
        return str(expr.name)
    elif isinstance(expr, (int, float, str)):
        return str(expr)
    else:
        return str(expr)


def get_array_size_from_node(node) -> Optional[Union[int, str]]:
    """Extract array size from an AST node."""
    if hasattr(node, "size"):
        if node.size is None:
            return None
        try:
            return int(node.size)
        except (ValueError, TypeError):
            return str(node.size)
    return None


def is_legacy_ast_node(node) -> bool:
    """Check if node uses legacy AST structure."""
    return hasattr(node, "vtype") and isinstance(getattr(node, "vtype", None), str)


def normalize_operator(operator: str) -> str:
    """Normalize operator representations."""
    op_map = {
        "PLUS": "+",
        "MINUS": "-",
        "MULTIPLY": "*",
        "DIVIDE": "/",
        "MODULO": "%",
        "EQUALS": "=",
        "EQUAL": "==",
        "NOT_EQUAL": "!=",
        "LESS_THAN": "<",
        "GREATER_THAN": ">",
        "LESS_EQUAL": "<=",
        "GREATER_EQUAL": ">=",
        "LOGICAL_AND": "&&",
        "LOGICAL_OR": "||",
        "BITWISE_AND": "&",
        "BITWISE_OR": "|",
        "BITWISE_XOR": "^",
        "BITWISE_NOT": "~",
        "LOGICAL_NOT": "!",
        "ASSIGN_ADD": "+=",
        "ASSIGN_SUB": "-=",
        "ASSIGN_MUL": "*=",
        "ASSIGN_DIV": "/=",
        "ASSIGN_MOD": "%=",
        "ASSIGN_AND": "&=",
        "ASSIGN_OR": "|=",
        "ASSIGN_XOR": "^=",
        "SHIFT_LEFT": "<<",
        "SHIFT_RIGHT": ">>",
        "ASSIGN_SHIFT_LEFT": "<<=",
        "ASSIGN_SHIFT_RIGHT": ">>=",
    }

    return op_map.get(operator, operator)


def safe_visit_node(visitor, node, method_name: str = None):
    """Safely visit a node with fallback handling."""
    if node is None:
        return ""

    if method_name:
        method = getattr(visitor, method_name, None)
        if method:
            return method(node)

    # Try generic visit
    if hasattr(visitor, "visit"):
        return visitor.visit(node)
    elif hasattr(visitor, "generate_expression"):
        return visitor.generate_expression(node)
    else:
        return str(node)


def extract_includes_from_ast(ast) -> List[str]:
    """Extract include statements from AST."""
    includes = []

    if hasattr(ast, "includes"):
        for include in ast.includes:
            if hasattr(include, "content"):
                includes.append(include.content)
            else:
                includes.append(str(include))

    return includes


def extract_structs_from_ast(ast) -> List:
    """Extract struct definitions from AST."""
    if hasattr(ast, "structs"):
        return ast.structs
    else:
        return []


def extract_functions_from_ast(ast) -> List:
    """Extract function definitions from AST."""
    if hasattr(ast, "functions"):
        return ast.functions
    else:
        return []


def extract_global_variables_from_ast(ast) -> List:
    """Extract global variables from AST."""
    if hasattr(ast, "global_variables"):
        return ast.global_variables
    else:
        return []


def extract_cbuffers_from_ast(ast) -> List:
    """Extract constant buffers from AST."""
    if hasattr(ast, "cbuffers"):
        return ast.cbuffers
    elif hasattr(ast, "constants"):
        return ast.constants
    else:
        return []

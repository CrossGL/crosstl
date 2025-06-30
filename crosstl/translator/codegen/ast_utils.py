"""
AST Utilities for CrossGL Code Generators

This module provides comprehensive utilities for working with the new AST structure,
including type conversion, semantic extraction, and compatibility functions for
all backends.
"""

from typing import Optional, List
from ..ast import (
    TypeNode,
    PrimitiveType,
    VectorType,
    MatrixType,
    ArrayType,
    PointerType,
    ReferenceType,
    FunctionType,
    GenericType,
    NamedType,
    StructMemberNode,
    VariableNode,
    ParameterNode,
    AttributeNode,
    FunctionNode,
    StructNode,
    ExpressionNode,
    StatementNode,
    BlockNode,
)


class ASTUtils:
    """Comprehensive utilities for AST processing and type conversion."""

    @staticmethod
    def get_type_string(type_node: TypeNode, backend: str = "generic") -> str:
        """Convert a TypeNode to a string representation for a specific backend."""
        if isinstance(type_node, PrimitiveType):
            return ASTUtils._map_primitive_type(type_node.name, backend)

        elif isinstance(type_node, VectorType):
            element_type = ASTUtils.get_type_string(type_node.element_type, backend)
            return ASTUtils._map_vector_type(element_type, type_node.size, backend)

        elif isinstance(type_node, MatrixType):
            element_type = ASTUtils.get_type_string(type_node.element_type, backend)
            return ASTUtils._map_matrix_type(
                element_type, type_node.rows, type_node.cols, backend
            )

        elif isinstance(type_node, ArrayType):
            element_type = ASTUtils.get_type_string(type_node.element_type, backend)
            if type_node.size is not None:
                if isinstance(type_node.size, int):
                    return f"{element_type}[{type_node.size}]"
                else:
                    # Size is an expression
                    return f"{element_type}[{ASTUtils.expression_to_string(type_node.size)}]"
            else:
                return f"{element_type}[]"

        elif isinstance(type_node, PointerType):
            pointee_type = ASTUtils.get_type_string(type_node.pointee_type, backend)
            return ASTUtils._map_pointer_type(
                pointee_type, type_node.is_mutable, backend
            )

        elif isinstance(type_node, ReferenceType):
            referenced_type = ASTUtils.get_type_string(
                type_node.referenced_type, backend
            )
            return ASTUtils._map_reference_type(
                referenced_type, type_node.is_mutable, backend
            )

        elif isinstance(type_node, FunctionType):
            return_type = ASTUtils.get_type_string(type_node.return_type, backend)
            param_types = [
                ASTUtils.get_type_string(pt, backend) for pt in type_node.param_types
            ]
            return ASTUtils._map_function_type(return_type, param_types, backend)

        elif isinstance(type_node, GenericType):
            return type_node.name  # Generic types stay as-is

        elif isinstance(type_node, NamedType):
            if type_node.generic_args:
                args = [
                    ASTUtils.get_type_string(arg, backend)
                    for arg in type_node.generic_args
                ]
                return f"{type_node.name}<{', '.join(args)}>"
            return type_node.name

        else:
            # Fallback for string types or unknown
            return str(type_node)

    @staticmethod
    def _map_primitive_type(type_name: str, backend: str) -> str:
        """Map primitive type names to backend-specific types."""
        type_mappings = {
            "generic": {
                "void": "void",
                "bool": "bool",
                "int": "int",
                "float": "float",
                "double": "double",
                "char": "char",
                "uint": "uint",
            },
            "metal": {
                "void": "void",
                "bool": "bool",
                "int": "int",
                "float": "float",
                "double": "double",
                "char": "int",
                "uint": "uint",
                "half": "half",
            },
            "directx": {
                "void": "void",
                "bool": "bool",
                "int": "int",
                "float": "float",
                "double": "double",
                "char": "int",
                "uint": "uint",
            },
            "opengl": {
                "void": "void",
                "bool": "bool",
                "int": "int",
                "float": "float",
                "double": "double",
                "char": "int",
                "uint": "uint",
            },
            "vulkan": {
                "void": "void",
                "bool": "bool",
                "int": "int",
                "float": "float",
                "double": "double",
                "char": "int",
                "uint": "uint",
            },
            "rust": {
                "void": "()",
                "bool": "bool",
                "int": "i32",
                "float": "f32",
                "double": "f64",
                "char": "i8",
                "uint": "u32",
            },
            "cuda": {
                "void": "void",
                "bool": "bool",
                "int": "int",
                "float": "float",
                "double": "double",
                "char": "char",
                "uint": "unsigned int",
            },
            "hip": {
                "void": "void",
                "bool": "bool",
                "int": "int",
                "float": "float",
                "double": "double",
                "char": "char",
                "uint": "unsigned int",
            },
            "mojo": {
                "void": "None",
                "bool": "Bool",
                "int": "Int32",
                "float": "Float32",
                "double": "Float64",
                "char": "Int8",
                "uint": "UInt32",
            },
        }

        mapping = type_mappings.get(backend, type_mappings["generic"])
        return mapping.get(type_name, type_name)

    @staticmethod
    def _map_vector_type(element_type: str, size: int, backend: str) -> str:
        """Map vector types to backend-specific representations."""
        if backend == "metal":
            return f"{element_type}{size}"
        elif backend == "directx":
            return f"{element_type}{size}"
        elif backend in ["opengl", "vulkan"]:
            if element_type == "float":
                return f"vec{size}"
            elif element_type == "int":
                return f"ivec{size}"
            elif element_type == "uint":
                return f"uvec{size}"
            elif element_type == "bool":
                return f"bvec{size}"
        elif backend == "rust":
            return f"Vec{size}<{element_type}>"
        elif backend in ["cuda", "hip"]:
            return f"{element_type}{size}"
        elif backend == "mojo":
            return f"SIMD[DType.{element_type.lower()}, {size}]"

        return f"{element_type}{size}"  # Fallback

    @staticmethod
    def _map_matrix_type(element_type: str, rows: int, cols: int, backend: str) -> str:
        """Map matrix types to backend-specific representations."""
        if backend == "metal":
            return f"{element_type}{cols}x{rows}"
        elif backend == "directx":
            return f"{element_type}{rows}x{cols}"
        elif backend in ["opengl", "vulkan"]:
            if rows == cols:
                return f"mat{rows}"
            else:
                return f"mat{cols}x{rows}"
        elif backend == "rust":
            return f"Mat{rows}x{cols}<{element_type}>"
        elif backend in ["cuda", "hip"]:
            return f"{element_type}{rows}x{cols}"
        elif backend == "mojo":
            return f"Matrix[DType.{element_type.lower()}, {rows}, {cols}]"

        return f"{element_type}{rows}x{cols}"  # Fallback

    @staticmethod
    def _map_pointer_type(pointee_type: str, is_mutable: bool, backend: str) -> str:
        """Map pointer types to backend-specific representations."""
        if backend == "rust":
            return f"*{'mut' if is_mutable else 'const'} {pointee_type}"
        elif backend in ["cuda", "hip"]:
            return f"{pointee_type}*"
        else:
            return f"{pointee_type}*"

    @staticmethod
    def _map_reference_type(
        referenced_type: str, is_mutable: bool, backend: str
    ) -> str:
        """Map reference types to backend-specific representations."""
        if backend == "rust":
            return f"&{'mut ' if is_mutable else ''}{referenced_type}"
        else:
            return f"{referenced_type}&"

    @staticmethod
    def _map_function_type(
        return_type: str, param_types: List[str], backend: str
    ) -> str:
        """Map function types to backend-specific representations."""
        params = ", ".join(param_types)
        if backend == "rust":
            return f"fn({params}) -> {return_type}"
        elif backend == "mojo":
            return f"fn({params}) -> {return_type}"
        else:
            return f"{return_type}({params})"

    @staticmethod
    def get_semantic_from_attributes(attributes: List[AttributeNode]) -> Optional[str]:
        """Extract semantic information from attribute list."""
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
            "TEXCOORD4",
            "TEXCOORD5",
            "TEXCOORD6",
            "TEXCOORD7",
            "COLOR0",
            "COLOR1",
        ]

        for attr in attributes:
            if attr.name in semantic_attrs:
                return attr.name

        return None

    @staticmethod
    def get_member_info(member: StructMemberNode, backend: str = "generic"):
        """Extract complete member information for code generation."""
        return {
            "name": member.name,
            "type": ASTUtils.get_type_string(member.member_type, backend),
            "semantic": ASTUtils.get_semantic_from_attributes(member.attributes),
            "attributes": member.attributes,
            "visibility": member.visibility,
            "default_value": member.default_value,
        }

    @staticmethod
    def get_variable_info(variable: VariableNode, backend: str = "generic"):
        """Extract complete variable information for code generation."""
        return {
            "name": variable.name,
            "type": ASTUtils.get_type_string(variable.var_type, backend),
            "semantic": ASTUtils.get_semantic_from_attributes(variable.attributes),
            "attributes": variable.attributes,
            "qualifiers": variable.qualifiers,
            "is_mutable": variable.is_mutable,
            "initial_value": variable.initial_value,
            "visibility": variable.visibility,
        }

    @staticmethod
    def get_parameter_info(parameter: ParameterNode, backend: str = "generic"):
        """Extract complete parameter information for code generation."""
        return {
            "name": parameter.name,
            "type": ASTUtils.get_type_string(parameter.param_type, backend),
            "semantic": ASTUtils.get_semantic_from_attributes(parameter.attributes),
            "attributes": parameter.attributes,
            "is_mutable": parameter.is_mutable,
            "default_value": parameter.default_value,
        }

    @staticmethod
    def get_function_info(function: FunctionNode, backend: str = "generic"):
        """Extract complete function information for code generation."""
        return {
            "name": function.name,
            "return_type": ASTUtils.get_type_string(function.return_type, backend),
            "parameters": [
                ASTUtils.get_parameter_info(p, backend) for p in function.parameters
            ],
            "qualifiers": function.qualifiers,
            "attributes": function.attributes,
            "visibility": function.visibility,
            "is_unsafe": function.is_unsafe,
            "is_async": function.is_async,
            "body": function.body,
        }

    @staticmethod
    def expression_to_string(expr: ExpressionNode) -> str:
        """Convert an expression node to a string representation."""
        # This is a simplified version - in practice, you'd need a full expression visitor
        if hasattr(expr, "value"):
            return str(expr.value)
        elif hasattr(expr, "name"):
            return str(expr.name)
        else:
            return str(expr)

    @staticmethod
    def is_legacy_ast_node(node) -> bool:
        """Check if a node is from the legacy AST structure."""
        # Check for old-style attributes
        return hasattr(node, "vtype") and isinstance(getattr(node, "vtype", None), str)

    @staticmethod
    def get_legacy_compatible_type(node, backend: str = "generic") -> str:
        """Get type string with legacy compatibility."""
        if ASTUtils.is_legacy_ast_node(node):
            # Old AST structure
            return getattr(node, "vtype", "float")
        else:
            # New AST structure
            if hasattr(node, "var_type"):
                return ASTUtils.get_type_string(node.var_type, backend)
            elif hasattr(node, "member_type"):
                return ASTUtils.get_type_string(node.member_type, backend)
            elif hasattr(node, "param_type"):
                return ASTUtils.get_type_string(node.param_type, backend)
            else:
                return "float"  # Fallback

    @staticmethod
    def get_legacy_compatible_semantic(node) -> Optional[str]:
        """Get semantic information with legacy compatibility."""
        if ASTUtils.is_legacy_ast_node(node):
            # Old AST structure
            return getattr(node, "semantic", None)
        else:
            # New AST structure
            if hasattr(node, "attributes"):
                return ASTUtils.get_semantic_from_attributes(node.attributes)
            else:
                return None

    @staticmethod
    def safe_get_body_statements(body):
        """Safely extract statements from function body, handling both old and new AST."""
        if body is None:
            return []
        elif isinstance(body, BlockNode):
            return body.statements
        elif isinstance(body, list):
            return body
        else:
            return [body]

    @staticmethod
    def safe_get_function_qualifier(function: FunctionNode) -> Optional[str]:
        """Safely get function qualifier, handling both old and new AST."""
        # New AST uses qualifiers list
        if hasattr(function, "qualifiers") and function.qualifiers:
            return function.qualifiers[0]
        # Legacy compatibility
        elif hasattr(function, "qualifier"):
            return function.qualifier
        else:
            return None

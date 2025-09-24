"""
Base Code Generation Infrastructure for all backends.
This module provides common code generation patterns to reduce duplication.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Union
from .base_ast import (
    ASTNode,
    BaseArrayAccessNode,
    BaseAssignmentNode,
    BaseBinaryOpNode,
    BaseBreakNode,
    BaseCaseNode,
    BaseContinueNode,
    BaseForNode,
    BaseFunctionCallNode,
    BaseFunctionNode,
    BaseIfNode,
    BaseMemberAccessNode,
    BaseReturnNode,
    BaseShaderNode,
    BaseStructNode,
    BaseSwitchNode,
    BaseTernaryOpNode,
    BaseUnaryOpNode,
    BaseVariableNode,
    BaseWhileNode,
    NodeType,
)


class TypeMapping:
    """Centralized type mapping system for all backends."""

    # Universal type mappings - all backends should support these concepts
    UNIVERSAL_TYPES = {
        "void": "void",
        "bool": "bool",
        "int": "int",
        "uint": "uint",
        "float": "float",
        "double": "double",
        # Vector types
        "vec2": "vec2",
        "vec3": "vec3",
        "vec4": "vec4",
        "ivec2": "ivec2",
        "ivec3": "ivec3",
        "ivec4": "ivec4",
        "uvec2": "uvec2",
        "uvec3": "uvec3",
        "uvec4": "uvec4",
        "bvec2": "bvec2",
        "bvec3": "bvec3",
        "bvec4": "bvec4",
        # Matrix types
        "mat2": "mat2",
        "mat3": "mat3",
        "mat4": "mat4",
        # Texture types
        "sampler2D": "sampler2D",
        "samplerCube": "samplerCube",
        "sampler3D": "sampler3D",
    }

    # Backend-specific type mappings
    BACKEND_MAPPINGS = {
        "cuda": {
            "vec2": "float2",
            "vec3": "float3",
            "vec4": "float4",
            "ivec2": "int2",
            "ivec3": "int3",
            "ivec4": "int4",
            "uvec2": "uint2",
            "uvec3": "uint3",
            "uvec4": "uint4",
            "mat2": "float2x2",
            "mat3": "float3x3",
            "mat4": "float4x4",
        },
        "metal": {
            "vec2": "float2",
            "vec3": "float3",
            "vec4": "float4",
            "ivec2": "int2",
            "ivec3": "int3",
            "ivec4": "int4",
            "uvec2": "uint2",
            "uvec3": "uint3",
            "uvec4": "uint4",
            "mat2": "float2x2",
            "mat3": "float3x3",
            "mat4": "float4x4",
            "sampler2D": "texture2d<float>",
            "samplerCube": "texturecube<float>",
        },
        "directx": {
            "vec2": "float2",
            "vec3": "float3",
            "vec4": "float4",
            "ivec2": "int2",
            "ivec3": "int3",
            "ivec4": "int4",
            "uvec2": "uint2",
            "uvec3": "uint3",
            "uvec4": "uint4",
            "mat2": "float2x2",
            "mat3": "float3x3",
            "mat4": "float4x4",
            "sampler2D": "Texture2D",
            "samplerCube": "TextureCube",
        },
        "opengl": {
            # GLSL uses the universal types mostly as-is
        },
        "vulkan": {
            # Vulkan SPIR-V uses GLSL-like types
        },
        "rust": {
            "void": "()",
            "int": "i32",
            "uint": "u32",
            "float": "f32",
            "double": "f64",
            "vec2": "Vec2<f32>",
            "vec3": "Vec3<f32>",
            "vec4": "Vec4<f32>",
            "ivec2": "Vec2<i32>",
            "ivec3": "Vec3<i32>",
            "ivec4": "Vec4<i32>",
            "uvec2": "Vec2<u32>",
            "uvec3": "Vec3<u32>",
            "uvec4": "Vec4<u32>",
            "mat2": "Mat2<f32>",
            "mat3": "Mat3<f32>",
            "mat4": "Mat4<f32>",
        },
        "mojo": {
            "void": "None",
            "int": "Int32",
            "uint": "UInt32",
            "float": "Float32",
            "double": "Float64",
            "vec2": "SIMD[DType.float32, 2]",
            "vec3": "SIMD[DType.float32, 3]",
            "vec4": "SIMD[DType.float32, 4]",
            "ivec2": "SIMD[DType.int32, 2]",
            "ivec3": "SIMD[DType.int32, 3]",
            "ivec4": "SIMD[DType.int32, 4]",
            "mat2": "Matrix[DType.float32, 2, 2]",
            "mat3": "Matrix[DType.float32, 3, 3]",
            "mat4": "Matrix[DType.float32, 4, 4]",
        },
        "hip": {
            "vec2": "float2",
            "vec3": "float3",
            "vec4": "float4",
            "ivec2": "int2",
            "ivec3": "int3",
            "ivec4": "int4",
            "uvec2": "uint2",
            "uvec3": "uint3",
            "uvec4": "uint4",
            "mat2": "float2x2",
            "mat3": "float3x3",
            "mat4": "float4x4",
        },
        "slang": {
            "vec2": "float2",
            "vec3": "float3",
            "vec4": "float4",
            "ivec2": "int2",
            "ivec3": "int3",
            "ivec4": "int4",
            "uvec2": "uint2",
            "uvec3": "uint3",
            "uvec4": "uint4",
            "mat2": "float2x2",
            "mat3": "float3x3",
            "mat4": "float4x4",
        },
    }

    @classmethod
    def map_type(cls, type_name: str, backend: str) -> str:
        """Map a universal type to a backend-specific type."""
        backend = backend.lower()

        # First try backend-specific mapping
        if backend in cls.BACKEND_MAPPINGS:
            backend_map = cls.BACKEND_MAPPINGS[backend]
            if type_name in backend_map:
                return backend_map[type_name]

        # Fallback to universal type
        return cls.UNIVERSAL_TYPES.get(type_name, type_name)

    @classmethod
    def add_backend_mapping(cls, backend: str, type_map: Dict[str, str]):
        """Add or update type mappings for a backend."""
        backend = backend.lower()
        if backend not in cls.BACKEND_MAPPINGS:
            cls.BACKEND_MAPPINGS[backend] = {}
        cls.BACKEND_MAPPINGS[backend].update(type_map)


class SemanticMapping:
    """Centralized semantic mapping system for all backends."""

    # Universal semantics that map to different backend equivalents
    UNIVERSAL_SEMANTICS = {
        # Vertex input semantics
        "Position": "position",
        "Normal": "normal",
        "Tangent": "tangent",
        "Binormal": "binormal",
        "TexCoord": "texcoord",
        "TexCoord0": "texcoord0",
        "TexCoord1": "texcoord1",
        "Color": "color",
        "VertexID": "vertex_id",
        "InstanceID": "instance_id",
        # Vertex output semantics
        "gl_Position": "position_output",
        # Fragment input semantics
        "gl_FragCoord": "fragment_coord",
        "gl_FrontFacing": "front_facing",
        # Fragment output semantics
        "gl_FragColor": "color_output",
        "gl_FragDepth": "depth_output",
    }

    BACKEND_MAPPINGS = {
        "directx": {
            "position": "SV_Position",
            "vertex_id": "SV_VertexID",
            "instance_id": "SV_InstanceID",
            "color_output": "SV_Target",
            "depth_output": "SV_Depth",
        },
        "metal": {
            "position": "[[position]]",
            "vertex_id": "[[vertex_id]]",
            "instance_id": "[[instance_id]]",
            "color_output": "[[color(0)]]",
            "depth_output": "[[depth(any)]]",
        },
        "opengl": {
            "position_output": "gl_Position",
            "color_output": "gl_FragColor",
            "depth_output": "gl_FragDepth",
        },
        "vulkan": {
            "position_output": "gl_Position",
            "color_output": "gl_FragColor",
            "depth_output": "gl_FragDepth",
        },
    }

    @classmethod
    def map_semantic(cls, semantic: str, backend: str) -> str:
        """Map a universal semantic to a backend-specific semantic."""
        backend = backend.lower()

        # First try backend-specific mapping
        if backend in cls.BACKEND_MAPPINGS:
            backend_map = cls.BACKEND_MAPPINGS[backend]
            if semantic in backend_map:
                return backend_map[semantic]

        # Check if it's a universal semantic
        if semantic in cls.UNIVERSAL_SEMANTICS:
            universal = cls.UNIVERSAL_SEMANTICS[semantic]
            # Try to map the universal semantic
            if backend in cls.BACKEND_MAPPINGS:
                backend_map = cls.BACKEND_MAPPINGS[backend]
                return backend_map.get(universal, semantic)

        # Return as-is if no mapping found
        return semantic


class OperatorMapping:
    """Centralized operator mapping system."""

    # Standard operators that are mostly consistent
    STANDARD_OPERATORS = {
        "=": "=",
        "+=": "+=",
        "-=": "-=",
        "*=": "*=",
        "/=": "/=",
        "%=": "%=",
        "+": "+",
        "-": "-",
        "*": "*",
        "/": "/",
        "%": "%",
        "==": "==",
        "!=": "!=",
        "<": "<",
        ">": ">",
        "<=": "<=",
        ">=": ">=",
        "&&": "&&",
        "||": "||",
        "!": "!",
        "&": "&",
        "|": "|",
        "^": "^",
        "~": "~",
        "<<": "<<",
        ">>": ">>",
        ".": ".",
        "->": "->",
        "?": "?",
        ":": ":",
    }

    # Backend-specific operator overrides
    BACKEND_OVERRIDES = {
        "rust": {
            "->": ".",  # Rust doesn't have pointer access
        },
        "mojo": {
            # Mojo might have different operator syntax
        },
    }

    @classmethod
    def map_operator(cls, operator: str, backend: str) -> str:
        """Map an operator to backend-specific syntax."""
        backend = backend.lower()

        # Check for backend override
        if backend in cls.BACKEND_OVERRIDES:
            override = cls.BACKEND_OVERRIDES[backend].get(operator)
            if override is not None:
                return override

        # Return standard mapping
        return cls.STANDARD_OPERATORS.get(operator, operator)


class BaseCodeGenerator(ABC):
    """Base code generator providing common functionality."""

    def __init__(self, backend_name: str):
        self.backend_name = backend_name.lower()
        self.indent_level = 0
        self.indent_string = "    "  # 4 spaces
        self.output_lines = []

        # Visitors for different node types
        self.visitors = self._initialize_visitors()

    def _initialize_visitors(self) -> Dict[NodeType, str]:
        """Initialize visitor method mappings."""
        return {
            NodeType.SHADER: "visit_shader_node",
            NodeType.FUNCTION: "visit_function_node",
            NodeType.STRUCT: "visit_struct_node",
            NodeType.VARIABLE: "visit_variable_node",
            NodeType.BINARY_OP: "visit_binary_op_node",
            NodeType.UNARY_OP: "visit_unary_op_node",
            NodeType.ASSIGNMENT: "visit_assignment_node",
            NodeType.FUNCTION_CALL: "visit_function_call_node",
            NodeType.MEMBER_ACCESS: "visit_member_access_node",
            NodeType.ARRAY_ACCESS: "visit_array_access_node",
            NodeType.IF: "visit_if_node",
            NodeType.FOR: "visit_for_node",
            NodeType.WHILE: "visit_while_node",
            NodeType.RETURN: "visit_return_node",
            NodeType.BREAK: "visit_break_node",
            NodeType.CONTINUE: "visit_continue_node",
            NodeType.TERNARY_OP: "visit_ternary_op_node",
            NodeType.SWITCH: "visit_switch_node",
            NodeType.CASE: "visit_case_node",
        }

    @abstractmethod
    def generate(self, ast_node: ASTNode) -> str:
        """Generate code from AST. Must be implemented by subclasses."""

    def visit(self, node: ASTNode) -> str:
        """Visit a node using the appropriate visitor method."""
        if node is None:
            return ""

        # Handle primitive types
        if isinstance(node, (str, int, float, bool)):
            return str(node)

        # Handle lists
        if isinstance(node, list):
            return self.visit_list(node)

        # Handle AST nodes
        if isinstance(node, ASTNode):
            visitor_name = self.visitors.get(node.node_type)
            if visitor_name and hasattr(self, visitor_name):
                visitor = getattr(self, visitor_name)
                return visitor(node)
            else:
                return self.generic_visit(node)

        return str(node)

    def generic_visit(self, node: ASTNode) -> str:
        """Generic visitor for unhandled node types."""
        return f"/* Unhandled node type: {type(node).__name__} */"

    def visit_list(self, nodes: List[ASTNode]) -> str:
        """Visit a list of nodes."""
        return "\n".join(self.visit(node) for node in nodes if node)

    def emit(self, code: str = "") -> None:
        """Emit a line of code with proper indentation."""
        if code.strip():
            self.output_lines.append(self.indent_string * self.indent_level + code)
        else:
            self.output_lines.append("")

    def emit_raw(self, code: str) -> None:
        """Emit code without indentation."""
        self.output_lines.append(code)

    def increase_indent(self) -> None:
        """Increase indentation level."""
        self.indent_level += 1

    def decrease_indent(self) -> None:
        """Decrease indentation level."""
        if self.indent_level > 0:
            self.indent_level -= 1

    def get_output(self) -> str:
        """Get the generated code."""
        return "\n".join(self.output_lines)

    def clear_output(self) -> None:
        """Clear the output buffer."""
        self.output_lines.clear()

    # Common type mapping
    def map_type(self, type_name: str) -> str:
        """Map a type to backend-specific syntax."""
        return TypeMapping.map_type(type_name, self.backend_name)

    def map_semantic(self, semantic: str) -> str:
        """Map a semantic to backend-specific syntax."""
        mapped = SemanticMapping.map_semantic(semantic, self.backend_name)
        if mapped and mapped != semantic:
            return f"@ {mapped}" if not mapped.startswith("@") else mapped
        return ""

    def map_operator(self, operator: str) -> str:
        """Map an operator to backend-specific syntax."""
        return OperatorMapping.map_operator(operator, self.backend_name)

    # Common visitor implementations

    def visit_shader_node(self, node: BaseShaderNode) -> str:
        """Visit shader node - common structure."""
        self.clear_output()
        self.emit(f"// Generated {self.backend_name.upper()} code")
        self.emit()

        # Generate includes/imports
        if node.includes:
            for include in node.includes:
                self.emit(self.visit(include))
            self.emit()

        # Generate structs
        for struct in node.structs:
            self.emit(self.visit(struct))
            self.emit()

        # Generate global variables
        for var in node.global_variables:
            self.emit(self.visit(var))

        if node.global_variables:
            self.emit()

        # Generate functions
        for func in node.functions:
            self.emit(self.visit(func))
            self.emit()

        return self.get_output()

    def visit_struct_node(self, node: BaseStructNode) -> str:
        """Visit struct node - common structure."""
        result = f"struct {node.name} {{\n"

        self.increase_indent()
        for member in node.members:
            member_code = self.visit(member)
            result += f"{self.indent_string * self.indent_level}{member_code};\n"
        self.decrease_indent()

        result += "};"
        return result

    def visit_function_node(self, node: BaseFunctionNode) -> str:
        """Visit function node - common structure."""
        # Build function signature
        return_type = self.map_type(node.return_type)
        params = []
        for param in node.params:
            param_type = self.map_type(param.vtype)
            params.append(f"{param_type} {param.name}")

        params_str = ", ".join(params)
        signature = f"{return_type} {node.name}({params_str})"

        # Add qualifiers
        if node.qualifiers:
            qualifiers_str = " ".join(node.qualifiers)
            signature = f"{qualifiers_str} {signature}"

        # Generate body
        if node.body:
            result = f"{signature} {{\n"
            self.increase_indent()
            for stmt in node.body:
                stmt_code = self.visit(stmt)
                if stmt_code:
                    result += f"{self.indent_string * self.indent_level}{stmt_code}\n"
            self.decrease_indent()
            result += "}"
        else:
            result = f"{signature};"

        return result

    def visit_variable_node(self, node: BaseVariableNode) -> str:
        """Visit variable node - common structure."""
        var_type = self.map_type(node.vtype)
        result = f"{var_type} {node.name}"

        if node.value:
            value_code = self.visit(node.value)
            result += f" = {value_code}"

        return result

    def visit_binary_op_node(self, node: BaseBinaryOpNode) -> str:
        """Visit binary operation node."""
        left = self.visit(node.left)
        right = self.visit(node.right)
        op = self.map_operator(node.op)
        return f"({left} {op} {right})"

    def visit_unary_op_node(self, node: BaseUnaryOpNode) -> str:
        """Visit unary operation node."""
        operand = self.visit(node.operand)
        op = self.map_operator(node.op)

        if node.is_postfix:
            return f"({operand}{op})"
        else:
            return f"({op}{operand})"

    def visit_assignment_node(self, node: BaseAssignmentNode) -> str:
        """Visit assignment node."""
        left = self.visit(node.left)
        right = self.visit(node.right)
        op = self.map_operator(node.operator)
        return f"{left} {op} {right};"

    def visit_function_call_node(self, node: BaseFunctionCallNode) -> str:
        """Visit function call node."""
        if isinstance(node.name, str):
            func_name = node.name
        else:
            func_name = self.visit(node.name)

        args = []
        for arg in node.args:
            args.append(self.visit(arg))

        args_str = ", ".join(args)
        return f"{func_name}({args_str})"

    def visit_member_access_node(self, node: BaseMemberAccessNode) -> str:
        """Visit member access node."""
        obj = self.visit(node.object)
        op = "->" if node.is_pointer else "."
        return f"{obj}{op}{node.member}"

    def visit_array_access_node(self, node: BaseArrayAccessNode) -> str:
        """Visit array access node."""
        array = self.visit(node.array)
        index = self.visit(node.index)
        return f"{array}[{index}]"

    def visit_if_node(self, node: BaseIfNode) -> str:
        """Visit if node."""
        condition = self.visit(node.condition)
        result = f"if ({condition}) {{\n"

        self.increase_indent()
        for stmt in node.if_body:
            stmt_code = self.visit(stmt)
            if stmt_code:
                result += f"{self.indent_string * self.indent_level}{stmt_code}\n"
        self.decrease_indent()

        result += f"{self.indent_string * self.indent_level}}}"

        if node.else_body:
            result += " else {\n"
            self.increase_indent()
            for stmt in node.else_body:
                stmt_code = self.visit(stmt)
                if stmt_code:
                    result += f"{self.indent_string * self.indent_level}{stmt_code}\n"
            self.decrease_indent()
            result += f"{self.indent_string * self.indent_level}}}"

        return result

    def visit_for_node(self, node: BaseForNode) -> str:
        """Visit for node."""
        init = self.visit(node.init) if node.init else ""
        condition = self.visit(node.condition) if node.condition else ""
        update = self.visit(node.update) if node.update else ""

        result = f"for ({init}; {condition}; {update}) {{\n"

        self.increase_indent()
        for stmt in node.body:
            stmt_code = self.visit(stmt)
            if stmt_code:
                result += f"{self.indent_string * self.indent_level}{stmt_code}\n"
        self.decrease_indent()

        result += f"{self.indent_string * self.indent_level}}}"
        return result

    def visit_while_node(self, node: BaseWhileNode) -> str:
        """Visit while node."""
        condition = self.visit(node.condition)
        result = f"while ({condition}) {{\n"

        self.increase_indent()
        for stmt in node.body:
            stmt_code = self.visit(stmt)
            if stmt_code:
                result += f"{self.indent_string * self.indent_level}{stmt_code}\n"
        self.decrease_indent()

        result += f"{self.indent_string * self.indent_level}}}"
        return result

    def visit_return_node(self, node: BaseReturnNode) -> str:
        """Visit return node."""
        if node.value:
            value = self.visit(node.value)
            return f"return {value};"
        else:
            return "return;"

    def visit_break_node(self, node: BaseBreakNode) -> str:
        """Visit break node."""
        return "break;"

    def visit_continue_node(self, node: BaseContinueNode) -> str:
        """Visit continue node."""
        return "continue;"

    def visit_ternary_op_node(self, node: BaseTernaryOpNode) -> str:
        """Visit ternary operation node."""
        condition = self.visit(node.condition)
        true_expr = self.visit(node.true_expr)
        false_expr = self.visit(node.false_expr)
        return f"({condition} ? {true_expr} : {false_expr})"

    def visit_switch_node(self, node: BaseSwitchNode) -> str:
        """Visit switch node."""
        expression = self.visit(node.expression)
        result = f"switch ({expression}) {{\n"

        self.increase_indent()
        for case in node.cases:
            result += f"{self.indent_string * self.indent_level}{self.visit(case)}\n"

        if node.default_case:
            result += f"{self.indent_string * self.indent_level}{self.visit(node.default_case)}\n"
        self.decrease_indent()

        result += f"{self.indent_string * self.indent_level}}}"
        return result

    def visit_case_node(self, node: BaseCaseNode) -> str:
        """Visit case node."""
        if node.value:  # Regular case
            value = self.visit(node.value)
            result = f"case {value}:\n"
        else:  # Default case
            result = "default:\n"

        self.increase_indent()
        for stmt in node.body:
            stmt_code = self.visit(stmt)
            if stmt_code:
                result += f"{self.indent_string * self.indent_level}{stmt_code}\n"
        result += f"{self.indent_string * self.indent_level}break;\n"
        self.decrease_indent()

        return result


class CrossGLToCrossGLConverter(BaseCodeGenerator):
    """Converter from language-specific AST to CrossGL intermediate representation."""

    def __init__(self, source_language: str):
        super().__init__("crossgl")
        self.source_language = source_language.lower()

    def generate(self, ast_node: ASTNode) -> str:
        """Generate CrossGL IR from language-specific AST."""
        self.clear_output()
        self.emit(f"// {self.source_language.upper()} to CrossGL conversion")
        self.emit()

        result = self.visit(ast_node)
        return result if isinstance(result, str) else self.get_output()


class CrossGLToTargetConverter(BaseCodeGenerator):
    """Converter from CrossGL IR to target language."""

    def __init__(self, target_language: str):
        super().__init__(target_language)
        self.target_language = target_language.lower()

    def generate(self, ast_node: ASTNode) -> str:
        """Generate target language code from CrossGL IR."""
        self.clear_output()
        self.emit(f"// CrossGL to {self.target_language.upper()} conversion")
        self.emit()

        result = self.visit(ast_node)
        return result if isinstance(result, str) else self.get_output()


class LanguageSpecificCodeGen(BaseCodeGenerator):
    """Base class for language-specific code generators."""

    def __init__(self, target_language: str):
        super().__init__(target_language)

    def generate_shader_wrapper(self, node: BaseShaderNode) -> str:
        """Generate language-specific shader wrapper."""
        # Default implementation - can be overridden
        return self.visit_shader_node(node)

    def generate_includes(self) -> str:
        """Generate language-specific includes."""
        # Override in subclasses
        return ""

    def generate_shader_entry_point(self, func: BaseFunctionNode, stage: str) -> str:
        """Generate language-specific shader entry point."""
        # Override in subclasses
        return self.visit_function_node(func)


# Factory for creating appropriate code generators
class CodeGenFactory:
    """Factory for creating code generators."""

    _crossgl_converter_registry: Dict[str, type] = {}
    _target_generator_registry: Dict[str, type] = {}

    @classmethod
    def register_crossgl_converter(cls, language: str, converter_class: type):
        """Register a CrossGL converter for a source language."""
        cls._crossgl_converter_registry[language.lower()] = converter_class

    @classmethod
    def register_target_generator(cls, language: str, generator_class: type):
        """Register a target code generator for a language."""
        cls._target_generator_registry[language.lower()] = generator_class

    @classmethod
    def create_crossgl_converter(
        cls, source_language: str
    ) -> CrossGLToCrossGLConverter:
        """Create a CrossGL converter for the source language."""
        language = source_language.lower()
        if language in cls._crossgl_converter_registry:
            return cls._crossgl_converter_registry[language]()
        else:
            return CrossGLToCrossGLConverter(language)

    @classmethod
    def create_target_generator(cls, target_language: str) -> CrossGLToTargetConverter:
        """Create a target generator for the target language."""
        language = target_language.lower()
        if language in cls._target_generator_registry:
            return cls._target_generator_registry[language]()
        else:
            return CrossGLToTargetConverter(language)


# Utility functions for code generation


def format_function_signature(
    return_type: str,
    name: str,
    params: List[str],
    qualifiers: Optional[List[str]] = None,
) -> str:
    """Format a function signature consistently."""
    qualifier_str = " ".join(qualifiers) + " " if qualifiers else ""
    params_str = ", ".join(params)
    return f"{qualifier_str}{return_type} {name}({params_str})"


def format_struct_definition(name: str, members: List[str]) -> str:
    """Format a struct definition consistently."""
    result = f"struct {name} {{\n"
    for member in members:
        result += f"    {member};\n"
    result += "};"
    return result


def format_attribute(name: str, args: Optional[List[str]] = None) -> str:
    """Format an attribute consistently."""
    if args:
        args_str = ", ".join(args)
        return f"@{name}({args_str})"
    else:
        return f"@{name}"


def escape_string_literal(value: str) -> str:
    """Escape a string literal for code generation."""
    # Basic escaping - can be enhanced
    value = value.replace("\\", "\\\\")
    value = value.replace('"', '\\"')
    value = value.replace("\n", "\\n")
    value = value.replace("\t", "\\t")
    return f'"{value}"'


def format_numeric_literal(value: Union[int, float], suffix: str = "") -> str:
    """Format a numeric literal with optional suffix."""
    return f"{value}{suffix}"

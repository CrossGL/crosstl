"""
HIP to CrossGL Code Generator

This module provides code generation functionality to convert HIP (HIP Is a Portable GPU Runtime)
AST nodes to CrossGL intermediate representation.
"""

from typing import List, Optional
from .HipAst import *
from ...translator.ast import *


class HipToCrossGLConverter:
    """Converts HIP AST to CrossGL AST"""

    def __init__(self):
        self.variable_map = {}
        self.function_map = {}
        self.struct_map = {}
        self.current_function = None
        self.in_device_code = False

        # HIP to CrossGL type mapping
        self.type_map = {
            # Basic types
            "int": "int",
            "float": "float",
            "double": "double",
            "bool": "bool",
            "char": "int",
            "unsigned int": "uint",
            "unsigned char": "uint",
            "void": "void",
            # HIP vector types
            "int2": "ivec2",
            "int3": "ivec3",
            "int4": "ivec4",
            "float2": "vec2",
            "float3": "vec3",
            "float4": "vec4",
            "double2": "dvec2",
            "double3": "dvec3",
            "double4": "dvec4",
            "uint2": "uvec2",
            "uint3": "uvec3",
            "uint4": "uvec4",
            # HIP matrix types (map to arrays for now)
            "float2x2": "mat2",
            "float3x3": "mat3",
            "float4x4": "mat4",
            # Memory types
            "hipDeviceptr_t": "buffer",
            "texture": "sampler2D",
            "surface": "image2D",
        }

        # HIP function mappings
        self.function_map_builtin = {
            # Math functions
            "sqrtf": "sqrt",
            "sinf": "sin",
            "cosf": "cos",
            "tanf": "tan",
            "asinf": "asin",
            "acosf": "acos",
            "atanf": "atan",
            "atan2f": "atan2",
            "expf": "exp",
            "exp2f": "exp2",
            "logf": "log",
            "log2f": "log2",
            "powf": "pow",
            "fabsf": "abs",
            "floorf": "floor",
            "ceilf": "ceil",
            "fmodf": "mod",
            "fminf": "min",
            "fmaxf": "max",
            "rsqrtf": "inversesqrt",
            # Vector functions
            "make_float2": "vec2",
            "make_float3": "vec3",
            "make_float4": "vec4",
            "make_int2": "ivec2",
            "make_int3": "ivec3",
            "make_int4": "ivec4",
            "dot": "dot",
            "cross": "cross",
            "normalize": "normalize",
            "length": "length",
            "distance": "distance",
            # Texture functions
            "tex2D": "texture",
            "tex3D": "texture",
            "texCubemap": "texture",
        }

    def convert(self, node: HipASTNode) -> ASTNode:
        """Convert HIP AST node to CrossGL AST node"""
        method_name = f"convert_{type(node).__name__}"
        converter = getattr(self, method_name, self.generic_convert)
        return converter(node)

    def generic_convert(self, node: HipASTNode) -> ASTNode:
        """Generic converter for unsupported nodes"""
        raise NotImplementedError(
            f"Conversion not implemented for {type(node).__name__}"
        )

    def convert_HipProgramNode(self, node: HipProgramNode) -> ProgramNode:
        """Convert HIP program to CrossGL program"""
        statements = []

        for stmt in node.statements:
            converted = self.convert(stmt)
            if converted:
                statements.append(converted)

        return ProgramNode(statements)

    def convert_HipFunctionNode(self, node: HipFunctionNode) -> FunctionNode:
        """Convert HIP function to CrossGL function"""
        self.current_function = node.name

        # Check for device qualifiers
        is_kernel = "__global__" in node.qualifiers
        "__device__" in node.qualifiers or "__host__" in node.qualifiers

        # Convert return type
        return_type = (
            self.convert_type(node.return_type) if node.return_type else "void"
        )

        # Convert parameters
        params = []
        for param in node.parameters:
            param_type = self.convert_type(param.param_type)
            param_node = ParameterNode(param.name, param_type)
            params.append(param_node)

        # Convert body
        body = None
        if node.body:
            body = self.convert(node.body)

        # Create function node
        func_node = FunctionNode(node.name, return_type, params, body, is_kernel)

        # Add shader attributes for kernel functions
        if is_kernel:
            func_node.shader_type = "compute"

        self.current_function = None
        return func_node

    def convert_HipStructNode(self, node: HipStructNode) -> StructNode:
        """Convert HIP struct to CrossGL struct"""
        members = []

        for field in node.fields:
            if isinstance(field, HipStructMemberNode):
                member_type = self.convert_type(field.member_type)
                member_node = VariableNode(field.name, member_type)
                members.append(member_node)

        return StructNode(node.name, members)

    def convert_HipClassNode(self, node: HipClassNode) -> StructNode:
        """Convert HIP class to CrossGL struct (simplified)"""
        members = []

        for member in node.members:
            if isinstance(member, HipMemberVariableNode):
                member_type = self.convert_type(member.member_type)
                member_node = VariableNode(member.name, member_type)
                members.append(member_node)
            elif isinstance(member, HipMemberFunctionNode):
                # Skip member functions for now in struct conversion
                pass

        return StructNode(node.name, members)

    def convert_HipVariableDeclarationNode(
        self, node: HipVariableDeclarationNode
    ) -> List[VariableNode]:
        """Convert HIP variable declaration to CrossGL variables"""
        variables = []

        for var in node.variables:
            var_type = self.convert_type(var.var_type)

            # Handle array dimensions
            if var.dimensions:
                for dim in var.dimensions:
                    if dim:
                        # Convert to array type
                        var_type = (
                            f"{var_type}[{self.convert_expression_to_string(dim)}]"
                        )
                    else:
                        var_type = f"{var_type}[]"

            var_node = VariableNode(var.name, var_type)

            # Handle initializer
            if var.initializer:
                var_node.value = self.convert(var.initializer)

            variables.append(var_node)

        return variables

    def convert_HipBlockNode(self, node: HipBlockNode) -> BlockNode:
        """Convert HIP block to CrossGL block"""
        statements = []

        for stmt in node.statements:
            converted = self.convert(stmt)
            if converted:
                if isinstance(converted, list):
                    statements.extend(converted)
                else:
                    statements.append(converted)

        return BlockNode(statements)

    def convert_HipExpressionStatementNode(
        self, node: HipExpressionStatementNode
    ) -> ExpressionStatementNode:
        """Convert HIP expression statement to CrossGL expression statement"""
        expr = self.convert(node.expression)
        return ExpressionStatementNode(expr)

    def convert_HipBinaryOpNode(self, node: HipBinaryOpNode) -> BinaryOpNode:
        """Convert HIP binary operation to CrossGL binary operation"""
        left = self.convert(node.left)
        right = self.convert(node.right)

        # Map HIP operators to CrossGL operators
        op_map = {
            "&&": "and",
            "||": "or",
            "==": "==",
            "!=": "!=",
            "<": "<",
            ">": ">",
            "<=": "<=",
            ">=": ">=",
            "+": "+",
            "-": "-",
            "*": "*",
            "/": "/",
            "%": "%",
            "&": "&",
            "|": "|",
            "^": "^",
            "<<": "<<",
            ">>": ">>",
        }

        operator = op_map.get(node.operator, node.operator)
        return BinaryOpNode(left, operator, right)

    def convert_HipUnaryOpNode(self, node: HipUnaryOpNode) -> UnaryOpNode:
        """Convert HIP unary operation to CrossGL unary operation"""
        operand = self.convert(node.operand)

        # Map HIP unary operators
        op_map = {
            "!": "not",
            "-": "-",
            "+": "+",
            "~": "~",
            "++": "++",
            "--": "--",
            "*": "*",  # dereference
            "&": "&",  # address-of
        }

        operator = op_map.get(node.operator, node.operator)
        return UnaryOpNode(operator, operand)

    def convert_HipIdentifierNode(self, node: HipIdentifierNode) -> IdentifierNode:
        """Convert HIP identifier to CrossGL identifier"""
        # Handle special HIP identifiers
        special_identifiers = {
            "threadIdx.x": "gl_LocalInvocationID.x",
            "threadIdx.y": "gl_LocalInvocationID.y",
            "threadIdx.z": "gl_LocalInvocationID.z",
            "blockIdx.x": "gl_WorkGroupID.x",
            "blockIdx.y": "gl_WorkGroupID.y",
            "blockIdx.z": "gl_WorkGroupID.z",
            "blockDim.x": "gl_WorkGroupSize.x",
            "blockDim.y": "gl_WorkGroupSize.y",
            "blockDim.z": "gl_WorkGroupSize.z",
            "gridDim.x": "gl_NumWorkGroups.x",
            "gridDim.y": "gl_NumWorkGroups.y",
            "gridDim.z": "gl_NumWorkGroups.z",
        }

        name = special_identifiers.get(node.name, node.name)
        return IdentifierNode(name)

    def convert_HipLiteralNode(self, node: HipLiteralNode) -> LiteralNode:
        """Convert HIP literal to CrossGL literal"""
        return LiteralNode(node.value, node.literal_type)

    def convert_HipFunctionCallNode(
        self, node: HipFunctionCallNode
    ) -> FunctionCallNode:
        """Convert HIP function call to CrossGL function call"""
        # Convert function name
        if isinstance(node.function, HipIdentifierNode):
            func_name = node.function.name

            # Map HIP built-in functions
            mapped_name = self.function_map_builtin.get(func_name, func_name)
            function = IdentifierNode(mapped_name)
        else:
            function = self.convert(node.function)

        # Convert arguments
        args = [self.convert(arg) for arg in node.arguments]

        return FunctionCallNode(function, args)

    def convert_HipArrayAccessNode(self, node: HipArrayAccessNode) -> ArrayAccessNode:
        """Convert HIP array access to CrossGL array access"""
        array = self.convert(node.array)
        index = self.convert(node.index)
        return ArrayAccessNode(array, index)

    def convert_HipMemberAccessNode(
        self, node: HipMemberAccessNode
    ) -> MemberAccessNode:
        """Convert HIP member access to CrossGL member access"""
        object_expr = self.convert(node.object)
        return MemberAccessNode(object_expr, node.member)

    def convert_HipPointerMemberAccessNode(
        self, node: HipPointerMemberAccessNode
    ) -> MemberAccessNode:
        """Convert HIP pointer member access to CrossGL member access"""
        # Convert -> to . for CrossGL
        object_expr = self.convert(node.object)
        return MemberAccessNode(object_expr, node.member)

    def convert_HipInitializerListNode(
        self, node: HipInitializerListNode
    ) -> ArrayLiteralNode:
        """Convert HIP initializer list to CrossGL array literal"""
        elements = [self.convert(elem) for elem in node.elements]
        return ArrayLiteralNode(elements)

    def convert_HipPreprocessorNode(
        self, node: HipPreprocessorNode
    ) -> Optional[ASTNode]:
        """Convert HIP preprocessor directive"""
        # Handle common preprocessor directives
        if node.directive == "include":
            # Skip includes for now
            return None
        elif node.directive == "define":
            # Could convert to const declarations
            return None
        else:
            # Skip other preprocessor directives
            return None

    def convert_HipExternBlockNode(self, node: HipExternBlockNode) -> List[ASTNode]:
        """Convert HIP extern block"""
        # Convert body statements
        statements = []
        for stmt in node.body:
            converted = self.convert(stmt)
            if converted:
                statements.append(converted)

        return statements

    def convert_HipEnumNode(self, node: HipEnumNode) -> List[VariableNode]:
        """Convert HIP enum to CrossGL constants"""
        constants = []
        current_value = 0

        for enum_value in node.values:
            if enum_value.value:
                # Use explicit value
                value = self.convert(enum_value.value)
            else:
                # Use incremental value
                value = LiteralNode(str(current_value), "int")
                current_value += 1

            const_node = VariableNode(enum_value.name, "const int")
            const_node.value = value
            constants.append(const_node)

        return constants

    def convert_HipTypedefNode(self, node: HipTypedefNode) -> Optional[ASTNode]:
        """Convert HIP typedef"""
        # For now, skip typedefs
        # Could be converted to type aliases in the future
        return None

    def convert_HipNamespaceNode(self, node: HipNamespaceNode) -> List[ASTNode]:
        """Convert HIP namespace"""
        # Flatten namespace contents
        statements = []
        for stmt in node.body:
            converted = self.convert(stmt)
            if converted:
                if isinstance(converted, list):
                    statements.extend(converted)
                else:
                    statements.append(converted)

        return statements

    def convert_HipUsingNode(self, node: HipUsingNode) -> Optional[ASTNode]:
        """Convert HIP using declaration"""
        # Skip using declarations for now
        return None

    def convert_HipTemplateNode(self, node: HipTemplateNode) -> ASTNode:
        """Convert HIP template"""
        # For now, just convert the template body
        return self.convert(node.body)

    def convert_type(self, type_node: HipTypeNode) -> str:
        """Convert HIP type to CrossGL type"""
        if not type_node:
            return "void"

        base_type = type_node.base_type

        # Apply qualifiers
        qualifiers = type_node.qualifiers or []
        if "const" in qualifiers:
            # Handle const qualifier
            pass

        # Map type
        crossgl_type = self.type_map.get(base_type, base_type)

        # Handle template arguments (for vector types)
        if type_node.template_args:
            # This would handle templated types
            pass

        # Handle pointer/reference modifiers
        modifiers = type_node.modifiers or []
        for modifier in modifiers:
            if modifier == "*":
                # Pointers are generally not supported in CrossGL shaders
                # Could be converted to buffer references
                crossgl_type = f"buffer<{crossgl_type}>"
            elif modifier == "&":
                # References are handled as regular parameters
                pass

        return crossgl_type

    def convert_expression_to_string(self, expr: HipExpressionNode) -> str:
        """Convert expression to string representation"""
        if isinstance(expr, HipLiteralNode):
            return str(expr.value)
        elif isinstance(expr, HipIdentifierNode):
            return expr.name
        elif isinstance(expr, HipBinaryOpNode):
            left = self.convert_expression_to_string(expr.left)
            right = self.convert_expression_to_string(expr.right)
            return f"({left} {expr.operator} {right})"
        else:
            return "0"  # Default fallback

    def add_builtin_functions(self) -> List[FunctionNode]:
        """Add HIP-specific builtin functions"""
        builtins = []

        # Add syncthreads function
        sync_func = FunctionNode(
            name="__syncthreads",
            return_type="void",
            parameters=[],
            body=BlockNode([]),
            is_builtin=True,
        )
        builtins.append(sync_func)

        # Add memory fence functions
        fence_func = FunctionNode(
            name="__threadfence",
            return_type="void",
            parameters=[],
            body=BlockNode([]),
            is_builtin=True,
        )
        builtins.append(fence_func)

        return builtins


def hip_to_crossgl(hip_ast: HipProgramNode) -> ProgramNode:
    """
    Convert HIP AST to CrossGL AST

    Args:
        hip_ast: HIP program AST node

    Returns:
        CrossGL program AST node
    """
    converter = HipToCrossGLConverter()
    crossgl_ast = converter.convert(hip_ast)

    # Add builtin functions
    builtins = converter.add_builtin_functions()
    if builtins:
        crossgl_ast.statements.extend(builtins)

    return crossgl_ast

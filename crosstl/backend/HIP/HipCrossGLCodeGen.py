"""
HIP to CrossGL Code Generator

This module provides code generation functionality to convert HIP (HIP Is a Portable GPU Runtime)
AST nodes to CrossGL intermediate representation.
"""

from typing import Any
from .HipAst import (
    ASTNode,
    ShaderNode,
    FunctionNode,
    KernelNode,
    StructNode,
    VariableNode,
    AssignmentNode,
    BinaryOpNode,
    UnaryOpNode,
    FunctionCallNode,
    AtomicOperationNode,
    SyncNode,
    MemberAccessNode,
    ArrayAccessNode,
    IfNode,
    ForNode,
    WhileNode,
    ReturnNode,
    BreakNode,
    ContinueNode,
    PreprocessorNode,
    HipBuiltinNode,
)
from ...translator.ast import (
    ShaderNode as CrossGLShaderNode,
    FunctionNode as CrossGLFunctionNode,
    VariableNode as CrossGLVariableNode,
    StructNode as CrossGLStructNode,
    BinaryOpNode as CrossGLBinaryOpNode,
    UnaryOpNode as CrossGLUnaryOpNode,
    FunctionCallNode as CrossGLFunctionCallNode,
    AssignmentNode as CrossGLAssignmentNode,
    ArrayAccessNode as CrossGLArrayAccessNode,
    MemberAccessNode as CrossGLMemberAccessNode,
)


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
            # Atomic functions
            "atomicAdd": "atomicAdd",
            "atomicSub": "atomicSub",
            "atomicMax": "atomicMax",
            "atomicMin": "atomicMin",
            "atomicExch": "atomicExchange",
            # Sync functions
            "__syncthreads": "barrier",
            "__threadfence": "memoryBarrier",
        }

        # Built-in variable mappings
        self.builtin_map = {
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

    def convert(self, node: Any) -> Any:
        """Convert HIP AST node to CrossGL AST node"""
        if node is None:
            return None

        # Handle different node types from HipParser
        if hasattr(node, "statements"):  # HipProgramNode
            return self.convert_program(node)
        elif isinstance(node, FunctionNode):
            return self.convert_function(node)
        elif isinstance(node, KernelNode):
            return self.convert_kernel(node)
        elif isinstance(node, StructNode):
            return self.convert_struct(node)
        elif isinstance(node, VariableNode):
            return self.convert_variable(node)
        elif isinstance(node, BinaryOpNode):
            return self.convert_binary_op(node)
        elif isinstance(node, UnaryOpNode):
            return self.convert_unary_op(node)
        elif isinstance(node, FunctionCallNode):
            return self.convert_function_call(node)
        elif isinstance(node, ArrayAccessNode):
            return self.convert_array_access(node)
        elif isinstance(node, MemberAccessNode):
            return self.convert_member_access(node)
        elif isinstance(node, AssignmentNode):
            return self.convert_assignment(node)
        elif isinstance(node, HipBuiltinNode):
            return self.convert_builtin(node)
        elif isinstance(node, PreprocessorNode):
            return self.convert_preprocessor(node)
        elif isinstance(node, str):
            return node  # Return string directly for identifiers
        elif isinstance(node, (int, float)):
            return str(node)  # Return string representation for literals
        elif isinstance(node, list):
            return [self.convert(item) for item in node]
        else:
            # Return as-is for unknown types
            return node

    def convert_program(self, node: Any) -> CrossGLShaderNode:
        """Convert HIP program to CrossGL shader"""
        functions = []
        structs = []
        variables = []
        cbuffers = []

        for stmt in node.statements:
            converted = self.convert(stmt)
            if converted:
                if isinstance(converted, CrossGLFunctionNode):
                    functions.append(converted)
                elif isinstance(converted, CrossGLStructNode):
                    structs.append(converted)
                elif isinstance(converted, CrossGLVariableNode):
                    variables.append(converted)

        # Create shader with required arguments
        shader = CrossGLShaderNode(structs, functions, variables, cbuffers)

        return shader

    def convert_function(self, node: FunctionNode) -> CrossGLFunctionNode:
        """Convert HIP function to CrossGL function"""
        self.current_function = node.name

        # Convert return type
        return_type = (
            self.convert_type(node.return_type) if node.return_type else "void"
        )

        # Convert parameters
        params = []
        if hasattr(node, "params") and node.params:
            for param in node.params:
                if isinstance(param, dict):
                    param_type = self.convert_type(param.get("type", "int"))
                    param_name = param.get("name", "param")
                    params.append(CrossGLVariableNode(param_name, param_type))
                else:
                    params.append(self.convert(param))

        # Convert body
        body = None
        if hasattr(node, "body") and node.body:
            body = self.convert(node.body)

        # Check if this is a kernel (has __global__ qualifier)
        is_kernel = hasattr(node, "qualifiers") and "__global__" in getattr(
            node, "qualifiers", []
        )

        func = CrossGLFunctionNode(node.name, return_type, params, body)

        if is_kernel:
            func.shader_type = "compute"

        self.current_function = None
        return func

    def convert_kernel(self, node: KernelNode) -> CrossGLFunctionNode:
        """Convert HIP kernel to CrossGL compute function"""
        return self.convert_function(node)

    def convert_struct(self, node: StructNode) -> CrossGLStructNode:
        """Convert HIP struct to CrossGL struct"""
        members = []

        if hasattr(node, "members") and node.members:
            for member in node.members:
                converted = self.convert(member)
                if converted:
                    members.append(converted)

        return CrossGLStructNode(node.name, members)

    def convert_variable(self, node: VariableNode) -> CrossGLVariableNode:
        """Convert HIP variable to CrossGL variable"""
        var_type = (
            self.convert_type(node.vtype)
            if hasattr(node, "vtype")
            else self.convert_type(getattr(node, "type", "int"))
        )
        var_name = node.name

        var = CrossGLVariableNode(var_name, var_type)

        if hasattr(node, "value") and node.value:
            var.value = self.convert(node.value)

        return var

    def convert_binary_op(self, node: BinaryOpNode) -> CrossGLBinaryOpNode:
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

        operator = op_map.get(node.op, node.op)
        return CrossGLBinaryOpNode(left, operator, right)

    def convert_unary_op(self, node: UnaryOpNode) -> CrossGLUnaryOpNode:
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

        operator = op_map.get(node.op, node.op)
        return CrossGLUnaryOpNode(operator, operand)

    def convert_function_call(self, node: FunctionCallNode) -> CrossGLFunctionCallNode:
        """Convert HIP function call to CrossGL function call"""
        # Convert function name
        if hasattr(node, "name"):
            func_name = node.name
        else:
            func_name = str(node.function) if hasattr(node, "function") else "unknown"

        # Map HIP built-in functions
        mapped_name = self.function_map_builtin.get(func_name, func_name)
        function = mapped_name  # Use string directly for function name

        # Convert arguments
        args = []
        if hasattr(node, "args") and node.args:
            args = [self.convert(arg) for arg in node.args]
        elif hasattr(node, "arguments") and node.arguments:
            args = [self.convert(arg) for arg in node.arguments]

        return CrossGLFunctionCallNode(function, args)

    def convert_array_access(self, node: ArrayAccessNode) -> CrossGLArrayAccessNode:
        """Convert HIP array access to CrossGL array access"""
        array = self.convert(node.array)
        index = self.convert(node.index)
        return CrossGLArrayAccessNode(array, index)

    def convert_member_access(self, node: MemberAccessNode) -> CrossGLMemberAccessNode:
        """Convert HIP member access to CrossGL member access"""
        object_expr = self.convert(node.object)
        return CrossGLMemberAccessNode(object_expr, node.member)

    def convert_assignment(self, node: AssignmentNode) -> CrossGLAssignmentNode:
        """Convert HIP assignment to CrossGL assignment"""
        left = self.convert(node.left)
        right = self.convert(node.right)
        operator = getattr(node, "operator", "=")
        return CrossGLAssignmentNode(left, right, operator)

    def convert_builtin(self, node: HipBuiltinNode) -> str:
        """Convert HIP built-in variable to CrossGL built-in"""
        if node.component:
            builtin_name = f"{node.builtin_name}.{node.component}"
        else:
            builtin_name = node.builtin_name

        mapped_name = self.builtin_map.get(builtin_name, builtin_name)
        return CrossGLIdentifierNode(mapped_name)

    def convert_preprocessor(self, node: PreprocessorNode) -> None:
        """Convert HIP preprocessor directive (mostly ignored)"""
        # Skip preprocessor directives in CrossGL
        return None

    def convert_type(self, type_info: Any) -> str:
        """Convert HIP type to CrossGL type"""
        if type_info is None:
            return "void"

        if isinstance(type_info, str):
            return self.type_map.get(type_info, type_info)

        # Handle complex type objects
        if hasattr(type_info, "name"):
            type_name = type_info.name
        elif hasattr(type_info, "base_type"):
            type_name = type_info.base_type
        else:
            type_name = str(type_info)

        return self.type_map.get(type_name, type_name)


def hip_to_crossgl(hip_ast: Any) -> CrossGLShaderNode:
    """Convert HIP AST to CrossGL AST"""
    converter = HipToCrossGLConverter()
    return converter.convert(hip_ast)

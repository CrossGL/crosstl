"""
CrossGL to HIP Code Generator

This module provides code generation functionality to convert CrossGL AST to HIP source code.
HIP (Heterogeneous-Compute Interface for Portability) is AMD's CUDA-compatible runtime API
for GPU programming.
"""

from ..ast import (
    ASTNode,
    CbufferNode,
    FunctionNode,
    ShaderNode,
    StructNode,
    VariableNode,
)


class HipCodeGen:
    """Generates HIP code from CrossGL AST"""

    def __init__(self):
        self.indent_level = 0
        self.code_lines = []
        self.current_function = None
        self.variable_counter = 0

        # CrossGL to HIP type mapping
        self.type_map = {
            # Basic types
            "int": "int",
            "float": "float",
            "double": "double",
            "bool": "bool",
            "void": "void",
            "uint": "unsigned int",
            # Vector types
            "vec2": "float2",
            "vec3": "float3",
            "vec4": "float4",
            "ivec2": "int2",
            "ivec3": "int3",
            "ivec4": "int4",
            "uvec2": "uint2",
            "uvec3": "uint3",
            "uvec4": "uint4",
            "dvec2": "double2",
            "dvec3": "double3",
            "dvec4": "double4",
            # Matrix types
            "mat2": "float2x2",
            "mat3": "float3x3",
            "mat4": "float4x4",
            "dmat2": "double2x2",
            "dmat3": "double3x3",
            "dmat4": "double4x4",
            # Texture types
            "sampler2D": "texture<float4, 2>",
            "sampler3D": "texture<float4, 3>",
            "samplerCube": "textureCube<float4>",
            "image2D": "surface<void, 2>",
            "buffer": "hipDeviceptr_t",
        }

        # CrossGL to HIP function mapping
        self.function_map = {
            # Math functions
            "sin": "sinf",
            "cos": "cosf",
            "tan": "tanf",
            "asin": "asinf",
            "acos": "acosf",
            "atan": "atanf",
            "atan2": "atan2f",
            "sinh": "sinhf",
            "cosh": "coshf",
            "tanh": "tanhf",
            "exp": "expf",
            "exp2": "exp2f",
            "log": "logf",
            "log2": "log2f",
            "sqrt": "sqrtf",
            "inversesqrt": "rsqrtf",
            "pow": "powf",
            "abs": "fabsf",
            "floor": "floorf",
            "ceil": "ceilf",
            "round": "roundf",
            "trunc": "truncf",
            "fract": "fracf",
            "mod": "fmodf",
            "min": "fminf",
            "max": "fmaxf",
            "clamp": "fmaxf(fminf",  # Special handling needed
            "mix": "lerp",
            "step": "step",
            "smoothstep": "smoothstep",
            # Vector functions
            "length": "length",
            "distance": "distance",
            "dot": "dot",
            "cross": "cross",
            "normalize": "normalize",
            "reflect": "reflect",
            "refract": "refract",
            # Geometric functions
            "faceforward": "faceforward",
            # Vector constructors
            "vec2": "make_float2",
            "vec3": "make_float3",
            "vec4": "make_float4",
            "ivec2": "make_int2",
            "ivec3": "make_int3",
            "ivec4": "make_int4",
            "uvec2": "make_uint2",
            "uvec3": "make_uint3",
            "uvec4": "make_uint4",
            # Texture functions
            "texture": "tex2D",
            "textureLod": "tex2DLod",
            "textureGrad": "tex2DGrad",
        }

        # Built-in variable mappings
        self.builtin_map = {
            "gl_LocalInvocationID.x": "threadIdx.x",
            "gl_LocalInvocationID.y": "threadIdx.y",
            "gl_LocalInvocationID.z": "threadIdx.z",
            "gl_WorkGroupID.x": "blockIdx.x",
            "gl_WorkGroupID.y": "blockIdx.y",
            "gl_WorkGroupID.z": "blockIdx.z",
            "gl_WorkGroupSize.x": "blockDim.x",
            "gl_WorkGroupSize.y": "blockDim.y",
            "gl_WorkGroupSize.z": "blockDim.z",
            "gl_NumWorkGroups.x": "gridDim.x",
            "gl_NumWorkGroups.y": "gridDim.y",
            "gl_NumWorkGroups.z": "gridDim.z",
            "gl_GlobalInvocationID.x": "(blockIdx.x * blockDim.x + threadIdx.x)",
            "gl_GlobalInvocationID.y": "(blockIdx.y * blockDim.y + threadIdx.y)",
            "gl_GlobalInvocationID.z": "(blockIdx.z * blockDim.z + threadIdx.z)",
        }

    def generate(self, node: ASTNode) -> str:
        """Generate HIP code from CrossGL AST"""
        self.code_lines = []
        self.indent_level = 0

        # Add necessary includes
        self.add_includes()

        # Generate code
        self.visit(node)

        return "\n".join(self.code_lines)

    def add_includes(self):
        """Add necessary HIP includes"""
        self.code_lines.extend(
            [
                "#include <hip/hip_runtime.h>",
                "#include <hip/hip_runtime_api.h>",
                "#include <hip/math_functions.h>",
                "#include <hip/device_functions.h>",
                "",
            ]
        )

    def indent(self) -> str:
        """Return current indentation string"""
        return "    " * self.indent_level

    def add_line(self, line: str = ""):
        """Add a line with current indentation"""
        if line:
            self.code_lines.append(self.indent() + line)
        else:
            self.code_lines.append("")

    def visit(self, node: ASTNode) -> str:
        """Visit a node and generate code"""
        method_name = f"visit_{type(node).__name__}"
        visitor = getattr(self, method_name, self.generic_visit)
        return visitor(node)

    def generic_visit(self, node: ASTNode) -> str:
        """Generic visitor for unsupported nodes"""
        raise NotImplementedError(
            f"Code generation not implemented for {type(node).__name__}"
        )

    def visit_ShaderNode(self, node: ShaderNode) -> str:
        """Visit shader node"""
        # Generate structs
        structs = getattr(node, "structs", [])
        for struct in structs:
            self.visit(struct)

        # Generate global variables
        global_vars = getattr(node, "global_variables", [])
        for var in global_vars:
            self.visit(var)

        # Generate cbuffers (legacy compatibility)
        cbuffers = getattr(node, "cbuffers", [])
        for cbuffer in cbuffers:
            self.visit(cbuffer)

        # Generate functions
        functions = getattr(node, "functions", [])
        for func in functions:
            self.visit(func)

        # Handle shader stages (new AST structure)
        if hasattr(node, "stages") and node.stages:
            for stage_type, stage in node.stages.items():
                if hasattr(stage, "entry_point"):
                    # Set the stage type context for proper qualifier handling
                    stage_name = (
                        str(stage_type).split(".")[-1].lower()
                        if hasattr(stage_type, "name")
                        else str(stage_type).lower()
                    )

                    # Temporarily set qualifier for compute stages
                    if stage_name == "compute" or "compute" in stage_name:
                        # Set the function qualifier to compute for proper __global__ generation
                        if hasattr(stage.entry_point, "qualifiers"):
                            if "compute" not in stage.entry_point.qualifiers:
                                stage.entry_point.qualifiers.append("compute")
                        else:
                            stage.entry_point.qualifiers = ["compute"]

                    self.visit(stage.entry_point)
                if hasattr(stage, "local_functions"):
                    for func in stage.local_functions:
                        self.visit(func)

        return ""

    def visit_FunctionNode(self, node: FunctionNode) -> str:
        """Visit function node"""
        self.current_function = node.name

        # Determine function qualifiers - handle both old and new AST
        qualifiers = []
        if hasattr(node, "qualifiers") and node.qualifiers:
            for qualifier in node.qualifiers:
                if "kernel" in qualifier or "compute" in qualifier:
                    qualifiers.append("__global__")
                elif "device" in qualifier:
                    qualifiers.append("__device__")
                else:
                    qualifiers.append("__device__")
        elif hasattr(node, "qualifier") and node.qualifier:
            if "kernel" in node.qualifier or "compute" in node.qualifier:
                qualifiers.append("__global__")
            elif "device" in node.qualifier:
                qualifiers.append("__device__")
            else:
                qualifiers.append("__device__")
        else:
            qualifiers.append("__device__")  # Default for HIP

        # Function signature - handle both old and new AST
        if hasattr(node, "return_type"):
            if hasattr(node.return_type, "name"):
                return_type = self.map_type(node.return_type.name)
            else:
                return_type = self.map_type(str(node.return_type))
        else:
            return_type = "void"

        # Handle parameters - support both old and new AST
        param_list = getattr(node, "parameters", getattr(node, "params", []))
        params = ", ".join(self.visit_parameter(param) for param in param_list)

        qualifier_str = " ".join(qualifiers)
        signature = f"{qualifier_str} {return_type} {node.name}({params})"

        self.add_line(signature)

        # Handle function body - support both old and new AST
        body = getattr(node, "body", [])
        if body:
            self.add_line("{")
            self.indent_level += 1

            if hasattr(body, "statements"):
                # New AST BlockNode structure
                for stmt in body.statements:
                    self.visit(stmt)
            elif isinstance(body, list):
                # Old AST structure
                for stmt in body:
                    self.visit(stmt)
            else:
                # Single statement
                self.visit(body)

            self.indent_level -= 1
            self.add_line("}")
        else:
            self.add_line(";")

        self.add_line()
        self.current_function = None
        return ""

    def visit_parameter(self, param) -> str:
        """Visit parameter node"""
        if isinstance(param, dict):
            param_type = self.map_type(param.get("type", "int"))
            param_name = param.get("name", "param")
        else:
            # Handle both old and new AST parameter structures
            if hasattr(param, "param_type"):
                if hasattr(param.param_type, "name"):
                    param_type = self.map_type(param.param_type.name)
                else:
                    param_type = self.map_type(str(param.param_type))
            elif hasattr(param, "vtype"):
                param_type = self.map_type(param.vtype)
            else:
                param_type = "int"

            param_name = getattr(param, "name", "param")

        return f"{param_type} {param_name}"

    def visit_StructNode(self, node: StructNode) -> str:
        """Visit struct node"""
        self.add_line(f"struct {node.name}")
        self.add_line("{")
        self.indent_level += 1

        members = getattr(node, "members", [])
        for member in members:
            if hasattr(member, "member_type"):
                # New AST structure - pass TypeNode directly to map_type
                member_type = self.map_type(member.member_type)
            elif hasattr(member, "vtype"):
                # Old AST structure
                member_type = self.map_type(member.vtype)
            elif hasattr(member, "var_type"):
                # Alternative structure
                member_type = self.map_type(str(member.var_type))
            else:
                member_type = "float"

            self.add_line(f"{member_type} {member.name};")

        self.indent_level -= 1
        self.add_line("};")
        self.add_line()
        return ""

    def visit_VariableNode(self, node: VariableNode) -> str:
        """Visit variable node"""
        # Handle both old and new AST structures
        if hasattr(node, "var_type"):
            if hasattr(node.var_type, "name"):
                var_type = self.map_type(node.var_type.name)
            else:
                var_type = self.map_type(str(node.var_type))
        elif hasattr(node, "vtype"):
            var_type = self.map_type(node.vtype)
        else:
            var_type = "int"

        # Handle initial value
        if hasattr(node, "initial_value") and node.initial_value:
            value = self.visit(node.initial_value)
            self.add_line(f"{var_type} {node.name} = {value};")
        elif hasattr(node, "value") and node.value:
            value = self.visit(node.value)
            self.add_line(f"{var_type} {node.name} = {value};")
        else:
            self.add_line(f"{var_type} {node.name};")

        return ""

    def visit_CbufferNode(self, node: CbufferNode) -> str:
        """Visit cbuffer node - convert to struct"""
        self.add_line(f"struct {node.name}")
        self.add_line("{")
        self.indent_level += 1

        for member in node.members:
            if isinstance(member, VariableNode):
                member_type = self.map_type(
                    getattr(member, "vtype", getattr(member, "var_type", "int"))
                )
                self.add_line(f"{member_type} {member.name};")

        self.indent_level -= 1
        self.add_line("};")
        self.add_line()
        return ""

    def visit_list(self, node_list) -> str:
        """Visit list of nodes"""
        for node in node_list:
            self.visit(node)
        return ""

    def visit_IfNode(self, node) -> str:
        """Visit if node"""
        condition = self.visit(node.if_condition)
        self.add_line(f"if ({condition})")
        self.add_line("{")
        self.indent_level += 1
        if isinstance(node.if_body, list):
            for stmt in node.if_body:
                self.visit(stmt)
        else:
            self.visit(node.if_body)
        self.indent_level -= 1
        self.add_line("}")

        if hasattr(node, "else_body") and node.else_body:
            self.add_line("else")
            self.add_line("{")
            self.indent_level += 1
            if isinstance(node.else_body, list):
                for stmt in node.else_body:
                    self.visit(stmt)
            else:
                self.visit(node.else_body)
            self.indent_level -= 1
            self.add_line("}")

        return ""

    def visit_ForNode(self, node) -> str:
        """Visit for loop node"""
        init = self.visit(node.init) if node.init else ""
        condition = self.visit(node.condition) if node.condition else ""
        update = self.visit(node.update) if node.update else ""

        self.add_line(f"for ({init}; {condition}; {update})")
        self.add_line("{")
        self.indent_level += 1
        if isinstance(node.body, list):
            for stmt in node.body:
                self.visit(stmt)
        else:
            self.visit(node.body)
        self.indent_level -= 1
        self.add_line("}")

        return ""

    def visit_ReturnNode(self, node) -> str:
        """Visit return node"""
        if node.value:
            value = self.visit(node.value)
            self.add_line(f"return {value};")
        else:
            self.add_line("return;")
        return ""

    def visit_AssignmentNode(self, node) -> str:
        """Visit assignment node"""
        left = self.visit(node.left)
        right = self.visit(node.right)
        return f"{left} = {right}"

    def visit_BinaryOpNode(self, node) -> str:
        """Visit binary operation node"""
        left = self.visit(node.left)
        right = self.visit(node.right)

        # Handle special operators
        if node.op == "and":
            return f"({left} && {right})"
        elif node.op == "or":
            return f"({left} || {right})"
        else:
            return f"({left} {node.op} {right})"

    def visit_UnaryOpNode(self, node) -> str:
        """Visit unary operation node"""
        operand = self.visit(node.operand)

        if node.op == "not":
            return f"!{operand}"
        elif node.op in ["++", "--"]:
            if hasattr(node, "postfix") and node.postfix:
                return f"{operand}{node.op}"
            else:
                return f"{node.op}{operand}"
        else:
            return f"{node.op}{operand}"

    def visit_FunctionCallNode(self, node) -> str:
        """Visit function call node"""
        func_name = node.name
        args = [self.visit(arg) for arg in node.args]

        # Map function name
        mapped_name = self.function_map.get(func_name, func_name)

        # Handle special functions
        if func_name == "clamp":
            if len(args) == 3:
                return f"fmaxf({args[1]}, fminf({args[2]}, {args[0]}))"
        elif func_name in ["texture", "tex2D"]:
            # Handle texture sampling
            if len(args) >= 2:
                return f"tex2D({args[0]}, {args[1]})"
        elif func_name == "barrier":
            return "__syncthreads()"
        elif func_name == "memoryBarrier":
            return "__threadfence()"

        args_str = ", ".join(args)
        return f"{mapped_name}({args_str})"

    def visit_str(self, node) -> str:
        """Visit string node"""
        return str(node)

    def visit_int(self, node) -> str:
        """Visit integer node"""
        return str(node)

    def visit_float(self, node) -> str:
        """Visit float node"""
        return str(node)

    def visit_ArrayAccessNode(self, node) -> str:
        """Visit array access node"""
        array = self.visit(node.array)
        index = self.visit(node.index)
        return f"{array}[{index}]"

    def visit_MemberAccessNode(self, node) -> str:
        """Visit member access node"""
        object_expr = self.visit(node.object)

        # Handle vector swizzling
        if node.member in ["x", "y", "z", "w", "r", "g", "b", "a"]:
            return f"{object_expr}.{node.member}"
        elif len(node.member) > 1 and all(c in "xyzw" for c in node.member):
            # Multi-component swizzle - might need special handling
            return f"{object_expr}.{node.member}"
        else:
            return f"{object_expr}.{node.member}"

    def visit_TernaryOpNode(self, node) -> str:
        """Visit ternary operation node"""
        condition = self.visit(node.condition)
        true_expr = self.visit(node.true_expr)
        false_expr = self.visit(node.false_expr)
        return f"({condition} ? {true_expr} : {false_expr})"

    def visit_LiteralNode(self, node) -> str:
        """Visit literal node"""
        if hasattr(node, "value"):
            if isinstance(node.value, str):
                return f'"{node.value}"'
            elif isinstance(node.value, bool):
                return "true" if node.value else "false"
            else:
                return str(node.value)
        return str(node)

    def visit_IdentifierNode(self, node) -> str:
        """Visit identifier node"""
        name = getattr(node, "name", str(node))
        # Handle built-in variables mapping
        return self.builtin_map.get(name, name)

    def visit_ExpressionStatementNode(self, node) -> str:
        """Visit expression statement node"""
        expr = self.visit(node.expression)
        self.add_line(f"{expr};")
        return ""

    def visit_BlockNode(self, node) -> str:
        """Visit block node"""
        if hasattr(node, "statements"):
            for stmt in node.statements:
                self.visit(stmt)
        return ""

    def visit_BreakNode(self, node) -> str:
        """Visit break statement node"""
        self.add_line("break;")
        return ""

    def visit_ContinueNode(self, node) -> str:
        """Visit continue statement node"""
        self.add_line("continue;")
        return ""

    def visit_EnumNode(self, node) -> str:
        """Visit enum declaration node"""
        self.add_line(f"enum {node.name}")
        self.add_line("{")
        self.indent_level += 1

        if hasattr(node, "variants") and node.variants:
            for i, variant in enumerate(node.variants):
                if hasattr(variant, "value") and variant.value:
                    value = self.visit(variant.value)
                    if i == len(node.variants) - 1:
                        self.add_line(f"{variant.name} = {value}")
                    else:
                        self.add_line(f"{variant.name} = {value},")
                else:
                    if i == len(node.variants) - 1:
                        self.add_line(f"{variant.name}")
                    else:
                        self.add_line(f"{variant.name},")

        self.indent_level -= 1
        self.add_line("};")
        self.add_line()
        return ""

    def convert_type_node_to_string(self, type_node) -> str:
        """Convert new AST TypeNode to string representation."""
        if hasattr(type_node, "name"):
            # PrimitiveType
            return type_node.name
        elif hasattr(type_node, "element_type") and hasattr(type_node, "size"):
            # VectorType or ArrayType
            if hasattr(type_node, "rows"):
                # MatrixType
                element_type = self.convert_type_node_to_string(type_node.element_type)
                return f"float{type_node.rows}x{type_node.cols}"
            elif str(type(type_node)).find("ArrayType") != -1:
                # ArrayType
                element_type = self.convert_type_node_to_string(type_node.element_type)
                if type_node.size is not None:
                    return f"{element_type}[{type_node.size}]"
                else:
                    return f"{element_type}[]"
            else:
                # VectorType
                element_type = self.convert_type_node_to_string(type_node.element_type)
                size = type_node.size
                if element_type == "float":
                    return f"float{size}"
                elif element_type == "int":
                    return f"int{size}"
                else:
                    return f"{element_type}{size}"
        else:
            return str(type_node)

    def map_type(self, type_name) -> str:
        """Map CrossGL type to HIP type"""
        # Handle TypeNode objects
        if hasattr(type_name, "name") or hasattr(type_name, "element_type"):
            type_str = self.convert_type_node_to_string(type_name)
        else:
            type_str = str(type_name)

        # Handle array types
        if "[" in type_str and "]" in type_str:
            base_type = type_str.split("[")[0]
            array_part = type_str[type_str.find("[") :]
            mapped_base = self.type_map.get(base_type, base_type)
            return f"{mapped_base}{array_part}"

        return self.type_map.get(type_str, type_str)

    def generate_kernel_wrapper(self, kernel_node: FunctionNode) -> str:
        """Generate host-side kernel launch wrapper"""
        wrapper_lines = []

        # Generate wrapper function
        wrapper_name = f"launch_{kernel_node.name}"
        params = []
        args = []

        for param in kernel_node.parameters:
            param_type = self.map_type(param.param_type)
            params.append(f"{param_type} {param.name}")
            args.append(param.name)

        # Add grid and block size parameters
        params.extend(["dim3 gridSize", "dim3 blockSize", "hipStream_t stream = 0"])

        wrapper_lines.extend(
            [
                f"void {wrapper_name}({', '.join(params)})",
                "{",
                f"    hipLaunchKernelGGL({kernel_node.name}, gridSize, blockSize, 0, stream, {', '.join(args)});",
                "}",
            ]
        )

        return "\n".join(wrapper_lines)


def generate_hip_code(ast: ShaderNode) -> str:
    """
    Generate HIP code from CrossGL AST

    Args:
        ast: CrossGL shader AST

    Returns:
        Generated HIP source code
    """
    generator = HipCodeGen()
    return generator.generate(ast)

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
            "vec2<f32>": "float2",
            "vec3<f32>": "float3",
            "vec4<f32>": "float4",
            "ivec2": "int2",
            "ivec3": "int3",
            "ivec4": "int4",
            "vec2<i32>": "int2",
            "vec3<i32>": "int3",
            "vec4<i32>": "int4",
            "uvec2": "uint2",
            "uvec3": "uint3",
            "uvec4": "uint4",
            "vec2<u32>": "uint2",
            "vec3<u32>": "uint3",
            "vec4<u32>": "uint4",
            "dvec2": "double2",
            "dvec3": "double3",
            "dvec4": "double4",
            "vec2<f64>": "double2",
            "vec3<f64>": "double3",
            "vec4<f64>": "double4",
            "bvec2": "uchar2",
            "bvec3": "uchar3",
            "bvec4": "uchar4",
            "vec2<bool>": "uchar2",
            "vec3<bool>": "uchar3",
            "vec4<bool>": "uchar4",
            "bool2": "uchar2",
            "bool3": "uchar3",
            "bool4": "uchar4",
            # Matrix types
            "mat2": "float2x2",
            "mat3": "float3x3",
            "mat4": "float4x4",
            "mat2x2": "float2x2",
            "mat2x3": "float2x3",
            "mat2x4": "float2x4",
            "mat3x2": "float3x2",
            "mat3x3": "float3x3",
            "mat3x4": "float3x4",
            "mat4x2": "float4x2",
            "mat4x3": "float4x3",
            "mat4x4": "float4x4",
            "dmat2": "double2x2",
            "dmat3": "double3x3",
            "dmat4": "double4x4",
            "dmat2x2": "double2x2",
            "dmat2x3": "double2x3",
            "dmat2x4": "double2x4",
            "dmat3x2": "double3x2",
            "dmat3x3": "double3x3",
            "dmat3x4": "double3x4",
            "dmat4x2": "double4x2",
            "dmat4x3": "double4x3",
            "dmat4x4": "double4x4",
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
            "vec2<f32>": "make_float2",
            "vec3<f32>": "make_float3",
            "vec4<f32>": "make_float4",
            "ivec2": "make_int2",
            "ivec3": "make_int3",
            "ivec4": "make_int4",
            "vec2<i32>": "make_int2",
            "vec3<i32>": "make_int3",
            "vec4<i32>": "make_int4",
            "uvec2": "make_uint2",
            "uvec3": "make_uint3",
            "uvec4": "make_uint4",
            "vec2<u32>": "make_uint2",
            "vec3<u32>": "make_uint3",
            "vec4<u32>": "make_uint4",
            "dvec2": "make_double2",
            "dvec3": "make_double3",
            "dvec4": "make_double4",
            "vec2<f64>": "make_double2",
            "vec3<f64>": "make_double3",
            "vec4<f64>": "make_double4",
            "bvec2": "make_uchar2",
            "bvec3": "make_uchar3",
            "bvec4": "make_uchar4",
            "vec2<bool>": "make_uchar2",
            "vec3<bool>": "make_uchar3",
            "vec4<bool>": "make_uchar4",
            "bool2": "make_uchar2",
            "bool3": "make_uchar3",
            "bool4": "make_uchar4",
            # Matrix constructors
            "mat2": "float2x2",
            "mat3": "float3x3",
            "mat4": "float4x4",
            "mat2x2": "float2x2",
            "mat2x3": "float2x3",
            "mat2x4": "float2x4",
            "mat3x2": "float3x2",
            "mat3x3": "float3x3",
            "mat3x4": "float3x4",
            "mat4x2": "float4x2",
            "mat4x3": "float4x3",
            "mat4x4": "float4x4",
            "dmat2": "double2x2",
            "dmat3": "double3x3",
            "dmat4": "double4x4",
            "dmat2x2": "double2x2",
            "dmat2x3": "double2x3",
            "dmat2x4": "double2x4",
            "dmat3x2": "double3x2",
            "dmat3x3": "double3x3",
            "dmat3x4": "double3x4",
            "dmat4x2": "double4x2",
            "dmat4x3": "double4x3",
            "dmat4x4": "double4x4",
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
        self.code_lines = []
        self.indent_level = 0

        self.add_includes()
        self.visit(node)

        return "\n".join(self.code_lines)

    def add_includes(self):
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
        return "    " * self.indent_level

    def add_line(self, line: str = ""):
        if line:
            self.code_lines.append(self.indent() + line)
        else:
            self.code_lines.append("")

    def visit(self, node: ASTNode) -> str:
        method_name = f"visit_{type(node).__name__}"
        visitor = getattr(self, method_name, self.generic_visit)
        return visitor(node)

    def generic_visit(self, node: ASTNode) -> str:
        raise NotImplementedError(
            f"Code generation not implemented for {type(node).__name__}"
        )

    def visit_ShaderNode(self, node: ShaderNode) -> str:
        structs = getattr(node, "structs", [])
        for struct in structs:
            self.visit(struct)

        global_vars = getattr(node, "global_variables", [])
        for var in global_vars:
            self.visit(var)

        cbuffers = getattr(node, "cbuffers", [])
        for cbuffer in cbuffers:
            self.visit(cbuffer)

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
        self.current_function = node.name

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
            qualifiers.append("__device__")

        if hasattr(node, "return_type"):
            return_type = self.map_type(node.return_type)
        else:
            return_type = "void"

        param_list = getattr(node, "parameters", getattr(node, "params", []))
        params = ", ".join(self.visit_parameter(param) for param in param_list)

        qualifier_str = " ".join(qualifiers)
        signature = f"{qualifier_str} {return_type} {node.name}({params})"

        self.add_line(signature)

        body = getattr(node, "body", [])
        if body:
            self.add_line("{")
            self.indent_level += 1
            self.emit_body(body)

            self.indent_level -= 1
            self.add_line("}")
        else:
            self.add_line(";")

        self.add_line()
        self.current_function = None
        return ""

    def visit_parameter(self, param) -> str:
        if isinstance(param, dict):
            param_type = param.get("type", "int")
            param_name = param.get("name", "param")
        else:
            if hasattr(param, "param_type"):
                param_type = param.param_type
            elif hasattr(param, "vtype"):
                param_type = param.vtype
            else:
                param_type = "int"

            param_name = getattr(param, "name", "param")

        return self.format_typed_declarator(param_type, param_name)

    def visit_StructNode(self, node: StructNode) -> str:
        self.add_line(f"struct {node.name}")
        self.add_line("{")
        self.indent_level += 1

        members = getattr(node, "members", [])
        for member in members:
            if hasattr(member, "member_type"):
                member_type = member.member_type
            elif hasattr(member, "vtype"):
                member_type = member.vtype
            elif hasattr(member, "var_type"):
                member_type = member.var_type
            else:
                member_type = "float"

            self.add_line(f"{self.format_typed_declarator(member_type, member.name)};")

        self.indent_level -= 1
        self.add_line("};")
        self.add_line()
        return ""

    def visit_VariableNode(self, node: VariableNode) -> str:
        self.add_line(f"{self.format_variable_declaration(node)};")
        return ""

    def format_variable_declaration(self, node: VariableNode) -> str:
        if hasattr(node, "var_type"):
            var_type = node.var_type
        elif hasattr(node, "vtype"):
            var_type = node.vtype
        else:
            var_type = "int"

        declaration = self.format_typed_declarator(var_type, node.name)
        initial_value = getattr(node, "initial_value", getattr(node, "value", None))
        if initial_value is not None:
            declaration += f" = {self.visit(initial_value)}"

        return declaration

    def visit_CbufferNode(self, node: CbufferNode) -> str:
        self.add_line(f"struct {node.name}")
        self.add_line("{")
        self.indent_level += 1

        for member in node.members:
            if isinstance(member, VariableNode):
                member_type = getattr(
                    member, "vtype", getattr(member, "var_type", "int")
                )
                declaration = self.format_typed_declarator(member_type, member.name)
                self.add_line(f"{declaration};")

        self.indent_level -= 1
        self.add_line("};")
        self.add_line()
        return ""

    def visit_list(self, node_list) -> str:
        for node in node_list:
            self.emit_statement(node)
        return ""

    def emit_statement(self, node):
        if node is None:
            return

        result = self.visit(node)
        if isinstance(result, str) and result.strip():
            self.add_line(f"{result};")

    def emit_body(self, body):
        if isinstance(body, list):
            for stmt in body:
                self.emit_statement(stmt)
        elif hasattr(body, "statements"):
            for stmt in body.statements:
                self.emit_statement(stmt)
        else:
            self.emit_statement(body)

    def visit_IfNode(self, node) -> str:
        condition = self.visit(node.if_condition)
        self.add_line(f"if ({condition})")
        self.add_line("{")
        self.indent_level += 1
        self.emit_body(node.if_body)
        self.indent_level -= 1
        self.add_line("}")

        if hasattr(node, "else_body") and node.else_body:
            self.add_line("else")
            self.add_line("{")
            self.indent_level += 1
            self.emit_body(node.else_body)
            self.indent_level -= 1
            self.add_line("}")

        return ""

    def visit_ForNode(self, node) -> str:
        if isinstance(node.init, VariableNode):
            init = self.format_variable_declaration(node.init)
        elif hasattr(node.init, "expression"):
            init = self.visit(node.init.expression)
        else:
            init = self.visit(node.init) if node.init else ""
        condition = self.visit(node.condition) if node.condition else ""
        update = self.visit(node.update) if node.update else ""

        self.add_line(f"for ({init}; {condition}; {update})")
        self.add_line("{")
        self.indent_level += 1
        self.emit_body(node.body)
        self.indent_level -= 1
        self.add_line("}")

        return ""

    def visit_WhileNode(self, node) -> str:
        condition = self.visit(node.condition) if node.condition else ""

        self.add_line(f"while ({condition})")
        self.add_line("{")
        self.indent_level += 1
        self.emit_body(node.body)
        self.indent_level -= 1
        self.add_line("}")

        return ""

    def visit_SwitchNode(self, node) -> str:
        expression = self.visit(node.expression)

        self.add_line(f"switch ({expression})")
        self.add_line("{")
        self.indent_level += 1
        for case in getattr(node, "cases", []):
            self.visit(case)
        self.indent_level -= 1
        self.add_line("}")

        return ""

    def visit_CaseNode(self, node) -> str:
        if getattr(node, "value", None) is None:
            self.add_line("default:")
        else:
            value = self.visit(node.value)
            self.add_line(f"case {value}:")

        self.indent_level += 1
        self.emit_body(getattr(node, "statements", []))
        self.indent_level -= 1

        return ""

    def visit_ReturnNode(self, node) -> str:
        if node.value:
            value = self.visit(node.value)
            self.add_line(f"return {value};")
        else:
            self.add_line("return;")
        return ""

    def visit_AssignmentNode(self, node) -> str:
        left = self.visit(node.left)
        right = self.visit(node.right)
        operator = getattr(node, "operator", getattr(node, "op", "="))
        return f"{left} {operator} {right}"

    def visit_BinaryOpNode(self, node) -> str:
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
        operand = self.visit(node.operand)

        if node.op == "not":
            return f"!{operand}"
        elif node.op in ["++", "--"]:
            if getattr(node, "is_postfix", getattr(node, "postfix", False)):
                return f"{operand}{node.op}"
            else:
                return f"{node.op}{operand}"
        else:
            return f"{node.op}{operand}"

    def visit_FunctionCallNode(self, node) -> str:
        func_expr = getattr(node, "function", node.name)
        func_name = None
        if hasattr(func_expr, "name"):
            func_name = func_expr.name
            callee = func_name
        elif isinstance(func_expr, str):
            func_name = func_expr
            callee = func_expr
        else:
            callee = self.visit(func_expr)
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
        target = mapped_name if mapped_name is not None else callee
        return f"{target}({args_str})"

    def visit_str(self, node) -> str:
        return str(node)

    def visit_int(self, node) -> str:
        return str(node)

    def visit_float(self, node) -> str:
        return str(node)

    def visit_ArrayAccessNode(self, node) -> str:
        array = self.visit(node.array)
        index = self.visit(node.index)
        return f"{array}[{index}]"

    def visit_MemberAccessNode(self, node) -> str:
        object_expr = self.visit(node.object)
        member_access = f"{object_expr}.{node.member}"
        if member_access in self.builtin_map:
            return self.builtin_map[member_access]

        # Handle vector swizzling
        if node.member in ["x", "y", "z", "w", "r", "g", "b", "a"]:
            return member_access
        elif len(node.member) > 1 and all(c in "xyzw" for c in node.member):
            # Multi-component swizzle - might need special handling
            return member_access
        else:
            return member_access

    def visit_TernaryOpNode(self, node) -> str:
        condition = self.visit(node.condition)
        true_expr = self.visit(node.true_expr)
        false_expr = self.visit(node.false_expr)
        return f"({condition} ? {true_expr} : {false_expr})"

    def format_literal(self, value, literal_type=None):
        if isinstance(value, bool):
            return "true" if value else "false"
        if literal_type == "bool" and isinstance(value, str):
            lower_value = value.lower()
            if lower_value in {"true", "false"}:
                return lower_value
        if literal_type == "char":
            escaped = self.escape_literal(value, quote="'")
            return f"'{escaped}'"
        if isinstance(value, str):
            escaped = self.escape_literal(value, quote='"')
            return f'"{escaped}"'
        return str(value)

    def escape_literal(self, value, quote):
        text = str(value)
        escaped = []
        for index, char in enumerate(text):
            if char == "\n":
                escaped.append("\\n")
            elif char == "\r":
                escaped.append("\\r")
            elif char == "\t":
                escaped.append("\\t")
            elif char == quote and (index == 0 or text[index - 1] != "\\"):
                escaped.append("\\" + char)
            else:
                escaped.append(char)
        return "".join(escaped)

    def visit_LiteralNode(self, node) -> str:
        literal_type = getattr(getattr(node, "literal_type", None), "name", None)
        return self.format_literal(node.value, literal_type)

    def visit_IdentifierNode(self, node) -> str:
        name = getattr(node, "name", str(node))
        # Handle built-in variables mapping
        return self.builtin_map.get(name, name)

    def visit_ExpressionStatementNode(self, node) -> str:
        expr = self.visit(node.expression)
        self.add_line(f"{expr};")
        return ""

    def visit_BlockNode(self, node) -> str:
        if hasattr(node, "statements"):
            self.emit_body(node.statements)
        return ""

    def visit_BreakNode(self, node) -> str:
        self.add_line("break;")
        return ""

    def visit_ContinueNode(self, node) -> str:
        self.add_line("continue;")
        return ""

    def visit_EnumNode(self, node) -> str:
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
            generic_args = getattr(type_node, "generic_args", [])
            if generic_args:
                args = ", ".join(
                    self.convert_type_node_to_string(arg) for arg in generic_args
                )
                return f"{type_node.name}<{args}>"
            return type_node.name
        elif hasattr(type_node, "element_type"):
            if hasattr(type_node, "rows"):
                element_type = self.convert_type_node_to_string(type_node.element_type)
                prefix = "dmat" if element_type == "double" else "mat"
                return f"{prefix}{type_node.rows}x{type_node.cols}"
            elif not hasattr(type_node, "size"):
                return str(type_node)
            elif str(type(type_node)).find("ArrayType") != -1:
                element_type = self.convert_type_node_to_string(type_node.element_type)
                if type_node.size is not None:
                    return f"{element_type}[{self.format_array_size(type_node.size)}]"
                else:
                    return f"{element_type}[]"
            else:
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

    def format_typed_declarator(self, type_name, name, dynamic_array_as_pointer=True):
        if hasattr(type_name, "name") or hasattr(type_name, "element_type"):
            type_name = self.convert_type_node_to_string(type_name)
        else:
            type_name = str(type_name)

        if "[" not in type_name or "]" not in type_name:
            return f"{self.map_type(type_name)} {name}"

        open_bracket = type_name.find("[")
        base_type = type_name[:open_bracket]
        array_suffix = type_name[open_bracket:]
        mapped_base = self.map_type(base_type)

        if dynamic_array_as_pointer and "[]" in array_suffix:
            array_suffix = array_suffix.replace("[]", "")
            return f"{mapped_base}* {name}{array_suffix}"

        return f"{mapped_base} {name}{array_suffix}"

    def format_array_size(self, size):
        if size is None:
            return ""
        if isinstance(size, int):
            return str(size)
        return self.visit(size)

    def generate_kernel_wrapper(self, kernel_node: FunctionNode) -> str:
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
    generator = HipCodeGen()
    return generator.generate(ast)

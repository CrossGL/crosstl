"""
CrossGL to HIP Code Generator

This module provides code generation functionality to convert CrossGL AST to HIP source code.
HIP (Heterogeneous-Compute Interface for Portability) is AMD's CUDA-compatible runtime API
for GPU programming.
"""

from ..ast import (
    ASTNode,
    ArrayAccessNode,
    ArrayLiteralNode,
    CbufferNode,
    FunctionNode,
    IdentifierNode,
    ShaderNode,
    StructNode,
    VariableNode,
)
from .resource_diagnostics import ResourceDiagnosticMixin
from .resource_query import ResourceQueryMixin
from .resource_arrays import format_array_declarator
from .vector_arithmetic import VectorArithmeticMixin


class HipCodeGen(VectorArithmeticMixin, ResourceQueryMixin, ResourceDiagnosticMixin):
    """Emit HIP source from the shared CrossGL translator AST."""

    resource_diagnostic_backend = "HIP"

    def __init__(self):
        """Initialize HIP type maps and per-generation visitor state."""
        self.indent_level = 0
        self.code_lines = []
        self.current_function = None
        self.variable_counter = 0
        self.variable_types = {}
        self.struct_member_types = {}
        self.function_return_types = {}
        self.helper_functions = {}
        self.query_resource_names = set()
        self.query_metadata_function_params = {}
        self.query_functions_by_name = {}
        self.current_function_name = None
        self.resource_query_info_required = False

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
            # Texture/resource types
            "sampler": "hipTextureObject_t",
            "sampler1D": "texture<float4, 1>",
            "sampler2D": "texture<float4, 2>",
            "sampler3D": "texture<float4, 3>",
            "samplerCube": "textureCube<float4>",
            "sampler2DArray": "hipTextureObject_t",
            "sampler2DShadow": "hipTextureObject_t",
            "sampler2DArrayShadow": "hipTextureObject_t",
            "samplerCubeShadow": "hipTextureObject_t",
            "samplerCubeArray": "hipTextureObject_t",
            "samplerCubeArrayShadow": "hipTextureObject_t",
            "sampler2DMS": "hipTextureObject_t",
            "sampler2DMSArray": "hipTextureObject_t",
            "image2D": "hipSurfaceObject_t",
            "image3D": "hipSurfaceObject_t",
            "imageCube": "hipSurfaceObject_t",
            "image2DArray": "hipSurfaceObject_t",
            "image2DMS": "hipSurfaceObject_t",
            "image2DMSArray": "hipSurfaceObject_t",
            "iimage2D": "hipSurfaceObject_t",
            "iimage3D": "hipSurfaceObject_t",
            "iimage2DArray": "hipSurfaceObject_t",
            "iimage2DMS": "hipSurfaceObject_t",
            "iimage2DMSArray": "hipSurfaceObject_t",
            "uimage2D": "hipSurfaceObject_t",
            "uimage3D": "hipSurfaceObject_t",
            "uimage2DArray": "hipSurfaceObject_t",
            "uimage2DMS": "hipSurfaceObject_t",
            "uimage2DMSArray": "hipSurfaceObject_t",
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
            "float2": "make_float2",
            "float3": "make_float3",
            "float4": "make_float4",
            "vec2<f32>": "make_float2",
            "vec3<f32>": "make_float3",
            "vec4<f32>": "make_float4",
            "ivec2": "make_int2",
            "ivec3": "make_int3",
            "ivec4": "make_int4",
            "int2": "make_int2",
            "int3": "make_int3",
            "int4": "make_int4",
            "vec2<i32>": "make_int2",
            "vec3<i32>": "make_int3",
            "vec4<i32>": "make_int4",
            "uvec2": "make_uint2",
            "uvec3": "make_uint3",
            "uvec4": "make_uint4",
            "uint2": "make_uint2",
            "uint3": "make_uint3",
            "uint4": "make_uint4",
            "vec2<u32>": "make_uint2",
            "vec3<u32>": "make_uint3",
            "vec4<u32>": "make_uint4",
            "dvec2": "make_double2",
            "dvec3": "make_double3",
            "dvec4": "make_double4",
            "double2": "make_double2",
            "double3": "make_double3",
            "double4": "make_double4",
            "vec2<f64>": "make_double2",
            "vec3<f64>": "make_double3",
            "vec4<f64>": "make_double4",
            "bvec2": "make_uchar2",
            "bvec3": "make_uchar3",
            "bvec4": "make_uchar4",
            "uchar2": "make_uchar2",
            "uchar3": "make_uchar3",
            "uchar4": "make_uchar4",
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
        """Generate complete HIP source for a CrossGL AST."""
        self.code_lines = []
        self.indent_level = 0
        self.variable_types = {}
        self.struct_member_types = {}
        self.function_return_types = self.collect_function_return_types(node)
        self.helper_functions = {}
        self.resource_query_info_required = False
        (
            self.query_resource_names,
            self.query_metadata_function_params,
        ) = self.collect_resource_query_requirements(node)
        self.query_functions_by_name = {
            getattr(func, "name", None): func
            for func in self.query_collect_functions(node)
        }
        self.query_functions_by_name = {
            name: func for name, func in self.query_functions_by_name.items() if name
        }

        self.add_includes()
        self.visit(node)
        self.insert_helper_functions()

        return "\n".join(self.code_lines)

    def add_includes(self):
        """Emit the standard HIP runtime include block."""
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
        """Return whitespace for the current indentation level."""
        return "    " * self.indent_level

    def add_line(self, line: str = ""):
        """Append one HIP output line using the current indentation level."""
        if line:
            self.code_lines.append(self.indent() + line)
        else:
            self.code_lines.append("")

    def visit(self, node: ASTNode) -> str:
        """Dispatch an AST node to its HIP visitor method."""
        method_name = f"visit_{type(node).__name__}"
        visitor = getattr(self, method_name, self.generic_visit)
        return visitor(node)

    def generic_visit(self, node: ASTNode) -> str:
        """Raise a clear error for unsupported AST nodes."""
        raise NotImplementedError(
            f"Code generation not implemented for {type(node).__name__}"
        )

    def visit_ShaderNode(self, node: ShaderNode) -> str:
        """Render a full shader/program AST as a HIP translation unit."""
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
        """Render a CrossGL function or compute entry point as HIP code."""
        saved_variable_types = self.variable_types.copy()
        self.current_function = node.name
        saved_current_function_name = self.current_function_name
        self.current_function_name = node.name

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
        param_declarations = []
        for param in param_list:
            param_declarations.append(self.visit_parameter(param))
            param_type = self.get_parameter_type(param)
            param_name = getattr(param, "name", getattr(param, "param_name", None))
            metadata_param = self.query_metadata_parameter(param_name, param_type)
            if metadata_param:
                param_declarations.append(metadata_param)
        params = ", ".join(param_declarations)

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
        self.variable_types = saved_variable_types
        self.current_function_name = saved_current_function_name
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

        self.register_variable_type(param_name, param_type)
        return self.format_typed_declarator(param_type, param_name)

    def visit_StructNode(self, node: StructNode) -> str:
        self.add_line(f"struct {node.name}")
        self.add_line("{")
        self.indent_level += 1

        members = getattr(node, "members", [])
        member_types = {}
        for member in members:
            if hasattr(member, "member_type"):
                member_type = member.member_type
            elif hasattr(member, "vtype"):
                member_type = member.vtype
            elif hasattr(member, "var_type"):
                member_type = member.var_type
            else:
                member_type = "float"

            member_types[member.name] = member_type
            self.add_line(f"{self.format_typed_declarator(member_type, member.name)};")

        self.struct_member_types[node.name] = member_types
        self.indent_level -= 1
        self.add_line("};")
        self.add_line()
        return ""

    def visit_VariableNode(self, node: VariableNode) -> str:
        var_type = self.get_variable_node_type(node)
        self.add_line(f"{self.format_variable_declaration(node)};")
        metadata_declaration = self.query_metadata_declaration(node.name, var_type)
        if metadata_declaration:
            self.add_line(f"{metadata_declaration};")
        return ""

    def format_variable_declaration(self, node: VariableNode) -> str:
        if hasattr(node, "var_type"):
            var_type = node.var_type
        elif hasattr(node, "vtype"):
            var_type = node.vtype
        else:
            var_type = "int"

        self.register_variable_type(node.name, var_type)
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
        """Render and append one statement node when it produces code."""
        if node is None:
            return

        result = self.visit(node)
        if isinstance(result, str) and result.strip():
            self.add_line(f"{result};")

    def emit_body(self, body):
        """Render a list-like or block-like function body."""
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
        if operator in {"+=", "-=", "*=", "/="}:
            lowered_right = self.lower_vector_binary_operation(
                node.left,
                left,
                node.right,
                right,
                operator[0],
            )
            if lowered_right is not None:
                return f"{left} = {lowered_right}"
        return f"{left} {operator} {right}"

    def visit_BinaryOpNode(self, node) -> str:
        left = self.visit(node.left)
        right = self.visit(node.right)

        # Handle special operators
        if node.op == "and":
            return f"({left} && {right})"
        elif node.op == "or":
            return f"({left} || {right})"
        lowered = self.lower_vector_binary_operation(
            node.left,
            left,
            node.right,
            right,
            node.op,
        )
        if lowered is not None:
            return lowered
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
        func_expr = (
            node.function if hasattr(node, "function") else getattr(node, "name", None)
        )
        func_name = None
        if hasattr(func_expr, "name"):
            func_name = func_expr.name
            callee = func_name
        elif isinstance(func_expr, str):
            func_name = func_expr
            callee = func_expr
        else:
            callee = self.visit(func_expr)
        raw_args = getattr(node, "args", getattr(node, "arguments", []))
        args = [self.visit(arg) for arg in raw_args]

        resource_call = self.generate_resource_call(func_name, raw_args, args)
        if resource_call is not None:
            return resource_call

        args = self.query_metadata_call_arguments(func_name, raw_args, args)
        vector_info = self.vector_type_info(func_name)
        if vector_info and len(args) == 1:
            arg_type = self.expression_result_type(raw_args[0])
            if arg_type is not None and not self.vector_type_info(arg_type):
                args = args * len(vector_info["components"])

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

    def insert_helper_functions(self):
        if not self.helper_functions:
            return
        helpers = []
        if self.resource_query_info_required:
            helpers.extend(
                [
                    "struct CglResourceQueryInfo {",
                    "    int width;",
                    "    int height;",
                    "    int depth;",
                    "    int elements;",
                    "    int levels;",
                    "    int samples;",
                    "};",
                    "",
                ]
            )
        for helper in self.helper_functions.values():
            helpers.extend(helper.splitlines())
            helpers.append("")
        self.code_lines[5:5] = helpers

    def register_variable_type(self, name, type_name):
        if not name or type_name is None:
            return
        if not isinstance(type_name, str):
            type_name = self.convert_type_node_to_string(type_name)
        self.variable_types[name] = type_name

    def get_expression_name(self, node):
        if isinstance(node, IdentifierNode):
            return node.name
        if isinstance(node, VariableNode):
            return node.name
        if isinstance(node, str):
            return node
        if isinstance(node, ArrayAccessNode):
            array_node = getattr(node, "array", getattr(node, "array_expr", None))
            return self.get_expression_name(array_node)
        return None

    def get_expression_type(self, node):
        name = self.get_expression_name(node)
        if name is None:
            return None
        return self.variable_types.get(name)

    def map_vector_arithmetic_type(self, type_name):
        return self.map_type(type_name)

    def require_surface_read_helper(self, helper_name):
        helpers = {
            "cgl_surf2Dread": (
                "template <typename T>\n"
                "__device__ T cgl_surf2Dread(hipSurfaceObject_t surfObj, int x, int y)\n"
                "{\n"
                "    T value;\n"
                "    surf2Dread(&value, surfObj, x, y);\n"
                "    return value;\n"
                "}"
            ),
            "cgl_surf3Dread": (
                "template <typename T>\n"
                "__device__ T cgl_surf3Dread(hipSurfaceObject_t surfObj, int x, int y, int z)\n"
                "{\n"
                "    T value;\n"
                "    surf3Dread(&value, surfObj, x, y, z);\n"
                "    return value;\n"
                "}"
            ),
            "cgl_surf2DLayeredread": (
                "template <typename T>\n"
                "__device__ T cgl_surf2DLayeredread(hipSurfaceObject_t surfObj, int x, int y, int layer)\n"
                "{\n"
                "    T value;\n"
                "    surf2DLayeredread(&value, surfObj, x, y, layer);\n"
                "    return value;\n"
                "}"
            ),
        }
        self.require_helper_function(helper_name, helpers[helper_name])

    def generate_resource_call(self, func_name, raw_args, args):
        if func_name in {"textureSize", "imageSize"}:
            return self.generate_dimension_query(func_name, raw_args, args)

        if func_name in {"textureSamples", "imageSamples"}:
            return self.generate_sample_count_query(func_name, raw_args, args)

        if func_name == "textureQueryLevels":
            return self.generate_texture_query_levels(raw_args)

        if func_name == "textureQueryLod" and len(args) >= 2:
            texture_type = self.resource_base_type(
                self.get_expression_type(raw_args[0])
            )
            if texture_type is not None:
                return self.unsupported_resource_query_call(
                    func_name, texture_type, args
                )

        if (
            func_name
            in {
                "texture",
                "textureLod",
                "textureGrad",
                "textureGather",
                "textureCompare",
                "textureCompareLod",
                "textureCompareGrad",
                "textureCompareOffset",
                "textureGatherCompare",
                "textureGatherCompareOffset",
            }
            and len(args) >= 2
        ):
            texture_type = self.resource_base_type(
                self.get_expression_type(raw_args[0])
            )
            if self.is_shadow_resource_type(texture_type):
                return self.unsupported_shadow_resource_call(
                    func_name, texture_type, args
                )

        if (
            func_name
            in {
                "textureGather",
                "textureGatherOffset",
                "textureGatherOffsets",
            }
            and len(args) >= 2
        ):
            texture_type = self.resource_base_type(
                self.get_expression_type(raw_args[0])
            )
            if texture_type is not None:
                return self.unsupported_sampled_resource_call(
                    func_name, texture_type, args
                )

        if func_name in {
            "imageAtomicAdd",
            "imageAtomicMin",
            "imageAtomicMax",
            "imageAtomicAnd",
            "imageAtomicOr",
            "imageAtomicXor",
            "imageAtomicExchange",
            "imageAtomicCompSwap",
        }:
            image_type = None
            if raw_args:
                image_type = self.resource_base_type(
                    self.get_expression_type(raw_args[0])
                )
            return self.unsupported_image_atomic_resource_call(
                func_name, image_type, args
            )

        if func_name in {"texture", "textureLod", "textureGrad"} and len(args) >= 2:
            texture_type = self.resource_base_type(
                self.get_expression_type(raw_args[0])
            )
            if self.is_multisample_resource_type(texture_type):
                return self.unsupported_multisample_resource_call(
                    func_name, texture_type, args
                )

            texture_name = args[0]
            coord = args[1]
            if texture_type == "sampler1D":
                if func_name == "texture":
                    return f"tex1D({texture_name}, {coord})"
                if func_name == "textureLod" and len(args) >= 3:
                    return f"tex1DLod({texture_name}, {coord}, {args[2]})"
                if func_name == "textureGrad" and len(args) >= 4:
                    return f"tex1DGrad({texture_name}, {coord}, {args[2]}, {args[3]})"

            if texture_type == "sampler2DArray":
                coord_args = (
                    f"{texture_name}, "
                    f"{self.coord_component(coord, 'x')}, "
                    f"{self.coord_component(coord, 'y')}, "
                    f"{self.coord_component(coord, 'z')}"
                )
                if func_name == "texture":
                    return f"tex2DLayered<float4>({coord_args})"
                if func_name == "textureLod" and len(args) >= 3:
                    return f"tex2DLayeredLod<float4>({coord_args}, {args[2]})"
                if func_name == "textureGrad" and len(args) >= 4:
                    return (
                        f"tex2DLayeredGrad<float4>"
                        f"({coord_args}, {args[2]}, {args[3]})"
                    )

            if texture_type == "sampler3D":
                coord_args = (
                    f"{texture_name}, "
                    f"{self.coord_component(coord, 'x')}, "
                    f"{self.coord_component(coord, 'y')}, "
                    f"{self.coord_component(coord, 'z')}"
                )
                if func_name == "texture":
                    return f"tex3D({coord_args})"
                if func_name == "textureLod" and len(args) >= 3:
                    return f"tex3DLod({coord_args}, {args[2]})"
                if func_name == "textureGrad" and len(args) >= 4:
                    return f"tex3DGrad({coord_args}, {args[2]}, {args[3]})"

            if texture_type == "samplerCube":
                coord_args = (
                    f"{texture_name}, "
                    f"{self.coord_component(coord, 'x')}, "
                    f"{self.coord_component(coord, 'y')}, "
                    f"{self.coord_component(coord, 'z')}"
                )
                if func_name == "texture":
                    return f"texCubemap({coord_args})"
                if func_name == "textureLod" and len(args) >= 3:
                    return f"texCubemapLod({coord_args}, {args[2]})"
                if func_name == "textureGrad" and len(args) >= 4:
                    return f"texCubemapGrad({coord_args}, {args[2]}, {args[3]})"

            if texture_type == "samplerCubeArray":
                coord_args = (
                    f"{texture_name}, "
                    f"{self.coord_component(coord, 'x')}, "
                    f"{self.coord_component(coord, 'y')}, "
                    f"{self.coord_component(coord, 'z')}, "
                    f"{self.coord_component(coord, 'w')}"
                )
                if func_name == "texture":
                    return f"texCubemapLayered<float4>({coord_args})"
                if func_name == "textureLod" and len(args) >= 3:
                    return f"texCubemapLayeredLod<float4>({coord_args}, {args[2]})"
                if func_name == "textureGrad" and len(args) >= 4:
                    return (
                        f"texCubemapLayeredGrad<float4>"
                        f"({coord_args}, {args[2]}, {args[3]})"
                    )

        if func_name == "texelFetch" and len(args) >= 3:
            texture_type = self.resource_base_type(
                self.get_expression_type(raw_args[0])
            )
            if self.is_multisample_resource_type(texture_type):
                return self.unsupported_multisample_resource_call(
                    func_name, texture_type, args
                )

            texture_name = args[0]
            coord = args[1]
            if texture_type == "sampler2D":
                return (
                    f"tex2D({texture_name}, "
                    f"{self.coord_component(coord, 'x')}, "
                    f"{self.coord_component(coord, 'y')})"
                )
            if texture_type == "sampler2DArray":
                return (
                    f"tex2DLayered<float4>({texture_name}, "
                    f"{self.coord_component(coord, 'x')}, "
                    f"{self.coord_component(coord, 'y')}, "
                    f"{self.coord_component(coord, 'z')})"
                )
            if texture_type == "sampler3D":
                return (
                    f"tex3D({texture_name}, "
                    f"{self.coord_component(coord, 'x')}, "
                    f"{self.coord_component(coord, 'y')}, "
                    f"{self.coord_component(coord, 'z')})"
                )

        if func_name == "imageLoad" and len(args) >= 2:
            image_type = self.resource_base_type(self.get_expression_type(raw_args[0]))
            if image_type is None:
                return None
            if self.is_multisample_resource_type(image_type):
                return self.unsupported_multisample_resource_call(
                    func_name, image_type, args
                )

            image_name = args[0]
            coord = args[1]
            value_type = self.image_value_type(image_type)
            x = self.surface_x_offset(coord, value_type)
            y = self.coord_component(coord, "y")

            if "3D" in image_type:
                self.require_surface_read_helper("cgl_surf3Dread")
                z = self.coord_component(coord, "z")
                return f"cgl_surf3Dread<{value_type}>({image_name}, {x}, {y}, {z})"
            if "Array" in image_type:
                self.require_surface_read_helper("cgl_surf2DLayeredread")
                layer = self.coord_component(coord, "z")
                return (
                    f"cgl_surf2DLayeredread<{value_type}>"
                    f"({image_name}, {x}, {y}, {layer})"
                )
            if "2D" in image_type:
                self.require_surface_read_helper("cgl_surf2Dread")
                return f"cgl_surf2Dread<{value_type}>({image_name}, {x}, {y})"

        if func_name == "imageStore" and len(args) >= 3:
            image_type = self.resource_base_type(self.get_expression_type(raw_args[0]))
            if image_type is None:
                return None
            if self.is_multisample_resource_type(image_type):
                return self.unsupported_multisample_resource_call(
                    func_name, image_type, args
                )

            image_name = args[0]
            coord = args[1]
            value = args[2]
            value_type = self.image_value_type(image_type)
            x = self.surface_x_offset(coord, value_type)
            y = self.coord_component(coord, "y")

            if "3D" in image_type:
                z = self.coord_component(coord, "z")
                return f"surf3Dwrite({value}, {image_name}, {x}, {y}, {z})"
            if "Array" in image_type:
                layer = self.coord_component(coord, "z")
                return f"surf2DLayeredwrite({value}, {image_name}, {x}, {y}, {layer})"
            if "2D" in image_type:
                return f"surf2Dwrite({value}, {image_name}, {x}, {y})"

        return None

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

    def visit_ArrayLiteralNode(self, node: ArrayLiteralNode) -> str:
        elements = ", ".join(self.visit(element) for element in node.elements)
        return f"{{{elements}}}"

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
        if (
            literal_type == "uint"
            and isinstance(value, int)
            and not isinstance(value, bool)
        ):
            return f"{value}u"
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
        """Map a CrossGL type name or type node to a HIP type string."""
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

        return format_array_declarator(
            mapped_base,
            name,
            array_suffix,
            dynamic_array_as_pointer=dynamic_array_as_pointer,
        )

    def format_array_size(self, size):
        if size is None:
            return ""
        if isinstance(size, int):
            return str(size)
        return self.visit(size)

    def generate_kernel_wrapper(self, kernel_node: FunctionNode) -> str:
        """Generate a host-side HIP launch wrapper for a kernel node."""
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
    """Generate HIP source from a CrossGL shader AST."""
    generator = HipCodeGen()
    return generator.generate(ast)

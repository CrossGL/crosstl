from ..ast import (
    AssignmentNode,
    ForNode,
    IfNode,
    ReturnNode,
    StructNode,
    VariableNode,
    ArrayAccessNode,
    ArrayNode,
    ShaderNode,
    FunctionNode,
    ExpressionStatementNode,
    IdentifierNode,
    BlockNode,
)
from .resource_diagnostics import ResourceDiagnosticMixin
from .resource_query import ResourceQueryMixin
from .resource_arrays import format_array_declarator
from .vector_arithmetic import VectorArithmeticMixin


class CudaCodeGen(VectorArithmeticMixin, ResourceQueryMixin, ResourceDiagnosticMixin):
    resource_diagnostic_backend = "CUDA"

    def __init__(self):
        self.indent_level = 0
        self.output = []
        self.variable_types = {}
        self.struct_member_types = {}
        self.function_return_types = {}
        self.helper_functions = {}
        self.query_resource_names = set()
        self.query_metadata_function_params = {}
        self.query_functions_by_name = {}
        self.current_function_name = None
        self.resource_query_info_required = False
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

    def generate(self, ast_node):
        self.output = []
        self.indent_level = 0
        self.variable_types = {}
        self.struct_member_types = {}
        self.function_return_types = self.collect_function_return_types(ast_node)
        self.helper_functions = {}
        self.resource_query_info_required = False
        (
            self.query_resource_names,
            self.query_metadata_function_params,
        ) = self.collect_resource_query_requirements(ast_node)
        self.query_functions_by_name = {
            getattr(func, "name", None): func
            for func in self.query_collect_functions(ast_node)
        }
        self.query_functions_by_name = {
            name: func for name, func in self.query_functions_by_name.items() if name
        }
        self.visit(ast_node)
        self.insert_helper_functions()
        return "\n".join(self.output)

    def visit(self, node):
        method_name = f"visit_{type(node).__name__}"
        visitor = getattr(self, method_name, self.generic_visit)
        return visitor(node)

    def generic_visit(self, node):
        if isinstance(node, str):
            return node
        elif isinstance(node, list):
            return [self.visit(item) for item in node]
        else:
            return str(node)

    def emit(self, code):
        if code.strip():
            self.output.append("    " * self.indent_level + code)
        else:
            self.output.append("")

    def emit_statement(self, node):
        if node is None:
            return

        result = self.visit(node)
        if isinstance(result, str) and result.strip():
            self.emit(f"{result};")

    def emit_body(self, body):
        if isinstance(body, list):
            for stmt in body:
                self.emit_statement(stmt)
        elif hasattr(body, "statements"):
            for stmt in body.statements:
                self.emit_statement(stmt)
        else:
            self.emit_statement(body)

    def visit_ShaderNode(self, node):
        self.emit("#include <cuda_runtime.h>")
        self.emit("#include <device_launch_parameters.h>")
        self.emit("")

        structs = getattr(node, "structs", [])
        for struct in structs:
            self.visit(struct)
            self.emit("")

        cbuffers = getattr(node, "cbuffers", [])
        for cbuffer in cbuffers:
            self.visit_cbuffer(cbuffer)
            self.emit("")

        global_vars = getattr(node, "global_variables", [])
        for var in global_vars:
            self.visit(var)
            self.emit("")

        functions = getattr(node, "functions", [])
        for func in functions:
            self.visit(func)
            self.emit("")

        # Handle legacy shader structure
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
                    self.emit("")

    def visit_FunctionNode(self, node):
        saved_variable_types = self.variable_types.copy()
        saved_current_function_name = self.current_function_name
        self.current_function_name = node.name
        qualifiers = []

        if hasattr(node, "qualifiers") and node.qualifiers:
            for qualifier in node.qualifiers:
                if qualifier == "compute":
                    qualifiers.append("__global__")
                elif qualifier in ["vertex", "fragment"]:
                    qualifiers.append("__device__")
                else:
                    qualifiers.append("__device__")
        elif hasattr(node, "qualifier") and node.qualifier:
            if node.qualifier == "compute":
                qualifiers.append("__global__")
            elif node.qualifier in ["vertex", "fragment"]:
                qualifiers.append("__device__")
            else:
                qualifiers.append("__device__")
        else:
            qualifiers.append("__device__")

        if hasattr(node, "return_type"):
            return_type = self.convert_crossgl_type_to_cuda(node.return_type)
        else:
            return_type = "void"

        qualifier_str = " ".join(qualifiers)

        params = []
        param_list = getattr(node, "parameters", getattr(node, "params", []))

        for param in param_list:
            if hasattr(param, "param_type"):
                param_type = param.param_type
            elif hasattr(param, "vtype"):
                param_type = param.vtype
            else:
                param_type = "void"

            self.register_variable_type(param.name, param_type)
            params.append(self.format_typed_declarator(param_type, param.name))
            metadata_param = self.query_metadata_parameter(param.name, param_type)
            if metadata_param:
                params.append(metadata_param)

        param_str = ", ".join(params)
        self.emit(f"{qualifier_str} {return_type} {node.name}({param_str}) {{")

        self.indent_level += 1

        body = getattr(node, "body", [])
        self.emit_body(body)

        self.indent_level -= 1
        self.emit("}")
        self.variable_types = saved_variable_types
        self.current_function_name = saved_current_function_name

    def visit_StructNode(self, node):
        self.emit(f"struct {node.name} {{")
        self.indent_level += 1

        members = getattr(node, "members", [])
        member_types = {}
        for member in members:
            if hasattr(member, "member_type"):
                member_type = member.member_type
            elif hasattr(member, "vtype"):
                member_type = member.vtype
            else:
                member_type = "float"

            member_types[member.name] = member_type
            self.emit(f"{self.format_typed_declarator(member_type, member.name)};")

        self.struct_member_types[node.name] = member_types
        self.indent_level -= 1
        self.emit("};")

    def format_variable_declaration(self, node):
        var_type = None

        if hasattr(node, "var_type"):
            var_type = node.var_type
        elif hasattr(node, "vtype"):
            var_type = node.vtype

        if var_type:
            self.register_variable_type(node.name, var_type)
            # Check for special memory qualifiers
            qualifiers = []
            if hasattr(node, "qualifiers"):
                for qualifier in node.qualifiers:
                    if "workgroup" in str(qualifier) or "shared" in str(qualifier):
                        qualifiers.append("__shared__")
                    elif "uniform" in str(qualifier):
                        qualifiers.append("__constant__")

            qualifier_str = " ".join(qualifiers)
            if qualifier_str:
                qualifier_str += " "

            declaration = (
                f"{qualifier_str}{self.format_typed_declarator(var_type, node.name)}"
            )
            initial_value = getattr(node, "initial_value", getattr(node, "value", None))
            if initial_value is not None:
                declaration += f" = {self.visit(initial_value)}"
            return declaration

        return node.name

    def format_typed_declarator(self, type_name, name, dynamic_array_as_pointer=True):
        if hasattr(type_name, "name") or hasattr(type_name, "element_type"):
            type_name = self.convert_type_node_to_string(type_name)
        else:
            type_name = str(type_name)

        if "[" not in type_name or "]" not in type_name:
            return f"{self.convert_crossgl_type_to_cuda(type_name)} {name}"

        open_bracket = type_name.find("[")
        base_type = type_name[:open_bracket]
        array_suffix = type_name[open_bracket:]
        mapped_base = self.convert_crossgl_type_to_cuda(base_type)

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

    def visit_VariableNode(self, node):
        var_type = self.get_variable_node_type(node)
        declaration = self.format_variable_declaration(node)
        if declaration != node.name:
            self.emit(f"{declaration};")
            metadata_declaration = self.query_metadata_declaration(node.name, var_type)
            if metadata_declaration:
                self.emit(f"{metadata_declaration};")
            return None

        return node.name

    def visit_ExpressionStatementNode(self, node):
        expr = self.visit(node.expression)
        if expr and expr.strip():
            self.emit(f"{expr};")

    def visit_IdentifierNode(self, node):
        name = getattr(node, "name", str(node))
        return self.builtin_map.get(name, name)

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

    def visit_LiteralNode(self, node):
        literal_type = getattr(getattr(node, "literal_type", None), "name", None)
        return self.format_literal(node.value, literal_type)

    def visit_AssignmentNode(self, node):
        target = self.visit(node.target)
        value = self.visit(node.value)
        operator = getattr(node, "operator", "=")
        if operator in {"+=", "-=", "*=", "/="}:
            lowered_value = self.lower_vector_binary_operation(
                node.target,
                target,
                node.value,
                value,
                operator[0],
            )
            if lowered_value is not None:
                self.emit(f"{target} = {lowered_value};")
                return
        self.emit(f"{target} {operator} {value};")

    def visit_BinaryOpNode(self, node):
        left = self.visit(node.left)
        right = self.visit(node.right)
        operator = getattr(node, "operator", getattr(node, "op", "+"))
        lowered = self.lower_vector_binary_operation(
            node.left,
            left,
            node.right,
            right,
            operator,
        )
        if lowered is not None:
            return lowered
        return f"({left} {operator} {right})"

    def visit_UnaryOpNode(self, node):
        operand = self.visit(node.operand)
        operator = getattr(node, "operator", getattr(node, "op", "+"))
        if getattr(node, "is_postfix", getattr(node, "postfix", False)):
            return f"{operand}{operator}"
        return f"{operator}{operand}"

    def visit_FunctionCallNode(self, node):
        """Visit function call"""
        if hasattr(node, "function"):
            func_name = self.visit(node.function)
        else:
            func_name = getattr(node, "name", "unknown")

        raw_args = []
        if hasattr(node, "arguments"):
            raw_args = node.arguments
        elif hasattr(node, "args"):
            raw_args = node.args

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
        args_str = ", ".join(args)

        # Convert built-in functions
        func_name = self.convert_builtin_function(func_name)
        return f"{func_name}({args_str})"

    def visit_MemberAccessNode(self, node):
        """Visit member access"""
        if hasattr(node, "object_expr"):
            obj = self.visit(node.object_expr)
        else:
            obj = self.visit(node.object)
        member_access = f"{obj}.{node.member}"
        return self.builtin_map.get(member_access, member_access)

    def visit_ArrayAccessNode(self, node):
        """Visit array access"""
        if hasattr(node, "array_expr"):
            array = self.visit(node.array_expr)
        else:
            array = self.visit(node.array)

        if hasattr(node, "index_expr"):
            index = self.visit(node.index_expr)
        else:
            index = self.visit(node.index)

        return f"{array}[{index}]"

    def visit_IfNode(self, node):
        """Visit if statement"""
        condition = self.visit(node.condition)
        self.emit(f"if ({condition}) {{")

        self.indent_level += 1

        # Handle then branch
        if hasattr(node, "then_branch"):
            self.emit_body(node.then_branch)
        elif hasattr(node, "if_body"):
            self.emit_body(node.if_body)

        self.indent_level -= 1

        # Handle else branch
        if hasattr(node, "else_branch") and node.else_branch:
            self.emit("} else {")
            self.indent_level += 1

            self.emit_body(node.else_branch)

            self.indent_level -= 1
        elif hasattr(node, "else_body") and node.else_body:
            self.emit("} else {")
            self.indent_level += 1
            self.emit_body(node.else_body)
            self.indent_level -= 1

        self.emit("}")

    def visit_ForNode(self, node):
        """Visit for loop"""
        init_str = ""
        if node.init:
            if isinstance(node.init, VariableNode):
                init_str = self.format_variable_declaration(node.init)
            elif hasattr(node.init, "expression"):
                init_str = self.visit(node.init.expression)
            else:
                init_str = self.visit(node.init)

        condition_str = ""
        if node.condition:
            condition_str = self.visit(node.condition)

        update_str = ""
        if node.update:
            update_str = self.visit(node.update)

        self.emit(f"for ({init_str}; {condition_str}; {update_str}) {{")

        self.indent_level += 1

        # Handle body
        if hasattr(node, "body"):
            self.emit_body(node.body)

        self.indent_level -= 1
        self.emit("}")

    def visit_WhileNode(self, node):
        """Visit while loop"""
        condition = self.visit(node.condition) if node.condition else ""
        self.emit(f"while ({condition}) {{")

        self.indent_level += 1

        if hasattr(node, "body"):
            self.emit_body(node.body)

        self.indent_level -= 1
        self.emit("}")

    def visit_SwitchNode(self, node):
        """Visit switch statement"""
        expression = self.visit(node.expression)
        self.emit(f"switch ({expression}) {{")

        self.indent_level += 1
        for case in getattr(node, "cases", []):
            self.visit(case)
        self.indent_level -= 1

        self.emit("}")

    def visit_CaseNode(self, node):
        """Visit switch case/default label"""
        if getattr(node, "value", None) is None:
            self.emit("default:")
        else:
            value = self.visit(node.value)
            self.emit(f"case {value}:")

        self.indent_level += 1
        for stmt in getattr(node, "statements", []):
            self.emit_statement(stmt)
        self.indent_level -= 1

    def visit_ReturnNode(self, node):
        """Visit return statement"""
        if node.value:
            value = self.visit(node.value)
            self.emit(f"return {value};")
        else:
            self.emit("return;")

    def visit_BreakNode(self, node):
        """Visit break statement"""
        self.emit("break;")

    def visit_ContinueNode(self, node):
        """Visit continue statement"""
        self.emit("continue;")

    def visit_BlockNode(self, node):
        """Visit block statement"""
        self.emit_body(node.statements)

    def convert_crossgl_type_to_cuda(self, crossgl_type):
        """Convert CrossGL types to CUDA equivalents"""
        if hasattr(crossgl_type, "name") or hasattr(crossgl_type, "element_type"):
            crossgl_type = self.convert_type_node_to_string(crossgl_type)
        else:
            crossgl_type = str(crossgl_type)

        type_mapping = {
            # Basic types
            "void": "void",
            "bool": "bool",
            "i8": "char",
            "u8": "unsigned char",
            "i16": "short",
            "u16": "unsigned short",
            "i32": "int",
            "u32": "unsigned int",
            "i64": "long long",
            "u64": "unsigned long long",
            "f32": "float",
            "f64": "double",
            "int": "int",
            "float": "float",
            "double": "double",
            # Vector types (with generics)
            "vec2<f32>": "float2",
            "vec3<f32>": "float3",
            "vec4<f32>": "float4",
            "vec2<f64>": "double2",
            "vec3<f64>": "double3",
            "vec4<f64>": "double4",
            "vec2<i32>": "int2",
            "vec3<i32>": "int3",
            "vec4<i32>": "int4",
            "vec2<u32>": "uint2",
            "vec3<u32>": "uint3",
            "vec4<u32>": "uint4",
            # Vector types (without generics - for compatibility)
            "vec2": "float2",
            "vec3": "float3",
            "vec4": "float4",
            "dvec2": "double2",
            "dvec3": "double3",
            "dvec4": "double4",
            "ivec2": "int2",
            "ivec3": "int3",
            "ivec4": "int4",
            "uvec2": "uint2",
            "uvec3": "uint3",
            "uvec4": "uint4",
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
            "sampler": "cudaTextureObject_t",
            "sampler1D": "texture<float4, 1>",
            "sampler2D": "texture<float4, 2>",
            "sampler3D": "texture<float4, 3>",
            "samplerCube": "textureCube<float4>",
            "sampler2DArray": "cudaTextureObject_t",
            "sampler2DShadow": "cudaTextureObject_t",
            "sampler2DArrayShadow": "cudaTextureObject_t",
            "samplerCubeShadow": "cudaTextureObject_t",
            "samplerCubeArray": "cudaTextureObject_t",
            "samplerCubeArrayShadow": "cudaTextureObject_t",
            "sampler2DMS": "cudaTextureObject_t",
            "sampler2DMSArray": "cudaTextureObject_t",
            "image2D": "cudaSurfaceObject_t",
            "image3D": "cudaSurfaceObject_t",
            "imageCube": "cudaSurfaceObject_t",
            "image2DArray": "cudaSurfaceObject_t",
            "image2DMS": "cudaSurfaceObject_t",
            "image2DMSArray": "cudaSurfaceObject_t",
            "iimage2D": "cudaSurfaceObject_t",
            "iimage3D": "cudaSurfaceObject_t",
            "iimage2DArray": "cudaSurfaceObject_t",
            "iimage2DMS": "cudaSurfaceObject_t",
            "iimage2DMSArray": "cudaSurfaceObject_t",
            "uimage2D": "cudaSurfaceObject_t",
            "uimage3D": "cudaSurfaceObject_t",
            "uimage2DArray": "cudaSurfaceObject_t",
            "uimage2DMS": "cudaSurfaceObject_t",
            "uimage2DMSArray": "cudaSurfaceObject_t",
            "buffer": "CUdeviceptr",
        }

        # Handle arrays
        if crossgl_type.startswith("array<") and crossgl_type.endswith(">"):
            # Extract element type and size
            inner = crossgl_type[6:-1]  # Remove "array<" and ">"
            if "," in inner:
                parts = inner.split(",")
                element_type = parts[0].strip()
                size = parts[1].strip()
                cuda_element_type = type_mapping.get(element_type, element_type)
                return f"{cuda_element_type}[{size}]"
            else:
                cuda_element_type = type_mapping.get(inner, inner)
                return f"{cuda_element_type}*"

        # Handle pointers
        if crossgl_type.startswith("ptr<") and crossgl_type.endswith(">"):
            element_type = crossgl_type[4:-1]  # Remove "ptr<" and ">"
            cuda_element_type = type_mapping.get(element_type, element_type)
            return f"{cuda_element_type}*"

        return type_mapping.get(crossgl_type, crossgl_type)

    def convert_builtin_function(self, func_name):
        """Convert CrossGL built-in functions to CUDA equivalents"""
        function_mapping = {
            # Math functions
            "sqrt": "sqrtf",
            "pow": "powf",
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
            "log": "logf",
            "log2": "log2f",
            "exp": "expf",
            "exp2": "exp2f",
            "inversesqrt": "rsqrtf",
            "abs": "fabsf",
            "round": "roundf",
            "trunc": "truncf",
            "mod": "fmodf",
            "min": "fminf",
            "max": "fmaxf",
            "floor": "floorf",
            "ceil": "ceilf",
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
            "dvec2": "make_double2",
            "dvec3": "make_double3",
            "dvec4": "make_double4",
            "double2": "make_double2",
            "double3": "make_double3",
            "double4": "make_double4",
            "vec2<f64>": "make_double2",
            "vec3<f64>": "make_double3",
            "vec4<f64>": "make_double4",
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
            # Atomic operations
            "atomicAdd": "atomicAdd",
            "atomicSub": "atomicSub",
            "atomicMax": "atomicMax",
            "atomicMin": "atomicMin",
            "atomicExchange": "atomicExch",
            "atomicCompareExchange": "atomicCAS",
            # Synchronization
            "workgroupBarrier": "__syncthreads",
            # Texture functions
            "texture": "tex2D",
            "textureLod": "tex2DLod",
            "textureGrad": "tex2DGrad",
        }

        return function_mapping.get(func_name, func_name)

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
        return self.convert_crossgl_type_to_cuda(type_name)

    def insert_helper_functions(self):
        if not self.helper_functions:
            return

        helper_lines = []
        if self.resource_query_info_required:
            helper_lines.extend(
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
            helper_lines.extend(helper.splitlines())
            helper_lines.append("")

        self.output[3:3] = helper_lines

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
                z = self.coord_component(coord, "z")
                return f"surf3Dread<{value_type}>({image_name}, {x}, {y}, {z})"
            if "Array" in image_type:
                layer = self.coord_component(coord, "z")
                return (
                    f"surf2DLayeredread<{value_type}>"
                    f"({image_name}, {x}, {y}, {layer})"
                )
            if "2D" in image_type:
                return f"surf2Dread<{value_type}>({image_name}, {x}, {y})"

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

    def visit_cbuffer(self, cbuffer):
        """Visit constant buffer (convert to CUDA constant memory)"""
        self.emit(f"// Constant buffer: {cbuffer.name}")
        for member in cbuffer.members:
            if hasattr(member, "member_type"):
                member_type = member.member_type
            else:
                member_type = member.vtype
            declaration = self.format_typed_declarator(member_type, member.name)
            self.emit(f"__constant__ {declaration};")

    def visit_ArrayNode(self, node):
        """Visit array declaration"""
        if hasattr(node, "element_type"):
            element_type = self.convert_crossgl_type_to_cuda(node.element_type)
        else:
            element_type = self.convert_crossgl_type_to_cuda(node.vtype)

        if node.size:
            self.emit(
                f"{element_type} {node.name}[{self.format_array_size(node.size)}];"
            )
        else:
            # Dynamic array - use pointer in CUDA
            self.emit(f"{element_type}* {node.name};")

    def visit_TernaryOpNode(self, node):
        """Visit ternary conditional operator"""
        condition = self.visit(node.condition)
        true_expr = self.visit(node.true_expr)
        false_expr = self.visit(node.false_expr)
        return f"({condition} ? {true_expr} : {false_expr})"

    def visit_list(self, node_list):
        """Visit a list of nodes"""
        results = []
        for node in node_list:
            result = self.visit(node)
            if result:
                results.append(result)
        return results

    def visit_str(self, node):
        """Visit string literals"""
        return node

    def visit_int(self, node):
        """Visit integer literals"""
        return str(node)

    def visit_float(self, node):
        """Visit float literals"""
        return str(node)

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
                    return f"vec{size}"
                elif element_type == "int":
                    return f"ivec{size}"
                elif element_type == "uint":
                    return f"uvec{size}"
                else:
                    return f"{element_type}{size}"
        else:
            return str(type_node)

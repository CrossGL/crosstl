"""CUDA to CrossGL Code Generator"""

import re

from .CudaAst import (
    ArrayAccessNode,
    AssignmentNode,
    CastNode,
    EnumNode,
    FunctionCallNode,
    InitializerListNode,
    KernelLaunchNode,
    MemberAccessNode,
    TypeAliasNode,
    UnaryOpNode,
    VariableNode,
)


class CudaToCrossGLConverter:
    """Serialize CUDA backend AST nodes back into CrossGL source."""

    CUDA_RUNTIME_ERROR_WRAPPER_NAMES = {
        "CHECK_CUDA",
        "CUDA_CHECK",
        "checkCuda",
        "checkCudaErrors",
        "gpuErrchk",
    }

    VECTOR_TYPE_MAPPING = {
        "half2": "vec2<f16>",
        "float2": "vec2<f32>",
        "float3": "vec3<f32>",
        "float4": "vec4<f32>",
        "double2": "vec2<f64>",
        "double3": "vec3<f64>",
        "double4": "vec4<f64>",
        "int2": "vec2<i32>",
        "int3": "vec3<i32>",
        "int4": "vec4<i32>",
        "uint2": "vec2<u32>",
        "uint3": "vec3<u32>",
        "uint4": "vec4<u32>",
        "char2": "vec2<i8>",
        "char3": "vec3<i8>",
        "char4": "vec4<i8>",
        "uchar2": "vec2<u8>",
        "uchar3": "vec3<u8>",
        "uchar4": "vec4<u8>",
        "short2": "vec2<i16>",
        "short3": "vec3<i16>",
        "short4": "vec4<i16>",
        "ushort2": "vec2<u16>",
        "ushort3": "vec3<u16>",
        "ushort4": "vec4<u16>",
        "long2": "vec2<i64>",
        "long3": "vec3<i64>",
        "long4": "vec4<i64>",
        "ulong2": "vec2<u64>",
        "ulong3": "vec3<u64>",
        "ulong4": "vec4<u64>",
        "longlong2": "vec2<i64>",
        "longlong3": "vec3<i64>",
        "longlong4": "vec4<i64>",
        "ulonglong2": "vec2<u64>",
        "ulonglong3": "vec3<u64>",
        "ulonglong4": "vec4<u64>",
    }
    VECTOR1_TYPE_MAPPING = {
        "char1": "i8",
        "uchar1": "u8",
        "short1": "i16",
        "ushort1": "u16",
        "int1": "i32",
        "uint1": "u32",
        "long1": "i64",
        "ulong1": "u64",
        "longlong1": "i64",
        "ulonglong1": "u64",
        "float1": "f32",
        "double1": "f64",
    }
    VECTOR_CONSTRUCTOR_MAPPING = {
        **VECTOR1_TYPE_MAPPING,
        **VECTOR_TYPE_MAPPING,
        **{
            f"make_{name}": mapped
            for name, mapped in {
                **VECTOR1_TYPE_MAPPING,
                **VECTOR_TYPE_MAPPING,
            }.items()
        },
    }
    CUDA_TEXTURE_TYPE_MAPPING = {
        "1": "sampler1D",
        "2": "sampler2D",
        "3": "sampler3D",
        "cudaTextureType1D": "sampler1D",
        "cudaTextureType1DLayered": "sampler1DArray",
        "cudaTextureType2D": "sampler2D",
        "cudaTextureType2DLayered": "sampler2DArray",
        "cudaTextureType3D": "sampler3D",
        "cudaTextureTypeCubemap": "samplerCube",
        "cudaTextureTypeCubemapLayered": "samplerCubeArray",
    }
    CUDA_SURFACE_TYPE_MAPPING = {
        "1": "image1D",
        "2": "image2D",
        "3": "image3D",
        "cudaSurfaceType1D": "image1D",
        "cudaSurfaceType1DLayered": "image1DArray",
        "cudaSurfaceType2D": "image2D",
        "cudaSurfaceType2DLayered": "image2DArray",
        "cudaSurfaceType3D": "image3D",
        "cudaSurfaceTypeCubemap": "imageCube",
        "cudaSurfaceTypeCubemapLayered": "imageCubeArray",
    }
    CUDA_TEXTURE_CALL_TYPE_HINTS = {
        "tex1D": "sampler1D",
        "tex1Dfetch": "sampler1D",
        "tex1DLod": "sampler1D",
        "tex1DGrad": "sampler1D",
        "tex2D": "sampler2D",
        "tex2DLod": "sampler2D",
        "tex2DGrad": "sampler2D",
        "tex2Dgather": "sampler2D",
        "tex3D": "sampler3D",
        "tex3DLod": "sampler3D",
        "tex3DGrad": "sampler3D",
        "texCubemap": "samplerCube",
        "texCubemapLod": "samplerCube",
        "texCubemapGrad": "samplerCube",
        "tex1DLayered": "sampler1DArray",
        "tex1DLayeredLod": "sampler1DArray",
        "tex1DLayeredGrad": "sampler1DArray",
        "tex2DLayered": "sampler2DArray",
        "tex2DLayeredLod": "sampler2DArray",
        "tex2DLayeredGrad": "sampler2DArray",
        "texCubemapLayered": "samplerCubeArray",
        "texCubemapLayeredLod": "samplerCubeArray",
        "texCubemapLayeredGrad": "samplerCubeArray",
    }
    CUDA_SURFACE_CALL_TYPE_HINTS = {
        "surf1Dread": "image1D",
        "surf1Dwrite": "image1D",
        "surf1DLayeredread": "image1DArray",
        "surf1DLayeredwrite": "image1DArray",
        "surf2Dread": "image2D",
        "surf2Dwrite": "image2D",
        "surf3Dread": "image3D",
        "surf3Dwrite": "image3D",
        "surf2DLayeredread": "image2DArray",
        "surf2DLayeredwrite": "image2DArray",
        "surfCubemapread": "imageCube",
        "surfCubemapwrite": "imageCube",
        "surfCubemapLayeredread": "imageCubeArray",
        "surfCubemapLayeredwrite": "imageCubeArray",
    }
    CUDA_SURFACE_VALUE_BYTE_SIZES = {
        "char": 1,
        "signed char": 1,
        "unsigned char": 1,
        "int8_t": 1,
        "uint8_t": 1,
        "uchar": 1,
        "short": 2,
        "unsigned short": 2,
        "int16_t": 2,
        "uint16_t": 2,
        "half": 2,
        "__half": 2,
        "__nv_bfloat16": 2,
        "int": 4,
        "signed int": 4,
        "uint": 4,
        "unsigned int": 4,
        "int32_t": 4,
        "uint32_t": 4,
        "float": 4,
        "long long": 8,
        "signed long long": 8,
        "unsigned long long": 8,
        "int64_t": 8,
        "uint64_t": 8,
        "double": 8,
        "char1": 1,
        "uchar1": 1,
        "short1": 2,
        "ushort1": 2,
        "int1": 4,
        "uint1": 4,
        "float1": 4,
        "double1": 8,
        "char2": 2,
        "uchar2": 2,
        "short2": 4,
        "ushort2": 4,
        "half2": 4,
        "__half2": 4,
        "__nv_bfloat162": 4,
        "int2": 8,
        "uint2": 8,
        "float2": 8,
        "double2": 16,
        "char3": 3,
        "uchar3": 3,
        "short3": 6,
        "ushort3": 6,
        "int3": 12,
        "uint3": 12,
        "float3": 12,
        "double3": 24,
        "char4": 4,
        "uchar4": 4,
        "short4": 8,
        "ushort4": 8,
        "int4": 16,
        "uint4": 16,
        "float4": 16,
        "double4": 32,
    }
    CUDA_DEVICE_GRAPH_LAUNCH_MODES = {
        "cudaStreamGraphFireAndForget": "fire-and-forget",
        "cudaStreamGraphTailLaunch": "tail",
        "cudaStreamGraphFireAndForgetAsSibling": "fire-and-forget sibling",
    }
    CUDA_INTRINSIC_OVERRIDE_NAMES = {"__usad4"}
    CROSSGL_RESERVED_IDENTIFIERS = {"buffer", "in", "var"}

    def __init__(self):
        self.indent_level = 0
        self.output = []
        self.identifier_name_scopes = [{}]
        self.packed_argument_scopes = []
        self.unique_ptr_scopes = [set()]
        self.type_alias_scopes = [{}]
        self.vector1_name_scopes = [{}]
        self.user_function_names = set()
        self.global_resource_object_type_hints = {}
        self.struct_resource_member_hints = {}
        self.resource_object_hint_scopes = []
        self.cooperative_group_scopes = [{}]
        self.cuda_async_sync_scopes = [{}]
        self.variable_type_scopes = [{}]
        self.warp_mask_value_scopes = [{}]
        self.namespace_aliases = {}
        self.cuda_record_names = set()
        self.generated_matrix_helper_types = set()
        self.global_resource_binding_count = 0
        self.anonymous_enum_count = 0

    def generate(self, ast_node):
        self.output = []
        self.indent_level = 0
        self.identifier_name_scopes = [{}]
        self.packed_argument_scopes = []
        self.unique_ptr_scopes = [set()]
        self.type_alias_scopes = [{}]
        self.vector1_name_scopes = [{}]
        self.generated_matrix_helper_types = self.collect_generated_matrix_helper_types(
            ast_node
        )
        self.user_function_names = self.collect_user_function_names(ast_node)
        self.global_resource_object_type_hints = (
            self.collect_global_resource_object_type_hints(ast_node)
        )
        self.struct_resource_member_hints = self.collect_struct_resource_member_hints(
            ast_node
        )
        self.cuda_record_names = self.collect_cuda_record_names(ast_node)
        self.resource_object_hint_scopes = []
        self.cooperative_group_scopes = [{}]
        self.cuda_async_sync_scopes = [{}]
        self.variable_type_scopes = [{}]
        self.warp_mask_value_scopes = [{}]
        self.namespace_aliases = getattr(ast_node, "namespace_aliases", {}) or {}
        self.global_resource_binding_count = 0
        self.anonymous_enum_count = 0
        self.visit(ast_node)
        return "\n".join(self.output)

    def visit(self, node):
        method_name = f"visit_{type(node).__name__}"
        visitor = getattr(self, method_name, self.generic_visit)
        return visitor(node)

    def generic_visit(self, node):
        if isinstance(node, str):
            if self.is_cuda_numeric_literal_text(node):
                return node.replace("'", "")
            if self.is_cuda_char_literal_text(node):
                return self.convert_cuda_char_literal_to_crossgl(node)
            if self.is_cuda_string_literal_text(node):
                return self.convert_cuda_string_literal_to_crossgl(node)
            if self.is_cuda_qualified_identifier_text(node):
                return self.sanitize_crossgl_type_identifier(node)
            return self.output_identifier_name(node)
        elif isinstance(node, list):
            return [self.visit(item) for item in node]
        else:
            return str(node)

    def is_cuda_numeric_literal_text(self, value):
        text = str(value)
        return (
            "'" in text and bool(text) and (text[0].isdigit() or text.startswith("."))
        )

    def is_cuda_char_literal_text(self, value):
        return re.fullmatch(r"(?:u8|u|U|L)?'(?:[^'\\\n]|\\.)+'", str(value)) is not None

    def convert_cuda_char_literal_to_crossgl(self, value):
        text = str(value)
        match = re.fullmatch(r"(?:u8|u|U|L)?'(.*)'", text)
        if match is None:
            return text

        content = match.group(1)
        hex_match = re.fullmatch(r"\\x([0-9A-Fa-f]+)", content)
        if hex_match is not None:
            return str(int(hex_match.group(1), 16))

        octal_match = re.fullmatch(r"\\([0-7]+)", content)
        if octal_match is not None:
            return str(int(octal_match.group(1), 8))

        return f"'{content}'"

    def is_cuda_string_literal_text(self, value):
        return re.fullmatch(r'(?:u8|u|U|L)?"(?:[^"\\]|\\.)*"', str(value)) is not None

    def convert_cuda_string_literal_to_crossgl(self, value):
        match = re.fullmatch(r'(?:u8|u|U|L)?(".*")', str(value))
        return match.group(1) if match is not None else str(value)

    def is_cuda_qualified_identifier_text(self, value):
        text = str(value)
        if not text or text[0] in "\"'":
            return False
        return "::" in text and (("<" in text and ">" in text) or "(" in text)

    def emit(self, code):
        if code.strip():
            self.output.append("    " * self.indent_level + code)
        else:
            self.output.append("")

    def push_identifier_name_scope(self):
        self.identifier_name_scopes.append({})

    def pop_identifier_name_scope(self):
        if len(self.identifier_name_scopes) > 1:
            self.identifier_name_scopes.pop()

    def register_identifier_name(self, name):
        if not self.is_simple_identifier(name):
            return name
        output_name = self.sanitize_identifier_name(name)
        if not self.identifier_name_scopes:
            self.identifier_name_scopes.append({})
        self.identifier_name_scopes[-1][name] = output_name
        return output_name

    def output_identifier_name(self, name):
        if not self.is_simple_identifier(name):
            return name
        for scope in reversed(self.identifier_name_scopes):
            if name in scope:
                return scope[name]
        return name

    def sanitize_identifier_name(self, name):
        if name in self.CROSSGL_RESERVED_IDENTIFIERS:
            return f"{name}_"
        return name

    def format_crossgl_array_extent(self, size):
        size = str(size).strip()
        if not size:
            return size
        simple_extent_pattern = r"[A-Za-z_][A-Za-z0-9_]*|(?:0[xX][0-9A-Fa-f]+|\d+)[uU]?"
        if re.fullmatch(simple_extent_pattern, size):
            return size

        replacements = {
            "*": "_mul_",
            "/": "_div_",
            "%": "_mod_",
            "+": "_plus_",
            "-": "_minus_",
        }
        extent = size
        for operator, word in replacements.items():
            extent = extent.replace(operator, word)
        extent = re.sub(r"[^A-Za-z0-9_]+", "_", extent).strip("_")
        extent = re.sub(r"_+", "_", extent)
        if not extent:
            extent = "expr"
        if extent[0].isdigit():
            extent = f"cuda_array_extent_{extent}"
        return self.sanitize_identifier_name(extent)

    def is_simple_identifier(self, name):
        return isinstance(name, str) and name.isidentifier()

    def sanitize_crossgl_type_identifier(self, name):
        parts = []
        for char in str(name):
            if char.isalnum() or char == "_":
                parts.append(char)
            elif char == ":":
                if not parts or parts[-1] != "_":
                    parts.append("_")
            elif not parts or parts[-1] != "_":
                parts.append("_")

        sanitized = "".join(parts).strip("_") or "anonymous"
        if sanitized[0].isdigit():
            sanitized = f"type_{sanitized}"
        return self.sanitize_identifier_name(sanitized)

    def convert_unknown_cuda_template_type_to_crossgl(self, cuda_type):
        base_name, template_args = self.parse_cpp_template(cuda_type)
        if not template_args:
            return None

        converted_parts = [self.sanitize_crossgl_type_identifier(base_name)]
        for arg in template_args:
            converted_arg = self.convert_cuda_type_to_crossgl(arg)
            converted_parts.append(self.sanitize_crossgl_type_identifier(converted_arg))
        return "_".join(part for part in converted_parts if part)

    def convert_cuda_parameter_pack_type_to_crossgl(self, cuda_type):
        if "..." not in str(cuda_type):
            return None

        base_type = str(cuda_type).replace("...", "").strip()
        if not base_type:
            return "parameter_pack"

        converted_base_type = self.convert_cuda_type_to_crossgl(base_type)
        return f"{self.sanitize_crossgl_type_identifier(converted_base_type)}_pack"

    def convert_cuda_elaborated_type_to_crossgl(self, cuda_type):
        normalized_type = str(cuda_type).strip()
        if normalized_type.startswith("enum "):
            type_name = normalized_type[len("enum ") :].strip()
            return self.convert_cuda_type_to_crossgl(type_name)
        for prefix in ("class ", "struct ", "union "):
            if normalized_type.startswith(prefix):
                type_name = normalized_type[len(prefix) :].strip()
                if "*" in type_name:
                    return self.convert_cuda_pointer_type(type_name)
                return self.sanitize_crossgl_type_identifier(type_name)
        return None

    def convert_cuda_scoped_type_to_crossgl(self, cuda_type):
        normalized_type = str(cuda_type).strip()
        if "::" not in normalized_type or ">::" not in normalized_type:
            return None

        return self.sanitize_crossgl_type_identifier(normalized_type)

    def convert_cuda_record_name_to_crossgl(self, name):
        template_record_name = self.convert_unknown_cuda_template_type_to_crossgl(name)
        if template_record_name is not None:
            return template_record_name
        return self.sanitize_crossgl_type_identifier(name)

    def collect_cuda_record_names(self, node):
        return {
            struct.name
            for struct in getattr(node, "structs", []) or []
            if hasattr(struct, "name")
        }

    def convert_cuda_function_name_to_crossgl(self, name):
        operator_name = self.convert_cuda_operator_function_name_to_crossgl(name)
        if operator_name is not None:
            return operator_name
        if self.is_simple_identifier(name):
            return self.sanitize_identifier_name(name)
        return self.sanitize_crossgl_type_identifier(name)

    def convert_cuda_operator_function_name_to_crossgl(self, name):
        if not isinstance(name, str) or not name.startswith("operator"):
            return None

        operator_symbol = name[len("operator") :].strip()
        operator_mapping = {
            "==": "equal",
            "!=": "not_equal",
            "<=>": "three_way",
            "<": "less",
            ">": "greater",
            "<=": "less_equal",
            ">=": "greater_equal",
            "+": "add",
            "-": "subtract",
            "*": "multiply",
            "/": "divide",
            "%": "modulo",
            "&&": "logical_and",
            "||": "logical_or",
            "!": "logical_not",
            "&": "bitwise_and",
            "|": "bitwise_or",
            "^": "bitwise_xor",
            "~": "bitwise_not",
            "<<": "shift_left",
            ">>": "shift_right",
            "[]": "subscript",
            "()": "call",
        }
        return f"operator_{operator_mapping.get(operator_symbol, 'overload')}"

    def crossgl_parameter_name(self, param, index):
        if getattr(param, "name", ""):
            return self.register_identifier_name(param.name)
        return f"_unused_param_{index}"

    def collect_user_function_names(self, node):
        names = set()

        def collect(current):
            if current is None:
                return
            if isinstance(current, list):
                for item in current:
                    collect(item)
                return

            name = getattr(current, "name", None)
            body = getattr(current, "body", None)
            if name is not None and body is not None:
                if not self.is_generated_matrix_helper_function(current):
                    names.add(name)

            for function in getattr(current, "functions", []):
                collect(function)
            for kernel in getattr(current, "kernels", []):
                collect(kernel)

        collect(node)
        names.discard(None)
        return names

    def collect_generated_matrix_helper_types(self, node):
        return {
            struct.name
            for struct in getattr(node, "structs", []) or []
            if self.is_generated_matrix_helper_struct(struct)
        }

    def native_matrix_helper_dimensions(self, type_name):
        match = re.fullmatch(r"(float|double)([234])x([234])", str(type_name))
        if match is None:
            return None
        scalar_type, columns, rows = match.groups()
        return scalar_type, int(columns), int(rows)

    def convert_native_matrix_helper_name_to_crossgl(self, type_name):
        if type_name not in self.generated_matrix_helper_types:
            return None

        dimensions = self.native_matrix_helper_dimensions(type_name)
        if dimensions is None:
            return None

        scalar_type, columns, rows = dimensions
        prefix = "dmat" if scalar_type == "double" else "mat"
        suffix = str(columns) if columns == rows else f"{columns}x{rows}"
        return f"{prefix}{suffix}"

    def is_generated_matrix_helper_struct(self, node):
        dimensions = self.native_matrix_helper_dimensions(getattr(node, "name", ""))
        if dimensions is None:
            return False

        scalar_type, columns, rows = dimensions
        members = getattr(node, "members", []) or []
        member_by_name = {
            getattr(member, "name", None): member
            for member in members
            if getattr(member, "name", None)
        }

        matrix_values = member_by_name.get("m")
        column_count = member_by_name.get("CGL_COLUMNS")
        row_count = member_by_name.get("CGL_ROWS")
        if matrix_values is None or column_count is None or row_count is None:
            return False

        return (
            getattr(matrix_values, "vtype", None) == f"{scalar_type}[{columns * rows}]"
            and getattr(column_count, "vtype", None) == "const int"
            and str(getattr(column_count, "value", "")).strip() == str(columns)
            and "static" in (getattr(column_count, "qualifiers", []) or [])
            and getattr(row_count, "vtype", None) == "const int"
            and str(getattr(row_count, "value", "")).strip() == str(rows)
            and "static" in (getattr(row_count, "qualifiers", []) or [])
        )

    def is_generated_matrix_helper_function(self, node):
        if getattr(node, "name", None) not in {"operator*", "transpose", "inverse"}:
            return False
        if not self.function_references_generated_matrix_helper_type(node):
            return False
        qualifiers = set(getattr(node, "qualifiers", []) or [])
        return {"__host__", "__device__", "inline"}.issubset(qualifiers)

    def function_references_generated_matrix_helper_type(self, node):
        candidate_types = [getattr(node, "return_type", "")]
        candidate_types.extend(getattr(param, "vtype", "") for param in node.params)
        return any(
            self.strip_type_qualifiers(candidate_type).strip()
            in self.generated_matrix_helper_types
            for candidate_type in candidate_types
        )

    def add_resource_object_type_hint(self, hints, name, resource_type):
        if not name or not resource_type:
            return
        if name in hints and hints[name] != resource_type:
            hints[name] = None
            return
        hints[name] = resource_type

    def collect_resource_object_type_hints(self, node, declared_names=None):
        hints = {}

        def collect(current):
            if current is None:
                return
            if isinstance(current, (list, tuple)):
                for item in current:
                    collect(item)
                return
            if isinstance(current, dict):
                for item in current.values():
                    collect(item)
                return
            if isinstance(current, (str, int, float, bool)):
                return
            if not hasattr(current, "__dict__"):
                return

            if isinstance(current, FunctionCallNode):
                hint = self.get_resource_object_call_hint(current.name)
                if hint is not None:
                    arg_index, resource_type = hint
                    if len(current.args) > arg_index:
                        self.add_resource_object_type_hint(
                            hints,
                            self.get_resource_object_expression_name(
                                current.args[arg_index]
                            ),
                            resource_type,
                        )

            for value in vars(current).values():
                collect(value)

        collect(node)
        for name in declared_names or []:
            hints.setdefault(name, None)
        return hints

    def collect_global_resource_object_type_hints(self, node):
        hints = {}
        global_names = self.collect_global_declared_variable_names(node)
        for function in getattr(node, "functions", []):
            self.collect_global_resource_object_type_hints_from_function(
                function, global_names, hints
            )
        for kernel in getattr(node, "kernels", []):
            self.collect_global_resource_object_type_hints_from_function(
                kernel, global_names, hints
            )
        return hints

    def collect_global_resource_object_type_hints_from_function(
        self, node, global_names, hints
    ):
        local_names = self.collect_declared_variable_names(node)

        def collect(current):
            if current is None:
                return
            if isinstance(current, (list, tuple)):
                for item in current:
                    collect(item)
                return
            if isinstance(current, dict):
                for item in current.values():
                    collect(item)
                return
            if isinstance(current, (str, int, float, bool)):
                return
            if not hasattr(current, "__dict__"):
                return

            if isinstance(current, FunctionCallNode):
                hint = self.get_resource_object_call_hint(current.name)
                if hint is not None:
                    arg_index, resource_type = hint
                    if len(current.args) > arg_index:
                        name = self.get_resource_object_expression_name(
                            current.args[arg_index]
                        )
                        if name in global_names and name not in local_names:
                            self.add_resource_object_type_hint(
                                hints, name, resource_type
                            )

            for value in vars(current).values():
                collect(value)

        collect(getattr(node, "body", []))

    def collect_global_declared_variable_names(self, node):
        return {
            var.name
            for var in getattr(node, "global_variables", [])
            if isinstance(var, VariableNode)
        }

    def collect_global_declared_variable_types(self, node):
        return {
            var.name: var.vtype
            for var in getattr(node, "global_variables", [])
            if isinstance(var, VariableNode)
        }

    def collect_struct_member_types(self, node):
        return {
            struct.name: {
                member.name: member.vtype
                for member in getattr(struct, "members", [])
                if isinstance(member, VariableNode)
            }
            for struct in getattr(node, "structs", [])
            if struct.name
        }

    def collect_declared_variable_names(self, node):
        names = {
            param.name
            for param in getattr(node, "params", [])
            if isinstance(param, VariableNode)
        }

        def collect(current):
            if current is None:
                return
            if isinstance(current, (list, tuple)):
                for item in current:
                    collect(item)
                return
            if isinstance(current, dict):
                for item in current.values():
                    collect(item)
                return
            if isinstance(current, (str, int, float, bool)):
                return
            if not hasattr(current, "__dict__"):
                return

            if isinstance(current, VariableNode):
                names.add(current.name)

            for value in vars(current).values():
                collect(value)

        collect(getattr(node, "body", []))
        return names

    def collect_declared_variable_types(self, node):
        types = {
            param.name: param.vtype
            for param in getattr(node, "params", [])
            if isinstance(param, VariableNode)
        }

        for current in self.walk_ast_values(getattr(node, "body", [])):
            if isinstance(current, VariableNode):
                types[current.name] = current.vtype
        return types

    def walk_ast_values(self, root):
        visited = set()

        def walk(current):
            if current is None:
                return
            if isinstance(current, (list, tuple)):
                for item in current:
                    yield from walk(item)
                return
            if isinstance(current, dict):
                for item in current.values():
                    yield from walk(item)
                return
            if isinstance(current, (str, int, float, bool)):
                return
            if not hasattr(current, "__dict__"):
                return

            current_id = id(current)
            if current_id in visited:
                return
            visited.add(current_id)
            yield current

            for value in vars(current).values():
                yield from walk(value)

        yield from walk(root)

    def collect_struct_resource_member_hints(self, node):
        struct_member_types = self.collect_struct_member_types(node)
        struct_names = set(struct_member_types)
        if not struct_names:
            return {}

        hints = {}
        global_variable_types = self.collect_global_declared_variable_types(node)
        for function in [
            *getattr(node, "functions", []),
            *getattr(node, "kernels", []),
        ]:
            variable_types = dict(global_variable_types)
            variable_types.update(self.collect_declared_variable_types(function))
            for current in self.walk_ast_values(getattr(function, "body", [])):
                if not isinstance(current, FunctionCallNode):
                    continue
                call_hint = self.get_resource_object_call_hint(current.name)
                if call_hint is None:
                    continue
                arg_index, resource_type = call_hint
                if len(current.args) <= arg_index:
                    continue

                member_access = self.resource_member_access_target(
                    current.args[arg_index]
                )
                if member_access is None:
                    continue

                struct_name = self.struct_type_for_resource_member_object(
                    member_access.object,
                    variable_types,
                    struct_names,
                    struct_member_types,
                )
                if struct_name is None:
                    continue
                self.add_struct_resource_member_hint(
                    hints, struct_name, member_access.member, resource_type
                )

        return hints

    def add_struct_resource_member_hint(
        self, hints, struct_name, member_name, resource_type
    ):
        key = (struct_name, member_name)
        if key in hints and hints[key] != resource_type:
            hints[key] = None
            return
        hints[key] = resource_type

    def resource_member_access_target(self, expression):
        if isinstance(expression, MemberAccessNode):
            return expression
        if isinstance(expression, ArrayAccessNode):
            return self.resource_member_access_target(expression.array)
        if isinstance(expression, CastNode):
            return self.resource_member_access_target(expression.expression)
        if isinstance(expression, UnaryOpNode):
            return self.resource_member_access_target(expression.operand)
        return None

    def struct_type_for_resource_member_object(
        self, expression, variable_types, struct_names, struct_member_types
    ):
        if isinstance(expression, str):
            return self.normalized_struct_type_name(
                variable_types.get(expression), struct_names
            )
        if isinstance(expression, ArrayAccessNode):
            return self.struct_type_for_resource_member_object(
                expression.array, variable_types, struct_names, struct_member_types
            )
        if isinstance(expression, CastNode):
            cast_struct = self.normalized_struct_type_name(
                expression.target_type, struct_names
            )
            if cast_struct is not None:
                return cast_struct
            return self.struct_type_for_resource_member_object(
                expression.expression, variable_types, struct_names, struct_member_types
            )
        if isinstance(expression, UnaryOpNode):
            return self.struct_type_for_resource_member_object(
                expression.operand, variable_types, struct_names, struct_member_types
            )
        if isinstance(expression, MemberAccessNode):
            owner_struct = self.struct_type_for_resource_member_object(
                expression.object,
                variable_types,
                struct_names,
                struct_member_types,
            )
            if owner_struct is None:
                return None
            member_type = struct_member_types.get(owner_struct, {}).get(
                expression.member
            )
            return self.normalized_struct_type_name(member_type, struct_names)
        return None

    def normalized_struct_type_name(self, type_name, struct_names):
        if not type_name:
            return None
        type_name = self.strip_type_qualifiers(type_name)
        type_name = type_name.split("[", 1)[0].replace("*", "").strip()
        return type_name if type_name in struct_names else None

    def get_resource_object_call_hint(self, function_name):
        base_name, _ = self.parse_cpp_template(function_name)
        if self.is_user_defined_function(base_name):
            return None
        if base_name in self.CUDA_TEXTURE_CALL_TYPE_HINTS:
            return 0, self.CUDA_TEXTURE_CALL_TYPE_HINTS[base_name]
        if base_name in self.CUDA_SURFACE_CALL_TYPE_HINTS:
            if base_name.endswith("write"):
                return 1, self.CUDA_SURFACE_CALL_TYPE_HINTS[base_name]
            return 0, self.CUDA_SURFACE_CALL_TYPE_HINTS[base_name]
        return None

    def get_resource_object_expression_name(self, expression):
        if isinstance(expression, str):
            return expression
        if isinstance(expression, ArrayAccessNode):
            return self.get_resource_object_expression_name(expression.array)
        if isinstance(expression, CastNode):
            return self.get_resource_object_expression_name(expression.expression)
        if isinstance(expression, UnaryOpNode):
            return self.get_resource_object_expression_name(expression.operand)
        return None

    def is_user_defined_function(self, func_name):
        return isinstance(func_name, str) and func_name in self.user_function_names

    def is_cuda_intrinsic_override_function(self, func_name):
        if not isinstance(func_name, str):
            return False
        normalized_name = func_name[2:] if func_name.startswith("::") else func_name
        return normalized_name in self.CUDA_INTRINSIC_OVERRIDE_NAMES

    def push_resource_object_hint_scope(self, hints):
        self.resource_object_hint_scopes.append(hints)

    def pop_resource_object_hint_scope(self):
        if self.resource_object_hint_scopes:
            self.resource_object_hint_scopes.pop()

    def lookup_resource_object_type_hint(self, name):
        for scope in reversed(self.resource_object_hint_scopes):
            if name in scope:
                return scope[name]
        return self.global_resource_object_type_hints.get(name)

    def emit_statement(self, stmt):
        if stmt is None:
            return
        if isinstance(stmt, list):
            for item in stmt:
                self.emit_statement(item)
            return

        if self.emit_cuda_runtime_call_statement(stmt):
            return
        if self.emit_cuda_output_parameter_call_statement(stmt):
            return

        result = self.visit(stmt)
        if isinstance(result, str) and result.strip():
            self.emit(f"{result};")

    def emit_cuda_runtime_call_statement(self, stmt):
        if not isinstance(stmt, FunctionCallNode):
            return False
        if self.is_user_defined_function(stmt.name):
            return False

        runtime_wrapper = self.format_cuda_runtime_wrapper_expression(stmt)
        if runtime_wrapper is not None:
            comments, _ = runtime_wrapper
            for comment in comments:
                self.emit(comment)
            return True

        runtime_value = self.format_cuda_runtime_value_expression(stmt)
        if runtime_value is not None:
            comments, _ = runtime_value
            for comment in comments:
                self.emit(comment)
            return True

        comments = self.format_cuda_runtime_call(stmt)
        if comments is None:
            return False

        for comment in comments:
            self.emit(comment)
        return True

    def emit_cuda_output_parameter_call_statement(self, stmt):
        if not isinstance(stmt, FunctionCallNode):
            return False

        raw_name = stmt.name if isinstance(stmt.name, str) else self.visit(stmt.name)
        if self.is_user_defined_function(raw_name):
            return False

        statements = self.format_cuda_output_parameter_call_statements(
            raw_name, stmt.args
        )
        if statements is None:
            return False

        for statement in statements:
            self.emit(statement)
        return True

    def format_cuda_output_parameter_call_statements(self, function_name, args):
        if isinstance(function_name, str) and function_name.startswith("::"):
            function_name = function_name[2:]

        if function_name == "__sincosf" and len(args) == 3:
            value = self.visit(args[0])
            sin_target = self.format_cuda_output_pointer_target(args[1])
            cos_target = self.format_cuda_output_pointer_target(args[2])
            if sin_target is None or cos_target is None:
                return None
            return [
                f"{sin_target} = sin({value});",
                f"{cos_target} = cos({value});",
            ]

        return None

    def format_cuda_output_pointer_target(self, arg):
        if isinstance(arg, CastNode):
            return self.format_cuda_output_pointer_target(arg.expression)
        if isinstance(arg, UnaryOpNode) and arg.op == "&":
            return self.visit(arg.operand)

        target = self.visit(arg)
        if not target:
            return None
        return f"(*{target})"

    def format_cuda_runtime_status_expression(self, value):
        if not isinstance(value, FunctionCallNode):
            return None
        if self.is_user_defined_function(value.name):
            return None

        wrapped_runtime = self.format_cuda_runtime_wrapper_expression(value)
        if wrapped_runtime is not None:
            return wrapped_runtime

        comments = self.format_cuda_runtime_call(value)
        if comments is None:
            return None

        return comments, self.format_cuda_success_literal(value.name)

    def format_cuda_success_literal(self, name):
        if isinstance(name, str) and name.startswith("cu") and len(name) > 2:
            if name[2].isupper():
                return "CUDA_SUCCESS"
        return "cudaSuccess"

    def format_cuda_runtime_expression(self, value):
        if isinstance(value, KernelLaunchNode):
            return self.format_kernel_launch_comments(value), "cudaSuccess"

        runtime_value = self.format_cuda_runtime_value_expression(value)
        if runtime_value is not None:
            return runtime_value
        return self.format_cuda_runtime_status_expression(value)

    def format_cuda_runtime_value_expression(self, value):
        if not isinstance(value, FunctionCallNode):
            return None
        if self.is_user_defined_function(value.name):
            return None

        name = value.name if isinstance(value.name, str) else self.visit(value.name)
        if name == "cudaGetCurrentGraphExec" and not value.args:
            return ["// CUDA device graph get current exec"], "0"
        return None

    def format_cuda_runtime_wrapper_expression(self, value):
        if not isinstance(value, FunctionCallNode):
            return None
        if self.is_user_defined_function(value.name):
            return None
        if value.name not in self.CUDA_RUNTIME_ERROR_WRAPPER_NAMES:
            return None
        if len(getattr(value, "args", []) or []) != 1:
            return None
        return self.format_cuda_runtime_expression(value.args[0])

    def format_cuda_runtime_inline_expression(self, value):
        runtime_expression = self.format_cuda_runtime_expression(value)
        if runtime_expression is None:
            return None

        comments, fallback = runtime_expression
        if not comments:
            return fallback

        comment = comments[-1]
        if comment.startswith("// "):
            comment = comment[3:]
        return f"(/* {comment} */ {fallback})"

    def format_cuda_runtime_call(self, node):
        args = [self.visit(arg) for arg in node.args]
        name = node.name

        if name in {"cudaMalloc", "cudaMallocManaged", "cudaMallocHost"}:
            if len(node.args) >= 2:
                target = self.format_runtime_pointer_target(node.args[0])
                size = self.visit(node.args[1])
                return [f"// CUDA memory allocate: {target}, bytes: {size}"]
        elif name in {"cudaFree", "cudaFreeHost"}:
            if args:
                return [f"// CUDA memory free: {args[0]}"]
        elif name in {"cudaMemcpy", "cudaMemcpyAsync"}:
            if len(args) >= 4:
                comment = (
                    f"// CUDA memory copy: {args[1]} -> {args[0]}, "
                    f"bytes: {args[2]}, kind: {args[3]}"
                )
                if len(args) >= 5:
                    comment += f", stream: {args[4]}"
                return [comment]
        elif name in {"cudaMemset", "cudaMemsetAsync"}:
            if len(args) >= 3:
                comment = (
                    f"// CUDA memory set: {args[0]}, value: {args[1]}, "
                    f"bytes: {args[2]}"
                )
                if len(args) >= 4:
                    comment += f", stream: {args[3]}"
                return [comment]
        elif name == "cuMemAlloc":
            if len(node.args) >= 2:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    f"// CUDA driver memory allocate: output: {output}, "
                    f"bytes: {args[1]}"
                ]
        elif name == "cuMemAllocManaged":
            if len(node.args) >= 3:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    "// CUDA driver memory allocate managed: "
                    f"output: {output}, bytes: {args[1]}, flags: {args[2]}"
                ]
        elif name in {"cuMemAllocHost", "cuMemHostAlloc"}:
            if len(node.args) >= 2:
                output = self.format_runtime_pointer_target(node.args[0])
                comment = (
                    f"// CUDA driver host memory allocate: output: {output}, "
                    f"bytes: {args[1]}"
                )
                if name == "cuMemHostAlloc" and len(args) >= 3:
                    comment += f", flags: {args[2]}"
                return [comment]
        elif name == "cuMemFree":
            if args:
                return [f"// CUDA driver memory free: {args[0]}"]
        elif name == "cuMemFreeHost":
            if args:
                return [f"// CUDA driver host memory free: {args[0]}"]
        elif name == "cuPointerGetAttribute":
            if len(node.args) >= 3:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    "// CUDA driver pointer get attribute: "
                    f"output: {output}, attribute: {args[1]}, pointer: {args[2]}"
                ]
        elif name == "cuPointerGetAttributes":
            if len(args) >= 4:
                return [
                    "// CUDA driver pointer get attributes: "
                    f"count: {args[0]}, attributes: {args[1]}, "
                    f"data: {args[2]}, pointer: {args[3]}"
                ]
        elif name == "cuMemGetAddressRange":
            if len(node.args) >= 3:
                base_output = self.format_runtime_pointer_target(node.args[0])
                size_output = self.format_runtime_pointer_target(node.args[1])
                return [
                    "// CUDA driver memory get address range: "
                    f"base output: {base_output}, size output: {size_output}, "
                    f"pointer: {args[2]}"
                ]
        elif name in {"cuMemGetInfo", "cuMemGetInfo_v2"}:
            if len(node.args) >= 2:
                free_output = self.format_runtime_pointer_target(node.args[0])
                total_output = self.format_runtime_pointer_target(node.args[1])
                suffix = " v2" if name == "cuMemGetInfo_v2" else ""
                return [
                    f"// CUDA driver memory get info{suffix}: "
                    f"free output: {free_output}, total output: {total_output}"
                ]
        elif name == "cuMemAdvise":
            if len(args) >= 4:
                return [
                    "// CUDA driver memory advise: "
                    f"pointer: {args[0]}, bytes: {args[1]}, "
                    f"advice: {args[2]}, device: {args[3]}"
                ]
        elif name == "cuMemAllocAsync":
            if len(node.args) >= 3:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    "// CUDA driver memory allocate async: "
                    f"output: {output}, bytes: {args[1]}, stream: {args[2]}"
                ]
        elif name == "cuMemFreeAsync":
            if len(args) >= 2:
                return [
                    "// CUDA driver memory free async: "
                    f"pointer: {args[0]}, stream: {args[1]}"
                ]
        elif name == "cuMemPoolCreate":
            if len(node.args) >= 2:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    "// CUDA driver memory pool create: "
                    f"output: {output}, props: {args[1]}"
                ]
        elif name == "cuMemPoolDestroy":
            if args:
                return [f"// CUDA driver memory pool destroy: pool: {args[0]}"]
        elif name == "cuMemPoolTrimTo":
            if len(args) >= 2:
                return [
                    f"// CUDA driver memory pool trim: "
                    f"pool: {args[0]}, bytes: {args[1]}"
                ]
        elif name in {"cuDeviceGetDefaultMemPool", "cuDeviceGetMemPool"}:
            if len(node.args) >= 2:
                output = self.format_runtime_pointer_target(node.args[0])
                pool_kind = (
                    "default memory pool"
                    if name == "cuDeviceGetDefaultMemPool"
                    else "memory pool"
                )
                return [
                    f"// CUDA driver device get {pool_kind}: "
                    f"output: {output}, device: {args[1]}"
                ]
        elif name == "cuDeviceSetMemPool":
            if len(args) >= 2:
                return [
                    f"// CUDA driver device set memory pool: "
                    f"device: {args[0]}, pool: {args[1]}"
                ]
        elif name in {"cuMemPoolSetAttribute", "cuMemPoolGetAttribute"}:
            if len(node.args) >= 3:
                value = self.format_runtime_pointer_target(node.args[2])
                action = "set" if name == "cuMemPoolSetAttribute" else "get"
                value_label = "value" if action == "set" else "output"
                return [
                    f"// CUDA driver memory pool {action} attribute: "
                    f"pool: {args[0]}, attribute: {args[1]}, "
                    f"{value_label}: {value}"
                ]
        elif name == "cuMemPoolSetAccess":
            if len(args) >= 3:
                return [
                    f"// CUDA driver memory pool set access: "
                    f"pool: {args[0]}, descriptors: {args[1]}, count: {args[2]}"
                ]
        elif name == "cuMemPoolGetAccess":
            if len(node.args) >= 3:
                flags_output = self.format_runtime_pointer_target(node.args[0])
                return [
                    f"// CUDA driver memory pool get access: "
                    f"flags output: {flags_output}, pool: {args[1]}, "
                    f"location: {args[2]}"
                ]
        elif name == "cuMemPoolExportToShareableHandle":
            if len(node.args) >= 4:
                handle_output = self.format_runtime_pointer_target(node.args[0])
                return [
                    f"// CUDA driver memory pool export shareable handle: "
                    f"handle output: {handle_output}, pool: {args[1]}, "
                    f"handle type: {args[2]}, flags: {args[3]}"
                ]
        elif name == "cuMemPoolImportFromShareableHandle":
            if len(node.args) >= 4:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    f"// CUDA driver memory pool import shareable handle: "
                    f"output: {output}, handle: {args[1]}, "
                    f"handle type: {args[2]}, flags: {args[3]}"
                ]
        elif name == "cuMemPoolExportPointer":
            if len(node.args) >= 2:
                share_data_output = self.format_runtime_pointer_target(node.args[0])
                return [
                    f"// CUDA driver memory pool export pointer: "
                    f"share data output: {share_data_output}, pointer: {args[1]}"
                ]
        elif name == "cuMemPoolImportPointer":
            if len(node.args) >= 3:
                pointer_output = self.format_runtime_pointer_target(node.args[0])
                share_data = self.format_runtime_pointer_target(node.args[2])
                return [
                    f"// CUDA driver memory pool import pointer: "
                    f"pointer output: {pointer_output}, pool: {args[1]}, "
                    f"share data: {share_data}"
                ]
        elif name == "cuMemAddressReserve":
            if len(node.args) >= 5:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    f"// CUDA driver virtual memory reserve: "
                    f"output: {output}, bytes: {args[1]}, "
                    f"alignment: {args[2]}, address: {args[3]}, "
                    f"flags: {args[4]}"
                ]
        elif name == "cuMemAddressFree":
            if len(args) >= 2:
                return [
                    f"// CUDA driver virtual memory free address: "
                    f"address: {args[0]}, bytes: {args[1]}"
                ]
        elif name == "cuMemCreate":
            if len(node.args) >= 4:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    f"// CUDA driver virtual memory create allocation: "
                    f"output: {output}, bytes: {args[1]}, "
                    f"props: {args[2]}, flags: {args[3]}"
                ]
        elif name == "cuMemRelease":
            if args:
                return [
                    f"// CUDA driver virtual memory release allocation: "
                    f"allocation: {args[0]}"
                ]
        elif name == "cuMemMap":
            if len(args) >= 5:
                return [
                    f"// CUDA driver virtual memory map: "
                    f"address: {args[0]}, bytes: {args[1]}, "
                    f"offset: {args[2]}, allocation: {args[3]}, "
                    f"flags: {args[4]}"
                ]
        elif name == "cuMemUnmap":
            if len(args) >= 2:
                return [
                    f"// CUDA driver virtual memory unmap: "
                    f"address: {args[0]}, bytes: {args[1]}"
                ]
        elif name == "cuMemSetAccess":
            if len(args) >= 4:
                return [
                    f"// CUDA driver virtual memory set access: "
                    f"address: {args[0]}, bytes: {args[1]}, "
                    f"descriptors: {args[2]}, count: {args[3]}"
                ]
        elif name == "cuMemGetAccess":
            if len(node.args) >= 3:
                flags_output = self.format_runtime_pointer_target(node.args[0])
                return [
                    f"// CUDA driver virtual memory get access: "
                    f"flags output: {flags_output}, location: {args[1]}, "
                    f"address: {args[2]}"
                ]
        elif name == "cuMemRetainAllocationHandle":
            if len(node.args) >= 2:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    f"// CUDA driver virtual memory retain allocation handle: "
                    f"output: {output}, address: {args[1]}"
                ]
        elif name == "cuMemExportToShareableHandle":
            if len(node.args) >= 4:
                handle_output = self.format_runtime_pointer_target(node.args[0])
                return [
                    f"// CUDA driver virtual memory export shareable handle: "
                    f"handle output: {handle_output}, allocation: {args[1]}, "
                    f"handle type: {args[2]}, flags: {args[3]}"
                ]
        elif name == "cuMemImportFromShareableHandle":
            if len(node.args) >= 3:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    f"// CUDA driver virtual memory import shareable handle: "
                    f"output: {output}, handle: {args[1]}, "
                    f"handle type: {args[2]}"
                ]
        elif name == "cuImportExternalMemory":
            if len(node.args) >= 2:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    f"// CUDA driver external memory import: "
                    f"output: {output}, handle: {args[1]}"
                ]
        elif name == "cuExternalMemoryGetMappedBuffer":
            if len(node.args) >= 3:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    f"// CUDA driver external memory mapped buffer: "
                    f"{args[1]}, desc: {args[2]}, output: {output}"
                ]
        elif name == "cuExternalMemoryGetMappedMipmappedArray":
            if len(node.args) >= 3:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    "// CUDA driver external memory mapped mipmapped array: "
                    f"{args[1]}, desc: {args[2]}, output: {output}"
                ]
        elif name == "cuDestroyExternalMemory":
            if args:
                return [f"// CUDA driver external memory destroy: {args[0]}"]
        elif name == "cuImportExternalSemaphore":
            if len(node.args) >= 2:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    f"// CUDA driver external semaphore import: "
                    f"output: {output}, handle: {args[1]}"
                ]
        elif name in {
            "cuSignalExternalSemaphoresAsync",
            "cuWaitExternalSemaphoresAsync",
        }:
            if len(args) >= 3:
                operation = (
                    "signal" if name == "cuSignalExternalSemaphoresAsync" else "wait"
                )
                comment = (
                    f"// CUDA driver external semaphore {operation}: "
                    f"semaphores: {args[0]}, params: {args[1]}, count: {args[2]}"
                )
                if len(args) >= 4:
                    comment += f", stream: {args[3]}"
                return [comment]
        elif name == "cuDestroyExternalSemaphore":
            if args:
                return [f"// CUDA driver external semaphore destroy: {args[0]}"]
        elif name in {"cuArrayCreate", "cuArrayCreate_v2"}:
            if len(node.args) >= 2:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    f"// CUDA driver array create: output: {output}, "
                    f"desc: {args[1]}"
                ]
        elif name in {"cuArray3DCreate", "cuArray3DCreate_v2"}:
            if len(node.args) >= 2:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    f"// CUDA driver array 3D create: output: {output}, "
                    f"desc: {args[1]}"
                ]
        elif name == "cuArrayDestroy":
            if args:
                return [f"// CUDA driver array destroy: array: {args[0]}"]
        elif name == "cuMipmappedArrayCreate":
            if len(node.args) >= 3:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    f"// CUDA driver mipmapped array create: output: {output}, "
                    f"desc: {args[1]}, levels: {args[2]}"
                ]
        elif name == "cuMipmappedArrayGetLevel":
            if len(node.args) >= 3:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    "// CUDA driver mipmapped array get level: "
                    f"output: {output}, mipmapped array: {args[1]}, "
                    f"level: {args[2]}"
                ]
        elif name == "cuMipmappedArrayDestroy":
            if args:
                return [
                    f"// CUDA driver mipmapped array destroy: mipmapped array: {args[0]}"
                ]
        elif name in {"cuArrayGetDescriptor", "cuArrayGetDescriptor_v2"}:
            if len(node.args) >= 2:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    "// CUDA driver array get descriptor: "
                    f"output: {output}, array: {args[1]}"
                ]
        elif name in {"cuArray3DGetDescriptor", "cuArray3DGetDescriptor_v2"}:
            if len(node.args) >= 2:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    "// CUDA driver array 3D get descriptor: "
                    f"output: {output}, array: {args[1]}"
                ]
        elif name == "cuArrayGetMemoryRequirements":
            if len(node.args) >= 3:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    "// CUDA driver array get memory requirements: "
                    f"output: {output}, array: {args[1]}, device: {args[2]}"
                ]
        elif name == "cuMipmappedArrayGetMemoryRequirements":
            if len(node.args) >= 3:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    "// CUDA driver mipmapped array get memory requirements: "
                    f"output: {output}, mipmapped array: {args[1]}, "
                    f"device: {args[2]}"
                ]
        elif name == "cuArrayGetPlane":
            if len(node.args) >= 3:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    f"// CUDA driver array get plane: output: {output}, "
                    f"array: {args[1]}, plane: {args[2]}"
                ]
        elif name == "cuArrayGetSparseProperties":
            if len(node.args) >= 2:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    "// CUDA driver array get sparse properties: "
                    f"output: {output}, array: {args[1]}"
                ]
        elif name == "cuMipmappedArrayGetSparseProperties":
            if len(node.args) >= 2:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    "// CUDA driver mipmapped array get sparse properties: "
                    f"output: {output}, mipmapped array: {args[1]}"
                ]
        elif name == "cuTexObjectCreate":
            if len(node.args) >= 4:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    "// CUDA driver texture object create: "
                    f"output: {output}, resource desc: {args[1]}, "
                    f"texture desc: {args[2]}, resource view desc: {args[3]}"
                ]
        elif name == "cuTexObjectDestroy":
            if args:
                return [
                    f"// CUDA driver texture object destroy: texture object: {args[0]}"
                ]
        elif name == "cuTexObjectGetResourceDesc":
            if len(node.args) >= 2:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    "// CUDA driver texture object get resource desc: "
                    f"output: {output}, texture object: {args[1]}"
                ]
        elif name == "cuTexObjectGetTextureDesc":
            if len(node.args) >= 2:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    "// CUDA driver texture object get texture desc: "
                    f"output: {output}, texture object: {args[1]}"
                ]
        elif name == "cuTexObjectGetResourceViewDesc":
            if len(node.args) >= 2:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    "// CUDA driver texture object get resource view desc: "
                    f"output: {output}, texture object: {args[1]}"
                ]
        elif name == "cuSurfObjectCreate":
            if len(node.args) >= 2:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    "// CUDA driver surface object create: "
                    f"output: {output}, resource desc: {args[1]}"
                ]
        elif name == "cuSurfObjectDestroy":
            if args:
                return [
                    f"// CUDA driver surface object destroy: surface object: {args[0]}"
                ]
        elif name == "cuSurfObjectGetResourceDesc":
            if len(node.args) >= 2:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    "// CUDA driver surface object get resource desc: "
                    f"output: {output}, surface object: {args[1]}"
                ]
        elif name == "cuTexRefSetArray":
            if len(args) >= 3:
                return [
                    "// CUDA driver texture reference set array: "
                    f"texture ref: {args[0]}, array: {args[1]}, flags: {args[2]}"
                ]
        elif name == "cuTexRefSetMipmappedArray":
            if len(args) >= 3:
                return [
                    "// CUDA driver texture reference set mipmapped array: "
                    f"texture ref: {args[0]}, mipmapped array: {args[1]}, "
                    f"flags: {args[2]}"
                ]
        elif name in {"cuTexRefSetAddress", "cuTexRefSetAddress_v2"}:
            if len(node.args) >= 4:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    "// CUDA driver texture reference set address: "
                    f"byte offset output: {output}, texture ref: {args[1]}, "
                    f"pointer: {args[2]}, bytes: {args[3]}"
                ]
        elif name in {"cuTexRefSetAddress2D", "cuTexRefSetAddress2D_v2"}:
            if len(args) >= 4:
                return [
                    "// CUDA driver texture reference set address 2D: "
                    f"texture ref: {args[0]}, desc: {args[1]}, "
                    f"pointer: {args[2]}, pitch: {args[3]}"
                ]
        elif name == "cuTexRefSetFormat":
            if len(args) >= 3:
                return [
                    "// CUDA driver texture reference set format: "
                    f"texture ref: {args[0]}, format: {args[1]}, "
                    f"components: {args[2]}"
                ]
        elif name == "cuTexRefSetAddressMode":
            if len(args) >= 3:
                return [
                    "// CUDA driver texture reference set address mode: "
                    f"texture ref: {args[0]}, dimension: {args[1]}, "
                    f"mode: {args[2]}"
                ]
        elif name == "cuTexRefSetFilterMode":
            if len(args) >= 2:
                return [
                    "// CUDA driver texture reference set filter mode: "
                    f"texture ref: {args[0]}, mode: {args[1]}"
                ]
        elif name == "cuTexRefSetMipmapFilterMode":
            if len(args) >= 2:
                return [
                    "// CUDA driver texture reference set mipmap filter mode: "
                    f"texture ref: {args[0]}, mode: {args[1]}"
                ]
        elif name == "cuTexRefSetMipmapLevelBias":
            if len(args) >= 2:
                return [
                    "// CUDA driver texture reference set mipmap level bias: "
                    f"texture ref: {args[0]}, bias: {args[1]}"
                ]
        elif name == "cuTexRefSetMipmapLevelClamp":
            if len(args) >= 3:
                return [
                    "// CUDA driver texture reference set mipmap level clamp: "
                    f"texture ref: {args[0]}, min level: {args[1]}, "
                    f"max level: {args[2]}"
                ]
        elif name == "cuTexRefSetMaxAnisotropy":
            if len(args) >= 2:
                return [
                    "// CUDA driver texture reference set max anisotropy: "
                    f"texture ref: {args[0]}, anisotropy: {args[1]}"
                ]
        elif name == "cuTexRefSetBorderColor":
            if len(args) >= 2:
                return [
                    "// CUDA driver texture reference set border color: "
                    f"texture ref: {args[0]}, color: {args[1]}"
                ]
        elif name == "cuTexRefSetFlags":
            if len(args) >= 2:
                return [
                    "// CUDA driver texture reference set flags: "
                    f"texture ref: {args[0]}, flags: {args[1]}"
                ]
        elif name in {"cuTexRefGetAddress", "cuTexRefGetAddress_v2"}:
            if len(node.args) >= 2:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    "// CUDA driver texture reference get address: "
                    f"output: {output}, texture ref: {args[1]}"
                ]
        elif name == "cuTexRefGetArray":
            if len(node.args) >= 2:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    "// CUDA driver texture reference get array: "
                    f"output: {output}, texture ref: {args[1]}"
                ]
        elif name == "cuTexRefGetMipmappedArray":
            if len(node.args) >= 2:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    "// CUDA driver texture reference get mipmapped array: "
                    f"output: {output}, texture ref: {args[1]}"
                ]
        elif name == "cuTexRefGetAddressMode":
            if len(node.args) >= 3:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    "// CUDA driver texture reference get address mode: "
                    f"output: {output}, texture ref: {args[1]}, "
                    f"dimension: {args[2]}"
                ]
        elif name == "cuTexRefGetFilterMode":
            if len(node.args) >= 2:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    "// CUDA driver texture reference get filter mode: "
                    f"output: {output}, texture ref: {args[1]}"
                ]
        elif name == "cuTexRefGetFormat":
            if len(node.args) >= 3:
                format_output = self.format_runtime_pointer_target(node.args[0])
                channel_output = self.format_runtime_pointer_target(node.args[1])
                return [
                    "// CUDA driver texture reference get format: "
                    f"format output: {format_output}, "
                    f"channel output: {channel_output}, texture ref: {args[2]}"
                ]
        elif name == "cuTexRefGetMipmapFilterMode":
            if len(node.args) >= 2:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    "// CUDA driver texture reference get mipmap filter mode: "
                    f"output: {output}, texture ref: {args[1]}"
                ]
        elif name == "cuTexRefGetMipmapLevelBias":
            if len(node.args) >= 2:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    "// CUDA driver texture reference get mipmap level bias: "
                    f"output: {output}, texture ref: {args[1]}"
                ]
        elif name == "cuTexRefGetMipmapLevelClamp":
            if len(node.args) >= 3:
                min_output = self.format_runtime_pointer_target(node.args[0])
                max_output = self.format_runtime_pointer_target(node.args[1])
                return [
                    "// CUDA driver texture reference get mipmap level clamp: "
                    f"min output: {min_output}, max output: {max_output}, "
                    f"texture ref: {args[2]}"
                ]
        elif name == "cuTexRefGetMaxAnisotropy":
            if len(node.args) >= 2:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    "// CUDA driver texture reference get max anisotropy: "
                    f"output: {output}, texture ref: {args[1]}"
                ]
        elif name == "cuTexRefGetBorderColor":
            if len(node.args) >= 2:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    "// CUDA driver texture reference get border color: "
                    f"output: {output}, texture ref: {args[1]}"
                ]
        elif name == "cuTexRefGetFlags":
            if len(node.args) >= 2:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    "// CUDA driver texture reference get flags: "
                    f"output: {output}, texture ref: {args[1]}"
                ]
        elif name == "cuSurfRefSetArray":
            if len(args) >= 3:
                return [
                    "// CUDA driver surface reference set array: "
                    f"surface ref: {args[0]}, array: {args[1]}, flags: {args[2]}"
                ]
        elif name == "cuSurfRefGetArray":
            if len(node.args) >= 2:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    "// CUDA driver surface reference get array: "
                    f"output: {output}, surface ref: {args[1]}"
                ]
        elif name == "cuGLInit":
            return ["// CUDA driver OpenGL initialize"]
        elif name in {"cuGLCtxCreate", "cuGLCtxCreate_v2"}:
            if len(node.args) >= 3:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    "// CUDA driver OpenGL context create: "
                    f"output: {output}, flags: {args[1]}, device: {args[2]}"
                ]
        elif name == "cuGLGetDevices":
            if len(node.args) >= 4:
                count_output = self.format_runtime_pointer_target(node.args[0])
                return [
                    "// CUDA driver OpenGL get devices: "
                    f"count output: {count_output}, devices: {args[1]}, "
                    f"max devices: {args[2]}, device list: {args[3]}"
                ]
        elif name == "cuGLRegisterBufferObject":
            if args:
                return [
                    "// CUDA driver OpenGL register buffer object: "
                    f"buffer object: {args[0]}"
                ]
        elif name == "cuGLSetBufferObjectMapFlags":
            if len(args) >= 2:
                return [
                    "// CUDA driver OpenGL set buffer object map flags: "
                    f"buffer object: {args[0]}, flags: {args[1]}"
                ]
        elif name in {"cuGLMapBufferObject", "cuGLMapBufferObject_v2"}:
            if len(node.args) >= 3:
                pointer_output = self.format_runtime_pointer_target(node.args[0])
                size_output = self.format_runtime_pointer_target(node.args[1])
                return [
                    "// CUDA driver OpenGL map buffer object: "
                    f"pointer output: {pointer_output}, "
                    f"size output: {size_output}, buffer object: {args[2]}"
                ]
        elif name in {
            "cuGLMapBufferObjectAsync",
            "cuGLMapBufferObjectAsync_v2",
        }:
            if len(node.args) >= 4:
                pointer_output = self.format_runtime_pointer_target(node.args[0])
                size_output = self.format_runtime_pointer_target(node.args[1])
                return [
                    "// CUDA driver OpenGL map buffer object async: "
                    f"pointer output: {pointer_output}, "
                    f"size output: {size_output}, buffer object: {args[2]}, "
                    f"stream: {args[3]}"
                ]
        elif name == "cuGLUnmapBufferObject":
            if args:
                return [
                    "// CUDA driver OpenGL unmap buffer object: "
                    f"buffer object: {args[0]}"
                ]
        elif name == "cuGLUnmapBufferObjectAsync":
            if len(args) >= 2:
                return [
                    "// CUDA driver OpenGL unmap buffer object async: "
                    f"buffer object: {args[0]}, stream: {args[1]}"
                ]
        elif name == "cuGLUnregisterBufferObject":
            if args:
                return [
                    "// CUDA driver OpenGL unregister buffer object: "
                    f"buffer object: {args[0]}"
                ]
        elif name == "cuGraphicsGLRegisterBuffer":
            if len(node.args) >= 3:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    "// CUDA driver graphics GL register buffer: "
                    f"output: {output}, buffer: {args[1]}, flags: {args[2]}"
                ]
        elif name == "cuGraphicsGLRegisterImage":
            if len(node.args) >= 4:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    "// CUDA driver graphics GL register image: "
                    f"output: {output}, image: {args[1]}, "
                    f"target: {args[2]}, flags: {args[3]}"
                ]
        elif name in {
            "cuGraphicsD3D9RegisterResource",
            "cuGraphicsD3D10RegisterResource",
            "cuGraphicsD3D11RegisterResource",
        }:
            if len(node.args) >= 3:
                output = self.format_runtime_pointer_target(node.args[0])
                api = name[len("cuGraphics") : -len("RegisterResource")]
                return [
                    f"// CUDA driver graphics {api} register resource: "
                    f"output: {output}, resource: {args[1]}, flags: {args[2]}"
                ]
        elif name == "cuGraphicsUnregisterResource":
            if args:
                return [
                    f"// CUDA driver graphics unregister resource: resource: {args[0]}"
                ]
        elif name in {"cuGraphicsMapResources", "cuGraphicsUnmapResources"}:
            if len(args) >= 3:
                operation = "map" if name == "cuGraphicsMapResources" else "unmap"
                return [
                    f"// CUDA driver graphics {operation} resources: "
                    f"count: {args[0]}, resources: {args[1]}, stream: {args[2]}"
                ]
        elif name in {
            "cuGraphicsResourceSetMapFlags",
            "cuGraphicsResourceSetMapFlags_v2",
        }:
            if len(args) >= 2:
                return [
                    "// CUDA driver graphics resource set map flags: "
                    f"resource: {args[0]}, flags: {args[1]}"
                ]
        elif name in {
            "cuGraphicsResourceGetMappedPointer",
            "cuGraphicsResourceGetMappedPointer_v2",
        }:
            if len(node.args) >= 3:
                pointer_output = self.format_runtime_pointer_target(node.args[0])
                size_output = self.format_runtime_pointer_target(node.args[1])
                return [
                    "// CUDA driver graphics mapped pointer: "
                    f"pointer output: {pointer_output}, "
                    f"size output: {size_output}, resource: {args[2]}"
                ]
        elif name == "cuGraphicsSubResourceGetMappedArray":
            if len(node.args) >= 4:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    "// CUDA driver graphics subresource mapped array: "
                    f"output: {output}, resource: {args[1]}, "
                    f"array index: {args[2]}, mip level: {args[3]}"
                ]
        elif name == "cuGraphicsResourceGetMappedMipmappedArray":
            if len(node.args) >= 2:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    "// CUDA driver graphics mapped mipmapped array: "
                    f"output: {output}, resource: {args[1]}"
                ]
        elif name in {"cuMemcpy", "cuMemcpyAsync"}:
            if len(args) >= 3:
                comment = (
                    f"// CUDA driver memory copy: {args[1]} -> {args[0]}, "
                    f"bytes: {args[2]}"
                )
                if name == "cuMemcpyAsync" and len(args) >= 4:
                    comment += f", stream: {args[3]}"
                return [comment]
        elif name in {
            "cuMemcpyHtoD",
            "cuMemcpyDtoH",
            "cuMemcpyDtoD",
            "cuMemcpyHtoDAsync",
            "cuMemcpyDtoHAsync",
            "cuMemcpyDtoDAsync",
        }:
            if len(args) >= 3:
                direction = name[len("cuMemcpy") :]
                if direction.endswith("Async"):
                    direction = direction[: -len("Async")]
                comment = (
                    f"// CUDA driver memory copy {direction}: "
                    f"{args[1]} -> {args[0]}, bytes: {args[2]}"
                )
                if name.endswith("Async") and len(args) >= 4:
                    comment += f", stream: {args[3]}"
                return [comment]
        elif name in {"cuMemcpy2D", "cuMemcpy2DAsync", "cuMemcpy2DUnaligned"}:
            if node.args:
                params = self.format_runtime_pointer_target(node.args[0])
                comment = f"// CUDA driver memory copy 2D: params: {params}"
                if name == "cuMemcpy2DAsync" and len(args) >= 2:
                    comment += f", stream: {args[1]}"
                return [comment]
        elif name in {"cuMemcpy3D", "cuMemcpy3DAsync"}:
            if node.args:
                params = self.format_runtime_pointer_target(node.args[0])
                comment = f"// CUDA driver memory copy 3D: params: {params}"
                if name == "cuMemcpy3DAsync" and len(args) >= 2:
                    comment += f", stream: {args[1]}"
                return [comment]
        elif name in {
            "cuMemsetD8",
            "cuMemsetD32",
            "cuMemsetD8Async",
            "cuMemsetD32Async",
        }:
            if len(args) >= 3:
                width = "D32" if "D32" in name else "D8"
                comment = (
                    f"// CUDA driver memory set {width}: {args[0]}, "
                    f"value: {args[1]}, count: {args[2]}"
                )
                if name.endswith("Async") and len(args) >= 4:
                    comment += f", stream: {args[3]}"
                return [comment]
        elif name in {
            "cuMemsetD2D8",
            "cuMemsetD2D32",
            "cuMemsetD2D8Async",
            "cuMemsetD2D32Async",
        }:
            if len(args) >= 5:
                width = "D32" if "D32" in name else "D8"
                comment = (
                    f"// CUDA driver memory set 2D {width}: {args[0]}, "
                    f"pitch: {args[1]}, value: {args[2]}, "
                    f"width: {args[3]}, height: {args[4]}"
                )
                if name.endswith("Async") and len(args) >= 6:
                    comment += f", stream: {args[5]}"
                return [comment]
        elif name == "cuModuleLoad":
            if len(node.args) >= 2:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    f"// CUDA driver module load: output: {output}, path: {args[1]}"
                ]
        elif name == "cuModuleLoadData":
            if len(node.args) >= 2:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    "// CUDA driver module load data: "
                    f"output: {output}, image: {args[1]}"
                ]
        elif name == "cuModuleLoadDataEx":
            if len(node.args) >= 5:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    "// CUDA driver module load data with options: "
                    f"output: {output}, image: {args[1]}, "
                    f"option count: {args[2]}, options: {args[3]}, "
                    f"option values: {args[4]}"
                ]
        elif name == "cuLinkCreate":
            if len(node.args) >= 4:
                output = self.format_runtime_pointer_target(node.args[3])
                return [
                    "// CUDA driver linker create: "
                    f"output: {output}, option count: {args[0]}, "
                    f"options: {args[1]}, option values: {args[2]}"
                ]
        elif name == "cuLinkAddData":
            if len(args) >= 8:
                return [
                    "// CUDA driver linker add data: "
                    f"state: {args[0]}, type: {args[1]}, data: {args[2]}, "
                    f"bytes: {args[3]}, name: {args[4]}, option count: {args[5]}, "
                    f"options: {args[6]}, option values: {args[7]}"
                ]
        elif name == "cuLinkAddFile":
            if len(args) >= 6:
                return [
                    "// CUDA driver linker add file: "
                    f"state: {args[0]}, type: {args[1]}, path: {args[2]}, "
                    f"option count: {args[3]}, options: {args[4]}, "
                    f"option values: {args[5]}"
                ]
        elif name == "cuLinkComplete":
            if len(node.args) >= 3:
                cubin_output = self.format_runtime_pointer_target(node.args[1])
                size_output = self.format_runtime_pointer_target(node.args[2])
                return [
                    "// CUDA driver linker complete: "
                    f"state: {args[0]}, cubin output: {cubin_output}, "
                    f"size output: {size_output}"
                ]
        elif name == "cuLinkDestroy":
            if args:
                return [f"// CUDA driver linker destroy: state: {args[0]}"]
        elif name == "cuModuleGetFunction":
            if len(node.args) >= 3:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    "// CUDA driver module get function: "
                    f"output: {output}, module: {args[1]}, name: {args[2]}"
                ]
        elif name == "cuModuleGetTexRef":
            if len(node.args) >= 3:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    "// CUDA driver module get texture reference: "
                    f"output: {output}, module: {args[1]}, name: {args[2]}"
                ]
        elif name == "cuModuleGetSurfRef":
            if len(node.args) >= 3:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    "// CUDA driver module get surface reference: "
                    f"output: {output}, module: {args[1]}, name: {args[2]}"
                ]
        elif name == "cuModuleGetGlobal":
            if len(node.args) >= 4:
                output = self.format_runtime_pointer_target(node.args[0])
                byte_output = self.format_runtime_pointer_target(node.args[1])
                return [
                    "// CUDA driver module get global: "
                    f"output: {output}, bytes output: {byte_output}, "
                    f"module: {args[2]}, name: {args[3]}"
                ]
        elif name == "cuModuleUnload":
            if args:
                return [f"// CUDA driver module unload: {args[0]}"]
        elif name in {"cuLaunchKernel", "cuLaunchCooperativeKernel"}:
            if len(args) >= 10:
                launch_kind = (
                    "launch cooperative kernel"
                    if name == "cuLaunchCooperativeKernel"
                    else "launch kernel"
                )
                comment = (
                    f"// CUDA driver {launch_kind}: function: {args[0]}, "
                    f"grid: {args[1]} x {args[2]} x {args[3]}, "
                    f"block: {args[4]} x {args[5]} x {args[6]}, "
                    f"shared memory: {args[7]}, stream: {args[8]}, "
                    f"params: {args[9]}"
                )
                if name == "cuLaunchKernel" and len(args) >= 11:
                    comment += f", extra: {args[10]}"
                return [comment]
        elif name == "cuLaunchCooperativeKernelMultiDevice":
            if len(node.args) >= 3:
                params = self.format_runtime_pointer_target(node.args[0])
                return [
                    "// CUDA driver launch cooperative kernel multi-device: "
                    f"params: {params}, count: {args[1]}, flags: {args[2]}"
                ]
        elif name in {
            "cuOccupancyMaxActiveBlocksPerMultiprocessor",
            "cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags",
        }:
            if len(node.args) >= 4:
                output = self.format_runtime_pointer_target(node.args[0])
                comment = (
                    "// CUDA driver occupancy active blocks: "
                    f"output: {output}, function: {args[1]}, "
                    f"block size: {args[2]}, dynamic shared memory: {args[3]}"
                )
                if name.endswith("WithFlags") and len(args) >= 5:
                    comment += f", flags: {args[4]}"
                return [comment]
        elif name in {
            "cuOccupancyMaxPotentialBlockSize",
            "cuOccupancyMaxPotentialBlockSizeWithFlags",
        }:
            if len(node.args) >= 6:
                min_grid_output = self.format_runtime_pointer_target(node.args[0])
                block_size_output = self.format_runtime_pointer_target(node.args[1])
                flags = (
                    args[6] if name.endswith("WithFlags") and len(args) >= 7 else "0"
                )
                return [
                    "// CUDA driver occupancy potential block size: "
                    f"min grid output: {min_grid_output}, "
                    f"block size output: {block_size_output}, "
                    f"function: {args[2]}, dynamic shared memory callback: {args[3]}, "
                    f"dynamic shared memory: {args[4]}, block size limit: {args[5]}, "
                    f"flags: {flags}"
                ]
        elif name == "cuFuncGetAttribute":
            if len(node.args) >= 3:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    "// CUDA driver function get attribute: "
                    f"output: {output}, attribute: {args[1]}, function: {args[2]}"
                ]
        elif name == "cuFuncSetAttribute":
            if len(args) >= 3:
                return [
                    "// CUDA driver function set attribute: "
                    f"function: {args[0]}, attribute: {args[1]}, value: {args[2]}"
                ]
        elif name == "cuFuncSetCacheConfig":
            if len(args) >= 2:
                return [
                    "// CUDA driver function set cache config: "
                    f"function: {args[0]}, config: {args[1]}"
                ]
        elif name == "cuFuncSetSharedMemConfig":
            if len(args) >= 2:
                return [
                    "// CUDA driver function set shared memory config: "
                    f"function: {args[0]}, config: {args[1]}"
                ]
        elif name == "cuInit":
            flags = args[0] if args else "0"
            return [f"// CUDA driver initialize: flags: {flags}"]
        elif name == "cuDriverGetVersion":
            if node.args:
                output = self.format_runtime_pointer_target(node.args[0])
                return [f"// CUDA driver get version: output: {output}"]
        elif name == "cuDeviceGet":
            if len(node.args) >= 2:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    f"// CUDA driver get device: output: {output}, "
                    f"ordinal: {args[1]}"
                ]
        elif name == "cuDeviceGetCount":
            if node.args:
                output = self.format_runtime_pointer_target(node.args[0])
                return [f"// CUDA driver get device count: output: {output}"]
        elif name == "cuDeviceGetName":
            if len(node.args) >= 3:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    f"// CUDA driver get device name: output: {output}, "
                    f"length: {args[1]}, device: {args[2]}"
                ]
        elif name == "cuDeviceGetUuid":
            if len(node.args) >= 2:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    f"// CUDA driver get device UUID: output: {output}, "
                    f"device: {args[1]}"
                ]
        elif name in {"cuDeviceTotalMem", "cuDeviceTotalMem_v2"}:
            if len(node.args) >= 2:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    "// CUDA driver get device total memory: "
                    f"output: {output}, device: {args[1]}"
                ]
        elif name == "cuDeviceGetAttribute":
            if len(node.args) >= 3:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    "// CUDA driver get device attribute: "
                    f"output: {output}, attribute: {args[1]}, device: {args[2]}"
                ]
        elif name == "cuDeviceComputeCapability":
            if len(node.args) >= 3:
                major_output = self.format_runtime_pointer_target(node.args[0])
                minor_output = self.format_runtime_pointer_target(node.args[1])
                return [
                    "// CUDA driver get device compute capability: "
                    f"major output: {major_output}, minor output: {minor_output}, "
                    f"device: {args[2]}"
                ]
        elif name == "cuDeviceGetProperties":
            if len(node.args) >= 2:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    "// CUDA driver get device properties: "
                    f"output: {output}, device: {args[1]}"
                ]
        elif name == "cuDeviceGetP2PAttribute":
            if len(node.args) >= 4:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    "// CUDA driver get device P2P attribute: "
                    f"output: {output}, attribute: {args[1]}, "
                    f"source device: {args[2]}, destination device: {args[3]}"
                ]
        elif name == "cuDeviceCanAccessPeer":
            if len(node.args) >= 3:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    "// CUDA driver device peer access query: "
                    f"output: {output}, device: {args[1]}, peer device: {args[2]}"
                ]
        elif name == "cuDeviceGetLuid":
            if len(node.args) >= 3:
                output = self.format_runtime_pointer_target(node.args[0])
                node_mask_output = self.format_runtime_pointer_target(node.args[1])
                return [
                    "// CUDA driver get device LUID: "
                    f"output: {output}, node mask output: {node_mask_output}, "
                    f"device: {args[2]}"
                ]
        elif name == "cuDeviceGetByPCIBusId":
            if len(node.args) >= 2:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    "// CUDA driver get device by PCI bus ID: "
                    f"output: {output}, bus ID: {args[1]}"
                ]
        elif name == "cuDeviceGetPCIBusId":
            if len(node.args) >= 3:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    "// CUDA driver get device PCI bus ID: "
                    f"output: {output}, length: {args[1]}, device: {args[2]}"
                ]
        elif name in {"cuCtxCreate", "cuCtxCreate_v2"}:
            if len(node.args) >= 3:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    f"// CUDA driver context create: output: {output}, "
                    f"flags: {args[1]}, device: {args[2]}"
                ]
        elif name in {"cuCtxDestroy", "cuCtxDestroy_v2"}:
            if args:
                return [f"// CUDA driver context destroy: {args[0]}"]
        elif name == "cuCtxSetCurrent":
            if args:
                return [f"// CUDA driver context set current: {args[0]}"]
        elif name == "cuCtxGetCurrent":
            if node.args:
                output = self.format_runtime_pointer_target(node.args[0])
                return [f"// CUDA driver context get current: output: {output}"]
        elif name in {"cuCtxPushCurrent", "cuCtxPushCurrent_v2"}:
            if args:
                return [f"// CUDA driver context push current: {args[0]}"]
        elif name in {"cuCtxPopCurrent", "cuCtxPopCurrent_v2"}:
            if node.args:
                output = self.format_runtime_pointer_target(node.args[0])
                return [f"// CUDA driver context pop current: output: {output}"]
        elif name == "cuCtxGetDevice":
            if node.args:
                output = self.format_runtime_pointer_target(node.args[0])
                return [f"// CUDA driver context get device: output: {output}"]
        elif name == "cuCtxGetFlags":
            if node.args:
                output = self.format_runtime_pointer_target(node.args[0])
                return [f"// CUDA driver context get flags: output: {output}"]
        elif name == "cuCtxGetId":
            if len(node.args) >= 2:
                output = self.format_runtime_pointer_target(node.args[1])
                return [
                    f"// CUDA driver context get id: context: {args[0]}, "
                    f"output: {output}"
                ]
        elif name == "cuCtxGetLimit":
            if len(node.args) >= 2:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    f"// CUDA driver context get limit: output: {output}, "
                    f"limit: {args[1]}"
                ]
        elif name == "cuCtxSetLimit":
            if len(args) >= 2:
                return [
                    f"// CUDA driver context set limit: limit: {args[0]}, "
                    f"value: {args[1]}"
                ]
        elif name == "cuCtxGetCacheConfig":
            if node.args:
                output = self.format_runtime_pointer_target(node.args[0])
                return [f"// CUDA driver context get cache config: output: {output}"]
        elif name == "cuCtxSetCacheConfig":
            if args:
                return [f"// CUDA driver context set cache config: config: {args[0]}"]
        elif name == "cuCtxGetSharedMemConfig":
            if node.args:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    "// CUDA driver context get shared memory config: "
                    f"output: {output}"
                ]
        elif name == "cuCtxSetSharedMemConfig":
            if args:
                return [
                    "// CUDA driver context set shared memory config: "
                    f"config: {args[0]}"
                ]
        elif name == "cuCtxEnablePeerAccess":
            if args:
                flags = args[1] if len(args) >= 2 else "0"
                return [
                    f"// CUDA driver context enable peer access: "
                    f"peer: {args[0]}, flags: {flags}"
                ]
        elif name == "cuCtxDisablePeerAccess":
            if args:
                return [f"// CUDA driver context disable peer access: peer: {args[0]}"]
        elif name == "cuCtxSynchronize":
            return ["// CUDA driver context synchronize"]
        elif name == "cuDevicePrimaryCtxRetain":
            if len(node.args) >= 2:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    "// CUDA driver device primary context retain: "
                    f"output: {output}, device: {args[1]}"
                ]
        elif name == "cuDevicePrimaryCtxGetState":
            if len(node.args) >= 3:
                flags_output = self.format_runtime_pointer_target(node.args[1])
                active_output = self.format_runtime_pointer_target(node.args[2])
                return [
                    "// CUDA driver device primary context get state: "
                    f"device: {args[0]}, flags output: {flags_output}, "
                    f"active output: {active_output}"
                ]
        elif name in {"cuDevicePrimaryCtxSetFlags", "cuDevicePrimaryCtxSetFlags_v2"}:
            if len(args) >= 2:
                return [
                    "// CUDA driver device primary context set flags: "
                    f"device: {args[0]}, flags: {args[1]}"
                ]
        elif name in {"cuDevicePrimaryCtxRelease", "cuDevicePrimaryCtxRelease_v2"}:
            if args:
                return [
                    "// CUDA driver device primary context release: "
                    f"device: {args[0]}"
                ]
        elif name in {"cuDevicePrimaryCtxReset", "cuDevicePrimaryCtxReset_v2"}:
            if args:
                return [
                    f"// CUDA driver device primary context reset: device: {args[0]}"
                ]
        elif name in {"cudaDeviceSynchronize", "cudaStreamSynchronize"}:
            if args:
                return [f"// CUDA synchronize: {args[0]}"]
            return ["// CUDA device synchronize"]
        elif name in {
            "cudaStreamCreate",
            "cudaStreamCreateWithFlags",
            "cudaStreamCreateWithPriority",
            "cudaStreamDestroy",
        }:
            if args:
                action = "destroy" if name == "cudaStreamDestroy" else "create"
                stream = (
                    self.format_runtime_pointer_target(node.args[0])
                    if action == "create"
                    else args[0]
                )
                comment = f"// CUDA stream {action}: {stream}"
                if (
                    name
                    in {
                        "cudaStreamCreateWithFlags",
                        "cudaStreamCreateWithPriority",
                    }
                    and len(args) >= 2
                ):
                    comment += f", flags: {args[1]}"
                if name == "cudaStreamCreateWithPriority" and len(args) >= 3:
                    comment += f", priority: {args[2]}"
                return [comment]
        elif name in {"cudaEventCreate", "cudaEventCreateWithFlags"}:
            if args:
                event = self.format_runtime_pointer_target(node.args[0])
                comment = f"// CUDA event create: {event}"
                if len(args) >= 2:
                    comment += f", flags: {args[1]}"
                return [comment]
        elif name == "cudaEventRecord":
            if args:
                comment = f"// CUDA event record: {args[0]}"
                if len(args) >= 2:
                    comment += f", stream: {args[1]}"
                return [comment]
        elif name == "cudaEventSynchronize":
            if args:
                return [f"// CUDA event synchronize: {args[0]}"]
        elif name == "cudaEventElapsedTime":
            if len(node.args) >= 3:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    f"// CUDA event elapsed time: {args[1]} -> {args[2]}, "
                    f"output: {output}"
                ]
        elif name == "cudaEventDestroy":
            if args:
                return [f"// CUDA event destroy: {args[0]}"]
        elif name == "cudaEventQuery":
            if args:
                return [f"// CUDA event query: {args[0]}"]
        elif name in {"cuEventCreate", "cuEventCreateWithFlags"}:
            if len(node.args) >= 2:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    f"// CUDA driver event create: output: {output}, "
                    f"flags: {args[1]}"
                ]
        elif name in {"cuEventRecord", "cuEventRecordWithFlags"}:
            if len(args) >= 2:
                if name == "cuEventRecordWithFlags" and len(args) >= 3:
                    return [
                        "// CUDA driver event record with flags: "
                        f"event: {args[0]}, stream: {args[1]}, flags: {args[2]}"
                    ]
                return [
                    f"// CUDA driver event record: event: {args[0]}, "
                    f"stream: {args[1]}"
                ]
        elif name in {"cuEventQuery", "cuEventSynchronize", "cuEventDestroy"}:
            if args:
                action = {
                    "cuEventQuery": "query",
                    "cuEventSynchronize": "synchronize",
                    "cuEventDestroy": "destroy",
                }[name]
                return [f"// CUDA driver event {action}: event: {args[0]}"]
        elif name == "cuEventElapsedTime":
            if len(node.args) >= 3:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    f"// CUDA driver event elapsed time: {args[1]} -> {args[2]}, "
                    f"output: {output}"
                ]
        elif name == "cuProfilerInitialize":
            if len(args) >= 3:
                return [
                    "// CUDA driver profiler initialize: "
                    f"config: {args[0]}, output: {args[1]}, mode: {args[2]}"
                ]
        elif name in {"cuProfilerStart", "cuProfilerStop"}:
            action = "start" if name == "cuProfilerStart" else "stop"
            return [f"// CUDA driver profiler {action}"]
        elif name == "cudaStreamWaitEvent":
            if len(args) >= 2:
                comment = f"// CUDA stream wait event: {args[0]} waits for {args[1]}"
                if len(args) >= 3:
                    comment += f", flags: {args[2]}"
                return [comment]
        elif name in {
            "cuStreamCreate",
            "cuStreamCreateWithFlags",
            "cuStreamCreateWithPriority",
        }:
            if len(node.args) >= 2:
                output = self.format_runtime_pointer_target(node.args[0])
                action = (
                    "create with priority"
                    if name == "cuStreamCreateWithPriority"
                    else "create"
                )
                comment = (
                    f"// CUDA driver stream {action}: output: {output}, "
                    f"flags: {args[1]}"
                )
                if name == "cuStreamCreateWithPriority" and len(args) >= 3:
                    comment += f", priority: {args[2]}"
                return [comment]
        elif name == "cuStreamDestroy":
            if args:
                return [f"// CUDA driver stream destroy: stream: {args[0]}"]
        elif name in {"cuStreamQuery", "cuStreamSynchronize"}:
            if args:
                action = "query" if name == "cuStreamQuery" else "synchronize"
                return [f"// CUDA driver stream {action}: stream: {args[0]}"]
        elif name == "cuStreamWaitEvent":
            if len(args) >= 2:
                flags = args[2] if len(args) >= 3 else "0"
                return [
                    f"// CUDA driver stream wait event: stream: {args[0]}, "
                    f"event: {args[1]}, flags: {flags}"
                ]
        elif name == "cuLaunchHostFunc":
            if len(args) >= 3:
                return [
                    "// CUDA driver stream launch host function: "
                    f"stream: {args[0]}, callback: {args[1]}, user data: {args[2]}"
                ]
        elif name == "cuStreamAddCallback":
            if len(args) >= 4:
                return [
                    "// CUDA driver stream add callback: "
                    f"stream: {args[0]}, callback: {args[1]}, "
                    f"user data: {args[2]}, flags: {args[3]}"
                ]
        elif name == "cuStreamAttachMemAsync":
            if len(args) >= 3:
                flags = args[3] if len(args) >= 4 else "0"
                return [
                    "// CUDA driver stream attach memory: "
                    f"stream: {args[0]}, pointer: {args[1]}, bytes: {args[2]}, "
                    f"flags: {flags}"
                ]
        elif name == "cuStreamBeginCapture":
            if len(args) >= 2:
                return [
                    "// CUDA driver stream begin capture: "
                    f"stream: {args[0]}, mode: {args[1]}"
                ]
        elif name == "cuStreamEndCapture":
            if len(node.args) >= 2:
                output = self.format_runtime_pointer_target(node.args[1])
                return [
                    "// CUDA driver stream end capture: "
                    f"stream: {args[0]}, graph output: {output}"
                ]
        elif name in {
            "cuStreamWaitValue32",
            "cuStreamWaitValue64",
            "cuStreamWriteValue32",
            "cuStreamWriteValue64",
        }:
            if len(args) >= 4:
                operation = "wait" if "Wait" in name else "write"
                width = "32-bit" if name.endswith("32") else "64-bit"
                return [
                    f"// CUDA driver stream {operation} {width} value: "
                    f"stream: {args[0]}, address: {args[1]}, "
                    f"value: {args[2]}, flags: {args[3]}"
                ]
        elif name == "cuStreamBatchMemOp":
            if len(args) >= 4:
                return [
                    "// CUDA driver stream batch memory operation: "
                    f"stream: {args[0]}, count: {args[1]}, "
                    f"params: {args[2]}, flags: {args[3]}"
                ]
        elif name in {"cuStreamGetAttribute", "cuStreamSetAttribute"}:
            if len(node.args) >= 3:
                action = "get" if name == "cuStreamGetAttribute" else "set"
                target = self.format_runtime_pointer_target(node.args[2])
                target_label = "output" if action == "get" else "value"
                return [
                    f"// CUDA driver stream {action} attribute: "
                    f"stream: {args[0]}, attribute: {args[1]}, "
                    f"{target_label}: {target}"
                ]
        elif name == "cuStreamCopyAttributes":
            if len(args) >= 2:
                return [
                    "// CUDA driver stream copy attributes: "
                    f"destination: {args[0]}, source: {args[1]}"
                ]
        elif name in {"cuStreamGetCtx", "cuStreamGetFlags", "cuStreamGetPriority"}:
            if len(node.args) >= 2:
                query_kind = {
                    "cuStreamGetCtx": "context",
                    "cuStreamGetFlags": "flags",
                    "cuStreamGetPriority": "priority",
                }[name]
                output = self.format_runtime_pointer_target(node.args[1])
                return [
                    f"// CUDA driver stream {query_kind} query: "
                    f"stream: {args[0]}, output: {output}"
                ]
        elif name == "cuStreamIsCapturing":
            if len(node.args) >= 2:
                output = self.format_runtime_pointer_target(node.args[1])
                return [
                    "// CUDA driver stream capture status query: "
                    f"stream: {args[0]}, output: {output}"
                ]
        elif name == "cuStreamGetCaptureInfo_v2":
            if len(node.args) >= 3:
                fields = [
                    f"stream: {args[0]}",
                    f"status output: {self.format_runtime_pointer_target(node.args[1])}",
                    f"id output: {self.format_runtime_pointer_target(node.args[2])}",
                ]
                if len(node.args) >= 4:
                    fields.append(
                        "graph output: "
                        f"{self.format_runtime_pointer_target(node.args[3])}"
                    )
                if len(node.args) >= 5:
                    fields.append(
                        "dependencies output: "
                        f"{self.format_runtime_pointer_target(node.args[4])}"
                    )
                if len(node.args) >= 6:
                    fields.append(
                        "dependency count output: "
                        f"{self.format_runtime_pointer_target(node.args[5])}"
                    )
                return [
                    "// CUDA driver stream capture info query: " + ", ".join(fields)
                ]
        elif name == "cuStreamUpdateCaptureDependencies":
            if len(args) >= 4:
                return [
                    "// CUDA driver stream update capture dependencies: "
                    f"stream: {args[0]}, dependencies: {args[1]}, "
                    f"dependency count: {args[2]}, flags: {args[3]}"
                ]
        elif name == "cudaStreamBeginCapture":
            if len(args) >= 2:
                return [
                    f"// CUDA stream begin capture: stream: {args[0]}, mode: {args[1]}"
                ]
        elif name == "cudaStreamEndCapture":
            if len(node.args) >= 2:
                output = self.format_runtime_pointer_target(node.args[1])
                return [
                    f"// CUDA stream end capture: stream: {args[0]}, "
                    f"graph output: {output}"
                ]
        elif name == "cudaGetLastError":
            return ["// CUDA get last error"]
        elif name == "cudaPeekAtLastError":
            return ["// CUDA peek at last error"]
        elif name in {
            "cudaGetTextureObjectResourceDesc",
            "cudaGetTextureObjectTextureDesc",
            "cudaGetTextureObjectResourceViewDesc",
        }:
            if len(node.args) >= 2:
                descriptor_kind = {
                    "cudaGetTextureObjectResourceDesc": "resource",
                    "cudaGetTextureObjectTextureDesc": "texture",
                    "cudaGetTextureObjectResourceViewDesc": "resource view",
                }[name]
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    f"// CUDA texture object {descriptor_kind} descriptor query: "
                    f"{args[1]}, output: {output}"
                ]
        elif name == "cudaGetSurfaceObjectResourceDesc":
            if len(node.args) >= 2:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    "// CUDA surface object resource descriptor query: "
                    f"{args[1]}, output: {output}"
                ]
        elif name == "cudaImportExternalMemory":
            if len(node.args) >= 2:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    f"// CUDA external memory import: output: {output}, "
                    f"handle: {args[1]}"
                ]
        elif name == "cudaExternalMemoryGetMappedBuffer":
            if len(node.args) >= 3:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    f"// CUDA external memory mapped buffer: {args[1]}, "
                    f"desc: {args[2]}, output: {output}"
                ]
        elif name == "cudaExternalMemoryGetMappedMipmappedArray":
            if len(node.args) >= 3:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    "// CUDA external memory mapped mipmapped array: "
                    f"{args[1]}, desc: {args[2]}, output: {output}"
                ]
        elif name == "cudaDestroyExternalMemory":
            if args:
                return [f"// CUDA external memory destroy: {args[0]}"]
        elif name == "cudaImportExternalSemaphore":
            if len(node.args) >= 2:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    f"// CUDA external semaphore import: output: {output}, "
                    f"handle: {args[1]}"
                ]
        elif name in {
            "cudaSignalExternalSemaphoresAsync",
            "cudaWaitExternalSemaphoresAsync",
        }:
            if len(args) >= 3:
                operation = (
                    "signal" if name == "cudaSignalExternalSemaphoresAsync" else "wait"
                )
                comment = (
                    f"// CUDA external semaphore {operation}: "
                    f"semaphores: {args[0]}, params: {args[1]}, count: {args[2]}"
                )
                if len(args) >= 4:
                    comment += f", stream: {args[3]}"
                return [comment]
        elif name == "cudaDestroyExternalSemaphore":
            if args:
                return [f"// CUDA external semaphore destroy: {args[0]}"]
        elif name == "cudaUserObjectCreate":
            if len(node.args) >= 5:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    f"// CUDA user object create: output: {output}, "
                    f"payload: {args[1]}, destroy callback: {args[2]}, "
                    f"initial references: {args[3]}, flags: {args[4]}"
                ]
        elif name in {"cudaUserObjectRetain", "cudaUserObjectRelease"}:
            if args:
                operation = "retain" if name == "cudaUserObjectRetain" else "release"
                count = args[1] if len(args) >= 2 else "1"
                return [
                    f"// CUDA user object {operation}: object: {args[0]}, "
                    f"references: {count}"
                ]
        elif name in {"cudaGraphRetainUserObject", "cudaGraphReleaseUserObject"}:
            if len(args) >= 2:
                operation = (
                    "retain" if name == "cudaGraphRetainUserObject" else "release"
                )
                count = args[2] if len(args) >= 3 else "1"
                comment = (
                    f"// CUDA graph {operation} user object: graph: {args[0]}, "
                    f"object: {args[1]}, references: {count}"
                )
                if name == "cudaGraphRetainUserObject":
                    flags = args[3] if len(args) >= 4 else "0"
                    comment += f", flags: {flags}"
                return [comment]
        elif name in {
            "cudaDeviceGetGraphMemAttribute",
            "cudaDeviceSetGraphMemAttribute",
        }:
            if len(node.args) >= 3:
                value = self.format_runtime_pointer_target(node.args[2])
                action = "get" if name == "cudaDeviceGetGraphMemAttribute" else "set"
                value_label = "output" if action == "get" else "value"
                return [
                    f"// CUDA device {action} graph memory attribute: "
                    f"device: {args[0]}, attribute: {args[1]}, "
                    f"{value_label}: {value}"
                ]
        elif name == "cudaDeviceGraphMemTrim":
            if args:
                return [f"// CUDA device graph memory trim: device: {args[0]}"]
        elif name == "cudaGraphCreate":
            if len(node.args) >= 2:
                output = self.format_runtime_pointer_target(node.args[0])
                return [f"// CUDA graph create: output: {output}, flags: {args[1]}"]
        elif name == "cudaGraphClone":
            if len(node.args) >= 2:
                output = self.format_runtime_pointer_target(node.args[0])
                return [f"// CUDA graph clone: output: {output}, source: {args[1]}"]
        elif name == "cudaGraphInstantiate":
            if len(node.args) >= 5:
                output = self.format_runtime_pointer_target(node.args[0])
                error_output = self.format_runtime_pointer_target(node.args[2])
                return [
                    f"// CUDA graph instantiate: output: {output}, graph: {args[1]}, "
                    f"error node output: {error_output}, log buffer: {args[3]}, "
                    f"log bytes: {args[4]}"
                ]
        elif name == "cudaGraphInstantiateWithFlags":
            if len(node.args) >= 3:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    f"// CUDA graph instantiate with flags: output: {output}, "
                    f"graph: {args[1]}, flags: {args[2]}"
                ]
        elif name == "cudaGraphInstantiateWithParams":
            if len(node.args) >= 3:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    f"// CUDA graph instantiate with params: output: {output}, "
                    f"graph: {args[1]}, params: {args[2]}"
                ]
        elif name == "cudaGraphConditionalHandleCreate":
            if len(node.args) >= 2:
                output = self.format_runtime_pointer_target(node.args[0])
                comment = (
                    f"// CUDA graph conditional handle create: output: {output}, "
                    f"graph: {args[1]}"
                )
                if len(args) >= 3:
                    comment += f", default launch value: {args[2]}"
                if len(args) >= 4:
                    comment += f", flags: {args[3]}"
                return [comment]
        elif name == "cudaGraphConditionalHandleCreate_v2":
            if len(node.args) >= 2:
                output = self.format_runtime_pointer_target(node.args[0])
                comment = (
                    f"// CUDA graph conditional handle create v2: output: {output}, "
                    f"graph: {args[1]}"
                )
                if len(args) >= 3:
                    comment += f", context: {args[2]}"
                if len(args) >= 4:
                    comment += f", default launch value: {args[3]}"
                if len(args) >= 5:
                    comment += f", flags: {args[4]}"
                return [comment]
        elif name == "cudaGraphSetConditional":
            if len(args) >= 2:
                return [
                    f"// CUDA graph set conditional: handle: {args[0]}, "
                    f"value: {args[1]}"
                ]
        elif name in {
            "cudaGraphKernelNodeSetEnabled",
            "cudaGraphKernelNodeSetGridDim",
        }:
            if len(args) >= 2:
                action = (
                    "set enabled"
                    if name == "cudaGraphKernelNodeSetEnabled"
                    else "set grid dim"
                )
                value_label = (
                    "enabled" if name == "cudaGraphKernelNodeSetEnabled" else "grid dim"
                )
                return [
                    f"// CUDA device graph kernel node {action}: "
                    f"node: {args[0]}, {value_label}: {args[1]}"
                ]
        elif name == "cudaGraphKernelNodeSetParam":
            if len(args) >= 3:
                comment = (
                    f"// CUDA device graph kernel node set param: "
                    f"node: {args[0]}, offset: {args[1]}, value: {args[2]}"
                )
                if len(args) >= 4:
                    comment += f", bytes: {args[3]}"
                return [comment]
        elif name == "cudaGraphKernelNodeUpdatesApply":
            if len(args) >= 2:
                return [
                    f"// CUDA device graph kernel node updates apply: "
                    f"updates: {args[0]}, count: {args[1]}"
                ]
        elif name == "cudaGraphDebugDotPrint":
            if len(args) >= 3:
                return [
                    f"// CUDA graph debug DOT print: graph: {args[0]}, "
                    f"path: {args[1]}, flags: {args[2]}"
                ]
        elif name == "cudaGraphLaunch":
            if len(args) >= 2:
                device_launch_mode = self.CUDA_DEVICE_GRAPH_LAUNCH_MODES.get(args[1])
                if device_launch_mode is not None:
                    return [
                        f"// CUDA graph device launch: exec: {args[0]}, "
                        f"mode: {device_launch_mode}"
                    ]
                return [f"// CUDA graph launch: exec: {args[0]}, stream: {args[1]}"]
        elif name in {"cudaGraphUpload", "cudaGraphExecUpload"}:
            if len(args) >= 2:
                operation = "exec upload" if name == "cudaGraphExecUpload" else "upload"
                return [
                    f"// CUDA graph {operation}: exec: {args[0]}, stream: {args[1]}"
                ]
        elif name == "cudaGraphExecDestroy":
            if args:
                return [f"// CUDA graph exec destroy: {args[0]}"]
        elif name == "cuGraphCreate":
            if len(node.args) >= 2:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    f"// CUDA driver graph create: output: {output}, "
                    f"flags: {args[1]}"
                ]
        elif name == "cuGraphClone":
            if len(node.args) >= 2:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    f"// CUDA driver graph clone: output: {output}, "
                    f"source: {args[1]}"
                ]
        elif name == "cuGraphInstantiate":
            if len(node.args) >= 5:
                output = self.format_runtime_pointer_target(node.args[0])
                error_output = self.format_runtime_pointer_target(node.args[2])
                return [
                    f"// CUDA driver graph instantiate: output: {output}, "
                    f"graph: {args[1]}, error node output: {error_output}, "
                    f"log buffer: {args[3]}, log bytes: {args[4]}"
                ]
            if len(node.args) >= 3:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    f"// CUDA driver graph instantiate: output: {output}, "
                    f"graph: {args[1]}, flags: {args[2]}"
                ]
        elif name == "cuGraphInstantiateWithFlags":
            if len(node.args) >= 3:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    "// CUDA driver graph instantiate with flags: "
                    f"output: {output}, graph: {args[1]}, flags: {args[2]}"
                ]
        elif name == "cuGraphInstantiateWithParams":
            if len(node.args) >= 3:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    "// CUDA driver graph instantiate with params: "
                    f"output: {output}, graph: {args[1]}, params: {args[2]}"
                ]
        elif name == "cuGraphUpload":
            if len(args) >= 2:
                return [
                    f"// CUDA driver graph upload: exec: {args[0]}, "
                    f"stream: {args[1]}"
                ]
        elif name == "cuGraphLaunch":
            if len(args) >= 2:
                return [
                    f"// CUDA driver graph launch: exec: {args[0]}, "
                    f"stream: {args[1]}"
                ]
        elif name == "cuGraphExecDestroy":
            if args:
                return [f"// CUDA driver graph exec destroy: {args[0]}"]
        elif name == "cuUserObjectCreate":
            if len(node.args) >= 5:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    f"// CUDA driver user object create: output: {output}, "
                    f"payload: {args[1]}, destroy callback: {args[2]}, "
                    f"initial references: {args[3]}, flags: {args[4]}"
                ]
        elif name in {"cuUserObjectRetain", "cuUserObjectRelease"}:
            if args:
                operation = "retain" if name == "cuUserObjectRetain" else "release"
                count = args[1] if len(args) >= 2 else "1"
                return [
                    f"// CUDA driver user object {operation}: object: {args[0]}, "
                    f"references: {count}"
                ]
        elif name in {"cuGraphRetainUserObject", "cuGraphReleaseUserObject"}:
            if len(args) >= 2:
                operation = "retain" if name == "cuGraphRetainUserObject" else "release"
                count = args[2] if len(args) >= 3 else "1"
                comment = (
                    f"// CUDA driver graph {operation} user object: "
                    f"graph: {args[0]}, object: {args[1]}, references: {count}"
                )
                if name == "cuGraphRetainUserObject":
                    flags = args[3] if len(args) >= 4 else "0"
                    comment += f", flags: {flags}"
                return [comment]
        elif name in {
            "cuDeviceGetGraphMemAttribute",
            "cuDeviceSetGraphMemAttribute",
        }:
            if len(node.args) >= 3:
                value = self.format_runtime_pointer_target(node.args[2])
                action = "get" if name == "cuDeviceGetGraphMemAttribute" else "set"
                value_label = "output" if action == "get" else "value"
                return [
                    f"// CUDA driver device {action} graph memory attribute: "
                    f"device: {args[0]}, attribute: {args[1]}, "
                    f"{value_label}: {value}"
                ]
        elif name == "cuDeviceGraphMemTrim":
            if args:
                return [f"// CUDA driver device graph memory trim: device: {args[0]}"]
        elif name == "cuGraphConditionalHandleCreate":
            if len(node.args) >= 5:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    "// CUDA driver graph conditional handle create: "
                    f"output: {output}, graph: {args[1]}, context: {args[2]}, "
                    f"default launch value: {args[3]}, flags: {args[4]}"
                ]
        elif name == "cuGraphAddNode":
            if len(node.args) >= 6:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    f"// CUDA driver graph add generic node: output: {output}, "
                    f"graph: {args[1]}, dependencies: {args[2]}, "
                    f"edge data: {args[3]}, dependency count: {args[4]}, "
                    f"params: {args[5]}"
                ]
        elif name in {
            "cuGraphAddKernelNode",
            "cuGraphAddMemcpyNode",
            "cuGraphAddMemsetNode",
            "cuGraphAddHostNode",
        }:
            if len(node.args) >= 5:
                node_kind = {
                    "cuGraphAddKernelNode": "kernel",
                    "cuGraphAddMemcpyNode": "memcpy",
                    "cuGraphAddMemsetNode": "memset",
                    "cuGraphAddHostNode": "host",
                }[name]
                output = self.format_runtime_pointer_target(node.args[0])
                comment = (
                    f"// CUDA driver graph add {node_kind} node: "
                    f"output: {output}, graph: {args[1]}, "
                    f"dependencies: {args[2]}, dependency count: {args[3]}, "
                    f"params: {args[4]}"
                )
                if name in {"cuGraphAddMemcpyNode", "cuGraphAddMemsetNode"}:
                    if len(args) >= 6:
                        comment += f", context: {args[5]}"
                return [comment]
        elif name == "cuGraphAddChildGraphNode":
            if len(node.args) >= 5:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    "// CUDA driver graph add child graph node: "
                    f"output: {output}, graph: {args[1]}, "
                    f"dependencies: {args[2]}, dependency count: {args[3]}, "
                    f"child graph: {args[4]}"
                ]
        elif name == "cuGraphAddEmptyNode":
            if len(node.args) >= 4:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    f"// CUDA driver graph add empty node: output: {output}, "
                    f"graph: {args[1]}, dependencies: {args[2]}, "
                    f"dependency count: {args[3]}"
                ]
        elif name in {"cuGraphAddEventRecordNode", "cuGraphAddEventWaitNode"}:
            if len(node.args) >= 5:
                node_kind = "record" if "Record" in name else "wait"
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    f"// CUDA driver graph add event {node_kind} node: "
                    f"output: {output}, graph: {args[1]}, "
                    f"dependencies: {args[2]}, dependency count: {args[3]}, "
                    f"event: {args[4]}"
                ]
        elif name in {
            "cuGraphAddExternalSemaphoresSignalNode",
            "cuGraphAddExternalSemaphoresWaitNode",
        }:
            if len(node.args) >= 5:
                node_kind = "signal" if "Signal" in name else "wait"
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    f"// CUDA driver graph add external semaphore {node_kind} "
                    f"node: output: {output}, graph: {args[1]}, "
                    f"dependencies: {args[2]}, dependency count: {args[3]}, "
                    f"params: {args[4]}"
                ]
        elif name == "cuGraphAddMemAllocNode":
            if len(node.args) >= 5:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    "// CUDA driver graph add memory alloc node: "
                    f"output: {output}, graph: {args[1]}, "
                    f"dependencies: {args[2]}, dependency count: {args[3]}, "
                    f"params: {args[4]}"
                ]
        elif name == "cuGraphAddMemFreeNode":
            if len(node.args) >= 5:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    f"// CUDA driver graph add memory free node: output: {output}, "
                    f"graph: {args[1]}, dependencies: {args[2]}, "
                    f"dependency count: {args[3]}, pointer: {args[4]}"
                ]
        elif name in {"cuGraphAddDependencies", "cuGraphRemoveDependencies"}:
            if len(args) >= 5:
                operation = "add" if name == "cuGraphAddDependencies" else "remove"
                return [
                    f"// CUDA driver graph {operation} dependencies: "
                    f"graph: {args[0]}, from: {args[1]}, to: {args[2]}, "
                    f"edge data: {args[3]}, count: {args[4]}"
                ]
            if len(args) >= 4:
                operation = "add" if name == "cuGraphAddDependencies" else "remove"
                return [
                    f"// CUDA driver graph {operation} dependencies: "
                    f"graph: {args[0]}, from: {args[1]}, to: {args[2]}, "
                    f"count: {args[3]}"
                ]
        elif name == "cuGraphNodeFindInClone":
            if len(node.args) >= 3:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    f"// CUDA driver graph node find in clone: output: {output}, "
                    f"original node: {args[1]}, clone graph: {args[2]}"
                ]
        elif name == "cuGraphNodeGetType":
            if len(node.args) >= 2:
                output = self.format_runtime_pointer_target(node.args[1])
                return [
                    f"// CUDA driver graph node get type: node: {args[0]}, "
                    f"output: {output}"
                ]
        elif name == "cuGraphDebugDotPrint":
            if len(args) >= 3:
                return [
                    f"// CUDA driver graph debug DOT print: graph: {args[0]}, "
                    f"path: {args[1]}, flags: {args[2]}"
                ]
        elif name in {"cuGraphExecGetFlags", "cuGraphExecGetId"}:
            if len(node.args) >= 2:
                output = self.format_runtime_pointer_target(node.args[1])
                if name == "cuGraphExecGetFlags":
                    return [
                        f"// CUDA driver graph exec get flags: exec: {args[0]}, "
                        f"output: {output}"
                    ]
                return [
                    f"// CUDA driver graph exec get id: exec: {args[0]}, "
                    f"output: {output}"
                ]
        elif name == "cuGraphGetId":
            if len(node.args) >= 2:
                output = self.format_runtime_pointer_target(node.args[1])
                return [
                    f"// CUDA driver graph get id: graph: {args[0]}, "
                    f"output: {output}"
                ]
        elif name == "cuGraphGetEdges":
            if len(node.args) >= 5:
                from_output = self.format_runtime_pointer_target(node.args[1])
                to_output = self.format_runtime_pointer_target(node.args[2])
                edge_data = self.format_runtime_pointer_target(node.args[3])
                count_output = self.format_runtime_pointer_target(node.args[4])
                return [
                    f"// CUDA driver graph get edges: graph: {args[0]}, "
                    f"from output: {from_output}, to output: {to_output}, "
                    f"edge data: {edge_data}, count output: {count_output}"
                ]
        elif name in {"cuGraphGetNodes", "cuGraphGetRootNodes"}:
            if len(node.args) >= 3:
                node_output = self.format_runtime_pointer_target(node.args[1])
                count_output = self.format_runtime_pointer_target(node.args[2])
                node_kind = "root nodes" if name == "cuGraphGetRootNodes" else "nodes"
                return [
                    f"// CUDA driver graph get {node_kind}: graph: {args[0]}, "
                    f"nodes output: {node_output}, count output: {count_output}"
                ]
        elif name == "cuGraphNodeGetContainingGraph":
            if len(node.args) >= 2:
                output = self.format_runtime_pointer_target(node.args[1])
                return [
                    "// CUDA driver graph node get containing graph: "
                    f"node: {args[0]}, output: {output}"
                ]
        elif name in {
            "cuGraphNodeGetDependencies",
            "cuGraphNodeGetDependentNodes",
        }:
            if len(node.args) >= 4:
                nodes_output = self.format_runtime_pointer_target(node.args[1])
                edge_data = self.format_runtime_pointer_target(node.args[2])
                count_output = self.format_runtime_pointer_target(node.args[3])
                if name == "cuGraphNodeGetDependencies":
                    return [
                        "// CUDA driver graph node get dependencies: "
                        f"node: {args[0]}, dependencies output: {nodes_output}, "
                        f"edge data: {edge_data}, count output: {count_output}"
                    ]
                return [
                    "// CUDA driver graph node get dependent nodes: "
                    f"node: {args[0]}, dependent nodes output: {nodes_output}, "
                    f"edge data: {edge_data}, count output: {count_output}"
                ]
        elif name in {"cuGraphNodeGetEnabled", "cuGraphNodeSetEnabled"}:
            if len(node.args) >= 3:
                if name == "cuGraphNodeGetEnabled":
                    output = self.format_runtime_pointer_target(node.args[2])
                    return [
                        f"// CUDA driver graph node get enabled: exec: {args[0]}, "
                        f"node: {args[1]}, output: {output}"
                    ]
                return [
                    f"// CUDA driver graph node set enabled: exec: {args[0]}, "
                    f"node: {args[1]}, enabled: {args[2]}"
                ]
        elif name in {"cuGraphNodeGetLocalId", "cuGraphNodeGetToolsId"}:
            if len(node.args) >= 2:
                output = self.format_runtime_pointer_target(node.args[1])
                id_kind = "local id" if name == "cuGraphNodeGetLocalId" else "tools id"
                return [
                    f"// CUDA driver graph node get {id_kind}: node: {args[0]}, "
                    f"output: {output}"
                ]
        elif name == "cuGraphChildGraphNodeGetGraph":
            if len(node.args) >= 2:
                output = self.format_runtime_pointer_target(node.args[1])
                return [
                    "// CUDA driver graph child graph node get graph: "
                    f"node: {args[0]}, output: {output}"
                ]
        elif name == "cuGraphMemAllocNodeGetParams":
            if len(args) >= 2:
                return [
                    f"// CUDA driver graph memory alloc node get params: "
                    f"node: {args[0]}, params: {args[1]}"
                ]
        elif name == "cuGraphMemFreeNodeGetParams":
            if len(node.args) >= 2:
                output = self.format_runtime_pointer_target(node.args[1])
                return [
                    f"// CUDA driver graph memory free node get params: "
                    f"node: {args[0]}, output: {output}"
                ]
        elif name in {"cuGraphNodeGetParams", "cuGraphNodeSetParams"}:
            if len(args) >= 2:
                action = "get" if name == "cuGraphNodeGetParams" else "set"
                return [
                    f"// CUDA driver graph node {action} params: "
                    f"node: {args[0]}, params: {args[1]}"
                ]
        elif name == "cuGraphExecNodeSetParams":
            if len(args) >= 3:
                return [
                    f"// CUDA driver graph exec set node params: "
                    f"exec: {args[0]}, node: {args[1]}, params: {args[2]}"
                ]
        elif name in {
            "cuGraphKernelNodeGetAttribute",
            "cuGraphKernelNodeSetAttribute",
        }:
            if len(node.args) >= 3:
                action = "get" if name.endswith("GetAttribute") else "set"
                value_label = "output" if action == "get" else "value"
                value = (
                    self.format_runtime_pointer_target(node.args[2])
                    if action == "get"
                    else args[2]
                )
                return [
                    f"// CUDA driver graph kernel node {action} attribute: "
                    f"node: {args[0]}, attribute: {args[1]}, "
                    f"{value_label}: {value}"
                ]
        elif name == "cuGraphKernelNodeCopyAttributes":
            if len(args) >= 2:
                return [
                    f"// CUDA driver graph kernel node copy attributes: "
                    f"source: {args[0]}, destination: {args[1]}"
                ]
        elif name in {
            "cuGraphKernelNodeGetParams",
            "cuGraphKernelNodeSetParams",
            "cuGraphMemcpyNodeGetParams",
            "cuGraphMemcpyNodeSetParams",
            "cuGraphMemsetNodeGetParams",
            "cuGraphMemsetNodeSetParams",
            "cuGraphHostNodeGetParams",
            "cuGraphHostNodeSetParams",
        }:
            if len(args) >= 2:
                node_kind = {
                    "cuGraphKernelNodeGetParams": "kernel",
                    "cuGraphKernelNodeSetParams": "kernel",
                    "cuGraphMemcpyNodeGetParams": "memcpy",
                    "cuGraphMemcpyNodeSetParams": "memcpy",
                    "cuGraphMemsetNodeGetParams": "memset",
                    "cuGraphMemsetNodeSetParams": "memset",
                    "cuGraphHostNodeGetParams": "host",
                    "cuGraphHostNodeSetParams": "host",
                }[name]
                action = "get" if name.endswith("GetParams") else "set"
                return [
                    f"// CUDA driver graph {node_kind} node {action} params: "
                    f"node: {args[0]}, params: {args[1]}"
                ]
        elif name in {
            "cuGraphExternalSemaphoresSignalNodeGetParams",
            "cuGraphExternalSemaphoresSignalNodeSetParams",
            "cuGraphExternalSemaphoresWaitNodeGetParams",
            "cuGraphExternalSemaphoresWaitNodeSetParams",
        }:
            if len(args) >= 2:
                node_kind = "signal" if "Signal" in name else "wait"
                action = "get" if name.endswith("GetParams") else "set"
                return [
                    f"// CUDA driver graph external semaphore {node_kind} "
                    f"node {action} params: node: {args[0]}, params: {args[1]}"
                ]
        elif name in {
            "cuGraphEventRecordNodeGetEvent",
            "cuGraphEventWaitNodeGetEvent",
        }:
            if len(node.args) >= 2:
                node_kind = "record" if "Record" in name else "wait"
                output = self.format_runtime_pointer_target(node.args[1])
                return [
                    f"// CUDA driver graph event {node_kind} node get event: "
                    f"node: {args[0]}, output: {output}"
                ]
        elif name in {
            "cuGraphEventRecordNodeSetEvent",
            "cuGraphEventWaitNodeSetEvent",
        }:
            if len(args) >= 2:
                node_kind = "record" if "Record" in name else "wait"
                return [
                    f"// CUDA driver graph event {node_kind} node set event: "
                    f"node: {args[0]}, event: {args[1]}"
                ]
        elif name in {"cuGraphExecUpdate", "cuGraphExecUpdate_v2"}:
            if len(node.args) >= 4:
                error_output = self.format_runtime_pointer_target(node.args[2])
                result_output = self.format_runtime_pointer_target(node.args[3])
                return [
                    f"// CUDA driver graph exec update: exec: {args[0]}, "
                    f"graph: {args[1]}, error node output: {error_output}, "
                    f"result output: {result_output}"
                ]
            if len(node.args) >= 3:
                result_info = self.format_runtime_pointer_target(node.args[2])
                update_kind = " update v2" if name.endswith("_v2") else " update"
                return [
                    f"// CUDA driver graph exec{update_kind}: exec: {args[0]}, "
                    f"graph: {args[1]}, result info output: {result_info}"
                ]
        elif name in {
            "cuGraphExecKernelNodeSetParams",
            "cuGraphExecMemcpyNodeSetParams",
            "cuGraphExecMemsetNodeSetParams",
            "cuGraphExecHostNodeSetParams",
        }:
            if len(args) >= 3:
                node_kind = {
                    "cuGraphExecKernelNodeSetParams": "kernel",
                    "cuGraphExecMemcpyNodeSetParams": "memcpy",
                    "cuGraphExecMemsetNodeSetParams": "memset",
                    "cuGraphExecHostNodeSetParams": "host",
                }[name]
                comment = (
                    f"// CUDA driver graph exec set {node_kind} node params: "
                    f"exec: {args[0]}, node: {args[1]}, params: {args[2]}"
                )
                if (
                    name
                    in {
                        "cuGraphExecMemcpyNodeSetParams",
                        "cuGraphExecMemsetNodeSetParams",
                    }
                    and len(args) >= 4
                ):
                    comment += f", context: {args[3]}"
                return [comment]
        elif name in {
            "cuGraphExecExternalSemaphoresSignalNodeSetParams",
            "cuGraphExecExternalSemaphoresWaitNodeSetParams",
        }:
            if len(args) >= 3:
                node_kind = "signal" if "Signal" in name else "wait"
                return [
                    f"// CUDA driver graph exec set external semaphore "
                    f"{node_kind} node params: exec: {args[0]}, "
                    f"node: {args[1]}, params: {args[2]}"
                ]
        elif name in {
            "cuGraphExecEventRecordNodeSetEvent",
            "cuGraphExecEventWaitNodeSetEvent",
        }:
            if len(args) >= 3:
                node_kind = "record" if "Record" in name else "wait"
                return [
                    f"// CUDA driver graph exec set event {node_kind} "
                    f"node event: exec: {args[0]}, node: {args[1]}, "
                    f"event: {args[2]}"
                ]
        elif name == "cuGraphExecChildGraphNodeSetParams":
            if len(args) >= 3:
                return [
                    "// CUDA driver graph exec set child graph node params: "
                    f"exec: {args[0]}, node: {args[1]}, child graph: {args[2]}"
                ]
        elif name in {
            "cudaGraphAddKernelNode",
            "cudaGraphAddMemcpyNode",
            "cudaGraphAddMemsetNode",
            "cudaGraphAddHostNode",
        }:
            if len(node.args) >= 5:
                node_kind = {
                    "cudaGraphAddKernelNode": "kernel",
                    "cudaGraphAddMemcpyNode": "memcpy",
                    "cudaGraphAddMemsetNode": "memset",
                    "cudaGraphAddHostNode": "host",
                }[name]
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    f"// CUDA graph add {node_kind} node: output: {output}, "
                    f"graph: {args[1]}, dependencies: {args[2]}, "
                    f"dependency count: {args[3]}, params: {args[4]}"
                ]
        elif name == "cuGraphAddBatchMemOpNode":
            if len(node.args) >= 5:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    "// CUDA driver graph add batch memory operation node: "
                    f"output: {output}, graph: {args[1]}, "
                    f"dependencies: {args[2]}, dependency count: {args[3]}, "
                    f"params: {args[4]}"
                ]
        elif name == "cudaGraphAddMemcpyNode1D":
            if len(node.args) >= 8:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    f"// CUDA graph add memcpy 1D node: output: {output}, "
                    f"graph: {args[1]}, dependencies: {args[2]}, "
                    f"dependency count: {args[3]}, dst: {args[4]}, src: {args[5]}, "
                    f"byte count: {args[6]}, kind: {args[7]}"
                ]
        elif name in {
            "cudaGraphAddMemcpyNodeFromSymbol",
            "cudaGraphAddMemcpyNodeToSymbol",
        }:
            if len(node.args) >= 9:
                output = self.format_runtime_pointer_target(node.args[0])
                if name == "cudaGraphAddMemcpyNodeFromSymbol":
                    return [
                        "// CUDA graph add memcpy-from-symbol node: "
                        f"output: {output}, graph: {args[1]}, "
                        f"dependencies: {args[2]}, dependency count: {args[3]}, "
                        f"dst: {args[4]}, symbol: {args[5]}, "
                        f"byte count: {args[6]}, offset: {args[7]}, "
                        f"kind: {args[8]}"
                    ]
                return [
                    "// CUDA graph add memcpy-to-symbol node: "
                    f"output: {output}, graph: {args[1]}, "
                    f"dependencies: {args[2]}, dependency count: {args[3]}, "
                    f"symbol: {args[4]}, src: {args[5]}, "
                    f"byte count: {args[6]}, offset: {args[7]}, kind: {args[8]}"
                ]
        elif name == "cudaGraphAddNode":
            if len(node.args) >= 6:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    f"// CUDA graph add generic node: output: {output}, "
                    f"graph: {args[1]}, dependencies: {args[2]}, "
                    f"edge data: {args[3]}, dependency count: {args[4]}, "
                    f"params: {args[5]}"
                ]
            if len(node.args) >= 5:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    f"// CUDA graph add generic node: output: {output}, "
                    f"graph: {args[1]}, dependencies: {args[2]}, "
                    f"dependency count: {args[3]}, params: {args[4]}"
                ]
        elif name == "cudaGraphAddChildGraphNode":
            if len(node.args) >= 5:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    f"// CUDA graph add child graph node: output: {output}, "
                    f"graph: {args[1]}, dependencies: {args[2]}, "
                    f"dependency count: {args[3]}, child graph: {args[4]}"
                ]
        elif name == "cudaGraphAddEmptyNode":
            if len(node.args) >= 4:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    f"// CUDA graph add empty node: output: {output}, "
                    f"graph: {args[1]}, dependencies: {args[2]}, "
                    f"dependency count: {args[3]}"
                ]
        elif name in {"cudaGraphAddEventRecordNode", "cudaGraphAddEventWaitNode"}:
            if len(node.args) >= 5:
                operation = (
                    "record" if name == "cudaGraphAddEventRecordNode" else "wait"
                )
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    f"// CUDA graph add event {operation} node: "
                    f"output: {output}, graph: {args[1]}, "
                    f"dependencies: {args[2]}, dependency count: {args[3]}, "
                    f"event: {args[4]}"
                ]
        elif name in {
            "cudaGraphAddExternalSemaphoresSignalNode",
            "cudaGraphAddExternalSemaphoresWaitNode",
        }:
            if len(node.args) >= 5:
                operation = (
                    "signal"
                    if name == "cudaGraphAddExternalSemaphoresSignalNode"
                    else "wait"
                )
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    f"// CUDA graph add external semaphore {operation} node: "
                    f"output: {output}, graph: {args[1]}, "
                    f"dependencies: {args[2]}, dependency count: {args[3]}, "
                    f"params: {args[4]}"
                ]
        elif name == "cudaGraphAddMemAllocNode":
            if len(node.args) >= 5:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    f"// CUDA graph add memory alloc node: output: {output}, "
                    f"graph: {args[1]}, dependencies: {args[2]}, "
                    f"dependency count: {args[3]}, params: {args[4]}"
                ]
        elif name == "cudaGraphAddMemFreeNode":
            if len(node.args) >= 5:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    f"// CUDA graph add memory free node: output: {output}, "
                    f"graph: {args[1]}, dependencies: {args[2]}, "
                    f"dependency count: {args[3]}, pointer: {args[4]}"
                ]
        elif name == "cudaGraphMemAllocNodeGetParams":
            if len(args) >= 2:
                return [
                    f"// CUDA graph memory alloc node get params: "
                    f"node: {args[0]}, params: {args[1]}"
                ]
        elif name == "cudaGraphMemFreeNodeGetParams":
            if len(node.args) >= 2:
                output = self.format_runtime_pointer_target(node.args[1])
                return [
                    f"// CUDA graph memory free node get params: "
                    f"node: {args[0]}, output: {output}"
                ]
        elif name in {"cudaGraphAddDependencies", "cudaGraphRemoveDependencies"}:
            if len(args) >= 5:
                operation = "add" if name == "cudaGraphAddDependencies" else "remove"
                return [
                    f"// CUDA graph {operation} dependencies: graph: {args[0]}, "
                    f"from: {args[1]}, to: {args[2]}, edge data: {args[3]}, "
                    f"count: {args[4]}"
                ]
            if len(args) >= 4:
                operation = "add" if name == "cudaGraphAddDependencies" else "remove"
                return [
                    f"// CUDA graph {operation} dependencies: graph: {args[0]}, "
                    f"from: {args[1]}, to: {args[2]}, count: {args[3]}"
                ]
        elif name == "cudaGraphNodeFindInClone":
            if len(node.args) >= 3:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    f"// CUDA graph node find in clone: output: {output}, "
                    f"original node: {args[1]}, clone graph: {args[2]}"
                ]
        elif name == "cudaGraphNodeGetType":
            if len(node.args) >= 2:
                output = self.format_runtime_pointer_target(node.args[1])
                return [
                    f"// CUDA graph node get type: node: {args[0]}, "
                    f"output: {output}"
                ]
        elif name == "cudaGraphGetId":
            if len(node.args) >= 2:
                output = self.format_runtime_pointer_target(node.args[1])
                return [f"// CUDA graph get id: graph: {args[0]}, output: {output}"]
        elif name == "cudaGraphGetEdges":
            if len(node.args) >= 5:
                from_output = self.format_runtime_pointer_target(node.args[1])
                to_output = self.format_runtime_pointer_target(node.args[2])
                edge_data = self.format_runtime_pointer_target(node.args[3])
                count_output = self.format_runtime_pointer_target(node.args[4])
                return [
                    f"// CUDA graph get edges: graph: {args[0]}, "
                    f"from output: {from_output}, to output: {to_output}, "
                    f"edge data: {edge_data}, count output: {count_output}"
                ]
        elif name in {"cudaGraphGetNodes", "cudaGraphGetRootNodes"}:
            if len(node.args) >= 3:
                node_output = self.format_runtime_pointer_target(node.args[1])
                count_output = self.format_runtime_pointer_target(node.args[2])
                node_kind = "root nodes" if name == "cudaGraphGetRootNodes" else "nodes"
                return [
                    f"// CUDA graph get {node_kind}: graph: {args[0]}, "
                    f"nodes output: {node_output}, count output: {count_output}"
                ]
        elif name == "cudaGraphNodeGetContainingGraph":
            if len(node.args) >= 2:
                output = self.format_runtime_pointer_target(node.args[1])
                return [
                    f"// CUDA graph node get containing graph: node: {args[0]}, "
                    f"output: {output}"
                ]
        elif name in {
            "cudaGraphNodeGetDependencies",
            "cudaGraphNodeGetDependentNodes",
        }:
            if len(node.args) >= 4:
                nodes_output = self.format_runtime_pointer_target(node.args[1])
                edge_data = self.format_runtime_pointer_target(node.args[2])
                count_output = self.format_runtime_pointer_target(node.args[3])
                if name == "cudaGraphNodeGetDependencies":
                    return [
                        f"// CUDA graph node get dependencies: node: {args[0]}, "
                        f"dependencies output: {nodes_output}, "
                        f"edge data: {edge_data}, count output: {count_output}"
                    ]
                return [
                    f"// CUDA graph node get dependent nodes: node: {args[0]}, "
                    f"dependent nodes output: {nodes_output}, "
                    f"edge data: {edge_data}, count output: {count_output}"
                ]
        elif name in {"cudaGraphNodeGetEnabled", "cudaGraphNodeSetEnabled"}:
            if len(node.args) >= 3:
                if name == "cudaGraphNodeGetEnabled":
                    output = self.format_runtime_pointer_target(node.args[2])
                    return [
                        f"// CUDA graph node get enabled: exec: {args[0]}, "
                        f"node: {args[1]}, output: {output}"
                    ]
                return [
                    f"// CUDA graph node set enabled: exec: {args[0]}, "
                    f"node: {args[1]}, enabled: {args[2]}"
                ]
        elif name in {"cudaGraphNodeGetLocalId", "cudaGraphNodeGetToolsId"}:
            if len(node.args) >= 2:
                output = self.format_runtime_pointer_target(node.args[1])
                id_kind = (
                    "local id" if name == "cudaGraphNodeGetLocalId" else "tools id"
                )
                return [
                    f"// CUDA graph node get {id_kind}: node: {args[0]}, "
                    f"output: {output}"
                ]
        elif name in {
            "cudaGraphEventRecordNodeGetEvent",
            "cudaGraphEventWaitNodeGetEvent",
        }:
            if len(node.args) >= 2:
                node_kind = (
                    "record" if name == "cudaGraphEventRecordNodeGetEvent" else "wait"
                )
                output = self.format_runtime_pointer_target(node.args[1])
                return [
                    f"// CUDA graph event {node_kind} node get event: "
                    f"node: {args[0]}, output: {output}"
                ]
        elif name in {
            "cudaGraphEventRecordNodeSetEvent",
            "cudaGraphEventWaitNodeSetEvent",
        }:
            if len(args) >= 2:
                node_kind = (
                    "record" if name == "cudaGraphEventRecordNodeSetEvent" else "wait"
                )
                return [
                    f"// CUDA graph event {node_kind} node set event: "
                    f"node: {args[0]}, event: {args[1]}"
                ]
        elif name == "cudaGraphChildGraphNodeGetGraph":
            if len(node.args) >= 2:
                output = self.format_runtime_pointer_target(node.args[1])
                return [
                    f"// CUDA graph child graph node get graph: "
                    f"node: {args[0]}, output: {output}"
                ]
        elif name in {"cudaGraphNodeGetParams", "cudaGraphNodeSetParams"}:
            if len(args) >= 2:
                action = "get" if name == "cudaGraphNodeGetParams" else "set"
                return [
                    f"// CUDA graph node {action} params: "
                    f"node: {args[0]}, params: {args[1]}"
                ]
        elif name in {
            "cuGraphBatchMemOpNodeGetParams",
            "cuGraphBatchMemOpNodeSetParams",
        }:
            if len(args) >= 2:
                action = "get" if name.endswith("GetParams") else "set"
                return [
                    f"// CUDA driver graph batch memory operation node {action} "
                    f"params: node: {args[0]}, params: {args[1]}"
                ]
        elif name == "cuGraphExecBatchMemOpNodeSetParams":
            if len(args) >= 3:
                return [
                    "// CUDA driver graph exec set batch memory operation node "
                    f"params: exec: {args[0]}, node: {args[1]}, params: {args[2]}"
                ]
        elif name in {
            "cudaGraphKernelNodeGetAttribute",
            "cudaGraphKernelNodeSetAttribute",
        }:
            if len(node.args) >= 3:
                action = "get" if name.endswith("GetAttribute") else "set"
                value_label = "output" if action == "get" else "value"
                value = (
                    self.format_runtime_pointer_target(node.args[2])
                    if action == "get"
                    else args[2]
                )
                return [
                    f"// CUDA graph kernel node {action} attribute: "
                    f"node: {args[0]}, attribute: {args[1]}, "
                    f"{value_label}: {value}"
                ]
        elif name == "cudaGraphKernelNodeCopyAttributes":
            if len(args) >= 2:
                return [
                    f"// CUDA graph kernel node copy attributes: "
                    f"source: {args[0]}, destination: {args[1]}"
                ]
        elif name in {
            "cudaGraphKernelNodeGetParams",
            "cudaGraphKernelNodeSetParams",
            "cudaGraphMemcpyNodeGetParams",
            "cudaGraphMemcpyNodeSetParams",
            "cudaGraphMemsetNodeGetParams",
            "cudaGraphMemsetNodeSetParams",
            "cudaGraphHostNodeGetParams",
            "cudaGraphHostNodeSetParams",
        }:
            if len(args) >= 2:
                node_kind = {
                    "cudaGraphKernelNodeGetParams": "kernel",
                    "cudaGraphKernelNodeSetParams": "kernel",
                    "cudaGraphMemcpyNodeGetParams": "memcpy",
                    "cudaGraphMemcpyNodeSetParams": "memcpy",
                    "cudaGraphMemsetNodeGetParams": "memset",
                    "cudaGraphMemsetNodeSetParams": "memset",
                    "cudaGraphHostNodeGetParams": "host",
                    "cudaGraphHostNodeSetParams": "host",
                }[name]
                action = "get" if name.endswith("GetParams") else "set"
                return [
                    f"// CUDA graph {node_kind} node {action} params: "
                    f"node: {args[0]}, params: {args[1]}"
                ]
        elif name == "cudaGraphMemcpyNodeSetParams1D":
            if len(args) >= 5:
                return [
                    f"// CUDA graph set memcpy 1D node params: "
                    f"node: {args[0]}, dst: {args[1]}, src: {args[2]}, "
                    f"byte count: {args[3]}, kind: {args[4]}"
                ]
        elif name in {
            "cudaGraphMemcpyNodeSetParamsFromSymbol",
            "cudaGraphMemcpyNodeSetParamsToSymbol",
        }:
            if len(args) >= 6:
                if name == "cudaGraphMemcpyNodeSetParamsFromSymbol":
                    return [
                        "// CUDA graph set memcpy-from-symbol node params: "
                        f"node: {args[0]}, dst: {args[1]}, "
                        f"symbol: {args[2]}, byte count: {args[3]}, "
                        f"offset: {args[4]}, kind: {args[5]}"
                    ]
                return [
                    "// CUDA graph set memcpy-to-symbol node params: "
                    f"node: {args[0]}, symbol: {args[1]}, src: {args[2]}, "
                    f"byte count: {args[3]}, offset: {args[4]}, "
                    f"kind: {args[5]}"
                ]
        elif name in {
            "cudaGraphExternalSemaphoresSignalNodeGetParams",
            "cudaGraphExternalSemaphoresSignalNodeSetParams",
            "cudaGraphExternalSemaphoresWaitNodeGetParams",
            "cudaGraphExternalSemaphoresWaitNodeSetParams",
        }:
            if len(args) >= 2:
                node_kind = "signal" if "Signal" in name else "wait"
                action = "get" if name.endswith("GetParams") else "set"
                return [
                    f"// CUDA graph external semaphore {node_kind} "
                    f"node {action} params: node: {args[0]}, params: {args[1]}"
                ]
        elif name == "cudaGraphExecUpdate":
            if len(node.args) >= 4:
                error_output = self.format_runtime_pointer_target(node.args[2])
                result_output = self.format_runtime_pointer_target(node.args[3])
                return [
                    f"// CUDA graph exec update: exec: {args[0]}, "
                    f"graph: {args[1]}, error node output: {error_output}, "
                    f"result output: {result_output}"
                ]
            if len(args) >= 3:
                return [
                    f"// CUDA graph exec update: exec: {args[0]}, "
                    f"graph: {args[1]}, result info: {args[2]}"
                ]
        elif name in {"cudaGraphExecGetFlags", "cudaGraphExecGetId"}:
            if len(node.args) >= 2:
                output = self.format_runtime_pointer_target(node.args[1])
                if name == "cudaGraphExecGetFlags":
                    return [
                        f"// CUDA graph exec get flags: exec: {args[0]}, "
                        f"output: {output}"
                    ]
                return [f"// CUDA graph exec get id: exec: {args[0]}, output: {output}"]
        elif name in {
            "cudaGraphExecKernelNodeSetParams",
            "cudaGraphExecMemcpyNodeSetParams",
            "cudaGraphExecMemsetNodeSetParams",
            "cudaGraphExecHostNodeSetParams",
        }:
            if len(args) >= 3:
                node_kind = {
                    "cudaGraphExecKernelNodeSetParams": "kernel",
                    "cudaGraphExecMemcpyNodeSetParams": "memcpy",
                    "cudaGraphExecMemsetNodeSetParams": "memset",
                    "cudaGraphExecHostNodeSetParams": "host",
                }[name]
                return [
                    f"// CUDA graph exec set {node_kind} node params: "
                    f"exec: {args[0]}, node: {args[1]}, params: {args[2]}"
                ]
        elif name == "cudaGraphExecMemcpyNodeSetParams1D":
            if len(args) >= 6:
                return [
                    f"// CUDA graph exec set memcpy 1D node params: "
                    f"exec: {args[0]}, node: {args[1]}, dst: {args[2]}, "
                    f"src: {args[3]}, byte count: {args[4]}, kind: {args[5]}"
                ]
        elif name in {
            "cudaGraphExecMemcpyNodeSetParamsFromSymbol",
            "cudaGraphExecMemcpyNodeSetParamsToSymbol",
        }:
            if len(args) >= 7:
                if name == "cudaGraphExecMemcpyNodeSetParamsFromSymbol":
                    return [
                        "// CUDA graph exec set memcpy-from-symbol node params: "
                        f"exec: {args[0]}, node: {args[1]}, dst: {args[2]}, "
                        f"symbol: {args[3]}, byte count: {args[4]}, "
                        f"offset: {args[5]}, kind: {args[6]}"
                    ]
                return [
                    "// CUDA graph exec set memcpy-to-symbol node params: "
                    f"exec: {args[0]}, node: {args[1]}, symbol: {args[2]}, "
                    f"src: {args[3]}, byte count: {args[4]}, "
                    f"offset: {args[5]}, kind: {args[6]}"
                ]
        elif name == "cudaGraphExecNodeSetParams":
            if len(args) >= 3:
                return [
                    f"// CUDA graph exec set node params: "
                    f"exec: {args[0]}, node: {args[1]}, params: {args[2]}"
                ]
        elif name in {
            "cudaGraphExecExternalSemaphoresSignalNodeSetParams",
            "cudaGraphExecExternalSemaphoresWaitNodeSetParams",
        }:
            if len(args) >= 3:
                node_kind = "signal" if "Signal" in name else "wait"
                return [
                    f"// CUDA graph exec set external semaphore {node_kind} "
                    f"node params: exec: {args[0]}, node: {args[1]}, "
                    f"params: {args[2]}"
                ]
        elif name in {
            "cudaGraphExecEventRecordNodeSetEvent",
            "cudaGraphExecEventWaitNodeSetEvent",
        }:
            if len(args) >= 3:
                node_kind = (
                    "record"
                    if name == "cudaGraphExecEventRecordNodeSetEvent"
                    else "wait"
                )
                return [
                    f"// CUDA graph exec set event {node_kind} node event: "
                    f"exec: {args[0]}, node: {args[1]}, event: {args[2]}"
                ]
        elif name == "cudaGraphExecChildGraphNodeSetParams":
            if len(args) >= 3:
                return [
                    f"// CUDA graph exec set child graph node params: "
                    f"exec: {args[0]}, node: {args[1]}, child graph: {args[2]}"
                ]
        elif name == "cudaGraphDestroyNode":
            if args:
                return [f"// CUDA graph destroy node: {args[0]}"]
        elif name == "cudaGraphDestroy":
            if args:
                return [f"// CUDA graph destroy: {args[0]}"]
        elif name in {"cudaLaunchKernelExC", "cudaLaunchKernelEx"}:
            if len(node.args) >= 2:
                config = self.format_runtime_pointer_target(node.args[0])
                function = self.format_cuda_kernel_function_reference(node.args[1])
                launch_args = node.args[2:]
                if name == "cudaLaunchKernelExC" and launch_args:
                    launch_args = self.resolve_packed_launch_args([launch_args[0]])

                comment = (
                    f"// CUDA launch kernel ex: config: {config}, "
                    f"function: {function}"
                )
                if launch_args:
                    args_str = ", ".join(
                        self.format_kernel_launch_arg(arg) for arg in launch_args
                    )
                    comment += f", args: {args_str}"
                return [comment]
        elif name == "cuGraphDestroyNode":
            if args:
                return [f"// CUDA driver graph destroy node: {args[0]}"]
        elif name == "cuGraphDestroy":
            if args:
                return [f"// CUDA driver graph destroy: {args[0]}"]

        return None

    def format_runtime_pointer_target(self, arg):
        if isinstance(arg, CastNode):
            return self.format_runtime_pointer_target(arg.expression)
        if isinstance(arg, UnaryOpNode) and arg.op == "&":
            return self.visit(arg.operand)
        return self.visit(arg)

    def format_cuda_kernel_function_reference(self, arg):
        if isinstance(arg, CastNode):
            return self.format_cuda_kernel_function_reference(arg.expression)
        if isinstance(arg, UnaryOpNode) and arg.op == "&":
            return self.visit(arg.operand)
        return self.visit(arg)

    def format_statement_fragment(self, stmt):
        if stmt is None:
            return ""
        if isinstance(stmt, list):
            return ", ".join(self.format_statement_fragment(item) for item in stmt)
        if isinstance(stmt, VariableNode):
            self.register_variable_type(stmt.name, stmt.vtype)
            var_type = self.convert_cuda_variable_type_to_crossgl(stmt.vtype, stmt.name)
            self.register_vector1_name(stmt.name, stmt.vtype)
            name = self.register_identifier_name(stmt.name)
            if stmt.value:
                value = self.visit(stmt.value)
                return f"var {name}: {var_type} = {value}"
            return f"var {name}: {var_type}"
        if isinstance(stmt, AssignmentNode):
            left = self.visit(stmt.left)
            right = self.visit(stmt.right)
            return f"{left} {stmt.operator} {right}"

        result = self.visit(stmt)
        return result if isinstance(result, str) else ""

    def visit_ShaderNode(self, node):
        """Render a CUDA shader/program AST as a CrossGL shader block."""
        self.emit("// CUDA to CrossGL conversion")

        if hasattr(node, "includes") and node.includes:
            for include in node.includes:
                self.visit(include)
            self.emit("")

        if hasattr(node, "structs") and node.structs:
            for struct in node.structs:
                if self.is_generated_matrix_helper_struct(struct):
                    continue
                self.visit(struct)
                self.emit("")

        if hasattr(node, "typedefs") and node.typedefs:
            for alias in node.typedefs:
                self.visit(alias)
            self.emit("")

        if hasattr(node, "global_variables") and node.global_variables:
            for var in node.global_variables:
                self.visit(var)
                self.emit("")

        if hasattr(node, "functions") and node.functions:
            for func in node.functions:
                if getattr(func, "body", None) is None:
                    continue
                if self.is_generated_matrix_helper_function(func):
                    continue
                self.emit(f"// Function: {func.name}")
                self.visit(func)
                self.emit("")

        if hasattr(node, "kernels") and node.kernels:
            for kernel in node.kernels:
                if getattr(kernel, "body", None) is None:
                    continue
                self.emit(f"// Kernel: {kernel.name}")
                self.visit_kernel_as_compute_shader(kernel)
                self.emit("")

    def visit_PreprocessorNode(self, node):
        if node.directive == "include":
            if "cuda_runtime.h" in node.content:
                self.emit("// CUDA runtime functionality built-in")
            elif "device_launch_parameters.h" in node.content:
                self.emit("// Device parameters built-in")
            else:
                self.emit(f"// {node.directive} {node.content}")
        else:
            self.emit(f"// {node.directive} {node.content}")

    def visit_StructNode(self, node):
        struct_name = self.convert_cuda_record_name_to_crossgl(node.name)
        self.emit(f"struct {struct_name} {{")
        self.indent_level += 1

        for member in node.members:
            member_type = self.convert_cuda_struct_member_type_to_crossgl(
                node.name, member.vtype, member.name
            )
            member_name = self.sanitize_identifier_name(member.name)
            self.emit(f"{member_type} {member_name};")

        self.indent_level -= 1
        self.emit("};")

    def visit_EnumNode(self, node):
        name = node.name or self.next_anonymous_enum_name()
        underlying = getattr(node, "underlying_type", None)
        suffix = (
            f" : {self.convert_cuda_type_to_crossgl(underlying)}" if underlying else ""
        )
        self.emit(f"enum {name}{suffix} {{")
        self.indent_level += 1

        members = getattr(node, "members", None) or getattr(node, "variants", [])
        for member in members:
            if isinstance(member, tuple):
                member_name, member_value = member
            else:
                member_name = getattr(member, "name", str(member))
                member_value = getattr(member, "value", None)

            if member_value is not None:
                value = self.visit(member_value)
                member_name = self.sanitize_identifier_name(member_name)
                self.emit(f"{member_name} = {value},")
            else:
                self.emit(f"{self.sanitize_identifier_name(member_name)},")

        self.indent_level -= 1
        self.emit("};")

    def next_anonymous_enum_name(self):
        name = f"anonymous_enum_{self.anonymous_enum_count}"
        self.anonymous_enum_count += 1
        return name

    def visit_FunctionNode(self, node):
        """Render a CUDA function node as a CrossGL function."""
        return_type = self.convert_cuda_type_to_crossgl(node.return_type)
        function_name = self.convert_cuda_function_name_to_crossgl(node.name)
        params = []

        self.push_resource_object_hint_scope(
            self.collect_resource_object_type_hints(
                node, self.collect_declared_variable_names(node)
            )
        )
        self.push_identifier_name_scope()
        self.push_vector1_name_scope()
        try:
            for index, param in enumerate(node.params):
                param_type = self.convert_cuda_variable_type_to_crossgl(
                    param.vtype, param.name
                )
                self.register_vector1_name(param.name, param.vtype)
                param_name = self.crossgl_parameter_name(param, index)
                params.append(f"{param_type} {param_name}")

            param_str = ", ".join(params)
            self.emit(f"{return_type} {function_name}({param_str}) {{")

            self.indent_level += 1
            self.push_packed_argument_scope()
            self.push_type_alias_scope()
            self.push_unique_ptr_scope()
            self.push_cooperative_group_scope()
            self.push_cuda_async_sync_scope()
            self.push_variable_type_scope()
            self.push_warp_mask_value_scope()
            for param in node.params:
                self.register_variable_type(param.name, param.vtype)
                self.register_unique_ptr_name(param.name, param.vtype)
                self.register_cooperative_group_parameter(param)
                self.register_cuda_async_sync_parameter(param)
            try:
                for stmt in node.body:
                    self.emit_statement(stmt)
            finally:
                self.pop_warp_mask_value_scope()
                self.pop_variable_type_scope()
                self.pop_cuda_async_sync_scope()
                self.pop_cooperative_group_scope()
                self.pop_unique_ptr_scope()
                self.pop_type_alias_scope()
                self.pop_packed_argument_scope()
                self.indent_level -= 1
        finally:
            self.pop_vector1_name_scope()
            self.pop_identifier_name_scope()
            self.pop_resource_object_hint_scope()

        self.emit("}")

    def visit_kernel_as_compute_shader(self, kernel):
        for attribute in getattr(kernel, "attributes", []) or []:
            attribute_text = str(attribute)
            if attribute_text.startswith("__launch_bounds__"):
                bounds = attribute_text[len("__launch_bounds__") :]
                self.emit(f"// CUDA launch bounds: {bounds}")
            elif attribute_text.startswith("__cluster_dims__"):
                dims = attribute_text[len("__cluster_dims__") :]
                self.emit(f"// CUDA cluster dims: {dims}")
            elif attribute_text.startswith("__block_size__"):
                size = attribute_text[len("__block_size__") :]
                self.emit(f"// CUDA block size: {size}")
        for param in kernel.params:
            if "__grid_constant__" in str(param.vtype).split():
                self.emit(f"// CUDA grid constant parameter: {param.name}")

        self.emit("@compute")
        self.emit("@workgroup_size(1, 1, 1)  // Default workgroup size")

        params = []
        self.push_resource_object_hint_scope(
            self.collect_resource_object_type_hints(
                kernel, self.collect_declared_variable_names(kernel)
            )
        )
        self.push_identifier_name_scope()
        self.push_vector1_name_scope()
        try:
            for index, param in enumerate(kernel.params):
                self.register_vector1_name(param.name, param.vtype)
                param_name = self.crossgl_parameter_name(param, index)
                if "*" in param.vtype:
                    element_type = self.convert_cuda_pointer_element_type(param.vtype)
                    element_type = self.crossgl_safe_generic_type_argument(element_type)
                    storage_type = self.format_crossgl_type_syntax(
                        f"array<{element_type}>"
                    )
                    binding = self.global_resource_binding_count + len(params)
                    params.append(
                        f"@group(0) @binding({binding}) var<storage, read_write> {param_name}: {storage_type}"
                    )
                else:
                    param_type = self.convert_cuda_variable_type_to_crossgl(
                        param.vtype, param.name
                    )
                    param_type = self.format_crossgl_type_syntax(param_type)
                    params.append(f"{param_type} {param_name}")

            self.emit(f"fn {kernel.name}(")
            self.indent_level += 1
            for i, param in enumerate(params):
                if i == len(params) - 1:
                    self.emit(f"{param}")
                else:
                    self.emit(f"{param},")
            self.indent_level -= 1
            self.emit(") {")

            self.indent_level += 1
            self.emit("let thread_id = gl_GlobalInvocationID;")
            self.emit("let block_id = gl_WorkGroupID;")
            self.emit("let thread_local_id = gl_LocalInvocationID;")
            self.emit("let block_dim = gl_WorkGroupSize;")
            self.emit("")

            self.push_packed_argument_scope()
            self.push_type_alias_scope()
            self.push_unique_ptr_scope()
            self.push_cooperative_group_scope()
            self.push_cuda_async_sync_scope()
            self.push_variable_type_scope()
            self.push_warp_mask_value_scope()
            for param in kernel.params:
                self.register_variable_type(param.name, param.vtype)
                self.register_unique_ptr_name(param.name, param.vtype)
                self.register_cooperative_group_parameter(param)
                self.register_cuda_async_sync_parameter(param)
            try:
                for stmt in kernel.body:
                    self.emit_statement(stmt)
            finally:
                self.pop_warp_mask_value_scope()
                self.pop_variable_type_scope()
                self.pop_cuda_async_sync_scope()
                self.pop_cooperative_group_scope()
                self.pop_unique_ptr_scope()
                self.pop_type_alias_scope()
                self.pop_packed_argument_scope()
                self.indent_level -= 1
        finally:
            self.pop_vector1_name_scope()
            self.pop_identifier_name_scope()
            self.pop_resource_object_hint_scope()

        self.emit("}")

    def visit_KernelLaunchNode(self, node):
        for comment in self.format_kernel_launch_comments(node):
            self.emit(comment)

    def format_kernel_launch_comments(self, node):
        kernel_name = self.visit(node.kernel_name)
        config = [self.visit(node.blocks), self.visit(node.threads)]
        if node.shared_mem is not None:
            config.append(self.visit(node.shared_mem))
        if node.stream is not None:
            config.append(self.visit(node.stream))

        comments = [f"// Kernel launch: {kernel_name}<<<{', '.join(config)}>>>()"]
        if node.args:
            args = self.resolve_packed_launch_args(node.args)
            args_str = ", ".join([self.format_kernel_launch_arg(arg) for arg in args])
            comments.append(f"// Arguments: {args_str}")
        return comments

    def visit_VariableNode(self, node):
        self.register_variable_type(node.name, node.vtype)

        cuda_async_sync = self.cuda_async_sync_declaration_metadata(node)
        if cuda_async_sync is not None:
            self.register_cuda_async_sync_name(node.name, cuda_async_sync)
            self.emit(
                self.format_cuda_async_sync_declaration(node.name, cuda_async_sync)
            )
            return

        cooperative_group = self.cooperative_group_declaration_metadata(node)
        if cooperative_group is not None:
            group_kind = cooperative_group["kind"]
            self.register_cooperative_group_name(node.name, cooperative_group)
            if group_kind == "thread_block":
                self.emit(
                    f"// cooperative_groups thread_block {node.name} maps to the current workgroup"
                )
            elif (
                group_kind == "thread_block_tile"
                and cooperative_group.get("tile_size")
                and cooperative_group.get("parent_kind") == "thread_block"
            ):
                self.emit(
                    f"// cooperative_groups thread_block_tile<{cooperative_group['tile_size']}> "
                    f"{node.name} maps to a tiled partition of the current workgroup"
                )
            else:
                self.emit(
                    f"// cooperative_groups {group_kind} for {node.name} not directly supported in CrossGL"
                )
            return

        var_type = self.convert_cuda_variable_type_to_crossgl(node.vtype, node.name)
        self.register_vector1_name(node.name, node.vtype)
        output_name = self.register_identifier_name(node.name)

        self.register_packed_argument_list(node)
        self.register_unique_ptr_name(node.name, node.vtype)
        if node.value:
            runtime_expression = self.format_cuda_runtime_expression(node.value)
            if runtime_expression is not None:
                comments, value = runtime_expression
                for comment in comments:
                    self.emit(comment)
                self.register_warp_mask_value(node.name, output_name, value)
                self.emit(f"var {output_name}: {var_type} = {value};")
                return

            value = self.visit(node.value)
            self.register_warp_mask_value(node.name, output_name, value)
            self.emit(f"var {output_name}: {var_type} = {value};")
        else:
            self.register_warp_mask_value(node.name, output_name, None)
            self.emit(f"var {output_name}: {var_type};")

    def push_packed_argument_scope(self):
        self.packed_argument_scopes.append({})

    def pop_packed_argument_scope(self):
        if self.packed_argument_scopes:
            self.packed_argument_scopes.pop()

    def push_unique_ptr_scope(self):
        self.unique_ptr_scopes.append(set())

    def pop_unique_ptr_scope(self):
        if len(self.unique_ptr_scopes) > 1:
            self.unique_ptr_scopes.pop()

    def push_cooperative_group_scope(self):
        self.cooperative_group_scopes.append({})

    def pop_cooperative_group_scope(self):
        if len(self.cooperative_group_scopes) > 1:
            self.cooperative_group_scopes.pop()

    def push_cuda_async_sync_scope(self):
        self.cuda_async_sync_scopes.append({})

    def pop_cuda_async_sync_scope(self):
        if len(self.cuda_async_sync_scopes) > 1:
            self.cuda_async_sync_scopes.pop()

    def push_variable_type_scope(self):
        self.variable_type_scopes.append({})

    def pop_variable_type_scope(self):
        if len(self.variable_type_scopes) > 1:
            self.variable_type_scopes.pop()

    def push_warp_mask_value_scope(self):
        self.warp_mask_value_scopes.append({})

    def pop_warp_mask_value_scope(self):
        if len(self.warp_mask_value_scopes) > 1:
            self.warp_mask_value_scopes.pop()

    def register_warp_mask_value(self, raw_name, output_name, value):
        if not self.warp_mask_value_scopes:
            self.warp_mask_value_scopes.append({})

        names = {name for name in (raw_name, output_name) if name}
        if not names:
            return

        if self.is_full_or_active_warp_mask(value):
            resolved_value = self.resolve_warp_mask_expression(value)
            for name in names:
                self.warp_mask_value_scopes[-1][name] = resolved_value
            return

        for name in names:
            self.warp_mask_value_scopes[-1].pop(name, None)

    def resolve_warp_mask_expression(self, mask):
        text = str(mask).strip()
        seen = set()

        while text and text not in seen:
            seen.add(text)
            replacement = None
            for scope in reversed(self.warp_mask_value_scopes):
                if text in scope:
                    replacement = scope[text]
                    break
            if replacement is None:
                return text
            text = str(replacement).strip()

        return text

    def register_variable_type(self, name, type_name):
        if name:
            self.variable_type_scopes[-1][name] = type_name

    def lookup_variable_type(self, name):
        for scope in reversed(self.variable_type_scopes):
            if name in scope:
                return scope[name]
        return None

    def register_cuda_async_sync_parameter(self, param):
        metadata = self.cuda_async_sync_metadata_from_type(param.vtype)
        if metadata is not None:
            self.register_cuda_async_sync_name(param.name, metadata)

    def register_cuda_async_sync_name(self, name, metadata):
        if name:
            self.cuda_async_sync_scopes[-1][name] = metadata

    def lookup_cuda_async_sync_metadata(self, name):
        for scope in reversed(self.cuda_async_sync_scopes):
            if name in scope:
                return scope[name]
        return None

    def register_cooperative_group_name(self, name, group_metadata):
        if name:
            self.cooperative_group_scopes[-1][name] = group_metadata

    def register_cooperative_group_parameter(self, param):
        metadata = self.cooperative_group_metadata_from_type(param.vtype)
        if metadata is not None:
            self.register_cooperative_group_name(param.name, metadata)

    def lookup_cooperative_group_name(self, name):
        metadata = self.lookup_cooperative_group_metadata(name)
        return metadata["kind"] if metadata is not None else None

    def lookup_cooperative_group_metadata(self, name):
        for scope in reversed(self.cooperative_group_scopes):
            if name in scope:
                group_metadata = scope[name]
                if isinstance(group_metadata, str):
                    return {"kind": group_metadata}
                return group_metadata
        return None

    def push_type_alias_scope(self):
        self.type_alias_scopes.append({})

    def pop_type_alias_scope(self):
        if len(self.type_alias_scopes) > 1:
            self.type_alias_scopes.pop()

    def push_vector1_name_scope(self):
        self.vector1_name_scopes.append({})

    def pop_vector1_name_scope(self):
        if len(self.vector1_name_scopes) > 1:
            self.vector1_name_scopes.pop()

    def register_vector1_name(self, name, type_name):
        if not name:
            return
        self.vector1_name_scopes[-1][name] = (
            self.cuda_vector1_scalar_type(type_name) is not None
        )

    def is_vector1_name(self, name):
        if not isinstance(name, str):
            return False
        for scope in reversed(self.vector1_name_scopes):
            if name in scope:
                return scope[name]
        return False

    def cuda_vector1_scalar_type(self, type_name):
        type_name = self.strip_type_qualifiers(type_name)
        if "*" in type_name or self.has_array_suffix(type_name):
            return None
        return self.VECTOR1_TYPE_MAPPING.get(type_name)

    def register_type_alias(self, name, alias_type):
        self.type_alias_scopes[-1][name] = alias_type

    def resolve_type_alias(self, type_name):
        type_name = self.strip_type_qualifiers(type_name)
        for scope in reversed(self.type_alias_scopes):
            if type_name in scope:
                return scope[type_name]
        return type_name

    def register_unique_ptr_name(self, name, type_name):
        if self.is_unique_ptr_type_name(type_name):
            self.unique_ptr_scopes[-1].add(name)

    def is_unique_ptr_expression(self, expr):
        if not isinstance(expr, str):
            return False
        return any(expr in scope for scope in reversed(self.unique_ptr_scopes))

    def register_packed_argument_list(self, node):
        if not self.packed_argument_scopes:
            return
        if self.is_packed_argument_list(node):
            self.packed_argument_scopes[-1][node.name] = (
                self.get_initializer_list_elements(node.value)
            )

    def is_packed_argument_list(self, node):
        if self.get_initializer_list_elements(getattr(node, "value", None)) is None:
            return False

        compact_type = getattr(node, "vtype", "").replace(" ", "")
        return compact_type in {"void*[]", "void**"}

    def get_initializer_list_elements(self, value):
        if isinstance(value, InitializerListNode):
            return value.elements
        if isinstance(value, CastNode) and isinstance(
            value.expression, InitializerListNode
        ):
            return value.expression.elements
        return None

    def resolve_packed_launch_args(self, args):
        if len(args) != 1:
            return args

        compound_elements = self.get_packed_compound_literal_elements(args[0])
        if compound_elements is not None:
            return compound_elements

        packed_arg_name = self.get_packed_argument_name(args[0])
        if packed_arg_name is None:
            return args

        for scope in reversed(self.packed_argument_scopes):
            if packed_arg_name in scope:
                return scope[packed_arg_name]

        return args

    def get_packed_argument_name(self, arg):
        if isinstance(arg, str):
            return arg
        if isinstance(arg, CastNode):
            return self.get_packed_argument_name(arg.expression)
        return None

    def get_packed_compound_literal_elements(self, arg):
        if not isinstance(arg, CastNode):
            return None

        compact_type = arg.target_type.replace(" ", "")
        if compact_type not in {"void*[]", "void**"}:
            return None

        return self.get_initializer_list_elements(arg.expression)

    def format_kernel_launch_arg(self, arg):
        if isinstance(arg, UnaryOpNode) and arg.op == "&":
            return self.visit(arg.operand)
        return self.visit(arg)

    def visit_SharedMemoryNode(self, node):
        cuda_async_sync = self.cuda_async_sync_metadata_from_type(node.vtype)
        if cuda_async_sync is not None:
            self.register_cuda_async_sync_name(node.name, cuda_async_sync)
            self.emit(
                self.format_cuda_async_sync_declaration(node.name, cuda_async_sync)
            )
            return

        # Convert to workgroup memory in CrossGL
        var_type = self.convert_cuda_variable_type_to_crossgl(node.vtype, node.name)
        self.register_vector1_name(node.name, node.vtype)
        if getattr(node, "is_dynamic_shared_memory", False):
            self.emit(
                f"// CUDA dynamic shared memory: {node.name} uses launch-time "
                "shared memory size"
            )
        if node.size:
            size = self.format_crossgl_array_extent(self.visit(node.size))
            self.emit(f"var<workgroup> {node.name}: array<{var_type}, {size}>;")
        else:
            self.emit(f"var<workgroup> {node.name}: {var_type};")

    def visit_ConstantMemoryNode(self, node):
        # Convert to uniform buffer in CrossGL
        var_type = self.convert_cuda_variable_type_to_crossgl(node.vtype, node.name)
        self.register_vector1_name(node.name, node.vtype)
        binding = self.allocate_global_resource_binding()
        if node.value:
            value = self.visit(node.value)
            self.emit(
                f"@group(0) @binding({binding}) var<uniform> {node.name}: {var_type} = {value};"
            )
        else:
            self.emit(
                f"@group(0) @binding({binding}) var<uniform> {node.name}: {var_type};"
            )

    def allocate_global_resource_binding(self):
        binding = self.global_resource_binding_count
        self.global_resource_binding_count += 1
        return binding

    def visit_AssignmentNode(self, node):
        left = self.visit(node.left)
        operator = getattr(node, "operator", "=")
        runtime_expression = (
            self.format_cuda_runtime_expression(node.right) if operator == "=" else None
        )
        if runtime_expression is not None:
            comments, value = runtime_expression
            for comment in comments:
                self.emit(comment)
            self.register_warp_mask_assignment(node.left, left, operator, value)
            self.emit(f"{left} = {value};")
            return None

        right = self.visit(node.right)
        self.register_warp_mask_assignment(node.left, left, operator, right)
        return f"{left} {operator} {right}"

    def register_warp_mask_assignment(self, raw_left, output_left, operator, value):
        raw_name = raw_left if isinstance(raw_left, str) else None
        if raw_name is None:
            return

        assigned_value = value if operator == "=" else None
        self.register_warp_mask_value(raw_name, output_left, assigned_value)

    def visit_BinaryOpNode(self, node):
        left = self.visit(node.left)
        right = self.visit(node.right)
        return f"({left} {node.op} {right})"

    def visit_UnaryOpNode(self, node):
        operand = self.visit(node.operand)
        if node.op.startswith("post"):
            return f"({operand}{node.op[4:]})"
        else:
            return f"({node.op}{operand})"

    def visit_FunctionCallNode(self, node):
        if self.is_get_method_call(node):
            return self.visit(node.name.object)

        cuda_async_sync_call = self.format_cuda_async_sync_call(node)
        if cuda_async_sync_call is not None:
            return cuda_async_sync_call

        cooperative_call = self.format_cooperative_group_call(node)
        if cooperative_call is not None:
            return cooperative_call

        raw_name = node.name if isinstance(node.name, str) else self.visit(node.name)
        runtime_expression = self.format_cuda_runtime_inline_expression(node)
        if runtime_expression is not None:
            return runtime_expression

        if raw_name == "lambda":
            return self.format_lambda_call(node.args)
        if raw_name == "cuda_fold_expression":
            return self.format_cuda_fold_expression(node.args)
        if raw_name == "cuda_comma_expression":
            return self.format_cuda_comma_expression(node.args)

        args = [self.visit(arg) for arg in node.args]
        args_str = ", ".join(args)
        make_unique = self.format_make_unique_call(raw_name, args)
        if make_unique is not None:
            return make_unique

        unique_ptr_init = self.format_unique_ptr_constructor_call(raw_name, args)
        if unique_ptr_init is not None:
            return unique_ptr_init

        if self.is_cuda_intrinsic_override_function(raw_name):
            integer_intrinsic = self.format_cuda_integer_intrinsic_call(raw_name, args)
            if integer_intrinsic is not None:
                return integer_intrinsic

        if self.is_user_defined_function(raw_name):
            return f"{raw_name}({args_str})"

        load_cache_intrinsic = self.format_cuda_load_cache_intrinsic_call(
            raw_name, node.args, args
        )
        if load_cache_intrinsic is not None:
            return load_cache_intrinsic

        memory_fence = self.format_cuda_memory_fence_call(raw_name, args)
        if memory_fence is not None:
            return memory_fence

        time_function = self.format_cuda_time_function_call(raw_name, args)
        if time_function is not None:
            return time_function

        nanosleep = self.format_cuda_nanosleep_call(raw_name, args)
        if nanosleep is not None:
            return nanosleep

        warp_intrinsic = self.format_cuda_warp_intrinsic_call(raw_name, args)
        if warp_intrinsic is not None:
            return warp_intrinsic

        fp16_intrinsic = self.format_cuda_fp16_intrinsic_call(raw_name, args)
        if fp16_intrinsic is not None:
            return fp16_intrinsic

        type_cast_intrinsic = self.format_cuda_type_cast_intrinsic_call(raw_name, args)
        if type_cast_intrinsic is not None:
            return type_cast_intrinsic

        integer_intrinsic = self.format_cuda_integer_intrinsic_call(raw_name, args)
        if integer_intrinsic is not None:
            return integer_intrinsic

        float_intrinsic = self.format_cuda_float_intrinsic_call(raw_name, args)
        if float_intrinsic is not None:
            return float_intrinsic

        block_sync_vote = self.format_cuda_block_sync_vote_call(raw_name, args)
        if block_sync_vote is not None:
            return block_sync_vote

        resource_call = self.format_cuda_resource_call(raw_name, args, node.args)
        if resource_call is not None:
            return resource_call

        func_name = self.convert_cuda_builtin_function(raw_name)
        return f"{func_name}({args_str})"

    def format_cuda_time_function_call(self, function_name, args):
        if not isinstance(function_name, str):
            return None

        normalized_name = (
            function_name[2:] if function_name.startswith("::") else function_name
        )
        if normalized_name not in {"clock", "clock64", "cuda::std::clock"}:
            return None

        display_name = normalized_name.rsplit("::", 1)[-1]
        return self.format_cuda_time_function_diagnostic(display_name, args)

    def format_cuda_time_function_diagnostic(self, function_name, args):
        args_text = ", ".join(args)
        return (
            f"(/* cuda time function {function_name}({args_text}) "
            "not directly supported in CrossGL */ 0)"
        )

    def format_cuda_comma_expression(self, args):
        args_text = ", ".join(self.visit(arg) for arg in args)
        return (
            f"(/* CUDA comma expression {args_text} "
            "not directly supported in CrossGL */ 0)"
        )

    def format_cuda_nanosleep_call(self, function_name, args):
        if not isinstance(function_name, str):
            return None

        normalized_name = (
            function_name[2:] if function_name.startswith("::") else function_name
        )
        if normalized_name != "__nanosleep":
            return None

        args_text = ", ".join(args)
        return (
            f"(/* cuda nanosleep __nanosleep({args_text}) "
            "not directly supported in CrossGL */ 0)"
        )

    def format_cuda_load_cache_intrinsic_call(
        self, function_name, raw_args, formatted_args
    ):
        if isinstance(function_name, str) and function_name.startswith("::"):
            function_name = function_name[2:]

        if function_name not in {
            "__ldca",
            "__ldcg",
            "__ldcs",
            "__ldcv",
            "__ldg",
            "__ldlu",
        }:
            return None

        if (
            len(raw_args) == 1
            and isinstance(raw_args[0], UnaryOpNode)
            and raw_args[0].op == "&"
        ):
            return self.visit(raw_args[0].operand)

        if len(raw_args) == 1:
            return f"(*{formatted_args[0]})"

        args_text = ", ".join(formatted_args)
        return (
            f"(/* cuda load cache intrinsic {function_name}({args_text}) "
            "not directly supported in CrossGL */ 0)"
        )

    def format_cuda_memory_fence_call(self, function_name, args):
        if not isinstance(function_name, str):
            return None

        normalized_name = self.resolve_namespace_alias_name(function_name)
        if normalized_name.startswith("::"):
            normalized_name = normalized_name[2:]

        if normalized_name != "cuda::atomic_thread_fence" or len(args) != 2:
            return None

        return "memoryBarrier()"

    def format_cuda_warp_intrinsic_call(self, function_name, args):
        if isinstance(function_name, str) and function_name.startswith("::"):
            function_name = function_name[2:]

        if function_name == "__activemask":
            if not args:
                return "WaveActiveBallot(true).x"
            return self.format_unsupported_cuda_warp_intrinsic(function_name, args)

        if function_name in {"__any_sync", "__all_sync", "__ballot_sync"}:
            if len(args) != 2 or not self.is_full_or_active_warp_mask(args[0]):
                return self.format_unsupported_cuda_warp_intrinsic(function_name, args)
            predicate = self.format_wave_predicate(args[1])
            if function_name == "__any_sync":
                return f"(WaveActiveAnyTrue({predicate}) ? 1 : 0)"
            if function_name == "__all_sync":
                return f"(WaveActiveAllTrue({predicate}) ? 1 : 0)"
            return f"WaveActiveBallot({predicate}).x"

        if function_name == "__match_any_sync":
            if len(args) == 2 and self.is_full_or_active_warp_mask(args[0]):
                return f"WaveMatch({args[1]}).x"
            return self.format_unsupported_cuda_warp_intrinsic(function_name, args)

        if function_name in {
            "__shfl_sync",
            "__shfl_up_sync",
            "__shfl_down_sync",
            "__shfl_xor_sync",
        }:
            if (
                len(args) not in {3, 4}
                or not self.is_full_or_active_warp_mask(args[0])
                or not self.is_full_warp_shuffle_width(args[3:] or None)
            ):
                return self.format_unsupported_cuda_warp_intrinsic(function_name, args)
            value, lane_or_delta = args[1], args[2]
            if function_name == "__shfl_sync":
                return f"WaveReadLaneAt({value}, {lane_or_delta})"
            if function_name == "__shfl_up_sync":
                return (
                    f"((WaveGetLaneIndex() >= ({lane_or_delta})) ? "
                    f"WaveReadLaneAt({value}, (WaveGetLaneIndex() - ({lane_or_delta}))) "
                    f": {value})"
                )
            if function_name == "__shfl_down_sync":
                return (
                    f"(((WaveGetLaneIndex() + ({lane_or_delta})) < 32) ? "
                    f"WaveReadLaneAt({value}, (WaveGetLaneIndex() + ({lane_or_delta}))) "
                    f": {value})"
                )
            return (
                f"WaveReadLaneAt({value}, " f"(WaveGetLaneIndex() ^ ({lane_or_delta})))"
            )

        reduce_map = {
            "__reduce_add_sync": "WaveActiveSum",
            "__reduce_min_sync": "WaveActiveMin",
            "__reduce_max_sync": "WaveActiveMax",
            "__reduce_and_sync": "WaveActiveBitAnd",
            "__reduce_or_sync": "WaveActiveBitOr",
            "__reduce_xor_sync": "WaveActiveBitXor",
        }
        if function_name in reduce_map:
            if len(args) == 2 and self.is_full_or_active_warp_mask(args[0]):
                return f"{reduce_map[function_name]}({args[1]})"
            return self.format_unsupported_cuda_warp_intrinsic(function_name, args)

        if function_name in {
            "__ballot",
            "__any",
            "__all",
            "__shfl",
            "__shfl_up",
            "__shfl_down",
            "__shfl_xor",
        }:
            return self.format_unsupported_cuda_warp_intrinsic(function_name, args)

        return None

    def is_full_warp_shuffle_width(self, width_args):
        if not width_args:
            return True
        if len(width_args) != 1:
            return False
        return self.is_full_warp_width(width_args[0])

    def is_full_or_active_warp_mask(self, mask):
        normalized = self.normalize_warp_mask_expression(
            self.resolve_warp_mask_expression(mask)
        )
        return normalized in {
            "0xffffffff",
            "0xffffffffu",
            "0xfffffffful",
            "0xffffffffffffffff",
            "0xffffffffffffffffu",
            "0xffffffffffffffffull",
            "uint_max",
            "ullong_max",
            "warp_full_mask",
            "full_mask",
            "waveactiveballot(true).x",
        }

    def is_full_warp_width(self, width):
        normalized = self.normalize_warp_mask_expression(width)
        return normalized in {"32", "32u", "warpsize", "warp_size", "warp_full_width"}

    def normalize_warp_mask_expression(self, mask):
        text = str(mask).strip()
        while text.startswith("(") and text.endswith(")"):
            inner = text[1:-1].strip()
            if not inner:
                break
            text = inner
        return text.replace(" ", "").lower()

    def format_wave_predicate(self, predicate):
        return f"({predicate} != 0)"

    def format_unsupported_cuda_warp_intrinsic(self, function_name, args):
        args_text = ", ".join(args)
        return (
            f"(/* cuda warp intrinsic {function_name}({args_text}) "
            "not directly supported in CrossGL */ 0)"
        )

    def format_cuda_block_sync_vote_call(self, function_name, args):
        if isinstance(function_name, str) and function_name.startswith("::"):
            function_name = function_name[2:]

        if function_name not in {
            "__syncthreads_count",
            "__syncthreads_and",
            "__syncthreads_or",
        }:
            return None

        return self.format_unsupported_cuda_block_sync_vote(function_name, args)

    def format_unsupported_cuda_block_sync_vote(self, function_name, args):
        args_text = ", ".join(args)
        return (
            f"(/* cuda thread block sync vote {function_name}({args_text}) "
            "not directly supported in CrossGL */ 0)"
        )

    def format_cuda_fp16_intrinsic_call(self, function_name, args):
        if isinstance(function_name, str) and function_name.startswith("::"):
            function_name = function_name[2:]

        if (
            function_name
            in {
                "__float2half",
                "__float2half_rn",
                "__float2half_rd",
                "__float2half_ru",
                "__float2half_rz",
            }
            and len(args) == 1
        ):
            return f"f16({args[0]})"
        if function_name == "__float2half2_rn" and len(args) == 1:
            return self.format_vector_constructor("vec2", [args[0], args[0]], "f16")
        if function_name == "__float22half2_rn" and len(args) == 1:
            return self.format_vector_constructor(
                "vec2",
                [
                    self.format_vector_component_access(args[0], "x"),
                    self.format_vector_component_access(args[0], "y"),
                ],
                "f16",
            )
        if function_name == "__floats2half2_rn" and len(args) == 2:
            return self.format_vector_constructor("vec2", args, "f16")
        if function_name == "__half22float2" and len(args) == 1:
            return self.format_vector_constructor(
                "vec2",
                [
                    self.format_vector_component_access(args[0], "x"),
                    self.format_vector_component_access(args[0], "y"),
                ],
                "f32",
            )
        if function_name == "__half2float" and len(args) == 1:
            return f"f32({args[0]})"
        if function_name == "__low2float" and len(args) == 1:
            return f"f32({self.format_vector_component_access(args[0], 'x')})"
        if function_name == "__high2float" and len(args) == 1:
            return f"f32({self.format_vector_component_access(args[0], 'y')})"
        if function_name == "__hadd2" and len(args) == 2:
            return f"({args[0]} + {args[1]})"
        if function_name == "__hmul2" and len(args) == 2:
            return f"({args[0]} * {args[1]})"
        if function_name == "__hfma2" and len(args) == 3:
            return f"fma({args[0]}, {args[1]}, {args[2]})"
        return None

    def format_cuda_type_cast_intrinsic_call(self, function_name, args):
        if isinstance(function_name, str) and function_name.startswith("::"):
            function_name = function_name[2:]

        bit_reinterpret_intrinsics = {
            "__double_as_longlong": "doubleBitsToLong",
            "__float_as_int": "floatBitsToInt",
            "__float_as_uint": "floatBitsToUint",
            "__int_as_float": "intBitsToFloat",
            "__longlong_as_double": "longBitsToDouble",
            "__uint_as_float": "uintBitsToFloat",
        }
        mapped_name = bit_reinterpret_intrinsics.get(function_name)
        if mapped_name is not None and len(args) == 1:
            return f"{mapped_name}({args[0]})"

        integer_conversion = self.format_cuda_float_to_integer_conversion_intrinsic(
            function_name, args
        )
        if integer_conversion is not None:
            return integer_conversion

        return None

    def format_cuda_float_to_integer_conversion_intrinsic(self, function_name, args):
        if len(args) != 1:
            return None

        conversion_intrinsics = {
            "__float2int_rd": ("i32", "floor"),
            "__float2int_rn": ("i32", "round"),
            "__float2int_ru": ("i32", "ceil"),
            "__float2int_rz": ("i32", None),
            "__float2uint_rd": ("u32", "floor"),
            "__float2uint_rn": ("u32", "round"),
            "__float2uint_ru": ("u32", "ceil"),
            "__float2uint_rz": ("u32", None),
            "__float2ll_rd": ("i64", "floor"),
            "__float2ll_rn": ("i64", "round"),
            "__float2ll_ru": ("i64", "ceil"),
            "__float2ll_rz": ("i64", None),
            "__float2ull_rd": ("u64", "floor"),
            "__float2ull_rn": ("u64", "round"),
            "__float2ull_ru": ("u64", "ceil"),
            "__float2ull_rz": ("u64", None),
            "__double2int_rd": ("i32", "floor"),
            "__double2int_rn": ("i32", "round"),
            "__double2int_ru": ("i32", "ceil"),
            "__double2int_rz": ("i32", None),
            "__double2uint_rd": ("u32", "floor"),
            "__double2uint_rn": ("u32", "round"),
            "__double2uint_ru": ("u32", "ceil"),
            "__double2uint_rz": ("u32", None),
            "__double2ll_rd": ("i64", "floor"),
            "__double2ll_rn": ("i64", "round"),
            "__double2ll_ru": ("i64", "ceil"),
            "__double2ll_rz": ("i64", None),
            "__double2ull_rd": ("u64", "floor"),
            "__double2ull_rn": ("u64", "round"),
            "__double2ull_ru": ("u64", "ceil"),
            "__double2ull_rz": ("u64", None),
        }
        conversion = conversion_intrinsics.get(function_name)
        if conversion is None:
            return None

        target_type, rounding_function = conversion
        value = args[0]
        if rounding_function is None:
            return f"{target_type}({value})"
        return f"{target_type}({rounding_function}({value}))"

    def format_cuda_integer_intrinsic_call(self, function_name, args):
        if isinstance(function_name, str) and function_name.startswith("::"):
            function_name = function_name[2:]

        if function_name == "__mul24" and len(args) == 2:
            left = self.format_cuda_signed_24_bit_operand(args[0])
            right = self.format_cuda_signed_24_bit_operand(args[1])
            return f"({left} * {right})"
        if function_name == "__umul24" and len(args) == 2:
            return f"(({args[0]} & 0x00ffffffu) * ({args[1]} & 0x00ffffffu))"
        if function_name == "__dp4a" and len(args) == 3:
            return f"(dot4I8Packed({args[0]}, {args[1]}) + {args[2]})"
        if function_name == "__byte_perm" and len(args) == 3:
            return self.format_cuda_byte_perm_intrinsic(args[0], args[1], args[2])
        if function_name in {"__ffs", "__ffsll"} and len(args) == 1:
            return f"(findLSB({args[0]}) + 1)"
        if function_name in {"__clz", "__clzll"} and len(args) == 1:
            return f"countLeadingZeros({args[0]})"
        if function_name in {"__brev", "__brevll"} and len(args) == 1:
            return f"bitfieldReverse({args[0]})"
        if function_name in {"__hadd", "__uhadd"} and len(args) == 2:
            return self.format_integer_average_floor(args[0], args[1])
        if function_name in {"__rhadd", "__urhadd"} and len(args) == 2:
            return self.format_integer_average_rounded(args[0], args[1])
        if function_name == "__sad" and len(args) == 3:
            return f"(abs({args[0]} - {args[1]}) + {args[2]})"
        if function_name == "__usad" and len(args) == 3:
            return (
                f"((({args[0]} > {args[1]}) ? ({args[0]} - {args[1]}) : "
                f"({args[1]} - {args[0]})) + {args[2]})"
            )
        if function_name == "__usad4" and len(args) in {2, 3}:
            bias = args[2] if len(args) == 3 else "0u"
            return self.format_cuda_usad4_intrinsic(args[0], args[1], bias)
        if function_name in {
            "__funnelshift_l",
            "__funnelshift_lc",
            "__funnelshift_r",
            "__funnelshift_rc",
        }:
            return self.format_cuda_integer_intrinsic_diagnostic(function_name, args)
        return None

    def format_cuda_usad4_intrinsic(self, left, right, bias):
        differences = [
            self.format_cuda_unsigned_byte_absdiff(left, right, shift)
            for shift in (0, 8, 16, 24)
        ]
        result = differences[0]
        for difference in differences[1:]:
            result = f"({result} + {difference})"

        if self.normalize_cuda_integer_literal(bias) == "0":
            return result
        return f"({result} + {bias})"

    def format_cuda_unsigned_byte_absdiff(self, left, right, shift):
        left_byte = self.format_cuda_packed_byte(left, shift)
        right_byte = self.format_cuda_packed_byte(right, shift)
        return (
            f"(({left_byte} > {right_byte}) ? ({left_byte} - {right_byte}) : "
            f"({right_byte} - {left_byte}))"
        )

    def format_cuda_packed_byte(self, value, shift):
        operand = self.format_bitwise_operand(value)
        if shift == 0:
            return f"({operand} & 0xffu)"
        return f"(({operand} >> {shift}) & 0xffu)"

    def format_cuda_integer_intrinsic_diagnostic(self, function_name, args):
        args_text = ", ".join(args)
        return (
            f"(/* cuda integer intrinsic {function_name}({args_text}) "
            "not directly supported in CrossGL */ 0)"
        )

    def format_cuda_byte_perm_intrinsic(self, left, right, selector):
        normalized_selector = self.normalize_cuda_integer_literal(selector)
        if normalized_selector in {"0x0123", "0x123"}:
            return self.format_cuda_byte_swap_32(left)

        return self.format_cuda_integer_intrinsic_diagnostic(
            "__byte_perm", [left, right, selector]
        )

    def format_cuda_byte_swap_32(self, value):
        operand = self.format_bitwise_operand(value)
        return (
            f"(((({operand} & 0x000000ffu) << 24) | "
            f"(({operand} & 0x0000ff00u) << 8)) | "
            f"((({operand} & 0x00ff0000u) >> 8) | "
            f"(({operand} & 0xff000000u) >> 24)))"
        )

    def normalize_cuda_integer_literal(self, value):
        text = str(value).strip().replace("'", "").lower()
        while text.endswith(("u", "l")):
            text = text[:-1]
        return text

    def format_bitwise_operand(self, value):
        text = str(value).strip()
        if text and all(char.isalnum() or char in "_." for char in text):
            return text
        return f"({text})"

    def format_integer_average_floor(self, left, right):
        return f"(({left} & {right}) + (({left} ^ {right}) >> 1))"

    def format_integer_average_rounded(self, left, right):
        return f"(({left} | {right}) - (({left} ^ {right}) >> 1))"

    def format_cuda_float_intrinsic_call(self, function_name, args):
        if isinstance(function_name, str) and function_name.startswith("::"):
            function_name = function_name[2:]

        fast_math_intrinsics = {
            "__cosf": ("cos", 1),
            "__exp10f": ("pow", 1),
            "__frsqrt_rn": ("inversesqrt", 1),
            "__fsqrt_rd": ("sqrt", 1),
            "__fsqrt_rn": ("sqrt", 1),
            "__fsqrt_ru": ("sqrt", 1),
            "__fsqrt_rz": ("sqrt", 1),
            "__expf": ("exp", 1),
            "__log2f": ("log2", 1),
            "__logf": ("log", 1),
            "__powf": ("pow", 2),
            "__sinf": ("sin", 1),
            "__tanf": ("tan", 1),
            "__tanhf": ("tanh", 1),
        }
        mapped_intrinsic = fast_math_intrinsics.get(function_name)
        if mapped_intrinsic is not None:
            mapped_name, arity = mapped_intrinsic
            if len(args) == arity:
                if function_name == "__exp10f":
                    return f"{mapped_name}(10.0f, {args[0]})"
                return f"{mapped_name}({', '.join(args)})"

        if function_name == "__saturatef" and len(args) == 1:
            return f"clamp({args[0]}, 0.0f, 1.0f)"
        if (
            function_name
            in {"__fdividef", "__fdiv_rd", "__fdiv_rn", "__fdiv_ru", "__fdiv_rz"}
            and len(args) == 2
        ):
            return f"({args[0]} / {args[1]})"
        if (
            function_name in {"__fadd_rd", "__fadd_rn", "__fadd_ru", "__fadd_rz"}
            and len(args) == 2
        ):
            return f"({args[0]} + {args[1]})"
        if (
            function_name in {"__fsub_rd", "__fsub_rn", "__fsub_ru", "__fsub_rz"}
            and len(args) == 2
        ):
            return f"({args[0]} - {args[1]})"
        if (
            function_name in {"__fmul_rd", "__fmul_rn", "__fmul_ru", "__fmul_rz"}
            and len(args) == 2
        ):
            return f"({args[0]} * {args[1]})"
        if (
            function_name in {"__frcp_rd", "__frcp_rn", "__frcp_ru", "__frcp_rz"}
            and len(args) == 1
        ):
            return f"(1.0f / {args[0]})"
        if (
            function_name
            in {
                "__fmaf_rd",
                "__fmaf_rn",
                "__fmaf_ru",
                "__fmaf_rz",
                "__fmaf_ieee_rd",
                "__fmaf_ieee_rn",
                "__fmaf_ieee_ru",
                "__fmaf_ieee_rz",
            }
            and len(args) == 3
        ):
            return f"fma({args[0]}, {args[1]}, {args[2]})"

        return None

    def format_cuda_signed_24_bit_operand(self, arg):
        return f"(({arg} << 8) >> 8)"

    def format_vector_component_access(self, expression, component):
        text = str(expression).strip()
        if text and all(char.isalnum() or char in "_." for char in text):
            return f"{text}.{component}"
        return f"({text}).{component}"

    def format_cooperative_group_call(self, node):
        if isinstance(node.name, MemberAccessNode):
            member = node.name.member
            group_metadata = self.resolve_cooperative_group_metadata(node.name.object)
            if group_metadata is None:
                return None
            return self.format_cooperative_group_member_call(
                group_metadata, member, node.args
            )

        raw_name = node.name if isinstance(node.name, str) else self.visit(node.name)
        base_call_name, _ = self.parse_cpp_template(raw_name)
        base_name = self.cooperative_group_base_name(base_call_name)
        if base_name in {"sync", "thread_rank", "size"} and len(node.args) == 1:
            group_metadata = self.resolve_cooperative_group_metadata(node.args[0])
            if group_metadata is not None:
                return self.format_cooperative_group_member_call(
                    group_metadata, base_name, []
                )
        if base_name in {"memcpy_async", "wait", "wait_prior"} and node.args:
            group_metadata = self.resolve_cooperative_group_metadata(node.args[0])
            if group_metadata is not None:
                return self.format_cooperative_group_member_call(
                    group_metadata, base_name, node.args[1:]
                )
        if base_name == "reduce" and node.args:
            group_metadata = self.resolve_cooperative_group_metadata(node.args[0])
            if group_metadata is not None:
                return self.format_cooperative_group_reduce_call(
                    group_metadata, node.args[1:]
                )
        return None

    def format_cooperative_group_member_call(self, group_metadata, member, args):
        group_kind = group_metadata["kind"]
        member_base_name, _ = self.parse_cpp_template(member)
        member_name = self.cooperative_group_base_name(member_base_name) or member
        if group_kind == "thread_block" and not args:
            if member_name == "sync":
                return "workgroupBarrier()"
            if member_name == "thread_rank":
                return "gl_LocalInvocationIndex"
            if member_name in {"size", "num_threads"}:
                return self.format_thread_block_size_expression()
            if member_name == "thread_index":
                return "gl_LocalInvocationID"
            if member_name == "dim_threads":
                return "gl_WorkGroupSize"

        if group_kind == "thread_block_tile" and not args:
            tile_size = group_metadata.get("tile_size")
            if (
                member_name == "sync"
                and tile_size
                and group_metadata.get("parent_kind") == "thread_block"
            ):
                return "workgroupBarrier()"
            if member_name in {"size", "num_threads"} and tile_size:
                return tile_size
            if (
                member_name == "thread_rank"
                and tile_size
                and group_metadata.get("parent_kind") == "thread_block"
            ):
                return f"(gl_LocalInvocationIndex % {tile_size})"

        if group_kind == "thread_block_tile" and args:
            tile_collective = self.format_thread_block_tile_collective_call(
                group_metadata, member_name, args
            )
            if tile_collective is not None:
                return tile_collective

        if member_name in {"sync", "memcpy_async", "wait", "wait_prior"}:
            return self.format_unsupported_cooperative_group_statement(
                group_kind, member_name, args
            )
        if member_name in {"thread_rank", "size", "num_threads", "block_rank"}:
            return self.format_unsupported_cooperative_group_expression(
                group_kind, member_name
            )
        if member_name in {"thread_index", "dim_threads", "dim_blocks"}:
            return self.format_unsupported_cooperative_group_expression(
                group_kind, member_name, "vec3<u32>(0, 0, 0)"
            )
        if member_name == "map_shared_rank" and args:
            return self.format_unsupported_cooperative_group_expression(
                group_kind, member_name, self.visit(args[0]), args=args
            )
        return self.format_unsupported_cooperative_group_expression(
            group_kind, member_name, args=args
        )

    def format_thread_block_tile_collective_call(
        self, group_metadata, member_name, args
    ):
        if (
            group_metadata.get("tile_size") != "32"
            or group_metadata.get("parent_kind") != "thread_block"
        ):
            return None

        formatted_args = [self.visit(arg) for arg in args]
        if member_name == "any" and len(formatted_args) == 1:
            predicate = self.format_wave_predicate(formatted_args[0])
            return f"WaveActiveAnyTrue({predicate})"
        if member_name == "all" and len(formatted_args) == 1:
            predicate = self.format_wave_predicate(formatted_args[0])
            return f"WaveActiveAllTrue({predicate})"
        if member_name == "ballot" and len(formatted_args) == 1:
            predicate = self.format_wave_predicate(formatted_args[0])
            return f"WaveActiveBallot({predicate}).x"
        if member_name == "shfl" and len(formatted_args) >= 2:
            return f"WaveReadLaneAt({formatted_args[0]}, {formatted_args[1]})"

        return None

    def format_cooperative_group_reduce_call(self, group_metadata, args):
        group_kind = group_metadata["kind"]
        if (
            group_kind == "thread_block_tile"
            and group_metadata.get("tile_size") == "32"
            and group_metadata.get("parent_kind") == "thread_block"
            and len(args) == 2
            and self.cooperative_group_reduce_operator(args[1]) == "plus"
        ):
            return f"WaveActiveSum({self.visit(args[0])})"

        return self.format_unsupported_cooperative_group_expression(
            group_kind, "reduce", args=args
        )

    def cooperative_group_reduce_operator(self, expression):
        if not isinstance(expression, FunctionCallNode) or expression.args:
            return None

        if not isinstance(expression.name, str):
            return None

        base_call_name, _ = self.parse_cpp_template(expression.name)
        return self.cooperative_group_base_name(base_call_name)

    def format_thread_block_size_expression(self):
        return "((gl_WorkGroupSize.x * gl_WorkGroupSize.y) * gl_WorkGroupSize.z)"

    def format_unsupported_cooperative_group_statement(
        self, group_kind, member, args=None
    ):
        args = args or []
        formatted_args = ", ".join(self.visit(arg) for arg in args)
        arg_suffix = f": {formatted_args}" if formatted_args else ""
        return (
            f"// cooperative_groups {group_kind}.{member} "
            f"not directly supported in CrossGL{arg_suffix}"
        )

    def format_unsupported_cooperative_group_expression(
        self, group_kind, member, fallback="0", args=None
    ):
        args = args or []
        formatted_args = ", ".join(self.visit(arg) for arg in args)
        arg_suffix = f"({formatted_args})" if formatted_args else ""
        return (
            f"(/* cooperative_groups {group_kind}.{member}{arg_suffix} "
            f"not directly supported in CrossGL */ {fallback})"
        )

    def format_cuda_async_sync_call(self, node):
        if isinstance(node.name, MemberAccessNode):
            metadata = self.resolve_cuda_async_sync_metadata(node.name.object)
            if metadata is None:
                return None
            return self.format_cuda_async_sync_member_call(
                metadata, node.name.member, node.args
            )

        raw_name = node.name if isinstance(node.name, str) else self.visit(node.name)
        base_call_name, _ = self.parse_cpp_template(raw_name)
        base_name = self.cooperative_group_base_name(base_call_name)
        if self.is_user_defined_function(base_name) or self.is_user_defined_function(
            raw_name
        ):
            return None

        if base_name == "memcpy_async":
            metadata = self.resolve_cuda_async_sync_metadata_from_args(node.args)
            if metadata is not None:
                return self.format_unsupported_cuda_async_statement(
                    metadata["kind"], base_name, node.args
                )

        if base_name == "init" and node.args:
            metadata = self.resolve_cuda_async_sync_metadata(
                self.unwrap_cuda_async_sync_target(node.args[0])
            )
            if metadata is not None and metadata["kind"] == "barrier":
                return self.format_unsupported_cuda_async_statement(
                    metadata["kind"], base_name, node.args
                )

        if base_name == "make_pipeline":
            return self.format_unsupported_cuda_async_expression("pipeline", base_name)

        primitive_member = self.cuda_pipeline_primitive_member_name(base_name)
        if primitive_member is not None:
            return self.format_unsupported_cuda_async_statement(
                "pipeline", primitive_member, node.args
            )

        return None

    def cuda_pipeline_primitive_member_name(self, base_name):
        primitive_mapping = {
            "__pipeline_memcpy_async": "memcpy_async",
            "__pipeline_commit": "commit",
            "__pipeline_wait_prior": "wait_prior",
            "__pipeline_arrive_on": "arrive_on",
        }
        return primitive_mapping.get(base_name)

    def format_cuda_async_sync_member_call(self, metadata, member, args):
        kind = metadata["kind"]
        member_base_name, _ = self.parse_cpp_template(member)
        member_name = self.cooperative_group_base_name(member_base_name) or member

        if kind == "barrier":
            if member_name in {"arrive", "arrival_token"}:
                return self.format_unsupported_cuda_async_expression(kind, member_name)
            if member_name in {"try_wait", "try_wait_parity"}:
                return self.format_unsupported_cuda_async_expression(
                    kind, member_name, "false"
                )
            return self.format_unsupported_cuda_async_statement(kind, member_name, args)

        if kind == "pipeline":
            return self.format_unsupported_cuda_async_statement(kind, member_name, args)

        return self.format_unsupported_cuda_async_statement(kind, member_name, args)

    def format_unsupported_cuda_async_statement(self, kind, member, args=None):
        args = args or []
        formatted_args = ", ".join(self.visit(arg) for arg in args)
        arg_suffix = f": {formatted_args}" if formatted_args else ""
        return f"// cuda {kind}.{member} not directly supported in CrossGL{arg_suffix}"

    def format_unsupported_cuda_async_expression(self, kind, member, fallback="0"):
        return (
            f"(/* cuda {kind}.{member} not directly supported in CrossGL */ {fallback})"
        )

    def format_cuda_async_sync_declaration(self, name, metadata):
        return f"// cuda {metadata['kind']} {name} not directly supported in CrossGL"

    def cuda_async_sync_declaration_metadata(self, node):
        declared_metadata = self.cuda_async_sync_metadata_from_type(node.vtype)
        factory_metadata = self.cuda_async_sync_factory_metadata(node.value)
        return factory_metadata or declared_metadata

    def cuda_async_sync_metadata_from_type(self, type_name):
        type_name = self.strip_cuda_async_sync_declarator(type_name)
        base_type, template_args = self.parse_cpp_template(type_name)
        base_name = self.cooperative_group_base_name(base_type)
        if base_name in {"barrier", "pipeline", "pipeline_shared_state"}:
            metadata = {"kind": base_name}
            if template_args:
                metadata["scope"] = template_args[0]
            return metadata
        return None

    def cuda_async_sync_factory_metadata(self, value):
        if not isinstance(value, FunctionCallNode):
            return None
        raw_name = value.name if isinstance(value.name, str) else self.visit(value.name)
        base_call_name, _ = self.parse_cpp_template(raw_name)
        base_name = self.cooperative_group_base_name(base_call_name)
        if base_name == "make_pipeline":
            return {"kind": "pipeline"}
        return None

    def resolve_cuda_async_sync_metadata_from_args(self, args):
        for arg in reversed(args):
            metadata = self.resolve_cuda_async_sync_metadata(arg)
            if metadata is not None and metadata["kind"] in {"barrier", "pipeline"}:
                return metadata
        return None

    def resolve_cuda_async_sync_metadata(self, expression):
        name = self.simple_identifier(expression)
        if name is None:
            return None
        return self.lookup_cuda_async_sync_metadata(name)

    def unwrap_cuda_async_sync_target(self, expression):
        if isinstance(expression, UnaryOpNode) and expression.op == "&":
            return expression.operand
        if isinstance(expression, CastNode):
            return self.unwrap_cuda_async_sync_target(expression.expression)
        return expression

    def strip_cuda_async_sync_declarator(self, type_name):
        type_name = self.strip_type_qualifiers(type_name).strip()
        if "[" in type_name:
            type_name = type_name.split("[", 1)[0].strip()

        while True:
            stripped = type_name.strip()
            for suffix in ("&&", "&", "*"):
                if stripped.endswith(suffix):
                    type_name = stripped[: -len(suffix)].strip()
                    break
            else:
                return stripped

    def cooperative_group_declaration_metadata(self, node):
        declared_metadata = self.cooperative_group_metadata_from_type(node.vtype)
        factory_metadata = self.cooperative_group_factory_metadata(node.value)
        return factory_metadata or declared_metadata

    def cooperative_group_kind_from_type(self, type_name):
        metadata = self.cooperative_group_metadata_from_type(type_name)
        return metadata["kind"] if metadata is not None else None

    def cooperative_group_metadata_from_type(self, type_name):
        type_name = self.strip_type_qualifiers(type_name)
        base_type, template_args = self.parse_cpp_template(type_name)
        base_name = self.cooperative_group_base_name(type_name)
        if base_name in {
            "thread_group",
            "thread_block",
            "grid_group",
            "multi_grid_group",
            "coalesced_group",
            "cluster_group",
        }:
            return {"kind": base_name}
        base_name = self.cooperative_group_base_name(base_type)
        if base_name and base_name.startswith("thread_block_tile"):
            metadata = {"kind": "thread_block_tile"}
            if template_args:
                metadata["tile_size"] = template_args[0]
            return metadata
        return None

    def cooperative_group_factory_kind(self, value):
        metadata = self.cooperative_group_factory_metadata(value)
        return metadata["kind"] if metadata is not None else None

    def cooperative_group_factory_metadata(self, value):
        if not isinstance(value, FunctionCallNode):
            return None
        raw_name = value.name if isinstance(value.name, str) else self.visit(value.name)
        base_call_name, template_args = self.parse_cpp_template(raw_name)
        base_name = self.cooperative_group_base_name(base_call_name)
        factory_mapping = {
            "this_thread_block": "thread_block",
            "this_grid": "grid_group",
            "this_multi_grid": "multi_grid_group",
            "this_cluster": "cluster_group",
            "coalesced_threads": "coalesced_group",
            "tiled_partition": "thread_block_tile",
        }
        group_kind = factory_mapping.get(base_name)
        if group_kind is None:
            return None

        metadata = {"kind": group_kind}
        if group_kind == "thread_block_tile":
            if template_args:
                metadata["tile_size"] = template_args[0]
            if value.args:
                parent = self.resolve_cooperative_group_metadata(value.args[0])
                if parent is not None:
                    metadata["parent_kind"] = parent["kind"]
        return metadata

    def resolve_cooperative_group_expression(self, expression):
        metadata = self.resolve_cooperative_group_metadata(expression)
        return metadata["kind"] if metadata is not None else None

    def resolve_cooperative_group_metadata(self, expression):
        name = self.simple_identifier(expression)
        group_metadata = self.lookup_cooperative_group_metadata(name)
        if group_metadata is not None:
            return group_metadata
        return self.cooperative_group_factory_metadata(expression)

    def simple_identifier(self, expression):
        if isinstance(expression, str):
            return expression
        return None

    def cooperative_group_base_name(self, name):
        if not isinstance(name, str):
            return None
        return name.rsplit("::", 1)[-1].split("<", 1)[0]

    def format_cuda_resource_call(self, function_name, args, raw_args=None):
        base_name, template_args = self.parse_cpp_template(function_name)
        if self.is_user_defined_function(base_name):
            return None

        value_type = template_args[0] if template_args else None
        if base_name == "tex1Dfetch":
            return self.format_cuda_texture_fetch_call(args)
        if base_name in {"tex1D", "tex1DLod", "tex1DGrad"}:
            return self.format_cuda_texture_call(base_name, args, "vec1", 1)
        if base_name in {"tex2D", "tex2DLod", "tex2DGrad"}:
            return self.format_cuda_texture_call(base_name, args, "vec2", 2)
        if base_name == "tex2Dgather":
            return self.format_cuda_texture_gather_call(args)
        if base_name in {"tex3D", "tex3DLod", "tex3DGrad"}:
            return self.format_cuda_texture_call(base_name, args, "vec3", 3)
        if base_name in {"texCubemap", "texCubemapLod", "texCubemapGrad"}:
            return self.format_cuda_texture_call(base_name, args, "vec3", 3)
        if base_name in {
            "tex1DLayered",
            "tex1DLayeredLod",
            "tex1DLayeredGrad",
        }:
            return self.format_cuda_texture_call(base_name, args, "vec2", 2)
        if base_name in {
            "tex2DLayered",
            "tex2DLayeredLod",
            "tex2DLayeredGrad",
        }:
            return self.format_cuda_texture_call(base_name, args, "vec3", 3)
        if base_name in {
            "texCubemapLayered",
            "texCubemapLayeredLod",
            "texCubemapLayeredGrad",
        }:
            return self.format_cuda_texture_call(base_name, args, "vec4", 4)

        if base_name in {
            "surf1Dread",
            "surf1DLayeredread",
            "surf2Dread",
            "surf3Dread",
            "surf2DLayeredread",
            "surfCubemapread",
            "surfCubemapLayeredread",
        }:
            dimensions = {
                "surf1Dread": 1,
                "surf1DLayeredread": 2,
                "surf2Dread": 2,
                "surf3Dread": 3,
                "surf2DLayeredread": 3,
                "surfCubemapread": 3,
                "surfCubemapLayeredread": 3,
            }[base_name]
            return self.format_cuda_surface_read(args, dimensions, value_type)

        if base_name in {
            "surf1Dwrite",
            "surf1DLayeredwrite",
            "surf2Dwrite",
            "surf3Dwrite",
            "surf2DLayeredwrite",
            "surfCubemapwrite",
            "surfCubemapLayeredwrite",
        }:
            dimensions = {
                "surf1Dwrite": 1,
                "surf1DLayeredwrite": 2,
                "surf2Dwrite": 2,
                "surf3Dwrite": 3,
                "surf2DLayeredwrite": 3,
                "surfCubemapwrite": 3,
                "surfCubemapLayeredwrite": 3,
            }[base_name]
            return self.format_cuda_surface_write(
                args, dimensions, value_type, raw_args
            )

        return None

    def format_cuda_texture_call(self, function_name, args, vector_name, dimensions):
        if len(args) < 2:
            return None

        if self.is_sparse_cuda_texture_call(function_name, args):
            return self.format_unsupported_cuda_texture_sparse_residency_call(
                function_name
            )

        extra_count = (
            2 if "Grad" in function_name else 1 if "Lod" in function_name else 0
        )
        coordinate_count = len(args) - 1 - extra_count
        if coordinate_count <= 0:
            return None

        texture_name = args[0]
        coordinate_args = args[1 : 1 + coordinate_count]
        if coordinate_count == 1:
            coordinate = coordinate_args[0]
            consumed = 2
        elif coordinate_count == dimensions:
            coordinate = self.format_vector_constructor(vector_name, coordinate_args)
            consumed = 1 + dimensions
        else:
            return None

        remaining = args[consumed:]
        if "Grad" in function_name:
            if len(remaining) < 2:
                return None
            return f"textureGrad({texture_name}, {coordinate}, {remaining[0]}, {remaining[1]})"
        if "Lod" in function_name:
            if not remaining:
                return None
            return f"textureLod({texture_name}, {coordinate}, {remaining[0]})"
        return f"texture({texture_name}, {coordinate})"

    def is_sparse_cuda_texture_call(self, function_name, args):
        sparse_arg_counts = {
            "tex1D": 3,
            "tex1DLod": 4,
            "tex1DGrad": 5,
            "tex2D": 4,
            "tex2DLod": 5,
            "tex2DGrad": 6,
            "tex3D": 5,
            "tex3DLod": 6,
            "tex3DGrad": 7,
            "texCubemap": 5,
            "texCubemapLod": 6,
            "texCubemapGrad": 7,
            "tex1DLayered": 4,
            "tex1DLayeredLod": 5,
            "tex1DLayeredGrad": 6,
            "tex2DLayered": 5,
            "tex2DLayeredLod": 6,
            "tex2DLayeredGrad": 7,
            "texCubemapLayered": 6,
            "texCubemapLayeredLod": 7,
            "texCubemapLayeredGrad": 8,
        }
        return len(args) == sparse_arg_counts.get(function_name)

    def format_unsupported_cuda_texture_sparse_residency_call(self, function_name):
        return self.format_unsupported_cuda_resource_expression(
            "texture",
            f"{function_name} sparse residency",
            "vec4<f32>(0.0, 0.0, 0.0, 0.0)",
        )

    def format_cuda_texture_gather_call(self, args):
        if len(args) == 2:
            texture_name, coordinate = args
            component = None
        elif len(args) in {3, 4}:
            texture_name = args[0]
            coordinate = self.format_vector_constructor("vec2", args[1:3])
            component = args[3] if len(args) == 4 else None
        else:
            return self.format_unsupported_cuda_resource_expression(
                "texture",
                "tex2Dgather sparse residency",
                "vec4<f32>(0.0, 0.0, 0.0, 0.0)",
            )

        if component is not None:
            return f"textureGather({texture_name}, {coordinate}, {component})"
        return f"textureGather({texture_name}, {coordinate})"

    def format_cuda_texture_fetch_call(self, args):
        if len(args) == 2:
            texture_name, coordinate = args
            return f"texelFetch({texture_name}, {coordinate}, 0)"
        return self.format_unsupported_cuda_resource_expression(
            "texture",
            "tex1Dfetch sparse residency",
            "vec4<f32>(0.0, 0.0, 0.0, 0.0)",
        )

    def format_cuda_surface_read(self, args, dimensions, value_type):
        if len(args) < dimensions + 1:
            return None
        surface_name = args[0]
        coord_args = [self.strip_surface_byte_offset(args[1], value_type)]
        coord_args.extend(args[2 : dimensions + 1])
        return f"imageLoad({surface_name}, {self.format_vector_constructor(f'vec{dimensions}', coord_args, 'i32')})"

    def format_cuda_surface_write(self, args, dimensions, value_type, raw_args=None):
        if len(args) < dimensions + 2:
            return None
        if value_type is None and raw_args:
            value_type = self.infer_cuda_surface_value_type(raw_args[0])
        value = args[0]
        surface_name = args[1]
        coord_args = [self.strip_surface_byte_offset(args[2], value_type)]
        coord_args.extend(args[3 : dimensions + 2])
        coord = self.format_vector_constructor(f"vec{dimensions}", coord_args, "i32")
        return f"imageStore({surface_name}, {coord}, {value})"

    def strip_surface_byte_offset(self, expression, value_type):
        text = str(expression).strip()
        if text.startswith("(") and text.endswith(")"):
            text = text[1:-1].strip()

        if value_type:
            suffix = f" * sizeof({value_type})"
            if text.endswith(suffix):
                return text[: -len(suffix)].strip()

            byte_size = self.cuda_surface_value_byte_size(value_type)
            if byte_size is not None:
                for literal in self.cuda_byte_size_literals(byte_size):
                    suffix = f" * {literal}"
                    if text.endswith(suffix):
                        return text[: -len(suffix)].strip()
                    prefix = f"{literal} * "
                    if text.startswith(prefix):
                        return text[len(prefix) :].strip()

        marker = " * sizeof("
        if marker in text and text.endswith(")"):
            return text.split(marker, 1)[0].strip()

        return expression

    def infer_cuda_surface_value_type(self, expression):
        if isinstance(expression, CastNode):
            return self.strip_type_qualifiers(expression.target_type)
        if isinstance(expression, ArrayAccessNode):
            array_type = self.cuda_surface_value_source_type(expression.array)
            return self.cuda_surface_element_type(array_type)
        if isinstance(expression, str):
            return self.cuda_surface_element_type(self.lookup_variable_type(expression))
        if isinstance(expression, UnaryOpNode) and expression.op == "*":
            return self.cuda_surface_element_type(
                self.cuda_surface_value_source_type(expression.operand)
            )
        return None

    def cuda_surface_value_source_type(self, expression):
        if isinstance(expression, str):
            return self.lookup_variable_type(expression)
        if isinstance(expression, ArrayAccessNode):
            return self.cuda_surface_element_type(
                self.cuda_surface_value_source_type(expression.array)
            )
        if isinstance(expression, CastNode):
            return expression.target_type
        if isinstance(expression, UnaryOpNode):
            return self.cuda_surface_value_source_type(expression.operand)
        return None

    def cuda_surface_element_type(self, type_name):
        if not type_name:
            return None
        type_name = self.strip_type_qualifiers(self.resolve_type_alias(type_name))
        if self.has_array_suffix(type_name):
            return type_name.split("[", 1)[0].strip()
        if "*" in type_name:
            return type_name.replace("*", "").strip()
        return type_name

    def cuda_surface_value_byte_size(self, value_type):
        if not value_type:
            return None
        value_type = self.strip_type_qualifiers(self.resolve_type_alias(value_type))
        value_type = value_type.replace("*", "").strip()
        return self.CUDA_SURFACE_VALUE_BYTE_SIZES.get(value_type)

    def cuda_byte_size_literals(self, byte_size):
        text = str(byte_size)
        return (text, f"{text}u", f"{text}U", f"{text}ul", f"{text}UL")

    def format_unsupported_cuda_resource_expression(self, kind, member, fallback):
        return (
            f"(/* cuda {kind}.{member} not directly supported in CrossGL */ {fallback})"
        )

    def format_vector_constructor(self, vector_name, args, element_type="f32"):
        if vector_name == "vec1":
            return args[0]
        return f"{vector_name}<{element_type}>({', '.join(args)})"

    def format_lambda_call(self, args):
        if not args:
            return "lambda()"

        rendered_args = [self.format_lambda_parameter(arg) for arg in args[:-1]] + [
            self.format_lambda_body(args[-1])
        ]
        return f"lambda({', '.join(rendered_args)})"

    def format_lambda_parameter(self, arg):
        if isinstance(arg, VariableNode):
            if arg.vtype:
                param_type = self.convert_cuda_variable_type_to_crossgl(
                    arg.vtype, arg.name
                )
                return f"{param_type} {arg.name}"
            return arg.name
        return self.format_lambda_body(arg)

    def format_lambda_body(self, arg):
        if isinstance(arg, str):
            return arg
        return self.visit(arg)

    def format_cuda_fold_expression(self, args):
        expression = args[0] if args else ""
        return (
            f"0 /* CUDA fold expression {expression} is not directly supported "
            "in CrossGL */"
        )

    def is_get_method_call(self, node):
        return (
            isinstance(node.name, MemberAccessNode)
            and node.name.member == "get"
            and not node.args
            and self.is_unique_ptr_expression(node.name.object)
        )

    def format_make_unique_call(self, function_name, args):
        base_name, template_args = self.parse_cpp_template(function_name)
        if not self.is_std_make_unique_base_name(base_name) or not template_args:
            return None

        target_type, is_array = self.unwrap_array_template_type(template_args[0])
        target_type = self.convert_cuda_type_to_crossgl(target_type)
        args_str = ", ".join(args)
        if is_array:
            return f"new_array<{target_type}>({args_str})"
        return f"new<{target_type}>({args_str})"

    def format_unique_ptr_constructor_call(self, function_name, args):
        base_name, _ = self.parse_cpp_template(function_name)
        if len(args) != 1:
            return None
        if not self.is_std_unique_ptr_base_name(
            base_name
        ) and not self.is_unique_ptr_type_name(function_name):
            return None

        return args[0]

    def visit_NewNode(self, node):
        target_type = self.convert_cuda_type_to_crossgl(node.target_type)
        if node.is_array:
            size = self.visit(node.size) if node.size is not None else ""
            return f"new_array<{target_type}>({size})"

        args = ", ".join(self.visit(arg) for arg in node.args)
        return f"new<{target_type}>({args})"

    def visit_DeleteNode(self, node):
        target = self.visit(node.expression)
        if node.is_array:
            self.emit(f"// delete array: {target}")
        else:
            self.emit(f"// delete: {target}")

    def visit_TypeAliasNode(self, node):
        self.register_type_alias(node.name, node.alias_type)
        if str(node.alias_type).strip().startswith(("struct ", "union ")):
            alias_type = node.alias_type
        elif self.is_decltype_type_name(node.alias_type):
            alias_type = node.alias_type
        else:
            alias_type = self.convert_cuda_type_to_crossgl(node.alias_type)
        self.emit(f"// typedef {alias_type} {node.name};")

    def format_atomic_argument(self, arg, index):
        if index == 0 and isinstance(arg, UnaryOpNode) and arg.op == "&":
            return self.visit(arg.operand)
        return self.visit(arg)

    def visit_AtomicOperationNode(self, node):
        args = [self.format_atomic_argument(arg, i) for i, arg in enumerate(node.args)]
        args_str = ", ".join(args)
        operation = node.operation
        scope = None
        for suffix, scope_name in (("_block", "block"), ("_system", "system")):
            if operation.endswith(suffix):
                operation = operation[: -len(suffix)]
                scope = scope_name
                break

        atomic_map = {
            "atomicAdd": "atomicAdd",
            "atomicSub": "atomicSub",
            "atomicMax": "atomicMax",
            "atomicMin": "atomicMin",
            "atomicExch": "atomicExchange",
            "atomicCAS": "atomicCompareExchange",
            "atomicAnd": "atomicAnd",
            "atomicOr": "atomicOr",
            "atomicXor": "atomicXor",
            "atomicInc": "atomicInc",
            "atomicDec": "atomicDec",
        }

        crossgl_func = atomic_map.get(operation, operation)
        expression = f"{crossgl_func}({args_str})"
        if scope is not None:
            return (
                f"(/* cuda {scope}-scope atomic {node.operation} lowered to "
                f"{crossgl_func}; scope not preserved */ {expression})"
            )
        return expression

    def visit_SyncNode(self, node):
        if node.sync_type == "__syncthreads":
            self.emit("workgroupBarrier();")
        elif node.sync_type == "__syncwarp":
            args = ", ".join(self.visit(arg) for arg in node.args)
            self.emit(f"// __syncwarp({args}) not directly supported in CrossGL")
        else:
            self.emit(f"// {node.sync_type}();")

    def visit_CudaAsmNode(self, node):
        volatility = " volatile" if node.is_volatile else ""
        self.emit(f"// CUDA inline PTX{volatility}: {node.template}")
        if node.outputs:
            self.emit(
                f"// CUDA inline PTX outputs: {self.format_cuda_asm_operands(node.outputs)}"
            )
        if node.inputs:
            self.emit(
                f"// CUDA inline PTX inputs: {self.format_cuda_asm_operands(node.inputs)}"
            )
        if node.clobbers:
            self.emit(f"// CUDA inline PTX clobbers: {', '.join(node.clobbers)}")

    def format_cuda_asm_operands(self, operands):
        formatted = []
        for operand in operands:
            prefix = (
                f"[{operand.symbolic_name}] "
                if operand.symbolic_name is not None
                else ""
            )
            expression = (
                self.visit(operand.expression)
                if operand.expression is not None
                else None
            )
            if expression is None:
                formatted.append(f"{prefix}{operand.constraint}")
            else:
                formatted.append(f"{prefix}{operand.constraint}({expression})")
        return ", ".join(formatted)

    def visit_CudaBuiltinNode(self, node):
        builtin_map = {
            "threadIdx": "gl_LocalInvocationID",
            "blockIdx": "gl_WorkGroupID",
            "gridDim": "gl_NumWorkGroups",
            "blockDim": "gl_WorkGroupSize",
            "warpSize": "32",  # Constant in CrossGL
        }

        base_name = builtin_map.get(node.builtin_name, node.builtin_name)
        if node.component:
            return f"{base_name}.{node.component}"
        else:
            return base_name

    def visit_MemberAccessNode(self, node):
        obj = self.visit(node.object)
        operator = "->" if getattr(node, "is_pointer", False) else "."
        if node.member == "x" and operator == "." and self.is_vector1_name(node.object):
            return obj
        member = self.sanitize_identifier_name(node.member)
        return f"{obj}{operator}{member}"

    def visit_ArrayAccessNode(self, node):
        array = self.visit(node.array)
        index = self.visit(node.index)
        return f"{array}[{index}]"

    def visit_InitializerListNode(self, node):
        elements = ", ".join(self.visit(element) for element in node.elements)
        return f"{{{elements}}}"

    def visit_DesignatedInitializerNode(self, node):
        designators = []
        for kind, target in node.designators:
            if kind == "index":
                designators.append(f"[{self.visit(target)}]")
            else:
                designators.append(f".{target}")

        value = self.visit(node.value)
        return f"{''.join(designators)} = {value}"

    def visit_IfNode(self, node):
        condition = self.visit(node.condition)
        self.emit(f"if ({condition}) {{")

        self.indent_level += 1
        if isinstance(node.if_body, list):
            for stmt in node.if_body:
                self.emit_statement(stmt)
        else:
            self.emit_statement(node.if_body)
        self.indent_level -= 1

        if node.else_body:
            self.emit("} else {")
            self.indent_level += 1
            if isinstance(node.else_body, list):
                for stmt in node.else_body:
                    self.emit_statement(stmt)
            else:
                self.emit_statement(node.else_body)
            self.indent_level -= 1

        self.emit("}")

    def visit_ForNode(self, node):
        scoped_init = isinstance(node.init, list)
        if scoped_init:
            self.emit("{")
            self.indent_level += 1
            for stmt in node.init:
                self.emit_statement(stmt)
            init = ""
        else:
            init = self.format_statement_fragment(node.init)
        condition = self.visit(node.condition) if node.condition else ""
        update = self.format_statement_fragment(node.update)

        self.emit(f"for ({init}; {condition}; {update}) {{")

        self.indent_level += 1
        if isinstance(node.body, list):
            for stmt in node.body:
                self.emit_statement(stmt)
        else:
            self.emit_statement(node.body)
        self.indent_level -= 1

        self.emit("}")
        if scoped_init:
            self.indent_level -= 1
            self.emit("}")

    def visit_RangeForNode(self, node):
        iterable = self.visit(node.iterable)
        self.emit(f"for {node.name} in {iterable} {{")

        self.indent_level += 1
        if isinstance(node.body, list):
            for stmt in node.body:
                self.emit_statement(stmt)
        else:
            self.emit_statement(node.body)
        self.indent_level -= 1

        self.emit("}")

    def visit_WhileNode(self, node):
        condition = self.visit(node.condition)
        self.emit(f"while ({condition}) {{")

        self.indent_level += 1
        if isinstance(node.body, list):
            for stmt in node.body:
                self.emit_statement(stmt)
        else:
            self.emit_statement(node.body)
        self.indent_level -= 1

        self.emit("}")

    def visit_DoWhileNode(self, node):
        condition = self.visit(node.condition)
        self.emit("do {")

        self.indent_level += 1
        if isinstance(node.body, list):
            for stmt in node.body:
                self.emit_statement(stmt)
        else:
            self.emit_statement(node.body)
        self.indent_level -= 1

        self.emit(f"}} while ({condition});")

    def visit_SwitchNode(self, node):
        expression = self.visit(node.expression)
        self.emit(f"switch ({expression}) {{")

        self.indent_level += 1
        ordered_cases = getattr(node, "ordered_cases", None)
        if ordered_cases is not None:
            for case in ordered_cases:
                if case.value is None:
                    self.emit("default:")
                    self.indent_level += 1
                    for stmt in case.body:
                        self.emit_statement(stmt)
                    self.indent_level -= 1
                else:
                    self.visit(case)
        else:
            for case in node.cases:
                self.visit(case)

            if node.default_case is not None:
                self.emit("default:")
                self.indent_level += 1
                for stmt in node.default_case:
                    self.emit_statement(stmt)
                self.indent_level -= 1

        self.indent_level -= 1
        self.emit("}")

    def visit_CaseNode(self, node):
        value = self.visit(node.value)
        self.emit(f"case {value}:")

        self.indent_level += 1
        for stmt in node.body:
            self.emit_statement(stmt)
        self.indent_level -= 1

    def visit_ReturnNode(self, node):
        if node.value:
            runtime_expression = self.format_cuda_runtime_expression(node.value)
            if runtime_expression is not None:
                comments, value = runtime_expression
                for comment in comments:
                    self.emit(comment)
                self.emit(f"return {value};")
                return

            if isinstance(node.value, InitializerListNode):
                elements = getattr(node.value, "elements", None) or []
                fallback = self.visit(elements[0]) if len(elements) == 1 else "0"
                self.emit(f"return {fallback} /* CUDA initializer list return */;")
                return

            value = self.visit(node.value)
            self.emit(f"return {value};")
        else:
            self.emit("return;")

    def visit_BreakNode(self, node):
        self.emit("break;")

    def visit_ContinueNode(self, node):
        self.emit("continue;")

    def visit_TernaryOpNode(self, node):
        condition = self.visit(node.condition)
        true_expr = self.visit(node.true_expr)
        false_expr = self.visit(node.false_expr)
        return f"({condition} ? {true_expr} : {false_expr})"

    def visit_CastNode(self, node):
        target_type = self.convert_cuda_type_to_crossgl(node.target_type)
        expression = self.visit(node.expression)
        return f"{target_type}({expression})"

    def convert_cuda_variable_type_to_crossgl(self, cuda_type, name):
        """Convert CUDA variable types, using call-site hints for resource handles."""
        resource_type = self.convert_cuda_resource_object_type(cuda_type, name)
        if resource_type is not None:
            return resource_type
        return self.convert_cuda_type_to_crossgl(cuda_type)

    def convert_cuda_struct_member_type_to_crossgl(self, struct_name, cuda_type, name):
        hint = self.struct_resource_member_hints.get((struct_name, name))
        if hint is not None:
            resource_type = self.convert_cuda_resource_object_type_with_hint(
                cuda_type, hint
            )
            if resource_type is not None:
                return resource_type
        return self.convert_cuda_type_to_crossgl(cuda_type)

    def convert_cuda_resource_object_type(self, cuda_type, name):
        hint = self.lookup_resource_object_type_hint(name)
        if hint is None:
            return None
        return self.convert_cuda_resource_object_type_with_hint(cuda_type, hint)

    def convert_cuda_resource_object_type_with_hint(self, cuda_type, hint):
        cuda_type = self.strip_type_qualifiers(cuda_type)

        if self.has_array_suffix(cuda_type):
            base_type = cuda_type.split("[", 1)[0].strip()
            mapped_type = self.convert_cuda_resource_object_type_with_hint(
                base_type, hint
            )
            if mapped_type is None:
                return None
            return self.wrap_mapped_cuda_array_type(cuda_type, mapped_type)

        if "*" in cuda_type:
            pointer_depth = cuda_type.count("*")
            base_type = cuda_type.replace("*", "").strip()
            mapped_type = self.convert_cuda_resource_object_base_type(base_type, hint)
            if mapped_type is None:
                return None
            for _ in range(pointer_depth):
                mapped_type = f"ptr<{mapped_type}>"
            return mapped_type

        return self.convert_cuda_resource_object_base_type(cuda_type, hint)

    def convert_cuda_resource_object_base_type(self, cuda_type, hint):
        cuda_type = self.strip_type_qualifiers(cuda_type)
        if cuda_type == "cudaTextureObject_t" and hint.startswith("sampler"):
            return hint
        if cuda_type == "cudaSurfaceObject_t" and "image" in hint:
            return hint
        return None

    def wrap_mapped_cuda_array_type(self, cuda_type, mapped_type):
        base_type = cuda_type.split("[", 1)[0].strip()
        dimensions = []
        remainder = cuda_type[len(base_type) :].strip()

        while remainder.startswith("["):
            close_index = remainder.find("]")
            if close_index == -1:
                break
            dimensions.append(remainder[1:close_index].strip())
            remainder = remainder[close_index + 1 :].strip()

        for size in reversed(dimensions):
            if size:
                mapped_type = (
                    f"array<{mapped_type}, {self.format_crossgl_array_extent(size)}>"
                )
            else:
                mapped_type = f"array<{mapped_type}>"

        return mapped_type

    def convert_cuda_type_to_crossgl(self, cuda_type):
        cuda_type = self.strip_dependent_template_disambiguators(cuda_type)
        cuda_type = self.strip_type_qualifiers(cuda_type)

        matrix_type = self.convert_native_matrix_helper_name_to_crossgl(cuda_type)
        if matrix_type is not None:
            return matrix_type

        elaborated_type = self.convert_cuda_elaborated_type_to_crossgl(cuda_type)
        if elaborated_type is not None:
            return elaborated_type

        parameter_pack_type = self.convert_cuda_parameter_pack_type_to_crossgl(
            cuda_type
        )
        if parameter_pack_type is not None:
            return parameter_pack_type

        type_mapping = {
            "void": "void",
            "bool": "bool",
            "char": "i8",
            "unsigned char": "u8",
            "short": "i16",
            "unsigned short": "u16",
            "int": "i32",
            "signed int": "i32",
            "uint": "u32",
            "unsigned int": "u32",
            "long": "i64",
            "signed long": "i64",
            "unsigned long": "u64",
            "long long": "i64",
            "signed long long": "i64",
            "unsigned long long": "u64",
            "long long unsigned int": "u64",
            "int8_t": "i8",
            "uint8_t": "u8",
            "int16_t": "i16",
            "uint16_t": "u16",
            "int32_t": "i32",
            "uint32_t": "u32",
            "int64_t": "i64",
            "uint64_t": "u64",
            "half": "f16",
            "__half": "f16",
            "__half2": "vec2<f16>",
            "__nv_bfloat16": "f16",
            "__nv_bfloat162": "vec2<f16>",
            "float": "f32",
            "double": "f64",
            "size_t": "u32",
            **self.VECTOR1_TYPE_MAPPING,
            **self.VECTOR_TYPE_MAPPING,
            "dim3": "vec3<u32>",
        }
        unqualified_type = self.unqualified_leading_cuda_std_type_name(cuda_type)
        if unqualified_type is not None and unqualified_type in type_mapping:
            return type_mapping[unqualified_type]

        cooperative_group_type = self.convert_cooperative_group_type(cuda_type)
        if cooperative_group_type is not None:
            return cooperative_group_type

        resource_type = self.convert_cuda_resource_type(cuda_type)
        if resource_type is not None:
            return resource_type

        unique_ptr_type = self.convert_unique_ptr_type(cuda_type)
        if unique_ptr_type is not None:
            return unique_ptr_type

        span_type = self.convert_cuda_span_type(cuda_type)
        if span_type is not None:
            return span_type

        nested_scoped_template_type = (
            self.convert_nested_scoped_template_type_to_crossgl(cuda_type)
        )
        if nested_scoped_template_type is not None:
            return nested_scoped_template_type

        enable_if_type = self.convert_enable_if_type(cuda_type)
        if enable_if_type is not None:
            return enable_if_type

        wmma_type = self.convert_cuda_wmma_type(cuda_type)
        if wmma_type is not None:
            return wmma_type

        if self.is_decltype_type_name(cuda_type):
            return "auto"

        expression_template_type = (
            self.convert_cuda_expression_template_type_to_crossgl(cuda_type)
        )
        if expression_template_type is not None:
            return expression_template_type

        if self.has_array_suffix(cuda_type):
            return self.convert_cuda_array_type(cuda_type, type_mapping)

        if "*" in cuda_type:
            return self.convert_cuda_pointer_type(cuda_type)

        template_type = None
        if cuda_type in self.cuda_record_names:
            template_type = self.convert_unknown_cuda_template_type_to_crossgl(
                cuda_type
            )
        if template_type is not None:
            return template_type

        mapped_type = type_mapping.get(cuda_type)
        if mapped_type is not None:
            return mapped_type
        scoped_type = self.convert_cuda_scoped_type_to_crossgl(cuda_type)
        if scoped_type is not None:
            return scoped_type
        return cuda_type

    def unqualified_leading_cuda_std_type_name(self, cuda_type):
        normalized_type = str(cuda_type)
        prefix = "::cuda::std::"
        if normalized_type.startswith(prefix):
            return normalized_type[len(prefix) :]
        return None

    def convert_nested_scoped_template_type_to_crossgl(self, cuda_type):
        base_name, template_args = self.parse_cpp_template(cuda_type)
        if not template_args or "::" not in str(cuda_type):
            return None
        if not any(
            self.template_arg_contains_scoped_template(arg) for arg in template_args
        ):
            return None

        return self.convert_scoped_template_type_to_crossgl_generic(
            base_name, template_args
        )

    def convert_scoped_template_type_to_crossgl_generic(self, base_name, template_args):
        converted_args = [
            self.convert_scoped_template_argument_to_crossgl(arg)
            for arg in template_args
        ]
        return (
            f"{self.sanitize_crossgl_type_identifier(base_name)}"
            f"<{', '.join(converted_args)}>"
        )

    def convert_scoped_template_argument_to_crossgl(self, type_name):
        base_name, template_args = self.parse_cpp_template(type_name)
        if template_args and "::" in base_name:
            return self.convert_scoped_template_type_to_crossgl_generic(
                base_name, template_args
            )
        return self.convert_cuda_type_to_crossgl(type_name)

    def template_arg_contains_scoped_template(self, type_name):
        base_name, template_args = self.parse_cpp_template(type_name)
        return bool(template_args and "::" in base_name)

    def strip_dependent_template_disambiguators(self, type_name):
        return str(type_name).replace("::template ", "::")

    def convert_cuda_expression_template_type_to_crossgl(self, cuda_type):
        _, template_args = self.parse_cpp_template(cuda_type)
        if not template_args:
            return None
        if not any(
            self.is_crossgl_unsafe_template_argument(arg) for arg in template_args
        ):
            return None
        return self.convert_unknown_cuda_template_type_to_crossgl(cuda_type)

    def is_crossgl_unsafe_template_argument(self, arg):
        return re.search(r"[()+\-*/%~!=|&?]", str(arg)) is not None

    def is_decltype_type_name(self, type_name):
        return isinstance(type_name, str) and type_name.strip().startswith("decltype(")

    def convert_cooperative_group_type(self, cuda_type):
        metadata = self.cooperative_group_metadata_from_type(cuda_type)
        if metadata is None:
            return None
        group_kind = metadata["kind"]
        if group_kind == "thread_block_tile" and metadata.get("tile_size"):
            return f"cooperative_groups_thread_block_tile_{metadata['tile_size']}"
        return f"cooperative_groups_{group_kind}"

    def format_crossgl_type_syntax(self, type_name):
        return str(type_name).replace(">>", "> >")

    def crossgl_safe_generic_type_argument(self, type_name):
        if "::" in str(type_name):
            return self.sanitize_crossgl_type_identifier(type_name)
        return type_name

    def convert_cuda_resource_type(self, cuda_type):
        base_name, template_args = self.parse_cpp_template(cuda_type)
        if base_name == "texture" and len(template_args) >= 2:
            return self.CUDA_TEXTURE_TYPE_MAPPING.get(template_args[1])
        if base_name == "surface" and len(template_args) >= 2:
            return self.CUDA_SURFACE_TYPE_MAPPING.get(template_args[1])
        return None

    def convert_unique_ptr_type(self, cuda_type):
        base_name, template_args = self.parse_cpp_template(cuda_type)
        if not self.is_unique_ptr_base_name(base_name) or not template_args:
            return None

        target_type, _ = self.unwrap_array_template_type(template_args[0])
        return f"ptr<{self.convert_cuda_type_to_crossgl(target_type)}>"

    def convert_cuda_span_type(self, cuda_type):
        base_name, template_args = self.parse_cpp_template(cuda_type)
        if base_name not in {"cuda::std::span", "std::span"} or not template_args:
            return None

        element_type = self.convert_cuda_type_to_crossgl(template_args[0])
        if len(template_args) >= 2 and not self.is_dynamic_span_extent(
            template_args[1]
        ):
            extent = self.format_crossgl_array_extent(template_args[1])
            return f"array<{element_type}, {extent}>"
        return f"array<{element_type}>"

    def convert_enable_if_type(self, cuda_type):
        base_name, template_args = self.parse_cpp_template(cuda_type)
        if base_name in {"std::enable_if_t", "::std::enable_if_t"} and template_args:
            return "void"
        return None

    def convert_cuda_wmma_type(self, cuda_type):
        normalized_type = self.resolve_namespace_alias_name(str(cuda_type).strip())
        base_name, template_args = self.parse_cpp_template(normalized_type)
        if not self.is_cuda_wmma_type_name(base_name):
            return None

        if not template_args:
            return self.sanitize_crossgl_type_identifier(normalized_type)

        converted_parts = [self.sanitize_crossgl_type_identifier(base_name)]
        for arg in template_args:
            converted_arg = self.convert_cuda_type_to_crossgl(arg)
            converted_parts.append(self.sanitize_crossgl_type_identifier(converted_arg))
        return "_".join(part for part in converted_parts if part)

    def is_cuda_wmma_type_name(self, type_name):
        return str(type_name).startswith(("wmma::", "nvcuda::wmma::"))

    def is_dynamic_span_extent(self, extent):
        extent = self.resolve_namespace_alias_name(str(extent).strip())
        if extent.startswith("::"):
            extent = extent[2:]
        return extent in {
            "dynamic_extent",
            "std::dynamic_extent",
            "cuda::std::dynamic_extent",
        }

    def is_unique_ptr_type_name(self, type_name):
        type_name = self.strip_type_qualifiers(type_name)
        type_name = self.resolve_type_alias(type_name)
        base_name, template_args = self.parse_cpp_template(type_name)
        return self.is_unique_ptr_base_name(base_name) and bool(template_args)

    def is_unique_ptr_base_name(self, base_name):
        return self.is_std_unique_ptr_base_name(base_name)

    def is_std_unique_ptr_base_name(self, base_name):
        return base_name in {"unique_ptr", "std::unique_ptr"}

    def is_std_make_unique_base_name(self, base_name):
        return base_name in {"make_unique", "std::make_unique"}

    def has_array_suffix(self, type_name):
        depth = 0
        for char in str(type_name):
            if char == "<":
                depth += 1
            elif char == ">":
                depth -= 1
            elif char == "[" and depth == 0:
                return True
        return False

    def unwrap_array_template_type(self, type_name):
        type_name = type_name.strip()
        if type_name.endswith("[]"):
            return type_name[:-2].strip(), True
        return type_name, False

    def parse_cpp_template(self, text):
        if not isinstance(text, str):
            return str(text), []

        start = text.find("<")
        if start == -1 or not text.endswith(">"):
            return self.resolve_namespace_alias_name(text), []

        base_name = self.resolve_namespace_alias_name(text[:start].strip())
        args = self.split_cpp_template_args(text[start + 1 : -1])
        return base_name, args

    def resolve_namespace_alias_name(self, name):
        if not isinstance(name, str) or not self.namespace_aliases:
            return name

        resolved = name
        for _ in range(len(self.namespace_aliases)):
            next_resolved = self.resolve_namespace_alias_name_once(resolved)
            if next_resolved == resolved:
                break
            resolved = next_resolved
        return resolved

    def resolve_namespace_alias_name_once(self, name):
        leading_scope = name.startswith("::")
        lookup_name = name[2:] if leading_scope else name
        alias, separator, suffix = lookup_name.partition("::")
        if not separator:
            return name

        target = self.namespace_aliases.get(alias)
        if target is None:
            return name

        resolved = f"{target}::{suffix}"
        if leading_scope and not resolved.startswith("::"):
            resolved = f"::{resolved}"
        return resolved

    def split_cpp_template_args(self, args_text):
        args = []
        depth = 0
        start = 0

        for index, char in enumerate(args_text):
            if char == "<":
                depth += 1
            elif char == ">":
                depth -= 1
            elif char == "," and depth == 0:
                args.append(args_text[start:index].strip())
                start = index + 1

        tail = args_text[start:].strip()
        if tail:
            args.append(tail)
        return args

    def convert_cuda_pointer_type(self, cuda_type):
        pointer_depth = cuda_type.count("*")
        base_type = cuda_type.replace("*", "").strip()
        base_type = self.strip_function_pointer_parameter_list(base_type)
        mapped_type = self.convert_cuda_type_to_crossgl(base_type)

        for _ in range(pointer_depth):
            mapped_type = self.wrap_crossgl_pointer_type(mapped_type)

        return mapped_type

    def convert_cuda_pointer_element_type(self, cuda_type):
        pointer_depth = cuda_type.count("*")
        base_type = cuda_type.replace("*", "").strip()
        base_type = self.strip_function_pointer_parameter_list(base_type)
        mapped_type = self.convert_cuda_type_to_crossgl(base_type)

        for _ in range(max(0, pointer_depth - 1)):
            mapped_type = self.wrap_crossgl_pointer_type(mapped_type)

        return mapped_type

    def strip_function_pointer_parameter_list(self, type_name):
        """Keep imported C++ function-pointer types reparsable in CrossGL."""
        return re.sub(r"\s*\(\s*\)\s*$", "", str(type_name)).strip()

    def wrap_crossgl_pointer_type(self, mapped_type):
        element_type = self.crossgl_safe_generic_type_argument(mapped_type)
        return f"ptr<{element_type}>"

    def strip_type_qualifiers(self, type_name):
        qualifiers = {
            "const",
            "volatile",
            "__restrict__",
            "__restrict",
            "restrict",
            "__grid_constant__",
            "typename",
            "&",
            "&&",
        }
        return " ".join(
            part for part in str(type_name).split() if part not in qualifiers
        )

    def convert_cuda_array_type(self, cuda_type, type_mapping):
        base_type = cuda_type.split("[", 1)[0].strip()
        dimensions = []
        remainder = cuda_type[len(base_type) :].strip()

        while remainder.startswith("["):
            close_index = remainder.find("]")
            if close_index == -1:
                break
            dimensions.append(remainder[1:close_index].strip())
            remainder = remainder[close_index + 1 :].strip()

        mapped_type = type_mapping.get(base_type)
        if mapped_type is None:
            mapped_type = self.convert_cuda_type_to_crossgl(base_type)
        for size in reversed(dimensions):
            if size:
                mapped_type = (
                    f"array<{mapped_type}, {self.format_crossgl_array_extent(size)}>"
                )
            else:
                mapped_type = f"array<{mapped_type}>"

        return mapped_type

    def convert_cuda_builtin_function(self, func_name):
        function_mapping = {
            "sqrtf": "sqrt",
            "powf": "pow",
            "sinf": "sin",
            "cosf": "cos",
            "tanf": "tan",
            "sinhf": "sinh",
            "coshf": "cosh",
            "tanhf": "tanh",
            "asinf": "asin",
            "acosf": "acos",
            "atanf": "atan",
            "logf": "log",
            "log2f": "log2",
            "expf": "exp",
            "__expf": "exp",
            "exp2f": "exp2",
            "fabsf": "abs",
            "rsqrtf": "inversesqrt",
            "roundf": "round",
            "rintf": "round",
            "nearbyintf": "round",
            "truncf": "trunc",
            "atan2f": "atan2",
            "fmaf": "fma",
            "fmodf": "mod",
            "fminf": "min",
            "fmaxf": "max",
            "lerp": "mix",
            "floorf": "floor",
            "ceilf": "ceil",
            "sqrt": "sqrt",
            "pow": "pow",
            "sin": "sin",
            "cos": "cos",
            "tan": "tan",
            "log": "log",
            "exp": "exp",
            "fabs": "abs",
            "rsqrt": "inversesqrt",
            "rint": "round",
            "nearbyint": "round",
            "fmod": "mod",
            "fmin": "min",
            "fmax": "max",
            "__threadfence": "memoryBarrier",
            "__threadfence_block": "memoryBarrier",
            "__threadfence_system": "memoryBarrier",
            "__popc": "bitCount",
            "__popcll": "bitCount",
            "floor": "floor",
            "ceil": "ceil",
            "bool": "bool",
            "char": "i8",
            "short": "i16",
            "int": "i32",
            "uint": "u32",
            "long": "i64",
            "int8_t": "i8",
            "uint8_t": "u8",
            "int16_t": "i16",
            "uint16_t": "u16",
            "int32_t": "i32",
            "uint32_t": "u32",
            "int64_t": "i64",
            "uint64_t": "u64",
            "half": "f16",
            "__half": "f16",
            "__half2": "vec2<f16>",
            "__nv_bfloat16": "f16",
            "__nv_bfloat162": "vec2<f16>",
            "float": "f32",
            "double": "f64",
            "size_t": "u32",
            **self.VECTOR_CONSTRUCTOR_MAPPING,
            "dim3": "vec3<u32>",
        }

        normalized_func_name = self.normalize_cuda_builtin_function_name(func_name)
        matrix_constructor = self.convert_native_matrix_helper_name_to_crossgl(
            normalized_func_name
        )
        if matrix_constructor is not None:
            return matrix_constructor
        return function_mapping.get(normalized_func_name, func_name)

    def normalize_cuda_builtin_function_name(self, func_name):
        if not isinstance(func_name, str):
            return func_name
        normalized_name = self.resolve_namespace_alias_name(func_name)
        if normalized_name.startswith("::"):
            normalized_name = normalized_name[2:]
        for namespace in ("cuda::std::", "std::"):
            if normalized_name.startswith(namespace):
                return normalized_name[len(namespace) :]
        return normalized_name

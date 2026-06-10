"""HIP to CrossGL Code Generator"""

import re

from .HipAst import (
    ArrayAccessNode,
    AssignmentNode,
    CastNode,
    DesignatedInitializerNode,
    EnumNode,
    FunctionCallNode,
    FunctionNode,
    HipDevicePropertyNode,
    InitializerListNode,
    MemberAccessNode,
    StructNode,
    TypeAliasNode,
    UnaryOpNode,
    VariableNode,
)


class HipToCrossGLConverter:
    """Serialize HIP backend AST nodes back into CrossGL source."""

    CPP_NUMERIC_LITERAL_WITH_SEPARATOR = re.compile(
        r"^(?=.*')"
        r"(?:"
        r"0[xX](?:(?:[0-9a-fA-F](?:'?[0-9a-fA-F])*)?\.(?:[0-9a-fA-F](?:'?[0-9a-fA-F])*)|(?:[0-9a-fA-F](?:'?[0-9a-fA-F])*)\.?)[pP][+-]?\d(?:'?\d)*"
        r"|"
        r"0[xX][0-9a-fA-F](?:'?[0-9a-fA-F])*"
        r"|0[bB][01](?:'?[01])*"
        r"|(?:\d(?:'?\d)*)(?:\.(?:\d(?:'?\d)*)?)?"
        r"(?:[eE][+-]?\d(?:'?\d)*)?"
        r"|(?:\d(?:'?\d)*)?\.(?:\d(?:'?\d)*)(?:[eE][+-]?\d(?:'?\d)*)?"
        r")"
        r"[fFdDlLuU]*$"
    )
    CPP_INTEGER_LITERAL = re.compile(
        r"^(?P<body>"
        r"0[xX][0-9a-fA-F](?:'?[0-9a-fA-F])*"
        r"|0[bB][01](?:'?[01])*"
        r"|\d(?:'?\d)*"
        r")(?P<suffix>[uUlL]*)$"
    )

    HIP_RUNTIME_ERROR_WRAPPER_NAMES = {
        "CHECK_HIP",
        "HIP_CHECK",
        "HIPCHECK",
        "checkHip",
        "checkHipErrors",
        "hipCheck",
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
    HIP_TEXTURE_TYPE_MAPPING = {
        "1": "sampler1D",
        "2": "sampler2D",
        "3": "sampler3D",
        "hipTextureType1D": "sampler1D",
        "hipTextureType1DLayered": "sampler1DArray",
        "hipTextureType2D": "sampler2D",
        "hipTextureType2DLayered": "sampler2DArray",
        "hipTextureType3D": "sampler3D",
        "hipTextureTypeCubemap": "samplerCube",
        "hipTextureTypeCubemapLayered": "samplerCubeArray",
        "cudaTextureType1D": "sampler1D",
        "cudaTextureType1DLayered": "sampler1DArray",
        "cudaTextureType2D": "sampler2D",
        "cudaTextureType2DLayered": "sampler2DArray",
        "cudaTextureType3D": "sampler3D",
        "cudaTextureTypeCubemap": "samplerCube",
        "cudaTextureTypeCubemapLayered": "samplerCubeArray",
    }
    HIP_SURFACE_TYPE_MAPPING = {
        "1": "image1D",
        "2": "image2D",
        "3": "image3D",
        "hipSurfaceType1D": "image1D",
        "hipSurfaceType1DLayered": "image1DArray",
        "hipSurfaceType2D": "image2D",
        "hipSurfaceType2DLayered": "image2DArray",
        "hipSurfaceType3D": "image3D",
        "hipSurfaceTypeCubemap": "imageCube",
        "hipSurfaceTypeCubemapLayered": "imageCubeArray",
        "cudaSurfaceType1D": "image1D",
        "cudaSurfaceType1DLayered": "image1DArray",
        "cudaSurfaceType2D": "image2D",
        "cudaSurfaceType2DLayered": "image2DArray",
        "cudaSurfaceType3D": "image3D",
        "cudaSurfaceTypeCubemap": "imageCube",
        "cudaSurfaceTypeCubemapLayered": "imageCubeArray",
    }
    HIP_TEXTURE_CALL_TYPE_HINTS = {
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
    HIP_SURFACE_CALL_TYPE_HINTS = {
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
    HIP_FUNCTION_ATTRIBUTE_MEMBERS = {
        "binaryVersion",
        "cacheModeCA",
        "constSizeBytes",
        "localSizeBytes",
        "maxDynamicSharedSizeBytes",
        "maxThreadsPerBlock",
        "numRegs",
        "preferredShmemCarveout",
        "ptxVersion",
        "sharedSizeBytes",
    }
    HIP_CHANNEL_DESCRIPTOR_MEMBERS = {"f", "w", "x", "y", "z"}
    HIP_ARRAY_DESCRIPTOR_MEMBERS = {"Format", "Height", "NumChannels", "Width"}
    HIP_ARRAY3D_DESCRIPTOR_MEMBERS = {
        "Depth",
        "Flags",
        "Format",
        "Height",
        "NumChannels",
        "Width",
    }
    HIP_EXTENT_MEMBERS = {"depth", "height", "width"}
    HIP_RESOURCE_DESCRIPTOR_MEMBERS = {"resType"}
    HIP_RESOURCE_DESCRIPTOR_NESTED_MEMBERS = {
        "res.linear.desc.f",
        "res.linear.desc.w",
        "res.linear.desc.x",
        "res.linear.desc.y",
        "res.linear.desc.z",
        "res.linear.sizeInBytes",
        "res.pitch2D.desc.f",
        "res.pitch2D.desc.w",
        "res.pitch2D.desc.x",
        "res.pitch2D.desc.y",
        "res.pitch2D.desc.z",
        "res.pitch2D.height",
        "res.pitch2D.pitchInBytes",
        "res.pitch2D.width",
    }
    HIP_TEXTURE_DESCRIPTOR_MEMBERS = {
        "disableTrilinearOptimization",
        "filterMode",
        "flags",
        "maxAnisotropy",
        "maxMipmapLevelClamp",
        "minMipmapLevelClamp",
        "mipmapFilterMode",
        "mipmapLevelBias",
        "normalizedCoords",
        "readMode",
        "sRGB",
    }
    HIP_TEXTURE_DESCRIPTOR_INDEXED_MEMBERS = {
        "addressMode[0]",
        "addressMode[1]",
        "addressMode[2]",
        "borderColor[0]",
        "borderColor[1]",
        "borderColor[2]",
        "borderColor[3]",
    }
    HIP_RESOURCE_VIEW_DESCRIPTOR_MEMBERS = {
        "depth",
        "firstLayer",
        "firstMipmapLevel",
        "format",
        "height",
        "lastLayer",
        "lastMipmapLevel",
        "width",
    }
    CROSSGL_RESERVED_IDENTIFIERS = {"buffer", "in", "precision"}
    CPP_OPERATOR_FUNCTION_NAME_PARTS = {
        "+": "plus",
        "-": "minus",
        "*": "mul",
        "/": "div",
        "%": "mod",
        "&": "bit_and",
        "|": "bit_or",
        "^": "bit_xor",
        "~": "bit_not",
        "!": "not",
        "=": "assign",
        "<": "lt",
        ">": "gt",
        "+=": "plus_assign",
        "-=": "minus_assign",
        "*=": "mul_assign",
        "/=": "div_assign",
        "%=": "mod_assign",
        "&=": "bit_and_assign",
        "|=": "bit_or_assign",
        "^=": "bit_xor_assign",
        "==": "eq",
        "!=": "ne",
        "<=>": "three_way",
        "<=": "le",
        ">=": "ge",
        "&&": "logical_and",
        "||": "logical_or",
        "<<": "shift_left",
        ">>": "shift_right",
        "<<=": "shift_left_assign",
        ">>=": "shift_right_assign",
        "++": "increment",
        "--": "decrement",
        "[]": "index",
        "()": "call",
    }
    CPP_SCALAR_TYPE_ALIASES = {
        **{
            f"std::{name}": name
            for name in (
                "int8_t",
                "uint8_t",
                "int16_t",
                "uint16_t",
                "int32_t",
                "uint32_t",
                "int64_t",
                "uint64_t",
                "size_t",
            )
        },
        **{
            f"cuda::std::{name}": name
            for name in (
                "int8_t",
                "uint8_t",
                "int16_t",
                "uint16_t",
                "int32_t",
                "uint32_t",
                "int64_t",
                "uint64_t",
                "size_t",
            )
        },
    }

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
        self.resource_object_hint_scopes = []
        self.cooperative_group_scopes = [{}]
        self.variable_type_scopes = [{}]
        self.device_property_source_scopes = [{}]
        self.device_attribute_source_scopes = [{}]
        self.device_query_source_scopes = [{}]
        self.member_query_source_scopes = [{}]
        self.suppress_device_property_member_access = 0
        self.suppress_device_attribute_value_access = 0
        self.suppress_device_query_value_access = 0
        self.suppress_identifier_name_rewrite = 0
        self.generated_matrix_helper_types = set()
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
        self.resource_object_hint_scopes = []
        self.cooperative_group_scopes = [{}]
        self.variable_type_scopes = [{}]
        self.device_property_source_scopes = [{}]
        self.device_attribute_source_scopes = [{}]
        self.device_query_source_scopes = [{}]
        self.member_query_source_scopes = [{}]
        self.suppress_device_property_member_access = 0
        self.suppress_device_attribute_value_access = 0
        self.suppress_device_query_value_access = 0
        self.suppress_identifier_name_rewrite = 0
        self.anonymous_enum_count = 0
        self.visit(ast_node)
        return "\n".join(self.output)

    def visit(self, node):
        """Dispatch a HIP backend AST node to its converter method."""
        method_name = f"visit_{type(node).__name__}"
        visitor = getattr(self, method_name, self.generic_visit)
        return visitor(node)

    def generic_visit(self, node):
        """Fallback converter for primitive values, lists, and unknown nodes."""
        if isinstance(node, str):
            string_literal = self.format_cpp_string_literal_expression(node)
            if string_literal is not None:
                return string_literal
            node = self.normalize_cpp_numeric_literal(node)
            attribute_expression = self.format_hip_device_attribute_read(node)
            if attribute_expression is not None:
                return attribute_expression
            query_expression = self.format_hip_device_query_read(node)
            if query_expression is not None:
                return query_expression
            user_defined_name = self.format_user_defined_function_call_name(node)
            if user_defined_name is not None:
                return user_defined_name
            return self.output_identifier_name(node)
        elif isinstance(node, list):
            return [self.visit(item) for item in node]
        else:
            return str(node)

    def normalize_cpp_numeric_literal(self, value):
        integer_literal = self.CPP_INTEGER_LITERAL.match(value)
        if integer_literal:
            body = integer_literal.group("body").replace("'", "")
            suffix = integer_literal.group("suffix").lower()
            if "u" in suffix:
                return f"{body}u"
            return body

        if "'" in value and self.CPP_NUMERIC_LITERAL_WITH_SEPARATOR.match(value):
            return value.replace("'", "")
        return value

    def format_cpp_string_literal_expression(self, value):
        if not isinstance(value, str):
            return None

        text = value.strip()
        if not self.maybe_cpp_string_literal_expression(text):
            return None

        parsed = self.parse_cpp_string_literal_sequence(text)
        if parsed is None:
            return None

        return self.quote_crossgl_string_literal(parsed)

    def maybe_cpp_string_literal_expression(self, text):
        if not text:
            return False
        return text.startswith(
            ('"', 'R"', 'u8"', 'u8R"', 'u"', 'uR"', 'U"', 'UR"', 'L"', 'LR"')
        )

    def parse_cpp_string_literal_sequence(self, text):
        parts = []
        index = 0

        while True:
            index = self.skip_cpp_string_literal_whitespace(text, index)
            if index >= len(text):
                break

            parsed = self.parse_cpp_string_literal_at(text, index)
            if parsed is None:
                return None

            content, index = parsed
            parts.append(content)

        if not parts:
            return None

        return "".join(parts)

    def skip_cpp_string_literal_whitespace(self, text, index):
        while index < len(text) and text[index].isspace():
            index += 1
        return index

    def parse_cpp_string_literal_at(self, text, index):
        prefix, literal_index = self.parse_cpp_string_literal_prefix(text, index)
        if literal_index >= len(text):
            return None

        if text.startswith('R"', literal_index):
            return self.parse_cpp_raw_string_literal(text, literal_index + 2)

        if text[literal_index] == '"':
            return self.parse_cpp_ordinary_string_literal(text, literal_index + 1)

        if prefix:
            return None

        return None

    def parse_cpp_string_literal_prefix(self, text, index):
        for prefix in ("u8", "u", "U", "L"):
            if text.startswith(prefix, index):
                return prefix, index + len(prefix)
        return "", index

    def parse_cpp_raw_string_literal(self, text, index):
        delimiter_end = text.find("(", index)
        if delimiter_end == -1:
            return None

        delimiter = text[index:delimiter_end]
        closing = ")" + delimiter + '"'
        content_start = delimiter_end + 1
        content_end = text.find(closing, content_start)
        if content_end == -1:
            return None

        return text[content_start:content_end], content_end + len(closing)

    def parse_cpp_ordinary_string_literal(self, text, index):
        chars = []

        while index < len(text):
            char = text[index]
            if char == '"':
                return "".join(chars), index + 1
            if char == "\\":
                decoded = self.decode_cpp_string_escape(text, index)
                if decoded is None:
                    return None
                value, index = decoded
                chars.append(value)
                continue
            chars.append(char)
            index += 1

        return None

    def decode_cpp_string_escape(self, text, index):
        escape_index = index + 1
        if escape_index >= len(text):
            return None

        escaped = text[escape_index]
        escape_map = {
            "\\": "\\",
            '"': '"',
            "'": "'",
            "?": "?",
            "a": "\a",
            "b": "\b",
            "f": "\f",
            "n": "\n",
            "r": "\r",
            "t": "\t",
            "v": "\v",
        }
        if escaped in escape_map:
            return escape_map[escaped], escape_index + 1

        if escaped == "\n":
            return "", escape_index + 1

        if escaped == "\r":
            next_index = escape_index + 1
            if next_index < len(text) and text[next_index] == "\n":
                next_index += 1
            return "", next_index

        if escaped == "x":
            digit_index = escape_index + 1
            while (
                digit_index < len(text)
                and text[digit_index] in "0123456789abcdefABCDEF"
            ):
                digit_index += 1
            if digit_index == escape_index + 1:
                return "x", digit_index
            return chr(int(text[escape_index + 1 : digit_index], 16)), digit_index

        if escaped in "01234567":
            digit_index = escape_index + 1
            while (
                digit_index < len(text)
                and digit_index < escape_index + 3
                and text[digit_index] in "01234567"
            ):
                digit_index += 1
            return chr(int(text[escape_index:digit_index], 8)), digit_index

        return escaped, escape_index + 1

    def quote_crossgl_string_literal(self, value):
        escaped = []
        for char in value:
            if char == "\\":
                escaped.append("\\\\")
            elif char == '"':
                escaped.append('\\"')
            elif char == "\n":
                escaped.append("\\n")
            elif char == "\r":
                escaped.append("\\r")
            elif char == "\t":
                escaped.append("\\t")
            elif char == "\b":
                escaped.append("\\b")
            elif char == "\f":
                escaped.append("\\f")
            elif char == "\v":
                escaped.append("\\v")
            elif char == "\a":
                escaped.append("\\a")
            elif char == "\0":
                escaped.append("\\0")
            else:
                escaped.append(char)
        return '"' + "".join(escaped) + '"'

    def emit(self, code):
        """Append a line of CrossGL output using the current indentation level."""
        if code.strip():
            self.output.append("    " * self.indent_level + code)
        else:
            self.output.append("")

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

            for stmt in getattr(current, "statements", []):
                collect(stmt)
            for function in getattr(current, "functions", []):
                collect(function)
            for kernel in getattr(current, "kernels", []):
                collect(kernel)

        collect(node)
        names.discard(None)
        return names

    def collect_generated_matrix_helper_types(self, node):
        structs = []
        for stmt in getattr(node, "statements", []) or []:
            if isinstance(stmt, StructNode):
                structs.append(stmt)
        structs.extend(getattr(node, "structs", []) or [])
        return {
            struct.name
            for struct in structs
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
        for param in getattr(node, "params", []) or []:
            if isinstance(param, dict):
                candidate_types.append(param.get("type", ""))
            else:
                candidate_types.append(getattr(param, "vtype", ""))
        return any(
            self.strip_type_qualifiers(candidate_type).strip()
            in self.generated_matrix_helper_types
            for candidate_type in candidate_types
        )

    def is_user_defined_function(self, func_name):
        return isinstance(func_name, str) and func_name in self.user_function_names

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
                hint = self.get_resource_object_call_hint(current.name, current.args)
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
        for stmt in getattr(node, "statements", []):
            if isinstance(stmt, FunctionNode):
                self.collect_global_resource_object_type_hints_from_function(
                    stmt, global_names, hints
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
                hint = self.get_resource_object_call_hint(current.name, current.args)
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
            stmt.name
            for stmt in getattr(node, "statements", [])
            if isinstance(stmt, VariableNode)
        }

    def collect_declared_variable_names(self, node):
        names = set()
        for param in getattr(node, "params", []) or []:
            if isinstance(param, dict):
                name = param.get("name")
            else:
                name = getattr(param, "name", None)
            if name:
                names.add(name)

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

    def get_resource_object_call_hint(self, function_name, args=None):
        base_name, _ = self.parse_cpp_template(function_name)
        if self.is_user_defined_function(base_name):
            return None
        if base_name in self.HIP_TEXTURE_CALL_TYPE_HINTS:
            return 0, self.HIP_TEXTURE_CALL_TYPE_HINTS[base_name]
        if base_name in self.HIP_SURFACE_CALL_TYPE_HINTS:
            if base_name.endswith("write"):
                return 1, self.HIP_SURFACE_CALL_TYPE_HINTS[base_name]
            if args and self.is_surface_pointer_output_argument(args[0]):
                return 1, self.HIP_SURFACE_CALL_TYPE_HINTS[base_name]
            return 0, self.HIP_SURFACE_CALL_TYPE_HINTS[base_name]
        return None

    def is_surface_pointer_output_argument(self, expression):
        if isinstance(expression, UnaryOpNode) and expression.op == "&":
            return True
        if isinstance(expression, CastNode):
            return self.is_surface_pointer_output_argument(expression.expression)
        if isinstance(expression, str):
            return self.is_surface_output_target(expression)
        return False

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

    def push_variable_type_scope(self):
        self.variable_type_scopes.append({})
        self.push_vector1_name_scope()
        self.device_property_source_scopes.append({})
        self.device_attribute_source_scopes.append({})
        self.device_query_source_scopes.append({})
        self.member_query_source_scopes.append({})

    def pop_variable_type_scope(self):
        if len(self.variable_type_scopes) > 1:
            self.variable_type_scopes.pop()
        self.pop_vector1_name_scope()
        if len(self.device_property_source_scopes) > 1:
            self.device_property_source_scopes.pop()
        if len(self.device_attribute_source_scopes) > 1:
            self.device_attribute_source_scopes.pop()
        if len(self.device_query_source_scopes) > 1:
            self.device_query_source_scopes.pop()
        if len(self.member_query_source_scopes) > 1:
            self.member_query_source_scopes.pop()

    def register_variable_type(self, name, type_name):
        if not name or not type_name:
            return
        if not self.variable_type_scopes:
            self.variable_type_scopes.append({})
        self.variable_type_scopes[-1][name] = type_name
        self.clear_device_property_source(name)
        self.clear_device_attribute_source(name)
        self.clear_device_query_source(name)
        self.clear_member_query_source(name)

    def lookup_variable_type(self, name):
        for scope in reversed(self.variable_type_scopes):
            if name in scope:
                return scope[name]
        return None

    def push_vector1_name_scope(self):
        self.vector1_name_scopes.append({})

    def pop_vector1_name_scope(self):
        if len(self.vector1_name_scopes) > 1:
            self.vector1_name_scopes.pop()

    def register_vector1_name(self, name, type_name):
        if not name:
            return
        if not self.vector1_name_scopes:
            self.vector1_name_scopes.append({})
        self.vector1_name_scopes[-1][name] = (
            self.hip_vector1_scalar_type(type_name) is not None
        )

    def is_vector1_name(self, name):
        if not isinstance(name, str):
            return False
        for scope in reversed(self.vector1_name_scopes):
            if name in scope:
                return scope[name]
        return False

    def hip_vector1_scalar_type(self, type_name):
        type_name = self.strip_type_qualifiers(type_name)
        if "*" in type_name or self.has_array_suffix(type_name):
            return None
        return self.VECTOR1_TYPE_MAPPING.get(type_name)

    def register_device_property_source(self, name, device_id):
        if not name or device_id is None:
            return
        if not self.device_property_source_scopes:
            self.device_property_source_scopes.append({})
        self.device_property_source_scopes[-1][name] = device_id
        self.clear_member_query_source(name)

    def clear_device_property_source(self, name):
        if not name:
            return
        if not self.device_property_source_scopes:
            self.device_property_source_scopes.append({})
        self.device_property_source_scopes[-1][name] = None

    def lookup_device_property_source(self, name):
        for scope in reversed(self.device_property_source_scopes):
            if name in scope:
                return scope[name]
        return None

    def register_device_attribute_source(self, name, attribute_name, device_id):
        if not name or attribute_name is None or device_id is None:
            return
        if not self.device_attribute_source_scopes:
            self.device_attribute_source_scopes.append({})
        self.device_attribute_source_scopes[-1][name] = (attribute_name, device_id)
        self.clear_device_query_source(name)
        self.clear_member_query_source(name)

    def clear_device_attribute_source(self, name):
        if not name:
            return
        if not self.device_attribute_source_scopes:
            self.device_attribute_source_scopes.append({})
        self.device_attribute_source_scopes[-1][name] = None

    def lookup_device_attribute_source(self, name):
        for scope in reversed(self.device_attribute_source_scopes):
            if name in scope:
                return scope[name]
        return None

    def register_device_query_source(self, name, query_name, device_id=None):
        if not name or query_name is None:
            return
        if not self.device_query_source_scopes:
            self.device_query_source_scopes.append({})
        self.device_query_source_scopes[-1][name] = (query_name, device_id)
        self.clear_device_property_source(name)
        self.clear_device_attribute_source(name)
        self.clear_member_query_source(name)

    def clear_device_query_source(self, name):
        if not name:
            return
        if not self.device_query_source_scopes:
            self.device_query_source_scopes.append({})
        self.device_query_source_scopes[-1][name] = None

    def lookup_device_query_source(self, name):
        for scope in reversed(self.device_query_source_scopes):
            if name in scope:
                return scope[name]
        return None

    def register_member_query_sources(self, name, member_queries):
        if not name or not member_queries:
            return
        if not self.member_query_source_scopes:
            self.member_query_source_scopes.append({})
        self.member_query_source_scopes[-1][name] = dict(member_queries)
        self.clear_device_property_source(name)
        self.clear_device_attribute_source(name)
        self.clear_device_query_source(name)

    def clear_member_query_source(self, name):
        if not name:
            return
        if not self.member_query_source_scopes:
            self.member_query_source_scopes.append({})
        self.member_query_source_scopes[-1][name] = None

    def lookup_member_query_source(self, name):
        for scope in reversed(self.member_query_source_scopes):
            if name in scope:
                return scope[name]
        return None

    def clear_device_metadata_source(self, name):
        self.clear_device_property_source(name)
        self.clear_device_attribute_source(name)
        self.clear_device_query_source(name)
        self.clear_member_query_source(name)

    def clear_lvalue_metadata_source(self, node):
        self.clear_device_metadata_source(self.get_runtime_pointer_target_name(node))
        root_name = self.get_lvalue_metadata_root_name(node)
        if root_name is not None:
            self.clear_device_metadata_source(root_name)

    def get_lvalue_metadata_root_name(self, node):
        if isinstance(node, CastNode):
            return self.get_lvalue_metadata_root_name(node.expression)
        if isinstance(node, UnaryOpNode):
            return self.get_lvalue_metadata_root_name(node.operand)
        if isinstance(node, MemberAccessNode):
            return self.get_lvalue_metadata_root_name(node.object)
        if isinstance(node, ArrayAccessNode):
            return self.get_lvalue_metadata_root_name(node.array)
        if isinstance(node, str):
            return node
        return None

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

    def register_variable_declaration_name(self, name):
        if self.is_simple_identifier(name):
            return self.register_identifier_name(name)

        output_name = self.sanitize_qualified_variable_name(name)
        if output_name is None:
            return name

        if not self.identifier_name_scopes:
            self.identifier_name_scopes.append({})
        self.identifier_name_scopes[-1][name] = output_name
        return output_name

    def output_identifier_name(self, name):
        if self.suppress_identifier_name_rewrite != 0:
            return name
        for scope in reversed(self.identifier_name_scopes):
            if name in scope:
                return scope[name]
        if not self.is_simple_identifier(name):
            return name
        return name

    def sanitize_identifier_name(self, name):
        if name in self.CROSSGL_RESERVED_IDENTIFIERS:
            return f"{name}_"
        return name

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

    def convert_hip_record_name_to_crossgl(self, name):
        base_name, template_args = self.parse_cpp_template(name)
        if not template_args:
            return self.sanitize_crossgl_type_identifier(name)

        parts = [self.sanitize_crossgl_type_identifier(base_name)]
        for arg in template_args:
            converted_arg = self.convert_hip_type_to_crossgl(arg)
            parts.append(self.sanitize_crossgl_type_identifier(converted_arg))
        return "_".join(part for part in parts if part)

    def format_crossgl_array_extent(self, size):
        size = str(size).strip()
        if not size:
            return size
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*|(?:0[xX][0-9A-Fa-f]+|\d+)[uU]?", size):
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
            extent = f"hip_array_extent_{extent}"
        return self.sanitize_identifier_name(extent)

    def sanitize_qualified_variable_name(self, name):
        if not isinstance(name, str) or "::" not in name:
            return None

        stripped = self.strip_cpp_template_arguments(name)
        chars = []
        previous_separator = False
        for char in stripped:
            if char.isalnum() or char == "_":
                chars.append(char)
                previous_separator = False
            elif chars and not previous_separator:
                chars.append("_")
                previous_separator = True

        output_name = "".join(chars)
        if not output_name:
            return None
        if output_name[0].isdigit():
            output_name = f"_{output_name}"
        return self.sanitize_identifier_name(output_name)

    def format_function_declaration_name(self, name):
        operator_name = self.format_cpp_operator_function_name(name)
        if operator_name is not None:
            return operator_name
        base_name, template_args = self.parse_cpp_template(name)
        if template_args:
            return self.format_function_declaration_name(base_name)
        if self.is_simple_identifier(name):
            return self.sanitize_identifier_name(name)
        if not isinstance(name, str) or "::" not in name:
            return self.sanitize_crossgl_type_identifier(name)

        normalized_name = name.replace("::~", "::destructor_")
        return self.sanitize_qualified_variable_name(normalized_name) or name

    def format_cpp_operator_function_name(self, name):
        if not isinstance(name, str):
            return None

        normalized_name = name.replace("::~", "::destructor_")
        parts = normalized_name.split("::")
        raw_operator = parts[-1]
        if not raw_operator.startswith("operator"):
            return None

        operator = raw_operator[len("operator") :]
        if not operator:
            return None

        suffix = self.CPP_OPERATOR_FUNCTION_NAME_PARTS.get(operator)
        if suffix is None:
            suffix = re.sub(r"[^A-Za-z0-9_]+", "_", operator).strip("_")
            suffix = re.sub(r"_+", "_", suffix)
        if not suffix:
            return None

        parts[-1] = f"operator_{suffix}"
        return self.sanitize_qualified_variable_name("::".join(parts)) or parts[-1]

    def format_function_call_name(self, name):
        user_defined_name = self.format_user_defined_function_call_name(name)
        if user_defined_name is not None:
            return user_defined_name
        return name

    def format_user_defined_function_call_name(self, name):
        if self.is_user_defined_function(name):
            return self.format_function_declaration_name(name)

        base_name, template_args = self.parse_cpp_template(name)
        if template_args and self.is_user_defined_function(base_name):
            return self.format_function_declaration_name(base_name)

        return None

    def strip_cpp_template_arguments(self, text):
        stripped = []
        depth = 0
        for char in text:
            if char == "<":
                depth += 1
                continue
            if char == ">" and depth > 0:
                depth -= 1
                continue
            if depth == 0:
                stripped.append(char)
        return "".join(stripped)

    def visit_lvalue_expression(self, node):
        self.suppress_device_property_member_access += 1
        self.suppress_device_attribute_value_access += 1
        self.suppress_device_query_value_access += 1
        try:
            return self.visit(node)
        finally:
            self.suppress_device_property_member_access -= 1
            self.suppress_device_attribute_value_access -= 1
            self.suppress_device_query_value_access -= 1

    def visit_runtime_argument_expression(self, node):
        self.suppress_device_property_member_access += 1
        self.suppress_device_attribute_value_access += 1
        self.suppress_device_query_value_access += 1
        try:
            return self.visit(node)
        finally:
            self.suppress_device_property_member_access -= 1
            self.suppress_device_attribute_value_access -= 1
            self.suppress_device_query_value_access -= 1

    def visit_source_comment_expression(self, node):
        self.suppress_identifier_name_rewrite += 1
        try:
            return self.visit(node)
        finally:
            self.suppress_identifier_name_rewrite -= 1

    def emit_statement(self, stmt):
        """Render and append one converted statement."""
        if isinstance(stmt, list):
            for item in stmt:
                self.emit_statement(item)
            return

        if self.emit_hip_runtime_call_statement(stmt):
            return

        result = self.visit(stmt)
        if isinstance(result, str) and result.strip():
            self.emit(f"{result};")

    def emit_hip_runtime_call_statement(self, stmt):
        if not isinstance(stmt, FunctionCallNode):
            return False
        if self.is_user_defined_function(stmt.name):
            return False

        runtime_wrapper = self.format_hip_runtime_wrapper_expression(stmt)
        if runtime_wrapper is not None:
            comments, _ = runtime_wrapper
            for comment in comments:
                self.emit(comment)
            return True

        comments = self.format_hip_runtime_call(stmt)
        if comments is None:
            return False

        for comment in comments:
            self.emit(comment)
        return True

    def format_hip_runtime_status_expression(self, value):
        if not isinstance(value, FunctionCallNode):
            return None
        if self.is_user_defined_function(value.name):
            return None

        wrapped_runtime = self.format_hip_runtime_wrapper_expression(value)
        if wrapped_runtime is not None:
            return wrapped_runtime

        comments = self.format_hip_runtime_call(value)
        if comments is None:
            return None

        if value.name == "hipGetSurfaceObjectResourceDesc":
            return comments, "hipErrorNotSupported"

        success_value = (
            "HIPRTC_SUCCESS" if value.name.startswith("hiprtc") else "hipSuccess"
        )
        return comments, success_value

    def format_hip_runtime_wrapper_expression(self, value):
        if not isinstance(value, FunctionCallNode):
            return None
        if self.is_user_defined_function(value.name):
            return None
        if value.name not in self.HIP_RUNTIME_ERROR_WRAPPER_NAMES:
            return None
        if len(getattr(value, "args", []) or []) != 1:
            return None
        return self.format_hip_runtime_status_expression(value.args[0])

    def format_hip_runtime_call(self, node):
        args = [self.visit_runtime_argument_expression(arg) for arg in node.args]
        name = node.name

        if name in {"hipMalloc", "hipMallocManaged"}:
            if len(node.args) >= 2:
                target = self.format_runtime_raw_output_target(node.args[0])
                size = self.visit(node.args[1])
                return [f"// HIP memory allocate: {target}, bytes: {size}"]
        elif name == "hipExtMallocWithFlags":
            if len(args) >= 3:
                target = self.format_runtime_raw_output_target(node.args[0])
                return [
                    f"// HIP extended memory allocate: {target}, "
                    f"bytes: {args[1]}, flags: {args[2]}"
                ]
        elif name == "hipMallocAsync":
            if len(args) >= 3:
                target = self.format_runtime_raw_output_target(node.args[0])
                return [
                    f"// HIP async memory allocate: {target}, bytes: {args[1]}, "
                    f"stream: {args[2]}"
                ]
        elif name == "hipMallocFromPoolAsync":
            if len(args) >= 4:
                target = self.format_runtime_raw_output_target(node.args[0])
                return [
                    f"// HIP async memory allocate from pool: {target}, "
                    f"bytes: {args[1]}, pool: {args[2]}, stream: {args[3]}"
                ]
        elif name in {"hipHostMalloc", "hipHostAlloc"}:
            if len(args) >= 2:
                target = self.format_runtime_raw_output_target(node.args[0])
                comment = f"// HIP host memory allocate: {target}, bytes: {args[1]}"
                if len(args) >= 3:
                    comment += f", flags: {args[2]}"
                return [comment]
        elif name == "hipMallocPitch":
            if len(args) >= 4:
                target = self.format_runtime_raw_output_target(node.args[0])
                pitch = self.format_runtime_raw_output_target(node.args[1])
                return [
                    f"// HIP pitched memory allocate: {target}, pitch: {pitch}, "
                    f"width: {args[2]}, height: {args[3]}"
                ]
        elif name == "hipMalloc3D":
            if len(args) >= 2:
                target = self.format_runtime_raw_output_target(node.args[0])
                return [f"// HIP 3D memory allocate: {target}, extent: {args[1]}"]
        elif name in {"hipMallocArray", "hipMalloc3DArray"}:
            if len(args) >= 4:
                target = self.format_runtime_raw_output_target(node.args[0])
                descriptor = self.format_runtime_pointer_target(node.args[1])
                if name == "hipMallocArray":
                    comment = (
                        f"// HIP array allocate: {target}, desc: {descriptor}, "
                        f"width: {args[2]}, height: {args[3]}"
                    )
                    if len(args) >= 5:
                        comment += f", flags: {args[4]}"
                else:
                    comment = (
                        f"// HIP 3D array allocate: {target}, desc: {descriptor}, "
                        f"extent: {args[2]}, flags: {args[3]}"
                    )
                return [comment]
        elif name == "hipMallocMipmappedArray":
            if len(args) >= 5:
                target = self.format_runtime_raw_output_target(node.args[0])
                descriptor = self.format_runtime_pointer_target(node.args[1])
                return [
                    f"// HIP mipmapped array allocate: output: {target}, "
                    f"desc: {descriptor}, extent: {args[2]}, levels: {args[3]}, "
                    f"flags: {args[4]}"
                ]
        elif name == "hipMipmappedArrayCreate":
            if len(args) >= 3:
                target = self.format_runtime_raw_output_target(node.args[0])
                descriptor = self.format_runtime_pointer_target(node.args[1])
                return [
                    f"// HIP mipmapped array create: output: {target}, "
                    f"descriptor: {descriptor}, levels: {args[2]}"
                ]
        elif name in {"hipGetMipmappedArrayLevel", "hipMipmappedArrayGetLevel"}:
            if len(args) >= 3:
                output = self.format_runtime_raw_output_target(node.args[0])
                return [
                    f"// HIP mipmapped array get level: output: {output}, "
                    f"mipmapped array: {args[1]}, level: {args[2]}"
                ]
        elif name == "hipMipmappedArrayDestroy":
            if args:
                return [f"// HIP free mipmapped array: {args[0]}"]
        elif name in {"hipFree", "hipHostFree", "hipFreeHost"}:
            if args:
                return [f"// HIP memory free: {args[0]}"]
        elif name == "hipFreeAsync":
            if len(args) >= 2:
                return [f"// HIP async memory free: {args[0]}, stream: {args[1]}"]
        elif name in {"hipFreeArray", "hipArrayDestroy"}:
            if args:
                return [f"// HIP array free: {args[0]}"]
        elif name in {"hipArrayCreate", "hipArray3DCreate"}:
            if len(args) >= 2:
                output = self.format_runtime_raw_output_target(node.args[0])
                descriptor = self.format_runtime_pointer_target(node.args[1])
                dimension = "3D " if name == "hipArray3DCreate" else ""
                return [
                    f"// HIP {dimension}array create: output: {output}, "
                    f"descriptor: {descriptor}"
                ]
        elif name in {"hipArrayGetDescriptor", "hipArray3DGetDescriptor"}:
            if len(args) >= 2:
                output = self.format_runtime_raw_output_target(node.args[0])
                output_name = self.get_runtime_pointer_target_name(node.args[0])
                dimension = "3D " if name == "hipArray3DGetDescriptor" else ""
                if output_name is not None:
                    if name == "hipArray3DGetDescriptor":
                        members = self.HIP_ARRAY3D_DESCRIPTOR_MEMBERS
                        query_prefix = "array.descriptor3D"
                    else:
                        members = self.HIP_ARRAY_DESCRIPTOR_MEMBERS
                        query_prefix = "array.descriptor"
                    self.register_member_query_sources(
                        output_name,
                        {
                            member: f"{query_prefix}.{member}({args[1]})"
                            for member in members
                        },
                    )
                return [
                    f"// HIP array get {dimension}descriptor: output: {output}, "
                    f"array: {args[1]}"
                ]
        elif name == "hipArrayGetInfo":
            if len(args) >= 4:
                descriptor_output = self.format_runtime_raw_output_target(node.args[0])
                extent_output = self.format_runtime_raw_output_target(node.args[1])
                flags_output = self.format_runtime_raw_output_target(node.args[2])
                descriptor_output_name = self.get_runtime_pointer_target_name(
                    node.args[0]
                )
                extent_output_name = self.get_runtime_pointer_target_name(node.args[1])
                flags_output_name = self.get_runtime_pointer_target_name(node.args[2])
                if descriptor_output_name is not None:
                    self.register_member_query_sources(
                        descriptor_output_name,
                        {
                            member: f"array.info.channelDesc.{member}({args[3]})"
                            for member in self.HIP_CHANNEL_DESCRIPTOR_MEMBERS
                        },
                    )
                if extent_output_name is not None:
                    self.register_member_query_sources(
                        extent_output_name,
                        {
                            member: f"array.info.extent.{member}({args[3]})"
                            for member in self.HIP_EXTENT_MEMBERS
                        },
                    )
                if flags_output_name is not None:
                    self.register_device_query_source(
                        flags_output_name,
                        f"array.info.flags({args[3]})",
                    )
                return [
                    f"// HIP array get info: desc output: {descriptor_output}, "
                    f"extent output: {extent_output}, flags output: {flags_output}, "
                    f"array: {args[3]}"
                ]
        elif name == "hipMemPoolCreate":
            if len(args) >= 2:
                output = self.format_runtime_raw_output_target(node.args[0])
                return [
                    f"// HIP memory pool create: output: {output}, properties: {args[1]}"
                ]
        elif name == "hipMemPoolDestroy":
            if args:
                return [f"// HIP memory pool destroy: {args[0]}"]
        elif name == "hipMemPoolTrimTo":
            if len(args) >= 2:
                return [
                    f"// HIP memory pool trim: pool: {args[0]}, "
                    f"minimum bytes: {args[1]}"
                ]
        elif name == "hipMemPoolSetAttribute":
            if len(args) >= 3:
                value = self.format_runtime_pointer_target(node.args[2])
                return [
                    f"// HIP memory pool set attribute: pool: {args[0]}, "
                    f"attribute: {args[1]}, value: {value}"
                ]
        elif name == "hipMemPoolGetAttribute":
            if len(args) >= 3:
                output = self.format_runtime_raw_output_target(node.args[2])
                output_name = self.get_runtime_pointer_target_name(node.args[2])
                if output_name is not None:
                    self.register_device_query_source(
                        output_name,
                        f"memoryPool.attribute({args[0]}, {args[1]})",
                    )
                return [
                    f"// HIP memory pool get attribute: pool: {args[0]}, "
                    f"attribute: {args[1]}, output: {output}"
                ]
        elif name == "hipMemPoolSetAccess":
            if len(args) >= 3:
                return [
                    f"// HIP memory pool set access: pool: {args[0]}, "
                    f"descriptors: {args[1]}, count: {args[2]}"
                ]
        elif name == "hipMemPoolGetAccess":
            if len(args) >= 3:
                output = self.format_runtime_raw_output_target(node.args[0])
                output_name = self.get_runtime_pointer_target_name(node.args[0])
                location = self.format_runtime_pointer_target(node.args[2])
                if output_name is not None:
                    self.register_device_query_source(
                        output_name,
                        f"memoryPool.accessFlags({args[1]}, {location})",
                    )
                return [
                    f"// HIP memory pool get access: output: {output}, "
                    f"pool: {args[1]}, location: {args[2]}"
                ]
        elif name == "hipMemPoolExportToShareableHandle":
            if len(args) >= 4:
                output = self.format_runtime_raw_output_target(node.args[0])
                return [
                    f"// HIP memory pool export to shareable handle: output: {output}, "
                    f"pool: {args[1]}, handle type: {args[2]}, flags: {args[3]}"
                ]
        elif name == "hipMemPoolImportFromShareableHandle":
            if len(args) >= 4:
                output = self.format_runtime_raw_output_target(node.args[0])
                return [
                    f"// HIP memory pool import from shareable handle: output: {output}, "
                    f"handle: {args[1]}, handle type: {args[2]}, flags: {args[3]}"
                ]
        elif name == "hipMemPoolExportPointer":
            if len(args) >= 2:
                output = self.format_runtime_raw_output_target(node.args[0])
                return [
                    f"// HIP memory pool export pointer: output: {output}, "
                    f"pointer: {args[1]}"
                ]
        elif name == "hipMemPoolImportPointer":
            if len(args) >= 3:
                output = self.format_runtime_raw_output_target(node.args[0])
                return [
                    f"// HIP memory pool import pointer: output: {output}, "
                    f"pool: {args[1]}, export data: {args[2]}"
                ]
        elif name == "hipDeviceGetDefaultMemPool":
            if len(args) >= 2:
                output = self.format_runtime_raw_output_target(node.args[0])
                return [
                    f"// HIP get default memory pool: output: {output}, "
                    f"device: {args[1]}"
                ]
        elif name == "hipDeviceSetMemPool":
            if len(args) >= 2:
                return [
                    f"// HIP set device memory pool: device: {args[0]}, pool: {args[1]}"
                ]
        elif name == "hipDeviceGetMemPool":
            if len(args) >= 2:
                output = self.format_runtime_raw_output_target(node.args[0])
                return [
                    f"// HIP get device memory pool: output: {output}, "
                    f"device: {args[1]}"
                ]
        elif name == "hipMemPrefetchAsync":
            if len(args) >= 4:
                return [
                    f"// HIP memory prefetch: pointer: {args[0]}, bytes: {args[1]}, "
                    f"device: {args[2]}, stream: {args[3]}"
                ]
        elif name == "hipMemPrefetchAsync_v2":
            if len(args) >= 5:
                return [
                    f"// HIP memory prefetch v2: pointer: {args[0]}, "
                    f"bytes: {args[1]}, location: {args[2]}, flags: {args[3]}, "
                    f"stream: {args[4]}"
                ]
        elif name == "hipMemAdvise":
            if len(args) >= 4:
                return [
                    f"// HIP memory advise: pointer: {args[0]}, bytes: {args[1]}, "
                    f"advice: {args[2]}, device: {args[3]}"
                ]
        elif name == "hipMemAdvise_v2":
            if len(args) >= 4:
                return [
                    f"// HIP memory advise v2: pointer: {args[0]}, "
                    f"bytes: {args[1]}, advice: {args[2]}, location: {args[3]}"
                ]
        elif name == "hipMemRangeGetAttribute":
            if len(args) >= 5:
                output = self.format_runtime_raw_output_target(node.args[0])
                output_name = self.get_runtime_pointer_target_name(node.args[0])
                if output_name is not None:
                    self.register_device_query_source(
                        output_name,
                        f"memory.rangeAttribute({args[2]}, {args[3]}, {args[4]})",
                    )
                return [
                    f"// HIP memory range get attribute: output: {output}, "
                    f"output bytes: {args[1]}, attribute: {args[2]}, "
                    f"pointer: {args[3]}, range bytes: {args[4]}"
                ]
        elif name == "hipMemRangeGetAttributes":
            if len(args) >= 6:
                return [
                    f"// HIP memory range get attributes: outputs: {args[0]}, "
                    f"output sizes: {args[1]}, attributes: {args[2]}, "
                    f"attribute count: {args[3]}, pointer: {args[4]}, "
                    f"range bytes: {args[5]}"
                ]
        elif name == "hipStreamAttachMemAsync":
            if len(args) >= 4:
                return [
                    f"// HIP stream attach memory: stream: {args[0]}, "
                    f"pointer: {args[1]}, bytes: {args[2]}, flags: {args[3]}"
                ]
        elif name == "hipMemGetAllocationGranularity":
            if len(args) >= 3:
                output = self.format_runtime_raw_output_target(node.args[0])
                output_name = self.get_runtime_pointer_target_name(node.args[0])
                properties = self.format_runtime_pointer_target(node.args[1])
                if output_name is not None:
                    self.register_device_query_source(
                        output_name,
                        "virtualMemory.allocationGranularity("
                        f"{properties}, {args[2]})",
                    )
                return [
                    f"// HIP virtual memory allocation granularity: output: {output}, "
                    f"properties: {args[1]}, option: {args[2]}"
                ]
        elif name == "hipMemCreate":
            if len(args) >= 4:
                output = self.format_runtime_raw_output_target(node.args[0])
                return [
                    f"// HIP virtual memory create allocation: output: {output}, "
                    f"bytes: {args[1]}, properties: {args[2]}, flags: {args[3]}"
                ]
        elif name == "hipMemRelease":
            if args:
                return [f"// HIP virtual memory release allocation: {args[0]}"]
        elif name == "hipMemAddressReserve":
            if len(args) >= 5:
                output = self.format_runtime_raw_output_target(node.args[0])
                return [
                    f"// HIP virtual memory reserve address: output: {output}, "
                    f"bytes: {args[1]}, alignment: {args[2]}, address: {args[3]}, "
                    f"flags: {args[4]}"
                ]
        elif name == "hipMemAddressFree":
            if len(args) >= 2:
                return [
                    f"// HIP virtual memory free address: pointer: {args[0]}, "
                    f"bytes: {args[1]}"
                ]
        elif name == "hipMemMap":
            if len(args) >= 5:
                return [
                    f"// HIP virtual memory map: pointer: {args[0]}, "
                    f"bytes: {args[1]}, offset: {args[2]}, handle: {args[3]}, "
                    f"flags: {args[4]}"
                ]
        elif name == "hipMemUnmap":
            if len(args) >= 2:
                return [
                    f"// HIP virtual memory unmap: pointer: {args[0]}, "
                    f"bytes: {args[1]}"
                ]
        elif name == "hipMemSetAccess":
            if len(args) >= 4:
                return [
                    f"// HIP virtual memory set access: pointer: {args[0]}, "
                    f"bytes: {args[1]}, descriptors: {args[2]}, count: {args[3]}"
                ]
        elif name == "hipMemGetAccess":
            if len(args) >= 3:
                output = self.format_runtime_raw_output_target(node.args[0])
                output_name = self.get_runtime_pointer_target_name(node.args[0])
                location = self.format_runtime_pointer_target(node.args[1])
                if output_name is not None:
                    self.register_device_query_source(
                        output_name,
                        f"virtualMemory.accessFlags({location}, {args[2]})",
                    )
                return [
                    f"// HIP virtual memory get access: output: {output}, "
                    f"location: {args[1]}, pointer: {args[2]}"
                ]
        elif name == "hipMemGetAllocationPropertiesFromHandle":
            if len(args) >= 2:
                output = self.format_runtime_raw_output_target(node.args[0])
                return [
                    f"// HIP virtual memory allocation properties: output: {output}, "
                    f"handle: {args[1]}"
                ]
        elif name == "hipMemRetainAllocationHandle":
            if len(args) >= 2:
                output = self.format_runtime_raw_output_target(node.args[0])
                return [
                    f"// HIP virtual memory retain allocation handle: output: {output}, "
                    f"address: {args[1]}"
                ]
        elif name == "hipMemExportToShareableHandle":
            if len(args) >= 4:
                output = self.format_runtime_raw_output_target(node.args[0])
                return [
                    f"// HIP virtual memory export shareable handle: output: {output}, "
                    f"handle: {args[1]}, handle type: {args[2]}, flags: {args[3]}"
                ]
        elif name == "hipMemImportFromShareableHandle":
            if len(args) >= 3:
                output = self.format_runtime_raw_output_target(node.args[0])
                return [
                    f"// HIP virtual memory import shareable handle: output: {output}, "
                    f"shareable handle: {args[1]}, handle type: {args[2]}"
                ]
        elif name == "hipHostRegister":
            if len(args) >= 3:
                return [
                    f"// HIP host memory register: {args[0]}, bytes: {args[1]}, "
                    f"flags: {args[2]}"
                ]
        elif name == "hipHostUnregister":
            if args:
                return [f"// HIP host memory unregister: {args[0]}"]
        elif name == "hipHostGetDevicePointer":
            if len(args) >= 3:
                output = self.format_runtime_raw_output_target(node.args[0])
                return [
                    f"// HIP host device pointer: output: {output}, host: {args[1]}, "
                    f"flags: {args[2]}"
                ]
        elif name == "hipHostGetFlags":
            if len(args) >= 2:
                output = self.format_runtime_raw_output_target(node.args[0])
                output_name = self.get_runtime_pointer_target_name(node.args[0])
                if output_name is not None:
                    self.register_device_query_source(
                        output_name,
                        f"hostMemory.flags({args[1]})",
                    )
                return [f"// HIP host memory flags: output: {output}, host: {args[1]}"]
        elif name in {"hipMemcpy", "hipMemcpyAsync", "hipMemcpyWithStream"}:
            if len(args) >= 4:
                comment = (
                    f"// HIP memory copy: {args[1]} -> {args[0]}, "
                    f"bytes: {args[2]}, kind: {args[3]}"
                )
                if len(args) >= 5:
                    comment += f", stream: {args[4]}"
                return [comment]
        elif name in {"hipMemcpyPeer", "hipMemcpyPeerAsync"}:
            if len(args) >= 5:
                comment = (
                    f"// HIP peer memory copy: source: {args[2]}, "
                    f"source device: {args[3]}, destination: {args[0]}, "
                    f"destination device: {args[1]}, bytes: {args[4]}"
                )
                if len(args) >= 6:
                    comment += f", stream: {args[5]}"
                return [comment]
        elif name in {"hipMemcpy2D", "hipMemcpy2DAsync"}:
            if len(args) >= 7:
                comment = (
                    f"// HIP 2D memory copy: {args[2]} -> {args[0]}, "
                    f"dst pitch: {args[1]}, src pitch: {args[3]}, "
                    f"width: {args[4]}, height: {args[5]}, kind: {args[6]}"
                )
                if len(args) >= 8:
                    comment += f", stream: {args[7]}"
                return [comment]
        elif name in {"hipMemcpyToArray", "hipMemcpyToArrayAsync"}:
            if len(args) >= 6:
                comment = (
                    f"// HIP memory copy to array: source: {args[3]}, "
                    f"destination array: {args[0]}, w offset: {args[1]}, "
                    f"h offset: {args[2]}, bytes: {args[4]}, kind: {args[5]}"
                )
                if len(args) >= 7:
                    comment += f", stream: {args[6]}"
                return [comment]
        elif name in {"hipMemcpyFromArray", "hipMemcpyFromArrayAsync"}:
            if len(args) >= 6:
                comment = (
                    f"// HIP memory copy from array: source array: {args[1]}, "
                    f"w offset: {args[2]}, h offset: {args[3]}, "
                    f"destination: {args[0]}, bytes: {args[4]}, kind: {args[5]}"
                )
                if len(args) >= 7:
                    comment += f", stream: {args[6]}"
                return [comment]
        elif name in {"hipMemcpy2DToArray", "hipMemcpy2DToArrayAsync"}:
            if len(args) >= 8:
                comment = (
                    f"// HIP 2D memory copy to array: source: {args[3]}, "
                    f"source pitch: {args[4]}, destination array: {args[0]}, "
                    f"w offset: {args[1]}, h offset: {args[2]}, width: {args[5]}, "
                    f"height: {args[6]}, kind: {args[7]}"
                )
                if len(args) >= 9:
                    comment += f", stream: {args[8]}"
                return [comment]
        elif name in {"hipMemcpy2DFromArray", "hipMemcpy2DFromArrayAsync"}:
            if len(args) >= 8:
                comment = (
                    f"// HIP 2D memory copy from array: source array: {args[2]}, "
                    f"w offset: {args[3]}, h offset: {args[4]}, "
                    f"destination: {args[0]}, destination pitch: {args[1]}, "
                    f"width: {args[5]}, height: {args[6]}, kind: {args[7]}"
                )
                if len(args) >= 9:
                    comment += f", stream: {args[8]}"
                return [comment]
        elif name == "hipMemcpyArrayToArray":
            if len(args) >= 8:
                return [
                    f"// HIP memory copy array to array: source array: {args[3]}, "
                    f"source w offset: {args[4]}, source h offset: {args[5]}, "
                    f"destination array: {args[0]}, destination w offset: {args[1]}, "
                    f"destination h offset: {args[2]}, bytes: {args[6]}, "
                    f"kind: {args[7]}"
                ]
        elif name == "hipMemcpy2DArrayToArray":
            if len(args) >= 9:
                return [
                    f"// HIP 2D memory copy array to array: source array: {args[3]}, "
                    f"source w offset: {args[4]}, source h offset: {args[5]}, "
                    f"destination array: {args[0]}, destination w offset: {args[1]}, "
                    f"destination h offset: {args[2]}, width: {args[6]}, "
                    f"height: {args[7]}, kind: {args[8]}"
                ]
        elif name in {"hipMemcpy3D", "hipMemcpy3DAsync"}:
            if args:
                params = self.format_runtime_pointer_target(node.args[0])
                comment = f"// HIP 3D memory copy: params: {params}"
                if len(args) >= 2:
                    comment += f", stream: {args[1]}"
                return [comment]
        elif name in {"hipMemcpy3DPeer", "hipMemcpy3DPeerAsync"}:
            if args:
                params = self.format_runtime_pointer_target(node.args[0])
                comment = f"// HIP 3D peer memory copy: params: {params}"
                if len(args) >= 2:
                    comment += f", stream: {args[1]}"
                return [comment]
        elif name in {"hipMemcpyParam2D", "hipMemcpyParam2DAsync"}:
            if args:
                params = self.format_runtime_pointer_target(node.args[0])
                comment = f"// HIP 2D parameterized memory copy: params: {params}"
                if len(args) >= 2:
                    comment += f", stream: {args[1]}"
                return [comment]
        elif name in {"hipDrvMemcpy3D", "hipDrvMemcpy3DAsync"}:
            if args:
                params = self.format_runtime_pointer_target(node.args[0])
                comment = f"// HIP driver 3D memory copy: params: {params}"
                if len(args) >= 2:
                    comment += f", stream: {args[1]}"
                return [comment]
        elif name == "hipMemcpyBatchAsync":
            if len(args) >= 9:
                fail_output = self.format_runtime_raw_output_target(node.args[7])
                return [
                    f"// HIP batched memory copy: destinations: {args[0]}, "
                    f"sources: {args[1]}, sizes: {args[2]}, count: {args[3]}, "
                    f"attributes: {args[4]}, attribute indices: {args[5]}, "
                    f"attribute count: {args[6]}, fail index output: {fail_output}, "
                    f"stream: {args[8]}"
                ]
        elif name == "hipMemcpy3DBatchAsync":
            if len(args) >= 5:
                fail_output = self.format_runtime_raw_output_target(node.args[2])
                return [
                    f"// HIP batched 3D memory copy: count: {args[0]}, "
                    f"operations: {args[1]}, fail index output: {fail_output}, "
                    f"flags: {args[3]}, stream: {args[4]}"
                ]
        elif name == "hipGetSymbolAddress":
            if len(args) >= 2:
                output = self.format_runtime_raw_output_target(node.args[0])
                return [
                    f"// HIP get symbol address: output: {output}, "
                    f"symbol: {args[1]}"
                ]
        elif name == "hipGetSymbolSize":
            if len(args) >= 2:
                output = self.format_runtime_raw_output_target(node.args[0])
                output_name = self.get_runtime_pointer_target_name(node.args[0])
                if output_name is not None:
                    self.register_device_query_source(
                        output_name,
                        f"symbol.size({args[1]})",
                    )
                return [f"// HIP get symbol size: output: {output}, symbol: {args[1]}"]
        elif name in {"hipMemcpyToSymbol", "hipMemcpyToSymbolAsync"}:
            if len(args) >= 3:
                comment = (
                    f"// HIP symbol copy to: {args[0]}, source: {args[1]}, "
                    f"bytes: {args[2]}"
                )
                if len(args) >= 4:
                    comment += f", offset: {args[3]}"
                if len(args) >= 5:
                    comment += f", kind: {args[4]}"
                if len(args) >= 6:
                    comment += f", stream: {args[5]}"
                return [comment]
        elif name in {"hipMemcpyFromSymbol", "hipMemcpyFromSymbolAsync"}:
            if len(args) >= 3:
                comment = (
                    f"// HIP symbol copy from: {args[1]}, destination: {args[0]}, "
                    f"bytes: {args[2]}"
                )
                if len(args) >= 4:
                    comment += f", offset: {args[3]}"
                if len(args) >= 5:
                    comment += f", kind: {args[4]}"
                if len(args) >= 6:
                    comment += f", stream: {args[5]}"
                return [comment]
        elif name in {"hipMemset", "hipMemsetAsync"}:
            if len(args) >= 3:
                comment = (
                    f"// HIP memory set: {args[0]}, value: {args[1]}, "
                    f"bytes: {args[2]}"
                )
                if len(args) >= 4:
                    comment += f", stream: {args[3]}"
                return [comment]
        elif name in {"hipMemset2D", "hipMemset2DAsync"}:
            if len(args) >= 5:
                comment = (
                    f"// HIP 2D memory set: {args[0]}, pitch: {args[1]}, "
                    f"value: {args[2]}, width: {args[3]}, height: {args[4]}"
                )
                if len(args) >= 6:
                    comment += f", stream: {args[5]}"
                return [comment]
        elif name in {"hipMemset3D", "hipMemset3DAsync"}:
            if len(args) >= 3:
                comment = (
                    f"// HIP 3D memory set: {args[0]}, value: {args[1]}, "
                    f"extent: {args[2]}"
                )
                if len(args) >= 4:
                    comment += f", stream: {args[3]}"
                return [comment]
        elif name == "hipMemAlloc":
            if len(args) >= 2:
                output = self.format_runtime_raw_output_target(node.args[0])
                return [
                    f"// HIP driver memory allocate: output: {output}, "
                    f"bytes: {args[1]}"
                ]
        elif name == "hipMemAllocPitch":
            if len(args) >= 5:
                output = self.format_runtime_raw_output_target(node.args[0])
                pitch = self.format_runtime_raw_output_target(node.args[1])
                return [
                    f"// HIP driver pitched memory allocate: output: {output}, "
                    f"pitch output: {pitch}, width: {args[2]}, height: {args[3]}, "
                    f"element bytes: {args[4]}"
                ]
        elif name == "hipMemFree":
            if args:
                return [f"// HIP driver memory free: {args[0]}"]
        elif name in {"hipMemAllocHost", "hipMemHostAlloc"}:
            if len(args) >= 2:
                output = self.format_runtime_raw_output_target(node.args[0])
                comment = (
                    f"// HIP driver host memory allocate: output: {output}, "
                    f"bytes: {args[1]}"
                )
                if len(args) >= 3:
                    comment += f", flags: {args[2]}"
                return [comment]
        elif name == "hipMemFreeHost":
            if args:
                return [f"// HIP driver host memory free: {args[0]}"]
        elif name == "hipMemHostGetDevicePointer":
            if len(args) >= 3:
                output = self.format_runtime_raw_output_target(node.args[0])
                return [
                    f"// HIP driver host device pointer: output: {output}, "
                    f"host: {args[1]}, flags: {args[2]}"
                ]
        elif name == "hipMemGetAddressRange":
            if len(args) >= 3:
                base_output = self.format_runtime_pointer_target(node.args[0])
                size_output = self.format_runtime_pointer_target(node.args[1])
                base_output_name = self.get_runtime_pointer_target_name(node.args[0])
                size_output_name = self.get_runtime_pointer_target_name(node.args[1])
                if base_output_name is not None:
                    self.register_device_query_source(
                        base_output_name,
                        f"memory.addressRange.base({args[2]})",
                    )
                if size_output_name is not None:
                    self.register_device_query_source(
                        size_output_name,
                        f"memory.addressRange.size({args[2]})",
                    )
                return [
                    f"// HIP driver memory address range: base output: {base_output}, "
                    f"size output: {size_output}, pointer: {args[2]}"
                ]
        elif name in {
            "hipMemcpyHtoD",
            "hipMemcpyHtoDAsync",
            "hipMemcpyDtoH",
            "hipMemcpyDtoHAsync",
            "hipMemcpyDtoD",
            "hipMemcpyDtoDAsync",
        }:
            if len(args) >= 3:
                copy_kind = {
                    "hipMemcpyHtoD": "host to device",
                    "hipMemcpyHtoDAsync": "host to device",
                    "hipMemcpyDtoH": "device to host",
                    "hipMemcpyDtoHAsync": "device to host",
                    "hipMemcpyDtoD": "device to device",
                    "hipMemcpyDtoDAsync": "device to device",
                }[name]
                comment = (
                    f"// HIP driver memory copy {copy_kind}: "
                    f"source: {args[1]}, destination: {args[0]}, bytes: {args[2]}"
                )
                if len(args) >= 4:
                    comment += f", stream: {args[3]}"
                return [comment]
        elif name in {"hipMemcpyAtoH", "hipMemcpyAtoHAsync"}:
            if len(args) >= 4:
                comment = (
                    "// HIP driver memory copy array to host: "
                    f"source array: {args[1]}, source offset: {args[2]}, "
                    f"destination host: {args[0]}, bytes: {args[3]}"
                )
                if len(args) >= 5:
                    comment += f", stream: {args[4]}"
                return [comment]
        elif name in {"hipMemcpyHtoA", "hipMemcpyHtoAAsync"}:
            if len(args) >= 4:
                comment = (
                    "// HIP driver memory copy host to array: "
                    f"source host: {args[2]}, destination array: {args[0]}, "
                    f"destination offset: {args[1]}, bytes: {args[3]}"
                )
                if len(args) >= 5:
                    comment += f", stream: {args[4]}"
                return [comment]
        elif name == "hipMemcpyAtoD":
            if len(args) >= 4:
                return [
                    "// HIP driver memory copy array to device: "
                    f"source array: {args[1]}, source offset: {args[2]}, "
                    f"destination device: {args[0]}, bytes: {args[3]}"
                ]
        elif name == "hipMemcpyDtoA":
            if len(args) >= 4:
                return [
                    "// HIP driver memory copy device to array: "
                    f"source device: {args[2]}, destination array: {args[0]}, "
                    f"destination offset: {args[1]}, bytes: {args[3]}"
                ]
        elif name == "hipMemcpyAtoA":
            if len(args) >= 5:
                return [
                    "// HIP driver memory copy array to array: "
                    f"source array: {args[2]}, source offset: {args[3]}, "
                    f"destination array: {args[0]}, destination offset: {args[1]}, "
                    f"bytes: {args[4]}"
                ]
        elif name in {
            "hipMemsetD2D8",
            "hipMemsetD2D8Async",
            "hipMemsetD2D16",
            "hipMemsetD2D16Async",
            "hipMemsetD2D32",
            "hipMemsetD2D32Async",
        }:
            if len(args) >= 5:
                width = {
                    "hipMemsetD2D8": "8-bit",
                    "hipMemsetD2D8Async": "8-bit",
                    "hipMemsetD2D16": "16-bit",
                    "hipMemsetD2D16Async": "16-bit",
                    "hipMemsetD2D32": "32-bit",
                    "hipMemsetD2D32Async": "32-bit",
                }[name]
                comment = (
                    f"// HIP driver 2D memory set {width}: pointer: {args[0]}, "
                    f"pitch: {args[1]}, value: {args[2]}, width: {args[3]}, "
                    f"height: {args[4]}"
                )
                if len(args) >= 6:
                    comment += f", stream: {args[5]}"
                return [comment]
        elif name in {
            "hipMemsetD8",
            "hipMemsetD8Async",
            "hipMemsetD16",
            "hipMemsetD16Async",
            "hipMemsetD32",
            "hipMemsetD32Async",
        }:
            if len(args) >= 3:
                width = {
                    "hipMemsetD8": "8-bit",
                    "hipMemsetD8Async": "8-bit",
                    "hipMemsetD16": "16-bit",
                    "hipMemsetD16Async": "16-bit",
                    "hipMemsetD32": "32-bit",
                    "hipMemsetD32Async": "32-bit",
                }[name]
                comment = (
                    f"// HIP driver memory set {width}: pointer: {args[0]}, "
                    f"value: {args[1]}, count: {args[2]}"
                )
                if len(args) >= 4:
                    comment += f", stream: {args[3]}"
                return [comment]
        elif name == "hipIpcGetMemHandle":
            if len(args) >= 2:
                output = self.format_runtime_raw_output_target(node.args[0])
                return [
                    f"// HIP IPC get memory handle: output: {output}, "
                    f"pointer: {args[1]}"
                ]
        elif name == "hipIpcOpenMemHandle":
            if len(args) >= 3:
                output = self.format_runtime_raw_output_target(node.args[0])
                return [
                    f"// HIP IPC open memory handle: output: {output}, "
                    f"handle: {args[1]}, flags: {args[2]}"
                ]
        elif name == "hipIpcCloseMemHandle":
            if args:
                return [f"// HIP IPC close memory handle: pointer: {args[0]}"]
        elif name == "hipIpcGetEventHandle":
            if len(args) >= 2:
                output = self.format_runtime_raw_output_target(node.args[0])
                return [
                    f"// HIP IPC get event handle: output: {output}, event: {args[1]}"
                ]
        elif name == "hipIpcOpenEventHandle":
            if len(args) >= 2:
                output = self.format_runtime_raw_output_target(node.args[0])
                return [
                    f"// HIP IPC open event handle: output: {output}, "
                    f"handle: {args[1]}"
                ]
        elif name in {
            "hipStreamWaitValue32",
            "hipStreamWaitValue64",
            "hipStreamWriteValue32",
            "hipStreamWriteValue64",
        }:
            if len(args) >= 4:
                operation = "wait" if "Wait" in name else "write"
                width = "32" if name.endswith("32") else "64"
                return [
                    f"// HIP stream {operation} value{width}: stream: {args[0]}, "
                    f"address: {args[1]}, value: {args[2]}, flags: {args[3]}"
                ]
        elif name == "hipStreamBatchMemOp":
            if len(args) >= 4:
                return [
                    f"// HIP stream batch memory op: stream: {args[0]}, "
                    f"count: {args[1]}, params: {args[2]}, flags: {args[3]}"
                ]
        elif name in {"hipStreamSynchronize"}:
            if args:
                return [f"// HIP synchronize: {args[0]}"]
        elif name == "hipStreamAddCallback":
            if len(args) >= 4:
                return [
                    f"// HIP stream add callback: stream: {args[0]}, "
                    f"callback: {args[1]}, user data: {args[2]}, flags: {args[3]}"
                ]
        elif name == "hipLaunchHostFunc":
            if len(args) >= 3:
                return [
                    f"// HIP launch host function: stream: {args[0]}, "
                    f"function: {args[1]}, user data: {args[2]}"
                ]
        elif name in {"hipConfigureCall", "__hipPushCallConfiguration"}:
            if len(args) >= 4:
                action = (
                    "push call configuration"
                    if name.startswith("__")
                    else "configure call"
                )
                return [
                    f"// HIP {action}: grid: {args[0]}, block: {args[1]}, "
                    f"shared memory: {args[2]}, stream: {args[3]}"
                ]
        elif name == "__hipPopCallConfiguration":
            if len(args) >= 4:
                grid_output = self.format_runtime_raw_output_target(node.args[0])
                block_output = self.format_runtime_raw_output_target(node.args[1])
                shared_output = self.format_runtime_raw_output_target(node.args[2])
                stream_output = self.format_runtime_raw_output_target(node.args[3])
                return [
                    f"// HIP pop call configuration: grid output: {grid_output}, "
                    f"block output: {block_output}, shared memory output: "
                    f"{shared_output}, stream output: {stream_output}"
                ]
        elif name == "hipSetupArgument":
            if len(args) >= 3:
                return [
                    f"// HIP setup kernel argument: value: {args[0]}, "
                    f"bytes: {args[1]}, offset: {args[2]}"
                ]
        elif name == "hipLaunchByPtr":
            if args:
                return [f"// HIP launch by pointer: function: {args[0]}"]
        elif name == "hipLaunchKernelExC":
            if len(args) >= 3:
                return [
                    f"// HIP launch kernel ex: config: {args[0]}, "
                    f"function: {args[1]}, args: {args[2]}"
                ]
        elif name == "hipDrvLaunchKernelEx":
            if len(args) >= 4:
                return [
                    f"// HIP driver launch kernel ex: config: {args[0]}, "
                    f"function: {args[1]}, params: {args[2]}, extra: {args[3]}"
                ]
        elif name == "hipExtLaunchKernel":
            if len(args) >= 9:
                function = self.format_runtime_pointer_target(node.args[0])
                return [
                    f"// HIP extended kernel launch: function: {function}, "
                    f"grid: {args[1]}, block: {args[2]}, args: {args[3]}, "
                    f"shared memory: {args[4]}, stream: {args[5]}, "
                    f"start event: {args[6]}, stop event: {args[7]}, "
                    f"flags: {args[8]}"
                ]
        elif name == "hipExtLaunchKernelGGL":
            if len(args) >= 8:
                call_args = ", ".join(args[8:]) if len(args) > 8 else "<none>"
                return [
                    f"// HIP extended kernel launch GGL: function: {args[0]}, "
                    f"grid: {args[1]}, block: {args[2]}, "
                    f"shared memory: {args[3]}, stream: {args[4]}, "
                    f"start event: {args[5]}, stop event: {args[6]}, "
                    f"flags: {args[7]}, args: {call_args}"
                ]
        elif name == "hipExtLaunchMultiKernelMultiDevice":
            if len(args) >= 3:
                return [
                    f"// HIP extended multi-kernel multi-device launch: "
                    f"params: {args[0]}, devices: {args[1]}, flags: {args[2]}"
                ]
        elif name == "hipDeviceGetStreamPriorityRange":
            if len(args) >= 2:
                least_output = self.format_runtime_raw_output_target(node.args[0])
                greatest_output = self.format_runtime_raw_output_target(node.args[1])
                least_output_name = self.get_runtime_pointer_target_name(node.args[0])
                greatest_output_name = self.get_runtime_pointer_target_name(
                    node.args[1]
                )
                if least_output_name is not None:
                    self.register_device_query_source(
                        least_output_name, "streamPriorityRange.least"
                    )
                if greatest_output_name is not None:
                    self.register_device_query_source(
                        greatest_output_name, "streamPriorityRange.greatest"
                    )
                return [
                    f"// HIP get stream priority range: "
                    f"least output: {least_output}, greatest output: {greatest_output}"
                ]
        elif name == "hipStreamGetFlags":
            if len(args) >= 2:
                output = self.format_runtime_raw_output_target(node.args[1])
                output_name = self.get_runtime_pointer_target_name(node.args[1])
                if output_name is not None:
                    self.register_device_query_source(
                        output_name, f"stream.flags({args[0]})"
                    )
                return [f"// HIP get stream flags: stream: {args[0]}, output: {output}"]
        elif name == "hipStreamGetPriority":
            if len(args) >= 2:
                output = self.format_runtime_raw_output_target(node.args[1])
                output_name = self.get_runtime_pointer_target_name(node.args[1])
                if output_name is not None:
                    self.register_device_query_source(
                        output_name, f"stream.priority({args[0]})"
                    )
                return [
                    f"// HIP get stream priority: stream: {args[0]}, output: {output}"
                ]
        elif name == "hipStreamBeginCapture":
            if len(args) >= 2:
                return [
                    f"// HIP stream begin capture: stream: {args[0]}, mode: {args[1]}"
                ]
        elif name == "hipStreamEndCapture":
            if len(args) >= 2:
                output = self.format_runtime_raw_output_target(node.args[1])
                return [
                    f"// HIP stream end capture: stream: {args[0]}, graph output: {output}"
                ]
        elif name == "hipStreamIsCapturing":
            if len(args) >= 2:
                output = self.format_runtime_raw_output_target(node.args[1])
                output_name = self.get_runtime_pointer_target_name(node.args[1])
                if output_name is not None:
                    self.register_device_query_source(
                        output_name, f"stream.captureStatus({args[0]})"
                    )
                return [
                    f"// HIP stream is capturing: stream: {args[0]}, output: {output}"
                ]
        elif name in {"hipStreamGetCaptureInfo", "hipStreamGetCaptureInfo_v2"}:
            if len(args) >= 3:
                status_output = self.format_runtime_raw_output_target(node.args[1])
                id_output = self.format_runtime_raw_output_target(node.args[2])
                status_output_name = self.get_runtime_pointer_target_name(node.args[1])
                id_output_name = self.get_runtime_pointer_target_name(node.args[2])
                if status_output_name is not None:
                    self.register_device_query_source(
                        status_output_name, f"stream.captureStatus({args[0]})"
                    )
                if id_output_name is not None:
                    self.register_device_query_source(
                        id_output_name, f"stream.captureId({args[0]})"
                    )
                comment = (
                    f"// HIP stream capture info: stream: {args[0]}, "
                    f"status output: {status_output}, id output: {id_output}"
                )
                if len(args) >= 6:
                    graph_output = self.format_runtime_raw_output_target(node.args[3])
                    dependencies_output = self.format_runtime_raw_output_target(
                        node.args[4]
                    )
                    count_output = self.format_runtime_raw_output_target(node.args[5])
                    count_output_name = self.get_runtime_pointer_target_name(
                        node.args[5]
                    )
                    if count_output_name is not None:
                        self.register_device_query_source(
                            count_output_name,
                            f"stream.captureDependencyCount({args[0]})",
                        )
                    comment += (
                        f", graph output: {graph_output}, "
                        f"dependencies output: {dependencies_output}, "
                        f"dependency count output: {count_output}"
                    )
                return [comment]
        elif name == "hipStreamUpdateCaptureDependencies":
            if len(args) >= 4:
                return [
                    f"// HIP stream update capture dependencies: stream: {args[0]}, "
                    f"dependencies: {args[1]}, count: {args[2]}, flags: {args[3]}"
                ]
        elif name == "hipDeviceSynchronize":
            return ["// HIP device synchronize"]
        elif name in {
            "hipStreamCreate",
            "hipStreamCreateWithFlags",
            "hipStreamCreateWithPriority",
            "hipStreamDestroy",
        }:
            if args:
                action = "destroy" if name == "hipStreamDestroy" else "create"
                stream = (
                    self.format_runtime_pointer_target(node.args[0])
                    if action == "create"
                    else args[0]
                )
                comment = f"// HIP stream {action}: {stream}"
                if (
                    name
                    in {
                        "hipStreamCreateWithFlags",
                        "hipStreamCreateWithPriority",
                    }
                    and len(args) >= 2
                ):
                    comment += f", flags: {args[1]}"
                if name == "hipStreamCreateWithPriority" and len(args) >= 3:
                    comment += f", priority: {args[2]}"
                return [comment]
        elif name in {"hipEventCreate", "hipEventCreateWithFlags"}:
            if args:
                event = self.format_runtime_pointer_target(node.args[0])
                comment = f"// HIP event create: {event}"
                if len(args) >= 2:
                    comment += f", flags: {args[1]}"
                return [comment]
        elif name == "hipEventRecord":
            if args:
                comment = f"// HIP event record: {args[0]}"
                if len(args) >= 2:
                    comment += f", stream: {args[1]}"
                return [comment]
        elif name == "hipEventSynchronize":
            if args:
                return [f"// HIP event synchronize: {args[0]}"]
        elif name == "hipEventElapsedTime":
            if len(node.args) >= 3:
                output = self.format_runtime_raw_output_target(node.args[0])
                output_name = self.get_runtime_pointer_target_name(node.args[0])
                if output_name is not None:
                    self.register_device_query_source(
                        output_name,
                        f"event.elapsedTime({args[1]}, {args[2]})",
                    )
                return [
                    f"// HIP event elapsed time: {args[1]} -> {args[2]}, "
                    f"output: {output}"
                ]
        elif name == "hipEventDestroy":
            if args:
                return [f"// HIP event destroy: {args[0]}"]
        elif name == "hipEventQuery":
            if args:
                return [f"// HIP event query: {args[0]}"]
        elif name == "hipStreamWaitEvent":
            if len(args) >= 2:
                comment = f"// HIP stream wait event: {args[0]} waits for {args[1]}"
                if len(args) >= 3:
                    comment += f", flags: {args[2]}"
                return [comment]
        elif name == "hipGetLastError":
            return ["// HIP get last error"]
        elif name == "hipPeekAtLastError":
            return ["// HIP peek at last error"]
        elif name == "hipGetDevice":
            if args:
                output = self.format_runtime_raw_output_target(node.args[0])
                output_name = self.get_runtime_pointer_target_name(node.args[0])
                if output_name is not None:
                    self.register_device_query_source(output_name, "currentDevice")
                return [f"// HIP get current device: output: {output}"]
        elif name == "hipGetDeviceCount":
            if args:
                output = self.format_runtime_raw_output_target(node.args[0])
                output_name = self.get_runtime_pointer_target_name(node.args[0])
                if output_name is not None:
                    self.register_device_query_source(output_name, "deviceCount")
                return [f"// HIP get device count: output: {output}"]
        elif name == "hipSetDevice":
            if args:
                return [f"// HIP set device: {args[0]}"]
        elif name == "hipSetValidDevices":
            if len(args) >= 2:
                return [
                    f"// HIP set valid devices: devices: {args[0]}, count: {args[1]}"
                ]
        elif name == "hipGetDeviceProperties":
            if len(args) >= 2:
                output = self.format_runtime_raw_output_target(node.args[0])
                output_name = self.get_runtime_pointer_target_name(node.args[0])
                if output_name is not None:
                    self.register_device_property_source(output_name, args[1])
                return [f"// HIP get device properties: {output}, device: {args[1]}"]
        elif name == "hipDeviceGetAttribute":
            if len(args) >= 3:
                output = self.format_runtime_raw_output_target(node.args[0])
                output_name = self.get_runtime_pointer_target_name(node.args[0])
                if output_name is not None:
                    self.register_device_attribute_source(output_name, args[1], args[2])
                return [
                    f"// HIP get device attribute: output: {output}, "
                    f"attribute: {args[1]}, device: {args[2]}"
                ]
        elif name == "hipDeviceGetName":
            if len(args) >= 3:
                output = self.format_runtime_raw_output_target(node.args[0])
                return [
                    f"// HIP get device name: output: {output}, "
                    f"length: {args[1]}, device: {args[2]}"
                ]
        elif name == "hipDeviceGetUuid":
            if len(args) >= 2:
                output = self.format_runtime_raw_output_target(node.args[0])
                return [f"// HIP get device UUID: output: {output}, device: {args[1]}"]
        elif name == "hipDeviceTotalMem":
            if len(args) >= 2:
                output = self.format_runtime_raw_output_target(node.args[0])
                output_name = self.get_runtime_pointer_target_name(node.args[0])
                if output_name is not None:
                    self.register_device_query_source(output_name, "totalMem", args[1])
                return [
                    f"// HIP get device total memory: output: {output}, "
                    f"device: {args[1]}"
                ]
        elif name == "hipDeviceComputeCapability":
            if len(args) >= 3:
                major_output = self.format_runtime_raw_output_target(node.args[0])
                minor_output = self.format_runtime_raw_output_target(node.args[1])
                major_output_name = self.get_runtime_pointer_target_name(node.args[0])
                minor_output_name = self.get_runtime_pointer_target_name(node.args[1])
                if major_output_name is not None:
                    self.register_device_query_source(
                        major_output_name, "computeCapability.major", args[2]
                    )
                if minor_output_name is not None:
                    self.register_device_query_source(
                        minor_output_name, "computeCapability.minor", args[2]
                    )
                return [
                    f"// HIP get device compute capability: "
                    f"major output: {major_output}, minor output: {minor_output}, "
                    f"device: {args[2]}"
                ]
        elif name == "hipChooseDevice":
            if len(args) >= 2:
                output = self.format_runtime_raw_output_target(node.args[0])
                output_name = self.get_runtime_pointer_target_name(node.args[0])
                if output_name is not None:
                    self.register_device_query_source(output_name, "selectedDevice")
                return [
                    f"// HIP choose device: output: {output}, properties: {args[1]}"
                ]
        elif name == "hipExtGetLinkTypeAndHopCount":
            if len(args) >= 4:
                link_type_output = self.format_runtime_raw_output_target(node.args[2])
                hop_count_output = self.format_runtime_raw_output_target(node.args[3])
                link_type_output_name = self.get_runtime_pointer_target_name(
                    node.args[2]
                )
                hop_count_output_name = self.get_runtime_pointer_target_name(
                    node.args[3]
                )
                if link_type_output_name is not None:
                    self.register_device_query_source(
                        link_type_output_name, f"linkType({args[0]}, {args[1]})"
                    )
                if hop_count_output_name is not None:
                    self.register_device_query_source(
                        hop_count_output_name, f"hopCount({args[0]}, {args[1]})"
                    )
                return [
                    f"// HIP get link type and hop count: device 1: {args[0]}, "
                    f"device 2: {args[1]}, link type output: {link_type_output}, "
                    f"hop count output: {hop_count_output}"
                ]
        elif name == "hipDeviceGetPCIBusId":
            if len(args) >= 3:
                output = self.format_runtime_raw_output_target(node.args[0])
                return [
                    f"// HIP get device PCI bus id: output: {output}, "
                    f"length: {args[1]}, device: {args[2]}"
                ]
        elif name == "hipDeviceGetByPCIBusId":
            if len(args) >= 2:
                output = self.format_runtime_raw_output_target(node.args[0])
                output_name = self.get_runtime_pointer_target_name(node.args[0])
                if output_name is not None:
                    self.register_device_query_source(
                        output_name, f"deviceByPCIBusId({args[1]})"
                    )
                return [
                    f"// HIP get device by PCI bus id: output: {output}, "
                    f"bus id: {args[1]}"
                ]
        elif name == "hipDeviceGetCacheConfig":
            if args:
                output = self.format_runtime_raw_output_target(node.args[0])
                output_name = self.get_runtime_pointer_target_name(node.args[0])
                if output_name is not None:
                    self.register_device_query_source(output_name, "cacheConfig")
                return [f"// HIP get device cache config: output: {output}"]
        elif name == "hipDeviceSetCacheConfig":
            if args:
                return [f"// HIP set device cache config: {args[0]}"]
        elif name == "hipDeviceGetSharedMemConfig":
            if args:
                output = self.format_runtime_raw_output_target(node.args[0])
                output_name = self.get_runtime_pointer_target_name(node.args[0])
                if output_name is not None:
                    self.register_device_query_source(output_name, "sharedMemConfig")
                return [f"// HIP get device shared memory config: output: {output}"]
        elif name == "hipDeviceSetSharedMemConfig":
            if args:
                return [f"// HIP set device shared memory config: {args[0]}"]
        elif name == "hipDeviceGetLimit":
            if len(args) >= 2:
                output = self.format_runtime_raw_output_target(node.args[0])
                output_name = self.get_runtime_pointer_target_name(node.args[0])
                if output_name is not None:
                    self.register_device_query_source(output_name, f"limit.{args[1]}")
                return [f"// HIP get device limit: output: {output}, limit: {args[1]}"]
        elif name == "hipDeviceSetLimit":
            if len(args) >= 2:
                return [f"// HIP set device limit: limit: {args[0]}, value: {args[1]}"]
        elif name == "hipDeviceReset":
            return ["// HIP device reset"]
        elif name == "hipGetDeviceFlags":
            if args:
                output = self.format_runtime_raw_output_target(node.args[0])
                output_name = self.get_runtime_pointer_target_name(node.args[0])
                if output_name is not None:
                    self.register_device_query_source(output_name, "deviceFlags")
                return [f"// HIP get device flags: output: {output}"]
        elif name == "hipSetDeviceFlags":
            if args:
                return [f"// HIP set device flags: {args[0]}"]
        elif name == "hipGetProcAddress":
            if len(args) >= 5:
                output = self.format_runtime_raw_output_target(node.args[1])
                status_output = self.format_runtime_raw_output_target(node.args[4])
                return [
                    f"// HIP get proc address: symbol: {args[0]}, output: {output}, "
                    f"version: {args[2]}, flags: {args[3]}, "
                    f"status output: {status_output}"
                ]
        elif name == "hipMemGetInfo":
            if len(args) >= 2:
                free_output = self.format_runtime_raw_output_target(node.args[0])
                total_output = self.format_runtime_raw_output_target(node.args[1])
                free_output_name = self.get_runtime_pointer_target_name(node.args[0])
                total_output_name = self.get_runtime_pointer_target_name(node.args[1])
                if free_output_name is not None:
                    self.register_device_query_source(
                        free_output_name,
                        "memory.info.free",
                    )
                if total_output_name is not None:
                    self.register_device_query_source(
                        total_output_name,
                        "memory.info.total",
                    )
                return [
                    f"// HIP memory info: free output: {free_output}, "
                    f"total output: {total_output}"
                ]
        elif name == "hipPointerGetAttributes":
            if len(args) >= 2:
                output = self.format_runtime_raw_output_target(node.args[0])
                return [
                    f"// HIP pointer attributes: output: {output}, pointer: {args[1]}"
                ]
        elif name == "hipDrvPointerGetAttributes":
            if len(args) >= 4:
                return [
                    f"// HIP driver pointer attributes: count: {args[0]}, "
                    f"attributes: {args[1]}, data: {args[2]}, pointer: {args[3]}"
                ]
        elif name == "hipPointerGetAttribute":
            if len(args) >= 3:
                output = self.format_runtime_raw_output_target(node.args[0])
                output_name = self.get_runtime_pointer_target_name(node.args[0])
                if output_name is not None:
                    self.register_device_query_source(
                        output_name,
                        f"pointer.attribute({args[1]}, {args[2]})",
                    )
                return [
                    f"// HIP pointer attribute: output: {output}, "
                    f"attribute: {args[1]}, pointer: {args[2]}"
                ]
        elif name == "hipPointerSetAttribute":
            if len(args) >= 3:
                value = self.format_runtime_pointer_target(node.args[0])
                return [
                    f"// HIP pointer set attribute: value: {value}, "
                    f"attribute: {args[1]}, pointer: {args[2]}"
                ]
        elif name == "hipMemPtrGetInfo":
            if len(args) >= 2:
                size_output = self.format_runtime_raw_output_target(node.args[1])
                size_output_name = self.get_runtime_pointer_target_name(node.args[1])
                if size_output_name is not None:
                    self.register_device_query_source(
                        size_output_name,
                        f"memoryPointer.size({args[0]})",
                    )
                return [
                    f"// HIP memory pointer info: pointer: {args[0]}, "
                    f"size output: {size_output}"
                ]
        elif name in {
            "hipOccupancyMaxPotentialBlockSize",
            "hipOccupancyMaxPotentialBlockSizeVariableSMem",
            "hipOccupancyMaxPotentialBlockSizeVariableSMemWithFlags",
        }:
            if len(args) >= 5:
                grid_output = self.format_runtime_raw_output_target(node.args[0])
                block_output = self.format_runtime_raw_output_target(node.args[1])
                grid_output_name = self.get_runtime_pointer_target_name(node.args[0])
                block_output_name = self.get_runtime_pointer_target_name(node.args[1])
                query_kind = {
                    "hipOccupancyMaxPotentialBlockSize": "maxPotentialBlockSize",
                    "hipOccupancyMaxPotentialBlockSizeVariableSMem": (
                        "maxPotentialBlockSizeVariableSMem"
                    ),
                    "hipOccupancyMaxPotentialBlockSizeVariableSMemWithFlags": (
                        "maxPotentialBlockSizeVariableSMemWithFlags"
                    ),
                }[name]
                query_operands = [args[2], args[3], args[4]]
                if len(args) >= 6:
                    query_operands.append(args[5])
                query_operands_text = ", ".join(query_operands)
                if grid_output_name is not None:
                    self.register_device_query_source(
                        grid_output_name,
                        f"occupancy.{query_kind}.grid({query_operands_text})",
                    )
                if block_output_name is not None:
                    self.register_device_query_source(
                        block_output_name,
                        f"occupancy.{query_kind}.block({query_operands_text})",
                    )
                comment = (
                    f"// HIP occupancy max potential block size: "
                    f"grid output: {grid_output}, block output: {block_output}, "
                    f"kernel: {args[2]}, dynamic shared memory: {args[3]}, "
                    f"block size limit: {args[4]}"
                )
                if len(args) >= 6:
                    comment += f", flags: {args[5]}"
                return [comment]
        elif name in {
            "hipOccupancyMaxActiveBlocksPerMultiprocessor",
            "hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags",
        }:
            if len(args) >= 4:
                output = self.format_runtime_raw_output_target(node.args[0])
                output_name = self.get_runtime_pointer_target_name(node.args[0])
                query_kind = (
                    "maxActiveBlocksPerMultiprocessorWithFlags"
                    if name.endswith("WithFlags")
                    else "maxActiveBlocksPerMultiprocessor"
                )
                query_operands = [args[1], args[2], args[3]]
                if len(args) >= 5:
                    query_operands.append(args[4])
                if output_name is not None:
                    self.register_device_query_source(
                        output_name,
                        f"occupancy.{query_kind}({', '.join(query_operands)})",
                    )
                comment = (
                    f"// HIP occupancy active blocks per multiprocessor: "
                    f"output: {output}, kernel: {args[1]}, "
                    f"block size: {args[2]}, dynamic shared memory: {args[3]}"
                )
                if len(args) >= 5:
                    comment += f", flags: {args[4]}"
                return [comment]
        elif name == "hipGetFuncBySymbol":
            if len(args) >= 2:
                output = self.format_runtime_pointer_target(node.args[0])
                self.clear_lvalue_metadata_source(node.args[0])
                return [
                    f"// HIP get function by symbol: output: {output}, symbol: {args[1]}"
                ]
        elif name == "hipFuncGetAttribute":
            if len(args) >= 3:
                output = self.format_runtime_raw_output_target(node.args[0])
                output_name = self.get_runtime_pointer_target_name(node.args[0])
                if output_name is not None:
                    self.register_device_query_source(
                        output_name,
                        f"function.attribute({args[1]}, {args[2]})",
                    )
                return [
                    f"// HIP function get attribute: output: {output}, "
                    f"attribute: {args[1]}, function: {args[2]}"
                ]
        elif name == "hipFuncGetAttributes":
            if len(args) >= 2:
                output = self.format_runtime_raw_output_target(node.args[0])
                output_name = self.get_runtime_pointer_target_name(node.args[0])
                if output_name is not None:
                    self.register_member_query_sources(
                        output_name,
                        {
                            member: f"function.attributes.{member}({args[1]})"
                            for member in self.HIP_FUNCTION_ATTRIBUTE_MEMBERS
                        },
                    )
                return [
                    f"// HIP function get attributes: output: {output}, "
                    f"function: {args[1]}"
                ]
        elif name == "hipFuncSetAttribute":
            if len(args) >= 3:
                return [
                    f"// HIP function set attribute: function: {args[0]}, "
                    f"attribute: {args[1]}, value: {args[2]}"
                ]
        elif name == "hipFuncSetCacheConfig":
            if len(args) >= 2:
                return [
                    f"// HIP function set cache config: function: {args[0]}, "
                    f"config: {args[1]}"
                ]
        elif name == "hipFuncSetSharedMemConfig":
            if len(args) >= 2:
                return [
                    f"// HIP function set shared memory config: function: {args[0]}, "
                    f"config: {args[1]}"
                ]
        elif name in {"hipModuleLoad", "hipModuleLoadData"}:
            if len(args) >= 2:
                output = self.format_runtime_raw_output_target(node.args[0])
                source_label = "file" if name == "hipModuleLoad" else "image"
                return [
                    f"// HIP module load: output: {output}, "
                    f"{source_label}: {args[1]}"
                ]
        elif name == "hipModuleLoadDataEx":
            if len(args) >= 5:
                output = self.format_runtime_raw_output_target(node.args[0])
                return [
                    f"// HIP module load data ex: output: {output}, image: {args[1]}, "
                    f"options: {args[2]}, option keys: {args[3]}, "
                    f"option values: {args[4]}"
                ]
        elif name == "hipModuleLoadFatBinary":
            if len(args) >= 2:
                output = self.format_runtime_raw_output_target(node.args[0])
                return [
                    f"// HIP module load fat binary: output: {output}, "
                    f"fat binary: {args[1]}"
                ]
        elif name == "hipModuleUnload":
            if args:
                return [f"// HIP module unload: {args[0]}"]
        elif name == "hipModuleGetFunction":
            if len(args) >= 3:
                output = self.format_runtime_pointer_target(node.args[0])
                self.clear_lvalue_metadata_source(node.args[0])
                return [
                    f"// HIP module get function: output: {output}, "
                    f"module: {args[1]}, name: {args[2]}"
                ]
        elif name == "hipModuleGetFunctionCount":
            if len(args) >= 2:
                output = self.format_runtime_pointer_target(node.args[0])
                output_name = self.get_runtime_pointer_target_name(node.args[0])
                if output_name is not None:
                    self.register_device_query_source(
                        output_name,
                        f"module.functionCount({args[1]})",
                    )
                else:
                    self.clear_lvalue_metadata_source(node.args[0])
                return [
                    f"// HIP module get function count: output: {output}, "
                    f"module: {args[1]}"
                ]
        elif name == "hipModuleGetGlobal":
            if len(args) >= 4:
                pointer_output = self.format_runtime_pointer_target(node.args[0])
                size_output = self.format_runtime_pointer_target(node.args[1])
                size_output_name = self.get_runtime_pointer_target_name(node.args[1])
                self.clear_lvalue_metadata_source(node.args[0])
                if size_output_name is not None:
                    self.register_device_query_source(
                        size_output_name,
                        f"module.global.size({args[2]}, {args[3]})",
                    )
                else:
                    self.clear_lvalue_metadata_source(node.args[1])
                return [
                    f"// HIP module get global: pointer output: {pointer_output}, "
                    f"size output: {size_output}, module: {args[2]}, name: {args[3]}"
                ]
        elif name == "hipModuleGetTexRef":
            if len(args) >= 3:
                output = self.format_runtime_pointer_target(node.args[0])
                self.clear_lvalue_metadata_source(node.args[0])
                return [
                    f"// HIP module get texture reference: output: {output}, "
                    f"module: {args[1]}, name: {args[2]}"
                ]
        elif name == "hipBindTexture":
            if len(args) >= 5:
                offset = self.format_runtime_raw_output_target(node.args[0])
                return [
                    "// HIP texture reference bind: "
                    f"offset output: {offset}, texture: {args[1]}, "
                    f"pointer: {args[2]}, desc: {args[3]}, bytes: {args[4]}"
                ]
        elif name == "hipBindTexture2D":
            if len(args) >= 7:
                offset = self.format_runtime_raw_output_target(node.args[0])
                return [
                    "// HIP texture reference bind 2D: "
                    f"offset output: {offset}, texture: {args[1]}, "
                    f"pointer: {args[2]}, desc: {args[3]}, width: {args[4]}, "
                    f"height: {args[5]}, pitch: {args[6]}"
                ]
        elif name == "hipBindTextureToArray":
            if len(args) >= 3:
                return [
                    "// HIP texture reference bind array: "
                    f"texture: {args[0]}, array: {args[1]}, desc: {args[2]}"
                ]
        elif name == "hipBindTextureToMipmappedArray":
            if len(args) >= 3:
                return [
                    "// HIP texture reference bind mipmapped array: "
                    f"texture: {args[0]}, mipmapped array: {args[1]}, "
                    f"desc: {args[2]}"
                ]
        elif name == "hipGetTextureReference":
            if len(args) >= 2:
                output = self.format_runtime_pointer_target(node.args[0])
                self.clear_lvalue_metadata_source(node.args[0])
                return [
                    f"// HIP get texture reference: output: {output}, "
                    f"symbol: {args[1]}"
                ]
        elif name == "hipGetTextureAlignmentOffset":
            if len(args) >= 2:
                output = self.format_runtime_raw_output_target(node.args[0])
                output_name = self.get_runtime_pointer_target_name(node.args[0])
                if output_name is not None:
                    self.register_device_query_source(
                        output_name,
                        f"textureReference.alignmentOffset({args[1]})",
                    )
                return [
                    "// HIP texture alignment offset query: "
                    f"output: {output}, texture: {args[1]}"
                ]
        elif name == "hipUnbindTexture":
            if args:
                return [f"// HIP texture reference unbind: {args[0]}"]
        elif name == "hipTexRefGetAddress":
            if len(args) >= 2:
                output = self.format_runtime_pointer_target(node.args[0])
                self.clear_lvalue_metadata_source(node.args[0])
                return [
                    "// HIP texture reference get address: "
                    f"output: {output}, texture: {args[1]}"
                ]
        elif name == "hipTexRefGetAddressMode":
            if len(args) >= 3:
                output = self.format_runtime_raw_output_target(node.args[0])
                output_name = self.get_runtime_pointer_target_name(node.args[0])
                if output_name is not None:
                    self.register_device_query_source(
                        output_name,
                        f"textureReference.addressMode({args[1]}, {args[2]})",
                    )
                return [
                    "// HIP texture reference get address mode: "
                    f"output: {output}, texture: {args[1]}, dim: {args[2]}"
                ]
        elif name in {
            "hipTexRefGetArray",
            "hipTexRefGetBorderColor",
            "hipTexRefGetFilterMode",
            "hipTexRefGetFlags",
            "hipTexRefGetMaxAnisotropy",
            "hipTexRefGetMipmapFilterMode",
            "hipTexRefGetMipmapLevelBias",
            "hipTexRefGetMipMappedArray",
            "hipTexRefGetMipmappedArray",
        }:
            if len(args) >= 2:
                output = self.format_runtime_raw_output_target(node.args[0])
                output_name = self.get_runtime_pointer_target_name(node.args[0])
                labels = {
                    "hipTexRefGetArray": "array",
                    "hipTexRefGetBorderColor": "border color",
                    "hipTexRefGetFilterMode": "filter mode",
                    "hipTexRefGetFlags": "flags",
                    "hipTexRefGetMaxAnisotropy": "max anisotropy",
                    "hipTexRefGetMipmapFilterMode": "mipmap filter mode",
                    "hipTexRefGetMipmapLevelBias": "mipmap level bias",
                    "hipTexRefGetMipMappedArray": "mipmapped array",
                    "hipTexRefGetMipmappedArray": "mipmapped array",
                }
                query_names = {
                    "hipTexRefGetFilterMode": "filterMode",
                    "hipTexRefGetFlags": "flags",
                    "hipTexRefGetMaxAnisotropy": "maxAnisotropy",
                    "hipTexRefGetMipmapFilterMode": "mipmapFilterMode",
                    "hipTexRefGetMipmapLevelBias": "mipmapLevelBias",
                }
                query_name = query_names.get(name)
                if output_name is not None and query_name is not None:
                    self.register_device_query_source(
                        output_name,
                        f"textureReference.{query_name}({args[1]})",
                    )
                elif output_name is not None and name == "hipTexRefGetBorderColor":
                    self.register_member_query_sources(
                        output_name,
                        {
                            f"[{index}]": (
                                f"textureReference.borderColor[{index}]({args[1]})"
                            )
                            for index in range(4)
                        },
                    )
                else:
                    self.clear_lvalue_metadata_source(node.args[0])
                return [
                    f"// HIP texture reference get {labels[name]}: "
                    f"output: {output}, texture: {args[1]}"
                ]
        elif name == "hipTexRefGetFormat":
            if len(args) >= 3:
                format_output = self.format_runtime_raw_output_target(node.args[0])
                channels_output = self.format_runtime_raw_output_target(node.args[1])
                format_output_name = self.get_runtime_pointer_target_name(node.args[0])
                channels_output_name = self.get_runtime_pointer_target_name(
                    node.args[1]
                )
                if format_output_name is not None:
                    self.register_device_query_source(
                        format_output_name,
                        f"textureReference.format({args[2]})",
                    )
                if channels_output_name is not None:
                    self.register_device_query_source(
                        channels_output_name,
                        f"textureReference.channelCount({args[2]})",
                    )
                return [
                    "// HIP texture reference get format: "
                    f"format output: {format_output}, "
                    f"channels output: {channels_output}, texture: {args[2]}"
                ]
        elif name == "hipTexRefGetMipmapLevelClamp":
            if len(args) >= 3:
                min_output = self.format_runtime_raw_output_target(node.args[0])
                max_output = self.format_runtime_raw_output_target(node.args[1])
                min_output_name = self.get_runtime_pointer_target_name(node.args[0])
                max_output_name = self.get_runtime_pointer_target_name(node.args[1])
                if min_output_name is not None:
                    self.register_device_query_source(
                        min_output_name,
                        f"textureReference.mipmapLevelClamp.min({args[2]})",
                    )
                if max_output_name is not None:
                    self.register_device_query_source(
                        max_output_name,
                        f"textureReference.mipmapLevelClamp.max({args[2]})",
                    )
                return [
                    "// HIP texture reference get mipmap level clamp: "
                    f"min output: {min_output}, max output: {max_output}, "
                    f"texture: {args[2]}"
                ]
        elif name == "hipTexRefSetAddress":
            if len(args) >= 4:
                offset = self.format_runtime_raw_output_target(node.args[0])
                return [
                    "// HIP texture reference set address: "
                    f"offset output: {offset}, texture: {args[1]}, "
                    f"pointer: {args[2]}, bytes: {args[3]}"
                ]
        elif name == "hipTexRefSetAddress2D":
            if len(args) >= 4:
                return [
                    "// HIP texture reference set address 2D: "
                    f"texture: {args[0]}, desc: {args[1]}, pointer: {args[2]}, "
                    f"pitch: {args[3]}"
                ]
        elif name in {
            "hipTexRefSetAddressMode",
            "hipTexRefSetArray",
            "hipTexRefSetBorderColor",
            "hipTexRefSetFilterMode",
            "hipTexRefSetFlags",
            "hipTexRefSetFormat",
            "hipTexRefSetMaxAnisotropy",
            "hipTexRefSetMipmapFilterMode",
            "hipTexRefSetMipmapLevelBias",
            "hipTexRefSetMipmapLevelClamp",
            "hipTexRefSetMipmappedArray",
        }:
            texture = args[0] if args else "<missing>"
            if name == "hipTexRefSetAddressMode" and len(args) >= 3:
                return [
                    "// HIP texture reference set address mode: "
                    f"texture: {texture}, dim: {args[1]}, mode: {args[2]}"
                ]
            if name == "hipTexRefSetArray" and len(args) >= 3:
                return [
                    "// HIP texture reference set array: "
                    f"texture: {texture}, array: {args[1]}, flags: {args[2]}"
                ]
            if name == "hipTexRefSetBorderColor" and len(args) >= 2:
                return [
                    "// HIP texture reference set border color: "
                    f"texture: {texture}, color: {args[1]}"
                ]
            if name == "hipTexRefSetFilterMode" and len(args) >= 2:
                return [
                    "// HIP texture reference set filter mode: "
                    f"texture: {texture}, mode: {args[1]}"
                ]
            if name == "hipTexRefSetFlags" and len(args) >= 2:
                return [
                    "// HIP texture reference set flags: "
                    f"texture: {texture}, flags: {args[1]}"
                ]
            if name == "hipTexRefSetFormat" and len(args) >= 3:
                return [
                    "// HIP texture reference set format: "
                    f"texture: {texture}, format: {args[1]}, "
                    f"components: {args[2]}"
                ]
            if name == "hipTexRefSetMaxAnisotropy" and len(args) >= 2:
                return [
                    "// HIP texture reference set max anisotropy: "
                    f"texture: {texture}, value: {args[1]}"
                ]
            if name == "hipTexRefSetMipmapFilterMode" and len(args) >= 2:
                return [
                    "// HIP texture reference set mipmap filter mode: "
                    f"texture: {texture}, mode: {args[1]}"
                ]
            if name == "hipTexRefSetMipmapLevelBias" and len(args) >= 2:
                return [
                    "// HIP texture reference set mipmap level bias: "
                    f"texture: {texture}, bias: {args[1]}"
                ]
            if name == "hipTexRefSetMipmapLevelClamp" and len(args) >= 3:
                return [
                    "// HIP texture reference set mipmap level clamp: "
                    f"texture: {texture}, min: {args[1]}, max: {args[2]}"
                ]
            if name == "hipTexRefSetMipmappedArray" and len(args) >= 3:
                return [
                    "// HIP texture reference set mipmapped array: "
                    f"texture: {texture}, mipmapped array: {args[1]}, "
                    f"flags: {args[2]}"
                ]
        elif name == "hipGetDriverEntryPoint":
            if len(args) >= 4:
                output = self.format_runtime_raw_output_target(node.args[1])
                status_output = self.format_runtime_raw_output_target(node.args[3])
                return [
                    f"// HIP get driver entry point: symbol: {args[0]}, "
                    f"output: {output}, flags: {args[2]}, "
                    f"status output: {status_output}"
                ]
        elif name in {"hipLibraryLoadData", "hipLibraryLoadFromFile"}:
            if len(args) >= 8:
                output = self.format_runtime_raw_output_target(node.args[0])
                source_label = "file" if name == "hipLibraryLoadFromFile" else "code"
                return [
                    f"// HIP library load: output: {output}, "
                    f"{source_label}: {args[1]}, jit options: {args[2]}, "
                    f"jit option values: {args[3]}, jit option count: {args[4]}, "
                    f"library options: {args[5]}, "
                    f"library option values: {args[6]}, "
                    f"library option count: {args[7]}"
                ]
        elif name == "hipLibraryUnload":
            if args:
                return [f"// HIP library unload: {args[0]}"]
        elif name == "hipLibraryGetKernel":
            if len(args) >= 3:
                output = self.format_runtime_pointer_target(node.args[0])
                self.clear_lvalue_metadata_source(node.args[0])
                return [
                    f"// HIP library get kernel: output: {output}, "
                    f"library: {args[1]}, name: {args[2]}"
                ]
        elif name == "hipLibraryGetKernelCount":
            if len(args) >= 2:
                output = self.format_runtime_pointer_target(node.args[0])
                output_name = self.get_runtime_pointer_target_name(node.args[0])
                if output_name is not None:
                    self.register_device_query_source(
                        output_name,
                        f"library.kernelCount({args[1]})",
                    )
                else:
                    self.clear_lvalue_metadata_source(node.args[0])
                return [
                    f"// HIP library get kernel count: output: {output}, "
                    f"library: {args[1]}"
                ]
        elif name == "hipLibraryEnumerateKernels":
            if len(args) >= 3:
                self.clear_lvalue_metadata_source(node.args[0])
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    f"// HIP library enumerate kernels: output: {output}, "
                    f"max kernels: {args[1]}, library: {args[2]}"
                ]
        elif name == "hipKernelGetLibrary":
            if len(args) >= 2:
                output = self.format_runtime_pointer_target(node.args[0])
                self.clear_lvalue_metadata_source(node.args[0])
                return [
                    f"// HIP kernel get library: output: {output}, "
                    f"kernel: {args[1]}"
                ]
        elif name == "hipKernelGetName":
            if len(args) >= 2:
                output = self.format_runtime_pointer_target(node.args[0])
                self.clear_lvalue_metadata_source(node.args[0])
                return [f"// HIP kernel get name: output: {output}, kernel: {args[1]}"]
        elif name == "hipKernelGetParamInfo":
            if len(args) >= 4:
                offset_output = self.format_runtime_pointer_target(node.args[2])
                size_output = self.format_runtime_pointer_target(node.args[3])
                offset_output_name = self.get_runtime_pointer_target_name(node.args[2])
                size_output_name = self.get_runtime_pointer_target_name(node.args[3])
                if offset_output_name is not None:
                    self.register_device_query_source(
                        offset_output_name,
                        f"kernel.param.offset({args[0]}, {args[1]})",
                    )
                else:
                    self.clear_lvalue_metadata_source(node.args[2])
                if size_output_name is not None:
                    self.register_device_query_source(
                        size_output_name,
                        f"kernel.param.size({args[0]}, {args[1]})",
                    )
                else:
                    self.clear_lvalue_metadata_source(node.args[3])
                return [
                    f"// HIP kernel get parameter info: kernel: {args[0]}, "
                    f"param index: {args[1]}, offset output: {offset_output}, "
                    f"size output: {size_output}"
                ]
        elif name == "hipKernelGetFunction":
            if len(args) >= 2:
                output = self.format_runtime_pointer_target(node.args[0])
                self.clear_lvalue_metadata_source(node.args[0])
                return [
                    f"// HIP kernel get function: output: {output}, "
                    f"kernel: {args[1]}"
                ]
        elif name == "hipKernelGetAttribute":
            if len(args) >= 4:
                output = self.format_runtime_pointer_target(node.args[0])
                output_name = self.get_runtime_pointer_target_name(node.args[0])
                if output_name is not None:
                    self.register_device_query_source(
                        output_name,
                        f"kernel.attribute({args[1]}, {args[2]}, {args[3]})",
                    )
                else:
                    self.clear_lvalue_metadata_source(node.args[0])
                return [
                    f"// HIP kernel get attribute: output: {output}, "
                    f"attribute: {args[1]}, kernel: {args[2]}, device: {args[3]}"
                ]
        elif name == "hipKernelSetAttribute":
            if len(args) >= 4:
                return [
                    f"// HIP kernel set attribute: attribute: {args[0]}, "
                    f"value: {args[1]}, kernel: {args[2]}, device: {args[3]}"
                ]
        elif name == "hipLinkCreate":
            if len(args) >= 4:
                output = self.format_runtime_raw_output_target(node.args[3])
                return [
                    f"// HIP link create: options: {args[0]}, "
                    f"option keys: {args[1]}, option values: {args[2]}, "
                    f"state output: {output}"
                ]
        elif name == "hipLinkAddFile":
            if len(args) >= 6:
                return [
                    f"// HIP link add file: state: {args[0]}, input type: {args[1]}, "
                    f"path: {args[2]}, options: {args[3]}, option keys: {args[4]}, "
                    f"option values: {args[5]}"
                ]
        elif name == "hipLinkAddData":
            if len(args) >= 8:
                return [
                    f"// HIP link add data: state: {args[0]}, input type: {args[1]}, "
                    f"image: {args[2]}, bytes: {args[3]}, name: {args[4]}, "
                    f"options: {args[5]}, option keys: {args[6]}, "
                    f"option values: {args[7]}"
                ]
        elif name == "hipLinkComplete":
            if len(args) >= 3:
                self.clear_lvalue_metadata_source(node.args[1])
                binary_output = self.format_runtime_pointer_target(node.args[1])
                self.clear_lvalue_metadata_source(node.args[2])
                size_output = self.format_runtime_pointer_target(node.args[2])
                size_output_name = self.get_runtime_pointer_target_name(node.args[2])
                if size_output_name is not None:
                    self.register_device_query_source(
                        size_output_name,
                        f"link.complete.size({args[0]})",
                    )
                return [
                    f"// HIP link complete: state: {args[0]}, "
                    f"binary output: {binary_output}, size output: {size_output}"
                ]
        elif name == "hipLinkDestroy":
            if args:
                return [f"// HIP link destroy: state: {args[0]}"]
        elif name == "hipMemGetHandleForAddressRange":
            if len(args) >= 5:
                output = self.format_runtime_raw_output_target(node.args[0])
                return [
                    f"// HIP memory get handle for address range: output: {output}, "
                    f"device pointer: {args[1]}, bytes: {args[2]}, "
                    f"handle type: {args[3]}, flags: {args[4]}"
                ]
        elif name == "hipModuleLaunchKernel":
            if len(args) >= 11:
                return [
                    f"// HIP module launch kernel: function: {args[0]}, "
                    f"grid: ({args[1]}, {args[2]}, {args[3]}), "
                    f"block: ({args[4]}, {args[5]}, {args[6]}), "
                    f"shared memory: {args[7]}, stream: {args[8]}, "
                    f"params: {args[9]}, extra: {args[10]}"
                ]
        elif name in {"hipExtModuleLaunchKernel", "hipHccModuleLaunchKernel"}:
            if len(args) >= 13:
                label = "extended" if name == "hipExtModuleLaunchKernel" else "HCC"
                comment = (
                    f"// HIP {label} module launch kernel: function: {args[0]}, "
                    f"global work size: ({args[1]}, {args[2]}, {args[3]}), "
                    f"local work size: ({args[4]}, {args[5]}, {args[6]}), "
                    f"shared memory: {args[7]}, stream: {args[8]}, "
                    f"params: {args[9]}, extra: {args[10]}, "
                    f"start event: {args[11]}, stop event: {args[12]}"
                )
                if len(args) >= 14:
                    comment += f", flags: {args[13]}"
                return [comment]
        elif name == "hipLaunchCooperativeKernel":
            if len(args) >= 6:
                return [
                    f"// HIP cooperative kernel launch: function: {args[0]}, "
                    f"grid: {args[1]}, block: {args[2]}, params: {args[3]}, "
                    f"shared memory: {args[4]}, stream: {args[5]}"
                ]
        elif name == "hipLaunchCooperativeKernelMultiDevice":
            if len(args) >= 3:
                return [
                    f"// HIP cooperative multi-device launch: params: {args[0]}, "
                    f"devices: {args[1]}, flags: {args[2]}"
                ]
        elif name == "hipModuleLaunchCooperativeKernel":
            if len(args) >= 10:
                return [
                    f"// HIP module cooperative kernel launch: function: {args[0]}, "
                    f"grid: ({args[1]}, {args[2]}, {args[3]}), "
                    f"block: ({args[4]}, {args[5]}, {args[6]}), "
                    f"shared memory: {args[7]}, stream: {args[8]}, "
                    f"params: {args[9]}"
                ]
        elif name == "hipModuleLaunchCooperativeKernelMultiDevice":
            if len(args) >= 3:
                return [
                    f"// HIP module cooperative multi-device launch: "
                    f"params: {args[0]}, devices: {args[1]}, flags: {args[2]}"
                ]

        rtc_comments = self.format_hip_rtc_runtime_call(name, node.args, args)
        if rtc_comments is not None:
            return rtc_comments

        interop_comments = self.format_hip_interop_runtime_call(name, node.args, args)
        if interop_comments is not None:
            return interop_comments

        driver_comments = self.format_hip_driver_runtime_call(name, node.args, args)
        if driver_comments is not None:
            return driver_comments

        graph_comments = self.format_hip_graph_runtime_call(name, node.args, args)
        if graph_comments is not None:
            return graph_comments

        if name in {"hipCreateTextureObject", "hipTexObjectCreate"}:
            if len(args) >= 4:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    f"// HIP texture object create: {output}, resource: {args[1]}, "
                    f"texture desc: {args[2]}, resource view: {args[3]}"
                ]
        elif name in {"hipDestroyTextureObject", "hipTexObjectDestroy"}:
            if args:
                return [f"// HIP texture object destroy: {args[0]}"]
        elif name in {
            "hipGetTextureObjectResourceDesc",
            "hipTexObjectGetResourceDesc",
        }:
            if len(args) >= 2:
                output = self.format_runtime_raw_output_target(node.args[0])
                output_name = self.get_runtime_pointer_target_name(node.args[0])
                if output_name is not None:
                    self.register_member_query_sources(
                        output_name,
                        {
                            member: f"textureObject.resourceDesc.{member}({args[1]})"
                            for member in (
                                self.HIP_RESOURCE_DESCRIPTOR_MEMBERS
                                | self.HIP_RESOURCE_DESCRIPTOR_NESTED_MEMBERS
                            )
                        },
                    )
                return [
                    f"// HIP texture object get resource desc: output: {output}, "
                    f"texture: {args[1]}"
                ]
        elif name in {"hipGetTextureObjectTextureDesc", "hipTexObjectGetTextureDesc"}:
            if len(args) >= 2:
                output = self.format_runtime_raw_output_target(node.args[0])
                output_name = self.get_runtime_pointer_target_name(node.args[0])
                if output_name is not None:
                    self.register_member_query_sources(
                        output_name,
                        {
                            member: f"textureObject.textureDesc.{member}({args[1]})"
                            for member in (
                                self.HIP_TEXTURE_DESCRIPTOR_MEMBERS
                                | self.HIP_TEXTURE_DESCRIPTOR_INDEXED_MEMBERS
                            )
                        },
                    )
                return [
                    f"// HIP texture object get texture desc: output: {output}, "
                    f"texture: {args[1]}"
                ]
        elif name in {
            "hipGetTextureObjectResourceViewDesc",
            "hipTexObjectGetResourceViewDesc",
        }:
            if len(args) >= 2:
                output = self.format_runtime_raw_output_target(node.args[0])
                output_name = self.get_runtime_pointer_target_name(node.args[0])
                if output_name is not None:
                    self.register_member_query_sources(
                        output_name,
                        {
                            member: (
                                f"textureObject.resourceViewDesc.{member}({args[1]})"
                            )
                            for member in self.HIP_RESOURCE_VIEW_DESCRIPTOR_MEMBERS
                        },
                    )
                return [
                    f"// HIP texture object get resource view desc: output: {output}, "
                    f"texture: {args[1]}"
                ]
        elif name == "hipGetChannelDesc":
            if len(args) >= 2:
                output = self.format_runtime_raw_output_target(node.args[0])
                output_name = self.get_runtime_pointer_target_name(node.args[0])
                if output_name is not None:
                    self.register_member_query_sources(
                        output_name,
                        {
                            member: f"array.channelDesc.{member}({args[1]})"
                            for member in self.HIP_CHANNEL_DESCRIPTOR_MEMBERS
                        },
                    )
                return [f"// HIP get channel desc: output: {output}, array: {args[1]}"]
        elif name == "hipCreateSurfaceObject":
            if len(args) >= 2:
                output = self.format_runtime_pointer_target(node.args[0])
                return [f"// HIP surface object create: {output}, resource: {args[1]}"]
        elif name == "hipGetSurfaceObjectResourceDesc":
            if len(args) >= 2:
                output = self.format_runtime_pointer_target(node.args[0])
                return [
                    "// HIP surface object resource descriptor query not supported "
                    f"by HIP runtime: surface: {args[1]}, output: {output}"
                ]
        elif name == "hipDestroySurfaceObject":
            if args:
                return [f"// HIP surface object destroy: {args[0]}"]

        return None

    def format_hip_interop_runtime_call(self, name, raw_args, args):
        if name == "hipProfilerStart":
            return ["// HIP profiler start"]
        if name == "hipProfilerStop":
            return ["// HIP profiler stop"]
        if name == "hipImportExternalMemory":
            if len(args) >= 2:
                output = self.format_runtime_raw_output_target(raw_args[0])
                return [
                    f"// HIP import external memory: output: {output}, "
                    f"descriptor: {args[1]}"
                ]
        if name == "hipDestroyExternalMemory":
            if args:
                return [f"// HIP destroy external memory: {args[0]}"]
        if name == "hipExternalMemoryGetMappedBuffer":
            if len(args) >= 3:
                output = self.format_runtime_raw_output_target(raw_args[0])
                return [
                    f"// HIP external memory mapped buffer: output: {output}, "
                    f"memory: {args[1]}, descriptor: {args[2]}"
                ]
        if name == "hipExternalMemoryGetMappedMipmappedArray":
            if len(args) >= 3:
                output = self.format_runtime_raw_output_target(raw_args[0])
                return [
                    f"// HIP external memory mapped mipmapped array: output: {output}, "
                    f"memory: {args[1]}, descriptor: {args[2]}"
                ]
        if name == "hipFreeMipmappedArray":
            if args:
                return [f"// HIP free mipmapped array: {args[0]}"]
        if name == "hipImportExternalSemaphore":
            if len(args) >= 2:
                output = self.format_runtime_raw_output_target(raw_args[0])
                return [
                    f"// HIP import external semaphore: output: {output}, "
                    f"descriptor: {args[1]}"
                ]
        if name == "hipDestroyExternalSemaphore":
            if args:
                return [f"// HIP destroy external semaphore: {args[0]}"]
        if name in {
            "hipSignalExternalSemaphoresAsync",
            "hipWaitExternalSemaphoresAsync",
        }:
            if len(args) >= 4:
                action = (
                    "signal" if name == "hipSignalExternalSemaphoresAsync" else "wait"
                )
                return [
                    f"// HIP {action} external semaphores: semaphores: {args[0]}, "
                    f"params: {args[1]}, count: {args[2]}, stream: {args[3]}"
                ]
        if name == "hipGraphicsGLRegisterBuffer":
            if len(args) >= 3:
                output = self.format_runtime_raw_output_target(raw_args[0])
                return [
                    f"// HIP OpenGL register buffer: output: {output}, "
                    f"buffer: {args[1]}, flags: {args[2]}"
                ]
        if name == "hipGraphicsGLRegisterImage":
            if len(args) >= 4:
                output = self.format_runtime_raw_output_target(raw_args[0])
                return [
                    f"// HIP OpenGL register image: output: {output}, "
                    f"image: {args[1]}, target: {args[2]}, flags: {args[3]}"
                ]
        if name == "hipGraphicsUnregisterResource":
            if args:
                return [f"// HIP graphics unregister resource: {args[0]}"]
        if name in {"hipGraphicsMapResources", "hipGraphicsUnmapResources"}:
            if len(args) >= 3:
                action = "map" if name == "hipGraphicsMapResources" else "unmap"
                return [
                    f"// HIP graphics {action} resources: count: {args[0]}, "
                    f"resources: {args[1]}, stream: {args[2]}"
                ]
        if name == "hipGraphicsResourceGetMappedPointer":
            if len(args) >= 3:
                pointer_output = self.format_runtime_raw_output_target(raw_args[0])
                size_output = self.format_runtime_raw_output_target(raw_args[1])
                return [
                    f"// HIP graphics mapped pointer: pointer output: {pointer_output}, "
                    f"size output: {size_output}, resource: {args[2]}"
                ]
        if name == "hipGraphicsSubResourceGetMappedArray":
            if len(args) >= 4:
                output = self.format_runtime_raw_output_target(raw_args[0])
                return [
                    f"// HIP graphics mapped subresource array: output: {output}, "
                    f"resource: {args[1]}, array index: {args[2]}, mip level: {args[3]}"
                ]
        return None

    def format_hip_rtc_runtime_call(self, name, raw_args, args):
        if name == "hiprtcVersion":
            if len(args) >= 2:
                major_output = self.format_runtime_raw_output_target(raw_args[0])
                minor_output = self.format_runtime_raw_output_target(raw_args[1])
                major_output_name = self.get_runtime_pointer_target_name(raw_args[0])
                minor_output_name = self.get_runtime_pointer_target_name(raw_args[1])
                if major_output_name is not None:
                    self.register_device_query_source(
                        major_output_name, "rtc.version.major"
                    )
                if minor_output_name is not None:
                    self.register_device_query_source(
                        minor_output_name, "rtc.version.minor"
                    )
                return [
                    f"// HIPRTC version: major output: {major_output}, "
                    f"minor output: {minor_output}"
                ]
        if name == "hiprtcCreateProgram":
            if len(args) >= 6:
                output = self.format_runtime_raw_output_target(raw_args[0])
                return [
                    f"// HIPRTC create program: output: {output}, source: {args[1]}, "
                    f"name: {args[2]}, headers: {args[3]}, header sources: {args[4]}, "
                    f"include names: {args[5]}"
                ]
        if name == "hiprtcDestroyProgram":
            if args:
                output = self.format_runtime_raw_output_target(raw_args[0])
                return [f"// HIPRTC destroy program: output: {output}"]
        if name == "hiprtcCompileProgram":
            if len(args) >= 3:
                return [
                    f"// HIPRTC compile program: program: {args[0]}, "
                    f"options: {args[1]}, option values: {args[2]}"
                ]
        if name in {
            "hiprtcGetCodeSize",
            "hiprtcGetBitcodeSize",
            "hiprtcGetProgramLogSize",
        }:
            if len(args) >= 2:
                output = self.format_runtime_raw_output_target(raw_args[1])
                output_name = self.get_runtime_pointer_target_name(raw_args[1])
                artifact = {
                    "hiprtcGetCodeSize": "code size",
                    "hiprtcGetBitcodeSize": "bitcode size",
                    "hiprtcGetProgramLogSize": "program log size",
                }[name]
                query = {
                    "hiprtcGetCodeSize": "rtc.program.code.size",
                    "hiprtcGetBitcodeSize": "rtc.program.bitcode.size",
                    "hiprtcGetProgramLogSize": "rtc.program.log.size",
                }[name]
                if output_name is not None:
                    self.register_device_query_source(
                        output_name, f"{query}({args[0]})"
                    )
                return [
                    f"// HIPRTC get {artifact}: program: {args[0]}, output: {output}"
                ]
        if name in {"hiprtcGetCode", "hiprtcGetBitcode", "hiprtcGetProgramLog"}:
            if len(args) >= 2:
                self.clear_lvalue_metadata_source(raw_args[1])
                artifact = {
                    "hiprtcGetCode": "code",
                    "hiprtcGetBitcode": "bitcode",
                    "hiprtcGetProgramLog": "program log",
                }[name]
                return [
                    f"// HIPRTC get {artifact}: program: {args[0]}, output: {args[1]}"
                ]
        if name == "hiprtcAddNameExpression":
            if len(args) >= 2:
                return [
                    f"// HIPRTC add name expression: program: {args[0]}, "
                    f"expression: {args[1]}"
                ]
        if name == "hiprtcGetLoweredName":
            if len(args) >= 3:
                output = self.format_runtime_raw_output_target(raw_args[2])
                return [
                    f"// HIPRTC get lowered name: program: {args[0]}, "
                    f"expression: {args[1]}, output: {output}"
                ]
        if name == "hiprtcLinkCreate":
            if len(args) >= 4:
                output = self.format_runtime_raw_output_target(raw_args[3])
                return [
                    f"// HIPRTC link create: options: {args[0]}, "
                    f"option keys: {args[1]}, option values: {args[2]}, "
                    f"state output: {output}"
                ]
        if name == "hiprtcLinkAddFile":
            if len(args) >= 6:
                return [
                    f"// HIPRTC link add file: state: {args[0]}, input type: {args[1]}, "
                    f"path: {args[2]}, options: {args[3]}, option keys: {args[4]}, "
                    f"option values: {args[5]}"
                ]
        if name == "hiprtcLinkAddData":
            if len(args) >= 8:
                return [
                    f"// HIPRTC link add data: state: {args[0]}, input type: {args[1]}, "
                    f"image: {args[2]}, bytes: {args[3]}, name: {args[4]}, "
                    f"options: {args[5]}, option keys: {args[6]}, "
                    f"option values: {args[7]}"
                ]
        if name == "hiprtcLinkComplete":
            if len(args) >= 3:
                binary_output = self.format_runtime_raw_output_target(raw_args[1])
                size_output = self.format_runtime_raw_output_target(raw_args[2])
                size_output_name = self.get_runtime_pointer_target_name(raw_args[2])
                if size_output_name is not None:
                    self.register_device_query_source(
                        size_output_name,
                        f"rtc.link.complete.size({args[0]})",
                    )
                return [
                    f"// HIPRTC link complete: state: {args[0]}, "
                    f"binary output: {binary_output}, size output: {size_output}"
                ]
        if name == "hiprtcLinkDestroy":
            if args:
                return [f"// HIPRTC link destroy: state: {args[0]}"]
        return None

    def format_hip_driver_runtime_call(self, name, raw_args, args):
        if name == "hipInit":
            if args:
                return [f"// HIP initialize runtime: flags: {args[0]}"]
        if name in {"hipDriverGetVersion", "hipRuntimeGetVersion"}:
            if args:
                output = self.format_runtime_raw_output_target(raw_args[0])
                version_kind = "driver" if name == "hipDriverGetVersion" else "runtime"
                output_name = self.get_runtime_pointer_target_name(raw_args[0])
                if output_name is not None:
                    self.register_device_query_source(
                        output_name, f"{version_kind}.version"
                    )
                return [f"// HIP get {version_kind} version: output: {output}"]
        if name == "hipDeviceGet":
            if len(args) >= 2:
                output = self.format_runtime_raw_output_target(raw_args[0])
                output_name = self.get_runtime_pointer_target_name(raw_args[0])
                if output_name is not None:
                    self.register_device_query_source(
                        output_name, f"deviceHandle({args[1]})"
                    )
                return [
                    f"// HIP get device handle: output: {output}, ordinal: {args[1]}"
                ]
        if name == "hipDeviceCanAccessPeer":
            if len(args) >= 3:
                output = self.format_runtime_raw_output_target(raw_args[0])
                output_name = self.get_runtime_pointer_target_name(raw_args[0])
                if output_name is not None:
                    self.register_device_query_source(
                        output_name, f"canAccessPeer({args[1]}, {args[2]})"
                    )
                return [
                    f"// HIP device can access peer: output: {output}, "
                    f"device: {args[1]}, peer device: {args[2]}"
                ]
        if name == "hipDeviceGetP2PAttribute":
            if len(args) >= 4:
                output = self.format_runtime_raw_output_target(raw_args[0])
                output_name = self.get_runtime_pointer_target_name(raw_args[0])
                if output_name is not None:
                    self.register_device_query_source(
                        output_name,
                        f"p2pAttribute.{args[1]}({args[2]}, {args[3]})",
                    )
                return [
                    f"// HIP get P2P attribute: output: {output}, "
                    f"attribute: {args[1]}, source device: {args[2]}, "
                    f"destination device: {args[3]}"
                ]
        if name in {"hipDeviceEnablePeerAccess", "hipDeviceDisablePeerAccess"}:
            if args:
                action = "enable" if name == "hipDeviceEnablePeerAccess" else "disable"
                comment = f"// HIP {action} peer access: peer device: {args[0]}"
                if len(args) >= 2:
                    comment += f", flags: {args[1]}"
                return [comment]
        if name == "hipCtxCreate":
            if len(args) >= 3:
                output = self.format_runtime_raw_output_target(raw_args[0])
                return [
                    f"// HIP context create: output: {output}, flags: {args[1]}, "
                    f"device: {args[2]}"
                ]
        if name == "hipCtxDestroy":
            if args:
                return [f"// HIP context destroy: {args[0]}"]
        if name == "hipCtxPopCurrent":
            if args:
                output = self.format_runtime_raw_output_target(raw_args[0])
                return [f"// HIP context pop current: output: {output}"]
        if name == "hipCtxPushCurrent":
            if args:
                return [f"// HIP context push current: {args[0]}"]
        if name == "hipCtxSetCurrent":
            if args:
                return [f"// HIP context set current: {args[0]}"]
        if name == "hipCtxGetCurrent":
            if args:
                output = self.format_runtime_raw_output_target(raw_args[0])
                return [f"// HIP context get current: output: {output}"]
        if name == "hipCtxGetDevice":
            if args:
                output = self.format_runtime_raw_output_target(raw_args[0])
                output_name = self.get_runtime_pointer_target_name(raw_args[0])
                if output_name is not None:
                    self.register_device_query_source(output_name, "context.device")
                return [f"// HIP context get device: output: {output}"]
        if name == "hipCtxGetApiVersion":
            if len(args) >= 2:
                output = self.format_runtime_raw_output_target(raw_args[1])
                output_name = self.get_runtime_pointer_target_name(raw_args[1])
                if output_name is not None:
                    self.register_device_query_source(
                        output_name, f"context.apiVersion({args[0]})"
                    )
                return [
                    f"// HIP context get API version: context: {args[0]}, "
                    f"output: {output}"
                ]
        if name in {"hipCtxGetCacheConfig", "hipCtxGetSharedMemConfig"}:
            if args:
                output = self.format_runtime_raw_output_target(raw_args[0])
                config_kind = "cache" if "Cache" in name else "shared memory"
                output_name = self.get_runtime_pointer_target_name(raw_args[0])
                if output_name is not None:
                    query_name = (
                        "context.cacheConfig"
                        if "Cache" in name
                        else "context.sharedMemConfig"
                    )
                    self.register_device_query_source(output_name, query_name)
                return [f"// HIP context get {config_kind} config: output: {output}"]
        if name in {"hipCtxSetCacheConfig", "hipCtxSetSharedMemConfig"}:
            if args:
                config_kind = "cache" if "Cache" in name else "shared memory"
                return [f"// HIP context set {config_kind} config: {args[0]}"]
        if name == "hipCtxGetFlags":
            if args:
                output = self.format_runtime_raw_output_target(raw_args[0])
                output_name = self.get_runtime_pointer_target_name(raw_args[0])
                if output_name is not None:
                    self.register_device_query_source(output_name, "context.flags")
                return [f"// HIP context get flags: output: {output}"]
        if name == "hipCtxSynchronize":
            return ["// HIP context synchronize"]
        if name == "hipDevicePrimaryCtxRetain":
            if len(args) >= 2:
                output = self.format_runtime_raw_output_target(raw_args[0])
                return [
                    f"// HIP primary context retain: output: {output}, "
                    f"device: {args[1]}"
                ]
        if name in {"hipDevicePrimaryCtxRelease", "hipDevicePrimaryCtxReset"}:
            if args:
                action = "release" if name == "hipDevicePrimaryCtxRelease" else "reset"
                return [f"// HIP primary context {action}: device: {args[0]}"]
        if name == "hipDevicePrimaryCtxSetFlags":
            if len(args) >= 2:
                return [
                    f"// HIP primary context set flags: device: {args[0]}, "
                    f"flags: {args[1]}"
                ]
        if name == "hipDevicePrimaryCtxGetState":
            if len(args) >= 3:
                flags_output = self.format_runtime_raw_output_target(raw_args[1])
                active_output = self.format_runtime_raw_output_target(raw_args[2])
                flags_output_name = self.get_runtime_pointer_target_name(raw_args[1])
                active_output_name = self.get_runtime_pointer_target_name(raw_args[2])
                if flags_output_name is not None:
                    self.register_device_query_source(
                        flags_output_name, "primaryContext.flags", args[0]
                    )
                if active_output_name is not None:
                    self.register_device_query_source(
                        active_output_name, "primaryContext.active", args[0]
                    )
                return [
                    f"// HIP primary context get state: device: {args[0]}, "
                    f"flags output: {flags_output}, active output: {active_output}"
                ]
        return None

    def format_hip_graph_runtime_call(self, name, raw_args, args):
        if name == "hipStreamBeginCaptureToGraph":
            if len(args) >= 6:
                return [
                    f"// HIP stream begin capture to graph: stream: {args[0]}, "
                    f"graph: {args[1]}, dependencies: {args[2]}, "
                    f"dependency data: {args[3]}, count: {args[4]}, mode: {args[5]}"
                ]
        if name == "hipThreadExchangeStreamCaptureMode":
            if args:
                output = self.format_runtime_raw_output_target(raw_args[0])
                return [f"// HIP exchange stream capture mode: output: {output}"]
        if name == "hipGraphCreate":
            if len(args) >= 2:
                output = self.format_runtime_raw_output_target(raw_args[0])
                return [f"// HIP graph create: output: {output}, flags: {args[1]}"]
        if name == "hipGraphDestroy":
            if args:
                return [f"// HIP graph destroy: {args[0]}"]
        if name == "hipGraphClone":
            if len(args) >= 2:
                output = self.format_runtime_raw_output_target(raw_args[0])
                return [f"// HIP graph clone: output: {output}, source: {args[1]}"]
        if name == "hipGraphAddNode":
            if len(args) >= 5:
                output = self.format_runtime_raw_output_target(raw_args[0])
                return [
                    f"// HIP graph add generic node: output: {output}, "
                    f"graph: {args[1]}, dependencies: {args[2]}, "
                    f"count: {args[3]}, params: {args[4]}"
                ]
        if name == "hipDrvGraphAddMemcpyNode":
            if len(args) >= 6:
                output = self.format_runtime_raw_output_target(raw_args[0])
                return [
                    f"// HIP driver graph add memcpy node: output: {output}, "
                    f"graph: {args[1]}, dependencies: {args[2]}, "
                    f"count: {args[3]}, params: {args[4]}, context: {args[5]}"
                ]
        if name == "hipDrvGraphAddMemsetNode":
            if len(args) >= 6:
                output = self.format_runtime_raw_output_target(raw_args[0])
                return [
                    f"// HIP driver graph add memset node: output: {output}, "
                    f"graph: {args[1]}, dependencies: {args[2]}, "
                    f"count: {args[3]}, params: {args[4]}, context: {args[5]}"
                ]
        if name == "hipDrvGraphAddMemFreeNode":
            if len(args) >= 5:
                output = self.format_runtime_raw_output_target(raw_args[0])
                return [
                    f"// HIP driver graph add memory free node: output: {output}, "
                    f"graph: {args[1]}, dependencies: {args[2]}, "
                    f"count: {args[3]}, pointer: {args[4]}"
                ]
        if name in {"hipDrvGraphMemcpyNodeGetParams", "hipDrvGraphMemcpyNodeSetParams"}:
            if len(args) >= 2:
                action = "get" if "GetParams" in name else "set"
                params = (
                    self.format_runtime_raw_output_target(raw_args[1])
                    if action == "get"
                    else args[1]
                )
                label = "params output" if action == "get" else "params"
                return [
                    f"// HIP driver graph memcpy node {action} params: "
                    f"node: {args[0]}, {label}: {params}"
                ]
        if name in {
            "hipDrvGraphExecMemcpyNodeSetParams",
            "hipDrvGraphExecMemsetNodeSetParams",
        }:
            if len(args) >= 4:
                node_kind = "memcpy" if "Memcpy" in name else "memset"
                return [
                    f"// HIP driver graph exec {node_kind} node set params: "
                    f"exec: {args[0]}, node: {args[1]}, params: {args[2]}, "
                    f"context: {args[3]}"
                ]
        if name in {
            "hipGraphAddEmptyNode",
            "hipGraphAddHostNode",
            "hipGraphAddKernelNode",
            "hipGraphAddMemcpyNode",
            "hipGraphAddMemsetNode",
        }:
            if len(args) >= 4:
                output = self.format_runtime_raw_output_target(raw_args[0])
                node_kind = {
                    "hipGraphAddEmptyNode": "empty",
                    "hipGraphAddHostNode": "host",
                    "hipGraphAddKernelNode": "kernel",
                    "hipGraphAddMemcpyNode": "memcpy",
                    "hipGraphAddMemsetNode": "memset",
                }[name]
                comment = (
                    f"// HIP graph add {node_kind} node: output: {output}, "
                    f"graph: {args[1]}, dependencies: {args[2]}, count: {args[3]}"
                )
                if len(args) >= 5:
                    comment += f", params: {args[4]}"
                return [comment]
        if name == "hipGraphAddChildGraphNode":
            if len(args) >= 5:
                output = self.format_runtime_raw_output_target(raw_args[0])
                return [
                    f"// HIP graph add child graph node: output: {output}, "
                    f"graph: {args[1]}, dependencies: {args[2]}, "
                    f"count: {args[3]}, child graph: {args[4]}"
                ]
        if name in {"hipGraphAddEventRecordNode", "hipGraphAddEventWaitNode"}:
            if len(args) >= 5:
                output = self.format_runtime_raw_output_target(raw_args[0])
                action = "record" if name == "hipGraphAddEventRecordNode" else "wait"
                return [
                    f"// HIP graph add event {action} node: output: {output}, "
                    f"graph: {args[1]}, dependencies: {args[2]}, "
                    f"count: {args[3]}, event: {args[4]}"
                ]
        if name in {
            "hipGraphAddMemAllocNode",
            "hipGraphAddMemFreeNode",
        }:
            if len(args) >= 5:
                output = self.format_runtime_raw_output_target(raw_args[0])
                node_kind = "alloc" if name == "hipGraphAddMemAllocNode" else "free"
                detail_label = "params" if node_kind == "alloc" else "pointer"
                return [
                    f"// HIP graph add memory {node_kind} node: output: {output}, "
                    f"graph: {args[1]}, dependencies: {args[2]}, "
                    f"count: {args[3]}, {detail_label}: {args[4]}"
                ]
        if name in {"hipGraphMemAllocNodeGetParams", "hipGraphMemFreeNodeGetParams"}:
            if len(args) >= 2:
                output = self.format_runtime_raw_output_target(raw_args[1])
                node_kind = "alloc" if "Alloc" in name else "free"
                detail_label = (
                    "params output" if node_kind == "alloc" else "pointer output"
                )
                return [
                    f"// HIP graph memory {node_kind} node get params: "
                    f"node: {args[0]}, {detail_label}: {output}"
                ]
        if name in {
            "hipGraphAddMemcpyNode1D",
            "hipGraphAddMemcpyNodeFromSymbol",
            "hipGraphAddMemcpyNodeToSymbol",
        }:
            if name == "hipGraphAddMemcpyNode1D" and len(args) >= 8:
                output = self.format_runtime_raw_output_target(raw_args[0])
                return [
                    f"// HIP graph add memcpy 1D node: output: {output}, "
                    f"graph: {args[1]}, dependencies: {args[2]}, count: {args[3]}, "
                    f"destination: {args[4]}, source: {args[5]}, bytes: {args[6]}, "
                    f"kind: {args[7]}"
                ]
            if len(args) >= 9:
                output = self.format_runtime_raw_output_target(raw_args[0])
                copy_kind = {
                    "hipGraphAddMemcpyNodeFromSymbol": "from symbol",
                    "hipGraphAddMemcpyNodeToSymbol": "to symbol",
                }[name]
                return [
                    f"// HIP graph add memcpy {copy_kind} node: output: {output}, "
                    f"graph: {args[1]}, dependencies: {args[2]}, count: {args[3]}, "
                    f"destination: {args[4]}, source: {args[5]}, bytes: {args[6]}, "
                    f"offset: {args[7]}, kind: {args[8]}"
                ]
        if name in {
            "hipGraphMemcpyNodeSetParams1D",
            "hipGraphMemcpyNodeSetParamsFromSymbol",
            "hipGraphMemcpyNodeSetParamsToSymbol",
        }:
            if name == "hipGraphMemcpyNodeSetParams1D" and len(args) >= 5:
                return [
                    f"// HIP graph memcpy 1D node set params: "
                    f"node: {args[0]}, destination: {args[1]}, source: {args[2]}, "
                    f"bytes: {args[3]}, kind: {args[4]}"
                ]
            if len(args) >= 6:
                copy_kind = {
                    "hipGraphMemcpyNodeSetParamsFromSymbol": "from symbol",
                    "hipGraphMemcpyNodeSetParamsToSymbol": "to symbol",
                }[name]
                return [
                    f"// HIP graph memcpy {copy_kind} node set params: "
                    f"node: {args[0]}, destination: {args[1]}, source: {args[2]}, "
                    f"bytes: {args[3]}, offset: {args[4]}, kind: {args[5]}"
                ]
        if name in {
            "hipGraphExecMemcpyNodeSetParams1D",
            "hipGraphExecMemcpyNodeSetParamsFromSymbol",
            "hipGraphExecMemcpyNodeSetParamsToSymbol",
        }:
            if name == "hipGraphExecMemcpyNodeSetParams1D" and len(args) >= 6:
                return [
                    f"// HIP graph exec memcpy 1D node set params: "
                    f"exec: {args[0]}, node: {args[1]}, destination: {args[2]}, "
                    f"source: {args[3]}, bytes: {args[4]}, kind: {args[5]}"
                ]
            if len(args) >= 7:
                copy_kind = {
                    "hipGraphExecMemcpyNodeSetParamsFromSymbol": "from symbol",
                    "hipGraphExecMemcpyNodeSetParamsToSymbol": "to symbol",
                }[name]
                return [
                    f"// HIP graph exec memcpy {copy_kind} node set params: "
                    f"exec: {args[0]}, node: {args[1]}, destination: {args[2]}, "
                    f"source: {args[3]}, bytes: {args[4]}, offset: {args[5]}, "
                    f"kind: {args[6]}"
                ]
        if name in {"hipGraphAddDependencies", "hipGraphRemoveDependencies"}:
            if len(args) >= 4:
                action = "add" if name == "hipGraphAddDependencies" else "remove"
                return [
                    f"// HIP graph {action} dependencies: graph: {args[0]}, "
                    f"from: {args[1]}, to: {args[2]}, count: {args[3]}"
                ]
        if name in {"hipGraphGetNodes", "hipGraphGetRootNodes"}:
            if len(args) >= 3:
                output = self.format_runtime_raw_output_target(raw_args[2])
                node_set = "nodes" if name == "hipGraphGetNodes" else "root nodes"
                return [
                    f"// HIP graph get {node_set}: graph: {args[0]}, "
                    f"nodes output: {args[1]}, count output: {output}"
                ]
        if name == "hipGraphGetEdges":
            if len(args) >= 4:
                output = self.format_runtime_raw_output_target(raw_args[3])
                return [
                    f"// HIP graph get edges: graph: {args[0]}, "
                    f"from output: {args[1]}, to output: {args[2]}, "
                    f"count output: {output}"
                ]
        if name in {"hipGraphNodeGetDependencies", "hipGraphNodeGetDependentNodes"}:
            if len(args) >= 3:
                output = self.format_runtime_raw_output_target(raw_args[2])
                node_set = (
                    "dependencies"
                    if name == "hipGraphNodeGetDependencies"
                    else "dependent nodes"
                )
                return [
                    f"// HIP graph node get {node_set}: node: {args[0]}, "
                    f"nodes output: {args[1]}, count output: {output}"
                ]
        if name == "hipGraphNodeFindInClone":
            if len(args) >= 3:
                output = self.format_runtime_raw_output_target(raw_args[0])
                return [
                    f"// HIP graph node find in clone: output: {output}, "
                    f"original: {args[1]}, clone graph: {args[2]}"
                ]
        if name == "hipGraphNodeGetType":
            if len(args) >= 2:
                output = self.format_runtime_raw_output_target(raw_args[1])
                return [
                    f"// HIP graph node get type: node: {args[0]}, output: {output}"
                ]
        if name == "hipGraphChildGraphNodeGetGraph":
            if len(args) >= 2:
                output = self.format_runtime_raw_output_target(raw_args[1])
                return [
                    f"// HIP graph child node get graph: "
                    f"node: {args[0]}, output: {output}"
                ]
        if name == "hipGraphKernelNodeCopyAttributes":
            if len(args) >= 2:
                return [
                    f"// HIP graph kernel node copy attributes: "
                    f"source: {args[0]}, destination: {args[1]}"
                ]
        if name in {"hipGraphKernelNodeGetAttribute", "hipGraphKernelNodeSetAttribute"}:
            if len(args) >= 3:
                action = "get" if name == "hipGraphKernelNodeGetAttribute" else "set"
                value = (
                    self.format_runtime_raw_output_target(raw_args[2])
                    if action == "get"
                    else self.format_runtime_pointer_target(raw_args[2])
                )
                label = "output" if action == "get" else "value"
                return [
                    f"// HIP graph kernel node {action} attribute: "
                    f"node: {args[0]}, attribute: {args[1]}, {label}: {value}"
                ]
        if name in {"hipGraphNodeSetEnabled", "hipGraphNodeGetEnabled"}:
            if len(args) >= 3:
                action = "get" if name == "hipGraphNodeGetEnabled" else "set"
                value = (
                    self.format_runtime_raw_output_target(raw_args[2])
                    if action == "get"
                    else args[2]
                )
                label = "output" if action == "get" else "value"
                return [
                    f"// HIP graph node {action} enabled: "
                    f"exec: {args[0]}, node: {args[1]}, {label}: {value}"
                ]
        if name == "hipGraphDestroyNode":
            if args:
                return [f"// HIP graph destroy node: {args[0]}"]
        if name == "hipGraphInstantiate":
            if len(args) >= 5:
                output = self.format_runtime_raw_output_target(raw_args[0])
                error_output = self.format_runtime_raw_output_target(raw_args[2])
                return [
                    f"// HIP graph instantiate: output: {output}, graph: {args[1]}, "
                    f"error node output: {error_output}, log buffer: {args[3]}, "
                    f"log bytes: {args[4]}"
                ]
        if name == "hipGraphInstantiateWithFlags":
            if len(args) >= 3:
                output = self.format_runtime_raw_output_target(raw_args[0])
                return [
                    f"// HIP graph instantiate with flags: output: {output}, "
                    f"graph: {args[1]}, flags: {args[2]}"
                ]
        if name == "hipGraphInstantiateWithParams":
            if len(args) >= 3:
                output = self.format_runtime_raw_output_target(raw_args[0])
                return [
                    f"// HIP graph instantiate with params: output: {output}, "
                    f"graph: {args[1]}, params: {args[2]}"
                ]
        if name == "hipGraphUpload":
            if len(args) >= 2:
                return [f"// HIP graph upload: exec: {args[0]}, stream: {args[1]}"]
        if name == "hipGraphLaunch":
            if len(args) >= 2:
                return [f"// HIP graph launch: exec: {args[0]}, stream: {args[1]}"]
        if name == "hipGraphExecDestroy":
            if args:
                return [f"// HIP graph exec destroy: {args[0]}"]
        if name == "hipGraphExecUpdate":
            if len(args) >= 4:
                error_output = self.format_runtime_raw_output_target(raw_args[2])
                result_output = self.format_runtime_raw_output_target(raw_args[3])
                return [
                    f"// HIP graph exec update: exec: {args[0]}, graph: {args[1]}, "
                    f"error node output: {error_output}, result output: {result_output}"
                ]
        if name == "hipGraphExecGetFlags":
            if len(args) >= 2:
                output = self.format_runtime_raw_output_target(raw_args[1])
                return [
                    f"// HIP graph exec get flags: exec: {args[0]}, output: {output}"
                ]
        if name in {"hipGraphNodeSetParams", "hipGraphExecNodeSetParams"}:
            if len(args) >= 2:
                if name == "hipGraphNodeSetParams":
                    return [
                        f"// HIP graph generic node set params: "
                        f"node: {args[0]}, params: {args[1]}"
                    ]
                if len(args) >= 3:
                    return [
                        f"// HIP graph exec generic node set params: "
                        f"exec: {args[0]}, node: {args[1]}, params: {args[2]}"
                    ]
        if name in {
            "hipGraphKernelNodeGetParams",
            "hipGraphKernelNodeSetParams",
            "hipGraphMemcpyNodeGetParams",
            "hipGraphMemcpyNodeSetParams",
            "hipGraphMemsetNodeGetParams",
            "hipGraphMemsetNodeSetParams",
            "hipGraphHostNodeGetParams",
            "hipGraphHostNodeSetParams",
        }:
            if len(args) >= 2:
                if "GetParams" in name:
                    self.clear_lvalue_metadata_source(raw_args[1])
                return [self.format_hip_graph_node_params_comment(name, args)]
        if name in {
            "hipGraphExecKernelNodeSetParams",
            "hipGraphExecMemcpyNodeSetParams",
            "hipGraphExecMemsetNodeSetParams",
            "hipGraphExecHostNodeSetParams",
            "hipGraphExecChildGraphNodeSetParams",
        }:
            if len(args) >= 3:
                node_kind = self.get_hip_graph_node_kind(name)
                return [
                    f"// HIP graph exec set {node_kind} node params: exec: {args[0]}, "
                    f"node: {args[1]}, params: {args[2]}"
                ]
        if name in {
            "hipGraphEventRecordNodeGetEvent",
            "hipGraphEventWaitNodeGetEvent",
        }:
            if len(args) >= 2:
                output = self.format_runtime_raw_output_target(raw_args[1])
                action = (
                    "record" if name == "hipGraphEventRecordNodeGetEvent" else "wait"
                )
                return [
                    f"// HIP graph event {action} node get event: "
                    f"node: {args[0]}, output: {output}"
                ]
        if name in {
            "hipGraphEventRecordNodeSetEvent",
            "hipGraphEventWaitNodeSetEvent",
        }:
            if len(args) >= 2:
                action = (
                    "record" if name == "hipGraphEventRecordNodeSetEvent" else "wait"
                )
                return [
                    f"// HIP graph event {action} node set event: "
                    f"node: {args[0]}, event: {args[1]}"
                ]
        if name in {
            "hipGraphExecEventRecordNodeSetEvent",
            "hipGraphExecEventWaitNodeSetEvent",
        }:
            if len(args) >= 3:
                action = (
                    "record"
                    if name == "hipGraphExecEventRecordNodeSetEvent"
                    else "wait"
                )
                return [
                    f"// HIP graph exec event {action} node set event: "
                    f"exec: {args[0]}, node: {args[1]}, event: {args[2]}"
                ]
        if name in {
            "hipGraphAddExternalSemaphoresSignalNode",
            "hipGraphAddExternalSemaphoresWaitNode",
        }:
            if len(args) >= 5:
                output = self.format_runtime_raw_output_target(raw_args[0])
                action = (
                    "signal"
                    if name == "hipGraphAddExternalSemaphoresSignalNode"
                    else "wait"
                )
                return [
                    f"// HIP graph add external semaphore {action} node: "
                    f"output: {output}, graph: {args[1]}, dependencies: {args[2]}, "
                    f"count: {args[3]}, params: {args[4]}"
                ]
        if name in {
            "hipGraphExternalSemaphoresSignalNodeGetParams",
            "hipGraphExternalSemaphoresSignalNodeSetParams",
            "hipGraphExternalSemaphoresWaitNodeGetParams",
            "hipGraphExternalSemaphoresWaitNodeSetParams",
        }:
            if len(args) >= 2:
                action = "signal" if "Signal" in name else "wait"
                direction = "get" if "GetParams" in name else "set"
                if direction == "get":
                    self.clear_lvalue_metadata_source(raw_args[1])
                return [
                    f"// HIP graph external semaphore {action} node {direction} params: "
                    f"node: {args[0]}, params: {args[1]}"
                ]
        if name in {
            "hipGraphExecExternalSemaphoresSignalNodeSetParams",
            "hipGraphExecExternalSemaphoresWaitNodeSetParams",
        }:
            if len(args) >= 3:
                action = "signal" if "Signal" in name else "wait"
                return [
                    f"// HIP graph exec external semaphore {action} node set params: "
                    f"exec: {args[0]}, node: {args[1]}, params: {args[2]}"
                ]
        if name in {"hipDeviceGetGraphMemAttribute", "hipDeviceSetGraphMemAttribute"}:
            if len(args) >= 3:
                action = "get" if name == "hipDeviceGetGraphMemAttribute" else "set"
                value = (
                    self.format_runtime_raw_output_target(raw_args[2])
                    if action == "get"
                    else self.format_runtime_pointer_target(raw_args[2])
                )
                label = "output" if action == "get" else "value"
                return [
                    f"// HIP device graph memory {action} attribute: "
                    f"device: {args[0]}, attribute: {args[1]}, {label}: {value}"
                ]
        if name == "hipDeviceGraphMemTrim":
            if args:
                return [f"// HIP device graph memory trim: device: {args[0]}"]
        if name == "hipGraphDebugDotPrint":
            if len(args) >= 3:
                return [
                    f"// HIP graph debug dot print: graph: {args[0]}, "
                    f"path: {args[1]}, flags: {args[2]}"
                ]
        if name == "hipUserObjectCreate":
            if len(args) >= 5:
                output = self.format_runtime_raw_output_target(raw_args[0])
                return [
                    f"// HIP user object create: output: {output}, "
                    f"resource: {args[1]}, destructor: {args[2]}, "
                    f"initial refcount: {args[3]}, flags: {args[4]}"
                ]
        if name in {"hipUserObjectRetain", "hipUserObjectRelease"}:
            if len(args) >= 2:
                action = "retain" if name == "hipUserObjectRetain" else "release"
                return [
                    f"// HIP user object {action}: object: {args[0]}, count: {args[1]}"
                ]
        if name in {"hipGraphRetainUserObject", "hipGraphReleaseUserObject"}:
            if len(args) >= 3:
                action = "retain" if name == "hipGraphRetainUserObject" else "release"
                comment = (
                    f"// HIP graph {action} user object: graph: {args[0]}, "
                    f"object: {args[1]}, count: {args[2]}"
                )
                if len(args) >= 4:
                    comment += f", flags: {args[3]}"
                return [comment]
        return None

    def format_hip_graph_node_params_comment(self, name, args):
        node_kind = self.get_hip_graph_node_kind(name)
        action = "get" if "GetParams" in name else "set"
        return (
            f"// HIP graph {node_kind} node {action} params: "
            f"node: {args[0]}, params: {args[1]}"
        )

    def get_hip_graph_node_kind(self, name):
        if "Kernel" in name:
            return "kernel"
        if "Memcpy" in name:
            return "memcpy"
        if "Memset" in name:
            return "memset"
        if "Host" in name:
            return "host"
        if "ChildGraph" in name:
            return "child graph"
        return "unknown"

    def format_hip_runtime_expression_call(self, node, args):
        name = node.name
        if name == "hipGetErrorString" and args:
            return f'/* HIP error string: {args[0]} */ ""'
        if name == "hipGetErrorName" and args:
            return f'/* HIP error name: {args[0]} */ ""'
        if name == "hiprtcGetErrorString" and args:
            return f'/* HIPRTC error string: {args[0]} */ ""'
        if name == "hipApiName" and args:
            return f'/* HIP API name: {args[0]} */ ""'
        if name == "hipKernelNameRef" and args:
            return f'/* HIP kernel name for function: {args[0]} */ ""'
        if name == "hipKernelNameRefByPtr" and len(args) >= 2:
            return (
                f"/* HIP kernel name for host function: {args[0]}, "
                f'stream: {args[1]} */ ""'
            )
        if name == "hipGetStreamDeviceId" and args:
            return f"/* HIP stream device id: {args[0]} */ 0"

        return self.format_hip_runtime_status_inline_expression(node)

    def format_hip_runtime_status_inline_expression(self, node):
        runtime_status = self.format_hip_runtime_status_expression(node)
        if runtime_status is None:
            return None

        comments, value = runtime_status
        detail = "; ".join(
            self.format_runtime_expression_comment_text(comment) for comment in comments
        )
        return f"(/* {detail} */ {value})"

    def format_runtime_expression_comment_text(self, comment):
        text = str(comment).strip()
        if text.startswith("// "):
            text = text[3:]
        elif text.startswith("//"):
            text = text[2:].lstrip()
        return text.replace("*/", "* /")

    def format_runtime_pointer_target(self, arg):
        if isinstance(arg, CastNode):
            return self.format_runtime_pointer_target(arg.expression)
        if isinstance(arg, UnaryOpNode) and arg.op == "&":
            return self.visit_lvalue_expression(arg.operand)
        return self.visit(arg)

    def format_runtime_raw_output_target(self, arg):
        self.clear_lvalue_metadata_source(arg)
        return self.format_runtime_pointer_target(arg)

    def get_runtime_pointer_target_name(self, arg):
        if isinstance(arg, CastNode):
            return self.get_runtime_pointer_target_name(arg.expression)
        if isinstance(arg, UnaryOpNode) and arg.op == "&":
            return self.get_runtime_pointer_target_name(arg.operand)
        if isinstance(arg, str):
            return arg
        return None

    def format_statement_fragment(self, stmt):
        if stmt is None:
            return ""
        if isinstance(stmt, list):
            return ", ".join(self.format_statement_fragment(item) for item in stmt)
        if isinstance(stmt, VariableNode):
            var_type = self.convert_hip_variable_type_to_crossgl(
                getattr(stmt, "vtype", "int"), stmt.name
            )
            self.register_vector1_name(stmt.name, getattr(stmt, "vtype", "int"))
            self.register_variable_type(stmt.name, var_type)
            if hasattr(stmt, "value") and stmt.value:
                value = self.format_variable_initializer_value(stmt.value)
                return f"var {stmt.name}: {var_type} = {value}"
            return f"var {stmt.name}: {var_type}"
        if isinstance(stmt, AssignmentNode):
            left = self.visit_lvalue_expression(stmt.left)
            right = self.visit(stmt.right)
            operator = getattr(stmt, "operator", "=")
            self.clear_lvalue_metadata_source(stmt.left)
            return f"{left} {operator} {right}"

        result = self.visit(stmt)
        return result if isinstance(result, str) else ""

    def visit_HipProgramNode(self, node):
        """Render a HIP program AST as a CrossGL shader block."""
        self.emit("// HIP to CrossGL conversion")

        for stmt in node.statements:
            if isinstance(stmt, FunctionNode):
                if self.is_generated_matrix_helper_function(stmt):
                    continue
                if hasattr(stmt, "qualifiers") and "__global__" in getattr(
                    stmt, "qualifiers", []
                ):
                    self.emit(f"// Kernel: {stmt.name}")
                    self.visit_kernel_as_compute_shader(stmt)
                else:
                    self.emit(f"// Function: {stmt.name}")
                    self.visit(stmt)
                self.emit("")
            elif isinstance(stmt, StructNode):
                if self.is_generated_matrix_helper_struct(stmt):
                    continue
                self.visit(stmt)
                self.emit("")
            elif isinstance(stmt, EnumNode):
                self.visit(stmt)
                self.emit("")
            elif isinstance(stmt, VariableNode):
                self.visit(stmt)
                self.emit("")
            elif isinstance(stmt, TypeAliasNode):
                self.visit(stmt)
                self.emit("")
            else:
                self.visit(stmt)

    def format_preprocessor_content(self, content):
        text = str(content).strip()
        compact = text.replace(" ", "")
        if compact.startswith("<") and compact.endswith(">"):
            return compact
        return text

    def visit_PreprocessorNode(self, node):
        content = self.format_preprocessor_content(node.content)
        if node.directive == "include":
            if "hip_runtime.h" in content:
                self.emit("// HIP runtime functionality built-in")
            elif "hip/hip_runtime_api.h" in content:
                self.emit("// HIP runtime API functionality built-in")
            else:
                self.emit(f"// include {content}".strip())
        elif content:
            self.emit(f"// {node.directive} {content}")
        else:
            self.emit(f"// {node.directive}")

    def visit_FunctionNode(self, node):
        """Render a HIP function node as a CrossGL function."""
        return_type = self.convert_hip_type_to_crossgl(
            node.return_type if hasattr(node, "return_type") else "void"
        )

        self.push_resource_object_hint_scope(
            self.collect_resource_object_type_hints(
                node, self.collect_declared_variable_names(node)
            )
        )
        self.push_variable_type_scope()
        self.push_identifier_name_scope()
        try:
            params = []

            if hasattr(node, "params") and node.params:
                for index, param in enumerate(node.params):
                    param_name = (
                        param.get("name", "param")
                        if isinstance(param, dict)
                        else getattr(param, "name", "param")
                    ) or f"_param{index}"
                    raw_type = (
                        param.get("type", "int")
                        if isinstance(param, dict)
                        else getattr(param, "vtype", "int")
                    )
                    param_type = self.convert_hip_variable_type_to_crossgl(
                        raw_type, param_name
                    )
                    output_name = self.register_identifier_name(param_name)
                    self.register_vector1_name(param_name, raw_type)
                    self.register_variable_type(param_name, param_type)
                    params.append(f"{param_type} {output_name}")

            param_str = ", ".join(params)
            function_name = self.format_function_declaration_name(node.name)
            self.emit(f"{return_type} {function_name}({param_str}) {{")

            self.indent_level += 1
            self.push_packed_argument_scope()
            self.push_type_alias_scope()
            self.push_unique_ptr_scope()
            self.push_cooperative_group_scope()
            if hasattr(node, "params") and node.params:
                for param in node.params:
                    self.register_unique_ptr_parameter(param)
                    self.register_cooperative_group_parameter(param)
            if hasattr(node, "body") and node.body:
                try:
                    if isinstance(node.body, list):
                        for stmt in node.body:
                            self.emit_statement(stmt)
                    else:
                        self.emit_statement(node.body)
                finally:
                    self.pop_cooperative_group_scope()
                    self.pop_unique_ptr_scope()
                    self.pop_type_alias_scope()
                    self.pop_packed_argument_scope()
                    self.indent_level -= 1
            else:
                self.pop_cooperative_group_scope()
                self.pop_unique_ptr_scope()
                self.pop_type_alias_scope()
                self.pop_packed_argument_scope()
                self.indent_level -= 1
        finally:
            self.pop_identifier_name_scope()
            self.pop_variable_type_scope()
            self.pop_resource_object_hint_scope()

        self.emit("}")

    def visit_kernel_as_compute_shader(self, kernel):
        """Render a HIP kernel as a CrossGL compute shader block."""
        for attribute in getattr(kernel, "attributes", []) or []:
            attribute_text = str(attribute)
            if attribute_text.startswith("__launch_bounds__"):
                bounds = attribute_text[len("__launch_bounds__") :]
                self.emit(f"// HIP launch bounds: {bounds}")
                continue

            flat_work_group_size = re.search(
                r"amdgpu_flat_work_group_size\((.*?)\)", attribute_text
            )
            if flat_work_group_size:
                bounds = re.sub(r"\s*,\s*", ", ", flat_work_group_size.group(1))
                self.emit(f"// HIP AMDGPU flat work group size: ({bounds})")

        self.emit("@compute")
        self.emit("@workgroup_size(1, 1, 1)  // Default workgroup size")

        params = []
        self.push_resource_object_hint_scope(
            self.collect_resource_object_type_hints(
                kernel, self.collect_declared_variable_names(kernel)
            )
        )
        self.push_variable_type_scope()
        self.push_identifier_name_scope()
        try:
            if hasattr(kernel, "params") and kernel.params:
                for index, param in enumerate(kernel.params):
                    if isinstance(param, dict):
                        raw_type = param.get("type", "int")
                        param_name = param.get("name", "param")
                    else:
                        raw_type = getattr(param, "vtype", "int")
                        param_name = getattr(param, "name", "param")
                    param_name = param_name or f"_param{index}"

                    if "*" in raw_type:
                        element_type = self.convert_hip_pointer_element_type(raw_type)
                        output_name = self.register_identifier_name(param_name)
                        self.register_variable_type(
                            param_name, f"array<{element_type}>"
                        )
                        params.append(
                            f"@group(0) @binding({len(params)}) var<storage, read_write> {output_name}: array<{element_type}>"
                        )
                    else:
                        param_type = self.convert_hip_variable_type_to_crossgl(
                            raw_type, param_name
                        )
                        output_name = self.register_identifier_name(param_name)
                        self.register_vector1_name(param_name, raw_type)
                        self.register_variable_type(param_name, param_type)
                        params.append(f"{param_type} {output_name}")

            kernel_name = self.format_function_declaration_name(kernel.name)
            self.emit(f"fn {kernel_name}(")
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

            if hasattr(kernel, "body") and kernel.body:
                self.push_packed_argument_scope()
                self.push_type_alias_scope()
                self.push_unique_ptr_scope()
                self.push_cooperative_group_scope()
                if hasattr(kernel, "params") and kernel.params:
                    for param in kernel.params:
                        self.register_unique_ptr_parameter(param)
                        self.register_cooperative_group_parameter(param)
                try:
                    if isinstance(kernel.body, list):
                        for stmt in kernel.body:
                            self.emit_statement(stmt)
                    else:
                        self.emit_statement(kernel.body)
                finally:
                    self.pop_cooperative_group_scope()
                    self.pop_unique_ptr_scope()
                    self.pop_type_alias_scope()
                    self.pop_packed_argument_scope()

            self.indent_level -= 1
            self.emit("}")
        finally:
            self.pop_identifier_name_scope()
            self.pop_variable_type_scope()
            self.pop_resource_object_hint_scope()

    def visit_StructNode(self, node):
        if getattr(node, "is_union", False):
            name = node.name or "anonymous"
            self.emit(
                f"// HIP union {name} represented as struct-like layout; "
                "overlapping storage is not modeled"
            )
            if not node.name:
                return

        struct_name = self.convert_hip_record_name_to_crossgl(node.name)
        self.emit(f"struct {struct_name} {{")
        self.indent_level += 1

        if hasattr(node, "members") and node.members:
            for member in node.members:
                if isinstance(member, VariableNode):
                    member_type = self.convert_hip_type_to_crossgl(
                        getattr(member, "vtype", "int")
                    )
                    self.emit(f"{member_type} {member.name};")

        self.indent_level -= 1
        self.emit("};")

    def visit_VariableNode(self, node):
        raw_name = node.name
        output_name = self.register_variable_declaration_name(raw_name)
        cooperative_group = self.cooperative_group_declaration_metadata(node)
        if cooperative_group is not None:
            group_kind = cooperative_group["kind"]
            self.register_cooperative_group_name(raw_name, cooperative_group)
            if group_kind == "thread_block":
                self.emit(
                    f"// cooperative_groups thread_block {output_name} maps to the current workgroup"
                )
            elif (
                group_kind == "thread_block_tile"
                and cooperative_group.get("tile_size")
                and cooperative_group.get("parent_kind") == "thread_block"
            ):
                self.emit(
                    f"// cooperative_groups thread_block_tile<{cooperative_group['tile_size']}> "
                    f"{output_name} maps to a tiled partition of the current workgroup"
                )
            else:
                self.emit(
                    f"// cooperative_groups {group_kind} for {output_name} not directly supported in CrossGL"
                )
            return

        var_type = self.convert_hip_variable_type_to_crossgl(
            getattr(node, "vtype", "int"), raw_name
        )
        qualifiers = set(getattr(node, "qualifiers", []) or [])

        self.register_vector1_name(raw_name, getattr(node, "vtype", "int"))
        self.register_packed_argument_list(node)
        self.register_unique_ptr_name(raw_name, getattr(node, "vtype", "int"))
        self.register_variable_type(raw_name, var_type)
        if output_name != raw_name:
            self.register_unique_ptr_name(output_name, getattr(node, "vtype", "int"))
            self.register_variable_type(output_name, var_type)
        if "__shared__" in qualifiers:
            if getattr(node, "is_dynamic_shared_memory", False):
                self.emit(
                    f"// HIP dynamic shared memory: {output_name} uses launch-time "
                    "shared memory size"
                )
            self.emit(f"var<workgroup> {output_name}: {var_type};")
            return

        if "__constant__" in qualifiers:
            if hasattr(node, "value") and node.value:
                value = self.format_variable_initializer_value(node.value)
                self.emit(
                    f"@group(0) @binding(0) var<uniform> {output_name}: "
                    f"{var_type} = {value};"
                )
            else:
                self.emit(
                    f"@group(0) @binding(0) var<uniform> {output_name}: {var_type};"
                )
            return

        if "__managed__" in qualifiers:
            self.emit(f"// HIP managed memory: {output_name}")

        dim3_default_value = self.format_hip_dim3_default_initializer(node)
        if dim3_default_value is not None:
            self.emit(f"var {output_name}: {var_type} = {dim3_default_value};")
            return

        if hasattr(node, "value") and node.value:
            dim3_brace_value = self.format_hip_dim3_brace_initializer(node)
            if dim3_brace_value is not None:
                self.emit(f"var {output_name}: {var_type} = {dim3_brace_value};")
                return

            runtime_status = self.format_hip_runtime_status_expression(node.value)
            if runtime_status is not None:
                comments, value = runtime_status
                for comment in comments:
                    self.emit(comment)
                self.emit(f"var {output_name}: {var_type} = {value};")
                return

            value = self.format_variable_initializer_value(node.value)
            self.emit(f"var {output_name}: {var_type} = {value};")
        else:
            self.emit(f"var {output_name}: {var_type};")

    def format_variable_initializer_value(self, value):
        string_value = self.format_single_string_initializer(value)
        if string_value is not None:
            return string_value
        scalar_value = self.format_single_scalar_initializer(value)
        if scalar_value is not None:
            return scalar_value
        return self.visit(value)

    def format_single_scalar_initializer(self, value):
        if not isinstance(value, InitializerListNode) or len(value.elements) != 1:
            return None
        initializer = value.elements[0]
        if isinstance(initializer, DesignatedInitializerNode):
            return None
        if not isinstance(initializer, FunctionCallNode):
            return None
        if not isinstance(initializer.name, str) or not initializer.name.startswith(
            "::"
        ):
            return None
        return self.visit(initializer)

    def format_single_string_initializer(self, value):
        if not isinstance(value, InitializerListNode) or len(value.elements) != 1:
            return None

        element = value.elements[0]
        if not isinstance(element, str):
            return None

        return self.format_cpp_string_literal_expression(element)

    def visit_TextureAccessNode(self, node):
        texture_name = self.visit(node.texture_name)
        coordinates = node.coordinates
        if isinstance(coordinates, (list, tuple)):
            rendered_coordinates = [
                self.visit(coordinate) for coordinate in coordinates
            ]
            if len(rendered_coordinates) == 1:
                coordinate = rendered_coordinates[0]
            else:
                coordinate = self.format_vector_constructor(
                    f"vec{len(rendered_coordinates)}", rendered_coordinates
                )
        else:
            coordinate = self.visit(coordinates)
        return f"texture({texture_name}, {coordinate})"

    def visit_SharedMemoryNode(self, node):
        var_type = self.convert_hip_variable_type_to_crossgl(node.vtype, node.name)
        output_name = self.register_identifier_name(node.name)
        self.register_variable_type(node.name, var_type)
        if getattr(node, "is_dynamic_shared_memory", False):
            self.emit(
                f"// HIP dynamic shared memory: {output_name} uses launch-time "
                "shared memory size"
            )
        if node.size is not None:
            size = self.format_crossgl_array_extent(self.visit(node.size))
            self.emit(f"var<workgroup> {output_name}: array<{var_type}, {size}>;")
        else:
            self.emit(f"var<workgroup> {output_name}: {var_type};")

    def visit_ConstantMemoryNode(self, node):
        var_type = self.convert_hip_variable_type_to_crossgl(node.vtype, node.name)
        output_name = self.register_identifier_name(node.name)
        self.register_variable_type(node.name, var_type)
        if node.value is not None:
            value = self.visit(node.value)
            self.emit(
                f"@group(0) @binding(0) var<uniform> {output_name}: "
                f"{var_type} = {value};"
            )
        else:
            self.emit(f"@group(0) @binding(0) var<uniform> {output_name}: {var_type};")

    def visit_HipErrorHandlingNode(self, node):
        error_type = self.visit(node.error_type)
        error_expr = self.visit(node.error_expr)
        runtime_expression = self.format_hip_runtime_expression_call(
            FunctionCallNode(error_type, [node.error_expr]), [error_expr]
        )
        if runtime_expression is not None:
            return runtime_expression
        return (
            f"(/* HIP error status: {error_type}, expr: {error_expr} */ {error_expr})"
        )

    def visit_HipDevicePropertyNode(self, node):
        property_name = self.visit(node.property_name)
        if node.device_id is None:
            return f"(/* HIP device property: {property_name} */ 0)"
        device_id = self.visit(node.device_id)
        return f"(/* HIP device property: {property_name}, device: {device_id} */ 0)"

    def visit_KernelLaunchNode(self, node):
        kernel_name = (
            node.kernel_name
            if isinstance(node.kernel_name, str)
            else self.visit(node.kernel_name)
        )
        config = [self.visit(node.blocks), self.visit(node.threads)]
        if node.shared_mem is not None:
            config.append(self.visit(node.shared_mem))
        if node.stream is not None:
            config.append(self.visit(node.stream))

        self.emit(f"// Kernel launch: {kernel_name}<<<{', '.join(config)}>>>()")
        if node.args:
            args = self.resolve_packed_launch_args(node.args)
            args_str = ", ".join([self.format_kernel_launch_arg(arg) for arg in args])
            self.emit(f"// Arguments: {args_str}")

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

    def register_cooperative_group_name(self, name, group_metadata):
        if name:
            self.cooperative_group_scopes[-1][name] = group_metadata

    def register_cooperative_group_parameter(self, param):
        if isinstance(param, dict):
            name = param.get("name", "")
            type_name = param.get("type", "")
        else:
            name = getattr(param, "name", "")
            type_name = getattr(param, "vtype", "")

        metadata = self.cooperative_group_metadata_from_type(type_name)
        if metadata is not None:
            self.register_cooperative_group_name(name, metadata)

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

    def register_type_alias(self, name, alias_type):
        self.type_alias_scopes[-1][name] = alias_type

    def resolve_type_alias(self, type_name):
        type_name = self.strip_type_qualifiers(type_name)
        for scope in reversed(self.type_alias_scopes):
            if type_name in scope:
                return scope[type_name]
        return type_name

    def register_unique_ptr_parameter(self, param):
        if isinstance(param, dict):
            self.register_unique_ptr_name(param.get("name", ""), param.get("type", ""))
        else:
            self.register_unique_ptr_name(
                getattr(param, "name", ""), getattr(param, "vtype", "")
            )

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

    def visit_AssignmentNode(self, node):
        left = self.visit_lvalue_expression(node.left)
        operator = getattr(node, "operator", "=")
        runtime_status = (
            self.format_hip_runtime_status_expression(node.right)
            if operator == "="
            else None
        )
        if runtime_status is not None:
            comments, value = runtime_status
            for comment in comments:
                self.emit(comment)
            self.clear_lvalue_metadata_source(node.left)
            self.emit(f"{left} = {value};")
            return

        right = self.visit(node.right)
        self.clear_lvalue_metadata_source(node.left)
        return f"{left} {operator} {right}"

    def visit_BinaryOpNode(self, node):
        left = self.visit(node.left)
        right = self.visit(node.right)
        return f"({left} {node.op} {right})"

    def visit_UnaryOpNode(self, node):
        mutates_operand = node.op in {"++", "--"} or (
            isinstance(node.op, str) and node.op.endswith("_POST")
        )
        if node.op == "&" or mutates_operand:
            operand = self.visit_lvalue_expression(node.operand)
        else:
            operand = self.visit(node.operand)
        if mutates_operand:
            self.clear_lvalue_metadata_source(node.operand)
        if isinstance(node.op, str) and node.op.endswith("_POST"):
            return f"({operand}{node.op[:-5]})"
        elif hasattr(node, "postfix") and node.postfix:
            return f"({operand}{node.op})"
        else:
            return f"({node.op}{operand})"

    def visit_FunctionCallNode(self, node):
        if self.is_get_method_call(node):
            return self.visit(node.name.object)

        if hasattr(node, "name"):
            func_name = node.name
        else:
            func_name = str(node.function) if hasattr(node, "function") else "unknown"

        if not isinstance(func_name, str):
            func_name = self.visit(func_name)

        if func_name == "lambda":
            return self.format_lambda_call(getattr(node, "args", []))

        raw_args = []
        if hasattr(node, "args") and node.args:
            raw_args = node.args
        elif hasattr(node, "arguments") and node.arguments:
            raw_args = node.arguments

        if func_name == "__hip_pack_expand__":
            if raw_args:
                return f"{self.visit(raw_args[0])}..."
            return "..."

        cooperative_call = self.format_cooperative_group_call(node)
        if cooperative_call is not None:
            return cooperative_call

        args = [self.visit(arg) for arg in raw_args]
        args_str = ", ".join(args)

        make_unique = self.format_make_unique_call(func_name, args)
        if make_unique is not None:
            return make_unique

        unique_ptr_init = self.format_unique_ptr_constructor_call(func_name, args)
        if unique_ptr_init is not None:
            return unique_ptr_init

        numeric_limits_call = self.format_std_numeric_limits_call(func_name, args)
        if numeric_limits_call is not None:
            return numeric_limits_call

        hip_intrinsic = self.format_hip_intrinsic_call(func_name, args)
        if hip_intrinsic is not None:
            return hip_intrinsic

        user_defined_call_name = self.format_user_defined_function_call_name(func_name)
        if user_defined_call_name is not None:
            return f"{user_defined_call_name}({args_str})"

        timer_intrinsic = self.format_hip_timer_intrinsic_call(func_name, args)
        if timer_intrinsic is not None:
            return timer_intrinsic

        load_cache_intrinsic = self.format_hip_load_cache_intrinsic_call(
            func_name, raw_args, args
        )
        if load_cache_intrinsic is not None:
            return load_cache_intrinsic

        runtime_expression = self.format_hip_runtime_expression_call(node, args)
        if runtime_expression is not None:
            return runtime_expression

        resource_call = self.format_hip_resource_call(func_name, args, raw_args)
        if resource_call is not None:
            return resource_call

        warp_intrinsic = self.format_hip_warp_intrinsic_call(func_name, args)
        if warp_intrinsic is not None:
            return warp_intrinsic

        dim3_constructor = self.format_hip_dim3_constructor_call(func_name, args)
        if dim3_constructor is not None:
            return dim3_constructor

        # Convert HIP built-in functions
        crossgl_func = self.convert_hip_builtin_function(func_name)
        if (
            crossgl_func == func_name
            and isinstance(func_name, str)
            and func_name.startswith("::")
            and not self.is_simple_identifier(crossgl_func)
        ):
            crossgl_func = self.format_function_declaration_name(func_name)
        return f"{crossgl_func}({args_str})"

    def format_std_numeric_limits_call(self, function_name, args):
        parsed = self.parse_cpp_template_static_member(function_name)
        if parsed is None:
            return None

        base_name, template_args, member_name = parsed
        if base_name not in {"std::numeric_limits", "cuda::std::numeric_limits"}:
            return None
        if len(template_args) != 1 or not re.fullmatch(
            r"[A-Za-z_][A-Za-z0-9_]*", member_name
        ):
            return None

        value_type = self.convert_hip_type_to_crossgl(template_args[0])
        args_str = ", ".join(args)
        return f"numeric_limits_{member_name}<{value_type}>({args_str})"

    def parse_cpp_template_static_member(self, text):
        if not isinstance(text, str):
            return None

        normalized = text[2:] if text.startswith("::") else text
        scope_index = self.find_last_scope_operator_outside_templates(normalized)
        if scope_index is None:
            return None

        template_name = normalized[:scope_index]
        member_name = normalized[scope_index + 2 :]
        if not member_name:
            return None

        base_name, template_args = self.parse_cpp_template(template_name)
        if not template_args:
            return None
        return base_name, template_args, member_name

    def find_last_scope_operator_outside_templates(self, text):
        depth = 0
        last_scope = None
        index = 0
        while index + 1 < len(text):
            char = text[index]
            if char == "<":
                depth += 1
            elif char == ">":
                depth = max(0, depth - 1)
            elif char == ":" and text[index + 1] == ":" and depth == 0:
                last_scope = index
                index += 1
            index += 1
        return last_scope

    def format_hip_intrinsic_call(self, function_name, args):
        fp16_intrinsic = self.format_hip_fp16_intrinsic_call(function_name, args)
        if fp16_intrinsic is not None:
            return fp16_intrinsic

        complex_intrinsic = self.format_hip_complex_intrinsic_call(function_name, args)
        if complex_intrinsic is not None:
            return complex_intrinsic

        type_cast_intrinsic = self.format_hip_type_cast_intrinsic_call(
            function_name, args
        )
        if type_cast_intrinsic is not None:
            return type_cast_intrinsic

        integer_intrinsic = self.format_hip_integer_intrinsic_call(function_name, args)
        if integer_intrinsic is not None:
            return integer_intrinsic

        float_intrinsic = self.format_hip_float_intrinsic_call(function_name, args)
        if float_intrinsic is not None:
            return float_intrinsic

        sync_vote_intrinsic = self.format_hip_sync_vote_intrinsic_call(
            function_name, args
        )
        if sync_vote_intrinsic is not None:
            return sync_vote_intrinsic

        return None

    def format_hip_timer_intrinsic_call(self, function_name, args):
        if isinstance(function_name, str) and function_name.startswith("::"):
            function_name = function_name[2:]

        if function_name not in {"clock", "clock64", "wall_clock64"}:
            return None

        args_text = ", ".join(args)
        return (
            f"(/* HIP device timer {function_name}({args_text}) "
            "not directly supported in CrossGL */ 0)"
        )

    def format_hip_load_cache_intrinsic_call(
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

        args_text = ", ".join(formatted_args)
        return (
            f"(/* hip load cache intrinsic {function_name}({args_text}) "
            "not directly supported in CrossGL */ 0)"
        )

    def format_hip_warp_intrinsic_call(self, function_name, args):
        if function_name in {"__activemask", "__ballot_sync"}:
            return self.format_unsupported_hip_warp_intrinsic(function_name, args)

        if function_name in {"__any", "__all"}:
            if len(args) != 1:
                return self.format_unsupported_hip_warp_intrinsic(function_name, args)
            predicate = self.format_wave_predicate(args[0])
            if function_name == "__any":
                return f"(WaveActiveAnyTrue({predicate}) ? 1 : 0)"
            return f"(WaveActiveAllTrue({predicate}) ? 1 : 0)"

        if function_name in {"__any_sync", "__all_sync"}:
            if len(args) != 2 or not self.is_full_or_active_warp_mask(args[0]):
                return self.format_unsupported_hip_warp_intrinsic(function_name, args)
            predicate = self.format_wave_predicate(args[1])
            if function_name == "__any_sync":
                return f"(WaveActiveAnyTrue({predicate}) ? 1 : 0)"
            return f"(WaveActiveAllTrue({predicate}) ? 1 : 0)"

        if function_name == "__shfl_sync":
            if (
                len(args) not in {3, 4}
                or not self.is_full_or_active_warp_mask(args[0])
                or self.hip_warp_shuffle_width_limit(args[3:] or None) is None
            ):
                return self.format_unsupported_hip_warp_intrinsic(function_name, args)
            return f"WaveReadLaneAt({args[1]}, {args[2]})"

        if function_name in {"__shfl_up_sync", "__shfl_down_sync"}:
            width_limit = self.hip_warp_shuffle_width_limit(args[3:] or None)
            if (
                len(args) not in {3, 4}
                or not self.is_full_or_active_warp_mask(args[0])
                or width_limit is None
            ):
                return self.format_unsupported_hip_warp_intrinsic(function_name, args)
            value, delta = args[1], args[2]
            if function_name == "__shfl_up_sync":
                return (
                    f"((WaveGetLaneIndex() >= ({delta})) ? "
                    f"WaveReadLaneAt({value}, (WaveGetLaneIndex() - ({delta}))) "
                    f": {value})"
                )
            return (
                f"(((WaveGetLaneIndex() + ({delta})) < {width_limit}) ? "
                f"WaveReadLaneAt({value}, (WaveGetLaneIndex() + ({delta}))) "
                f": {value})"
            )

        if function_name == "__shfl_xor_sync":
            if (
                len(args) not in {3, 4}
                or not self.is_full_or_active_warp_mask(args[0])
                or self.hip_warp_shuffle_width_limit(args[3:] or None) is None
            ):
                return self.format_unsupported_hip_warp_intrinsic(function_name, args)
            return f"WaveReadLaneAt({args[1]}, " f"(WaveGetLaneIndex() ^ {args[2]}))"

        if function_name in {
            "__ballot",
            "__shfl",
            "__shfl_up",
            "__shfl_down",
            "__shfl_xor",
        }:
            return self.format_unsupported_hip_warp_intrinsic(function_name, args)

        return None

    def is_full_warp_shuffle_width(self, width_args):
        return self.hip_warp_shuffle_width_limit(width_args) is not None

    def hip_warp_shuffle_width_limit(self, width_args):
        if not width_args:
            return "WaveGetLaneCount()"
        if len(width_args) != 1:
            return None
        normalized = self.normalize_warp_mask_expression(width_args[0])
        if normalized in {"32", "32u", "64", "64u"}:
            return normalized.rstrip("u")
        if normalized in {"warpsize", "warp_size", "warp_full_width"}:
            return "WaveGetLaneCount()"
        return None

    def is_full_or_active_warp_mask(self, mask):
        normalized = self.normalize_warp_mask_expression(mask)
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

    def format_unsupported_hip_warp_intrinsic(self, function_name, args):
        args_text = ", ".join(args)
        return (
            f"(/* hip warp intrinsic {function_name}({args_text}) "
            "not directly supported in CrossGL */ 0)"
        )

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
            if member_name in {"size", "num_threads"} and tile_size:
                return tile_size
            if (
                member_name == "thread_rank"
                and tile_size
                and group_metadata.get("parent_kind") == "thread_block"
            ):
                return f"(gl_LocalInvocationIndex % {tile_size})"

        if group_kind == "thread_block_tile" and member_name in {
            "shfl",
            "shfl_down",
            "shfl_up",
            "shfl_xor",
        }:
            return self.format_cooperative_group_tile_shuffle(
                group_metadata, member_name, args
            )

        if member_name in {"thread_rank", "size", "num_threads"}:
            return self.format_unsupported_cooperative_group_expression(
                group_kind, member_name
            )
        if member_name in {"thread_index", "dim_threads"}:
            return self.format_unsupported_cooperative_group_expression(
                group_kind, member_name, "vec3<u32>(0, 0, 0)"
            )
        return (
            f"// cooperative_groups {group_kind}.{member_name} "
            "not directly supported in CrossGL"
        )

    def format_cooperative_group_tile_shuffle(
        self, group_metadata, member_name, raw_args
    ):
        args = [self.visit(arg) for arg in raw_args]
        if len(args) != 2:
            return self.format_unsupported_cooperative_group_expression(
                group_metadata["kind"], member_name
            )

        value, lane_arg = args
        tile_size = group_metadata.get("tile_size")
        if not tile_size:
            return self.format_unsupported_cooperative_group_expression(
                group_metadata["kind"], member_name
            )

        lane = "WaveGetLaneIndex()"
        local_lane = f"({lane} % {tile_size})"
        if member_name == "shfl":
            source_lane = f"(({lane} - {local_lane}) + {lane_arg})"
            return f"WaveReadLaneAt({value}, {source_lane})"
        if member_name == "shfl_down":
            return (
                f"((({local_lane} + ({lane_arg})) < {tile_size}) ? "
                f"WaveReadLaneAt({value}, ({lane} + ({lane_arg}))) : {value})"
            )
        if member_name == "shfl_up":
            return (
                f"(({local_lane} >= ({lane_arg})) ? "
                f"WaveReadLaneAt({value}, ({lane} - ({lane_arg}))) : {value})"
            )

        source_local_lane = f"({local_lane} ^ ({lane_arg}))"
        source_lane = f"(({lane} - {local_lane}) + {source_local_lane})"
        return (
            f"(({source_local_lane} < {tile_size}) ? "
            f"WaveReadLaneAt({value}, {source_lane}) : {value})"
        )

    def format_thread_block_size_expression(self):
        return "((gl_WorkGroupSize.x * gl_WorkGroupSize.y) * gl_WorkGroupSize.z)"

    def format_unsupported_cooperative_group_expression(
        self, group_kind, member, fallback="0"
    ):
        return (
            f"(/* cooperative_groups {group_kind}.{member} "
            f"not directly supported in CrossGL */ {fallback})"
        )

    def cooperative_group_declaration_metadata(self, node):
        declared_metadata = self.cooperative_group_metadata_from_type(node.vtype)
        factory_metadata = self.cooperative_group_factory_metadata(node.value)
        return factory_metadata or declared_metadata

    def cooperative_group_metadata_from_type(self, type_name):
        type_name = self.normalize_cooperative_group_type_name(type_name)
        base_type, template_args = self.parse_cpp_template(type_name)
        base_name = self.cooperative_group_base_name(type_name)
        if base_name in {
            "thread_group",
            "thread_block",
            "grid_group",
            "multi_grid_group",
            "coalesced_group",
        }:
            return {"kind": base_name}
        base_name = self.cooperative_group_base_name(base_type)
        if base_name and base_name.startswith("thread_block_tile"):
            metadata = {"kind": "thread_block_tile"}
            if template_args:
                metadata["tile_size"] = template_args[0]
            return metadata
        return None

    def normalize_cooperative_group_type_name(self, type_name):
        text = str(type_name).strip()
        text = re.sub(
            r"\b(?:const|volatile|__restrict__|__restrict|restrict)\b", "", text
        )
        text = text.replace("&", " ").replace("*", " ")
        return " ".join(text.split())

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

    def convert_cooperative_group_type(self, hip_type):
        metadata = self.cooperative_group_metadata_from_type(hip_type)
        if metadata is None:
            return None
        group_kind = metadata["kind"]
        if group_kind == "thread_block_tile" and metadata.get("tile_size"):
            return f"cooperative_groups_thread_block_tile_{metadata['tile_size']}"
        return f"cooperative_groups_{group_kind}"

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

    def format_hip_fp16_intrinsic_call(self, function_name, args):
        if isinstance(function_name, str) and function_name.startswith("::"):
            function_name = function_name[2:]

        if function_name == "__half2float" and len(args) == 1:
            return f"f32({args[0]})"
        if function_name == "__float2half" and len(args) == 1:
            return f"f16({args[0]})"
        if function_name == "__float2half2_rn" and len(args) == 1:
            return self.format_vector_constructor("vec2", [args[0], args[0]], "f16")
        if function_name == "__floats2half2_rn" and len(args) == 2:
            return self.format_vector_constructor("vec2", args, "f16")
        if function_name == "__halves2half2" and len(args) == 2:
            return self.format_vector_constructor("vec2", args, "f16")
        if function_name == "__low2float" and len(args) == 1:
            return f"f32({self.format_vector_component_access(args[0], 'x')})"
        if function_name == "__high2float" and len(args) == 1:
            return f"f32({self.format_vector_component_access(args[0], 'y')})"
        if function_name == "__low2half" and len(args) == 1:
            return self.format_vector_component_access(args[0], "x")
        if function_name == "__high2half" and len(args) == 1:
            return self.format_vector_component_access(args[0], "y")
        if function_name == "__low2half2" and len(args) == 1:
            low = self.format_vector_component_access(args[0], "x")
            return self.format_vector_constructor("vec2", [low, low], "f16")
        if function_name == "__high2half2" and len(args) == 1:
            high = self.format_vector_component_access(args[0], "y")
            return self.format_vector_constructor("vec2", [high, high], "f16")
        if function_name in {"__lows2half2", "__low2half2"} and len(args) == 2:
            return self.format_vector_constructor(
                "vec2",
                [
                    self.format_vector_component_access(args[0], "x"),
                    self.format_vector_component_access(args[1], "x"),
                ],
                "f16",
            )
        if function_name == "__highs2half2" and len(args) == 2:
            return self.format_vector_constructor(
                "vec2",
                [
                    self.format_vector_component_access(args[0], "y"),
                    self.format_vector_component_access(args[1], "y"),
                ],
                "f16",
            )
        if function_name == "__lowhigh2highlow" and len(args) == 1:
            return self.format_vector_constructor(
                "vec2",
                [
                    self.format_vector_component_access(args[0], "y"),
                    self.format_vector_component_access(args[0], "x"),
                ],
                "f16",
            )
        if function_name == "__hadd2" and len(args) == 2:
            return f"({args[0]} + {args[1]})"
        if function_name == "__hmul2" and len(args) == 2:
            return f"({args[0]} * {args[1]})"
        if function_name == "__hfma2" and len(args) == 3:
            return f"fma({args[0]}, {args[1]}, {args[2]})"
        return None

    def format_hip_complex_intrinsic_call(self, function_name, args):
        if isinstance(function_name, str) and function_name.startswith("::"):
            function_name = function_name[2:]

        if function_name in {"make_hipComplex", "make_hipFloatComplex"}:
            if len(args) == 2:
                return self.format_vector_constructor("vec2", args, "f32")
            return None
        if function_name == "make_hipDoubleComplex":
            if len(args) == 2:
                return self.format_vector_constructor("vec2", args, "f64")
            return None

        if function_name in {"hipCrealf", "hipCreal"} and len(args) == 1:
            return self.format_vector_component_access(args[0], "x")
        if function_name in {"hipCimagf", "hipCimag"} and len(args) == 1:
            return self.format_vector_component_access(args[0], "y")

        if function_name in {"hipCaddf", "hipCadd"} and len(args) == 2:
            return f"({args[0]} + {args[1]})"
        if function_name in {"hipCsubf", "hipCsub"} and len(args) == 2:
            return f"({args[0]} - {args[1]})"
        if function_name == "hipCmulf" and len(args) == 2:
            return self.format_hip_complex_multiply(args[0], args[1], "f32")
        if function_name == "hipCmul" and len(args) == 2:
            return self.format_hip_complex_multiply(args[0], args[1], "f64")

        return None

    def format_hip_complex_multiply(self, left, right, scalar_type):
        left_real = self.format_vector_component_access(left, "x")
        left_imag = self.format_vector_component_access(left, "y")
        right_real = self.format_vector_component_access(right, "x")
        right_imag = self.format_vector_component_access(right, "y")
        return self.format_vector_constructor(
            "vec2",
            [
                f"(({left_real} * {right_real}) - ({left_imag} * {right_imag}))",
                f"(({left_real} * {right_imag}) + ({left_imag} * {right_real}))",
            ],
            scalar_type,
        )

    def format_hip_type_cast_intrinsic_call(self, function_name, args):
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

        return None

    def format_hip_integer_intrinsic_call(self, function_name, args):
        if isinstance(function_name, str) and function_name.startswith("::"):
            function_name = function_name[2:]

        if function_name == "__mul24" and len(args) == 2:
            left = self.format_hip_signed_24_bit_operand(args[0])
            right = self.format_hip_signed_24_bit_operand(args[1])
            return f"({left} * {right})"
        if function_name == "__umul24" and len(args) == 2:
            return f"(({args[0]} & 0x00ffffffu) * ({args[1]} & 0x00ffffffu))"
        if function_name == "__mulhi" and len(args) == 2:
            return self.format_hip_signed_multiply_high(args[0], args[1])
        if function_name == "__umulhi" and len(args) == 2:
            return self.format_hip_unsigned_multiply_high(args[0], args[1])
        if function_name == "__byte_perm" and len(args) == 3:
            return self.format_hip_byte_perm(args[0], args[1], args[2])
        if (
            function_name
            in {
                "__funnelshift_l",
                "__funnelshift_lc",
                "__funnelshift_r",
                "__funnelshift_rc",
            }
            and len(args) == 3
        ):
            return self.format_hip_funnelshift(function_name, args[0], args[1], args[2])
        if function_name == "__bitextract_u32" and len(args) == 3:
            return self.format_hip_unsigned_bit_extract(args[0], args[1], args[2])
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
        if function_name in {"__popc", "__popcll"} and len(args) == 1:
            return f"bitCount({args[0]})"
        return None

    def format_integer_average_floor(self, left, right):
        return f"(({left} & {right}) + (({left} ^ {right}) >> 1))"

    def format_integer_average_rounded(self, left, right):
        return f"(({left} | {right}) - (({left} ^ {right}) >> 1))"

    def format_hip_signed_24_bit_operand(self, arg):
        return f"(({arg} << 8) >> 8)"

    def format_hip_signed_multiply_high(self, left, right):
        return f"i32((i64({left}) * i64({right})) >> 32)"

    def format_hip_unsigned_multiply_high(self, left, right):
        return f"u32((u64({left}) * u64({right})) >> 32u)"

    def format_hip_unsigned_bit_extract(self, value, offset, width):
        width_value = self.parse_hip_integer_literal(width)
        if width_value is not None:
            if width_value <= 0:
                mask = "0u"
            elif width_value >= 32:
                mask = "0xffffffffu"
            else:
                mask = f"0x{(1 << width_value) - 1:x}u"
        else:
            mask = f"((1u << {width}) - 1u)"
        return f"(({value} >> {offset}) & {mask})"

    def format_hip_funnelshift(self, function_name, low, high, shift):
        shift_amount = f"({shift} & 31u)"
        if function_name.endswith("lc") or function_name.endswith("rc"):
            shift_amount = f"min({shift}, 32u)"

        source = f"((u64({high}) << 32u) | u64({low}))"
        if function_name.endswith("l") or function_name.endswith("lc"):
            return f"u32(({source} << {shift_amount}) >> 32u)"
        return f"u32({source} >> {shift_amount})"

    def format_hip_float_intrinsic_call(self, function_name, args):
        if isinstance(function_name, str) and function_name.startswith("::"):
            function_name = function_name[2:]

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

        return None

    def format_hip_byte_perm(self, left, right, selector):
        selector_value = self.parse_hip_integer_literal(selector)
        if selector_value is not None:
            return self.format_hip_static_byte_perm(left, right, selector_value)

        terms = []
        for result_index in range(4):
            selected = self.format_hip_dynamic_byte_perm_byte(
                left, right, selector, result_index
            )
            if result_index:
                selected = f"({selected} << {result_index * 8})"
            terms.append(selected)
        return f"({' | '.join(terms)})"

    def format_hip_static_byte_perm(self, left, right, selector_value):
        terms = []
        for result_index in range(4):
            source_index = (selector_value >> (result_index * 4)) & 0xF
            byte = self.format_hip_byte_source(left, right, source_index)
            if byte == "0":
                continue
            if result_index:
                byte = f"({byte} << {result_index * 8})"
            terms.append(byte)
        return f"({' | '.join(terms)})" if terms else "0"

    def format_hip_dynamic_byte_perm_byte(self, left, right, selector, result_index):
        selector_nibble = (
            f"(({selector} >> {result_index * 4}) & 0xf)"
            if result_index
            else f"({selector} & 0xf)"
        )
        fallback = "0"
        for source_index in reversed(range(8)):
            byte = self.format_hip_byte_source(left, right, source_index)
            fallback = f"(({selector_nibble} == {source_index}) ? {byte} : {fallback})"
        return fallback

    def format_hip_byte_source(self, left, right, source_index):
        if source_index < 0 or source_index > 7:
            return "0"
        source = left if source_index < 4 else right
        shift = (source_index % 4) * 8
        if shift:
            return f"(({source} >> {shift}) & 0xffu)"
        return f"({source} & 0xffu)"

    def parse_hip_integer_literal(self, value):
        text = self.strip_wrapping_parentheses(str(value).strip()).replace("'", "")
        if not text:
            return None
        while text and text[-1].lower() in {"u", "l"}:
            text = text[:-1]
        try:
            return int(text, 0)
        except ValueError:
            return None

    def format_hip_sync_vote_intrinsic_call(self, function_name, args):
        if isinstance(function_name, str) and function_name.startswith("::"):
            function_name = function_name[2:]

        if function_name not in {
            "__syncthreads_count",
            "__syncthreads_and",
            "__syncthreads_or",
        }:
            return None

        args_text = ", ".join(args)
        return (
            f"(/* HIP block-wide sync vote {function_name} predicate: {args_text} "
            "not directly supported in CrossGL */ 0)"
        )

    def format_vector_component_access(self, expression, component):
        text = str(expression).strip()
        if text and all(char.isalnum() or char in "_." for char in text):
            return f"{text}.{component}"
        return f"({text}).{component}"

    def format_hip_resource_call(self, function_name, args, raw_args=None):
        base_name, template_args = self.parse_cpp_template(function_name)
        if self.is_user_defined_function(base_name):
            return None

        value_type = template_args[0] if template_args else None
        if base_name == "tex1Dfetch":
            return self.format_hip_texture_fetch_call(args)
        if base_name in {"tex1D", "tex1DLod", "tex1DGrad"}:
            return self.format_hip_texture_call(base_name, args, "vec1", 1, raw_args)
        if base_name in {"tex2D", "tex2DLod", "tex2DGrad"}:
            return self.format_hip_texture_call(base_name, args, "vec2", 2, raw_args)
        if base_name == "tex2Dgather":
            return self.format_hip_texture_gather_call(args)
        if base_name in {"tex3D", "tex3DLod", "tex3DGrad"}:
            return self.format_hip_texture_call(base_name, args, "vec3", 3, raw_args)
        if base_name in {"texCubemap", "texCubemapLod", "texCubemapGrad"}:
            return self.format_hip_texture_call(base_name, args, "vec3", 3, raw_args)
        if base_name in {"tex1DLayered", "tex1DLayeredLod", "tex1DLayeredGrad"}:
            return self.format_hip_texture_call(base_name, args, "vec2", 2, raw_args)
        if base_name in {"tex2DLayered", "tex2DLayeredLod", "tex2DLayeredGrad"}:
            return self.format_hip_texture_call(base_name, args, "vec3", 3, raw_args)
        if base_name in {
            "texCubemapLayered",
            "texCubemapLayeredLod",
            "texCubemapLayeredGrad",
        }:
            return self.format_hip_texture_call(base_name, args, "vec4", 4, raw_args)

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
                "surfCubemapLayeredread": 4,
            }[base_name]
            return self.format_hip_surface_read(base_name, args, dimensions, value_type)

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
                "surfCubemapLayeredwrite": 4,
            }[base_name]
            return self.format_hip_surface_write(
                base_name,
                args,
                dimensions,
                value_type,
                value_is_pointer=base_name == "surfCubemapLayeredwrite",
            )

        return None

    def format_hip_texture_call(
        self, function_name, args, vector_name, dimensions, raw_args=None
    ):
        if len(args) < 2:
            return None

        if self.is_sparse_hip_texture_call(function_name, args):
            return self.format_unsupported_hip_texture_sparse_residency_call(
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
        raw_coordinate_args = (raw_args or [])[1 : 1 + coordinate_count]
        if coordinate_count == 1:
            rank = self.infer_texture_coordinate_rank(
                coordinate_args[0],
                raw_coordinate_args[0] if raw_coordinate_args else None,
            )
            if dimensions > 1 and rank is not None and rank != dimensions:
                return self.format_hip_texture_coordinate_rank_diagnostic(function_name)
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
            return (
                f"textureGrad({texture_name}, {coordinate}, "
                f"{remaining[0]}, {remaining[1]})"
            )
        if "Lod" in function_name:
            if not remaining:
                return None
            return f"textureLod({texture_name}, {coordinate}, {remaining[0]})"
        return f"texture({texture_name}, {coordinate})"

    def infer_texture_coordinate_rank(self, coordinate, raw_coordinate=None):
        expression_type = self.lookup_texture_coordinate_expression_type(
            coordinate, raw_coordinate
        )
        rank = self.vector_type_rank(expression_type)
        if rank is not None:
            return rank
        if self.is_scalar_type_name(expression_type):
            return 1

        text = self.strip_wrapping_parentheses(str(coordinate).strip())
        rank = self.vector_constructor_rank(text)
        if rank is not None:
            return rank
        if self.is_swizzle_expression(text):
            return len(text.rsplit(".", 1)[1])
        if self.is_scalar_literal(text):
            return 1
        return None

    def lookup_texture_coordinate_expression_type(self, coordinate, raw_coordinate):
        name = self.get_resource_object_expression_name(raw_coordinate)
        if name is None:
            name = self.strip_wrapping_parentheses(str(coordinate).strip())
        if self.is_simple_identifier(name):
            return self.lookup_variable_type(name)
        return None

    def vector_type_rank(self, type_name):
        if not type_name:
            return None
        text = str(type_name).strip()
        if text.startswith("vec") and len(text) >= 4 and text[3].isdigit():
            return int(text[3])
        mapped_type = self.VECTOR_TYPE_MAPPING.get(text)
        if mapped_type is not None:
            return self.vector_type_rank(mapped_type)
        return None

    def vector_constructor_rank(self, text):
        if text.startswith("vec") and len(text) >= 4 and text[3].isdigit():
            return int(text[3])
        return None

    def is_scalar_type_name(self, type_name):
        return type_name in {
            "bool",
            "f32",
            "f64",
            "i8",
            "u8",
            "i16",
            "u16",
            "i32",
            "u32",
            "i64",
            "u64",
        }

    def is_simple_identifier(self, text):
        if not isinstance(text, str) or not text:
            return False
        return text.replace("_", "").isalnum() and not text[0].isdigit()

    def is_swizzle_expression(self, text):
        if "." not in text:
            return False
        suffix = text.rsplit(".", 1)[1]
        return bool(suffix) and all(char in "xyzwrgba" for char in suffix)

    def is_scalar_literal(self, text):
        if not text:
            return False
        literal = text.lower()
        while literal.endswith(("f", "u", "l")):
            literal = literal[:-1]
        try:
            float(literal)
        except ValueError:
            return False
        return True

    def strip_wrapping_parentheses(self, text):
        while self.has_wrapping_parentheses(text):
            inner = text[1:-1].strip()
            if not inner:
                break
            text = inner
        return text

    def has_wrapping_parentheses(self, text):
        if not text.startswith("(") or not text.endswith(")"):
            return False
        depth = 0
        for index, char in enumerate(text):
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth == 0 and index != len(text) - 1:
                    return False
            if depth < 0:
                return False
        return depth == 0

    def is_sparse_hip_texture_call(self, function_name, args):
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

    def format_unsupported_hip_texture_sparse_residency_call(self, function_name):
        return self.format_unsupported_hip_resource_expression(
            "texture",
            f"{function_name} sparse residency",
            "vec4<f32>(0.0, 0.0, 0.0, 0.0)",
        )

    def format_hip_texture_gather_call(self, args):
        if len(args) == 2:
            texture_name, coordinate = args
            component = None
        elif len(args) in {3, 4}:
            texture_name = args[0]
            coordinate = self.format_vector_constructor("vec2", args[1:3])
            component = args[3] if len(args) == 4 else None
        else:
            return self.format_unsupported_hip_texture_sparse_residency_call(
                "tex2Dgather"
            )

        if component is not None:
            return f"textureGather({texture_name}, {coordinate}, {component})"
        return f"textureGather({texture_name}, {coordinate})"

    def format_hip_texture_coordinate_rank_diagnostic(self, function_name):
        return self.format_unsupported_hip_resource_expression(
            "texture",
            f"{function_name} coordinate rank mismatch",
            "vec4<f32>(0.0, 0.0, 0.0, 0.0)",
        )

    def format_hip_texture_fetch_call(self, args):
        if len(args) == 3:
            return self.format_unsupported_hip_resource_expression(
                "texture",
                "tex1Dfetch sparse residency",
                "vec4<f32>(0.0, 0.0, 0.0, 0.0)",
            )
        if len(args) != 2:
            return None
        texture_name, coordinate = args
        return f"texelFetch({texture_name}, {coordinate}, 0)"

    def format_unsupported_hip_resource_expression(self, kind, member, fallback):
        return (
            f"(/* hip {kind}.{member} not directly supported in CrossGL */ "
            f"{fallback})"
        )

    def format_hip_surface_read(self, function_name, args, dimensions, value_type):
        if not args:
            return None

        if self.is_surface_output_target(args[0]):
            if len(args) < 2:
                return None
            output_target = self.strip_surface_output_target(args[0])
            surface_name = args[1]
            coord_start = 2
            expected_arg_count = dimensions + 2
        else:
            output_target = None
            surface_name = args[0]
            coord_start = 1
            expected_arg_count = dimensions + 1

        if len(args) < expected_arg_count:
            if len(args) > coord_start:
                diagnostic = self.format_hip_surface_coordinate_shape_diagnostic(
                    function_name, value_type
                )
                if output_target is not None:
                    return f"{output_target} = {diagnostic}"
                return diagnostic
            return None

        coord_args = self.format_hip_surface_coordinate_args(
            args, coord_start, dimensions, value_type
        )
        if self.has_hip_surface_coordinate_shape_mismatch(coord_args):
            diagnostic = self.format_hip_surface_coordinate_shape_diagnostic(
                function_name, value_type
            )
            if output_target is not None:
                return f"{output_target} = {diagnostic}"
            return diagnostic

        coord = self.format_vector_constructor(f"vec{dimensions}", coord_args, "i32")
        image_load = f"imageLoad({surface_name}, {coord})"
        if output_target is not None:
            return f"{output_target} = {image_load}"
        return image_load

    def format_hip_surface_write(
        self,
        function_name,
        args,
        dimensions,
        value_type,
        value_is_pointer=False,
    ):
        if len(args) < dimensions + 2:
            if len(args) > 2:
                return self.format_hip_surface_coordinate_shape_diagnostic(
                    function_name, None
                )
            return None
        value = args[0]
        if value_is_pointer:
            value = self.strip_surface_output_target(value)
        surface_name = args[1]
        coord_args = self.format_hip_surface_coordinate_args(
            args, 2, dimensions, value_type
        )
        if self.has_hip_surface_coordinate_shape_mismatch(coord_args):
            return self.format_hip_surface_coordinate_shape_diagnostic(
                function_name, None
            )
        coord = self.format_vector_constructor(f"vec{dimensions}", coord_args, "i32")
        return f"imageStore({surface_name}, {coord}, {value})"

    def format_hip_surface_coordinate_args(
        self, args, coord_start, dimensions, value_type
    ):
        coord_args = [self.strip_surface_byte_offset(args[coord_start], value_type)]
        coord_args.extend(args[coord_start + 1 : coord_start + dimensions])
        return coord_args

    def has_hip_surface_coordinate_shape_mismatch(self, coord_args):
        for coordinate in coord_args:
            rank = self.infer_texture_coordinate_rank(coordinate)
            if rank is not None and rank != 1:
                return True
        return False

    def format_hip_surface_coordinate_shape_diagnostic(
        self, function_name, value_type=None
    ):
        fallback = self.format_hip_zero_value(value_type) if value_type else "0"
        return self.format_unsupported_hip_resource_expression(
            "surface",
            f"{function_name} coordinate shape mismatch",
            fallback,
        )

    def format_hip_zero_value(self, type_name):
        if not type_name:
            return "0"

        crossgl_type = self.convert_hip_type_to_crossgl(type_name)
        if (
            crossgl_type.startswith("vec")
            and len(crossgl_type) >= 4
            and crossgl_type[3].isdigit()
            and "<" in crossgl_type
            and crossgl_type.endswith(">")
        ):
            rank = int(crossgl_type[3])
            element_type = crossgl_type.split("<", 1)[1][:-1]
            zero = self.format_hip_zero_scalar(element_type)
            return f"{crossgl_type}({', '.join([zero] * rank)})"
        return self.format_hip_zero_scalar(crossgl_type)

    def format_hip_zero_scalar(self, type_name):
        if type_name == "bool":
            return "false"
        if type_name in {"f32", "f64"}:
            return "0.0"
        return "0"

    def is_surface_output_target(self, expression):
        text = str(expression).strip()
        if text.startswith("(") and text.endswith(")"):
            text = text[1:-1].strip()
        return text.startswith("&")

    def strip_surface_output_target(self, expression):
        text = str(expression).strip()
        if text.startswith("(") and text.endswith(")"):
            text = text[1:-1].strip()
        if text.startswith("&"):
            return text[1:].strip()
        return text

    def strip_surface_byte_offset(self, expression, value_type):
        text = str(expression).strip()
        if text.startswith("(") and text.endswith(")"):
            text = text[1:-1].strip()

        if value_type:
            suffix = f" * sizeof({value_type})"
            if text.endswith(suffix):
                return text[: -len(suffix)].strip()

        marker = " * sizeof("
        if marker in text and text.endswith(")"):
            return text.split(marker, 1)[0].strip()

        value_size = self.hip_surface_value_type_size(value_type)
        if value_size is not None:
            normalized = self.strip_surface_constant_byte_stride(text, value_size)
            if normalized is not None:
                return normalized

        return expression

    def hip_surface_value_type_size(self, value_type):
        if not value_type:
            return None

        text = str(value_type).strip()
        if text.startswith("const "):
            text = text[6:].strip()

        vector_match = re.fullmatch(
            r"(?:make_)?(char|uchar|short|ushort|int|uint|long|ulong|"
            r"longlong|ulonglong|float|double)([1-4])",
            text,
        )
        if vector_match:
            scalar_type, count = vector_match.groups()
            scalar_size = {
                "char": 1,
                "uchar": 1,
                "short": 2,
                "ushort": 2,
                "int": 4,
                "uint": 4,
                "long": 8,
                "ulong": 8,
                "longlong": 8,
                "ulonglong": 8,
                "float": 4,
                "double": 8,
            }.get(scalar_type)
            return scalar_size * int(count) if scalar_size is not None else None

        return {
            "char": 1,
            "signed char": 1,
            "unsigned char": 1,
            "int8_t": 1,
            "uint8_t": 1,
            "short": 2,
            "unsigned short": 2,
            "int16_t": 2,
            "uint16_t": 2,
            "int": 4,
            "unsigned int": 4,
            "float": 4,
            "int32_t": 4,
            "uint32_t": 4,
            "long": 8,
            "unsigned long": 8,
            "long long": 8,
            "unsigned long long": 8,
            "double": 8,
            "int64_t": 8,
            "uint64_t": 8,
        }.get(text)

    def strip_surface_constant_byte_stride(self, text, value_size):
        patterns = (
            (rf"^(.+?)\s*\*\s*{value_size}$", 1),
            (rf"^{value_size}\s*\*\s*(.+)$", 1),
        )
        for pattern, group in patterns:
            match = re.fullmatch(pattern, text)
            if match:
                coordinate = match.group(group).strip()
                if coordinate:
                    return self.strip_wrapping_parentheses(coordinate)
        return None

    def format_vector_constructor(self, vector_name, args, element_type="f32"):
        if vector_name == "vec1":
            return args[0]
        return f"{vector_name}<{element_type}>({', '.join(args)})"

    def format_hip_dim3_constructor_call(self, function_name, args):
        if isinstance(function_name, str) and function_name.startswith("::"):
            function_name = function_name[2:]
        if function_name != "dim3" or len(args) > 3:
            return None

        padded_args = list(args) + ["1"] * (3 - len(args))
        return f"vec3<u32>({', '.join(padded_args)})"

    def format_hip_dim3_default_initializer(self, node):
        if not self.is_dim3_variable_declaration(node):
            return None
        if getattr(node, "value", None) is not None:
            return None
        return self.format_hip_dim3_constructor_call("dim3", [])

    def format_hip_dim3_brace_initializer(self, node):
        if not self.is_dim3_variable_declaration(node):
            return None

        value = getattr(node, "value", None)
        if not isinstance(value, InitializerListNode) or len(value.elements) > 3:
            return None

        args = [self.visit(element) for element in value.elements]
        return self.format_hip_dim3_constructor_call("dim3", args)

    def is_dim3_variable_declaration(self, node):
        return self.strip_type_qualifiers(getattr(node, "vtype", "")) == "dim3"

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
                param_type = self.convert_hip_type_to_crossgl(arg.vtype)
                return f"{param_type} {arg.name}"
            return arg.name
        return self.format_lambda_body(arg)

    def format_lambda_body(self, arg):
        if isinstance(arg, str):
            return arg
        return self.visit(arg)

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

        crossgl_func = self.convert_hip_builtin_function(operation)
        expression = f"{crossgl_func}({args_str})"
        if scope is not None:
            return (
                f"(/* hip {scope}-scope atomic {node.operation} lowered to "
                f"{crossgl_func}; scope not preserved */ {expression})"
            )
        return expression

    def is_get_method_call(self, node):
        return (
            isinstance(getattr(node, "name", None), MemberAccessNode)
            and node.name.member == "get"
            and not getattr(node, "args", [])
            and self.is_unique_ptr_expression(node.name.object)
        )

    def format_make_unique_call(self, function_name, args):
        base_name, template_args = self.parse_cpp_template(function_name)
        if not self.is_std_make_unique_base_name(base_name) or not template_args:
            return None

        target_type, is_array = self.unwrap_array_template_type(template_args[0])
        target_type = self.convert_hip_type_to_crossgl(target_type)
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
        placement_args = getattr(node, "placement_args", None)
        if placement_args is not None:
            target_type = self.convert_hip_type_to_crossgl(node.target_type)
            args = ", ".join(self.visit(arg) for arg in node.args)
            placement = ", ".join(self.visit(arg) for arg in placement_args)
            fallback = self.visit(placement_args[0]) if placement_args else "nullptr"
            return (
                f"(/* HIP placement new: new({placement}) {target_type}({args}) "
                f"not directly supported in CrossGL */ {fallback})"
            )

        target_type = self.convert_hip_type_to_crossgl(node.target_type)
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
        if self.is_decltype_type_name(node.alias_type):
            alias_type = node.alias_type
        else:
            alias_type = self.convert_hip_type_to_crossgl(node.alias_type)
        if self.indent_level == 0:
            alias_type = self.normalize_top_level_type_alias(alias_type)
        self.emit(f"typedef {alias_type} {node.name};")

    def normalize_top_level_type_alias(self, alias_type):
        """Keep C++ trait aliases reparsable by CrossGL's top-level typedef parser."""
        member_alias = self.convert_type_member_alias(alias_type)
        if member_alias is not None:
            return member_alias

        template_alias = self.convert_reparsable_std_template_alias(alias_type)
        if template_alias is not None:
            return template_alias

        scoped_alias = self.convert_reparsable_scoped_alias(alias_type)
        if scoped_alias is not None:
            return scoped_alias

        return alias_type

    def convert_reparsable_scoped_alias(self, alias_type):
        if not isinstance(alias_type, str):
            return None

        alias_type = alias_type.strip()
        if "<" in alias_type or ">" in alias_type or "::" not in alias_type:
            return None
        if not re.fullmatch(
            r"[A-Za-z_][A-Za-z0-9_]*(?:::[A-Za-z_][A-Za-z0-9_]*)+",
            alias_type,
        ):
            return None

        return "ptr<void>"

    def convert_type_member_alias(self, alias_type):
        parsed = self.parse_cpp_template_member_type(alias_type)
        if parsed is None:
            return None

        base_name, template_args = parsed
        if not base_name.startswith("std::"):
            return None

        alias_base = f"{base_name.rsplit('::', 1)[-1]}_t"
        converted_args = [
            self.convert_top_level_alias_template_arg(arg) for arg in template_args
        ]
        return f"{alias_base}<{', '.join(converted_args)}>"

    def convert_reparsable_std_template_alias(self, alias_type):
        base_name, template_args = self.parse_cpp_template(alias_type)
        if not template_args or not base_name.startswith("std::"):
            return None

        alias_base = base_name.rsplit("::", 1)[-1]
        converted_args = [
            self.convert_top_level_alias_template_arg(arg) for arg in template_args
        ]
        return f"{alias_base}<{', '.join(converted_args)}>"

    def convert_top_level_alias_template_arg(self, arg):
        member_alias = self.convert_type_member_alias(arg)
        if member_alias is not None:
            return member_alias

        template_alias = self.convert_reparsable_std_template_alias(arg)
        if template_alias is not None:
            return template_alias

        return self.convert_hip_type_to_crossgl(arg)

    def parse_cpp_template_member_type(self, text):
        if not isinstance(text, str):
            return None

        text = text.strip()
        start = text.find("<")
        if start == -1:
            return None

        depth = 0
        end = None
        for index in range(start, len(text)):
            char = text[index]
            if char == "<":
                depth += 1
            elif char == ">":
                depth -= 1
                if depth == 0:
                    end = index
                    break

        if end is None or text[end + 1 :].strip() != "::type":
            return None

        base_name = text[:start].strip()
        template_args = self.split_cpp_template_args(text[start + 1 : end])
        return base_name, template_args

    def visit_MemberAccessNode(self, node):
        if self.suppress_device_property_member_access == 0:
            property_expression = self.format_hip_device_property_member_read(node)
            if property_expression is not None:
                return property_expression
        if self.suppress_device_query_value_access == 0:
            member_query_expression = self.format_hip_member_query_read(node)
            if member_query_expression is not None:
                return member_query_expression

        obj = self.visit(node.object)
        operator = "->" if getattr(node, "is_pointer", False) else "."
        if node.member == "x" and operator == "." and self.is_vector1_name(node.object):
            return obj
        return f"{obj}{operator}{node.member}"

    def format_hip_device_property_member_read(self, node):
        if not isinstance(getattr(node, "object", None), str):
            return None

        object_name = node.object
        object_type = self.lookup_variable_type(object_name)
        if not self.is_hip_device_property_object_type(
            object_type, getattr(node, "is_pointer", False)
        ):
            return None

        return self.visit_HipDevicePropertyNode(
            HipDevicePropertyNode(
                node.member, self.lookup_device_property_source(object_name)
            )
        )

    def is_hip_device_property_object_type(self, object_type, is_pointer_access):
        if is_pointer_access:
            return object_type == "ptr<hipDeviceProp_t>"
        return object_type == "hipDeviceProp_t"

    def format_hip_member_query_read(self, node):
        path = self.get_member_query_path(node)
        if path is None:
            return None

        object_name, member_path = path
        source = self.lookup_member_query_source(object_name)
        if not source:
            return None

        query_name = source.get(member_path)
        if query_name is None:
            return None

        return f"(/* HIP device query: {query_name} */ 0)"

    def get_member_query_path(self, node):
        path = []

        def collect(current):
            if isinstance(current, str):
                return current
            if isinstance(current, MemberAccessNode):
                if getattr(current, "is_pointer", False):
                    return None
                root = collect(current.object)
                if root is None:
                    return None
                path.append(current.member)
                return root
            if isinstance(current, ArrayAccessNode):
                root = collect(current.array)
                if root is None:
                    return None
                index = self.format_member_query_index(current.index)
                if index is None:
                    return None
                if path:
                    path[-1] = f"{path[-1]}[{index}]"
                else:
                    path.append(f"[{index}]")
                return root
            return None

        root_name = collect(node)
        if root_name is None or not path:
            return None
        return root_name, ".".join(path)

    def format_member_query_index(self, index):
        if isinstance(index, int):
            return str(index)
        if isinstance(index, str) and index.isdigit():
            return index
        return None

    def format_hip_device_attribute_read(self, name):
        if self.suppress_device_attribute_value_access != 0:
            return None

        source = self.lookup_device_attribute_source(name)
        if source is None:
            return None

        attribute_name, device_id = source
        return (
            f"(/* HIP device attribute: {attribute_name}, " f"device: {device_id} */ 0)"
        )

    def format_hip_device_query_read(self, name):
        if self.suppress_device_query_value_access != 0:
            return None

        source = self.lookup_device_query_source(name)
        if source is None:
            return None

        query_name, device_id = source
        if device_id is None:
            return f"(/* HIP device query: {query_name} */ 0)"
        return f"(/* HIP device query: {query_name}, device: {device_id} */ 0)"

    def visit_ArrayAccessNode(self, node):
        if self.suppress_device_query_value_access == 0:
            member_query_expression = self.format_hip_member_query_read(node)
            if member_query_expression is not None:
                return member_query_expression

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

    def visit_SyncNode(self, node):
        if node.sync_type == "__syncthreads":
            self.emit("workgroupBarrier();")
        elif node.sync_type == "hipDeviceSynchronize":
            self.emit("// HIP device synchronize")
        elif node.sync_type == "__syncwarp":
            args = ", ".join(self.visit(arg) for arg in node.args)
            self.emit(f"// __syncwarp({args}) not directly supported in CrossGL")
        else:
            self.emit(f"// {node.sync_type}();")

    def visit_HipAsmNode(self, node):
        volatility = " volatile" if node.is_volatile else ""
        self.emit(f"// HIP inline assembly{volatility}: {node.template}")
        if node.outputs:
            self.emit(
                f"// HIP inline assembly outputs: {self.format_hip_asm_operands(node.outputs)}"
            )
        if node.inputs:
            self.emit(
                f"// HIP inline assembly inputs: {self.format_hip_asm_operands(node.inputs)}"
            )
        if node.clobbers:
            self.emit(f"// HIP inline assembly clobbers: {', '.join(node.clobbers)}")

    def format_hip_asm_operands(self, operands):
        formatted = []
        for operand in operands:
            prefix = (
                f"[{operand.symbolic_name}] "
                if operand.symbolic_name is not None
                else ""
            )
            expression = (
                self.visit_source_comment_expression(operand.expression)
                if operand.expression is not None
                else None
            )
            if expression is None:
                formatted.append(f"{prefix}{operand.constraint}")
            else:
                formatted.append(f"{prefix}{operand.constraint}({expression})")
        return ", ".join(formatted)

    def visit_HipBuiltinNode(self, node):
        builtin_map = {
            "threadIdx": "gl_LocalInvocationID",
            "blockIdx": "gl_WorkGroupID",
            "gridDim": "gl_NumWorkGroups",
            "blockDim": "gl_WorkGroupSize",
            "warpSize": "32",
        }

        base_name = builtin_map.get(node.builtin_name, node.builtin_name)
        if hasattr(node, "component") and node.component:
            return f"{base_name}.{node.component}"
        else:
            return base_name

    def visit_ReturnNode(self, node):
        if hasattr(node, "value") and node.value:
            value = self.visit(node.value)
            self.emit(f"return {value};")
        else:
            self.emit("return;")

    def visit_BreakNode(self, node):
        self.emit("break;")

    def visit_ContinueNode(self, node):
        self.emit("continue;")

    def visit_IfNode(self, node):
        condition = self.visit(node.condition)
        self.emit(f"if ({condition}) {{")

        self.indent_level += 1
        if hasattr(node, "if_body") and node.if_body:
            if isinstance(node.if_body, list):
                for stmt in node.if_body:
                    self.emit_statement(stmt)
            else:
                self.emit_statement(node.if_body)
        self.indent_level -= 1

        if hasattr(node, "else_body") and node.else_body:
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
        init_node = node.init if hasattr(node, "init") else None
        scoped_init = isinstance(init_node, list)
        if scoped_init:
            self.emit("{")
            self.indent_level += 1
            for stmt in init_node:
                self.emit_statement(stmt)
            init = ""
        else:
            init = self.format_statement_fragment(init_node)
        condition = (
            self.visit(node.condition)
            if hasattr(node, "condition") and node.condition
            else ""
        )
        update = self.format_statement_fragment(
            node.update if hasattr(node, "update") else None
        )

        self.emit(f"for ({init}; {condition}; {update}) {{")

        self.indent_level += 1
        if hasattr(node, "body") and node.body:
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
        if hasattr(node, "body") and node.body:
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
        if hasattr(node, "body") and node.body:
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
        if hasattr(node, "body") and node.body:
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
                    for stmt in getattr(case, "body", []):
                        self.emit_statement(stmt)
                    self.indent_level -= 1
                else:
                    self.visit(case)
        else:
            for case in getattr(node, "cases", []):
                self.visit(case)

            if getattr(node, "default_case", None) is not None:
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
        for stmt in getattr(node, "body", []):
            self.emit_statement(stmt)
        self.indent_level -= 1

    def visit_TernaryOpNode(self, node):
        condition = self.visit(node.condition)
        true_expr = self.visit(node.true_expr)
        false_expr = self.visit(node.false_expr)
        return f"({condition} ? {true_expr} : {false_expr})"

    def visit_CastNode(self, node):
        target_type = self.convert_hip_type_to_crossgl(node.target_type)
        expression = self.visit(node.expression)
        return f"{target_type}({expression})"

    def convert_hip_variable_type_to_crossgl(self, hip_type, name):
        """Convert HIP variable types, using call-site hints for resource handles."""
        resource_type = self.convert_hip_resource_object_type(hip_type, name)
        if resource_type is not None:
            return resource_type
        return self.convert_hip_type_to_crossgl(hip_type)

    def convert_hip_resource_object_type(self, hip_type, name):
        hint = self.lookup_resource_object_type_hint(name)
        if hint is None:
            return None
        return self.convert_hip_resource_object_type_with_hint(hip_type, hint)

    def convert_hip_resource_object_type_with_hint(self, hip_type, hint):
        hip_type = self.strip_type_qualifiers(hip_type)

        if self.has_array_suffix(hip_type):
            base_type = hip_type.split("[", 1)[0].strip()
            mapped_type = self.convert_hip_resource_object_type_with_hint(
                base_type, hint
            )
            if mapped_type is None:
                return None
            return self.wrap_mapped_hip_array_type(hip_type, mapped_type)

        base_type, pointer_depth = self.split_pointer_declarators(hip_type)
        if pointer_depth:
            mapped_type = self.convert_hip_resource_object_base_type(base_type, hint)
            if mapped_type is None:
                return None
            for _ in range(pointer_depth):
                mapped_type = f"ptr<{mapped_type}>"
            return mapped_type

        return self.convert_hip_resource_object_base_type(hip_type, hint)

    def convert_hip_resource_object_base_type(self, hip_type, hint):
        hip_type = self.strip_type_qualifiers(hip_type)
        if hip_type == "hipTextureObject_t" and hint.startswith("sampler"):
            return hint
        if hip_type == "hipSurfaceObject_t" and "image" in hint:
            return hint
        return None

    def wrap_mapped_hip_array_type(self, hip_type, mapped_type):
        base_type = hip_type.split("[", 1)[0].strip()
        dimensions = []
        remainder = hip_type[len(base_type) :].strip()

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

    def convert_hip_type_to_crossgl(self, hip_type):
        """Map a HIP type name to the closest CrossGL type name."""
        if hip_type is None:
            return "void"

        if not isinstance(hip_type, str):
            hip_type = str(hip_type)

        hip_type = self.strip_type_qualifiers(hip_type)
        hip_type = self.strip_variadic_type_marker(hip_type)
        hip_type = self.strip_union_type_keyword(hip_type)
        hip_type = self.CPP_SCALAR_TYPE_ALIASES.get(hip_type, hip_type)
        matrix_type = self.convert_native_matrix_helper_name_to_crossgl(hip_type)
        if matrix_type is not None:
            return matrix_type
        cooperative_group_type = self.convert_cooperative_group_type(hip_type)
        if cooperative_group_type is not None:
            return cooperative_group_type

        type_mapping = {
            # Basic types
            "void": "void",
            "bool": "bool",
            "char": "i8",
            "signed char": "i8",
            "unsigned char": "u8",
            "short": "i16",
            "signed short": "i16",
            "short int": "i16",
            "signed short int": "i16",
            "unsigned short": "u16",
            "unsigned short int": "u16",
            "int": "i32",
            "signed": "i32",
            "unsigned": "u32",
            "signed int": "i32",
            "uint": "u32",
            "unsigned int": "u32",
            "long": "i64",
            "signed long": "i64",
            "long int": "i64",
            "signed long int": "i64",
            "unsigned long": "u64",
            "unsigned long int": "u64",
            "long unsigned int": "u64",
            "long long": "i64",
            "signed long long": "i64",
            "long long int": "i64",
            "signed long long int": "i64",
            "unsigned long long": "u64",
            "unsigned long long int": "u64",
            "long long unsigned int": "u64",
            "__int64": "i64",
            "signed __int64": "i64",
            "unsigned __int64": "u64",
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
            "float16_t": "f16",
            "float32_t": "f32",
            "float64_t": "f64",
            "rocwmma::float16_t": "f16",
            "rocwmma::float32_t": "f32",
            "rocwmma::float64_t": "f64",
            "hipComplex": "vec2<f32>",
            "hipFloatComplex": "vec2<f32>",
            "hipDoubleComplex": "vec2<f64>",
            "float": "f32",
            "double": "f64",
            "size_t": "u32",
            "hipArray": "ptr<void>",
            "hipArray_t": "ptr<void>",
            "hipTextureObject_t": "sampler",
            "hipSurfaceObject_t": "image2D",
            # HIP vector types
            **self.VECTOR1_TYPE_MAPPING,
            **self.VECTOR_TYPE_MAPPING,
            "dim3": "vec3<u32>",
        }

        unique_ptr_type = self.convert_unique_ptr_type(hip_type)
        if unique_ptr_type is not None:
            return unique_ptr_type

        std_container_type = self.convert_std_container_type(hip_type)
        if std_container_type is not None:
            return std_container_type

        enable_if_type = self.convert_enable_if_type(hip_type)
        if enable_if_type is not None:
            return enable_if_type

        resource_type = self.convert_hip_resource_type(hip_type)
        if resource_type is not None:
            return resource_type

        if self.has_array_suffix(hip_type):
            return self.convert_hip_array_type(hip_type, type_mapping)

        _, pointer_depth = self.split_pointer_declarators(hip_type)
        if pointer_depth:
            return self.convert_hip_pointer_type(hip_type)

        if self.is_decltype_type_name(hip_type):
            return "auto"

        return type_mapping.get(hip_type, hip_type)

    def is_decltype_type_name(self, type_name):
        return isinstance(type_name, str) and type_name.strip().startswith("decltype(")

    def strip_union_type_keyword(self, hip_type):
        if not isinstance(hip_type, str) or not hip_type.startswith("union "):
            return hip_type

        union_type = hip_type[len("union ") :].strip()
        if union_type.startswith("<anonymous>"):
            union_type = f"hip_anonymous_union{union_type[len('<anonymous>') :]}"
        return union_type

    def convert_hip_resource_type(self, hip_type):
        base_name, template_args = self.parse_cpp_template(hip_type)
        if base_name == "texture" and len(template_args) >= 2:
            return self.HIP_TEXTURE_TYPE_MAPPING.get(template_args[1])
        if base_name == "surface" and len(template_args) >= 2:
            return self.HIP_SURFACE_TYPE_MAPPING.get(template_args[1])
        return None

    def convert_unique_ptr_type(self, hip_type):
        base_name, template_args = self.parse_cpp_template(hip_type)
        if not self.is_unique_ptr_base_name(base_name) or not template_args:
            return None

        target_type, _ = self.unwrap_array_template_type(template_args[0])
        return f"ptr<{self.convert_hip_type_to_crossgl(target_type)}>"

    def convert_std_container_type(self, hip_type):
        base_name, template_args = self.parse_cpp_template(hip_type)
        if self.is_std_vector_base_name(base_name) and template_args:
            element_type = self.convert_hip_type_to_crossgl(template_args[0])
            return f"array<{element_type}>"
        if self.is_std_array_base_name(base_name) and len(template_args) >= 2:
            element_type = self.convert_hip_type_to_crossgl(template_args[0])
            size = self.format_crossgl_array_extent(template_args[1])
            return f"array<{element_type}, {size}>"
        if self.is_std_pair_base_name(base_name) and len(template_args) >= 2:
            first_type = self.convert_hip_type_to_crossgl(template_args[0])
            second_type = self.convert_hip_type_to_crossgl(template_args[1])
            return f"pair<{first_type}, {second_type}>"
        if self.is_std_tuple_base_name(base_name) and template_args:
            element_types = [
                self.convert_hip_type_to_crossgl(arg) for arg in template_args
            ]
            return f"tuple<{', '.join(element_types)}>"
        return None

    def convert_enable_if_type(self, hip_type):
        base_name, template_args = self.parse_cpp_template(hip_type)
        if base_name in {"std::enable_if_t", "::std::enable_if_t"} and template_args:
            return "void"
        return None

    def is_unique_ptr_type_name(self, type_name):
        type_name = self.strip_type_qualifiers(type_name)
        type_name = self.resolve_type_alias(type_name)
        base_name, template_args = self.parse_cpp_template(type_name)
        return self.is_unique_ptr_base_name(base_name) and bool(template_args)

    def is_unique_ptr_base_name(self, base_name):
        return self.is_std_unique_ptr_base_name(base_name)

    def is_std_unique_ptr_base_name(self, base_name):
        return base_name in {"unique_ptr", "std::unique_ptr"}

    def is_std_vector_base_name(self, base_name):
        return base_name in {"vector", "std::vector"}

    def is_std_array_base_name(self, base_name):
        return base_name in {"array", "std::array"}

    def is_std_pair_base_name(self, base_name):
        return base_name in {"pair", "std::pair"}

    def is_std_tuple_base_name(self, base_name):
        return base_name in {"tuple", "std::tuple"}

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

    def split_pointer_declarators(self, type_name):
        base_chars = []
        pointer_depth = 0
        template_depth = 0

        for char in str(type_name):
            if char == "<":
                template_depth += 1
                base_chars.append(char)
                continue
            if char == ">":
                template_depth = max(0, template_depth - 1)
                base_chars.append(char)
                continue
            if char == "*" and template_depth == 0:
                pointer_depth += 1
                continue
            base_chars.append(char)

        return "".join(base_chars).strip(), pointer_depth

    def parse_cpp_template(self, text):
        if not isinstance(text, str):
            return str(text), []

        start = text.find("<")
        if start == -1 or not text.endswith(">"):
            return text, []

        base_name = text[:start].strip()
        args = self.split_cpp_template_args(text[start + 1 : -1])
        return base_name, args

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

    def convert_hip_pointer_type(self, hip_type):
        base_type, pointer_depth = self.split_pointer_declarators(hip_type)
        base_type = self.strip_function_pointer_parameter_list(base_type)
        mapped_type = self.convert_hip_type_to_crossgl(base_type)

        for _ in range(pointer_depth):
            mapped_type = f"ptr<{mapped_type}>"

        return mapped_type

    def convert_hip_pointer_element_type(self, hip_type):
        base_type, pointer_depth = self.split_pointer_declarators(hip_type)
        base_type = self.strip_function_pointer_parameter_list(base_type)
        mapped_type = self.convert_hip_type_to_crossgl(base_type)

        for _ in range(max(0, pointer_depth - 1)):
            mapped_type = f"ptr<{mapped_type}>"

        return mapped_type

    def strip_function_pointer_parameter_list(self, type_name):
        """Keep imported C++ function-pointer types reparsable in CrossGL."""
        return re.sub(r"\s*\(\s*\)\s*$", "", str(type_name)).strip()

    def strip_type_qualifiers(self, type_name):
        qualifiers = {"const", "volatile", "__restrict__", "restrict", "&", "&&"}
        return " ".join(
            part for part in str(type_name).split() if part not in qualifiers
        )

    def strip_variadic_type_marker(self, type_name):
        return " ".join(part for part in str(type_name).split() if part != "...")

    def convert_hip_array_type(self, hip_type, type_mapping):
        base_type = hip_type.split("[", 1)[0].strip()
        dimensions = []
        remainder = hip_type[len(base_type) :].strip()

        while remainder.startswith("["):
            close_index = remainder.find("]")
            if close_index == -1:
                break
            dimensions.append(remainder[1:close_index].strip())
            remainder = remainder[close_index + 1 :].strip()

        mapped_type = type_mapping.get(base_type)
        if mapped_type is None:
            mapped_type = self.convert_hip_type_to_crossgl(base_type)
        for size in reversed(dimensions):
            if size:
                mapped_type = (
                    f"array<{mapped_type}, {self.format_crossgl_array_extent(size)}>"
                )
            else:
                mapped_type = f"array<{mapped_type}>"

        return mapped_type

    def convert_hip_builtin_function(self, func_name):
        function_mapping = {
            # Math functions
            "sqrtf": "sqrt",
            "powf": "pow",
            "sinf": "sin",
            "cosf": "cos",
            "tanf": "tan",
            "sinhf": "sinh",
            "coshf": "cosh",
            "tanhf": "tanh",
            "asinhf": "asinh",
            "acoshf": "acosh",
            "atanhf": "atanh",
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
            "truncf": "trunc",
            "atan2f": "atan2",
            "fmaf": "fma",
            "__fmaf_rn": "fma",
            "__fma_rn": "fma",
            "fmodf": "mod",
            "fminf": "min",
            "fmaxf": "max",
            "lerp": "mix",
            "floorf": "floor",
            "ceilf": "ceil",
            # Double precision variants
            "sqrt": "sqrt",
            "pow": "pow",
            "sin": "sin",
            "cos": "cos",
            "tan": "tan",
            "log": "log",
            "exp": "exp",
            "fabs": "abs",
            "rsqrt": "inversesqrt",
            "fmod": "mod",
            "fmin": "min",
            "fmax": "max",
            "floor": "floor",
            "ceil": "ceil",
            "bool": "bool",
            "char": "i8",
            "short": "i16",
            "int": "i32",
            "uint": "u32",
            "long": "i64",
            "__int64": "i64",
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
            "float": "f32",
            "double": "f64",
            "size_t": "u32",
            # Vector functions
            **self.VECTOR_CONSTRUCTOR_MAPPING,
            "dim3": "vec3<u32>",
            # Sync functions
            "__syncthreads": "workgroupBarrier",
            "__threadfence": "memoryBarrier",
            "__threadfence_block": "memoryBarrier",
            "__threadfence_system": "memoryBarrier",
            # Population count intrinsics
            "__builtin_popcount": "bitCount",
            "__builtin_popcountl": "bitCount",
            "__builtin_popcountll": "bitCount",
            "__popcnt16": "bitCount",
            "__popcnt": "bitCount",
            "__popcnt64": "bitCount",
            # Atomic functions
            "atomicAdd": "atomicAdd",
            "hipAtomicAdd": "atomicAdd",
            "atomicSub": "atomicSub",
            "hipAtomicSub": "atomicSub",
            "atomicMax": "atomicMax",
            "hipAtomicMax": "atomicMax",
            "atomicMin": "atomicMin",
            "hipAtomicMin": "atomicMin",
            "atomicExch": "atomicExchange",
            "hipAtomicExch": "atomicExchange",
            "atomicCAS": "atomicCompareExchange",
            "hipAtomicCAS": "atomicCompareExchange",
            "atomicAnd": "atomicAnd",
            "hipAtomicAnd": "atomicAnd",
            "atomicOr": "atomicOr",
            "hipAtomicOr": "atomicOr",
            "atomicXor": "atomicXor",
            "hipAtomicXor": "atomicXor",
            "atomicInc": "atomicInc",
            "hipAtomicInc": "atomicInc",
            "atomicDec": "atomicDec",
            "hipAtomicDec": "atomicDec",
        }

        normalized_func_name = self.normalize_hip_builtin_function_name(func_name)
        matrix_constructor = self.convert_native_matrix_helper_name_to_crossgl(
            normalized_func_name
        )
        if matrix_constructor is not None:
            return matrix_constructor
        return function_mapping.get(normalized_func_name, func_name)

    def normalize_hip_builtin_function_name(self, func_name):
        if not isinstance(func_name, str):
            return func_name

        normalized_name = func_name
        if normalized_name.startswith("::"):
            normalized_name = normalized_name[2:]

        for namespace in ("cuda::std::", "std::"):
            if normalized_name.startswith(namespace):
                return normalized_name[len(namespace) :]

        return normalized_name

    def visit_EnumNode(self, node):
        name = node.name or self.next_anonymous_enum_name()
        underlying = getattr(node, "underlying_type", None)
        suffix = (
            f" : {self.convert_hip_type_to_crossgl(underlying)}" if underlying else ""
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
                member_name = self.sanitize_enum_member_name(member_name)
                self.emit(f"{member_name} = {value},")
            else:
                self.emit(f"{self.sanitize_enum_member_name(member_name)},")

        self.indent_level -= 1
        self.emit("};")

    def sanitize_enum_member_name(self, name):
        return self.sanitize_identifier_name(name)

    def next_anonymous_enum_name(self):
        name = f"anonymous_enum_{self.anonymous_enum_count}"
        self.anonymous_enum_count += 1
        return name

    # Legacy method for backwards compatibility
    def convert(self, node):
        """Legacy convert method for compatibility"""
        return self.generate(node)


def hip_to_crossgl(hip_ast) -> str:
    """Convert HIP AST to CrossGL code string"""
    converter = HipToCrossGLConverter()
    return converter.generate(hip_ast)

"""Values and spellings in the Mojo generator: numeric vector operands, casts, literals, identifiers and type, operator and semantic mapping."""

import re

from ..ast import (
    ArrayAccessNode,
    AtomicOpNode,
    BinaryOpNode,
    BufferOpNode,
    CastNode,
    FunctionCallNode,
    MemberAccessNode,
    TernaryOpNode,
    TextureOpNode,
    UnaryOpNode,
)
from ..refusals import MojoRefusal
from .char_literals import char_literal_code
from .lowering_refusals import refuse_unsupported
from .mojo_exact_lookups import (
    mojo_generic_template_reason,
    mojo_operator,
)
from .mojo_tables import (
    MOJO_DTYPE_INFO,
    MOJO_MATRIX_TYPES,
    MOJO_RESOURCE_TYPE_MAPPING,
    MOJO_VECTOR_ARITHMETIC_OPS,
)


class MojoValueMixin:
    """Value members of ``MojoCodeGen``."""

    def generate_numeric_vector_binary_op(self, expr, target_type, target_context=None):
        op = self.map_operator(expr.op)
        if op not in MOJO_VECTOR_ARITHMETIC_OPS or target_type is None:
            return None

        target_info = self.vector_type_info(target_type)
        if target_info is None or target_info[0] == "DType.bool":
            return None

        left_type = self.expression_result_type(expr.left)
        right_type = self.expression_result_type(expr.right)
        left_info = self.vector_type_info(left_type)
        right_info = self.vector_type_info(right_type)
        if left_info is None and right_info is None:
            return None
        if left_info is not None and not self.compatible_numeric_vector_target(
            left_info, target_info
        ):
            return None
        if right_info is not None and not self.compatible_numeric_vector_target(
            right_info, target_info
        ):
            return None

        target_dtype = target_info[0]
        context = target_context or "vector binary expression"
        left = self.generate_numeric_vector_binary_operand(
            expr.left,
            target_dtype,
            context,
        )
        right = self.generate_numeric_vector_binary_operand(
            expr.right,
            target_dtype,
            context,
        )
        return f"({left} {op} {right})"

    def generate_numeric_vector_binary_operand(self, expr, target_dtype, context):
        expr_info = self.vector_type_info(self.expression_result_type(expr))
        if expr_info is not None:
            expr_text = self.generate_expression(expr)
            source_dtype = expr_info[0]
            if source_dtype == target_dtype:
                return expr_text
            return f"({expr_text}).cast[{target_dtype}]()"

        scalar_type = MOJO_DTYPE_INFO[target_dtype][0]
        return self.generate_constructor_scalar_expression(
            expr,
            target_dtype,
            f"{context} scalar operand as {scalar_type}",
        )

    def is_literal_expression(self, expr):
        return hasattr(expr, "__class__") and "Literal" in str(expr.__class__)

    def bool_cast_text(self, expr_text, target_dtype):
        # Mojo's Bool has no .cast method, so a bool value going to a numeric
        # type goes through Int first: Float32(Int(s)). Returns None when the
        # target has no scalar constructor form here, leaving the old output.
        constructor = {
            "DType.float16": "Float16",
            "DType.float32": "Float32",
            "DType.float64": "Float64",
            "DType.int8": "Int8",
            "DType.int16": "Int16",
            "DType.int32": "Int32",
            "DType.uint8": "UInt8",
            "DType.uint16": "UInt16",
            "DType.uint32": "UInt32",
        }.get(target_dtype)
        if constructor is None:
            return None
        return f"{constructor}(Int({expr_text}))"

    def cast_scalar_text(self, expr_text, source_dtype, target_dtype):
        if target_dtype is None or source_dtype is None or source_dtype == target_dtype:
            return expr_text
        if source_dtype == "DType.bool":
            bool_cast = self.bool_cast_text(expr_text, target_dtype)
            if bool_cast is not None:
                return bool_cast
        return f"({expr_text}).cast[{target_dtype}]()"

    def cast_vector_component(self, component, source_dtype, target_dtype):
        if target_dtype is None or source_dtype is None or source_dtype == target_dtype:
            return component
        if source_dtype == "DType.bool":
            bool_cast = self.bool_cast_text(component, target_dtype)
            if bool_cast is not None:
                return bool_cast
        return f"{component}.cast[{target_dtype}]()"

    def generate_constructor_scalar_expression(
        self, expr, target_dtype, target_context=None
    ):
        target_type = self.constructor_scalar_expression_target_type(expr, target_dtype)
        if target_context is not None:
            self.validate_expression_target_shape(expr, target_type, target_context)
            expr_text = self.generate_expression(expr, target_type, target_context)
        else:
            expr_text = self.generate_expression(expr)
        if self.is_literal_expression(expr):
            return expr_text
        return self.cast_scalar_text(
            expr_text, self.expression_mojo_dtype(expr), target_dtype
        )

    def vector_components_for_expression(self, expr, target_dtype=None):
        if isinstance(expr, MemberAccessNode):
            obj = self.generate_expression(expr.object)
            swizzle_indices = self.get_swizzle_indices(expr.member)
            if swizzle_indices is not None:
                source_info = self.vector_type_info(
                    self.expression_result_type(expr.object)
                )
                source_dtype = source_info[0] if source_info is not None else None
                return [
                    self.cast_vector_component(
                        f"{obj}[{index}]", source_dtype, target_dtype
                    )
                    for index in swizzle_indices
                ]

        expr_type = self.expression_result_type(expr)
        info = self.vector_type_info(expr_type)
        if info is None:
            return None

        source_dtype, source_width, _, _ = info
        if source_width <= 1:
            return None

        expr_text = self.generate_expression(expr)
        return [
            self.cast_vector_component(
                f"{expr_text}[{index}]", source_dtype, target_dtype
            )
            for index in range(source_width)
        ]

    def generate_swizzle(self, source_expr, obj, obj_type, member, swizzle_indices):
        if len(swizzle_indices) == 1:
            return f"{obj}[{swizzle_indices[0]}]"

        info = self.vector_type_info(obj_type)
        dtype = info[0] if info else "DType.float32"
        source_width = info[2] if info else 4
        if info is not None and self.is_duplicate_sensitive_expression(source_expr):
            helper_name = self.swizzle_helper_name(dtype, source_width, member)
            self.required_swizzle_helpers.add((dtype, source_width, member))
            return f"{helper_name}({obj})"

        _, _, pad_literal = MOJO_DTYPE_INFO.get(dtype, MOJO_DTYPE_INFO["DType.float32"])
        storage_width = 2 if len(swizzle_indices) == 2 else 4
        components = [f"{obj}[{index}]" for index in swizzle_indices]
        if len(swizzle_indices) == 3:
            components.append(pad_literal)

        return f"SIMD[{dtype}, {storage_width}]({', '.join(components)})"

    def is_duplicate_sensitive_expression(self, expr):
        if isinstance(
            expr,
            (
                FunctionCallNode,
                BinaryOpNode,
                TernaryOpNode,
                BufferOpNode,
                TextureOpNode,
                AtomicOpNode,
            ),
        ):
            return True
        if isinstance(expr, CastNode):
            return self.is_duplicate_sensitive_expression(expr.expression)
        if isinstance(expr, UnaryOpNode):
            return self.is_duplicate_sensitive_expression(expr.operand)
        if isinstance(expr, MemberAccessNode):
            return self.is_duplicate_sensitive_expression(expr.object)
        if isinstance(expr, ArrayAccessNode):
            return self.is_duplicate_sensitive_expression(
                expr.array
            ) or self.is_duplicate_sensitive_expression(expr.index)
        return False

    def format_literal(self, value, literal_type=None):
        if isinstance(value, bool):
            return "True" if value else "False"
        if literal_type == "bool" and isinstance(value, str):
            lower_value = value.lower()
            if lower_value == "true":
                return "True"
            if lower_value == "false":
                return "False"
        if literal_type == "char":
            return f"Int8({char_literal_code(value, MojoRefusal, 'Mojo')})"
        if isinstance(value, str):
            escaped = self.escape_literal(value)
            return f'"{escaped}"'
        return str(value)

    def escape_literal(self, value):
        text = str(value)
        escaped = []
        for index, char in enumerate(text):
            if char == "\n":
                escaped.append("\\n")
            elif char == "\r":
                escaped.append("\\r")
            elif char == "\t":
                escaped.append("\\t")
            elif char == '"' and (index == 0 or text[index - 1] != "\\"):
                escaped.append('\\"')
            else:
                escaped.append(char)
        return "".join(escaped)

    def mojo_identifier(self, name):
        """Return a Mojo-safe value or field identifier for source names."""
        name = str(name)
        if not name.isidentifier():
            return name
        if name in self.mojo_reserved_identifiers():
            return f"{name}_"
        return name

    def mojo_reserved_identifiers(self):
        return {
            "False",
            "None",
            "Self",
            "True",
            "alias",
            "and",
            "as",
            "assert",
            "async",
            "await",
            "borrowed",
            "break",
            "case",
            "class",
            "comptime",
            "constrained",
            "continue",
            "def",
            "elif",
            "else",
            "except",
            "finally",
            "fn",
            "for",
            "from",
            "global",
            "if",
            "import",
            "in",
            "inout",
            "is",
            "lambda",
            "let",
            "match",
            "mut",
            "nonlocal",
            "not",
            "or",
            "owned",
            "pass",
            "raise",
            "raises",
            "ref",
            "return",
            "self",
            "static",
            "struct",
            "trait",
            "try",
            "type",
            "var",
            "where",
            "while",
            "with",
            "yield",
        }

    def map_type(self, vtype):
        """Map a CrossGL type name or type node to a Mojo type string."""
        if vtype is None:
            return "Float32"

        if (
            hasattr(vtype, "name")
            or hasattr(vtype, "element_type")
            or hasattr(vtype, "pointee_type")
            or hasattr(vtype, "referenced_type")
        ):
            vtype_str = self.convert_type_node_to_string(vtype)
        else:
            vtype_str = str(vtype)

        pointee_type = self.parse_pointer_type_name(vtype_str)
        if pointee_type is not None:
            # Mojo 1.2 cannot infer a pointer's origin from a declaration; MutAnyOrigin is the origin a kernel's
            # buffer arguments carry and allows both reads and writes.
            return f"UnsafePointer[{self.map_type(pointee_type)}, MutAnyOrigin]"

        geometry_stream_type = self.geometry_stream_mapped_type(vtype_str)
        if geometry_stream_type is not None:
            return geometry_stream_type

        tessellation_patch_type = self.tessellation_patch_mapped_type(vtype_str)
        if tessellation_patch_type is not None:
            return tessellation_patch_type

        if self.is_array_type_name(vtype_str):
            base_type, size = self.parse_array_type_name(vtype_str)
            base_mapped = self.map_type(base_type)
            if size:
                return f"Array[{base_mapped}, {size}]"
            else:
                return f"List[{base_mapped}]"

        vtype_str = self.normalize_generic_vector_type_name(vtype_str)

        resource_alias = self.resource_type_alias(vtype_str)
        if resource_alias is not None:
            self.required_resource_types.add(resource_alias)
            return resource_alias

        buffer_info = self.buffer_resource_info(vtype_str)
        if buffer_info is not None:
            self.register_buffer_resource_type(buffer_info[0])
            return self.mapped_buffer_type(*buffer_info)

        if vtype_str in MOJO_MATRIX_TYPES:
            dtype, columns, rows = MOJO_MATRIX_TYPES[vtype_str]
            self.required_matrix_types.add((dtype, columns, rows))
            return self.matrix_type_name(dtype, columns, rows)

        generic_struct_type = self.generic_struct_mapped_type(vtype_str)
        if generic_struct_type is not None:
            return generic_struct_type

        generic_enum_type = self.generic_enum_mapped_type(vtype_str)
        if generic_enum_type is not None:
            return generic_enum_type

        if vtype_str in self.enum_types:
            return vtype_str

        mapped_type = self.type_mapping.get(vtype_str)
        if mapped_type is not None:
            if self.is_mojo_resource_type(mapped_type):
                self.required_resource_types.add(mapped_type)
            return mapped_type

        mojo_generic_type = self.mojo_native_generic_mapped_type(vtype_str)
        if mojo_generic_type is not None:
            return mojo_generic_type

        mapped_type = vtype_str
        if self.is_mojo_resource_type(mapped_type):
            self.required_resource_types.add(mapped_type)
        return mapped_type

    def mojo_native_generic_mapped_type(self, type_name):
        generic = self.parse_generic_type_name(type_name)
        if generic is None:
            return None

        base_type, generic_args = generic
        mapped_args = ", ".join(
            self.mojo_native_generic_argument(arg) for arg in generic_args
        )
        mapped_base = self.type_mapping.get(base_type)
        if mapped_base is None:
            reason = mojo_generic_template_reason(base_type)
            if reason is not None:
                return refuse_unsupported("Mojo", "type", reason, type_name)
            # A Mojo parameterised type (UnsafePointer, SIMD, Scalar) is spelled as written.
            mapped_base = base_type
        return f"{mapped_base}[{mapped_args}]"

    def mojo_native_generic_argument(self, arg):
        arg = str(arg).strip()
        nested_generic = self.mojo_native_generic_mapped_type(arg)
        if nested_generic is not None:
            return nested_generic
        mapped = self.type_mapping.get(arg)
        if mapped is not None:
            return mapped
        # A Mojo parameter value (Int64, MutAnyOrigin, a DType, a literal width) is spelled as written.
        return arg

    def is_resource_type_name(self, type_name):
        if self.is_array_type_name(type_name):
            base_type, _ = self.parse_array_type_name(type_name)
            return self.is_resource_type_name(base_type)
        if self.buffer_resource_info(type_name) is not None:
            return True
        if self.resource_type_alias(type_name) is not None:
            return True
        return self.is_mojo_resource_type(
            self.type_mapping.get(str(type_name), type_name)
        )

    def is_mojo_resource_type(self, type_name):
        return (
            type_name in set(MOJO_RESOURCE_TYPE_MAPPING.values())
            or self.typed_image_resource_info(type_name) is not None
            or self.typed_texture_resource_info(type_name) is not None
        )

    def map_operator(self, op):
        """``op`` spelled for Mojo; an operator CrossGL does not define is refused."""
        return mojo_operator(op)

    def map_semantic(self, semantic):
        """Map a CrossGL semantic to the Mojo backend attribute name."""
        if semantic:
            semantic = str(semantic)
            if semantic in self.semantic_map:
                return self.semantic_map[semantic]
            upper = semantic.upper()
            lower = semantic.lower()
            if upper in self.semantic_map:
                return self.semantic_map[upper]
            target_match = re.fullmatch(r"SV_TARGET(\d*)", upper)
            if target_match is not None:
                index = target_match.group(1) or "0"
                return f"color({index})"
            color_match = re.fullmatch(r"gl_fragcolor(\d*)", lower)
            if color_match is not None:
                index = color_match.group(1) or "0"
                return f"color({index})"
            texcoord_match = re.fullmatch(r"TEXCOORD(\d*)", upper)
            if texcoord_match is not None:
                index = texcoord_match.group(1)
                return f"texcoord{index}" if index else "texcoord"
            color_semantic_match = re.fullmatch(r"COLOR(\d*)", upper)
            if color_semantic_match is not None:
                index = color_semantic_match.group(1)
                return f"color{index}" if index else "color"
            if lower == "gl_fragdepth" or upper == "SV_DEPTH":
                return "depth(any)"
            if lower == "gl_position" or upper == "SV_POSITION":
                return "position"
            return semantic
        return ""

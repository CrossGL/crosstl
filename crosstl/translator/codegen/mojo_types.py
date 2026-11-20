"""Type names and result types in the Mojo generator: arrays, pointers, zero values, overload selection and expression result types."""

import re

from ..ast import (
    ArrayLiteralNode,
    BinaryOpNode,
    MatchNode,
    MemberAccessNode,
)
from .array_utils import evaluate_literal_int_expression
from .mojo_integer_promotion import promoted_type
from .mojo_tables import (
    MOJO_ATOMIC_OP_ALIASES,
    MOJO_BUFFER_OP_ALIASES,
    MOJO_BUFFER_STORE_RESOURCE_TYPES,
    MOJO_BYTE_ADDRESS_BUFFER_TYPES,
    MOJO_BYTE_ADDRESS_LOAD_METHODS,
    MOJO_BYTE_ADDRESS_STORE_METHODS,
    MOJO_DTYPE_INFO,
    MOJO_DTYPE_SUFFIX,
    MOJO_INTEGER_INDEX_TYPES,
    MOJO_MATRIX_TYPES,
    MOJO_SCALAR_DTYPES,
    MOJO_TYPED_BUFFER_LOAD_RESOURCE_TYPES,
    MOJO_VECTOR_ARITHMETIC_OPS,
    MOJO_VECTOR_COMPARISON_METHODS,
    SWIZZLE_SETS,
)


class MojoTypeMixin:
    """Type members of ``MojoCodeGen``."""

    def register_variable_type(self, name, var_type):
        if name and var_type:
            self.variable_types[name] = self.type_name(var_type)

    def type_name(self, type_value):
        if (
            hasattr(type_value, "name")
            or hasattr(type_value, "element_type")
            or hasattr(type_value, "pointee_type")
            or hasattr(type_value, "referenced_type")
        ):
            return self.convert_type_node_to_string(type_value)
        return str(type_value)

    def type_name_string(self, type_value):
        if type_value is None:
            return None
        return self.type_name(type_value)

    def identifier_replacement_result_type(self, name):
        replacement = self.expression_identifier_replacements.get(name)
        if replacement is None:
            return None
        return self.expression_path_result_type(replacement)

    def expression_path_result_type(self, expression):
        if not isinstance(expression, str) or not re.fullmatch(
            r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*", expression
        ):
            return None

        parts = expression.split(".")
        result_type = self.variable_types.get(parts[0])
        for member in parts[1:]:
            _struct_type, fields = self.struct_type_name_and_fields(result_type)
            if member not in fields:
                return None
            result_type = fields[member]
        return result_type

    def normalize_generic_vector_type_name(self, type_name):
        if not isinstance(type_name, str):
            return type_name
        match = re.fullmatch(r"vec([234])<\s*([^>]+)\s*>", type_name)
        if match is None:
            return type_name
        dtype = MOJO_SCALAR_DTYPES.get(match.group(2).strip())
        if dtype is None:
            return type_name
        canonical = f"vec{match.group(1)}<{MOJO_DTYPE_SUFFIX[dtype]}>"
        if canonical in self.vector_constructor_info:
            return canonical
        return type_name

    def is_array_type_name(self, type_name):
        return self.array_type_suffix(type_name) is not None

    def array_type_suffix(self, type_name):
        type_text = str(type_name or "").strip()
        if not type_text.endswith("]"):
            return None

        open_bracket = type_text.rfind("[")
        if open_bracket <= 0:
            return None

        size_text = type_text[open_bracket + 1 : -1].strip()
        if (
            not size_text
            or re.fullmatch(r"\d+", size_text)
            or re.fullmatch(r"[A-Z_][A-Z0-9_]*", size_text)
        ):
            return open_bracket, size_text
        return None

    def is_pointer_type_name(self, type_name):
        type_text = str(type_name or "").strip()
        return type_text.endswith("*")

    def parse_pointer_type_name(self, type_name):
        type_text = str(type_name or "").strip()
        if not self.is_pointer_type_name(type_text):
            return None
        return type_text[:-1].strip()

    def parse_array_type_name(self, type_name):
        type_text = str(type_name)
        array_suffix = self.array_type_suffix(type_text)
        if array_suffix is None:
            return type_text, None

        open_bracket, size_text = array_suffix
        element_type = type_text[:open_bracket]
        if not size_text:
            return element_type, None

        try:
            return element_type, int(size_text)
        except ValueError:
            return element_type, size_text

    def is_struct_type_name(self, type_name):
        if type_name is None:
            return False
        resolved_type = self.type_name(type_name)
        struct_type, _ = self.struct_type_name_and_fields(resolved_type)
        if struct_type is not None:
            return True
        return False

    def array_type_name(self, element_type, size):
        element_type_name = self.type_name(element_type)
        size = self.format_array_size(size)
        if size is None:
            return f"{element_type_name}[]"
        return f"{element_type_name}[{size}]"

    def array_storage_type(self, element_type, size):
        element_type_name = self.map_type(element_type)
        size = self.format_array_size(size)
        if size is None:
            return f"List[{element_type_name}]"
        return f"Array[{element_type_name}, {size}]"

    def array_initial_value(self, element_type, size):
        array_type = self.array_storage_type(element_type, size)
        if size is None:
            return f"{array_type}()"
        if self.should_zero_initialize_array_elements(element_type):
            return self.zero_array_value(element_type, size)
        return f"{array_type}(uninitialized=True)"

    def array_initial_value_for_type(self, type_name):
        element_type, size = self.parse_array_type_name(type_name)
        return self.array_initial_value(element_type, size)

    def should_zero_initialize_array_elements(self, element_type):
        return self.is_struct_type_name(element_type) or self.is_resource_type_name(
            element_type
        )

    def array_element_type(self, type_name):
        if not self.is_array_type_name(type_name):
            return None
        element_type, _ = self.parse_array_type_name(type_name)
        return element_type

    def pointer_element_type(self, type_name):
        return self.parse_pointer_type_name(type_name)

    def generate_array_literal_expression(self, expr, target_type=None):
        if target_type is not None and self.is_array_type_name(target_type):
            element_type, size = self.parse_array_type_name(target_type)
        else:
            element_type = self.infer_array_literal_element_type(expr)
            size = len(expr.elements)

        array_type = self.array_storage_type(element_type, size)
        elements = [
            self.generate_array_literal_element(
                element, element_type, f"array literal element {index}"
            )
            for index, element in enumerate(expr.elements, start=1)
        ]

        if size is not None:
            size = int(size)
            elements = elements[:size]
            while len(elements) < size:
                elements.append(self.zero_value_for_type(element_type))

        if array_type.startswith("Array[") and elements:
            # Mojo 1.2's Array has no positional element initializer; a list literal [a, b] is lowered to this
            # one, which also works where no declared type gives the literal its element type and length.
            return f"{array_type}({', '.join(elements)}, __list_literal__=None)"
        return f"{array_type}({', '.join(elements)})"

    def infer_array_literal_element_type(self, expr):
        if not expr.elements:
            return "float"
        return self.expression_result_type(expr.elements[0]) or "float"

    def generate_array_literal_element(
        self, element, element_type, context="array literal element"
    ):
        self.validate_expression_target_shape(element, element_type, context)
        if isinstance(element, ArrayLiteralNode) and self.is_array_type_name(
            element_type
        ):
            return self.generate_array_literal_expression(element, element_type)
        if isinstance(element, MatchNode):
            return self.generate_expression(element, element_type, context)
        target_dtype = MOJO_SCALAR_DTYPES.get(self.type_name(element_type))
        if target_dtype is not None:
            return self.generate_constructor_scalar_expression(element, target_dtype)
        return self.generate_expression(element, element_type, context)

    def zero_value_for_type(self, type_name):
        type_name = self.type_name(type_name)
        if self.is_pointer_type_name(type_name):
            return f"{self.map_type(type_name)}()"
        buffer_info = self.buffer_resource_info(type_name)
        if buffer_info is not None:
            return f"{self.mapped_buffer_type(*buffer_info)}()"

        if self.is_array_type_name(type_name):
            element_type, size = self.parse_array_type_name(type_name)
            return self.zero_array_value(element_type, size)

        struct_type, fields = self.struct_type_name_and_fields(type_name)
        if struct_type is not None:
            return self.zero_struct_value(
                struct_type, fields=fields, mapped_type=self.map_type(type_name)
            )

        vector_info = self.vector_type_info(type_name)
        if vector_info is not None:
            dtype, source_width, storage_width, pad_literal = vector_info
            zero = MOJO_DTYPE_INFO[dtype][2]
            components = [zero] * source_width
            if pad_literal is not None and len(components) == 3:
                components.append(pad_literal)
            return f"SIMD[{dtype}, {storage_width}]({', '.join(components)})"

        matrix_info = self.matrix_type_info(type_name)
        if matrix_info is not None:
            dtype, columns, rows = matrix_info
            return self.zero_matrix_value(dtype, columns, rows)

        dtype = MOJO_SCALAR_DTYPES.get(type_name)
        if dtype is not None:
            return MOJO_DTYPE_INFO[dtype][2]
        return f"{self.map_type(type_name)}()"

    def zero_array_value(self, element_type, size):
        array_type = self.array_storage_type(element_type, size)
        if size is None:
            return f"{array_type}()"

        try:
            element_count = int(size)
        except (TypeError, ValueError):
            return f"{array_type}(uninitialized=True)"

        if element_count <= 0:
            return f"{array_type}(uninitialized=True)"
        # Mojo 1.2's Array fills every element from one value.
        return f"{array_type}(fill={self.zero_value_for_type(element_type)})"

    def zero_struct_value(self, type_name, fields=None, mapped_type=None):
        fields = fields if fields is not None else self.struct_types.get(type_name, {})
        values = [
            self.zero_value_for_type(field_type) for field_type in fields.values()
        ]
        return f"{mapped_type or type_name}({', '.join(values)})"

    def zero_matrix_value(self, dtype, columns, rows):
        self.required_matrix_types.add((dtype, columns, rows))
        matrix_type = self.matrix_type_name(dtype, columns, rows)
        storage_rows = self.matrix_storage_rows(rows)
        zero = self.matrix_zero_literal(dtype)
        column_type = f"SIMD[{dtype}, {storage_rows}]"
        column_values = []
        for _ in range(columns):
            components = [zero] * rows
            if rows == 3:
                components.append(zero)
            column_values.append(f"{column_type}({', '.join(components)})")
        return f"{matrix_type}({', '.join(column_values)})"

    def generate_array_access_expression(self, expr):
        array_type = self.expression_result_type(expr.array)
        matrix_info = self.matrix_type_info(array_type)
        vector_info = self.vector_type_info(array_type)
        array_element_type = self.array_element_type(array_type)
        pointer_element_type = self.pointer_element_type(array_type)
        array = self.generate_expression(expr.array)
        index = self.generate_array_index_expression(
            expr.index,
            cast_integer_index=matrix_info is not None
            or vector_info is not None
            or array_element_type is not None
            or pointer_element_type is not None,
            preserve_integer_index_types=(
                {"int", "i32", "Int32"} if matrix_info is not None else None
            ),
        )

        if matrix_info is not None:
            column_index = self.literal_int_value(expr.index)
            if column_index is not None:
                return f"{array}.c{column_index}"

        return f"{array}[{index}]"

    def generate_array_index_expression(
        self,
        expr,
        cast_integer_index=False,
        preserve_integer_index_types=None,
    ):
        index = self.generate_expression(expr)
        if not cast_integer_index or self.literal_int_value(expr) is not None:
            return index

        index_type = self.expression_result_type(expr)
        if preserve_integer_index_types and index_type in preserve_integer_index_types:
            return index
        if index_type in MOJO_INTEGER_INDEX_TYPES:
            return f"Int({index})"
        return index

    def literal_int_value(self, expr):
        return evaluate_literal_int_expression(
            expr, getattr(self, "literal_int_constants", {})
        )

    def function_call_name(self, expr):
        func_expr = getattr(expr, "function", None)
        if func_expr is None:
            func_expr = expr.name
        if isinstance(func_expr, MemberAccessNode):
            return None
        if hasattr(func_expr, "name"):
            return func_expr.name
        if isinstance(func_expr, str):
            return func_expr
        return None

    def binary_expression_result_type(self, op, left_type, right_type):
        comparison = self.vector_comparison_info(op, left_type, right_type)
        if comparison is not None:
            return self.vector_type_name_for_dtype_width("DType.bool", comparison[1])
        left_info = self.vector_type_info(left_type)
        right_info = self.vector_type_info(right_type)
        mapped_op = self.map_operator(op)
        if left_info is not None and right_info is not None:
            if left_info == right_info:
                return left_type
            if (
                mapped_op in MOJO_VECTOR_ARITHMETIC_OPS
                and left_info[0] == right_info[0]
                and left_info[2] == right_info[2]
                and left_info[0] != "DType.bool"
            ):
                return self.vector_type_name_for_dtype_width(
                    left_info[0], max(left_info[1], right_info[1])
                )
            return left_type
        if left_info is not None:
            return left_type
        if right_info is not None:
            return right_type
        if mapped_op in MOJO_VECTOR_COMPARISON_METHODS or mapped_op in {"and", "or"}:
            # A comparison or logical operation on scalars yields a ``Bool``,
            # not a value of its operand type.
            return "bool"
        left_type = promoted_type(self, mapped_op, left_type)
        right_type = promoted_type(self, mapped_op, right_type)
        return left_type if left_type == right_type else left_type or right_type

    def selected_function_return_type(self, func_name, args):
        signature = self.select_function_overload_signature(func_name, args)
        if signature is not None:
            return signature["return_type"]
        return self.function_return_types.get(func_name)

    def selected_function_parameter_types(self, func_name, args):
        signature = self.select_function_overload_signature(func_name, args)
        if signature is not None:
            return signature["parameter_types"]
        return self.function_parameter_types.get(func_name, [])

    def select_function_overload_signature(self, func_name, args):
        overloads = self.function_overload_signatures.get(func_name) or []
        if not overloads:
            return None

        indexed = list(enumerate(overloads))
        matching_arity = [
            item for item in indexed if len(item[1]["parameter_types"]) == len(args)
        ]
        candidates = matching_arity or indexed
        arg_types = [self.expression_result_type(arg) for arg in args]
        best = None
        for index, signature in candidates:
            score = self.function_overload_match_score(
                arg_types, signature["parameter_types"]
            )
            if score is None:
                continue
            key = (score, index)
            if best is None or key < best[0]:
                best = (key, signature)

        if best is not None:
            return best[1]
        return candidates[-1][1] if candidates else None

    def function_overload_match_score(self, arg_types, parameter_types):
        score = 0
        for arg_type, parameter_type in zip(arg_types, parameter_types):
            part_score = self.function_overload_argument_match_score(
                arg_type, parameter_type
            )
            if part_score is None:
                return None
            score += part_score
        return score + abs(len(arg_types) - len(parameter_types)) * 100

    def function_overload_argument_match_score(self, arg_type, parameter_type):
        if arg_type is None or parameter_type is None:
            return 10

        arg_name = self.type_name(arg_type)
        parameter_name = self.type_name(parameter_type)
        if arg_name == parameter_name:
            return 0

        arg_vector = self.vector_type_info(arg_name)
        parameter_vector = self.vector_type_info(parameter_name)
        if arg_vector is not None or parameter_vector is not None:
            if arg_vector is None or parameter_vector is None:
                return None
            if arg_vector == parameter_vector:
                return 0
            if self.compatible_numeric_vector_target(arg_vector, parameter_vector):
                return 1
            if self.compatible_storage_vector_target(arg_vector, parameter_vector):
                return 2
            return None

        arg_dtype = MOJO_SCALAR_DTYPES.get(arg_name)
        parameter_dtype = MOJO_SCALAR_DTYPES.get(parameter_name)
        if arg_dtype is not None or parameter_dtype is not None:
            if arg_dtype is None or parameter_dtype is None:
                return None
            if "DType.bool" in {arg_dtype, parameter_dtype}:
                return None
            return 1

        return None

    def buffer_op_result_type(self, expr):
        operation = MOJO_BUFFER_OP_ALIASES.get(getattr(expr, "operation", ""))
        if operation is None:
            return None

        info = self.buffer_resource_info(
            self.expression_result_type(getattr(expr, "buffer_expr", None))
        )
        if operation == "Load":
            if info is None:
                return None
            return self.buffer_helper_element_type(*info)
        if operation in MOJO_BYTE_ADDRESS_LOAD_METHODS:
            if info is None or info[0] not in MOJO_BYTE_ADDRESS_BUFFER_TYPES:
                return None
            return self.byte_address_load_result_type(
                MOJO_BYTE_ADDRESS_LOAD_METHODS[operation]
            )
        if operation == "Consume":
            if info is None or info[0] != "ConsumeStructuredBuffer":
                return None
            return info[1]
        if operation in {
            "Append",
            "GetDimensions",
            "Store",
            *MOJO_BYTE_ADDRESS_STORE_METHODS,
        }:
            return "void"
        return None

    def member_function_result_type(self, expr):
        func_expr = getattr(expr, "function", None)
        if not isinstance(func_expr, MemberAccessNode):
            return None

        texture_result_type = self.texture_member_function_result_type(func_expr)
        if texture_result_type is not None:
            return texture_result_type

        member = getattr(func_expr, "member", None)
        obj_expr = getattr(func_expr, "object", getattr(func_expr, "object_expr", None))
        info = self.buffer_resource_info(self.expression_result_type(obj_expr))
        if info is None:
            return None

        buffer_type, element_type = info
        if buffer_type in MOJO_BYTE_ADDRESS_BUFFER_TYPES:
            if member in MOJO_BYTE_ADDRESS_LOAD_METHODS:
                return self.byte_address_load_result_type(
                    MOJO_BYTE_ADDRESS_LOAD_METHODS[member]
                )
            if member in MOJO_BYTE_ADDRESS_STORE_METHODS or member == "GetDimensions":
                return "void"

        if member == "Load" and buffer_type in MOJO_TYPED_BUFFER_LOAD_RESOURCE_TYPES:
            return element_type
        if member == "Consume" and buffer_type == "ConsumeStructuredBuffer":
            return element_type
        if member == "Store" and buffer_type in MOJO_BUFFER_STORE_RESOURCE_TYPES:
            return "void"
        if member == "Append" and buffer_type == "AppendStructuredBuffer":
            return "void"
        if member == "GetDimensions":
            return "void"
        return None

    def texture_member_function_result_type(self, func_expr):
        member = getattr(func_expr, "member", None)
        obj_expr = getattr(func_expr, "object", getattr(func_expr, "object_expr", None))
        resource_type = self.map_type(self.expression_result_type(obj_expr))
        if not self.is_texture_resource_type(resource_type):
            if not self.is_image_resource_type(resource_type):
                return None
            if member == "Load":
                return self.image_result_type_name(resource_type)
            if member == "Store":
                return "void"
            if member in {"GetDimensions", "textureSize"}:
                return self.resource_size_result_type_name(resource_type)
            if member == "imageSamples":
                return "int"
            if member in MOJO_ATOMIC_OP_ALIASES:
                return self.image_result_type_name(resource_type)
            return None

        if member in {
            "SampleCmp",
            "SampleCmpLevel",
            "SampleCmpLevelZero",
            "SampleCmpGrad",
        }:
            return "float"
        if member in {"GetDimensions", "textureSize"}:
            return self.resource_size_result_type_name(resource_type)
        if member == "textureQueryLod":
            return "vec2"
        if member == "textureQueryLevels":
            return "int"
        if member == "textureSamples":
            return "int"
        if member in {"CalculateLevelOfDetail", "CalculateLevelOfDetailUnclamped"}:
            return "float"
        return self.texture_result_type_name(resource_type)

    def resource_size_result_type_name(self, resource_type):
        size_type = self.resource_size_return_type(resource_type)
        if size_type == "Int32":
            return "int"
        if size_type == "SIMD[DType.int32, 2]":
            return "ivec2"
        if size_type == "SIMD[DType.int32, 4]":
            return "ivec3"
        return None

    def image_result_type_name(self, resource_type):
        typed_info = self.typed_image_resource_info(resource_type)
        if typed_info is not None:
            _, dtype, source_width = typed_info
            if source_width == 1:
                return MOJO_DTYPE_INFO[dtype][0]
            return self.vector_type_name_for_dtype_width(dtype, source_width)

        value_type = self.image_value_type(resource_type)
        if value_type == "Int32":
            return "int"
        if value_type == "UInt32":
            return "uint"
        return "vec4"

    def vector_type_info(self, type_name):
        if type_name is None:
            return None
        normalized_type = self.normalize_generic_vector_type_name(
            self.type_name(type_name)
        )
        if normalized_type in self.vector_constructor_info:
            return self.vector_constructor_info[normalized_type]
        return None

    def matrix_type_info(self, type_name):
        if type_name in MOJO_MATRIX_TYPES:
            return MOJO_MATRIX_TYPES[type_name]
        return None

    def vector_type_name_for_dtype_width(self, dtype, width):
        _, prefix, _ = MOJO_DTYPE_INFO[dtype]
        return f"{prefix}{width}"

    def swizzle_result_type(self, obj_type, component_count):
        info = self.vector_type_info(obj_type)
        dtype = info[0] if info else "DType.float32"
        scalar_type, prefix, _ = MOJO_DTYPE_INFO.get(
            dtype, MOJO_DTYPE_INFO["DType.float32"]
        )
        if component_count == 1:
            return scalar_type
        return f"{prefix}{component_count}"

    def get_swizzle_indices(self, member):
        if not member:
            return None
        for components in SWIZZLE_SETS.values():
            if all(component in components for component in member):
                return [components[component] for component in member]
        return None

    def expression_mojo_dtype(self, expr):
        expr_type = self.expression_result_type(expr)
        info = self.vector_type_info(expr_type)
        if info is not None:
            return info[0]
        return MOJO_SCALAR_DTYPES.get(expr_type)

    def is_scalar_bool_expression(self, expr):
        """Return whether ``expr`` evaluates to a scalar boolean.

        Scalar comparisons and logical expressions are recognized directly as
        their result type is inferred from their operands on some codegen paths.
        Vector comparisons and logical operations remain SIMD masks.
        """
        expr_type = self.expression_result_type(expr)
        if self.is_bool_scalar_type(expr_type):
            return True

        if isinstance(expr, BinaryOpNode):
            op = self.map_operator(expr.op)
            if op in MOJO_VECTOR_COMPARISON_METHODS or op in {"and", "or"}:
                return (
                    self.vector_type_info(self.expression_result_type(expr.left))
                    is None
                    and self.vector_type_info(self.expression_result_type(expr.right))
                    is None
                )

        return False

    def compatible_numeric_vector_target(self, source_info, target_info):
        if source_info is None or target_info is None:
            return False
        source_dtype, source_width, source_storage_width, _ = source_info
        target_dtype, target_width, target_storage_width, _ = target_info
        if source_width != target_width or source_storage_width != target_storage_width:
            return False
        if "DType.bool" in {source_dtype, target_dtype}:
            return False
        return source_dtype in MOJO_DTYPE_SUFFIX and target_dtype in MOJO_DTYPE_SUFFIX

    def compatible_storage_vector_target(self, source_info, target_info):
        if source_info is None or target_info is None:
            return False
        source_dtype, source_width, source_storage_width, _ = source_info
        target_dtype, target_width, target_storage_width, _ = target_info
        if source_dtype != target_dtype or source_dtype == "DType.bool":
            return False
        if source_storage_width != target_storage_width:
            return False
        return source_width <= target_width

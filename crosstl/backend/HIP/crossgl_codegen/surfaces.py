"""Surface reads and writes, byte offsets and surface value types in the HIP to CrossGL converter."""

import re

from ....translator.integer_literals import (
    IntegerLiteralError,
    integer_literal_parts,
)
from ...assignment_operands import grouped_operand
from ..HipAst import (
    ArrayAccessNode,
    BinaryOpNode,
    CastNode,
    FunctionCallNode,
    MemberAccessNode,
    UnaryOpNode,
    VariableNode,
)


class HipSurfaceMixin:
    """Surface members of ``HipToCrossGLConverter``."""

    def format_hip_surface_read(
        self, function_name, args, dimensions, value_type, raw_args=None
    ):
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

        if output_target is not None and value_type is None and raw_args:
            dest_node = raw_args[0]
            if isinstance(dest_node, UnaryOpNode) and dest_node.op == "&":
                value_type = self.infer_hip_surface_value_type(dest_node.operand)

        coord_args = self.format_hip_surface_coordinate_args(
            args, coord_start, dimensions, value_type, raw_args, function_name
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
        raw_args=None,
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
        raw_value = raw_args[0] if raw_args else None
        if value_is_pointer and isinstance(raw_value, UnaryOpNode):
            if raw_value.op == "&":
                raw_value = raw_value.operand
        if value_type is None and raw_value is not None:
            value_type = self.infer_hip_surface_value_type(raw_value)
        surface_name = args[1]
        coord_args = self.format_hip_surface_coordinate_args(
            args, 2, dimensions, value_type, raw_args, function_name
        )
        if self.has_hip_surface_coordinate_shape_mismatch(coord_args):
            return self.format_hip_surface_coordinate_shape_diagnostic(
                function_name, None
            )
        coord = self.format_vector_constructor(f"vec{dimensions}", coord_args, "i32")
        return f"imageStore({surface_name}, {coord}, {value})"

    def format_hip_surface_coordinate_args(
        self,
        args,
        coord_start,
        dimensions,
        value_type,
        raw_args=None,
        function_name=None,
    ):
        raw_offset = (
            raw_args[coord_start]
            if raw_args is not None and len(raw_args) > coord_start
            else None
        )
        # A vector-typed x coordinate is a shape mismatch, not a byte offset: leave
        # it unscaled so the shape check below reports it instead of refusing it
        # as an unprovable multiple of the element size.
        x_rank = self.infer_texture_coordinate_rank(args[coord_start], raw_offset)
        if x_rank is not None and x_rank != 1:
            coord_args = [args[coord_start]]
            coord_args.extend(args[coord_start + 1 : coord_start + dimensions])
            return coord_args
        coord_args = [
            self.scale_hip_surface_byte_offset(
                raw_offset, value_type, function_name or "surf"
            )
        ]
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

    def divide_hip_surface_byte_offset(self, node, byte_size, value_type):
        """Divide a HIP surface x byte offset by the element size, structurally.

        Returns ``(text, precedence)`` with the texel coordinate, or ``None`` when the
        offset cannot be shown to be a multiple of the element size. ``precedence`` is
        ``"atom"`` for a single operand, ``"mul"`` for a product and ``"add"`` for a sum
        or difference, so an embedding operator knows when to group the result. Every
        operand is kept as written: terms are never combined or dropped, so a call or an
        increment inside the offset still evaluates exactly once.
        """
        if node is None:
            return None
        if isinstance(node, str):
            literal = self.parse_hip_surface_integer_literal(node)
            if literal is None:
                return None
            return self.divide_hip_surface_constant(literal, byte_size)
        if isinstance(node, CastNode):
            return self.divide_hip_surface_byte_offset(
                node.expression, byte_size, value_type
            )
        if isinstance(node, UnaryOpNode):
            if node.op not in {"-", "+"}:
                return None
            inner = self.divide_hip_surface_byte_offset(
                node.operand, byte_size, value_type
            )
            if inner is None:
                return None
            text, precedence = inner
            if node.op == "+":
                return (text, precedence)
            grouped = f"({text})" if precedence != "atom" else text
            return (f"(-{grouped})", "atom")
        if isinstance(node, FunctionCallNode):
            if node.name != "sizeof" or len(node.args) != 1:
                return None
            return self.divide_hip_surface_sizeof(node.args[0], byte_size, value_type)
        if isinstance(node, BinaryOpNode):
            if node.op == "*":
                return self.divide_hip_surface_product(node, byte_size, value_type)
            if node.op in {"+", "-"}:
                left = self.divide_hip_surface_byte_offset(
                    node.left, byte_size, value_type
                )
                right = self.divide_hip_surface_byte_offset(
                    node.right, byte_size, value_type
                )
                if left is None or right is None:
                    return None
                left_text, _ = left
                right_text, right_precedence = right
                if node.op == "-":
                    if right_precedence == "add":
                        right_text = f"({right_text})"
                    return (f"{left_text} - {right_text}", "add")
                return (f"{left_text} + {right_text}", "add")
            return None
        return None

    def divide_hip_surface_constant(self, value, byte_size):
        """Divide a constant byte count by the element size, or return None."""
        if byte_size is None or value % byte_size != 0:
            return None
        return (str(value // byte_size), "atom")

    def divide_hip_surface_sizeof(self, arg, byte_size, value_type):
        """Divide a ``sizeof`` byte count by the element size, or return None."""
        if not isinstance(arg, str):
            return None
        if value_type is not None and self.strip_type_qualifiers(
            arg
        ) == self.strip_type_qualifiers(value_type):
            return ("1", "atom")
        size = self.hip_surface_value_type_size(arg)
        if size is None:
            return None
        return self.divide_hip_surface_constant(size, byte_size)

    def divide_hip_surface_product(self, node, byte_size, value_type):
        """Divide a product by the element size, or return None.

        One side must be a byte factor: an integer literal, or ``sizeof`` of the
        element type (one element size, whatever its byte count) or of a type with a
        known size. The other side is kept as written, so calls and increments
        inside it still evaluate exactly once.
        """
        left = self.hip_surface_byte_factor(node.left, value_type)
        right = self.hip_surface_byte_factor(node.right, value_type)
        if left is not None and right is not None:
            if left[0] == "size" and right[0] == "size":
                if byte_size is None:
                    return None
                return (str(byte_size), "atom")
            if left[0] == "size":
                return (str(right[1]), "atom")
            if right[0] == "size":
                return (str(left[1]), "atom")
            return self.divide_hip_surface_constant(left[1] * right[1], byte_size)
        if left is not None:
            return self.divide_hip_surface_product_factor(left, node.right, byte_size)
        if right is not None:
            return self.divide_hip_surface_product_factor(right, node.left, byte_size)
        return None

    def divide_hip_surface_product_factor(self, factor, other, byte_size):
        """Divide ``other * factor`` by the element size, or return None."""
        other_text = grouped_operand(self, other)
        if factor[0] == "size":
            return (other_text, "atom")
        count = factor[1]
        if byte_size is None or count % byte_size != 0:
            return None
        quotient = count // byte_size
        if quotient == 1:
            return (other_text, "atom")
        return (f"{other_text} * {quotient}", "mul")

    def hip_surface_byte_factor(self, node, value_type):
        """Classify a product operand as a byte factor.

        Returns ``("bytes", count)`` for an integer literal or a ``sizeof`` with a
        known size, ``("size",)`` for ``sizeof`` of the element type, and ``None``
        for anything else.
        """
        if isinstance(node, str):
            literal = self.parse_hip_surface_integer_literal(node)
            if literal is not None:
                return ("bytes", literal)
            return None
        if isinstance(node, CastNode):
            return self.hip_surface_byte_factor(node.expression, value_type)
        if (
            isinstance(node, FunctionCallNode)
            and node.name == "sizeof"
            and len(node.args) == 1
            and isinstance(node.args[0], str)
        ):
            if value_type is not None and self.strip_type_qualifiers(
                node.args[0]
            ) == self.strip_type_qualifiers(value_type):
                return ("size",)
            size = self.hip_surface_value_type_size(node.args[0])
            if size is not None:
                return ("bytes", size)
            return None
        return None

    def parse_hip_surface_integer_literal(self, text):
        """Parse a HIP integer literal to an int, or return None when it is not one."""
        spelling = str(text).replace("'", "")
        try:
            digits, _type = integer_literal_parts(spelling, legacy_octal=True)
        except IntegerLiteralError:
            return None
        try:
            return int(digits, 0)
        except ValueError:
            return None

    def infer_hip_surface_value_type(self, expression):
        """Element type of a ``surf*write`` value or a pointer-read destination, including member and array accesses."""
        value_type = self.hip_surface_expression_type(expression)
        return self.hip_surface_element_type(value_type)

    def hip_surface_expression_type(self, expression):
        """HIP type of a value expression, or None when it cannot be inferred."""
        if isinstance(expression, str):
            return self.hip_surface_normalize_type(
                self.lookup_variable_type(expression)
            )
        if isinstance(expression, CastNode):
            return self.strip_type_qualifiers(expression.target_type)
        if isinstance(expression, ArrayAccessNode):
            array_type = self.hip_surface_expression_type(expression.array)
            return self.hip_surface_element_type(array_type)
        if isinstance(expression, MemberAccessNode):
            return self.hip_surface_member_type(expression)
        if isinstance(expression, UnaryOpNode) and expression.op == "*":
            operand_type = self.hip_surface_expression_type(expression.operand)
            return self.hip_surface_element_type(operand_type)
        if isinstance(expression, FunctionCallNode) and isinstance(
            expression.name, str
        ):
            name = expression.name
            if name.startswith("::"):
                name = name[2:]
            if name.startswith("make_"):
                return name[len("make_") :]
            if (
                name in self.VECTOR_TYPE_MAPPING
                or name in self.VECTOR1_TYPE_MAPPING
                or name in self.HIP_SURFACE_SCALAR_TYPES
            ):
                return name
            return None
        return None

    def hip_surface_member_type(self, expression):
        """Type of a member access, from the struct definition or a vector component."""
        object_type = self.hip_surface_expression_type(expression.object)
        named_type = self.hip_surface_named_type(object_type)
        if named_type is None:
            return None
        definition = self.record_definition_from_type(named_type)
        if definition is not None:
            for member in getattr(definition, "members", []) or []:
                if (
                    isinstance(member, VariableNode)
                    and member.name == expression.member
                ):
                    return self.strip_type_qualifiers(member.vtype)
            return None
        return self.hip_surface_vector_component_scalar_type(
            named_type, expression.member
        )

    def hip_surface_named_type(self, type_name):
        """Strip qualifiers, aliases, pointers and array suffixes to a plain type name."""
        if not type_name:
            return None
        type_name = self.strip_type_qualifiers(self.resolve_type_alias(type_name))
        type_name = type_name.replace("*", "").split("[", 1)[0].strip()
        for prefix in ("struct ", "union "):
            if type_name.startswith(prefix):
                type_name = type_name[len(prefix) :].strip()
        return type_name or None

    def hip_surface_vector_component_scalar_type(self, base_type, member):
        """Scalar type of a single-component access such as ``float4.x``."""
        scalars = {
            "char": "char",
            "uchar": "unsigned char",
            "short": "short",
            "ushort": "unsigned short",
            "int": "int",
            "uint": "unsigned int",
            "long": "long",
            "ulong": "unsigned long",
            "longlong": "long long",
            "ulonglong": "unsigned long long",
            "half": "half",
            "__half": "half",
            "float": "float",
            "double": "double",
        }
        match = re.fullmatch(r"([A-Za-z_]+)(\d)", base_type)
        if match is None:
            return None
        scalar = scalars.get(match.group(1))
        if scalar is None:
            return None
        dimension = int(match.group(2))
        if member not in self.HIP_SURFACE_VECTOR_COMPONENTS[:dimension]:
            return None
        return scalar

    def hip_surface_element_type(self, type_name):
        if not type_name:
            return None
        type_name = self.strip_type_qualifiers(self.resolve_type_alias(type_name))
        if self.has_array_suffix(type_name):
            return self.hip_surface_normalize_type(type_name.split("[", 1)[0].strip())
        if "*" in type_name:
            return self.hip_surface_normalize_type(type_name.replace("*", "").strip())
        base, template_args = self.parse_cpp_template(type_name)
        if base in {"array", "ptr"} and template_args:
            return self.hip_surface_normalize_type(template_args[0])
        return self.hip_surface_normalize_type(type_name)

    def hip_surface_normalize_type(self, type_name):
        """Map a CrossGL type spelling back to the HIP spelling it came from.

        Variable types are registered in their CrossGL spelling (``f32``,
        ``vec4<f32>``, ``array<f32>``), while struct members, casts and template
        arguments keep the HIP spelling (``float``, ``float4``, ``float*``). The
        surface helpers compare and size HIP spellings, so a CrossGL spelling is
        translated back; anything else is returned unchanged.
        """
        if not type_name:
            return type_name
        text = self.strip_type_qualifiers(type_name).strip()
        crossgl_scalars = {
            "bool": "bool",
            "i8": "char",
            "u8": "uchar",
            "i16": "short",
            "u16": "ushort",
            "i32": "int",
            "u32": "uint",
            "i64": "long long",
            "u64": "unsigned long long",
            "f16": "half",
            "f32": "float",
            "f64": "double",
        }
        if text in crossgl_scalars:
            return crossgl_scalars[text]
        vector = re.fullmatch(r"vec([1-4])<([A-Za-z0-9]+)>", text)
        if vector is not None:
            scalar = crossgl_scalars.get(vector.group(2))
            if scalar is not None:
                return f"{scalar}{vector.group(1)}"
        return text

    def hip_surface_value_type_size(self, value_type):
        if not value_type:
            return None

        text = str(value_type).strip()
        if text.startswith("const "):
            text = text[6:].strip()

        vector_match = re.fullmatch(
            r"(?:make_)?(char|uchar|short|ushort|int|uint|long|ulong|"
            r"longlong|ulonglong|half|__half|float|double)([1-4])",
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
                "half": 2,
                "__half": 2,
                "float": 4,
                "double": 8,
            }.get(scalar_type)
            return scalar_size * int(count) if scalar_size is not None else None

        return {
            "bool": 1,
            "char": 1,
            "signed char": 1,
            "unsigned char": 1,
            "uchar": 1,
            "int8_t": 1,
            "uint8_t": 1,
            "short": 2,
            "unsigned short": 2,
            "ushort": 2,
            "int16_t": 2,
            "uint16_t": 2,
            "half": 2,
            "__half": 2,
            "int": 4,
            "signed int": 4,
            "unsigned int": 4,
            "uint": 4,
            "float": 4,
            "int32_t": 4,
            "uint32_t": 4,
            "long": 8,
            "unsigned long": 8,
            "ulong": 8,
            "long long": 8,
            "unsigned long long": 8,
            "longlong": 8,
            "ulonglong": 8,
            "double": 8,
            "int64_t": 8,
            "uint64_t": 8,
        }.get(text)

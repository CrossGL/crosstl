"""Constructor calls in the WGSL generator: scalar matrix constructors, generic constructors and vector constructor arguments."""

import re

from ..ast import (
    LiteralNode,
    UnaryOpNode,
)
from ..refusals import WGSLRefusal
from .generic_struct_utils import named_struct_constructor_arguments
from .wgsl_clip_distances import scalar_clip_distance_value
from .wgsl_matrix_resize import wgsl_matrix_resize


class WGSLConstructorMixin:
    """Constructor members of ``WGSLCodeGen``."""

    def generate_constructor(self, node):
        if node.named_arguments:
            # Fields come from the source declaration: the emitted name of a
            # structure such as ``filter`` is ``filter_``.
            struct_name = self.struct_type_name(node.constructor_type)
            struct = self._structs_by_name.get(struct_name)
            fields = [
                (member.name, member.member_type)
                for member in getattr(struct, "members", None) or []
                if hasattr(member, "member_type")
            ]
            arguments = named_struct_constructor_arguments(struct_name, node, fields)
            args = ", ".join(
                scalar_clip_distance_value(
                    self,
                    struct_name,
                    field_name,
                    self.generate_expression_for_target(arg, field_type),
                )
                for (field_name, _), (arg, field_type) in zip(fields, arguments)
            )
            return f"{self.type_name_string(node.constructor_type)}({args})"
        if len(node.arguments) == 1:
            narrowed = self.generate_vector_narrowing_conversion(
                node.constructor_type, node.arguments[0]
            )
            if narrowed is not None:
                return narrowed
            diagonal = self.generate_scalar_matrix_constructor(
                node.constructor_type, node.arguments[0]
            )
            if diagonal is not None:
                return diagonal
            resized = wgsl_matrix_resize(self, node.constructor_type, node.arguments[0])
            if resized is not None:
                return resized
        vector_shape = self.vector_shape(node.constructor_type)
        if vector_shape is not None:
            element_type, _size = vector_shape
            convert_vectors = len(node.arguments) > 1
            args = ", ".join(
                self.generate_vector_constructor_argument(
                    arg, element_type, convert_vectors
                )
                for arg in node.arguments
            )
        else:
            args = ", ".join(self.generate_expression(arg) for arg in node.arguments)
        return f"{self.type_name_string(node.constructor_type)}({args})"

    def generate_scalar_matrix_constructor(self, matrix_type, arg):
        """Expand a one-scalar matrix constructor such as ``mat2(s)``.

        CrossGL, like GLSL (4.60 section 5.4.2), builds a matrix with ``s`` on
        the diagonal and zeros elsewhere. WGSL has no such constructor, so the
        matrix is written column by column.
        """
        shape = self.matrix_shape(matrix_type)
        if shape is None:
            return None
        # ``-s`` and ``+s`` have the type of ``s``.
        operand = arg
        while (
            isinstance(operand, UnaryOpNode)
            and operand.operator in {"-", "+"}
            and not operand.is_postfix
        ):
            operand = operand.operand
        arg_type = self.expression_type(operand)
        try:
            arg_type_name = (
                self.type_name_string(arg_type) if arg_type is not None else None
            )
        except ValueError:
            arg_type_name = None
        is_numeric_literal = isinstance(operand, LiteralNode) and isinstance(
            operand.value, (int, float)
        )
        if arg_type_name not in {"f32", "f16", "i32", "u32"} and not (
            is_numeric_literal and not isinstance(operand.value, bool)
        ):
            return None
        element, columns, rows = shape
        if not self.is_repeatable_matrix_operand(arg):
            raise WGSLRefusal(
                "WGSL target writes a one-scalar matrix constructor "
                f"mat{columns}x{rows}<{element}>(s) one diagonal element at a "
                "time and requires an argument without side effects; assign the "
                "argument to a local variable first"
            )
        # An integer literal without the ``u`` suffix is an abstract integer in
        # WGSL and converts to the element type; ``1u`` is converted below.
        if (
            self.is_float_vector_abstract_int_argument(operand, element)
            and arg_type_name != "u32"
        ) or (is_numeric_literal and arg_type_name is None):
            diagonal = self.generate_expression(arg)
        elif operand is arg:
            diagonal = self.generate_expression_for_target(arg, element)
        else:
            diagonal = self.generate_expression(arg)
            if arg_type_name != element:
                diagonal = f"{element}({diagonal})"
        column_values = []
        for column in range(columns):
            components = [diagonal if row == column else "0.0" for row in range(rows)]
            column_values.append(f"vec{rows}<{element}>({', '.join(components)})")
        return f"mat{columns}x{rows}<{element}>({', '.join(column_values)})"

    def generic_constructor_type_name(self, name):
        parsed = self.generic_constructor_parts(name)
        if parsed is None:
            return None
        base_name, generic_args = parsed
        key = self.generic_struct_type_key(base_name, generic_args)
        specialized_name = self._generic_struct_specialized_names.get(key)
        if specialized_name is None:
            return None
        return self.type_identifier_name(specialized_name)

    def generic_constructor_parts(self, name):
        match = re.fullmatch(r"\s*([A-Za-z_][0-9A-Za-z_]*)\s*<(.+)>\s*", str(name))
        if match is None:
            return None
        base_name = match.group(1)
        if base_name not in self._generic_struct_definitions:
            return None
        generic_args = self.split_generic_constructor_arguments(match.group(2))
        if generic_args is None:
            return None
        return base_name, generic_args

    def split_generic_constructor_arguments(self, args_text):
        args = []
        depth = 0
        current = []
        for char in str(args_text):
            if char == "<":
                depth += 1
            elif char == ">":
                depth -= 1
                if depth < 0:
                    return None
            elif char == "," and depth == 0:
                args.append("".join(current).strip())
                current = []
                continue
            current.append(char)
        if depth != 0:
            return None
        if current or args_text:
            args.append("".join(current).strip())
        return args

    def generate_vector_constructor_argument(
        self, arg, element_type, convert_vectors=True
    ):
        arg_shape = self.vector_shape(self.expression_type(arg))
        if arg_shape is not None:
            rendered = self.generate_expression(arg)
            # A WGSL vector constructor converts only a whole vector passed alone, so a vector of
            # another component type that contributes components among other arguments is
            # converted first.
            element = self.scalar_type_name(element_type)
            if convert_vectors and element is not None and element != arg_shape[0]:
                return f"vec{arg_shape[1]}<{element}>({rendered})"
            return rendered
        if self.is_float_vector_abstract_int_argument(arg, element_type):
            return self.generate_expression(arg)
        return self.generate_expression_for_target(arg, element_type)

    def is_float_vector_abstract_int_argument(self, arg, element_type):
        return (
            self.scalar_type_name(element_type) == "f32"
            and isinstance(arg, LiteralNode)
            and isinstance(arg.value, int)
            and not isinstance(arg.value, bool)
        )

    def generate_type_constructor_call_arguments(self, function_name, arguments):
        vector_shape = self.vector_shape(function_name)
        if vector_shape is None:
            if self.map_type_name(function_name) == "u32" and len(arguments) == 1:
                return self.concrete_negative_integer_literal(
                    arguments[0], self.generate_expression(arguments[0])
                )
            return self.generate_call_arguments(function_name, arguments)
        element_type, _size = vector_shape
        convert_vectors = len(arguments) > 1
        return ", ".join(
            self.generate_vector_constructor_argument(
                arg, element_type, convert_vectors
            )
            for arg in arguments
        )

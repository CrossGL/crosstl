"""Metal renderings of the CrossGL expression forms that ``MetalCodeGen.generate_expression`` dispatches on: source strings, identifiers, variables, operators, array literals, subscripts, constructors, member accesses, conditional expressions and literal nodes."""

from ..ast import (
    ArrayLiteralNode,
    AssignmentNode,
    BinaryOpNode,
    CastNode,
    PointerReinterpretNode,
    TernaryOpNode,
    UnaryOpNode,
)
from ..refusals import (
    MetalRefusal,
)
from .array_utils import (
    split_array_type_suffix,
)
from .c_family_operands import (
    postfix_operand,
)
from .enum_utils import (
    enum_value_expression,
    generate_enum_constructor_expression,
)
from .generic_struct_utils import (
    generate_struct_constructor_expression,
)
from .metal_comparison_signs import (
    metal_sign_matched_comparison_operands,
)
from .metal_feature_errors import (
    UnsupportedMetalFeatureError,
)
from .relational_builtins import (
    metal_logical_not_operand,
)
from .storage_image_texels import (
    is_whole_texel_load,
)


class MetalExpressionFormMixin:
    """Expression-form members of ``MetalCodeGen.generate_expression``."""

    def generate_metal_raw_string_expression(
        self,
        expr,
    ):
        """Render an expression that is already a source string."""
        builtin_name = self.metal_builtin_expression_name(expr)
        if builtin_name != expr:
            return builtin_name
        if expr.rsplit("::", 1)[-1] == "None":
            option_default = self.option_none_default_expression()
            if option_default is not None:
                return option_default
        if expr in self.local_variable_types:
            return self.metal_local_identifier_name(expr)
        return expr

    def generate_metal_identifier_node_expression(
        self,
        expr,
    ):
        """Render an identifier node."""
        name = expr.name
        builtin_name = self.metal_builtin_expression_name(name)
        if builtin_name != name:
            return builtin_name
        if isinstance(name, str) and name.rsplit("::", 1)[-1] == "None":
            option_default = self.option_none_default_expression()
            if option_default is not None:
                return option_default
        buffer_block_value = self.unsupported_glsl_buffer_block_access_value(expr)
        if buffer_block_value is not None:
            return buffer_block_value
        if name in self.METAL_RAY_FLAG_VALUES:
            return f"{self.METAL_RAY_FLAG_VALUES[name]}u"
        if name in getattr(self, "enum_variant_constants", {}):
            return enum_value_expression(self, name)
        if (
            name not in self.local_variable_types
            and name in self.ambiguous_cbuffer_members
        ):
            raise MetalRefusal(
                f"Ambiguous cbuffer member reference '{name}' appears in multiple cbuffers"
            )
        if (
            name not in self.local_variable_types
            and name in self.cbuffer_member_references
        ):
            return self.cbuffer_member_references[name]
        if (
            name not in self.local_variable_types
            and name in self.metal_runtime_initialized_constants
        ):
            return f"{name}()"
        return self.metal_local_identifier_name(name)

    def generate_metal_variable_expression(
        self,
        expr,
    ):
        """Render a variable reference."""
        unsupported_value = self.unsupported_glsl_buffer_block_access_value(expr)
        if unsupported_value is not None:
            return unsupported_value
        unsupported_value = (
            self.unsupported_metal_acceleration_structure_array_expression(expr)
        )
        if unsupported_value is not None:
            return unsupported_value
        unsupported_value = self.unsupported_metal_ray_function_table_array_expression(
            expr
        )
        if unsupported_value is not None:
            return unsupported_value
        if hasattr(expr, "name"):
            builtin_name = self.metal_builtin_expression_name(expr.name)
            if builtin_name != expr.name:
                return builtin_name
            if isinstance(expr.name, str) and expr.name.rsplit("::", 1)[-1] == "None":
                option_default = self.option_none_default_expression()
                if option_default is not None:
                    return option_default
            if expr.name in self.METAL_RAY_FLAG_VALUES:
                return f"{self.METAL_RAY_FLAG_VALUES[expr.name]}u"
            if expr.name in self.local_variable_types:
                return self.metal_local_identifier_name(expr.name)
            enum_value = enum_value_expression(self, expr.name)
            if enum_value is not None:
                return enum_value
            # Not an enum value: a variable this program declares keeps its declared spelling.
            return expr.name
        else:
            return str(expr)

    def generate_metal_binary_expression(
        self,
        expr,
    ):
        """Render a binary operation."""
        operator = self.map_operator(expr.op)
        left = self.metal_narrow_vector_operand(
            expr.left, self.generate_binary_operand(expr.left, operator)
        )
        right = self.metal_narrow_vector_operand(
            expr.right, self.generate_binary_operand(expr.right, operator, True)
        )
        if operator in {"<", ">", "<=", ">="}:
            if self.map_type(self.expression_result_type(expr.left)) == "bool":
                left = f"int({left})"
            if self.map_type(self.expression_result_type(expr.right)) == "bool":
                right = f"int({right})"
        if operator in {"<", ">", "<=", ">=", "==", "!="}:
            left, right = metal_sign_matched_comparison_operands(
                self.map_type(self.expression_result_type(expr.left)),
                self.map_type(self.expression_result_type(expr.right)),
                left,
                right,
            )
        left, right, common = self.metal_mixed_vector_binary_operands(
            expr, operator, left, right
        )
        if operator == "%" and common in {"float", "half"}:
            return f"fmod({left}, {right})"
        matrix_arithmetic = self.metal_matrix_builtins.componentwise(
            operator, expr.left, expr.right, left, right
        )
        if matrix_arithmetic is not None:
            return matrix_arithmetic
        # The operands keep their own (common) type and the declaration
        # converts the result: converting each operand to the declared
        # type would turn an integer division into a floating one, or a
        # signed one into an unsigned one (#3166).
        return self.metal_narrow_vector_wrap(
            expr, operator, f"{left} {operator} {right}"
        )

    def generate_metal_array_literal_expression(
        self,
        expr,
    ):
        """Render an array literal."""
        element_type, array_suffix = split_array_type_suffix(
            self.current_expression_expected_type or ""
        )
        matrix_element_type = (
            element_type
            if array_suffix.count("[") == 1 and self.is_matrix_value_type(element_type)
            else None
        )
        # A structure's brace list gives each nested list its member's type,
        # and an array of structures' list gives each its element type, so a
        # matrix array member is written as the array above is.
        nested_types = []
        if element_type in self.struct_member_types and array_suffix:
            nested_types = [
                self.metal_for_in_array_element_type(
                    self.current_expression_expected_type
                )
            ] * len(expr.elements)
        elif element_type in self.struct_member_types:
            nested_types = list(self.struct_member_types[element_type].values())
        rendered_elements = []
        for index, element in enumerate(expr.elements):
            nested_type = (
                nested_types[index]
                if isinstance(element, ArrayLiteralNode) and index < len(nested_types)
                else None
            )
            rendered = self.generate_expression_with_expected(element, nested_type)
            # A scalar cannot initialize a matrix element of a brace list, so
            # write the matrix constructor, as GLSL output's mat3(0.0) does.
            if (
                matrix_element_type is not None
                and self.expression_component_count(element) == 1
            ):
                rendered = f"{self.map_type(matrix_element_type)}({rendered})"
            rendered_elements.append(rendered)
        elements = ", ".join(rendered_elements)
        return f"{{{elements}}}"

    def generate_metal_unary_expression(
        self,
        expr,
    ):
        """Render a unary operation."""
        operand_value = getattr(getattr(expr, "operand", None), "value", None)
        if (
            getattr(expr, "op", getattr(expr, "operator", None)) == "-"
            and not getattr(expr, "is_postfix", False)
            and isinstance(operand_value, int)
            and not isinstance(operand_value, bool)
            and operand_value == 1 << 63
            and self.map_type(
                getattr(getattr(expr.operand, "literal_type", None), "name", None)
            )
            not in {"ulong", "uint64_t"}
        ):
            return "(-9223372036854775807L - 1L)"
        negated_literal = self.metal_negated_wide_integer_literal(expr)
        if negated_literal is not None:
            return negated_literal
        local_reinterpret = self.generate_metal_local_reinterpret_read(expr)
        if local_reinterpret is not None:
            return local_reinterpret
        operator = self.map_operator(expr.op)
        if operator in {"++", "--"} and self.is_metal_signed_builtin_member(
            expr.operand
        ):
            # The uint storage steps with the int's bits; the value read is converted back.
            storage = self.generate_metal_signed_builtin_member_storage(expr.operand)
            if getattr(expr, "is_postfix", False):
                return f"int({storage}{operator})"
            return f"int({operator}{storage})"
        operand = self.generate_unary_operand(expr.operand)
        if operator in {"++", "--"}:
            byte_update = self.metal_narrow_integer_update_expression(
                expr, expr.operand, operator
            )
            if byte_update is not None:
                return byte_update
            # Buffer storage whose reads convert (a packed vector's
            # components) is updated in place through the storage itself.
            layout_store = self.metal_hlsl_layout_store_target(expr.operand)
            if layout_store is not None and not layout_store[1]:
                operand = layout_store[0]
        if getattr(expr, "is_postfix", False):
            return f"{postfix_operand(operand)}{operator}"
        if operator == "!":
            operand = metal_logical_not_operand(
                operand, self.expression_result_type(expr.operand)
            )
        operand = self.metal_narrow_vector_operand(expr.operand, operand)
        if operator == "-":
            matrix_negation = self.metal_matrix_builtins.negation(expr.operand, operand)
            if matrix_negation is not None:
                return matrix_negation
        if operator in {"+", "-"} and operand.startswith(("+", "-")):
            # Keep adjacent signs from becoming an increment or decrement.
            operand = f"({operand})"
        return self.metal_narrow_vector_wrap(expr, operator, f"{operator}{operand}")

    def generate_metal_array_access_expression(
        self,
        expr,
    ):
        """Render a subscript expression."""
        unsupported_value = self.unsupported_glsl_buffer_block_access_value(expr)
        if unsupported_value is not None:
            return unsupported_value
        stage_io_access = self.generate_metal_stage_io_lowered_array_access(expr)
        if stage_io_access is not None:
            return stage_io_access
        unsupported_value = (
            self.unsupported_metal_acceleration_structure_array_expression(expr)
        )
        if unsupported_value is not None:
            return unsupported_value
        unsupported_value = self.unsupported_metal_ray_function_table_array_expression(
            expr
        )
        if unsupported_value is not None:
            return unsupported_value
        block_load = self.generate_glsl_buffer_block_array_load(expr)
        if block_load is not None:
            return block_load
        layout_element = self.metal_hlsl_layout_element_access(expr)
        if layout_element is not None:
            access, conversion = layout_element
            return f"{conversion}({access})" if conversion else access
        if is_whole_texel_load(expr.array, self.function_call_name):
            # The subscript selects a component of the whole texel.
            array = self.generate_expression_with_expected(expr.array, None)
        else:
            array = self.generate_expression(expr.array)
        if isinstance(
            expr.array,
            (
                AssignmentNode,
                BinaryOpNode,
                TernaryOpNode,
                CastNode,
                PointerReinterpretNode,
                UnaryOpNode,
            ),
        ):
            array = f"({array})"
        index = self.generate_expression(expr.index)
        return f"{array}[{index}]"

    def generate_metal_constructor_expression(
        self,
        expr,
    ):
        """Render a constructor expression."""
        registered_conversion = self.metal_registered_structure_scalar_conversion(
            expr,
            getattr(expr, "constructor_type", None),
            getattr(expr, "arguments", []),
        )
        if registered_conversion is not None:
            return registered_conversion
        enum_constructor = generate_enum_constructor_expression(self, expr)
        if enum_constructor is not None:
            return enum_constructor
        constructor = generate_struct_constructor_expression(self, expr)
        if constructor is not None:
            return constructor
        constructor_type = getattr(expr, "constructor_type", None)
        matrix_constructor = self.generate_metal_matrix_constructor_override(
            constructor_type, getattr(expr, "arguments", [])
        )
        if matrix_constructor is not None:
            return matrix_constructor
        if self.is_builtin_value_constructor_type(constructor_type):
            metal_type = self.map_type(constructor_type)
            matrix_resize = self.generate_metal_matrix_resize_constructor(
                metal_type, getattr(expr, "arguments", [])
            ) or self.metal_matrix_builtins.component_constructor(
                metal_type, getattr(expr, "arguments", [])
            )
            if matrix_resize is not None:
                return matrix_resize
            args = ", ".join(
                self.generate_metal_value_constructor_arguments(
                    metal_type, getattr(expr, "arguments", [])
                )
            )
            if getattr(
                expr, "is_braced_constructor", False
            ) and self.is_vector_value_type(constructor_type):
                width = self.value_component_count(constructor_type)
                counts = [
                    self.expression_component_count(arg) for arg in expr.arguments
                ]
                if (
                    width
                    and all(count is not None for count in counts)
                    and sum(counts) > width
                ):
                    raise UnsupportedMetalFeatureError(
                        "vector-list-initialization",
                        f"Vector initializer for {metal_type} exceeds its {width} components",
                        reason="element-count-mismatch",
                        source_location=getattr(expr, "source_location", None),
                    )
                rendered = f"{metal_type}{{{args}}}"
            else:
                rendered = f"{metal_type}({args})"
            return (
                self.metal_narrow_integer_conversion_expression(
                    rendered, constructor_type
                )
                or rendered
            )
        return str(expr)

    def generate_metal_member_access_expression(
        self,
        expr,
    ):
        """Render a member access."""
        unsupported_value = self.unsupported_glsl_buffer_block_access_value(expr)
        if unsupported_value is not None:
            return unsupported_value
        stage_io_access = self.generate_metal_stage_io_lowered_member_access(expr)
        if stage_io_access is not None:
            return stage_io_access
        if id(
            expr
        ) not in self.metal_signed_member_raw_reads and self.is_metal_signed_builtin_member(
            expr
        ):
            return f"int({self.generate_metal_signed_builtin_member_storage(expr)})"
        unsupported_value = self.unsupported_metal_ray_function_table_array_expression(
            expr
        )
        if unsupported_value is not None:
            return unsupported_value
        block_load = self.generate_glsl_buffer_block_member_load(expr)
        if block_load is not None:
            return block_load
        suppress_suffix = self.member_access_is_image_load_component(expr)
        object_name = self.expression_name(expr.object)
        if (
            object_name in self.metal_static_struct_members
            and str(expr.member)
            in self.metal_static_struct_members.get(object_name, set())
            and object_name not in self.local_variable_types
            and object_name not in self.metal_program_scope_value_global_types
        ):
            return f"{object_name}::{expr.member}"
        previous_suppression = self.suppress_image_load_component_suffix
        self.suppress_image_load_component_suffix = (
            previous_suppression or suppress_suffix
        )
        try:
            obj = self.generate_expression_with_expected(expr.object, None)
        finally:
            self.suppress_image_load_component_suffix = previous_suppression
        if isinstance(
            expr.object, (BinaryOpNode, TernaryOpNode, UnaryOpNode)
        ) and not self.metal_expression_is_parenthesized(obj):
            # Member access binds tighter than every operator; keep the
            # operand grouping that CrossGL expresses through the tree.
            obj = f"({obj})"
        if self.member_access_uses_pointer_operator(expr):
            return f"{obj}->{expr.member}"
        layout_member = self.metal_hlsl_layout_member(expr)
        if (
            layout_member is not None
            and layout_member.read_conversion
            and layout_member.array_count is None
        ):
            return f"{layout_member.read_conversion}({obj}.{expr.member})"
        return f"{obj}.{expr.member}"

    def generate_metal_ternary_expression(
        self,
        expr,
    ):
        """Render a conditional expression."""
        true_code, false_code = self.metal_conditional_arms(
            expr,
            self.generate_expression(expr.true_expr),
            self.generate_expression(expr.false_expr),
        )
        condition = self.generate_expression(expr.condition)
        # `?:` is right-associative: an unparenthesized conditional
        # condition would regroup as part of the false arm.
        if isinstance(expr.condition, TernaryOpNode):
            condition = f"({condition})"
        return f"{condition} ? {true_code} : {false_code}"

    def generate_metal_literal_class_expression(
        self,
        expr,
    ):
        """Render a node whose class name marks it as a literal."""
        if hasattr(expr, "value"):
            value = expr.value
            literal_type = getattr(getattr(expr, "literal_type", None), "name", None)
            if isinstance(value, int) and not isinstance(value, bool):
                mapped_type = self.map_type(literal_type) if literal_type else None
                if mapped_type in {"long", "ulong", "int64_t", "uint64_t"}:
                    suffix = "ul" if mapped_type in {"ulong", "uint64_t"} else "l"
                    return f"{value}{suffix}"
            if (
                literal_type == "uint"
                and isinstance(value, int)
                and not isinstance(value, bool)
            ):
                return f"{value}u"
            if isinstance(value, bool):
                return "true" if value else "false"
            if isinstance(value, int):
                # Unsuffixed decimal literals beyond int are long in
                # Metal; the source type of such a literal is uint up to
                # 2^32 - 1 (as in GLSL and HLSL) and 64-bit above it.
                if value > 0x7FFFFFFFFFFFFFFF:
                    return f"{value}ul"
                if 0x7FFFFFFF < value <= 0xFFFFFFFF:
                    return f"{value}u"
                if value < -0x80000000:
                    return f"{value}l"
            if isinstance(value, str) and not (
                value.startswith('"') and value.endswith('"')
            ):
                return f'"{value}"'  # Add quotes for string literals
            return str(value)
        return str(expr)

    def generate_metal_identifier_class_expression(
        self,
        expr,
    ):
        """Render a node whose class name marks it as an identifier."""
        name = getattr(expr, "name", str(expr))
        if isinstance(name, str) and name.rsplit("::", 1)[-1] == "None":
            option_default = self.option_none_default_expression()
            if option_default is not None:
                return option_default
        unsupported_value = self.unsupported_glsl_buffer_block_access_value(expr)
        if unsupported_value is not None:
            return unsupported_value
        if name in self.METAL_RAY_FLAG_VALUES:
            return f"{self.METAL_RAY_FLAG_VALUES[name]}u"
        if name in getattr(self, "enum_variant_constants", {}):
            return enum_value_expression(self, name)
        if (
            name not in self.local_variable_types
            and name in self.ambiguous_cbuffer_members
        ):
            raise MetalRefusal(
                f"Ambiguous cbuffer member reference '{name}' appears in multiple cbuffers"
            )
        if (
            name not in self.local_variable_types
            and name in self.cbuffer_member_references
        ):
            return self.cbuffer_member_references[name]
        if (
            name not in self.local_variable_types
            and name in self.metal_runtime_initialized_constants
        ):
            return f"{name}()"
        return name

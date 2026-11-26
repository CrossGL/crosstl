"""Numeric conversions in the GLSL generator: struct construction, arithmetic and narrowing contracts, aggregate constructors, literals and partial vector constructors."""

import math
import re
import struct

from ..arithmetic_conversions import (
    ArithmeticScalarKind,
    ArithmeticType,
    UnrepresentableArithmeticConversion,
    arithmetic_type,
    arithmetic_type_name,
    narrow_integer_shape,
    promoted_integer_type_name,
    resolve_arithmetic_conversion,
    target_arithmetic_type,
)
from ..ast import (
    ArrayAccessNode,
    ArrayLiteralNode,
    ConstructorNode,
    FunctionCallNode,
    LiteralNode,
    UnaryOpNode,
)
from ..refusals import OpenGLRefusal
from ..structure_conversions import StructureConversionKind
from .array_utils import (
    evaluate_literal_int_expression,
    split_array_type_suffix,
)
from .bfloat_constants import (
    bfloat16_constant_float,
    scalar_constant_value,
)
from .comma_expressions import (
    c_family_comma_expression,
    is_comma_expression,
)
from .crossgl_vector_names import crossgl_named_vector_spelling
from .glsl_version_floor import glsl_version_reaches
from .opengl_values import (
    OpenGLAggregateInitializerError,
    OpenGLComplexArithmeticError,
    OpenGLScalarConversionError,
    OpenGLTemplateTypeError,
)


class OpenGLNumericConversionMixin:
    """Numeric conversion members of ``GLSLCodeGen``."""

    def glsl_struct_construction_error(
        self,
        expr,
        destination_type,
        source_type,
        conversion_kind,
        reason,
    ):
        if isinstance(conversion_kind, StructureConversionKind):
            conversion_kind = conversion_kind.value
        rendered_source_type = self.type_name_string(source_type) or "unknown"
        raise self.GLSL_STRUCT_CONSTRUCTION_ERROR(
            f"{self.GLSL_TARGET_DISPLAY_NAME} cannot preserve structure "
            "construction from "
            f"'{rendered_source_type}' to '{destination_type}': "
            f"{reason.replace('-', ' ')}",
            destination_type=destination_type,
            source_type=rendered_source_type,
            conversion_kind=conversion_kind,
            reason=reason,
            source_location=getattr(expr, "source_location", None),
        )

    def glsl_struct_constructor_conversion(self, expr, constructor, arguments):
        union = self.union_storage.construct(expr, constructor, arguments)
        if union is not None:
            return union
        source_constructor = self.glsl_source_type_identifier_name(constructor)
        if source_constructor not in self.structs_by_name:
            return None
        struct_node = self.structs_by_name[source_constructor]
        if not list(getattr(struct_node, "members", []) or []):
            if all(
                self.glsl_side_effect_free_expression(argument)
                for argument in arguments
            ):
                return f"{constructor}(0)"
            self.glsl_struct_construction_error(
                expr,
                constructor,
                None,
                "empty-structure-construction",
                "constructor-arguments-have-side-effects",
            )
        if not arguments:
            return self.glsl_registered_default_struct_construction(expr, constructor)
        if len(arguments) != 1:
            return None

        argument = arguments[0]
        source_type = self.glsl_source_expression_type(argument)
        if source_type is not None and self.map_type(source_type) == constructor:
            return self.generate_expression(argument)

        constructor_fields = list(
            self.struct_member_types.get(source_constructor, {}).values()
        )
        if len(constructor_fields) == 1:
            return None

        rendered = self.generate_expression(argument)
        promoted = self.glsl_registered_scalar_to_struct_conversion(
            expr,
            rendered,
            source_type,
            constructor,
            StructureConversionKind.SCALAR_PROMOTION.value,
        )
        if promoted is not None:
            return promoted

        source_info = self.glsl_value_type_info(source_type)
        source_mapped = self.map_type(source_type) if source_type is not None else None
        if source_type is None:
            conversion_kind = StructureConversionKind.UNKNOWN.value
            reason = "source-type-unknown"
        elif self.glsl_struct_node(source_mapped) is not None:
            conversion_kind = StructureConversionKind.STRUCTURE_CONVERSION.value
            reason = "structure-conversion-unsupported"
        elif source_info is not None and source_info["width"] == 1:
            conversion_kind = StructureConversionKind.SCALAR_PROMOTION.value
            reason = "scalar-conversion-unregistered"
        else:
            conversion_kind = StructureConversionKind.VALUE_CONVERSION.value
            reason = "source-shape-unsupported"
        self.glsl_struct_construction_error(
            expr,
            constructor,
            source_type,
            conversion_kind,
            reason,
        )

    def glsl_complex64_error(self, expr, operator, operand_types, reason):
        rendered_types = tuple(
            self.type_name_string(value) or "unknown" for value in operand_types
        )
        raise OpenGLComplexArithmeticError(
            "OpenGL cannot preserve complex arithmetic "
            f"'{operator}' for operand types {rendered_types}: "
            f"{reason.replace('-', ' ')}",
            operator=operator,
            operand_types=rendered_types,
            reason=reason,
            source_location=getattr(expr, "source_location", None),
        )

    def glsl_validate_complex64_arithmetic_representation(
        self,
        expr,
        operator,
        operand_types,
    ):
        if any(
            self.is_glsl_registered_complex64_type(operand_type)
            and not self.is_glsl_complex64_type(operand_type)
            for operand_type in operand_types
        ):
            self.glsl_complex64_error(
                expr,
                operator,
                operand_types,
                "registered-representation-arithmetic-unsupported",
            )

    def glsl_complex64_binary_expression(
        self,
        expr,
        left,
        right,
        operator,
        left_type=None,
        right_type=None,
    ):
        left_type = left_type or self.glsl_source_expression_type(expr.left)
        right_type = right_type or self.glsl_source_expression_type(expr.right)
        if not (
            self.is_glsl_registered_complex64_type(left_type)
            or self.is_glsl_registered_complex64_type(right_type)
        ):
            return None
        self.glsl_validate_complex64_arithmetic_representation(
            expr,
            operator,
            (left_type, right_type),
        )

        helpers = {
            "+": "crossgl_complex64_add",
            "-": "crossgl_complex64_sub",
            "*": "crossgl_complex64_mul",
            "/": "crossgl_complex64_div",
            "%": "crossgl_complex64_mod",
            "==": "crossgl_complex64_equal",
            "!=": "crossgl_complex64_not_equal",
            ">": "crossgl_complex64_greater",
            ">=": "crossgl_complex64_greater_equal",
            "<": "crossgl_complex64_less",
            "<=": "crossgl_complex64_less_equal",
        }
        helper = helpers.get(operator)
        if helper is None:
            self.glsl_complex64_error(
                expr,
                operator,
                (left_type, right_type),
                "operator-unsupported",
            )

        left_operand = self.glsl_complex64_operand(left, left_type, expr.left)
        right_operand = self.glsl_complex64_operand(right, right_type, expr.right)
        if left_operand is None or right_operand is None:
            self.glsl_complex64_error(
                expr,
                operator,
                (left_type, right_type),
                "operand-shape-unsupported",
            )
        self.required_glsl_complex64_helpers.add(helper)
        return f"{helper}({left_operand}, {right_operand})"

    def glsl_complex64_unary_expression(self, expr, operand, operator):
        operand_type = self.glsl_source_expression_type(expr.operand)
        if not self.is_glsl_registered_complex64_type(operand_type):
            return None
        self.glsl_validate_complex64_arithmetic_representation(
            expr,
            operator,
            (operand_type,),
        )
        if operator == "+":
            return operand
        if operator != "-":
            self.glsl_complex64_error(
                expr,
                operator,
                (operand_type,),
                "operator-unsupported",
            )
        helper = "crossgl_complex64_negate"
        self.required_glsl_complex64_helpers.add(helper)
        return f"{helper}({operand})"

    def glsl_complex64_compound_assignment_expression(
        self,
        node,
        left,
        right_node,
        operator,
        left_type,
    ):
        if not self.is_glsl_registered_complex64_type(left_type):
            return None
        if operator == "=":
            return None
        right_type = self.glsl_source_expression_type(right_node)
        self.glsl_validate_complex64_arithmetic_representation(
            node,
            operator,
            (left_type, right_type),
        )
        helpers = {
            "+=": "crossgl_complex64_add_assign",
            "-=": "crossgl_complex64_sub_assign",
            "*=": "crossgl_complex64_mul_assign",
            "/=": "crossgl_complex64_div_assign",
            "%=": "crossgl_complex64_mod_assign",
        }
        helper = helpers.get(operator)
        if helper is None:
            self.glsl_complex64_error(
                node,
                operator,
                (left_type, self.glsl_source_expression_type(right_node)),
                "operator-unsupported",
            )
        right = self.generate_expression(right_node)
        right_operand = self.glsl_complex64_operand(right, right_type, right_node)
        if right_operand is None:
            self.glsl_complex64_error(
                node,
                operator,
                (left_type, right_type),
                "operand-shape-unsupported",
            )
        self.required_glsl_complex64_helpers.add(helper)
        return f"{helper}({left}, {right_operand})"

    def generate_discarded_expression(self, expr):
        if is_comma_expression(expr):
            return c_family_comma_expression(
                expr,
                self.generate_expression,
                self.generate_discarded_expression,
                discarded=True,
            )
        previous_discarded_expression = self.current_discarded_expression
        self.current_discarded_expression = expr
        try:
            return self.generate_expression(expr)
        finally:
            self.current_discarded_expression = previous_discarded_expression

    def glsl_normalized_source_type(self, vtype):
        type_name = self.type_name_string(vtype)
        if not type_name:
            return None
        seen = []
        while type_name not in seen:
            seen.append(type_name)
            alias_target = self.resolve_glsl_type_alias(type_name)
            resolved = re.sub(
                r"\b(?:const|volatile|restrict|device|constant|thread|threadgroup)\b",
                "",
                alias_target,
            )
            resolved = re.sub(r"\s+", " ", resolved).strip()
            if resolved.endswith("&"):
                resolved = resolved[:-1].rstrip()
            resolved = crossgl_named_vector_spelling(resolved)
            if resolved == type_name == alias_target:
                return resolved
            type_name = resolved
        raise OpenGLRefusal(
            f"Cyclic OpenGL type alias: {' -> '.join([*seen, type_name])}"
        )

    def glsl_narrow_integer_contract(self, vtype):
        source_type = self.glsl_normalized_source_type(vtype)
        if not source_type:
            return None
        shape = narrow_integer_shape(source_type)
        if shape is None:
            return None
        kind, bits, width = shape
        return {
            "signed": kind is ArithmeticScalarKind.SIGNED_INTEGER,
            "bits": bits,
            "width": width,
            "mapped": self.map_type(source_type),
        }

    def glsl_half_width(self, vtype):
        source_type = self.glsl_normalized_source_type(vtype) or ""
        if source_type in {"half", "f16", "float16", "float16_t"}:
            return 1
        match = re.fullmatch(
            r"(?:half|packed_half|f16vec|float16_t)([234])", source_type
        )
        return int(match.group(1)) if match else None

    def glsl_bfloat_width(self, vtype):
        source_type = self.glsl_normalized_source_type(vtype) or ""
        if source_type in self.GLSL_BFLOAT16_ALIASES:
            return 1
        match = re.fullmatch(r"(?:bfloat|bfloat16vec)([234])", source_type)
        return int(match.group(1)) if match else None

    def glsl_narrow_float_width(self, vtype):
        return self.glsl_half_width(vtype) or self.glsl_bfloat_width(vtype)

    def glsl_apply_narrow_float_contract(self, value, expected_type, source_node):
        if self.glsl_bfloat_width(expected_type) is not None:
            return self.glsl_apply_bfloat_contract(value, expected_type, source_node)
        return self.glsl_apply_half_contract(value, expected_type, source_node)

    def glsl_apply_bfloat_contract(self, value, expected_type, source_node):
        width = self.glsl_bfloat_width(expected_type)
        source_type = self.glsl_source_expression_type(source_node)
        if self.GLSL_TARGET_DISPLAY_NAME != "OpenGL" or not glsl_version_reaches(
            self, "a bfloat16 conversion", 400
        ):
            self.glsl_scalar_conversion_error(
                source_node, source_type, expected_type, "bfloat-unsupported-profile"
            )
        if width == 1:
            constant = scalar_constant_value(
                source_node,
                constants=(
                    self.glsl_active_literal_int_constants()
                    if self.glsl_generating_global_initializer
                    else None
                ),
                function_names=self.function_return_types,
            )
            if constant is not None:
                rounded = bfloat16_constant_float(constant)
                if math.isfinite(rounded):
                    return repr(rounded)
        constructor = (
            isinstance(source_node, FunctionCallNode)
            and self.glsl_bfloat_width(self.function_call_name(source_node)) == width
            and self.function_call_name(source_node) not in self.function_return_types
        )
        operands = source_node.args if constructor else (source_node,)
        for operand in operands:
            info = self.glsl_value_type_info(self.glsl_source_expression_type(operand))
            exact_integer_literal = False
            if info is not None and info["family"] in {"int", "uint"}:
                literal = scalar_constant_value(
                    operand, function_names=self.function_return_types
                )
                literals = [literal] if literal is not None else None
                if literals is not None:
                    try:
                        exact_integer_literal = all(
                            v == struct.unpack("<f", struct.pack("<f", v))[0]
                            for v in literals
                        )
                    except (OverflowError, struct.error):
                        pass
            if info is not None and (
                (info["family"] == "float" and info["bits"] > 32)
                or (
                    info["family"] in {"int", "uint"}
                    and info["bits"] > 16
                    and not exact_integer_literal
                )
            ):
                if (
                    info["family"] in {"int", "uint"}
                    and info["bits"] in {32, 64}
                    and info["width"] == width == 1
                    and len(operands) == 1
                    and not self.glsl_generating_global_initializer
                ):
                    for builtin in ("findMSB", "uintBitsToFloat"):
                        if (
                            builtin in self.function_return_types
                            or builtin in self.global_variable_types
                        ):
                            self.glsl_scalar_conversion_error(
                                source_node,
                                source_type,
                                expected_type,
                                "bfloat-target-builtin-shadowed",
                            )
                    if self.glsl_bfloat_integer_helper_name is None:
                        used_names = self.glsl_module_used_identifier_names | getattr(
                            self, "glsl_numeric_helper_reserved_names", set()
                        )
                        self.glsl_bfloat_integer_helper_name = (
                            self.glsl_unique_identifier(
                                "crossgl_integer_to_bfloat", used_names
                            )
                        )
                        self.glsl_module_used_identifier_names.add(
                            self.glsl_bfloat_integer_helper_name
                        )
                    if info["bits"] == 64:
                        self.glsl_bfloat_wide_integer_helper_required = True
                    return f"{self.glsl_bfloat_integer_helper_name}({self.generate_expression(operand)})"
                self.glsl_scalar_conversion_error(
                    source_node, info["source"], expected_type, "bfloat-double-rounding"
                )
        if (
            constructor
            and operands
            and all(
                self.glsl_bfloat_width(self.glsl_source_expression_type(arg))
                is not None
                for arg in operands
            )
        ):
            return value
        literals = self.glsl_half_literal_components(source_node)
        if literals is not None:
            if len(literals) == 1:
                literals *= width
            if len(literals) == width:
                try:
                    bits = [
                        struct.unpack("<I", struct.pack("<f", v))[0] for v in literals
                    ]
                    rounded = [
                        struct.unpack(
                            "<f",
                            struct.pack(
                                "<I", (v + 0x7FFF + ((v >> 16) & 1)) & 0xFFFF0000
                            ),
                        )[0]
                        for v in bits
                    ]
                except (OverflowError, struct.error):
                    rounded = []
                if rounded and all(math.isfinite(v) for v in rounded):
                    if all(v & 0xFFFF == 0 for v in bits):
                        return value
                    values = ", ".join(repr(v) for v in rounded)
                    return values if width == 1 else f"vec{width}({values})"
        if self.glsl_generating_global_initializer:
            self.glsl_scalar_conversion_error(
                source_node, source_type, expected_type, "bfloat-global-initializer"
            )
        for builtin in ("floatBitsToUint", "uintBitsToFloat"):
            if (
                builtin in self.function_return_types
                or builtin in self.global_variable_types
            ):
                self.glsl_scalar_conversion_error(
                    source_node,
                    source_type,
                    expected_type,
                    "bfloat-target-builtin-shadowed",
                )
        for required_width in (1, width):
            if required_width not in self.glsl_bfloat_helper_names:
                used_names = self.glsl_module_used_identifier_names | getattr(
                    self, "glsl_numeric_helper_reserved_names", set()
                )
                name = self.glsl_unique_identifier(
                    f"crossgl_round_bfloat{required_width}", used_names
                )
                self.glsl_bfloat_helper_names[required_width] = name
                self.glsl_module_used_identifier_names.add(name)
        return f"{self.glsl_bfloat_helper_names[width]}({value})"

    def glsl_half_literal_components(self, expression):
        if isinstance(expression, LiteralNode):
            value = expression.value
            return [value] if isinstance(value, (int, float)) else None
        if isinstance(expression, UnaryOpNode) and self.map_operator(expression.op) in {
            "+",
            "-",
        }:
            values = self.glsl_half_literal_components(expression.operand)
            if values is not None:
                sign = -1 if self.map_operator(expression.op) == "-" else 1
                return [sign * value for value in values]
        if not isinstance(expression, FunctionCallNode):
            return None
        name = self.function_call_name(expression)
        if name in self.function_return_types:
            return None
        if self.glsl_bfloat_width(name) == 1:
            value = scalar_constant_value(
                expression, function_names=self.function_return_types
            )
            return [value] if value is not None else None
        mapped = self.glsl_constructor_type(name)
        if mapped not in {"float", "vec2", "vec3", "vec4"}:
            return None
        values = []
        for arg in expression.args:
            components = self.glsl_half_literal_components(arg)
            if components is None:
                return None
            values.extend(components)
        width = 1 if mapped == "float" else int(mapped[-1])
        if len(values) == 1:
            values *= width
        if len(values) != width:
            return None
        try:
            values = [
                struct.unpack("<f", struct.pack("<f", value))[0] for value in values
            ]
            if self.glsl_half_width(name) is not None:
                values = [
                    struct.unpack("<e", struct.pack("<e", value))[0] for value in values
                ]
        except (OverflowError, struct.error):
            return None
        return values

    def glsl_apply_half_contract(self, value, expected_type, source_node):
        width = self.glsl_half_width(expected_type)
        if width is None:
            return value
        source_type = self.glsl_source_expression_type(source_node)
        info = self.glsl_value_type_info(source_type)
        if self.GLSL_TARGET_DISPLAY_NAME != "OpenGL" or not glsl_version_reaches(
            self, "a half conversion", 400
        ):
            self.glsl_scalar_conversion_error(
                source_node, source_type, expected_type, "half-unsupported-profile"
            )
        if info is not None and info["family"] == "float" and info["bits"] > 32:
            self.glsl_scalar_conversion_error(
                source_node, source_type, expected_type, "half-double-rounding"
            )
        literals = self.glsl_half_literal_components(source_node)
        if literals is not None:
            if len(literals) == 1:
                literals *= width
            if len(literals) == width:
                try:
                    narrowed = [
                        struct.unpack(
                            "<e",
                            struct.pack(
                                "<e", struct.unpack("<f", struct.pack("<f", value))[0]
                            ),
                        )[0]
                        for value in literals
                    ]
                except (OverflowError, struct.error):
                    narrowed = []
                if narrowed and all(math.isfinite(value) for value in narrowed):
                    values = ", ".join(repr(value) for value in narrowed)
                    return values if width == 1 else f"vec{width}({values})"
        if self.glsl_generating_global_initializer:
            self.glsl_scalar_conversion_error(
                source_node, source_type, expected_type, "half-global-initializer"
            )
        # Repacking existing half components does not perform a numeric conversion.
        if (
            isinstance(source_node, FunctionCallNode)
            and self.glsl_half_width(self.function_call_name(source_node)) == width
            and source_node.args
            and all(
                self.glsl_half_width(self.glsl_source_expression_type(arg)) is not None
                for arg in source_node.args
            )
        ):
            return value
        for builtin in ("floatBitsToUint", "uintBitsToFloat", "findMSB"):
            if (
                builtin in self.function_return_types
                or builtin in self.global_variable_types
            ):
                self.glsl_scalar_conversion_error(
                    source_node,
                    source_type,
                    expected_type,
                    "half-target-builtin-shadowed",
                )
        for required_width in (1, width):
            if required_width in self.glsl_half_helper_names:
                continue
            used_names = set(self.glsl_module_used_identifier_names)
            used_names.update(
                getattr(self, "glsl_numeric_helper_reserved_names", set())
            )
            name = self.glsl_unique_identifier(
                f"crossgl_round_half{required_width}", used_names
            )
            self.glsl_half_helper_names[required_width] = name
            self.glsl_module_used_identifier_names.add(name)
        helper = self.glsl_half_helper_names[width]
        return f"{helper}({value})"

    def glsl_arithmetic_operand_type(self, vtype):
        source_type = self.glsl_normalized_source_type(vtype)
        if not source_type:
            return None
        try:
            mapped_type = self.map_type(source_type)
        except OpenGLTemplateTypeError:
            return None
        return arithmetic_type(source_type, mapped_type)

    def glsl_value_type_info(self, vtype):
        type_name = self.glsl_normalized_source_type(vtype)
        if not type_name:
            return None
        try:
            mapped = self.map_type(type_name)
        except OpenGLTemplateTypeError:
            return None
        target = target_arithmetic_type(mapped)
        if target is None:
            return None
        narrow = self.glsl_narrow_integer_contract(type_name)
        family = {
            ArithmeticScalarKind.BOOLEAN: "bool",
            ArithmeticScalarKind.SIGNED_INTEGER: "int",
            ArithmeticScalarKind.UNSIGNED_INTEGER: "uint",
            ArithmeticScalarKind.FLOATING: "float",
        }[target.kind]
        return {
            "source": type_name,
            "mapped": mapped,
            "family": family,
            "bits": target.bits,
            "width": target.lanes,
            "narrow": narrow,
        }

    def glsl_vector_type_for_component(self, component_type, width):
        component = target_arithmetic_type(component_type)
        if component is None or component.lanes != 1:
            return None
        return arithmetic_type_name(component.kind, component.bits, width)

    def glsl_arithmetic_type_from_info(self, info, *, width=None):
        kind = {
            "bool": ArithmeticScalarKind.BOOLEAN,
            "int": ArithmeticScalarKind.SIGNED_INTEGER,
            "uint": ArithmeticScalarKind.UNSIGNED_INTEGER,
            "float": ArithmeticScalarKind.FLOATING,
        }.get(info["family"])
        if kind is None:
            return None
        narrow = info["narrow"]
        bits = narrow["bits"] if narrow is not None else info["bits"]
        return ArithmeticType(
            info["source"],
            info["mapped"],
            kind,
            bits,
            info["width"] if width is None else width,
        )

    def glsl_promoted_integer_component(self, info):
        operand = self.glsl_arithmetic_type_from_info(info, width=1)
        return promoted_integer_type_name(operand) if operand is not None else None

    def glsl_promoted_integer_type(self, info):
        operand = self.glsl_arithmetic_type_from_info(info)
        return promoted_integer_type_name(operand) if operand is not None else None

    def glsl_arithmetic_conversion_plan(
        self,
        left_type,
        right_type,
        operator,
        *,
        source_node=None,
        fail_closed=False,
    ):
        left = self.glsl_arithmetic_operand_type(left_type)
        right = self.glsl_arithmetic_operand_type(right_type)
        if left is None or right is None:
            return None
        try:
            return resolve_arithmetic_conversion(
                left,
                right,
                operator,
                supported_integer_widths=(
                    self.GLSL_SUPPORTED_ARITHMETIC_INTEGER_WIDTHS
                ),
                supported_floating_widths=(
                    self.GLSL_SUPPORTED_ARITHMETIC_FLOATING_WIDTHS
                ),
            )
        except UnrepresentableArithmeticConversion as error:
            if fail_closed:
                self.glsl_arithmetic_conversion_error(
                    source_node,
                    operator,
                    left_type,
                    right_type,
                    error.attempted_common_type,
                    error.reason,
                )
            return None

    def glsl_arithmetic_conversion_error(
        self,
        source_node,
        operator,
        left_type,
        right_type,
        attempted_common_type,
        reason,
    ):
        operand_types = (
            self.type_name_string(left_type) or "unknown",
            self.type_name_string(right_type) or "unknown",
        )
        rendered_common_type = attempted_common_type or "unavailable"
        raise self.GLSL_ARITHMETIC_CONVERSION_ERROR(
            f"{self.GLSL_TARGET_DISPLAY_NAME} cannot preserve arithmetic operator "
            f"'{operator}' for source operands '{operand_types[0]}' and "
            f"'{operand_types[1]}'; attempted common type "
            f"'{rendered_common_type}': {reason.replace('-', ' ')}",
            operator=operator,
            operand_types=operand_types,
            attempted_common_type=attempted_common_type,
            reason=reason,
            source_location=getattr(source_node, "source_location", None),
        )

    def glsl_half_arithmetic_width(self, left_type, right_type):
        operands = [
            self.glsl_value_type_info(vtype) for vtype in (left_type, right_type)
        ]
        if any(operand is None for operand in operands):
            return None
        half_widths = [self.glsl_half_width(operand["source"]) for operand in operands]
        if not any(half_widths) or any(
            operand["family"] == "float" and half_width is None
            for operand, half_width in zip(operands, half_widths)
        ):
            return None
        return max(operand["width"] for operand in operands)

    def glsl_common_arithmetic_type(self, left_type, right_type, operator):
        plan = self.glsl_arithmetic_conversion_plan(left_type, right_type, operator)
        if plan is None:
            return None
        left = self.glsl_value_type_info(left_type)
        right = self.glsl_value_type_info(right_type)
        if left is None or right is None:
            return None
        half_width = self.glsl_half_arithmetic_width(left_type, right_type)
        if half_width is not None:
            return "half" + (str(half_width) if half_width > 1 else "")
        left_bfloat = self.glsl_bfloat_width(left["source"])
        right_bfloat = self.glsl_bfloat_width(right["source"])
        if left_bfloat is not None and right_bfloat is not None:
            return left["source"] if left_bfloat >= right_bfloat else right["source"]
        if (
            self.glsl_source_type_signature(left["source"])
            == self.glsl_source_type_signature(right["source"])
            and left["narrow"] is None
            and left["family"] != "bool"
        ):
            return left["source"]
        return plan.common_type

    def glsl_source_pointee_type(self, pointer_type):
        pointer_type = self.type_name_string(pointer_type)
        if not pointer_type:
            return None
        pointer_type = pointer_type.rstrip()
        if not pointer_type.endswith(("*", "&")):
            return None
        return pointer_type[:-1].rstrip()

    def glsl_binary_operand_conversion_types(
        self,
        left_type,
        right_type,
        operator,
        *,
        source_node=None,
        fail_closed=False,
    ):
        plan = self.glsl_arithmetic_conversion_plan(
            left_type,
            right_type,
            operator,
            source_node=source_node,
            fail_closed=fail_closed,
        )
        if plan is None:
            return None
        if self.glsl_half_arithmetic_width(left_type, right_type) is not None:
            # Float storage must not erase the source integer-to-half conversion.
            converted = []
            for source, target in (
                (left_type, plan.left_target_type),
                (right_type, plan.right_target_type),
            ):
                if self.glsl_arithmetic_operand_type(source).is_integer:
                    width = target_arithmetic_type(target).lanes
                    target = "half" + (str(width) if width > 1 else "")
                converted.append(target)
            return tuple(converted)
        right_target = plan.right_target_type
        if operator in {"<<", ">>"} and self.GLSL_TARGET_DISPLAY_NAME == "OpenGL":
            count_type = target_arithmetic_type(right_target)
            if count_type is not None and count_type.bits == 64:
                # Source-defined counts fit in 32 bits; native GL drivers may
                # reject wide counts even when glslang accepts the expression.
                right_target = arithmetic_type_name(
                    count_type.kind, 32, count_type.lanes
                )
        return plan.left_target_type, right_target

    def glsl_unary_source_result_type(self, expression):
        operator = self.map_operator(expression.op)
        if operator == "*":
            binding = self.glsl_workgroup_pointer_binding(
                expression.operand,
                self.glsl_workgroup_pointer_aliases(),
            )
            if binding is None:
                binding = self.glsl_storage_pointer_binding(
                    expression.operand,
                    self.glsl_storage_pointer_aliases(),
                )
            if binding is not None and binding.get("element_type") is not None:
                source_element = binding.get("source_element_type")
                return (
                    source_element
                    if self.glsl_narrow_float_width(source_element) is not None
                    else binding["element_type"]
                )

            pointee_type = self.glsl_source_pointee_type(
                self.glsl_source_expression_type(expression.operand)
            )
            if pointee_type is not None:
                return pointee_type

        operand_type = self.glsl_source_expression_type(expression.operand)
        converted_type = self.glsl_unary_operand_conversion_type(expression)
        return converted_type if operator in {"!", "+", "-", "~"} else operand_type

    def glsl_unary_operand_conversion_type(self, expression):
        operand_type = self.glsl_source_expression_type(expression.operand)
        operand = self.glsl_value_type_info(operand_type)
        if operand is None:
            return operand_type
        operator = self.map_operator(expression.op)
        if operator == "!":
            if operand["width"] == 1:
                return "bool"
            return self.glsl_vector_type_for_component("bool", operand["width"])
        if operator in {"+", "-", "~"}:
            return self.glsl_promoted_integer_type(operand) or operand_type
        return operand_type

    def glsl_scalar_conversion_error(self, expr, source_type, target_type, reason):
        raise OpenGLScalarConversionError(
            "OpenGL cannot preserve source conversion from "
            f"'{source_type}' to '{target_type}': {reason.replace('-', ' ')}",
            source_type=source_type,
            target_type=target_type,
            reason=reason,
            source_location=getattr(expr, "source_location", None),
        )

    def glsl_bool_conversion_expression(self, generated, actual, expected, expr):
        if expected["width"] == 1:
            if actual["width"] != 1:
                self.glsl_scalar_conversion_error(
                    expr,
                    actual["source"],
                    expected["source"],
                    "vector-to-scalar",
                )
            if actual["family"] == "bool":
                return generated
            return f"({generated} != {self.zero_value_expression(actual['mapped'])})"

        if actual["width"] not in {1, expected["width"]}:
            self.glsl_scalar_conversion_error(
                expr,
                actual["source"],
                expected["source"],
                "vector-width-mismatch",
            )
        if actual["width"] == 1:
            if actual["family"] == "bool":
                return f"{expected['mapped']}({generated})"
            comparison = (
                f"({generated} != {self.zero_value_expression(actual['mapped'])})"
            )
            return f"{expected['mapped']}({comparison})"
        if actual["family"] == "bool":
            return generated
        return f"notEqual({generated}, {self.zero_value_expression(actual['mapped'])})"

    def generate_glsl_boolean_context(self, expression):
        if isinstance(expression, bool):
            return self.generate_expression(expression)
        source_type = self.glsl_source_expression_type(expression)
        source = self.glsl_value_type_info(source_type)
        if source is None:
            self.glsl_scalar_conversion_error(
                expression,
                source_type or "<unresolved>",
                "bool",
                (
                    "boolean-context-source-type-unresolved"
                    if source_type is None
                    else "boolean-context-source-type-unsupported"
                ),
            )
        if source["family"] not in {"bool", "int", "uint", "float"}:
            self.glsl_scalar_conversion_error(
                expression,
                source["source"],
                "bool",
                "boolean-context-source-type-unsupported",
            )
        return self.generate_expression_with_expected(expression, "bool")

    def glsl_is_explicit_type_constructor_expression(self, expression, target_type):
        prefix = f"{target_type}("
        if not expression.startswith(prefix) or not expression.endswith(")"):
            return False
        depth = 0
        opening_index = len(prefix) - 1
        for index, character in enumerate(expression[opening_index:], opening_index):
            if character == "(":
                depth += 1
            elif character == ")":
                depth -= 1
                if depth == 0:
                    return index == len(expression) - 1
        return False

    def glsl_is_source_constructor_expression(self, expression):
        if isinstance(expression, ConstructorNode):
            return True
        if not isinstance(expression, FunctionCallNode):
            return False
        return (
            self.glsl_constructor_type(self.function_call_name(expression)) is not None
        )

    def glsl_apply_narrow_integer_contract(self, generated, value_type):
        narrow = self.glsl_narrow_integer_contract(value_type)
        if narrow is None:
            return generated
        return f"bitfieldExtract({generated}, 0, {narrow['bits']})"

    def glsl_explicit_literal_conversion(self, generated, expected):
        """Spell a non-negative integer literal in the expected scalar family."""
        if re.fullmatch(r"[0-9]+", generated):
            if expected["family"] == "float":
                return f"{generated}.0"
            if expected["family"] == "uint":
                return f"{generated}u"
        return f"{expected['mapped']}({generated})"

    def glsl_fixed_array_constructor_type(self, mapped_type, extent):
        base_type, array_suffix = split_array_type_suffix(str(mapped_type or ""))
        match = re.fullmatch(r"\[[^\]]*\](.*)", array_suffix)
        if match is None:
            return mapped_type
        return f"{base_type}[{extent}]{match.group(1)}"

    def glsl_indexed_aggregate_literal_shape(self, expr, access_depth):
        dimensions = []
        frontier = [expr]
        for depth in range(access_depth):
            non_aggregates = [
                element
                for element in frontier
                if not isinstance(element, ArrayLiteralNode)
            ]
            if non_aggregates:
                self.glsl_aggregate_error(
                    non_aggregates[0],
                    expected_type=self.current_expression_expected_type,
                    reason="indexed-shape-depth-mismatch",
                    detail=(
                        "the aggregate nesting depth is smaller than the "
                        "subscript depth"
                    ),
                )

            element_counts = {
                len(getattr(element, "elements", []) or []) for element in frontier
            }
            if 0 in element_counts:
                empty = next(
                    element
                    for element in frontier
                    if not (getattr(element, "elements", []) or [])
                )
                self.glsl_aggregate_error(
                    empty,
                    expected_type=self.current_expression_expected_type,
                    reason="empty-indexed-shape",
                    detail=(
                        f"dimension {depth + 1} is empty and has no fixed-array extent"
                    ),
                )
            if len(element_counts) != 1:
                self.glsl_aggregate_error(
                    frontier[0],
                    expected_type=self.current_expression_expected_type,
                    reason="ragged-indexed-shape",
                    detail=(
                        f"dimension {depth + 1} has inconsistent row lengths "
                        f"{sorted(element_counts)}"
                    ),
                )

            extent = next(iter(element_counts))
            dimensions.append(extent)
            frontier = [
                child
                for element in frontier
                for child in getattr(element, "elements", []) or []
            ]
        return dimensions, frontier

    def glsl_indexed_aggregate_literal_leaf_type(self, expr, leaves):
        leaf_types = [self.glsl_source_expression_type(leaf) for leaf in leaves]
        if not leaf_types or any(leaf_type is None for leaf_type in leaf_types):
            self.glsl_aggregate_error(
                expr,
                expected_type=None,
                reason="expected-type-unresolved",
            )

        inferred_type = leaf_types[0]
        for leaf_type in leaf_types[1:]:
            if self.glsl_source_type_signature(
                inferred_type
            ) == self.glsl_source_type_signature(leaf_type):
                continue
            common_type = self.glsl_common_arithmetic_type(
                inferred_type, leaf_type, "?:"
            )
            if common_type is None:
                self.glsl_aggregate_error(
                    expr,
                    expected_type=None,
                    reason="indexed-element-context-incompatible",
                    detail=(
                        "the indexed aggregate leaves do not have a unique "
                        "compatible result type"
                    ),
                )
            inferred_type = common_type
        return inferred_type

    def glsl_indexed_aggregate_literal_expression(self, expr):
        accesses = []
        base = expr
        while isinstance(base, ArrayAccessNode):
            accesses.append(base)
            base = base.array
        if not isinstance(base, ArrayLiteralNode):
            return None

        expected_type = self.type_name_string(self.current_expression_expected_type)
        dimensions, leaves = self.glsl_indexed_aggregate_literal_shape(
            base, len(accesses)
        )
        if not expected_type:
            expected_type = self.glsl_indexed_aggregate_literal_leaf_type(base, leaves)
        base_type, array_suffix = split_array_type_suffix(expected_type)
        indexed_type = (
            f"{base_type}"
            f"{''.join(f'[{extent}]' for extent in dimensions)}"
            f"{array_suffix}"
        )
        rendered = f"({self.generate_expression_with_expected(base, indexed_type)})"
        for access in reversed(accesses):
            index = self.glsl_resource_array_index_expression(access)
            rendered += f"[{index}]"
        return rendered

    def glsl_aggregate_error(
        self,
        expr,
        *,
        expected_type,
        reason,
        expected_element_count=None,
        detail=None,
    ):
        element_count = len(getattr(expr, "elements", []) or [])
        if reason == "expected-type-unresolved":
            message = (
                "OpenGL aggregate initializer requires a concrete destination type"
            )
        elif reason == "element-count-mismatch":
            message = (
                f"OpenGL aggregate initializer for '{expected_type}' expects "
                f"{expected_element_count} elements, got {element_count}"
            )
        elif detail:
            message = (
                f"OpenGL aggregate initializer for '{expected_type}' cannot be "
                f"lowered: {detail}"
            )
        else:
            message = (
                f"OpenGL aggregate initializer for '{expected_type}' cannot be lowered"
            )
        raise OpenGLAggregateInitializerError(
            message,
            expected_type=expected_type,
            element_count=element_count,
            expected_element_count=expected_element_count,
            reason=reason,
            source_location=getattr(expr, "source_location", None),
        )

    def glsl_fixed_array_aggregate_contract(self, expr, mapped_type):
        array_type = self.glsl_outer_array_type(mapped_type)
        if array_type is None:
            self.glsl_aggregate_error(
                expr,
                expected_type=mapped_type,
                reason="array-type-invalid",
                detail="the array suffix is invalid",
            )
        extent_text, element_type = array_type
        if not extent_text:
            self.glsl_aggregate_error(
                expr,
                expected_type=mapped_type,
                reason="unsized-array",
                detail="an unsized array has no constructor extent",
            )
        extent = evaluate_literal_int_expression(
            extent_text,
            self.glsl_active_literal_int_constants(),
        )
        if extent is None:
            self.glsl_aggregate_error(
                expr,
                expected_type=mapped_type,
                reason="array-extent-unresolved",
                detail=f"array extent '{extent_text}' is not a known integer constant",
            )
        if extent <= 0:
            self.glsl_aggregate_error(
                expr,
                expected_type=mapped_type,
                reason="array-extent-invalid",
                detail=f"array extent '{extent}' must be positive",
            )
        return extent, [element_type] * extent

    def glsl_matrix_dimensions(self, mapped_type):
        match = re.fullmatch(r"(d?)mat([234])(?:x([234]))?", mapped_type)
        if match is None:
            return None
        columns = int(match.group(2))
        rows = int(match.group(3) or match.group(2))
        component_type = "double" if match.group(1) else "float"
        column_type = f"{'d' if match.group(1) else ''}vec{rows}"
        return columns, rows, component_type, column_type

    def glsl_vector_initializer_component_count(self, expression):
        type_info = self.glsl_value_type_info(
            self.glsl_source_expression_type(expression)
        )
        if type_info is None:
            type_info = self.glsl_value_type_info(
                self.expression_result_type(expression)
            )
        if type_info is not None:
            return type_info["width"]
        return self.expression_component_count(expression)

    def generate_glsl_vector_aggregate_initializer(self, expr, mapped_type):
        if not self.is_vector_value_type(mapped_type):
            return None
        if self.glsl_is_zero_aggregate_initializer(expr):
            return self.glsl_value_initialized_expression(expr, mapped_type)

        width = self.value_component_count(mapped_type)
        component_type = self.vector_component_type(mapped_type)
        if not width or width <= 1 or component_type is None:
            return None

        rendered = []
        provided_components = 0
        for element in list(getattr(expr, "elements", []) or []):
            component_count = self.glsl_vector_initializer_component_count(element)
            if component_count is None:
                self.glsl_aggregate_error(
                    expr,
                    expected_type=mapped_type,
                    reason="vector-element-type-unresolved",
                    detail="a vector aggregate element has no known component width",
                )
            if provided_components + component_count > width:
                self.glsl_aggregate_error(
                    expr,
                    expected_type=mapped_type,
                    reason="element-count-mismatch",
                    expected_element_count=width,
                )
            rendered.append(
                self.generate_expression_with_expected(
                    element,
                    component_type if component_count == 1 else None,
                )
            )
            provided_components += component_count

        rendered.extend(
            self.glsl_value_initialized_expression(expr, component_type)
            for _index in range(provided_components, width)
        )
        return f"{mapped_type}({', '.join(rendered)})"

    def generate_glsl_partial_vector_constructor(
        self, expr, constructor, source_constructor
    ):
        arguments = list(getattr(expr, "arguments", getattr(expr, "args", [])) or [])
        if not self.is_vector_value_type(constructor) or len(arguments) <= 1:
            return None

        component_counts = [
            self.glsl_vector_initializer_component_count(argument)
            for argument in arguments
        ]
        if any(component_count is None for component_count in component_counts):
            return None

        width = self.value_component_count(constructor)
        provided_components = sum(component_counts)
        if not width or provided_components >= width:
            return None

        component_type = self.vector_component_type(constructor)
        if component_type is None:
            return None
        rendered = [
            self.generate_expression_with_expected(
                argument,
                component_type if component_count == 1 else None,
            )
            for argument, component_count in zip(arguments, component_counts)
        ]
        rendered.extend(
            self.zero_value_expression(component_type)
            for _index in range(provided_components, width)
        )
        generated = f"{constructor}({', '.join(rendered)})"
        return self.glsl_apply_narrow_integer_contract(generated, source_constructor)

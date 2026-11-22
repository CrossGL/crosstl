"""Integer and compound arithmetic in the HLSL generator: minimum-precision and native 16-bit contracts, wide integers, boolean operands, lvalues and operation stabilization."""

import re
from copy import deepcopy

from ..ast import (
    ArrayAccessNode,
    IdentifierNode,
    LiteralNode,
    MemberAccessNode,
    SwizzleNode,
    VariableNode,
)
from .directx_values import (
    DirectXBooleanCompoundAssignmentError,
    DirectXContextualConversionError,
    DirectXMinimumPrecisionIntegerArithmeticError,
    DirectXNative16BitArithmeticError,
)


class DirectXIntegerArithmeticMixin:
    """Integer arithmetic members of ``HLSLCodeGen``."""

    def hlsl_wide_integer_arithmetic_contract(
        self, node, operator, left_type, right_type
    ):
        if operator not in {
            "+",
            "-",
            "*",
            "/",
            "%",
            "&",
            "|",
            "^",
            "<<",
            ">>",
            "==",
            "!=",
            "<",
            "<=",
            ">",
            ">=",
            "?:",
        }:
            return None
        left = self.hlsl_integer_arithmetic_type_info(left_type)
        right = self.hlsl_integer_arithmetic_type_info(right_type)
        if (
            left is None
            or right is None
            or not any(
                info["base_type"] in {"int64_t", "uint64_t"} for info in (left, right)
            )
        ):
            return None

        reason = None
        if left["width"] > 1 and right["width"] > 1:
            if left["width"] != right["width"]:
                reason = "wide-integer-vector-width-mismatch"
            elif (
                operator not in {"<<", ">>"}
                and left["mapped_type"] != right["mapped_type"]
            ):
                reason = "wide-integer-vector-element-mismatch"
        if operator in {"<<", ">>"} and left["width"] == 1 and right["width"] > 1:
            reason = "wide-integer-scalar-vector-shift"
        if reason is not None:
            raise DirectXContextualConversionError(
                f"DirectX cannot preserve '{operator}' between '{left_type}' and "
                f"'{right_type}': incompatible source vector operands",
                source_type=left["mapped_type"],
                target_type=right["mapped_type"],
                reason=reason,
                source_location=getattr(node, "source_location", None),
            )

        # Metal converts a scalar to a vector's element type, but scalar pairs
        # use the usual arithmetic conversions. Shift operands promote separately.
        if operator in {"<<", ">>"}:
            left_base = self.hlsl_promoted_integer_arithmetic_base_type(left)
            right_base = self.hlsl_promoted_integer_arithmetic_base_type(right)
            width = left["width"]
        else:
            if left["width"] > 1 or right["width"] > 1:
                vector = left if left["width"] > 1 else right
                left_base = vector["base_type"]
            else:
                left_base = self.hlsl_common_integer_arithmetic_base_type(left, right)
            right_base = left_base
            width = max(left["width"], right["width"])
        return {
            "left": left,
            "right": right,
            "left_base": left_base,
            "right_base": right_base,
            "operation_type": left_base + (str(width) if width > 1 else ""),
        }

    def hlsl_wide_integer_operands(self, contract, left, right):
        return (
            self.hlsl_widened_integer_operand(
                left, contract["left"], contract["left_base"]
            ),
            self.hlsl_widened_integer_operand(
                right, contract["right"], contract["right_base"]
            ),
        )

    def hlsl_wide_integer_binary_expression(self, expr, left, right, operator):
        contract = self.hlsl_wide_integer_arithmetic_contract(
            expr,
            operator,
            self.expression_result_type(expr.left),
            self.expression_result_type(expr.right),
        )
        if contract is None:
            return None
        left, right = self.hlsl_wide_integer_operands(contract, left, right)
        return f"({left} {operator} {right})"

    def generate_hlsl_wide_integer_compound_assignment(
        self, node, target, value, operator, *, target_type=None, lhs=None
    ):
        operator = self.map_operator(operator)
        if operator not in {
            "+=",
            "-=",
            "*=",
            "/=",
            "%=",
            "&=",
            "|=",
            "^=",
            "<<=",
            ">>=",
        }:
            return None
        binary = operator[:-1]
        target_type = target_type or self.expression_result_type(target)
        value_type = self.expression_result_type(value)
        contract = self.hlsl_wide_integer_arithmetic_contract(
            node, binary, target_type, value_type
        )
        if contract is None:
            return None
        left, right = self.hlsl_wide_integer_operands(contract, "target", "value")
        # A shift count keeps its independent type even when no conversion is
        # needed; assignment-context rendering would impose the lvalue's shape.
        if (left, right) == ("target", "value") and binary not in {"<<", ">>"}:
            return None
        rhs = self.generate_expression_with_expected(value, None)
        if left == "target":
            # Native compound assignment already evaluates its lvalue once.
            lhs = lhs if lhs is not None else self.generate_expression(target)
            rhs = self.hlsl_widened_integer_operand(
                rhs, contract["right"], contract["right_base"]
            )
            return f"{lhs} {operator} {rhs}"
        if self.hlsl_expression_has_observable_side_effects(
            value, allow_integer_constructors=True, allow_buffer_loads=True
        ):
            raise DirectXContextualConversionError(
                "DirectX cannot preserve mixed-width compound assignment when "
                "the right operand may modify the copied assignment target; "
                "materialize the right operand before assignment",
                source_type=self.map_type(value_type),
                target_type=self.map_type(target_type),
                reason="wide-integer-compound-copy-in-alias",
                source_location=getattr(node, "source_location", None),
            )
        target_mapped = contract["left"]["mapped_type"]
        value_mapped = contract["right"]["mapped_type"]
        assignments = []
        if self.hlsl_expression_has_observable_side_effects(target):
            assignments, stable_target = self.hlsl_stabilize_inout_lvalue(
                node, target, target_mapped
            )
            lhs = self.generate_expression(stable_target)
        elif lhs is None:
            lhs = self.generate_expression(target)
        key = (target_mapped, value_mapped, binary, left, right)
        name = self.hlsl_wide_integer_compound_helpers.get(key)
        if name is None:
            used = self.hlsl_wide_integer_reserved_names | set(
                self.local_variable_types
            )
            used.update(self.global_variable_types)
            used.update(self.function_return_types)
            used.update(self.hlsl_wide_integer_compound_helpers.values())
            name = "__crossgl_integer_compound"
            while name in used:
                name += "_"
            self.hlsl_wide_integer_compound_helpers[key] = name
        call = f"{name}({lhs}, {rhs})"
        return f"({', '.join([*assignments, call])})" if assignments else call

    def hlsl_stabilize_inout_lvalue(
        self,
        node,
        target,
        target_type,
        *,
        operation="mixed-width assignment",
        reason="wide-integer-compound-unstable-lvalue",
    ):
        # DXC may evaluate an indexed inout argument on both copy-in and copy-out.
        # Declare private temporaries in the function, but evaluate their values
        # in the original expression so conditional and loop execution is intact.
        assignments = []

        def reject():
            raise DirectXContextualConversionError(
                f"DirectX cannot preserve the indexed {operation} "
                "without a scalar integer index and a stable storage owner",
                target_type=target_type,
                source_type=self.expression_result_type(target),
                reason=reason,
                source_location=getattr(node, "source_location", None),
            )

        def stabilize(expression):
            if isinstance(expression, (IdentifierNode, VariableNode)):
                return expression
            if isinstance(expression, MemberAccessNode):
                expression.object_expr = expression.object = stabilize(
                    expression.object_expr
                )
                return expression
            if isinstance(expression, SwizzleNode):
                expression.vector_expr = stabilize(expression.vector_expr)
                return expression
            if not isinstance(expression, ArrayAccessNode):
                reject()
            expression.array_expr = expression.array = stabilize(expression.array)
            index = expression.index
            if self.literal_int_value(index, {}) is not None:
                return expression
            index_type = self.expression_result_type(index)
            info = self.hlsl_boolean_compound_type_info(index_type)
            if (
                info is None
                or info["kind"] != "integer"
                or info["width"] != 1
                or self.hlsl_wide_integer_compound_indices is None
            ):
                reject()
            mapped_type = self.map_type(index_type)
            used = self.current_identifier_reserved_names | set(
                self.local_variable_types
            )
            used.update(self.hlsl_wide_integer_reserved_names)
            used.update(self.global_variable_types)
            name = self.hlsl_unique_local_identifier("__crossgl_integer_index", used)
            self.current_identifier_reserved_names.add(name)
            self.local_variable_types[name] = mapped_type
            self.hlsl_wide_integer_compound_indices.append((mapped_type, name))
            rendered = self.generate_expression_with_expected(index, None)
            assignments.append(f"{name} = {rendered}")
            expression.index_expr = expression.index = IdentifierNode(name)
            return expression

        return assignments, stabilize(deepcopy(target))

    def generate_hlsl_wide_integer_compound_helpers(self):
        code = ""
        for (
            target,
            value,
            operator,
            left,
            right,
        ), name in self.hlsl_wide_integer_compound_helpers.items():
            code += (
                f"{target} {name}(inout {target} target, {value} value) {{\n"
                f"    target = {target}({left} {operator} {right});\n"
                "    return target;\n}\n\n"
            )
        return code

    def hlsl_native_16_bit_arithmetic_error(
        self,
        node,
        operator,
        left_type,
        right_type,
        *,
        operation_type=None,
        reason,
        detail,
    ):
        mapped_left = self.map_type(left_type) if left_type else None
        mapped_right = self.map_type(right_type) if right_type else None
        raise DirectXNative16BitArithmeticError(
            "DirectX cannot preserve native 16-bit arithmetic for "
            f"'{operator}': {detail}",
            operator=operator,
            left_type=mapped_left,
            right_type=mapped_right,
            operation_type=operation_type,
            reason=reason,
            source_location=getattr(node, "source_location", None),
        )

    def hlsl_native_16_bit_arithmetic_contract(
        self,
        node,
        operator,
        left_type,
        right_type,
    ):
        if not getattr(self, "requires_hlsl_native_16_bit_types", False):
            return None
        if operator not in {
            "+",
            "-",
            "*",
            "/",
            "%",
            "&",
            "|",
            "^",
            "<<",
            ">>",
            "==",
            "!=",
            "<",
            "<=",
            ">",
            ">=",
        }:
            return None

        left_integer = self.hlsl_integer_arithmetic_type_info(left_type)
        right_integer = self.hlsl_integer_arithmetic_type_info(right_type)
        left_floating = self.hlsl_floating_arithmetic_type_info(left_type)
        right_floating = self.hlsl_floating_arithmetic_type_info(right_type)
        left_info = left_integer or left_floating
        right_info = right_integer or right_floating
        if left_info is None or right_info is None:
            return None
        if not any(
            info["base_type"] in self.HLSL_NATIVE_16_BIT_ARITHMETIC_BASE_TYPES
            for info in (left_info, right_info)
        ):
            return None

        left_width = left_info["width"]
        right_width = right_info["width"]
        if left_width > 1 and right_width > 1 and left_width != right_width:
            self.hlsl_native_16_bit_arithmetic_error(
                node,
                operator,
                left_type,
                right_type,
                reason="incompatible-vector-widths",
                detail=(
                    f"vector widths {left_width} and {right_width} cannot be "
                    "combined without changing the source operation"
                ),
            )

        if operator in {"<<", ">>"}:
            if left_integer is None or right_integer is None:
                self.hlsl_native_16_bit_arithmetic_error(
                    node,
                    operator,
                    left_type,
                    right_type,
                    reason="non-integral-shift-operand",
                    detail="Metal shift operands require independent integer promotion",
                )
            if left_width == 1 and right_width > 1:
                self.hlsl_native_16_bit_arithmetic_error(
                    node,
                    operator,
                    left_type,
                    right_type,
                    reason="scalar-left-vector-shift-unsupported",
                    detail=(
                        "a scalar left operand cannot preserve a vector shift-count "
                        "shape"
                    ),
                )
            narrow_vector_bases = self.hlsl_narrow_vector_shift_bases(
                left_integer, right_integer
            )
            if narrow_vector_bases is not None:
                left_operation_base, right_operation_base = narrow_vector_bases
            else:
                left_operation_base = self.hlsl_promoted_integer_arithmetic_base_type(
                    left_integer
                )
                right_operation_base = self.hlsl_promoted_integer_arithmetic_base_type(
                    right_integer
                )
            operation_base = left_operation_base
            result_width = left_width
        else:
            if left_width > 1 and right_width > 1:
                if left_info["mapped_type"] != right_info["mapped_type"]:
                    self.hlsl_native_16_bit_arithmetic_error(
                        node,
                        operator,
                        left_type,
                        right_type,
                        reason="incompatible-vector-element-types",
                        detail=(
                            "Metal does not implicitly convert between vector element "
                            f"types '{left_info['mapped_type']}' and "
                            f"'{right_info['mapped_type']}'"
                        ),
                    )
                operation_base = left_info["base_type"]
            elif left_width > 1 or right_width > 1:
                vector_info = left_info if left_width > 1 else right_info
                scalar_floating = right_floating if left_width > 1 else left_floating
                vector_is_integer = (
                    left_integer is not None
                    if left_width > 1
                    else right_integer is not None
                )
                if vector_is_integer and scalar_floating is not None:
                    self.hlsl_native_16_bit_arithmetic_error(
                        node,
                        operator,
                        left_type,
                        right_type,
                        reason="floating-to-integer-vector-conversion",
                        detail=(
                            "Metal does not implicitly convert a floating scalar to an "
                            f"integer vector '{vector_info['mapped_type']}'"
                        ),
                    )
                operation_base = vector_info["base_type"]
            else:
                floating_infos = [
                    info for info in (left_floating, right_floating) if info is not None
                ]
                if floating_infos:
                    if operator in {"%", "&", "|", "^"}:
                        return None
                    floating_bases = {info["base_type"] for info in floating_infos}
                    if "double" in floating_bases:
                        operation_base = "double"
                    elif "float" in floating_bases:
                        operation_base = "float"
                    elif "float16_t" in floating_bases:
                        operation_base = "float16_t"
                    else:
                        return None
                else:
                    operation_base = self.hlsl_common_integer_arithmetic_base_type(
                        left_integer,
                        right_integer,
                    )

            left_operation_base = operation_base
            right_operation_base = operation_base
            result_width = max(left_width, right_width)

        operation_type = operation_base + (
            "" if result_width == 1 else str(result_width)
        )
        return {
            "left": left_info,
            "right": right_info,
            "operation_base": operation_base,
            "left_operation_base": left_operation_base,
            "right_operation_base": right_operation_base,
            "operation_type": operation_type,
            "result_width": result_width,
        }

    def hlsl_native_16_bit_arithmetic_operand(
        self,
        rendered,
        type_info,
        operation_base,
    ):
        width = type_info["width"]
        target_type = operation_base if width == 1 else f"{operation_base}{width}"
        if type_info["mapped_type"] == target_type:
            return rendered
        if operation_base == "float16_t":
            floating_type = "float" + (str(width) if width > 1 else "")
            return self.hlsl_half_conversion_expression(
                f"{floating_type}({rendered})", width, (type_info["mapped_type"],)
            )
        if type_info["base_type"] == "float16_t" and operation_base in {
            "float",
            "double",
        }:
            for name in (
                "asuint16",
                "asuint",
                "asfloat",
                "__crossgl_binary16_to_float",
            ):
                if name in self.global_variable_types or (
                    name in self.function_return_types
                    and name not in self.hlsl_function_name_aliases
                ):
                    raise DirectXContextualConversionError(
                        f"DirectX cannot preserve binary16 widening: '{name}' is shadowed",
                        source_type=type_info["mapped_type"],
                        target_type=target_type,
                        reason="half-widening-intrinsic-shadowed",
                    )
            uint_type = "uint" if width == 1 else f"uint{width}"
            self.require_hlsl_explicit_bitcast_helper("binary16_to_float")
            decoded = (
                "__crossgl_binary16_to_float(" f"{uint_type}(asuint16({rendered})))"
            )
            return (
                decoded
                if operation_base == "float"
                else f"double{width if width > 1 else ''}({decoded})"
            )
        return f"{target_type}({rendered})"

    def generate_hlsl_native_16_bit_compound_assignment(
        self,
        node,
        target,
        value,
        operator,
        *,
        target_type=None,
        lhs=None,
    ):
        mapped_operator = self.map_operator(operator)
        binary_operator = {
            "+=": "+",
            "-=": "-",
            "*=": "*",
            "/=": "/",
        }.get(mapped_operator)
        if binary_operator is None:
            return None

        target_type = target_type or self.expression_result_type(target)
        target_info = self.hlsl_floating_arithmetic_type_info(target_type)
        if target_info is None or target_info["base_type"] != "float16_t":
            return None
        value_type = self.expression_result_type(value)
        contract = self.hlsl_native_16_bit_arithmetic_contract(
            node,
            binary_operator,
            target_type,
            value_type,
        )
        if contract is None:
            return None
        if self.hlsl_expression_has_observable_side_effects(target):
            self.hlsl_native_16_bit_arithmetic_error(
                node,
                binary_operator,
                target_type,
                value_type,
                operation_type=contract["operation_type"],
                reason="side-effecting-assignment-target",
                detail=(
                    "the assignment target would need to be evaluated more than "
                    "once; materialize its index or pointer before the assignment"
                ),
            )
        if self.hlsl_expression_has_observable_side_effects(
            value
        ) and not self.hlsl_minimum_precision_compound_target_is_stable(target):
            self.hlsl_native_16_bit_arithmetic_error(
                node,
                binary_operator,
                target_type,
                value_type,
                operation_type=contract["operation_type"],
                reason="rhs-may-change-assignment-target",
                detail=(
                    "the side-effecting right operand may change the storage "
                    "selected by the repeated assignment target; materialize its "
                    "index or pointer before the assignment"
                ),
            )

        lhs = lhs or self.generate_expression(target)
        rhs = self.generate_expression_with_expected(value, None)
        left = self.hlsl_native_16_bit_arithmetic_operand(
            lhs,
            contract["left"],
            contract["left_operation_base"],
        )
        right = self.hlsl_native_16_bit_arithmetic_operand(
            rhs,
            contract["right"],
            contract["right_operation_base"],
        )
        operation = f"({left} {binary_operator} {right})"
        narrowing = self.hlsl_contextual_floating_narrowing_expression(
            operation,
            target_type,
            contract["operation_type"],
            source_location=getattr(node, "source_location", None),
        )
        if narrowing is not None:
            return f"{lhs} = {narrowing}"
        return f"{lhs} = {target_info['mapped_type']}({operation})"

    def hlsl_native_16_bit_binary_expression(
        self,
        expr,
        left,
        right,
        operator,
    ):
        contract = self.hlsl_native_16_bit_arithmetic_contract(
            expr,
            operator,
            self.expression_result_type(getattr(expr, "left", None)),
            self.expression_result_type(getattr(expr, "right", None)),
        )
        if contract is None:
            return None
        left = self.hlsl_native_16_bit_arithmetic_operand(
            left,
            contract["left"],
            contract["left_operation_base"],
        )
        right = self.hlsl_native_16_bit_arithmetic_operand(
            right,
            contract["right"],
            contract["right_operation_base"],
        )
        return f"({left} {operator} {right})"

    def hlsl_minimum_precision_integer_arithmetic_error(
        self,
        node,
        operator,
        left_type,
        right_type,
        *,
        expected_type=None,
        context,
        reason,
        detail,
    ):
        mapped_left = self.map_type(left_type) if left_type else None
        mapped_right = self.map_type(right_type) if right_type else None
        mapped_expected = self.map_type(expected_type) if expected_type else None
        raise DirectXMinimumPrecisionIntegerArithmeticError(
            "DirectX cannot lower minimum-precision integer "
            f"'{operator}' in {context}: {detail}",
            operator=operator,
            left_type=mapped_left,
            right_type=mapped_right,
            expected_type=mapped_expected,
            context=context,
            reason=reason,
            source_location=getattr(node, "source_location", None),
        )

    def hlsl_promoted_integer_arithmetic_base_type(self, type_info):
        if type_info["minimum_precision"] or type_info["base_type"] in {
            "int16_t",
            "uint16_t",
        }:
            return "int"
        return type_info["base_type"]

    def hlsl_common_integer_arithmetic_base_type(self, left_info, right_info):
        if (
            left_info["base_type"] == "min16uint"
            and right_info["base_type"] == "min16uint"
        ):
            return "uint"
        left_base = self.hlsl_promoted_integer_arithmetic_base_type(left_info)
        right_base = self.hlsl_promoted_integer_arithmetic_base_type(right_info)
        if left_base == right_base:
            return left_base
        if "uint64_t" in {left_base, right_base}:
            return "uint64_t"
        if "int64_t" in {left_base, right_base}:
            return "int64_t"
        if "uint" in {left_base, right_base}:
            return "uint"
        return "int"

    def hlsl_minimum_precision_integer_operation_contract(
        self,
        node,
        operator,
        left_type,
        right_type,
        *,
        context,
    ):
        left_info = self.hlsl_integer_arithmetic_type_info(left_type)
        right_info = self.hlsl_integer_arithmetic_type_info(right_type)
        has_minimum_precision_operand = any(
            info is not None and info["minimum_precision"]
            for info in (left_info, right_info)
        )
        if not has_minimum_precision_operand:
            return None
        if self.hlsl_floating_arithmetic_type(
            left_type
        ) or self.hlsl_floating_arithmetic_type(right_type):
            return None
        if left_info is None or right_info is None:
            self.hlsl_minimum_precision_integer_arithmetic_error(
                node,
                operator,
                left_type,
                right_type,
                context=context,
                reason="unresolved-integer-operation-type",
                detail=(
                    "both operand types must resolve to scalar or vector integer "
                    "types before signedness-preserving widening"
                ),
            )
        left_width = left_info["width"]
        right_width = right_info["width"]
        if left_width > 1 and right_width > 1 and left_width != right_width:
            self.hlsl_minimum_precision_integer_arithmetic_error(
                node,
                operator,
                left_type,
                right_type,
                context=context,
                reason="incompatible-vector-widths",
                detail=(
                    f"vector widths {left_width} and {right_width} cannot be "
                    "combined without changing the source operation"
                ),
            )
        return {
            "left": left_info,
            "right": right_info,
            "base_type": self.hlsl_common_integer_arithmetic_base_type(
                left_info, right_info
            ),
            "result_width": max(left_width, right_width),
        }

    def hlsl_widened_integer_operand(self, rendered, type_info, base_type):
        width = type_info["width"]
        operation_type = base_type if width == 1 else f"{base_type}{width}"
        if type_info["mapped_type"] == operation_type:
            return rendered
        return f"{operation_type}({rendered})"

    def hlsl_minimum_precision_integer_operation_expression(
        self,
        node,
        operator,
        left,
        right,
        left_type,
        right_type,
        *,
        expected_type=None,
        context,
    ):
        if operator not in {"/", "%"}:
            return None
        contract = self.hlsl_minimum_precision_integer_operation_contract(
            node,
            operator,
            left_type,
            right_type,
            context=context,
        )
        if contract is None:
            return None

        left = self.hlsl_widened_integer_operand(
            left, contract["left"], contract["base_type"]
        )
        right = self.hlsl_widened_integer_operand(
            right, contract["right"], contract["base_type"]
        )
        rendered = f"({left} {operator} {right})"

        expected_info = self.hlsl_integer_arithmetic_type_info(expected_type)
        if expected_info is None or not expected_info["minimum_precision"]:
            return rendered
        if expected_info["width"] != contract["result_width"]:
            self.hlsl_minimum_precision_integer_arithmetic_error(
                node,
                operator,
                left_type,
                right_type,
                expected_type=expected_type,
                context=context,
                reason="incompatible-result-width",
                detail=(
                    f"the operation produces {contract['result_width']} component(s), "
                    f"but the required result has {expected_info['width']}"
                ),
            )
        return f"{expected_info['mapped_type']}({rendered})"

    def hlsl_minimum_precision_integer_binary_expression(
        self, expr, left, right, operator
    ):
        return self.hlsl_minimum_precision_integer_operation_expression(
            expr,
            operator,
            left,
            right,
            self.expression_result_type(getattr(expr, "left", None)),
            self.expression_result_type(getattr(expr, "right", None)),
            context="binary expression",
        )

    def hlsl_minimum_precision_compound_target_is_stable(self, target):
        if self.hlsl_repeatable_array_assignment_target(target):
            return True
        if not isinstance(target, ArrayAccessNode):
            return False

        index = getattr(target, "index", None)
        index_is_literal = isinstance(index, (int, float)) or (
            hasattr(index, "__class__") and "Literal" in str(index.__class__)
        )
        return (
            index_is_literal
            and self.hlsl_minimum_precision_compound_target_is_stable(
                getattr(target, "array", None)
            )
        )

    def generate_hlsl_minimum_precision_integer_compound_assignment(
        self,
        node,
        target,
        value,
        operator,
        *,
        lhs=None,
        target_type=None,
    ):
        operator = self.map_operator(operator)
        binary_operator = {"/=": "/", "%=": "%"}.get(operator)
        if binary_operator is None:
            return None

        target_type = target_type or self.expression_result_type(target)
        target_info = self.hlsl_integer_arithmetic_type_info(target_type)
        if target_info is None or not target_info["minimum_precision"]:
            return None
        value_type = self.expression_result_type(value)
        if self.hlsl_floating_arithmetic_type(value_type):
            return None
        if self.hlsl_expression_has_observable_side_effects(target):
            self.hlsl_minimum_precision_integer_arithmetic_error(
                node,
                binary_operator,
                target_type,
                value_type,
                expected_type=target_type,
                context="compound assignment",
                reason="side-effecting-assignment-target",
                detail=(
                    "the assignment target would need to be evaluated more than "
                    "once; materialize its index or pointer before the assignment"
                ),
            )
        if self.hlsl_expression_has_observable_side_effects(
            value
        ) and not self.hlsl_minimum_precision_compound_target_is_stable(target):
            self.hlsl_minimum_precision_integer_arithmetic_error(
                node,
                binary_operator,
                target_type,
                value_type,
                expected_type=target_type,
                context="compound assignment",
                reason="rhs-may-change-assignment-target",
                detail=(
                    "the side-effecting right operand may change the storage "
                    "selected by the repeated assignment target; materialize its "
                    "index or pointer before the assignment"
                ),
            )

        lhs = lhs or self.generate_expression(target)
        rhs = self.generate_expression_with_expected(value, None)
        operation = self.hlsl_minimum_precision_integer_operation_expression(
            node,
            binary_operator,
            lhs,
            rhs,
            target_type,
            value_type,
            expected_type=target_type,
            context="compound assignment",
        )
        if operation is None:
            return None
        return f"{lhs} = {operation}"

    def hlsl_boolean_compound_assignment_error(
        self,
        node,
        operator,
        target_type,
        right_type,
        *,
        reason,
        detail,
        operation_type=None,
    ):
        mapped_target = self.map_type(target_type) if target_type else None
        mapped_right = self.map_type(right_type) if right_type else None
        raise DirectXBooleanCompoundAssignmentError(
            "DirectX cannot lower boolean compound assignment "
            f"'{operator}': {detail}",
            operator=operator,
            target_type=mapped_target,
            right_type=mapped_right,
            operation_type=operation_type,
            context="compound assignment",
            reason=reason,
            source_location=getattr(node, "source_location", None),
        )

    def hlsl_boolean_compound_type_info(self, vtype):
        mapped_type = self.map_type(vtype) if vtype else None
        if mapped_type == "bool" or mapped_type in {"bool2", "bool3", "bool4"}:
            return {
                "base_type": "int",
                "width": 1 if mapped_type == "bool" else int(mapped_type[-1]),
                "kind": "integer",
                "boolean": True,
            }
        integer_info = self.hlsl_integer_arithmetic_type_info(vtype)
        if integer_info is not None:
            source_component = self.type_name_string(vtype).strip()
            generic_vector = re.fullmatch(r"vec[234]<(.+)>", source_component)
            suffixed_vector = re.fullmatch(r"(.+?)(?:vec)?[234]", source_component)
            if generic_vector is not None:
                source_component = generic_vector.group(1)
            elif suffixed_vector is not None:
                source_component = suffixed_vector.group(1)
            promoted_base = (
                "int"
                if source_component in self.HLSL_NARROW_INTEGER_TYPE_NAMES
                else self.hlsl_promoted_integer_arithmetic_base_type(integer_info)
            )
            return {
                "base_type": promoted_base,
                "width": integer_info["width"],
                "kind": "integer",
                "boolean": False,
            }
        if self.is_scalar_value_type(mapped_type):
            base_type, width = mapped_type, 1
        elif self.is_vector_value_type(mapped_type):
            base_type = self.vector_component_type(mapped_type)
            width = int(mapped_type[-1])
        else:
            return None
        if base_type not in {
            "double",
            "float",
            "float16_t",
            "half",
            "min16float",
            "min10float",
        }:
            return None
        return {
            "base_type": base_type,
            "width": width,
            "kind": "floating",
            "boolean": False,
        }

    def hlsl_boolean_compound_temporary(self, base_name, vtype):
        used_names = set(self.local_variable_types)
        used_names.update(self.global_variable_types)
        used_names.update(self.current_identifier_reserved_names)
        used_names.update(self.current_identifier_aliases.values())
        name = self.hlsl_unique_local_identifier(base_name, used_names)
        self.current_identifier_reserved_names.add(name)
        self.local_variable_types[name] = vtype
        return name

    def hlsl_stabilize_boolean_compound_lvalue(
        self,
        node,
        target,
        operator,
        target_type,
        right_type,
    ):
        def reject(reason, detail):
            self.hlsl_boolean_compound_assignment_error(
                node,
                operator,
                target_type,
                right_type,
                reason=reason,
                detail=detail,
            )

        if self.hlsl_union_access_ancestor(target) is not None:
            reject(
                "union-writeback-unrepresentable",
                (
                    "the target is a reinterpreted union view whose bool writeback "
                    "cannot preserve its storage contract"
                ),
            )
        if (
            self.glsl_buffer_block_array_access(target) is not None
            or self.glsl_buffer_block_member_access(target) is not None
        ):
            reject(
                "byte-address-writeback-unrepresentable",
                (
                    "the target uses ByteAddressBuffer storage and cannot expose a "
                    "repeatable bool lvalue"
                ),
            )

        prefix = []
        stabilized_target = deepcopy(target)

        def stabilize(expression):
            if isinstance(expression, (IdentifierNode, VariableNode)):
                return expression

            if isinstance(expression, MemberAccessNode):
                owner = stabilize(
                    getattr(
                        expression, "object_expr", getattr(expression, "object", None)
                    )
                )
                expression.object_expr = owner
                expression.object = owner
                return expression

            if isinstance(expression, SwizzleNode):
                vector = stabilize(getattr(expression, "vector_expr", None))
                expression.vector_expr = vector
                return expression

            if isinstance(expression, ArrayAccessNode):
                array = stabilize(
                    getattr(
                        expression, "array_expr", getattr(expression, "array", None)
                    )
                )
                expression.array_expr = array
                expression.array = array
                index = getattr(
                    expression, "index_expr", getattr(expression, "index", None)
                )
                index_info = self.hlsl_boolean_compound_type_info(
                    self.expression_result_type(index)
                )
                static_index = (
                    isinstance(index, int)
                    and not isinstance(index, bool)
                    or isinstance(index, LiteralNode)
                    and index_info is not None
                    and not index_info["boolean"]
                    and index_info["kind"] == "integer"
                    and self.literal_int_value(index, {}) is not None
                )
                if static_index:
                    return expression

                if (
                    self.hlsl_fixed_array_return_struct_metadata_for_type(
                        self.expression_result_type(array)
                    )
                    is not None
                ):
                    reject(
                        "dynamic-fixed-array-writeback-unrepresentable",
                        (
                            "a dynamically indexed fixed-array value is exposed "
                            "through a read helper and is not an HLSL lvalue"
                        ),
                    )

                if (
                    index_info is None
                    or index_info["kind"] != "integer"
                    or index_info["width"] != 1
                ):
                    reject(
                        "unresolved-index-type",
                        (
                            "each dynamic lvalue index must resolve to a scalar "
                            "integer type before it can be evaluated once"
                        ),
                    )

                promoted_index_type = index_info["base_type"]
                rendered_index = self.generate_expression_with_expected(index, None)
                index_name = self.hlsl_boolean_compound_temporary(
                    "__crossgl_bool_compound_index", promoted_index_type
                )
                prefix.append(
                    f"{promoted_index_type} {index_name} = "
                    f"{promoted_index_type}({rendered_index})"
                )
                stable_index = IdentifierNode(
                    index_name,
                    source_location=getattr(index, "source_location", None),
                )
                expression.index_expr = stable_index
                expression.index = stable_index
                return expression

            reject(
                "unstable-assignment-target",
                (
                    "only identifier, member, swizzle, and indexed lvalues can be "
                    "stabilized without an HLSL reference"
                ),
            )

        return prefix, stabilize(stabilized_target)

    def generate_hlsl_boolean_compound_assignment(
        self,
        node,
        target,
        value,
        operator,
        *,
        target_type=None,
        statement_context,
    ):
        operator = self.map_operator(operator)
        binary_operator = {
            "+=": "+",
            "-=": "-",
            "*=": "*",
            "/=": "/",
            "%=": "%",
            "<<=": "<<",
            ">>=": ">>",
        }.get(operator)
        target_type = target_type or self.expression_result_type(target)
        right_type = self.expression_result_type(value)
        target_info = self.hlsl_boolean_compound_type_info(target_type)
        if binary_operator is None:
            return None

        def reject(reason, detail, operation_type=None):
            self.hlsl_boolean_compound_assignment_error(
                node,
                operator,
                target_type,
                right_type,
                reason=reason,
                detail=detail,
                operation_type=operation_type,
            )

        mapped_target = self.map_type(target_type) if target_type else None
        if target_info is None or not target_info["boolean"]:
            matrix_shape = (
                self.hlsl_matrix_shape(mapped_target) if mapped_target else None
            )
            if matrix_shape is not None and matrix_shape[0] == "bool":
                reject(
                    "target-shape-unrepresentable",
                    "only scalar and vector bool lvalues have a defined writeback",
                )
            return None
        right_info = self.hlsl_boolean_compound_type_info(right_type)
        if right_info is None:
            reject(
                "unresolved-promoted-operand-type",
                (
                    "the right operand must resolve to a scalar or vector arithmetic "
                    "type before source promotion"
                ),
            )
        if right_info["width"] > 1 and right_info["width"] != target_info["width"]:
            reject(
                "incompatible-vector-widths",
                (
                    f"the {target_info['width']}-component target cannot consume a "
                    f"{right_info['width']}-component promoted right operand"
                ),
            )
        if binary_operator in {"<<", ">>", "%"} and right_info["kind"] != "integer":
            reason = (
                "non-integral-shift-operand"
                if binary_operator in {"<<", ">>"}
                else "non-integral-remainder-operand"
            )
            reject(
                reason,
                f"{binary_operator} operands require integral promotion",
            )

        if binary_operator in {"<<", ">>"}:
            operation_base = "int"
            right_base = right_info["base_type"]
        elif right_info["kind"] == "floating":
            operation_base = right_base = right_info["base_type"]
        else:
            right_base = right_info["base_type"]
            operation_base = (
                right_base if right_base in {"uint64_t", "int64_t", "uint"} else "int"
            )
            right_base = operation_base

        operation_type = operation_base + (
            "" if target_info["width"] == 1 else str(target_info["width"])
        )
        right_operation_type = right_base + (
            "" if right_info["width"] == 1 else str(right_info["width"])
        )
        if not statement_context:
            reject(
                "statement-context-required",
                (
                    "the promoted read, ordered right operand, and bool writeback "
                    "require statement temporaries"
                ),
                operation_type,
            )

        prefix, stable_target = self.hlsl_stabilize_boolean_compound_lvalue(
            node,
            target,
            operator,
            target_type,
            right_type,
        )
        lhs = self.generate_expression(stable_target)
        rhs = self.generate_expression_with_expected(value, None)
        lhs_name = self.hlsl_boolean_compound_temporary(
            "__crossgl_bool_compound_lhs", operation_type
        )
        rhs_name = self.hlsl_boolean_compound_temporary(
            "__crossgl_bool_compound_rhs", right_operation_type
        )
        operation = (
            f"({lhs_name} {binary_operator} {rhs_name}) != "
            f"{self.diagnostic_zero_value_for_type(operation_type)}"
        )
        return "\n".join(
            [
                *prefix,
                f"{operation_type} {lhs_name} = {operation_type}({lhs})",
                f"{right_operation_type} {rhs_name} = "
                f"{right_operation_type}({rhs})",
                f"{lhs} = ({operation})",
            ]
        )

    def hlsl_vector_swizzle(self, width):
        return {1: "x", 2: "xy", 3: "xyz", 4: "xyzw"}[width]

    def generate_hlsl_matrix_resize_constructor(self, target_type, args):
        target_dimensions = self.hlsl_matrix_dimensions(target_type)
        if target_dimensions is None or len(args or []) != 1:
            return None

        arg_type = self.expression_result_type(args[0])
        arg_dimensions = self.hlsl_matrix_dimensions(arg_type)
        if arg_dimensions is None:
            return None

        target_component, target_rows, target_columns = target_dimensions
        _arg_component, arg_rows, arg_columns = arg_dimensions
        if target_rows == arg_rows and target_columns == arg_columns:
            return None

        target_type = self.map_type(target_type)
        rendered_arg = self.generate_expression_with_expected(args[0], None)
        source_base = (
            rendered_arg
            if self.hlsl_expression_is_repeatable(args[0])
            else f"({rendered_arg})"
        )
        rows = []
        for row in range(target_rows):
            if row < arg_rows:
                source_row = f"{source_base}[{row}]"
                if target_columns == arg_columns:
                    rows.append(source_row)
                elif target_columns < arg_columns:
                    rows.append(
                        f"{source_row}.{self.hlsl_vector_swizzle(target_columns)}"
                    )
                else:
                    fill_values = [
                        "1.0" if column == row else "0.0"
                        for column in range(arg_columns, target_columns)
                    ]
                    rows.append(
                        f"{target_component}{target_columns}({source_row}, "
                        f"{', '.join(fill_values)})"
                    )
            else:
                values = [
                    "1.0" if column == row else "0.0"
                    for column in range(target_columns)
                ]
                rows.append(f"{target_component}{target_columns}({', '.join(values)})")
        return f"{target_type}({', '.join(rows)})"

    def hlsl_transpose_result_type(self, vtype):
        matrix_shape = self.hlsl_matrix_shape(vtype)
        if not matrix_shape:
            return vtype
        component_type, rows, columns = matrix_shape
        return f"{component_type}{columns}x{rows}"

    def hlsl_bool_arithmetic_expression(self, expr, left, right, op):
        if op not in {"+", "-", "*", "/", "%"}:
            return None
        result_type = self.map_type(self.expression_result_type(expr))
        if result_type != "bool":
            return None
        left_uint = f"(({left}) ? 1u : 0u)"
        right_uint = f"(({right}) ? 1u : 0u)"
        return f"(({left_uint} {op} {right_uint}) != 0u)"

    def hlsl_bfloat16_constructor_call(self, func_name, args, *, source_location=None):
        if not isinstance(func_name, str) or func_name in getattr(
            self, "function_return_types", {}
        ):
            return None

        vector_constructor = self.hlsl_bfloat16_vector_constructor_call(
            func_name, args, source_location=source_location
        )
        if vector_constructor is not None:
            return vector_constructor

        target_is_bfloat = self.is_hlsl_bfloat16_type(func_name)
        source_type = (
            self.hlsl_source_expression_type(args[0]) if len(args) == 1 else None
        )
        source_is_bfloat = self.is_hlsl_bfloat16_type(source_type)
        if not target_is_bfloat and not source_is_bfloat:
            return None

        if target_is_bfloat:
            if not args:
                return "0u"
            if len(args) != 1:
                raise self.directx_bfloat16_unsupported(
                    "DirectX bfloat16 construction requires zero or one scalar "
                    f"argument, got {len(args)}",
                    operation="bfloat16 constructor",
                    source_type=func_name,
                    reason="invalid-constructor-arity",
                    source_location=source_location,
                )
            rendered = self.generate_expression_with_expected(args[0], None)
            return self.hlsl_bfloat16_conversion_expression(
                rendered,
                func_name,
                source_type,
                source_location=source_location,
            )

        mapped_target = self.map_type(func_name)
        if not self.is_scalar_value_type(mapped_target):
            return None
        if len(args) != 1:
            raise self.directx_bfloat16_unsupported(
                "DirectX exact bfloat16 conversion requires a one-argument scalar "
                f"constructor, got '{func_name}'",
                operation="numeric conversion from bfloat16",
                source_type=source_type,
                reason="unsupported-conversion-target",
                source_location=source_location,
            )
        rendered = self.generate_expression_with_expected(args[0], None)
        return self.hlsl_bfloat16_conversion_expression(
            rendered,
            func_name,
            source_type,
            source_location=source_location,
        )

    def is_hlsl_complex64_type(self, type_text):
        return self.map_type(type_text) == "complex64_t"

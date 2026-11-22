"""Private pointer word views in the HLSL generator: scalar and complex64 views, fixed array and struct reinterpretation, registered structures, and compound results and constructors."""

import re
from hashlib import sha1

from ..ast import (
    ArrayAccessNode,
    ArrayNode,
    ArrayType,
    BinaryOpNode,
    FunctionCallNode,
    IdentifierNode,
    LiteralNode,
    MemberAccessNode,
    PointerReinterpretNode,
    PointerType,
    ReferenceType,
    TernaryOpNode,
    UnaryOpNode,
    VariableNode,
    VectorType,
)
from ..refusals import DirectXRefusal
from ..structure_conversions import (
    StructureConversionKind,
    StructureFieldValue,
    registered_structure_conversion_for_identity,
)
from . import hlsl_byte_records
from .directx_values import (
    DirectXComplexCompoundAssignmentError,
    DirectXPrivatePointerParameterError,
    DirectXStructConversionError,
)
from .generic_struct_utils import format_struct_constructor_expression
from .pointer_reinterpret import (
    PointerReinterpretationError,
    scalar_storage_layout,
)
from .private_member_views import hlsl_member_view_root


class DirectXPrivateWordViewMixin:
    """Private word view members of ``HLSLCodeGen``."""

    def hlsl_registered_structure_scalar_conversion_contract(
        self,
        destination_type,
        source_type,
        *,
        source_location=None,
    ):
        """Return one validated registered structure-to-scalar conversion.

        CrossGL carries structure fields rather than source C++ conversion
        operators.  Preserve a value conversion only when the source identity is
        registered, its ordered field shape is exact, and one field consumes the
        source value.  Unregistered lookalikes remain ordinary HLSL expressions.
        """
        destination_name = self.type_name_string(destination_type)
        source_name = self.type_name_string(source_type)
        if not destination_name or not source_name:
            return None

        mapped_destination = self.map_type(destination_name)
        if not self.is_scalar_value_type(mapped_destination):
            return None

        mapped_source = self.map_type(source_name)
        source_identities = {source_name, mapped_source}
        matching_contracts = {
            contract
            for identity in source_identities
            if (contract := registered_structure_conversion_for_identity(identity))
            is not None
        }
        if not matching_contracts:
            return None
        if len(matching_contracts) != 1:
            raise DirectXStructConversionError(
                "DirectX structure value conversion source identity is ambiguous: "
                f"'{source_name}'",
                destination_type=destination_name,
                source_type=source_name,
                conversion_kind=StructureConversionKind.VALUE_CONVERSION.value,
                reason="source-identity-ambiguous",
                source_location=source_location,
            )
        contract = next(iter(matching_contracts))

        member_types = self.struct_member_types.get(source_name)
        if member_types is None:
            member_types = self.struct_member_types.get(mapped_source)
        actual_shape = (
            None
            if member_types is None
            else tuple(
                (name, self.map_type(member_type))
                for name, member_type in member_types.items()
            )
        )
        expected_shape = tuple(
            (field.name, self.map_type(field.type_name)) for field in contract.fields
        )
        if actual_shape != expected_shape:
            reason = (
                "source-shape-unknown"
                if actual_shape is None
                else "source-shape-mismatch"
            )
            raise DirectXStructConversionError(
                "DirectX cannot preserve registered structure value conversion "
                f"from '{source_name}' to '{destination_name}': "
                f"{reason.replace('-', ' ')}",
                destination_type=destination_name,
                source_type=source_name,
                conversion_kind=StructureConversionKind.VALUE_CONVERSION.value,
                reason=reason,
                source_location=source_location,
            )

        converted_fields = [
            field
            for field in contract.fields
            if field.scalar_value is StructureFieldValue.CONVERTED_SOURCE
        ]
        if len(converted_fields) != 1:
            raise DirectXStructConversionError(
                "DirectX registered structure value conversion requires exactly "
                "one converted source field",
                destination_type=destination_name,
                source_type=source_name,
                conversion_kind=StructureConversionKind.VALUE_CONVERSION.value,
                reason="source-single-evaluation-unsupported",
                source_location=source_location,
            )
        (converted_field,) = converted_fields
        return mapped_destination, converted_field.name, converted_field.type_name

    def hlsl_registered_structure_scalar_conversion_expression(
        self,
        rendered,
        destination_type,
        source_type,
        *,
        source_location=None,
    ):
        conversion = self.hlsl_registered_structure_scalar_conversion_contract(
            destination_type,
            source_type,
            source_location=source_location,
        )
        if conversion is None:
            return None

        mapped_destination, source_field, source_field_type = conversion
        source_component = f"({rendered}).{source_field}"
        if self.is_hlsl_bfloat16_type(destination_type):
            return self.hlsl_float_to_bfloat16_expression(source_component)
        if mapped_destination == "bool":
            return f"({source_component} != 0.0)"
        if mapped_destination == "float16_t":
            # A native float16_t() conversion is not round-to-nearest-even on
            # every D3D device: WARP turns 65520.0f into 65504 instead of inf.
            return self.hlsl_half_conversion_expression(
                source_component,
                1,
                (source_field_type,),
                source_location=source_location,
            )
        return f"{mapped_destination}({source_component})"

    def hlsl_registered_structure_scalar_constructor_call(
        self,
        func_name,
        args,
        *,
        source_location=None,
    ):
        if (
            not isinstance(func_name, str)
            or func_name in getattr(self, "function_return_types", {})
            or len(args) != 1
        ):
            return None

        source_type = self.hlsl_source_expression_type(args[0])
        conversion = self.hlsl_registered_structure_scalar_conversion_contract(
            func_name,
            source_type,
            source_location=source_location,
        )
        if conversion is None:
            return None
        rendered = self.generate_expression_with_expected(args[0], None)
        return self.hlsl_registered_structure_scalar_conversion_expression(
            rendered,
            func_name,
            source_type,
            source_location=source_location,
        )

    def hlsl_complex64_operand(self, rendered, type_text):
        if self.is_hlsl_complex64_type(type_text):
            return rendered
        if self.is_hlsl_bfloat16_type(type_text):
            rendered = self.hlsl_bfloat16_to_float_expression(rendered)
        # Promote a real scalar to complex through the make helper; HLSL has no
        # struct constructor syntax, so `complex64_t(rendered, 0.0)` would fail
        # to compile.
        self.required_hlsl_complex64_helpers.add("__crossgl_complex64_make")
        return f"__crossgl_complex64_make({rendered}, 0.0)"

    def hlsl_complex64_constructor_call(self, func_name, args):
        """Lower a ``complex64_t(...)`` constructor call to the make helper.

        HLSL structs cannot declare constructors, so MLX's ``complex64_t(real,
        imag)`` construction (and its 1-/0-argument forms) must route through
        ``__crossgl_complex64_make`` instead of emitting invalid constructor
        syntax.
        """
        if not isinstance(func_name, str) or self.map_type(func_name) != "complex64_t":
            return None
        rendered = [self.generate_expression(argument) for argument in args or []]
        if len(rendered) == 0:
            real, imag = "0.0", "0.0"
        elif len(rendered) == 1:
            real, imag = rendered[0], "0.0"
        elif len(rendered) == 2:
            real, imag = rendered[0], rendered[1]
        else:
            return None
        self.required_hlsl_complex64_helpers.add("__crossgl_complex64_make")
        return f"__crossgl_complex64_make({real}, {imag})"

    def hlsl_complex64_binary_expression(self, expr, left, right, op):
        helper_names = {
            "+": "__crossgl_complex64_add",
            "-": "__crossgl_complex64_sub",
            "*": "__crossgl_complex64_mul",
            "/": "__crossgl_complex64_div",
            "%": "__crossgl_complex64_mod",
            "==": "__crossgl_complex64_equal",
            "!=": "__crossgl_complex64_not_equal",
            ">": "__crossgl_complex64_greater",
            ">=": "__crossgl_complex64_greater_equal",
            "<": "__crossgl_complex64_less",
            "<=": "__crossgl_complex64_less_equal",
        }
        helper_name = helper_names.get(op)
        if helper_name is None:
            return None

        left_type = self.expression_result_type(getattr(expr, "left", None))
        right_type = self.expression_result_type(getattr(expr, "right", None))
        if not (
            self.is_hlsl_complex64_type(left_type)
            or self.is_hlsl_complex64_type(right_type)
        ):
            return None

        self.required_hlsl_complex64_helpers.add(helper_name)
        return (
            f"{helper_name}("
            f"{self.hlsl_complex64_operand(left, left_type)}, "
            f"{self.hlsl_complex64_operand(right, right_type)})"
        )

    def hlsl_bfloat16_binary_expression(self, expr, left, right, op):
        vector_binary = self.hlsl_bfloat16_vector_binary_expression(
            expr, left, right, op
        )
        if vector_binary is not None:
            return vector_binary
        left_type = self.hlsl_source_expression_type(getattr(expr, "left", None))
        right_type = self.hlsl_source_expression_type(getattr(expr, "right", None))
        left_is_bfloat = self.is_hlsl_bfloat16_type(left_type)
        right_is_bfloat = self.is_hlsl_bfloat16_type(right_type)
        if not left_is_bfloat and not right_is_bfloat:
            return None

        supported = {"+", "-", "*", "/", "==", "!=", "<", "<=", ">", ">=", "&&", "||"}
        if op not in supported:
            raise self.directx_bfloat16_unsupported(
                "DirectX exact bfloat16 lowering does not support operator "
                f"'{op}' on bfloat16 payloads",
                operation=op,
                source_type=left_type if left_is_bfloat else right_type,
                reason="unsupported-bfloat16-operator",
                source_location=getattr(expr, "source_location", None),
            )

        if left_is_bfloat:
            left = self.hlsl_bfloat16_to_float_expression(left)
        if right_is_bfloat:
            right = self.hlsl_bfloat16_to_float_expression(right)
        if op in {"&&", "||"}:
            if left_is_bfloat:
                left = f"({left} != 0.0)"
            if right_is_bfloat:
                right = f"({right} != 0.0)"
        operation = f"({left} {op} {right})"
        if op in {"==", "!=", "<", "<=", ">", ">=", "&&", "||"}:
            return operation

        result_type = self.expression_result_type(expr)
        if self.is_hlsl_bfloat16_type(result_type):
            return self.hlsl_float_to_bfloat16_expression(operation)
        return operation

    def hlsl_complex64_unary_expression(self, expr, operand, op):
        if op != "-":
            return None
        operand_type = self.expression_result_type(getattr(expr, "operand", None))
        if not self.is_hlsl_complex64_type(operand_type):
            return None
        self.required_hlsl_complex64_helpers.add("__crossgl_complex64_negate")
        return f"__crossgl_complex64_negate({operand})"

    def hlsl_bfloat16_unary_expression(self, expr, operand, op):
        vector_unary = self.hlsl_bfloat16_vector_unary_expression(expr, operand, op)
        if vector_unary is not None:
            return vector_unary
        operand_type = self.hlsl_source_expression_type(getattr(expr, "operand", None))
        if not self.is_hlsl_bfloat16_type(operand_type):
            return None
        if op == "+":
            return f"(uint({operand}) & 0xffffu)"
        if op == "-":
            return f"((uint({operand}) ^ 0x8000u) & 0xffffu)"
        if op == "!":
            decoded = self.hlsl_bfloat16_to_float_expression(operand)
            return f"!({decoded} != 0.0)"
        if op in {"++", "--"}:
            if getattr(expr, "is_postfix", False):
                raise self.directx_bfloat16_unsupported(
                    "DirectX exact bfloat16 lowering cannot preserve postfix "
                    f"'{op}' without a statement temporary",
                    operation=op,
                    source_type=operand_type,
                    reason="postfix-temporary-required",
                    source_location=getattr(expr, "source_location", None),
                )
            operand_node = getattr(expr, "operand", None)
            if not self.hlsl_expression_is_repeatable(operand_node):
                raise self.directx_bfloat16_unsupported(
                    "DirectX exact bfloat16 increment requires a stable lvalue",
                    operation=op,
                    source_type=operand_type,
                    reason="side-effecting-increment-target",
                    source_location=getattr(expr, "source_location", None),
                )
            decoded = self.hlsl_bfloat16_to_float_expression(operand)
            delta = "+ 1.0" if op == "++" else "- 1.0"
            rounded = self.hlsl_float_to_bfloat16_expression(f"({decoded} {delta})")
            return f"({operand} = {rounded})"
        raise self.directx_bfloat16_unsupported(
            "DirectX exact bfloat16 lowering does not support unary operator "
            f"'{op}'",
            operation=op,
            source_type=operand_type,
            reason="unsupported-bfloat16-operator",
            source_location=getattr(expr, "source_location", None),
        )

    def hlsl_complex64_compound_assignment_error(
        self,
        node,
        operator,
        target_type,
        right_type,
        *,
        reason,
        detail,
    ):
        raise DirectXComplexCompoundAssignmentError(
            "DirectX cannot lower complex compound assignment "
            f"'{operator}': {detail}",
            operator=operator,
            target_type=self.map_type(target_type) if target_type else None,
            right_type=self.map_type(right_type) if right_type else None,
            reason=reason,
            source_location=getattr(node, "source_location", None),
        )

    def hlsl_complex64_compound_rhs_has_side_effects(self, expression):
        if isinstance(expression, FunctionCallNode):
            func_expr = getattr(
                expression, "function", getattr(expression, "name", None)
            )
            func_name = getattr(func_expr, "name", func_expr)
            if self.hlsl_metal_simd_shuffle_name(func_name) is not None:
                args = getattr(expression, "arguments", getattr(expression, "args", []))
                return any(
                    self.hlsl_expression_has_observable_side_effects(argument)
                    for argument in args
                )
        return self.hlsl_expression_has_observable_side_effects(expression)

    def generate_hlsl_complex64_compound_assignment(
        self,
        node,
        target,
        value,
        operator,
        *,
        target_type=None,
    ):
        operator = self.map_operator(operator)
        helper_name = {
            "+=": "__crossgl_complex64_add",
            "-=": "__crossgl_complex64_sub",
            "*=": "__crossgl_complex64_mul",
            "/=": "__crossgl_complex64_div",
            "%=": "__crossgl_complex64_mod",
        }.get(operator)
        target_type = target_type or self.expression_result_type(target)
        if helper_name is None or not self.is_hlsl_complex64_type(target_type):
            return None

        right_type = self.expression_result_type(value)
        if self.hlsl_expression_has_observable_side_effects(target):
            self.hlsl_complex64_compound_assignment_error(
                node,
                operator,
                target_type,
                right_type,
                reason="side-effecting-assignment-target",
                detail=(
                    "the assignment target would need to be evaluated more than "
                    "once; materialize its index or pointer before the assignment"
                ),
            )
        if self.hlsl_complex64_compound_rhs_has_side_effects(
            value
        ) and not self.hlsl_minimum_precision_compound_target_is_stable(target):
            self.hlsl_complex64_compound_assignment_error(
                node,
                operator,
                target_type,
                right_type,
                reason="rhs-may-change-assignment-target",
                detail=(
                    "the side-effecting right operand may change the storage selected "
                    "by the repeated assignment target; materialize its index or "
                    "pointer before the assignment"
                ),
            )

        lhs = self.generate_expression(target)
        rhs = self.generate_expression_with_expected(value, None)
        self.required_hlsl_complex64_helpers.add(helper_name)
        operation = (
            f"{helper_name}("
            f"{self.hlsl_complex64_operand(lhs, target_type)}, "
            f"{self.hlsl_complex64_operand(rhs, right_type)})"
        )
        return f"{lhs} = {operation}"

    def hlsl_boolean_expression_result_type(self, *operand_types):
        for operand_type in operand_types:
            mapped_type = self.map_type(operand_type)
            if self.is_vector_value_type(mapped_type):
                size = mapped_type[-1:]
                if size in {"2", "3", "4"}:
                    return f"bool{size}"
            matrix_shape = self.hlsl_matrix_shape(mapped_type)
            if matrix_shape:
                _, rows, columns = matrix_shape
                return f"bool{rows}x{columns}"
        return "bool"

    def hlsl_untyped_int_expression(self, expr):
        """Identify expressions DXC can keep in its legacy literal-int type."""
        if isinstance(expr, LiteralNode):
            return (
                isinstance(expr.value, int)
                and not isinstance(expr.value, bool)
                and self.hlsl_literal_result_type(expr) == "int"
            )
        if isinstance(expr, UnaryOpNode):
            return self.map_operator(expr.op) in {"+", "-", "~"} and (
                self.hlsl_untyped_int_expression(expr.operand)
            )
        if isinstance(expr, BinaryOpNode):
            return self.map_operator(expr.op) in {
                "+",
                "-",
                "*",
                "/",
                "%",
                "&",
                "|",
                "^",
            } and all(
                self.hlsl_untyped_int_expression(operand)
                for operand in (expr.left, expr.right)
            )
        if isinstance(expr, TernaryOpNode):
            return all(
                self.hlsl_untyped_int_expression(operand)
                for operand in (expr.true_expr, expr.false_expr)
            )
        return False

    def hlsl_literal_result_type(self, expr):
        literal_type = getattr(getattr(expr, "literal_type", None), "name", None)
        value = getattr(expr, "value", None)
        if not isinstance(value, int) or isinstance(value, bool):
            return literal_type

        mapped_type = self.map_type(literal_type) if literal_type else None
        if mapped_type in {"int64_t", "uint64_t"}:
            return mapped_type
        if mapped_type == "uint":
            return mapped_type
        if value > 0xFFFFFFFF:
            return "uint64_t"
        if value > 0x7FFFFFFF:
            return "uint"
        if value < -0x80000000:
            return "int64_t"
        return literal_type

    def hlsl_pointer_offset_result_type(self, operator, left_type, right_type):
        """Return the pointer's own type for ``p + n``, ``n + p`` and ``p - n``.

        Offsetting a storage pointer or buffer view keeps its element type, so
        ``(p + n)[i]`` reads the element ``p`` points to, never the offset's.
        """

        def addressable(vtype):
            return (
                self.hlsl_pointer_element_type(vtype) is not None
                or self.hlsl_typed_buffer_element_type(
                    vtype,
                    {
                        "Buffer",
                        "StructuredBuffer",
                        "RWBuffer",
                        "RWStructuredBuffer",
                        "RasterizerOrderedBuffer",
                        "RasterizerOrderedStructuredBuffer",
                    },
                )
                is not None
            )

        def offset(vtype):
            info = self.hlsl_integer_arithmetic_type_info(vtype)
            return info is not None and info["width"] == 1

        if operator not in {"+", "-"}:
            return None
        if addressable(left_type) and offset(right_type):
            return left_type
        if operator == "+" and offset(left_type) and addressable(right_type):
            return right_type
        return None

    def hlsl_fixed_array_return_struct_metadata_for_type(self, type_name):
        mapped_type = self.map_type(type_name)
        return self.hlsl_fixed_array_return_struct_metadata.get(mapped_type)

    def hlsl_fixed_array_member_base_expression(self, expr, rendered):
        if isinstance(
            expr,
            (
                ArrayAccessNode,
                FunctionCallNode,
                IdentifierNode,
                MemberAccessNode,
                VariableNode,
            ),
        ):
            return rendered
        if hasattr(expr, "__class__") and any(
            name in str(expr.__class__)
            for name in (
                "ArrayAccess",
                "FunctionCall",
                "Identifier",
                "MemberAccess",
                "Variable",
            )
        ):
            return rendered
        if str(rendered).startswith("("):
            return rendered
        return f"({rendered})"

    def hlsl_fixed_array_read_helper_name(self, struct_name):
        return f"{struct_name}_read"

    def generate_hlsl_fixed_array_return_access(self, array_expr, index_expr):
        array_type = self.expression_result_type(array_expr)
        metadata = self.hlsl_fixed_array_return_struct_metadata_for_type(array_type)
        if metadata is None:
            return None

        array = self.generate_expression(array_expr)
        literal_index = self.literal_int_value(index_expr, self.literal_int_constants)
        if literal_index is not None:
            if not 0 <= literal_index < metadata["size"]:
                raise DirectXRefusal(
                    f"DirectX fixed-array return index {literal_index} is out of "
                    f"range for {metadata['type_text']}"
                )
            field_name = self.hlsl_fixed_array_return_field_name(literal_index)
            base = self.hlsl_fixed_array_member_base_expression(array_expr, array)
            return f"{base}.{field_name}"

        struct_name = self.map_type(array_type)
        self.required_hlsl_fixed_array_read_helpers.add(struct_name)
        index = self.generate_expression(index_expr)
        return (
            f"{self.hlsl_fixed_array_read_helper_name(struct_name)}({array}, {index})"
        )

    def hlsl_resource_pointer_offset_sum(self, base, delta):
        if base in {None, "", "0", "0u"}:
            return delta
        if delta in {None, "", "0", "0u"}:
            return base
        return f"({base} + {delta})"

    def hlsl_private_pointer_word_view_error(
        self,
        expression,
        *,
        source_type,
        target_type,
        reason,
        detail,
        alignment=None,
        access="read",
    ):
        raise PointerReinterpretationError(
            "DirectX cannot preserve a private scalar view from "
            f"'{source_type}' to '{target_type}': {detail}",
            source_type=source_type,
            target_type=target_type,
            address_space=(
                getattr(getattr(expression, "target_type", None), "address_space", None)
                or "thread"
            ),
            alignment=alignment,
            access=access,
            target_backend="directx",
            reason=reason,
            source_location=getattr(expression, "source_location", None),
        )

    def hlsl_private_scalar_struct_reinterpret_node(self, expression):
        if not (
            isinstance(expression, UnaryOpNode)
            and not getattr(expression, "is_postfix", False)
            and self.map_operator(expression.op) == "*"
            and isinstance(expression.operand, PointerReinterpretNode)
        ):
            return None
        reinterpretation = expression.operand
        pointer_type = getattr(reinterpretation, "target_type", None)
        source_address = getattr(reinterpretation, "expression", None)
        source_value = getattr(source_address, "operand", None)
        if (
            not isinstance(source_address, UnaryOpNode)
            or self.map_operator(source_address.op) != "&"
            or not isinstance(source_value, (IdentifierNode, VariableNode))
            or not getattr(source_value, "name", None)
        ):
            return None
        address_space = str(getattr(pointer_type, "address_space", None) or "thread")
        if address_space.lower() not in {"thread", "private", "function"}:
            return None
        target_type = getattr(pointer_type, "pointee_type", None)
        target_name = self.type_name_string(target_type)
        if not target_name or scalar_storage_layout(target_name) is not None:
            return None
        struct = self.structs_by_name.get(target_name)
        if struct is None:
            struct = self.structs_by_name.get(self.map_type(target_name))
        return reinterpretation if struct is not None else None

    def hlsl_private_scalar_struct_reinterpret_contract(
        self,
        expression,
        expected_type=None,
    ):
        reinterpretation = self.hlsl_private_scalar_struct_reinterpret_node(expression)
        if reinterpretation is None:
            return None

        pointer_type = getattr(reinterpretation, "target_type", None)
        target_type_node = getattr(pointer_type, "pointee_type", None)
        target_type = self.type_name_string(target_type_node)
        mapped_target_type = self.map_type(target_type)
        struct = self.structs_by_name.get(target_type)
        if struct is None:
            struct = self.structs_by_name.get(mapped_target_type)

        def reject(reason, detail, *, source_type=None, alignment=None, access="read"):
            self.hlsl_private_pointer_word_view_error(
                reinterpretation,
                source_type=source_type or "unknown",
                target_type=target_type,
                reason=reason,
                detail=detail,
                alignment=alignment,
                access=access,
            )

        address_space = str(getattr(pointer_type, "address_space", None) or "thread")
        if address_space.lower() not in {"thread", "private", "function"}:
            reject(
                "unsupported-private-scalar-struct-address-space",
                f"address space '{address_space}' is not private storage",
            )

        source_address = getattr(reinterpretation, "expression", None)
        source_value = getattr(source_address, "operand", None)
        source_name = getattr(source_value, "name", None)
        if (
            not isinstance(source_address, UnaryOpNode)
            or self.map_operator(source_address.op) != "&"
            or not isinstance(source_value, (IdentifierNode, VariableNode))
            or not source_name
        ):
            reject(
                "private-scalar-struct-backing-unresolved",
                "the source is not the stable address of one scalar identifier",
            )

        source_type_node = (
            self.local_variable_source_types.get(source_name)
            or self.local_variable_types.get(source_name)
            or getattr(source_value, "var_type", getattr(source_value, "vtype", None))
        )
        source_type = self.type_name_string(source_type_node)
        source_layout = scalar_storage_layout(source_type)
        if source_layout is None:
            reject(
                "unsupported-private-scalar-struct-source-layout",
                "the addressed value is not a scalar with a known storage layout",
                source_type=source_type,
            )

        members = [
            member
            for member in getattr(struct, "members", []) or []
            if not self.hlsl_static_struct_member(member)
        ]
        if (
            len(members) != 1
            or getattr(struct, "generic_params", None)
            or getattr(struct, "inheritance", None)
            or getattr(struct, "attributes", None)
            or getattr(struct, "is_union", False)
            or self.hlsl_union_layout_for_type(target_type_node) is not None
        ):
            reject(
                "unsupported-private-scalar-struct-layout",
                "the destination must be a canonical attribute-free struct with "
                "exactly one instance member",
                source_type=source_type,
                alignment=source_layout.byte_width,
            )

        member = members[0]
        member_type_node = getattr(
            member,
            "member_type",
            getattr(member, "vtype", getattr(member, "element_type", None)),
        )
        if (
            isinstance(member, ArrayNode)
            or isinstance(member_type_node, (ArrayType, PointerType, ReferenceType))
            or getattr(member, "attributes", None)
            or getattr(member, "resource_qualifiers", None)
        ):
            reject(
                "unsupported-private-scalar-struct-layout",
                "the sole destination member is not an unqualified scalar value",
                source_type=source_type,
                alignment=source_layout.byte_width,
            )

        member_type = self.type_name_string(member_type_node)
        member_layout = scalar_storage_layout(member_type)
        if (
            member_layout is None
            or source_layout != member_layout
            or source_layout.bit_width not in {8, 16, 32}
        ):
            reject(
                "private-scalar-struct-layout-mismatch",
                "the source scalar and sole destination member do not have the "
                "same supported logical storage layout",
                source_type=source_type,
                alignment=member_layout.byte_width if member_layout else None,
            )

        if (
            expected_type is not None
            and self.map_type(expected_type) != mapped_target_type
        ):
            reject(
                "private-scalar-struct-destination-mismatch",
                f"the destination type '{self.type_name_string(expected_type)}' does "
                f"not match '{target_type}'",
                source_type=source_type,
                alignment=member_layout.byte_width,
            )

        rendered_value = self.generate_expression_with_expected(
            source_value,
            member_type_node,
        )
        if (
            member_layout.kind == "integer"
            and not member_layout.signed
            and member_layout.bit_width < 32
        ):
            mask = (1 << member_layout.bit_width) - 1
            rendered_value = f"({rendered_value} & 0x{mask:x}u)"

        return {
            "reinterpretation": reinterpretation,
            "target_type": target_type,
            "mapped_target_type": mapped_target_type,
            "member": member,
            "member_type": member_type_node,
            "member_layout": member_layout,
            "source_type": source_type,
            "rendered_value": rendered_value,
        }

    def hlsl_private_scalar_struct_reinterpret_initializer(
        self,
        expected_type,
        expression,
    ):
        contract = self.hlsl_private_scalar_struct_reinterpret_contract(
            expression,
            expected_type,
        )
        if contract is None:
            return None
        member = contract["member"]
        return (
            contract["mapped_target_type"],
            [(member.name, contract["member_type"])],
            [contract["rendered_value"]],
            [None],
        )

    def generate_hlsl_private_scalar_struct_reinterpret_read(self, expression):
        contract = self.hlsl_private_scalar_struct_reinterpret_contract(expression)
        if contract is None:
            return None
        return format_struct_constructor_expression(
            self,
            contract["mapped_target_type"],
            [contract["rendered_value"]],
        )

    def hlsl_private_pointer_word_component_layout(self, value_type):
        if isinstance(value_type, ArrayType):
            extent = self.literal_int_value(
                getattr(value_type, "size", None), self.literal_int_constants
            )
            if extent is None or extent <= 0:
                return None, (
                    "dynamic-private-view-extent",
                    "a backing array extent is not a positive constant",
                )
            element, failure = self.hlsl_private_pointer_word_component_layout(
                value_type.element_type
            )
            if failure is not None:
                return None, failure
            if element["kind"] not in {"scalar", "vector"}:
                return None, (
                    "unsupported-private-view-source-layout",
                    "nested backing arrays do not have a proven word layout",
                )
            return {
                **element,
                "kind": f"array_{element['kind']}",
                "array_extent": extent,
                "word_extent": (
                    None
                    if element["word_extent"] is None
                    else extent * element["word_extent"]
                ),
                "size_bytes": extent * element["size_bytes"],
            }, None

        type_name = self.type_name_string(value_type)
        scalar_layout = scalar_storage_layout(type_name)
        byte_scalar = hlsl_byte_records.byte_component(
            scalar_layout, self.map_type(type_name)
        )
        if byte_scalar is not None:
            return byte_scalar, None
        if (
            scalar_layout is not None
            and scalar_layout.bit_width == 32
            and self.map_type(type_name) in {"float", "int", "uint"}
        ):
            return {
                "kind": "scalar",
                "layout": scalar_layout,
                "mapped_type": self.map_type(type_name),
                "word_extent": 1,
                "size_bytes": 4,
                "alignment": 4,
            }, None

        if isinstance(value_type, VectorType):
            vector_width = self.literal_int_value(
                getattr(value_type, "size", None), self.literal_int_constants
            )
            element_name = self.type_name_string(value_type.element_type)
            component_layout = scalar_storage_layout(element_name)
            mapped_vector = self.map_type(type_name)
        else:
            mapped_vector = self.map_type(type_name)
            vector_match = re.fullmatch(r"(float|int|uint)([234])", mapped_vector or "")
            if vector_match is None:
                return None, (
                    "unsupported-private-view-source-layout",
                    "backing members must contain 32-bit scalar words",
                )
            vector_width = int(vector_match.group(2))
            component_layout = scalar_storage_layout(vector_match.group(1))

        if (
            component_layout is None
            or component_layout.bit_width != 32
            or vector_width not in {2, 3, 4}
        ):
            return None, (
                "unsupported-private-view-source-layout",
                "backing vectors must contain 32-bit scalar words",
            )
        if vector_width == 3:
            return None, (
                "padded-private-view-layout",
                "three-component vectors have an implicit fourth-word footprint",
            )
        return {
            "kind": "vector",
            "layout": component_layout,
            "mapped_type": mapped_vector,
            "vector_width": vector_width,
            "word_extent": vector_width,
            "size_bytes": vector_width * 4,
            "alignment": vector_width * 4,
        }, None

    def hlsl_private_pointer_word_view_backing(self, layout, backing_name):
        backing = self.hlsl_identifier_name(backing_name)
        if layout["backing_kind"] in {"array_member", "vector_member"}:
            backing += f".{self.hlsl_identifier_name(layout['member'])}"
        return backing

    def hlsl_private_pointer_word_segment_expression(
        self, value_name, segment, word_index
    ):
        value = value_name
        if segment.get("member"):
            value += f".{self.hlsl_identifier_name(segment['member'])}"
        kind = segment["kind"]
        if kind == "scalar":
            return value
        if kind in {"array_scalar", "vector"}:
            return f"{value}[uint({word_index})]"
        if kind == "array_vector":
            width = segment["vector_width"]
            return (
                f"{value}[uint(({word_index}) / {width}u)]"
                f"[uint(({word_index}) % {width}u)]"
            )
        return None

    def hlsl_private_pointer_word_bits_expression(self, expression, layout):
        if layout.kind == "floating" or layout.signed:
            return f"asuint({expression})"
        return expression

    def generate_hlsl_private_pointer_word_view_helpers(self):
        layouts = {
            layout["helper_name"]: layout
            for layout in self.hlsl_private_pointer_word_views.values()
            if layout.get("backing_kind") == "aggregate"
        }
        code = ""
        for helper_name, layout in sorted(layouts.items()):
            code += (
                f"uint {helper_name}(in {layout['parameter_type']} value, "
                "uint word_index) {\n"
            )
            for segment in layout["segments"]:
                start = segment["word_offset"]
                end = start + segment["word_extent"]
                relative_index = (
                    "word_index" if start == 0 else f"(word_index - {start}u)"
                )
                value = self.hlsl_private_pointer_word_segment_expression(
                    "value", segment, relative_index
                )
                bits = self.hlsl_private_pointer_word_bits_expression(
                    value, segment["layout"]
                )
                code += f"    if (word_index < {end}u) {{\n"
                code += f"        return {bits};\n"
                code += "    }\n"
            code += "    return 0u;\n"
            code += "}\n\n"
        return code

    def hlsl_private_pointer_word_view_layout(self, expression, source_type):
        source_name = self.type_name_string(source_type)
        pointer_type = getattr(expression, "target_type", None)
        target_type = getattr(pointer_type, "pointee_type", None)
        target_name = self.type_name_string(target_type)
        target_layout = scalar_storage_layout(target_name)
        alignment = target_layout.byte_width if target_layout else None

        def reject(reason, detail):
            self.hlsl_private_pointer_word_view_error(
                expression,
                source_type=source_name,
                target_type=target_name,
                reason=reason,
                detail=detail,
                alignment=alignment,
            )

        address_space = str(getattr(pointer_type, "address_space", None) or "thread")
        if address_space.lower() not in {"thread", "private", "function"}:
            reject(
                "unsupported-private-address-space",
                f"address space '{address_space}' is not private storage",
            )
        if (
            not isinstance(pointer_type, PointerType)
            or target_layout is None
            or target_layout.bit_width not in {8, 16, 32}
            or (target_layout.kind == "floating" and target_layout.bit_width != 32)
        ):
            reject(
                "unsupported-private-view-target-layout",
                "the target must be an 8-, 16-, or 32-bit scalar view",
            )

        struct_node = self.structs_by_name.get(source_name)
        direct_component, direct_failure = (
            self.hlsl_private_pointer_word_component_layout(source_type)
            if struct_node is None
            else (None, None)
        )
        if struct_node is None and direct_component is None:
            reason, detail = direct_failure or (
                "heterogeneous-private-view-layout",
                "the backing type has no fixed aggregate or vector word layout",
            )
            reject(reason, detail)
        struct_alignment, canonical_attributes = (
            hlsl_byte_records.record_alignment(
                struct_node,
                lambda value: self.literal_int_value(value, self.literal_int_constants),
            )
            if struct_node is not None
            else (None, True)
        )
        if struct_node is not None and (
            not canonical_attributes
            or getattr(struct_node, "inheritance", None)
            or self.hlsl_union_layout_for_type(source_type) is not None
        ):
            reject(
                "unsupported-private-view-aggregate-layout",
                "the backing struct has noncanonical layout metadata",
            )
        members = (
            [
                member
                for member in getattr(struct_node, "members", []) or []
                if not self.hlsl_static_struct_member(member)
            ]
            if struct_node is not None
            else []
        )
        if struct_node is not None and not members:
            reject(
                "heterogeneous-private-view-layout",
                "the backing struct has no instance words",
            )

        segments = []
        byte_extent = 0
        maximum_alignment = struct_alignment or 1
        if direct_component is not None:
            segments.append({**direct_component, "member": None, "word_offset": 0})
            byte_extent = direct_component["size_bytes"]
            maximum_alignment = max(maximum_alignment, direct_component["alignment"])
        else:
            for member in members:
                if getattr(member, "attributes", None):
                    reject(
                        "unsupported-private-view-aggregate-layout",
                        "a backing member has noncanonical layout metadata",
                    )
                component, failure = self.hlsl_private_pointer_word_component_layout(
                    getattr(member, "member_type", None)
                )
                if component is None:
                    reason, detail = failure
                    reject(reason, detail)
                alignment = component["alignment"]
                aligned_offset = (byte_extent + alignment - 1) // alignment * alignment
                if aligned_offset != byte_extent:
                    reject(
                        "padded-private-view-layout",
                        f"member '{member.name}' requires implicit alignment padding",
                    )
                segments.append(
                    {
                        **component,
                        "member": member.name,
                        "word_offset": byte_extent // 4,
                    }
                )
                byte_extent += component["size_bytes"]
                maximum_alignment = max(maximum_alignment, alignment)

        padded_extent = (
            (byte_extent + maximum_alignment - 1)
            // maximum_alignment
            * maximum_alignment
        )
        if padded_extent != byte_extent:
            reject(
                "padded-private-view-layout",
                "the backing aggregate has implicit trailing padding",
            )
        if not hlsl_byte_records.byte_segments_are_viewable(segments):
            reject(
                "heterogeneous-private-view-layout",
                "byte backings must be one fixed byte array",
            )
        word_extent = byte_extent // 4

        sole_segment = segments[0] if len(segments) == 1 else None
        if sole_segment is not None and sole_segment["kind"] == "array_scalar":
            backing_kind = "array_member" if struct_node is not None else "array"
            source_layout = sole_segment["layout"]
            element_extent = sole_segment["array_extent"]
            parameter_type = f"{self.map_type(source_layout.name)}[{element_extent}]"
            member_name = sole_segment.get("member")
            variant = (
                "word_view",
                source_layout.name,
                target_layout.name,
                element_extent,
            )
        elif sole_segment is not None and sole_segment["kind"] == "vector":
            backing_kind = "vector_member" if struct_node is not None else "vector"
            source_layout = sole_segment["layout"]
            parameter_type = sole_segment["mapped_type"]
            member_name = sole_segment.get("member")
            variant = (
                "word_view_vector",
                parameter_type,
                source_layout.name,
                target_layout.name,
                word_extent,
            )
        else:
            backing_kind = "aggregate"
            source_layout = scalar_storage_layout("uint")
            parameter_type = self.map_type(source_name)
            member_name = None
            segment_signature = tuple(
                (
                    segment.get("member"),
                    segment["kind"],
                    segment["layout"].name,
                    segment["word_extent"],
                    segment.get("array_extent"),
                    segment.get("vector_width"),
                )
                for segment in segments
            )
            variant = (
                "word_view_aggregate",
                parameter_type,
                target_layout.name,
                segment_signature,
            )
        helper_name = None
        if backing_kind == "aggregate":
            digest = sha1(repr(variant).encode("utf-8")).hexdigest()[:10]
            helper_name = self.hlsl_sanitized_identifier(
                f"__crossgl_private_word_{source_name}_{digest}"
            )
        layout = {
            "variant": variant,
            "source_type": source_name,
            "source_layout": source_layout,
            "target_layout": target_layout,
            "member": member_name,
            "segments": segments,
            "backing_kind": backing_kind,
            "parameter_type": parameter_type,
            "helper_name": helper_name,
            "word_extent": word_extent,
            "target_extent": byte_extent // target_layout.byte_width,
        }
        self.hlsl_private_pointer_word_views.setdefault(variant, layout)
        self.hlsl_private_pointer_reinterpret_word_views[id(expression)] = layout
        return layout

    def hlsl_private_pointer_word_view_for_variant(self, variant):
        return self.hlsl_private_pointer_word_views.get(variant)

    def hlsl_private_pointer_variant_extent(self, variant):
        layout = self.hlsl_private_pointer_word_view_for_variant(variant)
        return layout["target_extent"] if layout is not None else variant

    def hlsl_private_pointer_view_binding(self, expression):
        local_extents = self.function_private_pointer_local_array_extents.get(
            self.current_function_name, {}
        )
        scalar_parameters = self.function_private_pointer_scalar_parameters.get(
            self.current_function_name, set()
        )
        if isinstance(expression, PointerReinterpretNode):
            if self.hlsl_resource_pointer_binding(expression.expression) is not None:
                return None
            layout = self.hlsl_private_pointer_reinterpret_word_views.get(
                id(expression)
            )
            operand = getattr(expression, "expression", None)
            backing = getattr(operand, "operand", None)
            backing_name = getattr(backing, "name", None)
            if (
                layout is None
                or not isinstance(operand, UnaryOpNode)
                or operand.op != "&"
                or not backing_name
            ):
                self.hlsl_private_pointer_word_view_error(
                    expression,
                    source_type=self.type_name_string(
                        self.expression_result_type(operand)
                    )
                    or "unknown",
                    target_type=self.type_name_string(
                        getattr(expression.target_type, "pointee_type", None)
                    ),
                    reason="missing-private-view-backing",
                    detail="the reinterpretation has no direct local struct backing",
                )
            return {
                "backing": self.hlsl_private_pointer_word_view_backing(
                    layout, backing_name
                ),
                "offset": "0",
                "root_kind": "local_word_view",
                "word_view": layout["variant"],
            }

        if isinstance(expression, (str, IdentifierNode, VariableNode)):
            raw_name = (
                expression
                if isinstance(expression, str)
                else getattr(expression, "name", None)
            )
            if raw_name in self.current_hlsl_private_pointer_base_names:
                variant = self.current_hlsl_private_pointer_variant.get(raw_name)
                if self.hlsl_private_pointer_word_view_for_variant(variant) is not None:
                    return {
                        "backing": self.hlsl_identifier_name(raw_name),
                        "offset": self.current_hlsl_private_pointer_base_names[
                            raw_name
                        ],
                        "root_kind": "word_view_parameter",
                        "word_view": variant,
                    }
                if raw_name in scalar_parameters:
                    return {
                        "backing": self.hlsl_identifier_name(raw_name),
                        "offset": "0",
                        "root_kind": "scalar_parameter",
                    }
                return {
                    "backing": self.hlsl_identifier_name(raw_name),
                    "offset": self.current_hlsl_private_pointer_base_names[raw_name],
                    "root_kind": "parameter",
                }
            if raw_name in local_extents:
                return {
                    "backing": self.hlsl_identifier_name(raw_name),
                    "offset": "0",
                    "root_kind": "local",
                }
            return None

        if isinstance(expression, ArrayAccessNode):
            row = self.hlsl_private_pointer_array_row(
                expression,
                self.current_hlsl_private_pointer_base_names,
                local_extents,
                self.function_private_pointer_local_backing_types.get(
                    self.current_function_name, {}
                ),
                self.literal_int_constants,
            )
            if row is None:
                return None
            if any(
                self.hlsl_private_pointer_expression_has_side_effects(index)
                for index in row[3]
            ):
                raise DirectXPrivatePointerParameterError(
                    "DirectX cannot emit a side-effecting private array row index",
                    function_name=self.current_function_name,
                    reason="side-effecting-view-offset",
                    source_location=getattr(expression, "source_location", None),
                )
            return {
                "backing": self.generate_expression(expression),
                "offset": "0",
                "root_kind": "local",
            }

        if isinstance(expression, UnaryOpNode) and expression.op == "&":
            operand = expression.operand
            if isinstance(operand, (IdentifierNode, VariableNode)):
                name = getattr(operand, "name", None)
                if name not in local_extents and name not in scalar_parameters:
                    return {
                        "backing": self.hlsl_identifier_name(name),
                        "offset": "0",
                        "root_kind": "scalar",
                    }
            if isinstance(operand, MemberAccessNode):
                excluded = (
                    set(local_extents)
                    | set(scalar_parameters)
                    | set(self.current_hlsl_private_pointer_base_names)
                )
                if hlsl_member_view_root(self, operand, excluded) is None:
                    return None
                return {
                    "backing": self.generate_expression(operand),
                    "offset": "0",
                    "root_kind": "scalar",
                    "member_view": True,
                }
            if not isinstance(operand, ArrayAccessNode):
                return None
            binding = self.hlsl_private_pointer_view_binding(operand.array)
            delta_expression = operand.index
            operator = "+"
        elif isinstance(expression, BinaryOpNode) and expression.op in {"+", "-"}:
            binding = self.hlsl_private_pointer_view_binding(expression.left)
            delta_expression = expression.right
            operator = expression.op
            if binding is None and expression.op == "+":
                binding = self.hlsl_private_pointer_view_binding(expression.right)
                delta_expression = expression.left
            if binding is None:
                return None
        else:
            return None

        if self.hlsl_private_pointer_expression_has_side_effects(delta_expression):
            raise DirectXPrivatePointerParameterError(
                "DirectX cannot emit a side-effecting private pointer view offset",
                function_name=self.current_function_name,
                reason="side-effecting-view-offset",
                source_location=getattr(expression, "source_location", None),
            )
        rendered_delta = self.generate_expression(delta_expression)
        if operator == "-":
            rendered_delta = f"-({rendered_delta})"
        binding["offset"] = self.hlsl_resource_pointer_offset_sum(
            binding["offset"], rendered_delta
        )
        return binding

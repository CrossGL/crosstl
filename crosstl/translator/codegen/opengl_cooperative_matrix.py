"""Cooperative matrices in the GLSL generator: expected-type expressions, trailing zero built-ins, fragment and result contracts, multiply-accumulate and interface block values."""

from ..ast import (
    ArrayAccessNode,
    ArrayNode,
    BinaryOpNode,
    CooperativeMatrixOpNode,
    CooperativeMatrixType,
    IdentifierNode,
    LiteralNode,
    MemberAccessNode,
    UnaryOpNode,
)
from ..cooperative_matrix import (
    get_cooperative_matrix_fragment_mapping,
    has_cooperative_matrix_fragment_mapping,
)
from .array_utils import split_array_type_suffix
from .boolean_intrinsics import (
    is_boolean_type,
    ordered_boolean_minmax_width,
)
from .cooperative_matrix_diagonal import generate_glsl_cooperative_matrix_diagonal
from .glsl_narrow_integer_literals import narrowed_integer_literal
from .glsl_shared_array_values import (
    allow_shared_array_value,
    shared_array_value_root,
)
from .opengl_values import (
    OpenGLCooperativeMatrixError,
    OpenGLTrailingZeroBuiltinError,
    _opengl_cooperative_matrix_contract_type,
    _opengl_cooperative_matrix_detail_value,
    _opengl_cooperative_matrix_fragment_contract,
)


class OpenGLCooperativeMatrixMixin:
    """Cooperative matrix members of ``GLSLCodeGen``."""

    def generate_expression_with_expected(self, expr, expected_type):
        expected_type = self.type_name_string(expected_type)
        narrowed = narrowed_integer_literal(self, expr, expected_type)
        if narrowed is not None:
            return narrowed
        interface_value = self.glsl_interface_block_value_expression(
            expr, expected_type
        )
        if interface_value is not None:
            return interface_value
        previous_expected_type = self.current_expression_expected_type
        self.current_expression_expected_type = expected_type
        try:
            with allow_shared_array_value(
                self, shared_array_value_root(self, expr, expected_type)
            ):
                generated = self.generate_expression(expr)
        finally:
            self.current_expression_expected_type = previous_expected_type
        converted = self.glsl_expected_type_conversion(expr, generated, expected_type)
        half_width = self.glsl_half_width(expected_type)
        if half_width is not None and half_width == self.glsl_half_width(
            self.glsl_source_expression_type(expr)
        ):
            return converted
        bfloat_width = self.glsl_bfloat_width(expected_type)
        if bfloat_width is not None and bfloat_width == self.glsl_bfloat_width(
            self.glsl_source_expression_type(expr)
        ):
            return converted
        return self.glsl_apply_narrow_float_contract(converted, expected_type, expr)

    def glsl_interface_block_value_expression(self, expression, expected_type):
        root_name = self.glsl_interface_block_value_root_name(expression)
        if root_name is None:
            return None

        # The root's types are recorded under its emitted name, which is its alias if it has one.
        emitted_root_name = self.current_identifier_aliases.get(root_name)
        if emitted_root_name is None:
            emitted_root_name = root_name
        source_type = self.glsl_uniform_block_variable_types.get(emitted_root_name)
        is_storage_block = False
        if source_type is None:
            source_type = self.glsl_buffer_block_variable_types.get(emitted_root_name)
            is_storage_block = source_type is not None
        if source_type is None and emitted_root_name != root_name:
            source_type = self.glsl_uniform_block_variable_types.get(root_name)
            if source_type is None:
                source_type = self.glsl_buffer_block_variable_types.get(root_name)
                is_storage_block = source_type is not None
        if source_type not in self.glsl_interface_value_struct_names:
            return None

        expected_base_type, expected_array_suffix = split_array_type_suffix(
            expected_type or ""
        )
        if expected_array_suffix or expected_base_type.strip() != source_type:
            return None

        struct = self.structs_by_name.get(source_type)
        if struct is None:
            return None
        if is_storage_block and self.glsl_struct_has_unsized_array_member(struct):
            return None
        members = [
            member
            for member in getattr(struct, "members", []) or []
            if not self.glsl_static_struct_member(member)
        ]
        if not members:
            return None

        base_expression = self.generate_expression(expression)
        arguments = [
            f"{base_expression}.{self.glsl_struct_member_name(source_type, member.name)}"
            for member in members
        ]
        return f"{self.map_type(source_type)}({', '.join(arguments)})"

    def glsl_struct_has_unsized_array_member(self, struct):
        for member in getattr(struct, "members", []) or []:
            if isinstance(member, ArrayNode) and getattr(member, "size", None) is None:
                return True
            member_type = getattr(member, "member_type", None)
            while member_type is not None and hasattr(member_type, "element_type"):
                if (
                    "ArrayType" in member_type.__class__.__name__
                    and getattr(member_type, "size", None) is None
                ):
                    return True
                member_type = getattr(member_type, "element_type", None)
        return False

    def glsl_interface_block_value_root_name(self, expression):
        if isinstance(expression, IdentifierNode):
            return expression.name
        if isinstance(expression, str):
            return expression
        if isinstance(expression, ArrayAccessNode) or (
            hasattr(expression, "__class__")
            and "ArrayAccess" in expression.__class__.__name__
        ):
            return self.glsl_interface_block_value_root_name(
                getattr(expression, "array", getattr(expression, "array_expr", None))
            )
        return None

    def glsl_ordered_boolean_minmax_call(
        self,
        func_name,
        args,
        *,
        source_location=None,
    ):
        if func_name in self.function_return_types:
            return None
        argument_types = [
            self.map_type(self.expression_result_type(argument)) for argument in args
        ]
        width = ordered_boolean_minmax_width(func_name, argument_types)
        if width is None:
            if func_name in {"min", "max"} and any(
                is_boolean_type(type_name) for type_name in argument_types
            ):
                reason = (
                    "invalid-arity"
                    if len(args) != 2
                    else (
                        "operand-type-unresolved"
                        if any(not type_name for type_name in argument_types)
                        else "operand-type-mismatch"
                    )
                )
                raise self.GLSL_BOOLEAN_ORDERED_INTRINSIC_ERROR(
                    f"{self.GLSL_TARGET_DISPLAY_NAME} codegen cannot preserve "
                    "ordered boolean "
                    f"'{func_name}' for operand types {argument_types}",
                    operation=func_name,
                    operand_types=argument_types,
                    reason=reason,
                    source_location=source_location,
                )
            return None

        left, right = (self.generate_expression(argument) for argument in args)
        if width == 1:
            operator = "&&" if func_name == "min" else "||"
            return f"(({left}) {operator} ({right}))"

        helper_name = self.glsl_generated_module_identifier(
            ("ordered-boolean", func_name, width),
            f"crossgl_bool_{func_name}_bvec{width}",
        )
        self.required_glsl_boolean_order_helpers.add((helper_name, func_name, width))
        return f"{helper_name}({left}, {right})"

    def glsl_trailing_zero_builtin_is_shadowed(self, func_name):
        return (
            func_name in self.function_return_types
            or func_name in self.local_variable_types
            or func_name in self.global_variable_types
        )

    def glsl_trailing_zero_result_type(self, func_name, args):
        if (
            func_name not in self.GLSL_CLANG_TRAILING_ZERO_BUILTINS
            or self.glsl_trailing_zero_builtin_is_shadowed(func_name)
        ):
            return None
        return "int"

    def opengl_trailing_zero_builtin_error(
        self,
        func_name,
        *,
        reason,
        operand_type=None,
        argument_count=None,
        source_location=None,
    ):
        target_profile = getattr(self, "current_glsl_version_line", None)
        if self.GLSL_TARGET_DISPLAY_NAME != "OpenGL":
            target_profile = self.GLSL_TARGET_DISPLAY_NAME

        if reason == "invalid-arity":
            message = (
                f"OpenGL trailing-zero builtin '{func_name}' requires exactly "
                f"1 scalar integer operand, got {argument_count}"
            )
        elif reason == "unresolved-operand-type":
            message = (
                "OpenGL translation could not resolve source type metadata for "
                f"the operand of trailing-zero builtin '{func_name}'"
            )
        elif reason == "unsupported-operand-shape":
            message = (
                f"OpenGL trailing-zero builtin '{func_name}' requires a scalar "
                f"integer operand; got '{operand_type}'"
            )
        elif reason == "unsupported-integer-width":
            message = (
                f"OpenGL trailing-zero builtin '{func_name}' supports only 32-bit "
                f"or 64-bit integer operands; got '{operand_type}'"
            )
        elif reason == "operand-narrowing-required":
            message = (
                f"OpenGL trailing-zero builtin '{func_name}' has a 32-bit operand "
                f"contract and cannot consume '{operand_type}' without narrowing"
            )
        elif reason == "unsupported-profile":
            message = (
                f"OpenGL trailing-zero builtin '{func_name}' is unsupported for "
                f"target profile '{target_profile or '<unknown>'}'"
            )
        else:
            message = (
                f"OpenGL trailing-zero builtin '{func_name}' requires a 32-bit "
                f"or 64-bit integer operand; got '{operand_type}'"
            )
        return OpenGLTrailingZeroBuiltinError(
            message,
            builtin_name=func_name,
            operand_type=operand_type,
            argument_count=argument_count,
            target_profile=target_profile,
            reason=reason,
            source_location=source_location,
        )

    def validate_glsl_trailing_zero_profile(
        self, func_name, required_width, *, source_location=None
    ):
        version_line = getattr(self, "current_glsl_version_line", None)
        version_number = self.glsl_version_number(version_line)
        unsupported = (
            self.GLSL_TARGET_DISPLAY_NAME != "OpenGL"
            or self.glsl_version_is_es(version_line)
            or version_number is None
            or version_number < 400
            or (
                required_width == 64
                and self.GLSL_INT64_EXTENSION in getattr(
                    self, "current_glsl_disabled_extensions", set()
                )
            )
        )
        if unsupported:
            raise self.opengl_trailing_zero_builtin_error(
                func_name,
                reason="unsupported-profile",
                source_location=source_location,
            )

    def glsl_trailing_zero_operand_type_info(self, operand_type):
        source_type = self.type_name_string(operand_type)
        if not source_type:
            return None
        info = self.glsl_value_type_info(source_type)
        if (
            info is None
            or info["width"] != 1
            or info["family"] not in {"int", "uint"}
            or info["narrow"] is not None
            or info["bits"] not in {32, 64}
        ):
            return None
        return info

    def generate_glsl_trailing_zero_call(
        self, func_name, args, *, source_location=None
    ):
        if (
            func_name not in self.GLSL_CLANG_TRAILING_ZERO_BUILTINS
            or self.glsl_trailing_zero_builtin_is_shadowed(func_name)
        ):
            return None
        if len(args) != 1:
            raise self.opengl_trailing_zero_builtin_error(
                func_name,
                reason="invalid-arity",
                argument_count=len(args),
                source_location=source_location,
            )

        required_width = self.GLSL_CLANG_TRAILING_ZERO_WIDTHS[func_name]
        self.validate_glsl_trailing_zero_profile(
            func_name,
            required_width,
            source_location=source_location,
        )

        argument = args[0]
        operand_type = self.glsl_source_expression_type(argument)
        if operand_type is None:
            raise self.opengl_trailing_zero_builtin_error(
                func_name,
                reason="unresolved-operand-type",
                source_location=source_location,
            )

        source_type = self.type_name_string(operand_type)
        mapped_type = self.map_type(source_type)
        type_info = self.glsl_trailing_zero_operand_type_info(source_type)
        if type_info is None:
            value_info = self.glsl_value_type_info(source_type)
            if (value_info is not None and value_info["width"] != 1) or (
                self.is_vector_value_type(mapped_type)
                or self.is_matrix_value_type(mapped_type)
            ):
                reason = "unsupported-operand-shape"
            elif source_type in self.GLSL_NARROW_INTEGER_TYPE_NAMES or (
                value_info is not None and value_info["narrow"] is not None
            ):
                reason = "unsupported-integer-width"
            else:
                reason = "unsupported-operand-type"
            raise self.opengl_trailing_zero_builtin_error(
                func_name,
                reason=reason,
                operand_type=source_type,
                source_location=source_location,
            )

        if type_info["bits"] > required_width:
            raise self.opengl_trailing_zero_builtin_error(
                func_name,
                reason="operand-narrowing-required",
                operand_type=source_type,
                source_location=source_location,
            )

        unsigned_type = "uint64_t" if required_width == 64 else "uint"
        helper_name = self.glsl_trailing_zero_helper_name(unsigned_type)
        self.required_glsl_trailing_zero_helpers.add(unsigned_type)
        argument_code = self.generate_expression_with_expected(argument, None)
        if type_info["mapped"] != unsigned_type:
            argument_code = f"{unsigned_type}({argument_code})"
        return f"{helper_name}({argument_code})"

    def glsl_cooperative_matrix_error(
        self,
        message,
        *,
        operation,
        matrix_type,
        reason,
        source_location=None,
    ):
        fragment_contract = _opengl_cooperative_matrix_fragment_contract(matrix_type)
        raise OpenGLCooperativeMatrixError(
            message,
            operation=operation,
            matrix_type=matrix_type,
            **fragment_contract,
            reason=reason,
            source_location=(
                source_location
                if source_location is not None
                else getattr(matrix_type, "source_location", None)
            ),
        )

    @staticmethod
    def glsl_cooperative_matrix_contract_label(value):
        normalized = _opengl_cooperative_matrix_detail_value(value)
        if normalized is None:
            return None
        return str(normalized).strip()

    def glsl_cooperative_matrix_fragment_contract(
        self,
        matrix_type,
        *,
        operation="type",
        source_location=None,
    ):
        matrix_type = _opengl_cooperative_matrix_contract_type(matrix_type)
        if not isinstance(matrix_type, CooperativeMatrixType):
            self.glsl_cooperative_matrix_error(
                "OpenGL cooperative-matrix software lowering requires a canonical "
                "CooperativeMatrix type contract",
                operation=operation,
                matrix_type=matrix_type,
                reason="invalid-fragment-contract",
                source_location=source_location,
            )

        fragment_contract = _opengl_cooperative_matrix_fragment_contract(matrix_type)
        required_fields = {
            "scope": matrix_type.scope,
            "fragment_layout": fragment_contract["fragment_layout"],
            "subgroup_size": fragment_contract["subgroup_size"],
            "elements_per_lane": fragment_contract["elements_per_lane"],
            "fragment_provenance": fragment_contract["fragment_provenance"],
            "fragment_mapping": fragment_contract["fragment_mapping"],
            "fragment_mapping_provenance": fragment_contract[
                "fragment_mapping_provenance"
            ],
        }
        missing_fields = sorted(
            name
            for name, value in required_fields.items()
            if (
                self.glsl_cooperative_matrix_contract_label(value) is None
                or self.glsl_cooperative_matrix_contract_label(value).lower()
                in {"", "unspecified"}
            )
        )
        if missing_fields:
            self.glsl_cooperative_matrix_error(
                "OpenGL cooperative-matrix software lowering requires a complete "
                "lane-fragment contract; missing " + ", ".join(missing_fields),
                operation=operation,
                matrix_type=matrix_type,
                reason="missing-fragment-contract",
                source_location=source_location,
            )

        integer_fields = {
            "rows": matrix_type.rows,
            "columns": matrix_type.cols,
            "subgroup_size": matrix_type.subgroup_size,
            "elements_per_lane": matrix_type.elements_per_lane,
        }
        symbolic_fields = sorted(
            name for name, value in integer_fields.items() if type(value) is not int
        )
        label_fields = {
            "fragment_layout": matrix_type.fragment_layout,
            "fragment_provenance": matrix_type.fragment_provenance,
            "fragment_mapping": matrix_type.fragment_mapping,
            "fragment_mapping_provenance": matrix_type.fragment_mapping_provenance,
        }
        symbolic_fields.extend(
            sorted(
                name
                for name, value in label_fields.items()
                if not isinstance(value, str)
            )
        )
        if symbolic_fields:
            self.glsl_cooperative_matrix_error(
                "OpenGL cooperative-matrix software lowering requires concrete "
                "lane-fragment values; symbolic " + ", ".join(symbolic_fields),
                operation=operation,
                matrix_type=matrix_type,
                reason="symbolic-fragment-contract",
                source_location=source_location,
            )

        scope = self.glsl_cooperative_matrix_contract_label(matrix_type.scope)
        if not isinstance(matrix_type.scope, str):
            self.glsl_cooperative_matrix_error(
                "OpenGL cooperative-matrix software lowering requires a concrete "
                "execution scope",
                operation=operation,
                matrix_type=matrix_type,
                reason="symbolic-fragment-contract",
                source_location=source_location,
            )
        scope = scope.lower().replace("-", "_")
        if scope != "subgroup":
            self.glsl_cooperative_matrix_error(
                "OpenGL cooperative-matrix software lowering requires subgroup "
                f"scope for lane-local fragment ownership, got '{scope}'",
                operation=operation,
                matrix_type=matrix_type,
                reason="incompatible-fragment-contract",
                source_location=source_location,
            )

        mapping = get_cooperative_matrix_fragment_mapping(
            matrix_type.fragment_mapping,
            matrix_type.rows,
            matrix_type.cols,
            matrix_type.subgroup_size,
            matrix_type.elements_per_lane,
        )
        if mapping is None:
            mapping_name = self.glsl_cooperative_matrix_contract_label(
                matrix_type.fragment_mapping
            )
            registered = has_cooperative_matrix_fragment_mapping(mapping_name)
            reason = (
                "incompatible-fragment-mapping"
                if registered
                else "unregistered-fragment-mapping"
            )
            qualifier = "incompatible" if registered else "unregistered"
            self.glsl_cooperative_matrix_error(
                "OpenGL cooperative-matrix software lowering cannot use "
                f"{qualifier} fragment mapping '{mapping_name}' for "
                f"{matrix_type.rows}x{matrix_type.cols}, subgroup_size="
                f"{matrix_type.subgroup_size}, elements_per_lane="
                f"{matrix_type.elements_per_lane}",
                operation=operation,
                matrix_type=matrix_type,
                reason=reason,
                source_location=source_location,
            )

        component_source_type = str(
            self.type_name_string(matrix_type.element_type) or ""
        ).strip()
        mapped_component_type = self.map_type(matrix_type.element_type)
        expected_component_type = self.GLSL_COOPERATIVE_MATRIX_COMPONENT_TYPES.get(
            component_source_type
        )
        if (
            expected_component_type is None
            or mapped_component_type != expected_component_type
        ):
            self.glsl_cooperative_matrix_error(
                "OpenGL cooperative-matrix software lowering cannot represent "
                f"component type '{component_source_type}' as a lane-local scalar",
                operation=operation,
                matrix_type=matrix_type,
                reason="unsupported-component-type",
                source_location=source_location,
            )

        use = self.glsl_cooperative_matrix_contract_label(matrix_type.use)
        layout = self.glsl_cooperative_matrix_contract_label(matrix_type.layout)
        fragment_layout = self.glsl_cooperative_matrix_contract_label(
            matrix_type.fragment_layout
        )
        fragment_provenance = self.glsl_cooperative_matrix_contract_label(
            matrix_type.fragment_provenance
        )
        fragment_mapping_provenance = self.glsl_cooperative_matrix_contract_label(
            matrix_type.fragment_mapping_provenance
        )
        contract_key = (
            component_source_type,
            mapped_component_type,
            matrix_type.rows,
            matrix_type.cols,
            scope,
            use,
            layout,
            fragment_layout,
            matrix_type.subgroup_size,
            matrix_type.elements_per_lane,
            fragment_provenance,
            mapping.name,
            fragment_mapping_provenance,
        )
        existing = self.glsl_cooperative_matrix_fragment_contracts_by_key.get(
            contract_key
        )
        if existing is not None:
            return existing

        base_name = (
            "crossgl_cooperative_matrix_fragment_"
            f"{mapped_component_type}_{matrix_type.rows}x{matrix_type.cols}_"
            f"{matrix_type.subgroup_size}x{matrix_type.elements_per_lane}_"
            f"{mapping.name}"
        )
        type_name = self.glsl_generated_module_identifier(
            ("cooperative-matrix-fragment-type", contract_key),
            base_name,
        )
        contract = {
            "type_name": type_name,
            "contract_key": contract_key,
            "matrix_type": matrix_type,
            "component_type": component_source_type,
            "mapped_component_type": mapped_component_type,
            "rows": matrix_type.rows,
            "columns": matrix_type.cols,
            "scope": scope,
            "use": use,
            "layout": layout,
            "fragment_layout": fragment_layout,
            "subgroup_size": matrix_type.subgroup_size,
            "elements_per_lane": matrix_type.elements_per_lane,
            "fragment_provenance": fragment_provenance,
            "fragment_mapping": mapping.name,
            "fragment_mapping_provenance": fragment_mapping_provenance,
            "fragment_mapping_contract": mapping,
            "helpers": {},
        }
        self.glsl_cooperative_matrix_fragment_contracts[type_name] = contract
        self.glsl_cooperative_matrix_fragment_contracts_by_key[contract_key] = contract
        return contract

    def glsl_cooperative_matrix_contract_for_type(
        self,
        type_value,
        *,
        operation,
        source_location=None,
    ):
        if type_value is None:
            return None
        if isinstance(type_value, str):
            existing = self.glsl_cooperative_matrix_fragment_contracts.get(type_value)
            if existing is not None:
                return existing
        matrix_type = _opengl_cooperative_matrix_contract_type(type_value)
        if not isinstance(matrix_type, CooperativeMatrixType):
            return None
        return self.glsl_cooperative_matrix_fragment_contract(
            matrix_type,
            operation=operation,
            source_location=source_location,
        )

    def glsl_cooperative_matrix_expression_contract(self, expression, *, operation):
        source_location = getattr(expression, "source_location", None)
        candidates = [
            getattr(expression, "result_type", None),
            getattr(expression, "expression_type", None),
            self.glsl_source_expression_type(expression),
        ]
        for candidate in candidates:
            contract = self.glsl_cooperative_matrix_contract_for_type(
                candidate,
                operation=operation,
                source_location=source_location,
            )
            if contract is not None:
                return contract
        if isinstance(expression, CooperativeMatrixOpNode) and expression.arguments:
            if expression.operation != "element":
                return self.glsl_cooperative_matrix_expression_contract(
                    expression.arguments[0], operation=operation
                )
        return None

    def glsl_cooperative_matrix_operation_error(
        self,
        node,
        message,
        *,
        reason,
        matrix_type=None,
    ):
        if matrix_type is None:
            matrix_type = getattr(node, "result_type", None)
        if matrix_type is None and getattr(node, "arguments", None):
            operand_contract = self.glsl_cooperative_matrix_expression_contract(
                node.arguments[0], operation=node.operation
            )
            if operand_contract is not None:
                matrix_type = operand_contract["matrix_type"]
        self.glsl_cooperative_matrix_error(
            message,
            operation=node.operation,
            matrix_type=matrix_type,
            reason=reason,
            source_location=getattr(node, "source_location", None),
        )

    def glsl_cooperative_matrix_result_contract(self, node, operand_contracts):
        result_candidates = (
            ("result", getattr(node, "result_type", None)),
            ("expression", getattr(node, "expression_type", None)),
            ("contextual", self.current_expression_expected_type),
        )
        result_contract = None
        for source, candidate in result_candidates:
            if candidate is None:
                continue
            contract = self.glsl_cooperative_matrix_contract_for_type(
                candidate,
                operation=node.operation,
                source_location=getattr(node, "source_location", None),
            )
            if contract is None:
                candidate_name = self.type_name_string(candidate)
                if source == "contextual" and candidate_name in {"auto", "var"}:
                    continue
                self.glsl_cooperative_matrix_operation_error(
                    node,
                    "OpenGL cooperative-matrix lane-local operation "
                    f"'{node.operation}' received a non-cooperative {source} "
                    f"result type '{candidate_name}'",
                    reason="incompatible-result-contract",
                    matrix_type=(
                        operand_contracts[0]["matrix_type"]
                        if operand_contracts
                        else None
                    ),
                )
            if result_contract is None:
                result_contract = contract
            elif contract["contract_key"] != result_contract["contract_key"]:
                self.glsl_cooperative_matrix_operation_error(
                    node,
                    "OpenGL cooperative-matrix lane-local operation "
                    f"'{node.operation}' has incompatible declared result "
                    "contracts",
                    reason="incompatible-result-contract",
                    matrix_type=contract["matrix_type"],
                )
        return result_contract or (operand_contracts[0] if operand_contracts else None)

    def glsl_cooperative_matrix_helper_name(self, contract, operation):
        helper = contract["helpers"].get(operation)
        if helper is not None:
            return helper
        helper = self.glsl_generated_module_identifier(
            ("cooperative-matrix-fragment-helper", contract["contract_key"], operation),
            f"{contract['type_name']}_{operation}",
        )
        contract["helpers"][operation] = helper
        self.required_glsl_cooperative_matrix_helpers.add(
            (contract["type_name"], operation)
        )
        return helper

    def glsl_cooperative_matrix_multiply_accumulate_helper_name(
        self,
        result_contract,
        left_contract,
        right_contract,
        accumulator_contract,
    ):
        helper_key = (
            "multiply_accumulate",
            left_contract["contract_key"],
            right_contract["contract_key"],
            accumulator_contract["contract_key"],
        )
        helper = result_contract["helpers"].get(helper_key)
        if helper is not None:
            return helper
        helper = self.glsl_generated_module_identifier(
            (
                "cooperative-matrix-fragment-helper",
                result_contract["contract_key"],
                helper_key,
            ),
            f"{result_contract['type_name']}_multiply_accumulate",
        )
        result_contract["helpers"][helper_key] = helper
        self.required_glsl_cooperative_matrix_helpers.add(
            (
                result_contract["type_name"],
                "multiply_accumulate",
                left_contract["type_name"],
                right_contract["type_name"],
                accumulator_contract["type_name"],
            )
        )
        return helper

    def glsl_cooperative_matrix_multiply_accumulate_contracts(self, node):
        argument_count = len(node.arguments)
        # A plain product is a multiply-accumulate into a +0.0 accumulator:
        # Metal's simdgroup_matrix operator* matches
        # simdgroup_multiply_accumulate with a zero-filled matrix.
        multiply_only = node.operation == "multiply"
        if multiply_only and argument_count != 2:
            self.glsl_cooperative_matrix_operation_error(
                node,
                "OpenGL cooperative-matrix operation 'multiply' expects 2 "
                f"arguments, got {argument_count}",
                reason="invalid-argument-count",
            )
        if not multiply_only and argument_count not in {3, 4}:
            self.glsl_cooperative_matrix_operation_error(
                node,
                "OpenGL cooperative-matrix operation 'multiply_accumulate' "
                f"expects 3 or 4 arguments, got {argument_count}",
                reason="invalid-argument-count",
            )

        destination = node.arguments[0] if argument_count == 4 else None
        operand_offset = 1 if destination is not None else 0
        if multiply_only:
            left, right = node.arguments
            accumulator = None
            expressions = [left, right]
        else:
            left, right, accumulator = node.arguments[operand_offset:]
            expressions = [left, right, accumulator]
        if destination is not None:
            expressions.insert(0, destination)

        contracts = []
        for expression in expressions:
            contract = self.glsl_cooperative_matrix_expression_contract(
                expression, operation=node.operation
            )
            if contract is None:
                self.glsl_cooperative_matrix_operation_error(
                    node,
                    "OpenGL cooperative-matrix multiply-accumulate requires "
                    "complete destination and operand contracts",
                    reason="missing-operand-contract",
                )
            contracts.append(contract)

        if multiply_only:
            left_contract, right_contract = contracts
            accumulator_contract = None
            destination_contract = None
        elif destination is None:
            left_contract, right_contract, accumulator_contract = contracts
            destination_contract = None
        else:
            (
                destination_contract,
                left_contract,
                right_contract,
                accumulator_contract,
            ) = contracts

        result_contract = self.glsl_cooperative_matrix_result_contract(
            node, [left_contract if multiply_only else accumulator_contract]
        )
        if result_contract is None:
            self.glsl_cooperative_matrix_operation_error(
                node,
                "OpenGL cooperative-matrix multiply-accumulate requires a "
                "complete result contract",
                reason="missing-result-contract",
            )
        if multiply_only:
            accumulator_contract = result_contract

        result_compatible = [accumulator_contract]
        if destination_contract is not None:
            result_compatible.append(destination_contract)
        if any(
            contract["contract_key"] != result_contract["contract_key"]
            for contract in result_compatible
        ):
            self.glsl_cooperative_matrix_operation_error(
                node,
                "OpenGL cooperative-matrix multiply-accumulate requires "
                "identical destination, accumulator, and result contracts",
                reason="incompatible-result-contract",
                matrix_type=result_contract["matrix_type"],
            )

        if destination is not None and not self.glsl_reference_argument_is_lvalue(
            destination
        ):
            self.glsl_cooperative_matrix_operation_error(
                node,
                "OpenGL cooperative-matrix multiply-accumulate destination "
                "must be assignable",
                reason="destination-requires-lvalue",
                matrix_type=result_contract["matrix_type"],
            )

        if not (
            left_contract["rows"] == result_contract["rows"]
            and left_contract["columns"] == right_contract["rows"]
            and right_contract["columns"] == result_contract["columns"]
        ):
            self.glsl_cooperative_matrix_operation_error(
                node,
                "OpenGL cooperative-matrix multiply-accumulate received "
                "incompatible matrix dimensions",
                reason="incompatible-multiply-accumulate-contract",
                matrix_type=result_contract["matrix_type"],
            )

        operation_contracts = (
            result_contract,
            left_contract,
            right_contract,
            accumulator_contract,
        )
        exact_mapping = all(
            contract["rows"] == 8
            and contract["columns"] == 8
            and contract["subgroup_size"] == 32
            and contract["elements_per_lane"] == 2
            and contract["fragment_mapping"] == "tile_4x4_row_pair"
            and contract["mapped_component_type"] == "float"
            for contract in operation_contracts
        )
        if not exact_mapping:
            self.glsl_cooperative_matrix_operation_error(
                node,
                "OpenGL cooperative-matrix multiply-accumulate only supports "
                "the float tile_4x4_row_pair 8x8/32x2 fragment contract",
                reason="incompatible-multiply-accumulate-contract",
                matrix_type=result_contract["matrix_type"],
            )

        if self.software_subgroup_width == 32:
            self.validate_glsl_software_subgroup_function(
                "CooperativeMatrixMultiplyAccumulate",
                source_location=getattr(node, "source_location", None),
            )
        elif self.current_glsl_exact_subgroup_width != 32:
            actual_width = self.current_glsl_exact_subgroup_width
            actual = "unspecified" if actual_width is None else str(actual_width)
            self.glsl_cooperative_matrix_operation_error(
                node,
                "OpenGL cooperative-matrix multiply-accumulate requires an "
                f"exact 32-lane subgroup entry contract, got {actual}",
                reason="exact-subgroup-width-required",
                matrix_type=result_contract["matrix_type"],
            )

        return {
            "destination": destination,
            "left": left,
            "right": right,
            "accumulator": accumulator,
            "result_contract": result_contract,
            "left_contract": left_contract,
            "right_contract": right_contract,
            "accumulator_contract": accumulator_contract,
        }

    def generate_glsl_cooperative_matrix_multiply_accumulate(self, node):
        operation = self.glsl_cooperative_matrix_multiply_accumulate_contracts(node)
        if self.cooperative_matrix_mma_profile is not None:
            for name in ("floatBitsToUint", "uintBitsToFloat"):
                if (
                    name in self.function_return_types
                    or name in self.global_variable_types
                    or name in self.structs_by_name
                ):
                    self.glsl_cooperative_matrix_operation_error(
                        node,
                        f"OpenGL matrix rounding requires the unshadowed '{name}' builtin",
                        reason="rounding-builtin-shadowed",
                    )
        helper_name = self.glsl_cooperative_matrix_multiply_accumulate_helper_name(
            operation["result_contract"],
            operation["left_contract"],
            operation["right_contract"],
            operation["accumulator_contract"],
        )
        operands = [
            self.generate_expression(operation[name]) for name in ("left", "right")
        ]
        if operation["accumulator"] is None:
            result_contract = operation["result_contract"]
            component = result_contract["mapped_component_type"]
            count = result_contract["elements_per_lane"]
            zeros = ", ".join(f"{component}(0)" for _ in range(count))
            operands.append(
                f"{result_contract['type_name']}({component}[{count}]({zeros}))"
            )
        else:
            operands.append(self.generate_expression(operation["accumulator"]))
        call = f"{helper_name}({', '.join(operands)})"
        destination = operation["destination"]
        if destination is None:
            return call
        return f"({self.generate_expression(destination)} = {call})"

    def generate_glsl_cooperative_matrix_operation(self, node):
        operation = node.operation
        if operation in {"multiply_accumulate", "multiply"}:
            return self.generate_glsl_cooperative_matrix_multiply_accumulate(node)
        if operation == "fill":
            return self.generate_glsl_cooperative_matrix_fill(node)
        if operation == "diagonal":
            return generate_glsl_cooperative_matrix_diagonal(self, node)
        if operation in {"load", "store"}:
            return self.generate_glsl_cooperative_matrix_transfer(node)

        lane_local_arities = {
            "element": 2,
            "negate": 1,
            "elementwise_add": 2,
            "elementwise_subtract": 2,
            "elementwise_multiply": 2,
        }
        expected_arity = lane_local_arities.get(operation)
        if expected_arity is None:
            self.glsl_cooperative_matrix_operation_error(
                node,
                "OpenGL cooperative-matrix software lowering does not implement "
                f"non-lane-local operation '{operation}'",
                reason="non-lane-local-operation",
            )
        if len(node.arguments) != expected_arity:
            self.glsl_cooperative_matrix_operation_error(
                node,
                f"OpenGL cooperative-matrix operation '{operation}' expects "
                f"{expected_arity} arguments, got {len(node.arguments)}",
                reason="invalid-argument-count",
            )

        if operation == "element":
            matrix_contract = self.glsl_cooperative_matrix_expression_contract(
                node.arguments[0], operation=operation
            )
            if matrix_contract is None:
                self.glsl_cooperative_matrix_operation_error(
                    node,
                    "OpenGL cooperative-matrix element access requires an operand "
                    "with a complete lane-fragment contract",
                    reason="missing-operand-contract",
                )
            declared_result_type = getattr(node, "expression_type", None)
            if declared_result_type is not None:
                declared_matrix_type = _opengl_cooperative_matrix_contract_type(
                    declared_result_type
                )
                declared_component_type = (
                    None
                    if isinstance(declared_matrix_type, CooperativeMatrixType)
                    else self.map_type(declared_result_type)
                )
                if declared_component_type != matrix_contract["mapped_component_type"]:
                    self.glsl_cooperative_matrix_operation_error(
                        node,
                        "OpenGL cooperative-matrix element access result type "
                        "does not match the lane-fragment component type",
                        reason="incompatible-result-contract",
                        matrix_type=matrix_contract["matrix_type"],
                    )
            index_node = node.arguments[1]
            index_type = self.glsl_source_expression_type(index_node)
            mapped_index_type = self.map_type(index_type) if index_type else None
            if mapped_index_type not in {"int", "uint", "int64_t", "uint64_t"}:
                self.glsl_cooperative_matrix_operation_error(
                    node,
                    "OpenGL cooperative-matrix element access requires a scalar "
                    "integer lane-local index",
                    reason="invalid-element-index-type",
                    matrix_type=matrix_contract["matrix_type"],
                )
            literal_index = self.literal_int_value(
                index_node,
                getattr(self, "current_compile_time_int_constants", None),
            )
            if literal_index is not None and not (
                0 <= literal_index < matrix_contract["elements_per_lane"]
            ):
                self.glsl_cooperative_matrix_operation_error(
                    node,
                    "OpenGL cooperative-matrix lane-local element index is out "
                    f"of range: {literal_index}",
                    reason="element-index-out-of-range",
                    matrix_type=matrix_contract["matrix_type"],
                )
            matrix = self.generate_expression(node.arguments[0])
            index = self.generate_expression(index_node)
            return f"{matrix}.elements[{index}]"

        operand_contracts = []
        for argument in node.arguments:
            contract = self.glsl_cooperative_matrix_expression_contract(
                argument, operation=operation
            )
            if contract is None:
                self.glsl_cooperative_matrix_operation_error(
                    node,
                    "OpenGL cooperative-matrix lane-local operation "
                    f"'{operation}' requires complete operand contracts",
                    reason="missing-operand-contract",
                )
            operand_contracts.append(contract)

        result_contract = self.glsl_cooperative_matrix_result_contract(
            node, operand_contracts
        )
        if result_contract is None:
            self.glsl_cooperative_matrix_operation_error(
                node,
                "OpenGL cooperative-matrix lane-local operation "
                f"'{operation}' requires a complete result contract",
                reason="missing-result-contract",
            )
        incompatible = [
            contract
            for contract in operand_contracts
            if contract["contract_key"] != result_contract["contract_key"]
        ]
        if incompatible:
            self.glsl_cooperative_matrix_operation_error(
                node,
                "OpenGL cooperative-matrix lane-local operation "
                f"'{operation}' requires identical operand and result fragment "
                "contracts",
                reason="incompatible-fragment-contract",
                matrix_type=result_contract["matrix_type"],
            )

        helper_name = self.glsl_cooperative_matrix_helper_name(
            result_contract, operation
        )
        arguments = ", ".join(
            self.generate_expression(argument) for argument in node.arguments
        )
        return f"{helper_name}({arguments})"

    def generate_glsl_cooperative_matrix_fill(self, node):
        """Fill every lane-local element; the result type comes from context."""
        if len(node.arguments) != 1:
            self.glsl_cooperative_matrix_operation_error(
                node,
                "OpenGL cooperative-matrix fill expects 1 argument, got "
                f"{len(node.arguments)}",
                reason="invalid-argument-count",
            )
        contract = self.glsl_cooperative_matrix_result_contract(node, [])
        if contract is None:
            self.glsl_cooperative_matrix_operation_error(
                node,
                "OpenGL cooperative-matrix fill requires a declared matrix result",
                reason="missing-result-contract",
            )
        helper = self.glsl_cooperative_matrix_helper_name(contract, "fill")
        component = contract["mapped_component_type"]
        value = self.generate_expression_with_expected(node.arguments[0], component)
        return f"{helper}({component}({value}))"

    @staticmethod
    def glsl_cooperative_matrix_transfer_coordinate(lane, element):
        """The closed form of ``GLSL_COOPERATIVE_MATRIX_TRANSFER_MAPPINGS``."""
        row = ((lane >> 2) & 4) + ((lane >> 1) & 3)
        column = ((lane >> 2) & 2) * 2 + (lane & 1) * 2 + element
        return row, column

    def glsl_cooperative_matrix_pure_expression(self, expression):
        """Whether re-evaluating ``expression`` per element is unobservable."""
        if isinstance(expression, (IdentifierNode, LiteralNode)):
            return True
        if isinstance(expression, ArrayAccessNode):
            return self.glsl_cooperative_matrix_pure_expression(
                expression.array
            ) and self.glsl_cooperative_matrix_pure_expression(expression.index)
        if isinstance(expression, MemberAccessNode):
            return self.glsl_cooperative_matrix_pure_expression(expression.object_expr)
        if isinstance(expression, BinaryOpNode):
            if self.map_operator(expression.operator) in {"=", ","}:
                return False
            return self.glsl_cooperative_matrix_pure_expression(
                expression.left
            ) and self.glsl_cooperative_matrix_pure_expression(expression.right)
        if isinstance(expression, UnaryOpNode):
            if self.map_operator(expression.operator) not in {"+", "-", "!", "~"}:
                return False
            return self.glsl_cooperative_matrix_pure_expression(expression.operand)
        return False

    @staticmethod
    def glsl_cooperative_matrix_literal_bool(value):
        if value is None:
            return False
        if isinstance(value, LiteralNode) and isinstance(value.value, bool):
            return value.value
        if isinstance(value, LiteralNode) and value.value in {"true", "false"}:
            return value.value == "true"
        if isinstance(value, IdentifierNode) and value.name in {"true", "false"}:
            return value.name == "true"
        return None

    @staticmethod
    def glsl_cooperative_matrix_zero_origin(origin):
        if origin is None:
            return True

        def zero(value):
            return (
                isinstance(value, LiteralNode)
                and not isinstance(value.value, bool)
                and str(value.value).rstrip("uUlL") in {"0", "0x0"}
            )

        if zero(origin):
            return True
        components = getattr(origin, "args", None)
        if components is None:
            components = getattr(origin, "arguments", None)
        return bool(components) and all(zero(item) for item in components)

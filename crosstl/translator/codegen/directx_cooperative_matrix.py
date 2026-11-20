"""Cooperative matrices and atomic fences in the HLSL generator: built-in lowering, fragment operations, MMA and fill helpers, synchronization, bfloat16 operands and trailing zero counts."""

from ..ast import (
    ArrayAccessNode,
    AssignmentNode,
    BinaryOpNode,
    CooperativeMatrixOpNode,
    CooperativeMatrixType,
    FunctionCallNode,
    IdentifierNode,
    LiteralNode,
    MemberAccessNode,
    UnaryOpNode,
    VariableNode,
)
from ..cooperative_matrix import get_cooperative_matrix_fragment_mapping
from ..refusals import DirectXRefusal
from ..validation import expression_debug_name
from .boolean_intrinsics import (
    is_boolean_type,
    ordered_boolean_minmax_width,
)
from .comma_expressions import (
    c_family_comma_expression,
    is_comma_expression,
)
from .cooperative_matrix_diagonal import generate_hlsl_cooperative_matrix_diagonal
from .directx_values import (
    DirectXAtomicFenceLoweringError,
    DirectXBooleanOrderedIntrinsicError,
    DirectXContextualConversionError,
    _directx_cooperative_matrix_contract_type,
)


class DirectXCooperativeMatrixMixin:
    """Cooperative matrix members of ``HLSLCodeGen``."""

    def directx_cooperative_matrix_fragment_lowering(
        self,
        matrix_type,
        *,
        operation="type",
        source_location=None,
    ):
        matrix_type = _directx_cooperative_matrix_contract_type(matrix_type)
        if not isinstance(matrix_type, CooperativeMatrixType):
            raise self.directx_cooperative_matrix_unsupported(
                "DirectX cooperative matrix software lowering requires a "
                "canonical CooperativeMatrix type contract",
                operation=operation,
                matrix_type=matrix_type,
                reason="invalid-fragment-contract",
                source_location=source_location,
            )

        concrete_dimensions = {
            "rows": matrix_type.rows,
            "columns": matrix_type.cols,
            "subgroup_size": matrix_type.subgroup_size,
            "elements_per_lane": matrix_type.elements_per_lane,
        }
        missing_dimensions = [
            name for name, value in concrete_dimensions.items() if value is None
        ]
        if missing_dimensions:
            raise self.directx_cooperative_matrix_unsupported(
                "DirectX cooperative matrix software lowering requires concrete "
                f"{', '.join(missing_dimensions)} values",
                operation=operation,
                matrix_type=matrix_type,
                reason="missing-fragment-contract",
                source_location=source_location,
            )
        symbolic_dimensions = [
            name
            for name, value in concrete_dimensions.items()
            if type(value) is not int
        ]
        if symbolic_dimensions:
            raise self.directx_cooperative_matrix_unsupported(
                "DirectX cooperative matrix software lowering does not resolve "
                "symbolic fragment dimensions: "
                f"{', '.join(symbolic_dimensions)}",
                operation=operation,
                matrix_type=matrix_type,
                reason="symbolic-fragment-contract",
                source_location=source_location,
            )

        fragment_labels = {
            "fragment_layout": matrix_type.fragment_layout,
            "fragment_provenance": matrix_type.fragment_provenance,
            "fragment_mapping": matrix_type.fragment_mapping,
            "fragment_mapping_provenance": matrix_type.fragment_mapping_provenance,
        }
        missing_labels = [
            name
            for name, value in fragment_labels.items()
            if value is None
            or (isinstance(value, str) and value.strip().lower() in {"", "unspecified"})
        ]
        if missing_labels:
            raise self.directx_cooperative_matrix_unsupported(
                "DirectX cooperative matrix software lowering requires complete "
                "fragment mapping and provenance fields: "
                f"{', '.join(missing_labels)}",
                operation=operation,
                matrix_type=matrix_type,
                reason="missing-fragment-contract",
                source_location=source_location,
            )
        symbolic_labels = [
            name
            for name, value in fragment_labels.items()
            if not isinstance(value, str) or not value.strip()
        ]
        if symbolic_labels:
            raise self.directx_cooperative_matrix_unsupported(
                "DirectX cooperative matrix software lowering requires concrete "
                "fragment mapping labels: "
                f"{', '.join(symbolic_labels)}",
                operation=operation,
                matrix_type=matrix_type,
                reason="symbolic-fragment-contract",
                source_location=source_location,
            )

        qualifier_labels = {
            "scope": matrix_type.scope,
            "use": matrix_type.use,
            "layout": matrix_type.layout,
        }
        missing_qualifiers = [
            name
            for name, value in qualifier_labels.items()
            if value is None or (isinstance(value, str) and not value.strip())
        ]
        if missing_qualifiers:
            raise self.directx_cooperative_matrix_unsupported(
                "DirectX cooperative matrix software lowering requires complete "
                "matrix qualifier fields: "
                f"{', '.join(missing_qualifiers)}",
                operation=operation,
                matrix_type=matrix_type,
                reason="missing-fragment-contract",
                source_location=source_location,
            )
        symbolic_qualifiers = [
            name
            for name, value in qualifier_labels.items()
            if not isinstance(value, str)
        ]
        if symbolic_qualifiers:
            raise self.directx_cooperative_matrix_unsupported(
                "DirectX cooperative matrix software lowering requires concrete "
                "matrix qualifier labels: "
                f"{', '.join(symbolic_qualifiers)}",
                operation=operation,
                matrix_type=matrix_type,
                reason="symbolic-fragment-contract",
                source_location=source_location,
            )

        scope = matrix_type.scope.strip().lower().replace("-", "_")
        if scope != "subgroup":
            raise self.directx_cooperative_matrix_unsupported(
                "DirectX cooperative matrix software lowering requires subgroup "
                f"scope, got '{matrix_type.scope}'",
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
            raise self.directx_cooperative_matrix_unsupported(
                "DirectX cooperative matrix software lowering has no registered "
                f"fragment mapping for '{matrix_type.fragment_mapping}' with "
                f"{matrix_type.rows}x{matrix_type.cols} elements across "
                f"{matrix_type.subgroup_size} lanes and "
                f"{matrix_type.elements_per_lane} elements per lane",
                operation=operation,
                matrix_type=matrix_type,
                reason="unregistered-fragment-mapping",
                source_location=source_location,
            )

        source_component_type = str(
            self.type_name_string(matrix_type.element_type) or ""
        ).strip()
        component_type = self.DIRECTX_COOPERATIVE_MATRIX_COMPONENT_TYPES.get(
            source_component_type
        )
        if (
            component_type is None
            or self.map_type(matrix_type.element_type) != component_type
        ):
            raise self.directx_cooperative_matrix_unsupported(
                "DirectX cooperative matrix software lowering cannot safely "
                f"represent component type '{source_component_type or 'unknown'}'",
                operation=operation,
                matrix_type=matrix_type,
                reason="unsupported-component-type",
                source_location=source_location,
            )

        elements_per_lane = matrix_type.elements_per_lane
        if not 1 <= elements_per_lane <= 4:
            raise self.directx_cooperative_matrix_unsupported(
                "DirectX cooperative matrix software lowering currently requires "
                "between one and four lane-local elements; got "
                f"{elements_per_lane}",
                operation=operation,
                matrix_type=matrix_type,
                reason="unsupported-lane-fragment-width",
                source_location=source_location,
            )
        hlsl_type = (
            component_type
            if elements_per_lane == 1
            else f"{component_type}{elements_per_lane}"
        )
        lowering = {
            "source_component_type": source_component_type,
            "component_type": component_type,
            "hlsl_type": hlsl_type,
            "rows": matrix_type.rows,
            "columns": matrix_type.cols,
            "scope": scope,
            "use": matrix_type.use.strip(),
            "layout": matrix_type.layout.strip(),
            "fragment_layout": matrix_type.fragment_layout.strip(),
            "subgroup_size": matrix_type.subgroup_size,
            "elements_per_lane": elements_per_lane,
            "fragment_provenance": matrix_type.fragment_provenance.strip(),
            "fragment_mapping": mapping.name,
            "fragment_mapping_contract": mapping,
            "fragment_mapping_provenance": (
                matrix_type.fragment_mapping_provenance.strip()
            ),
            "matrix_type": matrix_type,
        }
        signature = self.directx_cooperative_matrix_contract_signature(lowering)
        existing = self.directx_cooperative_matrix_lowerings.get(signature)
        if existing is not None:
            return existing
        self.directx_cooperative_matrix_lowerings[signature] = lowering
        return lowering

    def directx_cooperative_matrix_expression_type(self, expr):
        if isinstance(expr, CooperativeMatrixOpNode):
            if expr.operation == "element":
                return None
            return (
                getattr(expr, "result_type", None)
                or getattr(expr, "expression_type", None)
                or self.current_expression_expected_type
            )
        if isinstance(expr, (IdentifierNode, VariableNode)):
            name = getattr(expr, "name", None)
            return self.local_variable_source_types.get(name) or (
                self.variable_type_by_name(name)
            )
        if isinstance(expr, str):
            return self.local_variable_source_types.get(expr) or (
                self.variable_type_by_name(expr)
            )
        return self.expression_result_type(expr)

    def directx_cooperative_matrix_operation_error(
        self,
        expr,
        message,
        *,
        matrix_type=None,
        reason,
    ):
        return self.directx_cooperative_matrix_unsupported(
            message,
            operation=expr.operation,
            matrix_type=matrix_type,
            reason=reason,
            source_location=getattr(expr, "source_location", None),
        )

    def generate_hlsl_cooperative_matrix_mma(self, expr):
        arguments = list(expr.arguments)
        # A plain product is a multiply-accumulate into a +0.0 accumulator:
        # Metal's simdgroup_matrix operator* matches
        # simdgroup_multiply_accumulate with a zero-filled matrix.
        multiply_only = expr.operation == "multiply"
        if multiply_only and len(arguments) != 2:
            raise self.directx_cooperative_matrix_operation_error(
                expr,
                "DirectX matrix multiply requires two operands",
                reason="invalid-operation-arity",
            )
        if not multiply_only and len(arguments) not in {3, 4}:
            raise self.directx_cooperative_matrix_operation_error(
                expr,
                "DirectX matrix multiply-accumulate requires three operands and "
                "an optional destination",
                reason="invalid-operation-arity",
            )
        destination = arguments[0] if len(arguments) == 4 else None
        contracts = []
        for argument in arguments:
            matrix_type = _directx_cooperative_matrix_contract_type(
                self.directx_cooperative_matrix_expression_type(argument)
            )
            if not isinstance(matrix_type, CooperativeMatrixType):
                raise self.directx_cooperative_matrix_operation_error(
                    expr,
                    "DirectX matrix multiply-accumulate requires concrete operand contracts",
                    reason="missing-operand-contract",
                )
            contracts.append(
                self.directx_cooperative_matrix_fragment_lowering(
                    matrix_type,
                    operation=expr.operation,
                    source_location=getattr(expr, "source_location", None),
                )
            )
        accumulator = contracts[-1]
        result_contracts = contracts[:1] if destination is not None else []
        for source, candidate in (
            ("result", getattr(expr, "result_type", None)),
            ("expression", getattr(expr, "expression_type", None)),
            ("contextual", self.current_expression_expected_type),
        ):
            if candidate is None or (
                source == "contextual"
                and self.type_name_string(candidate) in {"auto", "var"}
            ):
                continue
            matrix_type = _directx_cooperative_matrix_contract_type(candidate)
            if not isinstance(matrix_type, CooperativeMatrixType):
                raise self.directx_cooperative_matrix_operation_error(
                    expr,
                    "DirectX matrix result must retain its cooperative contract",
                    reason="incompatible-result-contract",
                )
            result_contracts.append(
                self.directx_cooperative_matrix_fragment_lowering(
                    matrix_type,
                    operation=expr.operation,
                    source_location=getattr(expr, "source_location", None),
                )
            )
        if any(
            self.directx_cooperative_matrix_contract_differences(accumulator, contract)
            for contract in result_contracts
        ):
            raise self.directx_cooperative_matrix_operation_error(
                expr,
                "DirectX matrix destination and result must match the accumulator",
                matrix_type=accumulator["matrix_type"],
                reason="incompatible-result-contract",
            )
        if not all(
            contract["component_type"] == "float"
            and contract["rows"] == contract["columns"] == 8
            and contract["subgroup_size"] == 32
            and contract["elements_per_lane"] == 2
            and contract["fragment_mapping"] == "tile_4x4_row_pair"
            for contract in contracts + result_contracts
        ):
            raise self.directx_cooperative_matrix_operation_error(
                expr,
                "DirectX matrix multiply-accumulate requires the float "
                "tile_4x4_row_pair 8x8/32x2 fragment contract",
                matrix_type=accumulator["matrix_type"],
                reason="incompatible-multiply-accumulate-contract",
            )
        if (
            destination is not None
            and not self.hlsl_typed_buffer_atomic_original_is_lvalue(destination)
        ):
            raise self.directx_cooperative_matrix_operation_error(
                expr,
                "DirectX matrix destination must be assignable",
                reason="destination-requires-lvalue",
            )
        invocation = self.hlsl_software_subgroup_invocation_expressions.get(
            self.current_function_name
        )
        if invocation is None:
            raise self.directx_cooperative_matrix_operation_error(
                expr,
                "DirectX matrix multiply-accumulate requires validated uniform control flow",
                reason="software-subgroup-contract-required",
            )
        if self.cooperative_matrix_mma_profile is not None:
            for name in ("asuint", "asfloat"):
                if (
                    name in self.function_return_types
                    or name in self.global_variable_types
                    or name in self.structs_by_name
                ):
                    raise self.directx_cooperative_matrix_operation_error(
                        expr,
                        f"DirectX matrix rounding requires the unshadowed '{name}' builtin",
                        reason="rounding-builtin-shadowed",
                    )
        self.required_hlsl_cooperative_matrix_mma = True
        helper = self.hlsl_software_subgroup_identifier(
            ("matrix", "mma"), "__crossgl_software_matrix_mma"
        )
        operands = ", ".join(
            self.generate_expression(argument) for argument in arguments[-3:]
        )
        if multiply_only:
            operands += ", float2(0.0f, 0.0f)"
        call = f"{helper}({operands}, uint({invocation}))"
        if destination is not None:
            return f"({self.generate_expression(destination)} = {call})"
        return call

    def directx_cooperative_matrix_result_lowering(self, expr):
        """The single fragment contract a context-typed result must take."""
        lowerings = []
        for source, candidate in (
            ("result", getattr(expr, "result_type", None)),
            ("expression", getattr(expr, "expression_type", None)),
            ("contextual", self.current_expression_expected_type),
        ):
            if candidate is None or (
                source == "contextual"
                and self.type_name_string(candidate) in {"auto", "var"}
            ):
                continue
            matrix_type = _directx_cooperative_matrix_contract_type(candidate)
            if not isinstance(matrix_type, CooperativeMatrixType):
                raise self.directx_cooperative_matrix_operation_error(
                    expr,
                    "DirectX matrix result must retain its cooperative contract",
                    reason="incompatible-result-contract",
                )
            lowerings.append(
                self.directx_cooperative_matrix_fragment_lowering(
                    matrix_type,
                    operation=expr.operation,
                    source_location=getattr(expr, "source_location", None),
                )
            )
        if not lowerings:
            return None
        if any(
            self.directx_cooperative_matrix_contract_differences(lowerings[0], other)
            for other in lowerings[1:]
        ):
            raise self.directx_cooperative_matrix_operation_error(
                expr,
                "DirectX matrix result contracts disagree",
                matrix_type=lowerings[0]["matrix_type"],
                reason="incompatible-result-contract",
            )
        return lowerings[0]

    def generate_hlsl_cooperative_matrix_fill(self, expr):
        """Fill every lane-local element; the result type comes from context."""
        if len(expr.arguments) != 1:
            raise self.directx_cooperative_matrix_operation_error(
                expr,
                "DirectX cooperative matrix fill requires exactly one value",
                reason="invalid-operation-arity",
            )
        lowering = self.directx_cooperative_matrix_result_lowering(expr)
        if lowering is None:
            raise self.directx_cooperative_matrix_operation_error(
                expr,
                "DirectX cooperative matrix fill requires a declared matrix result",
                reason="missing-result-contract",
            )
        component = lowering["component_type"]
        value = self.generate_expression_with_expected(expr.arguments[0], component)
        # One evaluation, replicated by the scalar-to-vector cast.
        return f"(({lowering['hlsl_type']})(({component})({value})))"

    def directx_cooperative_matrix_pure_expression(self, expression):
        """Whether re-evaluating ``expression`` per element is unobservable."""
        if isinstance(expression, (IdentifierNode, LiteralNode)):
            return True
        if isinstance(expression, ArrayAccessNode):
            return self.directx_cooperative_matrix_pure_expression(
                expression.array
            ) and self.directx_cooperative_matrix_pure_expression(expression.index)
        if isinstance(expression, MemberAccessNode):
            return self.directx_cooperative_matrix_pure_expression(
                expression.object_expr
            )
        if isinstance(expression, BinaryOpNode):
            if self.map_operator(expression.operator) in {"=", ","}:
                return False
            return self.directx_cooperative_matrix_pure_expression(
                expression.left
            ) and self.directx_cooperative_matrix_pure_expression(expression.right)
        if isinstance(expression, UnaryOpNode):
            if self.map_operator(expression.operator) not in {"+", "-", "!", "~"}:
                return False
            return self.directx_cooperative_matrix_pure_expression(expression.operand)
        return False

    def generate_hlsl_cooperative_matrix_transfer(self, expr):
        """Lower simdgroup_load/simdgroup_store to lane-local buffer accesses.

        Lane ``l`` holds the elements the fragment mapping assigns it, so a
        transfer needs no cross-lane exchange: each element reads or writes
        ``pointer[row * stride + column]``, or ``pointer[column * stride +
        row]`` for a transposed matrix.
        """
        from .GLSL_codegen import GLSLCodeGen

        operation = expr.operation
        arguments = list(expr.arguments)
        if not 2 <= len(arguments) <= 5:
            raise self.directx_cooperative_matrix_operation_error(
                expr,
                f"DirectX cooperative matrix {operation} requires 2 to 5 arguments",
                reason="invalid-operation-arity",
            )
        if operation == "load":
            matrix, pointer = arguments[:2]
        else:
            pointer, matrix = arguments[:2]
        stride = arguments[2] if len(arguments) > 2 else None
        origin = arguments[3] if len(arguments) > 3 else None
        transpose = GLSLCodeGen.glsl_cooperative_matrix_literal_bool(
            arguments[4] if len(arguments) > 4 else None
        )
        matrix_type = _directx_cooperative_matrix_contract_type(
            self.directx_cooperative_matrix_expression_type(matrix)
        )
        if not isinstance(matrix_type, CooperativeMatrixType):
            raise self.directx_cooperative_matrix_operation_error(
                expr,
                f"DirectX cooperative matrix {operation} requires a matrix with a "
                "concrete cooperative-matrix contract",
                reason="missing-operand-contract",
            )
        lowering = self.directx_cooperative_matrix_fragment_lowering(
            matrix_type,
            operation=operation,
            source_location=getattr(expr, "source_location", None),
        )
        mapping = lowering["fragment_mapping_contract"]
        if (
            lowering["fragment_mapping"]
            not in GLSLCodeGen.GLSL_COOPERATIVE_MATRIX_TRANSFER_MAPPINGS
            or (lowering["rows"], lowering["columns"]) != (8, 8)
            or (lowering["subgroup_size"], lowering["elements_per_lane"]) != (32, 2)
            or any(
                mapping.coordinate(lane, element)
                != GLSLCodeGen.glsl_cooperative_matrix_transfer_coordinate(
                    lane, element
                )
                for lane in range(32)
                for element in range(2)
            )
        ):
            raise self.directx_cooperative_matrix_operation_error(
                expr,
                f"DirectX cooperative matrix {operation} only supports the "
                "tile_4x4_row_pair 8x8/32x2 fragment contract",
                matrix_type=matrix_type,
                reason="incompatible-transfer-contract",
            )
        if transpose is None:
            raise self.directx_cooperative_matrix_operation_error(
                expr,
                f"DirectX cooperative matrix {operation} requires a literal "
                "transpose flag",
                matrix_type=matrix_type,
                reason="dynamic-transpose",
            )
        if not GLSLCodeGen.glsl_cooperative_matrix_zero_origin(origin):
            raise self.directx_cooperative_matrix_operation_error(
                expr,
                f"DirectX cooperative matrix {operation} only supports a zero "
                "matrix origin",
                matrix_type=matrix_type,
                reason="nonzero-matrix-origin",
            )
        for label, value in (("matrix", matrix), ("pointer", pointer)):
            if not self.directx_cooperative_matrix_pure_expression(value):
                raise self.directx_cooperative_matrix_operation_error(
                    expr,
                    f"DirectX cooperative matrix {operation} {label} must be free "
                    "of side effects",
                    matrix_type=matrix_type,
                    reason="impure-transfer-operand",
                )
        if stride is not None and not self.directx_cooperative_matrix_pure_expression(
            stride
        ):
            raise self.directx_cooperative_matrix_operation_error(
                expr,
                f"DirectX cooperative matrix {operation} stride must be free of "
                "side effects",
                matrix_type=matrix_type,
                reason="impure-transfer-operand",
            )
        if (
            operation == "load"
            and not self.hlsl_typed_buffer_atomic_original_is_lvalue(matrix)
        ):
            raise self.directx_cooperative_matrix_operation_error(
                expr,
                "DirectX cooperative matrix load destination must be assignable",
                matrix_type=matrix_type,
                reason="destination-requires-lvalue",
            )
        binding = self.hlsl_resource_pointer_binding(pointer)
        if (
            binding is None
            or binding.get("kind") == "workgroup-pointer"
            or binding.get("pointer_reinterpretation") is not None
            or self.hlsl_half_storage.binding_type(binding) is not None
            or self.map_type(binding.get("element_type")) != lowering["component_type"]
        ):
            raise self.directx_cooperative_matrix_operation_error(
                expr,
                f"DirectX cooperative matrix {operation} requires a storage-buffer "
                "pointer with the matrix component type",
                matrix_type=matrix_type,
                reason="unsupported-transfer-pointer",
            )
        invocation = self.hlsl_software_subgroup_invocation_expressions.get(
            self.current_function_name
        )
        if invocation is None:
            raise self.directx_cooperative_matrix_operation_error(
                expr,
                f"DirectX cooperative matrix {operation} requires validated "
                "software subgroups",
                matrix_type=matrix_type,
                reason="software-subgroup-contract-required",
            )
        self.required_hlsl_cooperative_matrix_fragment_offset = True
        helper = self.hlsl_software_subgroup_identifier(
            ("matrix", "fragment_offset"), "__crossgl_software_matrix_fragment_offset"
        )
        stride_text = (
            f"uint({self.generate_expression(stride)})"
            if stride is not None
            else f"{lowering['columns']}u"
        )
        transpose_text = "true" if transpose else "false"
        matrix_text = self.generate_expression(matrix)
        if not isinstance(matrix, (IdentifierNode, VariableNode)):
            matrix_text = f"({matrix_text})"
        parts = []
        for element in range(lowering["elements_per_lane"]):
            index = (
                f"{helper}(uint({invocation}) % 32u, {element}u, {stride_text}, "
                f"{transpose_text})"
            )
            access = self.generate_hlsl_resource_pointer_access(
                pointer, index, require_write=operation == "store"
            )
            if operation == "load":
                parts.append(f"{matrix_text}[{element}] = {access}")
            else:
                parts.append(f"{access} = {matrix_text}[{element}]")
        return f"({', '.join(parts)})"

    def generate_hlsl_cooperative_matrix_fragment_offset_helper(self):
        from .GLSL_codegen import GLSLCodeGen

        helper = self.hlsl_software_subgroup_identifier(
            ("matrix", "fragment_offset"), "__crossgl_software_matrix_fragment_offset"
        )
        row, column = GLSLCodeGen.GLSL_COOPERATIVE_MATRIX_TRANSFER_MAPPINGS[
            "tile_4x4_row_pair"
        ]
        return (
            f"uint {helper}(uint lane, uint element, uint stride, bool transpose) {{\n"
            "    // The registered lane coordinates of this fragment element.\n"
            f"    uint row = {row};\n"
            f"    uint column = {column};\n"
            "    return transpose ? column * stride + row : row * stride + column;\n"
            "}\n\n"
        )

    def directx_cooperative_matrix_operation_lowering(self, expr):
        operation = expr.operation
        result_type = getattr(expr, "result_type", None)
        if (
            operation in {"multiply_accumulate", "multiply"}
            and self.software_subgroup_width == 32
        ):
            return self.generate_hlsl_cooperative_matrix_mma(expr)
        if operation == "fill":
            return self.generate_hlsl_cooperative_matrix_fill(expr)
        if operation == "diagonal":
            return generate_hlsl_cooperative_matrix_diagonal(self, expr)
        if operation in {"load", "store"} and self.software_subgroup_width == 32:
            return self.generate_hlsl_cooperative_matrix_transfer(expr)
        if operation in {"load", "store"}:
            raise self.directx_cooperative_matrix_operation_error(
                expr,
                f"DirectX cooperative matrix {operation} requires 32-lane software "
                "subgroups",
                matrix_type=result_type,
                reason="exact-subgroup-width-required",
            )

        if operation not in self.DIRECTX_COOPERATIVE_MATRIX_LANE_LOCAL_OPERATIONS:
            raise self.directx_cooperative_matrix_operation_error(
                expr,
                "DirectX cooperative matrix software lowering is unavailable for "
                f"non-lane-local operation '{operation}'",
                matrix_type=result_type,
                reason="unsupported-operation",
            )

        required_arity = {
            "element": 2,
            "negate": 1,
            "elementwise_add": 2,
            "elementwise_subtract": 2,
            "elementwise_multiply": 2,
        }[operation]
        if len(expr.arguments) != required_arity:
            raise self.directx_cooperative_matrix_operation_error(
                expr,
                f"DirectX cooperative matrix {operation} requires exactly "
                f"{required_arity} operands; got {len(expr.arguments)}",
                matrix_type=result_type,
                reason="invalid-operation-arity",
            )

        matrix_arguments = (
            expr.arguments[:1] if operation in {"element", "negate"} else expr.arguments
        )
        operand_lowerings = []
        for operand_index, argument in enumerate(matrix_arguments):
            operand_type = self.directx_cooperative_matrix_expression_type(argument)
            operand_type = _directx_cooperative_matrix_contract_type(operand_type)
            if not isinstance(operand_type, CooperativeMatrixType):
                raise self.directx_cooperative_matrix_operation_error(
                    expr,
                    "DirectX cooperative matrix software lowering requires a "
                    f"concrete cooperative-matrix contract for operand "
                    f"{operand_index}",
                    matrix_type=result_type,
                    reason="missing-operand-contract",
                )
            operand_lowerings.append(
                self.directx_cooperative_matrix_fragment_lowering(
                    operand_type,
                    operation=operation,
                    source_location=getattr(expr, "source_location", None),
                )
            )

        if operation == "element":
            expression_type = getattr(expr, "expression_type", None)
            if expression_type is not None:
                mapped_expression_type = self.map_type(expression_type)
                component_type = operand_lowerings[0]["component_type"]
                if mapped_expression_type != component_type:
                    raise self.directx_cooperative_matrix_operation_error(
                        expr,
                        "DirectX cooperative matrix element access result type "
                        f"'{mapped_expression_type}' does not match fragment "
                        f"component type '{component_type}'",
                        matrix_type=operand_lowerings[0]["matrix_type"],
                        reason="incompatible-result-contract",
                    )

        result_lowerings = []
        if operation != "element":
            result_candidates = (
                ("result", result_type),
                ("expression", getattr(expr, "expression_type", None)),
                ("contextual", self.current_expression_expected_type),
            )
            seen_candidates = set()
            for source, candidate in result_candidates:
                if candidate is None:
                    continue
                candidate_signature = self.type_name_string(candidate)
                if candidate_signature in seen_candidates:
                    continue
                seen_candidates.add(candidate_signature)
                matrix_candidate = _directx_cooperative_matrix_contract_type(candidate)
                if not isinstance(matrix_candidate, CooperativeMatrixType):
                    if source == "contextual" and candidate_signature in {
                        "auto",
                        "var",
                    }:
                        continue
                    raise self.directx_cooperative_matrix_operation_error(
                        expr,
                        "DirectX cooperative matrix lane-local operation "
                        f"'{operation}' has a non-cooperative {source} result "
                        f"type '{candidate_signature}'",
                        matrix_type=operand_lowerings[0]["matrix_type"],
                        reason="incompatible-result-contract",
                    )
                result_lowerings.append(
                    self.directx_cooperative_matrix_fragment_lowering(
                        matrix_candidate,
                        operation=operation,
                        source_location=getattr(expr, "source_location", None),
                    )
                )

        result_lowering = result_lowerings[0] if result_lowerings else None
        reference_lowering = result_lowering or operand_lowerings[0]
        compared_lowerings = [*result_lowerings[1:], *operand_lowerings]
        for compared_lowering in compared_lowerings:
            differences = self.directx_cooperative_matrix_contract_differences(
                reference_lowering, compared_lowering
            )
            if differences:
                raise self.directx_cooperative_matrix_operation_error(
                    expr,
                    "DirectX cooperative matrix software lowering requires "
                    "identical lane-fragment contracts for every operand and "
                    f"result; differing fields: {', '.join(differences)}",
                    matrix_type=compared_lowering["matrix_type"],
                    reason="incompatible-fragment-contract",
                )

        if operation == "element":
            index_expr = expr.arguments[1]
            index_type = self.expression_result_type(index_expr)
            if index_type is None or self.map_type(index_type) not in {"int", "uint"}:
                raise self.directx_cooperative_matrix_operation_error(
                    expr,
                    "DirectX cooperative matrix element access requires a scalar "
                    "32-bit integer lane-local index",
                    matrix_type=reference_lowering["matrix_type"],
                    reason="invalid-element-index-type",
                )
            literal_index = self.literal_int_value(
                index_expr, self.literal_int_constants
            )
            if literal_index is not None and not (
                0 <= literal_index < reference_lowering["elements_per_lane"]
            ):
                raise self.directx_cooperative_matrix_operation_error(
                    expr,
                    "DirectX cooperative matrix lane-local element index "
                    f"{literal_index} is outside [0, "
                    f"{reference_lowering['elements_per_lane']})",
                    matrix_type=reference_lowering["matrix_type"],
                    reason="element-index-out-of-range",
                )
            matrix = self.generate_expression(expr.arguments[0])
            if reference_lowering["elements_per_lane"] == 1:
                if literal_index is None:
                    raise self.directx_cooperative_matrix_operation_error(
                        expr,
                        "DirectX scalar lane fragments require a statically proven "
                        "element index of zero",
                        matrix_type=reference_lowering["matrix_type"],
                        reason="unsupported-dynamic-element-index",
                    )
                return matrix
            index = self.generate_expression(index_expr)
            if not isinstance(expr.arguments[0], (IdentifierNode, VariableNode, str)):
                matrix = f"({matrix})"
            return f"{matrix}[{index}]"

        rendered = [self.generate_expression(argument) for argument in expr.arguments]
        if operation == "negate":
            return f"(-{rendered[0]})"
        operator = {
            "elementwise_add": "+",
            "elementwise_subtract": "-",
            "elementwise_multiply": "*",
        }[operation]
        return f"({rendered[0]} {operator} {rendered[1]})"

    def generate_discarded_expression(self, expr):
        """Render an expression whose value is unused, such as a for update."""
        if isinstance(expr, AssignmentNode):
            return self.generate_assignment(expr)
        if is_comma_expression(expr):
            return c_family_comma_expression(
                expr,
                self.generate_expression,
                self.generate_discarded_expression,
                discarded=True,
            )
        return self.generate_expression(expr)

    def hlsl_helper_invocation_expression(self, expr):
        """Lower CrossGL's ``gl_HelperInvocation`` to Shader Model 6.6 ``IsHelperLane()``.

        HLSL has no helper-invocation variable; the query is the intrinsic. A local or
        global that happens to use the name is left alone. The dx10 and dx11 profiles
        (Shader Model 4 and 5) have no such intrinsic, so the read is refused there.
        """
        if not isinstance(expr, (IdentifierNode, VariableNode)):
            return None
        if getattr(expr, "name", None) != "gl_HelperInvocation":
            return None
        for table in ("local_variable_types", "global_variable_types"):
            if "gl_HelperInvocation" in (getattr(self, table, None) or {}):
                return None
        if self.target_profile in {"dx10", "dx11"}:
            return self.refuse_lowering(
                "gl_HelperInvocation: IsHelperLane() requires Shader Model 6.6, "
                f"which DirectX profile {self.target_profile} does not provide",
                "false",
            )
        return "IsHelperLane()"

    def atomic_fence_operand_identifier(self, expr):
        name = self.expression_name(expr)
        if name is None:
            return None
        return str(name).lstrip(":").rsplit("::", 1)[-1]

    def atomic_fence_operand_text(self, expr):
        if isinstance(expr, BinaryOpNode) and expr.op == "|":
            return (
                f"{self.atomic_fence_operand_text(expr.left)} | "
                f"{self.atomic_fence_operand_text(expr.right)}"
            )
        name = self.atomic_fence_operand_identifier(expr)
        return name if name is not None else expression_debug_name(expr)

    def collect_atomic_fence_memory_flags(self, expr):
        if isinstance(expr, BinaryOpNode) and expr.op == "|":
            left = self.collect_atomic_fence_memory_flags(expr.left)
            right = self.collect_atomic_fence_memory_flags(expr.right)
            if left is None or right is None:
                return None
            return left | right
        name = self.atomic_fence_operand_identifier(expr)
        if name in self.HLSL_ATOMIC_FENCE_MEMORY_FLAGS:
            return frozenset({name})
        return None

    def directx_atomic_fence_contract_error(
        self,
        reason,
        args,
        *,
        source_location=None,
    ):
        rendered = [self.atomic_fence_operand_text(arg) for arg in args]
        return DirectXAtomicFenceLoweringError(
            reason,
            memory_flags=rendered[0] if rendered else None,
            memory_order=rendered[1] if len(rendered) > 1 else None,
            thread_scope=rendered[2] if len(rendered) > 2 else None,
            source_location=source_location,
        )

    def generate_hlsl_atomic_thread_fence(self, args, *, source_location=None):
        if len(args) != 3:
            raise self.directx_atomic_fence_contract_error(
                "invalid-argument-count",
                args,
                source_location=source_location,
            )

        flags = self.collect_atomic_fence_memory_flags(args[0])
        order = self.atomic_fence_operand_identifier(args[1])
        scope = self.atomic_fence_operand_identifier(args[2])
        if flags is None:
            raise self.directx_atomic_fence_contract_error(
                "unknown-memory-flags",
                args,
                source_location=source_location,
            )
        if order not in self.HLSL_ATOMIC_FENCE_MEMORY_ORDERS:
            raise self.directx_atomic_fence_contract_error(
                "unknown-memory-order",
                args,
                source_location=source_location,
            )
        if scope not in self.HLSL_ATOMIC_FENCE_THREAD_SCOPES:
            raise self.directx_atomic_fence_contract_error(
                "unknown-thread-scope",
                args,
                source_location=source_location,
            )

        if scope != "thread_scope_threadgroup":
            reason = (
                "unsupported-system-thread-scope"
                if scope == "thread_scope_system"
                else "unsupported-thread-scope"
            )
            raise self.directx_atomic_fence_contract_error(
                reason,
                args,
                source_location=source_location,
            )
        if order != "memory_order_acq_rel":
            raise self.directx_atomic_fence_contract_error(
                "unsupported-memory-order",
                args,
                source_location=source_location,
            )
        if flags != frozenset({"mem_threadgroup"}):
            raise self.directx_atomic_fence_contract_error(
                "unsupported-memory-flags",
                args,
                source_location=source_location,
            )

        return "GroupMemoryBarrier()"

    def synchronization_function_call(self, func_name, args, *, source_location=None):
        if not func_name or func_name in getattr(self, "function_return_types", {}):
            return None

        if func_name in self.HLSL_SOFTWARE_SUBGROUP_BARRIERS:
            if args:
                raise self.hlsl_software_subgroup_error(
                    "DirectX subgroup execution barriers require zero arguments",
                    operation=func_name,
                    reason="invalid-argument-count",
                    source_location=source_location,
                )
            if (
                self.software_subgroup_width is None
                or self.hlsl_software_subgroup_workgroup_size is None
                or self.current_function_name
                not in self.hlsl_software_subgroup_function_names
            ):
                raise self.hlsl_software_subgroup_error(
                    "DirectX subgroup execution barriers require checked software "
                    "subgroup lowering with uniform workgroup participation",
                    operation=func_name,
                    reason="execution-barrier-contract-unproven",
                    source_location=source_location,
                )
            # The subgroup barrier also orders buffer and image memory (GL_KHR_shader_subgroup).
            intrinsic = (
                "AllMemoryBarrierWithGroupSync"
                if func_name == "subgroupBarrier"
                else "GroupMemoryBarrierWithGroupSync"
            )
            if (
                intrinsic in self.global_variable_types
                or intrinsic in self.standard_math_constant_shadow_names
            ):
                raise self.hlsl_software_subgroup_error(
                    "DirectX subgroup execution barrier target intrinsic is shadowed "
                    "by a global source declaration",
                    operation=func_name,
                    reason="target-intrinsic-shadowed",
                    source_location=source_location,
                )
            return f"{intrinsic}()"

        if func_name == "atomicThreadFence":
            return self.generate_hlsl_atomic_thread_fence(
                args,
                source_location=source_location,
            )

        intrinsic = self.synchronization_intrinsic_name(func_name)
        if intrinsic is None:
            return None
        if args:
            raise DirectXRefusal(
                f"DirectX synchronization builtin '{func_name}' requires 0 "
                f"argument(s), got {len(args)}"
            )
        return f"{intrinsic}()"

    def synchronization_intrinsic_name(self, func_name):
        return self.HLSL_SYNCHRONIZATION_INTRINSICS.get(func_name)

    def hlsl_builtin_function_name(self, func_name, args=None):
        if not func_name or func_name in getattr(self, "function_return_types", {}):
            return None
        if func_name == "atan" and args is not None and len(args) == 2:
            return "atan2"
        if func_name == "fma" and args:
            # HLSL fma is the fused double-precision intrinsic; mad may be
            # evaluated unfused, which GLSL and SPIR-V Fma do not allow.
            argument_type = self.expression_result_type(args[0])
            mapped = self.map_type(argument_type) if argument_type else None
            if isinstance(mapped, str) and mapped.startswith("double"):
                return "fma"
        return self.function_map.get(func_name)

    def hlsl_builtin_function_call(self, func_name, args, *, source_location=None):
        if not func_name or func_name in getattr(self, "function_return_types", {}):
            return None
        bfloat_argument_types = [
            self.hlsl_source_expression_type(argument) for argument in args
        ]
        vector_builtin = self.generate_hlsl_bfloat16_vector_builtin_call(
            func_name,
            args,
            bfloat_argument_types,
            source_location=source_location,
        )
        if vector_builtin is not None:
            return vector_builtin
        if any(
            self.is_hlsl_bfloat16_type(argument_type)
            for argument_type in bfloat_argument_types
        ):
            return self.generate_hlsl_bfloat16_builtin_call(
                func_name,
                args,
                bfloat_argument_types,
                source_location=source_location,
            )
        metal_math_call = self.hlsl_metal_math_call(
            func_name, args, source_location=source_location
        )
        if metal_math_call is not None:
            return metal_math_call
        boolean_order_call = self.hlsl_ordered_boolean_minmax_call(
            func_name,
            args,
            source_location=source_location,
        )
        if boolean_order_call is not None:
            return boolean_order_call
        trailing_zero_call = self.generate_hlsl_trailing_zero_call(
            func_name,
            args,
            source_location=source_location,
        )
        if trailing_zero_call is not None:
            return trailing_zero_call
        if not args and func_name in {"quiet_NaN", "signaling_NaN"}:
            return "asfloat(0x7fc00000)"
        if not args and func_name == "infinity":
            return "asfloat(0x7f800000)"
        if (
            func_name in {"fwidthFine", "fwidthCoarse", "fwidth_fine", "fwidth_coarse"}
            and len(args) == 1
        ):
            # HLSL has no fine or coarse fwidth; GLSL 4.60 section 8.14
            # defines fwidthFine(p) as abs(dFdxFine(p)) + abs(dFdyFine(p)).
            # The GLSL importer writes fwidthFine as fwidth_fine.
            granularity = (
                "fine" if func_name in {"fwidthFine", "fwidth_fine"} else "coarse"
            )
            targets = ("abs", f"ddx_{granularity}", f"ddy_{granularity}")
            lowering = f"abs(ddx_{granularity}(p)) + abs(ddy_{granularity}(p))"
            shadowed = [
                target
                for target in targets
                if self.hlsl_function_name_is_shadowed(target)
            ]
            if shadowed:
                return self.refuse_lowering(
                    f"{func_name} call: HLSL has no {func_name}, and its lowering "
                    f"{lowering} would name the shader's own '{shadowed[0]}'",
                    "0.0",
                )
            # The lowering evaluates p twice, so p must not assign, increment
            # or call one of the shader's own functions.
            if any(
                isinstance(node, AssignmentNode)
                or (isinstance(node, UnaryOpNode) and node.op in {"++", "--"})
                or (
                    isinstance(node, FunctionCallNode)
                    and self.hlsl_function_name_is_shadowed(
                        self.function_call_name(node)
                    )
                )
                for node in self.walk_ast(args[0])
            ):
                return self.refuse_lowering(
                    f"{func_name} call: HLSL has no {func_name}, and its lowering "
                    f"{lowering} would evaluate an argument with side effects "
                    "twice",
                    "0.0",
                )
            arg = self.generate_expression(args[0])
            return f"(abs(ddx_{granularity}({arg})) + abs(ddy_{granularity}({arg})))"

        if (
            func_name in {"frac", "fract"}
            and len(args) == 1
            and self.hlsl_function_name_is_shadowed("frac")
        ):
            arg = self.generate_expression(args[0])
            return f"(({arg}) - floor({arg}))"

        if (
            func_name in {"lerp", "mix"}
            and len(args) == 3
            and self.hlsl_function_name_is_shadowed("lerp")
        ):
            left = self.generate_expression(args[0])
            right = self.generate_expression(args[1])
            weight = self.generate_expression(args[2])
            return f"({left} + (({right} - {left}) * {weight}))"

        if (
            func_name in {"inverseSqrt", "inversesqrt", "rsqrt"}
            and len(args) == 1
            and self.hlsl_function_name_is_shadowed("rsqrt")
        ):
            arg = self.generate_expression(args[0])
            return f"(1.0 / sqrt({arg}))"

        if func_name != "mod" or len(args) != 2:
            return None

        left = self.generate_expression(args[0])
        right = self.generate_expression(args[1])
        return f"(({left}) - (({right}) * floor(({left}) / ({right}))))"

    def hlsl_metal_math_call(self, func_name, args, *, source_location=None):
        aliases = {"fabs": ("abs", 1), "fmax": ("max", 2), "fmin": ("min", 2)}
        if func_name in aliases:
            target_name, arity = aliases[func_name]
        elif func_name == "select":
            target_name, arity = "select", 3
        else:
            return None
        if len(args) != arity:
            raise DirectXContextualConversionError(
                f"DirectX builtin '{func_name}' requires {arity} arguments, "
                f"got {len(args)}",
                source_type=None,
                target_type=target_name,
                reason="metal-math-invalid-arity",
                source_location=source_location,
            )
        if (
            self.hlsl_function_name_is_shadowed(target_name)
            or target_name in self.global_variable_types
        ):
            raise DirectXContextualConversionError(
                f"DirectX cannot lower '{func_name}' through shadowed "
                f"intrinsic '{target_name}'",
                source_type=None,
                target_type=target_name,
                reason="metal-math-target-shadowed",
                source_location=source_location,
            )
        if func_name == "select":
            condition_type = self.expression_result_type(args[2])
            mapped_condition = self.map_type(condition_type) if condition_type else None
            if not mapped_condition or not is_boolean_type(mapped_condition):
                raise DirectXContextualConversionError(
                    "DirectX canonical select requires a resolved boolean "
                    "scalar or vector condition",
                    source_type=mapped_condition,
                    target_type="bool or boolean vector",
                    reason=(
                        "select-condition-not-boolean"
                        if mapped_condition
                        else "select-condition-unresolved"
                    ),
                    source_location=source_location,
                )
            # Metal orders arguments as (false, true, condition); HLSL uses
            # (condition, true, false). Both evaluate the two value arguments.
            args = (args[2], args[1], args[0])
        rendered = ", ".join(self.generate_expression(argument) for argument in args)
        return f"{target_name}({rendered})"

    def hlsl_bfloat16_builtin_result_type(
        self, func_name, argument_types, argument_count
    ):
        contract = self.HLSL_BFLOAT16_BUILTIN_CONTRACTS.get(func_name)
        if contract is None or argument_count != contract[1]:
            return None
        if contract[2] != "bfloat":
            return contract[2]
        if not argument_types or not all(
            self.is_hlsl_bfloat16_type(argument_type)
            for argument_type in argument_types
        ):
            return None
        return argument_types[0]

    def generate_hlsl_bfloat16_builtin_call(
        self,
        func_name,
        args,
        argument_types,
        *,
        source_location=None,
    ):
        source_type = next(
            argument_type
            for argument_type in argument_types
            if self.is_hlsl_bfloat16_type(argument_type)
        )
        result_type = self.hlsl_bfloat16_builtin_result_type(
            func_name, argument_types, len(args)
        )
        contract = self.HLSL_BFLOAT16_BUILTIN_CONTRACTS.get(func_name)
        if result_type is None or contract is None:
            raise self.directx_bfloat16_unsupported(
                "DirectX exact bfloat16 lowering cannot route bfloat16 payloads "
                f"through builtin '{func_name}' without a proven source contract",
                operation=func_name,
                source_type=source_type,
                reason="unsupported-bfloat16-builtin",
                source_location=source_location,
            )

        rendered_args = []
        for argument, argument_type in zip(args, argument_types):
            rendered = self.generate_expression(argument)
            if self.is_hlsl_bfloat16_type(argument_type):
                rendered = self.hlsl_bfloat16_to_float_expression(rendered)
            rendered_args.append(rendered)
        call = f"{contract[0]}({', '.join(rendered_args)})"
        if self.is_hlsl_bfloat16_type(result_type):
            return self.hlsl_float_to_bfloat16_expression(call)
        return call

    def hlsl_ordered_boolean_minmax_call(
        self,
        func_name,
        args,
        *,
        source_location=None,
    ):
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
                raise DirectXBooleanOrderedIntrinsicError(
                    "DirectX codegen cannot preserve ordered boolean "
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
        operator = "&" if func_name == "min" else "|"
        return f"(({left}) {operator} ({right}))"

    def hlsl_trailing_zero_builtin_is_shadowed(self, func_name):
        return (
            func_name in getattr(self, "function_return_types", {})
            or func_name in self.local_variable_types
            or func_name in self.global_variable_types
        )

    def hlsl_trailing_zero_result_type(self, func_name, args):
        if (
            func_name not in self.HLSL_CLANG_TRAILING_ZERO_BUILTINS
            or self.hlsl_trailing_zero_builtin_is_shadowed(func_name)
        ):
            return None
        return "int"

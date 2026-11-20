"""Cooperative matrix types in the CrossGL parser: type parsing, contracts and their merging, and contract inference for operations and constructors."""

from copy import deepcopy

from ..ast import (
    ArrayAccessNode,
    ArrayType,
    ASTNode,
    CooperativeMatrixOpNode,
    CooperativeMatrixType,
    IdentifierNode,
    LiteralNode,
    NamedType,
    ReferenceType,
    TypeNode,
    UnaryOpNode,
)
from ..refusals import CrossGLSyntaxError
from .vocabulary import COOPERATIVE_MATRIX_FIELD_LABELS


class ParserCooperativeMatrixMixin:
    """Cooperative matrix members of ``Parser``."""

    def cooperative_matrix_type(self, generic_args):
        """Build the canonical cooperative matrix type from generic arguments."""
        if not 3 <= len(generic_args) <= 12:
            raise CrossGLSyntaxError(
                "CooperativeMatrix expects 3 to 12 generic arguments: "
                "element type, rows, columns, scope, use, layout, fragment "
                "layout, subgroup size, elements per lane, fragment provenance, "
                "fragment mapping, and fragment mapping provenance"
            )

        element_type = generic_args[0]
        if not isinstance(element_type, TypeNode):
            raise CrossGLSyntaxError(
                "CooperativeMatrix element argument must be a type"
            )

        rows = self.cooperative_matrix_dimension(generic_args[1])
        cols = self.cooperative_matrix_dimension(generic_args[2])
        scope = self.cooperative_matrix_contract_label(
            generic_args[3] if len(generic_args) > 3 else None,
            "subgroup",
        )
        use = self.cooperative_matrix_contract_label(
            generic_args[4] if len(generic_args) > 4 else None,
            "unspecified",
        )
        layout = self.cooperative_matrix_contract_label(
            generic_args[5] if len(generic_args) > 5 else None,
            "unspecified",
        )
        fragment_layout = self.cooperative_matrix_optional_contract_label(
            generic_args[6] if len(generic_args) > 6 else None,
        )
        subgroup_size = self.cooperative_matrix_optional_dimension(
            generic_args[7] if len(generic_args) > 7 else None
        )
        elements_per_lane = self.cooperative_matrix_optional_dimension(
            generic_args[8] if len(generic_args) > 8 else None
        )
        fragment_provenance = self.cooperative_matrix_optional_contract_label(
            generic_args[9] if len(generic_args) > 9 else None,
        )
        fragment_mapping = self.cooperative_matrix_optional_contract_label(
            generic_args[10] if len(generic_args) > 10 else None,
        )
        fragment_mapping_provenance = self.cooperative_matrix_optional_contract_label(
            generic_args[11] if len(generic_args) > 11 else None,
        )
        try:
            return CooperativeMatrixType(
                element_type,
                rows,
                cols,
                scope,
                use,
                layout,
                fragment_layout,
                subgroup_size,
                elements_per_lane,
                fragment_provenance,
                fragment_mapping,
                fragment_mapping_provenance,
            )
        except ValueError as error:
            raise CrossGLSyntaxError(str(error)) from error

    def cooperative_matrix_dimension(self, argument):
        if isinstance(argument, LiteralNode) and type(argument.value) is int:
            return argument.value
        if (
            isinstance(argument, UnaryOpNode)
            and argument.op in {"+", "-"}
            and isinstance(argument.operand, LiteralNode)
            and type(argument.operand.value) is int
        ):
            value = argument.operand.value
            return value if argument.op == "+" else -value
        return argument

    def cooperative_matrix_optional_dimension(self, argument):
        if argument is None:
            return None
        if isinstance(argument, LiteralNode) and isinstance(argument.value, bool):
            return argument.value
        if isinstance(argument, (IdentifierNode, NamedType)):
            if argument.name == CooperativeMatrixType.UNSPECIFIED_FRAGMENT_CONTRACT:
                return None
            if argument.name in {"true", "false"}:
                return argument.name == "true"
        return self.cooperative_matrix_dimension(argument)

    def cooperative_matrix_optional_contract_label(self, argument):
        label = self.cooperative_matrix_contract_label(
            argument,
            CooperativeMatrixType.UNSPECIFIED_FRAGMENT_CONTRACT,
        )
        if label == CooperativeMatrixType.UNSPECIFIED_FRAGMENT_CONTRACT:
            return None
        return label

    def cooperative_matrix_contract_label(self, argument, default):
        if argument is None:
            return default
        if isinstance(argument, (IdentifierNode, NamedType)):
            return argument.name
        if isinstance(argument, LiteralNode):
            return str(argument.value)
        return self.format_type_argument(argument)

    def cooperative_matrix_value_type(self, value_type):
        """Return the cooperative-matrix contract represented by a value type."""
        while isinstance(value_type, ReferenceType):
            value_type = value_type.referenced_type
        if isinstance(value_type, CooperativeMatrixType):
            return value_type
        return None

    def cooperative_matrix_expression_type(self, expression):
        """Resolve an expression type from parsed results or lexical bindings."""
        if isinstance(expression, IdentifierNode):
            return self.lookup_value_type(expression.name)
        if isinstance(expression, CooperativeMatrixOpNode):
            return expression.expression_type
        if isinstance(expression, ArrayAccessNode):
            # One element of an array of matrices, such as ``tiles[i]``.
            array_type = self.cooperative_matrix_expression_type(expression.array)
            while isinstance(array_type, ReferenceType):
                array_type = array_type.referenced_type
            if isinstance(array_type, ArrayType) and isinstance(
                array_type.element_type, CooperativeMatrixType
            ):
                return array_type.element_type
        return getattr(expression, "expression_type", None)

    def cooperative_matrix_operand_contract(self, expression, operation, label):
        """Resolve one required matrix operand or report the missing contract."""
        contract = self.cooperative_matrix_value_type(
            self.cooperative_matrix_expression_type(expression)
        )
        if contract is None:
            raise CrossGLSyntaxError(
                f"Cooperative matrix {operation} operand {label} has no "
                "cooperative-matrix contract"
            )
        return contract

    def cooperative_matrix_values_equal(self, left, right):
        """Compare concrete or symbolic contract values structurally."""
        if isinstance(left, ASTNode) or isinstance(right, ASTNode):
            if not isinstance(left, ASTNode) or not isinstance(right, ASTNode):
                return False
            return self.format_type_argument(left) == self.format_type_argument(right)
        return left == right

    def format_cooperative_matrix_contract_value(self, value):
        if isinstance(value, ASTNode):
            return self.format_type_argument(value)
        return repr(value)

    @staticmethod
    def cooperative_matrix_field_is_unspecified(field, value):
        if field in {"use", "layout"}:
            return value == CooperativeMatrixType.UNSPECIFIED_FRAGMENT_CONTRACT
        if field in {
            "fragment_layout",
            "subgroup_size",
            "elements_per_lane",
            "fragment_provenance",
            "fragment_mapping",
            "fragment_mapping_provenance",
        }:
            return value is None
        return False

    def merge_cooperative_matrix_field(
        self,
        operation,
        field,
        left,
        right,
        left_label,
        right_label,
    ):
        """Unify one matrix-contract field, treating explicit gaps as wildcards."""
        if self.cooperative_matrix_field_is_unspecified(field, left):
            return deepcopy(right)
        if self.cooperative_matrix_field_is_unspecified(field, right):
            return deepcopy(left)
        if self.cooperative_matrix_values_equal(left, right):
            return deepcopy(right)

        field_name = COOPERATIVE_MATRIX_FIELD_LABELS[field]
        raise CrossGLSyntaxError(
            f"Cooperative matrix {operation} has incompatible {field_name}: "
            f"{left_label} is "
            f"{self.format_cooperative_matrix_contract_value(left)}, but "
            f"{right_label} is "
            f"{self.format_cooperative_matrix_contract_value(right)}"
        )

    def build_cooperative_matrix_contract(self, operation, values):
        """Build an inferred contract and normalize AST-owned values."""
        try:
            return CooperativeMatrixType(
                deepcopy(values["element_type"]),
                deepcopy(values["rows"]),
                deepcopy(values["cols"]),
                values["scope"],
                values["use"],
                values["layout"],
                values["fragment_layout"],
                deepcopy(values["subgroup_size"]),
                deepcopy(values["elements_per_lane"]),
                values["fragment_provenance"],
                values["fragment_mapping"],
                values["fragment_mapping_provenance"],
            )
        except ValueError as error:
            raise CrossGLSyntaxError(
                f"Cooperative matrix {operation} inferred an invalid result "
                f"contract: {error}"
            ) from error

    def cooperative_matrix_contract_values(self, contract):
        return {
            "element_type": contract.element_type,
            "rows": contract.rows,
            "cols": contract.cols,
            "scope": contract.scope,
            "use": contract.use,
            "layout": contract.layout,
            "fragment_layout": contract.fragment_layout,
            "subgroup_size": contract.subgroup_size,
            "elements_per_lane": contract.elements_per_lane,
            "fragment_provenance": contract.fragment_provenance,
            "fragment_mapping": contract.fragment_mapping,
            "fragment_mapping_provenance": contract.fragment_mapping_provenance,
        }

    def merge_cooperative_matrix_contracts(
        self,
        operation,
        left,
        right,
        left_label,
        right_label,
        *,
        right_owned_fields=(),
    ):
        """Unify two compatible matrix contracts into a distinct AST type.

        Fields in ``right_owned_fields`` are taken from ``right`` unchanged.
        """
        left_values = self.cooperative_matrix_contract_values(left)
        right_values = self.cooperative_matrix_contract_values(right)
        values = {
            field: (
                deepcopy(right_values[field])
                if field in right_owned_fields
                else self.merge_cooperative_matrix_field(
                    operation,
                    field,
                    left_values[field],
                    right_values[field],
                    left_label,
                    right_label,
                )
            )
            for field in left_values
        }
        return self.build_cooperative_matrix_contract(operation, values)

    def constrain_cooperative_matrix_accumulator(self, operation, product, accumulator):
        """Validate a product shape while preserving accumulator representation.

        The accumulator keeps its own component type, which may differ from the
        operands' (MSL simdgroup_multiply_accumulate, SPIR-V
        OpCooperativeMatrixMulAddKHR).
        """
        values = self.cooperative_matrix_contract_values(accumulator)
        product_values = self.cooperative_matrix_contract_values(product)
        for field in ("rows", "cols", "scope", "subgroup_size"):
            values[field] = self.merge_cooperative_matrix_field(
                operation,
                field,
                product_values[field],
                values[field],
                "product",
                "accumulator operand",
            )
        return self.build_cooperative_matrix_contract(operation, values)

    def cooperative_matrix_contract_shape_matches(self, contract, rows, cols):
        return self.cooperative_matrix_values_equal(
            contract.rows, rows
        ) and self.cooperative_matrix_values_equal(contract.cols, cols)

    def common_cooperative_matrix_fragment_values(self, contracts, rows, cols):
        """Return fragment metadata shared by same-shaped operand contracts."""
        fragment_fields = (
            "fragment_layout",
            "elements_per_lane",
            "fragment_provenance",
            "fragment_mapping",
            "fragment_mapping_provenance",
        )
        result = {field: None for field in fragment_fields}
        if not all(
            self.cooperative_matrix_contract_shape_matches(contract, rows, cols)
            for contract in contracts
        ):
            return result

        for field in fragment_fields:
            concrete = [
                getattr(contract, field)
                for contract in contracts
                if getattr(contract, field) is not None
            ]
            if concrete and all(
                self.cooperative_matrix_values_equal(concrete[0], value)
                for value in concrete[1:]
            ):
                result[field] = deepcopy(concrete[0])

        if result["fragment_layout"] is None:
            result["fragment_mapping"] = None
            result["fragment_mapping_provenance"] = None
        elif result["fragment_mapping"] is None:
            result["fragment_mapping_provenance"] = None
        return result

    def infer_cooperative_matrix_multiply_contract(
        self,
        operation,
        left,
        right,
        left_label="left",
        right_label="right",
        *,
        same_component_type=True,
    ):
        """Infer the source-neutral result contract for matrix multiplication.

        A plain product needs one component type. A multiply-accumulate takes
        its A and B component types independently, as MSL and SPIR-V do.
        """
        if same_component_type:
            self.merge_cooperative_matrix_field(
                operation,
                "element_type",
                left.element_type,
                right.element_type,
                left_label,
                right_label,
            )
        self.merge_cooperative_matrix_field(
            operation,
            "scope",
            left.scope,
            right.scope,
            left_label,
            right_label,
        )
        if not self.cooperative_matrix_values_equal(left.cols, right.rows):
            raise CrossGLSyntaxError(
                f"Cooperative matrix {operation} has incompatible inner dimension: "
                f"{left_label} columns are "
                f"{self.format_cooperative_matrix_contract_value(left.cols)}, but "
                f"{right_label} rows are "
                f"{self.format_cooperative_matrix_contract_value(right.rows)}"
            )

        subgroup_size = self.merge_cooperative_matrix_field(
            operation,
            "subgroup_size",
            left.subgroup_size,
            right.subgroup_size,
            left_label,
            right_label,
        )
        fragments = self.common_cooperative_matrix_fragment_values(
            (left, right), left.rows, right.cols
        )
        return self.build_cooperative_matrix_contract(
            operation,
            {
                "element_type": left.element_type,
                "rows": left.rows,
                "cols": right.cols,
                "scope": left.scope,
                "use": CooperativeMatrixType.UNSPECIFIED_FRAGMENT_CONTRACT,
                "layout": CooperativeMatrixType.UNSPECIFIED_FRAGMENT_CONTRACT,
                "subgroup_size": subgroup_size,
                **fragments,
            },
        )

    def infer_cooperative_matrix_operation(self, operation_node):
        """Infer and attach a cooperative-matrix operation result contract."""
        operation = operation_node.operation
        arguments = operation_node.arguments
        if operation in {"load", "store"}:
            return operation_node
        if operation in {"fill", "diagonal"}:
            # A fill or diagonal names only its value; the declaration,
            # assignment or return it initializes supplies the matrix contract.
            if len(arguments) != 1:
                raise CrossGLSyntaxError(
                    f"Cooperative matrix {operation} expects 1 argument, got "
                    f"{len(arguments)}"
                )
            return operation_node

        if operation == "element":
            if not arguments:
                raise CrossGLSyntaxError(
                    "Cooperative matrix element requires a matrix operand"
                )
            matrix = self.cooperative_matrix_operand_contract(
                arguments[0], operation, "matrix"
            )
            operation_node.set_expression_type(deepcopy(matrix.element_type))
            return operation_node

        expected_arity = {
            "negate": (1,),
            "elementwise_add": (2,),
            "elementwise_subtract": (2,),
            "elementwise_multiply": (2,),
            "multiply": (2,),
            "multiply_accumulate": (3, 4),
        }[operation]
        if len(arguments) not in expected_arity:
            return operation_node

        if operation == "negate":
            result = deepcopy(
                self.cooperative_matrix_operand_contract(
                    arguments[0], operation, "matrix"
                )
            )
        elif operation.startswith("elementwise_"):
            left = self.cooperative_matrix_operand_contract(
                arguments[0], operation, "left"
            )
            right = self.cooperative_matrix_operand_contract(
                arguments[1], operation, "right"
            )
            result = self.merge_cooperative_matrix_contracts(
                operation, left, right, "left operand", "right operand"
            )
        elif operation == "multiply":
            left = self.cooperative_matrix_operand_contract(
                arguments[0], operation, "left"
            )
            right = self.cooperative_matrix_operand_contract(
                arguments[1], operation, "right"
            )
            result = self.infer_cooperative_matrix_multiply_contract(
                operation, left, right
            )
        else:
            destination = None
            if len(arguments) == 4:
                destination = self.cooperative_matrix_operand_contract(
                    arguments[0], operation, "destination"
                )
                arguments = arguments[1:]
            left = self.cooperative_matrix_operand_contract(
                arguments[0], operation, "left"
            )
            right = self.cooperative_matrix_operand_contract(
                arguments[1], operation, "right"
            )
            accumulator = self.cooperative_matrix_operand_contract(
                arguments[2], operation, "accumulator"
            )
            product = self.infer_cooperative_matrix_multiply_contract(
                operation, left, right, same_component_type=False
            )
            result = self.constrain_cooperative_matrix_accumulator(
                operation, product, accumulator
            )
            if destination is not None:
                # The destination's component type is the result's; MSL lets it
                # differ from the accumulator's.
                result = self.merge_cooperative_matrix_contracts(
                    operation,
                    result,
                    destination,
                    "inferred result",
                    "destination operand",
                    right_owned_fields=("element_type",),
                )

        operation_node.set_result_contract(result)
        return operation_node

    def parse_cooperative_matrix_constructor(self, generic_args):
        """Parse ``CooperativeMatrix<...>(value)``, the diagonal constructor.

        Like a GLSL ``matN(s)`` constructor, the one value fills the diagonal
        and every other element is zero; it is the canonical
        ``cooperative_matrix_diagonal`` with the written type as its result.
        """
        matrix_type = self.cooperative_matrix_type(generic_args)
        self.eat("LPAREN")
        arguments = self.parse_call_arguments()
        self.eat("RPAREN")
        if len(arguments) != 1:
            raise CrossGLSyntaxError(
                "CooperativeMatrix constructor expects 1 diagonal value, got "
                f"{len(arguments)}"
            )
        operation = CooperativeMatrixOpNode("diagonal", arguments)
        operation.set_result_contract(matrix_type)
        return operation

    @staticmethod
    def type_is_inferred_declaration(value_type):
        return value_type is None or (
            isinstance(value_type, NamedType) and value_type.name in {"auto", "var"}
        )

    def constrain_cooperative_matrix_result(self, expression, expected_type, context):
        """Apply an explicit declaration, assignment, or return type constraint."""
        if not isinstance(expression, CooperativeMatrixOpNode):
            return
        if self.type_is_inferred_declaration(expected_type):
            return

        if expression.operation == "element":
            if not self.cooperative_matrix_values_equal(
                expression.expression_type, expected_type
            ):
                raise CrossGLSyntaxError(
                    "Cooperative matrix element has incompatible contextual type: "
                    f"inferred component is "
                    f"{self.format_cooperative_matrix_contract_value(expression.expression_type)}, "
                    f"but {context} is "
                    f"{self.format_cooperative_matrix_contract_value(expected_type)}"
                )
            return

        expected_contract = self.cooperative_matrix_value_type(expected_type)
        if expected_contract is None:
            raise CrossGLSyntaxError(
                f"Cooperative matrix {expression.operation} requires a "
                f"cooperative-matrix result, but {context} is "
                f"{self.format_cooperative_matrix_contract_value(expected_type)}"
            )
        if expression.result_type is None:
            if expression.operation in {"fill", "diagonal"}:
                expression.set_result_contract(deepcopy(expected_contract))
            return
        result = self.merge_cooperative_matrix_contracts(
            expression.operation,
            expression.result_type,
            expected_contract,
            "inferred result",
            context,
        )
        expression.set_result_contract(result)

    def finish_variable_contract(self, variable):
        """Constrain an initializer and register its effective lexical type."""
        if variable.initial_value is not None:
            self.constrain_cooperative_matrix_result(
                variable.initial_value,
                variable.var_type,
                f"declaration {variable.name!r}",
            )

        effective_type = variable.var_type
        if self.type_is_inferred_declaration(effective_type):
            effective_type = self.cooperative_matrix_expression_type(
                variable.initial_value
            )
        self.register_value_type(variable.name, effective_type)
        if getattr(variable, "generic_params", None):
            self.parameterized_value_names.add(str(variable.name))
        return variable

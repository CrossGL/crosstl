"""Arithmetic result types, expression types and address provenance."""

import re

from ....translator.float_limit_constants import float_limit_constant
from ....translator.integer_literals import integer_literal_parts
from ....translator.standard_constants import standard_math_constant
from ..generic_call_return_types import (
    expression_call_return_type,
    same_arity_overloads,
)
from ..matrix_arithmetic import (
    NOT_MATRIX,
    matrix_arithmetic_shape,
)
from ..matrix_columns import matrix_columns_object
from ..MetalAst import (
    ArrayAccessNode,
    AssignmentNode,
    BinaryOpNode,
    CastNode,
    FunctionCallNode,
    MemberAccessNode,
    MethodCallNode,
    PostfixOpNode,
    TernaryOpNode,
    TextureSampleNode,
    UnaryOpNode,
    VariableNode,
    VectorConstructorNode,
)
from ..narrow_float_constants import narrow_float_constant
from ..texture_method_results import (
    is_depth_texture_name,
    texture_method_result_shape,
)
from ..vote_masks import vote_mask_call_metal_type
from .support import (
    MetalAddressProvenanceError,
    MetalArithmeticTypeResolutionError,
    MetalIndexedComponentTypeResolutionError,
)


class MetalExpressionTypeMixin:
    """Members of ``MetalToCrossGLConverter``: arithmetic result types, expression types and address provenance."""

    def metal_small_vector_member_type(self, vector_type, member, source_location=None):
        component_parts = self.metal_vector_component_parts(vector_type)
        if component_parts is None:
            return None
        element_type, width, width_text = component_parts
        if width is None:
            raise MetalIndexedComponentTypeResolutionError(
                vector_type,
                "swizzle",
                f"the vector width '{width_text}' is not concrete",
                source_location,
                str(member),
            )
        vector_parts = self.metal_small_vector_type_parts(vector_type)
        if vector_parts is None:
            return None
        element_type, width = vector_parts
        selector = str(member)
        component_indices = {
            "x": 0,
            "r": 0,
            "y": 1,
            "g": 1,
            "z": 2,
            "b": 2,
            "w": 3,
            "a": 3,
        }
        if re.fullmatch(r"s[0-9a-fA-F]+", selector):
            indices = [int(component, 16) for component in selector[1:]]
        else:
            indices = [
                component_indices.get(component, width) for component in selector
            ]
        if not indices or any(index >= width for index in indices):
            raise MetalIndexedComponentTypeResolutionError(
                vector_type,
                "swizzle",
                f"selector '{selector}' is outside the {width}-component vector",
                source_location,
                selector,
            )
        if len(indices) == 1:
            return element_type
        if len(indices) <= 4:
            return self.metal_vector_type_from_element(element_type, len(indices))
        raise MetalIndexedComponentTypeResolutionError(
            vector_type,
            "swizzle",
            f"selector '{selector}' has more than four components",
            source_location,
            selector,
        )

    def metal_scalar_arithmetic_type_info(self, metal_type):
        type_name = self.normalized_metal_type(self.resolve_type_alias(metal_type))
        enum_type = self.metal_enum_arithmetic_types.get(type_name)
        if enum_type is not None:
            return self.metal_scalar_arithmetic_type_info(enum_type)
        return self.metal_scalar_arithmetic_types.get(type_name)

    def metal_standard_math_constant_type(self, name):
        constant = standard_math_constant(name)
        if constant is not None and "metal" in constant.source_environments:
            return constant.scalar_type
        # INFINITY, NAN, FLT_MAX and the other binary32 limits are float constants (MSL section 6.6.1).
        if float_limit_constant(name) is not None:
            return "float"
        narrow_constant = narrow_float_constant(name)
        if narrow_constant is not None:
            return narrow_constant.metal_type
        return None

    def metal_literal_string_type(self, value):
        text = str(value).replace("'", "")
        if text in {"true", "false"}:
            return "bool"
        if re.fullmatch(r"'(?:[^'\\]|\\.)'", str(value)):
            return "char"

        integer = re.fullmatch(
            r"(?:0[xX][0-9a-fA-F]+|0[bB][01]+|[0-9]+)(?P<suffix>[uUlL]*)",
            text,
        )
        if integer is not None:
            return integer_literal_parts(text, legacy_octal=True)[1]

        floating = re.fullmatch(
            r"(?:"
            r"0[xX](?:[0-9a-fA-F]+\.?[0-9a-fA-F]*|\.[0-9a-fA-F]+)"
            r"[pP][+-]?[0-9]+|"
            r"(?:[0-9]+\.[0-9]*|\.[0-9]+|[0-9]+[eE][+-]?[0-9]+)"
            r"(?:[eE][+-]?[0-9]+)?"
            r")(?P<suffix>[fFhH]?)",
            text,
        )
        if floating is None:
            return None
        return "half" if floating.group("suffix").lower() == "h" else "float"

    def promoted_metal_integer_type(self, type_info):
        _family, signed, bits = type_info
        if bits < 32:
            return "int", True, 32
        if bits == 32:
            return ("int", True, 32) if signed else ("uint", False, 32)
        return ("int64_t", True, 64) if signed else ("uint64_t", False, 64)

    def metal_common_integer_type(self, left_info, right_info):
        left_name, left_signed, left_bits = self.promoted_metal_integer_type(left_info)
        right_name, right_signed, right_bits = self.promoted_metal_integer_type(
            right_info
        )
        if left_signed == right_signed:
            return left_name if left_bits >= right_bits else right_name
        signed_name, signed_bits = (
            (left_name, left_bits) if left_signed else (right_name, right_bits)
        )
        unsigned_name, unsigned_bits = (
            (right_name, right_bits) if left_signed else (left_name, left_bits)
        )
        if unsigned_bits >= signed_bits:
            return unsigned_name
        if signed_bits > unsigned_bits:
            return signed_name
        return "uint64_t" if signed_bits == 64 else "uint"

    def metal_vector_scalar_conversion(self, operator, left_type, right_type):
        if operator not in {
            "+",
            "-",
            "*",
            "/",
            "%",
            "&",
            "|",
            "^",
            "==",
            "!=",
            "<",
            "<=",
            ">",
            ">=",
        }:
            return None
        if any(
            self.metal_pointer_pointee_type_once(vtype) is not None
            for vtype in (left_type, right_type)
        ):
            return None
        left_vector = self.metal_small_vector_type_parts(left_type)
        right_vector = self.metal_small_vector_type_parts(right_type)
        if (left_vector is None) == (right_vector is None):
            return None
        index = 1 if left_vector is not None else 0
        element_type, _width = left_vector or right_vector
        scalar_type = (left_type, right_type)[index]
        element = self.metal_scalar_arithmetic_type_info(element_type)
        scalar = self.metal_scalar_arithmetic_type_info(scalar_type)
        if element is None or scalar is None:
            return None
        if element[0] == "integer" and scalar[0] == "integer" and element[2] != 1:
            return index, element_type
        # MSL converts a scalar operand to a floating vector's element type,
        # so ``half3 * float`` stays ``half3``. A wider floating scalar is the
        # case CrossGL would otherwise widen the vector for.
        if (
            element[0] == "floating"
            and scalar[0] == "floating"
            and scalar[2] > element[2]
            and operator in {"+", "-", "*", "/", "==", "!=", "<", "<=", ">", ">="}
            and not any(
                self.normalized_metal_type(self.resolve_type_alias(vtype))
                in self.metal_source_bfloat_types
                for vtype in (element_type, scalar_type)
            )
        ):
            return index, element_type
        return None

    def metal_vector_scalar_operand(self, rendered, source_type, element_type):
        if self.metal_scalar_arithmetic_type_info(
            source_type
        ) == self.metal_scalar_arithmetic_type_info(element_type):
            return rendered
        # Metal converts the scalar before the vector operation. Recording that
        # conversion in CrossGL also preserves it through saved intermediates.
        return f"{self.map_type(element_type)}({rendered})"

    def metal_bfloat_arithmetic_plan(
        self, operator, left_type, right_type, source_location=None
    ):
        """Retain the source floating family and its operand conversions."""
        if operator not in {"+", "-", "*", "/", "==", "!=", "<", "<=", ">", ">="}:
            return None
        types = (left_type, right_type)
        if any(
            self.metal_pointer_pointee_type_once(vtype) is not None
            or self.split_outer_metal_declarator_array_type(vtype) is not None
            for vtype in types
        ):
            return None
        vectors = [self.metal_small_vector_type_parts(vtype) for vtype in types]
        elements = [
            self.normalized_metal_type(
                self.resolve_type_alias(vector[0] if vector is not None else vtype)
            )
            for vtype, vector in zip(types, vectors)
        ]
        bfloat = [element in self.metal_source_bfloat_types for element in elements]
        if not any(bfloat):
            return None
        infos = [
            self.metal_scalar_arithmetic_type_info(element) for element in elements
        ]
        if any(info is None for info in infos):
            return None

        def invalid(reason):
            raise MetalArithmeticTypeResolutionError(
                operator, types, reason, source_location
            )

        conversions = [None, None]
        if any(vectors):
            vector_index = 0 if vectors[0] is not None else 1
            scalar_index = 1 - vector_index
            element, width = vectors[vector_index]
            if all(vectors):
                if vectors[0][1] != vectors[1][1] or not all(bfloat):
                    invalid("implicit conversion between these vector types is invalid")
            elif bfloat[vector_index]:
                if not bfloat[scalar_index]:
                    if infos[scalar_index][0] != "integer":
                        invalid("scalar rank is greater than or unordered with bfloat")
                    conversions[scalar_index] = "bfloat"
            else:
                if infos[vector_index][0] != "floating" or infos[vector_index][2] < 32:
                    invalid(
                        "bfloat scalar rank is greater than or unordered with vector element"
                    )
                conversions[scalar_index] = element
            return self.metal_vector_type_from_element(element, width), conversions

        if all(bfloat):
            return "bfloat", conversions
        other = 1 if bfloat[0] else 0
        if infos[other][0] == "integer":
            # Preserve both the operand conversion and every narrow intermediate.
            conversions[other] = "bfloat"
            return "bfloat", conversions
        if infos[other][2] < 32:
            invalid("bfloat and half have unordered floating-point ranks")
        conversions[1 - other] = elements[other]
        return elements[other], conversions

    def metal_scalar_binary_result_type(self, operator, left_type, right_type):
        bfloat_plan = self.metal_bfloat_arithmetic_plan(operator, left_type, right_type)
        if bfloat_plan is not None:
            return bfloat_plan[0]
        left_info = self.metal_scalar_arithmetic_type_info(left_type)
        right_info = self.metal_scalar_arithmetic_type_info(right_type)

        if operator in {"<<", ">>"}:
            if (
                left_info is None
                or right_info is None
                or left_info[0] != "integer"
                or right_info[0] != "integer"
            ):
                return None
            return self.promoted_metal_integer_type(left_info)[0]

        if left_info is None or right_info is None:
            left_name = self.normalized_metal_type(left_type)
            right_name = self.normalized_metal_type(right_type)
            if left_name == right_name and operator in {"+", "-", "*", "/", "%"}:
                return left_name
            return None

        if operator in {"&", "|", "^", "%"}:
            if left_info[0] != "integer" or right_info[0] != "integer":
                return None
            return self.metal_common_integer_type(left_info, right_info)
        if operator not in {"+", "-", "*", "/"}:
            return None
        if left_info[0] == "floating" or right_info[0] == "floating":
            floating_bits = max(
                info[2] for info in (left_info, right_info) if info[0] == "floating"
            )
            return {16: "half", 32: "float", 64: "double"}[floating_bits]
        return self.metal_common_integer_type(left_info, right_info)

    def metal_binary_expression_type(self, expr):
        left_type = self.expression_metal_type(expr.left)
        right_type = self.expression_metal_type(expr.right)
        selected_operator = self.resolve_metal_free_operator(
            expr.op,
            (expr.left, expr.right),
            getattr(expr, "source_location", None),
            argument_metal_types=(left_type, right_type),
        )
        if selected_operator is not None:
            return self.resolve_type_alias(
                getattr(selected_operator, "return_type", None)
            )
        bfloat_plan = self.metal_bfloat_arithmetic_plan(
            expr.op, left_type, right_type, getattr(expr, "source_location", None)
        )
        if bfloat_plan is not None and expr.op in {"+", "-", "*", "/"}:
            return bfloat_plan[0]
        if expr.op in {"==", "!=", "<", "<=", ">", ">=", "&&", "||"}:
            left_vector = self.metal_small_vector_type_parts(left_type)
            right_vector = self.metal_small_vector_type_parts(right_type)
            vector = left_vector or right_vector
            if expr.op not in {"&&", "||"} and vector is not None:
                if (
                    left_vector is not None
                    and right_vector is not None
                    and left_vector[1] != right_vector[1]
                ):
                    return None
                return f"bool{vector[1]}"
            return "bool"

        if left_type is None or right_type is None:
            return None
        pointer_source = self.metal_pointer_arithmetic_source(
            expr, left_type, right_type
        )
        if pointer_source is not None:
            return pointer_source[1]
        if (
            self.metal_pointer_pointee_type_once(left_type) is not None
            or self.metal_pointer_pointee_type_once(right_type) is not None
        ):
            return None
        conversion_plan = self.metal_builtin_lowered_conversion_plan(
            expr.op,
            (expr.left, expr.right),
            [left_type, right_type],
        )
        if conversion_plan is not None:
            _helpers, converted_types = conversion_plan
            left_type, right_type = converted_types
        left_wide = self.wide_vector_type_info(
            left_type,
            getattr(expr.left, "source_location", None),
        )
        right_wide = self.wide_vector_type_info(
            right_type,
            getattr(expr.right, "source_location", None),
        )
        if left_wide is not None or right_wide is not None:
            wide_type = left_wide or right_wide
            if (
                left_wide is not None
                and right_wide is not None
                and left_wide["key"] != right_wide["key"]
            ):
                return None
            if self.wide_vector_binary_operation_name(expr.op) is not None:
                return wide_type["source_type"]
            return None

        matrix_type = self.metal_matrix_arithmetic_type(expr.op, left_type, right_type)
        if matrix_type is not NOT_MATRIX:
            return matrix_type
        left_vector = self.metal_small_vector_type_parts(left_type)
        right_vector = self.metal_small_vector_type_parts(right_type)
        if left_vector is not None or right_vector is not None:
            vector = left_vector or right_vector
            if (
                self.metal_vector_scalar_conversion(expr.op, left_type, right_type)
                is not None
            ):
                return self.metal_vector_type_from_element(vector[0], vector[1])
            if (
                left_vector is not None
                and right_vector is not None
                and left_vector[1] != right_vector[1]
            ):
                return None
            left_element = left_vector[0] if left_vector is not None else left_type
            right_element = right_vector[0] if right_vector is not None else right_type
            result_element = self.metal_scalar_binary_result_type(
                expr.op, left_element, right_element
            )
            if result_element is None:
                return None
            return self.metal_vector_type_from_element(result_element, vector[1])

        return self.metal_scalar_binary_result_type(expr.op, left_type, right_type)

    def metal_matrix_arithmetic_type(self, op, left_type, right_type):
        """The type of MSL arithmetic with a matrix operand, or ``NOT_MATRIX`` without one."""
        left_matrix = self.metal_matrix_type_parts(left_type)
        right_matrix = self.metal_matrix_type_parts(right_type)
        if left_matrix is None and right_matrix is None:
            return NOT_MATRIX

        def operand(matrix, metal_type):
            if matrix is not None:
                return matrix[0], (matrix[1], matrix[2]), None
            vector = self.metal_small_vector_type_parts(metal_type)
            if vector is not None:
                return vector[0], None, vector[1]
            if self.metal_scalar_arithmetic_type_info(metal_type) is not None:
                return metal_type, None, 0
            return None, None, None

        left_element, left_shape, left_lanes = operand(left_matrix, left_type)
        right_element, right_shape, right_lanes = operand(right_matrix, right_type)
        shape = matrix_arithmetic_shape(
            op, left_shape, right_shape, left_lanes, right_lanes
        )
        if shape is None:
            return None
        element = self.resolve_type_alias((left_matrix or right_matrix)[0])
        other = right_element if left_matrix is not None else left_element
        scalar_operand = 0 in (left_lanes, right_lanes)
        # A vector or second matrix must share the matrix's element type; a scalar converts to it.
        if not scalar_operand and self.normalized_metal_type(
            self.resolve_type_alias(other)
        ) != self.normalized_metal_type(element):
            return None
        if shape[0] == "vector":
            return self.metal_vector_type_from_element(element, shape[1])
        return f"{element}{shape[1]}x{shape[2]}"

    def metal_pointer_arithmetic_operand_type(self, metal_type):
        element = self.split_outer_metal_declarator_array_type(metal_type)
        if (
            element is not None
            and self.split_outer_metal_declarator_array_type(element) is None
            and self.metal_pointer_pointee_type_once(element) is None
        ):
            return f"{element}*"
        return metal_type

    def metal_builtin_pointer_arithmetic_operand_index(
        self, operator, left_type, right_type
    ):
        """Return the pointer operand for a proven built-in pointer operation."""
        if operator not in {"+", "-"}:
            return None
        left_pointer = self.metal_pointer_pointee_type_once(
            self.metal_pointer_arithmetic_operand_type(left_type)
        )
        right_pointer = self.metal_pointer_pointee_type_once(
            self.metal_pointer_arithmetic_operand_type(right_type)
        )

        if left_pointer is not None and right_pointer is None:
            right_info = self.metal_scalar_arithmetic_type_info(right_type)
            if right_info is not None and right_info[0] == "integer":
                return 0
        if operator == "+" and right_pointer is not None and left_pointer is None:
            left_info = self.metal_scalar_arithmetic_type_info(left_type)
            if left_info is not None and left_info[0] == "integer":
                return 1
        return None

    def metal_pointer_arithmetic_source(self, expr, left_type=None, right_type=None):
        if not isinstance(expr, BinaryOpNode):
            return None
        if left_type is None:
            left_type = self.expression_metal_type(expr.left)
        if right_type is None:
            right_type = self.expression_metal_type(expr.right)
        pointer_operand = self.metal_builtin_pointer_arithmetic_operand_index(
            expr.op, left_type, right_type
        )
        if pointer_operand == 0:
            return expr.left, self.metal_pointer_arithmetic_operand_type(left_type)
        if pointer_operand == 1:
            return expr.right, self.metal_pointer_arithmetic_operand_type(right_type)
        return None

    def metal_member_type_qualifiers(self, expression):
        owner = self.normalized_metal_type(
            self.metal_source_overload_value_type(
                self.expression_metal_type(expression.object)
            )
        )
        declaration = self.struct_declarations.get(owner)
        for member in getattr(declaration, "members", ()) or ():
            if getattr(member, "name", None) != str(expression.member):
                continue
            qualifiers = set(self.metal_declaration_type_qualifiers(member))
            resolved = self.struct_member_types.get(owner, {}).get(member.name, "")
            pointee_qualifiers = getattr(member, "pointee_qualifiers", None)
            if "*" in resolved and pointee_qualifiers is not None:
                qualifiers = set(pointee_qualifiers)
                qualifiers.update(
                    self.resolved_struct_member_qualifiers.get(id(member), ())
                )
            for token in str(resolved).split():
                if token not in self.metal_source_overload_type_qualifiers:
                    break
                qualifiers.add(token)
            return tuple(
                qualifier
                for qualifier in self.metal_source_overload_type_qualifiers
                if qualifier in qualifiers
            )
        return ()

    def expression_metal_type_qualifiers(self, expr):
        if isinstance(expr, FunctionCallNode):
            selected = self.selected_metal_callable(expr)
            if selected is not None:
                return_type = self.selected_metal_callable_return_type(selected)
                if self.metal_pointer_pointee_type_once(return_type) is not None:
                    declaration = VariableNode(
                        selected.return_type,
                        "",
                        qualifiers=getattr(selected, "return_qualifiers", ()) or (),
                    )
                    qualifiers = set(self.resolved_declaration_qualifiers(declaration))
                    binding = self.scalar_alias_binding(
                        selected.return_type, context=selected
                    )
                    if binding is not None:
                        qualifiers.update(binding[1])
                    for token in str(return_type).split():
                        if token not in self.metal_source_overload_type_qualifiers:
                            break
                        qualifiers.add(token)
                    return self.normalized_metal_address_qualifiers(
                        qualifiers, getattr(expr, "source_location", None), return_type
                    )
        if isinstance(expr, TernaryOpNode):
            result_type = self.expression_metal_type(expr)
            if self.metal_pointer_pointee_type_once(result_type) is not None:
                qualifiers = set(self.expression_metal_type_qualifiers(expr.true_expr))
                qualifiers.update(
                    self.expression_metal_type_qualifiers(expr.false_expr)
                )
                return self.normalized_metal_address_qualifiers(
                    qualifiers, getattr(expr, "source_location", None), result_type
                )
        if isinstance(expr, CastNode):
            target_type = self.resolve_type_alias(expr.target_type)
            if self.metal_pointer_pointee_type_once(target_type) is not None:
                return self.normalized_metal_address_qualifiers(
                    getattr(expr, "qualifiers", ()),
                    getattr(expr, "source_location", None),
                    target_type,
                )
        if isinstance(expr, str):
            return self.current_variable_type_qualifiers.get(
                expr, self.global_variable_type_qualifiers.get(expr, ())
            )
        if isinstance(expr, VariableNode):
            name = getattr(expr, "name", None)
            if not name:
                return ()
            direct = self.metal_declaration_type_qualifiers(expr)
            if getattr(expr, "vtype", None) and direct:
                return direct
            return self.current_variable_type_qualifiers.get(
                name, self.global_variable_type_qualifiers.get(name, direct)
            )
        if isinstance(expr, BinaryOpNode):
            pointer_source = self.metal_pointer_arithmetic_source(expr)
            if pointer_source is not None:
                return self.expression_metal_type_qualifiers(pointer_source[0])
        if isinstance(expr, ArrayAccessNode):
            selected_type = self.expression_metal_type(expr)
            if self.metal_pointer_pointee_type_once(selected_type) is not None:
                # Standard arrays retain qualifiers inside their element type,
                # not on the array object that stores the pointer.
                qualifiers = []
                for token in str(selected_type).split():
                    if token not in self.metal_source_overload_type_qualifiers:
                        break
                    qualifiers.append(token)
                if qualifiers:
                    return tuple(qualifiers)
            return self.expression_metal_type_qualifiers(expr.array)
        if isinstance(expr, MemberAccessNode):
            declared_qualifiers = self.metal_member_type_qualifiers(expr)
            member_type = self.expression_metal_type(expr)
            if member_type is not None and "*" in member_type:
                # A pointer field names its pointee storage, independently of
                # the address space and cv qualification of its owning object.
                return declared_qualifiers
            owner_qualifiers = self.expression_metal_type_qualifiers(expr.object)
            return tuple(dict.fromkeys((*owner_qualifiers, *declared_qualifiers)))
        if isinstance(expr, UnaryOpNode):
            if expr.op == "&":
                provenance = self.metal_address_provenance(expr)
                return provenance[1]
            if expr.op == "*":
                return self.expression_metal_type_qualifiers(expr.operand)
        return ()

    def metal_address_provenance(self, expression):
        if not isinstance(expression, UnaryOpNode) or expression.op != "&":
            return None

        operand = expression.operand
        source_location = getattr(expression, "source_location", None) or getattr(
            operand, "source_location", None
        )
        operand_type = self.expression_metal_type(operand)

        if isinstance(operand, ArrayAccessNode):
            selection = self.metal_indexed_type_selection(operand)
            if selection["kind"] not in {"array", "pointer"}:
                raise MetalAddressProvenanceError(
                    f"{selection['kind']} subscript",
                    "the selected component is not independently addressable storage",
                    source_location,
                    selection["base_type"],
                )
            qualifiers = self.metal_addressable_storage_qualifiers(operand.array)
            if qualifiers is None:
                raise MetalAddressProvenanceError(
                    "indexed expression",
                    "the indexed base is not rooted in tracked storage",
                    source_location,
                    selection["base_type"],
                )
            return self.metal_address_pointer_provenance(
                selection["selected_type"], qualifiers, source_location, "subscript"
            )

        if isinstance(operand, MemberAccessNode):
            object_type = self.expression_metal_type(operand.object)
            if self.metal_vector_component_parts(object_type) is not None:
                raise MetalAddressProvenanceError(
                    "vector swizzle",
                    "Metal vector components do not provide portable addressable storage",
                    source_location,
                    object_type,
                )
            if self.metal_matrix_type_parts(object_type) is not None:
                raise MetalAddressProvenanceError(
                    "matrix component",
                    "Metal matrix components do not provide portable addressable storage",
                    source_location,
                    object_type,
                )
            qualifiers = self.metal_addressable_storage_qualifiers(operand.object)
            if operand_type is None or qualifiers is None:
                raise MetalAddressProvenanceError(
                    "member expression",
                    "the member is not rooted in tracked aggregate storage",
                    source_location,
                    object_type,
                )
            return self.metal_address_pointer_provenance(
                operand_type, qualifiers, source_location, "member expression"
            )

        if isinstance(operand, (str, VariableNode)):
            qualifiers = self.metal_addressable_storage_qualifiers(operand)
            if operand_type is None or qualifiers is None:
                raise MetalAddressProvenanceError(
                    "identifier",
                    "the identifier does not name tracked storage",
                    source_location,
                    operand_type,
                )
            name = operand if isinstance(operand, str) else operand.name
            parameter_names = {
                getattr(parameter, "name", None)
                for parameter in getattr(self.current_function, "params", ()) or ()
            }
            thread_local_pointer_slot = (
                bool(name)
                and name in self.current_variable_types
                and name not in self.global_variable_types
                and name not in parameter_names
            )
            return self.metal_address_pointer_provenance(
                operand_type,
                qualifiers,
                source_location,
                "identifier",
                allow_thread_local_pointer_slot=thread_local_pointer_slot,
            )

        if isinstance(operand, UnaryOpNode) and operand.op == "*":
            pointer_type = self.expression_metal_type(operand.operand)
            if self.metal_pointer_pointee_type_once(pointer_type) is None:
                raise MetalAddressProvenanceError(
                    "dereference",
                    "the dereferenced expression does not have a pointer type",
                    source_location,
                    pointer_type,
                )
            qualifiers = self.metal_addressable_storage_qualifiers(operand.operand)
            if qualifiers is None:
                raise MetalAddressProvenanceError(
                    "dereference",
                    "the pointer is not rooted in tracked storage",
                    source_location,
                    pointer_type,
                )
            return pointer_type, qualifiers

        raise MetalAddressProvenanceError(
            "temporary expression",
            "only stored lvalues can initialize an inferred pointer",
            source_location,
            operand_type,
        )

    def metal_address_pointer_provenance(
        self,
        selected_type,
        qualifiers,
        source_location,
        operand_kind,
        *,
        allow_thread_local_pointer_slot=False,
    ):
        value_type = self.metal_source_overload_value_type(selected_type)
        if value_type is None:
            raise MetalAddressProvenanceError(
                operand_kind,
                "the addressed value type is not concrete",
                source_location,
                selected_type,
            )
        pointer_pointee = self.metal_pointer_pointee_type_once(value_type)
        if pointer_pointee is not None:
            pointer_slot_is_portable = (
                allow_thread_local_pointer_slot
                and self.metal_pointer_pointee_type_once(pointer_pointee) is None
                and self.split_outer_metal_declarator_array_type(pointer_pointee)
                is None
                and self.metal_array_type_parts(pointer_pointee) is None
                and not self.is_metal_resource_type(pointer_pointee)
            )
            if pointer_slot_is_portable:
                return f"{value_type}*", self.normalized_metal_address_qualifiers(
                    qualifiers, source_location, value_type
                )
        if (
            pointer_pointee is not None
            or self.split_outer_metal_declarator_array_type(value_type) is not None
            or self.metal_array_type_parts(value_type) is not None
        ):
            raise MetalAddressProvenanceError(
                operand_kind,
                "pointer-to-pointer and pointer-to-array aliases are not portable",
                source_location,
                value_type,
            )
        if self.is_metal_resource_type(value_type):
            raise MetalAddressProvenanceError(
                operand_kind,
                "resource objects cannot be represented as storage pointers",
                source_location,
                value_type,
            )
        return f"{value_type}*", self.normalized_metal_address_qualifiers(
            qualifiers, source_location, value_type
        )

    def normalized_metal_address_qualifiers(
        self, qualifiers, source_location=None, base_type=None
    ):
        qualifier_set = set(qualifiers or ())
        address_spaces = qualifier_set & self.metal_source_overload_address_spaces
        if len(address_spaces) > 1:
            raise MetalAddressProvenanceError(
                "storage expression",
                "the source has conflicting Metal address spaces",
                source_location,
                base_type,
            )
        if not address_spaces:
            qualifier_set.add("thread")
        return tuple(
            qualifier
            for qualifier in self.metal_source_overload_type_qualifiers
            if qualifier in qualifier_set
        )

    def metal_addressable_storage_qualifiers(self, expression):
        if isinstance(expression, (str, VariableNode)):
            name = expression if isinstance(expression, str) else expression.name
            metal_type = self.expression_metal_type(expression)
            if metal_type is None or not name:
                return None
            if (
                name not in self.current_variable_types
                and name not in self.global_variable_types
                and not getattr(expression, "vtype", None)
            ):
                return None
            return self.normalized_metal_address_qualifiers(
                self.expression_metal_type_qualifiers(expression),
                getattr(expression, "source_location", None),
                metal_type,
            )

        if isinstance(expression, ArrayAccessNode):
            selected_type = self.expression_metal_type(expression)
            if self.metal_pointer_pointee_type_once(selected_type) is not None:
                return self.normalized_metal_address_qualifiers(
                    self.expression_metal_type_qualifiers(expression),
                    getattr(expression, "source_location", None),
                    selected_type,
                )
            selection = self.metal_indexed_type_selection(expression)
            if selection["kind"] not in {"array", "pointer"}:
                return None
            return self.metal_addressable_storage_qualifiers(expression.array)

        if isinstance(expression, MemberAccessNode):
            member_type = self.expression_metal_type(expression)
            if self.metal_pointer_pointee_type_once(member_type) is not None:
                return self.normalized_metal_address_qualifiers(
                    self.expression_metal_type_qualifiers(expression),
                    getattr(expression, "source_location", None),
                    member_type,
                )
            object_type = self.expression_metal_type(expression.object)
            if (
                self.metal_vector_component_parts(object_type) is not None
                or self.metal_matrix_type_parts(object_type) is not None
            ):
                return None
            return self.metal_addressable_storage_qualifiers(expression.object)

        if isinstance(expression, BinaryOpNode):
            pointer_source = self.metal_pointer_arithmetic_source(expression)
            if pointer_source is None:
                return None
            return self.metal_addressable_storage_qualifiers(pointer_source[0])

        if isinstance(expression, UnaryOpNode):
            if expression.op == "&":
                provenance = self.metal_address_provenance(expression)
                return provenance[1]
            if expression.op == "*":
                pointer_type = self.expression_metal_type(expression.operand)
                if self.metal_pointer_pointee_type_once(pointer_type) is not None:
                    return self.metal_addressable_storage_qualifiers(expression.operand)
        return None

    def expression_metal_type(self, expr):
        if expr is None:
            return None
        if isinstance(expr, bool):
            return "bool"
        if isinstance(expr, int):
            return "int"
        if isinstance(expr, float):
            return "float"
        if isinstance(expr, str):
            literal_type = self.metal_literal_string_type(expr)
            if literal_type is not None:
                return literal_type
            constructor_member_type = self.current_constructor_member_type(expr)
            if constructor_member_type is not None:
                return constructor_member_type
            tracked_type = self.current_variable_types.get(
                expr,
                self.global_variable_types.get(
                    expr, self.metal_enum_member_types.get(expr)
                ),
            )
            return (
                tracked_type
                or self.metal_standard_math_constant_type(expr)
                or self.static_struct_member_type(expr)
            )
        if isinstance(expr, VariableNode):
            name = getattr(expr, "name", None)
            if not name:
                return None
            if getattr(expr, "vtype", None):
                if self.is_metal_auto_declaration_type(expr.vtype):
                    inferred_type = self.current_variable_types.get(
                        name, self.global_variable_types.get(name)
                    )
                    if inferred_type is not None:
                        return inferred_type
                return self.metal_declaration_expression_type(expr)
            constructor_member_type = self.current_constructor_member_type(name)
            if constructor_member_type is not None:
                return constructor_member_type
            tracked_type = self.current_variable_types.get(
                name,
                self.global_variable_types.get(
                    name, self.metal_enum_member_types.get(name)
                ),
            )
            return (
                tracked_type
                or self.metal_standard_math_constant_type(name)
                or self.static_struct_member_type(name)
            )
        if isinstance(expr, ArrayAccessNode):
            matrix_element = self.metal_cooperative_matrix_element_access(expr)
            if matrix_element is not None:
                matrix, _index = matrix_element
                element_type, _columns, _rows = (
                    self.metal_cooperative_matrix_type_parts(
                        self.expression_metal_type(matrix)
                    )
                )
                return self.resolve_type_alias(element_type)
            selection = self.metal_indexed_type_selection(expr)
            if selection["kind"] == "aggregate":
                raise MetalIndexedComponentTypeResolutionError(
                    selection["base_type"],
                    "aggregate-subscript",
                    "the user-defined aggregate subscript result type cannot be "
                    "inferred",
                    getattr(expr, "source_location", None),
                    self.metal_index_expression_text(expr.index),
                )
            return selection["selected_type"]
        if isinstance(expr, MemberAccessNode):
            object_type = self.expression_metal_type(expr.object)
            if object_type is None:
                return None
            if matrix_columns_object(self, expr) is not None:
                # ``m.columns[i]`` selects the column ``m[i]``.
                return object_type
            wide_vector = self.wide_vector_type_info(
                object_type, getattr(expr, "source_location", None)
            )
            if wide_vector is not None:
                lane = self.wide_vector_lane_index(
                    str(expr.member), wide_vector["width"]
                )
                return wide_vector["element_type"] if lane is not None else None
            vector_member_type = self.metal_small_vector_member_type(
                object_type,
                expr.member,
                getattr(expr, "source_location", None),
            )
            if vector_member_type is not None:
                return vector_member_type
            object_type = self.normalized_metal_type(
                self.resolve_type_alias(object_type)
            )
            member_types = self.struct_member_types.get(object_type)
            if not member_types:
                return None
            return member_types.get(str(expr.member))
        if isinstance(expr, AssignmentNode):
            return self.expression_metal_type(expr.left)
        if isinstance(expr, BinaryOpNode):
            return self.metal_binary_expression_type(expr)
        if isinstance(expr, TernaryOpNode):
            true_type = self.expression_metal_type(expr.true_expr)
            false_type = self.expression_metal_type(expr.false_expr)
            if true_type is None:
                return false_type
            if false_type is None:
                return true_type
            true_descriptor = self.metal_source_overload_type_descriptor(true_type)
            false_descriptor = self.metal_source_overload_type_descriptor(false_type)
            if true_descriptor == false_descriptor:
                return true_type
            if self.binary16_widening_profile is not None:
                return self.metal_scalar_binary_result_type("+", true_type, false_type)
            return None
        if isinstance(expr, PostfixOpNode):
            return self.expression_metal_type(expr.operand)
        if isinstance(expr, UnaryOpNode):
            selected_operator = self.resolve_metal_free_operator(
                expr.op,
                (expr.operand,),
                getattr(expr, "source_location", None),
            )
            if selected_operator is not None:
                return self.resolve_type_alias(
                    getattr(selected_operator, "return_type", None)
                )
            if expr.op == "!":
                return "bool"
            if expr.op == "&":
                provenance = self.metal_address_provenance(expr)
                return provenance[0]
            if expr.op == "*":
                return self.metal_pointer_pointee_type_once(
                    self.expression_metal_type(expr.operand)
                )
            return self.expression_metal_type(expr.operand)
        if isinstance(expr, CastNode):
            return self.resolve_type_alias(expr.target_type)
        if isinstance(expr, VectorConstructorNode):
            return self.resolve_type_alias(expr.type_name)
        if isinstance(expr, TextureSampleNode):
            return self.metal_texture_method_result_type(expr.texture, "sample")
        if isinstance(expr, MethodCallNode):
            return self.metal_texture_method_result_type(expr.object, expr.method)
        if isinstance(expr, FunctionCallNode):
            if str(expr.name) == "sizeof" and len(expr.args) == 1:
                return "size_t"
            if str(expr.name) in {"decltype", "metal::decltype"}:
                if len(expr.args) != 1:
                    return None
                return self.expression_metal_type(expr.args[0])
            numeric_limits_match = re.fullmatch(
                r"(?:metal::)?numeric_limits<(.+)>::"
                r"(?:max|min|lowest|infinity|quiet_NaN|signaling_NaN|epsilon|denorm_min)",
                str(expr.name),
            )
            if numeric_limits_match is not None and not expr.args:
                return self.resolve_type_alias(numeric_limits_match.group(1).strip())
            target_match = re.fullmatch(r"(?:metal::)?as_type<(.+)>", expr.name)
            if target_match is not None:
                return self.resolve_type_alias(target_match.group(1).strip())
            filled_matrix_parts = self.metal_make_filled_matrix_parts(expr)
            if filled_matrix_parts is not None:
                element_type, columns, rows = filled_matrix_parts
                return f"{element_type}{columns}x{rows}"
            constructor_type = self.metal_constructor_result_type(expr.name)
            if constructor_type is not None:
                return constructor_type
            mask_type = vote_mask_call_metal_type(self, expr)
            if mask_type is not None:
                return mask_type
            selected_callable = self.selected_metal_callable(expr)
            if selected_callable is not None:
                return expression_call_return_type(
                    self,
                    selected_callable,
                    expr.args,
                    self.selected_metal_callable_return_type(selected_callable),
                    expr,
                )
            builtin_result_type = self.metal_math_builtin_result_type(expr)
            if builtin_result_type is not None:
                return builtin_result_type
            atomic_name = str(expr.name).lstrip(":")
            if atomic_name.startswith("metal::"):
                atomic_name = atomic_name[len("metal::") :]
            if (
                atomic_name in {
                    "atomic_load_explicit",
                    "atomic_fetch_sub_explicit",
                    *self.metal_atomic_intrinsics,
                }
                and len(expr.args)
                == (2 if atomic_name == "atomic_load_explicit" else 3)
                and not self.metal_user_function_overloads(expr.name)
            ):
                pointee = self.metal_pointer_pointee_type_once(
                    self.expression_metal_type(expr.args[0])
                )
                atomic_result = self.atomic_element_type(
                    self.normalized_metal_type(self.resolve_type_alias(pointee))
                )
                if atomic_result is not None:
                    return atomic_result
            unscoped_name = str(expr.name).rsplit("::", 1)[-1]
            if (
                unscoped_name in self.metal_wave_intrinsics
                and expr.args
                and self.map_metal_wave_function_name(expr.name, expr.args) is not None
            ):
                # The call lowers to the builtin, which no user overload of
                # its name matches: it has the builtin's type, not the type of
                # a user overload of the same arity.
                return self.expression_metal_type(expr.args[0])
            arity_candidates = same_arity_overloads(self, expr)
            if len(arity_candidates) == 1:
                return expression_call_return_type(
                    self,
                    arity_candidates[0],
                    expr.args,
                    getattr(arity_candidates[0], "return_type", None),
                    expr,
                )
            if (
                unscoped_name in self.metal_wave_intrinsics
                or unscoped_name == "simd_shuffle_and_fill_up"
            ) and expr.args:
                return self.expression_metal_type(expr.args[0])
        return None

    def metal_depth_texel_component(self, texture_expr, call):
        """Select the depth of a CrossGL texel read on an MSL depth texture.

        ``depthNd<T>::sample`` and ``read`` return the scalar ``T`` (MSL 3.2 section 6.12.7 onwards), while the
        CrossGL ``texture``/``texelFetch`` family returns ``vec4(depth, 0, 0, 1)`` for every texture, so the
        import keeps the source type by reading component ``x``.
        """
        texture_type = self.expression_metal_type(texture_expr)
        if not texture_type:
            return call
        base_name, _generic_args = self.generic_type_parts(
            self.resolve_type_alias(texture_type)
        )
        if is_depth_texture_name(base_name):
            return f"{call}.x"
        return call

    def storage_image_write_value(self, texture_expr, value_expr, value):
        """Spell the implicit conversion of a texture ``write`` value to the texel type.

        ``write(Tv color, ...)`` takes the texel type of the texture (MSL 3.2 section 6.12.16); a scalar argument
        converts to it like any scalar-to-vector initialisation (MSL 3.2 section 2.12), so ``tex.write(0, c)``
        stores ``float4(0.0)`` in a ``texture2d<float, access::write>``.
        """
        texel_type = self.metal_texture_method_result_type(texture_expr, "read")
        value_type = self.expression_metal_type(value_expr)
        if not texel_type or not value_type:
            return value
        value_type = self.metal_source_overload_value_type(value_type) or value_type
        if self.metal_scalar_arithmetic_type_info(value_type) is None:
            # Only a scalar converts implicitly; a vector value already has the texel type.
            return value
        return f"{self.map_type(self.normalized_metal_type(texel_type))}({value})"

    def metal_texture_method_result_type(self, texture_expr, method):
        """The MSL type of ``texture.method(...)`` when the texture's type fixes it, else ``None``."""
        texture_type = self.expression_metal_type(texture_expr)
        if not texture_type:
            return None
        base_name, generic_args = self.generic_type_parts(
            self.resolve_type_alias(texture_type)
        )
        if base_name is None:
            return None
        shape = texture_method_result_shape(base_name, generic_args, method)
        if shape is None:
            return None
        element_type, width = shape
        element_type = self.resolve_type_alias(element_type)
        if width == 1:
            return element_type
        return self.metal_vector_type_from_element(element_type, width)

    def metal_make_filled_matrix_template(self, expr):
        """The template arguments of a call to the MSL ``make_filled_matrix``, else ``None``.

        A user function of the same name keeps its own meaning.
        """
        if not isinstance(expr, FunctionCallNode) or len(expr.args) != 1:
            return None
        match = re.fullmatch(
            r"(?:metal::)?make_filled_matrix<(.*)>", str(expr.name).lstrip(":").strip()
        )
        if match is None:
            return None
        bare_name = str(expr.name).split("<", 1)[0].strip()
        if self.metal_user_function_overloads(bare_name):
            return None
        return match.group(1)

    def metal_make_filled_matrix_parts(self, expr):
        """``(T, Cols, Rows)`` of a ``make_filled_matrix<T, Cols[, Rows]>(value)`` call, else ``None``.

        MSL 4.1 section 6.7 declares ``template <typename T, int Cols, int Rows = Cols, typename U>
        matrix<T, Cols, Rows> make_filled_matrix(U value)``, which sets every element to ``T(value)``.
        Only the matrices CrossGL can spell (``float`` or ``half`` elements, 2 to 4 literal columns and rows)
        have parts.
        """
        template = self.metal_make_filled_matrix_template(expr)
        if template is None:
            return None
        match = re.fullmatch(r"\s*([^,<>]+?)\s*,\s*(\d+)\s*(?:,\s*(\d+)\s*)?", template)
        if match is None:
            return None
        element_type = self.normalized_metal_type(
            self.resolve_type_alias(match.group(1).strip())
        )
        if element_type not in {"float", "half"}:
            return None
        columns = int(match.group(2))
        rows = int(match.group(3)) if match.group(3) is not None else columns
        if not (2 <= columns <= 4 and 2 <= rows <= 4):
            return None
        return element_type, columns, rows

    def generate_metal_make_filled_matrix(self, expr, is_main=False):
        """Lower ``make_filled_matrix<T, Cols, Rows>(value)`` to a CrossGL matrix with every element ``T(value)``.

        CrossGL has no such function, and a one-argument matrix constructor sets only the diagonal, so each column
        is the ``vec<T, Rows>`` splat of the value. A value that might have side effects is evaluated once, by
        scaling a matrix of ones, instead of once per column. A matrix CrossGL cannot spell is refused.
        """
        template = self.metal_make_filled_matrix_template(expr)
        if template is None:
            return None
        parts = self.metal_make_filled_matrix_parts(expr)
        if parts is None:
            return self.refuse_import(
                f"Metal make_filled_matrix<{template.strip()}>: CrossGL matrices have float or half elements "
                "and 2 to 4 columns and rows given as integer literals",
                "0.0",
            )
        element_type, columns, rows = parts
        matrix_type = self.map_type(f"{element_type}{columns}x{rows}")
        column_type = self.map_type(f"{element_type}{rows}")
        value = self.generate_expression(expr.args[0], is_main)
        if self.discarded_expression_is_proven_side_effect_free(expr.args[0]):
            column = f"{column_type}({value})"
            return f"{matrix_type}({', '.join([column] * columns)})"
        ones = ", ".join([f"{column_type}(1.0)"] * columns)
        scalar_type = self.map_type(element_type)
        return f"({matrix_type}({ones}) * {scalar_type}({value}))"

    def metal_constructor_result_type(self, name):
        type_name = self.normalized_metal_type(self.resolve_type_alias(str(name)))
        if not type_name:
            return None
        if (
            type_name in self.type_map
            or self.metal_vector_type_parts(type_name) is not None
            or type_name in self.struct_member_types
        ):
            return type_name
        return None

    def expression_mapped_type(self, expr):
        metal_type = self.expression_metal_type(expr)
        storage_type = self.map_storage_texture_type(metal_type)
        if storage_type:
            return storage_type
        return (
            self.map_type(self.resolve_type_alias(metal_type)) if metal_type else None
        )

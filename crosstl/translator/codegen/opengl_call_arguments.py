"""Call arguments in the GLSL generator: source expression types, buffer blocks and atomics, structured buffer arguments, texture access and scalar expression kinds."""

import re

from ..ast import (
    ArrayAccessNode,
    BinaryOpNode,
    ConstructorNode,
    FunctionCallNode,
    IdentifierNode,
    MemberAccessNode,
    PointerAccessNode,
    RayTracingOpNode,
    TernaryOpNode,
    UnaryOpNode,
    VariableNode,
)
from ..matrix_shape import (
    crossgl_linear_algebra_product_type,
    crossgl_matrix_componentwise_type,
)
from ..refusals import OpenGLRefusal
from ..validation import (
    IMAGE_RESOURCE_INTRINSIC_NAMES,
    expression_debug_name,
)
from .array_utils import (
    evaluate_literal_int_expression,
    parse_array_type,
    split_array_type_suffix,
)
from .builtin_result_types import (
    GLSL_FAMILY,
    builtin_call_result_type,
    component_access_type,
)
from .for_in_utils import validate_range_reference_argument
from .generic_function_utils import generic_function_call_name
from .glsl_buffer_layout import glsl_buffer_block_node_type
from .glsl_shared_array_values import (
    allow_shared_array_value,
    shared_array_value_root,
)
from .glsl_word_views import glsl_word_view_integer_storage
from .image_access_contracts import (
    image_access_diagnostic_name,
    image_access_requirement_label,
    image_access_satisfies_requirement,
)
from .image_access_contracts import (
    image_atomic_value_arguments as shared_image_atomic_value_arguments,
)
from .image_access_contracts import (
    is_image_atomic_operation,
    is_image_resource_operation,
    numeric_scalar_expression_kind,
)
from .opengl_values import (
    OpenGLPrivatePointerParameterError,
    OpenGLReferenceParameterError,
    OpenGLWorkgroupPointerError,
)


class OpenGLCallArgumentMixin:
    """Call argument members of ``GLSLCodeGen``."""

    def glsl_source_expression_type(self, expression):
        if expression is None:
            return None
        if isinstance(expression, RayTracingOpNode) and expression.operation in {
            "ReportHit",
            "ReportIntersection",
        }:
            return "bool"  # reportIntersectionEXT returns whether the hit was accepted
        emulated_matrix_type = self.glsl_emulated_matrix_expression_type(expression)
        if emulated_matrix_type is not None:
            return emulated_matrix_type
        if isinstance(expression, VariableNode) or (
            hasattr(expression, "__class__")
            and "IdentifierNode" in str(expression.__class__)
        ):
            name = getattr(expression, "name", None)
            pointer_array = self.glsl_storage_pointer_aliases().get(name)
            if pointer_array is not None and pointer_array.get("kind") == (
                "storage-pointer-array"
            ):
                source_element_type = pointer_array.get(
                    "source_element_type"
                ) or pointer_array.get("element_type")
                extent = pointer_array.get("extent")
                if source_element_type and isinstance(extent, int) and extent > 0:
                    return f"{self.type_name_string(source_element_type)}*[{extent}]"
            return (
                self.local_variable_source_types.get(name)
                or self.local_variable_types.get(name)
                or self.glsl_constant_types.get(name)
                or self.glsl_buffer_block_variable_types.get(name)
                or self.global_variable_types.get(name)
                or self.glsl_builtin_identifier_result_type(name)
            )
        if isinstance(expression, ArrayAccessNode):
            pointer_view_plan = self.glsl_fixed_array_vector_pointer_view_access_plan(
                expression
            )
            if pointer_view_plan is not None:
                return pointer_view_plan["scalar_type"]
            # Resource specialization removes a fixed storage-pointer-array
            # parameter from the emitted helper signature and represents it as
            # scalar logical offsets.  Its per-name type metadata consequently
            # resembles a scalar pointer, but one subscript still selects a
            # pointer-array element (not the final pointee value).  Preserve
            # that source distinction for overloads such as
            # ``advance(inputs[0])``.
            array_expression = expression.array
            array_name = (
                self.expression_name(array_expression)
                if isinstance(array_expression, (str, IdentifierNode, VariableNode))
                else None
            )
            pointer_array = self.glsl_storage_pointer_aliases().get(array_name)
            if pointer_array is not None and pointer_array.get("kind") == (
                "storage-pointer-array"
            ):
                source_element_type = pointer_array.get(
                    "source_element_type"
                ) or pointer_array.get("element_type")
                source_element_name = self.type_name_string(source_element_type)
                if source_element_name:
                    return f"{source_element_name}*"

            array_type = self.glsl_source_expression_type(expression.array)
            type_name = self.type_name_string(array_type)
            if not type_name:
                return None
            outer_array = self.glsl_outer_array_type(type_name)
            if outer_array is not None:
                _extent, element_type = outer_array
                return element_type
            if type_name.rstrip().endswith(("*", "&")):
                return type_name.rstrip()[:-1].strip()
            if self.is_structured_buffer_type(type_name):
                element_type = self.structured_buffer_source_element_type(type_name)
                if element_type is not None:
                    return element_type
            half_width = self.glsl_half_width(type_name)
            if half_width is not None and half_width > 1:
                return "half"
            bfloat_width = self.glsl_bfloat_width(type_name)
            if bfloat_width is not None and bfloat_width > 1:
                return "bfloat"
            narrow = self.glsl_narrow_integer_contract(type_name)
            if narrow is not None and narrow["width"] > 1:
                return ("int" if narrow["signed"] else "uint") + str(narrow["bits"])
            # A vector subscript selects a component and a matrix subscript a column (GLSL 4.60 section 5.6).
            component_type = self.vector_component_type(
                type_name
            ) or component_access_type(type_name, GLSL_FAMILY)
            if component_type is not None:
                return component_type
            vector_info = self.glsl_value_type_info(type_name)
            if vector_info is not None and vector_info["width"] > 1:
                return self.vector_component_type(type_name) or type_name
            matrix_dimensions = self.glsl_matrix_dimensions(self.map_type(type_name))
            if matrix_dimensions is not None:
                # A subscript selects a column; non-square columns have the
                # row count, not the column count, of components.
                return matrix_dimensions[3]
            return type_name
        if isinstance(expression, BinaryOpNode):
            operator = self.map_operator(
                getattr(expression, "op", getattr(expression, "operator", None))
            )
            if operator == ",":
                return self.glsl_source_expression_type(expression.right)
            if operator in {"==", "!=", "<", "<=", ">", ">=", "&&", "||"}:
                left_type = self.glsl_source_expression_type(expression.left)
                right_type = self.glsl_source_expression_type(expression.right)
                vector_type = self.glsl_relational_vector_type(left_type, right_type)
                return self.glsl_bool_vector_type(vector_type) or "bool"
            left_type = self.glsl_source_expression_type(expression.left)
            right_type = self.glsl_source_expression_type(expression.right)
            if self.is_glsl_complex64_type(left_type) or self.is_glsl_complex64_type(
                right_type
            ):
                return "complex64_t"
            if left_type is None:
                return right_type
            if right_type is None:
                return left_type
            if operator == "*":
                product_type = crossgl_linear_algebra_product_type(
                    self.map_type(left_type), self.map_type(right_type)
                )
                if product_type is not None:
                    return product_type
            matrix_type = crossgl_matrix_componentwise_type(
                operator, self.map_type(left_type), self.map_type(right_type)
            )
            if matrix_type is not None:
                return matrix_type
            arithmetic_type = self.glsl_common_arithmetic_type(
                left_type, right_type, operator
            )
            if arithmetic_type is not None:
                return arithmetic_type
            if self.glsl_source_type_signature(
                left_type
            ) == self.glsl_source_type_signature(right_type):
                return left_type
            left_mapped = self.map_type(left_type)
            right_mapped = self.map_type(right_type)
            if left_mapped == "float" and right_mapped != "float":
                return left_type
            if right_mapped == "float" and left_mapped != "float":
                return right_type
            return left_type
        if isinstance(expression, UnaryOpNode):
            return self.glsl_unary_source_result_type(expression)
        if isinstance(expression, TernaryOpNode):
            true_type = self.glsl_source_expression_type(expression.true_expr)
            false_type = self.glsl_source_expression_type(expression.false_expr)
            if true_type is None:
                return false_type
            if false_type is None:
                return true_type
            if self.glsl_source_type_signature(
                true_type
            ) == self.glsl_source_type_signature(false_type):
                return true_type
            return self.glsl_common_arithmetic_type(true_type, false_type, "?:")
        if isinstance(expression, MemberAccessNode):
            object_type = self.glsl_source_expression_type(expression.object)
            object_name = self.type_name_string(object_type)
            if object_name:
                member_type = self.struct_member_types.get(object_name, {}).get(
                    str(expression.member)
                )
                if member_type is not None:
                    return member_type
            half_width = self.glsl_half_width(object_type)
            bfloat_width = self.glsl_bfloat_width(object_type)
            member = str(expression.member)
            narrow = self.glsl_narrow_integer_contract(object_type)
            if narrow is not None and narrow["width"] > 1 and 1 <= len(member) <= 4:
                for components in ("xyzw", "rgba", "stpq"):
                    if all(
                        component in components[: narrow["width"]]
                        for component in member
                    ):
                        if len(member) == 1:
                            return ("int" if narrow["signed"] else "uint") + str(
                                narrow["bits"]
                            )
                        return (
                            ("i" if narrow["signed"] else "u")
                            + str(narrow["bits"])
                            + f"vec{len(member)}"
                        )
            narrow_width = half_width or bfloat_width
            if narrow_width is not None and narrow_width > 1 and 1 <= len(member) <= 4:
                for components in ("xyzw", "rgba", "stpq"):
                    if all(
                        component in components[:narrow_width] for component in member
                    ):
                        return ("half" if half_width else "bfloat") + (
                            str(len(member)) if len(member) > 1 else ""
                        )
            return self.expression_result_type(expression)
        if isinstance(expression, ConstructorNode):
            return self.type_name_string(getattr(expression, "constructor_type", None))
        if isinstance(expression, FunctionCallNode):
            function_name = self.function_call_name(expression)
            bitcast_type = self.metal_as_type_target(function_name)
            if (
                bitcast_type is not None
                and function_name not in self.function_return_types
            ):
                return bitcast_type
            if self.glsl_constructor_type(function_name):
                return function_name
            logical_type = self.glsl_hlsl_logical_result_type(
                function_name, expression.arguments
            )
            if logical_type is not None:
                return logical_type
            unified_type = self.glsl_unified_builtin_result_type(
                function_name, expression.arguments
            )
            if unified_type is not None:
                return unified_type
            buffer_result_type = self.glsl_buffer_call_result_type(
                function_name,
                expression.arguments,
                source_type=True,
            )
            if buffer_result_type is not None:
                return buffer_result_type
            simd_result_type = self.glsl_plain_simd_call_result_type(
                function_name, expression.arguments
            )
            if simd_result_type is not None:
                return simd_result_type
            specialized_name = generic_function_call_name(
                self,
                function_name,
                expression.arguments,
            )
            if specialized_name in self.function_return_types:
                return self.function_return_types[specialized_name]
            resolved = self.resolve_glsl_function_overload(
                function_name,
                expression.arguments,
                call_node=expression,
            )
            if resolved is not None:
                return self.type_name_string(getattr(resolved, "return_type", None))
            componentwise_type = self.glsl_componentwise_unary_result_type(
                function_name, expression.arguments
            )
            if componentwise_type is not None:
                return componentwise_type
            signbit_result_type = self.glsl_signbit_result_type(
                function_name,
                expression.arguments,
                source_type=True,
            )
            if signbit_result_type is not None:
                return signbit_result_type
            isfinite_result_type = self.glsl_isfinite_result_type(
                function_name,
                expression.arguments,
                source_type=True,
            )
            if isfinite_result_type is not None:
                return isfinite_result_type
            componentwise_boolean_type = self.glsl_componentwise_boolean_result_type(
                function_name,
                expression.arguments,
                source_type=True,
            )
            if componentwise_boolean_type is not None:
                return componentwise_boolean_type
            boolean_reduction_type = self.glsl_boolean_reduction_result_type(
                function_name, expression.arguments
            )
            if boolean_reduction_type is not None:
                return boolean_reduction_type
            if function_name not in self.function_return_types:
                # Typed by its operands' source types, so ``transpose(W)`` of a uniform or built-in matrix is typed.
                builtin_type = builtin_call_result_type(
                    function_name,
                    [
                        self.type_name_string(
                            self.glsl_source_expression_type(argument)
                        )
                        for argument in expression.arguments or []
                    ],
                    GLSL_FAMILY,
                )
                if builtin_type is not None:
                    return builtin_type
        return self.expression_result_type(expression)

    def glsl_addressed_element_type_match_score(
        self,
        actual_type,
        expected_type,
    ):
        if actual_type is None or expected_type is None:
            return None
        if self.glsl_source_type_signature(
            actual_type
        ) != self.glsl_source_type_signature(expected_type):
            return None
        return 12

    def glsl_address_argument_match_score(self, argument, parameter):
        """Match ``&value`` without treating it as the scalar value itself."""

        operand_type = self.glsl_source_expression_type(argument.operand)
        if operand_type is None:
            return None
        parameter_type = getattr(
            parameter,
            "param_type",
            getattr(parameter, "vtype", None),
        )

        pointer_array_contract = self.glsl_storage_pointer_array_contract(parameter)
        if pointer_array_contract is not None:
            if pointer_array_contract.get("extent") != 1:
                return None
            operand_pointee = self.glsl_source_pointee_type(operand_type)
            return self.glsl_addressed_element_type_match_score(
                operand_pointee,
                pointer_array_contract.get("source_element_type"),
            )

        fixed_array_shape = self.glsl_fixed_array_parameter_shape(parameter)
        if fixed_array_shape is not None:
            if fixed_array_shape.get("extent") != 1:
                return None
            operand_name = self.type_name_string(operand_type)
            _operand_element, operand_extent = parse_array_type(operand_name)
            if (
                operand_extent is not None
                or self.glsl_source_pointee_type(operand_type) is not None
            ):
                return None
            return self.glsl_addressed_element_type_match_score(
                operand_type,
                fixed_array_shape.get("source_element_type"),
            )

        parameter_pointee = self.glsl_source_pointee_type(parameter_type)
        if parameter_pointee is not None:
            return self.glsl_addressed_element_type_match_score(
                operand_type,
                parameter_pointee,
            )
        return None

    def glsl_function_argument_match_score(
        self,
        argument,
        actual_type,
        parameter,
    ):
        if isinstance(argument, UnaryOpNode) and self.map_operator(argument.op) == "&":
            return self.glsl_address_argument_match_score(argument, parameter)
        parameter_type = getattr(
            parameter,
            "param_type",
            getattr(parameter, "vtype", None),
        )
        return self.glsl_function_type_match_score(actual_type, parameter_type)

    def glsl_function_type_match_score(self, actual_type, parameter_type):
        if actual_type is None or parameter_type is None:
            return None
        actual_signature = self.glsl_source_type_signature(actual_type)
        parameter_signature = self.glsl_source_type_signature(parameter_type)
        if actual_signature == parameter_signature:
            return 8
        actual = self.glsl_value_type_info(actual_type)
        parameter = self.glsl_value_type_info(parameter_type)
        if actual is not None and parameter is not None:
            if (
                actual["width"] == parameter["width"] == 1
                and actual["family"] in {"bool", "int", "uint"}
                and parameter["family"] in {"int", "uint"}
                and self.glsl_promoted_integer_type(actual) == parameter["mapped"]
                and parameter["narrow"] is None
            ):
                return 6
        actual_mapped = self.map_type(actual_type)
        parameter_mapped = self.map_type(parameter_type)
        if actual_mapped == parameter_mapped:
            return 2
        if self.is_scalar_value_type(actual_type) and self.is_scalar_value_type(
            parameter_type
        ):
            return 1
        return None

    def glsl_function_candidate_signature(self, function):
        parameters = ", ".join(self.glsl_source_function_signature(function))
        return_type = self.glsl_source_type_signature(
            getattr(function, "return_type", None)
        )
        return f"{getattr(function, 'name', '')}({parameters}) -> {return_type}"

    def function_call_argument_pairs(self, func_name, call_args, parameter_infos=None):
        if parameter_infos is None:
            parameter_infos = list(self.function_parameter_infos.get(func_name, []))
            if not parameter_infos:
                parameter_infos = [
                    (None, param_type, ())
                    for param_type in self.function_parameter_types.get(func_name, [])
                ]
        else:
            parameter_infos = list(parameter_infos)
        skipped_indices = self.skipped_function_parameter_indices(func_name)
        filtered_parameter_infos = [
            param_info
            for index, param_info in enumerate(parameter_infos)
            if index not in skipped_indices
        ]
        pairs = []
        for index, arg in enumerate(call_args):
            param_name, param_type, param_qualifiers = (
                filtered_parameter_infos[index]
                if index < len(filtered_parameter_infos)
                else (None, None, ())
            )
            pairs.append((arg, param_name, param_type, param_qualifiers))
        return pairs

    def generate_function_call_arguments(
        self, func_name, call_args, parameter_infos=None
    ):
        args = []
        for (
            arg,
            param_name,
            param_type,
            param_qualifiers,
        ) in self.function_call_argument_pairs(
            func_name, call_args, parameter_infos=parameter_infos
        ):
            generated = self.generate_function_call_argument(
                func_name, arg, param_name, param_type, param_qualifiers
            )
            if isinstance(generated, list):
                args.extend(generated)
            else:
                args.append(generated)
        return args

    def generate_function_call_argument(
        self, func_name, arg, param_name, param_type, param_qualifiers=()
    ):
        self.union_storage.validate_argument(arg, param_type, param_qualifiers)
        validate_range_reference_argument(self, arg, param_type, param_qualifiers)
        private_binding = self.glsl_private_pointer_call_argument_binding(
            func_name,
            param_name,
            arg,
        )
        if private_binding is not None:
            if private_binding.get("root_kind") in {
                "scalar",
                "scalar_parameter",
            }:
                if private_binding.get("offset") not in {
                    None,
                    "",
                    "0",
                    "0u",
                    "int(0)",
                    "int(0u)",
                }:
                    display_name = self.glsl_private_pointer_display_name(func_name)
                    raise OpenGLPrivatePointerParameterError(
                        "OpenGL scalar private pointer argument for "
                        f"'{display_name}.{param_name}' has a nonzero offset",
                        function_name=display_name,
                        parameter_name=param_name,
                        reason="nonzero-scalar-view-offset",
                        source_location=getattr(arg, "source_location", None),
                    )
                return private_binding["backing"]
            return [private_binding["backing"], private_binding["offset"]]
        if (
            isinstance(arg, UnaryOpNode)
            and self.map_operator(arg.op) == "&"
            and isinstance(arg.operand, ArrayAccessNode)
        ):
            _element_type, extent_expression = parse_array_type(
                self.type_name_string(param_type)
            )
            extent = evaluate_literal_int_expression(
                extent_expression,
                self.glsl_active_literal_int_constants(),
            )
            if extent == 1:
                raise OpenGLPrivatePointerParameterError(
                    "OpenGL cannot materialize the address of one fixed-array "
                    "element as a writable GLSL array parameter",
                    function_name=func_name,
                    parameter_name=param_name,
                    reason="fixed-array-addressed-element-view-unsupported",
                    source_location=getattr(arg, "source_location", None),
                )
        buffer_array = self.structured_buffer_array_parameter_info(
            param_type, param_name, func_name
        )
        if buffer_array is not None:
            generated_args = self.structured_buffer_array_data_arguments(
                arg, buffer_array["count"]
            )
            if self.structured_buffer_requires_counter(buffer_array["base_type"]):
                generated_args += self.structured_buffer_array_counter_arguments(
                    arg, buffer_array["count"]
                )
            return generated_args
        if (
            param_type is not None
            and self.is_structured_buffer_type(param_type)
            and not self.is_array_type(param_type)
        ):
            data_arg = self.structured_buffer_data_argument(arg)
            if self.structured_buffer_requires_counter(param_type):
                return [data_arg, self.structured_buffer_counter_data_argument(arg)]
            return data_arg
        parameter_qualifiers = set(param_qualifiers)
        if parameter_qualifiers.intersection({"out", "inout"}):
            self.float_atomic_storage.validate_reference(arg)
            if (
                "inout" in parameter_qualifiers
                and self.glsl_reference_parameter_type_name(param_type)
                and not self.glsl_reference_argument_is_lvalue(arg)
            ):
                argument_name = expression_debug_name(arg)
                raise OpenGLReferenceParameterError(
                    "OpenGL mutable reference parameter "
                    f"'{func_name}.{param_name or '<unnamed>'}' requires an "
                    f"assignable argument; received '{argument_name}'",
                    function_name=func_name,
                    parameter_name=param_name,
                    argument=argument_name,
                    reason="mutable-reference-requires-lvalue",
                    source_location=getattr(arg, "source_location", None),
                )
            with allow_shared_array_value(
                self, shared_array_value_root(self, arg, param_type)
            ):
                return self.generate_expression(arg)
        generated = self.glsl_wide_integer_literal_argument(
            arg,
            self.generate_expression_with_expected(arg, param_type),
            param_type,
        )
        return self.glsl_precise_argument(arg, generated, param_type)

    def glsl_wide_integer_literal_argument(self, argument, generated, param_type):
        """Convert an integer literal argument of a 64-bit parameter explicitly.

        Assignments and initializers accept a 32-bit literal for a 64-bit
        integer, but overload resolution does not: Mesa rejects a call such as
        ``f(0)`` to ``void f(uint64_t offset)`` because it applies no implicit
        conversion from ``int`` to a 64-bit integer.
        """
        if (
            param_type is None
            or not isinstance(generated, str)
            or not self.is_numeric_literal_expression(argument)
        ):
            return generated
        expected = self.glsl_value_type_info(self.type_name_string(param_type))
        if (
            expected is None
            or expected["family"] not in {"int", "uint"}
            or expected["bits"] != 64
            or expected["width"] != 1
            or expected["narrow"] is not None
            or self.glsl_is_explicit_type_constructor_expression(
                generated, expected["mapped"]
            )
        ):
            return generated
        return f"{expected['mapped']}({generated})"

    def glsl_reference_parameter_type_name(self, param_type):
        type_name = self.type_name_string(param_type)
        return bool(type_name and type_name.rstrip().endswith("&"))

    def glsl_reference_argument_is_lvalue(self, argument):
        if isinstance(argument, str):
            return bool(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", argument))
        if isinstance(argument, (IdentifierNode, VariableNode)):
            return bool(getattr(argument, "name", None))
        if isinstance(argument, (ArrayAccessNode, MemberAccessNode, PointerAccessNode)):
            return True
        return isinstance(argument, UnaryOpNode) and argument.op == "*"

    def structured_buffer_array_data_arguments(self, arg, count):
        arg_name = self.expression_name(arg)
        expanded = self.current_structured_buffer_array_parameters.get(arg_name)
        if expanded is not None:
            expanded_names = expanded["expanded_names"]
            return expanded_names[:count]

        arg_code = self.generate_expression(arg)
        instance_member = self.structured_buffer_instance_members.get(arg_name)
        if instance_member:
            return [f"{arg_code}[{index}].{instance_member}" for index in range(count)]
        return [f"{arg_code}[{index}]" for index in range(count)]

    def structured_buffer_data_argument(self, arg):
        arg_code = self.generate_expression(arg)
        buffer_name = self.expression_name(arg)
        instance_member = self.structured_buffer_instance_members.get(buffer_name)
        if instance_member:
            return f"{arg_code}.{instance_member}"
        return arg_code

    def structured_buffer_array_counter_arguments(self, arg, count):
        arg_name = self.expression_name(arg)
        expanded = self.current_structured_buffer_array_parameters.get(arg_name)
        if expanded is not None:
            counter_names = expanded.get("counter_expanded_names", [])
            return counter_names[:count]

        instance_name = self.structured_buffer_counter_instances.get(arg_name)
        member_name = self.structured_buffer_counter_members.get(arg_name)
        if instance_name and member_name:
            return [f"{instance_name}[{index}].{member_name}" for index in range(count)]
        return [self.structured_buffer_counter_data_argument(arg)] * count

    def structured_buffer_counter_data_argument(self, arg):
        array_access = self.structured_buffer_array_parameter_access(arg)
        if array_access is not None:
            info, selector = array_access
            counter_names = info.get("counter_expanded_names")
            if counter_names:
                counter_info = {**info, "expanded_names": counter_names}
                return self.structured_buffer_array_select_expression(
                    counter_info,
                    selector,
                    lambda name: name,
                )

        arg_name = self.expression_name(arg)
        counter_parameter = self.current_structured_buffer_counter_parameters.get(
            arg_name
        )
        if counter_parameter is not None:
            return counter_parameter

        if isinstance(arg, ArrayAccessNode) or (
            hasattr(arg, "__class__") and "ArrayAccess" in str(arg.__class__)
        ):
            array_expr = getattr(arg, "array", getattr(arg, "array_expr", None))
            index_expr = getattr(arg, "index", getattr(arg, "index_expr", None))
            array_name = self.expression_name(array_expr)
            instance_name = self.structured_buffer_counter_instances.get(array_name)
            member_name = self.structured_buffer_counter_members.get(array_name)
            if instance_name and member_name:
                index = self.glsl_index_expression(index_expr, array_expr)
                return f"{instance_name}[{index}].{member_name}"

        member_name = self.structured_buffer_counter_members.get(arg_name)
        if member_name and arg_name not in self.structured_buffer_counter_instances:
            return member_name
        return None

    def structured_buffer_array_parameter_access(self, buffer_arg):
        if not isinstance(buffer_arg, ArrayAccessNode) and not (
            hasattr(buffer_arg, "__class__")
            and "ArrayAccess" in str(buffer_arg.__class__)
        ):
            return None
        array_expr = getattr(
            buffer_arg, "array", getattr(buffer_arg, "array_expr", None)
        )
        array_name = self.expression_name(array_expr)
        info = self.current_structured_buffer_array_parameters.get(array_name)
        if info is None:
            return None
        selector = getattr(buffer_arg, "index", getattr(buffer_arg, "index_expr", None))
        return info, selector

    def structured_buffer_array_select_expression(self, info, selector, branch):
        expanded_names = info["expanded_names"]
        literal_index = self.literal_int_value(selector, self.literal_int_constants)
        if literal_index is not None and 0 <= literal_index < len(expanded_names):
            return branch(expanded_names[literal_index])

        selector_code = self.glsl_index_expression(
            selector,
            info.get("source_name"),
            indexed_extent=len(expanded_names),
        )
        selector_type = self.map_type(self.expression_result_type(selector))

        def selector_literal(index):
            return f"{index}u" if selector_type == "uint" else str(index)

        expression = branch(expanded_names[-1])
        for index in range(len(expanded_names) - 2, -1, -1):
            expression = (
                f"(({selector_code} == {selector_literal(index)}) ? "
                f"{branch(expanded_names[index])} : {expression})"
            )
        return expression

    def is_array_type(self, vtype):
        if hasattr(vtype, "element_type") and str(type(vtype)).find("ArrayType") != -1:
            return True
        if hasattr(vtype, "name") or hasattr(vtype, "element_type"):
            return False
        type_string = str(vtype)
        return "[" in type_string and "]" in type_string

    def is_explicit_sampler_argument(self, args):
        if len(args) < 3:
            return False
        return self.texture_call_uses_explicit_sampler(args)

    def texture_call_uses_explicit_sampler(self, args):
        if len(args) < 2:
            return False
        sampler_name = self.expression_name(args[1]) or self.generate_expression(
            args[1]
        )
        if (
            sampler_name in self.sampler_variables
            or sampler_name in self.current_sampler_parameters
        ):
            return True
        arg_type = self.expression_result_type(args[1])
        return arg_type is not None and self.is_sampler_type(arg_type)

    def texture_resource_type(self, texture_arg):
        return self.sampled_image_type(self.texture_declared_resource_type(texture_arg))

    def texture_declared_resource_type(self, texture_arg):
        texture_name = self.expression_name(texture_arg)
        if not texture_name:
            return None
        alias = self.current_resource_aliases.get(texture_name)
        if alias is not None:
            return alias.get("type")
        return self.current_texture_parameters.get(
            texture_name, self.texture_variable_types.get(texture_name)
        )

    def sampled_image_type(self, texture_type):
        base_type = self.resource_base_type(texture_type)
        texture_to_sampler = {
            "texture1D": "sampler1D",
            "texture1DArray": "sampler1DArray",
            "texture2D": "sampler2D",
            "texture3D": "sampler3D",
            "textureCube": "samplerCube",
            "texture2DArray": "sampler2DArray",
            "textureCubeArray": "samplerCubeArray",
        }
        sampled_type = texture_to_sampler.get(base_type)
        if sampled_type is None:
            return texture_type
        if "[" not in str(texture_type):
            return sampled_type
        _, array_suffix = split_array_type_suffix(str(texture_type))
        return f"{sampled_type}{array_suffix}"

    def is_separate_texture_object_type(self, texture_type):
        return self.resource_base_type(texture_type) in {
            "texture1D",
            "texture1DArray",
            "texture2D",
            "texture3D",
            "textureCube",
            "texture2DArray",
            "textureCubeArray",
        }

    def texture_argument_resource_type(self, texture_arg):
        texture_type = self.texture_resource_type(texture_arg)
        if texture_type is not None:
            return texture_type
        return self.expression_result_type(texture_arg)

    def validate_texture_resource_argument(self, func_name, args):
        if not args or func_name not in self.texture_resource_operation_names():
            return
        if self.texture_resource_type(args[0]) is not None:
            return
        arg_type = self.expression_result_type(args[0])
        if arg_type is not None and self.is_inferable_resource_array_type(arg_type):
            return

        texture_name = self.expression_name(args[0]) or str(args[0])
        raise OpenGLRefusal(
            f"OpenGL texture operation '{func_name}' requires a declared "
            f"texture or image resource argument: {texture_name}"
        )

    def validate_image_resource_argument(self, func_name, args):
        if not args or not is_image_resource_operation(
            func_name, IMAGE_RESOURCE_INTRINSIC_NAMES
        ):
            return
        texture_type = self.texture_argument_resource_type(args[0])
        if self.is_storage_image_type(texture_type):
            return
        texture_name = self.expression_name(args[0]) or str(args[0])
        raise OpenGLRefusal(
            f"OpenGL image operation '{func_name}' requires a storage "
            f"image resource argument: {texture_name}"
        )

    def validate_image_access_argument(self, func_name, args):
        if not args or not is_image_resource_operation(
            func_name, IMAGE_RESOURCE_INTRINSIC_NAMES
        ):
            return
        access = self.image_resource_access(args[0])
        if access is None or access == "read_write":
            return
        texture_name = expression_debug_name(args[0])
        if func_name == "imageLoad" and access == "write":
            raise OpenGLRefusal(
                f"OpenGL image operation '{func_name}' requires read-capable "
                f"storage image access for {texture_name}: got writeonly"
            )
        if func_name == "imageStore" and access == "read":
            raise OpenGLRefusal(
                f"OpenGL image operation '{func_name}' requires write-capable "
                f"storage image access for {texture_name}: got readonly"
            )
        if is_image_atomic_operation(func_name):
            access_name = "readonly" if access == "read" else "writeonly"
            raise OpenGLRefusal(
                f"OpenGL image operation '{func_name}' requires read-write "
                f"storage image access for {texture_name}: got {access_name}"
            )

    def validate_structured_buffer_access_argument(self, func_name, args):
        required_access = self.structured_buffer_operation_access_requirement(func_name)
        if required_access is None or not args:
            return
        access = self.structured_buffer_resource_access(args[0])
        if image_access_satisfies_requirement(required_access, access):
            return
        buffer_name = expression_debug_name(args[0])
        required_label = image_access_requirement_label(required_access)
        actual_label = image_access_diagnostic_name(access)
        raise OpenGLRefusal(
            f"OpenGL buffer operation '{func_name}' requires {required_label} "
            f"SSBO access for {buffer_name}: got {actual_label}"
        )

    def glsl_buffer_block_access_metadata(self, node):
        choices = self.resource_access_metadata_choices(node)
        self.validate_resource_access_metadata_consistency(node, choices)
        if not choices:
            return None
        return choices[0][1]

    def record_glsl_buffer_block_variable(self, name, node, vtype=None):
        if not name:
            return
        block_type = str(
            self.resource_base_type(vtype or glsl_buffer_block_node_type(node))
        )
        self.glsl_buffer_block_variable_names.add(name)
        self.glsl_buffer_block_variable_types[name] = block_type
        access = self.glsl_buffer_block_access_metadata(node)
        if access is None or access == "read_write":
            return
        self.glsl_buffer_block_variable_accesses[name] = access

    def glsl_buffer_block_member_base_name(self, expr):
        if isinstance(expr, MemberAccessNode):
            return self.glsl_buffer_block_member_base_name(expr.object)
        if isinstance(expr, ArrayAccessNode) or (
            hasattr(expr, "__class__") and "ArrayAccess" in str(expr.__class__)
        ):
            return self.glsl_buffer_block_member_base_name(
                getattr(expr, "array", getattr(expr, "array_expr", None))
            )
        return self.expression_name(expr)

    def is_glsl_buffer_block_member_expression(self, expr):
        if isinstance(expr, MemberAccessNode):
            return True
        if isinstance(expr, ArrayAccessNode) or (
            hasattr(expr, "__class__") and "ArrayAccess" in str(expr.__class__)
        ):
            return self.is_glsl_buffer_block_member_expression(
                getattr(expr, "array", getattr(expr, "array_expr", None))
            )
        return False

    def is_glsl_buffer_block_member_reference(self, expr):
        if not self.is_glsl_buffer_block_member_expression(expr):
            return False
        base_name = self.glsl_buffer_block_member_base_name(expr)
        return bool(base_name and base_name in self.glsl_buffer_block_variable_names)

    def glsl_buffer_block_member_access(self, expr):
        if not self.is_glsl_buffer_block_member_expression(expr):
            return None
        base_name = self.glsl_buffer_block_member_base_name(expr)
        if not base_name:
            return None
        return self.glsl_buffer_block_variable_accesses.get(base_name)

    def validate_glsl_buffer_block_member_access(self, expr, required_access):
        access = self.glsl_buffer_block_member_access(expr)
        if image_access_satisfies_requirement(required_access, access):
            return
        member_name = expression_debug_name(expr)
        required_label = image_access_requirement_label(required_access)
        actual_label = image_access_diagnostic_name(access)
        raise OpenGLRefusal(
            f"OpenGL buffer block member access for {member_name} requires "
            f"{required_label} buffer block access: got {actual_label}"
        )

    def validate_glsl_buffer_block_assignment_target(self, target, operator):
        if operator == "=":
            self.validate_glsl_buffer_block_member_access(target, "write")
            return
        self.validate_glsl_buffer_block_member_access(target, "read_write")

    def validate_glsl_buffer_block_atomic_call(self, func_name, args):
        if func_name not in self.GLSL_MEMORY_ATOMIC_FUNCTIONS or not args:
            return
        target = args[0]
        self.validate_glsl_buffer_block_member_access(target, "read_write")
        if not self.is_glsl_buffer_block_member_reference(target):
            return
        target_type = self.expression_result_type(target)
        if target_type is None:
            return
        mapped_type = self.map_type(target_type)
        if mapped_type not in self.GLSL_ATOMIC_MEMORY_ZEROS:
            target_name = expression_debug_name(target)
            raise OpenGLRefusal(
                f"OpenGL buffer block atomic '{func_name}' requires a scalar "
                f"int or uint buffer block member, or an int64_t or uint64_t "
                f"one, for {target_name}: got {mapped_type}"
            )
        self.validate_glsl_buffer_block_atomic_value_arguments(
            func_name,
            args,
            target,
            mapped_type,
        )

    def validate_glsl_buffer_block_atomic_value_arguments(
        self, func_name, args, target, target_type
    ):
        target_name = expression_debug_name(target)
        for value_arg, label in self.glsl_buffer_block_atomic_value_arguments(
            func_name, args
        ):
            value_type = self.glsl_buffer_block_atomic_argument_type(value_arg)
            if value_type is None or value_type == target_type:
                continue
            if (
                target_type in self.GLSL_INT64_SCALAR_TYPES
                and value_type in self.GLSL_ATOMIC_MEMORY_ZEROS
            ):
                # The 64-bit value operand is converted to the memory type.
                continue
            if self.glsl_unsigned_atomic_literal_is_compatible(
                value_arg, value_type, target_type
            ):
                continue
            raise OpenGLRefusal(
                f"OpenGL buffer block atomic '{func_name}' requires {target_type} "
                f"{label} argument for {target_name}: "
                f"{expression_debug_name(value_arg)} has type {value_type}"
            )

    def glsl_buffer_block_atomic_value_arguments(self, func_name, args):
        if func_name in {"atomicCompSwap", "atomicCompareExchangeWeak"}:
            for index, label in ((1, "compare"), (2, "value")):
                if len(args) > index:
                    yield args[index], label
            return
        if len(args) > 1:
            yield args[1], "value"

    def glsl_buffer_block_atomic_argument_type(self, expr):
        scalar_kind = self.scalar_expression_kind(expr)
        if scalar_kind is not None:
            return scalar_kind
        result_type = self.expression_result_type(expr)
        if result_type is None:
            return None
        return self.map_type(result_type)

    def glsl_unsigned_atomic_literal_is_compatible(
        self, value_arg, value_type, target_type
    ):
        if target_type != "uint" or value_type != "int":
            return False
        literal_value = self.literal_int_value(value_arg, self.literal_int_constants)
        return literal_value is not None and literal_value >= 0

    def glsl_expected_compare_storage(self, target):
        if isinstance(target, PointerAccessNode):
            target = MemberAccessNode(
                ArrayAccessNode(target.pointer_expr, 0), target.member
            )
        if isinstance(target, UnaryOpNode) and target.op == "*":
            target = ArrayAccessNode(target.operand, 0)
        if (
            isinstance(target, FunctionCallNode)
            and self.function_call_name(target) == "buffer_load"
            and "buffer_load" not in self.function_return_types
            and len(target.arguments) == 2
        ):
            target = ArrayAccessNode(*target.arguments)
        if isinstance(target, MemberAccessNode):
            storage = self.glsl_expected_compare_storage(target.object)
            if storage is not None:
                storage[
                    "target"
                ] += f".{self.glsl_member_access_name(target.object, target.member)}"
            return storage
        if not isinstance(target, ArrayAccessNode):
            return None
        container, index = target.array, target.index
        binding = self.glsl_workgroup_pointer_binding(
            container, self.glsl_workgroup_pointer_aliases()
        )
        if binding is None:
            binding = self.glsl_storage_pointer_binding(
                container, self.glsl_storage_pointer_aliases()
            )
        if binding is not None:
            if binding.get("pointer_reinterpretation"):
                raise OpenGLRefusal(
                    "OpenGL atomicCompareExchangeWeak requires an unambiguous typed storage view"
                )
            rendered = self.glsl_index_expression(index, container)
            offset = self.glsl_workgroup_pointer_offset_expression(binding)
            if offset not in {"0", "0u"}:
                rendered = (
                    offset
                    if rendered in {"0", "0u"}
                    else f"({offset} + int({rendered}))"
                )
            return {
                "target": f"{binding['root']}[index0]",
                "arguments": [f"uint({rendered})"],
            }
        access = self.structured_buffer_resource_access(container)
        if access is not None:
            if not image_access_satisfies_requirement("read_write", access):
                raise OpenGLRefusal(
                    "OpenGL atomicCompareExchangeWeak requires read-write storage"
                )
            root = self.generate_expression(container)
            index_expr = self.glsl_index_expression(index, container)
            return {"target": f"{root}[index0]", "arguments": [f"uint({index_expr})"]}
        storage = self.glsl_expected_compare_storage(container)
        if storage is not None:
            storage["target"] += f"[index{len(storage['arguments'])}]"
            storage["arguments"].append(
                f"uint({self.glsl_index_expression(index, container)})"
            )
        return storage

    def generate_glsl_expected_compare_call(self, args, word_storage=None):
        if len(args) != 3:
            raise OpenGLRefusal(
                "OpenGL atomicCompareExchangeWeak requires target, expected and desired"
            )
        kind = self.map_type(self.expression_result_type(args[0]))
        if (
            kind not in {"int", "uint"}
            or self.map_type(self.expression_result_type(args[1])) != kind
        ):
            raise OpenGLRefusal(
                "OpenGL atomicCompareExchangeWeak requires matching integer target and expected"
            )
        if word_storage is not None:
            storage = {
                "target": word_storage["target"],
                "arguments": list(word_storage["arguments"]),
            }
        else:
            self.validate_glsl_storage_pointer_mutation_target(args[0])
            storage = self.glsl_expected_compare_storage(args[0])
        if storage is None:
            raise OpenGLRefusal(
                "OpenGL atomicCompareExchangeWeak requires a storage buffer or shared target"
            )
        key = (kind, storage["target"], len(storage["arguments"]))
        helper = self.glsl_expected_compare_helpers.get(key)
        if helper is None:
            name = self.glsl_generated_module_identifier(
                ("compare-expected", key), "crossgl_compare_expected"
            )
            helper = {**storage, "name": name, "kind": kind}
            self.glsl_expected_compare_helpers[key] = helper
        arguments = storage["arguments"] + [
            self.generate_expression(args[1]),
            self.generate_expression_with_expected(args[2], kind),
        ]
        return f"{helper['name']}({', '.join(arguments)})"

    def generate_glsl_expected_compare_helpers(self):
        code = ""
        for helper in self.glsl_expected_compare_helpers.values():
            kind = helper["kind"]
            roles = ["expected", "desired", "observed", "matched"] + [
                f"index{i}" for i in range(len(helper["arguments"]))
            ]
            names = {
                role: self.glsl_generated_module_identifier(
                    ("compare-local", role), f"crossgl_atomic_{role}"
                )
                for role in roles
            }
            target = helper["target"]
            parameters = []
            for i in range(len(helper["arguments"])):
                name = names[f"index{i}"]
                parameters.append(f"uint {name}")
                target = target.replace(f"[index{i}]", f"[{name}]")
            parameters.extend(
                [f"inout {kind} {names['expected']}", f"{kind} {names['desired']}"]
            )
            code += (
                f"bool {helper['name']}({', '.join(parameters)}) {{\n"
                f"    {kind} {names['observed']} = atomicCompSwap({target}, {names['expected']}, {names['desired']});\n"
                f"    bool {names['matched']} = {names['observed']} == {names['expected']};\n"
                f"    if (!{names['matched']}) {names['expected']} = {names['observed']};\n"
                f"    return {names['matched']};\n"
                "}\n\n"
            )
        return code

    def generate_glsl_memory_atomic_call(self, func_name, args):
        if func_name not in self.GLSL_MEMORY_ATOMIC_FUNCTIONS or not args:
            return None
        if func_name in self.function_return_types:
            return None
        float_atomic = self.float_atomic_storage.atomic(func_name, args)
        if float_atomic is not None:
            return float_atomic
        self.require_glsl_int64_memory_atomics(args[0])
        word_storage = glsl_word_view_integer_storage(self, args[0])
        if func_name == "atomicCompareExchangeWeak":
            self.validate_glsl_buffer_block_atomic_call(func_name, args)
            return self.generate_glsl_expected_compare_call(
                args, word_storage=word_storage
            )
        if func_name == "atomicLoad" and len(args) != 1:
            raise OpenGLRefusal("OpenGL atomicLoad requires exactly one target")
        if func_name == "atomicStore" and len(args) != 2:
            raise OpenGLRefusal("OpenGL atomicStore requires a target and value")
        if func_name == "atomicLoad" and word_storage is None:
            self.validate_glsl_storage_pointer_mutation_target(args[0])
            root = args[0]
            while True:
                if isinstance(root, MemberAccessNode):
                    root = root.object
                elif isinstance(root, ArrayAccessNode):
                    root = root.array_expr
                elif (
                    isinstance(root, FunctionCallNode)
                    and self.function_call_name(root) == "buffer_load"
                    and "buffer_load" not in self.function_return_types
                    and len(root.arguments) == 2
                ):
                    root = root.arguments[0]
                else:
                    break
            access = self.structured_buffer_resource_access(root)
            if not image_access_satisfies_requirement("read_write", access):
                raise OpenGLRefusal(
                    "OpenGL atomicLoad requires read-write storage for its "
                    "unchanged-value atomic operation"
                )

        self.validate_glsl_buffer_block_atomic_call(func_name, args)

        pointer_binding = self.glsl_workgroup_pointer_binding(
            args[0], self.glsl_workgroup_pointer_aliases()
        )
        if pointer_binding is not None:
            target_type = pointer_binding.get("element_type")
            if target_type not in {"int", "uint"}:
                raise OpenGLWorkgroupPointerError(
                    f"OpenGL workgroup atomic '{func_name}' requires an int or "
                    f"uint element, got {target_type or 'an unresolved type'}",
                    parameter_name=self.expression_name(args[0]),
                    reason="atomic-element-type-unsupported",
                    source_location=getattr(args[0], "source_location", None),
                )
            self.validate_glsl_buffer_block_atomic_value_arguments(
                func_name,
                args,
                args[0],
                target_type,
            )
            target_expression = self.glsl_workgroup_pointer_element_expression(args[0])
        else:
            target_type = self.expression_result_type(args[0])
            target_type = (
                self.map_type(target_type) if target_type is not None else None
            )
            if target_type not in {"int", "uint"}:
                source_target = self.glsl_atomic_operand_family(args[0])
                if source_target in {"int", "uint"}:
                    target_type = source_target
            target_expression = (
                word_storage["lvalue"]
                if word_storage is not None
                else self.generate_expression(args[0])
            )
        if (
            func_name in {"atomicLoad", "atomicStore"}
            and target_type not in self.GLSL_ATOMIC_MEMORY_ZEROS
        ):
            raise OpenGLRefusal(
                f"OpenGL {func_name} requires a scalar int, uint, int64_t or "
                "uint64_t target"
            )
        rendered_args = [target_expression]
        value_arg_ids = {
            id(value_arg)
            for value_arg, _ in self.glsl_buffer_block_atomic_value_arguments(
                func_name, args
            )
        }

        for arg in args[1:]:
            if (
                id(arg) in value_arg_ids
                and target_type in self.GLSL_ATOMIC_MEMORY_ZEROS
            ):
                rendered_args.append(
                    self.generate_glsl_memory_atomic_value_argument(arg, target_type)
                )
            else:
                rendered_args.append(self.generate_expression(arg))

        native_name = "atomicExchange" if func_name == "atomicStore" else func_name
        if func_name == "atomicLoad":
            native_name = "atomicOr"
            rendered_args.append(self.GLSL_ATOMIC_MEMORY_ZEROS[target_type])
        return f"{native_name}({', '.join(rendered_args)})"

    def require_glsl_int64_memory_atomics(self, target):
        """Require the extension of the 64-bit integer atomic overloads when ``target`` is a 64-bit integer.

        GLSL 4.60 declares the buffer and shared-memory atomic functions for 32-bit ``int`` and ``uint`` memory only
        (section 8.11); GL_NV_shader_atomic_int64 adds their ``int64_t`` and ``uint64_t`` overloads.
        """
        info = self.glsl_value_type_info(self.glsl_source_expression_type(target))
        if (
            info is None
            or info["width"] != 1
            or info["bits"] != 64
            or info["family"] not in {"int", "uint"}
        ):
            return
        required = getattr(self, "glsl_stage_builtin_extensions", None)
        if required is None:
            required = self.glsl_stage_builtin_extensions = []
        if self.GLSL_INT64_ATOMICS_EXTENSION not in required:
            required.append(self.GLSL_INT64_ATOMICS_EXTENSION)

    def generate_glsl_memory_atomic_value_argument(self, arg, target_type):
        value_type = self.glsl_buffer_block_atomic_argument_type(arg)
        if value_type == target_type:
            return self.generate_expression_with_expected(arg, target_type)
        if self.glsl_unsigned_atomic_literal_is_compatible(
            arg, value_type, target_type
        ):
            literal_value = self.literal_int_value(arg, self.literal_int_constants)
            if literal_value is not None:
                return f"{literal_value}u"
        if target_type in self.GLSL_INT64_SCALAR_TYPES:
            # The 64-bit overloads take operands of the memory type too; widen a
            # 32-bit operand and reinterpret the other 64-bit signedness.
            info = self.glsl_value_type_info(self.glsl_source_expression_type(arg))
            if (
                info is not None
                and info["width"] == 1
                and info["family"] in {"int", "uint"}
            ):
                return self.glsl_converted_operand(arg, target_type)
            return self.generate_expression(arg)
        if self.glsl_atomic_operand_family(arg) in ({"int", "uint"} - {target_type}):
            # GLSL atomics take operands of the memory type; HLSL converts
            # between int and uint implicitly.
            return self.glsl_converted_operand(arg, target_type)
        return self.generate_expression(arg)

    def generate_glsl_buffer_block_mutation_target(self, expr):
        self.glsl_buffer_block_read_validation_suppression += 1
        try:
            return self.generate_expression(expr)
        finally:
            self.glsl_buffer_block_read_validation_suppression -= 1

    def image_atomic_value_arguments(self, func_name, args):
        image_type = self.texture_argument_resource_type(args[0])
        has_sample = self.is_multisample_storage_image_type(image_type)
        return shared_image_atomic_value_arguments(func_name, args, has_sample)

    def scalar_expression_kind(self, expr):
        return numeric_scalar_expression_kind(
            expr,
            self.expression_result_type,
            self.type_name_string,
            self.map_type,
        )

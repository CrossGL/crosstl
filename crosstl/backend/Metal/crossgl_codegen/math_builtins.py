"""Materialized standard-library wrappers and math builtin overload resolution."""

from ....translator.math_namespaces import metal_precise_math_operation
from ..dependent_builtins import (
    NOT_DEPENDENT,
    dependent_access,
    dependent_builtin_result_type,
    unbound_template_type_names,
)
from ..MetalAst import (
    CastNode,
    FunctionCallNode,
    ReturnNode,
    VariableNode,
    VectorConstructorNode,
)
from .support import (
    MetalBuiltinResultTypeResolutionError,
    MetalStandardLibraryWrapperLoweringError,
)


class MetalMathBuiltinMixin:
    """Members of ``MetalToCrossGLConverter``: materialized standard-library wrappers and math builtin overload resolution."""

    @staticmethod
    def metal_math_source_overload_is_stdlib_extension(function):
        namespace = str(getattr(function, "namespace", "") or "")
        return namespace == "metal" or namespace.startswith("metal::")

    def materialized_metal_stdlib_wrapper_intrinsics(self, function):
        namespace = str(getattr(function, "namespace", "") or "")
        if namespace != "metal" and not namespace.startswith("metal::"):
            return ()
        declaration_qualifiers = {
            str(qualifier)
            for qualifier in getattr(function, "declaration_qualifiers", []) or []
        }
        if "METAL_FUNC" not in declaration_qualifiers:
            return ()

        intrinsics = set()
        seen = set()

        def visit(node):
            if node is None or isinstance(node, (str, int, float, bool)):
                return
            if not isinstance(node, (dict, list, tuple, set)):
                node_id = id(node)
                if node_id in seen:
                    return
                seen.add(node_id)
            if isinstance(node, FunctionCallNode):
                call_name = str(getattr(node, "name", ""))
                if call_name.rsplit("::", 1)[-1].startswith("__metal_"):
                    intrinsics.add(call_name)
            for child in self.iter_ast_children(node):
                visit(child)

        visit(getattr(function, "body", None))
        function_name = str(getattr(function, "name", ""))
        if function_name in self.materialized_metal_stdlib_body_wrappers:
            intrinsics.add(f"metal::{function_name}")
        return tuple(sorted(intrinsics))

    def is_materialized_metal_stdlib_wrapper(self, function):
        return bool(self.materialized_metal_stdlib_wrapper_intrinsics(function))

    def materialized_metal_trunc_wrapper_is_canonical(self, function):
        """Accept a direct intrinsic wrapper, not additional source computation."""
        body = list(getattr(function, "body", None) or [])
        parameters = list(getattr(function, "params", None) or [])
        if (
            len(body) != 1
            or not isinstance(body[0], ReturnNode)
            or len(parameters) != 1
        ):
            return False

        def canonical_type(value):
            return self.map_type(self.resolve_type_alias(value))

        result_type = canonical_type(function.return_type)
        parameter = parameters[0]
        if (
            canonical_type(self.metal_source_overload_parameter_type(parameter))
            != result_type
        ):
            return False

        def unwrap_conversion(node, allowed_types):
            if isinstance(node, CastNode):
                if canonical_type(node.target_type) in allowed_types:
                    return node.expression
            elif isinstance(node, (FunctionCallNode, VectorConstructorNode)):
                name = getattr(node, "vector_type", None) or getattr(node, "name", None)
                if len(node.args) == 1 and canonical_type(name) in allowed_types:
                    return node.args[0]
            return node

        operation = unwrap_conversion(body[0].value, {result_type})
        if (
            not isinstance(operation, FunctionCallNode)
            or str(operation.name).rsplit("::", 1)[-1] != "__metal_trunc"
            or len(operation.args) not in {1, 2}
        ):
            return False
        if len(operation.args) == 2:
            mode = operation.args[1]
            mode_name = mode.name if isinstance(mode, VariableNode) else mode
            if not isinstance(mode_name, str) or mode_name not in {
                "true",
                "false",
                "__METAL_MAYBE_FAST_MATH__",
                "__METAL_FAST_MATH__",
                "__METAL_PRECISE_MATH__",
            }:
                return False
        input_types = {result_type}
        if result_type in {"f16", "float16", "bfloat16"}:
            input_types.add("float")
        operand = unwrap_conversion(operation.args[0], input_types)
        return isinstance(operand, VariableNode) and operand.name == parameter.name

    def validate_materialized_metal_stdlib_wrapper_call(self, expression):
        selected = self.selected_metal_callable(expression)
        intrinsics = self.materialized_metal_stdlib_wrapper_intrinsics(selected)
        if not intrinsics:
            return

        name = str(getattr(expression, "name", ""))
        unscoped_name = name.rsplit("::", 1)[-1]
        if unscoped_name == "trunc":
            if self.materialized_metal_trunc_wrapper_is_canonical(selected):
                return
            raise MetalStandardLibraryWrapperLoweringError(
                name, intrinsics, getattr(expression, "source_location", None)
            )
        if unscoped_name in self.metal_wave_intrinsics:
            public_operation = self.metal_wave_intrinsics[unscoped_name]
            internal_operations = tuple(
                self.metal_wave_intrinsics.get(str(intrinsic).rsplit("::", 1)[-1])
                for intrinsic in intrinsics
            )
            if internal_operations and all(
                operation == public_operation for operation in internal_operations
            ):
                return
            raise MetalStandardLibraryWrapperLoweringError(
                name,
                intrinsics,
                getattr(expression, "source_location", None),
            )
        if (
            unscoped_name in self.metal_math_intrinsics
            or unscoped_name in self.metal_bit_intrinsics
        ):
            return
        raise MetalStandardLibraryWrapperLoweringError(
            name,
            intrinsics,
            getattr(expression, "source_location", None),
        )

    def generate_materialized_bfloat_math_wrapper_call(self, expression, is_main=False):
        """Preserve float computation and the narrow return of bfloat wrappers."""
        public_name = str(getattr(expression, "name", "")).rsplit("::", 1)[-1]
        if public_name not in self.metal_math_intrinsics:
            return None
        selected = self.selected_metal_callable(expression)
        if not self.is_materialized_metal_stdlib_wrapper(selected):
            return None
        return_type = self.metal_source_overload_value_type(
            self.selected_metal_callable_return_type(selected)
        )
        normalized = self.normalized_metal_type(self.resolve_type_alias(return_type))
        if normalized not in self.metal_source_bfloat_types:
            return None
        rendered = self.generate_metal_fma_call(expression, is_main)
        if rendered is None:
            rendered = self.generate_metal_remainder_call(expression, is_main)
        if rendered is None:
            rendered = self.generate_metal_power_call(expression, is_main)
        if rendered is None:
            rendered = self.generate_metal_log2_call(expression, is_main)
        if rendered is None:
            rendered = self.generate_metal_round_call(expression, is_main)
        if rendered is None:
            function_name = self.map_function_call_name(
                expression.name,
                expression.args,
                source_offset=self.alias_source_offset(expression),
            )
            parameters = list(getattr(selected, "params", []) or [])
            if len(parameters) != len(expression.args):
                raise MetalStandardLibraryWrapperLoweringError(
                    str(expression.name),
                    self.materialized_metal_stdlib_wrapper_intrinsics(selected),
                    getattr(expression, "source_location", None),
                )
            arguments = []
            for parameter, argument in zip(parameters, expression.args):
                value = self.generate_expression(argument, is_main)
                parameter_type = self.metal_source_overload_parameter_type(parameter)
                normalized_parameter = self.normalized_metal_type(
                    self.resolve_type_alias(parameter_type)
                )
                if normalized_parameter in self.metal_source_bfloat_types:
                    value = f"float({value})"
                arguments.append(value)
            if metal_precise_math_operation(str(expression.name)) == function_name:
                # The binary32 call keeps its precise:: annotation for the
                # backends that honor it, as the float and half calls do.
                function_name = f"precise::{function_name}"
            rendered = f"{function_name}({', '.join(arguments)})"
        return f"{self.map_type(return_type)}({rendered})"

    def generate_materialized_bfloat_wave_wrapper_call(self, expression, is_main=False):
        """Preserve the float compute contract of Metal bfloat SIMD wrappers."""
        selected = self.selected_metal_callable(expression)
        intrinsics = self.materialized_metal_stdlib_wrapper_intrinsics(selected)
        if not intrinsics:
            return None

        public_name = str(getattr(expression, "name", "")).rsplit("::", 1)[-1]
        public_operation = self.metal_wave_intrinsics.get(public_name)
        internal_operations = tuple(
            self.metal_wave_intrinsics.get(str(intrinsic).rsplit("::", 1)[-1])
            for intrinsic in intrinsics
        )
        if (
            public_operation is None
            or not internal_operations
            or not all(
                operation == public_operation for operation in internal_operations
            )
        ):
            return None

        return_type = self.selected_metal_callable_return_type(selected)
        return_value_type = self.metal_source_overload_value_type(return_type)
        normalized_return_type = self.normalized_metal_type(
            self.resolve_type_alias(return_value_type)
        )
        if normalized_return_type not in self.metal_source_bfloat_types:
            return None

        parameters = list(getattr(selected, "params", []) or [])
        if len(parameters) != len(expression.args):
            raise MetalStandardLibraryWrapperLoweringError(
                str(getattr(expression, "name", "")),
                intrinsics,
                getattr(expression, "source_location", None),
            )

        arguments = []
        for index, (parameter, argument) in enumerate(zip(parameters, expression.args)):
            rendered = self.generate_metal_wave_argument(
                public_name, public_operation, expression, index, is_main
            )
            parameter_type = self.metal_source_overload_parameter_type(parameter)
            normalized_parameter_type = self.normalized_metal_type(
                self.resolve_type_alias(parameter_type)
            )
            if normalized_parameter_type in self.metal_source_bfloat_types:
                rendered = f"float({rendered})"
            arguments.append(rendered)

        result_type = self.map_type(return_value_type)
        return f"{result_type}({public_operation}({', '.join(arguments)}))"

    def resolve_metal_math_builtin_name(self, name, arguments):
        text = str(name).lstrip(":")
        builtin_name = self.map_metal_math_function_name(text, arguments)
        if builtin_name is None:
            if "::" in text or text not in self.metal_math_intrinsics:
                return None
            builtin_name = text

        source_overloads = self.metal_user_function_overloads(name)
        if source_overloads:
            binding, _function = self.resolve_metal_user_function_overload(
                name, arguments
            )
            if binding in {"user", "unknown"}:
                return None
            if "::" not in text and any(
                not self.metal_math_source_overload_is_stdlib_extension(function)
                for function in source_overloads
            ):
                return None
        if "::" not in text and (
            text in self.current_variable_types or text in self.global_variable_types
        ):
            return None
        return builtin_name

    @staticmethod
    def metal_math_builtin_namespace_mode(name):
        text = str(name).lstrip(":")
        if text.startswith(("metal::fast::", "fast::")):
            return "fast"
        if text.startswith(("metal::precise::", "precise::")):
            return "precise"
        return "default"

    def metal_math_builtin_candidate_base_types(self, function_name, family):
        mode = self.metal_math_builtin_namespace_mode(function_name)
        if mode != "default":
            if family in {
                "floating",
                "floating_vector",
                "floating_bfloat",
                "arithmetic",
                "sincos",
            }:
                return ("float",)
            return ()
        if family == "bool":
            return ("bool",)
        if family in {"floating", "floating_vector"}:
            return ("half", "float", "double")
        if family == "floating_bfloat":
            return ("bfloat", "half", "float", "double")
        if family == "arithmetic":
            return (
                "half",
                "float",
                "double",
                "int",
                "uint",
                "short",
                "ushort",
                "char",
                "uchar",
                "long",
                "ulong",
            )
        if family == "select":
            return (
                "bool",
                "bfloat",
                "half",
                "float",
                "double",
                "int",
                "uint",
                "short",
                "ushort",
                "char",
                "uchar",
                "long",
                "ulong",
            )
        if family == "sincos":
            return ("half", "float", "double")
        return ()

    def metal_math_builtin_type_info(self, metal_type):
        value_type = self.metal_source_overload_value_type(metal_type)
        if value_type is None:
            return None
        normalized_value = self.normalized_metal_type(value_type)
        if normalized_value.startswith("packed_"):
            return {
                "value_type": value_type,
                "category": "packed",
                "width": 1,
                "element_type": value_type,
                "identity": self.metal_source_overload_type_identity(value_type),
            }

        vector = self.metal_vector_component_parts(value_type)
        if vector is None:
            element_type = value_type
            width = 1
            identity = self.metal_source_overload_type_identity(value_type)
        else:
            element_type, width, _width_text = vector
            if width is None:
                return None
            identity = (
                "vector",
                width,
                self.metal_source_overload_type_identity(element_type),
            )
        normalized_element = self.normalized_metal_type(
            self.resolve_type_alias(element_type)
        )
        if normalized_element == "bool":
            category = "bool"
        elif normalized_element in self.metal_source_bfloat_types:
            category = "bfloat"
        else:
            scalar = self.metal_scalar_arithmetic_type_info(element_type)
            category = scalar[0] if scalar is not None else "object"
        return {
            "value_type": value_type,
            "category": category,
            "width": width,
            "element_type": element_type,
            "identity": identity,
        }

    def metal_math_builtin_candidate_widths(self, argument_types):
        widths = {
            info["width"]
            for argument_type in argument_types
            if argument_type is not None
            for info in [self.metal_math_builtin_type_info(argument_type)]
            if info is not None
        }
        return tuple(sorted(widths)) if widths else (1, "N")

    @staticmethod
    def metal_math_builtin_vector_type(base_type, width):
        if width == 1:
            return base_type
        if width == "N" or int(width) > 4:
            return f"metal::vec<{base_type}, {width}>"
        return f"{base_type}{width}"

    def metal_math_builtin_candidates(
        self, function_name, result_kind, family, arity, argument_types
    ):
        candidates = []
        widths = self.metal_math_builtin_candidate_widths(argument_types)
        if family == "floating_vector":
            widths = tuple(width for width in widths if width != 1)
        for width in widths:
            bool_type = self.metal_math_builtin_vector_type("bool", width)
            for base_type in self.metal_math_builtin_candidate_base_types(
                function_name, family
            ):
                value_type = self.metal_math_builtin_vector_type(base_type, width)
                if family == "select":
                    parameter_types = (value_type, value_type, bool_type)
                    display_parameter_types = parameter_types
                elif family == "sincos":
                    parameter_types = (value_type, value_type)
                    display_parameter_types = (
                        value_type,
                        f"thread {value_type}&",
                    )
                else:
                    parameter_types = (value_type,) * arity
                    display_parameter_types = parameter_types

                if result_kind == "bool_reduction":
                    result_type = "bool"
                elif result_kind == "bool_shape":
                    result_type = bool_type
                elif result_kind == "element":
                    result_type = base_type
                else:
                    result_type = value_type
                signature = (
                    f"{result_type} {function_name}"
                    f"({', '.join(display_parameter_types)})"
                )
                candidates.append(
                    {
                        "parameter_types": parameter_types,
                        "result_type": result_type,
                        "signature": signature,
                        "concrete": width != "N",
                        "exact_parameters": (1,) if family == "sincos" else (),
                    }
                )
        return candidates

    def metal_math_builtin_conversion_rank(self, actual_type, expected_type, family):
        actual = self.metal_math_builtin_type_info(actual_type)
        expected = self.metal_math_builtin_type_info(expected_type)
        if actual is None or expected is None:
            return None
        if actual["category"] == "packed" or expected["category"] == "packed":
            return None
        if actual["identity"] == expected["identity"]:
            return 4
        if actual["width"] != expected["width"]:
            if actual["width"] == 1 and expected["width"] != 1:
                element_rank = self.metal_math_builtin_conversion_rank(
                    actual["value_type"], expected["element_type"], family
                )
                return None if element_rank is None else max(1, element_rank - 1)
            return None
        if actual["width"] != 1:
            return None
        if "object" in {actual["category"], expected["category"]}:
            return None

        actual_name = self.normalized_metal_type(actual["element_type"])
        expected_name = self.normalized_metal_type(expected["element_type"])
        if actual["category"] == "bool":
            if expected["category"] == "integer":
                expected_scalar = self.metal_scalar_arithmetic_type_info(
                    expected["element_type"]
                )
                if expected_scalar == ("integer", True, 32):
                    return 2
                return 1
            return 1 if expected["category"] == "floating" else None
        if expected["category"] == "bool":
            return (
                1 if actual["category"] in {"bfloat", "floating", "integer"} else None
            )
        if actual["category"] == "bfloat":
            if expected["category"] == "floating":
                if expected_name == "float":
                    return 1 if family == "arithmetic" else 2
                if expected_name == "double":
                    return 1
                return None
            if expected["category"] == "integer":
                return 1
            return None
        if expected["category"] == "bfloat":
            return None

        actual_scalar = self.metal_scalar_arithmetic_type_info(actual["element_type"])
        expected_scalar = self.metal_scalar_arithmetic_type_info(
            expected["element_type"]
        )
        if actual_scalar is None or expected_scalar is None:
            return None
        actual_family, _actual_signed, actual_bits = actual_scalar
        expected_family, expected_signed, expected_bits = expected_scalar
        if actual_family == expected_family == "integer":
            if actual_bits < 32 and expected_bits == 32 and expected_signed:
                return 2
            return 1
        if actual_family == expected_family == "floating":
            return 2 if actual_bits < expected_bits else 1
        if actual_name == "bool" or expected_name == "bool":
            return 1
        return 1

    @staticmethod
    def metal_math_builtin_rank_dominates(left, right):
        return all(a >= b for a, b in zip(left, right)) and any(
            a > b for a, b in zip(left, right)
        )

    def metal_math_builtin_result_type(self, expression):
        builtin_name = self.resolve_metal_math_builtin_name(
            expression.name, expression.args
        )
        if builtin_name is None:
            return None
        rule = self.metal_math_builtin_result_rules.get(builtin_name)
        if rule is None:
            return None
        result_kind, family, arity = rule
        argument_types = [
            self.metal_source_overload_value_type(self.expression_metal_type(argument))
            for argument in expression.args
        ]
        unbound_names = unbound_template_type_names(
            self.current_function,
            set(self.type_aliases).union(*self.template_type_bindings),
        )
        dependent = dependent_builtin_result_type(
            result_kind,
            argument_types,
            unbound_names,
            dependent_unknown=any(
                argument_type is None
                and dependent_access(
                    argument, self.expression_metal_type, unbound_names
                )
                for argument, argument_type in zip(expression.args, argument_types)
            ),
        )
        if dependent is not NOT_DEPENDENT:
            # An uninstantiated template resolves the overload per instantiation.
            return dependent
        diagnostic_argument_types = [
            argument_type or "<unknown>" for argument_type in argument_types
        ]
        candidates = self.metal_math_builtin_candidates(
            str(expression.name), result_kind, family, arity, argument_types
        )
        source_location = getattr(expression, "source_location", None)
        if len(argument_types) != arity:
            raise MetalBuiltinResultTypeResolutionError(
                str(expression.name),
                diagnostic_argument_types,
                [candidate["signature"] for candidate in candidates],
                f"the builtin requires {arity} argument{'s' if arity != 1 else ''}",
                source_location,
            )

        concrete_candidates = [
            candidate for candidate in candidates if candidate["concrete"]
        ]
        partially_viable = []
        for candidate in concrete_candidates:
            ranks = []
            for index, (actual_type, expected_type) in enumerate(
                zip(argument_types, candidate["parameter_types"])
            ):
                if actual_type is None:
                    ranks.append(None)
                    continue
                if index in candidate["exact_parameters"]:
                    actual_info = self.metal_math_builtin_type_info(actual_type)
                    expected_info = self.metal_math_builtin_type_info(expected_type)
                    if (
                        actual_info is None
                        or expected_info is None
                        or actual_info["identity"] != expected_info["identity"]
                    ):
                        break
                rank = self.metal_math_builtin_conversion_rank(
                    actual_type, expected_type, family
                )
                if rank is None:
                    break
                ranks.append(rank)
            else:
                partially_viable.append((tuple(ranks), candidate))

        if any(argument_type is None for argument_type in argument_types):
            viable = (
                [candidate for _ranks, candidate in partially_viable]
                if concrete_candidates
                and any(argument_type is not None for argument_type in argument_types)
                else candidates
            )
            reason = (
                "one or more builtin argument types could not be inferred"
                if viable
                else "no builtin signature matches the known argument types"
            )
            raise MetalBuiltinResultTypeResolutionError(
                str(expression.name),
                diagnostic_argument_types,
                [candidate["signature"] for candidate in viable],
                reason,
                source_location,
            )

        if not partially_viable:
            raise MetalBuiltinResultTypeResolutionError(
                str(expression.name),
                diagnostic_argument_types,
                [candidate["signature"] for candidate in candidates],
                "no builtin signature matches the inferred argument types",
                source_location,
            )

        winners = [
            entry
            for entry in partially_viable
            if not any(
                other is not entry
                and self.metal_math_builtin_rank_dominates(other[0], entry[0])
                for other in partially_viable
            )
        ]
        if len(winners) != 1:
            raise MetalBuiltinResultTypeResolutionError(
                str(expression.name),
                diagnostic_argument_types,
                [candidate["signature"] for _ranks, candidate in winners],
                "multiple builtin signatures remain viable after type matching",
                source_location,
            )
        return winners[0][1]["result_type"]

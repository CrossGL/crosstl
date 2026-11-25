"""Free operator calls, user function overloads and Metal math function names."""

from ..generic_local_aliases import (
    generic_alias_constructor,
    refuse_unresolved_local_alias_type,
)
from ..MetalAst import InitializerListNode
from ..namespaced_functions import nominated_namespaces_in_scope
from .support import (
    MetalBuiltinOverloadResolutionError,
    MetalPreciseMathLoweringError,
    MetalSourceOverloadResolutionError,
)


class MetalOverloadResolutionMixin:
    """Members of ``MetalToCrossGLConverter``: free operator calls, user function overloads and Metal math function names."""

    def generate_metal_free_operator_call(
        self, operator, arguments, is_main=False, source_location=None
    ):
        selected = self.resolve_metal_free_operator(
            operator,
            arguments,
            source_location,
        )
        if selected is None:
            return None
        function_name = self.sanitize_identifier(self.function_output_name(selected))
        rendered_arguments = ", ".join(
            self.generate_metal_value_argument(argument, parameter, is_main)
            for argument, parameter in zip(arguments, selected.params)
        )
        return f"{function_name}({rendered_arguments})"

    def metal_source_overload_groups_for_name(self, function_name):
        explicitly_global = str(function_name).startswith("::")
        function_name = str(function_name).lstrip(":")
        signature_groups = self.metal_source_overload_groups.get(function_name)
        if signature_groups and function_name in getattr(
            self, "metal_builtin_name_collision_groups", set()
        ):
            namespace = (
                ""
                if explicitly_global
                else str(getattr(self.current_function, "namespace", ""))
            )
            for scope in self.namespace_lookup_scopes(namespace):
                qualified_name = f"{scope}::{function_name}" if scope else function_name
                visible_names = {qualified_name}
                if not explicitly_global:
                    visible_names.update(
                        f"{target}::{function_name}"
                        for target in nominated_namespaces_in_scope(
                            namespace, scope, self.visible_using_namespace_targets
                        )
                    )
                scoped = {
                    signature: declarations
                    for signature, declarations in signature_groups.items()
                    if str(
                        getattr(declarations[0], "qualified_name", declarations[0].name)
                    )
                    in visible_names
                }
                if scoped:
                    return scoped
            return None
        if signature_groups or "::" not in str(function_name):
            return signature_groups
        unscoped_name = str(function_name).rsplit("::", 1)[-1]
        unscoped_groups = self.metal_source_overload_groups.get(unscoped_name)
        if not unscoped_groups:
            return None
        qualified_groups = {}
        for signature, declarations in unscoped_groups.items():
            matching = [
                declaration
                for declaration in declarations
                if str(getattr(declaration, "qualified_name", "")) == str(function_name)
            ]
            if matching:
                qualified_groups[signature] = matching
        return qualified_groups or None

    def resolve_transported_metal_source_overload(
        self, function_name, arguments, source_location=None
    ):
        library_operation = self.qualified_metal_math_operation(function_name)
        if library_operation is None and str(function_name) in getattr(
            self, "metal_using_declared_library_names", set()
        ):
            library_operation = str(function_name)
        if library_operation in getattr(
            self, "metal_builtin_name_collision_groups", set()
        ):
            binding, function = self.resolve_metal_user_function_overload(
                function_name, arguments
            )
            if binding == "none" or (
                binding == "user"
                and self.is_materialized_metal_stdlib_wrapper(function)
            ):
                return None
        signature_groups = self.metal_source_overload_groups_for_name(function_name)
        if not signature_groups:
            return None

        arity_groups = {
            signature: declarations
            for signature, declarations in signature_groups.items()
            if len(getattr(declarations[0], "params", []) or []) == len(arguments)
        }
        if not arity_groups:
            return None

        argument_types = [
            self.metal_source_overload_value_type(self.expression_metal_type(argument))
            for argument in arguments
        ]
        diagnostic_argument_types = [
            argument_type or "<unknown>" for argument_type in argument_types
        ]
        candidates = [
            self.metal_source_overload_candidate_signature(declarations[0])
            for declarations in arity_groups.values()
        ]
        if any(
            argument_type is None and not isinstance(argument, InitializerListNode)
            for argument, argument_type in zip(arguments, argument_types)
        ):
            raise MetalSourceOverloadResolutionError(
                function_name,
                diagnostic_argument_types,
                candidates,
                "one or more argument types could not be inferred",
                source_location,
            )

        ranked = []
        for _signature, declarations in arity_groups.items():
            parameters = getattr(declarations[0], "params", []) or []
            ranks = []
            for argument, argument_type, parameter in zip(
                arguments, argument_types, parameters
            ):
                match_rank = self.metal_source_overload_argument_match_rank(
                    argument,
                    argument_type,
                    parameter,
                )
                if match_rank is None:
                    break
                ranks.append(match_rank)
            else:
                ranked.append((tuple(ranks), declarations))

        if not ranked:
            raise MetalSourceOverloadResolutionError(
                function_name,
                diagnostic_argument_types,
                candidates,
                "no source-compatible overload matches the inferred argument types",
                source_location,
            )

        def dominates(left, right):
            return all(a >= b for a, b in zip(left, right)) and any(
                a > b for a, b in zip(left, right)
            )

        winners = [
            entry
            for entry in ranked
            if not any(
                other is not entry and dominates(other[0], entry[0]) for other in ranked
            )
        ]
        if len(winners) != 1:
            raise MetalSourceOverloadResolutionError(
                function_name,
                diagnostic_argument_types,
                [
                    self.metal_source_overload_candidate_signature(declarations[0])
                    for _ranks, declarations in winners
                ],
                "multiple source-compatible overloads remain after type matching",
                source_location,
            )

        declarations = winners[0][1]
        return next(
            (
                declaration
                for declaration in declarations
                if getattr(declaration, "body", None)
            ),
            declarations[0],
        )

    def transported_metal_source_overload_name(
        self, function_name, arguments, source_location=None
    ):
        selected = self.resolve_transported_metal_source_overload(
            function_name, arguments, source_location
        )
        if selected is None:
            return function_name
        for argument, parameter in zip(
            arguments,
            getattr(selected, "params", []) or [],
        ):
            if not isinstance(argument, InitializerListNode):
                continue
            parameter_type = self.metal_source_overload_parameter_type(parameter)
            descriptor = self.metal_source_overload_type_descriptor(parameter_type)
            if descriptor is not None and descriptor[0] == "vector":
                argument._metal_source_overload_expected_type = parameter_type
        return self.metal_source_overload_output_names[id(selected)]

    def metal_user_function_overloads(self, function_name, *, source_offset=None):
        explicitly_global = str(function_name).startswith("::")
        unscoped_name = str(function_name).lstrip(":")
        if unscoped_name in getattr(self, "metal_builtin_name_collision_groups", set()):
            groups = self.metal_source_overload_groups_for_name(function_name)
            return [
                declaration
                for declarations in (groups or {}).values()
                for declaration in declarations
            ]
        function_name = str(function_name).lstrip(":")
        direct = list(self.user_function_overloads_by_name.get(function_name, []))
        if direct and function_name in self.metal_wave_intrinsics:
            namespace = (
                ""
                if explicitly_global
                else str(getattr(self.current_function, "namespace", "") or "")
            )
            for scope in self.namespace_lookup_scopes(namespace):
                visible = {f"{scope}::{function_name}" if scope else function_name}
                if not explicitly_global:
                    visible.update(
                        f"{target}::{function_name}"
                        for target in nominated_namespaces_in_scope(
                            namespace,
                            scope,
                            lambda directive_scope: self.visible_using_namespace_targets(
                                directive_scope, use_offset=source_offset
                            ),
                        )
                    )
                matching = [
                    function
                    for function in direct
                    if str(getattr(function, "qualified_name", None) or function.name)
                    in visible
                ]
                if matching:
                    return matching
            return []
        if direct or "::" not in str(function_name):
            return direct
        unscoped_name = str(function_name).rsplit("::", 1)[-1]
        return [
            function
            for function in self.user_function_overloads_by_name.get(unscoped_name, [])
            if str(getattr(function, "qualified_name", "")) == str(function_name)
        ]

    def resolve_metal_user_function_overload(
        self,
        function_name,
        args,
        *,
        allow_wave_lane_conversion=False,
        source_offset=None,
    ):
        if source_offset is None:
            source_offset = next(
                (
                    offset
                    for argument in args
                    if (offset := self.alias_source_offset(argument)) is not None
                ),
                None,
            )
        candidates = [
            function
            for function in self.metal_user_function_overloads(
                function_name,
                source_offset=source_offset,
            )
            if len(getattr(function, "params", []) or []) == len(args)
        ]
        if not candidates:
            return "none", None

        argument_types = [self.expression_mapped_type(arg) for arg in args]
        if any(argument_type is None for argument_type in argument_types):
            return "unknown", None

        matching = [
            function
            for function in candidates
            if self.metal_function_mapped_signature(function) == tuple(argument_types)
        ]
        if not matching and allow_wave_lane_conversion:
            scored = [
                (score, function)
                for function in candidates
                if (
                    score := self.metal_wave_user_overload_match_score(
                        function, argument_types
                    )
                )
                is not None
            ]
            if scored:
                best_score = max(score for score, _function in scored)
                matching = [
                    function for score, function in scored if score == best_score
                ]
        if not matching:
            return "none", None

        source_groups = {}
        for function in matching:
            source_groups.setdefault(
                self.metal_function_source_signature(function), []
            ).append(function)
        if len(source_groups) > 1:
            candidate_names = [
                self.metal_function_candidate_signature(function)
                for functions in source_groups.values()
                for function in functions[:1]
            ]
            raise MetalBuiltinOverloadResolutionError(
                function_name,
                argument_types,
                candidate_names,
            )

        declarations = next(iter(source_groups.values()))
        function = next(
            (
                declaration
                for declaration in declarations
                if getattr(declaration, "body", None)
            ),
            declarations[0],
        )
        return "user", function

    def metal_wave_user_overload_match_score(self, function, argument_types):
        parameter_types = self.metal_function_mapped_signature(function)
        if not argument_types or len(parameter_types) != len(argument_types):
            return None
        if parameter_types[0] != argument_types[0]:
            return None

        score = 8
        for actual_type, parameter_type in zip(argument_types[1:], parameter_types[1:]):
            if actual_type == parameter_type:
                score += 8
                continue
            actual_info = self.metal_scalar_arithmetic_type_info(actual_type)
            parameter_info = self.metal_scalar_arithmetic_type_info(parameter_type)
            if (
                actual_info is None
                or parameter_info is None
                or actual_info[0] != "integer"
                or parameter_info[0] != "integer"
            ):
                return None
            score += 2
        return score

    def metal_function_mapped_signature(self, function):
        return tuple(
            self.map_type(self.resolve_type_alias(getattr(param, "vtype", None)))
            for param in getattr(function, "params", []) or []
        )

    def metal_function_source_signature(self, function):
        return tuple(
            self.normalized_metal_type(
                self.resolve_type_alias(getattr(param, "vtype", None))
            )
            for param in getattr(function, "params", []) or []
        )

    def metal_function_candidate_signature(self, function):
        parameters = ", ".join(self.metal_function_source_signature(function))
        return f"{function.name}({parameters})"

    def map_metal_type_constructor_name(self, name):
        text = str(name)
        refuse_unresolved_local_alias_type(self, text)
        decltype_type = self.resolve_metal_decltype_type(text)
        is_local_alias = text in self.local_type_alias_names
        is_type_alias = text in self.type_aliases
        is_materialized_type = any(
            text in bindings for bindings in self.template_type_bindings
        )
        instance = self.alias_template_instance_parts(text)
        if (
            decltype_type is None
            and "::" not in text
            and text not in self.unscoped_metal_type_constructors
            and self.metal_vector_type_parts(text) is None
            and not is_local_alias
            and not is_type_alias
            and not is_materialized_type
            and (
                instance is None
                or self.alias_template_declaration_for_name(instance[0]) is None
            )
        ):
            return None
        normalized = self.normalized_metal_type(
            decltype_type
            or self.substitute_template_type_text(self.resolve_type_alias(text))
        )
        mapped = self.map_type(normalized)
        if (
            decltype_type is not None
            or normalized in self.type_map
            or normalized in self.struct_name_map
            or mapped != normalized
            or generic_alias_constructor(self, text, mapped)
        ):
            return mapped
        return None

    def map_metal_math_function_name(self, name, args=None):
        name = str(name).lstrip(":")
        for prefix in self.metal_math_namespace_prefixes:
            if not str(name).startswith(prefix):
                continue
            unscoped = str(name)[len(prefix) :]
            if unscoped not in self.metal_math_intrinsics:
                continue
            if unscoped != "copysign" or args is None:
                return unscoped
            binding, function = self.resolve_metal_user_function_overload(
                str(name), args
            )
            if binding == "user":
                if self.is_materialized_metal_stdlib_wrapper(function):
                    return unscoped
                return self.function_output_name(function)
            if binding == "unknown":
                return None
            return unscoped
        return None

    def map_metal_precise_math_function_name(self, name, args=None):
        text = str(name)
        if self.metal_math_builtin_namespace_mode(text) != "precise":
            return None
        operation = text.rsplit("::", 1)[-1]
        if operation == "atan2":
            return self.map_metal_precise_atan2_function_name(args)
        if operation not in {
            "acos",
            "asin",
            "acosh",
            "atan",
            "sin",
            "cos",
            "exp",
            "log",
            "sqrt",
            "rsqrt",
        }:
            return None
        arguments = list(args or [])
        source_location = (
            getattr(arguments[0], "source_location", None) if arguments else None
        )
        if len(arguments) != 1:
            raise MetalPreciseMathLoweringError(
                operation,
                None,
                "the operation requires exactly one operand",
                source_location,
            )

        operand_type = self.expression_metal_type(arguments[0])
        type_info = self.metal_math_builtin_type_info(operand_type)
        if type_info is None:
            raise MetalPreciseMathLoweringError(
                operation,
                operand_type,
                "the operand type could not be inferred",
                source_location,
            )
        if type_info["category"] not in {"floating", "bfloat"}:
            raise MetalPreciseMathLoweringError(
                operation,
                operand_type,
                f"only floating-point operands have a precise {operation} contract",
                source_location,
            )
        width = type_info["width"]
        if not isinstance(width, int) or width not in {1, 2, 3, 4}:
            raise MetalPreciseMathLoweringError(
                operation,
                operand_type,
                "only scalar and two- to four-lane vectors are representable",
                source_location,
            )

        scalar_type = self.metal_scalar_arithmetic_type_info(type_info["element_type"])
        binary32_operand = scalar_type == ("floating", True, 32) or (
            width == 1
            and (
                type_info["category"] == "bfloat"
                or (scalar_type is not None and scalar_type[2] < 32)
            )
        )
        if operation in {"sin", "cos"}:
            if not binary32_operand:
                raise MetalPreciseMathLoweringError(
                    operation,
                    operand_type,
                    "precise trigonometric range reduction requires binary32 operands",
                    source_location,
                )
            self.required_metal_precise_trig_widths[operation].add(width)
            return self.metal_precise_trig_helper_name(operation, width)
        if operation == "asin":
            self.required_metal_precise_asin_widths.add(width)
            return self.metal_precise_asin_helper_name(width)
        if operation == "atan":
            if not binary32_operand:
                raise MetalPreciseMathLoweringError(
                    operation,
                    operand_type,
                    "precise arctangent lowering requires binary32 operands",
                    source_location,
                )
            self.required_metal_precise_atan_widths.add(width)
            return self.metal_precise_atan_helper_name(width)
        if operation == "exp":
            if not binary32_operand:
                raise MetalPreciseMathLoweringError(
                    operation,
                    operand_type,
                    "precise exponential lowering requires binary32 operands",
                    source_location,
                )
            self.required_metal_precise_exp_widths.add(width)
            return self.metal_precise_exp_helper_name(width)
        if operation == "log":
            if not binary32_operand:
                raise MetalPreciseMathLoweringError(
                    operation,
                    operand_type,
                    "precise logarithm lowering requires binary32 operands",
                    source_location,
                )
            if self.current_function is None:
                raise MetalPreciseMathLoweringError(
                    operation,
                    operand_type,
                    "global initializers cannot call the runtime logarithm helper",
                    source_location,
                )
            self.required_metal_precise_log_widths.add(width)
            return self.metal_precise_log_helper_name(width)
        if operation == "sqrt":
            if not binary32_operand:
                raise MetalPreciseMathLoweringError(
                    operation,
                    operand_type,
                    "precise square-root lowering requires binary32 operands",
                    source_location,
                )
            if self.current_function is None:
                raise MetalPreciseMathLoweringError(
                    operation,
                    operand_type,
                    "global initializers cannot call the runtime square-root helper",
                    source_location,
                )
            self.required_metal_precise_sqrt_widths.add(width)
            return self.metal_precise_sqrt_helper_name(width)
        if operation == "rsqrt":
            if not binary32_operand:
                raise MetalPreciseMathLoweringError(
                    operation,
                    operand_type,
                    "precise reciprocal-square-root lowering requires binary32 operands",
                    source_location,
                )
            if self.current_function is None:
                raise MetalPreciseMathLoweringError(
                    operation,
                    operand_type,
                    "global initializers cannot call the runtime reciprocal-square-root helper",
                    source_location,
                )
            self.required_metal_precise_rsqrt_widths.add(width)
            return self.metal_precise_rsqrt_helper_name(width)
        if operation == "acosh":
            if not binary32_operand:
                raise MetalPreciseMathLoweringError(
                    operation,
                    operand_type,
                    "precise inverse hyperbolic lowering requires binary32 operands",
                    source_location,
                )
            self.required_metal_precise_acosh_widths.add(width)
            return self.metal_precise_acosh_helper_name(width)
        self.required_metal_precise_acos_widths.add(width)
        return self.metal_precise_acos_helper_name(width)

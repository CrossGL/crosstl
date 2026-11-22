"""Out-of-line call operators and lowered struct methods."""

import re

from ..MetalAst import FunctionNode
from .support import (
    MetalOutOfLineCallOperatorLoweringError,
    MetalStructMethodCallResolutionError,
)


class MetalStructMethodMixin:
    """Members of ``MetalToCrossGLConverter``: out-of-line call operators and lowered struct methods."""

    def prepare_out_of_line_call_operator_bindings(self, functions):
        """Bind preserved qualified bodies to existing lowered receiver helpers."""
        replacements = {}
        suppressed = set()
        for definition in functions or []:
            declaration = getattr(
                definition, "out_of_line_call_operator_declaration", None
            )
            owner = getattr(definition, "out_of_line_call_operator_owner", None)
            if not declaration or not owner:
                continue

            candidates = [
                function
                for function in functions or []
                if function is not definition
                and self.out_of_line_call_operator_helper_matches(
                    definition, function, owner
                )
            ]
            if not candidates:
                continue
            if len(candidates) != 1:
                raise MetalOutOfLineCallOperatorLoweringError(
                    owner,
                    declaration.get("signature", "operator()"),
                    tuple(
                        self.metal_source_overload_candidate_signature(candidate)
                        for candidate in candidates
                    ),
                    "multiple lowered helpers match the declaration contract",
                    declaration_location=declaration.get("source_location"),
                    definition_location=getattr(
                        definition, "declaration_source_location", None
                    ),
                    candidate_locations=tuple(
                        getattr(candidate, "declaration_source_location", None)
                        for candidate in candidates
                    ),
                )

            helper = candidates[0]
            existing = replacements.get(id(helper))
            if existing is not None:
                raise MetalOutOfLineCallOperatorLoweringError(
                    owner,
                    declaration.get("signature", "operator()"),
                    (self.metal_source_overload_candidate_signature(helper),),
                    "multiple qualified definitions target the same lowered helper",
                    declaration_location=declaration.get("source_location"),
                    definition_location=getattr(
                        definition, "declaration_source_location", None
                    ),
                    candidate_locations=(
                        getattr(helper, "declaration_source_location", None),
                    ),
                )

            helper_parameters = list(getattr(helper, "params", []) or [])
            transport_count = len(self.lowered_method_transport_parameters(helper))
            explicit_parameters = helper_parameters[1 + transport_count :]
            parameter_aliases = tuple(
                (definition_parameter.name, helper_parameter.name)
                for definition_parameter, helper_parameter in zip(
                    getattr(definition, "params", []) or [], explicit_parameters
                )
                if getattr(definition_parameter, "name", None)
                and getattr(helper_parameter, "name", None)
            )
            replacements[id(helper)] = {
                "definition": definition,
                "parameter_aliases": parameter_aliases,
            }
            suppressed.add(id(definition))

        self.out_of_line_call_operator_replacements = replacements
        self.suppressed_out_of_line_call_operator_ids = suppressed

    def out_of_line_call_operator_helper_matches(self, definition, helper, owner):
        if (
            not isinstance(helper, FunctionNode)
            or getattr(helper, "body", None) is None
        ):
            return False
        params = list(getattr(helper, "params", []) or [])
        if not params:
            return False
        receiver = params[0]
        if getattr(receiver, "name", None) != "self" or not self.reference_parameter(
            receiver
        ):
            return False

        receiver_owner = self.normalized_metal_type(
            self.resolve_type_alias(getattr(receiver, "vtype", None))
        )
        owner_name = self.normalized_metal_type(self.resolve_type_alias(owner))
        unqualified_owner = owner_name.rsplit("::", 1)[-1]
        if receiver_owner not in {owner_name, unqualified_owner}:
            return False

        helper_name = str(getattr(helper, "name", ""))
        helper_prefixes = {
            f"{owner_name}__operator_call",
            f"{unqualified_owner}__operator_call",
            f"{self.sanitize_identifier(owner_name)}__operator_call",
        }
        if not any(
            helper_name == prefix or helper_name.startswith(f"{prefix}__")
            for prefix in helper_prefixes
        ):
            return False

        definition_const = "const" in {
            str(qualifier).lower()
            for qualifier in getattr(definition, "method_qualifiers", []) or []
        }
        receiver_const = self.readonly_parameter(
            receiver, self.effective_declaration_qualifiers(receiver)
        )
        if definition_const != receiver_const:
            return False

        transport_count = len(self.lowered_method_transport_parameters(helper))
        explicit_parameters = params[1 + transport_count :]
        definition_parameters = list(getattr(definition, "params", []) or [])
        if len(explicit_parameters) != len(definition_parameters):
            return False
        if tuple(
            self.normalized_metal_parameter_type(parameter)
            for parameter in explicit_parameters
        ) != tuple(
            self.normalized_metal_parameter_type(parameter)
            for parameter in definition_parameters
        ):
            return False

        helper_return = self.normalized_metal_type(
            self.resolve_type_alias(getattr(helper, "return_type", None))
        )
        definition_return = self.normalized_metal_type(
            self.resolve_type_alias(getattr(definition, "return_type", None))
        )
        return helper_return == definition_return

    def lowered_struct_method_context(self, function):
        """Return the concrete receiver contract for a lowered instance helper."""
        if not isinstance(function, FunctionNode):
            return None
        if getattr(function, "is_metal_constructor_factory", False):
            owner = self.normalized_metal_type(
                self.resolve_type_alias(getattr(function, "constructor_owner", None))
            )
            if not owner:
                return None
            return {
                "owner": owner,
                "helper_prefix": self.map_struct_name(owner),
                "receiver": None,
                "constructor_factory": True,
            }
        params = list(getattr(function, "params", []) or [])
        if not params:
            return None
        receiver = params[0]
        if getattr(receiver, "name", None) != "self" or not self.reference_parameter(
            receiver
        ):
            return None

        owner = self.normalized_metal_type(
            self.resolve_type_alias(getattr(receiver, "vtype", None))
        )
        if not owner or (
            owner not in self.struct_declarations
            and owner not in self.struct_name_map
            and owner not in self.struct_name_map.values()
        ):
            return None

        helper_name = str(getattr(function, "name", ""))
        prefixes = list(
            dict.fromkeys(
                (owner, self.map_struct_name(owner), self.sanitize_identifier(owner))
            )
        )
        helper_prefix = next(
            (prefix for prefix in prefixes if helper_name.startswith(f"{prefix}__")),
            None,
        )
        if helper_prefix is None:
            shortened_prefixes = [
                helper_name[: match.start()]
                for match in re.finditer(r"__", helper_name)
                if match.start()
            ]
            helper_prefix = next(
                (
                    candidate
                    for candidate in reversed(shortened_prefixes)
                    if any(
                        concrete.startswith(f"{candidate}_") for concrete in prefixes
                    )
                ),
                None,
            )
        if helper_prefix is None:
            return None
        return {
            "owner": owner,
            "helper_prefix": helper_prefix,
            "receiver": receiver,
            "constructor_factory": False,
        }

    @staticmethod
    def lowered_method_transport_parameters(function):
        transported = []
        for param in list(getattr(function, "params", []) or [])[1:]:
            if not str(getattr(param, "name", "")).startswith("crosstl_ptr_"):
                break
            transported.append(param)
        return transported

    def lowered_method_parameter_contract(self, parameter):
        metal_type = self.resolve_type_alias(
            self.metal_declaration_expression_type(parameter)
        )
        qualifiers = tuple(self.effective_declaration_qualifiers(parameter))
        return re.sub(r"\s+", " ", str(metal_type).strip()), qualifiers

    def lowered_method_explicit_parameter_types(self, function, transport_count):
        params = list(getattr(function, "params", []) or [])
        explicit = params[1 + transport_count :]
        mapped = tuple(
            self.map_type(
                self.resolve_type_alias(self.metal_declaration_expression_type(param))
            )
            for param in explicit
        )
        source = tuple(
            self.normalized_metal_parameter_type(param) for param in explicit
        )
        return mapped, source

    def normalized_metal_parameter_type(self, parameter):
        return re.sub(
            r"\s+",
            " ",
            str(
                self.resolve_type_alias(
                    self.metal_declaration_expression_type(parameter)
                )
            ).strip(),
        )

    def lowered_method_candidate_signature(self, function, transport_count):
        _mapped, source = self.lowered_method_explicit_parameter_types(
            function, transport_count
        )
        return f"{function.name}({', '.join(source)})"

    def lowered_static_struct_method_owner(self, function):
        """Return the concrete owner encoded by one lowered static helper."""
        if not isinstance(function, FunctionNode):
            return None
        if getattr(function, "is_metal_constructor_factory", False):
            owner = self.normalized_metal_type(
                self.resolve_type_alias(getattr(function, "constructor_owner", None))
            )
            return owner or None

        instance_context = self.lowered_struct_method_context(function)
        if instance_context is not None:
            return instance_context["owner"]

        function_name = str(getattr(function, "name", ""))
        matches = []
        for owner in self.struct_name_map:
            prefixes = {
                owner,
                self.map_struct_name(owner),
                self.sanitize_identifier(owner),
            }
            matched_prefix = next(
                (
                    prefix
                    for prefix in sorted(prefixes, key=len, reverse=True)
                    if prefix and function_name.startswith(f"{prefix}__")
                ),
                None,
            )
            if matched_prefix is not None:
                matches.append((len(matched_prefix), owner))
        if not matches:
            return None
        longest = max(length for length, _owner in matches)
        owners = {owner for length, owner in matches if length == longest}
        return next(iter(owners)) if len(owners) == 1 else None

    def resolve_lowered_static_struct_owner(self, owner_name, lexical_owner):
        """Resolve a scoped static-call owner, including owner-local aliases."""
        candidate = self.normalize_qualified_type_name(owner_name).lstrip(":")
        if lexical_owner:
            resolved_alias = self.resolve_dependent_alias_type(
                lexical_owner,
                candidate,
                required=False,
            )
            if resolved_alias is not None:
                candidate = resolved_alias
        candidate = self.normalized_metal_type(
            self.resolve_type_alias(
                self.materialize_alias_template_type(candidate, required=False)
            )
        )
        if candidate in self.struct_name_map:
            return candidate
        reverse_names = {
            mapped: owner for owner, mapped in self.struct_name_map.items()
        }
        # A renamed structure resolves to its source name; any other owner keeps the spelling resolved above.
        if candidate in reverse_names:
            return reverse_names[candidate]
        return candidate

    def lowered_static_method_candidate_signature(self, function):
        parameters = ", ".join(
            self.normalized_metal_parameter_type(parameter)
            for parameter in getattr(function, "params", []) or []
        )
        return f"{function.name}({parameters})"

    def resolve_lowered_static_struct_method_call(self, expression):
        """Bind a retained static struct call to its concrete free helper."""
        raw_name = self.normalize_qualified_type_name(getattr(expression, "name", ""))
        lexical_owner = self.lowered_static_struct_method_owner(self.current_function)
        if not raw_name or lexical_owner is None:
            return None

        if "::" in raw_name:
            owner_name, method_name = raw_name.rsplit("::", 1)
            if not self.crossgl_identifier_pattern.fullmatch(method_name):
                return None
            # A real qualified free function remains authoritative.
            if self.metal_user_function_overloads(raw_name) or (
                self.metal_source_overload_groups_for_name(raw_name)
            ):
                return None
            owner = self.resolve_lowered_static_struct_owner(
                owner_name,
                lexical_owner,
            )
        else:
            method_name = raw_name
            if not self.crossgl_identifier_pattern.fullmatch(method_name):
                return None
            # Preserve lexical values/callables and ordinary free functions.
            if (
                method_name in self.current_variable_types
                or method_name in self.global_variable_types
                or method_name in self.user_function_overloads_by_name
                or self.metal_source_overload_groups_for_name(method_name)
            ):
                return None
            owner = lexical_owner

        helper_name = f"{self.map_struct_name(owner)}__{method_name}"
        all_candidates = [
            candidate
            for candidate in self.user_function_overloads_by_name.get(helper_name, [])
            if not self.lowered_static_method_is_instance_helper(candidate)
        ]
        if not all_candidates:
            return None

        arity_candidates = [
            candidate
            for candidate in all_candidates
            if len(getattr(candidate, "params", []) or []) == len(expression.args)
        ]
        argument_types = tuple(
            self.metal_source_overload_value_type(self.expression_metal_type(argument))
            for argument in expression.args
        )
        diagnostic_argument_types = tuple(
            argument_type or "<unknown>" for argument_type in argument_types
        )
        candidate_signatures = tuple(
            self.lowered_static_method_candidate_signature(candidate)
            for candidate in all_candidates
        )
        if not arity_candidates:
            raise MetalStructMethodCallResolutionError(
                owner,
                method_name,
                diagnostic_argument_types,
                candidate_signatures,
                "no static helper has the required argument count",
                getattr(expression, "source_location", None),
            )
        if any(argument_type is None for argument_type in argument_types):
            raise MetalStructMethodCallResolutionError(
                owner,
                method_name,
                diagnostic_argument_types,
                candidate_signatures,
                "one or more static-helper argument types could not be inferred",
                getattr(expression, "source_location", None),
            )

        source_groups = {}
        for candidate in arity_candidates:
            source_groups.setdefault(
                tuple(
                    self.normalized_metal_parameter_type(parameter)
                    for parameter in getattr(candidate, "params", []) or []
                ),
                [],
            ).append(candidate)

        ranked = []
        for source_signature, declarations in source_groups.items():
            parameters = list(getattr(declarations[0], "params", []) or [])
            ranks = []
            for argument, argument_type, parameter in zip(
                expression.args,
                argument_types,
                parameters,
            ):
                rank = self.metal_source_overload_argument_match_rank(
                    argument,
                    argument_type,
                    parameter,
                )
                if rank is None:
                    break
                ranks.append(rank)
            else:
                ranked.append((tuple(ranks), source_signature, declarations))

        if not ranked:
            raise MetalStructMethodCallResolutionError(
                owner,
                method_name,
                diagnostic_argument_types,
                candidate_signatures,
                "no source-compatible static-helper overload matches",
                getattr(expression, "source_location", None),
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
            raise MetalStructMethodCallResolutionError(
                owner,
                method_name,
                diagnostic_argument_types,
                tuple(
                    self.lowered_static_method_candidate_signature(declarations[0])
                    for _ranks, _source, declarations in winners
                ),
                "multiple source-compatible static-helper overloads remain",
                getattr(expression, "source_location", None),
            )

        declarations = winners[0][2]
        selected = next(
            (
                candidate
                for candidate in declarations
                if getattr(candidate, "body", None)
            ),
            declarations[0],
        )
        if self.reference_element_type(getattr(selected, "return_type", None)):
            raise MetalStructMethodCallResolutionError(
                owner,
                method_name,
                diagnostic_argument_types,
                (self.lowered_static_method_candidate_signature(selected),),
                "reference-returning static helpers require lvalue-preserving lowering",
                getattr(expression, "source_location", None),
            )
        return selected

    def generate_lowered_static_struct_method_call(self, expression, is_main=False):
        selected = self.resolve_lowered_static_struct_method_call(expression)
        if selected is None:
            return None
        function_name = self.sanitize_identifier(self.function_output_name(selected))
        arguments = ", ".join(
            self.generate_metal_value_argument(argument, parameter, is_main)
            for argument, parameter in zip(expression.args, selected.params)
        )
        return f"{function_name}({arguments})"

    def resolve_lowered_struct_method_call(self, expression):
        """Bind one retained implicit-this call to a concrete lowered helper."""
        context = self.lowered_struct_method_context(self.current_function)
        if context is None:
            return None

        method_name = str(getattr(expression, "name", ""))
        if not self.crossgl_identifier_pattern.fullmatch(method_name):
            return None
        # A value/callable or an ordinary free function with this name wins over
        # the structural lowered-helper inference.
        if (
            method_name in self.current_variable_types
            or method_name in self.global_variable_types
            or method_name in self.user_function_overloads_by_name
        ):
            return None

        helper_name = f"{context['helper_prefix']}__{method_name}"
        all_candidates = list(self.user_function_overloads_by_name.get(helper_name, []))
        if not all_candidates:
            return None

        current_transport = (
            []
            if context["constructor_factory"]
            else self.lowered_method_transport_parameters(self.current_function)
        )
        current_transport_names = [param.name for param in current_transport]
        current_transport_contracts = {
            param.name: self.lowered_method_parameter_contract(param)
            for param in current_transport
        }
        current_receiver_readonly = (
            False
            if context["constructor_factory"]
            else self.readonly_parameter(
                context["receiver"],
                self.effective_declaration_qualifiers(context["receiver"]),
            )
        )
        argument_types = tuple(
            self.expression_mapped_type(argument) for argument in expression.args
        )
        diagnostic_argument_types = tuple(
            argument_type or "<unknown>" for argument_type in argument_types
        )

        viable = []
        rejected_receiver = False
        for candidate in all_candidates:
            candidate_context = self.lowered_struct_method_context(candidate)
            params = list(getattr(candidate, "params", []) or [])
            if (
                candidate_context is None
                or candidate_context["owner"] != context["owner"]
                or len(params) != 1 + len(current_transport) + len(expression.args)
            ):
                continue
            candidate_transport = params[1 : 1 + len(current_transport)]
            candidate_transport_names = [param.name for param in candidate_transport]
            if candidate_transport_names != current_transport_names:
                continue
            if any(
                self.lowered_method_parameter_contract(param)
                != current_transport_contracts.get(param.name)
                for param in candidate_transport
            ):
                continue
            if context["constructor_factory"]:
                receiver_qualifiers = set(
                    self.metal_declaration_type_qualifiers(
                        candidate_context["receiver"]
                    )
                )
                candidate_spaces = (
                    receiver_qualifiers & self.metal_source_overload_address_spaces
                )
                candidate_space = next(iter(candidate_spaces), "thread")
                if candidate_space != self.current_constructor_receiver_address_space():
                    continue
            candidate_readonly = self.readonly_parameter(
                candidate_context["receiver"],
                self.effective_declaration_qualifiers(candidate_context["receiver"]),
            )
            if current_receiver_readonly and not candidate_readonly:
                rejected_receiver = True
                continue
            mapped_types, source_types = self.lowered_method_explicit_parameter_types(
                candidate, len(current_transport)
            )
            viable.append((candidate, mapped_types, source_types, candidate_readonly))

        candidate_signatures = tuple(
            self.lowered_method_candidate_signature(candidate, len(current_transport))
            for candidate in all_candidates
        )
        if not viable:
            reason = (
                "a read-only receiver cannot call a mutating sibling helper"
                if rejected_receiver
                else "no helper preserves the receiver and transported-resource contract"
            )
            raise MetalStructMethodCallResolutionError(
                context["owner"],
                method_name,
                diagnostic_argument_types,
                candidate_signatures,
                reason,
                getattr(expression, "source_location", None),
            )

        if any(argument_type is None for argument_type in argument_types):
            matches = viable
        else:
            matches = [
                candidate for candidate in viable if candidate[1] == argument_types
            ]
            if not matches:
                raise MetalStructMethodCallResolutionError(
                    context["owner"],
                    method_name,
                    diagnostic_argument_types,
                    tuple(
                        self.lowered_method_candidate_signature(
                            candidate, len(current_transport)
                        )
                        for candidate, _mapped, _source, _readonly in viable
                    ),
                    "no exact overload matches the available argument types",
                    getattr(expression, "source_location", None),
                )

        source_groups = {}
        for candidate in matches:
            source_groups.setdefault((candidate[2], candidate[3]), []).append(candidate)
        if len(source_groups) != 1:
            raise MetalStructMethodCallResolutionError(
                context["owner"],
                method_name,
                diagnostic_argument_types,
                tuple(
                    self.lowered_method_candidate_signature(
                        candidates[0][0], len(current_transport)
                    )
                    for candidates in source_groups.values()
                ),
                "multiple exact overloads remain after type matching",
                getattr(expression, "source_location", None),
            )

        declarations = next(iter(source_groups.values()))
        selected = next(
            (
                candidate
                for candidate, _mapped, _source, _readonly in declarations
                if getattr(candidate, "body", None)
            ),
            declarations[0][0],
        )
        if self.reference_element_type(getattr(selected, "return_type", None)):
            raise MetalStructMethodCallResolutionError(
                context["owner"],
                method_name,
                diagnostic_argument_types,
                (
                    self.lowered_method_candidate_signature(
                        selected, len(current_transport)
                    ),
                ),
                "reference-returning sibling helpers require lvalue-preserving lowering",
                getattr(expression, "source_location", None),
            )
        return selected, current_transport

    def generate_lowered_struct_method_call(self, expression, is_main=False):
        resolved = self.resolve_lowered_struct_method_call(expression)
        if resolved is None:
            return None
        selected, transported = resolved
        if getattr(self.current_function, "is_metal_constructor_factory", False):
            receiver = self.render_identifier(
                self.current_function.constructor_result_name
            )
        else:
            receiver = self.render_identifier("self")
        arguments = [receiver]
        arguments.extend(self.render_identifier(param.name) for param in transported)
        arguments.extend(
            self.generate_metal_value_argument(argument, parameter, is_main)
            for argument, parameter in zip(
                expression.args, selected.params[1 + len(transported) :]
            )
        )
        function_name = self.sanitize_identifier(self.function_output_name(selected))
        return f"{function_name}({', '.join(arguments)})"

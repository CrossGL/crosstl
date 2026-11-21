"""Static aggregate evaluation, boolean variable templates and free operator overload materialization."""

import re
from typing import (
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
)

from ..concrete_free_operators import materialize_concrete_free_operators
from ..integral_constant_locals import (
    fold_concrete_function_integral_constant_arithmetic,
    is_folded_integral_constant_operator,
)
from ..static_assertion_sources import concrete_static_assertion_conditions
from ..value_traits import (
    complete_template_arguments,
    enable_if_parts,
    struct_value_trait_definitions,
)
from .support import (
    _MetalBooleanVariableTemplate,
    _MetalConstexprAggregateValue,
    _MetalConstexprBinaryOperator,
    _MetalConstexprScalarValue,
    _MetalFreeOperatorDefinition,
    _MetalNamespaceVisibility,
    _MetalStructDefinition,
    _MetalTemplateStruct,
    _MetalTypeAliasBinding,
    _strip_global_metal_qualifiers,
)

_TEMPLATE_HEAD_RE = re.compile(r"\btemplate\s*<")
_FREE_OPERATOR_NAME_RE = re.compile(
    r"\boperator\s*(?P<operator>==|!=|<=|>=|[+\-*/%]=|[+\-*/%<>])\s*\("
)
# A free operator's name follows its template head before any of these characters.
_DECLARATION_BREAK_RE = re.compile(r"[;{}]")


class MetalStaticAggregateMixin:
    """Members of ``MetalPreprocessor``: static aggregate evaluation, boolean variable templates and free operator overload materialization."""

    def _evaluate_static_aggregate_assertion(
        self,
        expression: str,
        *,
        operators: Sequence[_MetalConstexprBinaryOperator],
        structs: List[_MetalStructDefinition],
        template_structs: Sequence[_MetalTemplateStruct],
        explicit_template_specializations: Set[Tuple[str, Tuple[str, ...]]],
        type_aliases: Dict[str, List[_MetalTypeAliasBinding]],
        position: int,
    ) -> Tuple[bool, Optional[int]]:
        comparison = self._split_static_top_level_operator(
            expression,
            ("==", "!="),
        )
        if comparison is None:
            return False, None
        left_text, comparison_operator, right_text = comparison
        left = self._evaluate_static_aggregate_member_expression(
            left_text,
            operators=operators,
            structs=structs,
            template_structs=template_structs,
            explicit_template_specializations=explicit_template_specializations,
            type_aliases=type_aliases,
            position=position,
        )
        right = self._evaluate_static_aggregate_member_expression(
            right_text,
            operators=operators,
            structs=structs,
            template_structs=template_structs,
            explicit_template_specializations=explicit_template_specializations,
            type_aliases=type_aliases,
            position=position,
        )
        if left is None and right is None:
            return False, None
        if left is None:
            if right is None:
                return False, None
            left = self._evaluate_static_scalar_expression(
                left_text,
                expected_type=right.type_name,
            )
        elif right is None:
            right = self._evaluate_static_scalar_expression(
                right_text,
                expected_type=left.type_name,
            )
        if left is None or right is None or left.type_name != right.type_name:
            return False, None
        equal = left.value == right.value
        return True, int(equal if comparison_operator == "==" else not equal)

    def _evaluate_static_aggregate_member_expression(
        self,
        expression: str,
        *,
        operators: Sequence[_MetalConstexprBinaryOperator],
        structs: List[_MetalStructDefinition],
        template_structs: Sequence[_MetalTemplateStruct],
        explicit_template_specializations: Set[Tuple[str, Tuple[str, ...]]],
        type_aliases: Dict[str, List[_MetalTypeAliasBinding]],
        position: int,
    ) -> Optional[_MetalConstexprScalarValue]:
        text = self._strip_static_outer_parentheses(expression)
        if not text.startswith("("):
            return None
        close_paren = self._find_matching_delimiter(text, 0, "(", ")")
        if close_paren is None:
            return None
        member_match = re.fullmatch(
            r"\s*\.\s*([A-Za-z_][A-Za-z0-9_]*)\s*",
            text[close_paren + 1 :],
        )
        if member_match is None:
            return None
        operation = self._split_top_level_binary_arithmetic(text[1:close_paren].strip())
        if operation is None:
            return None
        left_text, binary_operator, right_text = operation
        left = self._evaluate_static_aggregate_constructor(
            left_text,
            structs=structs,
            template_structs=template_structs,
            explicit_template_specializations=explicit_template_specializations,
            type_aliases=type_aliases,
            position=position,
        )
        right = self._evaluate_static_aggregate_constructor(
            right_text,
            structs=structs,
            template_structs=template_structs,
            explicit_template_specializations=explicit_template_specializations,
            type_aliases=type_aliases,
            position=position,
        )
        if left is None or right is None:
            return None
        result = self._evaluate_static_constexpr_binary_operator(
            binary_operator,
            left,
            right,
            operators=operators,
            structs=structs,
            template_structs=template_structs,
            explicit_template_specializations=explicit_template_specializations,
            type_aliases=type_aliases,
            position=position,
        )
        return None if result is None else result.member(member_match.group(1))

    def _evaluate_static_constexpr_binary_operator(
        self,
        binary_operator: str,
        left: _MetalConstexprAggregateValue,
        right: _MetalConstexprAggregateValue,
        *,
        operators: Sequence[_MetalConstexprBinaryOperator],
        structs: List[_MetalStructDefinition],
        template_structs: Sequence[_MetalTemplateStruct],
        explicit_template_specializations: Set[Tuple[str, Tuple[str, ...]]],
        type_aliases: Dict[str, List[_MetalTypeAliasBinding]],
        position: int,
    ) -> Optional[_MetalConstexprAggregateValue]:
        results: List[_MetalConstexprAggregateValue] = []
        for candidate in operators:
            if candidate.operator != binary_operator:
                continue
            template_parameters = set(candidate.template_parameters)
            bindings: Dict[str, str] = {}
            compatible = True
            for declared_type, value in zip(
                candidate.parameter_types,
                (left, right),
            ):
                self._infer_template_parameter_bindings_from_type(
                    declared_type,
                    value.type_name,
                    template_parameters,
                    bindings,
                )
                concrete_declared_type = self._substitute_static_type_bindings(
                    declared_type,
                    bindings,
                )
                if self._canonical_static_aggregate_type(
                    concrete_declared_type
                ) != self._canonical_static_aggregate_type(value.type_name):
                    compatible = False
                    break
            if not compatible or not template_parameters <= set(bindings):
                continue

            concrete_return_type = self._substitute_static_type_bindings(
                candidate.return_type,
                bindings,
            )
            canonical_return_type = self._canonical_static_aggregate_type(
                concrete_return_type
            )
            result_type_candidates = {
                value.type_name
                for value in (left, right)
                if self._canonical_static_aggregate_type(value.type_name)
                == canonical_return_type
            }
            if len(result_type_candidates) != 1:
                continue
            return_expression = self._replace_identifiers(
                candidate.return_expression,
                bindings,
            ).strip()
            if not return_expression.startswith("{"):
                continue
            close_brace = self._find_matching_delimiter(
                return_expression,
                0,
                "{",
                "}",
            )
            if close_brace != len(return_expression) - 1:
                continue
            arguments = self._split_top_level_commas(return_expression[1:close_brace])
            result = self._construct_static_aggregate(
                next(iter(result_type_candidates)),
                arguments,
                structs=structs,
                template_structs=template_structs,
                explicit_template_specializations=explicit_template_specializations,
                type_aliases=type_aliases,
                position=position,
                aggregate_bindings={
                    candidate.parameter_names[0]: left,
                    candidate.parameter_names[1]: right,
                },
            )
            if result is not None:
                results.append(result)
        return results[0] if len(results) == 1 else None

    def _evaluate_static_aggregate_constructor(
        self,
        expression: str,
        *,
        structs: List[_MetalStructDefinition],
        template_structs: Sequence[_MetalTemplateStruct],
        explicit_template_specializations: Set[Tuple[str, Tuple[str, ...]]],
        type_aliases: Dict[str, List[_MetalTypeAliasBinding]],
        position: int,
    ) -> Optional[_MetalConstexprAggregateValue]:
        text = self._strip_static_outer_parentheses(expression)
        open_brace = self._find_next_top_level_char(text, 0, "{")
        if open_brace is None:
            return None
        type_name = text[:open_brace].strip()
        if (
            re.fullmatch(
                r"(?:::)?[A-Za-z_][A-Za-z0-9_]*(?:::[A-Za-z_][A-Za-z0-9_]*)*",
                type_name,
            )
            is None
            and self._exact_template_type_id(type_name) is None
        ):
            return None
        close_brace = self._find_matching_delimiter(text, open_brace, "{", "}")
        if close_brace != len(text) - 1:
            return None
        return self._construct_static_aggregate(
            type_name,
            self._split_top_level_commas(text[open_brace + 1 : close_brace]),
            structs=structs,
            template_structs=template_structs,
            explicit_template_specializations=explicit_template_specializations,
            type_aliases=type_aliases,
            position=position,
        )

    def _concrete_template_struct_for_static_evaluation(
        self,
        type_name: str,
        *,
        template_structs: Sequence[_MetalTemplateStruct],
        explicit_template_specializations: Set[Tuple[str, Tuple[str, ...]]],
    ) -> Optional[Tuple[_MetalStructDefinition, int]]:
        template_id = self._exact_template_type_id(type_name)
        if template_id is None:
            return None
        template_name, template_arguments = template_id
        requested_name = template_name.lstrip(":")
        candidates = [
            template
            for template in template_structs
            if requested_name
            in {
                template.name,
                (
                    f"{template.namespace}::{template.name}"
                    if template.namespace
                    else template.name
                ).lstrip("::"),
            }
        ]
        # This evaluator proves source semantics; it must not approximate C++
        # partial/explicit-specialization ordering or choose among duplicate
        # declarations. Keep those cases unresolved instead.
        if len(candidates) != 1:
            return None
        template = candidates[0]
        if (
            self._template_struct_specialization_arguments(template) is not None
            or template.variadic_template_parameters
            or any(
                specialization_name == template.name
                for specialization_name, _arguments in explicit_template_specializations
            )
            or len(template_arguments) > len(template.template_parameters)
        ):
            return None

        concrete_arguments = self._template_arguments_with_resolved_defaults(
            template,
            template_arguments,
        )
        if concrete_arguments is None or len(concrete_arguments) != len(
            template.template_parameters
        ):
            return None
        substitutions = dict(zip(template.template_parameters, concrete_arguments))
        for parameter, declared_type in template.template_parameter_types.items():
            integral_type = self._static_constexpr_integral_type(
                self._replace_identifiers(declared_type, substitutions)
            )
            if integral_type is None:
                return None
            value = self._proven_integral_constant_value(
                integral_type,
                self._replace_identifiers(substitutions[parameter], substitutions),
            )
            if value is None:
                return None
            substitutions[parameter] = value

        concrete_arguments = [
            self._normalize_template_argument_text(substitutions[parameter])
            for parameter in template.template_parameters
        ]
        if any(
            not argument
            or self._type_references_template_parameter(
                argument,
                template.template_parameters,
            )
            for argument in concrete_arguments
        ):
            return None

        qualified_name = (
            f"{template.namespace}::{template.name}"
            if template.namespace
            else template.name
        )
        probe_name = self._template_specialization_identifier(
            f"__crosstl_static_{qualified_name}",
            concrete_arguments,
        )
        materialized = self._materialize_template_struct_with_name(
            template,
            concrete_arguments,
            probe_name,
        )
        if not materialized:
            return None
        materialized_structs = self._find_concrete_struct_definitions(materialized)
        if len(materialized_structs) != 1 or materialized_structs[0].name != probe_name:
            return None
        probe = materialized_structs[0]
        # Static aggregate evaluation models a plain, standard-layout struct.
        # Classes can carry inaccessible members, bases alter initialization and
        # object layout, unions overlap storage, and an unsupported parser layout
        # means the captured member list is incomplete. Never prove a source
        # assertion from any of those approximations.
        if (
            not probe.layout_supported
            or probe.base_clause
            or probe.aggregate_kind != "struct"
        ):
            return None
        # Non-dependent aliases in a template are bound at its declaration, not
        # at the later static_assert that requested this synthetic specialization.
        probe.span = template.span
        return probe, template.span[0]

    def _construct_static_aggregate(
        self,
        type_name: str,
        arguments: Sequence[str],
        *,
        structs: List[_MetalStructDefinition],
        template_structs: Sequence[_MetalTemplateStruct],
        explicit_template_specializations: Set[Tuple[str, Tuple[str, ...]]],
        type_aliases: Dict[str, List[_MetalTypeAliasBinding]],
        position: int,
        aggregate_bindings: Optional[
            Mapping[str, _MetalConstexprAggregateValue]
        ] = None,
    ) -> Optional[_MetalConstexprAggregateValue]:
        resolved_type = self._resolve_type_aliases_at(
            type_name,
            type_aliases,
            position,
        )
        normalized_type = self._normalize_inferred_type(resolved_type)
        matching_structs = [
            struct
            for struct in structs
            if normalized_type in {struct.name, struct.qualified_name}
        ]
        struct_position = position
        if len(matching_structs) == 1:
            struct = matching_structs[0]
        elif matching_structs:
            return None
        else:
            concrete_template = self._concrete_template_struct_for_static_evaluation(
                normalized_type,
                template_structs=template_structs,
                explicit_template_specializations=(explicit_template_specializations),
            )
            if concrete_template is None:
                return None
            struct, struct_position = concrete_template
        if not struct.data_members or any(
            member.array_suffix or member.is_pointer for member in struct.data_members
        ):
            return None

        results: List[_MetalConstexprAggregateValue] = []
        for constructor in struct.constructors:
            if (
                not constructor.is_constexpr
                or constructor.template_parameters
                or len(constructor.param_names) != len(arguments)
                or len(constructor.param_types) != len(arguments)
                or (
                    constructor.receiver_address_spaces
                    and "thread" not in constructor.receiver_address_spaces
                )
                or self._strip_template_argument_comments(constructor.body).strip()
            ):
                continue
            parameter_values: Dict[str, _MetalConstexprScalarValue] = {}
            for name, declared_type, argument in zip(
                constructor.param_names,
                constructor.param_types,
                arguments,
            ):
                scalar_type = self._static_constexpr_scalar_type(
                    self._resolve_type_aliases_at(
                        declared_type,
                        type_aliases,
                        struct_position,
                    )
                )
                if scalar_type is None:
                    break
                value = self._evaluate_static_scalar_expression(
                    argument,
                    expected_type=scalar_type,
                    aggregate_bindings=aggregate_bindings,
                )
                if value is None:
                    break
                parameter_values[name] = value
            else:
                member_values: List[Tuple[str, _MetalConstexprScalarValue]] = []
                for member in struct.data_members:
                    member_type = self._static_constexpr_scalar_type(
                        self._resolve_type_aliases_at(
                            member.type_text,
                            type_aliases,
                            struct_position,
                        )
                    )
                    initializer = constructor.init_map.get(
                        member.name,
                        member.default,
                    )
                    if member_type is None or initializer is None:
                        break
                    value = self._evaluate_static_scalar_expression(
                        initializer,
                        expected_type=member_type,
                        scalar_bindings=parameter_values,
                        aggregate_bindings=aggregate_bindings,
                    )
                    if value is None:
                        break
                    member_values.append((member.name, value))
                else:
                    results.append(
                        _MetalConstexprAggregateValue(
                            type_name=normalized_type,
                            members=tuple(member_values),
                        )
                    )
        return results[0] if len(results) == 1 else None

    _FREE_OPERATOR_LABELS = {
        "+": "add",
        "-": "subtract",
        "*": "multiply",
        "/": "divide",
        "%": "modulo",
        "==": "equal",
        "!=": "not_equal",
        "<": "less",
        ">": "greater",
        "<=": "less_equal",
        ">=": "greater_equal",
        "+=": "add_assign",
        "-=": "subtract_assign",
        "*=": "multiply_assign",
        "/=": "divide_assign",
        "%=": "modulo_assign",
    }
    _COMPOUND_FREE_OPERATORS = frozenset({"+=", "-=", "*=", "/=", "%="})

    def _find_boolean_variable_templates(
        self, code: str
    ) -> Dict[str, List[_MetalBooleanVariableTemplate]]:
        definitions: Dict[str, List[_MetalBooleanVariableTemplate]] = {}
        ignored = self._find_comment_and_literal_spans(code)
        namespace_spans = self._find_namespace_spans(code)
        aggregate_spans = sorted(
            [struct.span for struct in self._find_concrete_struct_definitions(code)]
            + [struct.span for struct in self._find_template_structs(code)]
        )
        cursor = 0
        while True:
            match = re.search(r"\btemplate\s*<", code[cursor:])
            if match is None:
                break
            template_start = cursor + match.start()
            cursor = template_start + len("template")
            if self._containing_span(template_start, ignored) is not None:
                continue
            if self._containing_span(template_start, aggregate_spans) is not None:
                continue
            angle_start = code.find("<", template_start, cursor + 4)
            angle_end = self._find_matching_template_param_angle(code, angle_start)
            if angle_end is None:
                continue
            header = re.match(
                r"\s*(?:static\s+)?constexpr\s+(?:constant\s+)?bool\s+"
                r"(?P<name>[A-Za-z_]\w*)",
                code[angle_end + 1 :],
            )
            if header is None:
                cursor = angle_end + 1
                continue
            name = header.group("name")
            name_end = angle_end + 1 + header.end()
            position = name_end
            while position < len(code) and code[position].isspace():
                position += 1
            is_partial = position < len(code) and code[position] == "<"
            if is_partial:
                argument_end = self._find_matching_angle(code, position)
                if argument_end is None:
                    cursor = position + 1
                    continue
                argument_patterns = tuple(
                    self._split_top_level_commas(code[position + 1 : argument_end])
                )
                position = argument_end + 1
            else:
                argument_patterns = tuple(
                    self._template_parameter_names(code[angle_start + 1 : angle_end])
                )
            equals = code.find("=", position)
            semicolon = code.find(";", position)
            if equals == -1 or semicolon == -1 or semicolon < equals:
                cursor = position
                continue
            prefix = code[position:equals]
            if prefix.strip():
                cursor = position
                continue
            expression = code[equals + 1 : semicolon].strip()
            parameter_names = tuple(
                self._template_parameter_names(code[angle_start + 1 : angle_end])
            )
            if expression and argument_patterns:
                definitions.setdefault(name, []).append(
                    _MetalBooleanVariableTemplate(
                        name=name,
                        template_parameters=parameter_names,
                        argument_patterns=argument_patterns,
                        expression=expression,
                        is_partial_specialization=is_partial,
                        span=(template_start, semicolon + 1),
                        namespace=self._namespace_at(namespace_spans, template_start),
                    )
                )
            cursor = semicolon + 1
        traits = struct_value_trait_definitions(
            self, code, _MetalBooleanVariableTemplate
        )
        for trait_name, trait_definitions in traits.items():
            definitions.setdefault(trait_name, []).extend(trait_definitions)
        return definitions

    def _evaluate_boolean_variable_template(
        self,
        name: str,
        arguments: Sequence[str],
        bindings: Dict[str, str],
        definitions: Dict[str, List[_MetalBooleanVariableTemplate]],
        stack: Tuple[Tuple[str, Tuple[str, ...]], ...],
        *,
        structs: Optional[Sequence[_MetalStructDefinition]] = None,
        type_aliases: Optional[Dict[str, List[_MetalTypeAliasBinding]]] = None,
        position: int = 0,
        namespace_visibility: Optional[_MetalNamespaceVisibility] = None,
        lookup_position: Optional[int] = None,
        member: str = "",
    ) -> bool:
        visible = self._visible_boolean_variable_templates(
            name, definitions, position, namespace_visibility, lookup_position
        )
        if any(definition.member != member for definition in visible):
            raise self._UnrecognizedConstraint(name)

        def canonical(argument: str, at: int) -> str:
            normalized = self._canonical_type_trait_operand(
                argument,
                lambda operand: self._canonicalize_type_aliases_at(
                    operand, type_aliases or {}, at
                ),
            )
            if normalized is None:
                raise self._UnrecognizedConstraint(name)
            return normalized

        concrete_arguments = complete_template_arguments(
            self,
            visible[0],
            [
                canonical(self._replace_identifiers(argument, bindings), position)
                for argument in arguments
            ],
        )
        if concrete_arguments is None:
            raise self._UnrecognizedConstraint(name)
        concrete_arguments = tuple(
            canonical(argument, position) for argument in concrete_arguments
        )
        qualified_name = f"{visible[0].namespace}::{visible[0].name}"
        key = (qualified_name, concrete_arguments)
        if key in stack or len(stack) >= 64:
            raise self._UnrecognizedConstraint(name)
        candidates = []
        for definition in visible:
            patterns = complete_template_arguments(
                self, visible[0], definition.argument_patterns
            )
            if patterns is None or len(patterns) != len(concrete_arguments):
                continue
            local_bindings: Dict[str, str] = {}
            constraints: List[Tuple[Tuple[str, str], str]] = []
            matched = True
            score = (
                2
                if not definition.template_parameters
                else 1 if definition.is_partial_specialization else 0
            )
            for pattern, concrete in zip(patterns, concrete_arguments):
                normalized_pattern = self._normalize_template_argument_text(pattern)
                constraint = (
                    enable_if_parts(self, normalized_pattern)
                    if definition.is_partial_specialization
                    else None
                )
                if constraint is not None:
                    # Decided once the other arguments have bound the parameters.
                    constraints.append((constraint, concrete))
                    continue
                if normalized_pattern in definition.template_parameters:
                    existing = local_bindings.get(normalized_pattern)
                    if existing is not None and existing != concrete:
                        matched = False
                        break
                    local_bindings[normalized_pattern] = concrete
                    continue
                self._infer_template_parameter_bindings_from_type(
                    normalized_pattern,
                    concrete,
                    set(definition.template_parameters),
                    local_bindings,
                )
                instantiated = canonical(
                    self._replace_identifiers(normalized_pattern, local_bindings),
                    definition.span[0],
                )
                source_specialization = self._materialized_struct_specializations.get(
                    concrete
                )
                source_concrete = (
                    f"{source_specialization[0]}<{', '.join(source_specialization[1])}>"
                    if source_specialization
                    else concrete
                )
                if instantiated != canonical(source_concrete, position):
                    matched = False
                    break
            if matched and all(
                parameter in local_bindings
                for parameter in definition.template_parameters
            ):
                for (condition, result_type), concrete in constraints:
                    enabled = self._evaluate_boolean_constraint(
                        condition,
                        local_bindings,
                        boolean_templates=definitions,
                        variable_stack=(*stack, key),
                        structs=structs,
                        type_aliases=type_aliases,
                        position=position,
                        namespace_visibility=namespace_visibility,
                        lookup_position=definition.span[0],
                    )
                    if not enabled or canonical(
                        self._replace_identifiers(result_type, local_bindings),
                        definition.span[0],
                    ) != canonical(concrete, position):
                        matched = False
                        break
                if matched:
                    candidates.append((score, definition, local_bindings))
        if not candidates:
            raise self._UnrecognizedConstraint(name)
        best_score = max(candidate[0] for candidate in candidates)
        winners = [candidate for candidate in candidates if candidate[0] == best_score]
        if len(winners) != 1:
            raise self._UnrecognizedConstraint(name)
        _score, definition, local_bindings = winners[0]
        return self._evaluate_boolean_constraint(
            definition.expression,
            local_bindings,
            boolean_templates=definitions,
            variable_stack=(*stack, key),
            structs=structs,
            type_aliases=type_aliases,
            position=position,
            namespace_visibility=namespace_visibility,
            lookup_position=definition.span[0],
        )

    def _visible_boolean_variable_templates(
        self,
        name: str,
        definitions: Dict[str, List[_MetalBooleanVariableTemplate]],
        position: int,
        visibility: Optional[_MetalNamespaceVisibility],
        lookup_position: Optional[int],
    ) -> List[_MetalBooleanVariableTemplate]:
        lookup = position if lookup_position is None else lookup_position
        reference_namespace = self._metal_call_reference_namespace(name)
        declarations = definitions.get(name.rsplit("::", 1)[-1], [])
        primaries = [
            definition
            for definition in declarations
            if not definition.is_partial_specialization
            and definition.span[0] <= lookup
            and self._metal_namespace_declaration_visible(
                definition.namespace, reference_namespace, lookup, visibility
            )
        ]
        context = (
            self._namespace_at(list(visibility.namespace_spans), lookup)
            if visibility
            else ""
        )
        namespaces = (
            self._metal_namespace_reference_candidates(
                reference_namespace,
                context,
                globally_qualified=reference_namespace.startswith("::"),
            )
            if reference_namespace is not None
            else self._metal_enclosing_namespaces(context)
        )
        for namespace in namespaces:
            local = [
                definition
                for definition in primaries
                if definition.namespace == namespace
            ]
            if local:
                # An imported competing name has no proven precedence here.
                imported = [
                    definition
                    for definition in primaries
                    if definition.namespace not in namespaces
                ]
                primaries = local + imported
                break
        if len(primaries) != 1:
            raise self._UnrecognizedConstraint(name)
        primary = primaries[0]
        return [primary] + [
            definition
            for definition in declarations
            if definition.is_partial_specialization
            and definition.namespace == primary.namespace
            and definition.span[0] <= position
        ]

    def _evaluate_boolean_constraint_with_variable_templates(
        self,
        expression: str,
        bindings: Dict[str, str],
        definitions: Dict[str, List[_MetalBooleanVariableTemplate]],
        *,
        stack: Tuple[Tuple[str, Tuple[str, ...]], ...] = (),
        position: int = 0,
        namespace_visibility: Optional[_MetalNamespaceVisibility] = None,
        lookup_position: Optional[int] = None,
    ) -> bool:
        return self._evaluate_boolean_constraint(
            expression,
            bindings,
            boolean_templates=definitions,
            variable_stack=stack,
            position=position,
            namespace_visibility=namespace_visibility,
            lookup_position=lookup_position,
        )

    def _free_operator_constraint_enabled(
        self,
        constraint: str,
        bindings: Dict[str, str],
        definitions: Dict[str, List[_MetalBooleanVariableTemplate]],
        *,
        position: int = 0,
        namespace_visibility: Optional[_MetalNamespaceVisibility] = None,
        lookup_position: Optional[int] = None,
    ) -> bool:
        text = _strip_global_metal_qualifiers(constraint.strip())
        enable_match = re.match(
            r"^(?:typename\s+)?(?:metal\s*::\s*)?enable_if_t\s*<", text
        )
        if enable_match is not None:
            angle_start = text.find("<", enable_match.end() - 1)
            angle_end = self._find_matching_template_param_angle(text, angle_start)
            if angle_end is None or text[angle_end + 1 :].strip():
                raise self._UnrecognizedConstraint(constraint)
            arguments = self._split_template_parameter_list(
                text[angle_start + 1 : angle_end]
            )
            if not arguments:
                raise self._UnrecognizedConstraint(constraint)
            text = arguments[0].strip()
        return self._evaluate_boolean_constraint_with_variable_templates(
            text,
            bindings,
            definitions,
            position=position,
            namespace_visibility=namespace_visibility,
            lookup_position=lookup_position,
        )

    def _find_free_operator_definitions(
        self, code: str
    ) -> List[_MetalFreeOperatorDefinition]:
        definitions: List[_MetalFreeOperatorDefinition] = []
        ignored = self._find_comment_and_literal_spans(code)
        self._find_namespace_spans(code)
        anonymous_namespace_spans = self._source_analysis(
            code
        ).anonymous_namespace_spans
        cursor = 0
        while True:
            # The cursor always follows one of > ( ) { }, or sits between a matched
            # "template" and its "<", so a match at or after it is the match that
            # code[cursor:] would give, without copying the rest of the source.
            template_match = _TEMPLATE_HEAD_RE.search(code, cursor)
            if template_match is None:
                break
            template_start = template_match.start()
            cursor = template_start + len("template")
            if self._containing_span(template_start, ignored) is not None:
                continue
            angle_start = code.find("<", template_start, cursor + 4)
            angle_end = self._find_matching_template_param_angle(code, angle_start)
            if angle_end is None:
                continue
            # The operator name must come before the next ';', '{' or '}', and the
            # name itself contains none of them, so the search can stop there
            # instead of running to the end of the source.
            declaration_break = _DECLARATION_BREAK_RE.search(code, angle_end + 1)
            operator_match = _FREE_OPERATOR_NAME_RE.search(
                code,
                angle_end + 1,
                len(code) if declaration_break is None else declaration_break.start(),
            )
            if operator_match is None:
                cursor = angle_end + 1
                continue
            declaration_prefix = code[angle_end + 1 : operator_match.start()]
            parameter_open = code.find(
                "(", operator_match.start(), operator_match.end()
            )
            parameter_close = self._find_matching_delimiter(
                code, parameter_open, "(", ")"
            )
            if parameter_close is None:
                cursor = operator_match.end()
                continue
            body_open = code.find("{", parameter_close + 1)
            declaration_end = code.find(";", parameter_close + 1)
            if body_open == -1 or (
                declaration_end != -1 and declaration_end < body_open
            ):
                cursor = parameter_close + 1
                continue
            body_end = self._find_matching_brace(code, body_open)
            if body_end is None:
                cursor = body_open + 1
                continue
            parameters = code[parameter_open + 1 : parameter_close]
            parameter_types = tuple(self._parameter_declared_types(parameters))
            if len(parameter_types) not in {1, 2} or any(
                parameter_type is None for parameter_type in parameter_types
            ):
                cursor = body_end
                continue
            template_text = code[angle_start + 1 : angle_end]
            template_records = self._parse_template_parameter_list(template_text)
            template_parameters = tuple(
                record.name for record in template_records if record.name
            )
            constraints = tuple(
                record.constraint_text
                for record in template_records
                if record.constraint_text
            )
            return_type = re.sub(
                r"\b(?:constexpr|consteval|inline|static|friend)\b",
                " ",
                declaration_prefix,
            )
            return_type = self._normalize_template_argument_text(return_type)
            if not return_type or not template_parameters:
                cursor = body_end
                continue
            definitions.append(
                _MetalFreeOperatorDefinition(
                    operator=operator_match.group("operator"),
                    template_text=template_text,
                    template_parameters=template_parameters,
                    template_constraints=constraints,
                    return_type=return_type,
                    parameters=parameters,
                    parameter_types=parameter_types,
                    body=code[body_open + 1 : body_end - 1],
                    span=(template_start, body_end),
                    internal_linkage=bool(
                        re.search(
                            r"\bstatic\b",
                            self._mask_comments_and_literals(declaration_prefix),
                        )
                    )
                    or any(
                        start <= template_start < end
                        for start, end in anonymous_namespace_spans
                    ),
                )
            )
            cursor = body_end
        return definitions

    def _substitute_materialized_free_operator_text(
        self,
        text: str,
        bindings: Dict[str, str],
    ) -> str:
        result = self._replace_identifiers(text, bindings)
        for materialized_name, (source_name, source_arguments) in sorted(
            self._materialized_struct_specializations.items()
        ):
            concrete_arguments = [
                self._normalize_template_argument_text(
                    self._replace_identifiers(argument, bindings)
                )
                for argument in source_arguments
            ]
            argument_patterns = [
                re.escape(argument).replace(r"\ ", r"\s*")
                for argument in concrete_arguments
            ]
            pattern = (
                rf"\b{re.escape(source_name)}\s*<\s*"
                + r"\s*,\s*".join(argument_patterns)
                + r"\s*>"
            )
            result = re.sub(pattern, materialized_name, result)
        return result

    def _compound_free_operator_value_form(
        self,
        return_type: str,
        parameters: str,
        parameter_types: Sequence[str],
        body: str,
    ) -> Optional[Tuple[str, str, Tuple[str, ...]]]:
        """Return a materialized ``operator op=`` that updates and returns a copy.

        MLX's complex ``operator+=`` takes ``thread complex_t<T>& a`` and a
        value ``b``, updates ``a`` member by member and returns ``a``. With a
        value ``b`` nothing else can alias ``a``, so running the same body on a
        copy and returning it computes the value the reference form stores:
        callers lower ``x op= y`` to ``x = helper(x, y)``. Only that shape
        qualifies: a thread (or unqualified) non-const reference first
        parameter, a value second parameter, a reference return of the same
        type and a body whose single ``return`` is its last statement and
        names the first parameter. Threadgroup and device overloads keep their
        templates, so their compound assignments stay unlowered.
        """
        names = self._parameter_identifier_names(parameters)
        declarations = [
            item.strip() for item in self._split_top_level_commas(parameters)
        ]
        if len(names) != 2 or len(declarations) != 2 or len(parameter_types) != 2:
            return None
        reference = re.fullmatch(
            r"(?:thread\s+)?(?P<type>[A-Za-z_]\w*)\s*&", parameter_types[0].strip()
        )
        if reference is None or "&" in parameter_types[1] or "*" in parameter_types[1]:
            return None
        value_type = reference.group("type")
        returned = re.fullmatch(
            r"(?:thread\s+)?(?P<type>[A-Za-z_]\w*)\s*&",
            self._normalize_template_argument_text(return_type).strip(),
        )
        if returned is None or returned.group("type") != value_type:
            return None
        masked = self._mask_comments_and_literals(body)
        returns = list(re.finditer(r"\breturn\b", masked))
        final = re.search(rf"\breturn\s+{re.escape(names[0])}\s*;\s*$", masked.rstrip())
        if len(returns) != 1 or final is None or final.start() != returns[0].start():
            return None
        if re.search(rf"&\s*{re.escape(names[0])}\b", masked):
            # A body that takes the parameter's address could leak the copy.
            return None
        rendered = f"{value_type} {names[0]}, {declarations[1]}"
        return value_type, rendered, (value_type, parameter_types[1])

    def _free_operator_helper_name(
        self, operator: str, parameter_types: Sequence[str]
    ) -> str:
        if operator == "-" and len(parameter_types) == 1:
            label = "negate"
        else:
            label = self._FREE_OPERATOR_LABELS[operator]
        identities = []
        for parameter_type in parameter_types:
            identity = re.sub(
                r"\b(?:const|volatile|thread|threadgroup|device|constant|restrict)\b",
                " ",
                parameter_type,
            )
            identity = re.sub(r"[&*]+", " ", identity)
            identity = self._normalize_template_argument_text(identity)
            identity = re.sub(r"[^A-Za-z0-9_]+", "_", identity).strip("_")
            identities.append(identity or "unknown")
        return f"crosstl_metal_operator_{label}__" + "__".join(identities)

    def _rewrite_materialized_free_operator_calls(
        self,
        body: str,
        parameters: str,
        parameter_types: Sequence[str],
        emitted_names: Set[str],
    ) -> str:
        """Bind direct dependent operator calls to earlier concrete helpers.

        MLX's complex comparison operators define ``<`` and ``<=`` in terms of
        direct calls to ``operator>`` and ``operator>=`` with reversed
        arguments. Once the surrounding aggregate is materialized, those calls
        have an exact source signature. Rewrite only a permutation of the
        current function parameters and only when the referenced concrete
        helper has already been emitted; every other dependent call remains
        unresolved and keeps the existing fail-closed behavior.
        """

        parameter_names = self._parameter_identifier_names(parameters)
        if len(parameter_names) != len(parameter_types):
            return body
        type_by_name = dict(zip(parameter_names, parameter_types))
        replacements: List[Tuple[int, int, str]] = []
        pattern = re.compile(r"\boperator\s*(?P<operator>==|!=|<=|>=|[+\-*/%<>])\s*\(")
        for match in pattern.finditer(body):
            open_paren = match.end() - 1
            close_paren = self._find_matching_delimiter(body, open_paren, "(", ")")
            if close_paren is None:
                continue
            arguments = [
                argument.strip()
                for argument in self._split_top_level_commas(
                    body[open_paren + 1 : close_paren]
                )
            ]
            if len(arguments) != len(parameter_names) or any(
                not re.fullmatch(r"[A-Za-z_]\w*", argument)
                or argument not in type_by_name
                for argument in arguments
            ):
                continue
            helper_name = self._free_operator_helper_name(
                match.group("operator"),
                tuple(type_by_name[argument] for argument in arguments),
            )
            if helper_name not in emitted_names:
                continue
            replacements.append(
                (
                    match.start(),
                    close_paren + 1,
                    f"{helper_name}({', '.join(arguments)})",
                )
            )
        if not replacements:
            return body
        return self._apply_text_replacements(body, replacements)

    def _materialize_free_operator_overloads(self, code: str) -> str:
        # An operator between two integral constants has a known constant result; fold it rather than instantiate it.
        code = fold_concrete_function_integral_constant_arithmetic(self, code)
        code = materialize_concrete_free_operators(self, code)
        if not self._materialized_struct_specializations or "operator" not in code:
            return code
        operators = self._find_free_operator_definitions(code)
        if not operators:
            return code
        boolean_templates = self._find_boolean_variable_templates(code)
        namespace_visibility = self._metal_namespace_visibility(code)
        replacements = []
        emitted_names: Set[str] = set()
        template_spans = self._find_template_declaration_spans(code)
        concrete_functions = self._find_non_template_function_definitions(
            code, template_spans
        )
        reachable_source = (
            "\n".join(
                [
                    code[function.span[0] : function.span[1]]
                    for function in concrete_functions
                ]
                # Concrete static assertions are emitted and lowered too.
                + concrete_static_assertion_conditions(self, code, template_spans)
            )
            if concrete_functions
            else None
        )
        source_aliases = self._collect_local_type_alias_bindings(
            code,
            [(0, len(code))],
            skip_spans=template_spans,
        )

        def is_reachable_materialized_type(name: str) -> bool:
            if reachable_source is None:
                return True
            spellings = {name}
            for alias, bindings in source_aliases.items():
                targets = {
                    self._normalize_inferred_type(binding.target)
                    for binding in bindings
                    if binding.target
                }
                if targets == {name}:
                    spellings.add(alias)
            return any(
                re.search(rf"\b{re.escape(spelling)}\b", reachable_source)
                for spelling in spellings
            )

        retained_structs = {
            name: specialization
            for name, specialization in (
                self._materialized_struct_specializations.items()
            )
            if re.search(rf"\b(?:struct|class)\s+{re.escape(name)}\b", code)
            and is_reachable_materialized_type(name)
        }
        for definition in operators:
            if is_folded_integral_constant_operator(self, definition):
                continue
            generated = []
            for materialized_name, (source_name, source_arguments) in sorted(
                retained_structs.items()
            ):
                if not any(
                    re.search(rf"\b{re.escape(source_name)}\s*<", parameter_type)
                    for parameter_type in definition.parameter_types
                ):
                    continue
                bindings: Dict[str, str] = {}
                for parameter_type in definition.parameter_types:
                    if not re.search(
                        rf"\b{re.escape(source_name)}\s*<", parameter_type
                    ):
                        continue
                    if definition.operator in self._COMPOUND_FREE_OPERATORS:
                        # ``thread complex_t<T>& a`` binds ``T`` through the
                        # referenced type; the value form drops the reference.
                        parameter_type = re.sub(
                            r"\b(?:thread|threadgroup|device|constant)\b|&",
                            " ",
                            parameter_type,
                        ).strip()
                    self._infer_template_parameter_bindings_from_type(
                        parameter_type,
                        materialized_name,
                        set(definition.template_parameters),
                        bindings,
                    )
                    if not any(
                        parameter in bindings
                        for parameter in definition.template_parameters
                        if re.search(rf"\b{re.escape(parameter)}\b", parameter_type)
                    ):
                        bindings = {}
                        break
                if not bindings:
                    continue
                lane_type = source_arguments[0] if len(source_arguments) == 1 else None
                for parameter in definition.template_parameters:
                    if parameter in bindings:
                        continue
                    scalar_parameter = any(
                        self._normalize_inferred_type(parameter_type) == parameter
                        for parameter_type in definition.parameter_types
                    )
                    if lane_type is None or not scalar_parameter:
                        bindings = {}
                        break
                    bindings[parameter] = lane_type
                if not bindings:
                    continue
                try:
                    if not all(
                        self._free_operator_constraint_enabled(
                            constraint,
                            bindings,
                            boolean_templates,
                            position=len(code),
                            namespace_visibility=namespace_visibility,
                            lookup_position=definition.span[0],
                        )
                        for constraint in definition.template_constraints
                    ):
                        continue
                except self._UnrecognizedConstraint:
                    continue
                return_type = self._substitute_materialized_free_operator_text(
                    definition.return_type,
                    bindings,
                )
                parameters = self._substitute_materialized_free_operator_text(
                    definition.parameters,
                    bindings,
                )
                parameter_types = tuple(self._parameter_declared_types(parameters))
                if any(parameter_type is None for parameter_type in parameter_types):
                    continue
                body = self._substitute_materialized_free_operator_text(
                    definition.body,
                    bindings,
                )
                body = self._rewrite_materialized_free_operator_calls(
                    body,
                    parameters,
                    parameter_types,
                    emitted_names,
                )
                if re.search(r"\boperator\s*(?:==|!=|<=|>=|[+\-*/%<>])\s*\(", body):
                    continue
                if definition.operator in self._COMPOUND_FREE_OPERATORS:
                    value_form = self._compound_free_operator_value_form(
                        return_type, parameters, parameter_types, body
                    )
                    if value_form is None:
                        continue
                    return_type, parameters, parameter_types = value_form
                helper_name = self._free_operator_helper_name(
                    definition.operator,
                    parameter_types,
                )
                if helper_name in emitted_names:
                    continue
                emitted_names.add(helper_name)
                # Template instantiations can recur in separately translated units;
                # private definitions must not coalesce across those units.
                linkage = "static inline" if definition.internal_linkage else "inline"
                generated.append(
                    f"{linkage} {return_type} {helper_name}({parameters}) {{"
                    f"{body}}}\n"
                )
            if generated:
                replacements.append(
                    (definition.span[0], definition.span[1], "\n".join(generated))
                )
        if not replacements:
            return code
        return self._apply_text_replacements(code, replacements)

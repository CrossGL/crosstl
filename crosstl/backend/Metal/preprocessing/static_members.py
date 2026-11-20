"""Static data member initializers and static assertions."""

import re
from typing import (
    Dict,
    List,
    Optional,
    Tuple,
)

from ..namespace_constants import NamespaceConstantValues
from ..static_member_constants import (
    discarded_instantiated_branch_spans,
    static_member_constant_substituter,
)
from .support import (
    IDENTIFIER_RE,
    METAL_STRING_EXPRESSION_PATTERN,
    MetalStaticAssertionError,
    _MetalConstexprBinaryOperator,
    _MetalConstexprFunction,
    _MetalDataMember,
    _MetalIntegralConstantBinding,
    _MetalStaticAssertion,
    _MetalStructDefinition,
    _MetalStructMethod,
    _MetalTemplateStruct,
)


class MetalStaticMemberMixin:
    """Members of ``MetalPreprocessor``: static data member initializers and static assertions."""

    def _resolved_static_data_member_initializers(
        self, struct: _MetalStructDefinition
    ) -> Dict[str, str]:
        members: Dict[str, _MetalDataMember] = {}
        initializers: Dict[str, str] = {}
        for member in struct.data_members:
            qualifiers = set(IDENTIFIER_RE.findall(member.type_text))
            if (
                member.default is None
                or "static" not in qualifiers
                or not qualifiers.intersection({"const", "constant", "constexpr"})
            ):
                continue
            initializer = member.default.strip()
            if not initializer:
                continue
            initializer = self._canonicalize_struct_scoped_type(
                initializer,
                struct,
                None,
            )
            members[member.name] = member
            initializers[member.name] = initializer
        if not members:
            return {}

        states: Dict[str, str] = {}
        resolved: Dict[str, str] = {}
        member_names = list(members)

        def resolve(name: str, depth: int = 0) -> Optional[str]:
            state = states.get(name)
            if state == "resolved":
                return resolved[name]
            if state in {"visiting", "failed"}:
                return None
            if depth >= self._STATIC_INITIALIZER_MAX_DEPENDENCY_DEPTH:
                states[name] = "failed"
                return None

            states[name] = "visiting"
            initializer = initializers[name]
            dependency_mapping: Dict[str, str] = {}
            for dependency in self._static_initializer_dependencies(
                initializer, member_names
            ):
                dependency_value = resolve(dependency, depth + 1)
                if dependency_value is None:
                    states[name] = "failed"
                    return None
                dependency_mapping[dependency] = f"({dependency_value})"
            if dependency_mapping:
                initializer = self._substitute_bare_member_references(
                    initializer, dependency_mapping
                )
            initializer = self._fold_static_integral_initializer(
                members[name], initializer
            )
            resolved[name] = initializer
            states[name] = "resolved"
            return initializer

        for name in member_names:
            resolve(name)
        return resolved

    def _substitute_static_data_member_initializers(
        self,
        struct: _MetalStructDefinition,
        method: _MetalStructMethod,
        body: str,
    ) -> str:
        # A lowered free function cannot resolve a bare reference to a static
        # data member that remains nested in the data-only struct. Inline the
        # initializer for initialized static constants, but only at bare,
        # unshadowed uses. Mutable static members are deliberately excluded.
        shadowed = set(method.parameter_names)
        shadowed.update(self._local_variable_names(body))
        mapping = {
            name: f"({initializer})"
            for name, initializer in self._resolved_static_data_member_initializers(
                struct
            ).items()
            if name not in shadowed
        }
        if not mapping:
            return body
        return self._substitute_bare_member_references(body, mapping)

    def _static_member_use_requires_declared_storage(
        self,
        masked_code: str,
        use_start: int,
    ) -> bool:
        """Return whether replacing a static field would lose lvalue semantics.

        Compile-time member substitution normally turns an lvalue naming static
        storage into a prvalue initializer.  That is not valid for address-taking,
        reference binding, or type-sensitive unevaluated operands.  Keep the
        declaration and the exact use in those contexts so later translation can
        either preserve the storage or fail closed with its normal provenance
        diagnostic instead of manufacturing an invalid address of a temporary.
        ``masked_code`` preserves source offsets while blanking comments/literals.
        """

        boundary = max(
            masked_code.rfind(token, 0, use_start) for token in (";", "{", "}")
        )
        statement_prefix = masked_code[boundary + 1 : use_start]
        declaration_boundary = max(
            masked_code.rfind(token, 0, use_start) for token in (";", "}")
        )
        declaration_prefix = masked_code[declaration_boundary + 1 : use_start]
        if re.search(
            r"&\s*[A-Za-z_]\w*\s*(?:=\s*)?(?:[({]\s*)*$",
            declaration_prefix,
        ):
            return True
        if re.search(
            r"\b(?:alignof|decltype|sizeof)\s*(?:\(\s*)*$",
            statement_prefix,
        ):
            return True

        def previous_significant(before: int) -> int:
            position = before - 1
            while position >= 0 and masked_code[position].isspace():
                position -= 1
            return position

        # Accept redundant grouping around the selected member: both
        # ``&Owner::field`` and ``&(Owner::field)`` require the same storage.
        cursor = use_start
        previous = previous_significant(cursor)
        while previous >= 0 and masked_code[previous] == "(":
            cursor = previous
            previous = previous_significant(cursor)
        if previous < 0 or masked_code[previous] != "&":
            return False

        ampersand = previous
        # The second character of a logical-and token is not unary address-of.
        if ampersand > 0 and masked_code[ampersand - 1 : ampersand + 1] == "&&":
            return False
        before_ampersand = previous_significant(ampersand)
        if before_ampersand < 0:
            return True
        if masked_code[before_ampersand] in "([{,:;=!?~+-*/%&|^<>":
            return True
        if not (
            masked_code[before_ampersand].isalnum()
            or masked_code[before_ampersand] == "_"
        ):
            return False

        word_start = before_ampersand
        while word_start > 0 and (
            masked_code[word_start - 1].isalnum() or masked_code[word_start - 1] == "_"
        ):
            word_start -= 1
        return masked_code[word_start : before_ampersand + 1] in {
            "case",
            "co_await",
            "co_return",
            "return",
            "throw",
        }

    def _evaluate_static_assertions(
        self, code: str, *, keep_unresolved: bool = False
    ) -> str:
        # With keep_unresolved only assertions over constexpr aggregates are decided here: a proven one is removed
        # and a false one raises. Integral and undecided conditions stay in the source for the translator's own
        # check (crosstl/translator/semantics/static_assertions.py), and CrossGL keeps their declarations.
        code = self._fold_concrete_type_trait_values(code)
        if "static_assert" not in code:
            return code

        template_spans = self._find_template_declaration_spans(code)
        assertions = self._find_concrete_static_assertions(code, template_spans)
        if not assertions:
            return code

        self._static_constexpr_helper_values = {}
        self._static_constexpr_helper_resolution_stack = []
        owner_spans = [(0, len(code))]
        type_aliases = self._collect_local_type_alias_bindings(
            code,
            owner_spans,
            skip_spans=template_spans,
        )
        structs = self._find_concrete_struct_definitions(code)
        static_member_constants = static_member_constant_substituter(self, structs)
        constants = self._collect_local_integral_constant_bindings(
            code,
            owner_spans,
            type_aliases,
            static_member_constants=static_member_constants,
        )
        discarded_spans = discarded_instantiated_branch_spans(
            self, code, template_spans, constants, static_member_constants
        )
        template_structs = self._find_template_structs(code)
        primary_template_structs_by_name: Dict[str, _MetalTemplateStruct] = {}
        for template_struct in template_structs:
            if self._template_struct_specialization_arguments(template_struct) is None:
                primary_template_structs_by_name.setdefault(
                    template_struct.name,
                    template_struct,
                )
        explicit_template_struct_specializations = (
            self._find_explicit_struct_specialization_keys(
                code,
                primary_template_structs_by_name,
            )
        )
        constexpr_functions: Dict[str, List[_MetalConstexprFunction]] = {}
        constexpr_binary_operators: Optional[List[_MetalConstexprBinaryOperator]] = None
        constexpr_index_built = False
        replacements: List[Tuple[int, int, str]] = []
        namespace_constants = None
        assertion_index = 0
        while assertion_index < len(assertions):
            assertion = assertions[assertion_index]
            if any(start <= assertion.span[0] < end for start, end in discarded_spans):
                # [stmt.if]/2: a discarded substatement of an instantiated
                # template is not instantiated, so its assertion never fires.
                replacements.append((*assertion.span, ""))
                assertion_index += 1
                continue
            condition_position = assertion.condition_span[0]
            visible_bindings = self._local_integral_constant_bindings_at(
                constants,
                condition_position,
            )
            visible_constants = {
                name: binding.value
                for name, binding in visible_bindings.items()
                if binding.value is not None
            }
            resolved_condition = self._substitute_template_argument_static_constants(
                static_member_constants(
                    assertion.condition,
                    self._visible_local_declared_types(visible_bindings),
                ),
                visible_constants,
            )
            if namespace_constants is None or namespace_constants[0] is not constants:
                namespace_constants = (
                    constants,
                    NamespaceConstantValues(
                        self._find_namespace_spans(code), constants
                    ),
                )
            resolved_condition = namespace_constants[1].substitute(
                resolved_condition, condition_position
            )
            resolved_condition, evaluation_condition, _unresolved_calls = (
                self._resolve_static_constexpr_calls(
                    resolved_condition,
                    constexpr_functions,
                    condition_position,
                )
            )
            resolved_condition = self._replace_concrete_sizeof_expressions(
                resolved_condition,
                lambda type_text: self._resolve_type_aliases_at(
                    type_text,
                    type_aliases,
                    condition_position,
                ),
                lambda type_text: self._concrete_aggregate_type_layout(
                    type_text,
                    structs,
                    type_aliases,
                    condition_position,
                    template_structs=template_structs,
                    explicit_template_specializations=(
                        explicit_template_struct_specializations
                    ),
                ),
            )
            evaluation_condition = self._replace_concrete_sizeof_expressions(
                evaluation_condition,
                lambda type_text: self._resolve_type_aliases_at(
                    type_text,
                    type_aliases,
                    condition_position,
                ),
                lambda type_text: self._concrete_aggregate_type_layout(
                    type_text,
                    structs,
                    type_aliases,
                    condition_position,
                    template_structs=template_structs,
                    explicit_template_specializations=(
                        explicit_template_struct_specializations
                    ),
                ),
            )
            folded, value = self._evaluate_static_integral_expression(
                evaluation_condition
            )
            if keep_unresolved and folded:
                # An integral condition is the translator's to check, and CrossGL keeps the declaration.
                assertion_index += 1
                continue
            if not folded:
                if constexpr_binary_operators is None:
                    constexpr_binary_operators = (
                        self._static_constexpr_binary_operators(code)
                    )
                folded, value = self._evaluate_static_aggregate_assertion(
                    evaluation_condition,
                    operators=constexpr_binary_operators,
                    structs=structs,
                    template_structs=template_structs,
                    explicit_template_specializations=(
                        explicit_template_struct_specializations
                    ),
                    type_aliases=type_aliases,
                    position=condition_position,
                )
            if folded and value:
                replacements.append((*assertion.span, ""))
                assertion_index += 1
                continue
            if not folded and not constexpr_index_built:
                constexpr_functions = self._static_constexpr_function_index(code)
                self._static_constexpr_helper_values = {}
                self._static_constexpr_helper_resolution_stack = []
                constants = self._collect_local_integral_constant_bindings(
                    code,
                    owner_spans,
                    type_aliases,
                    constexpr_functions,
                    static_member_constants=static_member_constants,
                )
                discarded_spans = discarded_instantiated_branch_spans(
                    self, code, template_spans, constants, static_member_constants
                )
                replacements = []
                assertion_index = 0
                constexpr_index_built = True
                continue

            if keep_unresolved and not folded:
                assertion_index += 1
                continue
            source_message = (
                self._evaluate_metal_string_expression(assertion.message)
                if assertion.message is not None
                else None
            )
            compact_condition = re.sub(r"\s+", " ", assertion.condition).strip()
            compact_resolved = re.sub(r"\s+", " ", resolved_condition).strip()
            if folded:
                suggested_action = (
                    "correct the source condition or select specialization values "
                    "that satisfy the compile-time constraint"
                )
                detail = (
                    f"Metal static assertion failed because '{compact_resolved}' "
                    "evaluated to false"
                )
                reason = "assertion-failed"
                unresolved_dependencies: Tuple[str, ...] = ()
            else:
                unresolved_dependencies = self._static_assertion_dependencies(
                    resolved_condition,
                    visible_bindings,
                )
                suggested_action = (
                    "materialize all template parameters and express assertion "
                    "dependencies as constexpr integral values before translation"
                )
                dependency_detail = (
                    "; unresolved dependencies: " + ", ".join(unresolved_dependencies)
                    if unresolved_dependencies
                    else ""
                )
                detail = (
                    "Metal static assertion could not be evaluated at translation "
                    f"time: '{compact_resolved}'{dependency_detail}"
                )
                reason = "condition-unresolved"
            if source_message:
                detail += f". Source message: {source_message}"
            detail += f". Suggested action: {suggested_action}."
            raise MetalStaticAssertionError(
                detail,
                reason=reason,
                expression=compact_condition,
                resolved_expression=compact_resolved,
                assertion_message=source_message,
                unresolved_dependencies=unresolved_dependencies,
                source_location=self._source_location_for_offsets(
                    code,
                    assertion.span[0],
                    assertion.span[1],
                ),
                suggested_action=suggested_action,
            )

        return self._apply_text_replacements(code, replacements)

    def _find_concrete_static_assertions(
        self,
        code: str,
        template_spans: List[Tuple[int, int]],
    ) -> List[_MetalStaticAssertion]:
        ignored_spans = self._find_comment_and_literal_spans(code)
        assertions: List[_MetalStaticAssertion] = []
        for match in re.finditer(r"\bstatic_assert\b", code):
            start = match.start()
            if self._containing_span(start, ignored_spans) is not None:
                continue
            if self._containing_span(start, template_spans) is not None:
                continue

            open_paren = match.end()
            while open_paren < len(code) and code[open_paren].isspace():
                open_paren += 1
            if open_paren >= len(code) or code[open_paren] != "(":
                self._raise_malformed_static_assertion(code, start, match.end())
            close_paren = self._find_matching_delimiter(
                code,
                open_paren,
                "(",
                ")",
            )
            if close_paren is None:
                self._raise_malformed_static_assertion(code, start, len(code))

            end = close_paren + 1
            while end < len(code) and code[end].isspace():
                end += 1
            if end >= len(code) or code[end] != ";":
                self._raise_malformed_static_assertion(
                    code,
                    start,
                    close_paren + 1,
                )
            end += 1
            condition, assertion_message, condition_offset = (
                self._split_static_assertion_arguments(
                    code[open_paren + 1 : close_paren]
                )
            )
            if not condition:
                self._raise_malformed_static_assertion(code, start, end)
            condition_start = open_paren + 1 + condition_offset
            assertions.append(
                _MetalStaticAssertion(
                    span=(start, end),
                    condition_span=(
                        condition_start,
                        condition_start + len(condition),
                    ),
                    condition=condition,
                    message=assertion_message,
                )
            )
        return assertions

    def _split_static_assertion_arguments(
        self, arguments: str
    ) -> Tuple[str, Optional[str], int]:
        comma_positions: List[int] = []
        paren_depth = 0
        bracket_depth = 0
        brace_depth = 0
        i = 0
        while i < len(arguments):
            if arguments[i] in "\"'":
                _literal, consumed = self._read_string(arguments, i)
                i += consumed
                continue
            if arguments.startswith("//", i):
                end = arguments.find("\n", i)
                i = len(arguments) if end == -1 else end + 1
                continue
            if arguments.startswith("/*", i):
                end = arguments.find("*/", i + 2)
                i = len(arguments) if end == -1 else end + 2
                continue
            character = arguments[i]
            if character == "(":
                paren_depth += 1
            elif character == ")":
                paren_depth = max(0, paren_depth - 1)
            elif character == "[":
                bracket_depth += 1
            elif character == "]":
                bracket_depth = max(0, bracket_depth - 1)
            elif character == "{":
                brace_depth += 1
            elif character == "}":
                brace_depth = max(0, brace_depth - 1)
            elif (
                character == ","
                and paren_depth == 0
                and bracket_depth == 0
                and brace_depth == 0
            ):
                comma_positions.append(i)
            i += 1

        for comma in reversed(comma_positions):
            candidate_message = arguments[comma + 1 :].strip()
            if re.fullmatch(
                METAL_STRING_EXPRESSION_PATTERN,
                candidate_message,
                re.DOTALL,
            ):
                raw_condition = arguments[:comma]
                condition = raw_condition.strip()
                return (
                    condition,
                    candidate_message,
                    len(raw_condition) - len(raw_condition.lstrip()),
                )
        condition = arguments.strip()
        return condition, None, len(arguments) - len(arguments.lstrip())

    def _raise_malformed_static_assertion(
        self, code: str, start: int, end: int
    ) -> None:
        suggested_action = (
            "provide a well-formed static_assert(condition[, message]) statement"
        )
        raise MetalStaticAssertionError(
            "Metal static assertion could not be parsed. Suggested action: "
            f"{suggested_action}.",
            reason="assertion-malformed",
            expression="",
            resolved_expression="",
            assertion_message=None,
            unresolved_dependencies=(),
            source_location=self._source_location_for_offsets(code, start, end),
            suggested_action=suggested_action,
        )

    def _static_assertion_dependencies(
        self,
        expression: str,
        visible_bindings: Dict[str, _MetalIntegralConstantBinding],
    ) -> Tuple[str, ...]:
        masked = self._mask_comments_and_literals(expression)
        ignored = {
            "alignof",
            "false",
            "sizeof",
            "true",
        }
        dependencies = {
            identifier
            for identifier in IDENTIFIER_RE.findall(masked)
            if identifier not in ignored
        }
        dependencies.update(
            name
            for name, binding in visible_bindings.items()
            if binding.value is None
            and re.search(rf"\b{re.escape(name)}\b", masked) is not None
        )
        return tuple(sorted(dependencies))

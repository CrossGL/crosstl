"""Partial and explicit struct specializations, template-id candidates and parameter inference from types."""

import re
from typing import (
    Dict,
    List,
    Optional,
    Set,
    Tuple,
)

from ..source_analysis_cache import memoized
from ..text_scanners import strip_top_level_default_value
from .support import (
    IDENTIFIER_RE,
    METAL_STRING_EXPRESSION_PATTERN,
    MetalTemplateSpecializationError,
    _keyword_before,
    _MetalTemplateFunction,
    _MetalTemplateStruct,
)


class MetalStructSpecializationMixin:
    """Members of ``MetalPreprocessor``: partial and explicit struct specializations, template-id candidates and parameter inference from types."""

    def _partial_struct_specialization_bindings(
        self,
        template: _MetalTemplateStruct,
        specialization_arguments: List[str],
        concrete_arguments: List[str],
        *,
        code: str,
        position: int,
    ) -> Optional[Dict[str, str]]:
        if template.variadic_template_parameters:
            return None
        omitted = specialization_arguments[len(concrete_arguments) :]
        if len(specialization_arguments) < len(concrete_arguments) or any(
            not re.search(r"\benable_if(?:_t)?\s*<", argument) for argument in omitted
        ):
            return None
        bindings: Dict[str, str] = {}
        parameter_names = set(template.template_parameters)
        arguments = list(zip(specialization_arguments, concrete_arguments))
        arguments.extend((argument, "") for argument in omitted)
        constrained_arguments = [
            pair for pair in arguments if re.search(r"\benable_if(?:_t)?\s*<", pair[0])
        ]
        # Establish structural non-viability before considering a dependent
        # enabler. An unevaluated enabler is not proof that the primary applies.
        for specialization_argument, concrete_argument in [
            pair for pair in arguments if pair not in constrained_arguments
        ]:
            normalized_pattern = (
                self._canonicalize_partial_struct_specialization_argument(
                    specialization_argument
                )
            )
            normalized_argument = (
                self._canonicalize_partial_struct_specialization_argument(
                    concrete_argument
                )
            )
            if normalized_pattern != normalized_argument and not (
                self._equivalent_template_boolean_literals(
                    normalized_pattern,
                    normalized_argument,
                )
            ):
                self._infer_template_parameter_bindings_from_type(
                    normalized_pattern,
                    normalized_argument,
                    parameter_names,
                    bindings,
                )
            resolved_pattern = self._normalize_template_argument_text(
                self._replace_identifiers(normalized_pattern, bindings)
            )
            if resolved_pattern != normalized_argument and not (
                self._equivalent_template_boolean_literals(
                    resolved_pattern,
                    normalized_argument,
                )
            ):
                return None
        for pattern, argument in constrained_arguments:
            try:
                result_type = self._constrained_partial_struct_argument(
                    pattern, bindings, template, code, position
                )
            except self._UnrecognizedConstraint:
                pass
            else:
                if result_type is None:
                    return None
                normalized_result = (
                    self._canonicalize_partial_struct_specialization_argument(
                        result_type
                    )
                )
                if normalized_result != (
                    self._canonicalize_partial_struct_specialization_argument(argument)
                ):
                    return None
                continue
            signature = self._template_specialization_signature(
                template.name, concrete_arguments
            )
            raise MetalTemplateSpecializationError(
                f"Metal struct specialization '{signature}' has a potentially viable "
                "enable_if constraint that has not been evaluated. Selecting its "
                "primary declaration could change the field types or storage layout.",
                requested_signature=signature,
                requested_arguments=tuple(concrete_arguments),
                callee_template=template.name,
                suggested_action=(
                    "use a supported enable_if predicate and concrete result type, "
                    "or provide an explicit concrete struct specialization"
                ),
            )
        if not parameter_names <= set(bindings):
            return None
        return bindings

    def _require_unique_constrained_struct_partial(
        self,
        name: str,
        arguments: List[str],
        matches: List[Tuple[_MetalTemplateStruct, Dict[str, str]]],
    ) -> None:
        if len(matches) < 2 or not any(
            re.search(r"\benable_if(?:_t)?\s*<", argument)
            for template, _bindings in matches
            for argument in (
                self._template_struct_specialization_arguments(template) or []
            )
        ):
            return
        signature = self._template_specialization_signature(name, arguments)
        raise MetalTemplateSpecializationError(
            f"Metal struct specialization '{signature}' has multiple viable partial "
            "specializations whose ordering is not supported. Selecting its primary "
            "declaration could change the field types or storage layout.",
            requested_signature=signature,
            requested_arguments=tuple(arguments),
            callee_template=name,
            suggested_action=(
                "provide an explicit concrete specialization until partial "
                "specialization ordering is supported"
            ),
        )

    def _constrained_partial_struct_argument(
        self,
        pattern: str,
        bindings: Dict[str, str],
        template: _MetalTemplateStruct,
        code: str,
        position: int,
    ) -> Optional[str]:
        match = re.match(
            r"^\s*(?:typename\s+)?(?:metal\s*::\s*)?enable_if(?P<alias>_t)?\s*<",
            pattern,
        )
        if match is None:
            raise self._UnrecognizedConstraint(pattern)
        start = match.end() - 1
        end = self._find_matching_template_param_angle(pattern, start)
        if end is None:
            raise self._UnrecognizedConstraint(pattern)
        suffix = pattern[end + 1 :].strip()
        if (match.group("alias") and suffix) or (
            not match.group("alias") and not re.fullmatch(r"::\s*type", suffix)
        ):
            raise self._UnrecognizedConstraint(pattern)
        arguments = self._split_template_parameter_list(pattern[start + 1 : end])
        if not 1 <= len(arguments) <= 2:
            raise self._UnrecognizedConstraint(pattern)
        # Every query below depends on the source text only, and a module with many
        # partial specializations evaluates one constraint per candidate.
        values = self._source_analysis(code).values
        aliases = memoized(
            values,
            "global_type_alias_bindings",
            lambda: self._collect_local_type_alias_bindings(
                code,
                [(0, len(code))],
                skip_spans=self._find_template_declaration_spans(code),
            ),
        )
        structs = memoized(
            values,
            "concrete_struct_definitions",
            lambda: self._find_concrete_struct_definitions(code),
        )
        if not self._evaluate_boolean_constraint(
            arguments[0],
            bindings,
            structs=structs,
            type_aliases=aliases,
            position=position,
            boolean_templates=memoized(
                values,
                "boolean_variable_templates",
                lambda: self._find_boolean_variable_templates(code),
            ),
            namespace_visibility=self._metal_namespace_visibility(code),
            lookup_position=template.span[0],
        ):
            return None
        result = self._replace_identifiers(
            arguments[1] if len(arguments) == 2 else "void", bindings
        )
        result = self._resolve_type_aliases_at(result, aliases, template.span[0])
        result = self._canonicalize_qualified_struct_type_aliases(result, structs)
        result = self._canonicalize_metal_standard_type_aliases(
            result,
            allow_unqualified_remove_cv_t=self._metal_unqualified_remove_cv_t_available(
                code
            ),
        )
        if not self._type_trait_operand_is_concrete(
            result,
            memoized(
                values,
                "concrete_type_trait_names",
                lambda: self._concrete_type_trait_names(code, structs),
            ),
            allow_unknown_named_types=False,
        ):
            raise self._UnrecognizedConstraint(pattern)
        return result

    def _canonicalize_partial_struct_specialization_argument(
        self,
        argument: str,
    ) -> str:
        text = self._normalize_struct_specialization_argument(
            self._normalize_template_argument_text(argument)
        )
        for _ in range(len(self._materialized_struct_specializations) + 1):
            replacements: List[Tuple[int, int, str]] = []
            for match in IDENTIFIER_RE.finditer(text):
                specialization = self._materialized_struct_specializations.get(
                    match.group(0)
                )
                if specialization is None:
                    continue
                source_name, source_arguments = specialization
                replacements.append(
                    (
                        match.start(),
                        match.end(),
                        f"{source_name}<{', '.join(source_arguments)}>",
                    )
                )
            if not replacements:
                break
            updated = self._apply_text_replacements(text, replacements)
            if updated == text:
                break
            text = updated
        replacements: List[Tuple[int, int, str]] = []
        for match in IDENTIFIER_RE.finditer(text):
            decomposed = self._scalar_and_width(match.group(0))
            if decomposed is None or decomposed[1] == 1:
                continue
            scalar, width = decomposed
            replacements.append((match.start(), match.end(), f"vec<{scalar}, {width}>"))
        text = self._apply_text_replacements(text, replacements)
        text = re.sub(
            r"(?<![A-Za-z0-9_:])(?:::)?metal\s*::\s*(?=vec\s*<)",
            "",
            text,
        )
        return self._normalize_template_argument_text(text)

    def _qualified_template_struct_alias_at(
        self,
        code: str,
        struct_span: Tuple[int, int],
    ) -> Optional[Tuple[str, Tuple[int, int]]]:
        cursor = struct_span[1]
        while cursor < len(code) and code[cursor].isspace():
            cursor += 1
        if not code.startswith("::", cursor):
            return None
        cursor += 2
        while cursor < len(code) and code[cursor].isspace():
            cursor += 1
        alias_match = IDENTIFIER_RE.match(code, cursor)
        if alias_match is None:
            return None

        replacement_start = struct_span[0]
        typename_start = _keyword_before(code, 0, replacement_start, "typename")
        if typename_start is not None:
            replacement_start = typename_start
        return alias_match.group(0), (replacement_start, alias_match.end())

    def _concrete_template_struct_scalar_alias_target(
        self,
        template: _MetalTemplateStruct,
        bindings: Dict[str, str],
        alias_name: str,
        *,
        allow_unqualified_remove_cv_t: bool = False,
    ) -> Optional[str]:
        body_start = template.source.find("{")
        if body_start == -1:
            return None
        body_end = self._find_matching_brace(template.source, body_start)
        if body_end is None:
            return None
        aliases = self._collect_struct_scope_type_aliases(
            template.source[body_start + 1 : body_end - 1]
        )
        if alias_name not in aliases:
            return None

        resolved = dict(bindings)
        for _ in range(len(aliases) + 1):
            previous = dict(resolved)
            for name, target in aliases.items():
                resolved[name] = self._normalize_template_argument_text(
                    self._replace_identifiers(target, resolved)
                )
            if resolved == previous:
                break
        target = re.sub(r"^typename\s+", "", resolved[alias_name]).strip()
        target = self._canonicalize_metal_standard_type_aliases(
            target,
            allow_unqualified_remove_cv_t=allow_unqualified_remove_cv_t,
            dependent_identifiers=set(template.template_parameters),
        )
        target = re.sub(r"^metal::", "", target)
        if target not in self._METAL_SCALAR_VECTOR_TYPES:
            return None
        return target

    def _explicit_struct_specialization_definitions(
        self,
        code: str,
        primary_templates_by_name: Dict[str, _MetalTemplateStruct],
    ) -> Dict[Tuple[str, Tuple[str, ...]], Tuple[Tuple[int, int], str, bool]]:
        """Return full explicit struct specialization definitions by key.

        Keys match ``_find_explicit_struct_specialization_keys``. Each value
        holds the ``template <> struct Name<...>`` header span, its class key
        and whether the body declares member functions.
        """
        definitions: Dict[
            Tuple[str, Tuple[str, ...]], Tuple[Tuple[int, int], str, bool]
        ] = {}
        type_aliases = self._collect_local_type_alias_bindings(
            code,
            [(0, len(code))],
            skip_spans=self._find_template_declaration_spans(code),
        )
        masked = self._mask_comments_and_literals(code)
        for match in re.finditer(
            r"\btemplate\s*<\s*>\s*(?P<keyword>struct|class)\s+"
            r"(?P<name>[A-Za-z_][A-Za-z0-9_:]*)\s*<",
            masked,
        ):
            angle_start = match.end() - 1
            angle_end = self._find_matching_angle(code, angle_start)
            if angle_end is None:
                continue
            body_start = angle_end + 1
            while body_start < len(code) and code[body_start].isspace():
                body_start += 1
            if body_start >= len(code) or code[body_start] != "{":
                continue
            body_end = self._find_matching_delimiter(code, body_start, "{", "}")
            if body_end is None:
                continue
            name = match.group("name").split("::")[-1]
            arguments = self._split_top_level_commas(code[angle_start + 1 : angle_end])
            primary_template = primary_templates_by_name.get(name)
            if primary_template is not None:
                arguments = (
                    self._template_arguments_with_resolved_defaults(
                        primary_template,
                        arguments,
                    )
                    or arguments
                )
            canonical_arguments = [
                self._resolve_type_aliases_at(argument, type_aliases, match.start())
                for argument in arguments
            ]
            key = self._struct_specialization_comparison_key(name, canonical_arguments)
            probe = f"struct __crossgl_explicit_probe {code[body_start:body_end + 1]};"
            structs = self._find_concrete_struct_definitions(probe)
            has_methods = bool(
                structs
                and (
                    structs[0].methods
                    or structs[0].template_methods
                    or structs[0].has_operator_call
                )
            )
            definitions[key] = (
                (match.start(), angle_end + 1),
                match.group("keyword"),
                has_methods,
            )
        return definitions

    def _concretize_method_explicit_struct_specializations(
        self,
        code: str,
        primary_templates_by_name: Dict[str, _MetalTemplateStruct],
        references: List[
            Tuple[str, List[str], Tuple[str, Tuple[str, ...]], Tuple[int, int]]
        ],
        materialized_names: Dict[Tuple[str, Tuple[str, ...]], str],
    ) -> Optional[str]:
        """Give referenced method-bearing explicit specializations concrete names.

        Member calls lower through a concrete receiver struct, but a full
        explicit specialization such as ``template <> struct CumProd<bool>``
        keeps its template header, so ``CumProd<bool> op; op.simd_scan(x)`` has
        no receiver to resolve. Renaming the definition in place keeps its
        declaration order and body. Data-only specializations (``Limits<bool>``)
        keep their existing static-member path unchanged.
        """
        if not references:
            return None
        definitions = self._explicit_struct_specialization_definitions(
            code, primary_templates_by_name
        )
        reserved = set(IDENTIFIER_RE.findall(self._mask_comments_and_literals(code)))
        renamed: Dict[
            Tuple[str, Tuple[str, ...]], Tuple[str, Tuple[str, Tuple[str, ...]]]
        ] = {}
        replacements: List[Tuple[int, int, str]] = []
        for name, arguments, key, span in references:
            definition = definitions.get(key)
            if definition is None or not definition[2]:
                continue
            if key not in renamed:
                specialization_key = self._template_specialization_key(name, arguments)
                concrete = self._template_specialization_identifier(
                    name, list(specialization_key[1])
                )
                while concrete in reserved:
                    concrete += "_"
                reserved.add(concrete)
                renamed[key] = (concrete, specialization_key)
                header_span, keyword, _has_methods = definition
                replacements.append(
                    (header_span[0], header_span[1], f"{keyword} {concrete}")
                )
            replacements.append((span[0], span[1], renamed[key][0]))
        if not renamed:
            return None
        for concrete, specialization_key in renamed.values():
            materialized_names[specialization_key] = concrete
            self._materialized_struct_specializations[concrete] = (
                specialization_key[0],
                tuple(specialization_key[1]),
            )
            primary = primary_templates_by_name.get(specialization_key[0])
            if primary is not None:
                self._materialized_struct_primary_templates[concrete] = primary
                self._materialized_struct_specialization_namespaces[concrete] = (
                    primary.namespace
                )
        return self._apply_text_replacements(code, replacements)

    def _find_explicit_struct_specialization_keys(
        self,
        code: str,
        primary_templates_by_name: Dict[str, _MetalTemplateStruct],
    ) -> Set[Tuple[str, Tuple[str, ...]]]:
        # Full explicit struct specializations suppress the primary template only
        # for their exact argument list. A name-wide suppression incorrectly blocks
        # unrelated instances such as `BlockMMA<float, ...>` when the source also
        # defines a `BlockMMA<complex64_t, ...>` specialization.
        keys: Set[Tuple[str, Tuple[str, ...]]] = set()
        type_aliases = self._collect_local_type_alias_bindings(
            code,
            [(0, len(code))],
            skip_spans=self._find_template_declaration_spans(code),
        )
        for match in re.finditer(
            r"\btemplate\s*<\s*>\s*(?:struct|class)\s+"
            r"(?P<name>[A-Za-z_][A-Za-z0-9_:]*)\s*<",
            code,
        ):
            angle_start = match.end() - 1
            angle_end = self._find_matching_angle(code, angle_start)
            if angle_end is None:
                continue
            name = match.group("name").split("::")[-1]
            arguments = self._split_top_level_commas(code[angle_start + 1 : angle_end])
            primary_template = primary_templates_by_name.get(name)
            if primary_template is not None:
                arguments = (
                    self._template_arguments_with_resolved_defaults(
                        primary_template,
                        arguments,
                    )
                    or arguments
                )
            canonical_arguments = [
                self._resolve_type_aliases_at(argument, type_aliases, match.start())
                for argument in arguments
            ]
            keys.add(
                self._struct_specialization_comparison_key(name, canonical_arguments)
            )
        return keys

    def _struct_specialization_comparison_key(
        self, struct_name: str, template_arguments: List[str]
    ) -> Tuple[str, Tuple[str, ...]]:
        return (
            struct_name,
            tuple(
                self._normalize_struct_specialization_argument(argument)
                for argument in template_arguments
            ),
        )

    def _normalize_struct_specialization_argument(self, argument: str) -> str:
        text = self._normalize_template_argument_text(argument)
        if not text:
            return ""

        pointer_start = len(text)
        angle_depth = paren_depth = bracket_depth = 0
        for index, char in enumerate(text):
            if char == "<":
                angle_depth += 1
            elif char == ">":
                angle_depth = max(0, angle_depth - 1)
            elif char == "(":
                paren_depth += 1
            elif char == ")":
                paren_depth = max(0, paren_depth - 1)
            elif char == "[":
                bracket_depth += 1
            elif char == "]":
                bracket_depth = max(0, bracket_depth - 1)
            elif char in "*&" and angle_depth == paren_depth == bracket_depth == 0:
                pointer_start = index
                break

        prefix = text[:pointer_start]
        suffix = text[pointer_start:]
        qualifier_spans: List[Tuple[int, int]] = []
        angle_depth = paren_depth = bracket_depth = 0
        index = 0
        while index < len(prefix):
            char = prefix[index]
            if char == "<":
                angle_depth += 1
            elif char == ">":
                angle_depth = max(0, angle_depth - 1)
            elif char == "(":
                paren_depth += 1
            elif char == ")":
                paren_depth = max(0, paren_depth - 1)
            elif char == "[":
                bracket_depth += 1
            elif char == "]":
                bracket_depth = max(0, bracket_depth - 1)
            elif (
                char.isalpha() or char == "_"
            ) and angle_depth == paren_depth == bracket_depth == 0:
                identifier, consumed = self._read_identifier(prefix, index)
                if identifier in {"const", "volatile"}:
                    qualifier_spans.append((index, index + consumed))
                index += consumed
                continue
            index += 1
        if not qualifier_spans:
            return text

        qualifiers = {prefix[start:end] for start, end in qualifier_spans}
        base = self._apply_text_replacements(
            prefix,
            [(start, end, "") for start, end in qualifier_spans],
        )
        base = re.sub(r"\s+", " ", base).strip()
        ordered_qualifiers = [
            qualifier for qualifier in ("const", "volatile") if qualifier in qualifiers
        ]
        return self._normalize_template_argument_text(
            " ".join([*ordered_qualifiers, base]) + suffix
        )

    def _template_arguments_with_resolved_defaults(
        self,
        template: _MetalTemplateStruct,
        template_arguments: List[str],
    ) -> Optional[List[str]]:
        if not self._template_arguments_satisfy_parameters(
            template,
            template_arguments,
        ):
            return None
        substitutions, variadic_bindings = self._template_argument_bindings(
            template,
            template_arguments,
        )
        resolved: List[str] = []
        for parameter in template.template_parameters:
            if parameter in template.variadic_template_parameters:
                resolved.extend(variadic_bindings.get(parameter, []))
                continue
            argument = substitutions.get(parameter)
            if argument is None:
                return None
            declared_type = template.template_parameter_types.get(parameter)
            if declared_type is not None:
                prior_substitutions = {
                    name: value
                    for name, value in substitutions.items()
                    if name != parameter
                }
                concrete_type = self._replace_identifiers(
                    declared_type,
                    prior_substitutions,
                )
                integral_type = self._static_constexpr_integral_type(concrete_type)
                if integral_type is not None:
                    concrete_argument = self._replace_identifiers(
                        argument,
                        prior_substitutions,
                    )
                    folded_argument = self._proven_integral_constant_value(
                        integral_type,
                        concrete_argument,
                    )
                    if folded_argument is not None:
                        argument = folded_argument
                        substitutions[parameter] = argument
            resolved.append(self._normalize_template_argument_text(argument))
        return resolved

    def _dedupe_explicit_template_function_calls(
        self,
        calls: List[Tuple[str, List[str], Tuple[int, int]]],
    ) -> List[Tuple[str, List[str], List[Tuple[int, int]]]]:
        grouped: Dict[
            Tuple[str, Tuple[str, ...]],
            Tuple[str, List[str], List[Tuple[int, int]]],
        ] = {}
        for function_name, template_arguments, span in calls:
            key = self._template_specialization_key(function_name, template_arguments)
            grouped_call = grouped.get(key)
            if grouped_call is None:
                grouped[key] = (function_name, template_arguments, [span])
            else:
                grouped_call[2].append(span)
        return list(grouped.values())

    def _host_name_from_attributes(self, declaration: str) -> str:
        match = re.search(
            r"\[\[\s*host_name\s*\(\s*(?P<host>"
            + METAL_STRING_EXPRESSION_PATTERN
            + r")\s*\)\s*\]\]",
            declaration,
            re.DOTALL,
        )
        if match is None:
            return ""
        return self._evaluate_metal_string_expression(match.group("host"))

    def _template_id_candidates(self, declaration: str) -> List[Tuple[str, List[str]]]:
        candidates: List[Tuple[str, List[str]]] = []
        i = 0
        while i < len(declaration):
            if declaration[i] in "\"'":
                _literal, consumed = self._read_string(declaration, i)
                i += consumed
                continue
            if declaration.startswith("//", i):
                end = declaration.find("\n", i)
                if end == -1:
                    break
                i = end + 1
                continue
            if declaration.startswith("/*", i):
                end = declaration.find("*/", i + 2)
                if end == -1:
                    break
                i = end + 2
                continue
            if declaration[i].isalpha() or declaration[i] == "_":
                ident, consumed = self._read_identifier(declaration, i)
                scoped_start = self._scoped_identifier_start(declaration, i)
                name = declaration[scoped_start : i + consumed]
                j = i + consumed
                while j < len(declaration) and declaration[j].isspace():
                    j += 1
                if j < len(declaration) and declaration[j] == "<":
                    angle_end = self._find_matching_angle(declaration, j)
                    if angle_end is not None:
                        arguments = self._split_top_level_commas(
                            declaration[j + 1 : angle_end]
                        )
                        if ident not in {"decltype", "static_cast", "as_type"}:
                            candidates.append((name, arguments))
                        i = angle_end + 1
                        continue
                i += consumed
                continue
            i += 1
        return candidates

    def _declared_function_name(self, declaration: str) -> Optional[str]:
        function_name = self._function_name_from_header(declaration)
        if function_name in {"decltype", "static_cast", "as_type"}:
            return None
        return function_name

    def _declared_template_id_candidate(
        self,
        declaration: str,
        declared_function_name: Optional[str],
    ) -> Optional[Tuple[str, List[str]]]:
        candidates = [
            (name, arguments)
            for name, arguments in self._template_id_candidates(declaration)
            if arguments
        ]
        if not candidates:
            return None
        if declared_function_name is None:
            return candidates[0]
        for name, arguments in candidates:
            if name.split("::")[-1] == declared_function_name:
                return name, arguments
        return None

    def _infer_declared_template_arguments(
        self,
        template: _MetalTemplateFunction,
        declaration: str,
    ) -> List[str]:
        template_parameters = list(template.template_parameters)
        if not template_parameters:
            return []
        template_parameter_set = set(template_parameters)
        template_param_types = self._function_parameter_type_texts(template.source)
        declaration_param_types = self._function_parameter_type_texts(declaration)
        if not template_param_types or not declaration_param_types:
            return []

        bindings: Dict[str, str] = {}
        for template_type, concrete_type in zip(
            template_param_types,
            declaration_param_types,
        ):
            self._infer_template_parameter_bindings_from_type(
                template_type,
                concrete_type,
                template_parameter_set,
                bindings,
            )
        if not bindings:
            return []

        arguments: List[str] = []
        defaults = getattr(template, "template_parameter_defaults", {}) or {}
        substitutions: Dict[str, str] = {}
        last_inferred_index = -1
        for index, name in enumerate(template_parameters):
            if name in bindings:
                argument = bindings[name]
                last_inferred_index = index
            else:
                default_argument = defaults.get(name)
                if default_argument is None:
                    return []
                argument = self._resolve_template_default_argument(
                    default_argument,
                    substitutions,
                    template,
                )
            substitutions[name] = argument
            arguments.append(argument)

        if last_inferred_index == -1:
            return []
        return arguments[: last_inferred_index + 1]

    def _function_parameter_type_texts(self, declaration: str) -> List[str]:
        open_paren = self._function_parameter_start(declaration)
        if open_paren is None:
            return []
        close_paren = self._find_matching_delimiter(declaration, open_paren, "(", ")")
        if close_paren is None:
            return []
        parameters = self._split_top_level_commas(
            declaration[open_paren + 1 : close_paren]
        )
        return [
            normalized
            for normalized in (
                self._normalize_function_parameter_type_text(parameter)
                for parameter in parameters
            )
            if normalized and normalized != "void"
        ]

    def _normalize_function_parameter_type_text(self, parameter: str) -> str:
        parameter = self._strip_top_level_default_value(parameter)
        attributes = " ".join(self._metal_attributes(parameter))
        type_text = self._strip_metal_attributes(parameter)
        type_text = re.sub(
            r"\s+\b[A-Za-z_][A-Za-z0-9_]*\s*$",
            "",
            type_text.strip(),
        )
        normalized = self._normalize_template_argument_text(type_text)
        if attributes:
            normalized_attributes = self._normalize_template_argument_text(attributes)
            if normalized:
                return f"{normalized} {normalized_attributes}"
            return normalized_attributes
        return normalized

    def _function_parameter_value_type(self, parameter: str) -> str:
        normalized = self._normalize_inferred_type(
            self._normalize_function_parameter_type_text(parameter)
        )
        return normalized.rstrip("&").strip()

    def _metal_attributes(self, text: str) -> List[str]:
        attributes: List[str] = []
        i = 0
        while i < len(text):
            if text.startswith("[[", i):
                end = text.find("]]", i + 2)
                if end == -1:
                    break
                attributes.append(text[i : end + 2])
                i = end + 2
                continue
            i += 1
        return attributes

    def _strip_metal_attributes(self, text: str) -> str:
        result = ""
        i = 0
        while i < len(text):
            if text.startswith("[[", i):
                end = text.find("]]", i + 2)
                if end == -1:
                    break
                i = end + 2
                continue
            result += text[i]
            i += 1
        return result

    def _strip_top_level_default_value(self, text: str) -> str:
        return strip_top_level_default_value(text, self._read_string)

    def _infer_template_parameter_bindings_from_type(
        self,
        template_type: str,
        concrete_type: str,
        template_parameters: Set[str],
        bindings: Dict[str, str],
    ) -> None:
        specialization = self._materialized_struct_specializations.get(
            self._normalize_inferred_type(concrete_type)
        )
        if specialization is not None:
            source_name = specialization[0]
            if re.search(
                rf"(?<![A-Za-z0-9_]){re.escape(source_name)}\s*<", template_type
            ):
                template_type = self._canonical_template_binding_pointee_type(
                    template_type, context=concrete_type
                )
                concrete_type = self._canonical_template_binding_pointee_type(
                    concrete_type
                )
        template_id = self._exact_template_type_id(template_type)
        concrete_id = self._exact_template_type_id(concrete_type)
        if template_id is not None and concrete_id is not None:
            if template_id[0] != concrete_id[0] or len(template_id[1]) != len(
                concrete_id[1]
            ):
                return
        captures: List[str] = []
        pattern_parts: List[str] = []
        position = 0
        for match in IDENTIFIER_RE.finditer(template_type):
            pattern_parts.append(re.escape(template_type[position : match.start()]))
            identifier = match.group(0)
            if identifier in template_parameters:
                pattern_parts.append("(.+?)")
                captures.append(identifier)
            else:
                pattern_parts.append(re.escape(identifier))
            position = match.end()
        pattern_parts.append(re.escape(template_type[position:]))
        if not captures:
            return

        match = re.fullmatch("".join(pattern_parts), concrete_type)
        if match is None:
            return
        inferred: Dict[str, str] = {}
        for index, parameter in enumerate(captures, start=1):
            value = self._normalize_template_argument_text(match.group(index))
            if not value:
                return
            existing = inferred.get(parameter, bindings.get(parameter))
            if existing is not None and existing != value:
                return
            inferred[parameter] = value
        bindings.update(inferred)

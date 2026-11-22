"""Library fallback lookup, explicit template struct instantiation and the elision of stateless compile-time globals."""

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

from crosstl.translator.codegen.metal_library_functions import MSL_LIBRARY_FUNCTIONS

from ..function_specialization import _metal_function_parameter_declarations
from ..library_fallback_lookup import (
    library_fallback_obstacle,
    library_using_declaration_present,
)
from ..library_overloads import (
    LIBRARY_OVERLOAD_NAMES,
    library_overload_precedence,
    library_type_spelling,
)
from ..simd_group_template_overloads import simd_group_library_overload_selected
from .support import (
    IDENTIFIER_RE,
    METAL_STRING_EXPRESSION_PATTERN,
    MetalTemplateSpecializationError,
    _MetalCompileTimeGlobal,
    _MetalConstexprCall,
    _MetalNamespaceVisibility,
    _MetalStructDefinition,
    _MetalTemplateFunction,
    _MetalTemplateStruct,
)


class MetalStatelessGlobalMixin:
    """Members of ``MetalPreprocessor``: library fallback lookup, explicit template struct instantiation and the elision of stateless compile-time globals."""

    def _library_lookup_obstacle(
        self,
        masked_code: str,
        call: _MetalConstexprCall,
        lookup_position: int,
        candidates: Sequence[_MetalTemplateFunction],
        namespace_visibility: _MetalNamespaceVisibility,
    ) -> Optional[str]:
        """Why lookup at ``call`` misses the Metal library function, or ``None`` when it finds it."""
        if library_using_declaration_present(masked_code, call.name):
            return None
        return library_fallback_obstacle(
            name=call.name,
            reference_namespace=self._metal_call_reference_namespace(
                call.qualified_name
            ),
            call_namespace=self._namespace_at(
                list(namespace_visibility.namespace_spans), lookup_position
            ),
            position=lookup_position,
            candidate_namespaces=[template.namespace for template in candidates],
            directives=namespace_visibility.using_namespace_directives,
            directive_targets=lambda directive: self._metal_namespace_reference_candidates(
                directive.target,
                directive.declaration_namespace,
                globally_qualified=directive.globally_qualified,
            ),
        )

    def _library_argument_types(
        self, argument_types: Sequence[Optional[str]]
    ) -> Optional[List[str]]:
        """The library spellings of scalar and vector argument types, or ``None`` if one is neither."""
        spellings = []
        for argument_type in argument_types:
            decomposed = (
                self._scalar_and_width(argument_type)
                if argument_type is not None
                else None
            )
            spelling = (
                library_type_spelling(*decomposed) if decomposed is not None else None
            )
            if spelling is None:
                return None
            spellings.append(spelling)
        return spellings

    def _library_parameter_type(self, type_text: str) -> Optional[str]:
        """The library spelling of a scalar or vector parameter, ``thread T &`` for a non-const reference."""
        text = self._normalize_template_argument_text(type_text)
        reference = text.endswith("&")
        if reference:
            text = text[:-1].strip()
            # A const reference binds any argument as a by-value parameter does.
            reference = "const" not in text.split(" ")
        decomposed = self._scalar_and_width(text)
        spelling = (
            library_type_spelling(*decomposed) if decomposed is not None else None
        )
        if spelling is None:
            return None
        return f"thread {spelling} &" if reference else spelling

    def _require_library_fallback_lookup(
        self,
        code: str,
        masked_code: str,
        call: _MetalConstexprCall,
        lookup_position: int,
        candidates: Sequence[_MetalTemplateFunction],
        argument_types: Sequence[Optional[str]],
        namespace_visibility: _MetalNamespaceVisibility,
        *,
        has_other_declarations: bool,
    ) -> None:
        """Refuse a call whose disabled templates hide the Metal library function it would fall back to."""
        if (
            has_other_declarations
            or self._library_argument_types(argument_types) is None
        ):
            # Other declarations or argument-dependent lookup can still supply a viable function.
            return
        obstacle = self._library_lookup_obstacle(
            masked_code, call, lookup_position, candidates, namespace_visibility
        )
        if obstacle is not None:
            self._raise_constrained_free_function_error(
                code,
                call,
                f"every constrained candidate is disabled and {obstacle}",
            )

    def _library_overload_selected(
        self,
        code: str,
        masked_code: str,
        call: _MetalConstexprCall,
        lookup_position: int,
        candidates: Sequence[_MetalTemplateFunction],
        argument_types: Sequence[Optional[str]],
        namespace_visibility: _MetalNamespaceVisibility,
        template: _MetalTemplateFunction,
        bindings: Mapping[str, str],
    ) -> bool:
        """Whether a Metal library overload, not the enabled ``template``, is the best function for ``call``."""
        simd_group_selected = simd_group_library_overload_selected(
            self, code, call, argument_types, template, bindings
        )
        if simd_group_selected is not None:
            return simd_group_selected
        if (
            call.name not in LIBRARY_OVERLOAD_NAMES
            and call.name not in MSL_LIBRARY_FUNCTIONS
        ):
            return False
        arguments = self._library_argument_types(argument_types)
        if arguments is None:
            return False
        if (
            self._library_lookup_obstacle(
                masked_code, call, lookup_position, candidates, namespace_visibility
            )
            is not None
        ):
            return False
        parameters = self._template_function_parameter_text(template) or ""
        declarations = _metal_function_parameter_declarations(
            self, f"{template.name}({parameters})"
        )
        parameter_types = [
            (
                None
                if variadic
                else self._library_parameter_type(
                    self._substitute_static_type_bindings(type_text, bindings)
                )
            )
            for type_text, _name, variadic in declarations
        ]
        precedence = (
            library_overload_precedence(call.name, arguments, parameter_types)
            if call.name in LIBRARY_OVERLOAD_NAMES
            else "unproven"
        )
        if precedence == "unproven":
            self._raise_constrained_free_function_error(
                code,
                call,
                f"the enabled template competes with the Metal library's '{call.name}' "
                "overloads, and which one overload resolution selects is unproven",
            )
        return precedence == "library"

    def _materialize_explicit_template_struct_instantiations(
        self,
        code: str,
        *,
        work_budget: Optional[object] = None,
        allow_partial_object_specializations: bool = True,
    ) -> str:
        # Struct counterpart of _materialize_explicit_template_function_calls
        # (issue #1354). Replaces concrete `StructName<args>` type references with
        # a materialized concrete struct, iterating so nested instantiations
        # (e.g. BlockLoader referencing BlockMMA<...>) resolve. Regression-safe:
        # any failure returns the unmodified source, so a kernel that previously
        # reached the template-materialization diagnostic is left unchanged rather
        # than emitting a half-rewritten translation.
        previous_sizeof_source = self._enter_sizeof_source(code)
        try:
            return self._materialize_explicit_template_struct_instantiations_impl(
                code,
                work_budget=work_budget,
                allow_partial_object_specializations=(
                    allow_partial_object_specializations
                ),
            )
        except MetalTemplateSpecializationError:
            raise
        except Exception:
            return code
        finally:
            self._active_sizeof_source = previous_sizeof_source

    def _materialize_explicit_template_struct_instantiations_impl(
        self,
        code: str,
        *,
        work_budget: Optional[object] = None,
        allow_partial_object_specializations: bool = False,
    ) -> str:
        concrete_name_counts: Dict[str, int] = {}
        for struct in self._find_concrete_struct_definitions(code):
            concrete_name_counts[struct.name] = (
                concrete_name_counts.get(struct.name, 0) + 1
            )
        # Project materialization can expose new references and run this pass
        # again. Reuse only a recorded specialization with one surviving concrete
        # declaration; missing or duplicate declarations must be handled again.
        materialized_names: Dict[Tuple[str, Tuple[str, ...]], str] = {
            (source_name, tuple(source_arguments)): specialized_name
            for specialized_name, (
                source_name,
                source_arguments,
            ) in self._materialized_struct_specializations.items()
            if concrete_name_counts.get(specialized_name) == 1
        }
        working = code
        # Each iteration resolves one "layer" of instantiations; the bound mirrors
        # the specialization budget so deeply nested or pathological inputs cannot
        # loop unbounded.
        max_iterations = self.max_template_specializations + 1
        for _ in range(max_iterations):
            working = self._canonicalize_struct_static_alias_owners(working)
            templates = self._find_template_structs(
                working, include_forward_declarations=True
            )
            if not templates:
                return working
            primary_templates = [
                template
                for template in templates
                if self._template_struct_specialization_arguments(template) is None
            ]
            templates_by_name = {}
            for template in primary_templates:
                if (
                    template.name not in templates_by_name
                    or not template.is_forward_declaration
                ):
                    templates_by_name[template.name] = template
            if not templates_by_name:
                return working
            explicit_specialization_keys = (
                self._find_explicit_struct_specialization_keys(
                    working,
                    templates_by_name,
                )
            )
            partial_specializations: Dict[
                str, List[Tuple[_MetalTemplateStruct, List[str]]]
            ] = {}
            for template in templates:
                specialization_arguments = (
                    self._template_struct_specialization_arguments(template)
                )
                if specialization_arguments is None:
                    continue
                partial_specializations.setdefault(template.name, []).append(
                    (template, specialization_arguments)
                )
            allow_unqualified_remove_cv_t = (
                self._metal_unqualified_remove_cv_t_available(working)
            )
            excluded_spans = self._find_template_declaration_spans(working)
            self._explicit_struct_reference_sink = []
            try:
                (
                    instantiations,
                    selected_partial_specializations,
                ) = self._find_explicit_template_struct_instantiations(
                    working,
                    templates_by_name,
                    excluded_spans,
                    explicit_specialization_keys,
                    partial_specializations,
                    allow_partial_object_specializations=(
                        allow_partial_object_specializations
                    ),
                    allow_unqualified_remove_cv_t=allow_unqualified_remove_cv_t,
                )
                explicit_references = self._explicit_struct_reference_sink
            finally:
                self._explicit_struct_reference_sink = None
            concretized = self._concretize_method_explicit_struct_specializations(
                working,
                templates_by_name,
                explicit_references,
                materialized_names,
            )
            if concretized is not None:
                # Spans changed; rediscover instantiations against the new text.
                working = concretized
                continue
            if not instantiations:
                return working

            replacements: List[Tuple[int, int, str]] = []
            new_materializations: List[str] = []
            for (
                struct_name,
                template_arguments,
                spans,
            ) in self._dedupe_explicit_template_function_calls(instantiations):
                key = self._template_specialization_key(struct_name, template_arguments)
                canonical_arguments = self._canonical_materialized_struct_arguments(
                    template_arguments, materialized_names
                )
                canonical_key = self._template_specialization_key(
                    struct_name, canonical_arguments
                )
                if canonical_key != key and canonical_key in materialized_names:
                    # `Tile<float, BaseFrag<float, 8, 8>>` and `Tile<float,
                    # BaseFrag_float_8_8>` name one type once the inner
                    # specialization exists, so they share one specialization.
                    replacements.extend(
                        (span[0], span[1], materialized_names[canonical_key])
                        for span in spans
                    )
                    continue
                partial_selection = selected_partial_specializations.get(key)
                primary_template = templates_by_name[struct_name]
                if partial_selection is None and not (
                    self._template_arguments_satisfy_parameters(
                        primary_template, template_arguments
                    )
                ):
                    continue
                selected_template = primary_template
                selected_bindings: Dict[str, str]
                if partial_selection is None:
                    selected_bindings, _variadic_bindings = (
                        self._template_argument_bindings(
                            primary_template,
                            template_arguments,
                        )
                    )
                else:
                    selected_template, selected_bindings = partial_selection

                alias_replacements: List[Tuple[int, int, str]] = []
                materialization_spans: List[Tuple[int, int]] = []
                for span in spans:
                    qualified_alias = self._qualified_template_struct_alias_at(
                        working,
                        span,
                    )
                    if qualified_alias is None:
                        materialization_spans.append(span)
                        continue
                    alias_name, alias_span = qualified_alias
                    alias_target = self._concrete_template_struct_scalar_alias_target(
                        selected_template,
                        selected_bindings,
                        alias_name,
                        allow_unqualified_remove_cv_t=(allow_unqualified_remove_cv_t),
                    )
                    if alias_target is None:
                        materialization_spans.append(span)
                        continue
                    alias_replacements.append(
                        (alias_span[0], alias_span[1], alias_target)
                    )
                replacements.extend(alias_replacements)
                if not materialization_spans:
                    continue

                specialized_name = materialized_names.get(key)
                if specialized_name is not None:
                    replacements.extend(
                        (span[0], span[1], specialized_name)
                        for span in materialization_spans
                    )
                    continue
                if len(materialized_names) >= self.max_template_specializations:
                    # Stay regression-safe instead of raising: leave the residual
                    # `Name<...>` so the existing template-materialization
                    # diagnostic fires exactly as it does today.
                    return code
                if work_budget is not None:
                    consume_unique = getattr(work_budget, "consume_unique", None)
                    work_key = ("struct", key[0], *key[1])
                    context = (
                        "reachable concrete struct "
                        f"'{self._template_specialization_signature(struct_name, template_arguments)}'"
                    )
                    first_span = materialization_spans[0]
                    if callable(consume_unique):
                        consume_unique(
                            work_key,
                            offset=first_span[0],
                            length=max(first_span[1] - first_span[0], 0),
                            context=context,
                        )
                    else:
                        work_budget.consume(
                            1,
                            offset=first_span[0],
                            length=max(first_span[1] - first_span[0], 0),
                            context=context,
                        )
                specialized_name = self._unique_struct_specialization_identifier(
                    struct_name, key
                )
                if partial_selection is None:
                    materialized = self._materialize_template_struct_with_name(
                        primary_template,
                        template_arguments,
                        specialized_name,
                    )
                else:
                    partial_template, partial_bindings = partial_selection
                    materialized = self._materialize_partial_template_struct_with_name(
                        partial_template,
                        partial_bindings,
                        specialized_name,
                    )
                if materialized:
                    replacements.extend(
                        (span[0], span[1], specialized_name)
                        for span in materialization_spans
                    )
                    materialized_names[key] = specialized_name
                    self._materialized_struct_specializations[specialized_name] = (
                        struct_name,
                        tuple(key[1]),
                    )
                    self._materialized_struct_primary_templates[specialized_name] = (
                        primary_template
                    )
                    selected_source_template = (
                        primary_template
                        if partial_selection is None
                        else partial_selection[0]
                    )
                    self._materialized_struct_specialization_namespaces[
                        specialized_name
                    ] = selected_source_template.namespace
                    new_materializations.append(materialized)

            if not replacements and not new_materializations:
                return working

            working = self._apply_text_replacements(working, replacements)
            if new_materializations:
                working = working.rstrip() + "\n\n" + "\n\n".join(new_materializations)
                if not working.endswith("\n"):
                    working += "\n"
        return working

    def _prune_unreferenced_template_struct_declarations(self, code: str) -> str:
        declarations = self._find_all_template_struct_declaration_spans(code)
        if not declarations:
            return code

        declarations_by_name: Dict[str, List[Tuple[int, int]]] = {}
        for name, span in declarations:
            declarations_by_name.setdefault(name, []).append(span)
        declaration_spans = sorted(span for _name, span in declarations)
        owners_by_span = {span: name for name, span in declarations}
        ignored_spans = self._find_comment_and_literal_spans(code)

        dependencies: Dict[str, Set[str]] = {
            name: set() for name in declarations_by_name
        }
        live_names: Set[str] = set()
        names_pattern = "|".join(
            re.escape(name)
            for name in sorted(declarations_by_name, key=len, reverse=True)
        )
        reference_pattern = re.compile(rf"\b(?P<name>{names_pattern})\s*<")
        for match in reference_pattern.finditer(code):
            if self._containing_span(match.start(), ignored_spans) is not None:
                continue
            referenced_name = match.group("name")
            containing = [
                span for span in declaration_spans if span[0] <= match.start() < span[1]
            ]
            if not containing:
                live_names.add(referenced_name)
                continue
            owner_span = min(containing, key=lambda span: span[1] - span[0])
            owner_name = owners_by_span[owner_span]
            if owner_name != referenced_name:
                dependencies[owner_name].add(referenced_name)

        pending = list(live_names)
        while pending:
            owner_name = pending.pop()
            for dependency in dependencies.get(owner_name, set()):
                if dependency in live_names:
                    continue
                live_names.add(dependency)
                pending.append(dependency)

        replacements = [
            (span[0], span[1], "")
            for name, span in declarations
            if name not in live_names
        ]
        if not replacements:
            return code
        return self._apply_text_replacements(code, replacements)

    def _find_all_template_struct_declaration_spans(
        self, code: str
    ) -> List[Tuple[str, Tuple[int, int]]]:
        declarations: List[Tuple[str, Tuple[int, int]]] = []
        for span in self._find_template_declaration_spans(code):
            angle_start = code.find("<", span[0], span[1])
            if angle_start == -1:
                continue
            angle_end = self._find_matching_template_param_angle(code, angle_start)
            if angle_end is None or angle_end >= span[1]:
                continue
            header = code[angle_end + 1 : span[1]]
            match = re.match(
                r"\s*(?:\[\[[^\]]*\]\]\s*)*(?:struct|class)\s+"
                r"(?P<name>[A-Za-z_][A-Za-z0-9_:]*)\b",
                header,
                re.DOTALL,
            )
            if match is None:
                continue
            declarations.append((match.group("name").split("::")[-1], span))
        return declarations

    def _elide_stateless_compile_time_globals(self, code: str) -> str:
        """Erase compile-time objects only after proving all observable effects.

        This runs before generic member lowering because empty variadic methods
        otherwise lose their bodies before overload selection. Any uncertain
        layout, lifetime operation, receiver use, or method body raises a
        source-located diagnostic instead of producing a dangling target call.
        """
        structs = self._find_concrete_struct_definitions(code)
        if not structs:
            return code
        globals_ = self._find_compile_time_struct_globals(code, structs)
        if not globals_:
            return code

        structs_by_name = {struct.qualified_name: struct for struct in structs}
        namespace_spans = self._find_namespace_spans(code)
        declaration_spans = [
            span
            for global_ in globals_
            for span in (global_.span, *global_.prior_declaration_spans)
        ]
        global_names = {global_.name for global_ in globals_}
        replacements: List[Tuple[int, int, str]] = []
        for global_ in globals_:
            struct = structs_by_name[global_.type_name]
            if not self._stateless_global_layout_is_proven_empty(code, struct):
                continue
            self._prove_stateless_global_constructor(
                code, global_, struct, global_names
            )
            self._prove_stateless_global_destructor(code, global_, struct)
            replacements.extend(
                self._stateless_global_call_replacements(
                    code,
                    global_,
                    struct,
                    namespace_spans,
                    declaration_spans,
                    global_names,
                )
            )
            replacements.extend(
                (start, end, "") for start, end in global_.prior_declaration_spans
            )
            replacements.append((global_.span[0], global_.span[1], ""))
        return self._apply_text_replacements(code, replacements)

    def _find_compile_time_struct_globals(
        self,
        code: str,
        structs: List[_MetalStructDefinition],
    ) -> List[_MetalCompileTimeGlobal]:
        spellings = {
            spelling
            for struct in structs
            for spelling in (struct.name, struct.qualified_name)
        }
        if not spellings:
            return []
        type_pattern = "|".join(
            self._qualified_type_pattern(spelling)
            for spelling in sorted(spellings, key=len, reverse=True)
        )
        declaration_pattern = re.compile(
            r"(?<![A-Za-z0-9_])"
            r"(?P<qualifiers>(?:(?:static|inline|constexpr|const|constant|"
            r"extern|typedef|thread_local|volatile)\s+)+)"
            rf"(?P<type>(?:::)?(?:{type_pattern}))\s+"
            r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)"
        )
        ignored_spans = self._find_comment_and_literal_spans(code)
        masked_code = self._mask_comments_and_literals(code)
        lexical_scopes = self._find_lexical_brace_scopes(code)
        namespace_spans = self._find_namespace_spans(code)
        namespace_scopes = {(start, end) for start, end, _name in namespace_spans}
        structs_by_qualified = {struct.qualified_name: struct for struct in structs}
        globals_: List[_MetalCompileTimeGlobal] = []

        for match in declaration_pattern.finditer(code):
            if self._containing_span(match.start(), ignored_spans) is not None:
                continue
            line_start = masked_code.rfind("\n", 0, match.start()) + 1
            line_prefix = masked_code[line_start : match.start()]
            if line_prefix.strip():
                previous = match.start() - 1
                while previous >= line_start and masked_code[previous].isspace():
                    previous -= 1
                if previous >= line_start and masked_code[previous] not in ";{}":
                    continue
            scope = self._innermost_lexical_scope(
                lexical_scopes, match.start(), len(code)
            )
            if scope != (0, len(code)) and scope not in namespace_scopes:
                continue
            qualifiers = set(IDENTIFIER_RE.findall(match.group("qualifiers")))
            if not ({"constant", "constexpr"} & qualifiers):
                continue
            if qualifiers & {"extern", "typedef", "thread_local", "volatile"}:
                continue
            namespace = self._namespace_at(namespace_spans, match.start())
            if self._defines_function_after(code, match.end("name")):
                # `constexpr T name() { ... }` is a function, whatever the scope of `T`.
                continue
            source_type = re.sub(r"\s*::\s*", "::", match.group("type"))
            if source_type.startswith("::"):
                source_type = source_type[2:]
            struct = structs_by_qualified.get(source_type)
            if struct is None:
                candidates = [
                    candidate
                    for candidate in structs
                    if candidate.name == source_type
                    and self._struct_namespace(candidate) == namespace
                ]
                if len(candidates) != 1:
                    self._raise_stateless_global_error(
                        code,
                        global_name=match.group("name"),
                        type_name=source_type,
                        reason="ambiguous-type",
                        start=match.start("type"),
                        end=match.end("type"),
                        blocking_use=match.group("type"),
                    )
                struct = candidates[0]

            parsed = self._parse_compile_time_global_initializer(
                code,
                match.end("name"),
                {source_type, struct.name, struct.qualified_name},
            )
            if parsed is None:
                continue
            arguments, declaration_end = parsed
            prior_declaration_spans = self._find_prior_compile_time_global_declarations(
                code,
                before=match.start(),
                name=match.group("name"),
                expected_type_names={
                    source_type,
                    struct.name,
                    struct.qualified_name,
                },
                namespace=namespace,
                lexical_scopes=lexical_scopes,
                namespace_scopes=namespace_scopes,
                namespace_spans=namespace_spans,
            )
            globals_.append(
                _MetalCompileTimeGlobal(
                    name=match.group("name"),
                    type_name=struct.qualified_name,
                    namespace=namespace,
                    arguments=tuple(arguments),
                    span=(match.start(), declaration_end),
                    name_span=(match.start("name"), match.end("name")),
                    receiver_address_space="constant",
                    prior_declaration_spans=tuple(prior_declaration_spans),
                )
            )
        return globals_

    def _find_prior_compile_time_global_declarations(
        self,
        code: str,
        *,
        before: int,
        name: str,
        expected_type_names: Set[str],
        namespace: str,
        lexical_scopes: List[Tuple[int, int]],
        namespace_scopes: Set[Tuple[int, int]],
        namespace_spans: List[Tuple[int, int, str]],
    ) -> List[Tuple[int, int]]:
        type_pattern = "|".join(
            self._qualified_type_pattern(spelling)
            for spelling in sorted(expected_type_names, key=len, reverse=True)
        )
        pattern = re.compile(
            r"(?<![A-Za-z0-9_])"
            r"(?P<qualifiers>(?:(?:static|inline|constexpr|const|constant|"
            r"extern|typedef|thread_local|volatile)\s+)+)"
            rf"(?P<type>(?:::)?(?:{type_pattern}))\s+"
            rf"{re.escape(name)}\s*;"
        )
        masked = self._mask_comments_and_literals(code)
        declarations: List[Tuple[int, int]] = []
        for match in pattern.finditer(masked, 0, before):
            qualifiers = set(IDENTIFIER_RE.findall(match.group("qualifiers")))
            if "extern" not in qualifiers or not (
                {"constant", "constexpr"} & qualifiers
            ):
                continue
            scope = self._innermost_lexical_scope(
                lexical_scopes, match.start(), len(code)
            )
            if scope != (0, len(code)) and scope not in namespace_scopes:
                continue
            declaration_namespace = self._namespace_at(namespace_spans, match.start())
            if declaration_namespace != namespace:
                continue
            source_type = re.sub(r"\s*::\s*", "::", match.group("type"))
            if source_type.startswith("::"):
                source_type = source_type[2:]
            if source_type not in expected_type_names:
                continue
            declarations.append(match.span())
        return declarations

    @staticmethod
    def _qualified_type_pattern(type_name: str) -> str:
        return r"\s*::\s*".join(re.escape(part) for part in type_name.split("::"))

    @staticmethod
    def _struct_namespace(struct: _MetalStructDefinition) -> str:
        if "::" not in struct.qualified_name:
            return ""
        return struct.qualified_name.rsplit("::", 1)[0]

    def _defines_function_after(self, code: str, name_end: int) -> bool:
        """Whether the declarator ending at ``name_end`` has a parameter list followed by a function body."""
        cursor = self._skip_cpp_trivia(code, name_end)
        if cursor >= len(code) or code[cursor] != "(":
            return False
        end = self._find_matching_delimiter(code, cursor, "(", ")")
        if end is None:
            return False
        after = self._skip_cpp_trivia(code, end + 1)
        return bool(
            re.match(
                r"(?:(?:const|noexcept|thread|device)\b\s*)*(?:\{|->)", code[after:]
            )
        )

    def _parse_compile_time_global_initializer(
        self, code: str, name_end: int, expected_type_names: Set[str]
    ) -> Optional[Tuple[List[str], int]]:
        cursor = self._skip_cpp_trivia(code, name_end)
        if cursor >= len(code):
            return None
        if code[cursor] == ";":
            return [], cursor + 1
        if code[cursor] in "({":
            opener = code[cursor]
            closer = ")" if opener == "(" else "}"
            end = self._find_matching_delimiter(code, cursor, opener, closer)
            if end is None:
                return None
            after = self._skip_cpp_trivia(code, end + 1)
            if after >= len(code) or code[after] != ";":
                return None
            text = code[cursor + 1 : end]
            # `Type value();` is a function declaration, not an object.
            if opener == "(" and (
                not text.strip()
                or self._looks_like_function_parameter_declarations(text)
            ):
                return None
            return self._split_initializer_arguments(text), after + 1
        if code[cursor] != "=":
            return None
        declaration_end = self._statement_end(code, cursor)
        if declaration_end is None:
            return None
        initializer = code[cursor + 1 : declaration_end - 1].strip()
        if initializer.startswith("{"):
            end = self._find_matching_delimiter(initializer, 0, "{", "}")
            if end == len(initializer) - 1:
                return (
                    self._split_initializer_arguments(initializer[1:end]),
                    declaration_end,
                )
        constructor = re.match(
            r"(?P<type>(?:::)?[A-Za-z_][A-Za-z0-9_]*"
            r"(?:\s*::\s*[A-Za-z_]\w*)*)\s*(?P<open>[({])",
            initializer,
        )
        if constructor is None:
            return None
        constructor_type = re.sub(r"\s*::\s*", "::", constructor.group("type"))
        if constructor_type.startswith("::"):
            constructor_type = constructor_type[2:]
        if constructor_type not in expected_type_names:
            return None
        opener = constructor.group("open")
        open_index = constructor.end() - 1
        closer = ")" if opener == "(" else "}"
        end = self._find_matching_delimiter(initializer, open_index, opener, closer)
        if end != len(initializer) - 1:
            return None
        return (
            self._split_initializer_arguments(initializer[open_index + 1 : end]),
            declaration_end,
        )

    def _split_initializer_arguments(self, text: str) -> List[str]:
        return [
            argument.strip()
            for argument in self._split_top_level_commas(text)
            if argument.strip()
        ]

    def _looks_like_function_parameter_declarations(self, text: str) -> bool:
        parameters = self._split_top_level_commas(text)
        if not parameters:
            return True
        for parameter in parameters:
            declarator, _default = self._split_top_level_assignment(parameter)
            declarator = declarator.strip()
            if not declarator or declarator in {"true", "false", "nullptr"}:
                return False
            if any(character in declarator for character in "\"'(){}"):
                return False
            if re.search(r"[+\-/%|^!?=]", declarator):
                return False
            if IDENTIFIER_RE.search(declarator) is None:
                return False
        return True

    def _stateless_global_layout_is_proven_empty(
        self,
        code: str,
        struct: _MetalStructDefinition,
    ) -> bool:
        if struct.base_clause or struct.data_members:
            return False
        body = code[struct.body_span[0] : struct.body_span[1]]
        return self._first_unproven_struct_state_declaration(body) is None

    def _first_unproven_struct_state_declaration(self, body: str) -> Optional[str]:
        i = 0
        while i < len(body):
            i = self._skip_cpp_trivia(body, i)
            if i >= len(body):
                return None
            access = re.match(r"(?:public|private|protected)\s*:", body[i:])
            if access is not None:
                i += access.end()
                continue

            declaration_start = i
            if re.match(r"template\s*<", body[i:]):
                angle_start = body.find("<", i)
                angle_end = self._find_matching_template_param_angle(body, angle_start)
                if angle_end is None:
                    return body[declaration_start:].strip()
                i = angle_end + 1

            brace = self._find_next_top_level_char(body, i, "{")
            semicolon = self._find_next_top_level_char(body, i, ";")
            if brace is not None and (semicolon is None or brace < semicolon):
                header = body[i:brace]
                brace_end = self._find_matching_brace(body, brace)
                if brace_end is None:
                    return body[declaration_start:].strip()
                if not self._stateless_member_header_is_function(header):
                    end = self._statement_end(body, declaration_start) or brace_end
                    return body[declaration_start:end].strip()
                i = brace_end
                trailing = self._skip_cpp_trivia(body, i)
                if trailing < len(body) and body[trailing] == ";":
                    i = trailing + 1
                continue
            if semicolon is None:
                trailing = body[declaration_start:].strip()
                return trailing or None
            declaration = body[i:semicolon].strip()
            if not declaration:
                i = semicolon + 1
                continue
            if self._declaration_is_type_alias(declaration):
                i = semicolon + 1
                continue
            if re.match(r"static_assert\s*\(", declaration):
                i = semicolon + 1
                continue
            if self._stateless_member_declaration_is_function(declaration):
                i = semicolon + 1
                continue
            return body[declaration_start : semicolon + 1].strip()
        return None

    def _stateless_member_header_is_function(self, header: str) -> bool:
        if re.search(r"\bvirtual\b", header):
            return False
        paren_start = self._function_parameter_start(header)
        if paren_start is None:
            return False
        prefix = header[:paren_start]
        if "=" in prefix or re.search(r"\[[^\]]*\]\s*$", prefix):
            return False
        return bool(
            re.search(
                r"(?:~\s*)?[A-Za-z_][A-Za-z0-9_]*\s*$|"
                r"\boperator(?:\s*\(\s*\)|\s*[^\s]+)\s*$",
                prefix,
            )
        )

    def _stateless_member_declaration_is_function(self, declaration: str) -> bool:
        if re.search(r"\bvirtual\b", declaration):
            return False
        if re.search(r"\(\s*[*&]", declaration):
            return False
        declarator, default = self._split_top_level_assignment(declaration)
        if default is not None and default.strip() not in {"default", "delete"}:
            return False
        return self._declaration_is_method_prototype(declarator)

    def _prove_stateless_global_constructor(
        self,
        code: str,
        global_: _MetalCompileTimeGlobal,
        struct: _MetalStructDefinition,
        global_names: Set[str],
    ) -> None:
        for argument in global_.arguments:
            referenced_globals = (
                set(IDENTIFIER_RE.findall(self._mask_comments_and_literals(argument)))
                & global_names
            )
            if referenced_globals:
                self._raise_stateless_global_error(
                    code,
                    global_=global_,
                    reason="initializer-observes-global-identity",
                    start=global_.name_span[0],
                    end=global_.span[1],
                    blocking_use=", ".join(sorted(referenced_globals)),
                    effect="object-identity",
                )
            if not self._stateless_initializer_argument_is_proven_constant(argument):
                self._raise_stateless_global_error(
                    code,
                    global_=global_,
                    reason="initializer-effect-unproven",
                    start=global_.name_span[0],
                    end=global_.span[1],
                    blocking_use=argument,
                    effect="initializer-evaluation",
                )

        if struct.name in struct.member_prototype_names:
            self._raise_stateless_global_error(
                code,
                global_=global_,
                reason="constructor-body-unresolved",
                start=global_.name_span[0],
                end=global_.span[1],
                blocking_use=struct.name,
            )

        matching = [
            constructor
            for constructor in struct.constructors
            if self._callable_accepts_argument_count(
                constructor.params_text, len(global_.arguments)
            )
            and global_.receiver_address_space in constructor.receiver_address_spaces
        ]
        if not matching:
            has_declared_constructor = bool(struct.constructors) or (
                struct.name in struct.member_prototype_names
            )
            if not global_.arguments and not has_declared_constructor:
                return
            self._raise_stateless_global_error(
                code,
                global_=global_,
                reason="constructor-unresolved",
                start=global_.name_span[0],
                end=global_.span[1],
                blocking_use=f"{struct.qualified_name}({', '.join(global_.arguments)})",
            )
        for constructor in matching:
            if not constructor.is_constexpr:
                self._raise_stateless_global_error(
                    code,
                    global_=global_,
                    reason="constructor-not-constexpr",
                    start=(
                        constructor.span[0]
                        if constructor.span
                        else global_.name_span[0]
                    ),
                    end=(
                        constructor.span[1]
                        if constructor.span
                        else global_.name_span[1]
                    ),
                    blocking_use=constructor.prefix.rstrip("("),
                )
            if constructor.init_text.strip():
                self._raise_stateless_global_error(
                    code,
                    global_=global_,
                    reason="constructor-initializer-not-empty",
                    start=(
                        constructor.span[0]
                        if constructor.span
                        else global_.name_span[0]
                    ),
                    end=(
                        constructor.span[1]
                        if constructor.span
                        else global_.name_span[1]
                    ),
                    blocking_use=constructor.init_text,
                    effect="object-state",
                )
            effect = self._stateless_body_effect(constructor.body, allow_return=False)
            if effect is not None:
                self._raise_stateless_global_error(
                    code,
                    global_=global_,
                    reason="constructor-has-effects",
                    start=(
                        constructor.span[0]
                        if constructor.span
                        else global_.name_span[0]
                    ),
                    end=(
                        constructor.span[1]
                        if constructor.span
                        else global_.name_span[1]
                    ),
                    blocking_use=constructor.body.strip(),
                    effect=effect,
                )

    def _prove_stateless_global_destructor(
        self,
        code: str,
        global_: _MetalCompileTimeGlobal,
        struct: _MetalStructDefinition,
    ) -> None:
        body_start, body_end = struct.body_span
        body = code[body_start:body_end]
        masked = self._mask_comments_and_literals(body)
        scopes = self._find_lexical_brace_scopes(body)
        pattern = re.compile(rf"~\s*{re.escape(struct.name)}\s*\(")
        for match in pattern.finditer(masked):
            if self._innermost_lexical_scope(scopes, match.start(), len(body)) != (
                0,
                len(body),
            ):
                continue
            open_paren = masked.find("(", match.start(), match.end())
            close_paren = self._find_matching_delimiter(body, open_paren, "(", ")")
            if close_paren is None:
                self._raise_stateless_global_error(
                    code,
                    global_=global_,
                    reason="destructor-unresolved",
                    start=body_start + match.start(),
                    end=body_start + match.end(),
                )
            cursor = self._skip_cpp_trivia(body, close_paren + 1)
            while True:
                qualifier = IDENTIFIER_RE.match(body, cursor)
                if qualifier is None or qualifier.group(0) not in {
                    "const",
                    "constant",
                    "device",
                    "thread",
                    "threadgroup",
                    "noexcept",
                }:
                    break
                cursor = self._skip_cpp_trivia(body, qualifier.end())
            if cursor < len(body) and body[cursor] == "{":
                close_brace = self._find_matching_delimiter(body, cursor, "{", "}")
                if close_brace is None:
                    effect = "unresolved-body"
                    close_brace = cursor
                else:
                    effect = self._stateless_body_effect(
                        body[cursor + 1 : close_brace], allow_return=False
                    )
                if effect is None:
                    continue
                end = close_brace + 1
            else:
                declaration_end = body.find(";", cursor)
                if declaration_end == -1:
                    declaration_end = cursor
                suffix = body[cursor:declaration_end].strip()
                if re.fullmatch(r"=\s*default", suffix):
                    continue
                effect = "deleted-body" if "delete" in suffix else "unresolved-body"
                end = declaration_end + 1
            self._raise_stateless_global_error(
                code,
                global_=global_,
                reason="destructor-has-effects",
                start=body_start + match.start(),
                end=body_start + end,
                blocking_use=body[match.start() : end].strip(),
                effect=effect,
            )

    def _stateless_global_call_replacements(
        self,
        code: str,
        global_: _MetalCompileTimeGlobal,
        struct: _MetalStructDefinition,
        namespace_spans: List[Tuple[int, int, str]],
        declaration_spans: List[Tuple[int, int]],
        global_names: Set[str],
    ) -> List[Tuple[int, int, str]]:
        masked = self._mask_comments_and_literals(code)
        replacements: List[Tuple[int, int, str]] = []
        position = min(
            (end for _start, end in global_.prior_declaration_spans),
            default=global_.span[1],
        )
        pattern = re.compile(rf"\b{re.escape(global_.name)}\b")
        while True:
            match = pattern.search(masked, position)
            if match is None:
                break
            position = match.end()
            if self._containing_span(match.start(), declaration_spans) is not None:
                continue
            receiver_start = self._stateless_global_receiver_start(
                code, match.start(), global_, namespace_spans
            )
            if receiver_start is None:
                continue
            call = self._parse_stateless_receiver_call(code, match.end())
            if call is None:
                self._raise_stateless_global_error(
                    code,
                    global_=global_,
                    reason="identity-observed",
                    start=receiver_start,
                    end=match.end(),
                    blocking_use=code[receiver_start : match.end()],
                    effect="object-identity",
                )
            method_name, arguments, call_end, is_statement = call
            if not is_statement:
                self._raise_stateless_global_error(
                    code,
                    global_=global_,
                    reason="call-result-observed",
                    start=receiver_start,
                    end=call_end,
                    method_name=method_name,
                    blocking_use=code[receiver_start:call_end],
                    effect="returned-value",
                )
            if method_name in struct.member_prototype_names:
                self._raise_stateless_global_error(
                    code,
                    global_=global_,
                    reason="method-body-unresolved",
                    start=receiver_start,
                    end=call_end,
                    method_name=method_name,
                    blocking_use=code[receiver_start:call_end],
                )
            candidates = [
                method
                for method in [*struct.methods, *struct.template_methods]
                if method.name == method_name
                and not method.is_static
                and method.is_const
                and global_.receiver_address_space in method.receiver_address_spaces
                and self._callable_accepts_argument_count(
                    method.parameters, len(arguments)
                )
            ]
            if not candidates:
                self._raise_stateless_global_error(
                    code,
                    global_=global_,
                    reason="method-overload-unresolved",
                    start=receiver_start,
                    end=call_end,
                    method_name=method_name,
                    blocking_use=code[receiver_start:call_end],
                )
            for method in candidates:
                if self._normalize_inferred_type(method.return_type) != "void":
                    self._raise_stateless_global_error(
                        code,
                        global_=global_,
                        reason="method-returns-value",
                        start=receiver_start,
                        end=call_end,
                        method_name=method_name,
                        blocking_use=method.return_type,
                        effect="returned-value",
                    )
                effect = self._stateless_body_effect(method.body, allow_return=True)
                if effect is not None:
                    self._raise_stateless_global_error(
                        code,
                        global_=global_,
                        reason="method-has-effects",
                        start=receiver_start,
                        end=call_end,
                        method_name=method_name,
                        blocking_use=method.body.strip(),
                        effect=effect,
                    )
            for argument in arguments:
                referenced_globals = (
                    set(
                        IDENTIFIER_RE.findall(
                            self._mask_comments_and_literals(argument)
                        )
                    )
                    & global_names
                )
                if referenced_globals:
                    self._raise_stateless_global_error(
                        code,
                        global_=global_,
                        reason="identity-observed",
                        start=receiver_start,
                        end=call_end,
                        method_name=method_name,
                        blocking_use=", ".join(sorted(referenced_globals)),
                        effect="object-identity",
                    )
            evaluated = [
                argument
                for argument in arguments
                if self._stateless_argument_requires_evaluation(argument)
            ]
            replacement = self._render_elided_call_arguments(evaluated)
            replacements.append((receiver_start, call_end, replacement))
            position = call_end
        return replacements

    def _stateless_global_receiver_start(
        self,
        code: str,
        name_start: int,
        global_: _MetalCompileTimeGlobal,
        namespace_spans: List[Tuple[int, int, str]],
    ) -> Optional[int]:
        line_start = code.rfind("\n", 0, name_start) + 1
        prefix = code[line_start:name_start]
        qualified = re.search(
            r"(?P<qualification>(?:::)?(?:[A-Za-z_]\w*\s*::\s*)+)$", prefix
        )
        if qualified is not None:
            qualification = re.sub(
                r"\s*::\s*", "::", qualified.group("qualification")
            ).strip(":")
            if qualification != global_.namespace:
                return None
            return line_start + qualified.start("qualification")
        previous = name_start - 1
        while previous >= 0 and code[previous].isspace():
            previous -= 1
        if previous >= 0 and code[previous] == ".":
            return None
        if previous >= 1 and code[previous - 1 : previous + 1] == "->":
            return None
        namespace = self._namespace_at(namespace_spans, name_start)
        if global_.namespace and not (
            namespace == global_.namespace
            or namespace.startswith(global_.namespace + "::")
        ):
            return None
        return name_start

    def _parse_stateless_receiver_call(
        self, code: str, name_end: int
    ) -> Optional[Tuple[str, List[str], int, bool]]:
        cursor = self._skip_cpp_trivia(code, name_end)
        if cursor >= len(code) or code[cursor] != ".":
            return None
        cursor = self._skip_cpp_trivia(code, cursor + 1)
        method = IDENTIFIER_RE.match(code, cursor)
        if method is None:
            return None
        method_name = method.group(0)
        cursor = self._skip_cpp_trivia(code, method.end())
        if cursor < len(code) and code[cursor] == "<":
            angle_end = self._find_matching_angle(code, cursor)
            if angle_end is None:
                return None
            cursor = self._skip_cpp_trivia(code, angle_end + 1)
        if cursor >= len(code) or code[cursor] != "(":
            return None
        close_paren = self._find_matching_delimiter(code, cursor, "(", ")")
        if close_paren is None:
            return None
        arguments = self._split_initializer_arguments(code[cursor + 1 : close_paren])
        call_end = self._skip_cpp_trivia(code, close_paren + 1)
        is_statement = call_end < len(code) and code[call_end] == ";"
        return (
            method_name,
            arguments,
            call_end + 1 if is_statement else close_paren + 1,
            is_statement,
        )

    def _callable_accepts_argument_count(self, parameters: str, count: int) -> bool:
        parts = [
            part.strip()
            for part in self._split_top_level_commas(parameters)
            if part.strip() and part.strip() != "void"
        ]
        variadic = any("..." in part for part in parts)
        fixed = [part for part in parts if "..." not in part]
        required = sum(
            1
            for parameter in fixed
            if self._split_top_level_assignment(parameter)[1] is None
        )
        return required <= count and (variadic or count <= len(fixed))

    def _stateless_initializer_argument_is_proven_constant(self, argument: str) -> bool:
        masked = self._mask_comments_and_literals(argument)
        if re.search(r"\+\+|--|\b(?:new|delete|throw|asm)\b", masked):
            return False
        if re.search(
            r"(?:<<|>>|[+\-*/%&|^])=|(?<![=!<>+\-*/%&|^])=(?!=)",
            masked,
        ):
            return False
        if re.search(r"\b[A-Za-z_]\w*\s*(?:<[^;{}()]*>)?\s*\(", masked):
            return False
        return not bool(re.search(r"\[|\]|->|\.", masked))

    def _stateless_body_effect(self, body: str, *, allow_return: bool) -> Optional[str]:
        masked = self._mask_comments_and_literals(body).strip()
        if not masked:
            return None
        if allow_return and re.fullmatch(r"(?:return\s*;\s*)+", masked):
            return None
        if re.search(r"\b(?:atomic\w*|barrier|threadgroup_barrier)\b", masked):
            return "atomic-or-synchronization"
        if re.search(r"\b(?:trap|abort|assert|discard_fragment)\b", masked):
            return "trap-or-termination"
        if re.search(
            r"\+\+|--|(?:<<|>>|[+\-*/%&|^])=|" r"(?<![=!<>+\-*/%&|^])=(?!=)",
            masked,
        ):
            return "write"
        if re.search(r"\breturn\s+[^;]", masked):
            return "returned-value"
        if re.search(r"\b[A-Za-z_]\w*\s*(?:<[^;{}()]*>)?\s*\(", masked):
            return "function-call"
        return "unproven-statement"

    def _stateless_argument_requires_evaluation(self, argument: str) -> bool:
        text = argument.strip()
        while text.startswith("(") and text.endswith(")"):
            close = self._find_matching_delimiter(text, 0, "(", ")")
            if close != len(text) - 1:
                break
            text = text[1:-1].strip()
        literal = METAL_STRING_EXPRESSION_PATTERN
        scalar = (
            r"(?:true|false|nullptr|[-+]?(?:0[xX][0-9A-Fa-f]+|"
            r"(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
            r"[uUlLfFhH]*)"
        )
        return re.fullmatch(rf"(?:{literal}|{scalar})", text) is None

    @staticmethod
    def _render_elided_call_arguments(arguments: List[str]) -> str:
        if not arguments:
            return ";"
        if len(arguments) == 1:
            return f"{arguments[0].strip()};"
        statements = "\n".join(f"  {argument.strip()};" for argument in arguments)
        return f"{{\n{statements}\n}}"

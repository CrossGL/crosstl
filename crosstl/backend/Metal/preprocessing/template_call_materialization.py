"""Source-analysis caches, compiler-diagnostic stripping, library matrix typedefs and the materialization of project, explicit and constrained template function calls."""

import re
from typing import (
    Dict,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
)

from ..function_specialization import (
    deducible_template_parameters,
    unique_specialization_identifier,
)
from ..namespaced_specializations import (
    place_in_namespace,
    specialization_reference,
)
from ..template_overloads import (
    declared_type_names,
    select_instantiated_overload,
)
from ..unreachable_constrained_callers import (
    materialize_with_unreachable_callers,
    unreachable_caller_scope,
)
from .support import (
    COMPILER_DIAGNOSTIC_START_RE,
    IDENTIFIER_RE,
    METAL_LIBRARY_MATRIX_TYPEDEF_RE,
    METAL_LIBRARY_MATRIX_TYPEDEFS,
    MSL_SOURCE_START_RE,
    MetalTemplateSpecializationError,
    _MetalConstexprCall,
    _MetalFunctionDefinition,
    _MetalNamespaceVisibility,
    _MetalSourceAnalysis,
    _MetalStructDefinition,
    _MetalStructMethod,
    _MetalTemplateFunction,
    _MetalTemplateStruct,
)


class MetalTemplateCallMaterializationMixin:
    """Members of ``MetalPreprocessor``: source-analysis caches, compiler-diagnostic stripping, library matrix typedefs and the materialization of project, explicit and constrained template function calls."""

    def _source_analysis(self, code: str) -> _MetalSourceAnalysis:
        return self._source_analysis_cache.get(code, _MetalSourceAnalysis)

    def _strip_leading_compiler_diagnostics(self, code: str) -> str:
        lines = code.splitlines(keepends=True)
        saw_diagnostic = False

        for index, line in enumerate(lines):
            if MSL_SOURCE_START_RE.match(line):
                if saw_diagnostic:
                    return "".join(lines[index:])
                return code

            stripped = line.strip()
            if not stripped:
                continue
            if COMPILER_DIAGNOSTIC_START_RE.match(line):
                saw_diagnostic = True
                continue
            if saw_diagnostic and (
                line.startswith((" ", "\t")) or stripped.startswith("^")
            ):
                continue
            return code

        return code

    def _canonicalize_metal_library_matrix_typedefs(self, code: str) -> str:
        """Spell the library 8x8 SIMD-group matrix typedefs as templates.

        ``simdgroup_float8x8`` is ``simdgroup_matrix<float, 8, 8>``. A name the
        source declares itself (struct, enum, typedef or alias) keeps the
        source's meaning, as do names inside comments and literals and names
        qualified by anything other than ``metal::``.
        """
        if "8x8" not in code:
            return code
        declared = set()
        for name in METAL_LIBRARY_MATRIX_TYPEDEFS:
            escaped = re.escape(name)
            if re.search(
                rf"\b(?:struct|class|union|enum)\s+{escaped}\b"
                rf"|\busing\s+{escaped}\s*="
                rf"|\btypedef\b[^;{{}}]*\b{escaped}\s*;",
                code,
            ):
                declared.add(name)
        ignored_spans = self._find_comment_and_literal_spans(code)
        replacements = []
        for match in METAL_LIBRARY_MATRIX_TYPEDEF_RE.finditer(code):
            name = match.group("name")
            if name in declared:
                continue
            if self._containing_span(match.start(), ignored_spans) is not None:
                continue
            replacements.append(
                (
                    match.start(),
                    match.end(),
                    (match.group("qualifier") or "")
                    + METAL_LIBRARY_MATRIX_TYPEDEFS[name],
                )
            )
        if not replacements:
            return code
        return self._apply_text_replacements(code, replacements)

    def _materialize_project_template_instantiations(
        self,
        code: str,
        *,
        enforce_specialization_limit: bool = True,
        host_names: Optional[Set[str]] = None,
    ) -> str:
        if not enforce_specialization_limit:
            # The project pipeline disables this limit only for template-hostile
            # targets, where residual source assertions cannot be emitted.
            self._fail_closed_static_assertions = True
        all_instantiations = self._find_project_template_instantiations(code)
        if not all_instantiations:
            return code
        instantiations = (
            all_instantiations
            if host_names is None
            else [
                instantiation
                for instantiation in all_instantiations
                if instantiation.host_name in host_names
            ]
        )
        # Explicit project instantiations (`instantiate_kernel(...)` and the
        # equivalent `template [[host_name(...)]] ... decltype(func<args>)
        # func<args>;` declarations) are the source's own enumerated request for
        # concrete kernels; their number is bounded by the source text rather than
        # by combinatorial discovery. `enforce_specialization_limit` keeps the
        # historical safety bail for template-FRIENDLY consumers (e.g. the direct
        # MetalLexer/`preprocess` path targeting CGL, which can keep the residual
        # templates when a source declares an unusually large instantiation
        # family). Template-HOSTILE targets cannot emit residual templates, so the
        # project pipeline disables the bail (`enforce_specialization_limit=False`)
        # and materializes every explicit instantiation selected for the artifact;
        # otherwise bailing here leaves the kernels and every helper template
        # reachable only from them unmaterialized and mis-reported as "missing
        # template arguments".
        if (
            enforce_specialization_limit
            and len(instantiations) > self.max_template_specializations
        ):
            return code

        templates = self._find_template_functions(code)
        if not templates or not instantiations:
            return self._apply_text_replacements(
                code,
                [
                    (instantiation.span[0], instantiation.span[1], "")
                    for instantiation in all_instantiations
                ],
            )

        templates_by_name: Dict[str, List[_MetalTemplateFunction]] = {}
        for template in templates:
            templates_by_name.setdefault(template.name, []).append(template)
        overload_type_names = (
            declared_type_names(code)
            if any(len(group) > 1 for group in templates_by_name.values())
            else frozenset()
        )
        replacements: List[Tuple[int, int, str]] = [
            (instantiation.span[0], instantiation.span[1], "")
            for instantiation in all_instantiations
        ]
        seen: Set[Tuple[str, Tuple[str, ...], str]] = set()

        for instantiation in instantiations:
            template = select_instantiated_overload(
                self,
                overload_type_names,
                templates_by_name.get(instantiation.function_name, ()),
                instantiation.template_arguments,
            )
            if template is None:
                continue
            template_arguments = self._template_arguments_with_defaults(
                code,
                template,
                instantiation.template_arguments,
            )
            key = (
                instantiation.function_name,
                tuple(
                    self._normalize_template_argument_text(argument)
                    for argument in template_arguments
                ),
                instantiation.host_name,
            )
            if key in seen:
                continue
            seen.add(key)
            materialized = self._materialize_template_function(
                code, template, instantiation
            )
            if materialized:
                template.materializations.append(materialized)

        for template in templates:
            if template.materializations:
                replacement = "\n\n".join(template.materializations)
                replacements.append((template.span[0], template.span[1], replacement))

        return self._apply_text_replacements(code, replacements)

    def _materialize_explicit_template_function_calls(
        self,
        code: str,
        *,
        work_budget: Optional[object] = None,
        materialized_names: Optional[Dict[Tuple[str, Tuple[str, ...]], str]] = None,
        excluded_template_names: Optional[Set[str]] = None,
        include_struct_members: bool = True,
    ) -> str:
        materialized_names = (
            materialized_names
            if materialized_names is not None
            else self._explicit_template_function_names
        )
        excluded_template_names = set(excluded_template_names or ())
        working = code

        while True:
            templates = self._find_template_functions(working)
            if not include_struct_members:
                struct_spans = [
                    struct.span
                    for struct in self._find_concrete_struct_definitions(working)
                ]
                templates = [
                    template
                    for template in templates
                    if self._containing_span(template.span[0], struct_spans) is None
                ]
            if not templates:
                return working

            # This frontend-only pass has no call-argument type environment and
            # therefore cannot select safely from an overloaded template set.
            # Repo translation excludes those names here and resolves them later
            # with the signature-aware project materializer instead of silently
            # choosing the last declaration.
            template_name_counts: Dict[str, int] = {}
            for template in templates:
                template_name_counts[template.name] = (
                    template_name_counts.get(template.name, 0) + 1
                )
            templates_by_name = {
                template.name: template
                for template in templates
                if template.name not in excluded_template_names
                and template_name_counts[template.name] == 1
            }
            template_spans = self._find_template_declaration_spans(working)
            reachable_function_spans = self._reachable_function_spans(
                working, template_spans
            )
            explicit_specialization_keys = (
                self._find_explicit_template_specialization_keys(working)
            )
            calls = self._find_explicit_template_function_calls(
                working,
                templates_by_name,
                template_spans,
                explicit_specialization_keys,
                reachable_function_spans,
            )
            if not calls:
                return working

            placements = self._template_specialization_placements(
                working, templates_by_name.values()
            )
            replacements: List[Tuple[int, int, str]] = []
            new_materializations: List[str] = []
            source_identifiers = None
            for (
                function_name,
                template_arguments,
                spans,
            ) in self._dedupe_explicit_template_function_calls(calls):
                key = self._template_specialization_key(
                    function_name, template_arguments
                )
                template = templates_by_name[function_name]
                if not self._template_arguments_satisfy_parameters(
                    template,
                    template_arguments,
                ):
                    continue
                if not template.variadic_template_parameters and (
                    set(template.template_parameters[len(template_arguments) :])
                    & deducible_template_parameters(self, template)
                ):
                    # A defaulted parameter the call can deduce takes the
                    # deduced type, not its default; argument deduction decides.
                    continue
                placement = placements.get(template.span, "")
                specialized_name = materialized_names.get(key)
                if specialized_name is not None:
                    reference = specialization_reference(placement, specialized_name)
                    replacements.extend((span[0], span[1], reference) for span in spans)
                    continue
                unique_count = len(materialized_names) + 1
                if len(materialized_names) >= self.max_template_specializations:
                    requested_signature = self._template_specialization_signature(
                        function_name, template_arguments
                    )
                    suggested_action = (
                        "raise max_template_specializations for this source pattern "
                        "or backend, or reduce explicit template helper "
                        "instantiations"
                    )
                    raise MetalTemplateSpecializationError(
                        "Metal template specialization limit exceeded while "
                        f"materializing '{requested_signature}'; "
                        f"{unique_count} unique concrete signatures requested, "
                        f"limit {self.max_template_specializations} from "
                        f"{self.template_specialization_limit_source}. "
                        f"Suggested action: {suggested_action}.",
                        limit=self.max_template_specializations,
                        limit_source=self.template_specialization_limit_source,
                        unique_specialization_count=unique_count,
                        requested_signature=requested_signature,
                        suggested_action=suggested_action,
                    )
                if work_budget is not None:
                    consume_unique = getattr(work_budget, "consume_unique", None)
                    work_key = ("function", key[0], *key[1])
                    context = (
                        "reachable concrete helper "
                        f"'{self._template_specialization_signature(function_name, template_arguments)}'"
                    )
                    if callable(consume_unique):
                        consume_unique(
                            work_key,
                            offset=spans[0][0],
                            length=max(spans[0][1] - spans[0][0], 0),
                            context=context,
                        )
                    else:
                        work_budget.consume(
                            1,
                            offset=spans[0][0],
                            length=max(spans[0][1] - spans[0][0], 0),
                            context=context,
                        )
                if source_identifiers is None:
                    source_identifiers = set(
                        IDENTIFIER_RE.findall(self._mask_comments_and_literals(working))
                    )
                specialized_name = unique_specialization_identifier(
                    self._template_specialization_identifier(
                        function_name, list(key[1])
                    ),
                    source_identifiers,
                    materialized_names.values(),
                )
                materialized = self._materialize_template_function_with_name(
                    template,
                    template_arguments,
                    specialized_name,
                    host_name=None,
                )
                if materialized:
                    reference = specialization_reference(placement, specialized_name)
                    replacements.extend((span[0], span[1], reference) for span in spans)
                    materialized_names[key] = specialized_name
                    new_materializations.append(
                        place_in_namespace(placement, materialized)[0]
                    )

            if not replacements and not new_materializations:
                return working

            working = self._apply_text_replacements(working, replacements)
            if new_materializations:
                working = working.rstrip() + "\n\n" + "\n\n".join(new_materializations)
                if not working.endswith("\n"):
                    working += "\n"

    def _materialize_inferred_constrained_template_function_calls(
        self, code: str, *, simd_group_overloads: bool = False
    ) -> str:
        return materialize_with_unreachable_callers(
            self,
            code,
            lambda source: self._materialize_constrained_template_calls_in_source(
                source, simd_group_overloads=simd_group_overloads
            ),
        )

    def _materialize_constrained_template_calls_in_source(
        self, code: str, *, simd_group_overloads: bool = False
    ) -> str:
        # Arguments that call source functions are typed from this source, which includes the
        # instantiations an earlier round appended.
        with self._call_result_scope.active(code):
            return self._materialize_constrained_template_calls_in_scope(
                code, simd_group_overloads=simd_group_overloads
            )

    def _constrained_call_associated_namespaces(
        self,
        argument_types: Sequence[str],
        structs: Sequence[_MetalStructDefinition],
        template_structs: Sequence[_MetalTemplateStruct],
        *,
        position: int,
        namespace_visibility: _MetalNamespaceVisibility,
    ) -> Optional[Set[str]]:
        """Prove namespaces for concrete class arguments of a dependent call.

        Relocated specializations retain their source namespace and type arguments.
        Primitive arguments contribute nothing; unresolved owners and inheritance
        cannot safely yield a partial overload set.
        """
        namespaces: Set[str] = set()
        pending = list(argument_types)
        visited = set()
        while pending:
            raw_type = pending.pop()
            type_name = self._normalize_inferred_type(raw_type).rstrip("*& ")
            if type_name in visited:
                continue
            visited.add(type_name)
            if len(visited) > 256:
                return None
            if type_name in self._METAL_SCALAR_VECTOR_TYPES or type_name == "void":
                continue
            specialization = self._materialized_struct_specializations.get(type_name)
            primary = None
            if specialization is not None:
                primary = self._materialized_struct_primary_templates.get(type_name)
                arguments = list(specialization[1])
            else:
                template_id = self._exact_template_type_id(type_name)
                if template_id is not None:
                    name, arguments = template_id
                    candidates = [
                        template
                        for template in template_structs
                        if self._template_struct_specialization_arguments(template)
                        is None
                        and (
                            (
                                f"{template.namespace}::{template.name}"
                                if template.namespace
                                else template.name
                            )
                            == name.lstrip(":")
                            or (
                                name == template.name
                                and self._metal_namespace_declaration_visible(
                                    template.namespace,
                                    None,
                                    position,
                                    namespace_visibility,
                                )
                            )
                        )
                    ]
                    if len(candidates) != 1:
                        return None
                    primary = candidates[0]
                    arguments = self._template_arguments_with_resolved_defaults(
                        primary, arguments
                    )
                    if arguments is None:
                        return None
            if primary is not None:
                if primary.variadic_template_parameters:
                    return None
                body_start = self._find_next_top_level_char(primary.source, 0, "{")
                if body_start is None or re.search(
                    r"(?<!:):(?!:)", primary.source[:body_start]
                ):
                    return None
                namespaces.add(primary.namespace)
                for parameter, argument in zip(primary.template_parameters, arguments):
                    if not primary.template_parameter_types.get(parameter):
                        pending.append(argument)
                continue
            candidates = [
                struct
                for struct in structs
                if (struct.qualified_name or struct.name).lstrip(":")
                == type_name.lstrip(":")
                or (
                    struct.name == type_name
                    and self._metal_namespace_declaration_visible(
                        self._struct_namespace(struct),
                        None,
                        position,
                        namespace_visibility,
                    )
                )
            ]
            if len(candidates) != 1 or candidates[0].base_clause:
                return None
            namespaces.add(self._struct_namespace(candidates[0]))
        return namespaces

    def _prune_unreferenced_call_operator_definitions(self, code: str) -> str:
        """Leave library definitions intact; discard only unused kernel helpers."""
        masked = self._mask_comments_and_literals(code)
        if re.search(r"::\s*operator\s*\(\s*\)", masked) is None:
            return code
        structs = self._find_concrete_struct_definitions(code)
        excluded = sorted(
            self._find_template_declaration_spans(code)
            + [struct.span for struct in structs]
        )
        functions = self._find_non_template_function_definitions(code, excluded)
        if not any(function.is_entry for function in functions):
            return code
        owners: Dict[str, List[_MetalStructDefinition]] = {}
        for struct in structs:
            owners.setdefault(struct.name, []).append(struct)
        definitions: Dict[str, List[_MetalFunctionDefinition]] = {}
        for function in functions:
            header = code[function.span[0] : function.body_span[0] - 1]
            if re.search(r"\boperator\s*\(\s*\)", header) is None:
                continue
            if re.search(r"\b(?:extern|visible)\b", header):
                continue
            owner = self._qualified_function_owner(header)
            # A unique unqualified owner avoids guessing namespace or template
            # bindings. Any remaining mention, even in a template, keeps it live.
            if owner not in owners or len(owners[owner]) != 1:
                continue
            struct = owners[owner][0]
            if struct.qualified_name != owner or not re.fullmatch(
                r"\s*;", code[struct.body_span[1] + 1 : struct.span[1]]
            ):
                continue
            definitions.setdefault(owner, []).append(function)
        replacements = []
        for owner, candidates in definitions.items():
            own_spans = sorted(
                [owners[owner][0].span] + [function.span for function in candidates]
            )
            if any(
                self._containing_span(match.start(), own_spans) is None
                for match in re.finditer(rf"\b{re.escape(owner)}\b", masked)
            ):
                continue
            for function in candidates:
                start, end = function.span
                # Keep original offsets and lines for subsequent diagnostics.
                blank = re.sub(r"[^\r\n]", " ", code[start:end])
                replacements.append((start, end, blank))
        return self._apply_text_replacements(code, replacements)

    def _prune_unreferenced_materialized_function_templates(self, code: str) -> str:
        """Drop an instantiated primary only after all raw references are gone.

        A concrete body can still expose calls needed by a later pass. Pruning
        happens after materialization, and references in other template bodies,
        function-pointer expressions and declarations conservatively retain it.

        Pruning repeats until nothing changes: a primary named only by another
        pruned primary (a helper called from a default template argument such as
        ``uint W = work_per_thread<T>()``) loses its last reference with it.
        """
        if not self._materialized_function_templates:
            return code
        while True:
            masked = self._mask_comments_and_literals(code)
            replacements = []
            for template in self._find_template_functions(code):
                if (
                    template.namespace,
                    template.source,
                ) not in self._materialized_function_templates:
                    continue
                start, end = template.span
                if any(
                    not start <= match.start() < end
                    for match in re.finditer(rf"\b{re.escape(template.name)}\b", masked)
                ):
                    continue
                replacements.append((start, end, ""))
            if not replacements:
                return code
            code = self._apply_text_replacements(code, replacements)

    def _find_free_function_overload_declarations(
        self,
        code: str,
        references: List[_MetalConstexprCall],
        *,
        excluded_spans: List[Tuple[int, int]],
        template_spans: List[Tuple[int, int]],
        namespace_visibility: _MetalNamespaceVisibility,
    ) -> Dict[str, List[_MetalConstexprCall]]:
        """Index signatures, including prototypes, outside the selected templates.

        The balanced call scanner also recognizes declaration parameter lists.
        Distinguish those from initializers and nested calls before using them to
        guard overload resolution; a missing definition does not remove an
        otherwise visible overload from the source language's candidate set.
        """
        declarations: Dict[str, List[_MetalConstexprCall]] = {}
        for reference in references:
            if self._containing_span(reference.span[0], excluded_spans) is not None:
                continue
            template_span = self._containing_span(reference.span[0], template_spans)
            if template_span is not None:
                angle_start = code.find("<", template_span[0])
                angle_end = self._find_matching_template_param_angle(code, angle_start)
                if angle_end is None or reference.span[0] <= angle_end:
                    continue
                declaration_start = angle_end + 1
            else:
                declaration_start = self._function_declaration_start(
                    code, reference.span[0]
                )
                for start, end, _namespace in namespace_visibility.namespace_spans:
                    if start <= reference.span[0] < end:
                        declaration_start = max(declaration_start, start)
            header = code[declaration_start : reference.span[1]]
            if self._function_parameter_start(header) != (
                reference.argument_open - declaration_start
            ):
                continue
            prefix = self._mask_comments_and_literals(
                code[declaration_start : reference.span[0]]
            ).strip()
            if not prefix or self._split_top_level_assignment(prefix)[1] is not None:
                continue
            declarations.setdefault(reference.name, []).append(reference)
        return declarations

    def _template_function_parameter_text(
        self, template: _MetalTemplateFunction
    ) -> Optional[str]:
        body_start = self._find_next_top_level_char(template.source, 0, "{")
        if body_start is None:
            return None
        header = template.source[:body_start]
        parameter_start = self._function_parameter_start(header)
        if parameter_start is None:
            return None
        parameter_end = self._find_matching_delimiter(header, parameter_start, "(", ")")
        if parameter_end is None:
            return None
        return header[parameter_start + 1 : parameter_end]

    def _template_function_method_signature(
        self, template: _MetalTemplateFunction, parameters: str
    ) -> _MetalStructMethod:
        """Share signature deduction between free calls and explicit bodies."""
        return _MetalStructMethod(
            name=template.name,
            free_name="",
            is_static=True,
            is_operator_call=False,
            return_type="",
            parameters=parameters,
            parameter_names=[],
            body="",
            span=template.span,
            template_parameters=list(template.template_parameters),
            template_parameter_types=dict(template.template_parameter_types),
            variadic_template_parameters=set(template.variadic_template_parameters),
            template_parameter_defaults=dict(template.template_parameter_defaults),
            template_constraints=list(template.template_constraints),
        )

    def _raise_constrained_free_function_error(
        self,
        code: str,
        call: _MetalConstexprCall,
        reason: str,
    ) -> None:
        # A failure in an unreachable body is soft; the soft pass needs to know where it was.
        unreachable_caller_scope(self).failure_offset = call.span[0]
        requested = re.sub(r"\s+", " ", code[call.span[0] : call.span[1]]).strip()
        suggested_action = (
            "give the intended overload a distinct name, specialize the free "
            "function explicitly, or restrict the overload "
            "set to recognized enable_if_t constraints and inferable argument types"
        )
        raise MetalTemplateSpecializationError(
            "Metal constrained free-function call "
            f"'{requested}' could not be resolved because {reason}. "
            f"Suggested action: {suggested_action}.",
            requested_signature=requested,
            suggested_action=suggested_action,
            source_location=self._source_location_for_offsets(
                code, call.span[0], call.span[1]
            ),
            callee_template=call.name,
            requested_arguments=tuple(call.arguments),
        )

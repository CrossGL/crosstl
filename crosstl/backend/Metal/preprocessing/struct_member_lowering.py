"""Lowering of struct member functions and explicit member specializations, and concrete member type aliases."""

import re
from dataclasses import replace
from typing import (
    Dict,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
)

from ..alias_template_owners import expose_alias_template_owners
from ..declarator_lists import split_struct_declarator_lists
from ..integral_constant_locals import (
    fold_concrete_function_integral_constant_arithmetic,
)
from ..qualified_member_types import resolve_qualified_struct_member_types
from ..static_member_constants import resolve_retained_static_member_owners
from .support import (
    MetalStatelessGlobalElisionError,
    MetalStructMethodError,
    MetalTemplateSpecializationError,
    _MetalCompileTimeGlobal,
    _MetalExplicitMemberSpecialization,
    _MetalStructDefinition,
    _MetalStructMethod,
    _PointerPromotionPlan,
    _PromotedPointerBinding,
)


class MetalStructMemberLoweringMixin:
    """Members of ``MetalPreprocessor``: lowering of struct member functions and explicit member specializations, and concrete member type aliases."""

    def _raise_stateless_global_error(
        self,
        code: str,
        *,
        reason: str,
        start: int,
        end: int,
        global_: Optional[_MetalCompileTimeGlobal] = None,
        global_name: Optional[str] = None,
        type_name: Optional[str] = None,
        method_name: Optional[str] = None,
        blocking_use: Optional[str] = None,
        effect: Optional[str] = None,
    ) -> None:
        resolved_name = (
            global_.name if global_ is not None else global_name or "<unknown>"
        )
        resolved_type = (
            global_.type_name if global_ is not None else type_name or "<unknown>"
        )
        if blocking_use is not None:
            blocking_use = re.sub(r"\s+", " ", blocking_use).strip()
            if len(blocking_use) > 240:
                blocking_use = blocking_use[:237].rstrip() + "..."
        raise MetalStatelessGlobalElisionError(
            f"Cannot safely elide compile-time global '{resolved_name}' of type "
            f"'{resolved_type}': {reason.replace('-', ' ')}.",
            global_name=resolved_name,
            type_name=resolved_type,
            reason=reason,
            source_location=self._source_location_for_offsets(code, start, end),
            method_name=method_name,
            blocking_use=blocking_use,
            effect=effect,
            suggested_action=(
                "Keep the object state and effects target-representable, or make the "
                "selected constructor, destructor, and constant receiver methods "
                "empty with unobserved object identity."
            ),
        )

    def _lower_struct_member_functions(self, code: str) -> str:
        # CrossGL structs are data-only, so the Metal frontend has historically
        # dropped struct member functions while keeping the now-dangling
        # `obj.method(...)` call sites (invalid output). This pass lowers each
        # concrete (non-template) struct's member functions to FREE functions and
        # rewrites the corresponding call sites BEFORE lexing/parsing. A CALLED
        # template member method is instantiated from its call-site argument
        # types and lowered to a concrete free function too. It runs after the
        # struct-template materializer so concrete structs produced from
        # templates (e.g. `Sum_float`) are lowered too. Regression-safe: an
        # UNEXPECTED failure returns the unmodified source, and the pass no-ops
        # when there are no struct methods (method-free structs stay
        # byte-identical). A DELIBERATE clean-fail for an unresolvable template
        # member method call (MetalStructMethodError) is re-raised so the
        # pipeline reports the kernel as a translation FAILURE instead of leaving
        # a dangling call.
        # Lowered calls type their arguments. A value computed from a compile-time loop index exists only once the
        # loop is expanded, and `auto m = mm * Int<4>{};` must already read `Int<4>{}`.
        code = self._lower_concrete_const_for_loop_callbacks(code)
        code = fold_concrete_function_integral_constant_arithmetic(self, code)
        evaluated = (
            self._evaluate_static_assertions(code)
            if self._fail_closed_static_assertions
            else code
        )
        previous_constexpr_functions = self._active_static_constexpr_functions
        self._active_static_constexpr_functions = (
            self._materialization_constexpr_function_index(evaluated)
        )
        previous_sizeof_source = self._enter_sizeof_source(evaluated)
        try:
            try:
                return self._lower_struct_member_functions_impl(evaluated)
            except (MetalStructMethodError, MetalTemplateSpecializationError):
                raise
            except Exception:
                return evaluated
        finally:
            self._active_static_constexpr_functions = previous_constexpr_functions
            self._active_sizeof_source = previous_sizeof_source

    def _lower_struct_member_functions_impl(self, code: str) -> str:
        code = self._prune_unreferenced_call_operator_definitions(code)
        template_declaration_spans = self._find_template_declaration_spans(code)
        self._source_type_alias_bindings = self._collect_local_type_alias_bindings(
            code,
            [(0, len(code))],
            skip_spans=template_declaration_spans,
        )
        structs = self._find_concrete_struct_definitions(code)
        struct_names = {struct.name for struct in structs}
        alias_bindings = self._source_type_alias_bindings
        split_code = split_struct_declarator_lists(
            code,
            struct_names
            | {
                name
                for name, bindings in alias_bindings.items()
                if any(binding.target in struct_names for binding in bindings)
            },
            self._find_comment_and_literal_spans(code),
            lambda name, position: name in struct_names
            or self._resolve_struct_type_alias_at(alias_bindings, name, position)
            in struct_names,
        )
        if split_code != code:
            # Every receiver of a declarator list now has its own declaration.
            return self._lower_struct_member_functions_impl(split_code)
        resolved_code = resolve_qualified_struct_member_types(
            self,
            code,
            structs,
            self._source_type_alias_bindings,
            sorted(template_declaration_spans + [struct.span for struct in structs]),
        )
        if resolved_code != code:
            code = resolved_code
            template_declaration_spans = self._find_template_declaration_spans(code)
            self._source_type_alias_bindings = self._collect_local_type_alias_bindings(
                code,
                [(0, len(code))],
                skip_spans=template_declaration_spans,
            )
            structs = self._find_concrete_struct_definitions(code)
        self._apply_explicit_member_specializations(code, structs)
        function_excluded_spans = sorted(
            template_declaration_spans + [struct.span for struct in structs]
        )
        unresolved_call_skip_spans = sorted(
            template_declaration_spans
            + self._unreachable_function_spans(code, function_excluded_spans)
        )
        self._reject_unresolved_temporary_functor_calls(
            code,
            {struct.name: struct for struct in structs},
            unresolved_call_skip_spans,
        )
        if not structs:
            return code
        aliased = self._rewrite_function_reference_alias_bindings(
            code, structs, function_excluded_spans
        )
        if aliased != code:
            # Captured storage adds declarations and moves every later span.
            return self._lower_struct_member_functions_impl(aliased)
        all_struct_spans = [struct.span for struct in structs]
        local_owner_spans = [
            function.body_span
            for function in self._find_non_template_function_definitions(
                code,
                all_struct_spans,
            )
        ]
        local_owner_spans.extend(
            body_span
            for struct in structs
            for body_span in struct.member_function_body_spans
        )
        # Only structs that actually declare at least one member function (a
        # concrete method OR a template method) need rewriting; everything else
        # stays untouched so method-free structs and existing kernels are
        # byte-identical. Function-local methods stay in the aggregate so the
        # reverse frontend can reject unsupported hoisting with source context;
        # moving them to file scope would leave a dangling local owner type.
        structs_with_methods = [
            struct
            for struct in structs
            if (struct.methods or struct.template_methods)
            and self._containing_span(struct.span[0], local_owner_spans) is None
        ]
        if not structs_with_methods:
            return self._canonicalize_qualified_struct_alias_templates(
                expose_alias_template_owners(self, code, structs), structs
            )

        # Structs with device/threadgroup pointer members cannot be lowered by
        # passing `self` by value (a pointer cannot live in a Function-storage
        # struct), so they take the pointer-member promotion path instead. A
        # struct is only promoted when it is actually CONSTRUCTED in a form we can
        # rewrite (every pointer expression is sourced from a construction
        # argument); otherwise it falls back to the ordinary path unchanged.
        promotion_plans: Dict[str, _PointerPromotionPlan] = {}
        for struct in structs_with_methods:
            plan = self._pointer_promotion_plan(struct, code=code)
            if plan is not None:
                promotion_plans[struct.name] = plan
        promoted_pointer_args: Dict[str, List[_PromotedPointerBinding]] = {}
        promoted_construction_replacements: List[Tuple[int, int, str]] = []
        promoted_names: Set[str] = set()
        if promotion_plans:
            (
                promoted_pointer_args,
                promoted_construction_replacements,
                promoted_names,
            ) = self._scan_pointer_struct_constructions(
                code, promotion_plans, all_struct_spans
            )
        promoted_structs = [
            struct for struct in structs_with_methods if struct.name in promoted_names
        ]
        # The ordinary lowering runs on every struct that is NOT promoted.
        structs_with_methods = [
            struct
            for struct in structs_with_methods
            if struct.name not in promoted_names
        ]

        struct_names = {struct.name for struct in structs_with_methods}
        # ALL concrete struct/union definitions (with or without methods), keyed
        # by name, so a member-access call argument (`obj.field` / `obj.field[i]`)
        # can resolve `field`'s type even when `obj`'s struct carries no methods
        # (e.g. a plain data carrier or a union). Kept separate from the
        # method-driven maps used for call rewriting.
        field_structs_by_name: Dict[str, _MetalStructDefinition] = {
            struct.name: struct for struct in structs
        }
        # Concrete methods keyed by struct name for direct call-site rewriting.
        methods_by_struct: Dict[str, Dict[str, _MetalStructMethod]] = {}
        # Template methods keyed by struct name then method name to the LIST of
        # overloads sharing that name (e.g. the two SFINAE `simd_reduce`
        # overloads). A single-overload name is just a one-element list, so the
        # non-overloaded path is unchanged; an overload set is resolved by
        # evaluating each candidate's SFINAE constraints at the call site.
        template_methods_by_struct: Dict[str, Dict[str, List[_MetalStructMethod]]] = {}
        operator_call_structs: Set[str] = set()
        structs_by_name: Dict[str, _MetalStructDefinition] = {}
        for struct in structs_with_methods:
            structs_by_name[struct.name] = struct
            method_map: Dict[str, _MetalStructMethod] = {}
            for method in struct.methods:
                method_map[method.name] = method
                if method.is_operator_call:
                    operator_call_structs.add(struct.name)
            methods_by_struct[struct.name] = method_map
            template_map: Dict[str, List[_MetalStructMethod]] = {}
            for method in struct.template_methods:
                template_map.setdefault(method.name, []).append(method)
                if method.is_operator_call:
                    operator_call_structs.add(struct.name)
            template_methods_by_struct[struct.name] = template_map

        for struct in structs_with_methods:
            for method in struct.methods:
                return_type = self._canonicalize_struct_scoped_type(
                    method.return_type, struct, field_structs_by_name
                )
                self._record_known_member_function_return_type(
                    method.free_name, return_type
                )

        instantiated_template_functions: Dict[str, str] = {}
        replacements: List[Tuple[int, int, str]] = []
        free_functions: List[str] = []
        for struct in structs_with_methods:
            data_only, lowered = self._render_lowered_struct(
                code,
                struct,
                instantiated_template_functions=instantiated_template_functions,
                template_methods_by_struct=template_methods_by_struct,
                methods_by_struct=methods_by_struct,
                operator_call_structs=operator_call_structs,
                structs_by_name=field_structs_by_name,
            )
            replacements.append((struct.span[0], struct.span[1], data_only))
            free_functions.extend(lowered)

        # Pointer-member-promoted structs: render the residual scalar struct,
        # emit each method as a free function taking `self` + the promoted pointer
        # parameters, and rewrite construction call sites (drop pointer arguments)
        # / method call sites (forward the pointer expressions).
        for struct in promoted_structs:
            plan = promotion_plans[struct.name]
            data_only, lowered = self._render_promoted_struct(code, struct, plan)
            replacements.append((struct.span[0], struct.span[1], data_only))
            free_functions.extend(lowered)
        replacements.extend(promoted_construction_replacements)
        if promoted_structs:
            replacements.extend(
                self._plan_promoted_function_forwarding(
                    code,
                    promoted_structs,
                    promotion_plans,
                    promoted_pointer_args,
                    all_struct_spans,
                )
            )
            replacements.extend(
                self._rewrite_promoted_call_sites(
                    code,
                    promoted_structs,
                    promotion_plans,
                    promoted_pointer_args,
                    all_struct_spans,
                )
            )

        # Rewrite call sites across the rest of the source (outside the structs
        # we are replacing, so receiver-less internal references are handled when
        # the method body is emitted, not here). Template-method call sites are
        # instantiated here; each unique (struct, method, bindings) instance adds
        # one concrete free function, deduplicated across call sites. The scan
        # skips EVERY method-bearing struct span (ordinary and promoted) so a
        # promoted struct's soon-to-be-replaced body is never rewritten in place.
        struct_spans = sorted(
            [struct.span for struct in structs_with_methods]
            + [struct.span for struct in promoted_structs]
        )
        # Member-access field resolution must see ALL struct/union spans (so a
        # field declaration is never mistaken for a local variable), and the
        # names of every struct/union type (so `Type var;` locals of a method-less
        # carrier are tracked).
        all_struct_names = {struct.name for struct in structs}
        call_replacements = self._rewrite_struct_member_call_sites(
            code,
            struct_names,
            methods_by_struct,
            template_methods_by_struct,
            operator_call_structs,
            struct_spans,
            field_structs_by_name,
            instantiated_template_functions,
            field_structs_by_name=field_structs_by_name,
            all_struct_spans=all_struct_spans,
            all_struct_names=all_struct_names,
        )
        replacements.extend(call_replacements)
        replacements.extend(
            self._preserve_rewritten_function_parameter_usage(
                code, call_replacements, function_excluded_spans, replacements
            )
        )
        free_functions.extend(instantiated_template_functions.values())

        rewritten = self._apply_text_replacements(code, replacements)
        if free_functions:
            rewritten = rewritten.rstrip() + "\n\n" + "\n\n".join(free_functions)
            if not rewritten.endswith("\n"):
                rewritten += "\n"
        rewritten = self._canonicalize_qualified_struct_alias_templates(
            expose_alias_template_owners(self, rewritten, structs), structs
        )
        return self._elide_used_resolved_static_data_members(
            resolve_retained_static_member_owners(self, rewritten),
            {struct.name for struct in [*structs_with_methods, *promoted_structs]},
        )

    def _apply_explicit_member_specializations(
        self,
        code: str,
        structs: Sequence[_MetalStructDefinition],
    ) -> None:
        specializations = self._find_explicit_member_specializations(code)
        if not specializations:
            return

        by_owner: Dict[
            Tuple[str, Tuple[str, ...]],
            List[_MetalExplicitMemberSpecialization],
        ] = {}
        seen: Dict[
            Tuple[str, Tuple[str, ...], Tuple[object, ...]],
            _MetalExplicitMemberSpecialization,
        ] = {}
        for specialization in specializations:
            owner_key = self._struct_specialization_comparison_key(
                specialization.owner_name,
                list(specialization.owner_arguments),
            )
            method_key = self._explicit_member_method_contract(specialization.method)
            duplicate_key = (*owner_key, method_key)
            previous = seen.get(duplicate_key)
            if previous is not None:
                requested = self._explicit_member_specialization_signature(
                    specialization
                )
                raise MetalTemplateSpecializationError(
                    "Metal explicit member specialization is defined more than "
                    f"once for '{requested}'.",
                    requested_signature=requested,
                    suggested_action=(
                        "keep one exact explicit member specialization for each "
                        "owner and method signature"
                    ),
                    source_location=self._source_location_for_offsets(
                        code,
                        specialization.span[0],
                        specialization.span[1],
                    ),
                    owner_struct_name=specialization.owner_qualified_name,
                    requested_arguments=specialization.owner_arguments,
                )
            seen[duplicate_key] = specialization
            by_owner.setdefault(owner_key, []).append(specialization)

        structs_by_name = {struct.name: struct for struct in structs}
        for struct in structs:
            materialized_owner = self._materialized_struct_specializations.get(
                struct.name
            )
            if materialized_owner is None:
                continue
            owner_name, owner_arguments = materialized_owner
            owner_key = self._struct_specialization_comparison_key(
                owner_name,
                list(owner_arguments),
            )
            owner_specializations = by_owner.get(owner_key, ())
            if not owner_specializations:
                continue

            methods = list(struct.methods)
            for specialization in owner_specializations:
                specialization_contract = self._explicit_member_method_contract(
                    specialization.method
                )
                candidates = [
                    (index, method)
                    for index, method in enumerate(methods)
                    if self._explicit_member_method_contract(
                        method, struct=struct, structs_by_name=structs_by_name
                    )
                    == specialization_contract
                ]
                if not candidates and any(
                    method.name == specialization.method.name for method in methods
                ):
                    # A same-named member whose signature no spelling matches; one
                    # already lowered to a free function leaves no member here.
                    requested = self._explicit_member_specialization_signature(
                        specialization
                    )
                    raise MetalTemplateSpecializationError(
                        "Metal explicit member specialization "
                        f"'{requested}' matches no member of its owner.",
                        requested_signature=requested,
                        suggested_action=(
                            "declare the specialized member with the parameter and "
                            "return types of a member of the primary template"
                        ),
                        source_location=self._source_location_for_offsets(
                            code,
                            specialization.span[0],
                            specialization.span[1],
                        ),
                        owner_struct_name=specialization.owner_qualified_name,
                        requested_arguments=specialization.owner_arguments,
                    )
                if len(candidates) != 1:
                    continue
                index, generic_method = candidates[0]
                methods[index] = replace(
                    specialization.method,
                    free_name=generic_method.free_name,
                    span=generic_method.span,
                )
            struct.methods = methods

    def _find_explicit_member_specializations(
        self,
        code: str,
    ) -> List[_MetalExplicitMemberSpecialization]:
        specializations: List[_MetalExplicitMemberSpecialization] = []
        ignored_spans = self._find_comment_and_literal_spans(code)
        position = 0
        pattern = re.compile(r"\btemplate\s*<\s*>\s*")
        while True:
            match = pattern.search(code, position)
            if match is None:
                break
            ignored = self._containing_span(match.start(), ignored_spans)
            if ignored is not None:
                position = ignored[1]
                continue

            declaration_start = match.end()
            body_start = self._find_next_top_level_char(
                code,
                declaration_start,
                "{",
            )
            semicolon = self._find_next_top_level_char(
                code,
                declaration_start,
                ";",
            )
            if body_start is None or (semicolon is not None and semicolon < body_start):
                position = declaration_start
                continue
            body_end = self._find_matching_brace(code, body_start)
            if body_end is None:
                position = body_start + 1
                continue

            header = code[declaration_start:body_start]
            parsed_header = self._explicit_member_specialization_header(header)
            if parsed_header is None:
                position = body_end
                continue
            (
                owner_name,
                owner_qualified_name,
                owner_arguments,
                dequalified_header,
            ) = parsed_header
            method_source = dequalified_header + code[body_start:body_end]
            method = self._parse_struct_method(
                owner_name,
                method_source,
                0,
                len(dequalified_header),
            )
            if method is not None:
                span = (match.start(), body_end)
                specializations.append(
                    _MetalExplicitMemberSpecialization(
                        owner_name=owner_name,
                        owner_qualified_name=owner_qualified_name,
                        owner_arguments=tuple(owner_arguments),
                        method=method,
                        span=span,
                    )
                )
            position = body_end
        return specializations

    def _explicit_member_specialization_header(
        self,
        header: str,
    ) -> Optional[Tuple[str, str, List[str], str]]:
        if re.match(r"\s*(?:struct|class|union)\b", header):
            return None

        operator_match = re.search(r"\boperator\s*\(\s*\)", header)
        if operator_match is not None:
            method_start = operator_match.start()
        else:
            paren_start = self._function_parameter_start(header)
            if paren_start is None:
                return None
            before_params = header[:paren_start]
            method_match = re.search(
                r"([A-Za-z_][A-Za-z0-9_]*)\s*$",
                before_params,
            )
            if method_match is None:
                return None
            method_start = method_match.start(1)

        scope_end = method_start
        while scope_end > 0 and header[scope_end - 1].isspace():
            scope_end -= 1
        if scope_end < 2 or header[scope_end - 2 : scope_end] != "::":
            return None
        scope_start = scope_end - 2
        owner_end = scope_start
        while owner_end > 0 and header[owner_end - 1].isspace():
            owner_end -= 1
        if owner_end == 0 or header[owner_end - 1] != ">":
            return None

        angle_end = owner_end - 1
        angle_start = self._matching_open_angle(header, angle_end)
        if angle_start is None:
            return None
        owner_match = re.search(
            r"(?P<owner>(?:(?:[A-Za-z_][A-Za-z0-9_]*)\s*::\s*)*"
            r"[A-Za-z_][A-Za-z0-9_]*)\s*$",
            header[:angle_start],
        )
        if owner_match is None:
            return None

        owner_qualified_name = re.sub(
            r"\s*::\s*",
            "::",
            owner_match.group("owner"),
        )
        owner_name = owner_qualified_name.rsplit("::", 1)[-1]
        owner_arguments = self._split_top_level_commas(
            header[angle_start + 1 : angle_end]
        )
        if not owner_arguments:
            return None
        dequalified_header = header[: owner_match.start("owner")] + header[scope_end:]
        return (
            owner_name,
            owner_qualified_name,
            owner_arguments,
            dequalified_header,
        )

    @staticmethod
    def _matching_open_angle(text: str, angle_end: int) -> Optional[int]:
        depth = 0
        for index in range(angle_end, -1, -1):
            character = text[index]
            if character == ">":
                depth += 1
            elif character == "<":
                depth -= 1
                if depth == 0:
                    return index
        return None

    def _explicit_member_method_contract(
        self,
        method: _MetalStructMethod,
        *,
        struct: Optional[_MetalStructDefinition] = None,
        structs_by_name: Optional[Dict[str, _MetalStructDefinition]] = None,
    ) -> Tuple[object, ...]:
        """The signature an explicit member specialization must repeat.

        A member of a materialized owner (``struct``) spells its types with the
        owner's member aliases, which resolve first, as do the member aliases of
        other concrete structs they name (``Traits_float::scalar_T``). Metal
        vectors compare by type, so ``vec<float, 2>``, ``metal::vec<float, 2>``
        and ``float2`` agree.
        """

        def canonical(type_text: str) -> str:
            text = self._normalize_template_argument_text(type_text)
            if struct is not None:
                text = self._canonicalize_struct_scoped_type(
                    text, struct, structs_by_name
                )
                text = self._resolve_concrete_member_type_aliases(
                    text, structs_by_name or {struct.name: struct}
                )
            return self._canonical_metal_vector_spelling(text)

        parameters = tuple(
            canonical(self._normalize_function_parameter_type_text(parameter))
            for parameter in self._split_top_level_commas(method.parameters)
            if parameter.strip() and parameter.strip() != "void"
        )
        return (
            method.name,
            method.is_static,
            method.is_operator_call,
            method.is_conversion_operator,
            canonical(method.return_type),
            parameters,
            self._method_receiver_contract_key(method),
        )

    _QUALIFIED_MEMBER_TYPE_RE = re.compile(
        r"(?<![A-Za-z0-9_:])(?:typename\s+)?(?P<owner>[A-Za-z_]\w*)\s*::\s*"
        r"(?P<alias>[A-Za-z_]\w*)"
    )

    def _resolve_concrete_member_type_aliases(
        self,
        text: str,
        structs_by_name: Dict[str, _MetalStructDefinition],
    ) -> str:
        """Replace ``Owner::alias`` with the alias's type for concrete owners."""

        def target(match: re.Match) -> str:
            owner = structs_by_name.get(match.group("owner"))
            aliased = owner.type_aliases.get(match.group("alias")) if owner else None
            if aliased is None:
                return match.group(0)
            aliased = re.sub(r"^typename\s+", "", aliased).strip()
            return self._canonicalize_struct_scoped_type(
                aliased, owner, structs_by_name
            )

        # Each pass resolves one level of aliases naming other structs' aliases.
        for _ in range(8):
            resolved = self._QUALIFIED_MEMBER_TYPE_RE.sub(target, text)
            if resolved == text:
                break
            text = self._normalize_template_argument_text(resolved)
        return text

    # Scalar types whose Metal vectors are also spelled ``<scalar><N>`` (MSL 2.2).
    _METAL_NAMED_VECTOR_SCALARS = frozenset(
        (
            "bool",
            "char",
            "uchar",
            "short",
            "ushort",
            "int",
            "uint",
            "long",
            "ulong",
            "half",
            "float",
            "bfloat",
        )
    )

    def _canonical_metal_vector_spelling(self, text: str) -> str:
        """Spell each concrete ``[metal::]vec<scalar, N>`` as ``<scalar><N>``."""

        def named(match: re.Match) -> str:
            scalar = match.group("scalar")
            if scalar not in self._METAL_NAMED_VECTOR_SCALARS:
                return match.group(0)
            return f"{scalar}{match.group('width')}"

        return re.sub(
            r"(?<![A-Za-z0-9_:])(?:::)?(?:metal::)?vec<\s*(?P<scalar>[A-Za-z_]\w*)"
            r"\s*,\s*(?P<width>[234])\s*>",
            named,
            text,
        )

    @staticmethod
    def _explicit_member_specialization_signature(
        specialization: _MetalExplicitMemberSpecialization,
    ) -> str:
        owner = specialization.owner_qualified_name
        arguments = ", ".join(specialization.owner_arguments)
        method = specialization.method
        return f"{owner}<{arguments}>::{method.name}({method.parameters})"

    def _reject_unresolved_temporary_functor_calls(
        self,
        code: str,
        structs_by_name: Dict[str, _MetalStructDefinition],
        template_declaration_spans: List[Tuple[int, int]],
    ) -> None:
        ignored_spans = sorted(
            self._find_comment_and_literal_spans(code) + template_declaration_spans
        )
        cursor = 0
        while cursor < len(code):
            ignored = self._containing_span(cursor, ignored_spans)
            if ignored is not None:
                cursor = ignored[1]
                continue
            if not (code[cursor].isalpha() or code[cursor] == "_"):
                # Only identifiers and ignored-span boundaries matter here.
                cursor = self._next_scan_stop(code, cursor + 1, ignored_spans)
                continue
            ident, consumed = self._read_identifier(code, cursor)
            ident_end = cursor + consumed
            if ident == "operator":
                cursor = ident_end
                continue
            constructor_start = ident_end
            while constructor_start < len(code) and code[constructor_start].isspace():
                constructor_start += 1
            temporary = self._temporary_functor_call(code, constructor_start)
            struct = structs_by_name.get(ident)
            if (
                temporary is not None
                and self._temporary_functor_is_candidate(code, cursor, temporary)
                and (
                    struct is None
                    or not (
                        struct.has_operator_call
                        or "operator()" in struct.member_prototype_names
                    )
                )
            ):
                self._raise_temporary_functor_error(
                    code,
                    ident,
                    cursor,
                    temporary,
                    reason="temporary-functor-type-unresolved",
                    detail=(
                        "the temporary type does not resolve to a concrete struct "
                        "with operator()"
                    ),
                )
            cursor = ident_end

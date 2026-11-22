"""Struct-scope type aliases and alias templates: collection, materialization and canonicalization."""

import copy
import re
from typing import (
    Dict,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
)

from ..pattern_cache import pattern as _cached_pattern
from ..source_analysis_cache import memoized
from ..struct_head_specifiers import (
    alignment_is_natural,
    skip_struct_head_specifiers,
)
from .support import (
    IDENTIFIER_RE,
    _MetalStructDefinition,
    _MetalTemplateFunction,
    _MetalTypeAliasBinding,
    _MetalTypeAliasTemplate,
)


class MetalStructScopedAliasMixin:
    """Members of ``MetalPreprocessor``: struct-scope type aliases and alias templates: collection, materialization and canonicalization."""

    def _find_concrete_struct_definitions(
        self, code: str
    ) -> List[_MetalStructDefinition]:
        """The concrete struct definitions of ``code``, computed once per source text.

        The scan depends on the text only, and one source is queried many times while it is materialized. Callers
        get their own shallow copies, so one may rebind a field such as ``span``; the member lists and tables in
        them are shared and must not be mutated.
        """
        definitions = memoized(
            self._source_analysis(code).values,
            "concrete_struct_definitions",
            lambda: tuple(self._scan_concrete_struct_definitions(code)),
        )
        return [copy.copy(definition) for definition in definitions]

    def _scan_concrete_struct_definitions(
        self, code: str
    ) -> List[_MetalStructDefinition]:
        # Locate every concrete (non-template) `struct/class/union Name { ... };`
        # and split its body into data members and method definitions. Template
        # structs/classes (`template <...> struct ...`) are skipped wholesale:
        # those are handled by the materializer, and their (possibly template)
        # methods are out of scope for lowering. Unions are captured for their
        # data members only (they carry no methods we lower) so a `union var;`
        # local and its members are type-resolvable for call-argument inference.
        template_spans = self._find_template_declaration_spans(code)
        namespace_spans = self._find_namespace_spans(code)
        definitions: List[_MetalStructDefinition] = []
        for match in re.finditer(r"\b(?P<aggregate_kind>struct|class|union)\s+", code):
            start = match.start()
            if self._containing_span(start, template_spans) is not None:
                continue
            name_start, head_alignments = skip_struct_head_specifiers(code, match.end())
            name, consumed = self._read_identifier(code, name_start)
            if not name or not consumed:
                continue
            after_name = name_start + consumed
            # Distinguish a definition (`struct Name { ... }`) from a forward
            # declaration / variable usage (`struct Name x;`). Skip anything that
            # is not immediately a `{` or a base-class clause `: ... {`.
            body_start = self._find_next_top_level_char(
                code, after_name, "{", stop_at=";"
            )
            semicolon = self._find_next_top_level_char(
                code, after_name, ";", stop_at="{"
            )
            if body_start is None or (semicolon is not None and semicolon < body_start):
                continue
            between = code[after_name:body_start]
            # Only a base-class clause may appear between the name and the body.
            stripped_between = between.strip()
            if stripped_between and not stripped_between.startswith(":"):
                continue
            body_end_after = self._find_matching_brace(code, body_start)
            if body_end_after is None:
                continue
            # The struct definition span includes the trailing semicolon so the
            # data-only replacement keeps the declaration well-formed.
            span_end = body_end_after
            trailing = code.find(";", body_end_after)
            if trailing != -1 and code[body_end_after:trailing].strip() == "":
                span_end = trailing + 1

            body = code[body_start + 1 : body_end_after - 1]
            parent = min(
                (
                    definition
                    for definition in definitions
                    if definition.span[0] < start < definition.span[1]
                ),
                key=lambda definition: definition.span[1] - definition.span[0],
                default=None,
            )
            namespace = self._namespace_at(namespace_spans, start)
            qualified_name = (
                f"{parent.qualified_name}::{name}"
                if parent is not None and parent.qualified_name
                else f"{namespace}::{name}" if namespace else name
            )
            member_prototype_names: Set[str] = set()
            (
                data_member_names,
                data_member_types,
                methods,
                template_methods,
                ordered_members,
                constructors,
                member_function_body_spans,
                layout_supported,
            ) = self._split_struct_body(
                name,
                body,
                body_start + 1,
                member_prototype_names=member_prototype_names,
            )
            # An alignas stricter than the natural alignment changes the layout.
            layout_supported = layout_supported and alignment_is_natural(
                head_alignments
            )
            definitions.append(
                _MetalStructDefinition(
                    name=name,
                    span=(start, span_end),
                    body_span=(body_start + 1, body_end_after - 1),
                    data_member_names=data_member_names,
                    methods=methods,
                    has_operator_call=any(m.is_operator_call for m in methods)
                    or any(m.is_operator_call for m in template_methods),
                    qualified_name=qualified_name,
                    template_methods=template_methods,
                    data_member_types=data_member_types,
                    data_members=ordered_members,
                    constructors=constructors,
                    member_function_body_spans=member_function_body_spans,
                    base_clause=stripped_between,
                    aggregate_kind=match.group("aggregate_kind"),
                    layout_supported=layout_supported,
                    member_prototype_names=member_prototype_names,
                    type_aliases=self._collect_struct_scope_type_aliases(body),
                    type_alias_templates=self._collect_struct_scope_type_alias_templates(
                        body
                    ),
                )
            )
        return definitions

    def _collect_struct_scope_type_aliases(self, body: str) -> Dict[str, str]:
        if "using" not in body and "typedef" not in body:
            return {}
        aliases: Dict[str, str] = {}
        ignored_spans = self._find_comment_and_literal_spans(body)
        lexical_scopes = self._find_lexical_brace_scopes(body)

        def is_struct_scope(position: int) -> bool:
            if self._containing_span(position, ignored_spans) is not None:
                return False
            return self._innermost_lexical_scope(
                lexical_scopes, position, len(body)
            ) == (0, len(body))

        for match in re.finditer(
            r"\busing\s+(?P<alias>[A-Za-z_]\w*)\s*=\s*" r"(?P<target>[^;{}]+?)\s*;",
            body,
            re.DOTALL,
        ):
            if not is_struct_scope(match.start()):
                continue
            target = self._normalize_template_argument_text(match.group("target"))
            if target:
                aliases[match.group("alias")] = target

        for match in re.finditer(
            r"\btypedef\s+(?P<target>[^;{}]+?)\s+" r"(?P<alias>[A-Za-z_]\w*)\s*;",
            body,
            re.DOTALL,
        ):
            if not is_struct_scope(match.start()):
                continue
            target = self._normalize_template_argument_text(match.group("target"))
            if target:
                aliases[match.group("alias")] = target
        return aliases

    def _collect_struct_scope_type_alias_templates(
        self, body: str
    ) -> Dict[str, _MetalTypeAliasTemplate]:
        if "template" not in body or "using" not in body:
            return {}
        templates: Dict[str, _MetalTypeAliasTemplate] = {}
        ignored_spans = self._find_comment_and_literal_spans(body)
        lexical_scopes = self._find_lexical_brace_scopes(body)

        for match in re.finditer(r"\btemplate\s*<", body):
            if self._containing_span(match.start(), ignored_spans) is not None:
                continue
            if self._innermost_lexical_scope(
                lexical_scopes, match.start(), len(body)
            ) != (0, len(body)):
                continue
            angle_start = body.find("<", match.start(), match.end())
            angle_end = self._find_matching_angle(body, angle_start)
            if angle_end is None:
                continue
            cursor = angle_end + 1
            while cursor < len(body) and body[cursor].isspace():
                cursor += 1
            alias_match = re.match(
                r"using\s+(?P<alias>[A-Za-z_]\w*)\s*=\s*",
                body[cursor:],
            )
            if alias_match is None:
                continue
            target_start = cursor + alias_match.end()
            target_end = body.find(";", target_start)
            if target_end == -1:
                continue
            parameters = self._parse_template_parameter_list(
                body[angle_start + 1 : angle_end]
            )
            target = self._normalize_template_argument_text(
                body[target_start:target_end]
            )
            if parameters and target:
                templates[alias_match.group("alias")] = _MetalTypeAliasTemplate(
                    parameters=parameters,
                    target=target,
                )
        return templates

    def _materialize_struct_scoped_alias_template(
        self,
        alias_template: _MetalTypeAliasTemplate,
        template_arguments: List[str],
        struct: _MetalStructDefinition,
    ) -> Optional[str]:
        parameters = alias_template.parameters
        if len(template_arguments) > len(parameters) or any(
            parameter.name is None or parameter.is_variadic for parameter in parameters
        ):
            return None

        bindings: Dict[str, str] = {}
        for index, parameter in enumerate(parameters):
            argument = (
                template_arguments[index]
                if index < len(template_arguments)
                else parameter.default
            )
            if argument is None or parameter.name is None:
                return None
            argument = self._normalize_template_argument_text(
                self._replace_identifiers(argument, bindings)
            )
            if not argument:
                return None
            bindings[parameter.name] = argument

        target = self._normalize_template_argument_text(
            self._replace_identifiers(alias_template.target, bindings)
        )
        parameter_names = {
            parameter.name for parameter in parameters if parameter.name is not None
        }
        if set(IDENTIFIER_RE.findall(target)) & parameter_names:
            return None

        constants = self._resolved_static_data_member_initializers(struct)
        if constants:
            target = self._resolve_qualified_static_constant_expression(
                target,
                {struct.name: struct},
            )
            target = self._substitute_template_argument_static_constants(
                target, constants
            )
        unresolved_static_constants = {
            member.name
            for member in struct.data_members
            if "static" in IDENTIFIER_RE.findall(member.type_text)
            and set(IDENTIFIER_RE.findall(member.type_text)).intersection(
                {"const", "constant", "constexpr"}
            )
            and member.name not in constants
        }
        if set(IDENTIFIER_RE.findall(target)) & unresolved_static_constants:
            return None

        target = re.sub(r"^typename\s+", "", target).strip()
        target = re.sub(r"::template\s+", "::", target)
        owner = struct.qualified_name or struct.name
        for alias in sorted(struct.type_alias_templates, key=len, reverse=True):
            target = re.sub(
                rf"(?<![A-Za-z0-9_:]){re.escape(alias)}(?=\s*<)",
                f"{owner}::{alias}",
                target,
            )
        return self._normalize_template_argument_text(target) or None

    def _canonicalize_struct_scoped_alias_templates(
        self,
        type_text: str,
        struct: _MetalStructDefinition,
        *,
        excluded_aliases: Optional[Set[str]] = None,
        qualified_only: bool = False,
        preserve_formatting: bool = False,
        allow_unqualified_owner: bool = True,
    ) -> str:
        templates = struct.type_alias_templates
        text = (
            str(type_text)
            if preserve_formatting
            else self._normalize_template_argument_text(type_text)
        )
        if not text or not templates:
            return text

        owner_names = {struct.qualified_name or struct.name}
        if allow_unqualified_owner:
            owner_names.add(struct.name)
        materialized_owner = self._materialized_struct_specializations.get(struct.name)
        if materialized_owner is not None:
            owner_names.add(materialized_owner[0])
        owner_pattern = "|".join(
            r"\s*::\s*".join(re.escape(part) for part in owner.split("::"))
            for owner in sorted(owner_names, key=len, reverse=True)
        )

        for _ in range(len(templates) + 2):
            ignored_spans = self._find_comment_and_literal_spans(text)
            replacements: List[Tuple[int, int, str]] = []
            for alias, alias_template in templates.items():
                if alias not in text:
                    # Both patterns below need the alias spelled in the text.
                    continue
                patterns = [
                    _cached_pattern(
                        rf"(?<![A-Za-z0-9_])(?:typename\s+)?"
                        rf"(?:::)?(?:{owner_pattern})"
                        rf"(?:\s*<[^<>{{}}()]*>)?\s*::\s*"
                        rf"(?:template\s+)?{re.escape(alias)}\s*<"
                    )
                ]
                if not qualified_only and alias not in (excluded_aliases or set()):
                    patterns.append(
                        _cached_pattern(rf"(?<![A-Za-z0-9_:]){re.escape(alias)}\s*<")
                    )
                for pattern in patterns:
                    for match in pattern.finditer(text):
                        if (
                            self._containing_span(match.start(), ignored_spans)
                            is not None
                        ):
                            continue
                        angle_start = text.rfind("<", match.start(), match.end())
                        angle_end = self._find_matching_angle(text, angle_start)
                        if angle_end is None:
                            continue
                        replacement = self._materialize_struct_scoped_alias_template(
                            alias_template,
                            self._split_top_level_commas(
                                text[angle_start + 1 : angle_end]
                            ),
                            struct,
                        )
                        if replacement is None:
                            continue
                        span = (match.start(), angle_end + 1)
                        if any(
                            not (span[1] <= start or span[0] >= end)
                            for start, end, _replacement in replacements
                        ):
                            continue
                        replacements.append((span[0], span[1], replacement))
            if not replacements:
                break
            text = self._apply_text_replacements(text, replacements)
        if preserve_formatting:
            return text
        return self._normalize_template_argument_text(text)

    def _canonicalize_qualified_struct_alias_templates(
        self,
        source: str,
        structs: List[_MetalStructDefinition],
    ) -> str:
        rewritten = source
        templated = [struct for struct in structs if struct.type_alias_templates]
        if not templated:
            return rewritten
        owner_name_counts: Dict[str, int] = {}
        for struct in structs:
            owner_name_counts[struct.name] = owner_name_counts.get(struct.name, 0) + 1
        for struct in templated:
            if not any(alias in rewritten for alias in struct.type_alias_templates):
                # Every rewrite needs an alias template spelled in the text.
                continue
            rewritten = self._canonicalize_struct_scoped_alias_templates(
                rewritten,
                struct,
                qualified_only=True,
                preserve_formatting=True,
                allow_unqualified_owner=owner_name_counts[struct.name] == 1,
            )
        return rewritten

    def _canonicalize_struct_static_alias_owners(self, code: str) -> str:
        """Expose concrete template owners before lowering their static calls."""
        if "::" not in code or not ("using" in code or "typedef" in code):
            return code
        structs = self._find_concrete_struct_definitions(code)
        structs_by_name = {struct.name: struct for struct in structs}
        replacements: List[Tuple[int, int, str]] = []
        for struct in structs:
            if not struct.type_aliases:
                continue
            aliases = "|".join(
                re.escape(name)
                for name in sorted(struct.type_aliases, key=len, reverse=True)
            )
            pattern = re.compile(rf"(?<![A-Za-z0-9_:.])(?P<alias>{aliases})(?=\s*::)")
            members = [*struct.methods, *struct.template_methods, *struct.constructors]
            constants = self._resolved_static_data_member_initializers(struct)
            for member in members:
                if member.span is None:
                    continue
                start, end = member.span
                body = code[start:end]
                ignored = self._find_comment_and_literal_spans(body)
                local_aliases = self._local_type_alias_shadow_scopes(body)
                local_bindings = self._collect_local_type_alias_bindings(
                    body, [(0, len(body))]
                )
                parameter_names = set(
                    getattr(
                        member, "parameter_names", getattr(member, "param_names", [])
                    )
                )
                for match in pattern.finditer(body):
                    alias = match.group("alias")
                    if (
                        self._containing_span(match.start(), ignored) is not None
                        or alias in parameter_names
                    ):
                        continue
                    shadowed = self._shadowed_type_aliases_at(
                        local_aliases, match.start()
                    )
                    target = self._canonicalize_type_aliases_at(
                        alias, local_bindings, match.start()
                    )
                    if target is None:
                        continue
                    target = self._canonicalize_struct_scoped_type(
                        target, struct, structs_by_name, excluded_aliases=shadowed
                    )
                    target = self._substitute_template_argument_static_constants(
                        target, constants
                    )
                    if (
                        target not in structs_by_name
                        and self._exact_template_type_id(target) is None
                    ):
                        continue
                    replacements.append(
                        (start + match.start(), start + match.end(), target)
                    )
        return self._apply_text_replacements(code, replacements)

    def _canonicalize_qualified_struct_type_aliases(
        self,
        source: str,
        structs: List[_MetalStructDefinition],
    ) -> str:
        aliased_structs = [struct for struct in structs if struct.type_aliases]
        if not aliased_structs:
            return source

        structs_by_name = {struct.name: struct for struct in structs}
        owner_name_counts: Dict[str, int] = {}
        for struct in structs:
            owner_name_counts[struct.name] = owner_name_counts.get(struct.name, 0) + 1

        def type_pattern(type_text: str) -> str:
            parts = re.split(r"(\s+|::|[<>,*&])", type_text)
            return "".join(
                (
                    r"\s*"
                    if not part or part.isspace()
                    else (
                        rf"\s*{re.escape(part)}\s*"
                        if part in {"::", "<", ">", ",", "*", "&"}
                        else re.escape(part)
                    )
                )
                for part in parts
            )

        # Every rule rewrites ``Owner::alias``, so a rule can match only text that spells its alias. Rules are built
        # in a fixed order as their aliases appear in the text; each owner's patterns and each alias's target are
        # resolved at most once per call.
        candidates: List[Tuple[_MetalStructDefinition, str]] = [
            (struct, alias)
            for struct in aliased_structs
            for alias in struct.type_aliases
            if alias not in struct.type_alias_templates
        ]
        if not candidates:
            return source
        owners_by_struct: Dict[int, List[str]] = {}
        targets: Dict[int, Optional[str]] = {}

        def owner_patterns_of(struct: _MetalStructDefinition) -> List[str]:
            owners = owners_by_struct.get(id(struct))
            if owners is not None:
                return owners
            owner_patterns = (
                {type_pattern(struct.qualified_name)}
                if struct.qualified_name
                else set()
            )
            if owner_name_counts[struct.name] == 1:
                owner_patterns.add(type_pattern(struct.name))
            specialization = self._materialized_struct_specializations.get(struct.name)
            if specialization is not None:
                source_name, source_arguments = specialization
                owner_patterns.add(
                    type_pattern(f"{source_name}<{', '.join(source_arguments)}>")
                )
            owners = owners_by_struct[id(struct)] = sorted(owner_patterns)
            return owners

        def rules_for(text: str) -> List[Tuple[re.Pattern[str], str]]:
            rules: List[Tuple[re.Pattern[str], str]] = []
            for index, (struct, alias) in enumerate(candidates):
                if alias not in text:
                    continue
                if index not in targets:
                    targets[index] = self._canonicalize_struct_scoped_type(
                        alias,
                        struct,
                        structs_by_name,
                    )
                target = targets[index]
                if not target or target == alias:
                    continue
                for owner_pattern in owner_patterns_of(struct):
                    rules.append(
                        (
                            _cached_pattern(
                                rf"(?<![A-Za-z0-9_:])(?:typename\s+)?"
                                rf"(?:::)?{owner_pattern}\s*::\s*"
                                rf"{re.escape(alias)}\b"
                            ),
                            target,
                        )
                    )
            return rules

        rewritten = source
        # A struct has at most three owner spellings, so this bounds the rule count.
        for _ in range(3 * len(candidates) + 1):
            rules = rules_for(rewritten)
            if not rules:
                break
            ignored_spans = self._find_comment_and_literal_spans(rewritten)
            replacements: List[Tuple[int, int, str]] = []
            for pattern, target in rules:
                for match in pattern.finditer(rewritten):
                    if self._containing_span(match.start(), ignored_spans) is not None:
                        continue
                    span = (match.start(), match.end())
                    if any(
                        not (span[1] <= start or span[0] >= end)
                        for start, end, _replacement in replacements
                    ):
                        continue
                    replacements.append((span[0], span[1], target))
            if not replacements:
                break
            updated = self._apply_text_replacements(rewritten, replacements)
            if updated == rewritten:
                break
            rewritten = updated
        return rewritten

    def _canonicalize_struct_scoped_type(
        self,
        type_text: str,
        struct: _MetalStructDefinition,
        structs_by_name: Optional[Dict[str, _MetalStructDefinition]],
        excluded_aliases: Optional[Set[str]] = None,
    ) -> str:
        text = self._normalize_template_argument_text(type_text)
        known_structs = structs_by_name or {struct.name: struct}
        text = self._canonicalize_struct_scoped_alias_templates(
            text,
            struct,
            excluded_aliases=excluded_aliases,
        )
        text = self._canonicalize_qualified_struct_alias_templates(
            text,
            list(known_structs.values()),
        )
        all_type_aliases = dict(struct.type_aliases)
        type_aliases = {
            alias: target
            for alias, target in all_type_aliases.items()
            if alias not in (excluded_aliases or set())
        }
        if not text or not all_type_aliases:
            return text
        text = re.sub(r"\btypename\s+", "", text)
        text = re.sub(r"::template\s+", "::", text)
        owner_names = {struct.name}
        materialized_owner = self._materialized_struct_specializations.get(struct.name)
        if materialized_owner is not None:
            owner_names.add(materialized_owner[0])
        owner_pattern = "|".join(
            re.escape(owner) for owner in sorted(owner_names, key=len, reverse=True)
        )
        for alias in sorted(all_type_aliases, key=len, reverse=True):
            if alias not in text:
                continue
            qualified_alias = _cached_pattern(
                rf"(?<![A-Za-z0-9_])"
                rf"(?:(?:[A-Za-z_]\w*)\s*::\s*)*"
                rf"(?:{owner_pattern})(?:\s*<[^<>{{}}()]*>)?\s*::\s*"
                rf"(?:template\s+)?{re.escape(alias)}\b"
            )
            qualified_target = re.sub(
                r"^typename\s+", "", all_type_aliases[alias]
            ).strip()
            qualified_target = re.sub(r"::template\s+", "::", qualified_target)
            text = qualified_alias.sub(qualified_target, text)

        def struct_alias_target(alias: str, seen: Set[str]) -> Optional[str]:
            if alias in seen:
                return None
            target = type_aliases.get(alias)
            if target is None:
                return None
            target = re.sub(r"^typename\s+", "", target).strip()
            if not IDENTIFIER_RE.fullmatch(target):
                return None
            if target in known_structs:
                return target
            return struct_alias_target(target, {*seen, alias})

        for _ in range(len(type_aliases) + 2):
            previous = text
            for alias in sorted(type_aliases, key=len, reverse=True):
                if alias not in text:
                    # Each rewrite below needs the alias spelled in the text.
                    continue
                owner = struct_alias_target(alias, set())
                escaped = re.escape(alias)
                if owner is not None:
                    text = _cached_pattern(rf"(?<![A-Za-z0-9_:]){escaped}(?=::)").sub(
                        owner, text
                    )
                text = _cached_pattern(rf"(?<![A-Za-z0-9_:]){escaped}(?=\s*<)").sub(
                    f"{struct.name}::{alias}", text
                )
                target = type_aliases[alias]
                target = re.sub(r"^typename\s+", "", target).strip()
                target = re.sub(r"::template\s+", "::", target)
                text = _cached_pattern(
                    rf"(?<![A-Za-z0-9_:]){escaped}(?![A-Za-z0-9_<])"
                ).sub(target, text)
            if text == previous:
                break
        text = self._canonicalize_qualified_struct_alias_templates(
            text,
            list(known_structs.values()),
        )
        return self._normalize_template_argument_text(text)

    def _canonicalize_struct_scoped_parameters(
        self,
        parameters: str,
        struct: _MetalStructDefinition,
        structs_by_name: Optional[Dict[str, _MetalStructDefinition]],
    ) -> str:
        if not (set(IDENTIFIER_RE.findall(parameters)) & set(struct.type_aliases)):
            return parameters.strip()
        canonical: List[str] = []
        for parameter in self._split_top_level_commas(parameters):
            if not parameter.strip():
                continue
            declaration, default = self._split_top_level_assignment(parameter)
            attributes = self._metal_attributes(declaration)
            plain_declaration = self._strip_metal_attributes(declaration).strip()
            name = self._declared_data_member_name(plain_declaration)
            member = (
                self._parse_ordered_data_member(name, plain_declaration)
                if name is not None
                else None
            )
            if member is not None:
                type_text = self._canonicalize_struct_scoped_type(
                    member.type_text, struct, structs_by_name
                )
                declaration = f"{type_text} {member.name}{member.array_suffix}"
                if attributes:
                    declaration += " " + " ".join(attributes)
            else:
                declaration = self._canonicalize_struct_scoped_type(
                    declaration, struct, structs_by_name
                )
            if default is not None:
                declaration = f"{declaration} = {default}"
            canonical.append(declaration)
        return ", ".join(canonical)

    def _local_type_alias_shadow_scopes(
        self, body: str
    ) -> Dict[str, List[Tuple[int, int, int]]]:
        aliases: Dict[str, List[Tuple[int, int, int]]] = {}
        declarations: List[Tuple[int, str]] = []
        for match in re.finditer(r"\busing\s+([A-Za-z_]\w*)\s*=\s*[^;{}]+;", body):
            declarations.append((match.start(), match.group(1)))
        for match in re.finditer(r"\btypedef\s+[^;{}]+\s+([A-Za-z_]\w*)\s*;", body):
            declarations.append((match.start(), match.group(1)))
        if not declarations:
            return aliases
        ignored_spans = self._find_comment_and_literal_spans(body)
        lexical_scopes = self._find_lexical_brace_scopes(body)
        for declaration_position, alias in sorted(declarations):
            if self._containing_span(declaration_position, ignored_spans) is not None:
                continue
            scope_start, scope_end = self._innermost_lexical_scope(
                lexical_scopes, declaration_position, len(body)
            )
            aliases.setdefault(alias, []).append(
                (declaration_position, scope_start, scope_end)
            )
        return aliases

    def _collect_local_type_alias_bindings(
        self,
        code: str,
        owner_spans: List[Tuple[int, int]],
        *,
        skip_spans: Optional[List[Tuple[int, int]]] = None,
    ) -> Dict[str, List[_MetalTypeAliasBinding]]:
        raw_aliases: List[Tuple[int, str, str]] = []

        for match in re.finditer(
            r"\busing\s+(?P<alias>[A-Za-z_]\w*)\s*=\s*" r"(?P<target>[^;{}]+?)\s*;",
            code,
            re.DOTALL,
        ):
            if self._containing_span(match.start(), owner_spans) is None:
                continue
            raw_aliases.append(
                (match.start(), match.group("alias"), match.group("target"))
            )

        for match in re.finditer(
            r"\btypedef\s+(?P<target>[^;{}]+?)\s+" r"(?P<alias>[A-Za-z_]\w*)\s*;",
            code,
            re.DOTALL,
        ):
            if self._containing_span(match.start(), owner_spans) is None:
                continue
            raw_aliases.append(
                (match.start(), match.group("alias"), match.group("target"))
            )

        if not raw_aliases:
            return {}
        ignored_spans = sorted(
            [*(skip_spans or []), *self._find_comment_and_literal_spans(code)]
        )
        raw_aliases = [
            alias
            for alias in raw_aliases
            if self._containing_span(alias[0], ignored_spans) is None
        ]
        if not raw_aliases:
            return {}
        aliases: Dict[str, List[_MetalTypeAliasBinding]] = {}
        lexical_scopes = self._find_lexical_brace_scopes(code)
        namespace_spans = self._find_namespace_spans(code)
        namespace_scopes = {(start, end) for start, end, _namespace in namespace_spans}
        for declaration_position, alias, target in sorted(raw_aliases):
            resolved_target = self._resolve_type_aliases_at(
                target,
                aliases,
                declaration_position,
            )
            scope_start, scope_end = self._innermost_lexical_scope(
                lexical_scopes, declaration_position, len(code)
            )
            aliases.setdefault(alias, []).append(
                _MetalTypeAliasBinding(
                    declaration_position=declaration_position,
                    scope_start=scope_start,
                    scope_end=scope_end,
                    target=resolved_target,
                )
            )
            namespace = self._namespace_at(namespace_spans, declaration_position)
            if namespace and (scope_start, scope_end) in namespace_scopes:
                # A namespace member's fully qualified spelling is visible from
                # every later lexical scope, not only from inside the namespace
                # braces where its unqualified spelling was declared.
                aliases.setdefault(f"{namespace}::{alias}", []).append(
                    _MetalTypeAliasBinding(
                        declaration_position=declaration_position,
                        scope_start=0,
                        scope_end=len(code),
                        target=resolved_target,
                    )
                )
        return aliases

    def _qualified_type_alias_references(
        self,
        text: str,
        aliases: Dict[str, List[_MetalTypeAliasBinding]],
    ) -> List[Tuple[int, int, str]]:
        """Return longest qualified alias prefixes without reading comments."""
        if "::" not in text or not aliases:
            return []
        ignored_spans = self._find_comment_and_literal_spans(text)
        references: List[Tuple[int, int, str]] = []
        pattern = re.compile(
            r"(?<![A-Za-z0-9_:])(?:::)?[A-Za-z_]\w*" r"(?:\s*::\s*[A-Za-z_]\w*)+"
        )
        for match in pattern.finditer(text):
            if self._containing_span(match.start(), ignored_spans) is not None:
                continue
            components = list(IDENTIFIER_RE.finditer(match.group(0)))
            for component_count in range(len(components), 1, -1):
                alias = "::".join(
                    component.group(0) for component in components[:component_count]
                )
                if alias not in aliases:
                    continue
                references.append(
                    (
                        match.start(),
                        match.start() + components[component_count - 1].end(),
                        alias,
                    )
                )
                break
        return references

    def _resolve_type_aliases_at(
        self,
        type_text: str,
        aliases: Dict[str, List[_MetalTypeAliasBinding]],
        position: int,
    ) -> str:
        replacements = {
            alias: resolved
            for alias in aliases
            if "::" not in alias
            and (
                resolved := self._resolve_struct_type_alias_at(aliases, alias, position)
            )
            is not None
        }
        resolved = self._normalize_template_argument_text(type_text)
        for _ in range(max(len(aliases), 1)):
            qualified_replacements = []
            for start, end, alias in self._qualified_type_alias_references(
                resolved,
                aliases,
            ):
                target = self._resolve_struct_type_alias_at(aliases, alias, position)
                if target is not None:
                    qualified_replacements.append((start, end, target))
            candidate = self._apply_text_replacements(
                resolved,
                qualified_replacements,
            )
            candidate = self._replace_identifiers(
                candidate,
                replacements,
                skip_template_names=True,
            )
            if candidate == resolved:
                break
            resolved = candidate
        return self._normalize_template_argument_text(resolved)

    def _canonicalize_type_aliases_at(
        self,
        type_text: str,
        aliases: Dict[str, List[_MetalTypeAliasBinding]],
        position: int,
        *,
        excluded_aliases: Optional[Set[str]] = None,
    ) -> Optional[str]:
        """Resolve aliases visible at ``position`` without crossing a cycle.

        Alias targets are resolved at their own declaration positions. This
        preserves lexical shadowing and prevents a later declaration from
        retroactively satisfying an unresolved or forward alias target. Fully
        qualified namespace aliases are resolved atomically before individual
        identifiers so a namespace prefix cannot be mistaken for a local alias.
        """
        normalized = self._normalize_template_argument_text(type_text)
        if not normalized:
            return None

        cache: Dict[Tuple[str, int, int, int], Optional[str]] = {}

        def canonicalize(
            text: str,
            context_position: int,
            resolving: Set[Tuple[str, int, int, int]],
        ) -> Optional[str]:
            candidate = self._normalize_template_argument_text(text)
            if not candidate:
                return None

            qualified_replacements: List[Tuple[int, int, str]] = []
            for start, end, identifier in self._qualified_type_alias_references(
                candidate,
                aliases,
            ):
                binding = self._resolve_type_alias_binding_at(
                    aliases,
                    identifier,
                    context_position,
                )
                if binding is None:
                    if any(
                        future.declaration_position > context_position
                        and future.scope_start <= context_position < future.scope_end
                        for future in aliases.get(identifier, [])
                    ):
                        return None
                    continue
                key = (
                    identifier,
                    binding.declaration_position,
                    binding.scope_start,
                    binding.scope_end,
                )
                if key in resolving or binding.target is None:
                    return None
                if key not in cache:
                    cache[key] = canonicalize(
                        binding.target,
                        binding.declaration_position,
                        {*resolving, key},
                    )
                resolved = cache[key]
                if resolved is None:
                    return None
                qualified_replacements.append((start, end, resolved))
            candidate = self._apply_text_replacements(
                candidate,
                qualified_replacements,
            )

            replacements: Dict[str, str] = {}
            identifiers = dict.fromkeys(
                match.group(0)
                for match in IDENTIFIER_RE.finditer(candidate)
                if not self._is_member_identifier_context(candidate, match.start())
            )
            for identifier in identifiers:
                if identifier in (excluded_aliases or set()):
                    continue
                binding = self._resolve_type_alias_binding_at(
                    aliases,
                    identifier,
                    context_position,
                )
                if binding is None:
                    if any(
                        future.declaration_position > context_position
                        and future.scope_start <= context_position < future.scope_end
                        for future in aliases.get(identifier, [])
                    ):
                        return None
                    continue
                key = (
                    identifier,
                    binding.declaration_position,
                    binding.scope_start,
                    binding.scope_end,
                )
                if key in resolving:
                    return None
                if key not in cache:
                    if binding.target is None:
                        return None
                    cache[key] = canonicalize(
                        binding.target,
                        binding.declaration_position,
                        {*resolving, key},
                    )
                resolved = cache[key]
                if resolved is None:
                    return None
                replacements[identifier] = resolved
            return self._normalize_template_argument_text(
                self._replace_identifiers(candidate, replacements)
            )

        return canonicalize(normalized, position, set())

    def _canonicalize_template_function_arguments_at(
        self,
        template: _MetalTemplateFunction,
        arguments: Sequence[str],
        aliases: Dict[str, List[_MetalTypeAliasBinding]],
        position: int,
    ) -> Optional[Tuple[str, ...]]:
        """Canonicalize only TYPE arguments of one function template at a point.

        Non-type arguments may legitimately use an identifier that is also a type
        alias in another scope, so applying alias replacement indiscriminately can
        change a value specialization. Variadic packs are mapped with the same
        fixed-tail rule as template argument binding.
        """
        canonical = [
            self._normalize_template_argument_text(argument) for argument in arguments
        ]
        non_type_parameters = set(template.template_parameter_types)
        variadic_parameters = set(template.variadic_template_parameters)
        argument_index = 0
        for parameter_index, parameter in enumerate(template.template_parameters):
            if parameter in variadic_parameters:
                remaining_fixed = (
                    len(template.template_parameters) - parameter_index - 1
                )
                argument_count = max(
                    0,
                    len(canonical) - argument_index - remaining_fixed,
                )
            else:
                argument_count = int(argument_index < len(canonical))
            if parameter not in non_type_parameters:
                for index in range(
                    argument_index,
                    min(argument_index + argument_count, len(canonical)),
                ):
                    resolved = self._canonicalize_type_aliases_at(
                        canonical[index],
                        aliases,
                        position,
                    )
                    if resolved is None:
                        return None
                    canonical[index] = resolved
            argument_index += argument_count
        return tuple(canonical)

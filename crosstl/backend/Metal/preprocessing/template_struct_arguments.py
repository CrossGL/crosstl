"""Qualified template struct references, static constants in template arguments and partial specialization matching."""

import re
from typing import (
    Dict,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
)

from .support import (
    IDENTIFIER_RE,
    _MetalStructDefinition,
    _MetalTemplateStruct,
    _MetalTypeAliasBinding,
)


class MetalTemplateStructArgumentMixin:
    """Members of ``MetalPreprocessor``: qualified template struct references, static constants in template arguments and partial specialization matching."""

    def _resolve_concrete_qualified_template_struct_aliases(
        self,
        type_text: str,
        struct_templates_by_name: Dict[str, _MetalTemplateStruct],
        partial_specializations: Dict[
            str, List[Tuple[_MetalTemplateStruct, List[str]]]
        ],
        type_aliases: Dict[str, List[_MetalTypeAliasBinding]],
        concrete_structs: List[_MetalStructDefinition],
        source_position: int,
        *,
        source_code: str,
        allow_unqualified_remove_cv_t: bool = False,
    ) -> str:
        resolved = self._normalize_template_argument_text(type_text)
        max_iterations = len(struct_templates_by_name) + 1
        for _ in range(max_iterations):
            replacements: List[Tuple[int, int, str]] = []
            position = 0
            while position < len(resolved):
                match = IDENTIFIER_RE.search(resolved, position)
                if match is None:
                    break
                position = match.end()
                name = match.group(0)
                primary_template = struct_templates_by_name.get(name)
                if primary_template is None:
                    continue
                angle_start = position
                while angle_start < len(resolved) and resolved[angle_start].isspace():
                    angle_start += 1
                if angle_start >= len(resolved) or resolved[angle_start] != "<":
                    continue
                angle_end = self._find_matching_angle(resolved, angle_start)
                if angle_end is None:
                    continue
                qualified_alias = self._qualified_template_struct_alias_at(
                    resolved,
                    (
                        self._scoped_identifier_start(resolved, match.start()),
                        angle_end + 1,
                    ),
                )
                if qualified_alias is None:
                    position = angle_start + 1
                    continue
                alias_name, alias_span = qualified_alias
                arguments = self._split_top_level_commas(
                    resolved[angle_start + 1 : angle_end]
                )
                arguments = (
                    self._template_arguments_with_resolved_defaults(
                        primary_template,
                        arguments,
                    )
                    or arguments
                )
                partial_match_arguments = [
                    self._canonicalize_metal_standard_type_aliases(
                        self._canonicalize_qualified_struct_type_aliases(
                            self._resolve_type_aliases_at(
                                argument,
                                type_aliases,
                                source_position,
                            ),
                            concrete_structs,
                        ),
                        allow_unqualified_remove_cv_t=(allow_unqualified_remove_cv_t),
                    )
                    for argument in arguments
                ]
                matching_partials: List[Tuple[_MetalTemplateStruct, Dict[str, str]]] = (
                    []
                )
                matching_patterns: List[List[str]] = []
                has_unmaterializable_partial = False
                for (
                    partial_template,
                    specialization_arguments,
                ) in partial_specializations.get(name, []):
                    canonical_specialization_arguments = [
                        self._canonical_partial_struct_specialization_pattern(
                            partial_template,
                            argument,
                            type_aliases,
                            allow_unqualified_remove_cv_t=(
                                allow_unqualified_remove_cv_t
                            ),
                        )
                        for argument in specialization_arguments
                    ]
                    if not self._template_arguments_may_match_partial_struct_specialization(
                        partial_template,
                        canonical_specialization_arguments,
                        partial_match_arguments,
                        code=source_code,
                        position=source_position,
                    ):
                        continue
                    bindings = self._partial_struct_specialization_bindings(
                        partial_template,
                        canonical_specialization_arguments,
                        partial_match_arguments,
                        code=source_code,
                        position=source_position,
                    )
                    if bindings is None:
                        has_unmaterializable_partial = True
                    else:
                        matching_partials.append((partial_template, bindings))
                        matching_patterns.append(canonical_specialization_arguments)
                matching_partials = self._most_specialized_struct_partials(
                    name,
                    arguments,
                    matching_partials,
                    matching_patterns,
                    code=source_code,
                    position=source_position,
                )
                if has_unmaterializable_partial or len(matching_partials) > 1:
                    position = angle_start + 1
                    continue
                if matching_partials:
                    selected_template, bindings = matching_partials[0]
                elif self._template_arguments_satisfy_parameters(
                    primary_template,
                    arguments,
                ):
                    selected_template = primary_template
                    bindings, _variadic_bindings = self._template_argument_bindings(
                        primary_template,
                        arguments,
                    )
                else:
                    position = angle_start + 1
                    continue
                target = self._concrete_template_struct_scalar_alias_target(
                    selected_template,
                    bindings,
                    alias_name,
                    allow_unqualified_remove_cv_t=(allow_unqualified_remove_cv_t),
                )
                if target is None:
                    position = angle_start + 1
                    continue
                replacements.append((alias_span[0], alias_span[1], target))
                position = alias_span[1]
            if not replacements:
                break
            updated = self._apply_text_replacements(resolved, replacements)
            if updated == resolved:
                break
            resolved = updated
        return self._normalize_template_argument_text(resolved)

    def _template_arguments_contain_qualified_struct_alias(
        self,
        arguments: Sequence[str],
        struct_template_names: Set[str],
    ) -> bool:
        for argument in arguments:
            position = 0
            while position < len(argument):
                match = IDENTIFIER_RE.search(argument, position)
                if match is None:
                    break
                position = match.end()
                if match.group(0) not in struct_template_names:
                    continue
                angle_start = position
                while angle_start < len(argument) and argument[angle_start].isspace():
                    angle_start += 1
                if angle_start >= len(argument) or argument[angle_start] != "<":
                    continue
                angle_end = self._find_matching_angle(argument, angle_start)
                if angle_end is None:
                    continue
                qualified_start = angle_end + 1
                while (
                    qualified_start < len(argument)
                    and argument[qualified_start].isspace()
                ):
                    qualified_start += 1
                if argument.startswith("::", qualified_start):
                    return True
                position = angle_start + 1
        return False

    def _replace_discovered_struct_template_references(
        self,
        argument: str,
        materialized_structs: Dict[str, Dict[Tuple[str, ...], str]],
    ) -> str:
        replacements: List[Tuple[int, int, str]] = []
        i = 0
        while i < len(argument):
            if argument[i] in "\"'":
                _literal, consumed = self._read_string(argument, i)
                i += consumed
                continue
            if argument.startswith("//", i):
                break
            if argument.startswith("/*", i):
                end = argument.find("*/", i + 2)
                if end == -1:
                    break
                i = end + 2
                continue
            if not (argument[i].isalpha() or argument[i] == "_"):
                i += 1
                continue
            ident, consumed = self._read_identifier(argument, i)
            known_specializations = materialized_structs.get(ident)
            if known_specializations is None:
                i += consumed
                continue
            angle_start = i + consumed
            while angle_start < len(argument) and argument[angle_start].isspace():
                angle_start += 1
            if angle_start >= len(argument) or argument[angle_start] != "<":
                i += consumed
                continue
            angle_end = self._find_matching_angle(argument, angle_start)
            if angle_end is None:
                i += consumed
                continue
            template_arguments = self._split_top_level_commas(
                argument[angle_start + 1 : angle_end]
            )
            key = self._template_specialization_key(ident, template_arguments)
            specialized_name = known_specializations.get(key[1])
            if specialized_name is None:
                i += consumed
                continue
            replacements.append(
                (
                    self._scoped_identifier_start(argument, i),
                    angle_end + 1,
                    specialized_name,
                )
            )
            i = angle_end + 1
        if not replacements:
            return argument
        return self._apply_text_replacements(argument, replacements)

    def _substitute_template_argument_static_constants(
        self, argument: str, constants: Dict[str, str]
    ) -> str:
        result: List[str] = []
        i = 0
        while i < len(argument):
            if argument[i] in "\"'":
                literal, consumed = self._read_string(argument, i)
                result.append(literal)
                i += consumed
                continue
            if argument.startswith("//", i):
                result.append(argument[i:])
                break
            if argument.startswith("/*", i):
                end = argument.find("*/", i + 2)
                if end == -1:
                    result.append(argument[i:])
                    break
                result.append(argument[i : end + 2])
                i = end + 2
                continue
            if argument[i].isalpha() or argument[i] == "_":
                identifier, consumed = self._read_identifier(argument, i)
                replacement = constants.get(identifier)
                previous = i - 1
                while previous >= 0 and argument[previous].isspace():
                    previous -= 1
                scoped = previous >= 0 and argument[previous] == "."
                if previous >= 0 and argument[previous] == ":":
                    qualifier_colon = previous - 1
                    while qualifier_colon >= 0 and argument[qualifier_colon].isspace():
                        qualifier_colon -= 1
                    scoped = qualifier_colon >= 0 and argument[qualifier_colon] == ":"
                following = i + consumed
                while following < len(argument) and argument[following].isspace():
                    following += 1
                if (
                    replacement is not None
                    and not scoped
                    and (following >= len(argument) or argument[following] != "(")
                ):
                    result.append(self._static_initializer_reference(replacement))
                else:
                    result.append(identifier)
                i += consumed
                continue
            result.append(argument[i])
            i += 1
        return "".join(result).strip()

    def _template_struct_specialization_arguments(
        self, template: _MetalTemplateStruct
    ) -> Optional[List[str]]:
        match = re.match(
            r"\s*(?:\[\[[^\]]*\]\]\s*)*(?:struct|class)\s+"
            + re.escape(template.name)
            + r"\s*<",
            template.source,
            re.DOTALL,
        )
        if match is None:
            return None
        angle_start = match.end() - 1
        angle_end = self._find_matching_angle(template.source, angle_start)
        if angle_end is None:
            return []
        return self._split_top_level_commas(
            template.source[angle_start + 1 : angle_end]
        )

    def _canonical_partial_struct_specialization_pattern(
        self,
        template: _MetalTemplateStruct,
        argument: str,
        type_aliases: Dict[str, List[_MetalTypeAliasBinding]],
        *,
        allow_unqualified_remove_cv_t: bool = False,
    ) -> str:
        # Concrete arguments are compared with aliases resolved, so a pattern
        # spelled with an alias (`S<complex64_t, U>` for `using complex64_t =
        # complex_t<float>`) must be resolved where the partial specialization
        # is declared. Its own template parameters shadow any alias.
        parameters = set(template.template_parameters)
        visible = {
            alias: bindings
            for alias, bindings in type_aliases.items()
            if alias not in parameters
        }
        return self._canonicalize_metal_standard_type_aliases(
            self._resolve_type_aliases_at(argument, visible, template.span[0]),
            allow_unqualified_remove_cv_t=allow_unqualified_remove_cv_t,
            dependent_identifiers=parameters,
        )

    def _template_arguments_may_match_partial_struct_specialization(
        self,
        template: _MetalTemplateStruct,
        specialization_arguments: List[str],
        concrete_arguments: List[str],
        *,
        code: str,
        position: int,
    ) -> bool:
        parameter_names = set(template.template_parameters)
        if template.variadic_template_parameters:
            return self._template_trait_variadic_specialization_may_match(
                tuple(specialization_arguments),
                concrete_arguments,
                parameter_names,
                template.variadic_template_parameters,
            )
        return (
            self._partial_struct_specialization_bindings(
                template,
                specialization_arguments,
                concrete_arguments,
                code=code,
                position=position,
            )
            is not None
        )

    @staticmethod
    def _equivalent_template_boolean_literals(left: str, right: str) -> bool:
        def value(text: str) -> Optional[bool]:
            normalized = text.strip()
            if normalized == "true":
                return True
            if normalized == "false":
                return False
            if re.fullmatch(r"\+?1(?:[uUlL]+)?", normalized):
                return True
            if re.fullmatch(r"[+-]?0(?:[uUlL]+)?", normalized):
                return False
            return None

        left_value = value(left)
        right_value = value(right)
        return (
            left_value is not None
            and right_value is not None
            and left_value == right_value
        )

"""Shared Metal free-function specialization identity and selection."""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence


def _strip_metal_attribute_blocks(text: str) -> str:
    return re.sub(r"\[\[[^\]]*\]\]", " ", str(text or ""))


def _normalize_metal_type_text(type_text: str) -> str:
    # Strip C/C++ comments first: a type spelling never legitimately contains a
    # comment, but extraction can pick one up (e.g. a trailing `// ...` after a
    # return type). Leaving it in makes downstream token scans treat comment
    # words as identifiers (e.g. flagging "Get" as a missing template parameter).
    text = re.sub(r"/\*.*?\*/", " ", type_text, flags=re.DOTALL)
    text = re.sub(r"//[^\n]*", " ", text)
    text = _strip_metal_attribute_blocks(text)
    text = re.sub(r"\b(?:struct|class|typename)\s+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s*([<>,*&\[\]()])\s*", r"\1", text)
    return text.strip()


def _metal_declared_type_and_name(declaration: str) -> tuple[str, str] | None:
    cleaned = _strip_metal_attribute_blocks(declaration)
    cleaned = cleaned.split("=", 1)[0].strip()
    cleaned = cleaned.rstrip(";").strip()
    if not cleaned or cleaned == "void":
        return None
    name_match = re.search(
        r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*(?:\[[^\]]*\]\s*)*$",
        cleaned,
    )
    if name_match is None:
        return None
    name = name_match.group("name")
    type_text = cleaned[: name_match.start()].strip()
    if not type_text:
        return None
    array_suffix = cleaned[name_match.end("name") :].strip()
    if array_suffix:
        type_text = f"{type_text}{array_suffix}"
    return _normalize_metal_type_text(type_text), name


def _metal_function_parameter_declarations(
    preprocessor: Any, header: str
) -> list[tuple[str, str, bool]]:
    open_paren = preprocessor._function_parameter_start(header)
    if open_paren is None:
        return []
    close_paren = preprocessor._find_matching_delimiter(header, open_paren, "(", ")")
    if close_paren is None:
        return []

    declarations: list[tuple[str, str, bool]] = []
    for parameter in preprocessor._split_top_level_commas(
        header[open_paren + 1 : close_paren]
    ):
        parameter = parameter.strip()
        if not parameter or parameter == "void":
            continue
        variadic = "..." in parameter
        normalized = parameter.replace("...", " ")
        parsed = _metal_declared_type_and_name(normalized)
        if parsed is None:
            declaration, _default = preprocessor._split_top_level_assignment(normalized)
            unnamed_type = _strip_metal_attribute_blocks(declaration).strip()
            if not unnamed_type:
                continue
            declarations.append(
                (_normalize_metal_type_text(unnamed_type), "", variadic)
            )
            continue
        type_text, name = parsed
        declarations.append((type_text, name, variadic))
    return declarations


def _template_parameter_values_from_arguments(
    preprocessor: Any,
    template: Any,
    arguments: Sequence[str],
) -> dict[str, str]:
    substitutions, variadic_bindings = preprocessor._template_argument_bindings(
        template,
        list(arguments),
    )
    parameters = {name: str(value) for name, value in substitutions.items()}
    for name, values in variadic_bindings.items():
        parameters[name] = ", ".join(str(value) for value in values)
    return parameters


def _explicit_template_function_specialization_for_selected_overload(
    *,
    preprocessor: Any,
    explicit_specializations: Mapping[
        tuple[str, tuple[str, ...], tuple[str, ...]],
        Mapping[str, Any],
    ],
    template: Any,
    function_name: str,
    arguments: Sequence[str],
    parameter_declarations: Sequence[tuple[str, str, bool]],
    declaration_source: str,
    declaration_type_aliases: Mapping[str, Sequence[Any]],
    argument_alias_contexts: Sequence[
        tuple[Mapping[str, Sequence[Any]], int, str]
    ] = (),
) -> Mapping[str, Any] | None:
    """Select an explicit body only when its concrete overload is proven exact.

    Metal/C++ overload identity is based on canonical types, not the source
    spelling of a visible ``typedef``/``using`` alias.  Resolve each spelling at
    the position where it occurs: call arguments at their call site, the primary
    signature at its declaration, and every explicit signature at its own
    declaration.  An unresolved forward/cyclic alias must fail closed instead of
    making a valid explicit body look absent and silently selecting the primary.
    """
    from crosstl.backend.Metal.preprocessor import (
        MetalTemplateSpecializationError,
    )

    specialization_key = preprocessor._template_specialization_key(
        function_name,
        arguments,
    )
    requested_signature = preprocessor._template_specialization_signature(
        function_name,
        list(specialization_key[1]),
    )
    named_candidates = [
        specialization
        for (
            name,
            _template_arguments,
            _signature,
        ), specialization in explicit_specializations.items()
        if name == specialization_key[0]
    ]
    if not named_candidates:
        return None

    def source_location(source: str, position: int, text: str) -> Any:
        return preprocessor._source_location_for_offsets(
            source,
            position,
            min(len(source), position + max(len(text), 1)),
        )

    def canonical_type(
        type_text: str,
        *,
        aliases: Mapping[str, Sequence[Any]],
        position: int,
        source: str,
        context: str,
        location: Any = None,
        excluded_aliases: set[str] | None = None,
    ) -> str:
        canonical = preprocessor._canonicalize_type_aliases_at(
            type_text,
            aliases,
            position,
            excluded_aliases=excluded_aliases,
        )
        if canonical is not None:
            return _normalize_metal_type_text(canonical)

        suggested_action = (
            "declare every type alias before the specialization use, remove "
            "cyclic aliases, and keep each concrete overload signature unique"
        )
        raise MetalTemplateSpecializationError(
            "Metal explicit free-function specialization cannot be selected "
            f"safely for overload '{requested_signature}': {context} "
            f"'{_normalize_metal_type_text(type_text)}' has type aliases that "
            "cannot be resolved with declaration-order and lexical-scope "
            f"fidelity. Suggested action: {suggested_action}.",
            requested_signature=requested_signature,
            suggested_action=suggested_action,
            source_location=(
                location
                if location is not None
                else source_location(source, position, type_text)
            ),
            callee_template=function_name,
            requested_arguments=tuple(specialization_key[1]),
        )

    def canonical_template_arguments(
        values: Sequence[str],
        *,
        alias_contexts: Sequence[tuple[Mapping[str, Sequence[Any]], int, str]],
        context: str,
        location: Any = None,
    ) -> tuple[str, ...]:
        canonical = [
            preprocessor._normalize_template_argument_text(value) for value in values
        ]
        template_parameters = list(getattr(template, "template_parameters", ()) or ())
        non_type_parameters = set(
            (getattr(template, "template_parameter_types", {}) or {}).keys()
        )
        variadic_parameters = set(
            getattr(template, "variadic_template_parameters", ()) or ()
        )
        argument_index = 0
        for parameter_index, parameter in enumerate(template_parameters):
            if parameter in variadic_parameters:
                remaining_fixed = len(template_parameters) - parameter_index - 1
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
                    for aliases, position, source in alias_contexts:
                        canonical[index] = canonical_type(
                            canonical[index],
                            aliases=aliases,
                            position=position,
                            source=source,
                            context=context,
                            location=location,
                        )
            argument_index += argument_count
        return tuple(canonical)

    selected_alias_contexts = tuple(argument_alias_contexts) or (
        (
            declaration_type_aliases,
            int(template.span[0]),
            declaration_source,
        ),
    )
    selected_arguments = canonical_template_arguments(
        specialization_key[1],
        alias_contexts=selected_alias_contexts,
        context="selected template argument",
    )

    candidates: list[tuple[Mapping[str, Any], tuple[str, ...]]] = []
    for specialization in named_candidates:
        if not specialization["templateArgumentsExplicit"]:
            # Omitted or empty argument lists are deduced from the concrete
            # function type. Exact parameter-signature matching below decides
            # whether this selected primary owns the specialization body.
            candidates.append((specialization, ()))
            continue
        specialization_position = int(specialization["span"][0])
        candidate_arguments = canonical_template_arguments(
            specialization["arguments"],
            alias_contexts=(
                (
                    declaration_type_aliases,
                    specialization_position,
                    declaration_source,
                ),
            ),
            context="explicit specialization template argument",
            location=specialization.get("sourceLocation"),
        )
        if candidate_arguments == selected_arguments[: len(candidate_arguments)]:
            candidates.append((specialization, candidate_arguments))
    if not candidates:
        return None

    parameters = _template_parameter_values_from_arguments(
        preprocessor,
        template,
        list(selected_arguments),
    )

    def concrete_signature(
        declarations: Sequence[tuple[str, str, bool]],
        *,
        aliases: Mapping[str, Sequence[Any]],
        position: int,
        source: str,
        context: str,
        substitute: bool,
        location: Any = None,
    ) -> tuple[str, ...]:
        signature = []
        for type_text, _name, variadic in declarations:
            resolved = (
                preprocessor._replace_identifiers(type_text, parameters)
                if substitute
                else type_text
            )
            normalized = canonical_type(
                resolved,
                aliases=aliases,
                position=position,
                source=source,
                context=context,
                location=location,
                excluded_aliases=(
                    set(getattr(template, "template_parameters", ()) or ())
                    if substitute
                    else None
                ),
            )
            signature.append(f"{normalized}..." if variadic else normalized)
        return tuple(signature)

    template_position = int(template.span[0])
    selected_signature = concrete_signature(
        parameter_declarations,
        aliases=declaration_type_aliases,
        position=template_position,
        source=declaration_source,
        context="selected primary parameter type",
        substitute=True,
    )
    matches = []
    for specialization, candidate_arguments in candidates:
        specialization_declarations = _metal_function_parameter_declarations(
            preprocessor,
            str(specialization["header"]),
        )
        specialization_signature = concrete_signature(
            specialization_declarations,
            aliases=declaration_type_aliases,
            position=int(specialization["span"][0]),
            source=declaration_source,
            context="explicit specialization parameter type",
            substitute=False,
            location=specialization.get("sourceLocation"),
        )
        if selected_signature != specialization_signature:
            continue
        if len(candidate_arguments) < len(selected_arguments):
            # A missing template argument is deduced from the concrete signature
            # before a primary default is used. Signature equality alone cannot
            # distinguish non-type parameters used only inside the function body.
            method = preprocessor._template_function_method_signature(
                template, preprocessor._template_function_parameter_text(template)
            )
            inferred = preprocessor._bind_template_method_parameters(
                method,
                list(specialization_signature),
                explicit_template_arguments=list(candidate_arguments),
                argument_type_aliases=declaration_type_aliases,
                argument_type_position=int(specialization["span"][0]),
            )
            if inferred is None:
                raise MetalTemplateSpecializationError(
                    "Metal explicit free-function specialization cannot be selected "
                    f"safely for overload '{requested_signature}': omitted template "
                    "arguments could not be deduced from its concrete signature",
                    requested_signature=requested_signature,
                    suggested_action="spell every explicit specialization template argument",
                    source_location=specialization.get("sourceLocation"),
                    callee_template=function_name,
                    requested_arguments=tuple(selected_arguments),
                )
            inferred_arguments = canonical_template_arguments(
                [inferred[name] for name in template.template_parameters],
                alias_contexts=(
                    (declaration_type_aliases, template_position, declaration_source),
                ),
                context="deduced explicit specialization template argument",
                location=specialization.get("sourceLocation"),
            )
            if inferred_arguments != selected_arguments:
                continue
        matches.append(specialization)

    if not matches:
        # Another overload alone may be explicitly specialized for the same
        # canonical template arguments. It must not replace or invalidate this
        # overload; materialize the selected primary body instead.
        return None

    if len(matches) > 1:
        raise MetalTemplateSpecializationError(
            "Metal explicit free-function specialization cannot be selected safely "
            f"for overload '{requested_signature}' with canonical parameter "
            f"signature {selected_signature!r}: more than one explicit body matches",
            requested_signature=requested_signature,
            suggested_action=(
                "keep one exact explicit specialization for each canonical "
                "concrete overload signature"
            ),
            source_location=matches[0].get("sourceLocation"),
            callee_template=function_name,
            requested_arguments=tuple(selected_arguments),
        )

    specialization = matches[0]
    specialization_position = int(specialization["span"][0])
    specialization_source = str(specialization["source"])
    specialization_header = str(specialization["header"])
    # The scanner masks comments/literals without changing length, so the raw
    # source header ends at the same offset as the stored masked header.
    raw_header = specialization_source[: len(specialization_header)]
    name_start, name_end = (int(value) for value in specialization["nameSpan"])
    open_paren = preprocessor._function_parameter_start(raw_header)
    close_paren = (
        preprocessor._find_matching_delimiter(raw_header, open_paren, "(", ")")
        if open_paren is not None
        else None
    )
    if open_paren is None or close_paren is None:
        suggested_action = (
            "keep the selected explicit specialization as a concrete function "
            "declaration with a parseable parameter list"
        )
        raise MetalTemplateSpecializationError(
            "Metal explicit free-function specialization cannot be materialized "
            f"safely for overload '{requested_signature}': its concrete "
            f"parameter list cannot be reconstructed. Suggested action: "
            f"{suggested_action}.",
            requested_signature=requested_signature,
            suggested_action=suggested_action,
            source_location=specialization.get("sourceLocation"),
            callee_template=function_name,
            requested_arguments=tuple(selected_arguments),
        )

    def canonical_alias_fragment(fragment: str, context: str) -> str:
        canonical = preprocessor._canonicalize_type_aliases_at(
            fragment,
            declaration_type_aliases,
            specialization_position,
        )
        if canonical is None:
            # Reuse the structured alias failure contract above.
            canonical_type(
                fragment,
                aliases=declaration_type_aliases,
                position=specialization_position,
                source=declaration_source,
                context=context,
                location=specialization.get("sourceLocation"),
            )
            raise AssertionError("unreachable")
        return canonical

    return_prefix = canonical_alias_fragment(
        raw_header[:name_start],
        "explicit specialization return type",
    ).rstrip()
    if return_prefix:
        return_prefix += " "

    raw_parameters = preprocessor._split_top_level_commas(
        raw_header[open_paren + 1 : close_paren]
    )
    parsed_parameters = _metal_function_parameter_declarations(
        preprocessor,
        raw_header,
    )
    concrete_parameters = [
        parameter for parameter in raw_parameters if parameter.strip() != "void"
    ]
    if len(concrete_parameters) != len(parsed_parameters):
        suggested_action = (
            "use ordinary named or unnamed concrete parameters in the explicit "
            "specialization"
        )
        raise MetalTemplateSpecializationError(
            "Metal explicit free-function specialization cannot be materialized "
            f"safely for overload '{requested_signature}': its parameter "
            f"declarators are ambiguous. Suggested action: {suggested_action}.",
            requested_signature=requested_signature,
            suggested_action=suggested_action,
            source_location=specialization.get("sourceLocation"),
            callee_template=function_name,
            requested_arguments=tuple(selected_arguments),
        )

    canonical_parameters: list[str] = []
    parsed_index = 0
    for raw_parameter in raw_parameters:
        if raw_parameter.strip() == "void":
            canonical_parameters.append("void")
            continue
        _type_text, parameter_name, _variadic = parsed_parameters[parsed_index]
        parsed_index += 1
        if not parameter_name:
            canonical_parameters.append(
                canonical_alias_fragment(
                    raw_parameter,
                    "explicit specialization parameter type",
                )
            )
            continue
        name_matches = list(
            re.finditer(rf"\b{re.escape(parameter_name)}\b", raw_parameter)
        )
        if not name_matches:
            suggested_action = (
                "use a concrete parameter declarator whose name can be retained "
                "during alias canonicalization"
            )
            raise MetalTemplateSpecializationError(
                "Metal explicit free-function specialization cannot be "
                f"materialized safely for overload '{requested_signature}': "
                f"parameter '{parameter_name}' cannot be located in its "
                f"declarator. Suggested action: {suggested_action}.",
                requested_signature=requested_signature,
                suggested_action=suggested_action,
                source_location=specialization.get("sourceLocation"),
                callee_template=function_name,
                requested_arguments=tuple(selected_arguments),
            )
        name_match = name_matches[-1]
        canonical_prefix = canonical_alias_fragment(
            raw_parameter[: name_match.start()],
            "explicit specialization parameter type",
        ).rstrip()
        declarator_suffix = raw_parameter[name_match.start() :].lstrip()
        canonical_parameters.append(f"{canonical_prefix} {declarator_suffix}".strip())

    canonical_header = (
        return_prefix
        + raw_header[name_start:name_end]
        + raw_header[name_end : open_paren + 1]
        + ", ".join(canonical_parameters)
        + raw_header[close_paren:]
    )
    canonical_specialization = dict(specialization)
    canonical_specialization["header"] = canonical_header
    canonical_specialization["source"] = (
        canonical_header + specialization_source[len(raw_header) :]
    )
    canonical_specialization["nameSpan"] = (
        len(return_prefix),
        len(return_prefix) + name_end - name_start,
    )
    canonical_specialization["parameterTypes"] = tuple(
        value[:-3] if value.endswith("...") else value for value in selected_signature
    )
    return canonical_specialization

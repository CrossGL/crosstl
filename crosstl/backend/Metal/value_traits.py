"""Boolean ``value`` members of struct templates, read as constraint predicates.

Metal code commonly spells a type trait as a struct template with a Boolean ``value`` member and partial or full
specializations, for example::

    template <typename T> struct is_wrapped { static constexpr bool value = false; };
    template <typename T> struct is_wrapped<wrapper<T>> { static constexpr bool value = true; };

and then constrains a specialization with ``enable_if_t<is_wrapped<R>::value>``, directly or through a variable
template. Each declaration found here is recorded in the same form as a Boolean variable template, with
``member="value"``, so constraint evaluation selects among the primary and its specializations with the rules it
already applies to variable templates. A declaration whose ``value`` cannot be read is not recorded, so evaluating
it stays unrecognized and the caller refuses the specialization.
"""

import re
from typing import Dict, List, Optional, Sequence, Tuple

_STRUCT_HEAD = re.compile(r"\s*(?:struct|class)\s+(?P<name>[A-Za-z_]\w*)\b")
_VALUE_MEMBER = re.compile(
    r"(?:\b(?:static|constexpr|constant|const|inline)\s+)+bool\s+value\s*=\s*"
    r"(?P<expression>[^;{}]+);"
)
_STATIC_SPECIFIER = re.compile(r"\bstatic\b")
_BOOL_BASE = re.compile(
    r"(?:public\s+)?(?:::)?(?:metal\s*::\s*)?(?P<base>true_type|false_type)"
)
_BOOL_CONSTANT_BASE = re.compile(
    r"(?:public\s+)?(?:::)?(?:metal\s*::\s*)?"
    r"(?:bool_constant\s*<(?P<bool>.*)>|integral_constant\s*<\s*bool\s*,(?P<integral>.*)>)",
    re.DOTALL,
)
_CACHE_LIMIT = 8


def _top_level_text(body: str) -> str:
    """``body`` with every nested brace group blanked, so only direct members remain."""
    parts: List[str] = []
    depth = 0
    for character in body:
        if character == "{":
            depth += 1
            parts.append(" ")
        elif character == "}":
            depth = max(depth - 1, 0)
            parts.append(" ")
        else:
            parts.append(character if depth == 0 else " ")
    return "".join(parts)


def _value_expression(body: str) -> Optional[str]:
    matches = [
        match
        for match in _VALUE_MEMBER.finditer(_top_level_text(body))
        if _STATIC_SPECIFIER.search(match.group(0))
    ]
    if len(matches) != 1:
        return None
    return matches[0].group("expression").strip() or None


def _base_expression(preprocessor, base_clause: str) -> Optional[str]:
    bases = preprocessor._split_top_level_commas(base_clause)
    if len(bases) != 1:
        return None
    base = bases[0].strip()
    match = _BOOL_BASE.fullmatch(base)
    if match is not None:
        return "true" if match.group("base") == "true_type" else "false"
    match = _BOOL_CONSTANT_BASE.fullmatch(base)
    if match is not None:
        expression = (match.group("bool") or match.group("integral") or "").strip()
        return f"({expression})" if expression else None
    base = re.sub(r"^(?:public|private|protected)\s+", "", base).strip()
    if re.fullmatch(r"(?:::)?[A-Za-z_][\w:]*\s*<.*>", base, re.DOTALL):
        # A trait derived from another trait inherits its ``value``.
        return f"{base}::value"
    return None


def _scan(preprocessor, code: str, definition_type) -> Dict[str, List[object]]:
    definitions: Dict[str, List[object]] = {}
    ignored = preprocessor._find_comment_and_literal_spans(code)
    namespace_spans = preprocessor._find_namespace_spans(code)
    # Member templates are not namespace-scope traits.
    aggregate_spans = sorted(
        [struct.span for struct in preprocessor._find_concrete_struct_definitions(code)]
        + [struct.span for struct in preprocessor._find_template_structs(code)]
    )
    cursor = 0
    while True:
        match = re.search(r"\btemplate\s*<", code[cursor:])
        if match is None:
            break
        template_start = cursor + match.start()
        cursor = template_start + len("template")
        if preprocessor._containing_span(template_start, ignored) is not None:
            continue
        if any(start < template_start < end for start, end in aggregate_spans):
            continue
        angle_start = code.find("<", template_start)
        angle_end = preprocessor._find_matching_template_param_angle(code, angle_start)
        if angle_end is None:
            continue
        head = _STRUCT_HEAD.match(code, angle_end + 1)
        if head is None:
            continue
        position = head.end()
        while position < len(code) and code[position].isspace():
            position += 1
        argument_patterns: Optional[Tuple[str, ...]] = None
        parameter_text = code[angle_start + 1 : angle_end]
        parameters = tuple(preprocessor._template_parameter_names(parameter_text))
        if position < len(code) and code[position] == "<":
            argument_end = preprocessor._find_matching_angle(code, position)
            if argument_end is None:
                continue
            argument_patterns = tuple(
                preprocessor._split_top_level_commas(code[position + 1 : argument_end])
            )
            position = argument_end + 1
        body_start = code.find("{", position)
        semicolon = code.find(";", position)
        if body_start == -1 or (semicolon != -1 and semicolon < body_start):
            continue
        body_end = preprocessor._find_matching_brace(code, body_start)
        if body_end is None:
            continue
        cursor = body_end
        base_clause = code[position:body_start].strip()
        if base_clause and not base_clause.startswith(":"):
            continue
        expression = _value_expression(code[body_start + 1 : body_end - 1])
        if expression is None and base_clause:
            expression = _base_expression(preprocessor, base_clause[1:])
        if expression is None:
            continue
        patterns = argument_patterns if argument_patterns is not None else parameters
        if not patterns:
            continue
        name = head.group("name")
        definitions.setdefault(name, []).append(
            definition_type(
                name=name,
                template_parameters=parameters,
                argument_patterns=patterns,
                expression=expression,
                is_partial_specialization=argument_patterns is not None,
                span=(template_start, body_end),
                namespace=preprocessor._namespace_at(namespace_spans, template_start),
                member="value",
                parameter_slots=tuple(
                    (parameter.name or "", parameter.default)
                    for parameter in preprocessor._parse_template_parameter_list(
                        parameter_text
                    )
                ),
            )
        )
    return definitions


def struct_value_trait_definitions(
    preprocessor, code: str, definition_type
) -> Dict[str, List[object]]:
    """Struct templates of ``code`` with a Boolean ``value``, keyed by name, in source order.

    ``definition_type`` is the Boolean variable-template record type; each entry carries ``member="value"``.
    """
    cache = preprocessor.__dict__.setdefault("_struct_value_trait_cache", {})
    definitions = cache.pop(code, None)
    if definitions is None:
        definitions = _scan(preprocessor, code, definition_type)
    cache[code] = definitions
    if len(cache) > _CACHE_LIMIT:
        del cache[next(iter(cache))]
    return {name: list(entries) for name, entries in definitions.items()}


_ENABLE_IF_START = re.compile(
    r"\s*(?:typename\s+)?(?:(?:::)?metal\s*::\s*)?enable_if(?P<alias>_t)?\s*<"
)


def enable_if_parts(preprocessor, pattern: str) -> Optional[Tuple[str, str]]:
    """The condition and result type of an ``enable_if`` specialization argument, or None for any other argument."""
    match = _ENABLE_IF_START.match(pattern)
    if match is None:
        return None
    start = match.end() - 1
    end = preprocessor._find_matching_template_param_angle(pattern, start)
    if end is None:
        return None
    suffix = pattern[end + 1 :].strip()
    if match.group("alias") and suffix:
        return None
    if not match.group("alias") and re.fullmatch(r"::\s*type", suffix) is None:
        return None
    arguments = preprocessor._split_template_parameter_list(pattern[start + 1 : end])
    if not 1 <= len(arguments) <= 2:
        return None
    result = arguments[1].strip() if len(arguments) == 2 else "void"
    return arguments[0].strip(), result


def complete_template_arguments(
    preprocessor, primary, arguments: Sequence[str]
) -> Optional[Tuple[str, ...]]:
    """``arguments`` followed by the primary's defaults for the parameters they omit.

    Returns ``arguments`` unchanged when the primary records no parameter list, and None when an omitted parameter
    has no default or there are too many arguments.
    """
    slots = getattr(primary, "parameter_slots", ())
    if not slots:
        return tuple(arguments)
    if len(arguments) > len(slots):
        return None
    substitutions = {
        name: argument for (name, _default), argument in zip(slots, arguments) if name
    }
    completed = list(arguments)
    for name, default in slots[len(arguments) :]:
        if default is None:
            return None
        value = preprocessor._normalize_template_argument_text(
            preprocessor._replace_identifiers(default, substitutions)
        )
        if name:
            substitutions[name] = value
        completed.append(value)
    return tuple(completed)


# metal_simdgroup: ``__is_valid_simdgroup_type<T>`` holds when ``make_scalar_t<T>`` is one of these scalars (double
# only with native doubles, which Apple GPUs lack) and T is a scalar or a 2-, 3- or 4-element vector.
_SIMDGROUP_SCALARS = frozenset(
    {
        "half",
        "float",
        "char",
        "uchar",
        "short",
        "ushort",
        "int",
        "uint",
        "int8_t",
        "uint8_t",
        "int16_t",
        "uint16_t",
        "int32_t",
        "uint32_t",
    }
)
_SIMDGROUP_TRAIT = re.compile(
    r"(?:(?:::)?metal\s*::\s*)?__is_valid_simdgroup_type\s*<(?P<operand>.*)>\s*::\s*value",
    re.DOTALL,
)


def simdgroup_type_trait(
    preprocessor, expression: str, resolve, struct_names
) -> Optional[Tuple[Optional[bool]]]:
    """Evaluate Metal's internal ``__is_valid_simdgroup_type<T>::value``.

    Returns None when ``expression`` is not that trait, otherwise a one-element tuple holding the value, or None as
    the value when the operand's type is not known well enough to decide. ``resolve`` canonicalizes a type operand.
    """
    match = _SIMDGROUP_TRAIT.fullmatch(expression.strip())
    if match is None:
        return None
    operand = resolve(match.group("operand"))
    if operand is None:
        return (None,)
    operand = re.sub(r"^(?:const|volatile)\s+|\s+(?:const|volatile)$", "", operand)
    packed = re.fullmatch(r"packed_([a-z]+)([234])", operand)
    scalar = (
        (packed.group(1), int(packed.group(2)))
        if packed is not None
        else preprocessor._scalar_and_width(operand)
    )
    if scalar is not None:
        base, width = scalar
        if width in (8, 16):
            return (None,)
        return (base in _SIMDGROUP_SCALARS and width in (1, 2, 3, 4),)
    if (
        re.fullmatch(r"[A-Za-z_][\w:]*\s*<.*>", operand, re.DOTALL)
        or re.fullmatch(r"(?:half|float|bfloat)[234]x[234]", operand)
        or operand in struct_names
    ):
        # A struct or matrix is its own make_scalar_t, which is none of the scalars.
        return (False,)
    return (None,)

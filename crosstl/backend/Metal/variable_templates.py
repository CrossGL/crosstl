"""Materialize Metal value variable templates at their concrete uses.

``template <typename T> constexpr constant uint packing_size = sizeof(uint) / sizeof(T);`` is
read as ``packing_size<uchar>`` once the enclosing template is concrete. CrossGL has no variable
templates, so each concrete use becomes the definition's initializer with the template arguments
substituted: a folded literal of the declared type when the initializer is an integral constant
expression, and otherwise the initializer converted to the declared type.

Boolean variable templates keep their own constraint machinery. A name with a partial or explicit
specialization, or with more than one primary definition, is left untouched for the existing
fail-closed paths, as is every use that still sits inside an unmaterialized template.
"""

import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

_DEFINITION = re.compile(
    r"\s*(?:static\s+)?constexpr\s+(?:static\s+)?(?:constant\s+)?(?:const\s+)?"
    r"(?P<type>(?:unsigned\s+(?:int|long|short|char)|unsigned|[A-Za-z_]\w*))\s+"
    r"(?P<name>[A-Za-z_]\w*)\s*(?P<partial><)?"
)

# Declared type -> literal suffix for a folded integral value.
_LITERAL_SUFFIX = {
    "uint": "u",
    "unsigned": "u",
    "unsigned int": "u",
    "uint32_t": "u",
    "int": "",
    "int32_t": "",
}
_INTEGRAL_TYPES = {
    "char",
    "uchar",
    "short",
    "ushort",
    "long",
    "ulong",
    "size_t",
    "ptrdiff_t",
    "int8_t",
    "uint8_t",
    "int16_t",
    "uint16_t",
    "int64_t",
    "uint64_t",
    "unsigned char",
    "unsigned short",
    "unsigned long",
}
_SPELLING = {
    "unsigned": "uint",
    "unsigned int": "uint",
    "unsigned char": "uchar",
    "unsigned short": "ushort",
    "unsigned long": "ulong",
}


@dataclass(frozen=True)
class ValueVariableTemplate:
    name: str
    value_type: str
    parameters: Tuple[str, ...]
    expression: str
    span: Tuple[int, int]


def find_value_variable_templates(preprocessor, code: str):
    """Return the primary value variable templates that have one definition."""
    ignored = preprocessor._find_comment_and_literal_spans(code)
    definitions: Dict[str, List[ValueVariableTemplate]] = {}
    specialized = set()
    for match in re.finditer(r"\btemplate\s*<", code):
        start = match.start()
        if preprocessor._containing_span(start, ignored) is not None:
            continue
        angle_start = code.find("<", start)
        angle_end = preprocessor._find_matching_template_param_angle(code, angle_start)
        if angle_end is None:
            continue
        header = _DEFINITION.match(code, angle_end + 1)
        if header is None:
            continue
        value_type = " ".join(header.group("type").split())
        name = header.group("name")
        if header.group("partial"):
            specialized.add(name)
            continue
        if value_type in {"bool", "struct", "class", "union", "enum", "auto"}:
            continue
        equals = header.end()
        while equals < len(code) and code[equals].isspace():
            equals += 1
        if equals >= len(code) or code[equals] != "=":
            continue
        semicolon = code.find(";", equals)
        if semicolon == -1:
            continue
        expression = code[equals + 1 : semicolon].strip()
        parameters = tuple(
            preprocessor._template_parameter_names(code[angle_start + 1 : angle_end])
        )
        if not expression or not parameters:
            continue
        definitions.setdefault(name, []).append(
            ValueVariableTemplate(
                name=name,
                value_type=value_type,
                parameters=parameters,
                expression=expression,
                span=(start, semicolon + 1),
            )
        )
    for match in re.finditer(r"\btemplate\s*<\s*>", code):
        header = _DEFINITION.match(code, match.end())
        if header is not None and header.group("partial"):
            specialized.add(header.group("name"))
    return {
        name: candidates[0]
        for name, candidates in definitions.items()
        if len(candidates) == 1 and name not in specialized
    }


def _render_value(
    preprocessor, definition: ValueVariableTemplate, arguments, array_bound=False
):
    expression = preprocessor._replace_identifiers(
        definition.expression, dict(zip(definition.parameters, arguments))
    )
    value_type = _SPELLING.get(definition.value_type, definition.value_type)
    integral = (
        definition.value_type in _LITERAL_SUFFIX
        or definition.value_type in _INTEGRAL_TYPES
    )
    if integral:
        ok, value = preprocessor._evaluate_static_integral_expression(expression)
        if ok and isinstance(value, int) and not isinstance(value, bool):
            if array_bound and value > 0:
                # An array bound converts to the size type; only its value matters.
                return str(value)
            suffix = _LITERAL_SUFFIX.get(definition.value_type)
            if suffix is not None and (suffix == "" or value >= 0):
                literal = f"{value}{suffix}"
                return f"({literal})" if value < 0 else literal
            return f"{value_type}({value})"
    return f"{value_type}({expression})"


def materialize_value_variable_templates(preprocessor, code: str) -> str:
    templates = find_value_variable_templates(preprocessor, code)
    if not templates:
        return code
    ignored = preprocessor._find_comment_and_literal_spans(code)
    template_spans = preprocessor._find_template_declaration_spans(code)
    pattern = re.compile(
        r"(?<![\w.])(?:(?:[A-Za-z_]\w*)?\s*::\s*)*\b(?P<name>"
        + "|".join(re.escape(name) for name in sorted(templates))
        + r")\s*<"
    )
    replacements = []
    for match in pattern.finditer(code):
        start = match.start()
        if preprocessor._containing_span(start, ignored) is not None:
            continue
        if preprocessor._containing_span(start, template_spans) is not None:
            continue
        definition = templates[match.group("name")]
        angle_start = match.end() - 1
        angle_end = preprocessor._find_matching_angle(code, angle_start)
        if angle_end is None:
            continue
        arguments = [
            preprocessor._normalize_template_argument_text(argument)
            for argument in preprocessor._split_top_level_commas(
                code[angle_start + 1 : angle_end]
            )
        ]
        if len(arguments) != len(definition.parameters) or not all(arguments):
            continue
        before = code[:start].rstrip()
        after = code[angle_end + 1 :].lstrip()
        array_bound = before.endswith("[") and after.startswith("]")
        replacements.append(
            (
                start,
                angle_end + 1,
                _render_value(preprocessor, definition, arguments, array_bound),
            )
        )
    if not replacements:
        return code
    return preprocessor._apply_text_replacements(code, replacements)

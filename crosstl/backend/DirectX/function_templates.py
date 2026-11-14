"""HLSL 2021 function templates imported as CrossGL generic functions.

``template <typename T> T twice(T x)`` is CrossGL's ``generic<T> T twice(T x)``: the parser keeps the template
parameter list on the function, and the importer emits it as the generic prefix. ``template <> float
twice<float>(float x)`` (and ``template <> float twice(float x)``, whose arguments are deduced from its signature)
replaces the body of the primary for ``T = float`` ([temp.expl.spec]/1); it becomes a concrete function marked
``@ explicit_specialization(twice, float)``, so every instantiation of ``twice`` for ``float`` runs its body. Calls
that spell template arguments (``twice<float2>(v)``) keep them in CrossGL spelling (``twice<vec2>(v)``).

A template value parameter (``template <typename T, uint N>``) is a CrossGL generic parameter that each call binds to
a value: ``sum<uint, 5>(x)`` imports as ``sum<uint, 5u>(x)``, the literal typed as the parameter is (see
:func:`template_argument`). CrossGL generic parameters have no value defaults, so a function template that gives a
value parameter one is refused.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from crosstl.translator.refusals import DirectXSyntaxError

from .namespaces import source_name

__all__ = [
    "TemplateParameter",
    "begin_template_declaration",
    "expire_template_declaration",
    "explicit_specialization_annotation",
    "parse_specialization_declarator_arguments",
    "skip_specialization_declarator_arguments_at",
    "take_template_parameters",
    "template_call_result_raw_type",
    "template_function_generic_prefix",
    "template_type_parameter_names",
    "templated_call_name",
]

_TYPE_PARAMETER_KEYWORDS = frozenset({"typename", "class"})
_TEMPLATED_NAME = re.compile(r"^(?P<base>[^<>]+?)\s*<(?P<arguments>.*)>$")
_INTEGER_ARGUMENT = re.compile(r"^[+-]?\d+[uUlL]*$")
_TEMPLATE_VECTOR = re.compile(
    r"^vector\s*<\s*(?P<element>\w+)\s*,\s*(?P<size>[1-4])\s*>$"
)
_HLSL_VECTOR = re.compile(r"^(?P<element>[A-Za-z_]\w*?)(?P<size>[1-4])$")


@dataclass(frozen=True)
class TemplateParameter:
    """One parameter of a ``template <...>`` list: ``kind`` is ``typename`` or the value parameter's type."""

    kind: str
    name: str | None
    default: str | None = None

    @property
    def is_type(self):
        return self.kind == "typename"


# Parsing ------------------------------------------------------------------------------------------------------


def _collect_argument_tokens(parser):
    """The tokens of one template argument, up to the ``,`` or ``>`` that ends it at nesting depth zero."""
    tokens = []
    depth = 0
    while parser.current_token[0] != "EOF":
        token_type = parser.current_token[0]
        if depth == 0 and token_type in {"COMMA", "GREATER_THAN"}:
            return tokens
        if token_type in {"LESS_THAN", "LPAREN", "LBRACKET"}:
            depth += 1
        elif token_type in {"GREATER_THAN", "RPAREN", "RBRACKET"}:
            depth -= 1
        elif token_type == "SHIFT_RIGHT":
            if depth < 2:
                raise DirectXSyntaxError(
                    "Unbalanced '>>' in a template parameter default"
                )
            depth -= 2
        tokens.append(parser.current_token)
        parser.eat(token_type)
    raise DirectXSyntaxError("Unterminated template parameter list")


def parse_template_parameter_list(parser):
    """Parse ``template <...>`` at the current token into :class:`TemplateParameter` entries."""
    parser.eat("IDENTIFIER")
    parser.eat("LESS_THAN")
    parameters = []
    while parser.current_token[0] not in {"GREATER_THAN", "EOF"}:
        token_type, value = parser.current_token[0], parser.current_token[1]
        if token_type == "IDENTIFIER" and value in _TYPE_PARAMETER_KEYWORDS:
            parser.eat("IDENTIFIER")
            kind = "typename"
        else:
            kind = str(parser.parse_type())
        name = None
        if parser.current_token[0] == "IDENTIFIER":
            name = parser.current_token[1]
            parser.eat("IDENTIFIER")
        default = None
        if parser.current_token[0] == "EQUALS":
            parser.eat("EQUALS")
            default = parser.format_generic_argument_tokens(
                _collect_argument_tokens(parser)
            )
        parameters.append(TemplateParameter(kind, name, default))
        if parser.current_token[0] != "COMMA":
            break
        parser.eat("COMMA")
    parser.eat("GREATER_THAN")
    return parameters


def begin_template_declaration(parser):
    """Consume a top-level ``template <...>`` prefix and keep its parameters for the declaration after it."""
    parser.pending_template_parameters = parse_template_parameter_list(parser)
    parser.pending_template_fresh = True


def expire_template_declaration(parser):
    """Drop a template prefix that the declaration after it did not take (one parser iteration later)."""
    if getattr(parser, "pending_template_parameters", None) is None:
        return
    if getattr(parser, "pending_template_fresh", False):
        parser.pending_template_fresh = False
        return
    parser.pending_template_parameters = None


def take_template_parameters(parser):
    """The pending template parameters for the function being parsed (``[]`` for ``template <>``), else ``None``."""
    parameters = getattr(parser, "pending_template_parameters", None)
    parser.pending_template_parameters = None
    parser.pending_template_fresh = False
    return parameters


def _explicit_specialization_pending(parser):
    return getattr(parser, "pending_template_parameters", None) == []


def skip_specialization_declarator_arguments_at(parser, index):
    """Skip ``<float>`` after the declarator of ``template <> float twice<float>(...)``; other indices pass."""
    if not _explicit_specialization_pending(parser):
        return index
    start = index
    if index >= len(parser.tokens) or parser.tokens[index][0] != "LESS_THAN":
        return start
    depth = 0
    while index < len(parser.tokens):
        token_type = parser.tokens[index][0]
        if token_type == "LESS_THAN":
            depth += 1
        elif token_type == "GREATER_THAN":
            depth -= 1
        elif token_type == "SHIFT_RIGHT":
            depth -= 2
        elif token_type in {"SEMICOLON", "LBRACE", "EOF"}:
            return start
        index += 1
        if depth <= 0:
            break
    # Only a function declarator (``twice<float>(``) spells specialization arguments.
    if index < len(parser.tokens) and parser.tokens[index][0] == "LPAREN":
        return index
    return start


def parse_specialization_declarator_arguments(parser, name):
    """``twice<float>`` for the declarator of an explicit specialization that spells its arguments."""
    if _explicit_specialization_pending(parser) and parser.current_token[0] == (
        "LESS_THAN"
    ):
        return parser.format_templated_name(name, parser.parse_generic_arguments())
    return name


# Emission -----------------------------------------------------------------------------------------------------


def _split_template_arguments(text):
    arguments, depth, current = [], 0, []
    for character in text:
        if character in "<([":
            depth += 1
        elif character in ">)]":
            depth -= 1
        if character == "," and depth == 0:
            arguments.append("".join(current).strip())
            current = []
            continue
        current.append(character)
    if "".join(current).strip():
        arguments.append("".join(current).strip())
    return arguments


def _split_templated_name(name):
    match = _TEMPLATED_NAME.match(str(name or "").strip())
    if match is None:
        return None, None
    return match.group("base").strip(), _split_template_arguments(
        match.group("arguments")
    )


def _template_parameters(func):
    return getattr(func, "template_parameters", None)


def template_type_parameter_names(func):
    """The names of ``func``'s template type parameters, for the types spelled in its signature and body."""
    return frozenset(
        parameter.name
        for parameter in _template_parameters(func) or []
        if parameter.is_type and parameter.name
    )


def _is_primary_template(func):
    return bool(_template_parameters(func))


def _is_explicit_specialization(func):
    return _template_parameters(func) == []


def _primary_templates(codegen, base):
    return [
        func
        for func in getattr(codegen, "source_functions", {}).get(base, [])
        if _is_primary_template(func)
    ]


def _crossgl_type_argument(codegen, argument):
    argument = str(argument).strip()
    if _INTEGER_ARGUMENT.match(argument):
        return argument
    return str(codegen.map_type(argument)).strip()


_UNSIGNED_VALUE_KINDS = frozenset(
    {"uint", "unsigned", "unsigned int", "uint32_t", "dword", "uint64_t", "uint16_t"}
)


def template_argument(codegen, parameter, argument):
    """The CrossGL spelling of one template argument for ``parameter``.

    A type argument takes its CrossGL type (``vec2`` for ``float2``). A value argument keeps its expression, and an
    integer literal for an unsigned parameter takes the ``u`` suffix, so the value CrossGL substitutes has the
    parameter's type (``3u`` for ``uint N``), as the HLSL template parameter converts it (C++17 17.3.2).
    """
    argument = str(argument).strip()
    if parameter is None or parameter.is_type:
        return _crossgl_type_argument(codegen, argument)
    kind = " ".join(str(parameter.kind).split())
    literal = re.fullmatch(r"([+-]?\d+|0[xX][0-9A-Fa-f]+)[uUlL]*", argument)
    if literal is not None:
        digits = literal.group(1)
        return f"{digits}u" if kind in _UNSIGNED_VALUE_KINDS else digits
    return argument


def template_value_parameter_names(func):
    """The names of ``func``'s template value parameters, which its body reads as values."""
    return frozenset(
        parameter.name
        for parameter in _template_parameters(func) or []
        if not parameter.is_type and parameter.name
    )


def template_function_generic_prefix(codegen, func):
    """``generic<T, U = vec2> `` for a function template, ``""`` for any other function."""
    parameters = _template_parameters(func)
    if not parameters:
        return ""
    rendered = []
    for parameter in parameters:
        if not parameter.is_type and parameter.default is not None:
            return codegen.refuse_import(
                f"HLSL function template '{source_name(func)}' gives the value parameter "
                f"'{parameter.kind} {parameter.name}' a default, which CrossGL generic functions cannot express"
            )
        if not parameter.is_type:
            # A value parameter is substituted by the value each call names (``scale<float, 3u>``).
            rendered.append(parameter.name)
            continue
        if not parameter.name:
            return codegen.refuse_import(
                f"HLSL function template '{func.name}' declares an unnamed type parameter"
            )
        text = parameter.name
        if parameter.default is not None:
            text += f" = {_crossgl_type_argument(codegen, parameter.default)}"
        rendered.append(text)
    return f"generic<{', '.join(rendered)}> "


def _type_base(codegen, type_value):
    return str(codegen.raw_type_base(type_value) or "").strip()


def _deduced_specialization_arguments(codegen, primary, func):
    """Template arguments of ``template <> R f(params)`` deduced from its signature against ``primary``'s."""
    names = [parameter.name for parameter in _template_parameters(primary)]
    bindings = {}
    pairs = [(primary.return_type, func.return_type)]
    primary_params = list(getattr(primary, "params", []) or [])
    params = list(getattr(func, "params", []) or [])
    if len(primary_params) != len(params):
        return None
    pairs.extend(
        (expected.vtype, actual.vtype)
        for expected, actual in zip(primary_params, params)
    )
    for expected, actual in pairs:
        expected_name = _type_base(codegen, expected)
        if expected_name in names:
            actual_name = _type_base(codegen, actual)
            if bindings.setdefault(expected_name, actual_name) != actual_name:
                return None
    defaults = {
        parameter.name: parameter.default for parameter in _template_parameters(primary)
    }
    arguments = []
    for name in names:
        argument = bindings.get(name, defaults.get(name))
        if argument is None:
            return None
        arguments.append(argument)
    return arguments


def _specialization_primary_and_arguments(codegen, func):
    base, arguments = _split_templated_name(func.name)
    if base is None:
        base = func.name
    primaries = _primary_templates(codegen, base)
    if len(primaries) != 1:
        return base, None, None
    primary = primaries[0]
    if arguments is None:
        arguments = _deduced_specialization_arguments(codegen, primary, func)
    return base, primary, arguments


def explicit_specialization_annotation(codegen, func):
    """`` @ explicit_specialization(twice, float)`` for ``template <> ... twice<float>(...)``, else ``""``."""
    if not _is_explicit_specialization(func):
        return ""
    base, primary, arguments = _specialization_primary_and_arguments(codegen, func)
    if primary is None:
        return codegen.refuse_import(
            f"HLSL explicit specialization '{func.name}' does not name exactly one function template"
        )
    if getattr(primary, "is_prototype", False) and not any(
        not getattr(candidate, "is_prototype", False)
        for candidate in _primary_templates(codegen, base)
    ):
        return codegen.refuse_import(
            f"HLSL explicit specialization '{func.name}' specializes a function template with no definition"
        )
    if arguments is None or len(arguments) != len(_template_parameters(primary)):
        return codegen.refuse_import(
            f"HLSL explicit specialization '{func.name}' does not determine every template argument of '{base}'"
        )
    spelled = ", ".join(
        _crossgl_type_argument(codegen, argument) for argument in arguments
    )
    primary_name = codegen.render_function_identifier(base)
    return f" @ explicit_specialization({primary_name}, {spelled})"


def _sanitized_suffix(text):
    return re.sub(r"\W+", "_", str(text)).strip("_")


def specialization_identifier_renames(codegen, functions, used_names):
    """Concrete CrossGL names for explicit specializations, which must not reuse the primary's name."""
    renames = {}
    used = set(used_names)
    for func in functions or []:
        if not _is_explicit_specialization(func) or getattr(
            func, "is_prototype", False
        ):
            continue
        base, _, arguments = _specialization_primary_and_arguments(codegen, func)
        stem = _sanitized_suffix(base)
        suffix = "_".join(_sanitized_suffix(argument) for argument in arguments or [])
        candidate = (
            f"{stem}_{suffix}_specialization" if suffix else f"{stem}_specialization"
        )
        while candidate in used:
            candidate = f"{candidate}_"
        used.add(candidate)
        renames[id(func)] = candidate
    return renames


def templated_call_name(codegen, func_name):
    """``twice<vec2>`` for a call of a function template that spells HLSL template arguments."""
    base, arguments = _split_templated_name(func_name)
    primaries = _primary_templates(codegen, base) if base is not None else []
    if not primaries:
        return func_name
    parameters = list(_template_parameters(primaries[0]) or [])
    mapped = ", ".join(
        template_argument(
            codegen, parameters[index] if index < len(parameters) else None, argument
        )
        for index, argument in enumerate(arguments)
    )
    return f"{codegen.render_function_identifier(base)}<{mapped}>"


def template_call_result_raw_type(codegen, expr):
    """The HLSL result type of a call of a function template, from its explicit or deduced arguments."""
    name = getattr(expr, "name", None)
    if not isinstance(name, str):
        return None
    base, explicit = _split_templated_name(name)
    if base is None:
        base, explicit = name, []
    primaries = [
        func
        for func in _primary_templates(codegen, base)
        if len(getattr(func, "params", []) or []) == len(expr.args)
    ]
    if len(primaries) != 1 or any(
        func.name == base and not _is_primary_template(func)
        for func in codegen.source_functions.get(base, [])
    ):
        return None
    primary = primaries[0]
    parameters = _template_parameters(primary)
    bindings = {
        parameter.name: argument for parameter, argument in zip(parameters, explicit)
    }
    names = {parameter.name for parameter in parameters}
    for param, argument in zip(primary.params, expr.args):
        expected = _type_base(codegen, param.vtype)
        vector = _TEMPLATE_VECTOR.match(expected)
        if expected not in names and (vector is None or vector["element"] not in names):
            continue
        actual = str(codegen.expression_value_raw_type(argument) or "").strip()
        if vector is not None:
            expected = vector["element"]
            actual = _vector_element(actual, vector["size"])
        if not actual:
            return None
        if bindings.setdefault(expected, actual) != actual:
            return None
    result = _type_base(codegen, primary.return_type)
    if result not in {parameter.name for parameter in parameters}:
        return (
            primary.return_type
            if not _mentions(primary.return_type, parameters)
            else None
        )
    if result in bindings:
        return bindings[result]
    for parameter in parameters:
        if parameter.name == result:
            return parameter.default
    return None


def _vector_element(raw_type, size):
    """``int`` for an ``int3`` or ``vector<int, 3>`` argument of a ``vector<T, 3>`` parameter, else ``None``."""
    for pattern in (_TEMPLATE_VECTOR, _HLSL_VECTOR):
        match = pattern.match(raw_type)
        if match is not None and match["size"] == size:
            return match["element"]
    return None


def _mentions(type_value, parameters):
    text = str(type_value or "")
    return any(
        parameter.name and re.search(rf"\b{re.escape(parameter.name)}\b", text)
        for parameter in parameters
    )

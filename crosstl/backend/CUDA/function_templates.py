"""CUDA and HIP function templates imported as CrossGL generic functions.

``template <typename T> __device__ T twice(T x)`` is CrossGL's ``generic<T> T twice(T x)``. The CUDA and HIP
parsers keep the template parameter list on the function (:func:`template_parameters`), and the importers write it
as the generic prefix (:func:`generic_prefix`); CrossGL deduces ``T`` at each call, or takes it from the call's
template arguments. Before, the importers dropped the template header, so ``T`` was an undeclared type in every
target.

A template value parameter (``template <int N>``) is a CrossGL generic parameter that each call binds to a value:
``scale<3>(x)`` keeps ``3``, and an integer literal for an unsigned parameter becomes ``3u``, so the value CrossGL
substitutes has the parameter's type ([temp.arg.nontype]/2 converts it). A type argument takes its CrossGL spelling
(``twice<vec2<f32>>(v)`` for ``twice<float2>(v)``), and a type parameter keeps a default (``typename U = float``
becomes ``U = f32``).

CrossGL generic parameters have no value defaults, so the importers refuse a function template that gives a value
parameter one with a typed import refusal (``CudaImportRefused``/``HipImportRefused``) rather than write a function
whose calls bind another value. A parameter pack, a template template parameter or an unnamed parameter has no
CrossGL generic form either; such a template, and a kernel template (an entry point CrossGL cannot instantiate),
are written as before, without the prefix, so a target that reaches its parameters refuses them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

__all__ = [
    "TemplateParameter",
    "collect_function_templates",
    "generic_call_name",
    "generic_prefix",
    "parse_template_parameters",
    "template_parameters",
]

_TYPE_KEYWORDS = ("typename", "class")
# The words a value parameter's type can end with, so ``template <unsigned int>`` declares no name.
_VALUE_TYPE_WORDS = frozenset(
    {"auto", "bool", "char", "int", "long", "short", "signed", "unsigned", "size_t"}
)
_PACK = re.compile(r"\.\s*\.\s*\.")
_TEMPLATED_NAME = re.compile(r"^(?P<base>[A-Za-z_]\w*)\s*<(?P<arguments>.*)>$", re.S)
_INTEGER_LITERAL = re.compile(r"([+-]?(?:0[xX][0-9A-Fa-f]+|\d+))[uUlL]*")
_UNSIGNED_KINDS = frozenset(
    {
        "unsigned",
        "unsigned int",
        "unsigned long",
        "unsigned long long",
        "unsigned short",
        "unsigned char",
        "uint",
        "uint8_t",
        "uint16_t",
        "uint32_t",
        "uint64_t",
        "size_t",
    }
)


@dataclass(frozen=True)
class TemplateParameter:
    """One parameter of a ``template <...>`` list.

    ``kind`` is ``"typename"`` for a type parameter, ``"template"`` for a template template parameter, and the
    declared type text (``"int"``, ``"unsigned int"``) for a value parameter. ``pack`` marks ``typename... Ts``.
    """

    kind: str
    name: str | None
    default: str | None = None
    pack: bool = False

    @property
    def is_type(self):
        return self.kind == "typename"


def _split_top_level(text, separator=","):
    """Split ``text`` at each ``separator`` outside ``<>``, ``()``, ``[]`` and ``{}``."""
    parts, depth, current = [], 0, []
    for character in text:
        if character in "<([{":
            depth += 1
        elif character in ">)]}":
            depth -= 1
        if character == separator and depth == 0:
            parts.append("".join(current).strip())
            current = []
            continue
        current.append(character)
    tail = "".join(current).strip()
    if tail:
        parts.append(tail)
    return parts


def _split_default(text):
    """``("int N", "4")`` for ``int N = 4``: the first ``=`` outside brackets starts the default."""
    depth = 0
    for index, character in enumerate(text):
        if character in "<([{":
            depth += 1
        elif character in ">)]}":
            depth -= 1
        elif character == "=" and depth == 0:
            return text[:index].strip(), text[index + 1 :].strip() or None
    return text.strip(), None


def _parameter(text):
    declaration, default = _split_default(text)
    pack = _PACK.search(declaration) is not None
    declaration = " ".join(_PACK.sub(" ", declaration).split())
    if declaration.startswith("template"):
        name = re.search(r"([A-Za-z_]\w*)\s*$", declaration)
        return TemplateParameter(
            "template", name.group(1) if name else None, default, pack
        )
    words = declaration.split()
    if words and words[0] in _TYPE_KEYWORDS:
        name = words[1] if len(words) > 1 else None
        return TemplateParameter("typename", name, default, pack)
    match = re.fullmatch(r"(?P<kind>.*?)(?P<name>[A-Za-z_]\w*)", declaration)
    if (
        match is None
        or not match.group("kind").strip()
        or match.group("name") in _VALUE_TYPE_WORDS
    ):
        # An unnamed value parameter (``template <int>``, ``template <unsigned int>``).
        return TemplateParameter(declaration or "auto", None, default, pack)
    return TemplateParameter(
        " ".join(match.group("kind").split()), match.group("name"), default, pack
    )


def parse_template_parameters(header):
    """The :class:`TemplateParameter` entries of a formatted ``<typename T, int N = 4>`` header."""
    header = str(header or "").strip()
    if header.startswith("<") and header.endswith(">"):
        header = header[1:-1]
    return [_parameter(part) for part in _split_top_level(header) if part]


def template_parameters(func):
    """The template parameters the parser kept on ``func``, or ``None`` for a function that is not a template."""
    return getattr(func, "template_parameters", None)


def collect_function_templates(program):
    """Map each function template declared in ``program`` (a CUDA shader or HIP program node) to its parameters.

    The CUDA AST lists functions in ``functions``; the HIP AST keeps declarations in ``statements``, including the
    bodies of namespace blocks. Function bodies are not searched.
    """
    templates = {}
    pending = [program]
    while pending:
        node = pending.pop()
        if isinstance(node, (list, tuple)):
            pending.extend(node)
            continue
        parameters = template_parameters(node)
        name = getattr(node, "name", None)
        if parameters and isinstance(name, str):
            templates.setdefault(name, parameters)
        for field in ("statements", "functions", "declarations"):
            children = getattr(node, field, None)
            if isinstance(children, (list, tuple)):
                pending.extend(children)
    return templates


def generic_prefix(codegen, func, language, convert_type):
    """``generic<T, U = f32> `` for a device function template, ``""`` for any other function.

    ``convert_type`` maps a CUDA or HIP type spelling to CrossGL. A parameter that CrossGL generics cannot carry is
    refused through ``codegen.refuse_import``.
    """
    parameters = template_parameters(func)
    if not parameters or "__global__" in (getattr(func, "qualifiers", None) or []):
        return ""
    if any(
        parameter.pack or parameter.kind == "template" or parameter.name is None
        for parameter in parameters
    ):
        # Parameter packs, template template parameters and unnamed parameters have no CrossGL generic form; the
        # function keeps its parameters' names, as before, and a target that reaches it refuses them.
        return ""
    rendered = []
    for parameter in parameters:
        if not parameter.is_type and parameter.default is not None:
            codegen.refuse_import(
                f"{language} function template '{func.name}' gives the value parameter "
                f"'{parameter.kind} {parameter.name}' a default, which CrossGL generic parameters cannot have"
            )
            return ""
        if parameter.is_type and parameter.default is not None:
            rendered.append(f"{parameter.name} = {convert_type(parameter.default)}")
        else:
            rendered.append(parameter.name)
    return f"generic<{', '.join(rendered)}> "


def generic_call_name(name, templates, convert_type):
    """``twice<vec2<f32>>`` for a call ``twice<float2>`` of a function template in ``templates``, else ``None``.

    ``templates`` maps a function template's name to its :class:`TemplateParameter` list.
    """
    if not isinstance(name, str):
        return None
    match = _TEMPLATED_NAME.match(name.strip())
    if match is None or match.group("base") not in templates:
        return None
    parameters = templates[match.group("base")]
    arguments = _split_top_level(match.group("arguments"))
    rendered = []
    for index, argument in enumerate(arguments):
        parameter = parameters[index] if index < len(parameters) else None
        if parameter is None or parameter.is_type:
            rendered.append(str(convert_type(argument)).strip())
            continue
        literal = _INTEGER_LITERAL.fullmatch(argument)
        if literal is not None:
            unsigned = (
                " ".join(parameter.kind.replace("const", " ").split())
                in _UNSIGNED_KINDS
            )
            rendered.append(f"{literal.group(1)}u" if unsigned else literal.group(1))
        else:
            rendered.append(argument)
    return f"{match.group('base')}<{', '.join(rendered)}>"

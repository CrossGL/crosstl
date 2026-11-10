"""Concrete types named by alias templates and by ``decltype`` of context-free arithmetic.

A header may name a function's result type through an alias template whose target is a ``decltype``::

    template <typename U, typename V>
    using common_dtype = decltype(U(0) + V(0));

    template <typename T, typename U>
    inline common_dtype<T, U> mul(const T x, const U y) { return x * y; }

Once ``mul`` is materialized for ``float`` and ``int``, its return type is ``common_dtype<float, int>``, which is
``decltype(float(0) + int(0))`` ([temp.alias]/2), which is ``float``: the operands are prvalues of built-in types,
so ``decltype`` denotes the type of the expression itself ([dcl.type.decltype]/1.5), and that type follows the usual
arithmetic conversions as ``arithmetic_conversions.py`` decides them.

``fold_expression_decltypes`` replaces ``decltype(e)`` when ``e`` names nothing but built-in scalar and vector
types, literals and ``static_cast``, and the importer types ``e`` as one scalar or vector type. Any other
``decltype`` stays as written, so a construct the importer cannot type is still refused where it is used.

``expand_alias_template_types`` replaces a reference to a namespace-scope alias template whose arguments are all
built-in scalar or vector types, when exactly one alias template of that name matches the reference's qualifier and
its substituted target folds to one scalar or vector type. Alias templates declared in an aggregate, with default
or variadic parameters, or with non-type parameters are left alone.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

__all__ = [
    "AliasTemplate",
    "expand_alias_template_types",
    "expand_concrete_alias_template_references",
    "find_alias_templates",
    "fold_expression_decltypes",
]

_DECLTYPE_RE = re.compile(r"(?<![\w:.])decltype\s*\(")
_METAL_QUALIFIER_RE = re.compile(r"(?<![\w:])(?:::\s*)?metal\s*::\s*")
_NUMERIC_LITERAL_RE = re.compile(
    r"(?<![\w.])(?:0[xX][0-9A-Fa-f']+|(?:\d[\d']*\.?[\d']*|\.\d[\d']*)(?:[eEpP][+-]?\d+)?)[A-Za-z]*"
)
_IDENTIFIER_RE = re.compile(r"[A-Za-z_]\w*")
_ALLOWED_WORDS = frozenset({"static_cast", "true", "false"})
_ALIAS_HEAD_RE = re.compile(r"\s*using\s+(?P<name>[A-Za-z_]\w*)\s*=")


@dataclass(frozen=True)
class AliasTemplate:
    """A namespace-scope ``template <typename...> using name = target;``."""

    name: str
    namespace: str
    parameters: tuple[str, ...]
    target: str


def _concrete_type(preprocessor: Any, text: str) -> str | None:
    """``text`` as a normalized built-in scalar or vector type, or None."""

    candidate = _METAL_QUALIFIER_RE.sub("", text).strip()
    if not candidate or not preprocessor._is_metal_scalar_or_vector_type(candidate):
        return None
    normalized = preprocessor._normalize_inferred_type(candidate)
    if not normalized or not preprocessor._is_metal_scalar_or_vector_type(normalized):
        return None
    return normalized


def _context_free_expression_type(preprocessor: Any, expression: str) -> str | None:
    """The type of ``expression`` when it names only built-in types, literals and casts."""

    expression = _METAL_QUALIFIER_RE.sub("", expression).strip()
    if not expression:
        return None
    words = _NUMERIC_LITERAL_RE.sub(" ", expression)
    for match in _IDENTIFIER_RE.finditer(words):
        word = match.group(0)
        if word in _ALLOWED_WORDS:
            continue
        if not preprocessor._is_metal_scalar_or_vector_type(word):
            return None
    scope = getattr(preprocessor, "_call_result_scope", None)
    if scope is None:
        inferred = preprocessor._infer_argument_type(expression, {}, {})
    else:
        # The expression calls no source function, so the call-result index is not consulted (or built).
        with scope.active(None):
            inferred = preprocessor._infer_argument_type(expression, {}, {})
    if inferred is None:
        return None
    return _concrete_type(preprocessor, inferred)


def fold_expression_decltypes(preprocessor: Any, text: str) -> str:
    """Replace each ``decltype`` of a context-free arithmetic expression with the type it denotes."""

    if "decltype" not in text:
        return text
    pieces: list[str] = []
    position = 0
    while True:
        match = _DECLTYPE_RE.search(text, position)
        if match is None:
            break
        open_paren = match.end() - 1
        close_paren = preprocessor._find_matching_delimiter(text, open_paren, "(", ")")
        if close_paren is None or close_paren < 0:
            break
        folded = _context_free_expression_type(
            preprocessor, text[open_paren + 1 : close_paren]
        )
        if folded is None:
            # An operand may itself be a decltype that folds; the outer one is retried by a later call.
            pieces.append(text[position : open_paren + 1])
            position = open_paren + 1
            continue
        pieces.append(text[position : match.start()])
        pieces.append(folded)
        position = close_paren + 1
    pieces.append(text[position:])
    return "".join(pieces)


def _aggregate_spans(preprocessor: Any, code: str) -> list[tuple[int, int]]:
    return sorted(
        [struct.span for struct in preprocessor._find_concrete_struct_definitions(code)]
        + [struct.span for struct in preprocessor._find_template_structs(code)]
    )


def _alias_parameters(preprocessor: Any, parameter_text: str) -> tuple[str, ...] | None:
    parameters = preprocessor._parse_template_parameter_list(parameter_text)
    names: list[str] = []
    for parameter in parameters:
        if (
            parameter.name is None
            or not parameter.is_type_parameter
            or parameter.is_variadic
            or parameter.default is not None
        ):
            return None
        names.append(parameter.name)
    return tuple(names) or None


def find_alias_templates(
    preprocessor: Any, code: str
) -> dict[str, list[AliasTemplate]]:
    """Namespace-scope alias templates of ``code`` with type parameters only, by name."""

    if "using" not in code:
        return {}
    masked = preprocessor._mask_comments_and_literals(code)
    namespace_spans = None
    aggregates = None
    found: dict[str, list[AliasTemplate]] = {}
    for match in re.finditer(r"\btemplate\s*<", masked):
        angle_start = match.end() - 1
        angle_end = preprocessor._find_matching_angle(code, angle_start)
        if angle_end is None:
            continue
        head = _ALIAS_HEAD_RE.match(masked, angle_end + 1)
        if head is None:
            continue
        semicolon = masked.find(";", head.end())
        if semicolon < 0:
            continue
        if aggregates is None:
            aggregates = _aggregate_spans(preprocessor, code)
            namespace_spans = preprocessor._find_namespace_spans(code)
        if any(start <= match.start() < end for start, end in aggregates):
            continue
        parameters = _alias_parameters(preprocessor, code[angle_start + 1 : angle_end])
        if parameters is None:
            continue
        name = head.group("name")
        found.setdefault(name, []).append(
            AliasTemplate(
                name=name,
                namespace=preprocessor._namespace_at(namespace_spans, match.start()),
                parameters=parameters,
                target=code[head.end() : semicolon].strip(),
            )
        )
    return found


def _qualifier_matches(qualifier: str, namespace: str) -> bool:
    components = [part for part in re.split(r"\s*::\s*", qualifier) if part]
    absolute = qualifier.lstrip().startswith("::")
    namespace_components = [part for part in namespace.split("::") if part]
    if absolute:
        return components == namespace_components
    return not components or namespace_components[-len(components) :] == components


def _alias_reference_type(
    preprocessor: Any,
    qualifier: str,
    candidates: Sequence[AliasTemplate],
    arguments: Sequence[str],
) -> str | None:
    matching = [
        alias
        for alias in candidates
        if len(alias.parameters) == len(arguments)
        and _qualifier_matches(qualifier, alias.namespace)
    ]
    if len(matching) != 1:
        return None
    bindings: dict[str, str] = {}
    for parameter, argument in zip(matching[0].parameters, arguments):
        concrete = _concrete_type(preprocessor, argument)
        if concrete is None:
            return None
        bindings[parameter] = concrete
    target = preprocessor._replace_identifiers(matching[0].target, bindings)
    target = re.sub(r"^\s*typename\s+", "", target)
    return _concrete_type(preprocessor, fold_expression_decltypes(preprocessor, target))


def expand_alias_template_types(
    preprocessor: Any,
    text: str,
    aliases: Mapping[str, Sequence[AliasTemplate]],
    namespace_at: Callable[[int], str] | None = None,
) -> str:
    """Replace references to ``aliases`` that name one built-in scalar or vector type.

    With ``namespace_at``, the namespace that encloses a position of ``text``, an unqualified reference considers only
    the alias templates its enclosing namespaces declare ([basic.lookup.unqual]/4).
    """

    names = [name for name in sorted(aliases) if name in text]
    if not names:
        return text
    pattern = re.compile(
        r"(?<![\w.])(?P<qualifier>(?:::\s*)?(?:[A-Za-z_]\w*\s*::\s*)*)"
        rf"(?P<name>{'|'.join(re.escape(name) for name in names)})\s*<"
    )
    pieces: list[str] = []
    position = 0
    while True:
        match = pattern.search(text, position)
        if match is None:
            break
        angle_start = match.end() - 1
        angle_end = preprocessor._find_matching_template_param_angle(text, angle_start)
        if angle_end is None:
            break
        arguments = preprocessor._split_top_level_commas(
            text[angle_start + 1 : angle_end]
        )
        candidates = aliases[match.group("name")]
        if namespace_at is not None and not match.group("qualifier").strip():
            enclosing = namespace_at(match.start("name"))
            candidates = [
                alias for alias in candidates if _encloses(alias.namespace, enclosing)
            ]
        replacement = _alias_reference_type(
            preprocessor,
            match.group("qualifier"),
            candidates,
            [argument.strip() for argument in arguments],
        )
        if replacement is None:
            pieces.append(text[position : angle_start + 1])
            position = angle_start + 1
            continue
        pieces.append(text[position : match.start()])
        pieces.append(replacement)
        position = angle_end + 1
    pieces.append(text[position:])
    return "".join(pieces)


def _encloses(namespace: str, enclosing: str) -> bool:
    """Whether ``namespace`` is ``enclosing`` or one of the namespaces that contain it."""

    outer = [part for part in namespace.split("::") if part]
    inner = [part for part in enclosing.split("::") if part]
    return inner[: len(outer)] == outer


def _member_alias_template_names(preprocessor: Any, code: str) -> set[str]:
    """Names of the alias templates that aggregates of ``code`` declare."""

    masked = preprocessor._mask_comments_and_literals(code)
    aggregates = None
    names: set[str] = set()
    for match in re.finditer(r"\btemplate\s*<", masked):
        angle_end = preprocessor._find_matching_angle(code, match.end() - 1)
        head = (
            None if angle_end is None else _ALIAS_HEAD_RE.match(masked, angle_end + 1)
        )
        if head is None:
            continue
        if aggregates is None:
            aggregates = _aggregate_spans(preprocessor, code)
        if any(start <= match.start() < end for start, end in aggregates):
            names.add(head.group("name"))
    return names


def expand_concrete_alias_template_references(preprocessor: Any, code: str) -> str:
    """Spell each reference to a namespace-scope alias template that names one built-in type as that type.

    A materialized instantiation of ``template <typename T, typename U> common_dtype<T, U> div(T, U)`` declares its
    result as ``common_dtype<float, float>``, which is ``float`` ([temp.alias]/2). Spelling it so lets the result type
    a call has, and the type of an ``auto`` local it initializes, be read from the declaration like any other.

    A name that an aggregate also declares as a member alias template is left alone: inside that aggregate the member
    hides the namespace-scope alias, and this scan does not tell the two apart.
    """

    aliases = find_alias_templates(preprocessor, code)
    if not aliases:
        return code
    hidden = _member_alias_template_names(preprocessor, code)
    aliases = {name: found for name, found in aliases.items() if name not in hidden}
    if not aliases:
        return code
    namespace_spans = preprocessor._find_namespace_spans(code)
    return expand_alias_template_types(
        preprocessor,
        code,
        aliases,
        lambda position: preprocessor._namespace_at(namespace_spans, position),
    )

"""Select the partial specialization of a Metal struct template that a template-id names.

A template-id ``tr<device half*>`` names the partial specialization whose argument patterns deduce from its
arguments, and the primary template only when no partial specialization matches ([temp.spec.partial.match]).
When several match, the most specialized one is used, and the template-id is ill-formed when none is more
specialized than all the others ([temp.spec.partial.order]).  Picking the primary template whenever no explicit
specialization is spelled the same way therefore gives a member such as ``tr<device half*>::type`` the primary's
meaning even though ``template <typename T> struct tr<device T*>`` redefines it.

The comparison is structural over normalized Metal spellings.  A pattern position that needs evaluation
(``enable_if_t<...>``, ``decltype``, arithmetic), an argument that is still dependent on an unbound template
parameter where the pattern spells something concrete, and a value that may be a named constant are
*undecided*: the specialization may or may not match, so the caller must not fall back to the primary template.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

from crosstl.translator.refusals import TranslationRefusal

MATCH = "match"
NO_MATCH = "no-match"
UNDECIDED = "undecided"

_TOKEN_RE = re.compile(r"[A-Za-z_]\w*|\d[\w.]*|::|->|<<|>>|&&|\|\||[=!<>]=|\S")
_INTEGER_RE = re.compile(r"(\d+)[uUlL]*")
_VECTOR_SHORTHAND_RE = re.compile(
    r"(bool|char|uchar|short|ushort|int|uint|long|ulong|half|float|bfloat)([234])"
)
_QUALIFIERS = frozenset(
    {
        "const",
        "volatile",
        "device",
        "constant",
        "thread",
        "threadgroup",
        "threadgroup_imageblock",
        "ray_data",
        "object_data",
    }
)
_IGNORED = frozenset({"typename", "struct", "class"})
_OPENERS = {"<": ">", "(": ")", "[": "]", "{": "}"}
_CLOSERS = frozenset(_OPENERS.values())
_SEGMENT_DELIMITERS = frozenset({"*", "&", "<", ">", ","})
# Spellings that a pattern can contain only when matching it needs evaluation.
_EVALUATED_NAMES = frozenset(
    {
        "enable_if",
        "enable_if_t",
        "decltype",
        "conditional",
        "conditional_t",
        "void_t",
        "sizeof",
        "alignof",
    }
)
_OPERATORS = frozenset(
    {
        "+",
        "-",
        "/",
        "%",
        "?",
        ":",
        "!",
        "~",
        "^",
        "|",
        "<<",
        ">>",
        "&&",
        "||",
        "==",
        "!=",
        "<=",
        ">=",
        "(",
    }
)


def argument_tokens(text: object) -> tuple[str, ...]:
    """Tokenize one template argument into the normalized form patterns are compared in.

    ``metal::`` qualification, elaborated-type keywords and integer-literal suffixes are dropped, a vector
    shorthand such as ``half2`` is spelled ``vec<half, 2>`` (Metal Shading Language Specification, section 2.2),
    and the qualifiers of each declarator segment are sorted, so ``device const T*`` and ``const device T*``
    compare equal.
    """

    raw = _TOKEN_RE.findall(re.sub(r"\[\[[^\]]*\]\]", " ", str(text or "")))
    tokens: list[str] = []
    index = 0
    while index < len(raw):
        token = raw[index]
        if token == "metal" and index + 1 < len(raw) and raw[index + 1] == "::":
            index += 2
            continue
        if token == "::" and not tokens and index + 1 < len(raw):
            index += 1
            continue
        index += 1
        if token in _IGNORED:
            continue
        integer = _INTEGER_RE.fullmatch(token)
        if integer is not None:
            tokens.append(integer.group(1))
            continue
        shorthand = _VECTOR_SHORTHAND_RE.fullmatch(token)
        if shorthand is not None:
            tokens.extend(
                ("vec", "<", shorthand.group(1), ",", shorthand.group(2), ">")
            )
            continue
        tokens.append(token)
    return _sort_segment_qualifiers(tokens)


def _sort_segment_qualifiers(tokens: Sequence[str]) -> tuple[str, ...]:
    result: list[str] = []
    segment: list[str] = []

    def flush() -> None:
        qualifiers = sorted(token for token in segment if token in _QUALIFIERS)
        result.extend(qualifiers)
        result.extend(token for token in segment if token not in _QUALIFIERS)
        segment.clear()

    for token in tokens:
        if token in _SEGMENT_DELIMITERS:
            flush()
            result.append(token)
        else:
            segment.append(token)
    flush()
    return tuple(result)


def _needs_evaluation(tokens: Sequence[str]) -> bool:
    for index, token in enumerate(tokens):
        if token in _EVALUATED_NAMES or token in _OPERATORS:
            return True
        if token == "*" and index + 1 < len(tokens):
            following = tokens[index + 1]
            # ``N * 2`` multiplies; a pointer declarator is followed by a delimiter or a qualifier.
            if following not in _SEGMENT_DELIMITERS and following not in _QUALIFIERS:
                return True
    return False


def _balanced_slice(tokens: Sequence[str]) -> bool:
    if not tokens or tokens[0] in _CLOSERS:
        return False
    stack: list[str] = []
    for token in tokens:
        if token in _OPENERS:
            stack.append(_OPENERS[token])
        elif token in _CLOSERS:
            if not stack or stack.pop() != token:
                return False
        elif token == "," and not stack:
            return False
    return not stack


_BOOL_LITERALS = {"true": "1", "false": "0"}


def _integer_spelling(literal: str) -> str:
    """``literal`` with ``true`` and ``false`` spelled as the integers a ``bool`` converts to."""
    if literal in _BOOL_LITERALS:
        return _BOOL_LITERALS[literal]
    return literal


def _literal_status(pattern: str, concrete: str, dependent: frozenset[str]) -> str:
    if pattern == concrete:
        return MATCH
    if concrete in dependent:
        return UNDECIDED
    pattern_value = _integer_spelling(pattern)
    concrete_value = _integer_spelling(concrete)
    pattern_number = pattern_value.isdigit()
    concrete_number = concrete_value.isdigit()
    if pattern_number and concrete_number:
        if pattern in _BOOL_LITERALS or concrete in _BOOL_LITERALS:
            # A bool parameter converts any nonzero value to ``true``.
            return (
                MATCH if (pattern_value != "0") == (concrete_value != "0") else NO_MATCH
            )
        return MATCH if int(pattern_value) == int(concrete_value) else NO_MATCH
    if pattern_number != concrete_number:
        # A named constant may have the literal's value.
        return UNDECIDED
    return NO_MATCH


def _combine(statuses: Iterable[str]) -> str:
    result = NO_MATCH
    for status in statuses:
        if status == MATCH:
            return MATCH
        if status == UNDECIDED:
            result = UNDECIDED
    return result


def _match_tokens(
    pattern: Sequence[str],
    concrete: Sequence[str],
    variables: frozenset[str],
    dependent: frozenset[str],
    bindings: dict[str, tuple[str, ...]],
) -> str:
    """Three-valued backtracking match of one argument pattern; ``bindings`` holds the first match."""

    def mismatch(concrete_index: int) -> str:
        return (
            UNDECIDED if dependent.intersection(concrete[concrete_index:]) else NO_MATCH
        )

    def bind(name: str, prefix: tuple[str, ...], pattern_index: int, start: int) -> str:
        """Bind the unbound variable ``name`` to ``prefix`` plus a balanced run from ``start``."""
        statuses = []
        for end in range(start + 1, len(concrete) + 1):
            candidate = prefix + tuple(concrete[start:end])
            if not _balanced_slice(candidate):
                continue
            bindings[name] = candidate
            status = visit(pattern_index + 1, end)
            if status == MATCH:
                return MATCH
            del bindings[name]
            statuses.append(status)
        return _combine(statuses)

    def run_end(tokens: Sequence[str], index: int) -> int:
        while index < len(tokens) and tokens[index] in _QUALIFIERS:
            index += 1
        return index

    def visit_qualifiers(pattern_index: int, concrete_index: int) -> str:
        # Qualifiers of one declarator segment compare as a set.  A qualifier the pattern does not spell is
        # deduced into the variable that forms the rest of the segment: ``device T*`` deduces ``T = const half``
        # from ``const device half*`` ([temp.deduct.call]/4).
        pattern_end = run_end(pattern, pattern_index)
        concrete_end = run_end(concrete, concrete_index)
        wanted = set(pattern[pattern_index:pattern_end])
        present = list(concrete[concrete_index:concrete_end])
        if not wanted.issubset(present):
            return mismatch(concrete_end)
        extra = tuple(token for token in present if token not in wanted)
        if not extra:
            return visit(pattern_end, concrete_end)
        absorbs = (
            pattern_end < len(pattern)
            and pattern[pattern_end] in variables
            and pattern[pattern_end] not in bindings
            and (
                pattern_end + 1 == len(pattern)
                or pattern[pattern_end + 1] in _SEGMENT_DELIMITERS
            )
        )
        if not absorbs:
            return mismatch(concrete_end)
        return bind(pattern[pattern_end], extra, pattern_end, concrete_end)

    def visit(pattern_index: int, concrete_index: int) -> str:
        if pattern_index == len(pattern):
            if concrete_index == len(concrete):
                return MATCH
            return mismatch(concrete_index)
        if concrete_index == len(concrete):
            return NO_MATCH
        token = pattern[pattern_index]
        if token in _QUALIFIERS:
            return visit_qualifiers(pattern_index, concrete_index)
        if token in variables:
            bound = bindings.get(token)
            if bound is not None:
                end = concrete_index + len(bound)
                if tuple(concrete[concrete_index:end]) == bound:
                    return visit(pattern_index + 1, end)
                return mismatch(concrete_index)
            return bind(token, (), pattern_index, concrete_index)
        status = _literal_status(token, concrete[concrete_index], dependent)
        if status == NO_MATCH:
            return NO_MATCH
        following = visit(pattern_index + 1, concrete_index + 1)
        if status == UNDECIDED and following != NO_MATCH:
            return UNDECIDED
        return following if status == MATCH else NO_MATCH

    return visit(0, 0)


@dataclass(frozen=True)
class SpecializationMatch:
    """The outcome of matching one partial specialization against concrete template arguments."""

    status: str
    bindings: Mapping[str, str] = field(default_factory=dict)
    reason: str = ""


def match_partial_specialization(
    pattern_arguments: Sequence[str],
    concrete_arguments: Sequence[str],
    parameters: Sequence[str],
    *,
    dependent_names: Iterable[str] = (),
) -> SpecializationMatch:
    """Match the argument patterns of a partial specialization against a template-id's arguments.

    ``parameters`` are the partial specialization's template parameters, ``concrete_arguments`` the template-id's
    arguments with the primary template's defaults already supplied, and ``dependent_names`` the template
    parameters still unbound where the template-id is spelled.  A match binds every parameter.
    """

    if len(pattern_arguments) != len(concrete_arguments):
        return SpecializationMatch(UNDECIDED, reason="the argument counts differ")
    variables = frozenset(parameters)
    dependent = frozenset(dependent_names) - variables
    bindings: dict[str, tuple[str, ...]] = {}
    for pattern_text, concrete_text in zip(pattern_arguments, concrete_arguments):
        pattern = argument_tokens(pattern_text)
        concrete = argument_tokens(concrete_text)
        if _needs_evaluation(pattern):
            if pattern == concrete:
                continue
            return SpecializationMatch(
                UNDECIDED, reason=f"the pattern '{pattern_text}' needs evaluation"
            )
        if _needs_evaluation(concrete) and variables.isdisjoint(pattern):
            if pattern == concrete:
                continue
            return SpecializationMatch(
                UNDECIDED, reason=f"the argument '{concrete_text}' needs evaluation"
            )
        status = _match_tokens(pattern, concrete, variables, dependent, bindings)
        if status != MATCH:
            reason = (
                f"'{concrete_text}' may match the pattern '{pattern_text}'"
                if status == UNDECIDED
                else ""
            )
            return SpecializationMatch(status, reason=reason)
    unbound = [name for name in parameters if name not in bindings]
    if unbound:
        return SpecializationMatch(
            UNDECIDED,
            reason="the pattern does not deduce "
            + ", ".join(repr(name) for name in unbound),
        )
    return SpecializationMatch(
        MATCH, {name: spell_tokens(bindings[name]) for name in parameters}
    )


def spell_tokens(tokens: Sequence[str]) -> str:
    """Spell normalized tokens as Metal source text."""

    text = ""
    previous = ""
    for token in tokens:
        if (
            text
            and (previous[:1].isalnum() or previous[:1] == "_")
            and (token[:1].isalnum() or token[:1] == "_")
        ):
            text += " "
        text += token
        if token == ",":
            text += " "
        previous = token
    return text


def more_specialized(
    candidate: tuple[Sequence[str], Sequence[str]],
    other: tuple[Sequence[str], Sequence[str]],
) -> bool:
    """Whether ``candidate`` is more specialized than ``other`` ([temp.spec.partial.order]).

    Each is ``(argument patterns, template parameters)``.  ``candidate`` is more specialized when ``other``'s
    patterns deduce from ``candidate``'s, with ``candidate``'s parameters as unique types, and not conversely.
    """

    def deduces(left, right) -> bool:
        return match_partial_specialization(left[0], right[0], left[1]).status == MATCH

    return deduces(other, candidate) and not deduces(candidate, other)


def select_partial_specialization(
    specializations: Sequence[tuple[object, Sequence[str], Sequence[str]]],
    concrete_arguments: Sequence[str],
    *,
    dependent_names: Iterable[str] = (),
):
    """Choose the partial specialization a template-id names.

    ``specializations`` holds ``(declaration, argument patterns, template parameters)`` entries.  Returns
    ``(status, declaration, bindings, reason)``: ``MATCH`` with the selected declaration and its parameter
    bindings, ``NO_MATCH`` when the primary template applies, or ``UNDECIDED`` with a reason when no choice is
    sound.  Several matches with no most specialized one are reported as ``UNDECIDED`` too, because the
    template-id is ill-formed.
    """

    dependent_names = tuple(dependent_names)
    matched = []
    undecided = []
    for declaration, patterns, parameters in specializations:
        outcome = match_partial_specialization(
            patterns, concrete_arguments, parameters, dependent_names=dependent_names
        )
        if outcome.status == MATCH:
            matched.append((declaration, patterns, parameters, outcome.bindings))
        elif outcome.status == UNDECIDED:
            undecided.append((patterns, outcome.reason))
    if undecided:
        patterns, reason = undecided[0]
        return (
            UNDECIDED,
            None,
            {},
            f"the partial specialization <{', '.join(patterns)}> may apply: {reason}",
        )
    if not matched:
        return NO_MATCH, None, {}, ""
    best = [
        entry
        for entry in matched
        if all(
            other is entry
            or more_specialized((entry[1], entry[2]), (other[1], other[2]))
            for other in matched
        )
    ]
    if len(best) != 1:
        return (
            UNDECIDED,
            None,
            {},
            "no matching partial specialization is more specialized than the others: "
            + "; ".join(f"<{', '.join(entry[1])}>" for entry in matched),
        )
    declaration, _patterns, _parameters, bindings = best[0]
    return MATCH, declaration, dict(bindings), ""


class UndecidedPartialSpecialization(TranslationRefusal):
    """A template-id may name a partial specialization that cannot be decided where it is spelled."""

    code = "project.translate.metal-partial-specialization-undecided"

    def __init__(self, owner: str, reason: str):
        super().__init__(reason)
        self.owner = owner
        self.reason = reason


def _template_parameter_names(declaration: object) -> list[str]:
    return [
        entry[1]
        for entry in getattr(declaration, "template_parameters", None) or []
        if isinstance(entry, (tuple, list)) and len(entry) >= 2 and entry[1]
    ]


def unbound_template_parameter_names(converter) -> set[str]:
    """Template parameters of the function or struct being converted that no active binding replaces."""

    from ..common_ast import FunctionNode, StructNode

    names: set[str] = set()
    for context in (
        getattr(converter, "current_type_resolution_context", None),
        getattr(converter, "current_function", None),
    ):
        if isinstance(context, (FunctionNode, StructNode)):
            names.update(_template_parameter_names(context))
    for scopes in (
        getattr(converter, "template_type_bindings", None) or [],
        getattr(converter, "template_value_bindings", None) or [],
    ):
        for scope in scopes:
            names.difference_update(scope)
    return names


def _active_bindings(converter) -> dict[str, str]:
    bindings: dict[str, str] = {}
    for scopes in (
        getattr(converter, "template_value_bindings", None) or [],
        getattr(converter, "template_type_bindings", None) or [],
    ):
        for scope in scopes:
            bindings.update({name: str(value) for name, value in scope.items()})
    return bindings


def _argument_spelling(converter, argument: object, context: object) -> str:
    """``canonical_struct_template_argument`` with alias resolution but token spacing preserved."""

    text = str(argument).strip()
    instance = converter.alias_template_instance_parts(text)
    if instance is not None:
        name, arguments = instance
        return (
            f"{name}<"
            + ", ".join(
                _argument_spelling(converter, nested, context) for nested in arguments
            )
            + ">"
        )
    binding = converter.scalar_alias_binding(text, context=context)
    if binding is not None:
        return " ".join([*binding[1], binding[0]])
    if context is None:
        return str(converter.resolve_local_type_aliases(text))
    return text


def _concrete_arguments(
    converter, arguments: Sequence[str], primary: object
) -> list[str]:
    bindings = _active_bindings(converter)
    concrete = [
        _argument_spelling(
            converter,
            converter.replace_alias_template_identifiers(argument, bindings),
            None,
        )
        for argument in arguments
    ]
    parameters = _template_parameter_names(primary)
    defaults = dict(getattr(primary, "template_parameter_defaults", None) or {})
    supplied = dict(zip(parameters, concrete))
    for name in parameters[len(concrete) :]:
        default = defaults.get(name)
        if default is None:
            break
        value = converter.replace_alias_template_identifiers(default, supplied).strip()
        supplied[name] = value
        concrete.append(_argument_spelling(converter, value, None))
    return concrete


def partial_specialization_for_owner(
    converter,
    owner: str,
    candidate_names: Iterable[str],
    arguments: Sequence[str],
    primary_nodes: Sequence[object],
):
    """Return ``(declaration, arguments)`` for the partial specialization ``owner`` names, or ``None``.

    ``None`` means that the primary template applies.  Raises :class:`UndecidedPartialSpecialization` when a
    visible partial specialization may apply but the choice cannot be made soundly.
    """

    candidate_names = set(candidate_names)
    specializations = []
    for (
        qualified_name,
        nodes,
    ) in converter.alias_template_structs_by_qualified_name.items():
        instance = converter.alias_template_instance_parts(qualified_name)
        if instance is None or instance[0] not in candidate_names:
            continue
        patterns = instance[1]
        for node in nodes:
            # Explicit specializations take part too: one spelled differently from the template-id
            # (``vec<float, 2>`` for ``float2``) or one that a dependent argument may name.
            parameters = _template_parameter_names(node)
            if not converter.declaration_visible_at_current_offset(node):
                continue
            if any(
                str(entry[0]).endswith("...")
                for entry in getattr(node, "template_parameters", None) or []
            ):
                raise UndecidedPartialSpecialization(
                    owner,
                    f"the variadic partial specialization '{qualified_name}' is not matched",
                )
            previous_context = converter.current_type_resolution_context
            converter.current_type_resolution_context = node
            try:
                spelled = [
                    _argument_spelling(converter, pattern, node) for pattern in patterns
                ]
            finally:
                converter.current_type_resolution_context = previous_context
            specializations.append((node, spelled, parameters))
    if not specializations or not primary_nodes:
        return None
    concrete = _concrete_arguments(converter, arguments, primary_nodes[0])
    status, declaration, bindings, reason = select_partial_specialization(
        specializations,
        concrete,
        dependent_names=sorted(unbound_template_parameter_names(converter)),
    )
    if status == UNDECIDED:
        raise UndecidedPartialSpecialization(owner, reason)
    if status == NO_MATCH:
        return None
    return declaration, [
        bindings[name] for name in _template_parameter_names(declaration)
    ]

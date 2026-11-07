"""Alias templates in type-trait operands.

``template <int val> using Int = integral_constant<int, val>;`` is not a
type of its own, so ``metal::is_same_v<StrY, Int<1>>`` with ``StrY`` deduced as
``Int<1>`` cannot be decided by looking ``Int`` up among the known types.
Substituting the alias's arguments into its target gives the template-id it
denotes (``integral_constant<int, 1>``), which the trait evaluator compares
like any other operand. Type and non-type parameters are both supported;
variadic or defaulted parameters, struct-scoped aliases and names declared by
more than one alias are left alone, so their operands stay unresolved and the
trait is refused as before.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

_ALIAS_HEAD_RE = re.compile(r"\s*using\s+(?P<name>[A-Za-z_]\w*)\s*=")
_REFERENCE_RE_CACHE: dict[tuple[str, ...], re.Pattern[str]] = {}
_MAX_EXPANSION_ROUNDS = 8


@dataclass(frozen=True)
class TraitAliasTemplate:
    """A namespace-scope ``template <...> using name = target;``."""

    name: str
    namespace: str
    parameters: tuple[str, ...]
    target: str


def _alias_parameters(preprocessor: Any, text: str) -> tuple[str, ...] | None:
    names: list[str] = []
    for parameter in preprocessor._parse_template_parameter_list(text):
        if parameter.name is None or parameter.is_variadic or parameter.default:
            return None
        names.append(parameter.name)
    return tuple(names) or None


def find_trait_alias_templates(
    preprocessor: Any, code: str
) -> dict[str, list[TraitAliasTemplate]]:
    """Namespace-scope alias templates of ``code`` by name, in source order."""

    if "using" not in code or "template" not in code:
        return {}
    masked = preprocessor._mask_comments_and_literals(code)
    aggregates: list[tuple[int, int]] | None = None
    namespace_spans = None
    found: dict[str, list[TraitAliasTemplate]] = {}
    for match in re.finditer(r"\btemplate\s*<", masked):
        angle_start = match.end() - 1
        angle_end = preprocessor._find_matching_template_param_angle(
            masked, angle_start
        )
        if angle_end is None:
            continue
        head = _ALIAS_HEAD_RE.match(masked, angle_end + 1)
        if head is None:
            continue
        semicolon = masked.find(";", head.end())
        if semicolon < 0:
            continue
        if aggregates is None:
            aggregates = sorted(
                [s.span for s in preprocessor._find_concrete_struct_definitions(code)]
                + [s.span for s in preprocessor._find_template_structs(code)]
            )
            namespace_spans = preprocessor._find_namespace_spans(code)
        if any(start <= match.start() < end for start, end in aggregates):
            continue
        parameters = _alias_parameters(preprocessor, code[angle_start + 1 : angle_end])
        if parameters is None:
            continue
        name = head.group("name")
        found.setdefault(name, []).append(
            TraitAliasTemplate(
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


def _reference_pattern(names: Sequence[str]) -> re.Pattern[str]:
    key = tuple(names)
    pattern = _REFERENCE_RE_CACHE.get(key)
    if pattern is None:
        pattern = re.compile(
            r"(?<![\w.])(?P<qualifier>(?:::\s*)?(?:[A-Za-z_]\w*\s*::\s*)*)"
            rf"(?P<name>{'|'.join(re.escape(name) for name in names)})\s*<"
        )
        _REFERENCE_RE_CACHE[key] = pattern
    return pattern


def _expand_once(
    preprocessor: Any, text: str, aliases: Mapping[str, Sequence[TraitAliasTemplate]]
) -> str:
    names = [name for name in sorted(aliases) if name in text]
    if not names:
        return text
    pattern = _reference_pattern(names)
    pieces: list[str] = []
    position = 0
    while True:
        match = pattern.search(text, position)
        if match is None:
            break
        angle_start = match.end() - 1
        # _find_matching_angle skips a `>` inside parentheses (`Int<(N > 1)>`).
        angle_end = preprocessor._find_matching_angle(text, angle_start)
        if angle_end is None:
            break
        arguments = [
            argument.strip()
            for argument in preprocessor._split_top_level_commas(
                text[angle_start + 1 : angle_end]
            )
        ]
        candidates = [
            alias
            for alias in aliases[match.group("name")]
            if len(alias.parameters) == len(arguments)
            and _qualifier_matches(match.group("qualifier"), alias.namespace)
        ]
        if len(candidates) != 1 or not all(arguments):
            pieces.append(text[position : angle_start + 1])
            position = angle_start + 1
            continue
        alias = candidates[0]
        # Arguments are substituted as written: the trait compares canonical
        # spellings, and both operands come from the same instantiation.
        bindings = dict(zip(alias.parameters, arguments))
        target = preprocessor._replace_identifiers(alias.target, bindings)
        target = re.sub(r"^\s*typename\s+", "", target)
        pieces.append(text[position : match.start()])
        pieces.append(target)
        position = angle_end + 1
    pieces.append(text[position:])
    return "".join(pieces)


def expand_trait_alias_templates(
    preprocessor: Any,
    text: str,
    aliases: Mapping[str, Sequence[TraitAliasTemplate]],
) -> str:
    """Replace alias-template references in ``text`` by the types they denote."""

    if not aliases:
        return text
    for _ in range(_MAX_EXPANSION_ROUNDS):
        expanded = _expand_once(preprocessor, text, aliases)
        if expanded == text:
            break
        text = expanded
    return text

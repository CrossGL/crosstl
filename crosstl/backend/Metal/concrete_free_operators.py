"""Name namespace-scope, non-template Metal free operators as callable helpers.

The Metal parser has no declarator for ``operator+``, so it skips a free operator definition. Template free operators
are instantiated by the preprocessor into ``crosstl_metal_operator_<label>__<types>`` functions, which the code
generator finds by that prefix and binds to ``a + b`` through source overload ranking. A concrete definition such as
``cplx operator+(cplx a, cplx b) { ... }`` had no such step: the parser dropped it and ``a + b`` fell back to the
built-in operator, discarding the user's definition.

:func:`materialize_concrete_free_operators` gives every concrete free operator at namespace scope the same helper
form, so its expressions resolve to the source definition like an instantiated template's do. A compound operator
(``operator+=``) gets the value form the template path uses, which callers lower to ``x = helper(x, y)``. A
definition whose shape the helper form cannot carry, such as a device reference operand or a direct call of an
operator the pass has not named, is left in place.
"""

import re
from typing import Any, List, Optional, Set, Tuple

_OPERATOR_RE = re.compile(
    r"\boperator\s*(?P<operator>==|!=|<=|>=|[+\-*/%]=|[+\-*/%<>])\s*\("
)
_DIRECT_OPERATOR_CALL_RE = re.compile(
    r"\boperator\s*(?:==|!=|<=|>=|[+\-*/%]=|[+\-*/%<>])\s*\("
)
_DECLARATION_SPECIFIERS_RE = re.compile(
    r"\b(?:constexpr|consteval|inline|static|extern)\b"
)
_GNU_ATTRIBUTE_RE = re.compile(r"__attribute__\s*\(\((?:[^()]|\([^()]*\))*\)\)")
_STANDARD_ATTRIBUTE_RE = re.compile(r"\[\[.*?\]\]", re.DOTALL)
# Templates, hidden friends, qualified member definitions, aliases and anything
# that reads as an expression are someone else's form.
_EXCLUDED_PREFIX_RE = re.compile(
    r"\b(?:template|friend|typedef|using|return|operator)\b|#|::\s*$|[=()]"
)


def _namespace_scope_matches(
    masked: str, matches: List["re.Match[str]"], namespace_opens: Set[int]
) -> List["re.Match[str]"]:
    """The ``matches`` that no brace encloses except namespace bodies, found in one sweep over the braces."""
    selected: List["re.Match[str]"] = []
    stack: List[bool] = []
    other_open = 0
    braces = re.finditer(r"[{}]", masked)
    brace = next(braces, None)
    for match in matches:
        while brace is not None and brace.start() < match.start():
            if brace.group() == "{":
                is_namespace = brace.start() in namespace_opens
                stack.append(is_namespace)
                other_open += not is_namespace
            elif stack:
                other_open -= not stack.pop()
            brace = next(braces, None)
        if other_open == 0:
            selected.append(match)
    return selected


def _declaration_start(masked: str, position: int) -> int:
    """The first position of the declaration that contains ``position``: just after the preceding ``;``, ``{`` or ``}``."""
    return max(masked.rfind(delimiter, 0, position) for delimiter in ";{}") + 1


def _return_type(preprocessor: Any, prefix: str) -> str:
    text = _STANDARD_ATTRIBUTE_RE.sub(" ", prefix)
    text = _GNU_ATTRIBUTE_RE.sub(" ", text)
    text = _DECLARATION_SPECIFIERS_RE.sub(" ", text)
    return preprocessor._normalize_template_argument_text(text).strip()


def _helper_definition(
    preprocessor: Any,
    code: str,
    masked: str,
    match: "re.Match[str]",
    emitted: Set[str],
    anonymous_spans: List[Tuple[int, int]],
) -> Optional[Tuple[int, int, str]]:
    """The replacement that turns the operator definition at ``match`` into a named helper, or ``None``."""
    operator = match.group("operator")
    start = _declaration_start(masked, match.start())
    masked_prefix = masked[start : match.start()]
    specifiers = _GNU_ATTRIBUTE_RE.sub(
        " ", _STANDARD_ATTRIBUTE_RE.sub(" ", masked_prefix)
    )
    if not specifiers.strip() or _EXCLUDED_PREFIX_RE.search(specifiers):
        return None
    start += len(masked_prefix) - len(masked_prefix.lstrip())
    open_paren = match.end() - 1
    close_paren = preprocessor._find_matching_delimiter(code, open_paren, "(", ")")
    if close_paren is None:
        return None
    body_open = masked.find("{", close_paren + 1)
    if (
        body_open == -1
        or _STANDARD_ATTRIBUTE_RE.sub("", masked[close_paren + 1 : body_open]).strip()
    ):
        # A declaration without a body, or a trailing specifier this pass does not model.
        return None
    body_end = preprocessor._find_matching_brace(code, body_open)
    if body_end is None:
        return None
    parameters = code[open_paren + 1 : close_paren].strip()
    parameter_types = tuple(preprocessor._parameter_declared_types(parameters))
    if len(parameter_types) not in {1, 2} or any(not item for item in parameter_types):
        return None
    if len(parameter_types) == 1 and operator not in {"-", "+"}:
        return None
    return_type = _return_type(preprocessor, code[start : match.start()])
    if not return_type:
        return None
    body = preprocessor._rewrite_materialized_free_operator_calls(
        code[body_open + 1 : body_end - 1], parameters, parameter_types, emitted
    )
    if _DIRECT_OPERATOR_CALL_RE.search(preprocessor._mask_comments_and_literals(body)):
        return None
    if operator in preprocessor._COMPOUND_FREE_OPERATORS:
        value_form = preprocessor._compound_free_operator_value_form(
            return_type, parameters, parameter_types, body
        )
        if value_form is None:
            return None
        return_type, parameters, parameter_types = value_form
    helper_name = preprocessor._free_operator_helper_name(operator, parameter_types)
    if helper_name in emitted:
        # Equal spellings in different namespaces; the code generator ranks
        # candidates by parameter type, so the name only has to be unique.
        suffix = 2
        while f"{helper_name}__{suffix}" in emitted:
            suffix += 1
        helper_name = f"{helper_name}__{suffix}"
    emitted.add(helper_name)
    internal = re.search(r"\bstatic\b", masked_prefix) is not None or any(
        span_start <= start < span_end for span_start, span_end in anonymous_spans
    )
    linkage = "static inline" if internal else "inline"
    signature = f"{linkage} {return_type} {helper_name}({parameters})"
    # Keep the body on its source lines so later diagnostics point at them.
    signature += "\n" * max(
        0, code.count("\n", start, body_open) - signature.count("\n")
    )
    return start, body_end, f"{signature} {{{body}}}"


def materialize_concrete_free_operators(preprocessor: Any, code: str) -> str:
    """Rewrite each namespace-scope, non-template free operator definition of ``code`` as a named helper."""
    if "operator" not in code or _OPERATOR_RE.search(code) is None:
        return code
    masked = preprocessor._mask_comments_and_literals(code)
    matches = list(_OPERATOR_RE.finditer(masked))
    if not matches:
        return code
    namespace_spans = preprocessor._find_namespace_spans(code)
    anonymous_spans = (
        preprocessor._source_analysis(code).anonymous_namespace_spans or []
    )
    namespace_opens = {span_start - 1 for span_start, _end, _name in namespace_spans}
    emitted: Set[str] = set()
    replacements: List[Tuple[int, int, str]] = []
    for match in _namespace_scope_matches(masked, matches, namespace_opens):
        replacement = _helper_definition(
            preprocessor, code, masked, match, emitted, anonymous_spans
        )
        if replacement is not None:
            replacements.append(replacement)
    if not replacements:
        return code
    return preprocessor._apply_text_replacements(code, replacements)

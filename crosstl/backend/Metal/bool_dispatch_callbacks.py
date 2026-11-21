"""Instantiate generic-lambda callbacks of a verified Boolean dispatch helper.

Compute kernels written against compile-time flags often turn a run-time ``bool`` into an integral constant with a
helper such as::

    template <typename F>
    void dispatch_bool(bool v, F f) {
      if (v) {
        f(true_type{});
      } else {
        f(false_type{});
      }
    }

    dispatch_bool(n % 32 == 0, [&](auto kAligned) {
      load<kAligned.value>(src, n);
    });

Each call of the generic lambda instantiates its body with the parameter bound to ``true_type`` or ``false_type``
([expr.prim.lambda.closure]), so ``kAligned.value`` is a constant expression there and may appear in ``if constexpr``
and as a template argument. This module replaces such a call statement by its two instantiations::

    { if ((n % 32 == 0)) { load<true>(src, n); } else { load<false>(src, n); } }

before template materialization, so the template-argument and ``if constexpr`` passes see concrete values.

The rewrite is applied only when it provably keeps the program's behaviour:

- The helper is the unique function of its name, takes ``(bool, F)`` with ``F`` its only template parameter, and its
  body is exactly the two-branch call above. The tags are ``true_type``/``false_type``, ``bool_constant<B>`` or
  ``integral_constant<bool, B>``; a tag type, ``bool_constant`` alias or ``integral_constant`` struct that the source
  declares itself must have the standard definition. The helper converts its condition to ``bool`` once, as the
  ``if`` does.
- The callback is a lambda with the ``[&]`` default capture and one ``auto`` parameter, so the body names the same
  entities, and its writes reach the same objects, once inlined. A body containing ``return`` (which would leave the
  enclosing function instead of the lambda) is left alone, and so is one that redeclares the parameter or contains a
  lambda with a parameter of the same name.
- Each use of the parameter is either ``p.value`` in a value context (not an operand of ``&``, ``sizeof``,
  ``decltype``, assignment or increment), or a bare ``p`` that is a leaf of ``!``, ``&&`` and ``||`` in an ``if`` or
  ``while`` condition, where only its truth value matters. A bare use needs the source's own ``integral_constant``
  with a conversion operator that returns ``value``; Metal's library tags have no such conversion. The source may
  overload ``&&`` and ``||`` for integral constants only with the standard shapes (``tv && uv`` on two constants, or
  ``true_type || T`` and ``false_type && T`` short circuits), which give the logical value. An overloaded operator
  evaluates its right operand where the built-in one may skip it, so under such overloads a bare left operand needs
  a right operand without side effects (an identifier, ``x.value`` or a Boolean literal).

A callback that synchronizes (it calls ``threadgroup_barrier``, ``simdgroup_barrier`` or a ``simd_``/``quad_``
function) and reads its parameter only as a leaf of ``if``, ``if constexpr`` and ``while`` conditions is inlined
once instead, with the parameter read from a ``bool`` the call's condition initializes, so its barriers and
collectives stay in the caller's control flow::

    { const bool kAligned_value = static_cast<bool>((n % 32 == 0)); ... if (kAligned_value) { ... } ... }

Calls that miss any condition are left unchanged for the existing callback lowering, which keeps its typed
diagnostics. Nested calls expand from the inside out, and the number of instantiated bodies is bounded by the
preprocessor's ``max_template_specializations`` like other compile-time expansions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .preprocessor import MetalPreprocessor

_QUALIFIER = r"(?:[A-Za-z_]\w*\s*::\s*)*"
_BOOL_TAG = (
    _QUALIFIER
    + r"(?:(?P<named>true_type|false_type)"
    + r"|bool_constant\s*<\s*(?P<bool_constant>true|false)\s*>"
    + r"|integral_constant\s*<\s*bool\s*,\s*(?P<integral_constant>true|false)\s*>)"
)
_HELPER = re.compile(
    r"template\s*<\s*(?:typename|class)\s+(?P<function_type>[A-Za-z_]\w*)\s*>\s*"
    r"(?:(?:inline|static|constexpr)\s+|__attribute__\s*\(\([^()]*\)\)\s*)*"
    r"void\s+(?P<name>[A-Za-z_]\w*)\s*\(\s*(?:const\s+)?bool\s+(?P<condition>[A-Za-z_]\w*)\s*,\s*"
    r"(?:const\s+)?(?:thread\s+)?(?P=function_type)\s*(?:&&|&)?\s*(?P<function>[A-Za-z_]\w*)\s*\)\s*\{"
)
_LAMBDA = re.compile(
    r"\s*\[\s*&\s*\]\s*\(\s*(?:const\s+)?auto\s*(?:const\s*)?(?:&&|&)?\s*"
    r"(?P<parameter>[A-Za-z_]\w*)\s*\)\s*(?:->\s*void\s*)?\{"
)
_LOGICAL_OPERATOR = re.compile(r"\boperator\s*(?P<operator>&&|\|\||!)\s*\(")
_INTEGRAL_CONSTANT_SPELLING = re.compile(
    r"\b(?:integral_constant|bool_constant|true_type|false_type)\b"
)


@dataclass
class _Helper:
    name: str
    span: tuple[int, int]
    # Whether the source overloads `&&` or `||` for integral constants.
    logical_overloads: bool
    # Whether a bare tag converts to its value: the library's integral_constant,
    # or a source definition whose conversion operator returns `value`.
    bare_conversion: bool


@dataclass
class _Budget:
    limit: int
    used: int = 0


def expand_bool_dispatch_callbacks(preprocessor: MetalPreprocessor, source: str) -> str:
    """Replace each provable helper call statement by its two instantiations."""
    if "auto" not in source or "[" not in source:
        return source
    masked = _masked(preprocessor, source)
    operators = _logical_operator_shapes(preprocessor, masked)
    if operators is None or not _tag_definitions_are_standard(preprocessor, masked):
        return source
    expanded: set[str] = set()
    while True:
        # Spans move as calls expand, so the helpers are found afresh each time.
        helper = next(
            (
                candidate
                for candidate in _helpers(preprocessor, source, bool(operators))
                if candidate.name not in expanded
            ),
            None,
        )
        if helper is None:
            return source
        expanded.add(helper.name)
        budget = _Budget(preprocessor.max_template_specializations)
        source = _expand_calls(
            preprocessor, source, helper, budget, depth=0, helper_span=helper.span
        )


def _masked(preprocessor: MetalPreprocessor, text: str) -> str:
    """``text`` with comments and literals blanked, keeping offsets."""
    characters = list(text)
    for start, end in preprocessor._find_comment_and_literal_spans(text):
        for index in range(start, end):
            if characters[index] != "\n":
                characters[index] = " "
    return "".join(characters)


def _tag_definitions_are_standard(preprocessor: MetalPreprocessor, masked: str) -> bool:
    """Whether every tag the source declares itself has the standard definition."""
    if re.search(r"\bstruct\s+integral_constant\b", masked) and not (
        preprocessor._integral_constant_contract_verified
    ):
        return False
    if re.search(
        r"\b(?:struct|class)\s+bool_constant\b|\btypedef\b[^;]*\bbool_constant\s*;",
        masked,
    ):
        return False
    for match in re.finditer(r"\busing\s+bool_constant\s*=", masked):
        prefix = masked[: match.start()].rstrip()
        if not re.search(r"template\s*<\s*bool\s+(?P<value>[A-Za-z_]\w*)\s*>$", prefix):
            return False
        value = re.search(r"<\s*bool\s+([A-Za-z_]\w*)\s*>$", prefix).group(1)
        definition = re.compile(
            rf"\s*{_QUALIFIER}integral_constant\s*<\s*bool\s*,\s*{re.escape(value)}\s*>\s*;"
        )
        if not definition.match(masked, match.end()):
            return False
    for tag, value in (("true_type", "true"), ("false_type", "false")):
        if re.search(
            rf"\b(?:struct|class)\s+{tag}\b|\btypedef\b[^;]*\b{tag}\s*;", masked
        ):
            return False
        definition = re.compile(
            rf"\s*{_QUALIFIER}(?:bool_constant\s*<\s*{value}\s*>"
            rf"|integral_constant\s*<\s*bool\s*,\s*{value}\s*>)\s*;"
        )
        for match in re.finditer(rf"\busing\s+{tag}\s*=", masked):
            if not definition.match(masked, match.end()):
                return False
    return True


def _logical_operator_shapes(
    preprocessor: MetalPreprocessor, masked: str
) -> set[str] | None:
    """The verified logical overload shapes on integral constants, or ``None``.

    ``None`` means the source overloads ``!``, ``&&`` or ``||`` for integral constants in some other way, so a bare
    parameter in such an expression may not have its built-in meaning.
    """
    shapes: set[str] = set()
    for match in _LOGICAL_OPERATOR.finditer(masked):
        open_paren = match.end() - 1
        close_paren = preprocessor._find_matching_delimiter(
            masked, open_paren, "(", ")"
        )
        if close_paren is None:
            return None
        parameters = masked[open_paren + 1 : close_paren]
        if not _INTEGRAL_CONSTANT_SPELLING.search(parameters):
            continue
        operator = match.group("operator")
        body_open = masked.find("{", close_paren)
        if body_open == -1 or masked[close_paren:body_open].strip(" \t\n").endswith(
            ";"
        ):
            return None
        body_close = preprocessor._find_matching_delimiter(masked, body_open, "{", "}")
        if body_close is None:
            return None
        body = re.sub(r"\s+", "", masked[body_open + 1 : body_close])
        shape = _logical_overload_shape(operator, parameters, body)
        if shape is None:
            return None
        shapes.add(shape)
    return shapes


def _logical_overload_shape(operator: str, parameters: str, body: str) -> str | None:
    """``"binary"`` or ``"short-circuit"`` for an overload with the logical value."""
    if operator == "!":
        return None
    compact = re.sub(r"\s+", "", parameters)
    # template <typename T, T tv, typename U, U uv>
    # auto operator&&(integral_constant<T, tv>, integral_constant<U, uv>) {
    #   constexpr auto res = tv && uv; ... return integral_constant<..., res>{}; }
    binary = re.fullmatch(
        rf"{_QUALIFIER}integral_constant<\w+,(?P<tv>\w+)>\w*,"
        rf"{_QUALIFIER}integral_constant<\w+,(?P<uv>\w+)>\w*",
        compact,
    )
    if binary is not None:
        result = re.match(
            rf"constexprauto(?P<res>\w+)={binary.group('tv')}{re.escape(operator)}"
            rf"{binary.group('uv')};",
            body,
        )
        if result is None:
            return None
        rest = re.sub(r"^using\w+=[^;]*;", "", body[result.end() :])
        returned = (
            rf"return{_QUALIFIER}integral_constant<[^;]*,{result.group('res')}>\{{\}};"
        )
        return "binary" if re.fullmatch(returned, rest) else None
    # `true_type || T` is true and `false_type && T` is false whatever T is.
    tag = "true_type" if operator == "||" else "false_type"
    short_circuit = re.fullmatch(
        rf"(?:{_QUALIFIER}{tag}\w*,\w+|\w+,{_QUALIFIER}{tag}\w*)", compact
    )
    if short_circuit is None:
        return None
    if not re.fullmatch(rf"return{_QUALIFIER}{tag}\{{\}};", body):
        return None
    return "short-circuit"


def _helpers(
    preprocessor: MetalPreprocessor, source: str, logical_overloads: bool
) -> list[_Helper]:
    masked = _masked(preprocessor, source)
    # Metal's own integral_constant has no conversion to its value, so a bare
    # tag needs the source's definition with the verified conversion operator.
    bare_conversion = bool(
        re.search(r"\bstruct\s+integral_constant\b", masked)
        and preprocessor._integral_constant_conversion_contract_verified
    )
    templates = preprocessor._find_template_functions(source)
    spans = [template.span for template in templates]
    non_template_names = {
        function.name
        for function in preprocessor._find_non_template_function_definitions(
            source, list(spans)
        )
    }
    helpers = []
    for template in templates:
        if len(template.template_parameters) != 1:
            continue
        if sum(other.name == template.name for other in templates) != 1:
            continue
        if template.name in non_template_names:
            continue
        start, end = template.span
        match = _HELPER.match(masked, start)
        if match is None or match.group("name") != template.name:
            continue
        body_open = match.end() - 1
        body_close = preprocessor._find_matching_delimiter(masked, body_open, "{", "}")
        if body_close is None or masked[body_close + 1 : end].strip():
            continue
        tags = _helper_dispatch_tags(
            masked[body_open + 1 : body_close],
            match.group("condition"),
            match.group("function"),
        )
        if tags is not None:
            library_tags = any(re.match(r"\s*metal\s*::", tag) for tag in tags)
            helpers.append(
                _Helper(
                    template.name,
                    template.span,
                    logical_overloads,
                    bare_conversion and not library_tags,
                )
            )
    return helpers


def _helper_dispatch_tags(
    body: str, condition: str, function: str
) -> tuple[str, str] | None:
    """The tags of ``if (v) { f(true tag{}); } else { f(false tag{}); }``, if ``body`` is that."""
    call = rf"{re.escape(function)}\s*\(\s*(?P<tag>{_BOOL_TAG})\s*\{{\s*\}}\s*\)\s*;"
    contract = re.compile(
        rf"\s*if\s*\(\s*{re.escape(condition)}\s*\)\s*\{{\s*(?P<first>.*?)\s*\}}"
        rf"\s*else\s*\{{\s*(?P<second>.*?)\s*\}}\s*",
        re.DOTALL,
    )
    match = contract.fullmatch(body)
    if match is None:
        return None
    tags = []
    values = []
    for branch in (match.group("first"), match.group("second")):
        tag = re.fullmatch(call, branch)
        if tag is None:
            return None
        named = tag.group("named")
        spelled = tag.group("bool_constant") or tag.group("integral_constant")
        values.append(named == "true_type" if named else spelled == "true")
        tags.append(tag.group("tag"))
    return (tags[0], tags[1]) if values == [True, False] else None


def _expand_calls(
    preprocessor: MetalPreprocessor,
    text: str,
    helper: _Helper,
    budget: _Budget,
    *,
    depth: int,
    helper_span: tuple[int, int] | None = None,
) -> str:
    if helper.name not in text:
        return text
    masked = _masked(preprocessor, text)
    pattern = re.compile(rf"\b{re.escape(helper.name)}\s*\(")
    output: list[str] = []
    cursor = 0
    position = 0
    while True:
        match = pattern.search(masked, position)
        if match is None:
            break
        position = match.end()
        if helper_span is not None and helper_span[0] <= match.start() < helper_span[1]:
            continue
        if _is_member_call(masked, match.start()):
            continue
        expansion = _expand_call(
            preprocessor, text, masked, match, helper, budget, depth=depth
        )
        if expansion is None:
            continue
        statement_start, statement_end, replacement = expansion
        output.append(text[cursor:statement_start])
        output.append(replacement)
        cursor = position = statement_end
    output.append(text[cursor:])
    return "".join(output)


def _is_member_call(masked: str, name_start: int) -> bool:
    previous = name_start - 1
    while previous >= 0 and masked[previous].isspace():
        previous -= 1
    return previous >= 0 and (
        masked[previous] == "." or masked[max(0, previous - 1) : previous + 1] == "->"
    )


def _expand_call(preprocessor, text, masked, match, helper, budget, *, depth):
    name_start = match.start()
    window_start = max(0, name_start - 256)
    qualifier = re.search(
        r"(?:::\s*)?(?:[A-Za-z_]\w*\s*::\s*)*$", masked[window_start:name_start]
    )
    qualified_start = window_start + qualifier.start() if qualifier else name_start
    open_paren = match.end() - 1
    close_paren = preprocessor._find_matching_delimiter(masked, open_paren, "(", ")")
    if close_paren is None:
        return None
    if not preprocessor._const_for_loop_is_standalone_statement(
        text, qualified_start, close_paren + 1
    ):
        return None
    semicolon = masked.index(";", close_paren + 1)
    arguments = _split_arguments(text, masked, open_paren, close_paren)
    if len(arguments) != 2:
        return None
    condition, callback = arguments
    lambda_match = _LAMBDA.match(callback)
    if lambda_match is None:
        return None
    body_open = lambda_match.end() - 1
    body_close = preprocessor._find_matching_delimiter(callback, body_open, "{", "}")
    if body_close is None or callback[body_close + 1 :].strip():
        return None
    if depth >= preprocessor.max_expansion_depth:
        return None
    parameter = lambda_match.group("parameter")
    before = budget.used
    body = _expand_calls(
        preprocessor,
        callback[body_open + 1 : body_close],
        helper,
        budget,
        depth=depth + 1,
    )
    inlinable = _body_is_inlinable(preprocessor, body, parameter)
    runtime = (
        _runtime_flag_form(preprocessor, body, parameter, condition, helper)
        if inlinable
        else None
    )
    if runtime is not None:
        # One body, which the nested calls' instantiations are part of.
        budget.used += 1
        if budget.used > budget.limit:
            _raise_expansion_limit(preprocessor, helper.name, budget)
        return qualified_start, semicolon + 1, runtime
    substituted = []
    if inlinable:
        for value in (True, False):
            instantiated = _substitute(preprocessor, body, parameter, value, helper)
            if instantiated is not None:
                substituted.append(instantiated)
    if len(substituted) != 2:
        # The call stays; the scan revisits its nested calls on its own.
        budget.used = before
        return None
    # Both copies carry the bodies the nested calls instantiated.
    budget.used = before + 2 * (budget.used - before + 1)
    if budget.used > budget.limit:
        _raise_expansion_limit(preprocessor, helper.name, budget)
    replacement = (
        f"{{ if (({condition.strip()})) {{{substituted[0]}}} "
        f"else {{{substituted[1]}}} }}"
    )
    return qualified_start, semicolon + 1, replacement


def _split_arguments(text: str, masked: str, open_paren: int, close_paren: int):
    """The call's arguments, split at commas outside brackets.

    A comma inside template arguments (``f<a, b>(x)``) also splits, which only
    makes the call fail the two-argument check and stay unchanged.
    """
    arguments = []
    depth = 0
    start = open_paren + 1
    for index in range(open_paren + 1, close_paren):
        character = masked[index]
        if character in "([{":
            depth += 1
        elif character in ")]}":
            depth -= 1
        elif character == "," and depth == 0:
            arguments.append(text[start:index])
            start = index + 1
    arguments.append(text[start:close_paren])
    return arguments


def _body_is_inlinable(preprocessor, body: str, parameter: str) -> bool:
    masked = _masked(preprocessor, body)
    nested = preprocessor._const_for_loop_nested_lambda_spans(body)
    outside = list(masked)
    for start, end in nested:
        for index in range(start, end):
            outside[index] = " "
    if re.search(r"\breturn\b", "".join(outside)):
        return False
    if parameter in preprocessor._local_variable_names(body):
        return False
    for start, end in nested:
        header_end = masked.find("{", start, end)
        if header_end != -1 and re.search(
            rf"\b{re.escape(parameter)}\b", masked[start:header_end]
        ):
            return False
    return True


def _substitute(preprocessor, body, parameter, value, helper) -> str | None:
    masked = _masked(preprocessor, body)
    literal = "true" if value else "false"
    pieces: list[str] = []
    cursor = 0
    for match in re.finditer(rf"\b{re.escape(parameter)}\b", masked):
        start = match.start()
        if preprocessor._is_member_identifier_context(masked, start):
            continue
        member = re.match(r"\s*\.\s*value\b", masked[match.end() :])
        if member is not None:
            end = match.end() + member.end()
            if not _value_use_is_safe(masked, start, end):
                return None
        elif _bare_use_is_condition_operand(masked, start, match.end(), helper):
            end = match.end()
        else:
            return None
        pieces.append(body[cursor:start])
        pieces.append(literal)
        cursor = end
    pieces.append(body[cursor:])
    return "".join(pieces)


def _value_use_is_safe(masked: str, start: int, end: int) -> bool:
    """Whether ``p.value`` at ``start:end`` is read as a value.

    Taking its address, its size or its type, or writing to it, needs the
    static member itself. A preceding ``&&`` is a logical operator, not ``&``.
    """
    prefix = masked[:start].rstrip()
    while prefix.endswith("("):
        prefix = prefix[:-1].rstrip()
    if re.search(r"(?:(?<!&)&|\+\+|--)$", prefix):
        return False
    if re.search(r"\b(?:alignof|decltype|sizeof|typeid)$", prefix):
        return False
    suffix = masked[end:].lstrip()
    if suffix.startswith(("++", "--", ".", "->", "(", "[")):
        return False
    return re.match(r"(?:(?:<<|>>|[+\-*/%&|^])?=)(?!=)", suffix) is None


_LOGICAL_LEFT = re.compile(r"(?:\(|!|&&|\|\|)$")
_LOGICAL_RIGHT = re.compile(r"\s*(?:\)|&&|\|\|)")
_CONDITION_KEYWORD = re.compile(r"(?:\bif(?:\s+constexpr)?|\bwhile)$")
_SIDE_EFFECT_FREE_OPERAND = re.compile(
    r"\s*(?:true|false|[A-Za-z_]\w*(?:\s*\.\s*value)?)\s*(?:\)|&&|\|\|)"
)


def _bare_use_is_condition_operand(
    masked: str, start: int, end: int, helper: _Helper
) -> bool:
    """Whether a bare use is a leaf of ``!``/``&&``/``||`` in an ``if`` or ``while`` condition.

    There the tag is only converted to ``bool``, so its value is all that matters.
    """
    if not helper.bare_conversion:
        return False
    return _condition_of_operand(masked, start, end, helper, bare=True) is not None


def _condition_of_operand(
    masked: str, start: int, end: int, helper: _Helper, *, bare: bool
) -> int | None:
    """The opening parenthesis of the ``if`` or ``while`` condition whose logical leaf is ``start:end``."""
    if not _LOGICAL_LEFT.search(masked[:start].rstrip()):
        return None
    right = _LOGICAL_RIGHT.match(masked, end)
    if right is None:
        return None
    if (
        bare
        and helper.logical_overloads
        and masked[right.start() : right.end()].strip() != ")"
        and _SIDE_EFFECT_FREE_OPERAND.match(masked, right.end()) is None
    ):
        # An overloaded operator evaluates its right operand, which the built-in
        # operator may skip, so that operand must not have effects.
        return None
    position = start
    while True:
        open_paren = _enclosing_open_paren(masked, position)
        if open_paren is None:
            return None
        prefix = masked[:open_paren].rstrip()
        if _CONDITION_KEYWORD.search(prefix):
            return open_paren
        # Any other parentheses must only group logical operands.
        if not _LOGICAL_LEFT.search(prefix):
            return None
        close_paren = _matching_close_paren(masked, open_paren)
        if close_paren is None or _LOGICAL_RIGHT.match(masked, close_paren + 1) is None:
            return None
        position = open_paren


# Calls that synchronize the threadgroup or exchange values between lanes.
_SYNCHRONIZING_CALL = re.compile(
    r"\b(?:threadgroup_barrier|simdgroup_barrier|simd_\w+|quad_\w+)\s*\("
)
_CONSTEXPR_IF = re.compile(r"\bif(?P<constexpr>\s+constexpr)$")


def _runtime_flag_form(preprocessor, body, parameter, condition, helper) -> str | None:
    """The callback inlined once, with its tag read from a run-time flag, if that is equivalent.

    Instantiating a body that synchronizes for each value puts its barriers and collectives under the
    dispatch condition, which may differ between the SIMD-groups of a threadgroup. Every invocation then
    reaches the same barriers only through different branches, which targets that need uniform control
    flow for barriers and emulated collectives refuse. When every use of the tag is a leaf of an ``if``,
    ``if constexpr`` or ``while`` condition, the body reads the same value from a ``bool`` the helper's
    condition initializes once::

        { const bool k_value = static_cast<bool>((c)); ... if (k_value) { ... } ... }

    Such an ``if constexpr`` becomes an ``if``, so both of its branches are instantiated, as the body's
    other ``if`` statements are. Bodies with any other use of the tag (a template argument, a constant
    initializer, a nested lambda's use), and bodies that do not synchronize, keep their two
    instantiations.
    """
    masked = _masked(preprocessor, body)
    if _SYNCHRONIZING_CALL.search(masked) is None:
        return None
    nested = preprocessor._const_for_loop_nested_lambda_spans(body)
    edits: list[tuple[int, int, str]] = []
    constexpr_spans: set[tuple[int, int]] = set()
    flag = _fresh_flag_name(masked + "\n" + condition, parameter)
    for match in re.finditer(rf"\b{re.escape(parameter)}\b", masked):
        start = match.start()
        if preprocessor._is_member_identifier_context(masked, start):
            continue
        if any(
            lambda_start <= start < lambda_end for lambda_start, lambda_end in nested
        ):
            return None
        member = re.match(r"\s*\.\s*value\b", masked[match.end() :])
        if member is not None:
            end = match.end() + member.end()
        elif helper.bare_conversion:
            end = match.end()
        else:
            return None
        open_paren = _condition_of_operand(
            masked, start, end, helper, bare=member is None
        )
        if open_paren is None:
            return None
        prefix = masked[:open_paren].rstrip()
        keyword = _CONSTEXPR_IF.search(prefix)
        if keyword is not None:
            constexpr_spans.add(keyword.span("constexpr"))
        edits.append((start, end, flag))
    if not edits:
        return None
    edits.extend((start, end, "") for start, end in constexpr_spans)
    pieces: list[str] = []
    cursor = 0
    for start, end, replacement in sorted(edits):
        pieces.append(body[cursor:start])
        pieces.append(replacement)
        cursor = end
    pieces.append(body[cursor:])
    return (
        f"{{ const bool {flag} = static_cast<bool>(({condition.strip()})); "
        f"{''.join(pieces)} }}"
    )


def _fresh_flag_name(text: str, parameter: str) -> str:
    """``<parameter>_value``, suffixed until no identifier of ``text`` has the name."""
    names = set(re.findall(r"[A-Za-z_]\w*", text))
    flag = f"{parameter}_value"
    index = 1
    while flag in names:
        flag = f"{parameter}_value_{index}"
        index += 1
    return flag


def _enclosing_open_paren(masked: str, position: int) -> int | None:
    depth = 0
    brackets = 0
    for index in range(position - 1, -1, -1):
        character = masked[index]
        if character == ")":
            depth += 1
        elif character == "(":
            if depth == 0:
                return index if brackets == 0 else None
            depth -= 1
        elif character == "]":
            brackets += 1
        elif character == "[":
            if brackets == 0:
                return None
            brackets -= 1
        elif character in "{};" and depth == 0:
            return None
    return None


def _matching_close_paren(masked: str, open_paren: int) -> int | None:
    depth = 0
    for index in range(open_paren, len(masked)):
        if masked[index] == "(":
            depth += 1
        elif masked[index] == ")":
            depth -= 1
            if depth == 0:
                return index
    return None


def _raise_expansion_limit(preprocessor, name: str, budget: _Budget) -> None:
    from .preprocessor import MetalTemplateSpecializationError

    suggested_action = (
        "raise max_template_specializations for this source pattern or reduce "
        "the nested Boolean dispatch"
    )
    raise MetalTemplateSpecializationError(
        f"Metal Boolean dispatch expansion limit exceeded for '{name}'; "
        f"{budget.used} callback instantiations requested, limit {budget.limit} "
        f"from {preprocessor.template_specialization_limit_source}. "
        f"Suggested action: {suggested_action}.",
        limit=budget.limit,
        limit_source=preprocessor.template_specialization_limit_source,
        unique_specialization_count=budget.used,
        required_work_items=budget.used,
        requested_signature=name,
        suggested_action=suggested_action,
    )

"""Integral-constant locals and static-member template arguments in concrete Metal function bodies.

``integral_constant_arithmetic`` folds ``a op b`` when both operands are integral constants. This module supplies two
more kinds of operand, and runs the fold over every concrete function definition, entry points included.

* **Block-scope locals.** A local declared as ``const Int<1> mm{};``, ``Int<1> mm = Int<1>{};`` or
  ``constexpr auto mm = Int<1>{};`` has the type ``Int<1>``. An integral constant has no non-static data members, so
  the result of ``mm * Int<4>{}`` depends only on the operand types and the operator template selects it at compile
  time ([over.match.oper]). Each use is bound to its declaration by C++ block scope ([basic.scope.block]): the latest
  declaration before the use whose enclosing block contains the use. A name is used only when every declaration of it
  in the body has one of the recognized forms, it is not a parameter of the function, and no lambda binds it as a
  parameter, so no other declaration can hide the one the scan picks.
* **Static-member template arguments.** ``Int<Frag::kRows>`` and ``Int<4>`` name the same type when the static member
  is a constant integral expression with value 4 ([temp.type]), so such an argument is replaced by its value. Only
  arguments that do not already evaluate are touched, and only when the member resolves through a concrete struct.
  The owner may be a block-scope alias (``using F = typename Tile::Frag_t;``) and a member alias of a concrete struct;
  both name the aliased type ([dcl.typedef]), so they are resolved first.

Concrete function bodies are folded before struct members are lowered, so a call such as ``Frag::load(v, C, m)`` with
``auto m = mm * Int<4>{};`` sees ``m`` as ``Int<4>`` and can be instantiated.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Callable, Dict, Optional, Tuple

from .static_member_constants import static_member_constant_substituter

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .preprocessor import MetalPreprocessor

__all__ = [
    "fold_concrete_function_integral_constant_arithmetic",
    "integral_constant_local_atoms",
    "is_folded_integral_constant_operator",
    "is_integral_constant_operand",
    "resolve_const_for_loop_bound_constants",
    "resolve_static_member_template_arguments",
]

_TEMPLATE_ID = re.compile(
    r"(?P<owner>(?:[A-Za-z_]\w*\s*::\s*)*)(?P<name>Int|integral_constant)\s*<"
    r"(?P<arguments>[^<>;{}]*)>"
)
_TEMPLATE_ID_TEXT = r"(?:[A-Za-z_]\w*\s*::\s*)*(?:Int|integral_constant)\s*<[^<>;{}]*>"
_LOCAL_DECLARATION = re.compile(
    r"(?P<qualifiers>(?:(?:const|constexpr|constant|thread)\s+)*)"
    rf"(?P<type>auto|{_TEMPLATE_ID_TEXT})\s*"
    r"(?P<name>(?<![\w:])[A-Za-z_]\w*)\s*"
    rf"(?:\{{\s*\}}|=\s*(?P<initializer>{_TEMPLATE_ID_TEXT})\s*\{{\s*\}})?\s*;"
)
_STATEMENT_BOUNDARIES = "{;}"

_CONSTRUCTION_BEFORE = re.compile(rf"{_TEMPLATE_ID_TEXT}\s*\{{\s*\}}\s*$")
_CONSTRUCTION_AFTER = re.compile(rf"\s*(?P<type>{_TEMPLATE_ID_TEXT})\s*\{{\s*\}}")
# A construction operand is a short template-id; looking further back cannot find one that ends at the operator.
_OPERAND_WINDOW = 256
_CONST_FOR_LOOP = re.compile(r"(?<![\w:])const_for_loop\s*<")

TypeParts = Callable[[str], Optional[Tuple[str, int]]]
StaticMembers = Callable[[str, Dict[str, str]], str]
_QUALIFIED_MEMBER = re.compile(
    r"(?<![\w:])(?P<owner>[A-Za-z_]\w*(?:\s*::\s*[A-Za-z_]\w*)*)\s*::\s*(?P<member>[A-Za-z_]\w*)\b"
)
_MAX_ALIAS_STEPS = 8
_PLAIN_TYPE_NAME = re.compile(r"[A-Za-z_]\w*(?:::[A-Za-z_]\w*)*")


def fold_concrete_function_integral_constant_arithmetic(
    preprocessor: MetalPreprocessor, code: str
) -> str:
    """Fold integral-constant arithmetic in every concrete (non-template) function definition of ``code``."""
    from .integral_constant_arithmetic import (
        fold_integral_constant_function_arithmetic,
        materialized_construction_pattern,
    )

    if (
        not preprocessor._integral_constant_binary_operators
        or not preprocessor._integral_constant_contract_verified
    ):
        return code
    template_spans = preprocessor._find_template_declaration_spans(code)
    functions = preprocessor._find_non_template_function_definitions(
        code, template_spans
    )
    static_members: list[StaticMembers] = []

    def resolver() -> StaticMembers:
        if not static_members:
            structs = preprocessor._find_concrete_struct_definitions(code)
            substitute = static_member_constant_substituter(preprocessor, structs)
            member_aliases = _member_aliases_by_name(structs)
            static_members.append(
                _OwnerResolvingSubstituter(preprocessor, substitute, member_aliases)
            )
        return static_members[0]

    materialized = materialized_construction_pattern(preprocessor)
    replacements: list[tuple[int, int, str]] = []
    for function in functions:
        body_start, body_end = function.body_span
        if _TEMPLATE_ID.search(code, body_start, body_end) is None and (
            materialized is None
            or materialized.search(code, body_start, body_end) is None
        ):
            continue
        start, end = function.span
        text = code[start:end]
        folded = fold_integral_constant_function_arithmetic(
            preprocessor, text, static_members=resolver
        )
        if folded != text:
            replacements.append((start, end, folded))
    if not replacements:
        return code
    return preprocessor._apply_text_replacements(code, replacements)


def resolve_static_member_template_arguments(
    preprocessor: MetalPreprocessor,
    body: str,
    static_members: Callable[[], StaticMembers],
    type_parts: TypeParts,
) -> str:
    """Replace ``Owner::member`` arguments of integral-constant template-ids with their constant values."""
    masked = preprocessor._mask_comments_and_literals(body)
    replacements: list[tuple[int, int, str]] = []
    for match in _TEMPLATE_ID.finditer(masked):
        arguments = match.group("arguments")
        if "::" not in arguments or _is_scope_operand(masked, match.start()):
            continue
        type_text = body[match.start() : match.end()]
        if type_parts(type_text) is not None:
            continue
        substitute = static_members()
        argument_text = body[match.start("arguments") : match.end("arguments")]
        resolved = substitute(argument_text, {})
        if resolved == argument_text and isinstance(
            substitute, _OwnerResolvingSubstituter
        ):
            resolved = substitute.through_aliases(body, match.start(), argument_text)
        if resolved == argument_text:
            continue
        candidate = (
            body[match.start() : match.start("arguments")]
            + resolved
            + body[match.end("arguments") : match.end()]
        )
        if type_parts(candidate) is None:
            continue
        replacements.append(
            (match.start("arguments"), match.end("arguments"), resolved)
        )
    if not replacements:
        return body
    return preprocessor._apply_text_replacements(body, replacements)


def _member_aliases_by_name(structs: list[object]) -> dict[str, dict[str, str]]:
    """Member type aliases of each concrete struct, by qualified name and by its name when that is unique."""
    counts: dict[str, int] = {}
    for struct in structs:
        counts[struct.name] = counts.get(struct.name, 0) + 1
    aliases: dict[str, dict[str, str]] = {}
    for struct in structs:
        aliases[struct.qualified_name or struct.name] = struct.type_aliases
        if counts[struct.name] == 1:
            aliases[struct.name] = struct.type_aliases
    return aliases


class _OwnerResolvingSubstituter:
    """A static-member substituter that can first resolve type-alias owners at a position of a body."""

    def __init__(
        self,
        preprocessor: MetalPreprocessor,
        substitute: StaticMembers,
        member_aliases: dict[str, dict[str, str]],
    ) -> None:
        self._preprocessor = preprocessor
        self._substitute = substitute
        self._member_aliases = member_aliases
        self._local_aliases: dict[int, tuple] = {}

    def __call__(self, expression: str, declared_types: dict[str, str]) -> str:
        return self._substitute(expression, declared_types)

    def through_aliases(self, body: str, position: int, arguments: str) -> str:
        key = id(body)
        cached = self._local_aliases.get(key)
        if cached is None or cached[0] is not body:
            cached = (
                body,
                self._preprocessor._collect_local_type_alias_bindings(
                    body, [(0, len(body))]
                ),
            )
            self._local_aliases = {key: cached}
        local_aliases = cached[1]
        replacements: list[tuple[int, int, str]] = []
        for match in _QUALIFIED_MEMBER.finditer(arguments):
            owner = re.sub(r"\s*::\s*", "::", match.group("owner"))
            resolved = self._resolve_owner(owner, local_aliases, position)
            if resolved != owner:
                replacements.append(
                    (match.start("owner"), match.end("owner"), resolved)
                )
        if not replacements:
            return arguments
        rewritten = self._preprocessor._apply_text_replacements(arguments, replacements)
        return self._substitute(rewritten, {})

    def _resolve_owner(self, owner: str, local_aliases, position: int) -> str:
        """Follow block-scope and struct-member aliases while they name a plain, possibly qualified, type name."""
        for _ in range(_MAX_ALIAS_STEPS):
            resolved = self._preprocessor._resolve_type_aliases_at(
                owner, local_aliases, position
            )
            resolved = re.sub(r"^\s*typename\s+", "", resolved).strip()
            head, separator, tail = resolved.rpartition("::")
            if separator:
                target = self._member_aliases.get(head.strip(), {}).get(tail.strip())
                if target is not None:
                    resolved = re.sub(r"^\s*typename\s+", "", target).strip()
            if resolved == owner or not _PLAIN_TYPE_NAME.fullmatch(resolved):
                return owner
            owner = resolved
        return owner


def integral_constant_local_atoms(
    preprocessor: MetalPreprocessor,
    masked: str,
    parameter_names: set[str],
    type_parts: TypeParts,
) -> list[tuple[int, int, str]]:
    """``(start, end, type)`` of every use of a block-scope integral-constant local in ``masked``."""
    declarations = _recognized_declarations(masked, type_parts)
    if not declarations:
        return []
    names = {name for name, *_rest in declarations}
    rejected = set(parameter_names) & names
    remainder = list(masked)
    for _name, start, end, _type, _block in declarations:
        remainder[start:end] = " " * (end - start)
    other_declarations = preprocessor._local_variable_names("".join(remainder))
    for name in sorted(names - rejected):
        if name in other_declarations or preprocessor._lambda_binds_identifier(
            masked, name
        ):
            rejected.add(name)
    accepted = names - rejected
    if not accepted:
        return []
    by_name: dict[str, list[tuple[int, int, str, tuple[int, int]]]] = {}
    for name, start, end, type_text, block in declarations:
        if name in accepted:
            by_name.setdefault(name, []).append((start, end, type_text, block))
    pattern = "|".join(re.escape(name) for name in sorted(accepted))
    atoms: list[tuple[int, int, str]] = []
    for match in re.finditer(rf"(?<![\w.])(?:{pattern})\b", masked):
        if _is_scope_operand(masked, match.start()) or _is_member_operand(
            masked, match.start()
        ):
            continue
        visible = [
            (start, type_text)
            for start, end, type_text, (block_start, block_end) in by_name[
                match.group(0)
            ]
            if end <= match.start() and block_start <= match.start() < block_end
        ]
        if visible:
            atoms.append((match.start(), match.end(), max(visible)[1]))
    return atoms


def _recognized_declarations(
    masked: str, type_parts: TypeParts
) -> list[tuple[str, int, int, str, tuple[int, int]]]:
    blocks: list[tuple[int, int]] | None = None
    declarations = []
    for match in _LOCAL_DECLARATION.finditer(masked):
        start = match.start()
        previous = start - 1
        while previous >= 0 and masked[previous].isspace():
            previous -= 1
        if previous >= 0 and masked[previous] not in _STATEMENT_BOUNDARIES:
            continue
        type_text = match.group("type")
        initializer = match.group("initializer")
        if type_text == "auto":
            if initializer is None:
                continue
            type_text = initializer
        declared = type_parts(type_text)
        if declared is None:
            continue
        if initializer is not None and type_parts(initializer) != declared:
            continue
        if blocks is None:
            blocks = _brace_blocks(masked)
        declarations.append(
            (
                match.group("name"),
                start,
                match.end(),
                re.sub(r"\s+", " ", type_text).strip(),
                _innermost_block(blocks, start, len(masked)),
            )
        )
    return declarations


def _brace_blocks(masked: str) -> list[tuple[int, int]]:
    blocks: list[tuple[int, int]] = []
    stack: list[int] = []
    for index, character in enumerate(masked):
        if character == "{":
            stack.append(index)
        elif character == "}" and stack:
            blocks.append((stack.pop(), index))
    return blocks


def _innermost_block(
    blocks: list[tuple[int, int]], position: int, length: int
) -> tuple[int, int]:
    enclosing = [(start, end) for start, end in blocks if start < position < end]
    if not enclosing:
        return 0, length
    return max(enclosing)


def _is_scope_operand(masked: str, start: int) -> bool:
    prefix = masked[:start].rstrip()
    return prefix.endswith("::")


def _is_member_operand(masked: str, start: int) -> bool:
    prefix = masked[:start].rstrip()
    return prefix.endswith((".", "->"))


def is_integral_constant_operand(
    preprocessor: MetalPreprocessor, source: str, start: int, end: int
) -> bool:
    """Whether ``source[start:end]`` is an operand of a verified integral-constant operator whose other operand is
    an integral-constant construction, as in ``idx * Int<4>{}`` or ``Int<4>{} * idx``.

    Overload resolution then selects the operator template and the result is an integral constant, so the operand must
    keep its integral-constant type: replacing it by its value would select the built-in operator instead.
    """
    operators = preprocessor._integral_constant_binary_operators
    following = end
    while following < len(source) and source[following].isspace():
        following += 1
    if following < len(source) and source[following] in operators:
        if not source.startswith(("*=", "/=", "+=", "-=", "**"), following):
            construction = _CONSTRUCTION_AFTER.match(source, following + 1)
            if construction is not None and _construction_parts(
                preprocessor, construction.group("type")
            ):
                return True
    previous = start - 1
    while previous >= 0 and source[previous].isspace():
        previous -= 1
    if previous < 0 or source[previous] not in operators:
        return False
    window = source[max(0, previous - _OPERAND_WINDOW) : previous]
    construction = _CONSTRUCTION_BEFORE.search(window)
    if construction is None:
        return False
    type_text = construction.group(0).rstrip()
    type_text = type_text[: type_text.rfind("{")].strip()
    return _construction_parts(preprocessor, type_text)


def _construction_parts(preprocessor: MetalPreprocessor, type_text: str) -> bool:
    """Whether ``type_text`` names a verified integral-constant template, whatever its (maybe dependent) value."""
    unqualified = re.sub(r"^(?:[A-Za-z_]\w*\s*::\s*)*", "", type_text.strip())
    if unqualified.startswith("Int"):
        return bool(preprocessor._int_alias_contract_verified)
    return bool(preprocessor._integral_constant_contract_verified)


def resolve_const_for_loop_bound_constants(
    preprocessor: MetalPreprocessor, code: str
) -> str:
    """Spell ``const_for_loop`` bounds that name block-scope constants by their values.

    In ``constexpr short TM = Tile::kRows; const_for_loop<0, TM, 1>(...)`` the bound ``TM`` is a constant expression
    ([expr.const]), so the loop expands exactly like ``const_for_loop<0, 2, 1>``. Only bounds of calls in concrete
    function bodies are changed, and only when every bound of the call then evaluates.
    """
    if _CONST_FOR_LOOP.search(code) is None:
        return code
    template_spans = preprocessor._find_template_declaration_spans(code)
    masked = preprocessor._mask_comments_and_literals(code)
    calls: list[tuple[int, int, list[str]]] = []
    owner_spans: set[tuple[int, int]] = set()
    for function in preprocessor._find_non_template_function_definitions(
        code, template_spans
    ):
        body_start, body_end = function.body_span
        for match in _CONST_FOR_LOOP.finditer(masked, body_start, body_end):
            angle_start = match.end() - 1
            angle_end = preprocessor._const_for_loop_template_argument_end(
                code, angle_start
            )
            if angle_end is None:
                continue
            bounds = [
                bound.strip()
                for bound in preprocessor._split_top_level_commas(
                    code[angle_start + 1 : angle_end]
                )
            ]
            if len(bounds) != 3 or all(_evaluates(preprocessor, b) for b in bounds):
                continue
            calls.append((angle_start, angle_end, bounds))
            owner_spans.add(function.body_span)
    if not calls:
        return code
    spans = sorted(owner_spans)
    type_aliases = preprocessor._collect_local_type_alias_bindings(code, spans)
    constants = preprocessor._collect_local_integral_constant_bindings(
        code,
        spans,
        type_aliases,
        preprocessor._materialization_constexpr_function_index(code),
        include_file_scope=True,
        static_member_constants=static_member_constant_substituter(
            preprocessor, preprocessor._find_concrete_struct_definitions(code)
        ),
    )
    replacements: list[tuple[int, int, str]] = []
    for angle_start, angle_end, bounds in calls:
        visible = preprocessor._local_integral_constants_at(constants, angle_start)
        if not visible:
            continue
        resolved = [
            preprocessor._substitute_template_argument_static_constants(bound, visible)
            for bound in bounds
        ]
        if resolved == bounds or not all(_evaluates(preprocessor, b) for b in resolved):
            continue
        replacements.append((angle_start + 1, angle_end, ", ".join(resolved)))
    if not replacements:
        return code
    return preprocessor._apply_text_replacements(code, replacements)


def _evaluates(preprocessor: MetalPreprocessor, expression: str) -> bool:
    folded, value = preprocessor._evaluate_static_integral_expression(expression)
    return folded and value is not None


def is_folded_integral_constant_operator(
    preprocessor: MetalPreprocessor, definition: object
) -> bool:
    """Whether a free operator template is the verified integral-constant operator that the fold evaluates.

    Its result is ``integral_constant<decltype(res), res>`` for a local constant ``res``, so an instance cannot be
    materialized as a function, and no call needs one: the fold replaces every call it can prove, and a call it
    cannot prove fails closed when the code generator finds no materialized operator.
    """
    if definition.operator not in preprocessor._integral_constant_binary_operators:
        return False
    if not preprocessor._integral_constant_contract_verified:
        return False
    return len(definition.parameter_types) == 2 and all(
        re.search(r"(?<![\w:])integral_constant\s*<", parameter_type)
        for parameter_type in definition.parameter_types
    )

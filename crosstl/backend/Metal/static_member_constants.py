"""Constant static data members of materialized Metal structs, and the assertions they decide.

Template materialization names a struct both by its materialized name (``Frag_float_1``) and, through a default
template argument, by the template-id it was materialized for (``Frag<float,1>``). A constant integral static data
member is a constant expression under either spelling ([expr.const]), so the helpers here fold it where a lowered
function or a data-only struct can no longer name it: assertion conditions, ``if constexpr`` conditions, call
arguments being typed, parameter array extents and other structs' static initializers.

They operate on the ``MetalPreprocessor`` passed in, whose source scanners and constant evaluator they reuse.
"""

import re
from typing import Any, Callable, Dict, List, Optional, Tuple

from .constexpr_branches import (
    discarded_constexpr_branch_spans,
    substitute_static_member_constants,
)

IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

__all__ = [
    "canonicalize_materialized_template_id_owners",
    "discarded_instantiated_branch_spans",
    "fold_argument_static_members",
    "materialized_struct_for_template_id",
    "resolve_retained_static_member_owners",
    "static_member_constant_substituter",
    "static_member_constant_value",
    "substitute_parameter_extent_static_members",
]


def resolve_retained_static_member_owners(preprocessor: Any, code: str) -> str:
    """Fold another struct's constants in a static member initializer.

    ``static constexpr constant int kRows = 2 * Frag_float_1::kElemRows;``
    is replaced by its value before static-member elision runs, because
    elision does not count uses inside struct definitions and may remove
    ``Frag_float_1::kElemRows`` while this initializer still names it.
    """
    if "::" not in code or "static" not in code:
        return code
    structs = preprocessor._find_concrete_struct_definitions(code)
    name_counts: Dict[str, int] = {}
    for struct in structs:
        name_counts[struct.name] = name_counts.get(struct.name, 0) + 1
    structs_by_owner: Dict[str, Any] = {}
    for struct in structs:
        structs_by_owner[struct.qualified_name or struct.name] = struct
        if name_counts[struct.name] == 1:
            structs_by_owner[struct.name] = struct
    replacements: List[Tuple[int, int, str]] = []
    for struct in structs:
        for member in struct.data_members:
            qualifiers = set(IDENTIFIER_RE.findall(member.type_text))
            if (
                member.default is None
                or member.span is None
                or "::" not in member.default
                or "static" not in qualifiers
                or not qualifiers.intersection({"const", "constexpr"})
            ):
                continue
            resolved = preprocessor._resolve_qualified_static_constant_expression(
                member.default, structs_by_owner
            )
            if "::" in resolved:
                continue
            folded, value = preprocessor._evaluate_static_integral_expression(resolved)
            if not folded or value is None:
                continue
            declaration = code[member.span[0] : member.span[1]]
            initializer = member.default.strip()
            equals = declaration.find("=")
            default_start = declaration.find(initializer, equals + 1)
            if (
                equals < 0
                or default_start < 0
                or declaration[equals + 1 : default_start].strip()
            ):
                continue
            replacements.append(
                (
                    member.span[0] + default_start,
                    member.span[0] + default_start + len(initializer),
                    preprocessor._static_integral_literal_text(int(value)),
                )
            )
    if not replacements:
        return code
    return preprocessor._apply_text_replacements(code, replacements)


def substitute_parameter_extent_static_members(
    preprocessor: Any,
    struct: Any,
    method: Any,
    parameters: str,
) -> str:
    """Inline constant static members named in parameter array extents.

    A member function may declare ``thread T vals[kRowsPerThread]``; the
    lowered free function is outside the struct's scope, so the extent is
    spelled with the member's initializer instead.
    """
    if "[" not in parameters:
        return parameters
    pieces: List[str] = []
    cursor = 0
    while True:
        open_bracket = parameters.find("[", cursor)
        if open_bracket < 0:
            break
        close_bracket = preprocessor._find_matching_delimiter(
            parameters, open_bracket, "[", "]"
        )
        if close_bracket is None:
            return parameters
        pieces.append(parameters[cursor : open_bracket + 1])
        pieces.append(
            preprocessor._substitute_static_data_member_initializers(
                struct,
                method,
                parameters[open_bracket + 1 : close_bracket],
            )
        )
        cursor = close_bracket
    pieces.append(parameters[cursor:])
    return "".join(pieces)


def static_member_constant_substituter(
    preprocessor: Any, structs: List[Any]
) -> Callable[[str, Dict[str, str]], str]:
    """Fold ``decltype(local)::member`` and ``Owner::member`` uses of constant integral static members."""
    name_counts: Dict[str, int] = {}
    for struct in structs:
        name_counts[struct.name] = name_counts.get(struct.name, 0) + 1
    structs_by_name: Dict[str, Any] = {}
    for struct in structs:
        structs_by_name[struct.qualified_name or struct.name] = struct
        if name_counts[struct.name] == 1:
            structs_by_name[struct.name] = struct
    values: Dict[Tuple[str, str], Optional[str]] = {}

    def lookup(owner: str, member: str) -> Optional[str]:
        key = (owner, member)
        if key not in values:
            value = static_member_constant_value(
                preprocessor, owner, member, structs_by_name
            )
            values[key] = None if value is None else str(value[0])
        return values[key]

    def substitute(expression: str, declared_types: Dict[str, str]) -> str:
        if not structs_by_name:
            return expression
        return substitute_static_member_constants(expression, declared_types, lookup)

    return substitute


def materialized_struct_for_template_id(
    preprocessor: Any, type_text: str
) -> Optional[str]:
    """The unique materialized struct a template-id such as ``Frag<float,1>`` names."""
    template_id = preprocessor._exact_template_type_id(type_text)
    if template_id is None:
        return None
    name, arguments = template_id
    key = (
        name.lstrip(":"),
        tuple(
            preprocessor._normalize_template_argument_text(item) for item in arguments
        ),
    )
    matches = [
        concrete
        for concrete, (
            template_name,
            template_arguments,
        ) in preprocessor._materialized_struct_specializations.items()
        if (
            template_name.lstrip(":"),
            tuple(
                preprocessor._normalize_template_argument_text(item)
                for item in template_arguments
            ),
        )
        == key
    ]
    return matches[0] if len(matches) == 1 else None


def static_member_constant_value(
    preprocessor: Any,
    owner: str,
    member: str,
    structs_by_name: Dict[str, Any],
) -> Optional[Tuple[int, str]]:
    """``(value, value type)`` of a constant integral static member of a concrete struct."""
    struct = structs_by_name.get(owner)
    if struct is None:
        concrete = materialized_struct_for_template_id(preprocessor, owner)
        struct = structs_by_name.get(concrete) if concrete else None
        if struct is None:
            return None
        owner = concrete
    if member not in preprocessor._resolved_static_data_member_initializers(struct):
        return None
    declared = next((item for item in struct.data_members if item.name == member), None)
    if declared is None or declared.array_suffix:
        return None
    value_types = [
        token
        for token in IDENTIFIER_RE.findall(declared.type_text)
        if token not in {"static", "constexpr", "constant", "const"}
    ]
    if len(value_types) != 1 or value_types[0] not in (
        preprocessor._METAL_INTEGRAL_SCALAR_TYPES
    ):
        return None
    folded, value = preprocessor._evaluate_static_integral_expression(
        preprocessor._resolve_qualified_static_constant_expression(
            f"{owner}::{member}", structs_by_name
        )
    )
    if not folded or value is None:
        return None
    return int(value), value_types[0]


def fold_argument_static_members(
    preprocessor: Any,
    argument: str,
    structs_by_name: Optional[Dict[str, Any]],
) -> str:
    """Fold ``Owner::member`` constants in a call argument for type inference only.

    Each use becomes a literal of the member's declared type, or a
    functional cast to it where no literal suffix spells that type, so the
    argument keeps the type C++ gives it.
    """
    if "::" not in argument or not structs_by_name:
        return argument

    literal_suffixes = {
        "int": "",
        "int32_t": "",
        "uint": "u",
        "uint32_t": "u",
        "long": "l",
        "int64_t": "l",
        "ulong": "ul",
        "uint64_t": "ul",
    }

    def lookup(owner: str, member: str) -> Optional[str]:
        value = static_member_constant_value(
            preprocessor, owner, member, structs_by_name
        )
        if value is None:
            return None
        number, value_type = value
        if value_type == "bool":
            return "true" if number else "false"
        suffix = literal_suffixes.get(value_type)
        if suffix is not None and number >= 0:
            return f"{number}{suffix}"
        return f"{value_type}({number})"

    return substitute_static_member_constants(argument, {}, lookup)


def discarded_instantiated_branch_spans(
    preprocessor: Any,
    code: str,
    template_spans: List[Tuple[int, int]],
    constants: Dict[str, List[Any]],
    static_member_constants: Callable[[str, Dict[str, str]], str],
) -> List[Tuple[int, int]]:
    """Substatements that ``if constexpr`` discards in materialized function templates."""
    if "constexpr" not in code or not preprocessor._materialized_function_names:
        return []
    bodies = [
        function.body_span
        for function in preprocessor._find_non_template_function_definitions(
            code, template_spans
        )
        if function.name in preprocessor._materialized_function_names
    ]
    if not bodies:
        return []

    def evaluate(start: int, end: int) -> Optional[bool]:
        visible = preprocessor._local_integral_constant_bindings_at(constants, start)
        condition = preprocessor._substitute_template_argument_static_constants(
            static_member_constants(
                code[start:end], preprocessor._visible_local_declared_types(visible)
            ),
            {
                name: binding.value
                for name, binding in visible.items()
                if binding.value is not None
            },
        )
        folded, value = preprocessor._evaluate_static_integral_expression(condition)
        return bool(value) if folded and value is not None else None

    return discarded_constexpr_branch_spans(
        preprocessor._mask_comments_and_literals(code), bodies, evaluate
    )


def canonicalize_materialized_template_id_owners(
    preprocessor: Any,
    code: str,
    structs_by_name: Dict[str, Any],
) -> str:
    """Spell a qualifying template-id such as ``Frag<float,1>::`` as its materialized struct."""
    if "::" not in code or "<" not in code:
        return code
    ignored = preprocessor._find_comment_and_literal_spans(code)
    replacements: List[Tuple[int, int, str]] = []
    for match in re.finditer(r"(?<![A-Za-z0-9_:.])[A-Za-z_]\w*\s*<", code):
        if preprocessor._containing_span(match.start(), ignored) is not None:
            continue
        angle_close = preprocessor._find_matching_angle(code, match.end() - 1)
        if angle_close is None:
            continue
        after = angle_close + 1
        while after < len(code) and code[after].isspace():
            after += 1
        if code[after : after + 2] != "::":
            continue
        concrete = materialized_struct_for_template_id(
            preprocessor, code[match.start() : angle_close + 1]
        )
        if concrete in structs_by_name:
            replacements.append((match.start(), angle_close + 1, concrete))
    if not replacements:
        return code
    return preprocessor._apply_text_replacements(code, replacements)

"""Bind reference parameters, and type member calls on named receivers.

Choosing among a struct's concrete member function overloads compares each
argument's type with each candidate's declared parameter type. Argument types
are value types, because an expression never has reference type
([expr.type]/1): a local ``float2 x`` has type ``float2``. A parameter declared
``thread float2& d`` binds ``x`` directly, which is an identity conversion
([over.ics.ref]/1), although its declared spelling ``float2&`` differs from the
argument's type.

:func:`argument_binds_parameter` accepts such a pair when the parameter is an
lvalue reference to the argument's value type, and either the argument is an
lvalue or the parameter is a ``const`` reference in the thread address space,
which also binds a temporary ([dcl.init.ref]/5). Rvalue reference parameters,
and arguments whose value category is not known, keep the exact-spelling rule,
so an overload set that needs either one stays refused.

:func:`member_call_result` types ``receiver.method(args)`` when ``receiver`` is
a local or parameter of a tracked struct type. The candidates are the
receiver's non-template member functions of that name that accept the argument
count. A ``const`` receiver admits only ``const`` member functions
([over.match.funcs]/4), and a receiver in an explicit address space admits only
member functions that accept it. When a ``const`` and a non-``const`` member
function have the same parameters, a non-``const`` receiver selects the
non-``const`` one ([over.ics.rank]/3.2.6). The call is typed only when every
remaining candidate gives the same result. Its value type is the declared return
type with struct-scoped aliases resolved and the reference removed; the result
is an lvalue when that return type is an lvalue reference ([expr.call]/14).
Template member functions, deduced return types and ``void`` leave the call
untyped.
"""

from __future__ import annotations

import re
from typing import Any, Mapping

__all__ = [
    "argument_binds_parameter",
    "argument_is_lvalue",
    "member_call_result",
]

_IDENTIFIER_RE = re.compile(r"[A-Za-z_]\w*")
_MEMBER_CALL_RE = re.compile(
    r"(?P<receiver>[A-Za-z_]\w*)\s*\.\s*(?P<method>[A-Za-z_]\w*)\s*\("
)
_QUALIFIED_MEMBER_RE = re.compile(
    r"(?P<owner>[A-Za-z_]\w*)\s*::\s*(?P<member>[A-Za-z_]\w*)"
)
_TRAILING_REFERENCE_RE = re.compile(r"\s*(?P<reference>&&|&)\s*$")
_ADDRESS_SPACES = frozenset(
    {
        "device",
        "constant",
        "thread",
        "threadgroup",
        "threadgroup_imageblock",
        "ray_data",
        "object_data",
    }
)
_LEADING_QUALIFIER_RE = re.compile(
    r"^(?:(?:const|volatile|typename|" + "|".join(sorted(_ADDRESS_SPACES)) + r")\s+)+"
)
# Each resolution step replaces one ``Owner::alias``; a chain longer than this
# is left untyped rather than followed indefinitely.
_MAX_QUALIFIED_ALIAS_STEPS = 8


def _split_reference(type_text: str) -> tuple[str, str]:
    match = _TRAILING_REFERENCE_RE.search(type_text)
    if match is None:
        return type_text.strip(), ""
    return type_text[: match.start()].strip(), match.group("reference")


def _reference_binds_temporaries(parameter_type: str) -> bool:
    referenced, reference = _split_reference(parameter_type)
    if reference != "&":
        return False
    tokens = set(re.findall(r"[A-Za-z_]\w*", referenced))
    if tokens & (_ADDRESS_SPACES - {"thread"}):
        return False
    if "*" in referenced:
        # `const T*&` refers to a non-const pointer; only `T* const&` is const.
        return re.search(r"\*\s*const\b[^*]*$", referenced) is not None
    return "const" in tokens


def _lvalue_path_end(pre: Any, expr: str, start: int) -> int:
    position = start
    while position < len(expr):
        rest = expr[position:]
        stripped = rest.lstrip()
        offset = position + len(rest) - len(stripped)
        if stripped.startswith("->") or stripped.startswith("."):
            step = 2 if stripped.startswith("->") else 1
            member = _IDENTIFIER_RE.match(expr, offset + step)
            if member is None:
                return position
            # A member name followed by a call is a call, not a member path.
            if expr[member.end() :].lstrip().startswith("("):
                return position
            position = member.end()
        elif stripped.startswith("["):
            close = pre._find_matching_delimiter(expr, offset, "[", "]")
            if close is None:
                return position
            position = close + 1
        else:
            return position
    return position


def argument_is_lvalue(
    pre: Any,
    argument: str,
    local_variable_types: Mapping[str, str],
    buffer_element_types: Any,
    structs_by_name: Mapping[str, Any] | None,
) -> bool:
    """Whether ``argument`` is an lvalue whose root object is a tracked name."""
    expr = pre._strip_enclosing_parens(
        pre._strip_template_argument_comments(argument).strip()
    )
    if not expr:
        return False
    member_call = member_call_result(pre, expr, local_variable_types, structs_by_name)
    if member_call is not None:
        return member_call[1]

    def tracked(name: str) -> bool:
        return name in local_variable_types or name in buffer_element_types

    if expr.startswith("*"):
        operand = pre._strip_enclosing_parens(expr[1:].strip())
        return _IDENTIFIER_RE.fullmatch(operand) is not None and tracked(operand)
    root = _IDENTIFIER_RE.match(expr)
    if root is None or not tracked(root.group(0)):
        return False
    return _lvalue_path_end(pre, expr, root.end()) == len(expr)


def argument_binds_parameter(
    pre: Any,
    declared_type: str,
    argument_type: str,
    parameter_type: str,
    argument: str,
    local_variable_types: Mapping[str, str],
    buffer_element_types: Any,
    structs_by_name: Mapping[str, Any] | None,
) -> bool:
    """Whether an argument of value type ``argument_type`` binds the parameter.

    ``declared_type`` and ``argument_type`` are the normalized spellings the
    overload comparison uses; ``parameter_type`` keeps the parameter's
    qualifiers so that a ``const`` reference can be told apart.
    """
    if declared_type == argument_type:
        return True
    referenced, reference = _split_reference(declared_type)
    if reference != "&" or referenced != argument_type.strip():
        return False
    if _reference_binds_temporaries(parameter_type):
        return True
    return argument_is_lvalue(
        pre, argument, local_variable_types, buffer_element_types, structs_by_name
    )


def _receiver_state(pre: Any, declared: str) -> tuple[str, bool, str] | None:
    text = pre._normalize_template_argument_text(str(declared))
    value, _reference = _split_reference(text)
    if not value or "*" in value or "[" in value:
        return None
    tokens = value.split(" ")
    is_const = "const" in tokens
    spaces = [token for token in tokens if token in _ADDRESS_SPACES]
    if len(spaces) > 1:
        return None
    space = spaces[0] if spaces else "thread"
    name = pre._normalize_inferred_type(value)
    return name, is_const, space


def _receiver_admits(method: Any, is_const: bool, space: str) -> bool:
    if method.is_static:
        return True
    if method.ref_qualifier == "&&":
        # A named receiver is an lvalue ([over.match.funcs]/5).
        return False
    if is_const and not method.is_const:
        return False
    spaces = tuple(method.receiver_address_spaces or ())
    return not spaces or space in spaces


def _parameter_signature(pre: Any, method: Any) -> tuple[str, ...]:
    return tuple(
        pre._normalize_template_argument_text(
            pre._normalize_function_parameter_type_text(parameter)
        )
        for parameter in pre._split_top_level_commas(method.parameters)
        if parameter.strip() and parameter.strip() != "void"
    )


def _resolve_qualified_aliases(
    pre: Any, type_text: str, structs_by_name: Mapping[str, Any]
) -> str | None:
    text = type_text
    for _step in range(_MAX_QUALIFIED_ALIAS_STEPS):
        prefix = _LEADING_QUALIFIER_RE.match(text)
        head = text[prefix.end() :] if prefix is not None else text
        value, reference = _split_reference(head)
        qualified = _QUALIFIED_MEMBER_RE.fullmatch(value)
        if qualified is None:
            return text
        owner = structs_by_name.get(qualified.group("owner"))
        if owner is None:
            return text
        resolved = pre._canonicalize_struct_scoped_type(
            qualified.group("member"), owner, structs_by_name
        )
        if not resolved or resolved == qualified.group("member"):
            return None
        qualifiers = text[: prefix.end()] if prefix is not None else ""
        text = f"{qualifiers}{resolved}{reference}"
    return None


def _candidate_result(
    pre: Any,
    method: Any,
    struct: Any,
    structs_by_name: Mapping[str, Any],
) -> tuple[str, bool] | None:
    return_type = (method.return_type or "").strip()
    if not return_type or re.search(r"\b(?:auto|decltype)\b", return_type):
        return None
    canonical = pre._canonicalize_struct_scoped_type(
        return_type, struct, structs_by_name
    )
    canonical = _resolve_qualified_aliases(pre, canonical, structs_by_name)
    if canonical is None:
        return None
    _value, reference = _split_reference(canonical)
    value_type = pre._normalize_inferred_expression_type(canonical)
    if not value_type or value_type == "void":
        return None
    return value_type, reference == "&"


def member_call_result(
    pre: Any,
    expr: str,
    local_variable_types: Mapping[str, str],
    structs_by_name: Mapping[str, Any] | None,
) -> tuple[str, bool] | None:
    """The value type of ``receiver.method(args)`` and whether it is an lvalue."""
    if not structs_by_name:
        return None
    match = _MEMBER_CALL_RE.match(expr)
    if match is None:
        return None
    close = pre._find_matching_delimiter(expr, match.end() - 1, "(", ")")
    if close != len(expr) - 1:
        return None
    declared = local_variable_types.get(match.group("receiver"))
    if not declared:
        return None
    state = _receiver_state(pre, declared)
    if state is None:
        return None
    struct_name, is_const, space = state
    struct = structs_by_name.get(struct_name)
    if struct is None:
        return None
    argument_count = len(
        [
            argument
            for argument in pre._split_top_level_commas(expr[match.end() : close])
            if argument.strip()
        ]
    )
    candidates = [
        method
        for method in struct.methods
        if method.name == match.group("method")
        and pre._callable_accepts_argument_count(method.parameters, argument_count)
    ]
    if not candidates or any(method.template_parameters for method in candidates):
        return None
    viable = [
        method for method in candidates if _receiver_admits(method, is_const, space)
    ]
    if not is_const:
        non_const_signatures = {
            _parameter_signature(pre, method)
            for method in viable
            if not method.is_static and not method.is_const
        }
        viable = [
            method
            for method in viable
            if method.is_static
            or not method.is_const
            or _parameter_signature(pre, method) not in non_const_signatures
        ]
    results = {
        _candidate_result(pre, method, struct, structs_by_name) for method in viable
    }
    if len(results) != 1:
        return None
    return results.pop()

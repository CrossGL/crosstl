"""Rename source identifiers that the Metal compiler parses as keywords.

GLSL and HLSL let a program name a parameter or variable ``kernel``,
``device``, ``thread``, ``and`` or ``new``. MSL is a C++14 dialect: its
function qualifiers (``kernel``, ``vertex``, ``fragment``), address spaces
(``device``, ``constant``, ``thread``, ``threadgroup``,
``threadgroup_imageblock``, ``ray_data``, ``object_data``), the ``half`` and
``bfloat`` types and the C++ keywords cannot be identifiers (Metal Shading
Language Specification 1.4, 4.1, 5.1 and 2.1). A parameter keeps its meaning
under any name, so each one spelled like a keyword is renamed together with
every reference in its function.
"""

from copy import deepcopy

from ..ast import (
    EnumNode,
    FunctionNode,
    IdentifierNode,
    MemberAccessNode,
    PointerAccessNode,
    StructNode,
    VariableNode,
)

# Every name in this set fails to compile as a Metal parameter or variable
# name (checked with the macOS Metal compiler); the spellings that GLSL and
# HLSL themselves reserve are kept so that every source language is covered.
METAL_KEYWORD_IDENTIFIERS = frozenset(
    {
        # Metal function qualifiers and address spaces.
        "kernel",
        "vertex",
        "fragment",
        "device",
        "constant",
        "thread",
        "threadgroup",
        "threadgroup_imageblock",
        "ray_data",
        "object_data",
        # Metal scalar type keywords.
        "half",
        "bfloat",
        # C++ keywords and alternative operator tokens.
        "alignas",
        "alignof",
        "and",
        "and_eq",
        "asm",
        "auto",
        "bitand",
        "bitor",
        "catch",
        "char",
        "char16_t",
        "char32_t",
        "class",
        "compl",
        "const_cast",
        "constexpr",
        "decltype",
        "delete",
        "dynamic_cast",
        "enum",
        "explicit",
        "export",
        "extern",
        "friend",
        "goto",
        "inline",
        "long",
        "mutable",
        "namespace",
        "new",
        "noexcept",
        "not",
        "not_eq",
        "nullptr",
        "operator",
        "or",
        "or_eq",
        "private",
        "protected",
        "public",
        "register",
        "reinterpret_cast",
        "short",
        "signed",
        "sizeof",
        "static",
        "static_assert",
        "static_cast",
        "template",
        "this",
        "thread_local",
        "throw",
        "try",
        "typedef",
        "typeid",
        "typename",
        "union",
        "unsigned",
        "using",
        "virtual",
        "volatile",
        "wchar_t",
        "xor",
        "xor_eq",
    }
)


def _iter_nodes(root):
    """Yield every AST node reachable from ``root`` once, iteratively."""
    seen = set()
    stack = [root]
    while stack:
        node = stack.pop()
        if node is None or isinstance(node, (str, int, float, bool)):
            continue
        if isinstance(node, (list, tuple, set)):
            stack.extend(node)
            continue
        if isinstance(node, dict):
            stack.extend(node.values())
            continue
        if not hasattr(node, "__dict__") or id(node) in seen:
            continue
        seen.add(id(node))
        yield node
        for key, value in vars(node).items():
            if key not in {"parent", "annotations"}:
                stack.append(value)


def _keyword_parameter_names(function):
    return {
        getattr(parameter, "name", None)
        for parameter in getattr(function, "parameters", None) or []
    } & METAL_KEYWORD_IDENTIFIERS


def _program_names(ast):
    return {
        name
        for node in _iter_nodes(ast)
        for name in (getattr(node, "name", None),)
        if isinstance(name, str)
    }


def _unused_name(name, used):
    candidate = f"{name}_"
    while candidate in used:
        candidate = f"{candidate}_"
    used.add(candidate)
    return candidate


def _member_containers(ast):
    for node in _iter_nodes(ast):
        if isinstance(node, StructNode):
            yield node
    yield from getattr(ast, "cbuffers", None) or []


def rename_metal_keyword_members(ast):
    """Return ``ast`` with keyword-spelled struct and cbuffer members renamed.

    A member is reached through ``.`` or ``->`` with its name, and a cbuffer
    member is also referenced by its bare name, so each keyword-spelled
    member name is replaced in every declaration, member access and bare
    reference of the program. Every struct sharing the name is renamed the
    same way, which keeps accesses through any of them valid. Accesses on an
    enum name (``Enum.value``) name enumerators, not members, and are kept.
    The input AST is not modified.
    """
    if not any(
        getattr(member, "name", None) in METAL_KEYWORD_IDENTIFIERS
        for container in _member_containers(ast)
        for member in getattr(container, "members", None) or []
    ):
        return ast
    ast = deepcopy(ast)
    used = _program_names(ast)
    used.update(
        str(node.member)
        for node in _iter_nodes(ast)
        if isinstance(node, (MemberAccessNode, PointerAccessNode))
    )
    renames = {}
    cbuffer_renames = {}
    cbuffers = {id(cbuffer) for cbuffer in getattr(ast, "cbuffers", None) or []}
    for container in _member_containers(ast):
        for member in getattr(container, "members", None) or []:
            name = getattr(member, "name", None)
            if name not in METAL_KEYWORD_IDENTIFIERS:
                continue
            if name not in renames:
                renames[name] = _unused_name(name, used)
            member.name = renames[name]
            if id(container) in cbuffers:
                cbuffer_renames[name] = renames[name]
    enum_names = {node.name for node in _iter_nodes(ast) if isinstance(node, EnumNode)}
    for node in _iter_nodes(ast):
        if isinstance(node, (MemberAccessNode, PointerAccessNode)):
            target = getattr(node, "object_expr", None) or getattr(
                node, "pointer_expr", None
            )
            if (
                isinstance(target, IdentifierNode)
                and getattr(target, "name", None) in enum_names
            ):
                continue
            new_name = renames.get(str(node.member))
            if new_name is not None:
                node.member = new_name
        elif isinstance(node, (IdentifierNode, VariableNode)):
            new_name = cbuffer_renames.get(getattr(node, "name", None))
            if new_name is not None:
                node.name = new_name
    return ast


def rename_metal_keyword_parameters(ast):
    """Return ``ast`` with keyword-spelled function parameters renamed.

    The replacement appends ``_`` until the name is used nowhere in the
    program, so it cannot capture another declaration. References are
    renamed only inside the parameter's own function; a nested declaration
    with the same name shadows the parameter and is renamed with it, which
    keeps every reference bound to the same declaration. The input AST is
    not modified.
    """
    if not any(
        _keyword_parameter_names(node)
        for node in _iter_nodes(ast)
        if isinstance(node, FunctionNode)
    ):
        return ast
    ast = deepcopy(ast)
    used = _program_names(ast)
    functions = [node for node in _iter_nodes(ast) if isinstance(node, FunctionNode)]
    for function in functions:
        reserved = _keyword_parameter_names(function)
        if not reserved:
            continue
        renames = {name: _unused_name(name, used) for name in sorted(reserved)}
        for parameter in function.parameters:
            new_name = renames.get(getattr(parameter, "name", None))
            if new_name is not None:
                parameter.name = new_name
        for node in _iter_nodes(getattr(function, "body", None)):
            if isinstance(node, (IdentifierNode, VariableNode)):
                new_name = renames.get(getattr(node, "name", None))
                if new_name is not None:
                    node.name = new_name
    return ast

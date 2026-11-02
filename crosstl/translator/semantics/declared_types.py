"""Undeclared type names: a CrossGL program that names a type it never declares is refused.

A type written as a plain name (``Foo v;``, ``Foo(1.0)``, ``(Foo)x``) is a CrossGL built-in or a structure,
enumeration, type alias, constant block or generic parameter the program declares. Any other name has no meaning, and
the generators would copy it into the target verbatim, where it is at best a compile error and at worst a different
type of the same spelling. :func:`refuse_undeclared_types` raises :class:`UndeclaredTypeError` for the first such
name, so the translation fails with a typed diagnostic instead.

Only plain names in code a stage entry can reach are checked (every function of a program without entries), and
trait bounds of generic parameters are not types. Neither is ``SAMPLER_FEEDBACK_MIN_MIP`` or
``SAMPLER_FEEDBACK_MIP_REGION_USED`` as the argument of a sampler feedback texture (``FeedbackTexture2D<KIND>``): any
other argument there is still read as a type, and :mod:`.feedback_textures` refuses one that is not a kind. Names
with generic arguments (``Texture2D<float4>``), namespace or member qualifiers
(``metal::texture2d``, ``Outer.Inner``) and parenthesised parts are resource or library spellings the generators
resolve themselves. The template argument of ``RayQuery<FLAGS>`` is a ray flag value, not a type (DXR, "RayQuery"),
and the generators check it themselves. The Metal, HLSL and GLSL generators emit a generic structure only as the
instantiations the program names (``Box<int>``), so for those targets a generic structure live code never names is
not read either: an importer keeps the template next to the specializations it materialized, and member aliases of
the template (``using complex_T = vec<scalar_T, 2>;``) have no CrossGL declaration.
"""

from __future__ import annotations

import re

from .. import ast as A
from ..diagnostics import CODE_PREFIX, TranslationDiagnostic
from .builtin_type_names import BUILTIN_TYPE_NAMES
from .feedback_textures import FEEDBACK_TEXTURE_TYPES, is_sampler_feedback_kind

__all__ = [
    "UndeclaredTypeError",
    "declared_type_names",
    "refuse_undeclared_types",
    "undeclared_type_uses",
]

_PLAIN_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
# Built-in types whose template arguments are values rather than types.
VALUE_TEMPLATE_TYPES = frozenset({"RayQuery"})
# Targets whose generators emit a generic structure only through the instantiations the program names.
INSTANTIATING_TARGETS = frozenset(
    {
        "metal",
        "directx",
        "hlsl",
        "dx11",
        "dx12",
        "d3d11",
        "d3d12",
        "opengl",
        "glsl",
        "webgl",
    }
)


class UndeclaredTypeError(TranslationDiagnostic):
    """A type name the program neither declares nor CrossGL defines."""

    def __init__(self, target, name, *, source_location=None):
        self.source_location = source_location
        self.type_name = name
        super().__init__(
            f"type '{name}' is not declared: it is neither a CrossGL built-in type nor a structure, "
            "enumeration, type alias or generic parameter of the program",
            target=target,
            code=f"{CODE_PREFIX}undeclared-type",
        )


def _children(node):
    for field, value in vars(node).items():
        if field in ("parent", "annotations", "source_location"):
            continue
        if isinstance(value, A.StageMap):
            yield from value.values()
        elif isinstance(value, A.ASTNode):
            yield value
        elif isinstance(value, (list, tuple)):
            for item in value:
                if isinstance(item, A.ASTNode):
                    yield item
        elif isinstance(value, dict):
            for item in value.values():
                if isinstance(item, A.ASTNode):
                    yield item


def _nodes(program, skip=frozenset()):
    """Every node reachable from ``program``, type nodes included, each once, in a stable order.

    The nodes whose ids are in ``skip`` are left out with everything below them.
    """
    seen = set(skip)
    stack = [program]
    while stack:
        node = stack.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        yield node
        stack.extend(reversed(list(_children(node))))


def _short(name):
    return str(name).rsplit("::", 1)[-1].rsplit(".", 1)[-1]


def _is_value_argument(node, argument):
    """Whether ``argument`` of the generic type ``node`` is a value rather than a type."""
    if node.name in VALUE_TEMPLATE_TYPES:
        return True
    return node.name in FEEDBACK_TEXTURE_TYPES and is_sampler_feedback_kind(argument)


def declared_type_names(program) -> frozenset:
    """The type names ``program`` declares: structures, constant blocks, enumerations, aliases, generic parameters."""
    names = set()
    for node in _nodes(program):
        name = getattr(node, "name", None)
        if not isinstance(name, str) or not name:
            continue
        if isinstance(
            node, (A.StructNode, A.EnumNode, A.GenericParameterNode, A.BufferNode)
        ):
            names.update((name, _short(name)))
        elif isinstance(node, A.GenericType):
            names.add(name)
        elif isinstance(node, A.VariableNode) and getattr(node, "is_type_alias", False):
            names.update((name, _short(name)))
    return frozenset(names)


def _function_lists(program):
    yield getattr(program, "functions", None) or []
    for stage in (getattr(program, "stages", None) or {}).values():
        yield getattr(stage, "local_functions", None) or []


def _referenced_names(node, skip):
    """The identifier and callee names used under ``node``, not descending into the nodes in ``skip``."""
    names = set()
    for child in _nodes(node, skip):
        if isinstance(child, A.IdentifierNode) and isinstance(child.name, str):
            names.add(child.name)
        elif isinstance(child, A.FunctionCallNode) and isinstance(child.function, str):
            names.add(child.function)
    return names


def _dead_functions(program):
    """Ids of the functions no stage entry reaches, which the generators drop; none when there is no entry.

    A function is live when it is a stage's entry point, or when a live function, a stage or a module-level
    declaration names it. A program without a stage entry is a library, so every function is checked.
    """
    helpers = {}
    for functions in _function_lists(program):
        for function in functions:
            if isinstance(function, A.FunctionNode) and isinstance(function.name, str):
                helpers.setdefault(function.name, []).append(function)
    entries = {
        id(stage.entry_point)
        for stage in (getattr(program, "stages", None) or {}).values()
        if isinstance(getattr(stage, "entry_point", None), A.FunctionNode)
    }
    if not entries:
        return set()
    every = {id(function) for group in helpers.values() for function in group}
    skip = every - entries
    live = set(entries)
    pending = sorted(_referenced_names(program, skip))
    seen = set()
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        seen.add(name)
        for function in helpers.get(name, []):
            if id(function) in live:
                continue
            live.add(id(function))
            pending.extend(sorted(_referenced_names(function, skip - {id(function)})))
    return every - live


def _type_references(node, skip):
    """The base names of the types and callees named under ``node``, not descending into the nodes in ``skip``."""
    names = set()
    for child in _nodes(node, skip):
        if isinstance(child, A.NamedType):
            name = child.name
        elif isinstance(child, A.FunctionCallNode):
            name = child.function
        else:
            continue
        if isinstance(name, str):
            base = name.split("<", 1)[0].strip()
            names.update((base, _short(base)))
    return names


def _uninstantiated_templates(program, skip):
    """Ids of the generic structures that no live code names, directly or through another named template."""
    templates = {}
    for node in _nodes(program, skip):
        if (
            isinstance(node, A.StructNode)
            and getattr(node, "generic_params", None)
            and isinstance(node.name, str)
        ):
            templates.setdefault(node.name, []).append(node)
    every = {id(template) for group in templates.values() for template in group}
    if not every:
        return set()
    hidden = set(skip) | every
    live = set()
    pending = sorted(_type_references(program, hidden))
    seen = set()
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        seen.add(name)
        for template in templates.get(name, []):
            if id(template) in live:
                continue
            live.add(id(template))
            pending.extend(sorted(_type_references(template, hidden - {id(template)})))
    return every - live


def _generic_function_names(program):
    """The names of the program's generic functions, whose explicit type arguments are types."""
    return {
        function.name
        for functions in _function_lists(program)
        for function in functions
        if isinstance(function, A.FunctionNode)
        and isinstance(function.name, str)
        and getattr(function, "generic_params", None)
    }


def _declared_value_names(program):
    """The names ``program`` declares as values: constants, variables and parameters.

    A generic call may pass one as an explicit argument, as in ``scaled<float, K>(x)`` for ``const int K``, which WGSL
    output substitutes into the specialization. Such a name is declared, so it is not an undeclared type.
    """
    return frozenset(
        node.name
        for node in _nodes(program)
        if isinstance(node, (A.ConstantNode, A.VariableNode, A.ParameterNode))
        and isinstance(node.name, str)
        and not getattr(node, "is_type_alias", False)
    )


def _call_type_argument_names(call, generic_functions):
    """The plain names among the explicit type arguments of ``call``, such as ``foo`` in ``pick<foo>(x)``.

    The parser keeps a call's type arguments in the callee name. Only calls of the program's own generic functions
    are read: the arguments of built-in calls, such as ``subgroup_broadcast_const<_, 0>``, may be values.
    """
    callee = call.function
    name = callee.name if isinstance(callee, A.IdentifierNode) else callee
    if not isinstance(name, str) or "<" not in name or not name.endswith(">"):
        return []
    base, arguments = name[:-1].split("<", 1)
    if base.strip() not in generic_functions:
        return []
    names, depth, start = [], 0, 0
    for index, character in enumerate(arguments):
        if character in "<[(":
            depth += 1
        elif character in ">])":
            depth -= 1
        elif character == "," and depth == 0:
            names.append(arguments[start:index].strip())
            start = index + 1
    names.append(arguments[start:].strip())
    return [name for name in names if _PLAIN_NAME.fullmatch(name)]


def undeclared_type_uses(program, target=None):
    """``(name, node)`` for each plain type name in ``program`` that is neither declared nor built in.

    Functions no stage entry reaches are not read: they cannot change what the program computes, and translation
    for a selected entry point drops them (a project's explicit specialization that was materialized under another
    name keeps its unresolved parameter types, for example). For a ``target`` in :data:`INSTANTIATING_TARGETS`, the
    generic structures no live code names are not read either, because those generators never emit them.
    """
    declared = declared_type_names(program)
    # Trait bounds (``generic<T: Numeric + Copy>``) constrain a parameter; they are not types any generator emits.
    bounds = {
        id(bound)
        for node in _nodes(program)
        if isinstance(node, A.GenericParameterNode)
        for bound in node.constraints
    }
    # RayQuery<RAY_FLAG_FORCE_OPAQUE> takes ray flags; the generators refuse operands that are not ray flags. A feedback
    # texture's argument is a value only when it is a sampler feedback kind; any other name is still read as a type.
    values = {
        id(argument)
        for node in _nodes(program)
        if isinstance(node, A.NamedType)
        for argument in node.generic_args
        if _is_value_argument(node, argument)
    }
    uses = []
    skip = _dead_functions(program) | bounds | values
    if str(target or "").lower() in INSTANTIATING_TARGETS:
        skip |= _uninstantiated_templates(program, skip)
    generic_functions = _generic_function_names(program)
    value_names = _declared_value_names(program) if generic_functions else frozenset()
    for node in _nodes(program, skip):
        if isinstance(node, A.FunctionCallNode):
            uses.extend(
                (name, node)
                for name in _call_type_argument_names(node, generic_functions)
                if name not in declared
                and name not in BUILTIN_TYPE_NAMES
                and name not in value_names
            )
            continue
        if not isinstance(node, A.NamedType) or node.generic_args:
            continue
        name = node.name
        if not isinstance(name, str) or not _PLAIN_NAME.fullmatch(name):
            continue
        if name in declared or name in BUILTIN_TYPE_NAMES:
            continue
        uses.append((name, node))
    return uses


def refuse_undeclared_types(program, target):
    """Raise :class:`UndeclaredTypeError` for the first undeclared plain type name in ``program``."""
    for name, node in undeclared_type_uses(program, target):
        raise UndeclaredTypeError(
            target, name, source_location=getattr(node, "source_location", None)
        )

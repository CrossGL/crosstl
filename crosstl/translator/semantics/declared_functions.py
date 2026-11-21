"""Undeclared functions: a CrossGL program that calls a function it never declares is refused.

A call written with a plain name (``f(x)``) names a function, generic function or structure method the program
declares, a type it converts to or constructs (``vec3(x)``, ``Light(...)``, ``vec<T, 2>{...}``), a value it calls, or
a built-in function. Any other name has no meaning. The Metal, HLSL and GLSL generators copied such a call into the
target as written, where it is at best a compile error and at worst a different function of the same spelling; only
SPIR-V output refused it. An importer that renames or drops a function but leaves a call to its old name produces
such a call too, and so does one that spells a source construct it cannot import as a call it never defines.
:func:`refuse_undeclared_functions` raises :class:`UndeclaredFunctionError` for the first such call, so the
translation fails with a typed diagnostic instead.

The built-in functions are those of :data:`~.builtin_function_names.BUILTIN_FUNCTION_NAMES`: CrossGL's own, and the
built-in functions of the languages the importers read, which an importer may carry into the IR for the generators
to map. A name reserved to the implementation (``__builtin_ctz``, ``__cgl_wrap_add``) is never a user function. Whether the target can express a built-in is left to the generators. A SPIR-V operation the importer kept as
a call (``spirvSDot``) is refused by :mod:`.spirv_instructions` with its own diagnostic.

As for undeclared types (:mod:`.declared_types`), every call is checked — no generator drops a plain function or
an explicit specialization no stage entry reaches — skipping the generic functions no code calls, which those
generators emit only through the calls that instantiate them (the Rust generator writes even those, so for
``rust`` their bodies are read), for a target that instantiates
generic structures the generic structures no code names, and only plain names: a callee with a namespace or
member qualifier (``metal::fast::sin``, ``obj.method``) is a library or method spelling the generators resolve
themselves. Explicit generic arguments (``pick<float>(x)``) are not part of the name.
"""

from __future__ import annotations

import re

from .. import ast as A
from ..diagnostics import CODE_PREFIX, TranslationDiagnostic
from .builtin_function_names import (
    BUILTIN_FUNCTION_NAMES,
    RESERVED_NAME_PREFIX,
    TYPE_TEMPLATE_NAMES,
)
from .builtin_type_names import BUILTIN_TYPE_NAMES
from .declared_types import (
    INSTANTIATING_TARGETS,
    _add_declared_type_name,
    _nodes,
    _short,
    _uninstantiated_templates,
    entry_dropped_function_ids,
)
from .spirv_instructions import is_spirv_operation_name

__all__ = [
    "UndeclaredFunctionError",
    "declared_function_names",
    "refuse_undeclared_functions",
    "undeclared_function_calls",
]

_PLAIN_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
# Targets whose generators write a generic function no call instantiates, so its body reaches the generated
# output and the names it uses are checked like any other body. Every other generator emits a generic function
# only through the calls that instantiate it.
GENERIC_WRITING_TARGETS = frozenset({"rust"})


class UndeclaredFunctionError(TranslationDiagnostic):
    """A call to a function the program neither declares nor CrossGL defines."""

    def __init__(self, target, name, *, source_location=None):
        self.source_location = source_location
        self.function_name = name
        super().__init__(
            f"function '{name}' is not declared: it is neither a CrossGL built-in function nor a function, "
            "structure method, type or callable value of the program",
            target=target,
            code=f"{CODE_PREFIX}undeclared-function",
        )


def _function_names(node):
    name = getattr(node, "name", None)
    if not isinstance(name, str) or not name:
        return ()
    base = name.split("<", 1)[0].strip()
    return (name, base, _short(base))


def declared_function_names(program) -> frozenset:
    """The names a call in ``program`` may use without a built-in: its functions, methods, types and values."""
    names = set()
    for node in _nodes(program):
        # The declared type names (:func:`~.declared_types.declared_type_names`), read in the same walk.
        _add_declared_type_name(node, names)
        if isinstance(
            node, (A.FunctionNode, A.VariableNode, A.ConstantNode, A.ParameterNode)
        ):
            names.update(_function_names(node))
    return frozenset(names)


def _callee_name(call):
    """The plain name ``call`` uses, without explicit generic arguments, or ``None`` when it needs no check.

    A qualified callee needs none, and neither does a type template constructed with explicit arguments.
    """
    callee = call.function
    name = callee.name if isinstance(callee, A.IdentifierNode) else callee
    if not isinstance(name, str):
        return None
    base, generic, _ = name.partition("<")
    base = base.strip()
    if not _PLAIN_NAME.fullmatch(base):
        return None
    explicit = generic or getattr(call, "generic_suffix_present", False)
    if base in TYPE_TEMPLATE_NAMES and explicit:
        return None
    return base


def _is_static_member(node):
    return isinstance(node, A.StructMemberNode) and any(
        str(getattr(attribute, "name", attribute)).lower() == "static"
        for attribute in getattr(node, "attributes", None) or []
    )


def _used_names(node, skip):
    names = set()
    for child in _nodes(node, skip):
        if isinstance(child, A.MemberAccessNode) and isinstance(child.member, str):
            names.add(child.member)
        elif isinstance(child, A.IdentifierNode) and isinstance(child.name, str):
            names.add(_short(child.name))
    return names


def _unused_static_members(program, skip):
    """Ids of the initialized static structure members no live code names, which the generators never emit.

    A static member is named through a member access or a qualified name (``Loader::reads``), and an initializer
    of a named static member names the members it reads, so names are followed to a fixed point, by unqualified
    name: a member of another structure with the same name keeps this one too.
    """
    statics = [
        node
        for node in _nodes(program, skip)
        if _is_static_member(node) and node.default_value is not None
    ]
    if not statics:
        return set()
    hidden = set(skip) | {id(member) for member in statics}
    used = _used_names(program, hidden)
    pending = list(statics)
    changed = True
    while changed:
        changed = False
        for member in list(pending):
            if member.name in used:
                pending.remove(member)
                used |= _used_names(member.default_value, hidden)
                changed = True
    return {id(member) for member in pending}


def _never_called_generic_functions(program, skip):
    """Ids of the generic functions no code calls, which the generators only emit through call sites.

    The mirror of :func:`~.declared_types._uninstantiated_templates`: except under the Rust generator, which
    writes even a generic function nothing instantiates, a generic function is written to the output only when a
    call instantiates it, so a body nothing calls never reaches the generated output, and the plain
    callee spellings the qualified-member lowering leaves behind inside it (``CTile::Frag_t::mma`` where ``CTile``
    is a parameter cannot be resolved, so the call stays a bare ``mma``) must not be refused. A name used anywhere
    outside the generic bodies counts as a call — through another generic body only once that body is live itself,
    following the same fixed point the unused static members follow.
    """
    generic = {}
    for node in _nodes(program, skip):
        if (
            isinstance(node, A.FunctionNode)
            and isinstance(node.name, str)
            and getattr(node, "generic_params", None)
        ):
            generic.setdefault(node.name, []).append(node)
    every = {id(function) for group in generic.values() for function in group}
    if not every:
        return set()
    hidden = set(skip) | every
    used = _used_names(program, hidden)
    live = set()
    pending = sorted(used & generic.keys())
    while pending:
        name = pending.pop()
        for function in generic.get(name, []):
            if id(function) in live:
                continue
            live.add(id(function))
            used = _used_names(function, hidden - {id(function)})
            pending.extend(sorted(used & generic.keys()))
    return every - live


def undeclared_function_calls(program, target=None, entry_point=None):
    """``(name, call)`` for each call in ``program`` whose plain callee is neither declared nor built in.

    Every function body is read: no generator drops a plain function or an explicit specialization no stage entry
    reaches, so a call in such a function reaches the generated output too. The generic functions no code calls
    are not read — those generators emit a generic function only through the calls that instantiate it — except
    for the Rust generator, which writes even those, so for ``target="rust"`` their bodies are read as well.
    Static structure members
    nothing names are not read, and for a ``target`` in :data:`~.declared_types.INSTANTIATING_TARGETS`, the
    generic structures no code names are not read either, because those generators emit a generic structure only
    through the instantiations it names. A translation that selects ``entry_point`` skips the functions no stage
    entry reaches, which it does not write (:func:`~.declared_types.entry_dropped_function_ids`).
    """
    declared = declared_function_names(program)
    calls = []
    skip = entry_dropped_function_ids(program, entry_point)
    if str(target or "").lower() not in GENERIC_WRITING_TARGETS:
        skip |= _never_called_generic_functions(program, skip)
    if str(target or "").lower() in INSTANTIATING_TARGETS:
        skip |= _uninstantiated_templates(program, skip)
    skip |= _unused_static_members(program, skip)
    for node in _nodes(program, skip):
        if not isinstance(node, A.FunctionCallNode):
            continue
        name = _callee_name(node)
        if name is None or name in declared or is_spirv_operation_name(name):
            continue
        if name in BUILTIN_FUNCTION_NAMES or name in BUILTIN_TYPE_NAMES:
            continue
        if name.startswith(RESERVED_NAME_PREFIX):
            continue
        calls.append((name, node))
    return calls


def refuse_undeclared_functions(program, target, entry_point=None):
    """Raise :class:`UndeclaredFunctionError` for the first call to an undeclared function in ``program``."""
    for name, node in undeclared_function_calls(program, target, entry_point):
        raise UndeclaredFunctionError(
            target, name, source_location=getattr(node, "source_location", None)
        )

"""Program-scope values with names C++ reserves, renamed for CUDA and HIP output.

CUDA and HIP output declare a CrossGL program's program-scope values in the global namespace of a C++ translation
unit: its resources, constants, constant-buffer members and stage variables. C++ reserves every name that begins with
an underscore for the implementation in the global namespace, and every name that contains a double underscore in
any scope (C++20 [lex.name] paragraph 3), so such a declaration may clash with a name the CUDA or HIP headers declare,
such as ``__half`` or ``_Float16``. NVCC also writes the C++ name of a program-scope variable into PTX as its symbol,
and PTX cannot spell a lone underscore: an identifier that starts with ``_`` needs at least one more character (PTX
ISA, "Identifiers"). ``ptxas`` then rejects the module with ``Parsing error near '_'``. SPIR-V disassembly names the
instance of an unnamed block ``_``, ``__0``, ``__1`` and so on, so a SPIR-V module with an unnamed buffer or uniform
block reached this.

:func:`rename_reserved_program_names` renames each such value on a copy of the AST, with every reference to it and
every parameter or local of the same name. Renaming every declaration of the name keeps each reference bound to the
declaration it named, and the new name, ``global_`` followed by the old one without its leading underscores and with
each run of underscores shortened to one, is unreserved and collides with no name of the program. A struct member
keeps its name, and so does a struct method's reference to a member: neither is in the global namespace. Functions
keep theirs: :mod:`.cuda_library_functions` renames the ones that start with ``__``, and NVCC mangles the C++ name of
any other device function into its PTX symbol.
"""

from __future__ import annotations

import re
from copy import deepcopy

from .. import ast as A
from ..semantics.rewrite import child_slots, get_child

__all__ = ["is_reserved_global_name", "rename_reserved_program_names"]

_UNDERSCORE_RUN = re.compile(r"_+")


def is_reserved_global_name(name) -> bool:
    """Whether C++ reserves ``name`` in the global namespace: it begins with ``_`` or contains ``__``."""
    return isinstance(name, str) and (name.startswith("_") or "__" in name)


def _stages(program):
    stages = getattr(program, "stages", None)
    if isinstance(stages, A.StageMap):
        return [
            stage
            for kind in list(stages.keys())
            for stage in stages.all_for_stage(kind)
        ]
    if isinstance(stages, dict):
        return list(stages.values())
    return list(stages or [])


def _program_values(program):
    """The values CUDA and HIP output declare at program scope."""
    values = list(getattr(program, "global_variables", None) or [])
    values += list(getattr(program, "constants", None) or [])
    for cbuffer in getattr(program, "cbuffers", None) or []:
        # The generators lower each constant-buffer member to a ``__constant__`` variable of its own.
        values += list(getattr(cbuffer, "members", None) or [])
    for stage in _stages(program):
        values += list(getattr(stage, "local_variables", None) or [])
    return [value for value in values if isinstance(getattr(value, "name", None), str)]


def _walk(node, seen):
    if not isinstance(node, A.ASTNode) or id(node) in seen:
        return
    seen.add(id(node))
    yield node
    for field, key in list(child_slots(node)):
        yield from _walk(get_child(node, field, key), seen)


def _program_names(program):
    """Every name the program spells, so that a new name collides with none of them."""
    names = set()
    for node in _walk(program, set()):
        for field in ("name", "identifier", "member"):
            value = getattr(node, field, None)
            if isinstance(value, str):
                names.add(value)
        function = getattr(node, "function", None)
        if isinstance(function, str):
            names.add(function)
    return names


def _fresh(name, taken):
    core = _UNDERSCORE_RUN.sub("_", name).strip("_")
    base = f"global_{core}" if core else "global"
    candidate = base
    suffix = 1
    while candidate in taken:
        candidate = f"{base}_{suffix}"
        suffix += 1
    taken.add(candidate)
    return candidate


def _struct_member_names(struct):
    return {
        member.name
        for member in getattr(struct, "members", None) or []
        if isinstance(getattr(member, "name", None), str)
    }


class _Renamer:
    def __init__(self, renames):
        self.renames = renames
        self.seen = set()

    def visit(self, node, members=frozenset()):
        """Rename in ``node``; ``members`` are the names a struct method reads as members of its struct."""
        if not isinstance(node, A.ASTNode) or id(node) in self.seen:
            return
        self.seen.add(id(node))
        if isinstance(node, A.StructNode):
            self.visit_struct(node)
            return
        if isinstance(node, (A.VariableNode, A.ParameterNode)):
            if node.name in self.renames and node.name not in members:
                node.name = self.renames[node.name]
        elif isinstance(node, A.IdentifierNode):
            if node.name in self.renames and node.name not in members:
                node.name = self.renames[node.name]
                node.identifier = node.name
        for field, key in list(child_slots(node)):
            child = get_child(node, field, key)
            if (
                isinstance(node, A.FunctionCallNode)
                and field == "function"
                and isinstance(child, A.IdentifierNode)
            ):
                # A callee names a function, which keeps its name.
                continue
            self.visit(child, members)

    def visit_struct(self, struct):
        members = frozenset(_struct_member_names(struct))
        for member in getattr(struct, "members", None) or []:
            # A member keeps its name; an initializer it carries may still read a renamed value.
            self.seen.add(id(member))
            for field, key in list(child_slots(member)):
                self.visit(get_child(member, field, key), members)
        for method in getattr(struct, "methods", None) or []:
            self.visit(method, members)


def rename_reserved_program_names(ast):
    """``ast``, or a copy whose program-scope values with names C++ reserves are renamed, with every use."""
    if not isinstance(ast, A.ShaderNode):
        return ast
    names = sorted(
        {
            value.name
            for value in _program_values(ast)
            if is_reserved_global_name(value.name)
        }
    )
    if not names:
        return ast
    ast = deepcopy(ast)
    taken = _program_names(ast)
    renames = {name: _fresh(name, taken) for name in names}
    # Constants and constant-buffer members are not variable nodes, so the walk below would not rename them.
    for value in _program_values(ast):
        if value.name in renames:
            value.name = renames[value.name]
    _Renamer(renames).visit(ast)
    return ast

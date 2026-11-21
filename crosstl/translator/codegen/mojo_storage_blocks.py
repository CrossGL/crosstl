"""Storage blocks that hold one runtime-sized array, as Mojo kernel pointer arguments.

A GLSL-style storage block whose only member is a runtime-sized array (``buffer Data { float data[]; };`` or
``buffer Out { Particle p[]; } outb;``) is that array: the buffer starts at its first element. Mojo has no global
variables, so a kernel receives such a buffer as an ``UnsafePointer`` to the element type, as it receives
``buffer float* p``. This pass rewrites each such block declaration into a runtime-sized array buffer of the block's
name, rewrites ``block.member`` to ``block`` in every function that does not declare the name itself, and drops the
block's structure, which then names nothing.

A block with any other member (a count before the array, or only sized arrays) stays a structure, which Mojo output
refuses outside a function.
"""

from __future__ import annotations

from ..ast import (
    ArrayType,
    IdentifierNode,
    MemberAccessNode,
    NamedType,
    VariableNode,
)
from .mojo_kernel_arguments import mojo_declared_names


def _block_array_member(node, structs):
    """The (member name, element type) of a single-runtime-array storage block declaration, or None."""
    if not isinstance(node, VariableNode):
        return None
    qualifiers = {str(qualifier).lower() for qualifier in node.qualifiers or []}
    if "buffer" not in qualifiers or getattr(node, "initial_value", None) is not None:
        return None
    var_type = getattr(node, "var_type", None)
    if not isinstance(var_type, NamedType) or var_type.generic_args:
        return None
    struct = structs.get(var_type.name)
    members = list(getattr(struct, "members", None) or [])
    if len(members) != 1:
        return None
    member_type = getattr(members[0], "member_type", None) or getattr(
        members[0], "var_type", None
    )
    if not isinstance(member_type, ArrayType) or member_type.size is not None:
        return None
    return members[0].name, member_type.element_type


def _replace_by_identity(root, replacements):
    """Replace every field reference to a node in ``replacements`` (keyed by id) under ``root``."""
    for node in list(root.walk()):
        fields = getattr(node, "__dict__", None)
        if not fields:
            continue
        for field, value in list(fields.items()):
            if field == "parent":
                continue
            if id(value) in replacements:
                setattr(node, field, replacements[id(value)])
            elif isinstance(value, list):
                for index, item in enumerate(value):
                    if id(item) in replacements:
                        value[index] = replacements[id(item)]


def _function_scopes(ast):
    """(function, stage or None) for every function the program declares, entry points included."""
    for func in getattr(ast, "functions", None) or []:
        yield func, None
    for stage in (getattr(ast, "stages", None) or {}).values():
        for func in getattr(stage, "local_functions", None) or []:
            yield func, stage
        entry_point = getattr(stage, "entry_point", None)
        if entry_point is not None:
            yield entry_point, stage


def _rewrite_member_reads(func, blocks):
    """Map each ``block.member`` in ``func`` to an identifier of the block, unless ``func`` declares the name."""
    parameters = {
        getattr(parameter, "name", None)
        for parameter in getattr(func, "parameters", getattr(func, "params", [])) or []
    }
    shadowed = parameters | mojo_declared_names(getattr(func, "body", None))
    replacements = {}
    for node in func.walk():
        if not isinstance(node, MemberAccessNode):
            continue
        target = getattr(node, "object", None)
        if not isinstance(target, IdentifierNode) or target.name in shadowed:
            continue
        member = blocks.get(target.name)
        if member is not None and node.member == member:
            replacements[id(node)] = IdentifierNode(target.name)
    if replacements:
        _replace_by_identity(func, replacements)


def lower_mojo_single_array_storage_blocks(ast):
    """Rewrite single-runtime-array storage blocks into runtime-sized array buffers, in place."""
    module_structs = {s.name: s for s in getattr(ast, "structs", None) or []}
    module_blocks = {}
    for node in getattr(ast, "global_variables", None) or []:
        found = _block_array_member(node, module_structs)
        if found is not None:
            module_blocks[node.name] = (node, found)
    stage_blocks = {}
    for stage_key, stage in (getattr(ast, "stages", None) or {}).items():
        structs = {
            **module_structs,
            **{s.name: s for s in getattr(stage, "local_structs", None) or []},
        }
        for node in getattr(stage, "local_variables", None) or []:
            found = _block_array_member(node, structs)
            if found is not None:
                stage_blocks.setdefault(stage_key, {})[node.name] = (node, found)
    if not module_blocks and not stage_blocks:
        return
    block_types = set()
    for node, (_member, element_type) in [
        *module_blocks.values(),
        *(entry for blocks in stage_blocks.values() for entry in blocks.values()),
    ]:
        block_types.add(node.var_type.name)
        node.var_type = ArrayType(element_type, None)
        node.vtype = node.var_type
    stages = getattr(ast, "stages", None) or {}
    stage_keys = {id(stage): key for key, stage in stages.items()}
    for func, stage in _function_scopes(ast):
        blocks = {name: entry[1][0] for name, entry in module_blocks.items()}
        if stage is not None:
            for name, entry in stage_blocks.get(stage_keys[id(stage)], {}).items():
                blocks[name] = entry[1][0]
        _rewrite_member_reads(func, blocks)
    # The block's structure names nothing once its declaration is an array.
    ast.structs = [s for s in ast.structs or [] if s.name not in block_types]
    for stage in stages.values():
        if getattr(stage, "local_structs", None):
            stage.local_structs = [
                s for s in stage.local_structs if s.name not in block_types
            ]

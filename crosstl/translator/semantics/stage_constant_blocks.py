"""Constant blocks declared at stage scope move to module scope.

CrossGL accepts a constant block (``uniform Params { ... };`` or ``cbuffer Params { ... };``) inside a stage as well
as at module scope; its members are names of that stage, as a module-scope block's members are names of the whole
program. The GLSL, HLSL, WGSL and Mojo generators declare a stage's blocks next to the module-scope ones. The Metal,
Slang, Rust, CUDA and HIP generators read only the module-scope list, so they dropped a stage's blocks and emitted
uses of undeclared members, and the SPIR-V generator refused the program for its unknown names.

For those targets :func:`hoist_stage_constant_blocks` moves each stage's blocks, in stage and declaration order, to
the end of the module-scope list, where the generators bind and pass them as they do any module-scope block. Two
stages may declare the same block: a block whose name, members and attributes match a module-scope block is the same
block (GLSL 4.60 section 4.3.9 matches uniform blocks across stages by name and member list) and is dropped from its
stage. A block that would change what a name means once it moves raises
:class:`StageConstantBlockConflictError`: a different block of the same name, or a block or member whose name a
module-scope declaration, another block's member or another stage's scope already declares.
"""

from __future__ import annotations

from .. import ast as A
from ..diagnostics import CODE_PREFIX, TranslationDiagnostic

__all__ = [
    "TARGETS",
    "StageConstantBlockConflictError",
    "hoist_stage_constant_blocks",
]

TARGETS = frozenset({"metal", "slang", "rust", "cuda", "hip", "vulkan", "spirv"})

_SHAPE_SKIPPED = frozenset({"parent", "annotations", "source_location"})


class StageConstantBlockConflictError(TranslationDiagnostic):
    """A stage's constant block cannot move to module scope without changing what a name refers to."""

    def __init__(self, message, target):
        super().__init__(
            message,
            target=target,
            code=f"{CODE_PREFIX}stage-constant-block-conflict",
        )


def hoist_stage_constant_blocks(program, target) -> int:
    """Move stage-scope constant blocks to module scope for ``target``; returns how many moved."""
    target = str(target or "").lower()
    if not isinstance(program, A.ShaderNode) or target not in TARGETS:
        return 0
    stages = _stages(program)
    if not any(getattr(stage, "local_cbuffers", None) for _, stage in stages):
        return 0
    program.cbuffers = list(getattr(program, "cbuffers", None) or [])
    blocks = {block.name: block for block in program.cbuffers}
    owners = _module_names(program)
    stage_names = {
        stage_key: _stage_scope_names(stage, exclude_blocks=True)
        for stage_key, stage in stages
    }
    moved = 0
    for stage_key, stage in stages:
        local_blocks = list(getattr(stage, "local_cbuffers", None) or [])
        if not local_blocks:
            continue
        stage_label = _stage_label(stage_key, stage)
        for block in local_blocks:
            existing = blocks.get(block.name)
            if existing is not None:
                if _shape(existing) != _shape(block):
                    raise StageConstantBlockConflictError(
                        f"the {stage_label} stage declares constant block '{block.name}' with members or "
                        "attributes that differ from another declaration of that block; give the blocks "
                        "different names or the same members",
                        target,
                    )
                continue
            for name in [block.name] + _member_names(block):
                owner = owners.get(name)
                if owner is None:
                    owner = _other_stage_declaring(name, stage_key, stage_names, stages)
                if owner is not None:
                    raise StageConstantBlockConflictError(
                        f"constant block '{block.name}' of the {stage_label} stage declares '{name}', which "
                        f"{owner} also declares; the block's names would become names of the whole program, "
                        "so rename one of them",
                        target,
                    )
            owners[block.name] = f"constant block '{block.name}'"
            for name in _member_names(block):
                owners[name] = f"constant block '{block.name}'"
            blocks[block.name] = block
            program.cbuffers.append(block)
            moved += 1
        stage.local_cbuffers = []
    return moved


def _stages(program):
    stages = getattr(program, "stages", None) or {}
    if isinstance(stages, dict):
        return list(stages.items())
    return [
        (getattr(stage, "stage", index), stage) for index, stage in enumerate(stages)
    ]


def _stage_label(stage_key, stage):
    value = getattr(stage, "stage", None) or stage_key
    return str(getattr(value, "value", value) or "stage").lower()


def _member_names(block):
    return [
        member.name
        for member in getattr(block, "members", None) or []
        if isinstance(getattr(member, "name", None), str)
    ]


def _module_names(program):
    """Module-scope names, each mapped to a description of what declares it."""
    owners = {}

    def add(nodes, kind):
        for node in nodes or []:
            name = getattr(node, "name", None)
            if isinstance(name, str) and name not in owners:
                owners[name] = f"module-scope {kind} '{name}'"

    add(getattr(program, "structs", None), "structure")
    add(getattr(program, "constants", None), "constant")
    add(getattr(program, "global_variables", None), "variable")
    add(getattr(program, "functions", None), "function")
    for block in getattr(program, "cbuffers", None) or []:
        if isinstance(getattr(block, "name", None), str):
            owners.setdefault(block.name, f"constant block '{block.name}'")
        for name in _member_names(block):
            owners.setdefault(name, f"constant block '{block.name}'")
    return owners


def _stage_scope_names(stage, exclude_blocks=False):
    """Names a stage declares at its own scope: variables, functions, structures and (optionally) blocks."""
    names = set()
    for field in ("local_variables", "local_functions", "local_structs"):
        for node in getattr(stage, field, None) or []:
            name = getattr(node, "name", None)
            if isinstance(name, str):
                names.add(name)
    if not exclude_blocks:
        for block in getattr(stage, "local_cbuffers", None) or []:
            names.add(block.name)
            names.update(_member_names(block))
    return names


def _other_stage_declaring(name, stage_key, stage_names, stages):
    """A description of another stage that declares ``name`` at its scope, or None."""
    for other_key, other in stages:
        if other_key == stage_key:
            continue
        if name in stage_names[other_key]:
            return f"the {_stage_label(other_key, other)} stage"
    return None


def _shape(node, path=None):
    """A structural key of a declaration that ignores source positions and analysis metadata."""
    path = path or frozenset()
    if isinstance(node, A.ASTNode):
        if id(node) in path:
            return ("cycle", type(node).__name__)
        path = path | {id(node)}
        return (
            type(node).__name__,
            tuple(
                (field, _shape(value, path))
                for field, value in sorted(vars(node).items())
                if field not in _SHAPE_SKIPPED
            ),
        )
    if isinstance(node, (list, tuple)):
        return tuple(_shape(item, path) for item in node)
    if isinstance(node, dict):
        return tuple(
            sorted((str(key), _shape(value, path)) for key, value in node.items())
        )
    if node is None or isinstance(node, (str, int, float, bool)):
        return node
    return (type(node).__name__, str(getattr(node, "value", node)))

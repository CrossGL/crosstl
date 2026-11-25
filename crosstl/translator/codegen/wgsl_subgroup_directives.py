"""The directives of a WGSL module that uses subgroup built-ins.

A WGSL module that calls a subgroup or quad built-in function, or declares a subgroup built-in input, must enable the
``subgroups`` extension (WGSL section 4.1.1, "Enable Extensions"; Tint refuses the module otherwise). Those built-ins
compute over the active invocations of the subgroup wherever they are called, as the CrossGL wave operations do
(GL_KHR_shader_subgroup; SPIR-V ``OpGroupNonUniform*``), and CrossGL asks for no uniform control flow around them. So
the module also turns off the ``subgroup_uniformity`` diagnostic, whose default severity, error, would require
subgroup-uniform control flow and lane arguments (WGSL, "Filterable Triggering Rules").

naga 30 refuses ``enable subgroups;`` as not yet implemented (gfx-rs/wgpu#5555) and reads the subgroup built-ins
without it, so a wgpu host removes that line before naga reads the module.
"""

from __future__ import annotations

import re

__all__ = ["SUBGROUP_DIRECTIVES", "add_subgroup_directives", "uses_subgroups"]

SUBGROUP_DIRECTIVES = ("enable subgroups;", "diagnostic(off, subgroup_uniformity);")

# WGSL's subgroup and quad built-in functions ("Subgroup Built-in Functions", "Quad Operations") and subgroup
# built-in values ("Built-in Values").
_FUNCTIONS = frozenset(
    {
        "quadBroadcast",
        "quadSwapDiagonal",
        "quadSwapX",
        "quadSwapY",
        "subgroupAdd",
        "subgroupAll",
        "subgroupAnd",
        "subgroupAny",
        "subgroupBallot",
        "subgroupBroadcast",
        "subgroupBroadcastFirst",
        "subgroupElect",
        "subgroupExclusiveAdd",
        "subgroupExclusiveMul",
        "subgroupInclusiveAdd",
        "subgroupInclusiveMul",
        "subgroupMax",
        "subgroupMin",
        "subgroupMul",
        "subgroupOr",
        "subgroupShuffle",
        "subgroupShuffleDown",
        "subgroupShuffleUp",
        "subgroupShuffleXor",
        "subgroupXor",
    }
)
_VALUES = frozenset(
    {"num_subgroups", "subgroup_id", "subgroup_invocation_id", "subgroup_size"}
)
_CALL = re.compile(r"\b([A-Za-z_]\w*)\s*\(")
_BUILTIN = re.compile(r"@builtin\(\s*(\w+)\s*\)")
_DECLARED = re.compile(r"\bfn\s+(\w+)")


def uses_subgroups(text: str) -> bool:
    """Whether the WGSL ``text`` declares a subgroup built-in input or calls a subgroup built-in function that no
    function of its own shadows."""
    if any(name in _VALUES for name in _BUILTIN.findall(text)):
        return True
    declared = set(_DECLARED.findall(text))
    return any(
        name in _FUNCTIONS and name not in declared for name in _CALL.findall(text)
    )


def add_subgroup_directives(lines: list[str]) -> None:
    """Insert :data:`SUBGROUP_DIRECTIVES` after the header comment of the module ``lines`` when it uses subgroups.
    Directives come before every declaration."""
    if uses_subgroups("\n".join(lines)):
        lines[1:1] = ["", *SUBGROUP_DIRECTIVES]

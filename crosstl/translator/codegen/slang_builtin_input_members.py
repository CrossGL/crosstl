"""Entry input structures with members Slang reads through an intrinsic.

A CrossGL fragment input structure can bind a member to a built-in that Slang has no system-value semantic for, such
as ``bool helper @ gl_HelperInvocation``. Slang reads that value with ``IsHelperLane()``, and the subgroup values with
``WaveGetLaneCount()``, ``WaveGetLaneIndex()`` and the ``WaveGetLane*Mask()`` functions (Slang
``docs/wave-intrinsics.md``). A member declared with the built-in's name as its semantic would be a user varying that
no stage writes, so the generator refuses it (``SlangSemanticsMixin.semantic_suffix``).

HLSL reads the same built-ins through the same intrinsics, and its generator already rewrites such a structure: the
entry takes a signature structure with the other members, and its body builds the program's structure, reading each
built-in member from the CrossGL built-in (:mod:`.hlsl_builtin_input_members`). Slang runs that pass with its own
table: the generator writes ``gl_HelperInvocation`` as ``IsHelperLane()`` (refused outside a fragment entry point)
and the subgroup built-ins through :data:`.slang_subgroup_builtins.SLANG_SUBGROUP_LANE_BUILTINS`.
"""

from __future__ import annotations

from .hlsl_builtin_input_members import lower_hlsl_builtin_input_members
from .slang_subgroup_builtins import SLANG_SUBGROUP_LANE_BUILTINS

__all__ = ["SLANG_INTRINSIC_INPUT_BUILTINS", "lower_slang_builtin_input_members"]

# CrossGL type spelling -> (scalar, vector size), the key the shared pass compares a member's type with.
_TYPE_KEYS = {"bool": ("bool", None), "uint": ("uint", None), "uvec4": ("uint", 4)}

# Built-in -> CrossGL type as (scalar, vector size), for the built-ins Slang reads only through an intrinsic.
SLANG_INTRINSIC_INPUT_BUILTINS = {
    "gl_HelperInvocation": _TYPE_KEYS["bool"],
    **{
        name: _TYPE_KEYS[crossgl_type]
        for name, (crossgl_type, _intrinsic) in sorted(
            SLANG_SUBGROUP_LANE_BUILTINS.items()
        )
    },
}


def lower_slang_builtin_input_members(ast):
    """``ast``, or a copy whose entry structures read their intrinsic-only built-in members in the entry body."""
    return lower_hlsl_builtin_input_members(ast, SLANG_INTRINSIC_INPUT_BUILTINS)

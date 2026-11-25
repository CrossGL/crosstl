"""CrossGL wave (subgroup) operations written as rust-gpu's ``spirv_std::arch`` subgroup functions.

spirv-std 0.11 (``crates/spirv-std/src/arch/subgroup.rs``) wraps each SPIR-V ``OpGroupNonUniform*`` instruction in a
function, and those instructions are what CrossGL's wave operations mean: they combine the *active* lanes of the
subgroup (:mod:`crosstl.translator.wave_operations`; GL_KHR_shader_subgroup). The Rust generator calls them:

- a reduction, scan or clustered reduction is ``subgroup_[inclusive_|exclusive_|clustered_]<kind><op>``, where the
  kind follows the operand's scalar type: ``i_``/``f_`` for ``add`` and ``mul``, ``s_``/``u_``/``f_`` for ``min`` and
  ``max``, nothing for the bitwise ``and``, ``or`` and ``xor`` of integers and ``logical_`` for those of ``bool``
  (``WaveActiveBitAnd`` of a ``bool`` is the logical and, as ``subgroupAnd(bool)`` is in GLSL). HLSL's
  ``WavePrefixSum`` and ``WavePrefixProduct`` are the exclusive scans. A clustered reduction takes its constant
  cluster size as spirv-std's ``CLUSTER_SIZE`` generic parameter, and is ``unsafe`` because the size must not
  exceed the subgroup;
- ``WaveActiveAnyTrue``, ``WaveActiveAllTrue`` and ``WaveActiveAllEqual`` are ``subgroup_any``, ``subgroup_all`` and
  ``subgroup_all_equal``; ``WaveIsFirstLane`` is ``subgroup_elect``;
- ``WaveActiveBallot`` is ``subgroup_ballot``, a ``UVec4`` mask; the ``WaveBallot*`` mask arithmetic is
  ``subgroup_ballot_bit_count``, ``subgroup_ballot_{inclusive,exclusive}_bit_count``, ``subgroup_ballot_find_lsb``,
  ``subgroup_ballot_find_msb``, ``subgroup_ballot_bit_extract`` and ``subgroup_inverse_ballot``; HLSL's
  ``WaveActiveCountBits`` and ``WavePrefixCountBits`` of a predicate count the bits of its ballot;
- ``WaveReadLaneFirst`` is ``subgroup_broadcast_first``. ``WaveReadLaneAt`` is ``subgroup_shuffle``
  (``OpGroupNonUniformShuffle``), which, unlike ``subgroup_broadcast``, is defined for a lane index that differs
  between lanes; ``WaveShuffleXor``, ``WaveShuffleUp`` and ``WaveShuffleDown`` are spirv-std's shuffles of the same
  names;
- ``QuadReadLaneAt`` is ``subgroup_quad_broadcast`` and ``QuadReadAcross{X,Y,Diagonal}`` is ``subgroup_quad_swap``
  in direction 0, 1 or 2 (``OpGroupNonUniformQuadSwap``). A compute stage without a derivative-group layout forms
  its quads from four consecutive subgroup invocations, as these instructions do in rust-gpu's ``GLCompute``
  entries; a stage that declares a derivative group is refused, since the generator writes no execution mode for
  it;
- ``WaveGetLaneIndex()`` and ``WaveGetLaneCount()`` are the ``subgroup_local_invocation_id`` and ``subgroup_size``
  built-ins, which :func:`rewrite_rust_gpu_lane_queries` substitutes before the generator binds entry parameters.

The partitioned operations and the rotations, which spirv-std lacks, reach the generator already lowered to these
(:mod:`crosstl.translator.semantics.partitioned_waves`, :mod:`crosstl.translator.semantics.subgroup_rotations`).
Every other operation, operand type or shape returns ``None``, and the caller refuses it.
"""

from __future__ import annotations

from ..ast import IdentifierNode, WaveOpNode
from ..wave_operations import (
    WAVE_CLUSTERED_OPERATIONS,
    WAVE_EXTENDED_PREFIX_OPERATIONS,
    cluster_size_value,
)
from .array_utils import evaluate_literal_int_expression
from .ast_walk import walk_ast_values
from .rust_gpu_math_calls import typed_receiver
from .spirv_compute_derivative_groups import DECLARED_GROUPS
from .stage_utils import stage_layout_entry

__all__ = [
    "generate_rust_gpu_wave_op",
    "rewrite_rust_gpu_lane_queries",
    "rust_gpu_wave_result_type",
]

_ARCH = "spirv_std::arch::"
_FLOATS = frozenset({"f32", "f64"})
_SIGNED = frozenset({"i8", "i16", "i32", "i64"})
_UNSIGNED = frozenset({"u8", "u16", "u32", "u64"})
# The glam vector types spirv-std implements ScalarOrVector for (crates/spirv-std/src/vector.rs).
_VECTOR_SCALARS = frozenset({"f32", "f64", "i32", "u32", "bool"})

_OPERATORS = {
    "sum": "add",
    "product": "mul",
    "min": "min",
    "max": "max",
    "and": "and",
    "or": "or",
    "xor": "xor",
}
# Reductions and scans: CrossGL operation -> (spirv-std operator, scope prefix).
_COMBINING = {
    "WaveActiveSum": ("add", ""),
    "WaveActiveProduct": ("mul", ""),
    "WaveActiveMin": ("min", ""),
    "WaveActiveMax": ("max", ""),
    "WaveActiveBitAnd": ("and", ""),
    "WaveActiveBitOr": ("or", ""),
    "WaveActiveBitXor": ("xor", ""),
    # HLSL's prefix sum and product exclude the calling lane.
    "WavePrefixSum": ("add", "exclusive_"),
    "WavePrefixProduct": ("mul", "exclusive_"),
    "WavePrefixInclusiveSum": ("add", "inclusive_"),
    "WavePrefixInclusiveProduct": ("mul", "inclusive_"),
    **{
        name: (operator, "inclusive_" if inclusive else "exclusive_")
        for name, (operator, inclusive) in WAVE_EXTENDED_PREFIX_OPERATIONS.items()
    },
}
_CLUSTERED = {
    name: _OPERATORS[operator] for name, operator in WAVE_CLUSTERED_OPERATIONS.items()
}
_VOTES = {"WaveActiveAnyTrue": "subgroup_any", "WaveActiveAllTrue": "subgroup_all"}
_BALLOT_MASK_OPERATIONS = {
    "WaveBallotBitCount": "subgroup_ballot_bit_count",
    "WaveBallotInclusiveBitCount": "subgroup_ballot_inclusive_bit_count",
    "WaveBallotExclusiveBitCount": "subgroup_ballot_exclusive_bit_count",
    "WaveBallotFindLSB": "subgroup_ballot_find_lsb",
    "WaveBallotFindMSB": "subgroup_ballot_find_msb",
}
_PREDICATE_COUNTS = {
    "WaveActiveCountBits": "subgroup_ballot_bit_count",
    "WavePrefixCountBits": "subgroup_ballot_exclusive_bit_count",
}
_LANE_READS = {
    "WaveReadLaneAt": "subgroup_shuffle",
    "WaveShuffleXor": "subgroup_shuffle_xor",
    "WaveShuffleUp": "subgroup_shuffle_up",
    "WaveShuffleDown": "subgroup_shuffle_down",
    "QuadReadLaneAt": "subgroup_quad_broadcast",
}
_QUAD_SWAPS = {"QuadReadAcrossX": 0, "QuadReadAcrossY": 1, "QuadReadAcrossDiagonal": 2}
_QUAD_OPERATIONS = frozenset({"QuadReadLaneAt", *_QUAD_SWAPS})
# CrossGL lane queries and the built-in inputs that hold them (rust_gpu_interface.COMPUTE_BUILTIN_INPUTS).
_LANE_QUERIES = {
    "WaveGetLaneIndex": "gl_SubgroupInvocationID",
    "WaveGetLaneCount": "gl_SubgroupSize",
}
_VALUE_RESULT = frozenset(
    {*_COMBINING, *_CLUSTERED, *_LANE_READS, *_QUAD_SWAPS, "WaveReadLaneFirst"}
)
_BOOL_RESULT = frozenset(
    {
        *_VOTES,
        "WaveActiveAllEqual",
        "WaveIsFirstLane",
        "WaveBallotBitExtract",
        "WaveInverseBallot",
    }
)
_UINT_RESULT = frozenset({*_BALLOT_MASK_OPERATIONS, *_PREDICATE_COUNTS})


def rewrite_rust_gpu_lane_queries(ast) -> None:
    """Replace ``WaveGetLaneIndex()`` and ``WaveGetLaneCount()`` in ``ast`` by the built-ins that hold them."""
    for node in list(walk_ast_values(ast)):
        attributes = getattr(node, "__dict__", None)
        if not attributes:
            continue
        for name, value in list(attributes.items()):
            if _lane_query(value) is not None:
                setattr(node, name, IdentifierNode(_lane_query(value)))
            elif isinstance(value, list):
                for index, item in enumerate(value):
                    builtin = _lane_query(item)
                    if builtin is not None:
                        value[index] = IdentifierNode(builtin)


def _lane_query(value):
    if isinstance(value, WaveOpNode) and not (getattr(value, "arguments", None) or []):
        return _LANE_QUERIES.get(getattr(value, "operation", None))
    return None


def _shape(generator, arg):
    """``(Rust scalar, vector size or 0)`` of an operand; ``(None, 0)`` when it is neither a scalar nor a vector."""
    type_name = generator.expression_result_type(arg)
    info = generator.vector_type_info(type_name)
    if info is not None:
        return generator.normalize_scalar_type(info["component_type"]), info["size"]
    return generator.normalize_scalar_type(type_name), 0


def _value_kind(scalar, size, operator):
    """The spirv-std kind prefix of ``operator`` on an operand, or None when spirv-std has no such function."""
    if size and scalar not in _VECTOR_SCALARS:
        return None
    if operator in {"and", "or", "xor"}:
        if scalar == "bool":
            return "logical_"
        return "" if scalar in _SIGNED | _UNSIGNED else None
    if scalar in _FLOATS:
        return "f_"
    if operator in {"add", "mul"}:
        return "i_" if scalar in _SIGNED | _UNSIGNED else None
    if scalar in _SIGNED:
        return "s_"
    return "u_" if scalar in _UNSIGNED else None


def _is_uvec4(generator, arg):
    return _shape(generator, arg) == ("u32", 4)


def _lane_index(generator, arg):
    """A lane index or delta as the ``u32`` spirv-std takes, or None when it is not an integer scalar."""
    scalar, size = _shape(generator, arg)
    if size or scalar not in _SIGNED | _UNSIGNED:
        return None
    value = generator.generate_expression(arg)
    return value if scalar == "u32" else f"(({value}) as u32)"


def _declares_derivative_group(generator):
    ast = getattr(generator, "rust_current_program", None)
    for stage in (getattr(ast, "stages", None) or {}).values():
        if any(stage_layout_entry(stage, name) is not None for name in DECLARED_GROUPS):
            return True
    return False


def generate_rust_gpu_wave_op(generator, operation, args):
    """The spirv-std expression for the CrossGL wave operation ``operation(args)``, or None when it has none."""
    args = list(args or [])
    if operation == "WaveIsFirstLane" and not args:
        return f"{_ARCH}subgroup_elect()"
    if not args:
        return None
    scalar, size = _shape(generator, args[0])
    if scalar is None:
        return None
    value = typed_receiver(generator.generate_expression(args[0]), scalar)
    if operation in _COMBINING and len(args) == 1:
        operator, scope = _COMBINING[operation]
        kind = _value_kind(scalar, size, operator)
        if kind is None:
            return None
        return f"{_ARCH}subgroup_{scope}{kind}{operator}({value})"
    if operation in _CLUSTERED and len(args) == 2:
        operator = _CLUSTERED[operation]
        kind = _value_kind(scalar, size, operator)
        cluster = cluster_size_value(evaluate_literal_int_expression(args[1]))
        if kind is None or cluster is None:
            return None
        return f"unsafe {{ {_ARCH}subgroup_clustered_{kind}{operator}::<{cluster}, _>({value}) }}"
    if operation in _VOTES and len(args) == 1:
        return (
            f"{_ARCH}{_VOTES[operation]}({value})"
            if (scalar, size) == ("bool", 0)
            else None
        )
    if operation == "WaveActiveAllEqual" and len(args) == 1:
        if size and scalar not in _VECTOR_SCALARS:
            return None
        return f"{_ARCH}subgroup_all_equal({value})"
    if operation == "WaveActiveBallot" and len(args) == 1:
        return (
            f"{_ARCH}subgroup_ballot({value})"
            if (scalar, size) == ("bool", 0)
            else None
        )
    if operation in _PREDICATE_COUNTS and len(args) == 1:
        if (scalar, size) != ("bool", 0):
            return None
        return f"{_ARCH}{_PREDICATE_COUNTS[operation]}({_ARCH}subgroup_ballot({value}))"
    if operation in _BALLOT_MASK_OPERATIONS and len(args) == 1:
        return (
            f"{_ARCH}{_BALLOT_MASK_OPERATIONS[operation]}({value})"
            if _is_uvec4(generator, args[0])
            else None
        )
    if operation == "WaveBallotBitExtract" and len(args) == 2:
        lane = _lane_index(generator, args[1])
        if lane is None or not _is_uvec4(generator, args[0]):
            return None
        return f"{_ARCH}subgroup_ballot_bit_extract({value}, {lane})"
    if operation == "WaveInverseBallot" and len(args) == 1:
        return (
            f"unsafe {{ {_ARCH}subgroup_inverse_ballot({value}) }}"
            if _is_uvec4(generator, args[0])
            else None
        )
    if size and scalar not in _VECTOR_SCALARS:
        return None
    if operation in _QUAD_OPERATIONS and _declares_derivative_group(generator):
        return None
    if operation == "WaveReadLaneFirst" and len(args) == 1:
        return f"{_ARCH}subgroup_broadcast_first({value})"
    if operation in _LANE_READS and len(args) == 2:
        lane = _lane_index(generator, args[1])
        if lane is None:
            return None
        return f"{_ARCH}{_LANE_READS[operation]}({value}, {lane})"
    if operation in _QUAD_SWAPS and len(args) == 1:
        return f"{_ARCH}subgroup_quad_swap::<{_QUAD_SWAPS[operation]}, _>({value})"
    return None


def rust_gpu_wave_result_type(generator, operation, args):
    """The CrossGL result type of a wave operation this module lowers, or None."""
    args = list(args or [])
    if operation in _BOOL_RESULT:
        return "bool"
    if operation in _UINT_RESULT:
        return "uint"
    if operation == "WaveActiveBallot":
        return "uvec4"
    if operation in _LANE_QUERIES:
        return "uint"
    if operation in _VALUE_RESULT and args:
        return generator.expression_result_type(args[0])
    return None

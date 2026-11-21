"""Metal lowering of quad operations in compute-like thread groups whose quads are 2x2 blocks of threads.

A CrossGL compute, mesh or task stage forms the quads of ``QuadReadAcrossX``, ``QuadReadAcrossY``,
``QuadReadAcrossDiagonal``, ``QuadReadLaneAt``, ``QuadAll`` and ``QuadAny`` (#4840):

- from four consecutive subgroup lanes when it declares no derivative group, as GLSL and SPIR-V do;
- from four consecutive local invocation indices with ``layout(derivative_group_linearKHR) in;``;
- from 2x2 blocks of the x and y local invocation ids with ``layout(derivative_group_quadsKHR) in;``, where the
  thread at ``(x, y)`` has quad lane ``(x & 1) | (y & 1) << 1``. The HLSL importer writes this layout for a thread
  group with even x and y, the Shader Model 6.6 rule.

A stage that takes derivatives and declares no group derives one by that rule, but Metal output refuses
derivatives outside fragment functions, so that case never reaches this module.

A Metal quad-group is four consecutive threads of a SIMD-group, and the SIMD-groups of a threadgroup hold
consecutive values of ``thread_index_in_threadgroup`` (x fastest). The first two groupings are therefore Metal's
quad-groups, and 2x2 quads agree with them when the thread group is two threads wide. For a width ``X`` that is a
power of two from 4 to 16 and an even height, the two rows of a 2x2 quad lie in one 32-thread SIMD-group, the
SIMD-group width of every Apple GPU family, and their SIMD lanes differ from the thread's lane in bit 0 and in
bit ``log2(X)``:

- ``QuadReadAcrossX`` stays ``quad_shuffle_xor(v, 1)``, because the lane it reads is in the same quad-group;
- ``QuadReadAcrossY`` and ``QuadReadAcrossDiagonal`` are ``simd_shuffle_xor`` with ``X`` and ``X + 1``;
- ``QuadReadLaneAt(v, i)`` is ``simd_shuffle`` of the quad's first lane plus ``(i & 1) + (i >> 1) * X``.

``QuadAny`` and ``QuadAll`` test the four quad lanes of ``simd_ballot`` against ``simd_active_threads_mask`` in
every compute-like stage, linear quads included (row distance 2), so that only active threads vote, as in HLSL
and ``OpGroupNonUniformQuadAnyKHR``. On an Apple M2 Max, ``quad_any`` in a branch with no other side effects votes
with the threads the branch excludes as well.

2x2 quads in any other thread group, in a thread group with an odd width or height, or in one whose size the
pipeline specializes or the host chooses, a stage that declares both groups, and a function that stages with
different quad layouts share are refused with ``MetalRefusal``. Metal's own ``quad_*`` calls import with no
derivative group and are never remapped.
"""

from __future__ import annotations

from ..ast import FunctionCallNode, WaveOpNode
from ..host_workgroup_size import HOST_WORKGROUP_SIZE_KEY, has_explicit_workgroup_size
from ..refusals import MetalRefusal
from .specialized_workgroup_size import has_specialized_workgroup_size
from .stage_utils import compute_local_size, normalize_stage_name

__all__ = [
    "GRID_QUAD_OPERATIONS",
    "collect_metal_grid_quad_widths",
    "grid_quad_row_width",
    "metal_grid_quad_lane_move",
    "metal_grid_quad_lane_requirements",
    "metal_grid_quad_vote",
    "stage_derivative_group",
]

# The SIMD-group width of the Apple GPU families (Metal Feature Set Tables, "SIMD-group size").
METAL_SIMD_GROUP_WIDTH = 32

GRID_QUAD_LANE_MOVES = frozenset(
    {"QuadReadAcrossX", "QuadReadAcrossY", "QuadReadAcrossDiagonal", "QuadReadLaneAt"}
)
GRID_QUAD_VOTES = frozenset({"QuadAll", "QuadAny"})
GRID_QUAD_OPERATIONS = GRID_QUAD_LANE_MOVES | GRID_QUAD_VOTES
# The operations whose remapped form reads the thread's SIMD lane.
_LANE_READING_OPERATIONS = frozenset({"QuadReadLaneAt", "QuadAll", "QuadAny"})
_COMPUTE_LIKE_STAGES = frozenset({"compute", "mesh", "task", "object", "amplification"})
# The row distance of quads of four consecutive threads, which Metal's quad-groups already hold.
LINEAR_QUADS = 2
# The derivative group layouts of GL_KHR_compute_shader_derivatives (and its NV predecessor).
_DERIVATIVE_GROUP_LAYOUTS = {
    "derivative_group_quadsKHR": "quads",
    "derivative_group_quadsNV": "quads",
    "derivative_group_linearKHR": "linear",
    "derivative_group_linearNV": "linear",
}
_ACROSS_MASKS = {
    "QuadReadAcrossX": 1,
    "QuadReadAcrossY": 2,
    "QuadReadAcrossDiagonal": 3,
}


def _operation_name(node):
    if isinstance(node, WaveOpNode):
        return getattr(node, "operation", None)
    if isinstance(node, FunctionCallNode):
        name = getattr(node, "name", None)
        return getattr(name, "name", name)
    return None


def _direct_grid_quad_operations(codegen, func):
    operations = set()
    for node in codegen.iter_ast_nodes(getattr(func, "body", None) or []):
        operation = _operation_name(node)
        if operation in GRID_QUAD_OPERATIONS:
            operations.add(operation)
    return operations


def grid_quad_row_width(size):
    """The SIMD-lane distance between the two rows of the 2x2 quads of a thread group of ``size``.

    ``2`` means the quads are four consecutive threads, Metal's quad-groups; ``0`` means the thread group has no
    2x2 quads (an odd width or height) or Metal's SIMD-group functions cannot reach them.
    """
    x, y, _z = size
    if x % 2 or y % 2:
        return 0
    if x & (x - 1) == 0 and 2 * x <= METAL_SIMD_GROUP_WIDTH:
        return x
    return 0


def stage_derivative_group(stage):
    """``"quads"``, ``"linear"`` or None: the ``layout(derivative_group_*) in;`` that ``stage`` declares."""
    groups = set()
    for qualifier in getattr(stage, "layout_qualifiers", None) or []:
        if getattr(qualifier, "direction", None) not in (None, "in"):
            continue
        for entry in getattr(qualifier, "entries", None) or []:
            group = _DERIVATIVE_GROUP_LAYOUTS.get(str(getattr(entry, "name", "")))
            if group is not None:
                groups.add(group)
    if len(groups) > 1:
        raise MetalRefusal(
            "a CrossGL stage declares both layout(derivative_group_quadsKHR) and "
            "layout(derivative_group_linearKHR); its quads can be formed only one way"
        )
    return next(iter(groups), None)


def _stage_row_width(stage_name, entry_name, stage):
    config = getattr(stage, "execution_config", None) or {}
    if stage_derivative_group(stage) != "quads":
        # Four consecutive subgroup lanes or local invocation indices: Metal's quad-groups.
        return LINEAR_QUADS
    if config.get(HOST_WORKGROUP_SIZE_KEY) and not has_explicit_workgroup_size(config):
        raise MetalRefusal(
            f"Metal {stage_name} function '{entry_name}' reads 2x2 quads of a thread group whose size the "
            "host chooses; the quads Metal's SIMD-group functions read depend on that size"
        )
    if has_specialized_workgroup_size(config):
        raise MetalRefusal(
            f"Metal {stage_name} function '{entry_name}' reads 2x2 quads of a thread group whose size the "
            "pipeline specializes; the quads Metal's SIMD-group functions read depend on that size"
        )
    try:
        size = tuple(int(str(value).strip(), 0) for value in compute_local_size(config))
    except ValueError:
        raise MetalRefusal(
            f"Metal {stage_name} function '{entry_name}' reads quads of a thread group whose size is not a "
            "constant; Metal output needs the size to find the SIMD lanes of its 2x2 quads"
        ) from None
    width = grid_quad_row_width(size)
    x, y, z = size
    if x % 2 or y % 2:
        raise MetalRefusal(
            f"Metal {stage_name} function '{entry_name}' declares layout(derivative_group_quadsKHR) for a "
            f"{x}x{y}x{z} thread group; 2x2 quads need an even width and height "
            "(GL_KHR_compute_shader_derivatives)"
        )
    if width == 0:
        raise MetalRefusal(
            f"Metal {stage_name} function '{entry_name}': the {x}x{y}x{z} thread group forms its quads from "
            "2x2 blocks of threads, and Metal reaches them with SIMD-group shuffles only when the width is "
            f"2 or a power of two up to {METAL_SIMD_GROUP_WIDTH // 2}; a Metal quad-group is four "
            "consecutive threads"
        )
    return width


def _reachable(start, calls):
    seen = set()
    pending = [start]
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        seen.add(name)
        pending.extend(sorted(calls.get(name, ())))
    return seen


def collect_metal_grid_quad_widths(codegen, ast, functions):
    """Map each function that compute-like stages call and that reads quads to the row width of their quads.

    Functions that are absent keep Metal's ``quad_*`` functions.
    """
    direct = {}
    calls = {}
    for func in functions or []:
        name = getattr(func, "name", None)
        if not name:
            continue
        operations = _direct_grid_quad_operations(codegen, func)
        if operations:
            direct[name] = operations
        calls[name] = codegen.called_user_function_names(func)
    if not direct:
        return {}

    layouts = {}
    for stage_type, stage in (getattr(ast, "stages", None) or {}).items():
        entry = getattr(stage, "entry_point", None)
        entry_name = getattr(entry, "name", None)
        if not entry_name:
            continue
        users = sorted(name for name in _reachable(entry_name, calls) if name in direct)
        if not users:
            continue
        stage_name = normalize_stage_name(stage_type)
        width = None
        if stage_name in _COMPUTE_LIKE_STAGES:
            width = _stage_row_width(stage_name, entry_name, stage)
        for name in users:
            layouts.setdefault(name, set()).add(width)

    widths = {}
    for name in sorted(layouts):
        found = layouts[name]
        if found == {None, LINEAR_QUADS} and not direct[name] & GRID_QUAD_VOTES:
            # Linear quads read lanes with the same quad_* calls as the other stages.
            found = {None}
        if len(found) > 1:
            raise MetalRefusal(
                f"Metal function '{name}' reads quads in stages whose quads are formed differently; Metal "
                "output would need one copy of the function per quad layout"
            )
        (width,) = found
        if width is not None:
            widths[name] = width
    return widths


def _current_width(codegen):
    widths = getattr(codegen, "metal_grid_quad_widths", None) or {}
    return widths.get(getattr(codegen, "current_function_name", None))


def metal_grid_quad_lane_requirements(codegen, func):
    """``{"WaveGetLaneIndex"}`` when the quad operations of ``func`` read the thread's SIMD lane."""
    widths = getattr(codegen, "metal_grid_quad_widths", None) or {}
    width = widths.get(getattr(func, "name", None))
    if width is None:
        return set()
    reading = GRID_QUAD_VOTES if width == LINEAR_QUADS else _LANE_READING_OPERATIONS
    if _direct_grid_quad_operations(codegen, func) & reading:
        return {"WaveGetLaneIndex"}
    return set()


def _lane(codegen, operation, arguments):
    lane = codegen.current_metal_wave_lane_index_parameter
    if lane is None:
        return None, codegen.metal_wave_diagnostic_expression(
            operation,
            arguments,
            "in a compute-like stage requires a thread_index_in_simdgroup value",
        )
    return lane, None


def metal_grid_quad_lane_move(codegen, operation, arguments, value):
    """The SIMD-group read of ``value`` that ``operation`` makes in the current function, or None.

    ``value`` is the generated payload. None means the current function reads lanes with Metal's quad-groups.
    """
    width = _current_width(codegen)
    if width in (None, LINEAR_QUADS) or operation not in GRID_QUAD_LANE_MOVES:
        return None
    if operation == "QuadReadAcrossX":
        return f"quad_shuffle_xor({value}, ushort(1))"
    if operation in _ACROSS_MASKS:
        mask = width if operation == "QuadReadAcrossY" else width + 1
        return f"simd_shuffle_xor({value}, ushort({mask}))"
    quad_lane = codegen.literal_int_value(arguments[1])
    if quad_lane is None:
        return codegen.metal_wave_diagnostic_expression(
            operation,
            arguments,
            "in a thread group of 2x2 quads requires a constant quad lane index",
        )
    lane, diagnostic = _lane(codegen, operation, arguments)
    if diagnostic is not None:
        return diagnostic
    offset = (quad_lane & 1) + ((quad_lane >> 1) & 1) * width
    return f"simd_shuffle({value}, ushort(({lane} & ~{width + 1}u) | {offset}u))"


def metal_grid_quad_vote(codegen, operation, arguments):
    """The ``QuadAll``/``QuadAny`` of the current function's quads over their active threads, or None.

    Apple's compiler moves ``quad_any`` and ``quad_all`` out of a branch whose other effects it can speculate, so
    that threads the branch excludes vote too; ``simd_ballot`` and ``simd_active_threads_mask`` stay in place.
    """
    width = _current_width(codegen)
    if width is None or operation not in GRID_QUAD_VOTES:
        return None
    lane, diagnostic = _lane(codegen, operation, arguments)
    if diagnostic is not None:
        return diagnostic
    predicate = codegen.generate_expression(arguments[0])
    quad = f"(simd_vote::vote_t({3 | 3 << width}u) << ({lane} & ~{width + 1}u))"
    ballot = f"(simd_vote::vote_t(simd_ballot({predicate})) & {quad})"
    if operation == "QuadAny":
        return f"({ballot} != simd_vote::vote_t(0u))"
    return f"({ballot} == (simd_vote::vote_t(simd_active_threads_mask()) & {quad}))"

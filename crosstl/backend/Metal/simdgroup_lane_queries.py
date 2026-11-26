"""Import Metal's lane-index intrinsic, which any function may call.

A kernel parameter declared ``[[thread_index_in_simdgroup]]`` gives the calling lane's index in its SIMD-group, but
only the kernel itself can declare it. Helper functions read the same value with the compiler intrinsic::

    const ushort lane = __metal_get_thread_index_in_simdgroup(ushort());

The value-initialised argument only selects the result type. The intrinsic returns the value the attribute would bind
for the calling thread, so it becomes CrossGL's ``WaveGetLaneIndex()``, which every function may call, converted to
that type. The Metal emitter threads ``WaveGetLaneIndex()`` into helper functions as a
``[[thread_index_in_simdgroup]]`` parameter of the kernel; DirectX and OpenGL read their own lane index.

Any other argument form is refused rather than passed through unchanged.
"""

from __future__ import annotations

from crosstl.backend.common_ast import VectorConstructorNode
from crosstl.translator.refusals import MetalSourceRefusal

LANE_INDEX_INTRINSIC = "__metal_get_thread_index_in_simdgroup"
_INTEGER_RESULT_TYPES = frozenset(
    {
        "ushort",
        "uint16_t",
        "short",
        "int16_t",
        "uint",
        "uint32_t",
        "unsigned int",
        "int",
        "int32_t",
    }
)


class MetalLaneQueryError(MetalSourceRefusal):
    """A call of Metal's lane-index intrinsic whose result type is not a value-initialised integer type."""


def generate_simdgroup_lane_query(converter, expression) -> str | None:
    """The CrossGL spelling of a lane-index intrinsic call, or None for any other call."""
    name = str(getattr(expression, "name", "")).lstrip(":")
    if name.startswith("metal::"):
        name = name[len("metal::") :]
    if name != LANE_INDEX_INTRINSIC or name in converter.user_function_names:
        return None
    arguments = list(getattr(expression, "args", None) or ())
    result_type = None
    if len(arguments) == 1 and isinstance(arguments[0], VectorConstructorNode):
        if not arguments[0].args:
            result_type = str(arguments[0].vector_type or "").strip()
    if result_type not in _INTEGER_RESULT_TYPES:
        raise MetalLaneQueryError(
            f"{LANE_INDEX_INTRINSIC} takes one value-initialised integer type "
            "argument, such as ushort(), that selects its result type"
        )
    return f"{converter.map_type(result_type)}(WaveGetLaneIndex())"

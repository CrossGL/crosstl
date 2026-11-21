"""The derivative group of a compute-like SPIR-V entry that takes derivatives or uses quad operations.

SPV_KHR_compute_shader_derivatives lets ``GLCompute``, ``MeshEXT`` and ``TaskEXT`` entries take derivatives and
query an implicit level of detail. The entry declares how its invocations form quads:

- ``DerivativeGroupQuadsKHR`` (capability ``ComputeDerivativeGroupQuadsKHR``) takes 2x2 quads of the x and y
  dimensions, and needs both dimensions to be even;
- ``DerivativeGroupLinearKHR`` (capability ``ComputeDerivativeGroupLinearKHR``) takes four consecutive local
  invocation indices, and needs the invocation count to be a multiple of four.

HLSL Shader Model 6.6 forms compute quads the same way: 2x2 quads when the x and y dimensions are even, and
otherwise four consecutive values of ``SV_GroupIndex``, which needs the thread count to be a multiple of four.
DXC's SPIR-V output chooses the mode by the same rule.

The workgroup size is part of the dispatch contract: the host sizes its grid by it, and ``gl_LocalInvocationID``,
``gl_GlobalInvocationID`` and workgroup memory are indexed by it. It is therefore never changed to fit a mode; a
size that fits neither mode is refused.

The mode also decides which invocations a quad operation (``OpGroupNonUniformQuadSwap`` and
``OpGroupNonUniformQuadBroadcast``) reads in these execution models: the Vulkan specification forms compute quads by
the declared derivative group, and otherwise from four consecutive subgroup invocations. HLSL Shader Model 6.6 forms
the quads of its quad operations by the derivative rule above, and DXC declares the same mode for an entry that uses
one, so an entry that reaches a quad operation declares the group as an entry that takes derivatives does. A
workgroup that fits neither mode has no HLSL quad layout (DXC's DXIL output accepts such a kernel and forms its quads
from wave lanes, and its SPIR-V output fails validation), so its quad operations keep Vulkan's subgroup quads and
the size is not refused.
"""

from __future__ import annotations

DERIVATIVES = "compute derivatives"
QUAD_OPERATIONS = "quad operations in compute, mesh and task entries"
QUADS = ("DerivativeGroupQuadsKHR", "ComputeDerivativeGroupQuadsKHR")
LINEAR = ("DerivativeGroupLinearKHR", "ComputeDerivativeGroupLinearKHR")


def compute_derivative_group(size):
    """The ``(execution mode, capability)`` that forms quads in a workgroup of ``size``, or None if neither can."""
    x, y, z = size
    if x % 2 == 0 and y % 2 == 0:
        return QUADS
    if (x * y * z) % 4 == 0:
        return LINEAR
    return None


def mark_quad_operation(generator):
    """Record that the function being generated uses a quad operation."""
    if generator.current_function_id is not None:
        generator.quad_operation_function_ids.add(generator.current_function_id)


def record_function_call(generator, callee_id):
    """Record that the function being generated calls the function ``callee_id``."""
    if generator.current_function_id is not None:
        generator.function_callee_ids.setdefault(
            generator.current_function_id, set()
        ).add(callee_id)


def entry_uses_quad_operations(generator, function_id) -> bool:
    """Whether the entry ``function_id``, or a function it calls, uses a quad operation."""
    pending = [function_id]
    visited = set()
    while pending:
        current = pending.pop()
        if current in visited:
            continue
        visited.add(current)
        if current in generator.quad_operation_function_ids:
            return True
        pending.extend(sorted(generator.function_callee_ids.get(current, ())))
    return False


def entry_derivative_group_purpose(generator, function_id):
    """Why the entry ``function_id`` needs a derivative group, or None when it needs none."""
    if generator.requires_compute_derivatives:
        return DERIVATIVES
    if entry_uses_quad_operations(generator, function_id):
        return QUAD_OPERATIONS
    return None


def emit_compute_derivative_group(generator, function_id, size, purpose=DERIVATIVES):
    """Declare the derivative group of the entry ``function_id`` of workgroup ``size``.

    Derivatives need a group, so a size that fits neither is refused; quad operations keep Vulkan's subgroup quads.
    """
    group = compute_derivative_group(size)
    if group is None:
        if purpose == QUAD_OPERATIONS:
            return
        x, y, z = size
        generator.refuse_lowering(
            f"SPIR-V {purpose} need a workgroup with even x and y dimensions "
            f"(DerivativeGroupQuadsKHR) or a multiple of four invocations (DerivativeGroupLinearKHR); "
            f"the {x}x{y}x{z} workgroup has neither, and changing it would change the dispatch"
        )
        return
    mode, capability = group
    generator.require_extension("SPV_KHR_compute_shader_derivatives")
    generator.require_capability(capability)
    generator.emit(f"OpExecutionMode %{function_id.id} {mode}")

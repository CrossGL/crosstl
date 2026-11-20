"""The derivative group of a compute-like SPIR-V entry that takes derivatives.

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
"""

from __future__ import annotations

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


def emit_compute_derivative_group(generator, function_id, size):
    """Declare the derivative group of the entry ``function_id`` of workgroup ``size``, or refuse the size."""
    group = compute_derivative_group(size)
    if group is None:
        x, y, z = size
        generator.refuse_lowering(
            f"SPIR-V compute derivatives need a workgroup with even x and y dimensions "
            f"(DerivativeGroupQuadsKHR) or a multiple of four invocations (DerivativeGroupLinearKHR); "
            f"the {x}x{y}x{z} workgroup has neither, and changing it would change the dispatch"
        )
        return
    mode, capability = group
    generator.require_capability(capability)
    generator.emit(f"OpExecutionMode %{function_id.id} {mode}")

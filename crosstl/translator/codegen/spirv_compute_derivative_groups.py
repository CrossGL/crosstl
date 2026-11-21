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

A CrossGL stage can also declare its group with ``layout(derivative_group_quadsKHR) in;`` or
``layout(derivative_group_linearKHR) in;`` (GL_KHR_compute_shader_derivatives; the NV spellings mean the same). The
group decides which invocations form the quads of its derivatives and of its quad operations
(``OpGroupNonUniformQuadSwap``, ``OpGroupNonUniformQuadBroadcast`` and the SPV_KHR_quad_control votes): the Vulkan
specification forms compute quads by the declared group, and otherwise from four consecutive subgroup invocations,
which is what a CrossGL stage without the layout means (#4840). A declared group therefore takes precedence over the
rule above, and a stage whose quad operations need no derivatives declares no group.
"""

from __future__ import annotations

from .stage_utils import stage_layout_entries

QUADS = ("DerivativeGroupQuadsKHR", "ComputeDerivativeGroupQuadsKHR")
LINEAR = ("DerivativeGroupLinearKHR", "ComputeDerivativeGroupLinearKHR")

# The CrossGL stage layouts that declare a group, by the lower-case name stage_utils normalizes them to.
DECLARED_GROUPS = {
    "derivative_group_quadskhr": QUADS,
    "derivative_group_quadsnv": QUADS,
    "derivative_group_linearkhr": LINEAR,
    "derivative_group_linearnv": LINEAR,
}


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


def declared_derivative_group(generator, stage):
    """The ``(execution mode, capability)`` the stage's ``layout(derivative_group_*) in;`` declares, or None."""
    groups = []
    for entry in stage_layout_entries(stage, "in"):
        group = DECLARED_GROUPS.get(str(getattr(entry, "name", "")).lower())
        if group is not None and group not in groups:
            groups.append(group)
    if len(groups) > 1:
        generator.refuse_lowering(
            "a stage declares both layout(derivative_group_quadsKHR) and layout(derivative_group_linearKHR); "
            "SPIR-V forms its quads one way"
        )
        return None
    return groups[0] if groups else None


def emit_declared_derivative_group(generator, function_id, group, size):
    """Declare the group ``group`` the entry ``function_id`` names, or refuse a workgroup ``size`` it cannot divide.

    ``size`` is None for a size the pipeline specializes, whose values the host must fit to the declared group, as
    GLSL leaves them.
    """
    mode, capability = group
    if size is not None:
        x, y, z = size
        if group == QUADS and (x % 2 or y % 2):
            generator.refuse_lowering(
                f"layout(derivative_group_quadsKHR) forms 2x2 quads, which need even workgroup x and y "
                f"dimensions; the {x}x{y}x{z} workgroup has none, and changing it would change the dispatch"
            )
            return
        if group == LINEAR and (x * y * z) % 4:
            generator.refuse_lowering(
                f"layout(derivative_group_linearKHR) forms quads of four consecutive invocations, which need a "
                f"multiple of four invocations; the {x}x{y}x{z} workgroup has {x * y * z}"
            )
            return
    generator.require_capability(capability)
    generator.require_extension("SPV_KHR_compute_shader_derivatives")
    generator.emit(f"OpExecutionMode %{function_id.id} {mode}")

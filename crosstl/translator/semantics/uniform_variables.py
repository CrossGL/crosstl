"""Module-scope uniform address-space variables in CrossGL, declared as the constant blocks every generator binds.

CrossGL also writes a constant buffer the WGSL way, as a module-scope variable in the ``uniform`` address space with
a group and a binding, ``@group(0) @binding(2) var<uniform> x: T`` (WGSL, "Address Spaces"). The CUDA and HIP
importers write ``__constant__`` memory in this form. The variable is the buffer's whole content, read-only in every
invocation and set by the host. The generators other than SPIR-V's printed the declaration as a plain global: HLSL
placed it in the ``$Globals`` constant buffer and GLSL in the default uniform block, so the group and binding the host
binds were lost, and GLSL for Vulkan, which has no default uniform block, rejected it.

For every target without its own spelling, :func:`lower_uniform_variables` declares each such variable as a constant
block of the same group and binding, ``cbuffer x_block @group(g) @binding(b) { T x; }``, keeping the variable's other
attributes (such as ``@buffer_layout``). The block holds one ``T`` at byte 0 and its member keeps the variable's name,
so every reference reads the block member unchanged. A block is laid out by the rules of a constant block: those of
the target, or the ``@buffer_layout`` it carries (:mod:`.constant_buffer_layouts`).

A variable with an initializer holds that value and nothing writes it in any invocation, so it becomes the program
constant ``const T x = init;``. That is the meaning of an OpenCL program-scope ``__constant`` variable, which must be
initialized and which the host cannot write (OpenCL C 3.0, "__constant (or constant)"). It is also what a CUDA
``__constant__`` variable with an initializer holds until the host writes it with ``cudaMemcpyToSymbol``; a host
that does must bind a constant buffer instead. The generators had dropped the initializer: HLSL kept it only as the
reflected default of a ``$Globals`` member, which no runtime applies, and MSL declared a buffer argument the host
had to fill.

CUDA and HIP declare the variable natively, as ``__constant__`` memory, and CrossGL and WGSL output keep it as
written.
"""

from __future__ import annotations

from .. import ast as A
from .compute_entry_values import _global_names, _unique
from .storage_blocks import _literal_int

__all__ = ["KEEPING_TARGETS", "lower_uniform_variables"]

KEEPING_TARGETS = frozenset({"", "cgl", "crossgl", "cuda", "hip", "wgsl"})


def _is_uniform_variable(node) -> bool:
    if not isinstance(node, A.VariableNode) or not getattr(
        node, "is_var_address_space", False
    ):
        return False
    qualifiers = [
        str(q).lower()
        for q in (
            getattr(node, "address_space_qualifiers", None)
            or getattr(node, "qualifiers", None)
            or []
        )
    ]
    return qualifiers == ["uniform"]


def _literal_counts(type_node):
    """``array<T, N>`` keeps its count as written; a block member's count is an expression."""
    while isinstance(type_node, A.ArrayType):
        size = type_node.size
        if isinstance(size, (int, str)) and not isinstance(size, bool):
            count = _literal_int(size)
            if count is not None:
                type_node.size = A.LiteralNode(count, A.PrimitiveType("int"))
        type_node = type_node.element_type
    return type_node


def lower_uniform_variables(program, target) -> None:
    """Declare ``program``'s module-scope ``var<uniform>`` variables as constant blocks for ``target``."""
    if str(target or "") in KEEPING_TARGETS or not isinstance(program, A.ShaderNode):
        return
    variables = [
        v
        for v in getattr(program, "global_variables", None) or []
        if _is_uniform_variable(v)
    ]
    if not variables:
        return
    if getattr(program, "cbuffers", None) is None:
        program.cbuffers = []
    if getattr(program, "constants", None) is None:
        program.constants = []
    used = _global_names(program)
    for variable in variables:
        _literal_counts(variable.var_type)
        program.global_variables.remove(variable)
        if getattr(variable, "initial_value", None) is not None:
            constant = A.ConstantNode(
                variable.name, variable.var_type, variable.initial_value
            )
            constant.source_location = getattr(variable, "source_location", None)
            program.constants.append(constant)
            continue
        block = A.StructNode(
            _unique(f"{variable.name}_block", used),
            [A.StructMemberNode(variable.name, variable.var_type)],
            attributes=list(variable.attributes or []),
        )
        block.is_cbuffer = True
        block.source_location = getattr(variable, "source_location", None)
        program.cbuffers.append(block)

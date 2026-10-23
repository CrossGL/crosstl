"""SPIR-V descriptor arrays of uniform blocks: ``ConstantBuffer<T> name[N]`` and GLSL ``uniform B { ... } name[N]``.

An array of uniform blocks binds one buffer per element: the descriptor set layout binding has ``N`` descriptors
(Vulkan 1.3 section 15.6.1), and SPIR-V declares it as a ``Uniform`` variable whose type is an array of a
``Block``-decorated struct (SPIR-V 1.6 section 2.16.2, the form glslang and DXC emit). An unsized array is a
runtime descriptor array, which needs the ``RuntimeDescriptorArray`` capability from ``SPV_EXT_descriptor_indexing``
(core in SPIR-V 1.5). ``name[i].member`` is then one access chain: the descriptor index, then the member.

The block members are laid out with std140, the layout of a uniform block (GLSL 4.60 section 7.6.2.2) and the one
the single-buffer ``ConstantBuffer<T>`` lowering uses. A block that declares another ``@ buffer_layout`` is refused,
as is an array whose element is not a struct.

A descriptor index marked ``nonuniformEXT`` or ``NonUniformResourceIndex`` makes the access chain ``NonUniform``
through the generator's usual propagation; indexing an array of uniform blocks that way also needs the
``UniformBufferArrayNonUniformIndexing`` capability (SPIR-V 1.6 section 3.31), which
:func:`require_non_uniform_block_array_indexing` adds.

A constant buffer bound by an HLSL ``register(bN, spaceM)`` takes its slot through the register mapping of
:mod:`spirv_hlsl_registers`, so :func:`lowered_constant_buffer_nodes` lists every declaration lowered from
``ConstantBuffer<T>``, one buffer or an array of them.
"""

from __future__ import annotations

from ..semantics.constant_buffers import (
    is_constant_buffer_array,
    is_lowered_constant_buffer,
)
from .glsl_block_layout import declared_buffer_layout

__all__ = [
    "declare_uniform_block_array",
    "is_uniform_block_array_node",
    "lowered_constant_buffer_nodes",
    "require_non_uniform_block_array_indexing",
]


def is_uniform_block_array_node(node) -> bool:
    """Whether ``node`` declares one uniform-block descriptor per array element."""
    return is_constant_buffer_array(node)


def lowered_constant_buffer_nodes(ast):
    """The module and stage declarations lowered from ``ConstantBuffer<T>``, which HLSL ``register`` slots bind."""
    nodes = list(getattr(ast, "global_variables", None) or [])
    for stage in (getattr(ast, "stages", None) or {}).values():
        nodes.extend(getattr(stage, "local_variables", None) or [])
    return [node for node in nodes if is_lowered_constant_buffer(node)]


def _refuse(node, detail):
    from .SPIRV_codegen import UnsupportedSPIRVFeatureError

    return UnsupportedSPIRVFeatureError(
        "array of uniform blocks",
        f"array of constant buffers '{node.name}' {detail}",
        missing_capabilities=("spirv.uniform_block_array",),
    )


def declare_uniform_block_array(codegen, node, value_type):
    """Declare ``node`` (``uniform T name[N]`` marked as a block array) as ``N`` uniform-block descriptors."""
    layout = declared_buffer_layout(getattr(node, "attributes", None)) or "std140"
    if layout != "std140":
        raise _refuse(
            node,
            f"declares the {layout} layout; the SPIR-V emitter lays out arrays of uniform blocks with std140 only",
        )
    array_info = codegen.array_type_info_from_type(value_type)
    if array_info is None:
        raise _refuse(node, "has no array type")
    element_type, size = array_info
    if codegen.current_struct_members.get(element_type.type.base_type) is None:
        raise _refuse(node, "holds no struct, so it has no uniform block type")
    block_type = codegen.storage_layout_type(element_type, "std140")
    members = codegen.current_struct_members.get(block_type.type.base_type)
    if members is None:
        members = codegen.current_struct_members[element_type.type.base_type]
    codegen.decorate_cbuffer_type(block_type, members)
    if size is None:
        codegen.require_capability("RuntimeDescriptorArray")
        codegen.require_extension("SPV_EXT_descriptor_indexing")
    array_type = codegen.register_array_type(block_type, size)

    var_id = codegen.create_variable(array_type, "Uniform", node.name)
    descriptor_set, binding = codegen.resource_descriptor_slot(node)
    codegen.decorations.append(
        f"OpDecorate %{var_id.id} DescriptorSet {descriptor_set}"
    )
    codegen.decorations.append(f"OpDecorate %{var_id.id} Binding {binding}")
    codegen.uniform_buffers.append(var_id)
    _block_array_ids(codegen).add(var_id.id)
    codegen.register_global_variable_name(node.name, var_id, "Uniform")
    return var_id


def _block_array_ids(codegen):
    ids = getattr(codegen, "uniform_block_array_ids", None)
    if ids is None:
        ids = set()
        codegen.uniform_block_array_ids = ids
    return ids


def require_non_uniform_block_array_indexing(codegen, base_id, indices):
    """Require ``UniformBufferArrayNonUniformIndexing`` when ``indices[0]`` selects a block descriptor non-uniformly."""
    if not indices or getattr(base_id, "id", None) not in _block_array_ids(codegen):
        return
    if codegen.is_non_uniform_value(indices[0]):
        codegen.require_capability("UniformBufferArrayNonUniformIndexing")

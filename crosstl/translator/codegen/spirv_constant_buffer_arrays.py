"""SPIR-V descriptor arrays of uniform blocks: CrossGL ``ConstantBuffer<T> name[N]`` and ``ConstantBuffer<T> name[]``.

An array of constant buffers binds one buffer per element: HLSL ``ConstantBuffer<T> lights[4] : register(b0)``
and GLSL ``uniform Lights { ... } lights[4];`` (GLSL 4.60 section 4.3.9, "an array of uniform blocks"). Vulkan
binds that as one descriptor-set binding with a descriptor count of ``N`` (Vulkan 1.3 section 15.6.4): a
``Uniform`` variable whose type is an ``OpTypeArray`` of a ``Block``-decorated struct (Vulkan 1.3 section 15.8.1,
"Descriptor Set Interface"), as glslang and DXC declare it. An unsized array is an
``OpTypeRuntimeArray`` of blocks and needs the ``RuntimeDescriptorArray`` capability (SPV_EXT_descriptor_indexing,
core in SPIR-V 1.5).

The block is a std140 clone of ``T`` of its own, so a ``T`` held by value or nested in another block keeps its
own layout and no nested struct is ever decorated ``Block``. Member reads go through the ordinary access chains
(``%var %index %member``), and reading a whole element converts the std140 clone back to a ``T`` value as every
other uniform-block read does.

Indexing the array with a ``NonUniformResourceIndex`` or ``nonuniformEXT`` index decorates the access chain
``NonUniform`` and declares ``UniformBufferArrayNonUniformIndexing`` (SPIR-V 1.6 section 3.31, capability 5306):
without it Vulkan requires the index into a uniform-buffer array to be dynamically uniform (Vulkan 1.3
appendix A, "Shader Capabilities").

An HLSL constant buffer keeps its ``register(bN, spaceM)`` slot: :func:`constant_buffer_register_nodes` hands every
module- or stage-scope declaration bound to a ``b`` register (a ``ConstantBuffer<T>`` lowered to ``uniform T``, or an
array of them) to the register allocator of :mod:`.spirv_hlsl_registers`, which places it at binding ``N`` of set
``M`` as it places textures, views and samplers.

Arrays of arrays of constant buffers and arrays of ``ConstantBuffer`` of a non-struct type have no faithful lowering
here and are refused through the emitter's lowering diagnostics.
"""

from __future__ import annotations

from ..semantics.constant_buffers import constant_buffer_element
from .spirv_hlsl_registers import _register

__all__ = [
    "constant_buffer_array_element",
    "constant_buffer_register_nodes",
    "declare_constant_buffer_array",
    "require_uniform_buffer_array_non_uniform_indexing",
]

_BLOCK_LAYOUT = "uniform_block"


def constant_buffer_array_element(codegen, node):
    """``(type name of T, array extent)`` for a ``ConstantBuffer<T> name[...]`` declaration, else ``None``.

    The extent is ``None`` for an unsized array, an ``int`` for a literal extent or the name of a specialization
    constant, as :meth:`split_outer_array_type` reports it.
    """
    declared = getattr(node, "var_type", None) or getattr(node, "vtype", None)
    if declared is None:
        return None
    type_name = codegen.type_name_from_value(declared)
    outer = codegen.split_outer_array_type(type_name)
    if outer is None:
        return None
    element_name, extent = outer
    element = constant_buffer_element(element_name.strip())
    if element is None:
        return None
    return codegen.type_name_from_value(element), extent, element_name


def _block_type(codegen, struct_name):
    """The ``Block``-decorated std140 clone of the struct ``struct_name``, created and decorated once."""
    blocks = codegen.__dict__.setdefault("constant_buffer_array_blocks", {})
    if struct_name in blocks:
        return blocks[struct_name]
    struct_type = codegen.struct_types[struct_name]
    members = [
        (codegen.storage_layout_type(member_type, "std140"), member_name)
        for member_type, member_name in codegen.current_struct_members[struct_name]
    ]
    block = codegen.register_layout_struct_type(struct_type, _BLOCK_LAYOUT, members)
    codegen.decorate_cbuffer_type(block, members)
    blocks[struct_name] = block
    return block


def _placeholder(codegen, node):
    float_type = codegen.register_primitive_type("float")
    return codegen.create_variable(float_type, "Private", node.name)


def declare_constant_buffer_array(codegen, node):
    """Declare ``ConstantBuffer<T> name[N]`` as a ``Uniform`` array of ``Block`` structs, or return ``None``.

    Returns ``None`` when ``node`` is not an array of constant buffers. A declaration this lowering cannot express
    is refused through :meth:`refuse_lowering` and gets a ``Private`` placeholder, so that generation reaches every
    other refusal; no module is returned while a refusal is pending.
    """
    info = constant_buffer_array_element(codegen, node)
    if info is None:
        return None
    struct_name, extent, element_name = info
    if "[" in element_name:
        codegen.refuse_lowering(
            f"array of arrays of constant buffers '{node.name}': a Vulkan descriptor-set binding holds a "
            "one-dimensional array of descriptors (VUID-StandaloneSpirv-UniformConstant-04655)"
        )
        return _placeholder(codegen, node)
    if struct_name not in codegen.struct_types:
        codegen.refuse_lowering(
            f"array of constant buffers '{node.name}' of '{struct_name}': a Vulkan uniform block is a "
            "Block-decorated struct"
        )
        return _placeholder(codegen, node)

    block = _block_type(codegen, struct_name)
    array_type = codegen.register_array_type(block, extent)
    if extent is None:
        codegen.require_capability("RuntimeDescriptorArray")
        codegen.require_extension("SPV_EXT_descriptor_indexing")
    var_id = codegen.create_variable(array_type, "Uniform", node.name)
    descriptor_set, binding = codegen.resource_descriptor_slot(node)
    codegen.decorations.append(
        f"OpDecorate %{var_id.id} DescriptorSet {descriptor_set}"
    )
    codegen.decorations.append(f"OpDecorate %{var_id.id} Binding {binding}")
    codegen.uniform_buffers.append(var_id)
    codegen.__dict__.setdefault("constant_buffer_array_variables", set()).add(var_id.id)
    codegen.register_global_variable_name(node.name, var_id, "Uniform")
    return var_id


def require_uniform_buffer_array_non_uniform_indexing(codegen, base_id, indices):
    """Declare ``UniformBufferArrayNonUniformIndexing`` when a non-uniform index selects a constant buffer."""
    variables = codegen.__dict__.get("constant_buffer_array_variables")
    if not variables or getattr(base_id, "id", None) not in variables or not indices:
        return
    if codegen.is_non_uniform_value(indices[0]):
        codegen.require_non_uniform_descriptor_indexing()
        codegen.require_capability("UniformBufferArrayNonUniformIndexing")


def constant_buffer_register_nodes(codegen, ast):
    """Module- and stage-scope declarations bound to an HLSL ``b`` register, in declaration order."""
    nodes = list(getattr(ast, "global_variables", None) or [])
    for stage in (getattr(ast, "stages", None) or {}).values():
        nodes.extend(getattr(stage, "local_variables", None) or [])
    for node in nodes:
        register = _register(codegen, node)
        if register is not None and register[0] == "b":
            yield node

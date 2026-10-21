"""SPIR-V push-constant blocks for CrossGL ``cbuffer`` declarations marked ``@push_constant``.

GLSL ``layout(push_constant) uniform`` blocks, HLSL ``[[vk::push_constant]]`` buffers and SPIR-V ``PushConstant``
variables all import as ``cbuffer Name @push_constant { ... }``. Vulkan reads such a block from the command buffer
(``vkCmdPushConstants``), not from a descriptor, so the module must declare it in the ``PushConstant`` storage class
with no ``DescriptorSet`` or ``Binding`` (Vulkan 1.3 section 14.5.1). Its members follow the standard storage-buffer
layout, which is GLSL's ``std430`` (Vulkan 1.3 section 15.6.4, GL_KHR_vulkan_glsl), unless the block names another
one with ``@ buffer_layout(std140)`` or ``@ buffer_layout(scalar)`` (GLSL ``layout(push_constant, std140)``,
GL_EXT_scalar_block_layout); a ``Uniform`` block at the ``std140`` offsets would both bind through the wrong interface
and move every array and scalar after a vector.

A stage statically uses at most one push-constant variable (VUID-StandaloneSpirv-OpEntryPoint-06674), and GLSL and
HLSL both allow a single push-constant block per stage, so a second block is refused.
"""

from __future__ import annotations

__all__ = [
    "declare_push_constant_block",
    "declare_push_constant_variable",
    "is_push_constant_block",
    "push_constant_layout",
    "uniform_block_storage_layout",
]

PUSH_CONSTANT_STORAGE_CLASS = "PushConstant"
PUSH_CONSTANT_LAYOUTS = ("std430", "std140", "scalar")


def push_constant_layout(node) -> str:
    """The GLSL block layout of a push-constant block: the one it declares, else ``std430``."""
    from .glsl_block_layout import declared_buffer_layout

    layout = declared_buffer_layout(getattr(node, "attributes", None))
    if layout is None:
        return "std430"
    if layout not in PUSH_CONSTANT_LAYOUTS:
        from .SPIRV_codegen import UnsupportedSPIRVFeatureError

        raise UnsupportedSPIRVFeatureError(
            "push-constant buffer layout",
            f"SPIR-V push-constant block '{node.name}' declares the {layout} layout; SPIR-V lays push constants "
            "out with explicit std430, std140 or scalar offsets",
            missing_capabilities=(f"spirv.push_constant.{layout}_layout",),
        )
    return layout


def is_push_constant_block(codegen, node) -> bool:
    """Whether a CrossGL cbuffer or ``uniform T name`` declaration carries the ``@push_constant`` attribute."""
    return codegen.has_attribute(node, "push_constant")


def uniform_block_storage_layout(codegen, node, row_major_members=()) -> str:
    """The layout of a ``Uniform`` block: ``std140``, or the ``std430`` or ``scalar`` layout it declares.

    A GLSL ``layout(std430) uniform`` or ``layout(scalar) uniform`` block (GL_EXT_scalar_block_layout) keeps its
    offsets; Vulkan accepts them with the ``uniformBufferStandardLayout`` or ``scalarBlockLayout`` feature. Row-major
    members of such a block are refused, as in storage buffers.
    """
    from .glsl_block_layout import (
        SCALAR_BLOCK_LAYOUT,
        STD430_LAYOUT,
        declared_buffer_layout,
    )

    layout = declared_buffer_layout(getattr(node, "attributes", None))
    if layout not in (STD430_LAYOUT, SCALAR_BLOCK_LAYOUT):
        return "std140"
    if row_major_members:
        from .SPIRV_codegen import UnsupportedSPIRVFeatureError

        raise UnsupportedSPIRVFeatureError(
            "row-major uniform-block member",
            f"{layout} uniform block {node.name} has row-major matrix members",
            missing_capabilities=("spirv.row_major_layout",),
        )
    return layout


def _claim_push_constant_slot(codegen, name: str) -> None:
    from .SPIRV_codegen import UnsupportedSPIRVFeatureError

    existing = getattr(codegen, "push_constant_block_name", None)
    if existing is not None:
        raise UnsupportedSPIRVFeatureError(
            "multiple push-constant blocks",
            f"SPIR-V push-constant block '{name}' follows '{existing}': a stage statically uses at "
            "most one PushConstant variable (VUID-StandaloneSpirv-OpEntryPoint-06674), and GLSL and HLSL allow "
            "one push-constant block per stage; merge the members into one block",
            missing_capabilities=("spirv.push_constant.multiple_blocks",),
        )
    codegen.push_constant_block_name = name


def declare_push_constant_block(codegen, cbuffer_node, members):
    """Declare ``cbuffer_node`` as one ``Block`` variable in the ``PushConstant`` storage class.

    ``members`` holds ``(logical_type, name)`` pairs; each member is stored through its layout clone (see
    :func:`push_constant_layout`) so the logical types used by function and workgroup variables keep no explicit
    layout decorations. Returns the variable and the laid-out members.
    """
    layout = push_constant_layout(cbuffer_node)
    _claim_push_constant_slot(codegen, cbuffer_node.name)
    laid_out = [
        (codegen.storage_layout_type(member_type, layout), member_name)
        for member_type, member_name in members
    ]
    block_type = codegen.register_struct_type(cbuffer_node.name, laid_out)
    codegen.decorate_storage_buffer_block_type(
        block_type, layout, block_decoration="Block"
    )
    variable = codegen.create_variable(
        block_type, PUSH_CONSTANT_STORAGE_CLASS, cbuffer_node.name
    )
    return variable, laid_out


def declare_push_constant_variable(codegen, node, logical_type):
    """Declare ``uniform T name @push_constant`` (HLSL ``[[vk::push_constant]] ConstantBuffer<T>``).

    The block type is a clone of ``T`` of its own, so the ``Block`` decoration never lands on a ``T`` that a
    storage buffer also holds as an array element. Returns the variable, or ``None`` when ``T`` is not a struct.
    """
    members = codegen.current_struct_members.get(logical_type.type.base_type)
    if members is None:
        return None
    _claim_push_constant_slot(codegen, node.name)
    laid_out = [
        (codegen.storage_layout_type(member_type, "std430"), member_name)
        for member_type, member_name in members
    ]
    block_type = codegen.register_layout_struct_type(
        logical_type, "push_constant", laid_out
    )
    codegen.decorate_storage_buffer_block_type(
        block_type, "std430", block_decoration="Block"
    )
    variable = codegen.create_variable(
        block_type, PUSH_CONSTANT_STORAGE_CLASS, node.name
    )
    codegen.register_global_variable_name(
        node.name, variable, PUSH_CONSTANT_STORAGE_CLASS
    )
    return variable

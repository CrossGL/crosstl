"""Push-constant blocks keep the std430 layout CrossGL gives them when the target's constant buffers use other rules.

A CrossGL push-constant block, ``cbuffer Name @push_constant { ... }`` or ``ConstantBuffer<T> name @push_constant``
(also spelled ``uniform T name @push_constant``), is laid out by the std430 rules unless it names another layout
with ``@buffer_layout`` (Vulkan 1.3 section 15.6.4, GL_KHR_vulkan_glsl; :mod:`..codegen.push_constant_blocks`). The
HLSL generator laid an unmarked block out by the HLSL constant-buffer packing rules instead (D3D11.3 functional
specification, section 7.3), which put ``vec3 dir`` after ``float scale`` at byte 4 where the host wrote it at byte
16, and declared ``uniform T name`` as a ``$Globals`` value rather than a constant buffer.

:func:`mark_push_constant_layouts` writes the std430 layout on every unmarked push-constant block for those targets,
and declares ``uniform T name @push_constant`` as the ``ConstantBuffer<T> name`` it stands for. For GLSL, which has
push constants only when compiled for Vulkan, the ``cbuffer`` form is marked as well, so the OpenGL uniform block that
stands for it reads the same std430 offsets as the Vulkan block; without the mark it took std140 offsets, and a member
``@ offset`` std140 could not reach was refused. An array of
push-constant blocks (a stage has one) and a push-constant value that is not a struct are refused with
:class:`..codegen.push_constant_blocks.PushConstantBlockError`. A ``ConstantBuffer<T>``
then becomes a constant block holding one ``T`` (:mod:`.constant_buffer_layouts`), and the generator places every
member at its std430 offset (:mod:`..codegen.hlsl_push_constants`).
"""

from __future__ import annotations

from .. import ast as A
from ..codegen.glsl_block_layout import declared_buffer_layout
from ..codegen.push_constant_blocks import PushConstantBlockError, is_push_constant
from .constant_buffers import constant_buffer_element
from .structured_layouts import HLSL_TARGETS

__all__ = ["BLOCK_TARGETS", "TARGETS", "mark_push_constant_layouts"]

# Targets whose constant buffers are not laid out by the std430 rules. GLSL output declares a push-constant block's
# Vulkan form itself (:mod:`..codegen.push_constant_blocks`); the marked layout keeps its OpenGL uniform block at the
# same offsets rather than at its std140 ones.
TARGETS = HLSL_TARGETS
BLOCK_TARGETS = TARGETS | {"opengl", "glsl"}
DEFAULT_LAYOUT = "std430"


def _stages(program):
    stages = getattr(program, "stages", None) or {}
    return list(stages.values() if isinstance(stages, dict) else stages)


def _struct_names(program):
    names = {
        struct.name
        for struct in getattr(program, "structs", None) or []
        if isinstance(getattr(struct, "name", None), str)
    }
    for stage in _stages(program):
        names.update(
            struct.name
            for struct in getattr(stage, "local_structs", None) or []
            if isinstance(getattr(struct, "name", None), str)
        )
    return names


def _mark(node) -> bool:
    if declared_buffer_layout(getattr(node, "attributes", None)) is not None:
        return False
    if not isinstance(getattr(node, "attributes", None), list):
        node.attributes = list(getattr(node, "attributes", None) or [])
    node.attributes.append(
        A.AttributeNode("buffer_layout", [A.IdentifierNode(DEFAULT_LAYOUT)])
    )
    return True


def _as_constant_buffer(variable, struct_names) -> None:
    """Declare ``uniform T name`` with a struct ``T`` as ``ConstantBuffer<T> name``; refuse any other push constant
    that is not a ``ConstantBuffer<T>``."""
    var_type = getattr(variable, "var_type", None)
    if constant_buffer_element(var_type) is not None:
        return
    if isinstance(var_type, A.ArrayType):
        raise PushConstantBlockError(
            "directx",
            variable.name,
            "it is an array; a stage has one push-constant block",
        )
    if (
        not isinstance(var_type, A.NamedType)
        or var_type.generic_args
        or var_type.name not in struct_names
        or set(getattr(variable, "qualifiers", None) or []) - {"uniform"}
    ):
        raise PushConstantBlockError(
            "directx",
            variable.name,
            "only a constant block, a ConstantBuffer<T> or a uniform struct is pushed",
        )
    block_type = A.NamedType("ConstantBuffer", [var_type])
    variable.var_type = block_type
    variable.vtype = block_type
    variable.qualifiers = [
        qualifier for qualifier in variable.qualifiers if qualifier != "uniform"
    ]


def _blocks(program):
    yield from getattr(program, "cbuffers", None) or []
    for stage in _stages(program):
        yield from getattr(stage, "local_cbuffers", None) or []


def _variables(program):
    yield from getattr(program, "global_variables", None) or []
    for stage in _stages(program):
        yield from getattr(stage, "local_variables", None) or []


def mark_push_constant_layouts(program, target) -> int:
    """Give each unmarked push-constant block of ``program`` the std430 layout for a target whose constant buffers
    use other rules; returns how many blocks were marked."""
    target = str(target or "").lower()
    if target not in BLOCK_TARGETS or not isinstance(program, A.ShaderNode):
        return 0
    count = 0
    for block in _blocks(program):
        if is_push_constant(block):
            count += _mark(block)
    if target not in TARGETS:
        return count
    struct_names = _struct_names(program)
    for variable in _variables(program):
        if not isinstance(variable, A.VariableNode) or not is_push_constant(variable):
            continue
        _as_constant_buffer(variable, struct_names)
        if constant_buffer_element(getattr(variable, "var_type", None)) is not None:
            count += _mark(variable)
    return count

"""The rust-gpu shader interface of a CrossGL program.

rust-gpu (https://github.com/Rust-GPU/rust-gpu, 0.11) compiles a ``#![no_std]`` crate that links ``spirv_std``. A
shader takes every resource and built-in input as a parameter of its entry point, marked with a ``#[spirv(...)]``
attribute, and has no mutable statics. This module classifies the CrossGL declarations that become such parameters:

- a storage buffer pointer (``buffer float* data;``) is a storage-buffer slice, ``&mut [f32]`` (``&[f32]`` when it is
  ``readonly``);
- a uniform value (``uniform float scale;``) is a reference into a uniform block, ``&f32``, which the entry point
  copies into a local of the same name, since nothing may write it;
- workgroup memory (``shared float tile[64];``) is ``&mut [f32; 64]`` in the Workgroup storage class;
- a compute built-in input (``gl_GlobalInvocationID``) is a parameter with that built-in's attribute.

Helpers receive each one they use, directly or through the helpers they call, as an ordinary trailing parameter
(:func:`.rust_resource_bindings.resource_parameters`): a slice or workgroup array by mutable reference, a uniform
value and a built-in input by value.
"""

from __future__ import annotations

from ..ast import (
    ArrayType,
    IdentifierNode,
    PointerType,
    PrimitiveType,
    VariableNode,
    VectorType,
)
from .ast_walk import walk_ast_values

# The crate attributes and imports every generated crate starts with. ``Float`` gives the scalar float types the
# methods (``x.sqrt()``) that rust-gpu lowers to GLSL.std.450 instructions; glam supplies the vector and matrix
# types.
RUST_GPU_PRELUDE = (
    "#![no_std]\n"
    "use spirv_std::glam::*;\n"
    "use spirv_std::num_traits::Float;\n"
    "use spirv_std::spirv;\n\n"
)

# Compute built-in inputs: the CrossGL names (GLSL and HLSL spellings) with the rust-gpu built-in attribute and the
# CrossGL type of the value (spirv_std's built-in table, rustc_codegen_spirv/src/symbols.rs, BUILTINS).
COMPUTE_BUILTIN_INPUTS = {
    "gl_GlobalInvocationID": ("global_invocation_id", "uvec3"),
    "gl_LocalInvocationID": ("local_invocation_id", "uvec3"),
    "gl_WorkGroupID": ("workgroup_id", "uvec3"),
    "gl_NumWorkGroups": ("num_workgroups", "uvec3"),
    "gl_WorkGroupSize": None,
    "gl_LocalInvocationIndex": ("local_invocation_index", "uint"),
    "gl_SubgroupInvocationID": ("subgroup_local_invocation_id", "uint"),
    "gl_SubgroupSize": ("subgroup_size", "uint"),
    "gl_SubgroupID": ("subgroup_id", "uint"),
    "gl_NumSubgroups": ("num_subgroups", "uint"),
    "SV_DispatchThreadID": ("global_invocation_id", "uvec3"),
    "SV_GroupThreadID": ("local_invocation_id", "uvec3"),
    "SV_GroupID": ("workgroup_id", "uvec3"),
    "SV_GroupIndex": ("local_invocation_index", "uint"),
}

# The rust-gpu attribute of each built-in input stage-parameter semantic, as
# :meth:`RustTypeDefinitionsMixin.rust_stage_parameter_semantic_key` normalizes it.
COMPUTE_SEMANTIC_BUILTINS = {
    "gl_globalinvocationid": ("global_invocation_id", "uvec3"),
    "gl_localinvocationid": ("local_invocation_id", "uvec3"),
    "gl_workgroupid": ("workgroup_id", "uvec3"),
    "gl_numworkgroups": ("num_workgroups", "uvec3"),
    "gl_localinvocationindex": ("local_invocation_index", "uint"),
    "SV_DISPATCHTHREADID": ("global_invocation_id", "uvec3"),
    "SV_GROUPTHREADID": ("local_invocation_id", "uvec3"),
    "SV_GROUPID": ("workgroup_id", "uvec3"),
    "SV_GROUPINDEX": ("local_invocation_index", "uint"),
}

STORAGE_SLICE_KIND = "storage_slice"
UNIFORM_VALUE_KIND = "uniform_value"
WORKGROUP_KIND = "workgroup"
BUILTIN_INPUT_KIND = "builtin_input"
INTERFACE_KINDS = frozenset(
    {STORAGE_SLICE_KIND, UNIFORM_VALUE_KIND, WORKGROUP_KIND, BUILTIN_INPUT_KIND}
)

_RESOURCE_TYPE_PREFIXES = (
    "sampler",
    "texture",
    "image",
    "Texture",
    "RWTexture",
    "Sampler",
    "Image",
    "isampler",
    "usampler",
    "iimage",
    "uimage",
    "subpassInput",
    "accelerationStructure",
    "RaytracingAccelerationStructure",
)


def _qualifiers(node) -> set[str]:
    return {str(q).lower() for q in getattr(node, "qualifiers", None) or []}


def storage_slice_parts(node):
    """``(pointee type, writable)`` for a storage buffer pointer declaration, else ``None``."""
    if not isinstance(node, VariableNode):
        return None
    var_type = getattr(node, "var_type", None)
    if not isinstance(var_type, PointerType):
        return None
    qualifiers = _qualifiers(node)
    if "buffer" not in qualifiers:
        return None
    return var_type.pointee_type, "readonly" not in qualifiers


def is_uniform_value(node, type_name) -> bool:
    """Whether ``node`` is a plain uniform value: a scalar, vector, matrix, array or struct, not a resource."""
    if not isinstance(node, VariableNode) or "uniform" not in _qualifiers(node):
        return False
    if getattr(node, "initial_value", getattr(node, "value", None)) is not None:
        return False
    if isinstance(getattr(node, "var_type", None), PointerType):
        return False
    return not str(type_name).startswith(_RESOURCE_TYPE_PREFIXES)


def is_workgroup_array(node) -> bool:
    """Whether ``node`` is workgroup memory with a fixed size: a ``shared`` scalar, vector, struct or array."""
    if not isinstance(node, VariableNode):
        return False
    if not _qualifiers(node) & {"shared", "workgroup", "groupshared"}:
        return False
    return getattr(node, "initial_value", getattr(node, "value", None)) is None


def builtin_input_infos(ast):
    """A resource record, keyed by name, for every compute built-in input ``ast`` refers to by name.

    The records have the shape :func:`.rust_resource_bindings.resource_parameters` threads: a synthetic declaration
    and the CrossGL type, plus the rust-gpu built-in attribute. ``SV_DispatchThreadID`` and
    ``gl_GlobalInvocationID`` are one built-in; each spelling a program uses is its own parameter.
    """
    names = sorted(
        {
            value.name
            for value in walk_ast_values(ast)
            if isinstance(value, IdentifierNode)
            and COMPUTE_BUILTIN_INPUTS.get(value.name)
        }
    )
    infos = {}
    for name in names:
        builtin, type_name = COMPUTE_BUILTIN_INPUTS[name]
        node = VariableNode(name, _crossgl_type(type_name))
        infos[name] = {
            "node": node,
            "type_name": type_name,
            "kind": BUILTIN_INPUT_KIND,
            "builtin": builtin,
        }
    return infos


def _crossgl_type(type_name):
    if type_name == "uvec3":
        return VectorType(PrimitiveType("uint"), 3)
    return PrimitiveType(type_name)


def slice_type_name(element_type_name: str, writable: bool) -> str:
    """The CrossGL-side spelling of a storage slice the Rust type mapping renders as ``&mut [T]`` or ``&[T]``."""
    return f"{'&mut ' if writable else '&'}{element_type_name}[]"


def workgroup_type_name(type_name: str) -> str:
    return f"&mut {type_name}"


def is_sized_array_type(var_type) -> bool:
    return isinstance(var_type, ArrayType)

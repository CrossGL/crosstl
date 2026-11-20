"""Exact name tables for the Rust generator, with a refusal for every name outside them.

The generator looked names up as ``mapping.get(name, name)``, which writes a name the table does not know into the
output as it stands. Each lookup is now exact:

- Operators resolve through :mod:`.operator_spellings`; an operator CrossGL does not define is refused. Rust has no
  unary ``+`` (The Rust Reference, "Negation operators": only ``-`` and ``!``), so the generator writes the operand.
- Ray-tracing operations have one runtime function each, with the argument counts the runtime accepts; any other
  count is refused instead of being passed on with the wrong arity.
- A call the program does not declare, and that the generator neither renames nor lowers, must name a function of
  the Rust runtime modules (``gpu`` and ``math``). A built-in of another language written as it stands calls a
  function nothing defines, so it is refused.
- A semantic ``semantic_map`` does not rename is a user varying only if it lies outside the reserved ``gl_`` and
  ``SV_`` namespaces; a built-in with no Rust name is refused.
- A generic type whose template is a built-in type template of a source language (``vector<float, 3>``,
  ``RWStructuredBuffer<T>``) is written only through ``type_mapping``; user and Rust templates are spelled the same.
- A resource is classified by its Rust runtime spelling, never by a prefix of a user type's name.

Every refusal goes through the generation's refusal sink, so the program is reported with
:class:`~.rust_refusals.RustLoweringRefused` and never returned.
"""

from __future__ import annotations

from ..semantics.builtin_function_names import (
    BUILTIN_FUNCTION_NAMES,
    TYPE_TEMPLATE_NAMES,
)
from ..semantics.builtin_type_names import BUILTIN_TYPE_NAMES
from .operator_spellings import map_c_family_operator

__all__ = [
    "RUST_PRIMITIVE_SOURCE_TYPES",
    "RUST_RAY_STAGE_METADATA_NAMES",
    "RUST_RAY_TRACING_INTRINSICS",
    "RUST_RESOURCE_KINDS",
    "call_names_resolving_to",
    "rust_foreign_call_reason",
    "rust_generic_template_reason",
    "rust_operator",
    "rust_resource_kind_of_spelling",
    "rust_unmapped_semantic_reason",
]

RUST_TARGET_LABEL = "Rust"

# CrossGL ray-tracing operation -> (runtime function, accepted argument counts). ``trace_ray`` takes the
# acceleration structure, ray flags, instance mask, hit-group offset, geometry multiplier, miss index, a ``RayDesc``
# and the payload; a call with the eleven DXR/GLSL operands (origin, Tmin, direction and Tmax spelled out) is packed
# into a ``RayDesc``. ``report_hit`` takes the hit distance, hit kind and attributes; the attributes may be omitted.
RUST_RAY_TRACING_INTRINSICS = {
    "TraceRay": ("trace_ray", (8, 11)),
    "CallShader": ("call_shader", (2,)),
    "ReportHit": ("report_hit", (2, 3)),
    "IgnoreHit": ("ignore_hit", (0,)),
    "AcceptHitAndEndSearch": ("accept_hit_and_end_search", (0,)),
}

# CrossGL ray-tracing stage -> the stage name the Rust output records in its stage metadata comment.
RUST_RAY_STAGE_METADATA_NAMES = {
    "ray_any_hit": "any_hit",
    "ray_callable": "callable",
    "ray_closest_hit": "closest_hit",
    "ray_generation": "ray_generation",
    "ray_intersection": "intersection",
    "ray_miss": "miss",
}

# Rust primitive scalar -> the CrossGL type of the same representation. Every Rust primitive the generator writes is
# listed, so a source type is never read back from a Rust spelling it does not know.
RUST_PRIMITIVE_SOURCE_TYPES = {
    "bool": "bool",
    "i8": "i8",
    "u8": "u8",
    "i16": "short",
    "u16": "ushort",
    "i32": "int",
    "u32": "uint",
    "i64": "long",
    "u64": "ulong",
    "f16": "half",
    "f32": "float",
    "f64": "double",
}

_IMAGE_SHAPES = ("1D", "1DArray", "2D", "2DArray", "2DMS", "2DMSArray", "3D", "Cube")
_TEXTURE_SHAPES = _IMAGE_SHAPES + ("CubeArray",)
_DEPTH_TEXTURE_SHAPES = ("2D", "2DArray", "Cube", "CubeArray")

# Rust runtime resource type (the base of its spelling, without generic arguments) -> resource kind. A user type is
# classified by its own declaration, never by a prefix of its name, so a structure named ``TextureParams`` stays a
# structure.
RUST_RESOURCE_KINDS = {
    "RayTracingAccelerationStructure": "acceleration_structure",
    "Sampler": "sampler",
    **{f"Image{shape}": "image" for shape in _IMAGE_SHAPES},
    **{f"Texture{shape}": "texture" for shape in _TEXTURE_SHAPES},
    **{f"DepthTexture{shape}": "texture" for shape in _DEPTH_TEXTURE_SHAPES},
}

_RESERVED_SEMANTIC_PREFIXES = ("gl_", "sv_")


def rust_operator(op):
    """``op`` spelled for Rust, refusing an operator CrossGL does not define."""
    return map_c_family_operator(op, RUST_TARGET_LABEL)


def call_names_resolving_to(mapping, targets):
    """The call names ``mapping`` resolves into ``targets``: the names it renames into one, and the targets it keeps."""
    targets = frozenset(targets)
    renamed = {name for name, spelling in mapping.items() if spelling in targets}
    return frozenset(renamed | {target for target in targets if target not in mapping})


def rust_runtime_spelling(generator, name):
    """Whether ``name`` is a function of the Rust runtime modules the generated program imports."""
    if not isinstance(name, str):
        return False
    if name in generator.rust_runtime_function_names():
        return True
    return generator.rust_imported_helper_module(name) is not None


def rust_foreign_call_reason(generator, source_name, rust_name):
    """Why a call to ``source_name``, written as ``rust_name``, has no Rust callee, or ``None`` when it has one.

    Only a built-in function name of a source language is refused: a call to anything else names a constructor, an
    enum variant or a declared function, which the program itself defines.
    """
    if not isinstance(source_name, str) or source_name not in BUILTIN_FUNCTION_NAMES:
        return None
    if source_name in generator.function_map or rust_name != source_name:
        return None
    if generator.rust_function_name_is_shadowed(source_name):
        return None
    if rust_runtime_spelling(generator, rust_name):
        return None
    return (
        f"call to {source_name}: {source_name} is a built-in function of another language and the Rust runtime "
        "defines no function of that name"
    )


def rust_unmapped_semantic_reason(semantic):
    """Why ``semantic``, which the Rust generator does not rename, has no Rust spelling, or ``None``.

    An unrenamed semantic is written as a user varying; a name in the reserved ``gl_`` or ``SV_`` namespace is a
    built-in variable instead, and one the generator does not know has no Rust equivalent.
    """
    if str(semantic).lower().startswith(_RESERVED_SEMANTIC_PREFIXES):
        return f"semantic {semantic}: names a built-in variable with no Rust equivalent"
    return None


def rust_generic_template_reason(template):
    """Why the generic template ``template``, which ``type_mapping`` does not map, cannot be written, or ``None``."""
    if template in BUILTIN_TYPE_NAMES or template in TYPE_TEMPLATE_NAMES:
        return f"type {template}<...>: built-in type template with no Rust spelling"
    return None


def rust_resource_kind_of_spelling(rust_base):
    """The resource kind of the Rust runtime type ``rust_base``, or ``None`` when it is not a resource type."""
    return RUST_RESOURCE_KINDS.get(rust_base)

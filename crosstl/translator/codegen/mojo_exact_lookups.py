"""Exact name tables for the Mojo generator, with a refusal for every name outside them.

The generator looked names up as ``mapping.get(name, name)``, which writes a name the table does not know into the
output as it stands. Each lookup is now exact:

- Operators resolve through :mod:`.operator_spellings`, with Mojo's keyword spellings of the logical operators
  (``and``, ``or``, ``not``: the Mojo manual, "Operators and expressions"); an operator CrossGL does not define is
  refused.
- Ray-tracing operations have one runtime helper each, with the argument counts the helper accepts; any other count
  is refused instead of being passed on with the wrong arity.
- A resource is classified by its Mojo runtime spelling, never by a prefix of a user type's name.
- A generic type whose template is a built-in type template of a source language (``vector<float, 3>``,
  ``matrix<float, 2, 3>``) is never written as Mojo's own parameter list: ``matrix[Float32, 2, 3]`` names nothing
  in Mojo, and HLSL and MSL do not even agree on which of the two numbers counts the rows.

Every refusal goes through the generation's refusal sink, so the program is reported with the generator's typed
``MojoLoweringRefused`` and never returned.
"""

from __future__ import annotations

from ..semantics.builtin_function_names import TYPE_TEMPLATE_NAMES
from ..semantics.builtin_type_names import BUILTIN_TYPE_NAMES, MOJO_TYPE_NAMES
from .operator_spellings import c_family_operator_spelling, refuse_unknown_operator

__all__ = [
    "MOJO_LOGICAL_OPERATORS",
    "MOJO_RAY_TRACING_INTRINSICS",
    "MOJO_RESOURCE_KINDS",
    "mojo_generic_template_reason",
    "mojo_operator",
    "mojo_resource_kind_of_spelling",
]

MOJO_TARGET_LABEL = "Mojo"

# C-family logical operator -> Mojo keyword.
MOJO_LOGICAL_OPERATORS = {"&&": "and", "||": "or", "!": "not"}

# CrossGL ray-tracing operation -> (runtime helper, accepted argument counts). ``trace_ray`` takes the acceleration
# structure, ray flags, instance mask, hit-group offset, geometry multiplier, miss index, a ``RayDesc`` and the
# payload; a call with the eleven DXR/GLSL operands is packed into a ``RayDesc``. ``report_hit`` takes the hit
# distance, hit kind and attributes; the attributes may be omitted.
MOJO_RAY_TRACING_INTRINSICS = {
    "TraceRay": ("trace_ray", (8, 11)),
    "CallShader": ("call_shader", (2,)),
    "ReportHit": ("report_hit", (2, 3)),
    "IgnoreHit": ("ignore_hit", (0,)),
    "AcceptHitAndEndSearch": ("accept_hit_and_end_search", (0,)),
}

_IMAGE_SHAPES = ("1D", "1DArray", "2D", "2DArray", "2DMS", "2DMSArray", "3D")
_TEXTURE_SHAPES = _IMAGE_SHAPES + ("Cube", "CubeArray")
# Depth-comparison textures exist for the shapes GLSL has shadow samplers for (GLSL 4.60 section 4.1.7).
_SHADOW_TEXTURE_SHAPES = ("1D", "1DArray", "2D", "2DArray", "Cube", "CubeArray")

# Mojo runtime resource type -> resource kind: every resource spelling the generator writes. Integer and unsigned
# images (``IImage2D``, ``UImage3D``) and integer, unsigned and depth-comparison textures (``Texture2DInt``,
# ``TextureCubeShadow``) are the same kind as float ones. A user type is classified by its own declaration, never by
# a prefix of its name.
MOJO_RESOURCE_KINDS = {
    "RayTracingAccelerationStructure": "acceleration_structure",
    "Sampler": "sampler",
    "ImageCube": "image",
    **{
        f"{prefix}Image{shape}": "image"
        for prefix in ("", "I", "U")
        for shape in _IMAGE_SHAPES
    },
    **{
        f"Texture{shape}{suffix}": "texture"
        for shape in _TEXTURE_SHAPES
        for suffix in ("", "Int", "UInt")
    },
    **{f"Texture{shape}Shadow": "texture" for shape in _SHADOW_TEXTURE_SHAPES},
}


def mojo_operator(op):
    """``op`` spelled for Mojo, refusing an operator CrossGL does not define.

    ``None`` and ``""`` stand for a node without an operator and are answered as they are, as
    :func:`.operator_spellings.map_c_family_operator` does.
    """
    if op is None or op == "":
        return op
    spelling = c_family_operator_spelling(op)
    if spelling is None:
        return refuse_unknown_operator(op, MOJO_TARGET_LABEL)
    if spelling in MOJO_LOGICAL_OPERATORS:
        return MOJO_LOGICAL_OPERATORS[spelling]
    return spelling


def mojo_generic_template_reason(template):
    """Why the generic template ``template``, which ``type_mapping`` does not map, cannot be written, or ``None``.

    Mojo's own parameterised types (``SIMD``, ``Scalar``, ``UnsafePointer``) and the program's own generic types are
    written as they are spelled; a built-in template of another language names nothing in Mojo.
    """
    if template in MOJO_TYPE_NAMES:
        return None
    if template in BUILTIN_TYPE_NAMES or template in TYPE_TEMPLATE_NAMES:
        return f"type {template}<...>: built-in type template with no Mojo spelling"
    return None


def mojo_resource_kind_of_spelling(mojo_type):
    """The resource kind of the Mojo runtime type ``mojo_type``, or ``None`` when it is not a resource type."""
    return MOJO_RESOURCE_KINDS.get(mojo_type)

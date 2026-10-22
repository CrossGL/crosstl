"""SPIR-V texel shapes of storage-image reads and writes, and scalar swizzles.

CrossGL ``imageLoad`` yields four components whatever the format, filled as
``(r, 0, 0, 1)`` for formats with fewer channels (see ``storage_image_texels``).
``OpImageRead`` does the same: Vulkan requires its result to have four
components (VUID-StandaloneSpirv-Result-04780) and fills the missing ones the
same way ("Conversion to RGBA").

GLSL ``imageStore`` takes four components and ignores the channels the format
lacks, while the texel of an ``OpImageWrite`` only needs as many components as
the format has (VUID-RuntimeSpirv-OpImageWrite-07112). A wider stored value of
the texel's component type keeps its leading components.

GLSL also swizzles scalars (``value.x``, ``value.rr``; GLSL 4.60 section
5.5), which sources such as ``imageLoad(img, p).x`` rely on.

Formats other than ``SHADER_IMAGE_FORMATS`` need the StorageImageExtendedFormats
capability.

A storage image whose format the runtime chooses (HLSL typed UAVs) has
``Unknown`` format, but an image atomic needs an ``R32i``/``R32ui``/``R32f``
format (VUID-StandaloneSpirv-OpImageTexelPointer-04658), the only formats the
source's atomics accept. Such an image that atomics target takes that format.
"""

import re

from .ast_walk import walk_ast_values
from .storage_image_texels import TEXEL_COMPONENTS, load_context_allowed

__all__ = [
    "SHADER_IMAGE_FORMATS",
    "atomic_storage_image_format",
    "emit_four_component_image_read",
    "image_load_in_context",
    "narrow_image_texel",
]

_IMAGE_ATOMICS = frozenset(
    {
        "imageAtomicAdd",
        "imageAtomicMin",
        "imageAtomicMax",
        "imageAtomicAnd",
        "imageAtomicOr",
        "imageAtomicXor",
        "imageAtomicExchange",
        "imageAtomicCompSwap",
    }
)
_ATOMIC_FORMATS = {"iimage": "r32i", "uimage": "r32ui", "image": "r32f"}

# Image Format operands that need only the Shader capability (SPIR-V 3.11).
# The others need StorageImageExtendedFormats.
SHADER_IMAGE_FORMATS = frozenset(
    {
        "Unknown",
        "Rgba32f",
        "Rgba16f",
        "R32f",
        "Rgba8",
        "Rgba8Snorm",
        "Rgba32i",
        "Rgba16i",
        "Rgba8i",
        "R32i",
        "Rgba32ui",
        "Rgba16ui",
        "Rgba8ui",
        "R32ui",
    }
)


def _component_shape(codegen, type_id):
    """``(component type name, component count)`` of a scalar or vector type."""
    base = type_id.type.base_type
    vector_info = codegen.vector_component_type_and_count(base)
    if vector_info is not None:
        return vector_info[0], vector_info[1]
    return codegen.normalize_primitive_name(base), 1


def emit_four_component_image_read(
    codegen, result_type, image_id, coord_id, operands, format_channels=None
):
    """Emit ``OpImageRead`` of the four-component texel of ``result_type``'s components.

    ``format_channels`` is recorded for :func:`image_load_in_context`.
    """
    component_name, count = _component_shape(codegen, result_type)
    read_type = result_type
    if count != 4:
        read_type = codegen.register_vector_type(
            codegen.register_primitive_type(component_name), 4
        )
    id_value = codegen.get_id()
    codegen.emit(
        f"%{id_value} = OpImageRead %{read_type.id} "
        f"%{image_id.id} %{coord_id.id}{operands}"
    )
    codegen.value_types[id_value] = read_type
    if not hasattr(codegen, "image_load_format_channels"):
        codegen.image_load_format_channels = {}
    codegen.image_load_format_channels[id_value] = format_channels
    return type(coord_id)(id_value, read_type.type)


def _context_component_count(codegen, type_name):
    """Components of the scalar or vector type ``type_name``, else ``None``."""
    if not isinstance(type_name, str):
        return None
    vector_info = codegen.vector_component_type_and_count(type_name)
    if vector_info is not None:
        return vector_info[1]
    scalar = codegen.normalize_primitive_name(type_name)
    if scalar in {"float", "double", "half", "int", "uint", "bool"}:
        return 1
    return None


def image_load_in_context(codegen, expression, value_id, context_type_name):
    """The leading components of an ``imageLoad`` the context is narrower than.

    CrossGL lets a scalar context, or one as wide as the format, take the
    leading components of the four-component texel (``storage_image_texels``).
    A narrower context the format does not allow is refused.
    """
    if value_id is None or codegen.function_call_name(expression) != "imageLoad":
        return value_id
    value_type = codegen.value_types.get(value_id.id)
    if value_type is None:
        return value_id
    component_name, count = _component_shape(codegen, value_type)
    context_count = _context_component_count(codegen, context_type_name)
    if count != TEXEL_COMPONENTS or context_count in (None, TEXEL_COMPONENTS):
        return value_id
    format_channels = getattr(codegen, "image_load_format_channels", {}).get(
        value_id.id
    )
    if not load_context_allowed(format_channels, context_count):
        codegen.refuse_lowering(
            f"imageLoad of a {format_channels}-channel image in a "
            f"{context_count}-component context"
        )
        return value_id
    component_type = codegen.register_primitive_type(component_name)
    if context_count == 1:
        return codegen.composite_extract(value_id, component_type, 0)
    narrow_type = codegen.register_vector_type(component_type, context_count)
    return codegen.vector_shuffle(value_id, narrow_type, list(range(context_count)))


_FILL_CONSTANTS = {"float": (0.0, 1.0), "int": (0, 1), "uint": (0, 1)}


def _filled_image_texel(codegen, value_id, value_count, texel_name, texel_count):
    """A two- or three-component stored value widened with the ``(0, 0, 1)`` fill."""
    fill_values = _FILL_CONSTANTS.get(texel_name)
    if fill_values is None:
        return value_id
    component_type = codegen.register_primitive_type(texel_name)
    zero, one = fill_values
    fill = [zero, zero, one][value_count - 1 : texel_count - 1]
    components = [
        codegen.composite_extract(value_id, component_type, index)
        for index in range(value_count)
    ]
    components.extend(
        codegen.register_constant(value, component_type) for value in fill
    )
    texel_type = codegen.register_vector_type(component_type, texel_count)
    return codegen.composite_construct(texel_type, components)


def narrow_image_texel(codegen, value_id, texel_type):
    """``value_id`` shaped as ``texel_type``: leading components or the texel fill.

    A wider stored value keeps its leading components; a two- or
    three-component value narrower than the texel takes the fill
    (``storage_image_texels``). Scalars are left to the scalar splat.
    """
    value_type = codegen.value_types.get(value_id.id)
    if value_type is None:
        return value_id
    value_name, value_count = _component_shape(codegen, value_type)
    texel_name, texel_count = _component_shape(codegen, texel_type)
    if value_name != texel_name:
        return value_id
    if 1 < value_count < texel_count:
        return _filled_image_texel(
            codegen, value_id, value_count, texel_name, texel_count
        )
    if value_count <= texel_count:
        return value_id
    if texel_count == 1:
        return codegen.composite_extract(value_id, texel_type, 0)
    return codegen.vector_shuffle(value_id, texel_type, list(range(texel_count)))


def _atomic_image_names(codegen):
    ast = getattr(codegen, "spirv_program_ast", None)
    cached = getattr(codegen, "_spirv_atomic_image_names", None)
    if cached is not None and cached[0] is ast:
        return cached[1]
    names = set()
    for node in walk_ast_values(ast):
        if type(node).__name__ != "FunctionCallNode" or not getattr(node, "args", None):
            continue
        callee = (
            node.name
            if isinstance(node.name, str)
            else getattr(node.name, "name", None)
        )
        target = node.args[0]
        while type(target).__name__ == "ArrayAccessNode":
            target = target.array
        target_name = (
            target if isinstance(target, str) else getattr(target, "name", None)
        )
        if callee in _IMAGE_ATOMICS and isinstance(target_name, str):
            names.add(target_name)
    codegen._spirv_atomic_image_names = (ast, names)
    return names


def atomic_storage_image_format(codegen, node, type_name):
    """``r32i``/``r32ui``/``r32f`` for an unformatted storage image that image atomics target, else ``None``."""
    name = getattr(node, "name", None)
    if not isinstance(name, str) or name not in _atomic_image_names(codegen):
        return None
    match = re.match(r"(iimage|uimage|image)", str(type_name))
    return _ATOMIC_FORMATS[match.group(1)] if match else None

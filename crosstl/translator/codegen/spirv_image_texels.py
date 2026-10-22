"""SPIR-V texel shapes of storage-image reads and writes, and scalar swizzles.

CrossGL ``imageLoad`` on an image with a one- or two-channel format yields that
many components, but Vulkan requires the result of ``OpImageRead`` to have four
components (VUID-StandaloneSpirv-Result-04780). The read is emitted with four
components and narrowed afterwards.

GLSL ``imageStore`` takes four components and ignores the channels the format
lacks, while the texel of an ``OpImageWrite`` only needs as many components as
the format has (VUID-RuntimeSpirv-OpImageWrite-07112). A wider stored value of
the texel's component type keeps its leading components.

GLSL also swizzles scalars (``value.x``, ``value.rr``; GLSL 4.60 section
5.5), which sources such as ``imageLoad(img, p).x`` rely on.

A storage image whose format the runtime chooses (HLSL typed UAVs) has
``Unknown`` format, but an image atomic needs an ``R32i``/``R32ui``/``R32f``
format (VUID-StandaloneSpirv-OpImageTexelPointer-04658), the only formats the
source's atomics accept. Such an image that atomics target takes that format.

An ``OpTypeImage`` format operand outside the core set needs a capability: the
SPIR-V 1.6 "Image Format" table gives ``StorageImageExtendedFormats`` for the
one- and two-channel and packed formats.
"""

import re

from .ast_walk import walk_ast_values

__all__ = [
    "atomic_storage_image_format",
    "emit_four_component_image_read",
    "image_format_capability",
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

# SPIR-V 1.6, section 3.11 "Image Format": formats outside the Shader set.
_EXTENDED_IMAGE_FORMATS = frozenset(
    {
        "Rg32f",
        "Rg16f",
        "R11fG11fB10f",
        "R16f",
        "Rgba16",
        "Rgb10A2",
        "Rg16",
        "Rg8",
        "R16",
        "R8",
        "Rgba16Snorm",
        "Rg16Snorm",
        "Rg8Snorm",
        "R16Snorm",
        "R8Snorm",
        "Rg32i",
        "Rg16i",
        "Rg8i",
        "R16i",
        "R8i",
        "Rgb10a2ui",
        "Rg32ui",
        "Rg16ui",
        "Rg8ui",
        "R16ui",
        "R8ui",
    }
)


def image_format_capability(image_format):
    """The capability an ``OpTypeImage`` format operand requires, or ``None`` for core formats."""
    if image_format in _EXTENDED_IMAGE_FORMATS:
        return "StorageImageExtendedFormats"
    return None


def _component_shape(codegen, type_id):
    """``(component type name, component count)`` of a scalar or vector type."""
    base = type_id.type.base_type
    vector_info = codegen.vector_component_type_and_count(base)
    if vector_info is not None:
        return vector_info[0], vector_info[1]
    return codegen.normalize_primitive_name(base), 1


def emit_four_component_image_read(codegen, result_type, image_id, coord_id, operands):
    """Emit ``OpImageRead`` with a four-component result narrowed to ``result_type``."""
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
    texel = type(coord_id)(id_value, read_type.type)
    if count == 4:
        return texel
    if count == 1:
        return codegen.composite_extract(texel, result_type, 0)
    return codegen.vector_shuffle(texel, result_type, list(range(count)))


def narrow_image_texel(codegen, value_id, texel_type):
    """Leading components of a stored value wider than ``texel_type``, else ``value_id``."""
    value_type = codegen.value_types.get(value_id.id)
    if value_type is None:
        return value_id
    value_name, value_count = _component_shape(codegen, value_type)
    texel_name, texel_count = _component_shape(codegen, texel_type)
    if value_name != texel_name or value_count <= texel_count:
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

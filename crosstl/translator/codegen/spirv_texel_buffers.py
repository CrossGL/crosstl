"""SPIR-V image types of CrossGL texel buffers.

A texel buffer is an ``OpTypeImage`` with ``Dim`` ``Buffer``. ``samplerBuffer``
(HLSL ``Buffer<T>``, MSL ``texture_buffer<T, access::read>``) has ``Sampled`` 1
and binds a ``UNIFORM_TEXEL_BUFFER`` descriptor; it takes no sampler, so it is
declared as the image itself rather than as an ``OpTypeSampledImage`` (SPIR-V 1.6
forbids sampled images of ``Dim`` ``Buffer``). ``imageBuffer`` has ``Sampled`` 2
and binds a ``STORAGE_TEXEL_BUFFER`` descriptor. They need the ``SampledBuffer``
and ``ImageBuffer`` capabilities (SPIR-V section 3.31, Vulkan "Texel Buffer"
descriptor types).
"""

import re
from typing import Dict, Optional

__all__ = ["emit_texel_buffer_fetch", "texel_buffer_type_info"]

_TEXEL_BUFFER = re.compile(r"([iu]?)(samplerBuffer|textureBuffer|imageBuffer)")
_COMPONENT_TYPES = {"": "float", "i": "int", "u": "uint"}


def texel_buffer_type_info(type_name: str) -> Optional[Dict[str, object]]:
    """Image metadata of a texel-buffer type name, or ``None`` for other types."""
    match = _TEXEL_BUFFER.fullmatch(str(type_name))
    if match is None:
        return None
    prefix, base = match.groups()
    storage = base == "imageBuffer"
    return {
        "kind": "storage_image" if storage else "texture",
        "component_type": _COMPONENT_TYPES[prefix],
        "dim": "Buffer",
        "depth": 0,
        "arrayed": 0,
        "multisampled": 0,
        "sampled": 2 if storage else 1,
        "format": "Unknown",
    }


def emit_texel_buffer_fetch(codegen, function_name, args):
    """``OpImageFetch`` for ``texelFetch(samplerBuffer, int)``, or ``None`` for other fetches.

    A texel-buffer fetch takes only the coordinate: it has no level of detail, sample or offset operand.
    """
    if function_name not in {"texelFetch", "texelFetchOffset"} or not args:
        return None
    metadata = codegen.resource_metadata_for_value(args[0])
    if not metadata or metadata.get("dim") != "Buffer":
        return None
    result_type = codegen.resource_access_result_type(metadata)
    if metadata.get("kind") != "texture" or function_name != "texelFetch":
        codegen.refuse_lowering(
            f"{function_name} is not an operation on storage texel buffers"
        )
        return codegen.default_value_for_type(result_type)
    if len(args) != 2:
        codegen.refuse_lowering(
            f"texelFetch on a texel buffer takes a buffer and a coordinate, got {len(args)} operands"
        )
        return codegen.default_value_for_type(result_type)
    coord_id = args[1]
    if not codegen.validate_sampled_texture_coordinate(
        function_name, metadata, coord_id, integer=True
    ):
        return codegen.default_value_for_type(result_type)
    image_id = codegen.image_operand_for_query(args[0], metadata)
    if image_id is None:
        return codegen.default_value_for_type(result_type)
    id_value = codegen.get_id()
    codegen.emit(
        f"%{id_value} = OpImageFetch %{result_type.id} %{image_id.id} %{coord_id.id}"
    )
    codegen.value_types[id_value] = result_type
    # The SPIR-V value class lives in the generator module, which imports this one.
    return type(coord_id)(id_value, result_type.type)

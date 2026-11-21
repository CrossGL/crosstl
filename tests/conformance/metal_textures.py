"""Textures and samplers for the Metal compute kernels the harness runs on an Apple GPU.

A Metal kernel takes a texture as an argument with ``[[texture(n)]]`` and a sampler with ``[[sampler(n)]]`` (Metal
Shading Language Specification, "Attributes to Locate Buffers, Textures, and Samplers"); ``array<T, N>`` of either
takes the N slots from ``n``. :func:`declaration` reads one such argument into the record the job carries, and
:func:`bind` gives the encoder a resource for every record.

Every texture holds texels that are a fixed function of its slot, layer and texel index, so two kernels read the same
values only where they bind the same textures to the same slots. Every sampler has the same state: normalized
coordinates, nearest filtering, repeat addressing and no mipmap filter. A native probe may sample several textures
through one sampler where its CrossGL translation binds one sampler per combined sampler, as for each element of a
``sampler2D tex[2]``, and the host gives both the same state. That state differs from an inline
``sampler(mag_filter::linear, min_filter::linear)``, whose linear filter and clamp-to-edge addressing ignore the
host's sampler, so a translation that drops a bound sampler computes something else.

A texture a kernel may write starts with the same texels and is not read back: the probes that write one store to a
single texel from every thread, so its final value depends on the order the threads ran in.
"""

from __future__ import annotations

import re
import struct

from .interface import InterfaceChanged

# Texel extents: 1D textures and texture buffers are 64 texels wide, so that a texel buffer read at any index a
# float input converts to stays in bounds; the others are 4 texels on a side with 2 array layers.
_WIDE = 64
_SIDE = 4
_LAYERS = 2
_SHAPES = {
    # type: (MTLTextureType name, width, height, depth, array length, slices per layer)
    "texture1d": ("MTLTextureType1D", _WIDE, 1, 1, 1, 1),
    "texture1d_array": ("MTLTextureType1DArray", _WIDE, 1, 1, _LAYERS, 1),
    "texture2d": ("MTLTextureType2D", _SIDE, _SIDE, 1, 1, 1),
    "texture2d_array": ("MTLTextureType2DArray", _SIDE, _SIDE, 1, _LAYERS, 1),
    "texture3d": ("MTLTextureType3D", _SIDE, _SIDE, _SIDE, 1, 1),
    "texturecube": ("MTLTextureTypeCube", _SIDE, _SIDE, 1, 1, 6),
    "texturecube_array": ("MTLTextureTypeCubeArray", _SIDE, _SIDE, 1, _LAYERS, 6),
    "texture_buffer": ("MTLTextureTypeTextureBuffer", _WIDE, 1, 1, 1, 1),
}
# Component type: (MTLPixelFormat name of a four-component format, struct code, bytes per component).
_FORMATS = {
    "float": ("MTLPixelFormatRGBA32Float", "f", 4),
    "half": ("MTLPixelFormatRGBA16Float", "e", 2),
    "int": ("MTLPixelFormatRGBA32Sint", "i", 4),
    "uint": ("MTLPixelFormatRGBA32Uint", "I", 4),
    "short": ("MTLPixelFormatRGBA16Sint", "h", 2),
    "ushort": ("MTLPixelFormatRGBA16Uint", "H", 2),
}
_ACCESS = frozenset({"sample", "read", "write", "read_write"})
_ATTRIBUTE = re.compile(r"\[\[\s*(texture|sampler)\s*\(\s*(\d+)\s*\)\s*\]\]")
_ARRAY = re.compile(r"^(?:const\s+)?array\s*<\s*(.+)\s*,\s*(\d+)\s*>\s*\w+\s*$")
_TEXTURE = re.compile(
    r"^(\w+)\s*<\s*(\w+)\s*(?:,\s*access\s*::\s*(\w+)\s*)?>\s*(?:\w+\s*)?$"
)


def declaration(parameter: str) -> dict:
    """The record of the kernel argument ``parameter``, which carries a ``[[texture(n)]]`` or ``[[sampler(n)]]``.

    Raises :class:`InterfaceChanged` for a texture the harness cannot create: depth and multisampled textures, and
    component types other than the ones in ``_FORMATS``.
    """
    found = _ATTRIBUTE.search(parameter)
    if found is None:
        raise InterfaceChanged(f"No texture or sampler slot in {parameter!r}")
    kind, slot = found.group(1), int(found.group(2))
    declared = " ".join(parameter[: found.start()].split())
    count = 1
    array = _ARRAY.match(declared)
    if array:
        declared, count = array.group(1).strip() + " x", int(array.group(2))
    declared = declared.rsplit(" ", 1)[0] if " " in declared else declared
    if kind == "sampler":
        if declared != "sampler":
            raise InterfaceChanged(f"Unexpected Metal sampler type {declared!r}")
        return {"kind": "sampler", "slot": slot, "count": count}
    texture = _TEXTURE.match(declared)
    if texture is None or texture.group(1) not in _SHAPES:
        raise InterfaceChanged(f"Unexpected Metal texture type {declared!r}")
    element, access = texture.group(2), texture.group(3) or "sample"
    if element not in _FORMATS or access not in _ACCESS:
        raise InterfaceChanged(f"Unexpected Metal texture type {declared!r}")
    return {
        "kind": "texture",
        "slot": slot,
        "count": count,
        "type": texture.group(1),
        "element": element,
        "access": access,
    }


def resources(records: list[dict]) -> dict:
    """The job entries for the texture and sampler ``records`` of a kernel; none when it declares neither.

    Raises :class:`InterfaceChanged` when two records claim one slot.
    """
    if not records:
        return {}
    for kind in ("texture", "sampler"):
        slots = [
            record["slot"] + offset
            for record in records
            if record["kind"] == kind
            for offset in range(record["count"])
        ]
        if len(slots) != len(set(slots)):
            raise InterfaceChanged(f"Two Metal {kind} arguments share a slot: {slots}")
    ordered = sorted(records, key=lambda record: (record["kind"], record["slot"]))
    return {"resources": ordered}


def texels(record: dict, slot: int) -> list[bytes]:
    """The contents of the texture ``record`` binds at ``slot``: one byte string per slice, in slice order."""
    _, width, height, depth, layers, faces = _SHAPES[record["type"]]
    _, code, _ = _FORMATS[record["element"]]
    per_slice = width * height * depth * 4
    slices = []
    for index in range(layers * faces):
        values = []
        for texel in range(per_slice):
            word = (texel * 7 + index * 13 + slot * 29) % 61
            values.append(word * 0.125 + 0.0625 if code in "fe" else word + slot)
        slices.append(struct.pack(f"<{per_slice}{code}", *values))
    return slices


def _texture(Metal, device, record, slot):
    type_name, width, height, depth, layers, _ = _SHAPES[record["type"]]
    format_name, _, component = _FORMATS[record["element"]]
    descriptor = Metal.MTLTextureDescriptor.new()
    descriptor.setTextureType_(getattr(Metal, type_name))
    descriptor.setPixelFormat_(getattr(Metal, format_name))
    descriptor.setWidth_(width)
    descriptor.setHeight_(height)
    descriptor.setDepth_(depth)
    descriptor.setArrayLength_(layers)
    descriptor.setStorageMode_(Metal.MTLStorageModeShared)
    usage = Metal.MTLTextureUsageShaderRead
    if record["access"] in {"write", "read_write"}:
        usage |= Metal.MTLTextureUsageShaderWrite
    descriptor.setUsage_(usage)
    row = width * 4 * component
    if record["type"] == "texture_buffer":
        data = texels(record, slot)[0]
        backing = device.newBufferWithBytes_length_options_(
            data, len(data), Metal.MTLResourceStorageModeShared
        )
        if backing is None:
            raise RuntimeError("Metal texture buffer allocation failed")
        descriptor.setResourceOptions_(Metal.MTLResourceStorageModeShared)
        texture = backing.newTextureWithDescriptor_offset_bytesPerRow_(
            descriptor, 0, row
        )
        if texture is None:
            raise RuntimeError("Metal texture buffer creation failed")
        return texture, backing
    texture = device.newTextureWithDescriptor_(descriptor)
    if texture is None:
        raise RuntimeError(f"Metal {record['type']} allocation failed")
    region = Metal.MTLRegionMake3D(0, 0, 0, width, height, depth)
    for index, data in enumerate(texels(record, slot)):
        texture.replaceRegion_mipmapLevel_slice_withBytes_bytesPerRow_bytesPerImage_(
            region, 0, index, data, row, row * height
        )
    return texture, None


def _sampler(Metal, device):
    descriptor = Metal.MTLSamplerDescriptor.new()
    filtering = Metal.MTLSamplerMinMagFilterNearest
    addressing = Metal.MTLSamplerAddressModeRepeat
    descriptor.setMinFilter_(filtering)
    descriptor.setMagFilter_(filtering)
    descriptor.setSAddressMode_(addressing)
    descriptor.setTAddressMode_(addressing)
    descriptor.setRAddressMode_(addressing)
    descriptor.setNormalizedCoordinates_(True)
    state = device.newSamplerStateWithDescriptor_(descriptor)
    if state is None:
        raise RuntimeError("Metal sampler creation failed")
    return state


def bind(Metal, device, encoder, job) -> list:
    """Set a texture or sampler on ``encoder`` for every slot the job's resources take.

    Returns the objects the encoder refers to, which the caller keeps until the command buffer completes.
    """
    kept = []
    for record in job.get("resources", ()):
        for offset in range(record["count"]):
            slot = record["slot"] + offset
            if record["kind"] == "sampler":
                state = _sampler(Metal, device)
                encoder.setSamplerState_atIndex_(state, slot)
                kept.append(state)
                continue
            texture, backing = _texture(Metal, device, record, slot)
            encoder.setTexture_atIndex_(texture, slot)
            kept += [texture, backing]
    return kept

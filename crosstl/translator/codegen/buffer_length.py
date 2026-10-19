"""Buffer length queries: the hub's ``buffer_dimensions(buffer)`` and GLSL's runtime-array ``length()``.

MSL has no way to ask a buffer for its size: a buffer argument is a pointer (MSL 3.2 section 2.8), so a length
needs a host-filled sidecar that changes the interface, and the Metal emitter refuses the query with
:class:`BufferLengthUnsupportedError` until the artifact's reflection metadata can describe that sidecar.


HLSL has no expression for a buffer's length. ``GetDimensions`` is a method that writes out parameters
(``StructuredBuffer::GetDimensions(out uint count, out uint stride)``, ``Buffer::GetDimensions(out uint count)``
and ``ByteAddressBuffer::GetDimensions(out uint bytes)``). The DirectX emitter printed ``buffer_dimensions(b)`` and
``block.member.length()`` unchanged, so DXC rejected the output. They are rendered here through one helper function
per resource type, which HLSL accepts because resources may be passed by value to functions::

    uint cgl_buffer_count(StructuredBuffer<int> b) { uint count; uint stride; b.GetDimensions(count, stride);
                                                    return count; }

A GLSL buffer block lowered to a byte-address buffer keeps the std430 or std140 offset and stride of its trailing
runtime array, whose length is the number of whole elements the bound buffer holds after the offset (GLSL 4.60
section 4.1.9: the length of a runtime-sized array is inferred from the size of the buffer):
``int((cgl_buffer_bytes(b) - offset) / stride)``. A sized array's ``length()`` is its declared size.
"""

from __future__ import annotations

from ..diagnostics import CODE_PREFIX, TranslationDiagnostic

__all__ = [
    "BYTE_ADDRESS_BUFFERS",
    "COUNTED_BUFFERS",
    "BufferLengthUnsupportedError",
    "HlslBufferLength",
]

# Resources whose GetDimensions reports elements: (count, stride) for structured buffers, count for typed buffers.
COUNTED_BUFFERS = {
    "StructuredBuffer": 2,
    "RWStructuredBuffer": 2,
    "RasterizerOrderedStructuredBuffer": 2,
    "Buffer": 1,
    "RWBuffer": 1,
    "RasterizerOrderedBuffer": 1,
}
BYTE_ADDRESS_BUFFERS = frozenset(
    {"ByteAddressBuffer", "RWByteAddressBuffer", "RasterizerOrderedByteAddressBuffer"}
)


class BufferLengthUnsupportedError(TranslationDiagnostic):
    """A buffer length query the target cannot answer."""

    def __init__(self, target, reason):
        self.missing_capabilities = (f"{target}.buffer-length",)
        super().__init__(
            reason, target=target, code=f"{CODE_PREFIX}{target}-buffer-length-unsupported"
        )


class HlslBufferLength:
    """Renders length queries and collects the helper functions they call, one per HLSL resource type."""

    def __init__(self):
        self.helpers = {}

    def element_count(self, buffer_code, resource_name, hlsl_type):
        """``buffer_dimensions(b)`` as a uint element count, or ``None`` when ``b`` reports no element count."""
        outputs = COUNTED_BUFFERS.get(resource_name)
        if outputs is None or not hlsl_type:
            return None
        if outputs == 2:
            body = "uint count; uint stride; b.GetDimensions(count, stride); return count;"
        else:
            body = "uint count; b.GetDimensions(count); return count;"
        self.helpers[("cgl_buffer_count", hlsl_type)] = (
            f"uint cgl_buffer_count({hlsl_type} b) {{ {body} }}\n"
        )
        return f"cgl_buffer_count({buffer_code})"

    def byte_size(self, buffer_code, resource_name):
        if resource_name not in BYTE_ADDRESS_BUFFERS:
            return None
        self.helpers[("cgl_buffer_bytes", resource_name)] = (
            f"uint cgl_buffer_bytes({resource_name} b) "
            "{ uint bytes; b.GetDimensions(bytes); return bytes; }\n"
        )
        return f"cgl_buffer_bytes({buffer_code})"

    def runtime_array_length(self, buffer_code, resource_name, offset, stride):
        """The int ``length()`` of a byte-address buffer's runtime array at ``offset`` with ``stride``."""
        size = self.byte_size(buffer_code, resource_name)
        if size is None or not stride:
            return None
        return f"int(({size} - {offset}u) / {stride}u)"

    def code(self) -> str:
        return "".join(self.helpers[key] for key in sorted(self.helpers))

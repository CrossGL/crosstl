"""CUDA and HIP emission of CrossGL sampled texel buffers.

A CrossGL ``samplerBuffer`` (``isamplerBuffer``, ``usamplerBuffer``) is a one-dimensional array of formatted texels
that is read with ``texelFetch(buffer, index)``. Its CUDA counterpart is a texture object whose resource is linear
memory (``cudaResourceTypeLinear``; HIP ``hipResourceTypeLinear``), and ``tex1Dfetch<T>(texture, index)`` reads one
texel of it by integer index, with no filtering and no coordinate normalization (CUDA C++ Programming Guide,
"Texture Object API"; HIP mirrors it). The texture hardware converts the texel format as ``texelFetch`` does, and
``tex1Dfetch`` returns a four-component vector of the sampler's element type, which is what ``texelFetch`` returns.

GLSL defines only two operations on a texel buffer: ``texelFetch`` and ``textureSize`` (GLSL 4.60, 8.9.1 and
8.9.2). ``textureSize`` has no lod argument and reads the width the host records in the resource's query metadata,
like the other texture size queries of these generators. Any other operation on a texel buffer is refused.
"""

from ..unhandled import UNHANDLED

SAMPLED_TEXEL_BUFFER_VALUE_TYPES = {
    "samplerBuffer": "float4",
    "isamplerBuffer": "int4",
    "usamplerBuffer": "uint4",
}

# The GLSL built-in families that take a texture or image first: a texel buffer passed to one of them other than
# texelFetch or textureSize is refused.
TEXTURE_OPERATION_PREFIXES = (
    "texture",
    "texel",
    "sparseTexture",
    "sparseTexel",
    "image",
)

TEXEL_BUFFER_QUERY_SPEC = {"dimensions": ("width",), "mip": False, "samples": False}


def sampled_texel_buffer_type(type_name):
    """The CrossGL sampled texel-buffer type ``type_name`` names, or ``None``."""
    text = str(type_name or "").strip()
    if "[" in text:
        text = text.split("[", 1)[0].strip()
    return text if text in SAMPLED_TEXEL_BUFFER_VALUE_TYPES else None


class CudaTexelBufferMixin:
    """Texel-buffer calls of the CUDA and HIP generators.

    The generator provides ``resource_base_type``, ``resource_expression_type``, ``refuse_lowering`` and
    ``resource_backend_name``.
    """

    def texel_buffer_resource_type(self, raw_args):
        """The texel-buffer type of the first argument of a resource call, or ``None``."""
        if not raw_args:
            return None
        return sampled_texel_buffer_type(
            self.resource_base_type(self.resource_expression_type(raw_args[0]))
        )

    def texel_buffer_call(self, func_name, raw_args, args):
        """``tex1Dfetch`` for ``texelFetch`` on a texel buffer, a refusal for other texture operations on one.

        ``textureSize`` and the calls whose first argument is not a texel buffer return ``UNHANDLED``; the
        dimension query path reads the width from the query metadata (:data:`TEXEL_BUFFER_QUERY_SPEC`).
        """
        buffer_type = self.texel_buffer_resource_type(raw_args)
        if buffer_type is None or not func_name.startswith(TEXTURE_OPERATION_PREFIXES):
            return UNHANDLED
        if func_name == "textureSize":
            if len(args) != 1:
                return self.refuse_lowering(
                    f"{self.resource_backend_name()} texel buffer call: textureSize on "
                    f"{buffer_type} takes no level-of-detail argument",
                    "0",
                )
            return UNHANDLED
        if func_name != "texelFetch":
            return self.refuse_lowering(
                f"{self.resource_backend_name()} texel buffer call: '{func_name}' is not an "
                f"operation on a {buffer_type}; GLSL reads a texel buffer only with texelFetch",
                self.zero_float4,
            )
        if len(args) != 2:
            return self.refuse_lowering(
                f"{self.resource_backend_name()} texel buffer call: texelFetch on "
                f"{buffer_type} takes the buffer and one integer index",
                self.zero_float4,
            )
        value_type = SAMPLED_TEXEL_BUFFER_VALUE_TYPES[buffer_type]
        return f"tex1Dfetch<{value_type}>({args[0]}, {args[1]})"

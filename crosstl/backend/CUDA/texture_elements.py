"""The element type of a CUDA or HIP texture-object fetch, ``tex2D<T>(tex, x, y)``, in CrossGL.

A texture object (``cudaTextureObject_t``, ``hipTextureObject_t``) has no element type of its own: each fetch names
it as its template argument, ``T`` (CUDA C++ Programming Guide, "Texture Object API"; HIP's
``texture_fetch_functions.h`` declares the same templates). ``T`` is a 1-, 2- or 4-component vector or a scalar of
``float``, ``int``, ``unsigned int``, ``char``, ``unsigned char``, ``short`` or ``unsigned short`` (CUDA has no
3-channel texture formats). A float ``T`` reads a float texture or, under ``cudaReadModeNormalizedFloat``, an
8- or 16-bit integer texture as normalized values; an integer ``T`` reads an integer texture under
``cudaReadModeElementType``.

The importers wrote every texture object as a float sampler and every fetch as the sampler's 4-component result,
whatever ``T`` was: ``tex1Dfetch<float>`` assigned a ``vec4<f32>`` to an ``f32``, which GLSL and Metal reject, and
``tex2D<int4>`` read an integer texture through a float sampler. Now:

- the sampler takes ``T``'s kind, ``isampler*`` for signed and ``usampler*`` for unsigned ``T``;
- the result keeps ``T``'s components (``.x`` for a scalar, ``.xy`` for two) and converts an 8- or 16-bit ``T``
  from the sampler's 32-bit components, which are its values (GLSL 4.60 section 8.9: an integer sampler returns the
  texel's integer value);
- ``tex1Dfetch`` reads a texture bound to linear memory (``cudaResourceTypeLinear``), whose element ``i`` it returns
  without filtering or addressing: a texel buffer, CrossGL's ``samplerBuffer`` read by ``texelFetch(buffer, i)``.

A texture reference, ``texture<T, dim, mode> tex``, names its element in its declaration instead, and its fetches
(``tex1Dfetch(tex, i)``, ``tex2D(tex, x, y)``) take no template argument. A fetch returns ``T`` under
``cudaReadModeElementType`` (``hipReadModeElementType``), the default, and, under ``cudaReadModeNormalizedFloat``, a
float with ``T``'s components for an 8- or 16-bit integer ``T`` (CUDA C++ Programming Guide, "Texture Reference
API"). The reference takes the sampler of that value, and a 1D reference read by ``tex1Dfetch`` is bound to linear
memory, so it is a texel buffer as well.

A ``T`` outside this list (``float3``, ``__half``, a template parameter) keeps the float sampler and the
4-component result, as before.
"""

import re

__all__ = [
    "TEXTURE_SAMPLE_COORDINATES",
    "TextureElement",
    "TextureElementMixin",
    "texture_element",
    "typed_texture_result",
    "typed_texture_sampler",
]

# The sampling fetches with their coordinate vector and its component count (a layer is one more coordinate).
TEXTURE_SAMPLE_COORDINATES = {
    **dict.fromkeys(("tex1D", "tex1DLod", "tex1DGrad"), ("vec1", 1)),
    **dict.fromkeys(("tex2D", "tex2DLod", "tex2DGrad"), ("vec2", 2)),
    **dict.fromkeys(("tex3D", "tex3DLod", "tex3DGrad"), ("vec3", 3)),
    **dict.fromkeys(("texCubemap", "texCubemapLod", "texCubemapGrad"), ("vec3", 3)),
    **dict.fromkeys(
        ("tex1DLayered", "tex1DLayeredLod", "tex1DLayeredGrad"), ("vec2", 2)
    ),
    **dict.fromkeys(
        ("tex2DLayered", "tex2DLayeredLod", "tex2DLayeredGrad"), ("vec3", 3)
    ),
    **dict.fromkeys(
        ("texCubemapLayered", "texCubemapLayeredLod", "texCubemapLayeredGrad"),
        ("vec4", 4),
    ),
}

_KINDS = {
    "f32": ("", False),
    "i32": ("i", False),
    "u32": ("u", False),
    "i8": ("i", True),
    "i16": ("i", True),
    "u8": ("u", True),
    "u16": ("u", True),
}
_VECTOR = re.compile(r"^vec([124])<(\w+)>$")
_SWIZZLES = {1: ".x", 2: ".xy", 4: ""}


class TextureElement:
    """The CrossGL form of a fetch's element type ``T``: the sampler's prefix and the result's conversion."""

    def __init__(self, crossgl_type, prefix, components, narrow):
        self.crossgl_type = crossgl_type
        self.prefix = prefix
        self.components = components
        self.narrow = narrow

    def sampler(self, sampler_type):
        """``isampler2D`` for an ``int`` element of a ``sampler2D``; float elements keep the float sampler."""
        if not self.prefix or not str(sampler_type).startswith("sampler"):
            return sampler_type
        return f"{self.prefix}{sampler_type}"

    def result(self, fetch):
        """The fetch ``fetch`` (a 4-component sampler result) as a value of ``T``."""
        value = f"{fetch}{_SWIZZLES[self.components]}"
        if self.narrow:
            return f"{self.crossgl_type}({value})"
        return value


def texture_element(crossgl_type):
    """The :class:`TextureElement` of a fetch whose ``T`` imports as ``crossgl_type``, or ``None`` for another ``T``."""
    if not isinstance(crossgl_type, str):
        return None
    text = crossgl_type.replace(" ", "")
    match = _VECTOR.match(text)
    components, scalar = (int(match.group(1)), match.group(2)) if match else (1, text)
    kind = _KINDS.get(scalar)
    if kind is None:
        return None
    prefix, narrow = kind
    return TextureElement(text, prefix, components, narrow)


def typed_texture_result(fetch, crossgl_type):
    """The fetch ``fetch`` as a value of the element type that imports as ``crossgl_type``."""
    if fetch is None:
        return None
    element = texture_element(crossgl_type)
    return fetch if element is None else element.result(fetch)


def typed_texture_sampler(fetch_name, sampler_type, crossgl_type):
    """The CrossGL sampler type a texture object read by ``fetch_name`` (``tex1Dfetch``, ``tex2D``) takes."""
    if fetch_name == "tex1Dfetch":
        sampler_type = "samplerBuffer"
    element = texture_element(crossgl_type)
    return sampler_type if element is None else element.sampler(sampler_type)


def reference_fetch_type(crossgl_type, read_mode):
    """The CrossGL type a fetch of ``texture<T, dim, read_mode>`` returns, when ``T`` imports as ``crossgl_type``."""
    element = texture_element(crossgl_type)
    if element is None:
        return None
    if element.prefix and str(read_mode or "").endswith("ReadModeNormalizedFloat"):
        return "f32" if element.components == 1 else f"vec{element.components}<f32>"
    return element.crossgl_type


class TextureElementMixin:
    """Element types of texture fetches, shared by the CUDA and HIP importers.

    The importer provides ``texture_element_crossgl_type(template_args)``, ``parse_cpp_template``,
    ``strip_type_qualifiers``, ``get_resource_object_expression_name`` and ``global_surface_variable_types`` (the
    declared type of each global).
    """

    def texture_reference_fetch_type(self, texture_type):
        """The CrossGL type a fetch of the texture reference type ``texture_type`` returns, or ``None``."""
        base_name, template_args = self.parse_cpp_template(
            self.strip_type_qualifiers(str(texture_type))
        )
        if base_name != "texture" or not template_args:
            return None
        read_mode = template_args[2] if len(template_args) >= 3 else None
        return reference_fetch_type(
            self.texture_element_crossgl_type(template_args[:1]), read_mode
        )

    def texture_reference_sampler(self, texture_type, mapping, texel_buffer=False):
        """The CrossGL sampler of ``texture<T, dim, mode>``, a texel buffer when ``texel_buffer`` and ``dim`` is 1D."""
        _, template_args = self.parse_cpp_template(
            self.strip_type_qualifiers(str(texture_type))
        )
        dimension = template_args[1] if len(template_args) >= 2 else "1"
        sampler_type = mapping.get(dimension)
        if sampler_type is None:
            return None
        fetch_name = (
            "tex1Dfetch" if texel_buffer and sampler_type == "sampler1D" else None
        )
        return typed_texture_sampler(
            fetch_name, sampler_type, self.texture_reference_fetch_type(texture_type)
        )

    def format_typed_texture_result(self, fetch, value_type, raw_args=None):
        """The fetch ``fetch`` as the value it returns: ``T`` of ``tex2D<T>``, or a texture reference's element."""
        if value_type:
            return typed_texture_result(
                fetch, self.texture_element_crossgl_type([value_type])
            )
        if not raw_args:
            return fetch
        name = self.get_resource_object_expression_name(raw_args[0])
        declared = (getattr(self, "global_surface_variable_types", None) or {}).get(
            name
        )
        if declared is None:
            return fetch
        return typed_texture_result(fetch, self.texture_reference_fetch_type(declared))

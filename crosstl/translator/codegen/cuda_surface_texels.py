"""Surface elements of CrossGL storage images in CUDA and HIP output.

A CUDA or HIP surface holds raw bytes. ``surf2Dread<T>`` reads ``sizeof(T)`` bytes at a byte offset and
``surf2Dwrite`` writes the bytes of its value, with no format conversion (CUDA C++ Programming Guide, "Surface
Memory"; HIP mirrors the API). A CrossGL storage image names its texel format (``layout(r32f)``,
``@format(rgba8)``), or, when the bound view decides the format, its texel type (``@texel(float)``), and
``imageLoad`` returns, ``imageStore`` takes, a four-component value that the format converts (GLSL 4.60, 8.12
"Image Functions"; Vulkan, "Texel Input Format Conversion" and "Texel Output Format Conversion").

The surface element of an image is its texel as memory holds it: ``float`` for ``r32f``, ``uchar4`` for ``rgba8``,
and the x byte offset counts elements of that size. A load widens the element to four components, the missing
ones reading as 0 and alpha as 1, and maps 8- and 16-bit unorm and snorm components to floats; a store narrows the
value to the element, converting a normalized component by clamping it and rounding to nearest. A 16-bit float
channel is stored as the bits of a ``__half`` (cuda_fp16.h, hip_fp16.h) and converted with round to nearest even.
An 8- or 16-bit integer channel (``rgba8ui``, ``r16i``) widens to the 32-bit integer of its signedness on a load,
zero- or sign-extended, and a store keeps its low bits: an integer the format cannot represent is stored as an
undefined value (Vulkan, "Texel Output Format Conversion"), so truncation is one of the values allowed. The packed
formats have no conversion here and are refused. An image without a format or texel type keeps the generator's
four-component default.
"""

import re
from dataclasses import dataclass

from ..ast import ArrayAccessNode, FunctionCallNode, IdentifierNode, VariableNode
from ..validation import IMAGE_FORMAT_METADATA_NAMES
from .cuda_vector_conversions import convert_vector_value
from .hlsl_image_formats import CROSSGL_TEXEL_TYPES, crossgl_image_component_kind

COMPONENTS = "xyzw"

# The CUDA scalar type of one stored channel, and the prefix of its vector types (uchar4, short2).
_STORAGE_VECTOR_PREFIXES = {
    "float": "float",
    "int": "int",
    "uint": "uint",
    "unsigned char": "uchar",
    "signed char": "char",
    "unsigned short": "ushort",
    "short": "short",
}

# The four-component CUDA type of a CrossGL texel of each component kind.
TEXEL_VECTOR_TYPES = {"float": "float4", "int": "int4", "uint": "uint4"}

_FILL = {"float": ("0.0f", "1.0f"), "int": ("0", "1"), "uint": ("0u", "1u")}

# The 32-bit CUDA scalar that an 8- or 16-bit integer channel widens to.
_INTEGER_SCALARS = {"int": "int", "uint": "unsigned int"}

_FORMAT_SHAPE = re.compile(r"(rgba|rg|r)(8|16|32)(f|i|ui|_snorm)?")
_CHANNEL_COUNTS = {"r": 1, "rg": 2, "rgba": 4}


@dataclass(frozen=True)
class SurfaceTexel:
    """The surface element of a storage image and how a load or store converts it."""

    name: str
    kind: str
    count: int
    storage: str
    encoding: str = "value"
    bits: int = 32

    @property
    def element(self):
        """The CUDA type a surface access reads or writes: ``float``, ``uchar4``."""
        if self.count == 1:
            return self.storage
        return f"{_STORAGE_VECTOR_PREFIXES[self.storage]}{self.count}"

    @property
    def vector(self):
        """The four-component CUDA type of the CrossGL value: ``float4``, ``int4`` or ``uint4``."""
        return TEXEL_VECTOR_TYPES[self.kind]

    def value_type(self, width):
        """The CUDA type of a CrossGL value of ``width`` components of this texel's kind."""
        return self.kind if width == 1 else f"{self.kind}{width}"

    def passes_through(self, width):
        """Whether a ``width``-component value is the element itself, so no conversion is needed."""
        return width == self.count and self.encoding == "value"

    def load_helper(self, width=4):
        """``(name, text)`` of the device function that widens one element to a ``width``-component value.

        The value is the leading ``width`` components of the four-component texel ``imageLoad`` returns.
        """
        name = f"cgl_{self.name}_texel" + ("" if width == 4 else str(width))
        zero, one = _FILL[self.kind]
        channels = [
            self._load_component("c" if self.count == 1 else f"c.{component}")
            for component in COMPONENTS[: self.count]
        ]
        # Green and blue a format lacks read as 0, alpha as 1.
        channels = (channels + [zero, zero][: max(0, 3 - self.count)] + [one])[:4]
        result_type = self.value_type(width)
        value = (
            channels[0]
            if width == 1
            else f"make_{result_type}({', '.join(channels[:width])})"
        )
        return name, (
            f"__device__ inline {result_type} {name}({self.element} c)\n"
            "{\n"
            f"    return {value};\n"
            "}"
        )

    def store_helper(self, width=4):
        """``(name, text)`` of the device function that narrows a ``width``-component value to one element.

        A scalar is splatted; a value narrower than the element takes the texel fill for the channels it lacks.
        """
        name = f"cgl_{self.name}_element" + ("" if width == 4 else str(width))
        zero, one = _FILL[self.kind]
        fill = [zero, zero, zero, one]
        sources = [
            (
                "v"
                if width == 1
                else f"v.{COMPONENTS[index]}" if index < width else fill[index]
            )
            for index in range(self.count)
        ]
        channels = [self._store_component(source) for source in sources]
        value = (
            channels[0]
            if self.count == 1
            else f"make_{self.element}({', '.join(channels)})"
        )
        return name, (
            f"__device__ inline {self.element} {name}({self.value_type(width)} v)\n"
            "{\n"
            f"    return {value};\n"
            "}"
        )

    def _normalized_maximum(self):
        bits = self.bits - 1 if self.encoding == "snorm" else self.bits
        return f"{(1 << bits) - 1}.0f"

    def _load_component(self, component):
        if self.encoding == "unorm":
            return f"{component} / {self._normalized_maximum()}"
        if self.encoding == "snorm":
            # Both -2^(b-1) and -2^(b-1)+1 read as -1.0 (Vulkan, "Conversion from Normalized Fixed-Point").
            return f"fmaxf({component} / {self._normalized_maximum()}, -1.0f)"
        if self.encoding == "half":
            return f"__half2float(__ushort_as_half({component}))"
        if self.encoding == "integer":
            # Zero- or sign-extended by the signedness of the storage type.
            return f"({_INTEGER_SCALARS[self.kind]}){component}"
        return component

    def _store_component(self, component):
        maximum = self._normalized_maximum()
        if self.encoding == "unorm":
            return (
                f"({self.storage})__float2uint_rn(__saturatef({component}) * {maximum})"
            )
        if self.encoding == "snorm":
            return (
                f"({self.storage})__float2int_rn("
                f"fminf(fmaxf({component}, -1.0f), 1.0f) * {maximum})"
            )
        if self.encoding == "half":
            return f"__half_as_ushort(__float2half_rn({component}))"
        if self.encoding == "integer":
            return f"({self.storage}){component}"
        return component


@dataclass(frozen=True)
class UnsupportedSurfaceTexel:
    """A storage image whose format or texel type has no surface element here."""

    reason: str


def format_surface_texel(image_format):
    """The surface element of a CrossGL image format, or an :class:`UnsupportedSurfaceTexel`."""
    match = _FORMAT_SHAPE.fullmatch(image_format)
    if match is None:
        return UnsupportedSurfaceTexel(
            f"format {image_format} packs its channels into bit fields, and the "
            "generator has no conversion for it"
        )
    channels, bits, suffix = match.group(1), int(match.group(2)), match.group(3)
    count = _CHANNEL_COUNTS[channels]
    if bits == 32 and suffix in {"f", "i", "ui"}:
        kind = {"f": "float", "i": "int", "ui": "uint"}[suffix]
        return SurfaceTexel(image_format, kind, count, kind)
    if bits == 16 and suffix == "f":
        return SurfaceTexel(image_format, "float", count, "unsigned short", "half", 16)
    if bits in (8, 16) and suffix is None:
        storage = "unsigned char" if bits == 8 else "unsigned short"
        return SurfaceTexel(image_format, "float", count, storage, "unorm", bits)
    if bits in (8, 16) and suffix == "_snorm":
        storage = "signed char" if bits == 8 else "short"
        return SurfaceTexel(image_format, "float", count, storage, "snorm", bits)
    if bits in (8, 16) and suffix in {"i", "ui"}:
        signed = suffix == "i"
        storage = {
            (8, True): "signed char",
            (8, False): "unsigned char",
            (16, True): "short",
            (16, False): "unsigned short",
        }[(bits, signed)]
        kind = "int" if signed else "uint"
        return SurfaceTexel(image_format, kind, count, storage, "integer", bits)
    return UnsupportedSurfaceTexel(
        f"format {image_format} stores {bits}-bit channels without a type, and the "
        "generator has no conversion between them and the four-component value"
    )


def texel_type_surface_texel(arguments):
    """The surface element of an ``@texel`` attribute's arguments, or an :class:`UnsupportedSurfaceTexel`."""
    spelled = f"@texel({', '.join(arguments)})"
    if len(arguments) != 1 or arguments[0] not in CROSSGL_TEXEL_TYPES:
        return UnsupportedSurfaceTexel(
            f"{spelled} leaves the size of a texel to the bound view, which a surface "
            "does not have"
        )
    kind, count = CROSSGL_TEXEL_TYPES[arguments[0]]
    if count == 3:
        return UnsupportedSurfaceTexel(
            f"{spelled} has three components, and a surface element is 1, 2, 4, 8 or "
            "16 bytes"
        )
    return SurfaceTexel(f"texel_{arguments[0]}", kind, count, kind)


def declared_surface_texel(node, image_type, attribute_value_to_string):
    """The surface element a storage image declaration gives, an :class:`UnsupportedSurfaceTexel`, or ``None``."""
    image_format = None
    texel_arguments = None
    for attribute in getattr(node, "attributes", None) or []:
        name = str(getattr(attribute, "name", "") or "").strip().lower()
        arguments = [
            str(attribute_value_to_string(argument)).strip()
            for argument in getattr(attribute, "arguments", None) or []
        ]
        if name in IMAGE_FORMAT_METADATA_NAMES:
            image_format = name
        elif name == "format" and arguments and arguments[0].lower() != "unknown":
            image_format = arguments[0].lower()
        elif name == "texel":
            texel_arguments = arguments
    if image_format is not None:
        texel = format_surface_texel(image_format)
    elif texel_arguments is not None:
        texel = texel_type_surface_texel(texel_arguments)
    else:
        return None
    image_kind = crossgl_image_component_kind(image_type)
    if isinstance(texel, SurfaceTexel) and texel.kind != image_kind:
        return UnsupportedSurfaceTexel(
            f"{image_type} holds {image_kind} texels, but its format or texel type "
            f"holds {texel.kind} ones"
        )
    return texel


_CUDA_VALUE_WIDTH = re.compile(
    r"(?:float|int|uint|half|double|bool|char|uchar|short|ushort|long|ulong|longlong|ulonglong)([1-4])?"
)
_CROSSGL_VECTOR_WIDTH = re.compile(
    r"(?:(?:b|i|u|f16|float16|int16|uint16)?vec([2-4])|vec([2-4])<[^>]+>|cgl_half([2-4]))"
)
_SCALAR_VALUE_TYPES = frozenset(
    {
        "unsigned int",
        "unsigned",
        "signed int",
        "unsigned char",
        "signed char",
        "unsigned short",
        "short int",
        "unsigned short int",
        "long int",
        "unsigned long",
        "unsigned long int",
        "long long",
        "long long int",
        "unsigned long long",
        "unsigned long long int",
        "int8_t",
        "uint8_t",
        "int16_t",
        "uint16_t",
        "int32_t",
        "uint32_t",
        "int64_t",
        "uint64_t",
        "__half",
        "i8",
        "u8",
        "i16",
        "u16",
        "i32",
        "u32",
        "i64",
        "u64",
        "f16",
        "f32",
        "f64",
        "int8",
        "uint8",
        "int16",
        "uint16",
        "int32",
        "uint32",
        "int64",
        "uint64",
        "float16",
        "float32",
        "float64",
        "float16_t",
        "min16float",
        "min16int",
        "min16uint",
    }
)


def cuda_value_width(type_name):
    """Components of a scalar or vector value type: 1 for ``uint``/``u32``, 2 for ``float2``/``uvec2``; ``None`` otherwise."""
    text = str(type_name or "").strip()
    if text in _SCALAR_VALUE_TYPES:
        return 1
    match = _CUDA_VALUE_WIDTH.fullmatch(text)
    if match is not None:
        return int(match.group(1) or 1)
    vector_match = _CROSSGL_VECTOR_WIDTH.fullmatch(text)
    if vector_match is not None:
        return int(next(group for group in vector_match.groups() if group is not None))
    return None


def _call_arguments(node):
    return getattr(node, "args", getattr(node, "arguments", None))


class CudaSurfaceTexelMixin:
    """Storage-image surface elements of the CUDA and HIP generators.

    The generator provides ``attribute_value_to_string``, ``is_storage_image_type``, ``resource_base_type``,
    ``helper_functions``, ``refuse_lowering``, ``resource_backend_name``, ``expression_result_type``,
    ``raw_function_call_name`` and the ``surface_texels`` map, which :meth:`register_storage_image_texel` fills from
    each declaration.

    CrossGL lets a load stand in a context narrower than four components and a store take a scalar
    (:mod:`.storage_image_texels`): :meth:`visit_in_texel_context` renders an ``imageLoad`` that initializes or is
    assigned to a scalar or a narrow vector as that many leading components.
    """

    surface_texel_context = None

    def register_storage_image_texel(self, name, type_name, node):
        """Record the surface element of storage image ``name``, or forget the name for any other type."""
        if not self.is_storage_image_type(type_name) or node is None:
            self.surface_texels.pop(name, None)
            return
        texel = declared_surface_texel(
            node,
            self.resource_base_type(type_name),
            self.attribute_value_to_string,
        )
        if texel is None:
            self.surface_texels.pop(name, None)
        else:
            self.surface_texels[name] = texel

    def storage_image_texel(self, image_arg):
        """The surface element of the image ``image_arg`` names, or ``None`` for the default."""
        while isinstance(image_arg, ArrayAccessNode):
            image_arg = getattr(
                image_arg, "array_expr", getattr(image_arg, "array", None)
            )
        if isinstance(image_arg, (IdentifierNode, VariableNode)):
            return self.surface_texels.get(image_arg.name)
        if isinstance(image_arg, str):
            return self.surface_texels.get(image_arg)
        return None

    def surface_texel_refusal(self, func_name, image_name, texel, placeholder):
        """Refuse ``func_name`` on an image whose format has no surface element."""
        return self.refuse_lowering(
            f"{self.resource_backend_name()} {func_name} on storage image "
            f"'{image_name}': {texel.reason}",
            placeholder,
        )

    def visit_in_texel_context(self, node, target_type):
        """Render ``node`` for a value of ``target_type``; an ``imageLoad`` there loads that many components.

        A vector of another component type is converted to ``target_type`` (:mod:`.cuda_vector_conversions`).
        """
        width = cuda_value_width(target_type)
        arguments = (
            _call_arguments(node) if isinstance(node, FunctionCallNode) else None
        )
        if (
            width is None
            or width == 4
            or not arguments
            or self.raw_function_call_name(node) != "imageLoad"
            or not isinstance(self.storage_image_texel(arguments[0]), SurfaceTexel)
        ):
            return convert_vector_value(self, node, target_type, self.visit(node))
        previous = self.surface_texel_context
        self.surface_texel_context = (arguments, width)
        try:
            return self.visit(node)
        finally:
            self.surface_texel_context = previous

    def surface_texel_load_width(self, raw_args):
        """Components the ``imageLoad`` with arguments ``raw_args`` loads: 4 unless its context is narrower."""
        context = self.surface_texel_context
        if context is not None and context[0] is raw_args:
            return context[1]
        return 4

    def surface_texel_load(self, read, texel, width=4):
        """``read``, the element a surface read returns, as the ``width``-component value ``imageLoad`` gives."""
        if texel is None or texel.passes_through(width):
            return read
        name, text = texel.load_helper(width)
        self.helper_functions.setdefault(name, text)
        return f"{name}({read})"

    def surface_texel_store_value(self, raw_value, value, texel):
        """The element ``surf*write`` stores for the value ``imageStore`` takes; a scalar is splatted."""
        if texel is None:
            return value
        width = cuda_value_width(self.expression_result_type(raw_value))
        if width not in {1, 2, 3}:
            width = 4
        if texel.passes_through(width):
            return value
        name, text = texel.store_helper(width)
        self.helper_functions.setdefault(name, text)
        return f"{name}({value})"

    def resource_call_result_type(self, func_name, raw_args):
        """``imageLoad`` of an image with a surface element returns its four-component value."""
        if func_name == "imageLoad" and raw_args:
            texel = self.storage_image_texel(raw_args[0])
            if isinstance(texel, SurfaceTexel):
                width = self.surface_texel_load_width(raw_args)
                return texel.vector if width == 4 else texel.value_type(width)
        return super().resource_call_result_type(func_name, raw_args)

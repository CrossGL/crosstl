"""HLSL emission of CrossGL texel buffers.

CrossGL ``samplerBuffer`` is an HLSL ``Buffer<T>`` and ``imageBuffer`` an
``RWBuffer<T>`` (``RasterizerOrderedBuffer<T>`` when rasterizer ordered), with
the ``i``/``u`` prefix selecting the ``int``/``uint`` element family.

CrossGL texel functions follow GLSL: ``texelFetch`` and ``imageLoad`` return
four components, filled as ``(r, g, 0, 1)`` for formats with fewer channels,
and ``imageStore`` takes four. An HLSL texel buffer instead declares how many
components it stores. A sampled buffer is declared with four components, which
HLSL fills the same way. A storage buffer is declared with the channel count
of its explicit format, or a single component when it is the target of an
image atomic (atomics need a single-channel 32-bit integer format), and four
otherwise. Loads widen to four components and stores keep the declared ones.
"""

import re

from .image_access_contracts import (
    explicit_image_format,
    image_format_channel_count,
    image_format_component_kind,
)

SAMPLED_TEXEL_BUFFER_TYPES = {
    "samplerBuffer": "float",
    "isamplerBuffer": "int",
    "usamplerBuffer": "uint",
}
STORAGE_TEXEL_BUFFER_TYPES = {
    "imageBuffer": "float",
    "iimageBuffer": "int",
    "uimageBuffer": "uint",
}
TEXEL_BUFFER_TYPES = {**SAMPLED_TEXEL_BUFFER_TYPES, **STORAGE_TEXEL_BUFFER_TYPES}

# Default HLSL declarations, used for function parameters and unformatted buffers.
HLSL_TEXEL_BUFFER_TYPES = {
    "samplerBuffer": "Buffer<float4>",
    "isamplerBuffer": "Buffer<int4>",
    "usamplerBuffer": "Buffer<uint4>",
    "imageBuffer": "RWBuffer<float4>",
    "iimageBuffer": "RWBuffer<int4>",
    "uimageBuffer": "RWBuffer<uint4>",
}

IMAGE_ATOMIC_INTRINSICS = {
    "imageAtomicAdd": "InterlockedAdd",
    "imageAtomicMin": "InterlockedMin",
    "imageAtomicMax": "InterlockedMax",
    "imageAtomicAnd": "InterlockedAnd",
    "imageAtomicOr": "InterlockedOr",
    "imageAtomicXor": "InterlockedXor",
    "imageAtomicExchange": "InterlockedExchange",
    "imageAtomicCompSwap": "InterlockedCompareExchange",
}

TEXEL_BUFFER_OPERATIONS = frozenset(
    {"texelFetch", "textureSize", "imageLoad", "imageStore", "imageSize"}
    | set(IMAGE_ATOMIC_INTRINSICS)
)

ZERO = {"float": "0.0", "int": "0", "uint": "0u"}
ONE = {"float": "1.0", "int": "1", "uint": "1u"}


_HLSL_TEXEL_BUFFER = re.compile(
    r"(Buffer|RWBuffer|RasterizerOrderedBuffer)<\s*(float|int|uint)[1-4]?\s*>"
)


def texel_buffer_base_type(type_name):
    """CrossGL texel-buffer type of ``type_name`` without array suffixes.

    Stage parameters already carry their HLSL type, so ``Buffer<T>`` and
    ``RWBuffer<T>`` map back to the CrossGL type of the same family.
    """
    text = str(type_name or "").strip()
    if "[" in text:
        text = text.split("[", 1)[0].strip()
    if text in TEXEL_BUFFER_TYPES:
        return text
    match = _HLSL_TEXEL_BUFFER.fullmatch(text)
    if match is None:
        return None
    prefix = {"float": "", "int": "i", "uint": "u"}[match.group(2)]
    base = "samplerBuffer" if match.group(1) == "Buffer" else "imageBuffer"
    return prefix + base


def explicit_hlsl_texel_buffer_type(type_name):
    """The HLSL ``Buffer<T>``-family spelling of ``type_name``, or ``None``."""
    text = str(type_name or "").strip()
    if "[" in text:
        text = text.split("[", 1)[0].strip()
    match = _HLSL_TEXEL_BUFFER.fullmatch(text)
    if match is None:
        return None
    return re.sub(r"\s+", "", text)


def hlsl_texel_buffer_element(hlsl_type):
    """``(scalar, component count)`` of an HLSL ``Buffer<T>``/``RWBuffer<T>``."""
    text = str(hlsl_type)
    element = text.split("<", 1)[1].rsplit(">", 1)[0].strip()
    if element[-1:].isdigit():
        return element[:-1], int(element[-1])
    return element, 1


class HLSLTexelBuffers:
    """Texel-buffer declarations, operations and helpers for one HLSL program."""

    def __init__(self, codegen):
        self.codegen = codegen
        self.declared_types = {}
        self.atomic_targets = set()
        self.helpers = {}

    def prepare(self, ast):
        """Record the texel buffers that image atomics target."""
        from ..ast import FunctionCallNode

        for node in self.codegen.walk_ast(ast):
            if (
                isinstance(node, FunctionCallNode)
                and self.codegen.expression_name(node.name) in IMAGE_ATOMIC_INTRINSICS
                and node.args
            ):
                name = self.codegen.expression_name(node.args[0])
                if name:
                    self.atomic_targets.add(name)

    def declaration_type(self, var_name, vtype, node, mapped_type):
        """HLSL declaration type of a global texel buffer, or ``mapped_type``."""
        type_text = self.codegen.type_name_string(vtype)
        crossgl_type = texel_buffer_base_type(type_text)
        if crossgl_type is None:
            return mapped_type
        family = TEXEL_BUFFER_TYPES[crossgl_type]
        spelled = explicit_hlsl_texel_buffer_type(type_text)
        if spelled is not None:
            # An HLSL spelling states the element type, as a format would.
            hlsl_type = self.codegen.rasterizer_ordered_resource_type(spelled, node)
        elif crossgl_type in SAMPLED_TEXEL_BUFFER_TYPES:
            hlsl_type = HLSL_TEXEL_BUFFER_TYPES[crossgl_type]
        else:
            hlsl_type = self.storage_declaration_type(
                var_name, crossgl_type, family, node
            )
            hlsl_type = self.codegen.rasterizer_ordered_resource_type(hlsl_type, node)
        self.declared_types[var_name] = hlsl_type
        return hlsl_type

    def storage_declaration_type(self, var_name, crossgl_type, family, node):
        image_format = explicit_image_format(
            node, self.codegen.attribute_value_to_string
        )
        if image_format is not None:
            format_family = image_format_component_kind(image_format)
            if format_family != family:
                self.codegen.refuse_lowering(
                    f"texel buffer '{var_name}': {crossgl_type} holds {family} "
                    f"texels but its format {image_format} holds {format_family}"
                )
            count = image_format_channel_count(image_format)
            return f"RWBuffer<{family}{'' if count == 1 else count}>"
        if var_name in self.atomic_targets:
            return f"RWBuffer<{family}>"
        return HLSL_TEXEL_BUFFER_TYPES[crossgl_type]

    def resource_type(self, arg):
        """``(CrossGL type, HLSL type)`` of a texel-buffer argument, or ``None``."""
        codegen = self.codegen
        name = codegen.expression_name(arg)
        result_type = codegen.type_name_string(codegen.expression_result_type(arg))
        crossgl_type = texel_buffer_base_type(result_type)
        if crossgl_type is None:
            return None
        hlsl_type = None
        if crossgl_type != str(result_type).split("[", 1)[0].strip():
            hlsl_type = str(result_type).split("[", 1)[0].strip()
        elif name and name not in codegen.local_variable_types:
            hlsl_type = self.declared_types.get(name)
        if hlsl_type is None:
            hlsl_type = HLSL_TEXEL_BUFFER_TYPES[crossgl_type]
        return crossgl_type, hlsl_type

    def generate_call(self, func_name, args):
        """HLSL for a texel-buffer operation, or ``None`` for other calls."""
        if func_name not in TEXEL_BUFFER_OPERATIONS or not args:
            return None
        resource = self.resource_type(args[0])
        if resource is None:
            return None
        crossgl_type, hlsl_type = resource
        sampled = crossgl_type in SAMPLED_TEXEL_BUFFER_TYPES
        expected = (
            {"texelFetch": 2, "textureSize": 1}
            if sampled
            else {"imageLoad": 2, "imageStore": 3, "imageSize": 1}
        )
        if func_name in IMAGE_ATOMIC_INTRINSICS and not sampled:
            return self.generate_atomic(func_name, args, crossgl_type, hlsl_type)
        if func_name not in expected:
            return self.refuse(
                f"'{func_name}' is not an operation on a {crossgl_type} texel buffer"
            )
        if len(args) != expected[func_name]:
            return self.refuse(
                f"texel-buffer '{func_name}' takes {expected[func_name]} "
                f"argument(s), got {len(args)}"
            )
        buffer = self.codegen.generate_expression(args[0])
        if func_name in {"textureSize", "imageSize"}:
            return f"{self.size_helper(hlsl_type)}({buffer})"
        coord = self.codegen.generate_expression(args[1])
        if func_name == "texelFetch":
            return f"{buffer}.Load({coord})"
        if func_name == "imageLoad":
            return self.widen(f"{buffer}[{coord}]", hlsl_type)
        value = self.codegen.generate_expression(args[2])
        return f"{buffer}[{coord}] = {self.narrow(value, hlsl_type)}"

    def widen(self, value, hlsl_type):
        scalar, count = hlsl_texel_buffer_element(hlsl_type)
        if count == 4:
            return value
        fill = [ZERO[scalar], ZERO[scalar], ONE[scalar]][count - 1 :]
        return f"{scalar}4({value}, {', '.join(fill)})"

    def narrow(self, value, hlsl_type):
        _, count = hlsl_texel_buffer_element(hlsl_type)
        if count == 4:
            return value
        return f"({value}).{'xyz'[:count]}"

    def generate_atomic(self, func_name, args, crossgl_type, hlsl_type):
        scalar, count = hlsl_texel_buffer_element(hlsl_type)
        value_count = 2 if func_name == "imageAtomicCompSwap" else 1
        if scalar not in {"int", "uint"} or count != 1:
            return self.refuse(
                f"'{func_name}' on {crossgl_type} needs a single-channel 32-bit "
                f"integer texel buffer, which {hlsl_type} is not"
            )
        if len(args) != 2 + value_count:
            return self.refuse(
                f"texel-buffer '{func_name}' takes {2 + value_count} argument(s), "
                f"got {len(args)}"
            )
        rendered = [self.codegen.generate_expression(arg) for arg in args]
        name = f"__crossgl_{func_name}_{self.helper_suffix(hlsl_type)}"
        values = [f"{scalar} value{index}" for index in range(value_count)]
        value_args = ", ".join(f"value{index}" for index in range(value_count))
        self.helpers[name] = (
            f"{scalar} {name}({hlsl_type} buffer, uint coord, {', '.join(values)})\n"
            "{\n"
            f"    {scalar} original;\n"
            f"    {IMAGE_ATOMIC_INTRINSICS[func_name]}(buffer[coord], {value_args}, "
            "original);\n"
            "    return original;\n"
            "}\n"
        )
        return f"{name}({', '.join(rendered)})"

    def size_helper(self, hlsl_type):
        name = f"__crossgl_texel_buffer_size_{self.helper_suffix(hlsl_type)}"
        self.helpers[name] = (
            f"int {name}({hlsl_type} buffer)\n"
            "{\n"
            "    uint width;\n"
            "    buffer.GetDimensions(width);\n"
            "    return int(width);\n"
            "}\n"
        )
        return name

    @staticmethod
    def helper_suffix(hlsl_type):
        return "".join(ch if ch.isalnum() else "_" for ch in hlsl_type).strip("_")

    def refuse(self, reason):
        self.codegen.refuse_lowering(f"DirectX {reason}")
        return "0"

    def code(self):
        return "".join(self.helpers[name] + "\n" for name in sorted(self.helpers))

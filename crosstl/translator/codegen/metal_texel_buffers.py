"""Metal emission of CrossGL texel buffers.

CrossGL ``samplerBuffer`` is a Metal ``texture_buffer<T, access::read>`` and
``imageBuffer`` a ``texture_buffer<T>`` whose access follows its
``@readonly``/``@writeonly`` qualifiers (``read_write`` otherwise), with the
``i``/``u`` prefix selecting the ``int``/``uint`` element type.

``texelFetch`` and ``imageLoad`` read four components and ``imageStore`` writes
four, as Metal's ``read`` and ``write`` do; Metal fills missing channels as
GLSL does. Sizes are ``get_width()``. Image atomics use the texture atomic
members of MSL 3.1, which need a single-channel 32-bit integer pixel format.
"""

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

METAL_TEXEL_BUFFER_TYPES = {
    **{
        name: f"texture_buffer<{scalar}, access::read>"
        for name, scalar in SAMPLED_TEXEL_BUFFER_TYPES.items()
    },
    **{
        name: f"texture_buffer<{scalar}, access::read_write>"
        for name, scalar in STORAGE_TEXEL_BUFFER_TYPES.items()
    },
}

IMAGE_ATOMIC_METHODS = {
    "imageAtomicAdd": "atomic_fetch_add",
    "imageAtomicMin": "atomic_fetch_min",
    "imageAtomicMax": "atomic_fetch_max",
    "imageAtomicAnd": "atomic_fetch_and",
    "imageAtomicOr": "atomic_fetch_or",
    "imageAtomicXor": "atomic_fetch_xor",
    "imageAtomicExchange": "atomic_exchange",
}

TEXEL_BUFFER_OPERATIONS = frozenset(
    {
        "texelFetch",
        "textureSize",
        "imageLoad",
        "imageStore",
        "imageSize",
        "imageAtomicCompSwap",
    }
    | set(IMAGE_ATOMIC_METHODS)
)


def texel_buffer_base_type(type_name):
    """CrossGL texel-buffer type of ``type_name`` without array suffixes."""
    text = str(type_name or "").strip()
    if "[" in text:
        text = text.split("[", 1)[0].strip()
    return text if text in TEXEL_BUFFER_TYPES else None


class MetalTexelBuffers:
    """Texel-buffer operations and helpers for one Metal program."""

    def __init__(self, codegen):
        self.codegen = codegen
        self.helpers = {}

    def metal_type(self, arg, crossgl_type):
        """Declared Metal type of a texel-buffer argument."""
        declared = self.codegen.texture_resource_type(arg)
        if declared is None or "texture_buffer<" not in str(declared):
            declared = METAL_TEXEL_BUFFER_TYPES[crossgl_type]
        return str(declared).split("[", 1)[0].strip()

    def generate_call(self, func_name, args):
        """Metal for a texel-buffer operation, or ``None`` for other calls."""
        if func_name not in TEXEL_BUFFER_OPERATIONS or not args:
            return None
        codegen = self.codegen
        crossgl_type = texel_buffer_base_type(
            codegen.type_name_string(
                codegen.texture_argument_raw_resource_type(args[0])
            )
        )
        if crossgl_type is None:
            return None
        metal_type = self.metal_type(args[0], crossgl_type)
        sampled = crossgl_type in SAMPLED_TEXEL_BUFFER_TYPES
        expected = (
            {"texelFetch": 2, "textureSize": 1}
            if sampled
            else {
                "imageLoad": 2,
                "imageStore": 3,
                "imageSize": 1,
                "imageAtomicCompSwap": 4,
                **{name: 3 for name in IMAGE_ATOMIC_METHODS},
            }
        )
        if func_name not in expected:
            return self.refuse(
                f"'{func_name}' is not an operation on a {crossgl_type} texel buffer"
            )
        if len(args) != expected[func_name]:
            return self.refuse(
                f"texel-buffer '{func_name}' takes {expected[func_name]} "
                f"argument(s), got {len(args)}"
            )
        buffer = codegen.generate_expression(args[0])
        if func_name in {"textureSize", "imageSize"}:
            return f"int({buffer}.get_width())"
        coord = f"uint({codegen.generate_expression(args[1])})"
        access = metal_type.rsplit("access::", 1)[-1].rstrip("> ")
        if func_name in {"texelFetch", "imageLoad"}:
            if access == "write":
                return self.refuse(f"'{func_name}' reads the write-only {metal_type}")
            return f"{buffer}.read({coord})"
        if access == "read":
            return self.refuse(f"'{func_name}' writes the read-only {metal_type}")
        values = [codegen.generate_expression(arg) for arg in args[2:]]
        if func_name == "imageStore":
            return f"{buffer}.write({values[0]}, {coord})"
        return self.generate_atomic(func_name, buffer, coord, values, metal_type)

    def generate_atomic(self, func_name, buffer, coord, values, metal_type):
        scalar = metal_type.split("<", 1)[1].split(",", 1)[0].strip()
        if scalar not in {"int", "uint"} or "access::read_write" not in metal_type:
            return self.refuse(
                f"'{func_name}' needs a read-write int or uint texel buffer, "
                f"got {metal_type}"
            )
        if func_name != "imageAtomicCompSwap":
            method = IMAGE_ATOMIC_METHODS[func_name]
            return f"{buffer}.{method}({coord}, {values[0]}).x"
        suffix = "".join(ch if ch.isalnum() else "_" for ch in metal_type).strip("_")
        name = f"__crossgl_texel_buffer_compswap_{suffix}"
        self.helpers[name] = (
            f"static {scalar} {name}({metal_type} buffer, uint coord, "
            f"{scalar} compareValue, {scalar} value) {{\n"
            f"    {scalar}4 original;\n"
            "    do {\n"
            "        original.x = compareValue;\n"
            "    } while (!buffer.atomic_compare_exchange_weak(coord, &original, "
            "value) && original.x == compareValue);\n"
            "    return original.x;\n"
            "}\n\n"
        )
        return f"{name}({buffer}, {coord}, {values[0]}, {values[1]})"

    def refuse(self, reason):
        self.codegen.refuse_lowering(f"Metal {reason}")
        return "0"

    def code(self):
        return "".join(self.helpers[name] for name in sorted(self.helpers))

"""Lossless transport between MLX bfloat16 words and native buffer carriers."""

from crosstl.project.runtime_value_encoding import BFLOAT16_BITS, FLOAT32_BITS

GUARD = [0x3EAB] * 32


def encoding(target):
    if target not in {"metal", "directx", "opengl"}:
        raise ValueError("Unsupported bfloat storage target")
    return {"metal": BFLOAT16_BITS, "directx": None, "opengl": FLOAT32_BITS}[target]


def pack(values, target):
    encoding(target)
    values = list(values)
    if any(type(word) is not int or not 0 <= word <= 0xFFFF for word in values):
        raise ValueError("Bfloat storage requires unsigned 16-bit words")
    return [word << 16 for word in values] if target == "opengl" else values


def unpack(values, target):
    """Reject inexact carriers rather than rounding a native result on the host."""
    encoding(target)
    values = list(values)
    if target == "opengl":
        if any(
            type(word) is not int or not 0 <= word <= 0xFFFFFFFF or word & 0xFFFF
            for word in values
        ):
            raise ValueError(
                "Native bfloat carrier is not an exact bfloat16 representation"
            )
        return [word >> 16 for word in values]
    return pack(values, target)

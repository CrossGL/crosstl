"""Layer selection for sampled array textures.

CrossGL follows GLSL, Vulkan and Direct3D: a sampled array texture receives the
layer as a floating-point coordinate component, and the sampler rounds it to the
nearest integer before clamping it to ``[0, layers - 1]``.

* GLSL 4.60, section 8.9: the layer used is ``max(0, min(d - 1, floor(layer + 0.5)))``.
* Vulkan 1.3, "Texel Coordinate Systems": ``layer = clamp(RNE(a), 0, d - 1)``.
* Direct3D samples ``Texture*Array`` with the array index rounded to the
  nearest integer.

Metal takes the layer of ``sample``, ``gather`` and ``sample_compare`` as an
integer argument instead. The implicit float-to-``uint`` conversion truncates
toward zero, so a layer coordinate of 0.6 selects layer 0 instead of layer 1.
The emitters round explicitly with ``rint``, which rounds halfway cases to the
nearest even integer (MSL 4.1, section 6.5), the same rule as Vulkan's RNE.

A negative layer must select layer 0, but converting a negative float to
``uint`` is undefined in MSL as in C++, and the compiler folds a constant such as
``uint(rint(-1.0))`` to an out-of-range index that the hardware clamps to the
last layer. The emitters therefore clamp the coordinate to ``[0, 2047]`` before
rounding. 2047 is the largest valid layer (``MTLTexture.arrayLength`` is at most
2048), so the conversion is always defined and the hardware's own clamp to
``d - 1`` finishes the job. The coordinate is converted to ``float`` first so
that ``clamp`` resolves for ``half`` coordinates too; the conversion is exact.

Integer texel fetches (``texelFetch``/``read``) already address a layer with an
integer and must not be rounded.
"""

from __future__ import annotations

__all__ = [
    "METAL_MAX_ARRAY_LAYER",
    "metal_integer_scalar_type",
    "metal_sampled_array_layer",
    "metal_sampled_layer_source_argument",
]


# MTLTexture.arrayLength is at most 2048 on every Metal GPU family.
METAL_MAX_ARRAY_LAYER = 2047


_METAL_INTEGER_SCALARS = frozenset(
    {
        "char",
        "uchar",
        "short",
        "ushort",
        "int",
        "uint",
        "long",
        "ulong",
        "size_t",
        "ptrdiff_t",
        "int8_t",
        "uint8_t",
        "int16_t",
        "uint16_t",
        "int32_t",
        "uint32_t",
        "int64_t",
        "uint64_t",
    }
)


_METAL_TYPE_QUALIFIERS = frozenset(
    {"const", "constant", "device", "thread", "threadgroup", "volatile"}
)


def metal_sampled_array_layer(component: str) -> str:
    """Return the Metal layer argument for a floating-point layer coordinate."""
    return f"uint(rint(clamp(float({component}), 0.0, {METAL_MAX_ARRAY_LAYER:.1f})))"


def metal_integer_scalar_type(metal_type: str | None) -> bool:
    """Return True when ``metal_type`` names a Metal integer scalar type."""
    if not metal_type:
        return False
    base = str(metal_type).strip()
    while base.endswith("&"):
        base = base[:-1].strip()
    words = [word for word in base.split() if word not in _METAL_TYPE_QUALIFIERS]
    if len(words) != 1:
        return False
    base = words[0]
    if base.startswith("metal::"):
        base = base[len("metal::") :]
    return base in _METAL_INTEGER_SCALARS


def metal_sampled_layer_source_argument(layer: str, metal_type: str | None) -> str:
    """Return the CrossGL layer component for a Metal sampled-array layer argument.

    Metal converts the layer argument to ``uint`` before sampling, so a
    floating-point layer expression is truncated. CrossGL rounds the layer
    component of the coordinate, so the import keeps Metal's truncation
    explicit. The conversion is applied whenever the argument type is not a
    known integer type, because ``uint`` of a ``uint`` is the identity.
    """
    if metal_integer_scalar_type(metal_type):
        return layer
    return f"uint({layer})"

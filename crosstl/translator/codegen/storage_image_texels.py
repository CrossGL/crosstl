"""Shapes of CrossGL storage-image texels.

CrossGL storage-image texels follow GLSL (GLSL 4.60 section 8.12): ``imageLoad``
returns a four-component ``gvec4`` and ``imageStore`` takes one, whatever the
image format. A format with fewer channels fills the components it lacks with
``(0, 0, 1)`` on a load and ignores them on a store. Vulkan ("Conversion to
RGBA"), Metal (``read`` returns ``vec<T, 4>``) and Direct3D (typed UAV format
conversion) define the same fill.

CrossGL also accepts shorthands. A load in a scalar context or in a context as
wide as the format takes the leading components, and a four-channel format
also accepts a three-component context. A stored scalar is splatted, as GLSL
and Metal vector constructors and HLSL scalar promotion splat it. A stored
value as wide as the format, or a three-component value for a four-channel
format, takes the fill above.

Backends whose resources hold fewer than four components (HLSL ``RWTexture2D<T>``
with a narrow ``T``) widen what they load with the same fill and keep the
leading components of what they store.
"""

TEXEL_COMPONENTS = 4

__all__ = [
    "TEXEL_COMPONENTS",
    "image_load_swizzle_extent",
    "is_whole_texel_load",
    "leading_components_suffix",
    "load_context_allowed",
    "padded_texel_expression",
    "reshaped_texel_expression",
    "store_value_allowed",
    "texel_store_value",
    "texel_vector_type",
]

_LEADING_SWIZZLES = {1: ".x", 2: ".xy", 3: ".xyz"}
_SWIZZLE_SETS = ("xyzw", "rgba", "stpq")
_FILL_VALUES = {
    "float": ("0.0", "1.0"),
    "half": ("0.0h", "1.0h"),
    "int": ("0", "1"),
    "uint": ("0u", "1u"),
}


def _shorthand_width(format_channels, count):
    if count is None or count in {1, TEXEL_COMPONENTS}:
        return True
    if format_channels is None:
        return False
    if count == format_channels:
        return True
    return format_channels == TEXEL_COMPONENTS and count == 3


def load_context_allowed(format_channels, context_channels):
    """Whether an ``imageLoad`` may appear in a context of ``context_channels``."""
    return _shorthand_width(format_channels, context_channels)


def store_value_allowed(format_channels, value_channels):
    """Whether ``imageStore`` accepts a value of ``value_channels`` components."""
    return _shorthand_width(format_channels, value_channels)


def leading_components_suffix(count):
    """Swizzle selecting the leading ``count`` components of a four-wide texel."""
    if count is None:
        return ""
    return _LEADING_SWIZZLES.get(count, "")


def texel_vector_type(component_kind, vector_spelling):
    """Four-component vector type name for ``component_kind``.

    ``vector_spelling`` maps a component kind to its vector type prefix, such as
    ``{"float": "vec", "int": "ivec", "uint": "uvec"}`` for GLSL or the kind
    itself for HLSL and Metal.
    """
    prefix = vector_spelling.get(component_kind)
    if prefix is None:
        return None
    return f"{prefix}{TEXEL_COMPONENTS}"


def _fill(component_kind, value_channels, target_channels):
    zero, one = _FILL_VALUES.get(component_kind, ("0", "1"))
    # Components 1, 2 and 3 (green, blue, alpha) fill as 0, 0 and 1.
    return [zero, zero, one][value_channels - 1 : target_channels - 1]


def padded_texel_expression(constructor, component_kind, value, value_channels):
    """``value`` of ``value_channels`` components padded to a four-wide texel.

    The missing components are filled as a load from a narrower format fills
    them: green and blue with zero, alpha with one.
    """
    if value_channels is None or value_channels >= TEXEL_COMPONENTS:
        return value
    fill = _fill(component_kind, value_channels, TEXEL_COMPONENTS)
    return f"{constructor}({value}, {', '.join(fill)})"


def texel_store_value(constructor, component_kind, value, value_channels):
    """``value`` spelled as the four-component texel ``imageStore`` writes.

    A scalar is splatted, as a vector constructor and HLSL scalar promotion
    splat it; a two- or three-component value takes the texel fill.
    """
    if value_channels is None or value_channels >= TEXEL_COMPONENTS:
        return value
    if value_channels == 1:
        return f"{constructor}({value})"
    return padded_texel_expression(constructor, component_kind, value, value_channels)


def reshaped_texel_expression(
    vector_prefix, component_kind, value, value_channels, target_channels
):
    """``value`` of ``value_channels`` components as ``target_channels`` components.

    A narrower target takes the leading components; a wider one is built as
    ``{vector_prefix}{target_channels}(value, ...)`` with the texel fill.
    ``value`` must be a postfix expression when it is narrowed.
    """
    if (
        value_channels is None
        or target_channels is None
        or value_channels == target_channels
    ):
        return value
    if target_channels < value_channels:
        return f"{value}{_LEADING_SWIZZLES[target_channels]}"
    fill = _fill(component_kind, value_channels, target_channels)
    return f"{vector_prefix}{target_channels}({value}, {', '.join(fill)})"


def image_load_swizzle_extent(object_expr, member, expression_name):
    """Components a swizzle ``member`` of an ``imageLoad`` call reads, or ``None``.

    ``member`` reads the leading ``n`` components of the texel; ``None`` when
    ``object_expr`` is not an ``imageLoad`` call or ``member`` not a swizzle.
    """
    name = getattr(object_expr, "name", None)
    if name is None or expression_name(name) != "imageLoad":
        return None
    if not isinstance(member, str) or not member or len(member) > TEXEL_COMPONENTS:
        return None
    for components in _SWIZZLE_SETS:
        if all(letter in components for letter in member):
            return max(components.index(letter) for letter in member) + 1
    return None


WHOLE_TEXEL_LOADS = frozenset({"imageLoad", "texelFetch"})


def is_whole_texel_load(expression, call_name):
    """Whether ``expression`` loads a texel whose components a subscript selects.

    ``imageLoad(img, p)[i]`` picks component ``i`` of the four-component texel,
    so the subscripted element's context must not narrow the load.
    """
    from ..ast import FunctionCallNode

    if not isinstance(expression, FunctionCallNode):
        return False
    return call_name(expression) in WHOLE_TEXEL_LOADS

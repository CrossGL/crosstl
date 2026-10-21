"""GLSL keywords an emitted identifier must not spell.

GLSL 4.60 section 3.6 reserves every built-in type name as a keyword, so a block or variable named ``imageBuffer``,
``vec3`` or ``texture2D`` does not compile, though each is an ordinary identifier in HLSL and MSL (and an
emitter that names a storage block ``<name>Buffer`` makes ``imageBuffer`` from a buffer called ``image``).
"""

from __future__ import annotations

__all__ = ["GLSL_TYPE_KEYWORDS"]


def _type_keywords():
    names = {"void", "bool", "int", "uint", "float", "double", "atomic_uint"}
    names |= {"true", "false", "struct", "precision", "subroutine", "discard"}
    for width in (2, 3, 4):
        names |= {f"{prefix}vec{width}" for prefix in ("", "d", "b", "i", "u")}
        names |= {f"{prefix}mat{width}" for prefix in ("", "d")}
        names |= {
            f"{prefix}mat{width}x{rows}" for prefix in ("", "d") for rows in (2, 3, 4)
        }
    shapes = (
        "1D",
        "2D",
        "3D",
        "Cube",
        "2DRect",
        "1DArray",
        "2DArray",
        "CubeArray",
        "Buffer",
        "2DMS",
        "2DMSArray",
    )
    for kind in ("sampler", "image", "texture"):
        names |= {
            f"{prefix}{kind}{shape}" for prefix in ("", "i", "u") for shape in shapes
        }
    shadows = ("1D", "2D", "Cube", "2DRect", "1DArray", "2DArray", "CubeArray")
    names |= {f"sampler{shape}Shadow" for shape in shadows}
    names |= {"sampler", "samplerShadow", "samplerExternalOES"}
    names |= {
        f"{prefix}subpassInput{ms}" for prefix in ("", "i", "u") for ms in ("", "MS")
    }
    return frozenset(names)


GLSL_TYPE_KEYWORDS = _type_keywords()

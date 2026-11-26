"""Subscripts of textures in Metal output.

CrossGL reads a texel with ``imageLoad`` or ``texelFetch`` and writes one with ``imageStore``. MSL's texture types
have no subscript operator (Metal rejects ``t[p]`` with "type 'texture2d<float, access::read_write>' does not provide
a subscript operator"), so the generator refuses a subscript of a texture instead of writing it as it stands. A
subscript of an array of textures selects one texture and is written normally.
"""

from __future__ import annotations

from ..ast import IdentifierNode

__all__ = ["metal_texture_subscript_refusal"]


def metal_texture_subscript_refusal(codegen, expr):
    """Refuse ``texture[p]`` on a texture or storage image, which MSL reads and writes only through methods.

    CrossGL reads a texel with ``imageLoad``/``texelFetch`` and writes one with ``imageStore``; MSL's texture types
    have no subscript operator ("type 'texture2d<float, access::read_write>' does not provide a subscript
    operator"), so a subscript written as it stands does not compile. Returns the refusal's placeholder, or ``None``
    when ``expr`` does not subscript a texture.
    """
    base = getattr(expr, "array", None)
    textures = {
        getattr(variable, "name", None): array_size
        for variable, _index, _type, array_size in (
            getattr(codegen, "texture_variables", ()) or ()
        )
    }
    if isinstance(base, IdentifierNode):
        name = base.name
        if name not in textures or textures[name] is not None:
            return None
    else:
        # ``textures[i][p]`` subscripts an element of an array of textures.
        inner = getattr(base, "array", None)
        if (
            not isinstance(inner, IdentifierNode)
            or textures.get(inner.name, None) is None
        ):
            return None
        name = inner.name
    if name in (getattr(codegen, "local_variable_types", None) or {}):
        return None
    return codegen.refuse_lowering(
        f"subscript of texture '{name}': Metal textures have no subscript operator; CrossGL reads a texel with "
        "imageLoad or texelFetch and writes one with imageStore",
        "0",
    )

"""Result types of MSL texture member functions.

MSL 3.2 section 6.12 fixes the value a texture member function returns from the texture's type alone:

- a colour texture ``textureNd<T, access>`` (``texture1d`` ... ``texturecube_array``, ``texture2d_ms``,
  ``texture2d_ms_array`` and ``texture_buffer``) returns ``vec<T, 4>`` from ``sample``, ``read`` and ``gather``;
- a depth texture ``depthNd<T, access>`` returns the scalar ``T`` from ``sample``, ``sample_compare`` and
  ``read``, and ``vec<T, 4>`` from ``gather`` and ``gather_compare``;
- the size queries (``get_width``, ``get_height``, ``get_depth``, ``get_array_size``, ``get_num_mip_levels``,
  ``get_num_samples``) return ``uint``, and ``calculate_clamped_lod``/``calculate_unclamped_lod`` return
  ``float``.

The importer needs these types to resolve overloads such as ``smoothstep(a, b, tex.sample(s, uv).x)``. Methods
outside this table, and textures whose element type is not spelled, have no result type here, so callers keep
their own typed refusal.
"""

from .texture_types import METAL_DEPTH_TEXTURE_TYPES, METAL_SAMPLED_TEXTURE_TYPES

__all__ = ["is_depth_texture_name", "texture_method_result_shape"]

_SIZE_QUERY_METHODS = frozenset(
    {
        "get_width",
        "get_height",
        "get_depth",
        "get_array_size",
        "get_num_mip_levels",
        "get_num_samples",
    }
)
_LOD_QUERY_METHODS = frozenset({"calculate_clamped_lod", "calculate_unclamped_lod"})
_COLOUR_VECTOR_METHODS = frozenset({"sample", "read", "gather"})
_DEPTH_SCALAR_METHODS = frozenset({"sample", "sample_compare", "read"})
_DEPTH_VECTOR_METHODS = frozenset({"gather", "gather_compare"})


def texture_method_result_shape(base_name, generic_args, method):
    """Return ``(element_type, width)`` for ``texture.method(...)``, or ``None`` when MSL does not fix it.

    ``base_name`` is the unqualified texture template name (``texture2d``, ``depth2d_array``) and
    ``generic_args`` its template arguments as spelled. ``width`` is 1 for a scalar result and 4 for
    ``vec<T, 4>``.
    """
    method = str(method)
    if method in _SIZE_QUERY_METHODS:
        if (
            base_name in METAL_SAMPLED_TEXTURE_TYPES
            or base_name in METAL_DEPTH_TEXTURE_TYPES
        ):
            return "uint", 1
        return None
    if method in _LOD_QUERY_METHODS:
        if (
            base_name in METAL_SAMPLED_TEXTURE_TYPES
            or base_name in METAL_DEPTH_TEXTURE_TYPES
        ):
            return "float", 1
        return None
    if not generic_args:
        return None
    element = _element_type(generic_args[0])
    if not element:
        return None
    if base_name in METAL_SAMPLED_TEXTURE_TYPES:
        return (element, 4) if method in _COLOUR_VECTOR_METHODS else None
    if base_name in METAL_DEPTH_TEXTURE_TYPES:
        if method in _DEPTH_SCALAR_METHODS:
            return element, 1
        if method in _DEPTH_VECTOR_METHODS:
            return element, 4
    return None


def is_depth_texture_name(base_name):
    """Whether ``base_name`` names an MSL depth texture template (``depth2d``, ``depthcube_array``, ...)."""
    return base_name in METAL_DEPTH_TEXTURE_TYPES


def _element_type(argument):
    element = str(argument).strip()
    while element.startswith("metal::"):
        element = element[len("metal::") :].strip()
    return element

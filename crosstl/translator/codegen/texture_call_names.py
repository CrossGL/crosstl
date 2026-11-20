"""The call names a generator's sampled-texture path handles, so it can pass over other calls cheaply.

The Metal generator tries every call with two or more arguments as a texture operation before it generates the
call itself, and the caller generates the call's arguments again when the attempt returns nothing. The attempt
used to render arguments too: seven argument validators each asked whether the second argument is a sampler by
rendering it, and the coordinate split rendered the first and the coordinate. Each level of nested calls
multiplied the work, so a chain of nested calls cost time exponential in its depth. Now:

- past the resource queries, the attempt returns a result only for a call that names one of these operations or
  whose first argument is a texture or image resource, so it stops before rendering anything otherwise
  (:func:`names_sampled_texture_operation`);
- whether the second argument is a sampler is answered once per attempt;
- that answer renders the argument only when the program declares a sampler whose name the text could match.
"""

from __future__ import annotations

from .image_access_contracts import (
    is_projected_texture_operation,
    is_texel_fetch_basic_operation,
    is_texel_fetch_offset_operation,
    is_texture_compare_operation,
    is_texture_gather_compare_operation,
    is_texture_gather_operation,
    is_texture_query_lod_operation,
    is_texture_sample_basic_operation,
    is_texture_sample_grad_operation,
    is_texture_sample_lod_operation,
    is_texture_sample_offset_operation,
    is_texture_sample_operation,
)

__all__ = ["SAMPLED_TEXTURE_OPERATION_PREDICATES", "names_sampled_texture_operation"]

# Every predicate that guards a result of ``MetalCodeGen.generate_texture_call_value`` after its arguments are
# generated. A predicate added there belongs here too.
SAMPLED_TEXTURE_OPERATION_PREDICATES = (
    is_texture_sample_operation,
    is_texture_sample_basic_operation,
    is_texture_sample_lod_operation,
    is_texture_sample_grad_operation,
    is_texture_sample_offset_operation,
    is_projected_texture_operation,
    is_texture_gather_operation,
    is_texture_compare_operation,
    is_texture_gather_compare_operation,
    is_texture_query_lod_operation,
    is_texel_fetch_basic_operation,
    is_texel_fetch_offset_operation,
)


def names_sampled_texture_operation(func_name) -> bool:
    """Whether ``func_name`` is one of the operations in :data:`SAMPLED_TEXTURE_OPERATION_PREDICATES`."""
    return any(
        predicate(func_name) for predicate in SAMPLED_TEXTURE_OPERATION_PREDICATES
    )

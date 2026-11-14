"""Immutable module-scope variables as SPIR-V constants.

MSL program-scope variables in the ``constant`` address space must be initialized and cannot be written (MSL 4.1
section 4.2), and HLSL ``static const`` and GLSL ``const`` globals cannot be written either. Shared headers use them
for sizes and factors: ``constant uint Width = 4; constant uint Total = Width * 3;``. The SPIR-V generator declared
each as a ``Private`` variable with an initializer. A later initializer that names one then loaded it with ``OpLoad``
at module scope, which ``spirv-val`` rejects ("Load must appear in a block").

:func:`module_constant` makes such a variable a named constant instead, when its initializer folds to a constant
instruction and its type is a scalar, vector or matrix. References then use the constant, and later initializers fold
through it as they do through ``const`` declarations. Arrays and structures stay variables, because a dynamic index
needs a variable to point into.
"""

from __future__ import annotations

from .spirv_constant_folding import (
    spirv_matrix_shape,
    spirv_scalar_kind,
    spirv_vector_shape,
)

__all__ = ["module_constant"]

_IMMUTABLE = frozenset({"const", "constant"})


def _is_immutable(node) -> bool:
    qualifiers = {str(q).lower() for q in getattr(node, "qualifiers", None) or []}
    return bool(qualifiers & _IMMUTABLE)


def _is_value_type(generator, var_type) -> bool:
    return (
        spirv_scalar_kind(generator, var_type.type.base_type) is not None
        or spirv_vector_shape(generator, var_type) is not None
        or spirv_matrix_shape(generator, var_type) is not None
    )


def module_constant(generator, node, var_type, initializer):
    """Register ``node`` as a named constant and return its id, or ``None`` to keep it a variable."""
    if initializer is None or not _is_immutable(node):
        return None
    if not generator.is_constant_instruction(initializer):
        return None
    if not _is_value_type(generator, var_type):
        return None
    generator.emit_named_constant_debug_name(node.name, initializer)
    generator.named_constants[node.name] = initializer
    value = generator.constant_values_by_id.get(initializer.id)
    if isinstance(value, (bool, int, float)):
        generator.literal_scalar_constants[node.name] = value
    return initializer

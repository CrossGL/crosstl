"""Fold scalar constant initializers for the SPIR-V generator.

SPIR-V module scope can only hold constant instructions, so a CrossGL
``const`` whose initializer is an operator expression, a conversion or a math
built-in call has to be evaluated during translation. Evaluation uses
:mod:`scalar_constant_folding`, which rounds binary32 values after every
operation and wraps integers, so the constant has the value the shader would
have computed.

Names resolve only at module scope: inside a function a local could shadow a
module-scope constant, and the generator's ordinary expression path handles
that case.
"""

from __future__ import annotations

from ..ast import BinaryOpNode, CastNode, FunctionCallNode, TernaryOpNode, UnaryOpNode
from ..float_limit_constants import float_limit_constant
from ..standard_constants import standard_math_constant
from .scalar_constant_folding import (
    FLOAT_KINDS,
    SCALAR_KINDS,
    ScalarConstant,
    ScalarConstantFolder,
    canonical_float_value,
    convert_scalar_constant,
    fold_math_builtin,
)

# Initializers that need evaluation. Bare literals and names keep the
# generator's existing literal path, so their spelling does not change.
_EVALUATED_NODES = (
    BinaryOpNode,
    CastNode,
    FunctionCallNode,
    TernaryOpNode,
    UnaryOpNode,
)


def spirv_scalar_kind(generator, type_value) -> str | None:
    """Return the scalar kind (``float``, ``int``, ...) of *type_value*, if any."""
    if type_value is None:
        return None
    name = (
        type_value if isinstance(type_value, str) else getattr(type_value, "name", None)
    )
    if not isinstance(name, str):
        return None
    kind = generator.normalize_primitive_name(name)
    return kind if kind in SCALAR_KINDS else None


def _integer_constant(value: int) -> ScalarConstant:
    for kind, low, high in (
        ("int", -(1 << 31), (1 << 31) - 1),
        ("uint", 0, (1 << 32) - 1),
        ("i64", -(1 << 63), (1 << 63) - 1),
    ):
        if low <= value <= high:
            return ScalarConstant(kind, value)
    return ScalarConstant("u64", value)


def _resolve_name(generator, name: str) -> ScalarConstant | None:
    if generator.current_function_name is not None:
        return None
    constant_id = generator.named_constants.get(name)
    if constant_id is not None:
        # Specialization constants have no entry here, so expressions over
        # them keep the OpSpecConstantOp path.
        value = generator.constant_values_by_id.get(constant_id.id)
        type_id = generator.value_types.get(constant_id.id)
        base_type = getattr(getattr(type_id, "type", None), "base_type", None)
        kind = spirv_scalar_kind(generator, base_type)
        if value is None or kind is None:
            return None
        return ScalarConstant(kind, value)
    int_value = generator.active_literal_int_constants().get(name)
    if isinstance(int_value, int) and not isinstance(int_value, bool):
        return _integer_constant(int_value)
    limit = float_limit_constant(name)
    if limit is not None:
        return ScalarConstant("float", limit.value)
    standard = standard_math_constant(name)
    if standard is not None:
        kind = spirv_scalar_kind(generator, standard.scalar_type)
        if kind in FLOAT_KINDS:
            return ScalarConstant(kind, float(standard.decimal))
    return None


def spirv_fold_scalar(generator, expr, kind: str | None) -> ScalarConstant | None:
    """Fold *expr*, converted to *kind* when given, or return ``None``."""
    folder = ScalarConstantFolder(
        lambda name: _resolve_name(generator, name),
        lambda type_value: spirv_scalar_kind(generator, type_value),
        fold_builtin=fold_math_builtin,
    )
    folded = folder.fold(expr)
    if folded is None or kind is None:
        return folded
    return convert_scalar_constant(folded, kind)


def _python_value(folded: ScalarConstant):
    if folded.kind in FLOAT_KINDS:
        return canonical_float_value(folded.kind, folded.value)
    return folded.value


def spirv_evaluated_scalar_initializer(generator, expr, type_id):
    """Return the Python value of an evaluated scalar initializer, or ``None``.

    Only operator expressions, conversions and calls are handled; bare
    literals and names are left to the generator's literal path.
    """
    if not isinstance(expr, _EVALUATED_NODES):
        return None
    kind = spirv_scalar_kind(generator, type_id.type.base_type)
    if kind is None:
        return None
    folded = spirv_fold_scalar(generator, expr, kind)
    return None if folded is None else _python_value(folded)


def spirv_folded_scalar_constant(generator, expr, target_type) -> object | None:
    """Return an ``OpConstant`` id for a foldable scalar expression, or ``None``."""
    if not isinstance(expr, _EVALUATED_NODES):
        return None
    if target_type is not None:
        kind = spirv_scalar_kind(generator, target_type.type.base_type)
        if kind is None:
            return None
        type_id = target_type
    else:
        kind = None
        type_id = None
    folded = spirv_fold_scalar(generator, expr, kind)
    if folded is None:
        return None
    if type_id is None:
        type_id = generator.register_primitive_type(folded.kind)
    return generator.register_constant(_python_value(folded), type_id)


__all__ = (
    "spirv_evaluated_scalar_initializer",
    "spirv_fold_scalar",
    "spirv_folded_scalar_constant",
    "spirv_scalar_kind",
)

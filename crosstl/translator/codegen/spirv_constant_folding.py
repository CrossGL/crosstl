"""Fold module-scope constant initializers for the SPIR-V generator.

SPIR-V module scope can only hold constant instructions, so a CrossGL
``const`` whose initializer is an operator expression, a conversion, a swizzle
or a math built-in call has to be evaluated during translation. Evaluation uses
:mod:`aggregate_constant_folding` and the scalar, vector and matrix folders
it builds on, which round binary32 values after every
operation and wraps integers, so the constant has the value the shader would
have computed.

Names resolve only at module scope: inside a function a local could shadow a
module-scope constant, and the generator's ordinary expression path handles
that case.
"""

from __future__ import annotations

import re

from ..ast import (
    ArrayAccessNode,
    BinaryOpNode,
    CastNode,
    FunctionCallNode,
    MatrixType,
    MemberAccessNode,
    SwizzleNode,
    TernaryOpNode,
    UnaryOpNode,
    VectorType,
)
from ..float_limit_constants import float_limit_constant
from ..standard_constants import standard_math_constant
from .aggregate_constant_folding import AggregateConstant, AggregateConstantFolder
from .matrix_constant_folding import (
    MatrixConstant,
    convert_matrix_constant,
    make_matrix,
)
from .scalar_constant_folding import (
    FLOAT_KINDS,
    SCALAR_KINDS,
    ScalarConstant,
    canonical_float_value,
    convert_scalar_constant,
    fold_math_builtin,
)
from .vector_constant_folding import (
    VectorConstant,
    convert_vector_constant,
    make_vector,
)

_MATRIX_NAME = re.compile(r"(d)?mat([234])(?:x([234]))?")

# Initializers that need evaluation. Bare literals and names keep the
# generator's existing literal path, so their spelling does not change.
_EVALUATED_NODES = (
    ArrayAccessNode,
    BinaryOpNode,
    CastNode,
    FunctionCallNode,
    MemberAccessNode,
    SwizzleNode,
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


def _is_spirv_type(type_value) -> bool:
    return hasattr(type_value, "id") and hasattr(type_value, "type")


def spirv_vector_shape(generator, type_value) -> tuple[str, int] | None:
    """Return ``(kind, size)`` when *type_value* is a foldable vector type."""
    if type_value is None:
        return None
    if _is_spirv_type(type_value):
        info = generator.vector_type_info_from_type(type_value)
        if info is None or info[0] is None:
            return None
        component_type, size = info
        kind = spirv_scalar_kind(generator, component_type.type.base_type)
        return None if kind is None else (kind, size)
    if isinstance(type_value, VectorType):
        kind = spirv_scalar_kind(generator, type_value.element_type)
        return None if kind is None else (kind, type_value.size)
    name = (
        type_value if isinstance(type_value, str) else getattr(type_value, "name", None)
    )
    if not isinstance(name, str):
        return None
    info = generator.vector_component_type_and_count(name)
    if info is None:
        return None
    component_name, size = info
    kind = spirv_scalar_kind(generator, component_name)
    return None if kind is None or not 2 <= size <= 4 else (kind, size)


def spirv_matrix_shape(generator, type_value) -> tuple[str, int, int] | None:
    """Return ``(kind, columns, rows)`` when *type_value* is a matrix type."""
    if type_value is None:
        return None
    if _is_spirv_type(type_value):
        info = generator.matrix_type_info_from_type(type_value)
        if info is None or info[0] is None:
            return None
        column_type, columns = info
        column_shape = spirv_vector_shape(generator, column_type)
        if column_shape is None or column_shape[0] not in FLOAT_KINDS:
            return None
        return column_shape[0], columns, column_shape[1]
    if isinstance(type_value, MatrixType):
        kind = spirv_scalar_kind(generator, type_value.element_type)
        if kind not in FLOAT_KINDS:
            return None
        return kind, type_value.cols, type_value.rows
    name = (
        type_value if isinstance(type_value, str) else getattr(type_value, "name", None)
    )
    if not isinstance(name, str):
        return None
    match = _MATRIX_NAME.fullmatch(generator.normalize_hlsl_matrix_type(name))
    if match is None:
        return None
    is_double, columns, rows = match.groups()
    kind = "double" if is_double else "float"
    return kind, int(columns), int(rows or columns)


def _integer_constant(value: int) -> ScalarConstant:
    for kind, low, high in (
        ("int", -(1 << 31), (1 << 31) - 1),
        ("uint", 0, (1 << 32) - 1),
        ("i64", -(1 << 63), (1 << 63) - 1),
    ):
        if low <= value <= high:
            return ScalarConstant(kind, value)
    return ScalarConstant("u64", value)


def _scalar_value_of(generator, constant_id: int) -> ScalarConstant | None:
    value = generator.constant_values_by_id.get(constant_id)
    type_id = generator.value_types.get(constant_id)
    base_type = getattr(getattr(type_id, "type", None), "base_type", None)
    kind = spirv_scalar_kind(generator, base_type)
    if value is None or kind is None:
        return None
    return ScalarConstant(kind, value)


def _vector_value_of(generator, constant_id: int) -> VectorConstant | None:
    shape = spirv_vector_shape(generator, generator.value_types.get(constant_id))
    if shape is None:
        return None
    for table in (generator.vector_constants, generator.composite_constants):
        for (_type_key, component_ids), registered in table.items():
            if registered.id != constant_id:
                continue
            components = []
            for component_id in component_ids:
                value = generator.constant_values_by_id.get(component_id)
                if value is None:
                    return None
                components.append(ScalarConstant(shape[0], value))
            vector = make_vector(components)
            return vector if vector is not None and vector.size == shape[1] else None
    return None


def _matrix_value_of(generator, constant_id: int) -> MatrixConstant | None:
    shape = spirv_matrix_shape(generator, generator.value_types.get(constant_id))
    if shape is None:
        return None
    for (_type_key, column_ids), registered in generator.composite_constants.items():
        if registered.id != constant_id:
            continue
        columns = [_vector_value_of(generator, column_id) for column_id in column_ids]
        matrix = make_matrix(columns)
        return matrix if matrix is not None and matrix.shape == shape else None
    return None


def _aggregate_value_of(generator, constant_id: int) -> AggregateConstant | None:
    """Return the folded array or struct held by composite constant *constant_id*."""
    for (_type_key, element_ids), registered in generator.composite_constants.items():
        if registered.id != constant_id:
            continue
        elements = []
        for element_id in element_ids:
            element = _constant_value_of(generator, element_id)
            if element is None:
                return None
            elements.append(element)
        value_type = generator.value_types.get(constant_id)
        base_type = getattr(getattr(value_type, "type", None), "base_type", None)
        members = generator.current_struct_members.get(base_type)
        if members is None:
            return AggregateConstant(tuple(elements))
        if len(members) != len(elements):
            return None
        names = tuple(member_name for _member_type, member_name in members)
        return AggregateConstant(tuple(elements), names)
    return None


def _constant_value_of(generator, constant_id: int):
    """Return the folded value of the constant with id *constant_id*, or ``None``."""
    for lookup in (_scalar_value_of, _vector_value_of, _matrix_value_of):
        value = lookup(generator, constant_id)
        if value is not None:
            return value
    if spirv_vector_shape(generator, generator.value_types.get(constant_id)):
        return None
    if spirv_matrix_shape(generator, generator.value_types.get(constant_id)):
        return None
    return _aggregate_value_of(generator, constant_id)


def _resolve_name(generator, name: str):
    if generator.current_function_name is not None:
        return None
    constant_id = generator.named_constants.get(name)
    if constant_id is not None:
        # Specialization constants have no entry in the value tables, so
        # expressions over them keep the OpSpecConstantOp path.
        return _constant_value_of(generator, constant_id.id)
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


def spirv_fold_constant(generator, expr):
    """Fold *expr* to a scalar, vector or matrix constant, or return ``None``."""
    folder = AggregateConstantFolder(
        lambda name: _resolve_name(generator, name),
        lambda type_value: spirv_scalar_kind(generator, type_value),
        lambda type_value: spirv_vector_shape(generator, type_value),
        lambda type_value: spirv_matrix_shape(generator, type_value),
        fold_builtin=fold_math_builtin,
    )
    return folder.fold(expr)


def spirv_fold_scalar(generator, expr, kind: str | None) -> ScalarConstant | None:
    """Fold *expr* to a scalar, converted to *kind* when given, or ``None``."""
    folded = spirv_fold_constant(generator, expr)
    if not isinstance(folded, ScalarConstant) or kind is None:
        return folded if isinstance(folded, ScalarConstant) else None
    return convert_scalar_constant(folded, kind)


def _python_value(folded: ScalarConstant):
    if folded.kind in FLOAT_KINDS:
        return canonical_float_value(folded.kind, folded.value)
    return folded.value


def spirv_evaluated_scalar_initializer(generator, expr, type_id):
    """Return the Python value of an evaluated scalar initializer, or ``None``.

    Only operator expressions, conversions, swizzles and calls are handled;
    bare literals and names are left to the generator's literal path.
    """
    if not isinstance(expr, _EVALUATED_NODES):
        return None
    kind = spirv_scalar_kind(generator, type_id.type.base_type)
    if kind is None:
        return None
    folded = spirv_fold_scalar(generator, expr, kind)
    return None if folded is None else _python_value(folded)


def _register_vector(generator, folded, target_type):
    if not isinstance(folded, (ScalarConstant, VectorConstant)):
        return None
    if target_type is None:
        if not isinstance(folded, VectorConstant):
            return None
        component_type = generator.register_primitive_type(folded.kind)
        vector_type = generator.register_vector_type(component_type, folded.size)
    else:
        kind, size = spirv_vector_shape(generator, target_type)
        component_type, _size = generator.vector_type_info_from_type(target_type)
        if isinstance(folded, ScalarConstant):
            # A scalar initializer of a vector constant fills every component.
            folded = make_vector((folded,) * size)
        if folded is None or folded.size != size:
            return None
        folded = convert_vector_constant(folded, kind)
        if folded is None:
            return None
        vector_type = target_type
    components = [
        generator.register_constant(_python_value(component), component_type)
        for component in folded.components
    ]
    return generator.register_vector_constant(vector_type, components)


def _register_matrix(generator, folded, target_type):
    if not isinstance(folded, MatrixConstant):
        return None
    if target_type is None:
        component_type = generator.register_primitive_type(folded.kind)
        column_type = generator.register_vector_type(component_type, folded.row_count)
        matrix_type = generator.register_matrix_type(column_type, folded.column_count)
    else:
        kind, columns, rows = spirv_matrix_shape(generator, target_type)
        if folded.shape[1:] != (columns, rows):
            return None
        folded = convert_matrix_constant(folded, kind)
        if folded is None:
            return None
        matrix_type = target_type
        column_type, _columns = generator.matrix_type_info_from_type(target_type)
        component_type, _rows = generator.vector_type_info_from_type(column_type)
    # Components first, then columns, as the matrix constructor path orders them.
    components = [
        [
            generator.register_constant(_python_value(component), component_type)
            for component in column.components
        ]
        for column in folded.columns
    ]
    columns = [
        generator.register_vector_constant(column_type, column) for column in components
    ]
    return generator.register_composite_constant(matrix_type, columns)


def _defers_to_constructor_path(generator, expr) -> bool:
    """Whether a matrix constructor is left to the generator's constructor path.

    That path already folds constructors over scalars and vectors; it cannot
    take a matrix argument, so only those constructors are folded here.
    """
    if not isinstance(expr, FunctionCallNode):
        return False
    callee = getattr(expr, "function", getattr(expr, "name", None))
    if spirv_matrix_shape(generator, getattr(callee, "name", callee)) is None:
        return False
    return not any(
        isinstance(spirv_fold_constant(generator, argument), MatrixConstant)
        for argument in getattr(expr, "args", None) or []
    )


def _array_constructor_constant(generator, expr):
    """Return the ``OpConstantComposite`` of a ``T[N](...)`` constructor, or ``None``."""
    if not isinstance(expr, FunctionCallNode):
        return None
    callee = getattr(expr, "function", getattr(expr, "name", None))
    callee_name = getattr(callee, "name", callee)
    if not isinstance(callee_name, str):
        return None
    array_info = generator.split_outer_array_type(callee_name)
    if array_info is None:
        return None
    element_type_name, size = array_info
    arguments = list(getattr(expr, "args", None) or [])
    if size is None:
        size = len(arguments)
    if not isinstance(size, int) or size != len(arguments) or size == 0:
        return None
    element_type = generator.map_crossgl_type(element_type_name)
    elements = []
    for argument in arguments:
        element = generator.process_constant_expression(argument, element_type)
        if element is None or not generator.is_constant_instruction(element):
            return None
        elements.append(element)
    array_type = generator.register_array_type(element_type, size)
    return generator.register_composite_constant(array_type, elements)


def spirv_folded_constant(generator, expr, target_type) -> object | None:
    """Return a constant id for a foldable initializer, or ``None``."""
    if not isinstance(expr, _EVALUATED_NODES):
        return None
    array_constant = _array_constructor_constant(generator, expr)
    if array_constant is not None:
        return array_constant
    if _defers_to_constructor_path(generator, expr):
        return None
    if target_type is not None and spirv_matrix_shape(generator, target_type):
        return _register_matrix(
            generator, spirv_fold_constant(generator, expr), target_type
        )
    if target_type is not None and spirv_vector_shape(generator, target_type):
        return _register_vector(
            generator, spirv_fold_constant(generator, expr), target_type
        )
    if target_type is not None:
        kind = spirv_scalar_kind(generator, target_type.type.base_type)
        if kind is None:
            return None
        folded = spirv_fold_scalar(generator, expr, kind)
        if folded is None:
            return None
        return generator.register_constant(_python_value(folded), target_type)
    folded = spirv_fold_constant(generator, expr)
    if isinstance(folded, VectorConstant):
        return _register_vector(generator, folded, None)
    if isinstance(folded, MatrixConstant):
        return _register_matrix(generator, folded, None)
    if folded is None:
        return None
    type_id = generator.register_primitive_type(folded.kind)
    return generator.register_constant(_python_value(folded), type_id)


__all__ = (
    "spirv_evaluated_scalar_initializer",
    "spirv_fold_constant",
    "spirv_fold_scalar",
    "spirv_folded_constant",
    "spirv_matrix_shape",
    "spirv_scalar_kind",
    "spirv_vector_shape",
)

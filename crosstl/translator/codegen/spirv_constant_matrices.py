"""Module-scope SPIR-V constants for CrossGL matrix constructors.

A ``const mat4 m = mat4(...)`` outside a function must become an ``OpConstantComposite`` of column constants:
SPIR-V 1.6 section 2.16.1 allows ``OpCompositeConstruct`` only inside a block, and an initializer that cannot be
folded must not silently become the zero matrix. The constructor rules are GLSL 4.60 section 5.4.2: the scalar
components of every argument fill the matrix column by column, and a single scalar argument fills the diagonal.
"""

from __future__ import annotations

from ..ast import FunctionCallNode

__all__ = ["constant_matrix_constructor"]


def constant_matrix_constructor(gen, expr, callee_name, target_type):
    """The ``OpConstantComposite`` for a matrix constructor of constants, or None when it is not one."""
    if not isinstance(expr, FunctionCallNode):
        return None
    matrix_type = target_type
    if matrix_type is None or gen.matrix_type_info_from_type(matrix_type) is None:
        if not isinstance(callee_name, str):
            return None
        try:
            matrix_type = gen.map_crossgl_type(callee_name)
        except Exception:  # noqa: BLE001 - an unknown callee is not a matrix type
            return None
    matrix_info = gen.matrix_type_info_from_type(matrix_type)
    if matrix_info is None:
        return None
    column_type, cols = matrix_info
    column_info = gen.vector_type_info_from_type(column_type)
    if column_info is None:
        return None
    component_type, rows = column_info
    args = list(getattr(expr, "args", None) or [])
    components: list = []
    for arg in args:
        values = _constant_components(gen, arg, component_type)
        if values is None:
            return None
        components.extend(values)
    if len(args) == 1 and len(components) == 1:
        zero = gen.register_constant(0.0, component_type)
        diagonal = components[0]
        components = [
            diagonal if row == col else zero
            for col in range(cols)
            for row in range(rows)
        ]
    if len(components) != cols * rows:
        return None
    columns = [
        gen.register_vector_constant(
            column_type, components[col * rows : (col + 1) * rows]
        )
        for col in range(cols)
    ]
    return gen.register_composite_constant(matrix_type, columns)


def _constant_components(gen, arg, component_type) -> list | None:
    value = gen.process_constant_expression(arg, None)
    if value is None or not gen.is_constant_instruction(value):
        return None
    value_type = gen.value_types.get(value.id)
    vector_info = gen.vector_type_info_from_type(value_type)
    if vector_info is None:
        if gen.matrix_type_info_from_type(value_type) is not None:
            return None
        return _coerced(gen, [value], component_type)
    for key, constant in gen.vector_constants.items():
        if constant.id == value.id:
            scalars = {scalar.id: scalar for scalar in gen.constants.values()}
            return _coerced(
                gen,
                [scalars.get(component_id) for component_id in key[1]],
                component_type,
            )
    return None


def _coerced(gen, values, component_type) -> list | None:
    result = []
    for value in values:
        if value is None:
            return None
        converted = gen.coerce_scalar_constant_to_type(value, component_type)
        converted_type = gen.value_types.get(converted.id)
        if converted_type is None or converted_type.id != component_type.id:
            return None
        result.append(converted)
    return result

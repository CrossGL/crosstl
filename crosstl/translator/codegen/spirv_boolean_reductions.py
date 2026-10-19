"""SPIR-V lowering of CrossGL's ``any`` and ``all``.

``any(v)`` and ``all(v)`` reduce a vector to one ``bool`` (GLSL 4.60 section 8.7); SPIR-V spells them ``OpAny`` and
``OpAll`` on a bool vector (SPIR-V 1.6 section 3.52.15).  CrossGL also accepts HLSL's forms: a numeric vector is
first converted componentwise to bool by comparing with zero, and a scalar reduces to itself converted to ``bool``
(HLSL intrinsic reference, ``any`` and ``all``).  An operand that converts to neither is refused.
"""

from __future__ import annotations

BOOLEAN_REDUCTIONS = {"any": "OpAny", "all": "OpAll"}


def spirv_boolean_reduction(emitter, function_name, value):
    """Emit ``any(value)`` or ``all(value)`` and return the ``bool`` result id."""
    bool_type = emitter.register_primitive_type("bool")
    vector = emitter.vector_component_type_and_count(value.type.base_type)
    if vector is None:
        converted = emitter.ensure_bool_value(value)
        if not emitter.value_has_type(converted, bool_type):
            raise emitter.builtin_lowering_error(
                function_name,
                f"{function_name} needs a scalar or vector operand convertible to bool",
            )
        return converted
    _, count = vector
    bool_vector = emitter.register_vector_type(bool_type, count)
    converted = emitter.convert_value_to_type(value, bool_vector)
    if not emitter.value_has_type(converted, bool_vector):
        raise emitter.builtin_lowering_error(
            function_name,
            f"{function_name} needs a vector operand convertible to a bool vector",
        )
    id_value = emitter.get_id()
    emitter.emit(
        f"%{id_value} = {BOOLEAN_REDUCTIONS[function_name]} %{bool_type.id} %{converted.id}"
    )
    result = type(value)(id_value, bool_type.type)
    emitter.value_types[id_value] = bool_type
    return result

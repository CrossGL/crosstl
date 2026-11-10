"""Copy-in/copy-out for arguments of ``out`` and ``inout`` parameters on SPIR-V output.

GLSL 4.60 section 6.1.1 passes ``out`` and ``inout`` parameters by value-result: the arguments are evaluated once,
left to right; the l-value of an ``out`` or ``inout`` argument is fixed at its turn, an ``inout`` value is copied in
then, and the parameter's final value is copied back when the function returns. HLSL gives ``out`` and ``inout``
the same copy semantics. The SPIR-V generator inlines every function that has such a parameter and used to bind the
parameter straight to the caller's pointer. That aliases the caller's variable for the whole call, so a later
argument that writes the variable (``scale(m, bump(m))``) changed the value the callee read, and a callee write was
visible through every other name of the same variable before the call returned.

The argument now gets its own ``Function`` variable of the parameter's type, as glslang emits it: the l-value's
pointer is taken at the argument's turn, an ``inout`` value is loaded and stored into the variable then, the
parameter name is bound to the variable, and after the inlined body the variable is stored back through that same
pointer, in parameter order. An ``inout`` parameter the callee never reads still copies back the value it copied in,
which undoes a later argument's write to the same variable.

Metal ``thread``/``device``/``threadgroup`` references reach CrossGL as ``inout`` with an address space, and C++
style ``T&`` parameters as reference types. Both keep reference semantics and stay bound to the caller's pointer, as
do values the generator cannot copy as a whole (resources, pointers and runtime-sized arrays).
"""

from __future__ import annotations

VALUE_RESULT_WRITE_BACK = "value_result_write_back"

# Address spaces that make an ``inout`` parameter a Metal reference rather than a value-result parameter.
REFERENCE_ADDRESS_SPACES = frozenset(
    {
        "constant",
        "device",
        "object_data",
        "ray_data",
        "thread",
        "threadgroup",
        "threadgroup_imageblock",
    }
)


def value_result_qualifier(gen, param) -> str | None:
    """``"out"`` or ``"inout"`` when ``param`` is passed by value-result, else None."""
    qualifiers = gen.parameter_qualifier_names(param)
    if qualifiers & REFERENCE_ADDRESS_SPACES:
        return None
    param_type = getattr(param, "param_type", getattr(param, "vtype", None))
    type_name = str(gen.type_name_from_value(param_type) or "").strip()
    if type_name.startswith("&") or type_name.endswith("&"):
        return None
    if "inout" in qualifiers or {"in", "out"} <= qualifiers:
        return "inout"
    if "out" in qualifiers:
        return "out"
    return None


def prepare_value_result_binding(
    gen, param, param_name: str, pointer_arg, source_value_type, param_value_type
) -> dict | None:
    """A copy-in/copy-out binding of ``param`` to ``pointer_arg``, or None to keep the caller's pointer."""
    qualifier = value_result_qualifier(gen, param)
    if qualifier is None:
        return None
    temporary_type = param_value_type or source_value_type
    if not _copyable(gen, temporary_type) or not _copyable(gen, source_value_type):
        return None
    if temporary_type.id != source_value_type.id and not (
        gen.aggregate_types_are_layout_compatible(source_value_type, temporary_type)
    ):
        return None

    temporary = gen.create_variable(temporary_type, "Function")
    if qualifier == "inout":
        value = gen.load_from_variable(pointer_arg, source_value_type)
        converted = gen.convert_value_to_type(value, temporary_type)
        if not gen.value_has_type(converted, temporary_type):
            gen.refuse_lowering(
                f"SPIR-V cannot copy the argument of inout parameter {param_name} "
                "into the parameter's type"
            )
            return None
        gen.store_to_variable(temporary, converted)
    return {
        "kind": "reference",
        "name": param_name,
        "value": temporary,
        "copy_pointer": False,
        VALUE_RESULT_WRITE_BACK: (pointer_arg, source_value_type, temporary_type),
    }


def write_back_value_result_binding(gen, binding) -> None:
    """Store a value-result parameter's final value back through the pointer taken at its argument's turn."""
    pointer_arg, source_value_type, temporary_type = binding[VALUE_RESULT_WRITE_BACK]
    value = gen.load_from_variable(binding["value"], temporary_type)
    converted = gen.convert_value_to_type(value, source_value_type)
    if not gen.value_has_type(converted, source_value_type):
        gen.refuse_lowering(
            f"SPIR-V cannot copy parameter {binding['name']} back to its argument"
        )
        return
    gen.store_to_variable(pointer_arg, converted)


def _copyable(gen, type_id, depth: int = 0) -> bool:
    """Whether a value of ``type_id`` can be loaded and stored as a whole: numbers and fixed aggregates of them."""
    if type_id is None or depth > 32:
        return False
    if type_id.type.storage_class:
        return False
    for name, primitive in gen.primitive_types.items():
        if primitive.id == type_id.id:
            return name != "void"
    if (
        gen.vector_type_info_from_type(type_id) is not None
        or gen.matrix_type_info_from_type(type_id) is not None
    ):
        return True
    array_info = gen.array_type_info_from_type(type_id)
    if array_info is not None:
        element_type, size = array_info
        return size is not None and _copyable(gen, element_type, depth + 1)
    members = gen.current_struct_members.get(type_id.type.base_type)
    if members is not None:
        return all(_copyable(gen, member, depth + 1) for member, _name in members)
    return False

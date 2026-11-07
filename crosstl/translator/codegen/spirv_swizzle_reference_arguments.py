"""Copy-in/copy-out for a swizzle of several vector components passed to ``out`` or ``inout``.

GLSL 4.60 section 6.1.1 passes ``out`` and ``inout`` parameters by copy: every argument is evaluated
once, left to right, at the call; the l-value of an ``out`` or ``inout`` argument is fixed then, an
``inout`` value is copied in, and the parameter's final value is copied back to that l-value when the
function returns. HLSL gives ``out`` and ``inout`` the same copy semantics. ``v.xz`` is such an
l-value as long as no component repeats (GLSL 4.60 section 5.8), but SPIR-V cannot point at it:
``OpAccessChain`` selects a single vector component per index (SPIR-V 1.6, section 3.56.7), so the
pointer-preserving inliner has nothing to bind the parameter to.

The argument gets its own ``Function`` variable of the swizzle's type instead. The base vector's
pointer is taken when the argument is evaluated, the selected components are copied in for an
``inout`` (or reference) parameter, the parameter name is bound to the variable, and after the
inlined body the variable's value is stored back into the selected components through that same
pointer. In buffers and workgroup memory each selected component is read and written through its own
pointer, so the other components of the vector are never stored.
"""

from __future__ import annotations

from ..ast import MemberAccessNode

SWIZZLE_WRITE_BACK = "swizzle_write_back"


def prepare_swizzle_reference_binding(
    gen, func_name: str, index: int, param, param_name: str, arg, param_value_type
) -> dict | None:
    """A copy-in/copy-out binding for ``arg`` when it is a multi-component swizzle of an l-value.

    Returns None when ``arg`` is not such a swizzle, leaving the caller's own diagnostics in place.
    """
    member = getattr(arg, "member", None)
    if not isinstance(arg, MemberAccessNode) or not isinstance(member, str):
        return None
    if len(member) < 2:
        return None
    base_pointer = gen.assignable_pointer_from_expression(arg.object)
    if base_pointer is None or not base_pointer.type.storage_class:
        return None
    update_info = gen.vector_swizzle_update_info(base_pointer, member)
    if update_info is None:
        return None
    vector_type, indices, member_type, swizzle_type = update_info
    if len(set(indices)) != len(indices):
        raise gen.unsupported_reference_argument_error(
            func_name,
            index,
            param_name,
            arg,
            feature="duplicate-swizzle-reference-argument",
            reason=(
                f"swizzle '{member}' repeats a component, so it is not an l-value "
                "(GLSL 4.60 section 5.8)"
            ),
        )
    if param_value_type is not None and param_value_type.id != swizzle_type.id:
        raise gen.unsupported_reference_argument_error(
            func_name,
            index,
            param_name,
            arg,
            feature="incompatible-reference-argument",
            reason=(
                f"argument type {swizzle_type.type.base_type} is incompatible "
                f"with {param_value_type.type.base_type}"
            ),
        )

    temporary = gen.create_variable(swizzle_type, "Function")
    qualifiers = gen.parameter_qualifier_names(param)
    if not ("out" in qualifiers and "inout" not in qualifiers):
        gen.store_to_variable(
            temporary,
            _load_swizzle(
                gen, base_pointer, vector_type, indices, member_type, swizzle_type
            ),
        )
    return {
        "kind": "reference",
        "name": param_name,
        "value": temporary,
        "copy_pointer": False,
        SWIZZLE_WRITE_BACK: (base_pointer, member, swizzle_type),
    }


def _load_swizzle(gen, base_pointer, vector_type, indices, member_type, swizzle_type):
    shared_components = gen.load_vector_components(base_pointer, indices, member_type)
    if shared_components is not None:
        return gen.composite_construct(swizzle_type, shared_components)
    vector_value = gen.load_from_variable(base_pointer, vector_type)
    return gen.vector_shuffle(vector_value, swizzle_type, indices)


def write_back_swizzle_reference_bindings(gen, prepared_bindings) -> None:
    """Copy every swizzle argument's final value back, in parameter order, after the inlined body."""
    for binding in prepared_bindings:
        target = binding.get(SWIZZLE_WRITE_BACK)
        if target is None:
            continue
        base_pointer, member, swizzle_type = target
        value = gen.load_from_variable(binding["value"], swizzle_type)
        gen.store_to_vector_swizzle(base_pointer, member, value)

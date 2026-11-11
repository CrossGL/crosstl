"""DirectX writes and atomics through same-width 32-bit storage word views."""

from ..ast import ArrayAccessNode, UnaryOpNode
from .pointer_reinterpret import PointerReinterpretationError
from .storage_word_views import (
    hlsl_word_bit_cast,
    is_same_width_word_view,
    layout_scalar_kind,
    word_view_index,
)


def hlsl_word_view_element(generator, binding, index_expression):
    """Return the backing word lvalue of one view element, or ``None``."""

    reinterpretation = (binding or {}).get("pointer_reinterpretation")
    if not is_same_width_word_view(reinterpretation):
        return None
    rendered_index = (
        str(index_expression)
        if isinstance(index_expression, (int, float))
        else generator.generate_expression(index_expression)
    )
    rendered_index = generator.hlsl_resource_pointer_offset_sum(
        binding.get("offset"), rendered_index
    )
    word = word_view_index(binding.get("byte_offset", "0"), rendered_index)
    member = reinterpretation.get("source_member")
    suffix = f".{member}" if member else ""
    return {
        "root": binding["root"],
        "word": f"uint({word})",
        "suffix": suffix,
        "lvalue": f"{binding['root']}[uint({word})]{suffix}",
        "source_layout": reinterpretation["source_layout"],
        "target_layout": reinterpretation["target_layout"],
    }


def _atomic_signedness_error(element, logical_kind, physical_kind):
    return PointerReinterpretationError(
        "DirectX integer atomics through a reinterpreted storage view need "
        f"the backing word's signedness: the view is {logical_kind}, the "
        f"backing word is {physical_kind}",
        source_type=element["source_layout"].name,
        target_type=element["target_layout"].name,
        address_space="storage",
        alignment=4,
        access="read_write",
        target_backend="directx",
        reason="atomic-signedness-mismatch",
    )


def hlsl_word_view_atomic_lvalue(generator, target):
    """Return the backing word lvalue of an integer atomic target, or ``None``.

    An integer atomic through a word view operates on the backing word, so
    the view and the word must agree on signedness for ``min``/``max`` and
    for the returned original value. Any other reinterpreted view is refused
    because its element is not one addressable word.
    """

    if isinstance(target, UnaryOpNode) and getattr(target, "op", None) == "*":
        pointer, index = target.operand, 0
    elif isinstance(target, ArrayAccessNode):
        pointer, index = target.array, target.index
    else:
        return None
    if not generator.hlsl_resource_pointer_alias_expression(pointer):
        return None
    binding = generator.hlsl_resource_pointer_binding(pointer)
    reinterpretation = (binding or {}).get("pointer_reinterpretation")
    if reinterpretation is None:
        return None
    if not is_same_width_word_view(reinterpretation):
        raise PointerReinterpretationError(
            "DirectX atomics through a reinterpreted storage view need one "
            "32-bit scalar per backing word",
            source_type=reinterpretation["source_layout"].name,
            target_type=reinterpretation["target_layout"].name,
            address_space="storage",
            alignment=reinterpretation["target_layout"].byte_width,
            access="read_write",
            target_backend="directx",
            reason="atomic-requires-word-view",
            source_location=getattr(target, "source_location", None),
        )
    generator.generate_hlsl_resource_pointer_access(
        pointer, index, require_write=True, word_view=True
    )
    element = hlsl_word_view_element(generator, binding, index)
    physical_kind = layout_scalar_kind(element["source_layout"])
    logical_kind = layout_scalar_kind(element["target_layout"])
    if logical_kind != physical_kind:
        raise _atomic_signedness_error(element, logical_kind, physical_kind)
    return element["lvalue"]


def hlsl_word_view_atomic_storage(generator, binding, index, resource_type):
    """Describe a word-view atomic target for the shared atomic helpers."""

    element = hlsl_word_view_element(generator, binding, index)
    if element is None:
        return None
    physical_kind = layout_scalar_kind(element["source_layout"])
    logical_kind = layout_scalar_kind(element["target_layout"])
    if logical_kind != "float" and logical_kind != physical_kind:
        raise _atomic_signedness_error(element, logical_kind, physical_kind)
    return {
        "resource_type": generator.map_type(
            generator.directx_resource_declaration_type(resource_type)
        ),
        "arguments": [element["root"], element["word"]],
        "index_types": ["uint"],
        "suffix": "[index0]" + element["suffix"],
        "physical_kind": physical_kind,
    }


def hlsl_word_float_atomic_body(operation, target, names, physical_kind):
    """Return a float atomic helper body over an integer backing word."""

    zero = "0u" if physical_kind == "uint" else "0"
    to_bits = "asuint" if physical_kind == "uint" else "asint"
    bits, expected, desired = names["observed"], names["expected"], names["desired"]
    value, original = names["value"], names["original"]
    if operation == "atomicLoad":
        return (
            f"    {physical_kind} {bits};\n"
            f"    InterlockedOr({target}, {zero}, {bits});\n"
            f"    {original} = asfloat({bits});\n"
        )
    if operation in {"atomicExchange", "atomicStore"}:
        return (
            f"    {physical_kind} {bits};\n"
            f"    InterlockedExchange({target}, {to_bits}({value}), {bits});\n"
            f"    {original} = asfloat({bits});\n"
        )
    return (
        f"    {physical_kind} {bits};\n"
        f"    InterlockedOr({target}, {zero}, {bits});\n"
        "    [allow_uav_condition]\n"
        "    while (true) {\n"
        f"        {physical_kind} {expected} = {bits};\n"
        f"        precise float {desired} = asfloat({expected}) + {value};\n"
        f"        InterlockedCompareExchange({target}, {expected}, "
        f"{to_bits}({desired}), {bits});\n"
        f"        if ({bits} == {expected}) {{\n"
        f"            {original} = asfloat({expected});\n"
        "            return;\n"
        "        }\n"
        "    }\n"
    )


def generate_hlsl_word_view_assignment(generator, node, target, value, op):
    """Store through a same-width word view, or return ``None``.

    The view element is the whole backing word, so a store writes the bit cast
    of the value. A compound store reads and writes that word, which needs an
    index without side effects so it is evaluated once per access.
    """

    if isinstance(target, UnaryOpNode) and getattr(target, "op", None) == "*":
        pointer, index = target.operand, 0
    elif isinstance(target, ArrayAccessNode):
        pointer, index = target.array, target.index
    else:
        return None
    if not generator.hlsl_resource_pointer_alias_expression(pointer):
        return None
    binding = generator.hlsl_resource_pointer_binding(pointer)
    if not is_same_width_word_view((binding or {}).get("pointer_reinterpretation")):
        return None
    # Validates writable storage before the store is rendered.
    generator.generate_hlsl_resource_pointer_access(
        pointer, index, require_write=True, word_view=True
    )
    element = hlsl_word_view_element(generator, binding, index)
    source_layout, target_layout = element["source_layout"], element["target_layout"]
    logical_type = layout_scalar_kind(target_layout)
    rhs = generator.generate_expression_with_expected(
        value, logical_type, source_location=getattr(node, "source_location", None)
    )
    if op == "=":
        stored = rhs
    else:
        if any(
            generator.hlsl_private_pointer_expression_has_side_effects(part)
            for part in (pointer, index)
            if not isinstance(part, (int, float))
        ):
            raise PointerReinterpretationError(
                "DirectX compound stores through a reinterpreted storage view "
                "need a stable element index",
                source_type=source_layout.name,
                target_type=target_layout.name,
                address_space="storage",
                alignment=4,
                access="read_write",
                target_backend="directx",
                reason="side-effecting-word-view-index",
                source_location=getattr(node, "source_location", None),
            )
        current = hlsl_word_bit_cast(element["lvalue"], source_layout, target_layout)
        stored = f"({current} {op[:-1]} ({rhs}))"
        if logical_type != "float":
            stored = f"{logical_type}{stored}"
    return f"{element['lvalue']} = " + hlsl_word_bit_cast(
        stored, target_layout, source_layout
    )

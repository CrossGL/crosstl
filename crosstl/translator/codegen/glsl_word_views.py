"""OpenGL writes and atomics through same-width 32-bit storage word views."""

from ..ast import ArrayAccessNode, UnaryOpNode
from .pointer_reinterpret import PointerReinterpretationError
from .storage_word_views import (
    glsl_word_bit_cast,
    is_same_width_word_view,
    layout_scalar_kind,
    word_view_index,
)


def glsl_word_view_target(generator, expression):
    """Return ``(pointer, index, binding)`` for a word-view element lvalue."""

    if isinstance(expression, ArrayAccessNode):
        pointer, index = expression.array, expression.index
    elif (
        isinstance(expression, UnaryOpNode)
        and generator.map_operator(expression.op) == "*"
        and not getattr(expression, "is_postfix", False)
    ):
        pointer, index = expression.operand, 0
    else:
        return None
    binding = generator.glsl_storage_pointer_binding(
        pointer, generator.glsl_storage_pointer_aliases()
    )
    if binding is None or not is_same_width_word_view(
        binding.get("pointer_reinterpretation")
    ):
        return None
    return pointer, index, binding


def glsl_word_view_element(generator, pointer, index, binding):
    """Render the backing word lvalue of one view element."""

    reinterpretation = binding["pointer_reinterpretation"]
    rendered_offset = generator.glsl_workgroup_pointer_offset_expression(binding)
    rendered_index = generator.glsl_index_expression(index, pointer)
    if rendered_offset in {"0", "0u"}:
        composed = rendered_index
    elif rendered_index in {"0", "0u"}:
        composed = rendered_offset
    else:
        composed = f"({rendered_offset} + int({rendered_index}))"
    word = word_view_index(binding.get("byte_offset", "0"), composed)
    member = reinterpretation.get("source_member")
    suffix = f".{member}" if member else ""
    return {
        "word": word,
        "suffix": suffix,
        "lvalue": f"{binding['root']}[int({word})]{suffix}",
        "target": f"{binding['root']}[index0]{suffix}",
        "source_layout": reinterpretation["source_layout"],
        "target_layout": reinterpretation["target_layout"],
    }


def glsl_word_view_float_atomic_storage(generator, expression):
    """Describe a float atomic on a word view for the float atomic helpers.

    The helpers update one ``uint`` word by compare-and-swap, so the backing
    word must be ``uint``. Returns ``None`` when ``expression`` is not a view.
    """

    target = glsl_word_view_target(generator, expression)
    if target is None:
        return None
    pointer, index, binding = target
    generator.validate_glsl_storage_pointer_binding_access(
        binding, "read_write", expression
    )
    element = glsl_word_view_element(generator, pointer, index, binding)
    if layout_scalar_kind(element["source_layout"]) != "uint":
        raise PointerReinterpretationError(
            "OpenGL float atomics through a reinterpreted storage view need "
            "uint backing words",
            source_type=element["source_layout"].name,
            target_type=element["target_layout"].name,
            address_space="storage",
            alignment=4,
            access="read_write",
            target_backend="opengl",
            reason="float-atomic-backing-word-kind",
            source_location=getattr(expression, "source_location", None),
        )
    return {"target": element["target"], "arguments": [f"uint({element['word']})"]}


def generate_glsl_word_view_assignment(generator, node, left, right, op):
    """Store through a same-width word view, or return ``None``."""

    target = glsl_word_view_target(generator, left)
    if target is None:
        return None
    pointer, index, binding = target
    generator.validate_glsl_storage_pointer_binding_access(
        binding, "write" if op == "=" else "read_write", left
    )
    element = glsl_word_view_element(generator, pointer, index, binding)
    source_layout, target_layout = element["source_layout"], element["target_layout"]
    logical_type = layout_scalar_kind(target_layout)
    rhs = generator.generate_expression_with_expected(right, logical_type)
    if op == "=":
        stored = rhs
    else:
        if any(
            generator.glsl_private_pointer_expression_has_side_effects(part)
            for part in (pointer, index)
            if not isinstance(part, (int, float))
        ):
            raise PointerReinterpretationError(
                "OpenGL compound stores through a reinterpreted storage view "
                "need a stable element index",
                source_type=source_layout.name,
                target_type=target_layout.name,
                address_space="storage",
                alignment=4,
                access="read_write",
                target_backend="opengl",
                reason="side-effecting-word-view-index",
                source_location=getattr(node, "source_location", None),
            )
        current = glsl_word_bit_cast(element["lvalue"], source_layout, target_layout)
        stored = f"{logical_type}({current} {op[:-1]} ({rhs}))"
    return f"{element['lvalue']} = " + glsl_word_bit_cast(
        stored, target_layout, source_layout
    )


def glsl_word_view_integer_storage(generator, expression):
    """Return the backing word of an integer atomic target, or ``None``.

    An integer atomic through a word view operates on the backing word, so
    the view and the word must agree on signedness for ``min``/``max`` and
    the returned original value. Any other reinterpreted view is refused,
    because its element is not one addressable word.
    """

    target = glsl_word_view_target(generator, expression)
    if target is None:
        binding = generator.glsl_storage_pointer_mutation_binding(
            expression, generator.glsl_storage_pointer_aliases()
        )
        reinterpretation = (binding or {}).get("pointer_reinterpretation")
        if reinterpretation is not None:
            raise PointerReinterpretationError(
                "OpenGL atomics through a reinterpreted storage view need one "
                "32-bit scalar per backing word",
                source_type=reinterpretation["source_layout"].name,
                target_type=reinterpretation["target_layout"].name,
                address_space="storage",
                alignment=reinterpretation["target_layout"].byte_width,
                access="read_write",
                target_backend="opengl",
                reason="atomic-requires-word-view",
                source_location=getattr(expression, "source_location", None),
            )
        return None
    pointer, index, binding = target
    generator.validate_glsl_storage_pointer_binding_access(
        binding, "read_write", expression
    )
    element = glsl_word_view_element(generator, pointer, index, binding)
    physical_kind = layout_scalar_kind(element["source_layout"])
    logical_kind = layout_scalar_kind(element["target_layout"])
    if logical_kind != physical_kind:
        raise PointerReinterpretationError(
            "OpenGL integer atomics through a reinterpreted storage view need "
            f"the backing word's signedness: the view is {logical_kind}, the "
            f"backing word is {physical_kind}",
            source_type=element["source_layout"].name,
            target_type=element["target_layout"].name,
            address_space="storage",
            alignment=4,
            access="read_write",
            target_backend="opengl",
            reason="atomic-signedness-mismatch",
            source_location=getattr(expression, "source_location", None),
        )
    return {
        "kind": logical_kind,
        "lvalue": element["lvalue"],
        "target": element["target"],
        "arguments": [f"uint({element['word']})"],
    }

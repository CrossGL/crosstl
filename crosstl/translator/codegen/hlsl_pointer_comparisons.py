"""Comparisons and truth tests of buffer pointers in HLSL output.

CUDA, HIP and OpenCL C kernels compare their buffer pointers (``p != end``, ``p == base + i``) and test them for
null (``inp ? inp[i] : out[i]``, ``if (!inp)``), which skips an optional input the host passes as a null pointer.
HLSL has no pointers: the generator keeps a pointer into a buffer as the buffer and an element offset
(:meth:`hlsl_resource_pointer_binding`), and an HLSL resource is always bound, so it has no null state and no truth
value. DXC rejects a resource as a condition ("conditional operator only supports condition with scalar, vector, or
matrix types convertable to bool") and as a comparison operand, and the generator used to write both, with the
pointer locals it had replaced by offsets left undeclared.

- Two pointers into the same buffer, with the same element view, compare as their offsets: within one array the
  address order is the element order (C++ [expr.rel], [expr.eq]).
- Every other pointer comparison, and every truth test of a pointer, is refused: pointers into two buffers have no
  common offset, and whether the host bound a buffer is not visible to the shader (comparing with ``nullptr`` or
  ``0``, ``!p``, ``p && q``, ``p ? a : b``, ``if (p)``, ``bool(p)``).
"""

from __future__ import annotations

from ..ast import (
    ArrayAccessNode,
    BinaryOpNode,
    IdentifierNode,
    PointerReinterpretNode,
    TernaryOpNode,
    UnaryOpNode,
    VariableNode,
)

__all__ = [
    "hlsl_pointer_comparison",
    "refuse_hlsl_pointer_truth_value",
]

_COMPARISONS = frozenset({"==", "!=", "<", "<=", ">", ">="})
_LOGICAL = frozenset({"&&", "||"})
# Keys of a binding whose offset does not count elements of its own element type.
_VIEW_KEYS = ("pointer_reinterpretation", "byte_offset")


def _operator(node):
    return str(getattr(node, "operator", getattr(node, "op", "")))


def is_hlsl_pointer_value(codegen, expression) -> bool:
    """Whether ``expression`` is a pointer into a buffer, rather than a value read through one."""
    if isinstance(expression, (IdentifierNode, VariableNode)):
        return codegen.hlsl_resource_pointer_binding(expression) is not None
    if isinstance(expression, BinaryOpNode):
        operator = _operator(expression)
        if operator == "+":
            return is_hlsl_pointer_value(
                codegen, expression.left
            ) or is_hlsl_pointer_value(codegen, expression.right)
        if operator == "-":
            # pointer - pointer is an element count.
            return is_hlsl_pointer_value(
                codegen, expression.left
            ) and not is_hlsl_pointer_value(codegen, expression.right)
        return False
    if isinstance(expression, UnaryOpNode) and _operator(expression) == "&":
        operand = expression.operand
        return isinstance(operand, ArrayAccessNode) and is_hlsl_pointer_value(
            codegen, operand.array
        )
    if isinstance(expression, PointerReinterpretNode):
        return is_hlsl_pointer_value(codegen, expression.expression)
    if isinstance(expression, TernaryOpNode):
        return is_hlsl_pointer_value(
            codegen, expression.true_expr
        ) or is_hlsl_pointer_value(codegen, expression.false_expr)
    return False


def _describe(codegen, expression):
    name = codegen.expression_name(expression) if expression is not None else None
    return f"'{name}'" if name else "a buffer pointer"


def _truth_value_refusal(codegen, expression):
    return codegen.refuse_lowering(
        f"buffer pointer {_describe(codegen, expression)} used as a truth value in "
        f"function '{codegen.current_function_name}': an HLSL resource is always bound "
        "and has no null state",
        "false",
    )


def refuse_hlsl_pointer_truth_value(codegen, expression):
    """Refuse ``expression`` in a boolean context when it is a buffer pointer; ``None`` otherwise."""
    if is_hlsl_pointer_value(codegen, expression):
        return _truth_value_refusal(codegen, expression)
    return None


def _element_type(codegen, binding):
    element_type = binding.get("element_type")
    return None if element_type is None else codegen.map_type(str(element_type))


def _same_view(codegen, left, right) -> bool:
    if left.get("root") is None or left.get("root") != right.get("root"):
        return False
    if left.get("kind") != right.get("kind"):
        return False
    if _element_type(codegen, left) != _element_type(codegen, right):
        return False
    return all(
        str(left.get(key) or "0") in {"0", "0u", "None"}
        and str(right.get(key) or "0") in {"0", "0u", "None"}
        for key in _VIEW_KEYS
    )


def hlsl_pointer_comparison(codegen, expression, mapped_op):
    """Render a comparison or logical operator whose operand is a buffer pointer; ``None`` when none is."""
    if mapped_op not in _COMPARISONS and mapped_op not in _LOGICAL:
        return None
    left, right = expression.left, expression.right
    left_pointer = is_hlsl_pointer_value(codegen, left)
    right_pointer = is_hlsl_pointer_value(codegen, right)
    if not (left_pointer or right_pointer):
        return None
    if mapped_op in _LOGICAL:
        return _truth_value_refusal(codegen, left if left_pointer else right)
    if not (left_pointer and right_pointer):
        pointer = left if left_pointer else right
        return codegen.refuse_lowering(
            f"buffer pointer {_describe(codegen, pointer)} compared with a non-pointer "
            f"value in function '{codegen.current_function_name}': an HLSL resource is "
            "always bound and has no null state or address",
            "false",
        )
    left_binding = codegen.hlsl_resource_pointer_binding(left)
    right_binding = codegen.hlsl_resource_pointer_binding(right)
    if not _same_view(codegen, left_binding, right_binding):
        return codegen.refuse_lowering(
            f"comparison of buffer pointers {_describe(codegen, left)} and "
            f"{_describe(codegen, right)} in function '{codegen.current_function_name}': "
            "HLSL compares element offsets only within one buffer viewed as one element type",
            "false",
        )
    left_offset = str(left_binding.get("offset") or "0")
    right_offset = str(right_binding.get("offset") or "0")
    return f"(({left_offset}) {mapped_op} ({right_offset}))"

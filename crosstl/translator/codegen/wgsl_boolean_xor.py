"""WGSL spelling of an exclusive or on booleans.

CrossGL, like HLSL, Metal and the C family, accepts ``^`` and ``^=`` on ``bool`` and boolean vectors. WGSL defines
``&`` and ``|`` for ``bool`` and ``vecN<bool>``, but ``^`` only for integer scalars and vectors (WGSL, "Bit
Expressions"). For booleans, exclusive or is inequality, which WGSL defines component-wise for ``bool`` and
``vecN<bool>`` (WGSL, "Comparison Expressions"), so ``p ^ q`` becomes ``(p != q)``.

``t ^= v`` becomes ``t = (t != v)``. That reads the target twice, so a target whose spelling calls a function is
bound once through a pointer (``let ptr = &t; *ptr = (*ptr != v);``) where a statement can hold that binding, and
is refused where it cannot, such as a ``for`` update.
"""

from __future__ import annotations

from ..ast import ASTNode, BinaryOpNode, FunctionCallNode, UnaryOpNode

__all__ = [
    "wgsl_boolean_xor",
    "wgsl_boolean_xor_assignment",
    "wgsl_boolean_xor_assignment_statement",
]


_BOOLEAN_RESULT_OPERATORS = frozenset({"==", "!=", "<", ">", "<=", ">=", "&&", "||"})
_BITWISE_OPERATORS = frozenset({"&", "|", "^"})


def _is_boolean_type(codegen, vtype):
    if vtype is None:
        return False
    if codegen.is_bool_type(vtype):
        return True
    shape = codegen.vector_shape(vtype)
    return shape is not None and shape[0] == "bool"


def _is_boolean(codegen, expr):
    # Comparisons and logical operators yield bool; the generator types only arithmetic.
    if isinstance(expr, BinaryOpNode):
        if expr.operator in _BOOLEAN_RESULT_OPERATORS:
            return True
        if expr.operator in _BITWISE_OPERATORS:
            return _boolean_operands(codegen, expr.left, expr.right)
    if isinstance(expr, UnaryOpNode) and expr.operator == "!":
        return True
    return _is_boolean_type(codegen, codegen.expression_type(expr))


def _boolean_operands(codegen, left, right):
    return _is_boolean(codegen, left) and _is_boolean(codegen, right)


def _calls_function(node):
    return isinstance(node, ASTNode) and any(
        isinstance(child, FunctionCallNode) for child in node.walk()
    )


def wgsl_boolean_xor(codegen, left, right, operands):
    """``(l != r)`` for a ``^`` whose operands are both boolean, else ``None``.

    ``operands`` renders the two operands; it is the bitwise operand helper, so a scalar meeting a boolean vector is
    replicated to the vector type first, as WGSL ``!=`` needs operands of one type.
    """
    if not _boolean_operands(codegen, left, right):
        return None
    left_rendered, right_rendered = operands(codegen, left, right)
    return f"({left_rendered} != {right_rendered})"


def _xor_value(codegen, node):
    target_type = codegen.expression_type(node.target)
    return codegen.generate_expression_for_target(node.value, target_type)


def _applies(codegen, node):
    return node.operator == "^=" and _boolean_operands(codegen, node.target, node.value)


def wgsl_boolean_xor_assignment(codegen, node):
    """``t = (t != v)`` for ``t ^= v`` on booleans, else ``None``.

    Raises ``ValueError`` when ``t`` calls a function, because this form evaluates it twice.
    """
    if not _applies(codegen, node):
        return None
    if _calls_function(node.target):
        raise ValueError(
            "WGSL target cannot lower a boolean ^= whose target calls a function "
            "outside a statement: WGSL has no ^ on bool, and t = (t != v) would "
            "evaluate the target twice"
        )
    target = codegen.generate_expression(node.target)
    return f"{target} = ({target} != {_xor_value(codegen, node)})"


def wgsl_boolean_xor_assignment_statement(codegen, node, indent):
    """The statement lines for ``t ^= v`` on booleans, else ``None``."""
    if not _applies(codegen, node):
        return None
    pad = "    " * indent
    if not _calls_function(node.target):
        return f"{pad}{wgsl_boolean_xor_assignment(codegen, node)};"
    target = codegen.generate_expression(node.target)
    value = _xor_value(codegen, node)
    pointer = codegen.declare_synthetic_local_identifier("xor_target")
    return "\n".join(
        [
            f"{pad}let {pointer} = &{target};",
            f"{pad}*{pointer} = (*{pointer} != {value});",
        ]
    )

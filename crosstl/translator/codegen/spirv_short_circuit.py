"""Short-circuit evaluation of ``&&``, ``||`` and ``?:`` in SPIR-V.

GLSL, MSL, C++ and HLSL 2021 evaluate the right operand of a scalar ``&&`` or ``||`` only when
the left operand leaves the result open, and evaluate only the selected arm of a scalar ``?:``
(GLSL 4.60 section 5.9, C++ [expr.log.and], [expr.log.or] and [expr.cond]). ``OpLogicalAnd``,
``OpLogicalOr`` and ``OpSelect`` take already evaluated operands, so lowering through them runs
every operand. That changes the program when an operand writes (an assignment, ``++``, an inlined
call that stores to an ``inout`` argument or to memory) or when the left operand guards it
(``i < n && data[i] > 0`` reads out of bounds; ``d != 0 && n / d > 1`` divides by zero, which
SPIR-V leaves undefined).

Such operands are lowered as glslang lowers them: the guarded operand gets its own block inside an
``OpSelectionMerge`` construct, and the merge block joins the values with ``OpPhi``. Operands that
can neither write nor fault keep the single-instruction form, which is what every existing module
emits for them. Vector operands are component-wise and never short-circuit.

A loop condition that branches can no longer share the loop header with ``OpLoopMerge``, because
the merge instruction must close the header block. :func:`evaluate_loop_condition` moves such a
condition into its own block behind the header, which is again glslang's layout.
"""

import re
from typing import List, Optional, Tuple

from ..ast import (
    ArrayLiteralNode,
    BinaryOpNode,
    CastNode,
    ConstantNode,
    ConstructorNode,
    FunctionCallNode,
    IdentifierNode,
    LiteralNode,
    MemberAccessNode,
    TernaryOpNode,
    UnaryOpNode,
    VariableNode,
)

SHORT_CIRCUIT_OPERATORS = ("&&", "||")

# Pure value conversions; every other call may write through an argument or fault.
_CONVERSION_CALL = re.compile(
    r"(?:bool|int|uint|float|double|half"
    r"|u?int(?:8|16|32|64)(?:_t)?|float(?:16|32|64)(?:_t)?"
    r"|[biud]?vec[234]|f16vec[234]|i64vec[234]|u64vec[234])"
)
# Operators whose evaluation can neither write nor fault on any operand values.
_PURE_BINARY_OPERATORS = frozenset(
    {
        "+",
        "-",
        "*",
        "&",
        "|",
        "^",
        "<<",
        ">>",
        "==",
        "!=",
        "<",
        ">",
        "<=",
        ">=",
        "&&",
        "||",
    }
)
_PURE_UNARY_OPERATORS = frozenset({"-", "+", "!", "~"})


def _call_name(node: FunctionCallNode) -> Optional[str]:
    function = node.function
    if isinstance(function, str):
        return function
    name = getattr(function, "name", None)
    return name if isinstance(name, str) else None


def _nonzero_literal(node) -> bool:
    if isinstance(node, bool):
        return False
    if isinstance(node, (int, float)):
        return node != 0
    if isinstance(node, (LiteralNode, ConstantNode)):
        value = getattr(node, "value", None)
        if isinstance(value, str):
            try:
                value = float(value.rstrip("uUlLfFhH"))
            except ValueError:
                return False
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and value != 0
        )
    return False


def expression_is_trivial(node) -> bool:
    """Return whether evaluating ``node`` can neither write state nor fault.

    Such an operand may run unconditionally, so ``&&``, ``||`` and ``?:`` keep their
    single-instruction lowering for it. Array indexing, calls other than value conversions,
    division by a non-literal and pointer dereferences are not trivial.
    """
    if node is None or isinstance(node, (bool, int, float, str)):
        return True
    if isinstance(node, (LiteralNode, ConstantNode, IdentifierNode, VariableNode)):
        return True
    if isinstance(node, MemberAccessNode):
        return expression_is_trivial(node.object_expr)
    if isinstance(node, UnaryOpNode):
        return node.op in _PURE_UNARY_OPERATORS and expression_is_trivial(node.operand)
    if isinstance(node, BinaryOpNode):
        if node.op in ("/", "%"):
            pure = _nonzero_literal(node.right)
        else:
            pure = node.op in _PURE_BINARY_OPERATORS
        return (
            pure
            and expression_is_trivial(node.left)
            and expression_is_trivial(node.right)
        )
    if isinstance(node, TernaryOpNode):
        return all(
            expression_is_trivial(part)
            for part in (node.condition, node.true_expr, node.false_expr)
        )
    if isinstance(node, CastNode):
        return expression_is_trivial(node.expression)
    if isinstance(node, FunctionCallNode):
        name = _call_name(node)
        return (
            name is not None
            and _CONVERSION_CALL.fullmatch(name) is not None
            and all(expression_is_trivial(argument) for argument in node.arguments)
        )
    if isinstance(node, ConstructorNode):
        return all(expression_is_trivial(argument) for argument in node.arguments)
    if isinstance(node, ArrayLiteralNode):
        return all(expression_is_trivial(element) for element in node.elements)
    return False


def _is_vector_type(codegen, typed) -> bool:
    """Return whether a SPIR-V type or value (anything with ``.type``) is a vector."""
    return (
        typed is not None
        and codegen.vector_component_type_and_count(typed.type.base_type) is not None
    )


def _placeholder_float(codegen):
    return codegen.register_constant(0.0, codegen.register_primitive_type("float"))


def _new_label(codegen):
    from .SPIRV_codegen import SpirvId, SpirvType

    return SpirvId(codegen.get_id(), SpirvType("label"))


def _open_block(codegen, label) -> None:
    codegen.emit(f"%{label.id} = OpLabel")
    codegen.current_label = label.id


def _close_block(codegen, merge_label, value, incoming: List[Tuple]) -> bool:
    """Branch the current block to ``merge_label`` and record ``value`` as its phi input."""
    if codegen.current_block_has_terminator():
        codegen.cooperative_matrix_control_flow_diverged = True
        return False
    incoming.append((value, codegen.current_label))
    codegen.create_branch(merge_label)
    return True


def _emit_phi(codegen, result_type, incoming: List[Tuple]):
    from .SPIRV_codegen import SpirvId

    id_value = codegen.get_id()
    if incoming:
        operands = " ".join(f"%{value.id} %{block}" for value, block in incoming)
        codegen.emit(f"%{id_value} = OpPhi %{result_type.id} {operands}")
    else:
        # Every operand left the function, so the merge block has no predecessor and its
        # value is never observed.
        codegen.emit(f"%{id_value} = OpUndef %{result_type.id}")
    codegen.value_types[id_value] = result_type
    return SpirvId(id_value, result_type.type)


def _evaluate_guarded(codegen, expr):
    codegen.cooperative_matrix_selection_depth += 1
    try:
        return codegen.process_expression(expr)
    finally:
        codegen.cooperative_matrix_selection_depth -= 1


def lower_short_circuit_logical(codegen, expr) -> Optional[object]:
    """Lower ``a && b`` or ``a || b`` so that ``b`` runs only when ``a`` leaves the result open.

    Returns None, before emitting anything, when ``b`` is trivial or no block is open; the caller
    then uses ``OpLogicalAnd``/``OpLogicalOr``.
    """
    if expr.op not in SHORT_CIRCUIT_OPERATORS or codegen.current_label is None:
        return None
    if expression_is_trivial(expr.right):
        return None
    if _is_vector_type(codegen, codegen.infer_expression_result_type(expr.right)):
        return None

    left = codegen.process_expression(expr.left)
    if left is None:
        return _placeholder_float(codegen)
    if _is_vector_type(codegen, left):
        # Component-wise vector logic evaluates both operands in every source language.
        right = codegen.process_expression(expr.right)
        if right is None:
            return _placeholder_float(codegen)
        result_type, left, right = codegen.logical_result_type_and_operands(left, right)
        return codegen.binary_operation(expr.op, result_type, left, right)

    bool_type = codegen.register_primitive_type("bool")
    left = codegen.ensure_bool_value(left)
    incoming = [(left, codegen.current_label)]
    guarded_label = _new_label(codegen)
    merge_label = _new_label(codegen)
    codegen.create_selection_merge(merge_label)
    if expr.op == "&&":
        codegen.create_conditional_branch(left, guarded_label, merge_label)
    else:
        codegen.create_conditional_branch(left, merge_label, guarded_label)

    _open_block(codegen, guarded_label)
    right = _evaluate_guarded(codegen, expr.right)
    if right is None:
        right = codegen.register_constant(False, bool_type)
    elif _is_vector_type(codegen, right):
        codegen.refuse_lowering(
            f"'{expr.op}' between a scalar and a vector operand has no short-circuit form"
        )
        right = codegen.register_constant(False, bool_type)
    else:
        right = codegen.ensure_bool_value(right)
    _close_block(codegen, merge_label, right, incoming)

    _open_block(codegen, merge_label)
    return _emit_phi(codegen, bool_type, incoming)


def _ternary_arms_need_guard(expr) -> bool:
    return not (
        expression_is_trivial(expr.true_expr) and expression_is_trivial(expr.false_expr)
    )


def lower_short_circuit_ternary(codegen, expr) -> Optional[object]:
    """Lower ``c ? x : y`` so that only the selected arm runs.

    Returns None, before emitting anything, when both arms are trivial or no block is open; the
    caller then uses ``OpSelect``. A vector condition selects per component, so both arms run.
    """
    if codegen.current_label is None or not _ternary_arms_need_guard(expr):
        return None
    if _is_vector_type(codegen, codegen.infer_expression_result_type(expr.condition)):
        return None

    condition = codegen.process_expression(expr.condition)
    bool_type = codegen.register_primitive_type("bool")
    if condition is None:
        condition = codegen.register_constant(False, bool_type)
    if _is_vector_type(codegen, condition):
        true_value = codegen.process_expression(expr.true_expr)
        false_value = codegen.process_expression(expr.false_expr)
        if true_value is None or false_value is None:
            return _placeholder_float(codegen)
        result_type, true_value, false_value = codegen.normalize_ternary_values(
            true_value, false_value
        )
        return codegen.select_operation(result_type, condition, true_value, false_value)
    condition = codegen.ensure_bool_value(condition)

    true_label = _new_label(codegen)
    false_label = _new_label(codegen)
    merge_label = _new_label(codegen)
    codegen.create_selection_merge(merge_label)
    codegen.create_conditional_branch(condition, true_label, false_label)

    _open_block(codegen, true_label)
    true_value = _evaluate_guarded(codegen, expr.true_expr)
    if true_value is None:
        true_value = _placeholder_float(codegen)
    true_block = codegen.current_label
    true_open = not codegen.current_block_has_terminator()
    # The true arm's conversion is spliced in here once the false arm fixes the result type.
    true_end = len(codegen.code_lines)
    if true_open:
        codegen.create_branch(merge_label)
    else:
        codegen.cooperative_matrix_control_flow_diverged = True

    _open_block(codegen, false_label)
    false_value = _evaluate_guarded(codegen, expr.false_expr)
    if false_value is None:
        false_value = _placeholder_float(codegen)
    false_open = not codegen.current_block_has_terminator()

    result_type = codegen.ternary_result_type(true_value, false_value)
    incoming: List[Tuple] = []
    if true_open:
        true_value = _convert_at(codegen, true_end, true_value, result_type)
        incoming.append((true_value, true_block))
    if false_open:
        false_value = codegen.convert_value_to_type(false_value, result_type)
        incoming.append((false_value, codegen.current_label))
        codegen.create_branch(merge_label)
    else:
        codegen.cooperative_matrix_control_flow_diverged = True

    for value, _ in incoming:
        value_type = codegen.value_types.get(value.id, value)
        if value_type.type.base_type != result_type.type.base_type:
            codegen.refuse_lowering(
                "conditional-expression arms have no common type: "
                f"'{true_value.type.base_type}' and '{false_value.type.base_type}'"
            )
            break

    _open_block(codegen, merge_label)
    return _emit_phi(codegen, result_type, incoming)


def _convert_at(codegen, position: int, value, result_type):
    """Convert ``value`` with the instructions placed at ``position`` in the function body."""
    saved_lines = codegen.code_lines
    codegen.code_lines = []
    try:
        converted = codegen.convert_value_to_type(value, result_type)
    finally:
        emitted = codegen.code_lines
        codegen.code_lines = saved_lines
    saved_lines[position:position] = emitted
    return converted


def evaluate_loop_condition(
    codegen, condition_expr, header_label, merge_label, continue_label
):
    """Evaluate a ``for``/``while`` condition and close the loop header with ``OpLoopMerge``.

    The condition is evaluated in the header. If its evaluation opened blocks (a short-circuit
    operand or an inlined call with control flow), the header is closed by ``OpLoopMerge`` and a
    branch to a new block that holds the condition, and phi inputs naming the header are renamed
    to that block. Returns the bool condition; the caller emits the conditional branch.
    """
    start = len(codegen.code_lines)
    condition = codegen.process_loop_control_expression(condition_expr)
    if condition is None:
        condition = codegen.register_constant(True, codegen.primitive_types["bool"])
    condition = codegen.ensure_bool_value(condition)

    loop_merge = f"OpLoopMerge %{merge_label.id} %{continue_label.id} None"
    if codegen.current_label == header_label.id:
        codegen.emit(loop_merge)
        return condition

    condition_label = _new_label(codegen)
    header_reference = re.compile(rf"%{header_label.id}(?!\w)")
    lines = codegen.code_lines
    for index in range(start, len(lines)):
        if " = OpPhi " in lines[index]:
            lines[index] = header_reference.sub(f"%{condition_label.id}", lines[index])
    lines[start:start] = [
        loop_merge,
        f"OpBranch %{condition_label.id}",
        f"%{condition_label.id} = OpLabel",
    ]
    return condition

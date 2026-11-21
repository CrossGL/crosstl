"""``OpPtrAccessChain`` and ``OpInBoundsPtrAccessChain`` in the SPIR-V importer.

These differ from ``OpAccessChain`` in their first index, ``Element``. ``Base`` points at one element of an array, and
``Element`` moves that pointer by ``Element`` elements within the same array before the remaining indexes walk into it
as an ``OpAccessChain`` would (SPIR-V specification, ``OpPtrAccessChain``). CrossGL has no pointer to an array
element, so the importer writes the moved pointer as the same array subscripted at the sum of the two indexes:
``a[i]`` moved by ``e`` is ``a[i + e]``. A constant ``Element`` of 0 leaves ``Base`` where it is.

The array subscript is exact only when the pointer type's ``ArrayStride`` is the stride of the array ``Base`` points
into. A module whose arrays of ``Base``'s pointee type state another stride, or whose ``Base`` is not an element of an
array, is refused.
"""

from __future__ import annotations

from .access_chains import chain_base_pointee, walk_indices
from .ray_tracing_import import SpirvInstructionUnsupportedError
from .VulkanAst import ArrayAccessNode, BinaryOpNode, VariableNode

__all__ = ["pointer_access_chain_expression"]


def _decoration_value(decorations, type_id, name):
    for decoration, operands in decorations.get(type_id, []) or []:
        if decoration == name and operands:
            return str(operands[0])
    return None


def _is_literal_zero(expression):
    text = expression.name if isinstance(expression, VariableNode) else expression
    return isinstance(text, (str, int)) and str(text).strip() in {"0", "0u"}


def _check_stride(opcode, pointer_type_id, pointee_type_id, types, decorations):
    """Refuse when an array of the pointee type states a stride other than the pointer type's ``ArrayStride``."""
    pointer_stride = _decoration_value(decorations, pointer_type_id, "ArrayStride")
    if pointer_stride is None:
        return
    for type_id, info in sorted(types.items()):
        if info.get("kind") not in {"array", "runtime_array"}:
            continue
        if info.get("element_type") != pointee_type_id:
            continue
        stride = _decoration_value(decorations, type_id, "ArrayStride")
        if stride is not None and stride != pointer_stride:
            raise SpirvInstructionUnsupportedError(
                opcode,
                f"moves a pointer by ArrayStride {pointer_stride}, but an array of its "
                f"element type has ArrayStride {stride}; CrossGL indexes arrays by their own stride",
            )


def pointer_access_chain_expression(
    parser,
    opcode,
    operands,
    *,
    expressions,
    names,
    decorations,
    member_names,
    types,
    variables_by_id,
    constants,
    expression_type_ids,
):
    """The CrossGL expression of an ``OpPtrAccessChain`` whose operands are ``operands``."""
    result_type_id, base_operand, element_operand = operands[:3]
    remaining = operands[3:]

    def operand_expression(operand):
        return parser.spirv_assembly_operand_expression(
            operand,
            expressions,
            names,
            decorations,
            constants,
            variables_by_id=variables_by_id,
            types=types,
        )

    base = operand_expression(base_operand)
    pointee = chain_base_pointee(
        base_operand, types, variables_by_id, expression_type_ids
    )
    if parser.spirv_integer_constant_operand(element_operand, constants) != 0:
        base_pointer_type = (expression_type_ids or {}).get(base_operand)
        if pointee is not None and base_pointer_type is not None:
            _check_stride(opcode, base_pointer_type, pointee, types, decorations)
        if not isinstance(base, ArrayAccessNode):
            raise SpirvInstructionUnsupportedError(
                opcode,
                "moves a pointer that does not point at an array element; "
                "CrossGL has no pointer arithmetic outside an array",
            )
        element = operand_expression(element_operand)
        index = (
            element
            if _is_literal_zero(base.index)
            else BinaryOpNode(base.index, "+", element)
        )
        base = ArrayAccessNode(base.array, index)
    return walk_indices(
        parser,
        base,
        pointee,
        remaining,
        operand_expression,
        member_names,
        types,
        constants,
    )

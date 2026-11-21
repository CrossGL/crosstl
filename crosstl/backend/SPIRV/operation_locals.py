"""Locals for the intermediate values of an imported SPIR-V instruction that its lowering reads more than once.

Some instructions have no single CrossGL operator and are lowered to an expression that reads an intermediate value
several times, such as ``OpSMod``, whose truncated remainder appears four times in its sign correction
(:mod:`crosstl.backend.SPIRV.integer_signedness_import`). Repeating such a value inline makes the CrossGL grow
exponentially when the instruction's operand is itself an inlined instance of it. :func:`bind_operation_local`
declares the value as a local where the instruction is defined, which is where SPIR-V evaluates it, and returns the
variable that reads it. The function's statement list is still flat here; structured control flow is rebuilt from it
afterwards, and :mod:`crosstl.backend.SPIRV.scoped_declarations` hoists a local whose reads end up outside its
block.
"""

from __future__ import annotations

from .VulkanAst import AssignmentNode, VariableNode

__all__ = ["bind_operation_local"]


def _declared_names(statements):
    for statement in statements:
        target = statement.left if isinstance(statement, AssignmentNode) else statement
        if isinstance(target, VariableNode) and target.vtype:
            yield target.name


def bind_operation_local(
    parser, result_id, role, type_name, expression, statements, names, decorations
):
    """Declare ``expression`` as a local named after ``result_id`` and ``role``; return the variable reading it.

    The name is ``<value>_<role>``, or the first ``<value>_<role>_N`` that no ``OpName`` of the module and no local
    declared so far in the function uses.
    """
    base = parser.spirv_assembly_value_name(
        result_id, names, decorations, prefix="value"
    )
    taken = set(names.values())
    taken.update(_declared_names(statements))
    name = f"{base}_{role}"
    suffix = 1
    while name in taken:
        name = f"{base}_{role}_{suffix}"
        suffix += 1
    statements.append(AssignmentNode(VariableNode(type_name, name), expression))
    return VariableNode("", name)

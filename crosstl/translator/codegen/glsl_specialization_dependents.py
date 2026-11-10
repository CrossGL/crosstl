"""Order and qualify the GLSL constants that read specialization constants.

GLSL output declares the global specialization constants after the other leading constants, so a constant that
reads one, directly or through another constant, has to follow them.

When the shader is compiled to SPIR-V, such a constant stays a constant only if the compiler can build it as a
specialization-constant instruction. SPIR-V 1.6 section 3.52.7 limits ``OpSpecConstantOp`` in Shader modules to
integer and Boolean arithmetic, bitwise operations, comparisons, selection, integer width conversions and
composite extraction; it has no floating-point operation. ``glslang`` builds ``OpSpecConstantComposite`` only for
integer and Boolean vectors, and rejects every other constant initializer that reads a specialization constant
("global const initializers must be constant"). Those constants are declared as global variables in the SPIR-V
branch instead; a global initializer runs before the entry point (GLSL 4.60 section 4.3), and the source never
writes a constant, so every read sees the same value.
"""

import re

from ..ast import (
    BinaryOpNode,
    FunctionCallNode,
    IdentifierNode,
    LiteralNode,
    MemberAccessNode,
    SwizzleNode,
    TernaryOpNode,
    UnaryOpNode,
    VariableNode,
)
from .constant_ordering import expression_reference_names

SPECIALIZATION_OPERATION_TYPES = frozenset(
    {"int", "uint", "bool"}
    | {f"{prefix}vec{size}" for prefix in "iub" for size in (2, 3, 4)}
)
SPECIALIZATION_BINARY_OPERATORS = frozenset(
    {
        "+",
        "-",
        "*",
        "/",
        "%",
        "<<",
        ">>",
        "&",
        "|",
        "^",
        "&&",
        "||",
        "==",
        "!=",
        "<",
        ">",
        "<=",
        ">=",
    }
)
SPECIALIZATION_UNARY_OPERATORS = frozenset({"-", "+", "~", "!"})
INTEGER_OR_BOOLEAN_LITERAL = re.compile(
    r"(?:true|false|True|False|0[xX][0-9a-fA-F]+[uU]?|[0-9]+[uU]?)"
)


def constant_source_name(node):
    name = getattr(node, "name", getattr(node, "variable_name", None))
    return name if isinstance(name, str) else None


def constant_initializer(node):
    if hasattr(node, "const_type"):
        return getattr(node, "value", None)
    return getattr(node, "initial_value", getattr(node, "value", None))


def partition_constants_by_specialization_dependency(constants, specialization_names):
    """Split constants into those declared before and after the specialization constants.

    A constant moves after them when its initializer reads a specialization constant or a constant that moved.
    Specialization constants themselves keep their place, and both groups keep source order.
    """
    after = set(specialization_names)
    leading = []
    dependent = []
    for constant in constants or []:
        name = constant_source_name(constant)
        if name not in specialization_names and (
            expression_reference_names(constant_initializer(constant)) & after
        ):
            dependent.append(constant)
            if name is not None:
                after.add(name)
        else:
            leading.append(constant)
    return leading, dependent


def runtime_specialization_dependents(
    constants, specialization_constants, mapped_type, mapped_type_name
):
    """Return the ids of constants a SPIR-V compile cannot build as specialization-constant instructions.

    ``constants`` are all non-specialization constants in source order, ``specialization_constants`` the
    specialization constant declarations, ``mapped_type(node)`` gives the GLSL type of a declaration and
    ``mapped_type_name(name)`` the GLSL type a constructor name builds.
    """
    specialization_names = {
        name
        for node in specialization_constants
        if (name := constant_source_name(node)) is not None
    }
    if not specialization_names:
        return set()
    operation_names = {
        name
        for node in specialization_constants
        if (name := constant_source_name(node)) is not None
        and mapped_type(node) in SPECIALIZATION_OPERATION_TYPES
    }
    dependent_names = set(specialization_names)
    runtime_ids = set()
    for constant in constants or []:
        name = constant_source_name(constant)
        initializer = constant_initializer(constant)
        integral = mapped_type(constant) in SPECIALIZATION_OPERATION_TYPES
        if not (expression_reference_names(initializer) & dependent_names):
            if integral and name is not None:
                operation_names.add(name)
            continue
        if name is not None:
            dependent_names.add(name)
        if integral and is_specialization_operation(
            initializer, operation_names, mapped_type_name
        ):
            if name is not None:
                operation_names.add(name)
        else:
            runtime_ids.add(id(constant))
    return runtime_ids


def is_specialization_operation(expression, operation_names, mapped_type_name):
    """Return whether an initializer uses only integer and Boolean specialization-constant operations."""
    if isinstance(expression, LiteralNode):
        value = expression.value
        if isinstance(value, bool) or isinstance(value, int):
            return True
        return isinstance(value, str) and bool(
            INTEGER_OR_BOOLEAN_LITERAL.fullmatch(value)
        )
    if isinstance(expression, (IdentifierNode, VariableNode)):
        return getattr(expression, "name", None) in operation_names
    if isinstance(expression, BinaryOpNode):
        return (
            expression.operator in SPECIALIZATION_BINARY_OPERATORS
            and is_specialization_operation(
                expression.left, operation_names, mapped_type_name
            )
            and is_specialization_operation(
                expression.right, operation_names, mapped_type_name
            )
        )
    if isinstance(expression, UnaryOpNode):
        return (
            not expression.is_postfix
            and expression.operator in SPECIALIZATION_UNARY_OPERATORS
            and is_specialization_operation(
                expression.operand, operation_names, mapped_type_name
            )
        )
    if isinstance(expression, TernaryOpNode):
        return all(
            is_specialization_operation(part, operation_names, mapped_type_name)
            for part in (
                expression.condition,
                expression.true_expr,
                expression.false_expr,
            )
        )
    if isinstance(expression, FunctionCallNode):
        function = expression.function
        function_name = getattr(function, "name", function)
        if not isinstance(function_name, str):
            return False
        return mapped_type_name(
            function_name
        ) in SPECIALIZATION_OPERATION_TYPES and all(
            is_specialization_operation(argument, operation_names, mapped_type_name)
            for argument in expression.arguments or []
        )
    if isinstance(expression, SwizzleNode):
        return is_specialization_operation(
            expression.vector_expr, operation_names, mapped_type_name
        )
    if isinstance(expression, MemberAccessNode):
        return is_specialization_operation(
            expression.object_expr, operation_names, mapped_type_name
        )
    return False

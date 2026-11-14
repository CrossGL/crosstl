"""C integer promotion for Mojo's narrow integers.

C17 section 6.3.1.1 converts an operand whose integer type is narrower than
``int`` to ``int`` before an arithmetic, bitwise, shift or relational operator
applies, so ``char`` and ``short`` arithmetic cannot wrap at eight or sixteen
bits. Mojo's ``Int8``, ``Int16``, ``UInt8`` and ``UInt16`` arithmetic wraps at
its own width, so such an operand is widened to ``Int32`` first.
"""

MOJO_NARROW_INTEGER_TYPES = frozenset({"Int8", "Int16", "UInt8", "UInt16"})

# The binary operators whose operands undergo the integer promotions (C17
# sections 6.5.5 to 6.5.12).
C_PROMOTING_BINARY_OPERATORS = frozenset(
    {
        "+",
        "-",
        "*",
        "/",
        "%",
        "&",
        "|",
        "^",
        "<<",
        ">>",
        "<",
        "<=",
        ">",
        ">=",
        "==",
        "!=",
    }
)


def mojo_type_is_narrow_integer(codegen, type_name):
    if type_name is None:
        return False
    try:
        return codegen.map_type(type_name) in MOJO_NARROW_INTEGER_TYPES
    except (TypeError, ValueError):
        return False


def promoted_operand(codegen, operand, code):
    """``code`` widened to ``Int32`` when ``operand`` has a narrow integer type."""
    if mojo_type_is_narrow_integer(codegen, codegen.expression_result_type(operand)):
        return f"Int32({code})"
    return code


def promoted_type(codegen, op, type_name):
    """The CrossGL type of an operand of ``op`` after the integer promotions."""
    if op in C_PROMOTING_BINARY_OPERATORS and mojo_type_is_narrow_integer(
        codegen, type_name
    ):
        return "int"
    return type_name

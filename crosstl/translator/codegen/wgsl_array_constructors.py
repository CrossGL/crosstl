"""WGSL value constructors for CrossGL array constructors.

CrossGL writes an array value as a constructor call, ``float[3](a, b, c)`` or ``float[](a, b, c)``, wherever an
expression may stand (GLSL 4.60 section 5.4.4, "Array Constructors"). WGSL has the same construct as a value
constructor, ``array<f32, 3>(a, b, c)``, which is also an expression (WGSL, "Array Value Constructors"), so the
constructor is written in place rather than hoisted into a declaration.

The element type and count are always spelled out. An inferred ``array(a, b, c)`` would take its element type from
the arguments, so ``uint[2](1, 2)`` would become an ``array<i32, 2>``. An unsized constructor, ``float[](...)``,
has as many elements as arguments (GLSL 4.60 section 4.1.9). A sized constructor whose size is an integer literal
must have exactly that many arguments, as in GLSL and WGSL, and a constructor of an array of arrays, which the
CrossGL call spelling does not describe, is refused rather than written verbatim.
"""

from __future__ import annotations

import re

from ..ast import IdentifierNode, MatrixType, NamedType, PrimitiveType, VectorType

__all__ = ["wgsl_array_constructor"]

_ARRAY_CONSTRUCTOR = re.compile(r"\s*([A-Za-z_]\w*)\s*\[\s*(\w*)\s*\]\s*")


def _element_type(codegen, spelling):
    """The type ``spelling`` names when it is a value type the shader can construct, else ``None``.

    The CrossGL parser reads a call through a subscripted name, such as ``table[0](x)``, with the same callee
    spelling as an array constructor, so the part before the subscript must name a type.
    """
    from ..lexer import Lexer
    from ..parser import Parser

    try:
        vtype = Parser(Lexer(f"{spelling} x;").tokens).parse_type()
    except Exception:
        return None
    if isinstance(vtype, (VectorType, MatrixType)):
        return vtype
    if isinstance(vtype, PrimitiveType) and vtype.name != "void":
        return vtype
    if isinstance(vtype, NamedType) and vtype.name in codegen._structs_by_name:
        return vtype
    return None


def wgsl_array_constructor(codegen, node):
    """WGSL ``array<T, N>(...)`` for a CrossGL array constructor call, or ``None`` for any other call."""
    callee = node.function
    if not isinstance(callee, IdentifierNode) or "[" not in str(callee.name):
        return None
    function_name = str(callee.name).strip()
    element_type = _element_type(codegen, function_name.split("[", 1)[0])
    if element_type is None:
        return None
    match = _ARRAY_CONSTRUCTOR.fullmatch(function_name)
    if match is None:
        raise ValueError(
            f"WGSL target does not support array constructor {function_name}"
        )
    size = match.group(2)
    arguments = list(node.arguments or [])
    if not arguments:
        raise ValueError(
            f"WGSL target does not support the empty array constructor {function_name}"
        )
    if not size:
        size = str(len(arguments))
    elif size.isdigit():
        if int(size) != len(arguments):
            raise ValueError(
                f"WGSL array constructor {function_name} has {len(arguments)} "
                f"arguments; it needs exactly {int(size)}"
            )
        size = str(int(size))
    else:
        size = codegen.identifier_name(size)
    element = codegen.type_name_string(element_type)
    values = ", ".join(codegen.generate_expression(argument) for argument in arguments)
    return f"array<{element}, {size}>({values})"

"""WGSL value constructors for CrossGL array constructors.

CrossGL writes an array value as a constructor call, ``float[3](a, b, c)`` or ``float[](a, b, c)``, wherever an
expression may stand (GLSL 4.60 section 5.4.4, "Array Constructors"). WGSL has the same construct as a value
constructor, ``array<f32, 3>(a, b, c)``, which is also an expression (WGSL, "Array Value Constructors"), so the
constructor is written in place rather than hoisted into a declaration.

The element type and count are always spelled out. An inferred ``array(a, b, c)`` would take its element type from
the arguments, so ``uint[2](1, 2)`` would become an ``array<i32, 2>``. An unsized constructor, ``float[](...)``,
has as many elements as arguments (GLSL 4.60 section 4.1.9). A sized constructor whose size is an integer literal
may give fewer arguments than that size: the elements it leaves out are the zero value ``T()`` (WGSL, "Zero Value
Expressions"), as GLSL output writes them and as a short declared initializer is filled. One with more arguments than
its size and an empty one are refused with :class:`WGSLAggregateInitializerError`.

A constructor of an array of arrays, ``float[2][3](float[3](...), float[3](...))``, takes one argument per element of
its first dimension, each an array of the remaining dimensions (GLSL 4.60 section 4.1.9, "Arrays"), so it is written
as ``array<array<f32, 3>, 2>(...)``. The CrossGL parser reads its callee as the first dimension's spelling
subscripted by the others (``ArrayAccessNode(ArrayAccessNode(float, 2), 3)``). Only the first dimension may be left
unsized: WGSL has no runtime-sized array element of an array value.
"""

from __future__ import annotations

import re

from ..ast import (
    ArrayAccessNode,
    ArrayType,
    IdentifierNode,
    LiteralNode,
    MatrixType,
    NamedType,
    PrimitiveType,
    VectorType,
)
from ..refusals import WGSLRefusal

__all__ = [
    "wgsl_array_constructor",
    "wgsl_array_constructor_type",
    "wgsl_array_element",
]

_ARRAY_CONSTRUCTOR = re.compile(r"\s*([A-Za-z_]\w*)\s*((?:\[\s*\w*\s*\]\s*)+)")
_DIMENSION = re.compile(r"\[\s*(\w*)\s*\]")


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


def _subscript_spelling(index):
    """The size spelling of one subscript of an array constructor callee, or ``None`` when it is not a size."""
    if isinstance(index, LiteralNode) and isinstance(index.value, int):
        return str(index.value)
    if isinstance(index, IdentifierNode) and re.fullmatch(
        r"[A-Za-z_]\w*", str(index.name)
    ):
        return str(index.name)
    return None


def _callee_spelling(callee):
    """The constructor spelling, such as ``float[2][3]``, of an array constructor callee, or ``None``."""
    subscripts = []
    while isinstance(callee, ArrayAccessNode):
        size = _subscript_spelling(callee.index)
        if size is None:
            return None
        subscripts.append(f"[{size}]")
        callee = callee.array
    if not isinstance(callee, IdentifierNode):
        return None
    spelling = str(callee.name).strip() + "".join(reversed(subscripts))
    return spelling if "[" in spelling else None


def _array_constructor_callee(codegen, node):
    """``(spelling, element type)`` of a CrossGL array constructor call, or ``None`` for any other call."""
    function_name = _callee_spelling(node.function)
    if function_name is None:
        return None
    element_type = _element_type(codegen, function_name.split("[", 1)[0])
    if element_type is None:
        return None
    return function_name, element_type


def _dimensions(function_name):
    """The size spellings of ``function_name``'s dimensions, first dimension first, or ``None``."""
    match = _ARRAY_CONSTRUCTOR.fullmatch(function_name)
    if match is None:
        return None
    return [size.strip() for size in _DIMENSION.findall(match.group(2))]


def _sized_type(element_type, size):
    if size.isdigit():
        return ArrayType(element_type, int(size))
    return ArrayType(element_type, IdentifierNode(size))


def _element_array_type(function_name, element_type, sizes):
    """The type of one argument of a constructor of ``sizes``: ``element_type`` or an array of the later sizes."""
    if any(not size for size in sizes[1:]):
        raise WGSLRefusal(
            f"WGSL target does not support array constructor {function_name}: only its first dimension may "
            "be unsized"
        )
    for size in reversed(sizes[1:]):
        element_type = _sized_type(element_type, size)
    return element_type


def wgsl_array_constructor_type(codegen, node):
    """The ``ArrayType`` a CrossGL array constructor call constructs, or ``None`` for any other call."""
    callee = _array_constructor_callee(codegen, node)
    if callee is None:
        return None
    function_name, element_type = callee
    sizes = _dimensions(function_name)
    if sizes is None or any(not size for size in sizes[1:]):
        return None
    element_type = _element_array_type(function_name, element_type, sizes)
    if not sizes[0]:
        return ArrayType(element_type, len(node.arguments or []))
    return _sized_type(element_type, sizes[0])


def _aggregate_error(message, **details):
    """The typed refusal, with a project diagnostic code, for an array constructor WGSL cannot write."""
    # Imported here because wgsl_codegen imports this module.
    from .wgsl_codegen import WGSLAggregateInitializerError

    return WGSLAggregateInitializerError(message, **details)


def wgsl_array_constructor(codegen, node):
    """WGSL ``array<T, N>(...)`` for a CrossGL array constructor call, or ``None`` for any other call."""
    callee = _array_constructor_callee(codegen, node)
    if callee is None:
        return None
    function_name, element_type = callee
    sizes = _dimensions(function_name)
    if sizes is None:
        raise WGSLRefusal(
            f"WGSL target does not support array constructor {function_name}"
        )
    size = sizes[0]
    element_type = _element_array_type(function_name, element_type, sizes)
    arguments = list(node.arguments or [])
    if not arguments:
        raise _aggregate_error(
            f"WGSL target does not support the empty array constructor {function_name}"
        )
    element = codegen.type_name_string(element_type)
    padding = 0
    if not size:
        size = str(len(arguments))
    elif size.isdigit():
        expected = int(size)
        if len(arguments) > expected:
            array_type = f"array<{element}, {expected}>"
            raise _aggregate_error(
                f"WGSL array constructor {function_name} for '{array_type}' "
                f"expects at most {expected} elements, got {len(arguments)}",
                expected_type=array_type,
                element_count=len(arguments),
                expected_element_count=expected,
            )
        padding = expected - len(arguments)
        size = str(expected)
    else:
        size = codegen.identifier_name(size)
    values = [
        wgsl_array_element(codegen, argument, element_type) for argument in arguments
    ]
    values += [f"{element}()"] * padding
    return f"array<{element}, {size}>({', '.join(values)})"


def wgsl_array_element(codegen, value, element_type):
    """WGSL for ``value`` as one element of an array constructor or initializer of ``element_type`` elements.

    A scalar element of a matrix array, as in ``mat3[2](0.0)``, stands for the matrix ``mat3(0.0)``, with the
    scalar on the diagonal and zero elsewhere (GLSL 4.60 section 5.4.2), as Metal output writes it. WGSL has no
    conversion from a scalar to a matrix, so the matrix is written out. Only array elements are read this way: a
    scalar stored to a matrix, as in ``m += 1.0`` or HLSL's ``float3x3 m = 1.0;``, applies to every component.
    """
    diagonal = codegen.generate_scalar_matrix_constructor(element_type, value)
    if diagonal is not None:
        return diagonal
    return codegen.generate_expression_for_target(value, element_type)

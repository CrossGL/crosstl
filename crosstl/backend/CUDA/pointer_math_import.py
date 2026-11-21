"""CUDA and HIP math functions that write results through pointers, imported with CrossGL ``out`` arguments.

CUDA and HIP declare the C math functions that return more than one part with pointer parameters (CUDA Math API,
single and double precision mathematical functions; HIP math API): ``sincosf(float x, float* s, float* c)``,
``sincospif``, ``frexpf(float x, int* e)``, ``modff(float x, float* i)``, ``remquof(float x, float y, int* q)``,
the ``__sincosf`` intrinsic and the ``double`` forms without the ``f``. CrossGL spells the same functions with an
``out`` argument for each part, ``sincosf(x, s, c)`` (see ``crosstl/translator/codegen/c_pointer_math.py``), so the
importers pass the value each pointer addresses: ``&s`` becomes ``s`` and any other pointer ``p`` becomes ``*p``.
The call written verbatim passed ``&s`` where CrossGL expects an assignable value, and the CUDA and HIP generators
refused the round trip.

``frexpf`` and ``modff`` become CrossGL's own ``frexp`` and ``modf``, which every target declares and which are exact,
so the float spelling would only narrow the targets. The other functions keep their CUDA names: CrossGL's ``sincos``
has the precision of the target, while ``sincosf`` is within 2 ulp over the whole binary32 range, and ``sincospi``
and ``remquo`` exist only in the C family. A ``sincosf`` or ``sincos`` statement whose operand has no side effects is
written as the two assignments ``s = __cgl_libm_sin(x); c = __cgl_libm_cos(x);``, which keep that bound on every
target (:data:`SINE_COSINE_SPLITS`).
"""

from __future__ import annotations

# Function name -> (argument count, positions of the pointer arguments).
POINTER_OUTPUT_FUNCTIONS = {
    "sincos": (3, (1, 2)),
    "sincosf": (3, (1, 2)),
    "__sincosf": (3, (1, 2)),
    "sincospi": (3, (1, 2)),
    "sincospif": (3, (1, 2)),
    "frexp": (2, (1,)),
    "frexpf": (2, (1,)),
    "modf": (2, (1,)),
    "modff": (2, (1,)),
    "remquo": (3, (2,)),
    "remquof": (3, (2,)),
}
# The CrossGL name of each function: frexpf and modff become frexp and modf, the others keep their CUDA names.
CROSSGL_NAMES = {
    "sincos": "sincos",
    "sincosf": "sincosf",
    "__sincosf": "__sincosf",
    "sincospi": "sincospi",
    "sincospif": "sincospif",
    "frexp": "frexp",
    "frexpf": "frexp",
    "modf": "modf",
    "modff": "modf",
    "remquo": "remquo",
    "remquof": "remquof",
}
# A sine-and-cosine statement -> the CrossGL functions its two assignments call. ``__sincosf`` is the fast intrinsic,
# so it takes the target's own precision, as ``__sinf`` and ``__cosf`` do.
SINE_COSINE_SPLITS = {
    "sincosf": ("__cgl_libm_sin", "__cgl_libm_cos"),
    "sincos": ("__cgl_libm_sin", "__cgl_libm_cos"),
    "__sincosf": ("sin", "cos"),
}
_PURE_UNARY_OPERATORS = frozenset({"-", "+", "!", "~"})
_ASSIGNING_OPERATORS = frozenset(
    {"=", "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=", "<<=", ">>=", ","}
)


def _bare_name(name):
    return name.removeprefix("::") if isinstance(name, str) else None


def pointer_output_positions(name, arguments):
    """The positions of the pointer arguments of a call to one of :data:`POINTER_OUTPUT_FUNCTIONS`, or ``None``."""
    entry = POINTER_OUTPUT_FUNCTIONS.get(_bare_name(name))
    if entry is None or len(arguments) != entry[0]:
        return None
    return entry[1]


def pointed_value(argument, unary_node_type):
    """The value a pointer argument addresses: the operand of ``&x``, or ``*p`` built with ``unary_node_type``.

    A cast of the pointer, as in ``(float*)&s``, addresses the same value.
    """
    kind = type(argument).__name__
    if kind == "CastNode":
        return pointed_value(argument.expression, unary_node_type)
    if kind == "UnaryOpNode" and getattr(argument, "op", None) == "&":
        return argument.operand
    return unary_node_type("*", argument)


def format_pointer_output_call(converter, name, arguments, unary_node_type):
    """The CrossGL call for a C math call that writes parts through pointers, or ``None`` for any other call.

    ``converter`` renders each argument; ``unary_node_type`` builds the ``*p`` dereference of a pointer that is not
    written ``&x``.
    """
    positions = pointer_output_positions(name, arguments)
    if positions is None:
        return None
    rendered = [
        converter.visit(
            pointed_value(argument, unary_node_type) if index in positions else argument
        )
        for index, argument in enumerate(arguments)
    ]
    bare = _bare_name(name)
    return f"{CROSSGL_NAMES[bare]}({', '.join(rendered)})"


def is_pure_expression(node):
    """Whether evaluating ``node`` twice reads the same values and changes nothing: names, literals, element and member
    reads, casts, and arithmetic over them. Calls, assignments and increments are not.
    """
    if isinstance(node, (str, int, float)):
        return True
    kind = type(node).__name__
    if kind == "UnaryOpNode":
        return node.op in _PURE_UNARY_OPERATORS and is_pure_expression(node.operand)
    if kind == "BinaryOpNode":
        return (
            node.op not in _ASSIGNING_OPERATORS
            and is_pure_expression(node.left)
            and is_pure_expression(node.right)
        )
    if kind == "ArrayAccessNode":
        return is_pure_expression(node.array) and is_pure_expression(node.index)
    if kind == "MemberAccessNode":
        return is_pure_expression(node.object)
    if kind == "CastNode":
        return is_pure_expression(node.expression)
    if kind == "TernaryOpNode":
        return all(
            is_pure_expression(getattr(node, part))
            for part in ("condition", "true_expr", "false_expr")
        )
    return False


def sine_cosine_statements(converter, name, arguments, unary_node_type):
    """The two CrossGL assignments a ``sincosf(x, s, c)`` statement becomes, or ``None`` when it stays one call.

    The operand is written twice, so only an operand :func:`is_pure_expression` accepts is split.
    """
    split = SINE_COSINE_SPLITS.get(_bare_name(name))
    if split is None or len(arguments) != 3 or not is_pure_expression(arguments[0]):
        return None
    value = converter.visit(arguments[0])
    sine = converter.visit(pointed_value(arguments[1], unary_node_type))
    cosine = converter.visit(pointed_value(arguments[2], unary_node_type))
    return [f"{sine} = {split[0]}({value});", f"{cosine} = {split[1]}({value});"]

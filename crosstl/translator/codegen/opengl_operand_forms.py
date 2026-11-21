"""GLSL forms of CrossGL operands that GLSL types more strictly, and the built-in calls GLSL declares no overload for.

- **Case labels.** GLSL 4.60 section 6.2 requires the case labels and the switch selector to have the same type, and
  GLSL 3.30 and GLSL ES have no implicit conversion that would give it to them. CrossGL follows C (C11 6.8.4.2): each
  label is converted to the type of the selector. A label of a 32-bit ``int`` or ``uint`` selector is therefore
  written in the selector's type: an integer literal as its value modulo 2^32 (``case 4294967295:`` of a ``uint``
  selector is ``case 4294967295u:``, of an ``int`` selector ``case -1:``), and any other label converted with a
  constructor. The literal was written as an ``int64_t`` literal before, which GLSL does not accept as a label.
- **Bit-field offsets.** GLSL 4.60 section 8.8 declares the ``offset`` and ``bits`` operands of ``bitfieldExtract``
  and ``bitfieldInsert`` as ``int``. CrossGL accepts any integer there: Metal's ``extract_bits`` and ``insert_bits``
  take ``uint`` (MSL section 6.13) and SPIR-V's ``OpBitField*`` instructions take any integer type. Another integer
  type is converted with ``int()``. Both functions are undefined unless ``0 <= offset``, ``0 <= bits`` and
  ``offset + bits`` is at most the bit width (``bitfield_builtins.py``), so every defined operand is below 2^31 and the
  conversion keeps it.
- **Cross products.** GLSL declares ``cross`` for ``vec3`` and ``dvec3`` only (section 8.5). Other operands are
  refused.
- **Double-precision transcendentals.** GLSL declares the angle, trigonometric and exponential functions of sections
  8.1 and 8.2 for ``genFType`` only; ``sqrt`` and ``inversesqrt`` are the only functions of section 8.2 with a
  ``genDType`` overload. A 64-bit floating-point operand of one of them has no GLSL form, and converting it to
  ``float`` would change its range and precision, so the call is refused.
"""

from __future__ import annotations

from ..diagnostics import CODE_PREFIX
from ..refusals import OpenGLRefusal
from ..unhandled import UNHANDLED

__all__ = [
    "FLOAT_ONLY_FUNCTIONS",
    "OpenGLBuiltinOperandError",
    "glsl_case_label",
    "glsl_operand_form_call",
]

# CrossGL functions whose GLSL built-in is declared for genFType only (GLSL 4.60 sections 8.1 and 8.2). ``atan2`` is
# written as ``atan`` and ``log10`` through ``log``.
FLOAT_ONLY_FUNCTIONS = frozenset(
    {
        "radians",
        "degrees",
        "sin",
        "cos",
        "tan",
        "asin",
        "acos",
        "atan",
        "atan2",
        "sinh",
        "cosh",
        "tanh",
        "asinh",
        "acosh",
        "atanh",
        "pow",
        "exp",
        "log",
        "exp2",
        "log2",
        "log10",
    }
)
# Function -> positions of the operands GLSL declares as ``int``.
_BITFIELD_INT_OPERANDS = {"bitfieldExtract": (1, 2), "bitfieldInsert": (2, 3)}
_LABEL_TYPES = frozenset({"int", "uint"})


class OpenGLBuiltinOperandError(OpenGLRefusal):
    """A built-in call whose operand types have no GLSL overload."""

    code = f"{CODE_PREFIX}opengl-builtin-operand-unsupported"
    missing_capabilities = ("opengl.builtin-operand-types",)

    def __init__(self, message, *, operation, operand_types, reason, span=None):
        self.operation = operation
        self.operand_types = tuple(operand_types)
        self.diagnostic_reason = reason
        super().__init__(message, span=span)


def _integer_literal(node):
    """The value of an integer literal, optionally negated, or ``None``."""
    kind = type(node).__name__
    if kind == "LiteralNode":
        value = getattr(node, "value", None)
        if isinstance(value, bool) or not isinstance(value, int):
            return None
        return value
    if kind == "UnaryOpNode" and getattr(node, "op", None) in {"-", "+"}:
        inner = _integer_literal(getattr(node, "operand", None))
        if inner is None:
            return None
        return -inner if node.op == "-" else inner
    return None


def _type_info(generator, node):
    """``glsl_value_type_info`` of the CrossGL type of ``node``, or ``None`` when it is unknown."""
    source = generator.glsl_source_expression_type(node)
    return generator.glsl_value_type_info(source) if source is not None else None


def glsl_case_label(generator, value, selector):
    """The GLSL text of the case label ``value`` of a switch on ``selector``, in the type of the selector."""
    selector_info = _type_info(generator, selector)
    if (
        selector_info is None
        or selector_info["width"] != 1
        or selector_info["bits"] != 32
        or selector_info["mapped"] not in _LABEL_TYPES
    ):
        return generator.generate_expression(value)
    selector_type = selector_info["mapped"]
    literal = _integer_literal(value)
    if literal is not None:
        word = literal % (1 << 32)
        if selector_type == "uint":
            return f"{word}u"
        if word == 1 << 31:
            # -2147483648 is the negation of a literal GLSL cannot write as an int.
            return "(-2147483647 - 1)"
        return str(word - (1 << 32) if word > (1 << 31) else word)
    rendered = generator.generate_expression(value)
    label_info = _type_info(generator, value)
    if label_info is None or label_info["mapped"] == selector_type:
        return rendered
    if label_info["family"] in {"int", "uint"} and label_info["width"] == 1:
        return f"{selector_type}({rendered})"
    return rendered


def _operand_types(generator, args):
    infos = [_type_info(generator, arg) for arg in args]
    names = tuple(info["mapped"] if info else "unknown" for info in infos)
    return infos, names


def _refuse(expr, message, *, operation, operand_types, reason):
    raise OpenGLBuiltinOperandError(
        message,
        operation=operation,
        operand_types=operand_types,
        reason=reason,
        span=getattr(expr, "source_location", None),
    )


def glsl_operand_form_call(generator, expr, func_name):
    """Render a built-in call whose operands need a GLSL form, refuse one GLSL cannot express, or ``UNHANDLED``."""
    if not isinstance(func_name, str) or func_name in generator.function_return_types:
        return UNHANDLED
    args = list(getattr(expr, "args", None) or [])
    if func_name in FLOAT_ONLY_FUNCTIONS:
        infos, names = _operand_types(generator, args)
        if any(
            info and info["family"] == "float" and info["bits"] == 64 for info in infos
        ):
            _refuse(
                expr,
                f"OpenGL cannot translate '{func_name}' of {', '.join(names)}: GLSL declares it for 32-bit "
                "floating-point operands only (GLSL 4.60 sections 8.1 and 8.2), and converting a 64-bit operand "
                "would change its range and precision",
                operation=func_name,
                operand_types=names,
                reason="double-precision-operand",
            )
        return UNHANDLED
    if func_name == "cross" and len(args) == 2:
        infos, names = _operand_types(generator, args)
        if all(info is not None for info in infos) and any(
            info["family"] != "float"
            or info["width"] != 3
            or info["bits"] not in {32, 64}
            for info in infos
        ):
            _refuse(
                expr,
                f"OpenGL cannot translate 'cross' of {', '.join(names)}: GLSL declares the cross product for vec3 "
                "and dvec3 operands only (GLSL 4.60 section 8.5)",
                operation="cross",
                operand_types=names,
                reason="non-3-component-operand",
            )
        return UNHANDLED
    positions = _BITFIELD_INT_OPERANDS.get(func_name)
    if positions is None or len(args) != positions[-1] + 1:
        return UNHANDLED
    converted = []
    for index in positions:
        info = _type_info(generator, args[index])
        if (
            info is not None
            and info["width"] == 1
            and info["family"] in {"int", "uint"}
            and (info["family"], info["bits"]) != ("int", 32)
        ):
            converted.append(index)
    if not converted:
        return UNHANDLED
    rendered = []
    for index, arg in enumerate(args):
        text = generator.generate_expression(arg)
        rendered.append(f"int({text})" if index in converted else text)
    return f"{func_name}({', '.join(rendered)})"

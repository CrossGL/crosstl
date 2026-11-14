"""Metal lowering of CrossGL debug printing.

CrossGL ``printf(format, values...)`` follows the Vulkan debug printf format (GL_EXT_debug_printf,
NonSemantic.DebugPrintf): C conversions, plus ``%v<N><conversion>`` that prints the ``N`` components of a vector
separated by ``", "``. Metal 3.2 prints from a shader with ``os_log_default.log(format, values...)`` from
``<metal_logging>`` (Metal Shading Language Specification 3.2, section 6.20 "Logging Functions"); the host enables
the log with ``MTLCompileOptions.enableLogging``, much as Vulkan prints only under the validation layer.

Metal's format is OpenCL C's, whose vector conversion separates the components with ``","`` and needs an ``hl``
length for 32-bit components, so a vector conversion is written out component by component instead: ``%v3f``
becomes ``%f, %f, %f`` over ``t.x``, ``t.y`` and ``t.z``, which prints the same text. A call with a vector value
evaluates every value once, left to right, into ``auto`` locals first.

The format must be a string literal, each conversion must have a value and every value a conversion; a ``*``
width or precision, a ``%s`` and a ``%n`` are refused, as Metal's compiler rejects them.
"""

from __future__ import annotations

import re

from ..diagnostics import CODE_PREFIX, TranslationDiagnostic

__all__ = [
    "METAL_DEBUG_PRINTF_FUNCTIONS",
    "MetalDebugPrintfError",
    "metal_debug_printf_statement",
    "uses_debug_printf",
]

METAL_DEBUG_PRINTF_FUNCTIONS = frozenset({"printf", "debugPrintfEXT"})

_CONVERSION = re.compile(
    r"%(?P<flags>[-+ #0]*)(?P<width>\*|\d*)(?P<precision>\.(?:\*|\d*))?"
    r"(?P<vector>v[234])?(?P<length>hh|hl|h|ll|l)?(?P<conversion>[diouxXfFeEgGaAc%]|.?)",
    re.DOTALL,
)
_COMPONENTS = "xyzw"


class MetalDebugPrintfError(TranslationDiagnostic):
    """A debug print whose format or values Metal's ``os_log`` cannot print the same way."""

    def __init__(self, detail, *, source_location=None):
        self.source_location = source_location
        self.missing_capabilities = ("metal.debug-printf",)
        super().__init__(
            f"debug printf: {detail}",
            target="metal",
            code=f"{CODE_PREFIX}metal-debug-printf-unsupported",
        )


def _call_name(node):
    function = getattr(node, "function", None) or getattr(node, "name", None)
    return getattr(function, "name", function)


def _call_arguments(node):
    return list(getattr(node, "arguments", None) or getattr(node, "args", None) or [])


def _is_debug_printf_call(codegen, node) -> bool:
    name = _call_name(node)
    return (
        type(node).__name__ == "FunctionCallNode"
        and name in METAL_DEBUG_PRINTF_FUNCTIONS
        and name not in (getattr(codegen, "user_function_names", None) or ())
    )


def uses_debug_printf(codegen, ast) -> bool:
    """Whether ``ast`` makes a debug print, so that the module includes ``<metal_logging>``."""
    walk = getattr(ast, "walk", None)
    nodes = walk() if callable(walk) else codegen.iter_ast_nodes(ast)
    return any(_is_debug_printf_call(codegen, node) for node in nodes)


def _format_literal(node):
    literal_type = getattr(node, "literal_type", None)
    value = getattr(node, "value", None)
    if getattr(literal_type, "name", None) == "string" and isinstance(value, str):
        return value
    return None


def _rewrite_format(text, arguments, refuse):
    """Split ``text`` into Metal format pieces; returns ``(pieces, uses_vectors)``.

    Each piece is a literal string or ``(spec, argument, width)`` with ``width`` ``None`` for a scalar conversion.
    """
    pieces = []
    remaining = list(arguments)
    position = 0
    uses_vectors = False
    for match in _CONVERSION.finditer(text):
        pieces.append(text[position : match.start()])
        position = match.end()
        conversion = match.group("conversion")
        if conversion == "%":
            pieces.append("%%")
            continue
        if conversion in {"", "s", "n", "p"} or len(conversion) != 1:
            refuse(f"conversion '{match.group(0)}' has no Metal os_log equivalent")
        if "*" in (match.group("width") + (match.group("precision") or "")):
            refuse(f"conversion '{match.group(0)}' takes its width from an argument")
        if not remaining:
            refuse(f"conversion '{match.group(0)}' has no value to print")
        vector = match.group("vector")
        spec = "%" + "".join(
            match.group(part) or ""
            for part in ("flags", "width", "precision", "length", "conversion")
        )
        width = int(vector[1]) if vector else None
        uses_vectors = uses_vectors or width is not None
        pieces.append((spec, remaining.pop(0), width))
    pieces.append(text[position:])
    if remaining:
        refuse(f"{len(remaining)} value(s) have no conversion in the format")
    return pieces, uses_vectors


def metal_debug_printf_statement(codegen, expression):
    """The Metal statement text for a ``printf`` expression statement, or ``None`` if it is not one."""
    if not _is_debug_printf_call(codegen, expression):
        return None
    location = getattr(expression, "source_location", None)

    def refuse(detail):
        raise MetalDebugPrintfError(detail, source_location=location)

    arguments = _call_arguments(expression)
    text = _format_literal(arguments[0]) if arguments else None
    if text is None:
        refuse("the format must be a string literal")
    pieces, uses_vectors = _rewrite_format(text, arguments[1:], refuse)
    for piece in pieces:
        if isinstance(piece, tuple) and piece[2] is not None:
            width = codegen.vector_value_width(codegen.expression_result_type(piece[1]))
            if width is not None and width != piece[2]:
                refuse(f"'%v{piece[2]}' prints a {width}-component value")
    hoisted = []
    rendered = {}
    if uses_vectors:
        for argument in arguments[1:]:
            index = getattr(codegen, "debug_printf_locals", 0)
            codegen.debug_printf_locals = index + 1
            name = f"_crosstl_log{index}"
            hoisted.append(f"auto {name} = {codegen.generate_expression(argument)};")
            rendered[id(argument)] = name
    format_text = ""
    values = []
    for piece in pieces:
        if isinstance(piece, str):
            format_text += piece
            continue
        spec, argument, width = piece
        value = rendered.get(id(argument)) or codegen.generate_expression(argument)
        if width is None:
            format_text += spec
            values.append(value)
            continue
        format_text += ", ".join([spec] * width)
        values.extend(f"{value}.{_COMPONENTS[index]}" for index in range(width))
    call = "os_log_default.log(" + ", ".join([f'"{format_text}"', *values]) + ");"
    if not hoisted:
        return call
    return "\n".join(["{", *(f"    {line}" for line in hoisted), f"    {call}", "}"])

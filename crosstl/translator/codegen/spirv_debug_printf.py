"""SPIR-V lowering of CrossGL debug printing.

CrossGL ``printf(format, values...)`` is the shader debug print of HLSL (DXC's SPIR-V back end), Slang and GLSL's
GL_EXT_debug_printf ``debugPrintfEXT``. SPIR-V expresses it with the ``NonSemantic.DebugPrintf`` extended
instruction set: instruction 1, ``DebugPrintf``, takes the ``OpString`` holding the format and then the values to
print, and returns ``void``. A non-semantic instruction set needs SPV_KHR_non_semantic_info before SPIR-V 1.6.

The Vulkan validation layers interpret the format; they print 32- and 64-bit integer and floating-point scalars and
vectors (``%v<N><conversion>`` prints a vector), so any other operand is refused. The format must be a string
literal. Its C escape sequences are decoded, as glslang does, and the ``OpString`` operand re-escapes only the
quote and the backslash, the two characters SPIR-V assembly escapes.
"""

from __future__ import annotations

import re

__all__ = ["DEBUG_PRINTF_FUNCTIONS", "spirv_debug_printf_call"]

DEBUG_PRINTF_FUNCTIONS = frozenset({"printf", "debugPrintfEXT"})

_PRINTABLE_COMPONENTS = frozenset({"float", "double", "int", "uint", "i64", "u64"})
_SIMPLE_ESCAPES = {
    "n": "\n",
    "t": "\t",
    "r": "\r",
    "v": "\v",
    "f": "\f",
    "a": "\a",
    "b": "\b",
    "\\": "\\",
    '"': '"',
    "'": "'",
    "?": "?",
}
_ESCAPE = re.compile(r"\\(x[0-9A-Fa-f]{1,2}|[0-7]{1,3}|.)", re.DOTALL)


def decode_format_literal(text: str):
    """Return the characters of the C string literal body ``text``, or ``None`` for an unknown escape."""
    failed = []

    def replace(match):
        escape = match.group(1)
        if escape in _SIMPLE_ESCAPES:
            return _SIMPLE_ESCAPES[escape]
        if escape[0] == "x" and len(escape) > 1:
            return chr(int(escape[1:], 16))
        if escape[0] in "01234567":
            return chr(int(escape, 8))
        failed.append(escape)
        return ""

    decoded = _ESCAPE.sub(replace, text)
    return None if failed else decoded


def spirv_string_operand(text: str) -> str:
    """Quote ``text`` as a SPIR-V assembly literal string."""
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _format_text(node):
    literal_type = getattr(node, "literal_type", None)
    if getattr(literal_type, "name", None) != "string" or not isinstance(
        getattr(node, "value", None), str
    ):
        return None
    return decode_format_literal(node.value)


def _is_printable(codegen, value) -> bool:
    base_type = str(getattr(value.type, "base_type", ""))
    if base_type in _PRINTABLE_COMPONENTS:
        return True
    vector = codegen.vector_component_type_and_count(base_type)
    return vector is not None and vector[0] in _PRINTABLE_COMPONENTS


def debug_printf_import_id(codegen) -> int:
    """The id of the module's ``NonSemantic.DebugPrintf`` import, declaring it on first use."""
    import_id = getattr(codegen, "debug_printf_import_id", None)
    if import_id is None:
        import_id = codegen.get_id()
        codegen.debug_printf_import_id = import_id
        codegen.require_extension("SPV_KHR_non_semantic_info")
        codegen.emit(f'%{import_id} = OpExtInstImport "NonSemantic.DebugPrintf"')
    return import_id


def debug_string_id(codegen, text: str) -> int:
    """The id of an ``OpString`` holding ``text``; the module layout places the strings first in its debug section."""
    strings = getattr(codegen, "debug_string_ids", None)
    if strings is None:
        strings = codegen.debug_string_ids = {}
        codegen.debug_strings = []
    if text not in strings:
        strings[text] = codegen.get_id()
        codegen.debug_strings.append(
            f"%{strings[text]} = OpString {spirv_string_operand(text)}"
        )
    return strings[text]


def spirv_debug_printf_call(codegen, function_name: str, arguments, spirv_id_class):
    """Emit ``DebugPrintf`` for the CrossGL call ``function_name(arguments...)``; returns the ``void`` result."""
    void_type = codegen.register_primitive_type("void")
    text = _format_text(arguments[0]) if arguments else None
    if text is None:
        codegen.refuse_lowering(
            f"'{function_name}' needs a string-literal format with C escape sequences "
            "as its first operand (NonSemantic.DebugPrintf takes the format as an OpString)"
        )
        return None
    operands = []
    for index, argument in enumerate(arguments[1:], start=1):
        value = codegen.process_expression(argument)
        if value is None:
            return None
        if not _is_printable(codegen, value):
            codegen.refuse_lowering(
                f"'{function_name}' operand {index} has type '{value.type.base_type}'; "
                "NonSemantic.DebugPrintf prints only 32- or 64-bit integer or floating-point "
                "scalars and vectors"
            )
            return None
        operands.append(f"%{value.id}")
    import_id = debug_printf_import_id(codegen)
    format_id = debug_string_id(codegen, text)
    result_id = codegen.get_id()
    codegen.emit(
        " ".join(
            [
                f"%{result_id} = OpExtInst %{void_type.id} %{import_id} 1 %{format_id}",
                *operands,
            ]
        )
    )
    return spirv_id_class(result_id, void_type.type)

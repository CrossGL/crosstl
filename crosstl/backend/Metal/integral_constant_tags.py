"""Concrete Metal standard-library integral constants as ordinary tag structs.

``metal::bool_constant<B>``, ``metal::integral_constant<T, v>``,
``metal::true_type`` and ``metal::false_type`` are empty tag types: code
passes ``metal::bool_constant<true>{}`` to select an overload or a template
specialization, and reads ``decltype(tag)::value``. Once template
materialization has bound every argument, a tag such as
``metal::bool_constant<true>`` has nothing left to deduce, but CrossGL has no
spelling for a library class template, so the importer emitted the
undeclared type ``bool_constant<true_>``.

Each concrete tag is replaced by a synthesized struct with the same
``static constexpr`` ``value``, for example::

    struct metal_integral_constant_bool_true {
      static constexpr constant bool value = true;
    };

which every backend already lowers like a user-defined tag. Two spellings of
the same constant map to the same struct, so overloads and specializations
that differ only in the tag stay distinct and equal tags stay equal.

Left alone, so that the existing handling (or refusal) applies:

- tags whose arguments are not integer constants (still dependent);
- base clauses (``struct is_x : metal::true_type``), which the value-trait
  reader interprets;
- unqualified spellings when the source declares an entity of the same name
  itself (MLX declares its own ``integral_constant``), or does not open
  ``namespace metal``.
"""

from __future__ import annotations

import re
from typing import Any

_TAG_RE = re.compile(
    r"(?<![\w:.])(?P<qualifier>(?:::\s*)?metal\s*::\s*)?"
    r"(?P<name>bool_constant|integral_constant|true_type|false_type)\b"
)
_USING_METAL_RE = re.compile(r"\busing\s+namespace\s+metal\s*;")
_BASE_CLAUSE_RE = re.compile(r"\b(?:struct|class)\b[^;{}()]*:[^;{}]*$")
_INTEGRAL_TYPES = {
    "bool": "bool",
    "char": "char",
    "uchar": "uchar",
    "short": "short",
    "ushort": "ushort",
    "int": "int",
    "uint": "uint",
    "long": "long",
    "ulong": "ulong",
    "int8_t": "char",
    "uint8_t": "uchar",
    "int16_t": "short",
    "uint16_t": "ushort",
    "int32_t": "int",
    "uint32_t": "uint",
    "int64_t": "long",
    "uint64_t": "ulong",
    "size_t": "ulong",
    "unsigned": "uint",
    "unsigned int": "uint",
    "signed": "int",
    "signed int": "int",
    "unsigned char": "uchar",
    "signed char": "char",
    "unsigned short": "ushort",
    "unsigned long": "ulong",
}


def _declares_name(masked: str, name: str) -> bool:
    return (
        re.search(
            rf"\b(?:struct|class|union|enum)\s+{name}\b"
            rf"|\busing\s+{name}\s*="
            rf"|\btypedef\b[^;]*\b{name}\s*;",
            masked,
        )
        is not None
    )


def _integral_value(preprocessor: Any, text: str) -> int | None:
    stripped = text.strip()
    if stripped in {"true", "false"}:
        return 1 if stripped == "true" else 0
    ok, value = preprocessor._evaluate_static_integral_expression(stripped)
    if not ok or value is None:
        return None
    return int(value)


def _tag_identity(
    preprocessor: Any, name: str, arguments: list[str]
) -> tuple[str, int] | None:
    """``(canonical value type, value)`` for a concrete tag, or ``None``."""
    if name == "true_type":
        return ("bool", 1) if not arguments else None
    if name == "false_type":
        return ("bool", 0) if not arguments else None
    if name == "bool_constant":
        if len(arguments) != 1:
            return None
        value = _integral_value(preprocessor, arguments[0])
        return None if value is None else ("bool", 1 if value else 0)
    if len(arguments) != 2:
        return None
    value_type = _INTEGRAL_TYPES.get(re.sub(r"\s+", " ", arguments[0].strip()))
    if value_type is None:
        return None
    value = _integral_value(preprocessor, arguments[1])
    if value is None:
        return None
    if value_type == "bool":
        value = 1 if value else 0
    return value_type, value


def _struct_name(value_type: str, value: int) -> str:
    if value_type == "bool":
        return f"metal_integral_constant_bool_{'true' if value else 'false'}"
    spelled = f"m{-value}" if value < 0 else str(value)
    return f"metal_integral_constant_{value_type}_{spelled}"


def _struct_definition(name: str, value_type: str, value: int) -> str:
    if value_type == "bool":
        rendered = "true" if value else "false"
    elif value_type in {"uint", "ulong", "uchar", "ushort"}:
        rendered = f"{value}u"
    else:
        rendered = str(value)
    return (
        f"struct {name} {{\n"
        f"  static constexpr constant {value_type} value = {rendered};\n"
        f"}};\n"
    )


def _insertion_point(code: str, masked: str) -> int:
    using = _USING_METAL_RE.search(masked)
    if using is not None:
        return code.find("\n", using.end()) + 1 or len(code)
    includes = list(re.finditer(r"^[ \t]*#[ \t]*include\b[^\n]*\n?", masked, re.M))
    return includes[-1].end() if includes else 0


def concretize_integral_constant_tags(preprocessor: Any, code: str) -> str:
    """Replace concrete library integral-constant tags by synthesized structs."""

    if not any(
        name in code
        for name in ("bool_constant", "integral_constant", "true_type", "false_type")
    ):
        return code
    masked = preprocessor._mask_comments_and_literals(code)
    opens_metal = _USING_METAL_RE.search(masked) is not None
    user_declared = {
        name
        for name in ("bool_constant", "integral_constant", "true_type", "false_type")
        if _declares_name(masked, name)
    }
    replacements: list[tuple[int, int, str]] = []
    definitions: dict[str, str] = {}
    for match in _TAG_RE.finditer(masked):
        name = match.group("name")
        if match.group("qualifier") is None and (
            not opens_metal or name in user_declared
        ):
            continue
        boundary = max(masked.rfind(token, 0, match.start()) for token in ";{}")
        if _BASE_CLAUSE_RE.search(masked[boundary + 1 : match.start()]):
            continue
        end = match.end()
        arguments: list[str] = []
        if name in {"bool_constant", "integral_constant"}:
            angle = end
            while angle < len(masked) and masked[angle].isspace():
                angle += 1
            if angle >= len(masked) or masked[angle] != "<":
                continue
            close = preprocessor._find_matching_angle(code, angle)
            if close is None:
                continue
            arguments = preprocessor._split_top_level_commas(code[angle + 1 : close])
            end = close + 1
        identity = _tag_identity(preprocessor, name, arguments)
        if identity is None:
            continue
        struct_name = _struct_name(*identity)
        definitions.setdefault(struct_name, _struct_definition(struct_name, *identity))
        replacements.append((match.start(), end, struct_name))
    if not replacements:
        return code
    rewritten = preprocessor._apply_text_replacements(code, replacements)
    point = _insertion_point(
        rewritten, preprocessor._mask_comments_and_literals(rewritten)
    )
    prelude = "".join(definitions[name] for name in sorted(definitions))
    return rewritten[:point] + prelude + rewritten[point:]


_TAG_VALUE_RE = re.compile(
    r"(?P<type>(?:(?:::\s*)?metal\s*::\s*)?(?:bool_constant|integral_constant)\s*<.*>"
    r"|(?:(?:::\s*)?metal\s*::\s*)?(?:true_type|false_type))"
    r"\s*(?:\{\s*\}|\(\s*\))",
    re.DOTALL,
)


def library_tag_value_type(preprocessor: Any, expression: str) -> str | None:
    """The tag type a value-initialized concrete library tag (``metal::bool_constant<true>{}``) has."""

    match = _TAG_VALUE_RE.fullmatch(expression.strip())
    if match is None:
        return None
    type_text = match.group("type")
    name_match = _TAG_RE.match(type_text)
    if name_match is None or name_match.group("qualifier") is None:
        return None
    arguments: list[str] = []
    end = name_match.end()
    if name_match.group("name") in {"bool_constant", "integral_constant"}:
        angle = type_text.find("<", end)
        close = preprocessor._find_matching_angle(type_text, angle)
        if close != len(type_text.rstrip()) - 1:
            return None
        arguments = preprocessor._split_top_level_commas(type_text[angle + 1 : close])
    if _tag_identity(preprocessor, name_match.group("name"), arguments) is None:
        return None
    return preprocessor._normalize_template_argument_text(type_text)

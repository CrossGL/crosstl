"""CrossGL semantics for Metal ``[[color(n)]]`` attachment indices.

Metal names colour attachments by index, and samples often spell the index with an enumerator of a header shared
with the host (``[[color(RenderTargetAlbedo)]]``). A fragment output's attachment is its output location, so a
symbolic output index becomes ``@location(name)``, which the binding-constant pass folds on the CrossGL program. A
fragment *input* with ``[[color(n)]]`` reads the attachment back (framebuffer fetch); CrossGL spells it
``@gl_FragColor<n>`` and needs the integer here, so it is resolved from the unscoped enumerators declared in the
source and refused otherwise. That holds for the members of a structure a fragment function takes as a parameter
other than its ``[[stage_in]]`` input (:func:`framebuffer_fetch_struct_names`) as well as for plain parameters.
"""

import re

from .MetalAst import EnumNode

_INTEGER_LITERAL = re.compile(r"(0[xX][0-9a-fA-F]+|0[0-7]*|[1-9][0-9]*)[uUlL]*")


class MetalColorAttachmentIndexError(ValueError):
    """Raised when a framebuffer-fetch colour input names an index that is not an integer."""

    project_diagnostic_code = "project.translate.metal-color-input-index-unresolved"
    missing_capabilities = ("metal.symbolic-framebuffer-fetch-index",)

    def __init__(self, index):
        self.index = index
        super().__init__(
            f"Cannot import Metal fragment input [[color({index})]]: a "
            "framebuffer-fetch colour input needs an attachment index that is an "
            "integer literal or an enumerator with a literal value"
        )


def _literal_value(text):
    match = _INTEGER_LITERAL.fullmatch(str(text).strip())
    if match is None:
        return None
    digits = match.group(1)
    if digits[:2] in ("0x", "0X"):
        return int(digits, 16)
    if len(digits) > 1 and digits.startswith("0"):
        return int(digits, 8)
    return int(digits)


def enumerator_index_values(enums):
    """Map each unscoped enumerator with a literal or implicitly incremented value to that value.

    An enumerator declared in several enumerations, or whose value is an expression, is left out.
    """
    values = {}
    ambiguous = set()
    for enum in enums or []:
        if not isinstance(enum, EnumNode) or getattr(enum, "is_scoped", False):
            continue
        next_value = 0
        for name, value in getattr(enum, "members", []) or []:
            current = next_value if value is None else _literal_value(value)
            name = str(name)
            if name in values or name in ambiguous:
                ambiguous.add(name)
                values.pop(name, None)
            elif current is not None:
                values[name] = current
            next_value = None if current is None else current + 1
    return values


_TYPE_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_TYPE_QUALIFIERS = frozenset({"const", "thread", "volatile", "struct"})


def framebuffer_fetch_struct_names(functions, struct_names):
    """The structures a fragment function takes as a parameter that is not its ``[[stage_in]]`` input.

    Such a parameter is a framebuffer-fetch input, so ``[[color(n)]]`` on its members reads attachment ``n`` back
    (Metal Shading Language Specification, section 5.2.3.4, "Fragment Function Input Attributes"). The members are
    declared in the structure, before the importer sees the parameter, so the names are collected up front.
    """
    struct_names = frozenset(struct_names)
    names = set()
    for function in functions or []:
        if str(getattr(function, "qualifier", "") or "").lower() != "fragment":
            continue
        for parameter in getattr(function, "params", None) or []:
            attributes = {
                str(getattr(attribute, "name", "")).lower()
                for attribute in getattr(parameter, "attributes", None) or []
            }
            if "stage_in" in attributes:
                continue
            words = [
                word
                for word in _TYPE_NAME.findall(str(getattr(parameter, "vtype", "")))
                if word not in _TYPE_QUALIFIERS
            ]
            if len(words) == 1 and words[0] in struct_names:
                names.add(words[0])
    return names


def color_attachment_semantic(argument, *, is_input, enumerator_values):
    """Return the CrossGL semantic for ``[[color(argument)]]`` without the leading ``@``."""
    index = _literal_value(argument)
    if index is None and is_input:
        index = enumerator_values.get(str(argument).strip())
        if index is None:
            raise MetalColorAttachmentIndexError(argument)
    if index is None:
        return f"location({argument})"
    return "gl_FragColor" if index == 0 else f"gl_FragColor{index}"

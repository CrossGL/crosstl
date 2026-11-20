"""Exact lookups for the SPIR-V importer.

The importer used to look names up with ``mapping.get(key, key)``. For a ``BuiltIn`` decoration that wrote the
SPIR-V enumerant (``FullyCoveredEXT``) into CrossGL as an ordinary variable name, so the built-in silently became a
user input or output. :func:`builtin_variable_name` refuses such a decoration with
:class:`SpirvBuiltinUnsupportedError` instead.

The other lookups have a default that is the table's meaning, and the helpers below name it:

- :func:`keyword_safe_identifier` reads a table of the identifiers that are CrossGL keywords and their replacements.
  Any other identifier is already a valid CrossGL name.
- :func:`constant_or_literal` resolves an operand that is either the ``<id>`` of a constant or a literal. A literal is
  its own value; an ``<id>`` the table does not hold is not a constant, and the callers' integer parse rejects it.
"""

from __future__ import annotations

from crosstl.translator.diagnostics import CODE_PREFIX, TranslationDiagnostic

__all__ = [
    "SpirvBuiltinUnsupportedError",
    "builtin_variable_name",
    "constant_or_literal",
    "keyword_safe_identifier",
]


class SpirvBuiltinUnsupportedError(TranslationDiagnostic):
    """A ``BuiltIn`` decoration whose built-in variable CrossGL does not name."""

    def __init__(self, builtin, storage_class=None):
        self.builtin = builtin
        self.storage_class = storage_class
        self.missing_capabilities = (f"spirv.builtin.{builtin}",)
        where = f" ({storage_class} storage class)" if storage_class else ""
        super().__init__(
            f"SPIR-V BuiltIn {builtin}{where} has no CrossGL built-in variable; "
            "importing it as an ordinary variable would drop its meaning",
            target="cgl",
            code=f"{CODE_PREFIX}spirv-builtin-unsupported",
        )


def builtin_variable_name(names, builtin, storage_class=None):
    """The CrossGL built-in variable ``names`` gives the SPIR-V ``BuiltIn`` enumerant ``builtin``.

    Raises :class:`SpirvBuiltinUnsupportedError` when ``names`` has none.
    """
    if builtin in names:
        return names[builtin]
    raise SpirvBuiltinUnsupportedError(builtin, storage_class)


def keyword_safe_identifier(replacements, identifier):
    """``identifier``, or its replacement when it is a CrossGL keyword ``replacements`` lists."""
    if identifier in replacements:
        return replacements[identifier]
    return identifier


def constant_or_literal(constants, operand):
    """The value of the constant ``<id>`` ``operand``; ``operand`` itself when it is not a constant's ``<id>``."""
    if operand in constants:
        return constants[operand]
    return operand

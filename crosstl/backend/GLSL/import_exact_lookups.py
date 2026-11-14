"""Exact name and operator lookups for the GLSL importer.

The importer used to look operators and names up with ``mapping.get(key, key)``, which writes any key the table does
not know into CrossGL as it was spelled in GLSL. This module replaces those lookups:

- :func:`crossgl_operator` spells a GLSL operator in CrossGL. GLSL 4.60 section 5.1 lists the operators; all but
  ``^^`` have the same spelling and meaning in CrossGL. ``^^`` is defined only on scalar booleans, where it equals
  ``!=``. Any other operator raises :class:`GLSLImportOperatorError` instead of reaching the output.
- :func:`folded_operator` is the same lookup for constant folding, where an unknown operator means "not a constant".
- :func:`declared_name` reads a table that holds only the declarations the importer renamed. A name the table does
  not hold was not renamed, so it keeps its spelling: that default is the table's meaning, not a fallback.
"""

from __future__ import annotations

from crosstl.translator.codegen.operator_spellings import c_family_operator_spelling
from crosstl.translator.diagnostics import CODE_PREFIX, TranslationDiagnostic

__all__ = [
    "GLSLImportOperatorError",
    "GLSL_ONLY_OPERATORS",
    "crossgl_operator",
    "declared_name",
    "folded_operator",
]

# GLSL operators, or GLSL lexer token names, that CrossGL spells differently. ``^^`` and ``!=`` agree on the scalar
# booleans that GLSL 4.60 section 5.9 allows as ``^^`` operands. The importer parenthesizes every binary
# expression, so the different precedence of ``!=`` does not change the parse.
GLSL_ONLY_OPERATORS = {
    "^^": "!=",
    "LOGICAL_XOR": "!=",
    "LOGICAL_NOT": "!",
    "SHIFT_LEFT": "<<",
    "SHIFT_RIGHT": ">>",
    "PLUS_EQUALS": "+=",
    "MINUS_EQUALS": "-=",
    "MULTIPLY_EQUALS": "*=",
    "DIVIDE_EQUALS": "/=",
    "MOD_EQUALS": "%=",
}


class GLSLImportOperatorError(TranslationDiagnostic):
    """A GLSL expression operator that has no CrossGL spelling."""

    def __init__(self, operator):
        self.operator = operator
        self.missing_capabilities = ("opengl.import.operator",)
        super().__init__(
            f"GLSL operator {operator!r} has no CrossGL spelling",
            target="cgl",
            code=f"{CODE_PREFIX}opengl-import-operator-unsupported",
        )


def folded_operator(operator):
    """The CrossGL spelling of ``operator``, or ``None`` when CrossGL has none."""
    if not isinstance(operator, str):
        return None
    if operator in GLSL_ONLY_OPERATORS:
        return GLSL_ONLY_OPERATORS[operator]
    return c_family_operator_spelling(operator)


def crossgl_operator(operator):
    """The CrossGL spelling of the GLSL ``operator``; raises :class:`GLSLImportOperatorError` if it has none."""
    spelling = folded_operator(operator)
    if spelling is None:
        raise GLSLImportOperatorError(operator)
    return spelling


def declared_name(renames, name):
    """``name`` as the CrossGL program spells it.

    ``renames`` holds only the declarations the importer renamed, to avoid a CrossGL keyword or a clash with another
    declaration. A name it does not hold was not renamed, so its GLSL spelling is its CrossGL spelling. Callees that
    are neither declared nor CrossGL built-ins are refused later, by ``refuse_undeclared_functions``.
    """
    if name in renames:
        return renames[name]
    return name

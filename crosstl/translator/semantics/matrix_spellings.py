"""Non-square ``floatNxM`` matrix spellings: refused, because their shape depends on the language they come from.

CrossGL matrices have GLSL's meaning (ADR 0004): ``matCxR`` has C columns of R rows. The same spelling
``float2x3`` names two different matrices in the languages CrossGL imports: MSL reads it as 2 columns of 3 rows
(MSL 4.1 section 2.3), CrossGL ``mat2x3``, while HLSL and Slang read it as 2 rows of 3 columns, CrossGL
``mat3x2``. The generators disagreed on it: the Metal generator kept MSL's reading, the GLSL generator took HLSL's,
and the HLSL and SPIR-V generators mixed the two. :func:`refuse_ambiguous_matrix_spellings` raises
:class:`AmbiguousMatrixSpellingError` for the first non-square floating-point ``floatNxM`` type, constructor or cast
of a program, so the translation fails with a typed diagnostic instead of changing the matrix's shape. The importers
write ``matCxR`` and ``dmatCxR``.

Square spellings (``float3x3``) have one shape in every language and stay aliases of ``matN``. Half-precision
``halfNxM`` spellings are refused by :mod:`.half` for every shape. Integer and boolean matrices, which only HLSL
has, keep HLSL's row-first spelling (ADR 0004) and are not affected.
"""

from __future__ import annotations

import re

from .. import ast as A
from ..diagnostics import CODE_PREFIX, TranslationDiagnostic
from .declared_types import _nodes, declared_type_names

__all__ = [
    "AmbiguousMatrixSpellingError",
    "ambiguous_matrix_spelling",
    "refuse_ambiguous_matrix_spellings",
]

_SPELLING = re.compile(r"(?:simd_)?(float|double|min16float|min10float)([234])x([234])")
_PREFIX = {"double": "dmat"}


def ambiguous_matrix_spelling(name):
    """``(msl_reading, hlsl_reading)`` CrossGL spellings of a non-square ``floatNxM`` name, or ``None``."""
    match = _SPELLING.fullmatch(str(name or ""))
    if match is None or match.group(2) == match.group(3):
        return None
    prefix = _PREFIX.get(match.group(1), "mat")
    first, second = match.group(2), match.group(3)
    return f"{prefix}{first}x{second}", f"{prefix}{second}x{first}"


class AmbiguousMatrixSpellingError(TranslationDiagnostic):
    """A non-square ``floatNxM`` matrix spelling, whose shape CrossGL cannot tell."""

    def __init__(self, target, name, *, source_location=None):
        self.source_location = source_location
        self.type_name = name
        columns_first, rows_first = ambiguous_matrix_spelling(name)
        super().__init__(
            f"matrix type '{name}' has no single shape in CrossGL: MSL reads it as {name[-3]} columns of "
            f"{name[-1]} rows ({columns_first}), HLSL and Slang as {name[-3]} rows of {name[-1]} columns "
            f"({rows_first}); write {columns_first} or {rows_first} (CrossGL matCxR has C columns of R rows)",
            target=target,
            code=f"{CODE_PREFIX}ambiguous-matrix-spelling",
        )


def _spelled_name(node):
    """The plain type name ``node`` spells as a type, a constructor or a cast, or ``None``."""
    if isinstance(node, A.NamedType) and not node.generic_args:
        return node.name
    if isinstance(node, A.PrimitiveType):
        return node.name
    if isinstance(node, A.FunctionCallNode):
        function = getattr(node, "function", None)
        if isinstance(function, A.IdentifierNode) and not getattr(
            node, "generic_args", None
        ):
            return function.name
    return None


def refuse_ambiguous_matrix_spellings(program, target):
    """Raise :class:`AmbiguousMatrixSpellingError` for the first non-square ``floatNxM`` spelling in ``program``.

    A structure, alias or other declaration the program itself names ``float2x3`` is its own type and is kept.
    """
    declared = declared_type_names(program)
    for node in _nodes(program):
        name = _spelled_name(node)
        if not isinstance(name, str) or name in declared:
            continue
        if ambiguous_matrix_spelling(name) is not None:
            raise AmbiguousMatrixSpellingError(
                target, name, source_location=getattr(node, "source_location", None)
            )

"""One conforming preprocessor for the C++-based (Metal, HLSL) and GLSL front ends.

``preprocess(text, dialect="msl", path=..., include_paths=..., defines=...)`` returns a
:class:`PreprocessResult` with the macro-replaced text, a line map to the original files, the include closure
and the final macro table. ``dialect`` is ``"c++"``, ``"glsl"``, ``"msl"``, ``"hlsl"`` or a :class:`Dialect`
from :func:`msl_dialect` or :func:`hlsl_dialect` for a specific compiler configuration. Every construct it
cannot reproduce exactly is refused with a typed :class:`~crosstl.translator.diagnostics.TranslationDiagnostic`
whose code starts with ``project.translate.preprocessor-``.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Union

from .dialects import CPP, DIALECTS, GLSL, Dialect, HeaderMacros
from .dialects import dialect as resolve_dialect
from .engine import (
    FileSystem,
    MemoryFileSystem,
    Preprocessor,
    PreprocessorError,
    PreprocessResult,
)
from .macros import Macro
from .predefined import ConfigurationError, hlsl_dialect, msl_dialect

__all__ = [
    "CPP",
    "DIALECTS",
    "GLSL",
    "ConfigurationError",
    "Dialect",
    "FileSystem",
    "HeaderMacros",
    "Macro",
    "MemoryFileSystem",
    "PreprocessResult",
    "Preprocessor",
    "PreprocessorError",
    "hlsl_dialect",
    "msl_dialect",
    "preprocess",
]


def preprocess(
    text: str,
    dialect: str | Dialect = "c++",
    *,
    path: str | None = None,
    include_paths: Sequence[str] = (),
    system_paths: Sequence[str] = (),
    defines: dict[str, str | None] | None = None,
    undefines: Sequence[str] = (),
    file_system: FileSystem | None = None,
) -> PreprocessResult:
    """Preprocess ``text`` (the contents of ``path``, if given) for ``dialect``."""
    engine = Preprocessor(
        resolve_dialect(dialect),
        include_paths=include_paths,
        system_paths=system_paths,
        defines=defines,
        undefines=undefines,
        file_system=file_system,
    )
    return engine.run(text, path)

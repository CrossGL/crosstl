"""Metal source directives beyond the shared HLSL preprocessor: ``#import`` and fail-closed unknown directives.

The Metal compiler is Clang, which accepts the Objective-C ``#import`` directive in every C-family language. It
includes a file the way ``#include`` does, but at most once: a file that was already entered, by ``#include`` or by
``#import``, is skipped, and every later ``#include`` of an imported file is skipped as well (Clang's
``HeaderSearch::ShouldEnterIncludeFile``). Apple's sample code uses it for headers shared between Metal and
Objective-C sources.

A directive the preprocessor does not interpret must not reach the translated program: a target compiler would
reject it at best, and at worst it would change meaning silently. :class:`MetalDirectiveMixin` therefore refuses any
active directive other than the ones that survive preprocessing on purpose (``#include <...>`` of the Metal standard
library and ``#pragma``) with :class:`MetalPreprocessorDirectiveError`.
"""

from __future__ import annotations

import re

from crosstl.backend.DirectX.preprocessor import PRESERVED_SYSTEM_INCLUDE_SENTINEL

# Directives the shared preprocessor deliberately leaves in its output.
_PRESERVED_DIRECTIVES = frozenset({"", "include", "pragma"})
_DIRECTIVE_LINE = re.compile(r"^\s*#\s*([A-Za-z_][A-Za-z0-9_]*)?(.*)$")


class MetalPreprocessorDirectiveError(SyntaxError):
    """An active Metal preprocessor directive that cannot be carried into the translated program."""

    project_diagnostic_code = (
        "project.translate.metal-preprocessor-directive-unsupported"
    )
    missing_capabilities = ("metal.preprocessor-directive",)

    def __init__(self, directive, detail):
        self.directive = directive
        self.detail = detail
        super().__init__(f"Metal preprocessor directive '#{directive}' {detail}")


class MetalDirectiveMixin:
    """Adds ``#import`` and fail-closed directive handling to an :class:`HLSLPreprocessor` subclass."""

    def preprocess(self, code: str, file_path: str | None = None) -> str:
        self._import_directive_pending = False
        self._entered_source_files = set()
        if file_path:
            self._entered_source_files.add(self._canonical_path(file_path))
        processed = super().preprocess(code, file_path=file_path)
        # Entry-point discovery only lists kernel names; the translation of the same source refuses the directive.
        if not getattr(self, "_discovering_entry_points", False):
            reject_unknown_directives(processed)
        return processed

    def _parse_directive(self, line: str):
        directive, rest = super()._parse_directive(line)
        # The flag only describes the directive parsed last; the include handler runs before any other
        # directive is parsed, so it never sees a stale value.
        self._import_directive_pending = directive == "import"
        if directive == "import":
            return "include", rest
        return directive, rest

    def _handle_include(self, rest: str, file_path: str | None):
        importing = getattr(self, "_import_directive_pending", False)
        self._import_directive_pending = False
        entered = getattr(self, "_entered_source_files", None)
        if entered is None:
            entered = self._entered_source_files = set()
        included = super()._handle_include(rest, file_path)
        if isinstance(included, tuple):
            text, candidate = included
            # A system header that is not on the include path stays as a preserved ``#include <...>`` line.
            if (
                text
                and candidate
                and not text.startswith(PRESERVED_SYSTEM_INCLUDE_SENTINEL)
            ):
                key = self._canonical_path(candidate)
                if importing and key in entered:
                    return "", candidate
                if importing:
                    self.pragma_once_files.add(key)
                entered.add(key)
        return included


def reject_unknown_directives(processed: str) -> None:
    """Raise :class:`MetalPreprocessorDirectiveError` for any directive line the preprocessor left uninterpreted."""
    if "#" not in processed:
        return
    for line in processed.splitlines():
        match = _DIRECTIVE_LINE.match(line)
        if match is None:
            continue
        directive = match.group(1) or ""
        if directive not in _PRESERVED_DIRECTIVES:
            raise MetalPreprocessorDirectiveError(
                directive,
                "is not supported and cannot be carried into the translated program",
            )

"""Preprocessor support for Slang source imports."""

from typing import Dict, List, Optional

from crosstl.backend.DirectX.preprocessor import HLSLPreprocessor


class SlangPreprocessor(HLSLPreprocessor):
    """Small Slang preprocessor used before lexing imported source files."""

    def __init__(
        self,
        include_paths: Optional[List[str]] = None,
        defines: Optional[Dict[str, str]] = None,
        strict: bool = False,
        max_expansion_depth: int = 64,
    ):
        super().__init__(
            include_paths=include_paths,
            defines=defines,
            strict=strict,
            max_expansion_depth=max_expansion_depth,
        )

    def _join_multiline_function_macro_call(self, lines, start):
        line = lines[start]
        consumed = 1
        while self._function_macro_call_balance(line) > 0 and start + consumed < len(
            lines
        ):
            next_line = lines[start + consumed]
            if next_line.lstrip().startswith("#"):
                break
            line += "\n" + next_line
            consumed += 1
        return line, consumed

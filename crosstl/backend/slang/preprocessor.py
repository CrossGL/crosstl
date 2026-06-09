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

    def preprocess(self, code: str, file_path: Optional[str] = None) -> str:
        return super().preprocess(self._strip_block_comments(code), file_path)

    def _strip_block_comments(self, code: str) -> str:
        result = []
        i = 0
        in_block_comment = False
        string_delimiter = None
        escaped = False

        while i < len(code):
            char = code[i]
            next_char = code[i + 1] if i + 1 < len(code) else ""

            if in_block_comment:
                if char == "\n":
                    result.append(char)
                elif char == "*" and next_char == "/":
                    result.append(" ")
                    in_block_comment = False
                    i += 1
                i += 1
                continue

            if string_delimiter:
                result.append(char)
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == string_delimiter:
                    string_delimiter = None
                i += 1
                continue

            if char in "\"'":
                string_delimiter = char
                result.append(char)
                i += 1
                continue

            if char == "/" and next_char == "/":
                while i < len(code) and code[i] != "\n":
                    result.append(code[i])
                    i += 1
                continue

            if char == "/" and next_char == "*":
                result.append(" ")
                in_block_comment = True
                i += 2
                continue

            result.append(char)
            i += 1

        return "".join(result)

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

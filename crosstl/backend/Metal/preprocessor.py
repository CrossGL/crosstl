"""Preprocessor support for Metal source imports."""

from typing import Dict, List, Optional

from crosstl.backend.DirectX.preprocessor import HLSLPreprocessor

PRESERVED_INCLUDE_SENTINEL = "__CROSSGL_METAL_PRESERVED_INCLUDE__ "


class MetalPreprocessor(HLSLPreprocessor):
    """Small Metal preprocessor used before lexing imported source files."""

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
        processed = super().preprocess(code, file_path=file_path)
        return processed.replace(PRESERVED_INCLUDE_SENTINEL, "#include ")

    def _expand_macros(
        self,
        text: str,
        line_num: int,
        in_expression: bool,
        file_path: Optional[str] = None,
    ) -> str:
        if not in_expression and self._has_incomplete_function_macro_call(text):
            return text
        return super()._expand_macros(text, line_num, in_expression, file_path)

    def _has_incomplete_function_macro_call(self, text: str) -> bool:
        i = 0
        while i < len(text):
            if text[i] in "\"'":
                _literal, consumed = self._read_string(text, i)
                i += consumed
                continue
            if text.startswith("//", i):
                return False
            if text.startswith("/*", i):
                end = text.find("*/", i + 2)
                if end == -1:
                    return False
                i = end + 2
                continue
            if text[i].isalpha() or text[i] == "_":
                ident, consumed = self._read_identifier(text, i)
                macro = self.macros.get(ident)
                i += consumed
                if macro is None or not macro.is_function_like():
                    continue
                j = i
                while j < len(text) and text[j].isspace():
                    j += 1
                if (
                    j < len(text)
                    and text[j] == "("
                    and not self._call_closes_on_line(text, j)
                ):
                    return True
                continue
            i += 1
        return False

    def _call_closes_on_line(self, text: str, start: int) -> bool:
        depth = 0
        i = start
        while i < len(text):
            if text[i] in "\"'":
                _literal, consumed = self._read_string(text, i)
                i += consumed
                continue
            if text.startswith("//", i):
                return False
            if text.startswith("/*", i):
                end = text.find("*/", i + 2)
                if end == -1:
                    return False
                i = end + 2
                continue
            if text[i] == "(":
                depth += 1
            elif text[i] == ")":
                depth -= 1
                if depth == 0:
                    return True
            i += 1
        return False

    def _handle_include(self, rest: str, file_path: Optional[str]) -> Optional[str]:
        included = super()._handle_include(rest, file_path)
        if included is not None or self.strict:
            return included
        include_target = rest.strip()
        if include_target.startswith("<") and include_target.endswith(">"):
            return f"{PRESERVED_INCLUDE_SENTINEL}{include_target}"
        return None

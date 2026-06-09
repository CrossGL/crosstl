"""Preprocessor support for Metal source imports."""

import os
import re
from typing import Dict, List, Optional, Set

from crosstl.backend.DirectX.preprocessor import HLSLPreprocessor, Macro

PRESERVED_INCLUDE_SENTINEL = "__CROSSGL_METAL_PRESERVED_INCLUDE__ "
CLANG_FEATURE_TEST_MACROS = {
    "__has_attribute",
    "__has_builtin",
    "__has_extension",
    "__has_feature",
    "__has_include",
    "__has_include_next",
}
COMPILER_DIAGNOSTIC_START_RE = re.compile(
    r'^\s*(?:<[^>\n]+>[^:\n]*|"[^"\n]+"|[^:\n]+):\d+:\d+:?\s+' r"(?:warning|note):"
)
MSL_SOURCE_START_RE = re.compile(
    r"^\s*(?:"
    r"#include\b|#pragma\b|using\b|namespace\b|template\b|"
    r"struct\b|class\b|enum\b|typedef\b|constant\b|"
    r"\[\[|kernel\b|vertex\b|fragment\b|"
    r"(?:inline|static|constexpr|const|device|thread|threadgroup|void|"
    r"float|half|double|int|uint|long|ulong|short|ushort|char|uchar|bool)\b"
    r")"
)


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
        self.macros.setdefault(
            "TARGET_OS_SIMULATOR",
            Macro(name="TARGET_OS_SIMULATOR", replacement="0"),
        )
        for name in CLANG_FEATURE_TEST_MACROS:
            self.macros.setdefault(name, Macro(name=name, replacement="0"))

    def preprocess(self, code: str, file_path: Optional[str] = None) -> str:
        code = self._strip_leading_compiler_diagnostics(code)
        processed = super().preprocess(code, file_path=file_path)
        return processed.replace(PRESERVED_INCLUDE_SENTINEL, "#include ")

    def _strip_leading_compiler_diagnostics(self, code: str) -> str:
        lines = code.splitlines(keepends=True)
        saw_diagnostic = False

        for index, line in enumerate(lines):
            if MSL_SOURCE_START_RE.match(line):
                if saw_diagnostic:
                    return "".join(lines[index:])
                return code

            stripped = line.strip()
            if not stripped:
                continue
            if COMPILER_DIAGNOSTIC_START_RE.match(line):
                saw_diagnostic = True
                continue
            if saw_diagnostic and (
                line.startswith((" ", "\t")) or stripped.startswith("^")
            ):
                continue
            return code

        return code

    def _expand_macros(
        self,
        text: str,
        line_num: int,
        in_expression: bool,
        file_path: Optional[str] = None,
        disabled_macros: Optional[Set[str]] = None,
    ) -> str:
        if in_expression:
            text = self._expand_clang_feature_test_macros(text, file_path)
        if not in_expression and self._has_incomplete_function_macro_call(text):
            return text
        return super()._expand_macros(
            text, line_num, in_expression, file_path, disabled_macros
        )

    def _expand_clang_feature_test_macros(
        self, text: str, file_path: Optional[str]
    ) -> str:
        result = ""
        i = 0
        while i < len(text):
            if text[i] in "\"'":
                literal, consumed = self._read_string(text, i)
                result += literal
                i += consumed
                continue
            if text.startswith("//", i):
                result += text[i:]
                break
            if text.startswith("/*", i):
                end = text.find("*/", i + 2)
                if end == -1:
                    result += text[i:]
                    break
                result += text[i : end + 2]
                i = end + 2
                continue
            if text[i].isalpha() or text[i] == "_":
                ident, consumed = self._read_identifier(text, i)
                if ident in CLANG_FEATURE_TEST_MACROS:
                    replacement, consumed_call = self._expand_feature_test_call(
                        ident, text, i + consumed, file_path
                    )
                    if replacement is not None:
                        result += replacement
                        i += consumed + consumed_call
                        continue
                result += ident
                i += consumed
                continue
            result += text[i]
            i += 1
        return result

    def _expand_feature_test_call(
        self, name: str, text: str, start: int, file_path: Optional[str]
    ):
        i = start
        while i < len(text) and text[i].isspace():
            i += 1
        if i >= len(text) or text[i] != "(":
            return None, 0
        args, consumed = self._parse_macro_args(text, i)
        if not consumed or i + consumed > len(text):
            return None, 0
        if name in {"__has_include", "__has_include_next"}:
            value = self._has_include(args[0] if args else "", file_path)
        else:
            value = False
        return ("1" if value else "0"), i + consumed - start

    def _has_include(self, include_arg: str, file_path: Optional[str]) -> bool:
        include_arg = self._strip_macro_comments(include_arg).strip()
        match = re.match(r'([<"])([^>"]+)[>"]$', include_arg)
        if not match:
            return False

        delimiter, target = match.groups()
        search_paths: List[str] = []
        if delimiter == '"' and file_path:
            search_paths.append(os.path.dirname(file_path))
        search_paths.extend(self.include_paths)

        return any(os.path.isfile(os.path.join(base, target)) for base in search_paths)

    def _join_multiline_function_macro_call(self, lines: List[str], start: int):
        return lines[start], 1

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

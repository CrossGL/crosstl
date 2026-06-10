"""Preprocessor support for GLSL source imports."""

import os
import re
from typing import Dict, List, Optional, Tuple

from crosstl.backend.DirectX.preprocessor import HLSLPreprocessor, Macro

HASH_BRACKETED_MARKER_RE = re.compile(r"#\s*\[\s*[A-Za-z_][A-Za-z0-9_]*\s*\]")
EXTENSION_DIRECTIVE_RE = re.compile(r"^\s*#\s*extension\s+([A-Za-z_][A-Za-z0-9_]*)\s*:")
EXTENSION_COMPANION_MACROS = {
    "GL_ARM_tensors": (
        "GL_ARM_tensors_bfloat16",
        "GL_ARM_tensors_float_e5m2",
        "GL_ARM_tensors_float_e4m3",
    ),
}


class _GLSLDirectivePreprocessor(HLSLPreprocessor):
    """HLSL preprocessor variant with GLSL comment/directive semantics."""

    def _handle_include(
        self, rest: str, file_path: Optional[str]
    ) -> Optional[Tuple[str, str]]:
        match = re.match(r"\s*([<\"])([^>\"]+)[>\"]", rest)
        if not match:
            return None
        delimiter = match.group(1)
        target = match.group(2)

        search_paths: List[str] = []
        if delimiter == '"' and file_path:
            search_paths.append(os.path.dirname(file_path))
        search_paths.extend(self.include_paths)

        targets = [target]
        normalized_target = target.replace("\\", os.sep)
        if normalized_target != target:
            targets.append(normalized_target)

        for base in search_paths:
            for include_target in targets:
                candidate = os.path.join(base, include_target)
                if os.path.isfile(candidate):
                    return self._read_source_file(candidate), candidate

        if self.strict:
            raise FileNotFoundError(f"Include not found: {target}")
        return None

    def _handle_define(self, rest: str):
        rest = rest.lstrip()
        name_match = re.match(r"[A-Za-z_][A-Za-z0-9_]*", rest)
        name = name_match.group(0) if name_match else None

        super()._handle_define(rest)

        macro = self.macros.get(name) if name else None
        if (
            macro is not None
            and not macro.is_function_like()
            and "##" in macro.replacement
        ):
            macro.replacement = self._replace_params(macro.replacement, {})

    def _split_logical_lines(self, code: str) -> List[str]:
        return super()._split_logical_lines(
            self._strip_comments(code, preserve_block_newlines=False)
        )

    def _strip_comments(self, code: str, preserve_block_newlines: bool = True) -> str:
        result = []
        i = 0
        while i < len(code):
            ch = code[i]
            if ch in "\"'":
                literal, consumed = self._read_string(code, i)
                result.append(literal)
                i += consumed
                continue
            if code.startswith("//", i):
                i = self._skip_line_comment_text(code, i)
                continue
            if code.startswith("/*", i):
                consumed, newline_count = self._skip_block_comment_text(code, i)
                result.append(" ")
                if preserve_block_newlines:
                    result.extend("\n" for _ in range(newline_count))
                i += consumed
                continue
            result.append(ch)
            i += 1
        return "".join(result)

    def _skip_line_comment_text(self, code: str, start: int) -> int:
        search_from = start
        while True:
            end = code.find("\n", search_from)
            if end == -1:
                return len(code)
            previous = end - 2 if end > 0 and code[end - 1] == "\r" else end - 1
            if previous >= 0 and code[previous] == "\\":
                search_from = end + 1
                continue
            return end

    def _skip_block_comment_text(self, code: str, start: int) -> Tuple[int, int]:
        end = code.find("*/", start + 2)
        if end == -1:
            raise SyntaxError("Unterminated block comment")
        comment = code[start : end + 2]
        return len(comment), comment.count("\n")


class GLSLPreprocessor:
    """GLSL preprocessor wrapper that preserves #version placement rules."""

    def __init__(
        self,
        include_paths: Optional[List[str]] = None,
        defines: Optional[Dict[str, str]] = None,
        strict: bool = True,
        max_expansion_depth: int = 64,
    ):
        self.strict = strict
        self._preprocessor = _GLSLDirectivePreprocessor(
            include_paths=include_paths,
            defines=defines,
            strict=strict,
            max_expansion_depth=max_expansion_depth,
        )

    def preprocess(self, code: str, file_path: Optional[str] = None) -> str:
        self._ensure_version_first(code)
        include_paths = self._preprocessor.include_paths
        macros = dict(self._preprocessor.macros)
        self._apply_version_macros(code)
        self._apply_declared_extension_macros(code)
        implicit_paths = self._implicit_include_paths(file_path)
        if implicit_paths:
            self._preprocessor.include_paths = [
                *include_paths,
                *(path for path in implicit_paths if path not in include_paths),
            ]
        try:
            processed = self._preprocessor.preprocess(code, file_path=file_path)
        finally:
            self._preprocessor.include_paths = include_paths
            self._preprocessor.macros = macros
        self._ensure_version_first(processed)
        return processed

    def _apply_version_macros(self, code: str):
        match = re.search(
            r"(?m)^\s*#\s*version\s+([0-9]+)(?:\s+([A-Za-z_][A-Za-z0-9_]*))?",
            code,
        )
        if not match:
            return

        version = match.group(1)
        profile = (match.group(2) or "").lower()
        self._define_macro("__VERSION__", version)
        if profile == "es":
            self._define_macro("GL_ES", "1")
            return
        if profile == "compatibility":
            self._define_macro("GL_compatibility_profile", "1")
            return
        self._define_macro("GL_core_profile", "1")

    def _apply_declared_extension_macros(self, code: str):
        for line in self._preprocessor._split_logical_lines(code):
            match = EXTENSION_DIRECTIVE_RE.match(line)
            if match:
                extension_name = match.group(1)
                if extension_name != "all":
                    self._define_macro(extension_name, "1")
                    for companion in EXTENSION_COMPANION_MACROS.get(extension_name, ()):
                        self._define_macro(companion, "1")

    def _define_macro(self, name: str, value: str):
        self._preprocessor.macros.setdefault(
            name, Macro(name=name, params=None, replacement=value)
        )

    def _implicit_include_paths(self, file_path: Optional[str]) -> List[str]:
        if file_path is None:
            return []

        include_paths: List[str] = []
        current_dir = os.path.abspath(os.path.dirname(file_path))
        include_paths.append(current_dir)
        while True:
            candidate = os.path.join(current_dir, "includes", "glsl")
            if os.path.isdir(candidate):
                include_paths.append(candidate)

            common_dir = os.path.join(current_dir, "common")
            if os.path.isdir(common_dir):
                include_paths.append(common_dir)

            parent_dir = os.path.dirname(current_dir)
            if parent_dir == current_dir:
                break
            current_dir = parent_dir

        return include_paths

    def _ensure_version_first(self, code: str):
        version_index = self._find_version_index(code)
        if version_index is None:
            return
        if self._has_tokens_before(code, version_index):
            if self.strict:
                raise SyntaxError("#version must appear before any other tokens")

    def _find_version_index(self, code: str) -> Optional[int]:
        i = 0
        while i < len(code):
            ch = code[i]
            if ch.isspace():
                i += 1
                continue
            if code.startswith("//", i):
                i = self._skip_line_comment(code, i)
                continue
            if code.startswith("/*", i):
                i = self._skip_block_comment(code, i)
                continue
            if ch == "#":
                marker_end = self._skip_hash_bracketed_marker_line(code, i)
                if marker_end is not None:
                    i = marker_end
                    continue
                directive = self._read_directive(code, i)
                if directive == "version":
                    return i
                i = self._skip_line_comment(code, i)
                continue
            return None
        return None

    def _has_tokens_before(self, code: str, version_index: int) -> bool:
        i = 0
        while i < version_index:
            ch = code[i]
            if ch.isspace():
                i += 1
                continue
            if code.startswith("//", i):
                i = self._skip_line_comment(code, i)
                continue
            if code.startswith("/*", i):
                i = self._skip_block_comment(code, i)
                continue
            marker_end = self._skip_hash_bracketed_marker_line(code, i)
            if marker_end is not None:
                i = marker_end
                continue
            return True
        return False

    def _skip_hash_bracketed_marker_line(self, code: str, start: int) -> Optional[int]:
        match = HASH_BRACKETED_MARKER_RE.match(code, start)
        if not match:
            return None

        i = match.end()
        while i < len(code) and code[i] in " \t\r\f\v":
            i += 1
        if i >= len(code):
            return i
        if code[i] == "\n":
            return i + 1
        if code.startswith("//", i):
            return self._skip_line_comment(code, i)
        if code.startswith("/*", i):
            after_comment = self._skip_block_comment(code, i)
            while after_comment < len(code) and code[after_comment] in " \t\r\f\v":
                after_comment += 1
            if after_comment >= len(code):
                return after_comment
            if code[after_comment] == "\n":
                return after_comment + 1
        return None

    def _read_directive(self, code: str, start: int) -> str:
        match = re.match(r"#\s*([A-Za-z_][A-Za-z0-9_]*)", code[start:])
        if not match:
            return ""
        return match.group(1)

    def _skip_line_comment(self, code: str, start: int) -> int:
        end = code.find("\n", start)
        if end == -1:
            return len(code)
        return end + 1

    def _skip_block_comment(self, code: str, start: int) -> int:
        end = code.find("*/", start + 2)
        if end == -1:
            return len(code)
        return end + 2

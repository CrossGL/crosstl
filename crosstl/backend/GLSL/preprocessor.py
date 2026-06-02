"""Preprocessor support for GLSL source imports."""

import os
import re
from typing import Dict, List, Optional

from crosstl.backend.DirectX.preprocessor import HLSLPreprocessor


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
        self._preprocessor = HLSLPreprocessor(
            include_paths=include_paths,
            defines=defines,
            strict=strict,
            max_expansion_depth=max_expansion_depth,
        )

    def preprocess(self, code: str, file_path: Optional[str] = None) -> str:
        self._ensure_version_first(code)
        include_paths = self._preprocessor.include_paths
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
        self._ensure_version_first(processed)
        return processed

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
            return True
        return False

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

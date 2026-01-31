import re
from typing import Dict, List, Optional

from crosstl.backend.DirectX.preprocessor import HLSLPreprocessor


class GLSLPreprocessor:
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
        processed = self._preprocessor.preprocess(code, file_path=file_path)
        self._ensure_version_first(processed)
        return processed

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

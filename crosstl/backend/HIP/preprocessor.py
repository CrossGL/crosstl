"""Preprocessor support for HIP source imports."""

from typing import Dict, List, Optional, Tuple

from crosstl.backend.DirectX.preprocessor import HLSLPreprocessor

PRESERVED_INCLUDE_SENTINEL = "__CROSSGL_HIP_PRESERVED_INCLUDE__ "


class HipPreprocessor(HLSLPreprocessor):
    """Small HIP preprocessor used before lexing imported source files."""

    DEFAULT_PLATFORM_DEFINES = {
        "__HIP_PLATFORM_AMD__": "1",
        "__linux__": "1",
        "__unix__": "1",
    }
    HIP_PLATFORM_DEFINE_NAMES = {
        "__HIP_PLATFORM_AMD__",
        "__HIP_PLATFORM_NVIDIA__",
    }
    HOST_PLATFORM_DEFINE_NAMES = {
        "__APPLE__",
        "__MACH__",
        "__linux__",
        "__unix__",
        "_WIN32",
        "_WIN64",
        "WIN32",
        "WIN64",
        "linux",
    }

    def __init__(
        self,
        include_paths: Optional[List[str]] = None,
        defines: Optional[Dict[str, str]] = None,
        strict: bool = False,
        max_expansion_depth: int = 64,
    ):
        merged_defines = defines
        if not strict:
            merged_defines = dict(self.DEFAULT_PLATFORM_DEFINES)
            if defines:
                if any(name in defines for name in self.HOST_PLATFORM_DEFINE_NAMES):
                    for name in self.HOST_PLATFORM_DEFINE_NAMES:
                        merged_defines.pop(name, None)
                if any(name in defines for name in self.HIP_PLATFORM_DEFINE_NAMES):
                    for name in self.HIP_PLATFORM_DEFINE_NAMES:
                        merged_defines.pop(name, None)
                merged_defines.update(defines)

        super().__init__(
            include_paths=include_paths,
            defines=merged_defines,
            strict=strict,
            max_expansion_depth=max_expansion_depth,
        )

    def preprocess(self, code: str, file_path: Optional[str] = None) -> str:
        code = self._mask_comments(code)
        processed = super().preprocess(code, file_path=file_path)
        return processed.replace(PRESERVED_INCLUDE_SENTINEL, "#include ")

    def _mask_comments(self, code: str) -> str:
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
                end = code.find("\n", i)
                if end == -1:
                    result.append(" " * (len(code) - i))
                    break
                result.append(" " * (end - i))
                result.append("\n")
                i = end + 1
                continue
            if code.startswith("/*", i):
                end = code.find("*/", i + 2)
                comment = code[i:] if end == -1 else code[i : end + 2]
                result.append(
                    "".join(
                        "\n" if comment_ch == "\n" else " " for comment_ch in comment
                    )
                )
                i += len(comment)
                continue
            result.append(ch)
            i += 1

        return "".join(result)

    def _handle_include(self, rest: str, file_path: Optional[str]):
        included = super()._handle_include(rest, file_path)
        if isinstance(included, tuple):
            included_text, included_path = included
            return self._mask_comments(included_text), included_path
        if included is not None:
            return self._mask_comments(included)
        if included is not None or self.strict:
            return included
        include_target = rest.strip()
        if include_target.startswith("<") and include_target.endswith(">"):
            return f"{PRESERVED_INCLUDE_SENTINEL}{include_target}"
        return None

    def _join_multiline_function_macro_call(
        self, lines: List[str], start: int
    ) -> Tuple[str, int]:
        if self._function_macro_call_balance(lines[start]) <= 0:
            return super()._join_multiline_function_macro_call(lines, start)

        line = self._strip_macro_comments(lines[start])
        consumed = 1

        while self._function_macro_call_balance(line) > 0 and start + consumed < len(
            lines
        ):
            next_line = lines[start + consumed]
            if next_line.lstrip().startswith("#"):
                break
            line += " " + self._strip_macro_comments(next_line.lstrip())
            consumed += 1

        return line, consumed

"""Preprocessor support for OpenCL source imports."""

import re
from typing import Dict, List, Optional, Tuple

from crosstl.backend.DirectX.macro_expansion import OPENCL_MACROS
from crosstl.backend.DirectX.preprocessor import HLSLPreprocessor, LogicalLines, Macro

PRESERVED_INCLUDE_SENTINEL = "__CROSSGL_OPENCL_PRESERVED_INCLUDE__ "


class OpenCLPreprocessor(HLSLPreprocessor):
    """Small OpenCL preprocessor used before lexing imported source files."""

    macro_dialect = OPENCL_MACROS

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
        for name in ("VECTOR_SIZE_I", "VECTOR_SIZE_J", "VECTOR_SIZE_K"):
            self.macros.setdefault(name, Macro(name=name, replacement="1"))
        for name in (
            "DECLARE_INPUT_MAT_N",
            "DECLARE_OUTPUT_MAT_N",
            "DECLARE_INDEX_N",
            "PROCESS_ELEM_N",
        ):
            self.macros.setdefault(name, Macro(name=name, replacement=""))

    def preprocess(self, code: str, file_path: Optional[str] = None) -> str:
        if not self.strict:
            code = self._downgrade_feature_gated_errors(code)
        processed = super().preprocess(code, file_path=file_path)
        return processed.replace(PRESERVED_INCLUDE_SENTINEL, "#include ")

    def _downgrade_feature_gated_errors(self, code: str) -> str:
        return re.sub(r"(?m)^(\s*)#(\s*)error\b", r"\1#\2warning", code)

    def _split_logical_lines(self, code: str) -> List[str]:
        # Comments are blanked after the lines are spliced and joined, so a comment that a directive line
        # opens stays part of that directive.
        lines = super()._split_logical_lines(code)
        return LogicalLines(
            [self._mask_comments(line) for line in lines], lines.physical
        )

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
                result.append(self._mask_comment_text(code[i:end]))
                result.append("\n")
                i = end + 1
                continue
            if code.startswith("/*", i):
                end = code.find("*/", i + 2)
                comment = code[i:] if end == -1 else code[i : end + 2]
                result.append(self._mask_comment_text(comment))
                i += len(comment)
                continue
            result.append(ch)
            i += 1

        return "".join(result)

    def _mask_comment_text(self, comment: str) -> str:
        result = []
        for index, comment_ch in enumerate(comment):
            if comment_ch == "\n":
                result.append("\n")
            elif comment_ch == "\\" and self._is_line_continuation(comment, index):
                result.append("\\")
            else:
                result.append(" ")
        return "".join(result)

    def _is_line_continuation(self, text: str, index: int) -> bool:
        next_newline = text.find("\n", index + 1)
        if next_newline == -1:
            return not text[index + 1 :].strip()
        return not text[index + 1 : next_newline].strip()

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
        conditional_stack: List[Dict[str, bool]] = []

        def is_active() -> bool:
            return all(frame["active"] for frame in conditional_stack)

        while self._function_macro_call_balance(line) > 0 and start + consumed < len(
            lines
        ):
            next_line = lines[start + consumed]
            stripped = next_line.lstrip()
            if stripped.startswith("#"):
                directive, rest = self._parse_directive(stripped)
                if directive in ("if", "ifdef", "ifndef"):
                    parent_active = is_active()
                    condition = self._evaluate_condition(directive, rest, 0, None)
                    conditional_stack.append(
                        {
                            "parent_active": parent_active,
                            "active": parent_active and condition,
                            "branch_taken": condition,
                        }
                    )
                elif directive == "elif" and conditional_stack:
                    frame = conditional_stack[-1]
                    if frame["parent_active"] and not frame["branch_taken"]:
                        condition = self._evaluate_expression(
                            self._expand_macros(rest, 0, True, None)
                        )
                        frame["active"] = condition
                        frame["branch_taken"] = condition
                    else:
                        frame["active"] = False
                elif directive == "else" and conditional_stack:
                    frame = conditional_stack[-1]
                    frame["active"] = (
                        frame["parent_active"] and not frame["branch_taken"]
                    )
                    frame["branch_taken"] = True
                elif directive == "endif" and conditional_stack:
                    conditional_stack.pop()
                line += "\n"
                consumed += 1
                continue
            # One line break per physical line, so __LINE__ counts the lines the invocation spans.
            line += "\n"
            if is_active():
                line += self._strip_macro_comments(stripped)
            consumed += 1

        return line, consumed

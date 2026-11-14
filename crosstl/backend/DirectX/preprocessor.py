"""Preprocessor support for DirectX HLSL source imports."""

import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from crosstl.translator.preprocessor.macros import UnterminatedInvocation
from crosstl.translator.refusals import DirectXSyntaxError

from .macro_expansion import (
    HLSL_MACROS,
    MacroDialect,
    code_extent,
    comments_to_spaces,
    evaluate_condition,
    expand_condition,
    expand_line,
)

PRESERVED_SYSTEM_INCLUDE_SENTINEL = "__CROSSGL_HLSL_PRESERVED_INCLUDE__ "
# ``\w`` on a str pattern is exactly ``str.isalnum()`` or ``_``.
IDENTIFIER_CONTINUATION_RE = re.compile(r"\w*")


@dataclass
class Macro:
    """Object-like or function-like HLSL preprocessor macro."""

    name: str
    params: Optional[List[str]] = None
    replacement: str = ""
    is_variadic: bool = False

    def is_function_like(self) -> bool:
        return self.params is not None


class LogicalLines(list):
    """Logical source lines; ``physical[i]`` is the number of physical lines that line ``i`` spans."""

    def __init__(self, lines: List[str], physical: List[int]):
        super().__init__(lines)
        self.physical = physical


class HLSLPreprocessor:
    """Small HLSL preprocessor used before lexing imported source files.

    Directives are interpreted line by line here; macro replacement follows the C++ rules through the shared
    preprocessor's expander (:mod:`.macro_expansion`), configured by ``macro_dialect`` for the compiler whose
    source a subclass reads.
    """

    macro_dialect: MacroDialect = HLSL_MACROS

    def __init__(
        self,
        include_paths: Optional[List[str]] = None,
        defines: Optional[Dict[str, str]] = None,
        strict: bool = False,
        max_expansion_depth: int = 64,
    ):
        self.include_paths = include_paths or []
        self.macros: Dict[str, Macro] = {}
        self.strict = strict
        self.max_expansion_depth = max_expansion_depth
        self.pragma_once_files: Set[str] = set()
        # State of the built-in macros (macro_expansion): __COUNTER__, __INCLUDE_LEVEL__ and __BASE_FILE__,
        # the physical line of each part of a joined line, and whether a replaced line ends in a
        # function-like macro name that a '(' on a following line would still invoke.
        self._macro_counter = 0
        self._include_depth = 0
        self._base_file: Optional[str] = None
        self._joined_line_numbers: Optional[List[int]] = None
        self._invocation_continues = False

        if defines:
            for name, value in defines.items():
                self.macros[name] = Macro(
                    name=name, params=None, replacement=str(value)
                )

    def preprocess(self, code: str, file_path: Optional[str] = None) -> str:
        self._macro_counter = 0
        self._include_depth = 0
        self._base_file = file_path
        logical_lines = self._split_logical_lines(code)
        output_lines = self._process_lines(logical_lines, file_path)
        return "\n".join(output_lines).replace(
            PRESERVED_SYSTEM_INCLUDE_SENTINEL, "#include "
        )

    def _split_logical_lines(self, code: str) -> List[str]:
        """Logical lines of ``code``: backslash-newline splices are removed ([lex.phases]/1.2), and a line that
        opens a block comment continues to the line that closes it, because a comment is one space
        ([lex.phases]/1.3): a directive goes on after its comment (which becomes a space) and a ``#`` inside a
        comment starts no directive. A text line keeps the comment and its line breaks.
        """
        lines = code.splitlines()
        logical_lines: List[str] = []
        physical: List[int] = []
        buffer = ""
        spliced = 0
        comment_at = -1  # where the block comment that ``buffer`` leaves open starts
        checked = 0  # ``buffer`` before this offset holds no comment end
        for line in lines:
            stripped = line.rstrip()
            if stripped.endswith("\\"):
                buffer += stripped[:-1]
                spliced += 1
                continue
            buffer += line
            if comment_at >= 0 and "*/" not in buffer[checked:]:
                extent = comment_at
            else:
                start = max(comment_at, 0)
                extent = start + code_extent(buffer[start:])
            if extent < len(buffer):
                comment_at = extent
                buffer += "\n"
                checked = len(buffer) - 1
                spliced += 1
                continue
            if comment_at >= 0 and self._is_directive_line(buffer):
                buffer = comments_to_spaces(buffer)
            logical_lines.append(buffer)
            physical.append(spliced + 1)
            buffer = ""
            spliced = 0
            comment_at = -1
        if buffer:
            if comment_at >= 0:
                buffer = buffer[:-1]  # the source ends inside a block comment
            logical_lines.append(buffer)
            physical.append(spliced)
        return LogicalLines(logical_lines, physical)

    def _is_directive_line(self, line: str) -> bool:
        """Whether ``line`` starts with ``#``, or with its alternative token ``%:`` where the dialect has
        digraphs ([lex.digraph])."""
        stripped = line.lstrip()
        return stripped.startswith("#") or (
            self.macro_dialect.mode.digraphs and stripped.startswith("%:")
        )

    def _paren_balance(self, text: str) -> int:
        balance = 0
        i = 0
        while i < len(text):
            if text[i] in "\"'":
                _literal, consumed = self._read_string(text, i)
                i += consumed
                continue
            if text.startswith("//", i):
                # A joined invocation keeps its line breaks; the comment ends with its line.
                newline = text.find("\n", i + 2)
                if newline == -1:
                    break
                i = newline + 1
                continue
            if text.startswith("/*", i):
                end = text.find("*/", i + 2)
                if end == -1:
                    break
                i = end + 2
                continue
            if text[i] == "(":
                balance += 1
            elif text[i] == ")" and balance > 0:
                balance -= 1
            i += 1
        return balance

    def _process_lines(self, lines: List[str], file_path: Optional[str]) -> List[str]:
        output: List[str] = []
        conditional_stack: List[Dict[str, bool]] = []
        current_line = 1
        current_file = file_path

        def is_active() -> bool:
            return all(frame["active"] for frame in conditional_stack)

        physical = getattr(lines, "physical", None)
        idx = 0
        while idx < len(lines):
            raw_line = lines[idx]
            line = raw_line
            stripped = line.lstrip()
            active = is_active()
            consumed_lines = 1

            if self._is_directive_line(stripped):
                if stripped.startswith("%:"):
                    stripped = "#" + stripped[2:]
                directive, rest = self._parse_directive(stripped)
                if directive in ("if", "ifdef", "ifndef"):
                    # The condition of a group nested in a skipped group is not evaluated ([cpp.cond]/13).
                    condition = active and self._evaluate_condition(
                        directive, rest, current_line, current_file
                    )
                    parent_active = active
                    active_now = parent_active and condition
                    conditional_stack.append(
                        {
                            "parent_active": parent_active,
                            "active": active_now,
                            "branch_taken": condition,
                        }
                    )
                elif directive == "elif":
                    if not conditional_stack:
                        raise DirectXSyntaxError("#elif without #if")
                    frame = conditional_stack[-1]
                    if frame["parent_active"] and not frame["branch_taken"]:
                        condition = self._evaluate_expression(
                            self._expand_macros(rest, current_line, True, current_file)
                        )
                        frame["active"] = condition
                        frame["branch_taken"] = condition
                    else:
                        frame["active"] = False
                elif (
                    directive in ("elifdef", "elifndef") and self.macro_dialect.elifdef
                ):
                    if not conditional_stack:
                        raise DirectXSyntaxError(f"#{directive} without #if")
                    frame = conditional_stack[-1]
                    if frame["parent_active"] and not frame["branch_taken"]:
                        condition = self._evaluate_condition(
                            directive[2:], rest, current_line, current_file
                        )
                        frame["active"] = condition
                        frame["branch_taken"] = condition
                    else:
                        frame["active"] = False
                elif directive == "else":
                    if not conditional_stack:
                        raise DirectXSyntaxError("#else without #if")
                    frame = conditional_stack[-1]
                    frame["active"] = (
                        frame["parent_active"] and not frame["branch_taken"]
                    )
                    frame["branch_taken"] = True
                elif directive == "endif":
                    if not conditional_stack:
                        raise DirectXSyntaxError("#endif without #if")
                    conditional_stack.pop()
                elif not active:
                    pass
                elif directive == "define":
                    self._handle_define(rest)
                elif directive == "undef":
                    name = rest.strip()
                    if name in self.macros:
                        del self.macros[name]
                elif directive == "include":
                    included = self._handle_include(rest, file_path)
                    if included is not None:
                        if isinstance(included, tuple):
                            included_text, included_path = included
                        else:
                            included_text = included
                            included_path = file_path
                        nested_lines = self._split_logical_lines(included_text)
                        self._include_depth += 1
                        try:
                            output.extend(
                                self._process_lines(nested_lines, included_path)
                            )
                        finally:
                            self._include_depth -= 1
                elif directive == "line":
                    line_info = self._handle_line_directive(rest)
                    if line_info is not None:
                        line_number, line_file = line_info
                        current_line = line_number - 1
                        if line_file is not None:
                            current_file = line_file
                elif directive == "pragma":
                    if active and self._is_pragma_once(rest) and current_file:
                        self.pragma_once_files.add(self._canonical_path(current_file))
                    elif active:
                        output.append(line)
                elif directive in ("error", "warning"):
                    if directive == "error" or self.strict:
                        raise DirectXSyntaxError(f"#{directive}: {rest.strip()}")
                elif (
                    not directive
                    and not self._strip_macro_comments(stripped[1:]).strip()
                ):
                    pass  # the null directive ([cpp.null])
                else:
                    if active:
                        output.append(line)
            else:
                if active:
                    line, consumed_lines = self._join_multiline_function_macro_call(
                        lines, idx
                    )
                    expanded, consumed_lines = self._expand_text_line(
                        lines, idx, line, consumed_lines, current_line, current_file
                    )
                    output.append(expanded)
            if physical is not None:
                current_line += sum(physical[idx : idx + consumed_lines])
            else:
                current_line += consumed_lines
            idx += consumed_lines

        if conditional_stack:
            raise DirectXSyntaxError("Unterminated #if block")

        return output

    def _expand_text_line(
        self,
        lines: List[str],
        start: int,
        line: str,
        consumed: int,
        line_num: int,
        file_path: Optional[str],
    ) -> Tuple[str, int]:
        """Replace the macros of the text line ``line`` (``lines[start:start + consumed]`` joined), joining
        following lines while a macro invocation continues on them: its argument list is not terminated, or the
        replaced line ends in a function-like macro name and the next line that is not blank starts with ``(``.
        As in the shared preprocessor, a directive ends the text that an invocation can take in.
        """
        physical = getattr(lines, "physical", None)
        while True:
            self._invocation_continues = False
            self._joined_line_numbers = self._physical_line_numbers(
                physical, start, consumed, line_num
            )
            try:
                expanded = self._expand_macros(line, line_num, False, file_path)
            except UnterminatedInvocation:
                following = start + consumed
                if following >= len(lines) or self._is_directive_line(lines[following]):
                    raise
                line += "\n" + lines[following]
                consumed += 1
                continue
            finally:
                self._joined_line_numbers = None
            more = (
                self._lines_to_invocation_parenthesis(lines, start + consumed)
                if self._invocation_continues
                else 0
            )
            if not more:
                return expanded, consumed
            for following in range(start + consumed, start + consumed + more):
                line += "\n" + lines[following]
            consumed += more

    @staticmethod
    def _physical_line_numbers(
        physical: Optional[List[int]], start: int, consumed: int, line_num: int
    ) -> Optional[List[int]]:
        if physical is None or consumed == 1:
            return None
        numbers = []
        for count in physical[start : start + consumed]:
            numbers.append(line_num)
            line_num += count
        return numbers

    def _lines_to_invocation_parenthesis(self, lines: List[str], start: int) -> int:
        """How many lines from ``start`` to take in so that the first token after blank lines, a ``(``, joins
        the line before; 0 when that token is not ``(`` or a directive or an open block comment comes first.
        """
        index = start
        while index < len(lines):
            text = lines[index]
            if self._is_directive_line(text) or code_extent(text) < len(text):
                return 0
            code = self._strip_macro_comments(text).strip()
            if code:
                return index - start + 1 if code.startswith("(") else 0
            index += 1
        return 0

    def _join_multiline_function_macro_call(
        self, lines: List[str], start: int
    ) -> Tuple[str, int]:
        line = lines[start]
        consumed = 1
        while self._function_macro_call_balance(line) > 0 and start + consumed < len(
            lines
        ):
            next_line = lines[start + consumed]
            if next_line.lstrip().startswith("#"):
                break
            # Keep the physical line break, so a // comment ends where its line does.
            line += "\n" + next_line.lstrip()
            consumed += 1
        return line, consumed

    def _function_macro_call_balance(self, text: str) -> int:
        for name, macro in self.macros.items():
            if not macro.is_function_like():
                continue
            pattern = rf"\b{re.escape(name)}\s*\("
            for match in re.finditer(pattern, text):
                balance = self._paren_balance(text[match.start() :])
                if balance > 0:
                    return balance
        return 0

    def _parse_directive(self, line: str) -> Tuple[str, str]:
        match = re.match(r"#\s*([A-Za-z_][A-Za-z0-9_]*)\s*(.*)", line)
        if not match:
            return "", ""
        return match.group(1), match.group(2)

    def _evaluate_condition(
        self,
        directive: str,
        rest: str,
        line_num: int,
        file_path: Optional[str],
    ) -> bool:
        if directive in ("ifdef", "ifndef"):
            parts = self._strip_macro_comments(rest).strip().split(maxsplit=1)
            name = parts[0] if parts else ""
            defined = name in self.macros or name in self.macro_dialect.builtins
            return defined if directive == "ifdef" else not defined
        return bool(
            self._evaluate_expression(
                self._expand_macros(rest, line_num, True, file_path)
            )
        )

    def _evaluate_expression(self, expr: str) -> int:
        return int(evaluate_condition(self, expr))

    def _handle_define(self, rest: str):
        rest = rest.lstrip()
        name_match = re.match(r"[A-Za-z_][A-Za-z0-9_]*", rest)
        if not name_match:
            return
        name = name_match.group(0)
        after = rest[name_match.end() :]
        if after.startswith("("):
            params, remainder, is_variadic = self._parse_macro_params(after)
            replacement = self._strip_macro_comments(remainder.lstrip())
            self.macros[name] = Macro(
                name=name,
                params=params,
                replacement=replacement,
                is_variadic=is_variadic,
            )
        else:
            replacement = self._strip_macro_comments(after.lstrip())
            self.macros[name] = Macro(name=name, params=None, replacement=replacement)

    def _strip_macro_comments(self, text: str) -> str:
        result = ""
        i = 0
        while i < len(text):
            if text[i] in "\"'":
                literal, consumed = self._read_string(text, i)
                result += literal
                i += consumed
                continue
            if text.startswith("//", i):
                break
            if text.startswith("/*", i):
                end = text.find("*/", i + 2)
                if end == -1:
                    break
                result += " "
                i = end + 2
                continue
            result += text[i]
            i += 1
        return result.rstrip()

    def _parse_macro_params(self, text: str) -> Tuple[List[str], str, bool]:
        assert text[0] == "("
        depth = 0
        params_text = ""
        i = 0
        while i < len(text):
            ch = text[i]
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    params_text = text[1:i]
                    remainder = text[i + 1 :]
                    break
            i += 1
        else:
            params_text = text[1:]
            remainder = ""

        params = [p.strip() for p in params_text.split(",") if p.strip()]
        is_variadic = False
        if params and params[-1] == "...":
            params[-1] = "__VA_ARGS__"
            is_variadic = True
        elif params and params[-1].endswith("..."):
            # GNU named variable argument ``rest...``: ``rest`` takes the variable arguments.
            params[-1] = params[-1][: -len("...")].strip()
            is_variadic = True
        return params, remainder, is_variadic

    def _handle_include(
        self, rest: str, file_path: Optional[str]
    ) -> Optional[Tuple[str, str]]:
        match = re.match(r"\s*([<\"])([^>\"]+)[>\"]", rest)
        if not match:
            return None
        delimiter = match.group(1)
        target = match.group(2)

        candidate = self._find_header(target, delimiter == '"', file_path)
        if candidate is not None:
            if self._canonical_path(candidate) in self.pragma_once_files:
                return "", candidate
            return self._read_source_file(candidate), candidate

        if self.strict:
            raise FileNotFoundError(f"Include not found: {target}")
        if delimiter == "<":
            return f"{PRESERVED_SYSTEM_INCLUDE_SENTINEL}<{target}>", file_path or ""
        return None

    def _find_header(
        self,
        target: str,
        quoted: bool,
        file_path: Optional[str],
        after_current: bool = False,
    ) -> Optional[str]:
        """The file that ``#include "target"`` (``quoted``) or ``#include <target>`` names: a quoted header is
        looked for next to ``file_path`` first, then on the include path. ``after_current`` searches only the
        include path entries after the one that holds ``file_path``, as ``#include_next`` does; in a file that no
        entry holds it searches the whole include path, as Clang does."""
        search_paths: List[str] = []
        if quoted and file_path and not after_current:
            search_paths.append(os.path.dirname(file_path))
        paths = list(self.include_paths)
        if after_current and file_path:
            directory = self._canonical_path(os.path.dirname(file_path))
            for index, base in enumerate(paths):
                root = self._canonical_path(base)
                if directory == root or directory.startswith(root + os.sep):
                    paths = paths[index + 1 :]
                    break
        search_paths.extend(paths)
        for base in search_paths:
            candidate = os.path.join(base, target)
            if os.path.isfile(candidate):
                return candidate
        return None

    def _is_pragma_once(self, rest: str) -> bool:
        parts = self._strip_macro_comments(rest).strip().split()
        return bool(parts) and parts[0] == "once"

    def _canonical_path(self, filepath: str) -> str:
        return os.path.realpath(os.path.abspath(filepath))

    def _read_source_file(self, filepath: str) -> str:
        with open(filepath, "rb") as handle:
            data = handle.read()
        if data.startswith((b"\xff\xfe", b"\xfe\xff")):
            return data.decode("utf-16", errors="replace")
        return data.decode("utf-8-sig", errors="replace")

    def _handle_line_directive(self, rest: str) -> Optional[Tuple[int, Optional[str]]]:
        parts = rest.strip().split()
        if not parts:
            return None
        try:
            line_number = int(parts[0])
        except ValueError:
            return None
        filename = None
        match = re.search(r'"([^"\\]*(?:\\.[^"\\]*)*)"', rest)
        if match:
            filename = match.group(1).replace(r"\\", "\\").replace(r"\"", '"')
        return line_number, filename

    def _expand_macros(
        self,
        text: str,
        line_num: int,
        in_expression: bool,
        file_path: Optional[str] = None,
        disabled_macros: Optional[Set[str]] = None,
    ) -> str:
        if in_expression:
            return expand_condition(self, text, line_num, file_path, disabled_macros)
        return expand_line(self, text, line_num, file_path, disabled_macros)

    def _read_identifier(self, text: str, start: int) -> Tuple[str, int]:
        if start < 0:
            # A regex position cannot be negative; keep str indexing semantics.
            i = start
            while i < len(text) and (text[i].isalnum() or text[i] == "_"):
                i += 1
            return text[start:i], i - start
        identifier = IDENTIFIER_CONTINUATION_RE.match(text, start).group()
        return identifier, len(identifier)

    def _read_string(self, text: str, start: int) -> Tuple[str, int]:
        quote = text[start]
        i = start + 1
        while i < len(text):
            if text[i] == "\\":
                i += 2
                continue
            if text[i] == quote:
                return text[start : i + 1], i - start + 1
            i += 1
        return text[start:], len(text) - start

    def _stringize(self, value: str) -> str:
        collapsed = re.sub(r"\s+", " ", value.strip())
        escaped = collapsed.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'

"""Function header text, renaming, delimiter matching, text replacement and function-like macro calls."""

import os
import re
from typing import (
    Dict,
    List,
    Optional,
    Tuple,
)

from ..text_scanners import (
    function_parameter_start,
    mask_comments_and_literals,
    split_top_level_commas,
)
from .support import (
    CLANG_FEATURE_TEST_MACROS,
    IDENTIFIER_RE,
    METAL_ANGLE_OPERATOR_RE,
    METAL_ENTRY_FUNCTION_RE,
    METAL_STATEMENT_DELIMITER_RE,
    METAL_TEMPLATE_DELIMITER_RE,
    MetalMacroExpansionError,
)


class MetalSourceTextMixin:
    """Members of ``MetalPreprocessor``: function header text, renaming, delimiter matching, text replacement and function-like macro calls."""

    def _expand_variadic_function_parameters(
        self, source: str, variadic_bindings: Dict[str, List[str]]
    ) -> str:
        header_end = source.find("{")
        if header_end == -1:
            return source
        header = source[:header_end]
        open_paren = header.find("(")
        if open_paren == -1:
            return source
        close_paren = self._find_matching_delimiter(header, open_paren, "(", ")")
        if close_paren is None:
            return source

        parameters = self._split_top_level_commas(header[open_paren + 1 : close_paren])
        expanded_parameters: List[str] = []
        renames: Dict[str, str] = {}
        for parameter in parameters:
            pack = self._variadic_function_parameter_name(parameter)
            if pack is None:
                expanded_parameters.append(parameter)
                continue
            type_name, value_name = pack
            bound_types = variadic_bindings.get(type_name, [])
            if not bound_types:
                continue
            for index, bound_type in enumerate(bound_types):
                generated_name = f"{value_name}_{index}"
                expanded = parameter.replace("...", "")
                expanded = re.sub(
                    rf"\b{re.escape(type_name)}\b",
                    bound_type,
                    expanded,
                    count=1,
                )
                expanded = re.sub(
                    rf"\b{re.escape(value_name)}\b",
                    generated_name,
                    expanded,
                    count=1,
                )
                expanded_parameters.append(expanded.strip())
            renames[value_name] = f"{value_name}_0"

        rebuilt_header = (
            header[: open_paren + 1]
            + ", ".join(expanded_parameters)
            + header[close_paren:]
        )
        body = source[header_end:]
        if renames:
            body = self._replace_identifiers(body, renames)
        return rebuilt_header + body

    def _variadic_function_parameter_name(
        self, parameter: str
    ) -> Optional[Tuple[str, str]]:
        if "..." not in parameter:
            return None
        cleaned = parameter.replace("...", " ")
        tokens = IDENTIFIER_RE.findall(cleaned)
        if len(tokens) < 2:
            return None
        return tokens[-2], tokens[-1]

    def _function_parameter_start(self, header: str) -> Optional[int]:
        return function_parameter_start(header, self._read_string)

    def _function_parameter_list_span(self, header: str) -> Optional[Tuple[int, int]]:
        # Locate the (open, close) offsets of a function header's REAL parameter
        # list. `operator()` is special: its declarator itself contains an empty
        # `()`, so the parameter list is the paren group that FOLLOWS it rather
        # than the first top-level `(` (which the plain scanner returns). Mirrors
        # the operator-aware handling in `_parse_struct_method` so an out-of-line
        # `Ret S::operator()(params)` definition exposes its parameters for
        # call-argument inference just like an in-struct declaration does.
        operator_match = re.search(r"\boperator\s*\(\s*\)", header)
        if operator_match is not None:
            relative_start = self._function_parameter_start(
                header[operator_match.end() :]
            )
            if relative_start is None:
                return None
            paren_start = relative_start + operator_match.end()
        else:
            paren_start = self._function_parameter_start(header)
            if paren_start is None:
                return None
        paren_end = self._find_matching_delimiter(header, paren_start, "(", ")")
        if paren_end is None:
            return None
        return paren_start, paren_end

    def _header_declares_entry(self, header: str) -> bool:
        """Whether a function definition's header declares a stage entry point.

        A stage qualifier (``kernel``, ``vertex``, ...) or ``[[kernel]]`` precedes the parameter list. A parameter
        named like a qualifier (``device mlx_atomic<T>* object``), a comment before the header and a string
        literal (``[[host_name("vertex_copy")]]``) are not qualifiers, so only the masked declaration text before
        the parameter list is searched.
        """
        masked = mask_comments_and_literals(header)
        parameter_start = self._function_parameter_start(masked)
        declaration = masked if parameter_start is None else masked[:parameter_start]
        return METAL_ENTRY_FUNCTION_RE.search(declaration) is not None

    def _function_name_from_header(self, header: str) -> Optional[str]:
        paren_index = self._function_parameter_start(header)
        if paren_index is None:
            return None
        before_params = header[:paren_index].rstrip()
        match = re.search(r"([A-Za-z_][A-Za-z0-9_:]*)\s*$", before_params)
        if match is None:
            return None
        return match.group(1).split("::")[-1]

    def _rename_function_definition(
        self, source: str, old_name: str, new_name: str
    ) -> str:
        body_start = source.find("{")
        header = source if body_start == -1 else source[:body_start]
        pattern = re.compile(rf"\b{re.escape(old_name)}\s*(?=\()")
        matches = list(pattern.finditer(header))
        if not matches:
            return source
        match = matches[-1]
        return source[: match.start()] + new_name + source[match.end() :]

    def _rename_struct_definition(
        self,
        source: str,
        old_name: str,
        new_name: str,
        *,
        strip_specialization_arguments: bool = False,
    ) -> str:
        pattern = re.compile(rf"\b(struct|class|union)\s+{re.escape(old_name)}\b")
        match = pattern.search(source)
        if match is None:
            return source
        declaration_end = match.end()
        if strip_specialization_arguments:
            angle_start = declaration_end
            while angle_start < len(source) and source[angle_start].isspace():
                angle_start += 1
            if angle_start < len(source) and source[angle_start] == "<":
                angle_end = self._find_matching_angle(source, angle_start)
                if angle_end is None:
                    return source
                declaration_end = angle_end + 1
        return (
            source[: match.start()]
            + f"{match.group(1)} {new_name}"
            + source[declaration_end:]
        )

    def _rename_materialized_struct_constructors(
        self, source: str, old_name: str, new_name: str
    ) -> str:
        """Rename constructor declarators with a materialized template owner."""

        definitions = self._find_concrete_struct_definitions(source)
        owner = next(
            (
                item
                for item in definitions
                if item.name == old_name or item.name == new_name
            ),
            None,
        )
        if owner is None:
            return source

        replacements: List[Tuple[int, int, str]] = []
        pattern = re.compile(rf"\b{re.escape(old_name)}\s*(?=\()")
        for constructor in owner.constructors:
            if constructor.span is None:
                continue
            start, end = constructor.span
            match = pattern.search(source, start, end)
            if match is not None:
                replacements.append((match.start(), match.end(), new_name))
        return self._apply_text_replacements(source, replacements)

    def _materialized_function_identifier(self, host_name: str, fallback: str) -> str:
        identifier = re.sub(r"\W", "_", host_name)
        if not identifier or identifier[0].isdigit():
            identifier = f"{fallback}_{identifier}" if identifier else fallback
        return identifier

    def _evaluate_metal_string_expression(self, expression: str) -> str:
        strings = re.findall(r'"((?:\\.|[^"\\])*)"|\'((?:\\.|[^\'\\])*)\'', expression)
        if strings:
            return "".join(
                self._unescape_metal_string(double or single)
                for double, single in strings
            )
        return expression.strip()

    def _unescape_metal_string(self, value: str) -> str:
        return bytes(value, "utf-8").decode("unicode_escape")

    def _split_top_level_commas(self, text: str) -> List[str]:
        return split_top_level_commas(text, self._read_string)

    def _replace_identifiers(
        self,
        text: str,
        replacements: Dict[str, str],
        *,
        skip_template_names: bool = False,
    ) -> str:
        # With ``skip_template_names`` an identifier followed by ``<`` is kept: a
        # plain alias takes no template arguments, so ``Op<`` after
        # ``using Op = Op<T>;`` names the template the alias refers to.
        result = ""
        i = 0
        while i < len(text):
            if text[i] in "\"'":
                literal, consumed = self._read_string(text, i)
                result += literal
                i += consumed
                continue
            if text.startswith("//", i):
                end = text.find("\n", i)
                if end == -1:
                    result += text[i:]
                    break
                result += text[i:end]
                i = end
                continue
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
                if self._is_member_identifier_context(text, i) or (
                    skip_template_names
                    and ident in replacements
                    and text[i + consumed :].lstrip().startswith("<")
                ):
                    result += ident
                else:
                    # An identifier that is not replaced is copied as written.
                    result += replacements[ident] if ident in replacements else ident
                i += consumed
                continue
            result += text[i]
            i += 1
        return result

    def _is_member_identifier_context(self, text: str, index: int) -> bool:
        def previous_code_character(before: int) -> int:
            previous = before - 1
            while previous >= 0:
                while previous >= 0 and text[previous].isspace():
                    previous -= 1
                if previous < 0:
                    return -1

                # Scan only the current physical line. This detects a preceding
                # `//` comment (and same-line literals/block comments) without a
                # whole-source pass in this identifier-level hot path.
                line_start = text.rfind("\n", 0, previous + 1) + 1
                line_prefix = text[line_start : previous + 1]
                relative_previous = previous - line_start
                for ignored_start, ignored_end in self._scan_comment_and_literal_spans(
                    line_prefix
                ):
                    if not (ignored_start <= relative_previous < ignored_end):
                        continue
                    if line_prefix.startswith(("//", "/*"), ignored_start):
                        previous = line_start + ignored_start - 1
                        break
                    # A string/character literal is a real token, not whitespace.
                    return -1
                else:
                    # A block comment can begin on an earlier line. If its closing
                    # delimiter is the previous token, skip the matching opener;
                    # comments are non-nesting in Metal/C++ source.
                    if previous >= 1 and text[previous - 1 : previous + 1] == "*/":
                        opening = text.rfind("/*", 0, previous - 1)
                        prior_close = text.rfind("*/", 0, previous - 1)
                        if opening > prior_close:
                            previous = opening - 1
                            continue
                    return previous
            return -1

        previous = previous_code_character(index)
        if previous < 0:
            return False
        # ``...`` ends a GNU range designator, ``[0 ... N - 1]``, or opens a fold, ``(... + xs)``: the
        # identifier after it is no member.
        if text[previous] == "." and text[max(0, previous - 2) : previous + 1] != "...":
            return True
        if previous >= 1 and text[previous - 1 : previous + 1] in {"::", "->"}:
            return True

        # A dependent member-template call uses a disambiguating ``template``
        # token between its object/scope qualifier and member name, for example
        # ``value.template convert<T>()`` or ``pointer->template convert<T>()``.
        # Treat the selected identifier as a member in those forms too; otherwise
        # free-template materialization and alias replacement can claim it by
        # basename when an unrelated free function has the same name.
        word_end = previous + 1
        word_start = word_end
        while word_start > 0 and (
            text[word_start - 1].isalnum() or text[word_start - 1] == "_"
        ):
            word_start -= 1
        if text[word_start:word_end] != "template":
            return False
        qualifier_end = previous_code_character(word_start)
        if qualifier_end < 0:
            return False
        if text[qualifier_end] == ".":
            return True
        return qualifier_end >= 1 and text[qualifier_end - 1 : qualifier_end + 1] in {
            "::",
            "->",
        }

    def _find_matching_angle(self, code: str, start: int) -> Optional[int]:
        """Find a template-id close without consuming value-expression operators.

        A non-type template argument can contain a parenthesized comparison such
        as ``Looped<1, int, (1 > 2)>``.  The generic delimiter matcher treats the
        comparison ``>`` as the template close, truncating the final argument.
        Track the delimiter context in which each angle was opened so a ``>``
        nested in parentheses/brackets/braces cannot close an outer template-id.
        Any unmatched ``<`` opened inside a closing value-expression delimiter is
        a comparison and is discarded; a valid nested template-id closes first.

        A template argument list is balanced, so a closing delimiter that has no
        opener after ``start``, or a ``;`` outside braces, ends the search: the
        ``<`` at ``start`` is a comparison, and scanning on would cost the rest
        of the module for every comparison a caller probes.
        """
        if start < 0 or start >= len(code) or code[start] != "<":
            return None

        paren_depth = 0
        bracket_depth = 0
        brace_depth = 0
        angle_contexts = [(0, 0, 0)]
        index = start + 1
        while index < len(code):
            match = METAL_TEMPLATE_DELIMITER_RE.search(code, index)
            if match is None:
                return None
            index = match.start()
            if code[index] in "\"'":
                _literal, consumed = self._read_string(code, index)
                index += consumed
                continue
            if code.startswith("//", index):
                end = code.find("\n", index)
                if end == -1:
                    return None
                index = end + 1
                continue
            if code.startswith("/*", index):
                end = code.find("*/", index + 2)
                if end == -1:
                    return None
                index = end + 2
                continue

            character = code[index]
            if code.startswith("<<", index):
                index += 2
                continue
            context = (paren_depth, bracket_depth, brace_depth)
            if character == "(":
                paren_depth += 1
            elif character == ")":
                if paren_depth == 0:
                    return None
                while len(angle_contexts) > 1 and angle_contexts[-1][0] >= paren_depth:
                    angle_contexts.pop()
                paren_depth -= 1
            elif character == "[":
                bracket_depth += 1
            elif character == "]":
                if bracket_depth == 0:
                    return None
                while (
                    len(angle_contexts) > 1 and angle_contexts[-1][1] >= bracket_depth
                ):
                    angle_contexts.pop()
                bracket_depth -= 1
            elif character == "{":
                brace_depth += 1
            elif character == "}":
                if brace_depth == 0:
                    return None
                while len(angle_contexts) > 1 and angle_contexts[-1][2] >= brace_depth:
                    angle_contexts.pop()
                brace_depth -= 1
            elif character == ";" and brace_depth == 0:
                return None
            elif character == "<":
                angle_contexts.append(context)
            elif character == ">" and angle_contexts[-1] == context:
                angle_contexts.pop()
                if not angle_contexts:
                    return index
            index += 1
        return None

    def _find_matching_brace(self, code: str, start: int) -> Optional[int]:
        end = self._find_matching_delimiter(code, start, "{", "}")
        return None if end is None else end + 1

    def _find_matching_delimiter(
        self, code: str, start: int, opener: str, closer: str
    ) -> Optional[int]:
        if start < 0 or start >= len(code) or code[start] != opener:
            return None
        delimiters = re.compile("[" + re.escape(opener + closer + "\"'/") + "]")
        depth = 0
        i = start
        while i < len(code):
            match = delimiters.search(code, i)
            if match is None:
                return None
            i = match.start()
            if code[i] in "\"'":
                _literal, consumed = self._read_string(code, i)
                i += consumed
                continue
            if code.startswith("//", i):
                end = code.find("\n", i)
                if end == -1:
                    return None
                i = end + 1
                continue
            if code.startswith("/*", i):
                end = code.find("*/", i + 2)
                if end == -1:
                    return None
                i = end + 2
                continue
            if code[i] == opener:
                depth += 1
            elif code[i] == closer:
                depth -= 1
                if depth == 0:
                    return i
            i += 1
        return None

    def _find_next_top_level_char(
        self,
        code: str,
        start: int,
        target: str,
        *,
        track_angles: bool = True,
        stop_at: str = "",
    ) -> Optional[int]:
        # Skip identifiers and whitespace without allocating a source substring.
        # Operator names still need their own angle-bracket handling below.
        delimiters = re.compile(
            "["
            + re.escape("()[]\"'/" + target + stop_at + ("<>" if track_angles else ""))
            + "]"
            + (r"|\boperator" if track_angles else "")
        )
        paren_depth = 0
        bracket_depth = 0
        angle_depth = 0
        i = start
        while i < len(code):
            match = delimiters.search(code, i)
            if match is None:
                return None
            i = match.start()
            ch = code[i]
            if ch == '"' or ch == "'":
                _literal, consumed = self._read_string(code, i)
                i += consumed
                continue
            if ch == "/":
                if code.startswith("//", i):
                    end = code.find("\n", i)
                    if end == -1:
                        return None
                    i = end + 1
                    continue
                if code.startswith("/*", i):
                    end = code.find("*/", i + 2)
                    if end == -1:
                        return None
                    i = end + 2
                    continue
            if (
                (ch == target or ch in stop_at)
                and paren_depth == 0
                and bracket_depth == 0
                and (not track_angles or angle_depth == 0)
            ):
                # A terminator excludes bodies belonging to later declarations.
                return i if ch == target else None
            if track_angles and ch == "o":
                operator_name = METAL_ANGLE_OPERATOR_RE.match(code, i)
                if operator_name is not None:
                    # An overloaded operator name does not open a template-id.
                    i = operator_name.end()
                    continue
            if ch == "(":
                paren_depth += 1
            elif ch == ")":
                paren_depth = max(0, paren_depth - 1)
            elif ch == "[":
                bracket_depth += 1
            elif ch == "]":
                bracket_depth = max(0, bracket_depth - 1)
            elif track_angles and ch == "<":
                angle_depth += 1
            elif track_angles and ch == ">":
                angle_depth = max(0, angle_depth - 1)
            i += 1
        return None

    def _statement_end(self, code: str, start: int) -> Optional[int]:
        paren_depth = 0
        bracket_depth = 0
        angle_depth = 0
        brace_depth = 0
        i = start
        while i < len(code):
            match = METAL_STATEMENT_DELIMITER_RE.search(code, i)
            if match is None:
                return None
            i = match.start()
            ch = code[i]
            if ch == '"' or ch == "'":
                _literal, consumed = self._read_string(code, i)
                i += consumed
                continue
            if ch == "/":
                if code.startswith("//", i):
                    end = code.find("\n", i)
                    if end == -1:
                        return len(code)
                    i = end + 1
                    continue
                if code.startswith("/*", i):
                    end = code.find("*/", i + 2)
                    if end == -1:
                        return None
                    i = end + 2
                    continue
            if (
                ch == ";"
                and paren_depth == 0
                and bracket_depth == 0
                and angle_depth == 0
                and brace_depth == 0
            ):
                return i + 1
            if ch == "(":
                paren_depth += 1
            elif ch == ")":
                paren_depth = max(0, paren_depth - 1)
            elif ch == "[":
                bracket_depth += 1
            elif ch == "]":
                bracket_depth = max(0, bracket_depth - 1)
            elif ch == "<":
                angle_depth += 1
            elif ch == ">":
                angle_depth = max(0, angle_depth - 1)
            elif ch == "{":
                brace_depth += 1
            elif ch == "}":
                brace_depth = max(0, brace_depth - 1)
            i += 1
        return None

    def _apply_text_replacements(
        self, code: str, replacements: List[Tuple[int, int, str]]
    ) -> str:
        replacements = sorted(replacements, key=lambda item: item[0])
        result = []
        pos = 0
        for start, end, replacement in replacements:
            if start < pos:
                continue
            result.append(code[pos:start])
            result.append(replacement)
            pos = end
        result.append(code[pos:])
        return "".join(result)

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

    def _join_multiline_function_macro_call(
        self, lines: List[str], start: int
    ) -> Tuple[str, int]:
        line = lines[start]
        consumed = 1
        while self._find_incomplete_function_macro_call(
            line
        ) is not None and start + consumed < len(lines):
            next_line = lines[start + consumed]
            if next_line.lstrip().startswith("#"):
                break
            # Preserve a physical line boundary so a // comment in one argument
            # cannot consume the remaining invocation.
            line += "\n" + next_line.lstrip()
            consumed += 1
        return line, consumed

    def _find_incomplete_function_macro_call(
        self, text: str
    ) -> Optional[Tuple[str, int]]:
        i = 0
        while i < len(text):
            if text[i] in "\"'":
                _literal, consumed = self._read_string(text, i)
                i += consumed
                continue
            if text.startswith("//", i):
                newline = text.find("\n", i + 2)
                if newline == -1:
                    return None
                i = newline + 1
                continue
            if text.startswith("/*", i):
                end = text.find("*/", i + 2)
                if end == -1:
                    return None
                i = end + 2
                continue
            if text[i].isalpha() or text[i] == "_":
                ident_start = i
                ident, consumed = self._read_identifier(text, i)
                i += consumed
                macro = self.macros.get(ident)
                if macro is None:
                    continue
                if not macro.is_function_like():
                    continue
                j = i
                while j < len(text) and text[j].isspace():
                    j += 1
                if j < len(text) and text[j] == "(":
                    call_end = self._function_macro_call_end(text, j)
                    if call_end is None:
                        return ident, ident_start
                    i = call_end
                continue
            i += 1
        return None

    def _function_macro_call_end(self, text: str, start: int) -> Optional[int]:
        depth = 0
        i = start
        while i < len(text):
            if text[i] in "\"'":
                _literal, consumed = self._read_string(text, i)
                i += consumed
                continue
            if text.startswith("//", i):
                newline = text.find("\n", i + 2)
                if newline == -1:
                    return None
                i = newline + 1
                continue
            if text.startswith("/*", i):
                end = text.find("*/", i + 2)
                if end == -1:
                    return None
                i = end + 2
                continue
            if text[i] == "(":
                depth += 1
            elif text[i] == ")":
                depth -= 1
                if depth == 0:
                    return i + 1
            i += 1
        return None

    def _parse_macro_args(self, text: str, start: int) -> Tuple[List[str], int]:
        assert text[start] == "("
        args: List[str] = []
        current: List[str] = []
        depth = 0
        i = start
        while i < len(text):
            if text[i] in "\"'":
                literal, consumed = self._read_string(text, i)
                current.append(literal)
                i += consumed
                continue
            if text.startswith("//", i):
                newline = text.find("\n", i + 2)
                if newline == -1:
                    return args, 0
                current.append(" ")
                i = newline + 1
                continue
            if text.startswith("/*", i):
                end = text.find("*/", i + 2)
                if end == -1:
                    return args, 0
                current.append(" ")
                i = end + 2
                continue

            ch = text[i]
            if ch == "(":
                depth += 1
                if depth > 1:
                    current.append(ch)
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    args.append("".join(current).strip())
                    return args, i - start + 1
                current.append(ch)
            elif ch == "," and depth == 1:
                args.append("".join(current).strip())
                current = []
            else:
                current.append(ch)
            i += 1
        return args, 0

    @staticmethod
    def _raise_incomplete_function_macro_call(
        text: str,
        macro_name: str,
        start: int,
        line_num: int,
        file_path: Optional[str],
    ) -> None:
        line_offset = text.count("\n", 0, start)
        previous_newline = text.rfind("\n", 0, start)
        line = line_num + line_offset
        column = start - previous_newline
        location = {
            "line": line,
            "column": column,
            "length": len(macro_name),
            "endLine": line,
            "endColumn": column + len(macro_name),
        }
        if file_path is not None:
            location["file"] = file_path
        raise MetalMacroExpansionError(
            f"Cannot expand function-like Metal macro '{macro_name}': "
            "the invocation is not terminated before the end of its source region.",
            macro_name=macro_name,
            reason="unterminated-function-macro-invocation",
            source_location=location,
            suggested_action=(
                "Close the invocation with balanced parentheses before the next "
                "preprocessor directive or end of file."
            ),
        )

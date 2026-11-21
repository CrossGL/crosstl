"""Standard type aliases, template parameter lists, defaults and constraints, and namespace scans."""

import re
from copy import deepcopy
from typing import (
    Dict,
    List,
    Optional,
    Set,
    Tuple,
)

from ..source_analysis_cache import memoized
from ..template_function_parameters import template_function_pointer_parameter
from ..text_scanners import strip_comments_outside_strings
from .support import (
    IDENTIFIER_RE,
    METAL_NAMESPACE_DELIMITER_RE,
    SOURCE_ANALYSIS_CACHE_LIMIT,
    _MetalTemplateFunction,
    _TemplateParameter,
)


class MetalTemplateParameterMixin:
    """Members of ``MetalPreprocessor``: standard type aliases, template parameter lists, defaults and constraints, and namespace scans."""

    def _metal_unqualified_remove_cv_t_available(self, code: str) -> bool:
        """Whether unqualified ``remove_cv_t`` can denote Metal's stdlib alias.

        Only a global ``using namespace metal`` admits the unqualified spelling.
        Any source type declaration with the same name disables this shortcut for
        the whole materialization snapshot: resolving through a possibly
        possibly shadowed declaration would be a silent specialization guess.
        """
        return memoized(
            self._source_analysis(code).values,
            "unqualified_remove_cv_t_available",
            lambda: self._scan_unqualified_remove_cv_t_available(code),
        )

    def _scan_unqualified_remove_cv_t_available(self, code: str) -> bool:
        ignored_spans = self._find_comment_and_literal_spans(code)
        masked = self._mask_comments_and_literals(code)
        if re.search(
            r"\b(?:struct|class|union)\s+remove_cv_t\b"
            r"|\busing\s+remove_cv_t\s*="
            r"|\btypedef\b[^;{}]*\bremove_cv_t\s*;",
            masked,
        ):
            return False
        position = 0
        while True:
            match = re.search(r"\btemplate\s*<", code[position:])
            if match is None:
                break
            start = position + match.start()
            position = start + len("template")
            if self._containing_span(start, ignored_spans) is not None:
                continue
            angle_start = code.find("<", start)
            angle_end = self._find_matching_template_param_angle(code, angle_start)
            if angle_end is None:
                continue
            cursor = self._skip_cpp_trivia(code, angle_end + 1)
            alias_match = re.match(
                r"using\s+remove_cv_t\s*=",
                code[cursor:],
            )
            if alias_match is not None:
                return False

        lexical_scopes = self._find_lexical_brace_scopes(code)
        for match in re.finditer(
            r"\busing\s+namespace\s+(?:::)?metal\s*;",
            code,
        ):
            if self._containing_span(match.start(), ignored_spans) is not None:
                continue
            if self._innermost_lexical_scope(
                lexical_scopes,
                match.start(),
                len(code),
            ) == (0, len(code)):
                return True
        return False

    def _remove_top_level_cv_qualifiers(self, type_text: str) -> Optional[str]:
        """Resolve the value of a concrete ``metal::remove_cv_t`` safely.

        Pointee qualifiers precede the final ``*`` and must remain intact.  For a
        pointer only trailing cv tokens qualify the pointer itself; references
        are already non-cv types.  Array declarators stay unresolved because
        their element-cv rules cannot be reconstructed from this lightweight
        textual type view without guessing.
        """

        text = self._normalize_template_argument_text(type_text)
        if not text:
            return None
        qualifier_spans: List[Tuple[int, int]] = []
        pointer_positions: List[int] = []
        has_reference = False
        has_array = False
        angle_depth = paren_depth = bracket_depth = 0
        position = 0
        while position < len(text):
            char = text[position]
            if char == "<":
                angle_depth += 1
            elif char == ">":
                angle_depth = max(0, angle_depth - 1)
            elif char == "(":
                paren_depth += 1
            elif char == ")":
                paren_depth = max(0, paren_depth - 1)
            elif char == "[" and angle_depth == 0 and paren_depth == 0:
                has_array = True
                bracket_depth += 1
            elif char == "]" and bracket_depth:
                bracket_depth -= 1
            elif angle_depth == paren_depth == bracket_depth == 0:
                if char == "*":
                    pointer_positions.append(position)
                elif char == "&":
                    has_reference = True
            position += 1

        for match in IDENTIFIER_RE.finditer(text):
            prefix = text[: match.start()]
            if (
                prefix.count("<") == prefix.count(">")
                and prefix.count("(") == prefix.count(")")
                and prefix.count("[") == prefix.count("]")
                and match.group(0) in {"const", "volatile"}
            ):
                qualifier_spans.append((match.start(), match.end()))
        if not qualifier_spans or has_reference:
            return text
        if has_array:
            return None
        if pointer_positions:
            final_pointer = pointer_positions[-1]
            qualifier_spans = [
                span for span in qualifier_spans if span[0] > final_pointer
            ]
            if not qualifier_spans:
                return text
        resolved = self._apply_text_replacements(
            text,
            [(start, end, "") for start, end in qualifier_spans],
        )
        return self._normalize_template_argument_text(resolved) or None

    def _canonicalize_metal_standard_type_aliases(
        self,
        type_text: str,
        *,
        allow_unqualified_remove_cv_t: bool = False,
        dependent_identifiers: Optional[Set[str]] = None,
    ) -> str:
        """Recursively canonicalize proven concrete Metal stdlib type aliases.

        Today this recognizes ``metal::remove_cv_t`` (and its unqualified form
        only under a proven global namespace import).  Unknown, dependent,
        malformed, or unsupported aliases are retained verbatim so downstream
        residual-template diagnostics remain fail closed.
        """

        resolved = self._normalize_template_argument_text(type_text)
        if not resolved:
            return resolved
        dependent = set(dependent_identifiers or ())
        pattern = re.compile(
            r"(?<![A-Za-z0-9_:])"
            r"(?P<qualified>(?:(?:::)?metal\s*::\s*)?)"
            r"remove_cv_t\s*<"
        )
        max_iterations = resolved.count("<") + 1
        for _ in range(max_iterations):
            replacement: Optional[Tuple[int, int, str]] = None
            matches = list(pattern.finditer(resolved))
            for match in reversed(matches):
                qualified = bool(match.group("qualified"))
                if not qualified and not allow_unqualified_remove_cv_t:
                    continue
                angle_start = resolved.rfind("<", match.start(), match.end())
                angle_end = self._find_matching_angle(resolved, angle_start)
                if angle_end is None:
                    continue
                arguments = self._split_top_level_commas(
                    resolved[angle_start + 1 : angle_end]
                )
                if len(arguments) != 1:
                    continue
                argument = self._canonicalize_metal_standard_type_aliases(
                    arguments[0],
                    allow_unqualified_remove_cv_t=(allow_unqualified_remove_cv_t),
                    dependent_identifiers=dependent,
                )
                if set(IDENTIFIER_RE.findall(argument)) & dependent:
                    continue
                target = self._remove_top_level_cv_qualifiers(argument)
                if target is None:
                    continue
                replacement = (match.start(), angle_end + 1, target)
                break
            if replacement is None:
                break
            updated = self._apply_text_replacements(resolved, [replacement])
            if updated == resolved:
                break
            resolved = self._normalize_template_argument_text(updated)
        return resolved

    def _strip_template_argument_comments(self, value: str) -> str:
        return strip_comments_outside_strings(value, self._read_string)

    # ------------------------------------------------------------------ #
    # `<` disambiguation for template-PARAMETER lists.                    #
    # ------------------------------------------------------------------ #
    # A template-parameter list may contain a SFINAE non-type parameter whose
    # constraint embeds comparison operators, e.g.
    #   template <typename T, metal::enable_if_t<sizeof(T) < 8, bool> = true>
    # The generic `_find_matching_angle` / `_split_top_level_commas` treat the
    # comparison `<` in `sizeof(T) < 8` as a template-open angle, so the angle
    # nesting never balances and the parameter list (and the method it heads) is
    # dropped. The helpers below scan a template-parameter list treating a `<`
    # as a template-argument opener ONLY when it immediately follows a
    # template-name identifier character (e.g. `enable_if_t<`, `is_integral_v<`);
    # a `<`/`>` used as a comparison (`sizeof(T) < 8`, `<=`, `>=`) does NOT change
    # angle depth. These are deliberately SEPARATE from the template-ARGUMENT
    # handling so other `<` parsing is left untouched.

    def _find_matching_template_param_angle(
        self, code: str, start: int
    ) -> Optional[int]:
        # Index of the `>` that closes the template-parameter-list `<` at `start`,
        # ignoring comparison `<`/`>` inside SFINAE constraints. Returns None when
        # unbalanced.
        if start < 0 or start >= len(code) or code[start] != "<":
            return None
        # The `<` at `start` is the parameter-list opener by construction (it is
        # the `<` of `template <`), so it always counts as depth 1 regardless of
        # what precedes it; the open-disambiguation only applies to later `<`.
        depth = 1
        i = start + 1
        n = len(code)
        while i < n:
            ch = code[i]
            if ch in "\"'":
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
            if ch == "<":
                if code[i + 1 : i + 2] == "=":
                    # `<=` comparison — never an angle; skip both chars.
                    i += 2
                    continue
                if self._template_param_angle_is_open(code, i):
                    depth += 1
                i += 1
                continue
            if ch == ">":
                if code[i + 1 : i + 2] == "=":
                    # `>=` comparison — never an angle; skip both chars.
                    i += 2
                    continue
                # `->` is a member access, not an angle close.
                if i > 0 and code[i - 1] == "-":
                    i += 1
                    continue
                if depth > 0:
                    depth -= 1
                    if depth == 0:
                        return i
                i += 1
                continue
            i += 1
        return None

    def _template_param_angle_is_open(self, code: str, index: int) -> bool:
        # A `<` opens a template-argument list (rather than being a comparison)
        # when the preceding significant token ends in an identifier. Comments
        # and whitespace do not change C++ token adjacency, so skip both here.
        # Comparisons such as `sizeof(T) < 8` remain distinguishable because the
        # preceding token ends in `)`.
        previous = index - 1
        skipped_trivia = False
        while previous >= 0:
            while previous >= 0 and code[previous].isspace():
                previous -= 1
                skipped_trivia = True
            if previous < 0:
                return False
            if previous >= 1 and code[previous - 1 : previous + 1] == "*/":
                comment_start = code.rfind("/*", 0, previous - 1)
                if comment_start == -1:
                    return False
                previous = comment_start - 1
                skipped_trivia = True
                continue
            line_start = code.rfind("\n", 0, previous + 1) + 1
            comment_start = code.find("//", line_start, previous + 1)
            if comment_start != -1:
                previous = comment_start - 1
                skipped_trivia = True
                continue
            break
        prev = code[previous]
        if not (prev.isalnum() or prev == "_"):
            return False
        if not skipped_trivia:
            return True

        parameter_start = max(
            code.rfind(",", 0, previous + 1),
            code.rfind("<", 0, previous + 1),
        )
        parameter_prefix = code[parameter_start + 1 : previous + 1]
        parameter_prefix = re.sub(
            r"/\*.*?\*/|//[^\n]*",
            " ",
            parameter_prefix,
            flags=re.DOTALL,
        )
        return (
            re.match(
                r"\s*(?:typename|class)\s+" r"[A-Za-z_][A-Za-z0-9_]*(?:\s*\.\.\.)?\s*=",
                parameter_prefix,
            )
            is not None
        )

    def _split_template_parameter_list(self, text: str) -> List[str]:
        # Top-level comma split of a template-parameter list that respects the
        # SFINAE `<` disambiguation above (so `enable_if_t<sizeof(T) < 8, bool>`
        # stays one parameter while the comma separating two parameters splits).
        parts: List[str] = []
        current = ""
        paren_depth = 0
        bracket_depth = 0
        brace_depth = 0
        angle_depth = 0
        i = 0
        n = len(text)
        while i < n:
            ch = text[i]
            if ch in "\"'":
                literal, consumed = self._read_string(text, i)
                current += literal
                i += consumed
                continue
            if text.startswith("//", i):
                end = text.find("\n", i + 2)
                if end == -1:
                    current += text[i:]
                    break
                current += text[i : end + 1]
                i = end + 1
                continue
            if text.startswith("/*", i):
                end = text.find("*/", i + 2)
                if end == -1:
                    current += text[i:]
                    break
                current += text[i : end + 2]
                i = end + 2
                continue
            if ch == "(":
                paren_depth += 1
            elif ch == ")":
                paren_depth = max(0, paren_depth - 1)
            elif ch == "[":
                bracket_depth += 1
            elif ch == "]":
                bracket_depth = max(0, bracket_depth - 1)
            elif ch == "{":
                brace_depth += 1
            elif ch == "}":
                brace_depth = max(0, brace_depth - 1)
            elif ch == "<":
                if text[i + 1 : i + 2] == "=":
                    current += "<="
                    i += 2
                    continue
                if self._template_param_angle_is_open(text, i):
                    angle_depth += 1
            elif ch == ">":
                if text[i + 1 : i + 2] == "=":
                    current += ">="
                    i += 2
                    continue
                if not (i > 0 and text[i - 1] == "-") and angle_depth > 0:
                    angle_depth -= 1
            elif (
                ch == ","
                and paren_depth == 0
                and bracket_depth == 0
                and brace_depth == 0
                and angle_depth == 0
            ):
                parts.append(current.strip())
                current = ""
                i += 1
                continue
            current += ch
            i += 1
        if current.strip():
            parts.append(current.strip())
        return parts

    def _parse_template_parameter_list(self, text: str) -> List[_TemplateParameter]:
        # Parse a template-parameter list into structured records, classifying
        # each entry as a bindable type parameter or an anonymous SFINAE
        # constraint. A non-type parameter that DOES name a bindable value (e.g.
        # `int N`) is recorded as a bindable parameter too (existing behavior for
        # `<typename T, int N>`); only an ANONYMOUS non-type parameter (no
        # trailing declarator name) becomes a constraint.
        records: List[_TemplateParameter] = []
        for raw in self._split_template_parameter_list(text):
            parameter = raw.strip()
            if not parameter:
                continue
            default: Optional[str] = None
            declarator = parameter
            eq = self._template_parameter_default_split(parameter)
            if eq is not None:
                declarator, default = eq
            declarator_without_comments = re.sub(
                r"/\*.*?\*/",
                " ",
                declarator,
                flags=re.DOTALL,
            )
            declarator_without_comments = re.sub(
                r"//[^\n]*", " ", declarator_without_comments
            )
            is_variadic = "..." in declarator_without_comments
            declarator_no_pack = declarator_without_comments.replace("...", " ")
            tokens = IDENTIFIER_RE.findall(declarator_no_pack)
            if not tokens:
                continue
            if tokens[0] in {"typename", "class"}:
                # `typename T` / `class T` (optionally `typename...`). A trailing
                # name is bindable. Retain anonymous parameters so struct
                # specialization can preserve their position and default.
                records.append(
                    _TemplateParameter(
                        name=tokens[-1] if len(tokens) >= 2 else None,
                        is_type_parameter=True,
                        is_variadic=is_variadic,
                        default=default,
                    )
                )
                continue
            # A function pointer or reference parameter,
            # `void (*load)(device const T *, short, thread U &)`, names itself
            # inside its declarator, not at the end of it.
            function_declarator = template_function_pointer_parameter(
                declarator_no_pack
            )
            if function_declarator is not None:
                name, declared_type = function_declarator
                records.append(
                    _TemplateParameter(
                        name=name,
                        is_type_parameter=False,
                        is_variadic=is_variadic,
                        default=default,
                        declared_type=declared_type,
                    )
                )
                continue
            # A non-type parameter. If its declarator ends in a bindable name
            # (`int N`) keep it as a (non-type) bindable parameter; otherwise it
            # is an anonymous SFINAE constraint (`metal::enable_if_t<...> [= v]`).
            if self._non_type_parameter_has_name(declarator_no_pack):
                name = tokens[-1]
                name_match = re.search(rf"\b{re.escape(name)}\s*$", declarator_no_pack)
                declared_type = ""
                if name_match is not None:
                    declared_type = self._normalize_inferred_type(
                        declarator_no_pack[: name_match.start()]
                    )
                records.append(
                    _TemplateParameter(
                        name=name,
                        is_type_parameter=False,
                        is_variadic=is_variadic,
                        default=default,
                        declared_type=declared_type or None,
                    )
                )
                continue
            records.append(
                _TemplateParameter(
                    name=None,
                    is_type_parameter=False,
                    is_variadic=is_variadic,
                    default=default,
                    constraint_text=declarator.strip(),
                )
            )
        return records

    def _template_parameter_default_split(
        self, parameter: str
    ) -> Optional[Tuple[str, str]]:
        # Split a template parameter on its TOP-LEVEL `=` default separator,
        # ignoring comments, `=` inside angles/parens/brackets, and the
        # `==`/`<=`/`>=`/`!=` comparison operators that appear in SFINAE
        # constraints. Returns
        # (declarator, default) or None when there is no default.
        paren_depth = 0
        bracket_depth = 0
        brace_depth = 0
        angle_depth = 0
        i = 0
        n = len(parameter)
        while i < n:
            if parameter.startswith("//", i):
                newline = parameter.find("\n", i + 2)
                if newline == -1:
                    break
                i = newline + 1
                continue
            if parameter.startswith("/*", i):
                comment_end = parameter.find("*/", i + 2)
                if comment_end == -1:
                    break
                i = comment_end + 2
                continue
            ch = parameter[i]
            if ch in "\"'":
                _literal, consumed = self._read_string(parameter, i)
                i += consumed
                continue
            if ch == "(":
                paren_depth += 1
            elif ch == ")":
                paren_depth = max(0, paren_depth - 1)
            elif ch == "[":
                bracket_depth += 1
            elif ch == "]":
                bracket_depth = max(0, bracket_depth - 1)
            elif ch == "{":
                brace_depth += 1
            elif ch == "}":
                brace_depth = max(0, brace_depth - 1)
            elif ch == "<":
                if parameter[i + 1 : i + 2] == "=":
                    i += 2
                    continue
                if self._template_param_angle_is_open(parameter, i):
                    angle_depth += 1
            elif ch == ">":
                if parameter[i + 1 : i + 2] == "=":
                    i += 2
                    continue
                if not (i > 0 and parameter[i - 1] == "-") and angle_depth > 0:
                    angle_depth -= 1
            elif ch == "=":
                # Skip comparison operators `==`, `!=`, `<=`, `>=`.
                nxt = parameter[i + 1 : i + 2]
                prv = parameter[i - 1] if i > 0 else ""
                if nxt == "=" or prv in "=!<>":
                    i += 2 if nxt == "=" else 1
                    continue
                if (
                    paren_depth == 0
                    and bracket_depth == 0
                    and brace_depth == 0
                    and angle_depth == 0
                ):
                    return parameter[:i].strip(), parameter[i + 1 :].strip()
            i += 1
        return None

    def _non_type_parameter_has_name(self, declarator: str) -> bool:
        # A non-type template parameter `Type name` ends in a bindable identifier
        # that is NOT part of a trailing template-id. An anonymous SFINAE
        # parameter ends in `>` (the closing angle of `enable_if_t<...>`), so it
        # has no trailing name. Conservative: only a declarator whose final
        # top-level token is a bare identifier (not immediately following a `>`
        # close of its own type) is treated as named.
        text = declarator.strip()
        if not text:
            return False
        if text.endswith(">"):
            return False
        # The last token must be an identifier preceded by type text (so a single
        # token such as `bool` — a lone type with no name — is NOT a named
        # parameter; that is the `enable_if_t<...>` value-type leftover case).
        match = re.search(r"([A-Za-z_][A-Za-z0-9_]*)\s*$", text)
        if match is None:
            return False
        before = text[: match.start()].strip()
        return bool(before)

    def _template_parameter_names(self, template_parameters: str) -> List[str]:
        # Bindable parameter names only: type parameters (`typename T`) and named
        # non-type parameters (`int N`). Anonymous SFINAE constraint parameters
        # are deliberately EXCLUDED — they are not bound from call arguments.
        return [
            parameter.name
            for parameter in self._parse_template_parameter_list(template_parameters)
            if parameter.name is not None
        ]

    def _template_parameter_constraints(self, template_parameters: str) -> List[str]:
        # SFINAE constraint texts from the anonymous non-type parameters, in
        # source order, for overload selection.
        return [
            parameter.constraint_text
            for parameter in self._parse_template_parameter_list(template_parameters)
            if parameter.is_constraint and parameter.constraint_text
        ]

    def _template_parameter_defaults(self, template_parameters: str) -> Dict[str, str]:
        defaults: Dict[str, str] = {}
        for parameter in self._parse_template_parameter_list(template_parameters):
            if parameter.name is None or parameter.default is None:
                continue
            defaults[parameter.name] = parameter.default
        return defaults

    def _variadic_template_parameter_names(self, template_parameters: str) -> Set[str]:
        return {
            parameter.name
            for parameter in self._parse_template_parameter_list(template_parameters)
            if parameter.is_variadic
            and parameter.is_type_parameter
            and parameter.name is not None
        }

    def _template_arguments_with_defaults(
        self,
        code: str,
        template: _MetalTemplateFunction,
        template_arguments: List[str],
    ) -> List[str]:
        del code
        arguments = list(template_arguments)
        if self._template_arguments_satisfy_parameters(template, arguments):
            return arguments

        defaults = getattr(template, "template_parameter_defaults", {}) or {}
        for parameter in template.template_parameters[len(arguments) :]:
            if parameter in template.variadic_template_parameters:
                break
            default = defaults.get(parameter)
            if default is None:
                break
            arguments.append(default)
        return arguments

    def _is_atomic_non_type_template_id(self, value: str) -> bool:
        """Whether ``value`` is exactly one qualified template-id.

        A function template specialization such as ``radix2<float>`` is a
        primary expression when used as a non-type template argument.  Wrapping
        it before textual callback substitution changes ``radix_func(x)`` into
        ``(radix2<float>)(x)``, which hides the call from later materialization.
        Require both a qualified template name and a balanced angle list that
        consumes the complete value so arithmetic around a template-id remains
        grouped as a compound expression.
        """

        text = str(value).strip()
        match = re.match(
            r"(?:::)?[A-Za-z_][A-Za-z0-9_]*" r"(?:\s*::\s*[A-Za-z_][A-Za-z0-9_]*)*\s*<",
            text,
        )
        if match is None:
            return False
        angle_start = match.end() - 1
        angle_end = self._find_matching_angle(text, angle_start)
        return angle_end == len(text) - 1

    def _is_atomic_qualified_non_type_template_id(self, value: str) -> bool:
        """Whether ``value`` is one qualified-id containing a template-id."""

        text = str(value).strip()
        match = re.match(
            r"(?:::)?[A-Za-z_][A-Za-z0-9_]*" r"(?:\s*::\s*[A-Za-z_][A-Za-z0-9_]*)*\s*<",
            text,
        )
        if match is None:
            return False
        angle_start = match.end() - 1
        angle_end = self._find_matching_angle(text, angle_start)
        if angle_end is None:
            return False

        position = angle_end + 1
        while True:
            while position < len(text) and text[position].isspace():
                position += 1
            if position == len(text):
                return True
            if not text.startswith("::", position):
                return False
            position += 2
            while position < len(text) and text[position].isspace():
                position += 1
            template_marker = re.match(r"template\b\s*", text[position:])
            if template_marker is not None:
                position += template_marker.end()
            name = re.match(r"[A-Za-z_][A-Za-z0-9_]*", text[position:])
            if name is None:
                return False
            position += name.end()
            while position < len(text) and text[position].isspace():
                position += 1
            if position < len(text) and text[position] == "<":
                angle_end = self._find_matching_angle(text, position)
                if angle_end is None:
                    return False
                position = angle_end + 1

    def _canonicalize_template_materialization_parameter(self, value: str) -> str:
        """Remove redundant outer grouping from one report-only parameter.

        Non-type substitutions retain their context-safe grouped source spelling,
        while materialization metadata records the standalone value without
        balanced outer parentheses.  Removing those parentheses is safe only for
        the report value as a whole; generated source continues to use the
        grouped spelling so compound-expression precedence remains unchanged.
        """

        text = str(value)
        normalized = self._normalize_template_argument_text(text)
        candidate = normalized
        while candidate.startswith("(") and candidate.endswith(")"):
            closing = self._find_matching_delimiter(candidate, 0, "(", ")")
            if closing != len(candidate) - 1:
                break
            inner = candidate[1:-1].strip()
            if not inner:
                break
            candidate = inner
        if candidate != normalized:
            return candidate
        return text

    def _group_non_type_template_substitution(self, value: str) -> str:
        """Return a context-safe spelling for one non-type template value.

        Identifier replacement is textual, so substituting an arithmetic default
        such as ``BCOLS / n_reads`` into ``tgp_size / TCOLS`` must retain the
        substituted expression as one operand.  Atomic values and complete
        qualified template-ids already bind as one expression; compound values
        receive one balanced outer pair.  The helper is deliberately limited to
        parameters recorded as non-type by callers so type substitutions never
        acquire expression parentheses.
        """

        text = str(value).strip()
        if not text or not self.group_non_type_template_substitutions:
            return text
        if text.startswith("(") and text.endswith(")"):
            closing = self._find_matching_delimiter(text, 0, "(", ")")
            if closing == len(text) - 1:
                return text
        scalar = (
            r"(?:true|false|nullptr|[-+]?(?:0[xX][0-9A-Fa-f]+|0[bB][01]+|"
            r"(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
            r"[uUlLfFhH]*)"
        )
        qualified_identifier = (
            r"(?:::)?[A-Za-z_][A-Za-z0-9_]*" r"(?:::[A-Za-z_][A-Za-z0-9_]*)*"
        )
        if re.fullmatch(
            rf"(?:{scalar}|{qualified_identifier})", text
        ) or self._is_atomic_non_type_template_id(text):
            return text
        return f"({text})"

    def _template_argument_bindings(
        self,
        template: _MetalTemplateFunction,
        template_arguments: List[str],
    ) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
        substitutions: Dict[str, str] = {}
        variadic_bindings: Dict[str, List[str]] = {}
        defaults = getattr(template, "template_parameter_defaults", {}) or {}
        non_type_parameters = set(
            getattr(template, "template_parameter_types", {}) or {}
        )

        def bound_value(name: str, value: str) -> str:
            if name not in non_type_parameters:
                return value
            return self._group_non_type_template_substitution(value)

        argument_index = 0
        for parameter_index, name in enumerate(template.template_parameters):
            if name in template.variadic_template_parameters:
                remaining_fixed = (
                    len(template.template_parameters) - parameter_index - 1
                )
                variadic_count = max(
                    0,
                    len(template_arguments) - argument_index - remaining_fixed,
                )
                values = template_arguments[
                    argument_index : argument_index + variadic_count
                ]
                variadic_bindings[name] = values
                substitutions[name] = values[0] if values else "void"
                argument_index += variadic_count
                continue
            if argument_index < len(template_arguments):
                substitutions[name] = bound_value(
                    name,
                    template_arguments[argument_index],
                )
                argument_index += 1
                continue
            default_argument = defaults.get(name)
            if default_argument is None:
                continue
            substitutions[name] = bound_value(
                name,
                self._resolve_template_default_argument(
                    default_argument,
                    substitutions,
                    template,
                ),
            )
        return substitutions, variadic_bindings

    def _template_arguments_satisfy_parameters(
        self,
        template: _MetalTemplateFunction,
        template_arguments: List[str],
    ) -> bool:
        defaults = getattr(template, "template_parameter_defaults", {}) or {}
        argument_index = 0
        parameter_count = len(template.template_parameters)
        for parameter_index, name in enumerate(template.template_parameters):
            if name in template.variadic_template_parameters:
                remaining_fixed = parameter_count - parameter_index - 1
                remaining_arguments = len(template_arguments) - argument_index
                if remaining_arguments < remaining_fixed:
                    return False
                variadic_count = max(0, remaining_arguments - remaining_fixed)
                argument_index += variadic_count
                continue
            if argument_index < len(template_arguments):
                argument_index += 1
                continue
            if name not in defaults:
                return False
        return argument_index == len(template_arguments)

    def _resolve_template_default_argument(
        self,
        default_argument: str,
        substitutions: Dict[str, str],
        template: _MetalTemplateFunction,
    ) -> str:
        non_type_parameters = set(
            getattr(template, "template_parameter_types", {}) or {}
        )
        grouped_substitutions = {
            name: (
                self._group_non_type_template_substitution(value)
                if name in non_type_parameters
                else value
            )
            for name, value in substitutions.items()
        }
        resolved = self._replace_identifiers(
            str(default_argument),
            grouped_substitutions,
        )
        resolved = self._normalize_template_argument_text(resolved)
        trait_resolved = self._resolve_template_type_trait(
            resolved,
            getattr(template, "template_type_traits", {}) or {},
            getattr(template, "namespace", ""),
        )
        return trait_resolved or resolved

    def _resolve_template_type_trait(
        self,
        type_text: str,
        traits: Dict[str, Dict[str, object]],
        namespace: str = "",
    ) -> Optional[str]:
        text = str(type_text or "").strip()
        match = re.fullmatch(
            r"(?:typename\s+)?(?P<name>(?:::)?[A-Za-z_][A-Za-z0-9_:]*)\s*"
            r"<(?P<args>.*)>\s*::\s*type",
            text,
            re.DOTALL,
        )
        if match is None:
            return None
        raw_trait_name = match.group("name")
        globally_qualified = raw_trait_name.startswith("::")
        trait_name = raw_trait_name.lstrip(":")
        lookup_names = self._namespace_lookup_names(
            trait_name,
            namespace,
            globally_qualified,
        )
        trait = next(
            (traits[name] for name in lookup_names if name in traits),
            None,
        )
        if not trait:
            return None
        allow_unqualified_remove_cv_t = bool(
            trait.get("allow_unqualified_remove_cv_t", False)
        )
        arguments = [
            self._canonicalize_metal_standard_type_aliases(
                self._normalize_template_argument_text(argument),
                allow_unqualified_remove_cv_t=allow_unqualified_remove_cv_t,
            )
            for argument in self._split_top_level_commas(match.group("args"))
        ]
        specializations = trait.get("specializations", {})
        specialized = specializations.get(tuple(arguments))
        if specialized is not None:
            specialized_type = (
                specialized if isinstance(specialized, str) else specialized.get("type")
            )
            if isinstance(specialized_type, str) and specialized_type:
                return self._canonicalize_metal_standard_type_aliases(
                    specialized_type,
                    allow_unqualified_remove_cv_t=allow_unqualified_remove_cv_t,
                )

        partial_matches: List[Tuple[Tuple[int, int, int], str]] = []
        unresolved_variadic_match = False
        for patterns, specialization in specializations.items():
            if len(patterns) != len(arguments) or tuple(patterns) == tuple(arguments):
                continue
            if not isinstance(specialization, dict):
                continue
            parameters = set(specialization.get("parameters", ()))
            variadic_parameters = set(specialization.get("variadic_parameters", ()))
            specialization_type = specialization.get("type")
            if not parameters or not isinstance(specialization_type, str):
                continue
            if variadic_parameters:
                if self._template_trait_variadic_specialization_may_match(
                    patterns,
                    arguments,
                    parameters,
                    variadic_parameters,
                ):
                    unresolved_variadic_match = True
                continue
            bindings: Dict[str, str] = {}
            matched = True
            for pattern, argument in zip(patterns, arguments):
                normalized_pattern = self._normalize_template_argument_text(pattern)
                if normalized_pattern != argument and not (
                    self._equivalent_template_boolean_literals(
                        normalized_pattern,
                        argument,
                    )
                ):
                    self._infer_template_parameter_bindings_from_type(
                        normalized_pattern,
                        argument,
                        parameters,
                        bindings,
                    )
                resolved_pattern = self._normalize_template_argument_text(
                    self._replace_identifiers(normalized_pattern, bindings)
                )
                if resolved_pattern != argument and not (
                    self._equivalent_template_boolean_literals(
                        resolved_pattern,
                        argument,
                    )
                ):
                    matched = False
                    break
            if matched:
                partial_matches.append(
                    (
                        self._template_trait_specialization_specificity(
                            patterns,
                            parameters,
                        ),
                        self._normalize_template_argument_text(
                            self._replace_identifiers(specialization_type, bindings)
                        ),
                    )
                )
        if unresolved_variadic_match:
            return None
        if partial_matches:
            highest_specificity = max(score for score, _result in partial_matches)
            best_results = {
                result
                for score, result in partial_matches
                if score == highest_specificity
            }
            if len(best_results) == 1:
                return self._canonicalize_metal_standard_type_aliases(
                    next(iter(best_results)),
                    allow_unqualified_remove_cv_t=allow_unqualified_remove_cv_t,
                )
            return None
        parameters = trait.get("parameters", [])
        default_type = trait.get("default")
        if not isinstance(default_type, str) or len(arguments) < len(parameters):
            return None
        substitutions = dict(zip(parameters, arguments))
        resolved = self._replace_identifiers(default_type, substitutions)
        return self._canonicalize_metal_standard_type_aliases(
            self._normalize_template_argument_text(resolved),
            allow_unqualified_remove_cv_t=allow_unqualified_remove_cv_t,
        )

    @staticmethod
    def _namespace_lookup_names(
        name: str,
        namespace: str,
        globally_qualified: bool = False,
    ) -> List[str]:
        if globally_qualified or not namespace:
            return [name]
        namespace_parts = [part for part in namespace.split("::") if part]
        candidates = [
            "::".join([*namespace_parts[:depth], name])
            for depth in range(len(namespace_parts), 0, -1)
        ]
        candidates.append(name)
        return list(dict.fromkeys(candidates))

    @staticmethod
    def _template_trait_variadic_specialization_may_match(
        patterns: Tuple[str, ...],
        arguments: List[str],
        parameters: Set[str],
        variadic_parameters: Set[str],
    ) -> bool:
        if len(patterns) != len(arguments):
            minimum_arguments = len(patterns) - sum(
                1
                for pattern in patterns
                if any(
                    re.fullmatch(
                        rf"\s*{re.escape(parameter)}\s*\.\.\.\s*",
                        pattern,
                    )
                    for parameter in variadic_parameters
                )
            )
            return len(arguments) >= minimum_arguments and minimum_arguments < len(
                patterns
            )
        for pattern, argument in zip(patterns, arguments):
            normalized_pattern = re.sub(r"\s+", "", pattern)
            normalized_argument = re.sub(r"\s+", "", argument)
            pattern_parts: List[str] = []
            position = 0
            for match in IDENTIFIER_RE.finditer(normalized_pattern):
                pattern_parts.append(
                    re.escape(normalized_pattern[position : match.start()])
                )
                identifier = match.group(0)
                position = match.end()
                if identifier in variadic_parameters:
                    pattern_parts.append(".*?")
                    if normalized_pattern[position : position + 3] == "...":
                        position += 3
                elif identifier in parameters:
                    pattern_parts.append(".+?")
                else:
                    pattern_parts.append(re.escape(identifier))
            pattern_parts.append(re.escape(normalized_pattern[position:]))
            if re.fullmatch("".join(pattern_parts), normalized_argument) is None:
                return False
        return True

    @staticmethod
    def _template_trait_specialization_specificity(
        patterns: Tuple[str, ...],
        parameters: Set[str],
    ) -> Tuple[int, int, int]:
        parameter_occurrences: List[str] = []
        literal_characters = 0
        for pattern in patterns:
            position = 0
            for match in IDENTIFIER_RE.finditer(pattern):
                literal_characters += len(
                    re.sub(r"\s+", "", pattern[position : match.start()])
                )
                identifier = match.group(0)
                if identifier in parameters:
                    parameter_occurrences.append(identifier)
                else:
                    literal_characters += len(identifier)
                position = match.end()
            literal_characters += len(re.sub(r"\s+", "", pattern[position:]))
        repeated_constraints = len(parameter_occurrences) - len(
            set(parameter_occurrences)
        )
        return (
            literal_characters,
            repeated_constraints,
            -len(set(parameter_occurrences)),
        )

    @staticmethod
    def _skip_cpp_trivia(code: str, position: int) -> int:
        cursor = position
        while cursor < len(code):
            if code[cursor].isspace():
                cursor += 1
                continue
            if code.startswith("//", cursor):
                newline = code.find("\n", cursor + 2)
                if newline == -1:
                    return len(code)
                cursor = newline + 1
                continue
            if code.startswith("/*", cursor):
                comment_end = code.find("*/", cursor + 2)
                if comment_end == -1:
                    return len(code)
                cursor = comment_end + 2
                continue
            break
        return cursor

    def _read_namespace_components(
        self,
        code: str,
        position: int,
    ) -> Tuple[Tuple[str, ...], int]:
        components: List[str] = []
        cursor = self._skip_cpp_trivia(code, position)
        while cursor < len(code):
            name_match = IDENTIFIER_RE.match(code, cursor)
            if name_match is None:
                break
            components.append(name_match.group(0))
            cursor = self._skip_cpp_trivia(code, name_match.end())
            if not code.startswith("::", cursor):
                break
            cursor = self._skip_cpp_trivia(code, cursor + 2)
        return tuple(components), cursor

    def _find_namespace_spans(self, code: str) -> List[Tuple[int, int, str]]:
        analysis = self._source_analysis(code)
        if analysis.namespace_spans is None:
            analysis.namespace_spans = self._scan_namespace_spans(code)
        return analysis.namespace_spans

    def _scan_namespace_spans(self, code: str) -> List[Tuple[int, int, str]]:
        spans: List[Tuple[int, int, str]] = []
        anonymous_spans: List[Tuple[int, int]] = []
        brace_stack: List[Tuple[int, Optional[Tuple[str, ...]], str]] = []
        active_namespace: List[str] = []
        i = 0
        while i < len(code):
            match = METAL_NAMESPACE_DELIMITER_RE.search(code, i)
            if match is None:
                break
            i = match.start()
            if code[i] in "\"'":
                _literal, consumed = self._read_string(code, i)
                i += consumed
                continue
            if code.startswith("//", i):
                newline = code.find("\n", i + 2)
                if newline == -1:
                    break
                i = newline + 1
                continue
            if code.startswith("/*", i):
                comment_end = code.find("*/", i + 2)
                if comment_end == -1:
                    break
                i = comment_end + 2
                continue
            if (
                code.startswith("namespace", i)
                and (i == 0 or not (code[i - 1].isalnum() or code[i - 1] == "_"))
                and (
                    i + len("namespace") == len(code)
                    or not (
                        code[i + len("namespace")].isalnum()
                        or code[i + len("namespace")] == "_"
                    )
                )
            ):
                components, cursor = self._read_namespace_components(
                    code,
                    i + len("namespace"),
                )
                cursor = self._skip_cpp_trivia(code, cursor)
                if cursor < len(code) and code[cursor] == "{":
                    full_namespace = "::".join([*active_namespace, *components])
                    brace_stack.append((cursor, components, full_namespace))
                    active_namespace.extend(components)
                    i = cursor + 1
                    continue
            if code[i] == "{":
                brace_stack.append((i, None, ""))
            elif code[i] == "}" and brace_stack:
                open_brace, components, full_namespace = brace_stack.pop()
                if components is not None:
                    spans.append((open_brace + 1, i, full_namespace))
                    if components:
                        del active_namespace[-len(components) :]
                    else:
                        anonymous_spans.append((open_brace + 1, i))
            i += 1
        self._source_analysis(code).anonymous_namespace_spans = anonymous_spans
        return spans

    @staticmethod
    def _namespace_at(
        namespace_spans: List[Tuple[int, int, str]],
        position: int,
    ) -> str:
        containing = [span for span in namespace_spans if span[0] <= position < span[1]]
        if not containing:
            return ""
        return max(containing, key=lambda span: span[0])[2]

    def _find_template_type_traits(
        self,
        code: str,
        namespace_spans: Optional[List[Tuple[int, int, str]]] = None,
    ) -> Dict[str, Dict[str, object]]:
        if namespace_spans is None:
            namespace_spans = self._find_namespace_spans(code)
        key = tuple(namespace_spans)
        cache = self._source_analysis(code).template_type_traits
        cached = cache.pop(key, None)
        if cached is None:
            cached = self._scan_template_type_traits(code, list(key))
            cache = self._source_analysis(code).template_type_traits
        cache[key] = cached
        if len(cache) > SOURCE_ANALYSIS_CACHE_LIMIT:
            del cache[next(iter(cache))]
        return deepcopy(cached)

    def _scan_template_type_traits(
        self,
        code: str,
        namespace_spans: List[Tuple[int, int, str]],
    ) -> Dict[str, Dict[str, object]]:
        traits: Dict[str, Dict[str, object]] = {}
        pos = 0
        while True:
            match = re.search(r"\btemplate\s*<", code[pos:])
            if match is None:
                break
            start = pos + match.start()
            angle_start = code.find("<", start)
            angle_end = self._find_matching_template_param_angle(code, angle_start)
            if angle_end is None:
                pos = start + len("template")
                continue
            declaration_start = angle_end + 1
            body_start = self._find_next_top_level_char(
                code, declaration_start, "{", stop_at=";"
            )
            semicolon = self._find_next_top_level_char(
                code, declaration_start, ";", stop_at="{"
            )
            if body_start is None or (semicolon is not None and semicolon < body_start):
                pos = declaration_start
                continue
            body_end = self._find_matching_brace(code, body_start)
            if body_end is None:
                pos = body_start + 1
                continue
            header = code[declaration_start:body_start]
            body = code[body_start + 1 : body_end - 1]
            self._record_template_type_trait(
                traits,
                code[angle_start + 1 : angle_end],
                header,
                body,
                self._namespace_at(namespace_spans, start),
            )
            pos = body_end
        allow_unqualified_remove_cv_t = self._metal_unqualified_remove_cv_t_available(
            code
        )
        for trait in traits.values():
            trait["allow_unqualified_remove_cv_t"] = allow_unqualified_remove_cv_t
        return traits

    def _record_template_type_trait(
        self,
        traits: Dict[str, Dict[str, object]],
        parameter_text: str,
        header: str,
        body: str,
        namespace: str = "",
    ) -> None:
        alias_type = self._template_type_trait_alias_type(body)
        if not alias_type:
            return
        header_match = re.search(
            r"\b(?:struct|class)\s+([A-Za-z_][A-Za-z0-9_:]*)\s*"
            r"(?:<(?P<args>[^{};]*)>)?",
            header,
            re.DOTALL,
        )
        if header_match is None:
            return
        name = header_match.group(1)
        if "::" not in name and namespace:
            name = f"{namespace}::{name}"
        trait = traits.setdefault(
            name,
            {
                "parameters": [],
                "default": None,
                "specializations": {},
            },
        )
        specialization_args = header_match.group("args")
        if specialization_args is not None:
            arguments = tuple(
                self._normalize_template_argument_text(argument)
                for argument in self._split_top_level_commas(specialization_args)
            )
            trait.setdefault("specializations", {})[arguments] = {
                "type": alias_type,
                "parameters": self._template_parameter_names(parameter_text),
                "variadic_parameters": sorted(
                    self._variadic_template_parameter_names(parameter_text)
                ),
            }
            return
        trait["parameters"] = self._template_parameter_names(parameter_text)
        trait["default"] = alias_type

    def _template_type_trait_alias_type(self, body: str) -> Optional[str]:
        match = re.search(
            r"\busing\s+type\s*=\s*(?P<type>[^;{}]+)\s*;",
            body,
            re.DOTALL,
        )
        if match is None:
            return None
        return self._normalize_template_argument_text(match.group("type"))

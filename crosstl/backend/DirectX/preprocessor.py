import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class Macro:
    name: str
    params: Optional[List[str]] = None
    replacement: str = ""
    is_variadic: bool = False

    def is_function_like(self) -> bool:
        return self.params is not None


class HLSLPreprocessor:
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

        if defines:
            for name, value in defines.items():
                self.macros[name] = Macro(
                    name=name, params=None, replacement=str(value)
                )

    def preprocess(self, code: str, file_path: Optional[str] = None) -> str:
        logical_lines = self._split_logical_lines(code)
        output_lines = self._process_lines(logical_lines, file_path)
        return "\n".join(output_lines)

    def _split_logical_lines(self, code: str) -> List[str]:
        lines = code.splitlines()
        logical_lines: List[str] = []
        buffer = ""
        for line in lines:
            stripped = line.rstrip()
            if stripped.endswith("\\"):
                buffer += stripped[:-1]
                continue
            buffer += line
            logical_lines.append(buffer)
            buffer = ""
        if buffer:
            logical_lines.append(buffer)
        return logical_lines

    def _process_lines(self, lines: List[str], file_path: Optional[str]) -> List[str]:
        output: List[str] = []
        conditional_stack: List[Dict[str, bool]] = []
        current_line = 1
        line_override: Optional[int] = None

        def is_active() -> bool:
            return all(frame["active"] for frame in conditional_stack)

        for raw_line in lines:
            line = raw_line
            stripped = line.lstrip()
            active = is_active()

            if stripped.startswith("#"):
                directive, rest = self._parse_directive(stripped)
                if directive in ("if", "ifdef", "ifndef"):
                    condition = self._evaluate_condition(directive, rest, current_line)
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
                        raise SyntaxError("#elif without #if")
                    frame = conditional_stack[-1]
                    if frame["parent_active"] and not frame["branch_taken"]:
                        condition = self._evaluate_expression(
                            self._expand_macros(rest, current_line, True)
                        )
                        frame["active"] = condition
                        frame["branch_taken"] = condition
                    else:
                        frame["active"] = False
                elif directive == "else":
                    if not conditional_stack:
                        raise SyntaxError("#else without #if")
                    frame = conditional_stack[-1]
                    frame["active"] = (
                        frame["parent_active"] and not frame["branch_taken"]
                    )
                    frame["branch_taken"] = True
                elif directive == "endif":
                    if not conditional_stack:
                        raise SyntaxError("#endif without #if")
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
                    included_text = self._handle_include(rest, file_path)
                    if included_text is not None:
                        nested_lines = self._split_logical_lines(included_text)
                        output.extend(self._process_lines(nested_lines, file_path))
                elif directive == "line":
                    line_override = self._handle_line_directive(rest)
                    if line_override is not None:
                        current_line = line_override
                elif directive in ("error", "warning"):
                    if directive == "error" or self.strict:
                        raise SyntaxError(f"#{directive}: {rest.strip()}")
                else:
                    if active:
                        output.append(line)
            else:
                if active:
                    expanded = self._expand_macros(line, current_line, False)
                    output.append(expanded)
            current_line += 1

        if conditional_stack:
            raise SyntaxError("Unterminated #if block")

        return output

    def _parse_directive(self, line: str) -> Tuple[str, str]:
        match = re.match(r"#\s*([A-Za-z_][A-Za-z0-9_]*)\s*(.*)", line)
        if not match:
            return "", ""
        return match.group(1), match.group(2)

    def _evaluate_condition(self, directive: str, rest: str, line_num: int) -> bool:
        if directive == "ifdef":
            name = rest.strip()
            return name in self.macros
        if directive == "ifndef":
            name = rest.strip()
            return name not in self.macros
        return bool(
            self._evaluate_expression(self._expand_macros(rest, line_num, True))
        )

    def _evaluate_expression(self, expr: str) -> int:
        tokenizer = _ExpressionTokenizer(expr)
        parser = _ExpressionParser(tokenizer)
        return parser.parse_expression()

    def _handle_define(self, rest: str):
        rest = rest.lstrip()
        name_match = re.match(r"[A-Za-z_][A-Za-z0-9_]*", rest)
        if not name_match:
            return
        name = name_match.group(0)
        after = rest[name_match.end() :]
        if after.startswith("("):
            params, remainder, is_variadic = self._parse_macro_params(after)
            replacement = remainder.lstrip()
            self.macros[name] = Macro(
                name=name,
                params=params,
                replacement=replacement,
                is_variadic=is_variadic,
            )
        else:
            replacement = after.lstrip()
            self.macros[name] = Macro(name=name, params=None, replacement=replacement)

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
            params[-1] = params[-1].replace("...", "")
            params.append("__VA_ARGS__")
            is_variadic = True
        return params, remainder, is_variadic

    def _handle_include(self, rest: str, file_path: Optional[str]) -> Optional[str]:
        match = re.match(r"\s*([<\"])([^>\"]+)[>\"]", rest)
        if not match:
            return None
        delimiter = match.group(1)
        target = match.group(2)

        search_paths: List[str] = []
        if delimiter == '"' and file_path:
            search_paths.append(os.path.dirname(file_path))
        search_paths.extend(self.include_paths)

        for base in search_paths:
            candidate = os.path.join(base, target)
            if os.path.isfile(candidate):
                with open(candidate, "r", encoding="utf-8") as handle:
                    return handle.read()

        if self.strict:
            raise FileNotFoundError(f"Include not found: {target}")
        return None

    def _handle_line_directive(self, rest: str) -> Optional[int]:
        parts = rest.strip().split()
        if not parts:
            return None
        try:
            return int(parts[0])
        except ValueError:
            return None

    def _expand_macros(self, text: str, line_num: int, in_expression: bool) -> str:
        result = ""
        i = 0
        depth = 0
        while i < len(text):
            ch = text[i]
            if ch in "\"'":
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

            if ch.isalpha() or ch == "_":
                ident, consumed = self._read_identifier(text, i)
                i += consumed
                if in_expression and ident == "defined":
                    value, consumed_def = self._parse_defined(text, i)
                    result += "1" if value else "0"
                    i += consumed_def
                    continue

                macro = self.macros.get(ident)
                if macro is None:
                    result += "0" if in_expression else ident
                    continue

                if macro.is_function_like():
                    j = i
                    while j < len(text) and text[j].isspace():
                        j += 1
                    if j < len(text) and text[j] == "(":
                        args, consumed_args = self._parse_macro_args(text, j)
                        i = j + consumed_args
                        replaced = self._expand_function_macro(macro, args)
                        result += self._expand_macros(replaced, line_num, in_expression)
                        continue
                    result += ident
                    continue

                replaced = macro.replacement if macro.replacement is not None else ""
                result += self._expand_macros(replaced, line_num, in_expression)
                continue

            result += ch
            i += 1
            depth += 1
            if depth > 100000:
                break
        return result

    def _read_identifier(self, text: str, start: int) -> Tuple[str, int]:
        i = start
        while i < len(text) and (text[i].isalnum() or text[i] == "_"):
            i += 1
        return text[start:i], i - start

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

    def _parse_defined(self, text: str, start: int) -> Tuple[bool, int]:
        i = start
        while i < len(text) and text[i].isspace():
            i += 1
        if i < len(text) and text[i] == "(":
            i += 1
            while i < len(text) and text[i].isspace():
                i += 1
            ident, consumed = self._read_identifier(text, i)
            i += consumed
            while i < len(text) and text[i] != ")":
                i += 1
            if i < len(text) and text[i] == ")":
                i += 1
            return ident in self.macros, i - start

        ident, consumed = self._read_identifier(text, i)
        i += consumed
        return ident in self.macros, i - start

    def _parse_macro_args(self, text: str, start: int) -> Tuple[List[str], int]:
        assert text[start] == "("
        args: List[str] = []
        current = ""
        depth = 0
        i = start
        while i < len(text):
            ch = text[i]
            if ch == "(":
                depth += 1
                if depth > 1:
                    current += ch
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    args.append(current.strip())
                    return args, i - start + 1
                current += ch
            elif ch == "," and depth == 1:
                args.append(current.strip())
                current = ""
            else:
                current += ch
            i += 1
        return args, i - start

    def _expand_function_macro(self, macro: Macro, args: List[str]) -> str:
        params = macro.params or []
        param_map: Dict[str, str] = {}

        if macro.is_variadic and len(args) >= len(params):
            fixed_count = len(params) - 1
            for idx in range(fixed_count):
                param_map[params[idx]] = args[idx] if idx < len(args) else ""
            param_map["__VA_ARGS__"] = ", ".join(args[fixed_count:])
        else:
            for idx, name in enumerate(params):
                param_map[name] = args[idx] if idx < len(args) else ""

        return self._replace_params(macro.replacement, param_map)

    def _replace_params(self, replacement: str, param_map: Dict[str, str]) -> str:
        tokens = self._tokenize_replacement(replacement)
        output: List[str] = []
        i = 0
        while i < len(tokens):
            tok_type, _tok_val = tokens[i]
            if tok_type == "hash" and i + 1 < len(tokens):
                next_type, next_val = tokens[i + 1]
                if next_type == "ident" and next_val in param_map:
                    output.append(self._stringize(param_map[next_val]))
                    i += 2
                    continue
            if tok_type == "paste" and output:
                if i + 1 < len(tokens):
                    next_val = self._token_value(tokens[i + 1], param_map)
                    prev = output.pop()
                    output.append(prev + next_val)
                    i += 2
                    continue
            output.append(self._token_value(tokens[i], param_map))
            i += 1
        return "".join(output)

    def _stringize(self, value: str) -> str:
        collapsed = re.sub(r"\s+", " ", value.strip())
        escaped = collapsed.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'

    def _tokenize_replacement(self, text: str) -> List[Tuple[str, str]]:
        tokens: List[Tuple[str, str]] = []
        i = 0
        while i < len(text):
            if text.startswith("##", i):
                tokens.append(("paste", "##"))
                i += 2
                continue
            if text[i] == "#":
                tokens.append(("hash", "#"))
                i += 1
                continue
            if text[i].isspace():
                start = i
                while i < len(text) and text[i].isspace():
                    i += 1
                tokens.append(("ws", text[start:i]))
                continue
            if text[i].isalpha() or text[i] == "_":
                ident, consumed = self._read_identifier(text, i)
                tokens.append(("ident", ident))
                i += consumed
                continue
            if text[i] in "\"'":
                literal, consumed = self._read_string(text, i)
                tokens.append(("literal", literal))
                i += consumed
                continue
            tokens.append(("sym", text[i]))
            i += 1
        return tokens

    def _token_value(self, token: Tuple[str, str], param_map: Dict[str, str]) -> str:
        tok_type, tok_val = token
        if tok_type == "ident" and tok_val in param_map:
            return param_map[tok_val]
        return tok_val


class _ExpressionTokenizer:
    def __init__(self, expr: str):
        self.expr = expr
        self.pos = 0

    def next_token(self) -> Tuple[str, Optional[int]]:
        while self.pos < len(self.expr) and self.expr[self.pos].isspace():
            self.pos += 1
        if self.pos >= len(self.expr):
            return "EOF", None

        multi = [
            "||",
            "&&",
            "==",
            "!=",
            "<=",
            ">=",
            "<<",
            ">>",
        ]
        for op in multi:
            if self.expr.startswith(op, self.pos):
                self.pos += len(op)
                return op, None

        ch = self.expr[self.pos]
        self.pos += 1
        if ch.isdigit():
            start = self.pos - 1
            while self.pos < len(self.expr) and (
                self.expr[self.pos].isalnum() or self.expr[self.pos] in "xX"
            ):
                self.pos += 1
            text = self.expr[start : self.pos]
            return "NUMBER", self._parse_number(text)
        if ch.isalpha() or ch == "_":
            start = self.pos - 1
            while self.pos < len(self.expr) and (
                self.expr[self.pos].isalnum() or self.expr[self.pos] == "_"
            ):
                self.pos += 1
            return "NUMBER", 0
        return ch, None

    def _parse_number(self, text: str) -> int:
        text = re.sub(r"[uUlL]+$", "", text)
        if text.startswith(("0x", "0X")):
            return int(text, 16)
        if text.startswith(("0b", "0B")):
            return int(text, 2)
        if text.startswith("0") and len(text) > 1:
            try:
                return int(text, 8)
            except ValueError:
                return int(text, 10)
        try:
            return int(float(text))
        except ValueError:
            return 0


class _ExpressionParser:
    def __init__(self, tokenizer: _ExpressionTokenizer):
        self.tokenizer = tokenizer
        self.current = self.tokenizer.next_token()

    def _eat(self, token_type: str):
        if self.current[0] != token_type:
            raise SyntaxError(f"Expected {token_type}, got {self.current[0]}")
        self.current = self.tokenizer.next_token()

    def parse_expression(self) -> int:
        return self._parse_logical_or()

    def _parse_logical_or(self) -> int:
        left = self._parse_logical_and()
        while self.current[0] == "||":
            self._eat("||")
            right = self._parse_logical_and()
            left = 1 if (left or right) else 0
        return left

    def _parse_logical_and(self) -> int:
        left = self._parse_bitwise_or()
        while self.current[0] == "&&":
            self._eat("&&")
            right = self._parse_bitwise_or()
            left = 1 if (left and right) else 0
        return left

    def _parse_bitwise_or(self) -> int:
        left = self._parse_bitwise_xor()
        while self.current[0] == "|":
            self._eat("|")
            right = self._parse_bitwise_xor()
            left = left | right
        return left

    def _parse_bitwise_xor(self) -> int:
        left = self._parse_bitwise_and()
        while self.current[0] == "^":
            self._eat("^")
            right = self._parse_bitwise_and()
            left = left ^ right
        return left

    def _parse_bitwise_and(self) -> int:
        left = self._parse_equality()
        while self.current[0] == "&":
            self._eat("&")
            right = self._parse_equality()
            left = left & right
        return left

    def _parse_equality(self) -> int:
        left = self._parse_relational()
        while self.current[0] in ("==", "!="):
            op = self.current[0]
            self._eat(op)
            right = self._parse_relational()
            left = 1 if (left == right if op == "==" else left != right) else 0
        return left

    def _parse_relational(self) -> int:
        left = self._parse_shift()
        while self.current[0] in ("<", "<=", ">", ">="):
            op = self.current[0]
            self._eat(op)
            right = self._parse_shift()
            if op == "<":
                left = 1 if left < right else 0
            elif op == "<=":
                left = 1 if left <= right else 0
            elif op == ">":
                left = 1 if left > right else 0
            else:
                left = 1 if left >= right else 0
        return left

    def _parse_shift(self) -> int:
        left = self._parse_additive()
        while self.current[0] in ("<<", ">>"):
            op = self.current[0]
            self._eat(op)
            right = self._parse_additive()
            left = left << right if op == "<<" else left >> right
        return left

    def _parse_additive(self) -> int:
        left = self._parse_multiplicative()
        while self.current[0] in ("+", "-"):
            op = self.current[0]
            self._eat(op)
            right = self._parse_multiplicative()
            left = left + right if op == "+" else left - right
        return left

    def _parse_multiplicative(self) -> int:
        left = self._parse_unary()
        while self.current[0] in ("*", "/", "%"):
            op = self.current[0]
            self._eat(op)
            right = self._parse_unary()
            if op == "*":
                left = left * right
            elif op == "/":
                left = left // right if right != 0 else 0
            else:
                left = left % right if right != 0 else 0
        return left

    def _parse_unary(self) -> int:
        if self.current[0] in ("+", "-", "!", "~"):
            op = self.current[0]
            self._eat(op)
            value = self._parse_unary()
            if op == "+":
                return value
            if op == "-":
                return -value
            if op == "!":
                return 0 if value else 1
            return ~value
        return self._parse_primary()

    def _parse_primary(self) -> int:
        if self.current[0] == "NUMBER":
            value = self.current[1] or 0
            self._eat("NUMBER")
            return value
        if self.current[0] == "(":
            self._eat("(")
            value = self.parse_expression()
            self._eat(")")
            return value
        return 0

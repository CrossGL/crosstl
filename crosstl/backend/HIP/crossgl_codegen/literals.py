"""Visitor dispatch, C++ numeric and string literals, and line emission in the HIP to CrossGL converter."""

from ....translator.integer_literals import crossgl_legacy_octal


class HipLiteralMixin:
    """Literal members of ``HipToCrossGLConverter``."""

    def visit(self, node):
        """Dispatch a HIP backend AST node to its converter method."""
        method_name = f"visit_{type(node).__name__}"
        visitor = getattr(self, method_name, self.generic_visit)
        return visitor(node)

    def generic_visit(self, node):
        """Fallback converter for primitive values, lists, and unknown nodes."""
        if isinstance(node, str):
            enum_value = self.resolve_enum_value_alias(node)
            if enum_value is not None:
                return enum_value
            string_literal = self.format_cpp_string_literal_expression(node)
            if string_literal is not None:
                return string_literal
            node = self.normalize_cpp_numeric_literal(node)
            attribute_expression = self.format_hip_device_attribute_read(node)
            if attribute_expression is not None:
                return attribute_expression
            query_expression = self.format_hip_device_query_read(node)
            if query_expression is not None:
                return query_expression
            user_defined_name = self.format_user_defined_function_call_name(node)
            if user_defined_name is not None:
                return user_defined_name
            return self.output_identifier_name(node)
        elif isinstance(node, list):
            return [self.visit(item) for item in node]
        else:
            return str(node)

    def normalize_cpp_numeric_literal(self, value):
        integer_literal = self.CPP_INTEGER_LITERAL.match(value)
        if integer_literal:
            body = integer_literal.group("body").replace("'", "")
            suffix = integer_literal.group("suffix").lower()
            if "u" in suffix:
                return crossgl_legacy_octal(f"{body}u")
            return crossgl_legacy_octal(body)

        if "'" in value and self.CPP_NUMERIC_LITERAL_WITH_SEPARATOR.match(value):
            return value.replace("'", "")
        return value

    def format_cpp_string_literal_expression(self, value):
        if not isinstance(value, str):
            return None

        text = value.strip()
        if not self.maybe_cpp_string_literal_expression(text):
            return None

        parsed = self.parse_cpp_string_literal_sequence(text)
        if parsed is None:
            return None

        return self.quote_crossgl_string_literal(parsed)

    def maybe_cpp_string_literal_expression(self, text):
        if not text:
            return False
        return text.startswith(
            ('"', 'R"', 'u8"', 'u8R"', 'u"', 'uR"', 'U"', 'UR"', 'L"', 'LR"')
        )

    def parse_cpp_string_literal_sequence(self, text):
        parts = []
        index = 0

        while True:
            index = self.skip_cpp_string_literal_whitespace(text, index)
            if index >= len(text):
                break

            parsed = self.parse_cpp_string_literal_at(text, index)
            if parsed is None:
                return None

            content, index = parsed
            parts.append(content)

        if not parts:
            return None

        return "".join(parts)

    def skip_cpp_string_literal_whitespace(self, text, index):
        while index < len(text) and text[index].isspace():
            index += 1
        return index

    def parse_cpp_string_literal_at(self, text, index):
        prefix, literal_index = self.parse_cpp_string_literal_prefix(text, index)
        if literal_index >= len(text):
            return None

        if text.startswith('R"', literal_index):
            return self.parse_cpp_raw_string_literal(text, literal_index + 2)

        if text[literal_index] == '"':
            return self.parse_cpp_ordinary_string_literal(text, literal_index + 1)

        if prefix:
            return None

        return None

    def parse_cpp_string_literal_prefix(self, text, index):
        for prefix in ("u8", "u", "U", "L"):
            if text.startswith(prefix, index):
                return prefix, index + len(prefix)
        return "", index

    def parse_cpp_raw_string_literal(self, text, index):
        delimiter_end = text.find("(", index)
        if delimiter_end == -1:
            return None

        delimiter = text[index:delimiter_end]
        closing = ")" + delimiter + '"'
        content_start = delimiter_end + 1
        content_end = text.find(closing, content_start)
        if content_end == -1:
            return None

        return text[content_start:content_end], content_end + len(closing)

    def parse_cpp_ordinary_string_literal(self, text, index):
        chars = []

        while index < len(text):
            char = text[index]
            if char == '"':
                return "".join(chars), index + 1
            if char == "\\":
                decoded = self.decode_cpp_string_escape(text, index)
                if decoded is None:
                    return None
                value, index = decoded
                chars.append(value)
                continue
            chars.append(char)
            index += 1

        return None

    def decode_cpp_string_escape(self, text, index):
        escape_index = index + 1
        if escape_index >= len(text):
            return None

        escaped = text[escape_index]
        escape_map = {
            "\\": "\\",
            '"': '"',
            "'": "'",
            "?": "?",
            "a": "\a",
            "b": "\b",
            "f": "\f",
            "n": "\n",
            "r": "\r",
            "t": "\t",
            "v": "\v",
        }
        if escaped in escape_map:
            return escape_map[escaped], escape_index + 1

        if escaped == "\n":
            return "", escape_index + 1

        if escaped == "\r":
            next_index = escape_index + 1
            if next_index < len(text) and text[next_index] == "\n":
                next_index += 1
            return "", next_index

        if escaped == "x":
            digit_index = escape_index + 1
            while (
                digit_index < len(text)
                and text[digit_index] in "0123456789abcdefABCDEF"
            ):
                digit_index += 1
            if digit_index == escape_index + 1:
                return "x", digit_index
            return chr(int(text[escape_index + 1 : digit_index], 16)), digit_index

        if escaped in "01234567":
            digit_index = escape_index + 1
            while (
                digit_index < len(text)
                and digit_index < escape_index + 3
                and text[digit_index] in "01234567"
            ):
                digit_index += 1
            return chr(int(text[escape_index:digit_index], 8)), digit_index

        return escaped, escape_index + 1

    def quote_crossgl_string_literal(self, value):
        escaped = []
        for char in value:
            if char == "\\":
                escaped.append("\\\\")
            elif char == '"':
                escaped.append('\\"')
            elif char == "\n":
                escaped.append("\\n")
            elif char == "\r":
                escaped.append("\\r")
            elif char == "\t":
                escaped.append("\\t")
            elif char == "\b":
                escaped.append("\\b")
            elif char == "\f":
                escaped.append("\\f")
            elif char == "\v":
                escaped.append("\\v")
            elif char == "\a":
                escaped.append("\\a")
            elif char == "\0":
                escaped.append("\\0")
            else:
                escaped.append(char)
        return '"' + "".join(escaped) + '"'

    def emit(self, code):
        """Append a line of CrossGL output using the current indentation level."""
        if code.strip():
            self.output.append("    " * self.indent_level + code)
        else:
            self.output.append("")

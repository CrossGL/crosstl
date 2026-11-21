"""Literals and names in the CUDA to CrossGL converter: numeric, character and string literals, identifier scopes, type identifier sanitizing, record and function names, and generated matrix helpers."""

import re

from ....translator.integer_literals import crossgl_legacy_octal
from ..keyword_renames import unused_identifier_name


class CudaImportNameMixin:
    """Name members of ``CudaToCrossGLConverter``."""

    def is_cuda_numeric_literal_text(self, value):
        text = str(value)
        if "'" in text and bool(text) and (text[0].isdigit() or text.startswith(".")):
            return True
        if re.fullmatch(r"0\d+", text):
            # A leading zero makes an integer literal octal (C++17 [lex.icon]).
            return True
        return (
            re.fullmatch(
                r"(?i)(?:0x[0-9a-f]+|0b[01]+|0o[0-7]+|\d+)(?:u|l|ul|lu|ull|llu)",
                text,
            )
            is not None
        )

    def convert_cuda_numeric_literal_to_crossgl(self, value):
        text = str(value).replace("'", "")
        match = re.fullmatch(
            r"(?i)((?:0x[0-9a-f]+|0b[01]+|0o[0-7]+|\d+))(u|l|ul|lu|ull|llu)",
            text,
        )
        if match is None:
            return crossgl_legacy_octal(text)

        digits, suffix = match.groups()
        if "u" in suffix.lower():
            unsigned_suffix = "U" if "U" in suffix else "u"
            return crossgl_legacy_octal(f"{digits}{unsigned_suffix}")
        return crossgl_legacy_octal(digits)

    def is_cuda_char_literal_text(self, value):
        return re.fullmatch(r"(?:u8|u|U|L)?'(?:[^'\\\n]|\\.)+'", str(value)) is not None

    def convert_cuda_char_literal_to_crossgl(self, value):
        text = str(value)
        match = re.fullmatch(r"(?:u8|u|U|L)?'(.*)'", text)
        if match is None:
            return text

        content = match.group(1)
        hex_match = re.fullmatch(r"\\x([0-9A-Fa-f]+)", content)
        if hex_match is not None:
            return str(int(hex_match.group(1), 16))

        octal_match = re.fullmatch(r"\\([0-7]+)", content)
        if octal_match is not None:
            return str(int(octal_match.group(1), 8))

        return f"'{content}'"

    def is_cuda_string_literal_text(self, value):
        return re.fullmatch(r'(?:u8|u|U|L)?"(?:[^"\\]|\\.)*"', str(value)) is not None

    def convert_cuda_string_literal_to_crossgl(self, value):
        match = re.fullmatch(r'(?:u8|u|U|L)?(".*")', str(value))
        return match.group(1) if match is not None else str(value)

    def is_cuda_qualified_identifier_text(self, value):
        text = str(value)
        if not text or text[0] in "\"'":
            return False
        return "::" in text and (("<" in text and ">" in text) or "(" in text)

    def emit(self, code):
        if code.strip():
            self.output.append("    " * self.indent_level + code)
        else:
            self.output.append("")

    def push_identifier_name_scope(self):
        self.identifier_name_scopes.append({})

    def pop_identifier_name_scope(self):
        if len(self.identifier_name_scopes) > 1:
            self.identifier_name_scopes.pop()

    def register_identifier_name(self, name):
        if not self.is_simple_identifier(name):
            return name
        output_name = self.sanitize_identifier_name(name)
        if not self.identifier_name_scopes:
            self.identifier_name_scopes.append({})
        self.identifier_name_scopes[-1][name] = output_name
        return output_name

    def output_identifier_name(self, name):
        if not self.is_simple_identifier(name):
            return name
        for scope in reversed(self.identifier_name_scopes):
            if name in scope:
                return scope[name]
        return name

    def sanitize_identifier_name(self, name):
        if name in self.CROSSGL_RESERVED_IDENTIFIERS:
            return unused_identifier_name(name, self.source_identifier_names)
        return name

    def rename_keyword_record_names(self, type_text):
        """Rename the structs that a type spelling such as ``let [2]`` names."""
        return re.sub(
            r"[A-Za-z_]\w*",
            lambda word: (
                self.sanitize_identifier_name(word.group())
                if word.group() in self.cuda_record_names
                else word.group()
            ),
            type_text,
        )

    def format_crossgl_array_extent(self, size):
        size = str(size).strip()
        if not size:
            return size
        simple_extent_pattern = r"[A-Za-z_][A-Za-z0-9_]*|(?:0[xX][0-9A-Fa-f]+|\d+)[uU]?"
        if re.fullmatch(simple_extent_pattern, size):
            return size

        replacements = {
            "*": "_mul_",
            "/": "_div_",
            "%": "_mod_",
            "+": "_plus_",
            "-": "_minus_",
        }
        extent = size
        for operator, word in replacements.items():
            extent = extent.replace(operator, word)
        extent = re.sub(r"[^A-Za-z0-9_]+", "_", extent).strip("_")
        extent = re.sub(r"_+", "_", extent)
        if not extent:
            extent = "expr"
        if extent[0].isdigit():
            extent = f"cuda_array_extent_{extent}"
        return self.sanitize_identifier_name(extent)

    def is_simple_identifier(self, name):
        return isinstance(name, str) and name.isidentifier()

    def sanitize_crossgl_type_identifier(self, name):
        parts = []
        for char in str(name):
            if char.isascii() and (char.isalnum() or char == "_"):
                parts.append(char)
            elif char == ":":
                if not parts or parts[-1] != "_":
                    parts.append("_")
            elif not parts or parts[-1] != "_":
                parts.append("_")

        raw_name = str(name)
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", raw_name):
            sanitized = raw_name
        else:
            sanitized = "".join(parts).strip("_") or "anonymous"
        if sanitized[0].isdigit():
            sanitized = f"type_{sanitized}"
        return self.sanitize_identifier_name(sanitized)

    def convert_unknown_cuda_template_type_to_crossgl(self, cuda_type):
        base_name, template_args = self.parse_cpp_template(cuda_type)
        if not template_args:
            return None

        converted_parts = [self.sanitize_crossgl_type_identifier(base_name)]
        for arg in template_args:
            converted_arg = self.convert_cuda_type_to_crossgl(arg)
            converted_parts.append(self.sanitize_crossgl_type_identifier(converted_arg))
        return "_".join(part for part in converted_parts if part)

    def convert_cuda_parameter_pack_type_to_crossgl(self, cuda_type):
        if "..." not in str(cuda_type):
            return None

        base_type = str(cuda_type).replace("...", "").strip()
        if not base_type:
            return "parameter_pack"

        converted_base_type = self.convert_cuda_type_to_crossgl(base_type)
        return f"{self.sanitize_crossgl_type_identifier(converted_base_type)}_pack"

    def convert_cuda_elaborated_type_to_crossgl(self, cuda_type):
        normalized_type = str(cuda_type).strip()
        if normalized_type.startswith("enum "):
            type_name = normalized_type[len("enum ") :].strip()
            return self.convert_cuda_type_to_crossgl(type_name)
        for prefix in ("class ", "struct ", "union "):
            if normalized_type.startswith(prefix):
                type_name = normalized_type[len(prefix) :].strip()
                # An array suffix binds before the pointer: `struct Pair* p[2]` is an array of pointers.
                if "[" in type_name:
                    return self.convert_cuda_type_to_crossgl(type_name)
                if "*" in type_name:
                    return self.convert_cuda_pointer_type(type_name)
                return self.sanitize_crossgl_type_identifier(type_name)
        return None

    def convert_cuda_scoped_type_to_crossgl(self, cuda_type):
        normalized_type = str(cuda_type).strip()
        if "::" not in normalized_type or ">::" not in normalized_type:
            return None

        return self.sanitize_crossgl_type_identifier(normalized_type)

    def convert_cuda_record_name_to_crossgl(self, name):
        template_record_name = self.convert_unknown_cuda_template_type_to_crossgl(name)
        if template_record_name is not None:
            return template_record_name
        return self.sanitize_crossgl_type_identifier(name)

    def collect_cuda_record_names(self, node):
        return {
            struct.name
            for struct in getattr(node, "structs", []) or []
            if hasattr(struct, "name")
        }

    def convert_cuda_function_name_to_crossgl(self, name):
        operator_name = self.convert_cuda_operator_function_name_to_crossgl(name)
        if operator_name is not None:
            return operator_name
        if self.is_simple_identifier(name):
            return self.sanitize_identifier_name(name)
        return self.sanitize_crossgl_type_identifier(name)

    def convert_cuda_operator_function_name_to_crossgl(self, name):
        if not isinstance(name, str) or not name.startswith("operator"):
            return None

        operator_symbol = name[len("operator") :].strip()
        operator_mapping = {
            "==": "equal",
            "!=": "not_equal",
            "<=>": "three_way",
            "<": "less",
            ">": "greater",
            "<=": "less_equal",
            ">=": "greater_equal",
            "+": "add",
            "-": "subtract",
            "*": "multiply",
            "/": "divide",
            "%": "modulo",
            "&&": "logical_and",
            "||": "logical_or",
            "!": "logical_not",
            "&": "bitwise_and",
            "|": "bitwise_or",
            "^": "bitwise_xor",
            "~": "bitwise_not",
            "<<": "shift_left",
            ">>": "shift_right",
            "[]": "subscript",
            "()": "call",
        }
        return f"operator_{operator_mapping.get(operator_symbol, 'overload')}"

    def crossgl_parameter_name(self, param, index):
        if getattr(param, "name", ""):
            return self.register_identifier_name(param.name)
        return f"_unused_param_{index}"

    def collect_user_function_names(self, node):
        names = set()

        def collect(current):
            if current is None:
                return
            if isinstance(current, list):
                for item in current:
                    collect(item)
                return

            name = getattr(current, "name", None)
            body = getattr(current, "body", None)
            if name is not None and body is not None:
                if not self.is_generated_matrix_helper_function(current):
                    names.add(name)

            for function in getattr(current, "functions", []):
                collect(function)
            for kernel in getattr(current, "kernels", []):
                collect(kernel)

        collect(node)
        names.discard(None)
        return names

    def collect_generated_matrix_helper_types(self, node):
        return {
            struct.name
            for struct in getattr(node, "structs", []) or []
            if self.is_generated_matrix_helper_struct(struct)
        }

    def native_matrix_helper_dimensions(self, type_name):
        match = re.fullmatch(r"(float|double)([234])x([234])", str(type_name))
        if match is None:
            return None
        scalar_type, columns, rows = match.groups()
        return scalar_type, int(columns), int(rows)

    def convert_native_matrix_helper_name_to_crossgl(self, type_name):
        if type_name not in self.generated_matrix_helper_types:
            return None

        dimensions = self.native_matrix_helper_dimensions(type_name)
        if dimensions is None:
            return None

        scalar_type, columns, rows = dimensions
        prefix = "dmat" if scalar_type == "double" else "mat"
        suffix = str(columns) if columns == rows else f"{columns}x{rows}"
        return f"{prefix}{suffix}"

    def is_generated_matrix_helper_struct(self, node):
        dimensions = self.native_matrix_helper_dimensions(getattr(node, "name", ""))
        if dimensions is None:
            return False

        scalar_type, columns, rows = dimensions
        members = getattr(node, "members", []) or []
        member_by_name = {
            getattr(member, "name", None): member
            for member in members
            if getattr(member, "name", None)
        }

        matrix_values = member_by_name.get("m")
        column_count = member_by_name.get("CGL_COLUMNS")
        row_count = member_by_name.get("CGL_ROWS")
        if matrix_values is None or column_count is None or row_count is None:
            return False

        return (
            getattr(matrix_values, "vtype", None) == f"{scalar_type}[{columns * rows}]"
            and getattr(column_count, "vtype", None) == "const int"
            and str(getattr(column_count, "value", "")).strip() == str(columns)
            and "static" in (getattr(column_count, "qualifiers", []) or [])
            and getattr(row_count, "vtype", None) == "const int"
            and str(getattr(row_count, "value", "")).strip() == str(rows)
            and "static" in (getattr(row_count, "qualifiers", []) or [])
        )

    def is_generated_matrix_helper_function(self, node):
        if getattr(node, "name", None) not in {"operator*", "transpose", "inverse"}:
            return False
        if not self.function_references_generated_matrix_helper_type(node):
            return False
        qualifiers = set(getattr(node, "qualifiers", []) or [])
        return {"__host__", "__device__", "inline"}.issubset(qualifiers)

    def function_references_generated_matrix_helper_type(self, node):
        candidate_types = [getattr(node, "return_type", "")]
        candidate_types.extend(getattr(param, "vtype", "") for param in node.params)
        return any(
            self.strip_type_qualifiers(candidate_type).strip()
            in self.generated_matrix_helper_types
            for candidate_type in candidate_types
        )

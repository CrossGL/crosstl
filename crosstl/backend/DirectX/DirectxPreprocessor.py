import os


class DirectxPreprocessor:
    def __init__(self):
        self.macros = {}
        self.condition_stack = []

    def preprocess(self, code):
        # Preprocess the input HLSL shader code.
        lines = code.splitlines()
        processed_lines = []
        skip_lines = False

        for line in lines:
            stripped_line = line.strip()

            if stripped_line.startswith("#include"):
                processed_lines.append(self.handle_include(stripped_line))
            elif stripped_line.startswith("#define"):
                self.handle_define(stripped_line)
            elif stripped_line.startswith("#ifdef"):
                skip_lines = not self.handle_ifdef(stripped_line)
            elif stripped_line.startswith("#endif"):
                skip_lines = self.handle_endif()
            elif stripped_line.startswith("#else"):
                skip_lines = not skip_lines
            elif not skip_lines:
                processed_lines.append(self.expand_macros(line))

        return "\n".join(processed_lines)

    def handle_include(self, line):
        # Handle #include directive.
        file_path = line.split()[1].strip('"<>"')
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Included file '{file_path}' not found.")
        with open(file_path, "r") as file:
            return file.read()

    def handle_define(self, line):
        # Handle #define directive.
        parts = line.split(maxsplit=2)
        if len(parts) == 3:
            self.macros[parts[1]] = parts[2]
        else:
            self.macros[parts[1]] = ""

    def handle_ifdef(self, line):
        # Handle #ifdef directive.
        macro = line.split(maxsplit=1)[1]
        is_defined = macro in self.macros
        self.condition_stack.append(is_defined)
        return is_defined

    def handle_endif(self):
        # Handle #endif directive.
        if not self.condition_stack:
            raise SyntaxError("#endif without matching #ifdef")
        self.condition_stack.pop()
        return False if self.condition_stack and not self.condition_stack[-1] else True

    def expand_macros(self, line):
        # Expand defined macros in the line.
        for macro, value in self.macros.items():
            line = line.replace(macro, value)

        # Check for undefined macros.
        if any(macro not in self.macros for macro in line.split()):
            raise ValueError(f"Undefined macro encountered in line: {line}")

        return line

"""Instruction tokenizing in the SPIR-V parser, including normalization of legacy glslang disassembly."""

import re
import shlex
import struct

from crosstl.translator.refusals import SpirvSyntaxError

from ..assembly_validation import (
    check_defined_ids,
    check_type_declaration_order,
    check_unique_result_ids,
)


class VulkanInstructionMixin:
    """Instruction members of ``VulkanParser``."""

    def parse_spirv_assembly_instructions(self, code):
        instructions = []
        if self.looks_like_legacy_glslang_spirv_disassembly(code):
            code = self.normalize_legacy_glslang_spirv_disassembly(code)

        tokens = self.spirv_assembly_tokens(code)
        index = 0
        id_operands = []

        while index < len(tokens):
            instruction_start = index
            operand_start = (
                instruction_start + 3
                if self.is_spirv_assembly_result_instruction_start(
                    tokens, instruction_start
                )
                else instruction_start + 1
            )
            next_index = self.next_spirv_assembly_instruction_index(
                tokens, operand_start
            )
            instruction_tokens = tokens[instruction_start:next_index]
            parts = [text for text, _quoted, _line_number in instruction_tokens]
            instruction_line_number = instruction_tokens[0][2]

            result_id = None
            if len(parts) >= 3 and parts[1] == "=":
                result_id = parts[0]
                opcode = parts[2]
                operands = parts[3:]
            else:
                opcode = parts[0]
                operands = parts[1:]

            if not opcode.startswith("Op"):
                raise SpirvSyntaxError(
                    "Expected SPIR-V opcode on line "
                    f"{instruction_line_number}, got {opcode}"
                )
            unquoted = [
                (text, line_number)
                for text, quoted, line_number in instruction_tokens[
                    len(parts) - len(operands) :
                ]
                if not quoted
            ]
            for text, line_number in unquoted:
                if text == "=":
                    raise SpirvSyntaxError(
                        f"Expected SPIR-V opcode after '=' on line {line_number}"
                    )
            id_operands.append(
                (
                    instruction_line_number,
                    opcode,
                    [text for text, _line in unquoted if text.startswith("%")],
                )
            )
            instructions.append((result_id, opcode, operands, instruction_line_number))
            index = next_index

        check_unique_result_ids(instructions)
        check_type_declaration_order(instructions)
        check_defined_ids(
            id_operands, {result_id for result_id, *_ in instructions if result_id}
        )
        return instructions

    def looks_like_legacy_glslang_spirv_disassembly(self, code):
        return bool(
            re.search(
                r"(?m)^\s*(?:"
                r"Capability\s+\w+\b"
                r"|MemoryModel\s+\w+\s+\w+\b"
                r"|EntryPoint\s+\w+\s+\d+\b"
                r"|\d+(?:\([^)]*\))?:\s+(?:Type[A-Za-z0-9_]+|Function|Label|Variable)\b"
                r")",
                code,
            )
        )

    def normalize_legacy_glslang_spirv_disassembly(self, code):
        normalized_lines = []
        module_index = -1
        id_namespace = None
        for raw_line in code.splitlines():
            if re.match(r"\s*//\s*Module Version\b", raw_line):
                module_index += 1
                id_namespace = f"m{module_index}_"
                continue

            line = raw_line.split("//", 1)[0].strip()
            if not line:
                continue
            try:
                parts = shlex.split(line, comments=False, posix=True)
            except ValueError as error:
                if self.can_skip_unparseable_legacy_glslang_spirv_line(line):
                    continue
                if self.legacy_glslang_spirv_line_has_instruction_shape(line):
                    raise SpirvSyntaxError(
                        f"Invalid glslang SPIR-V disassembly syntax: {error}"
                    ) from error
                continue

            instruction = self.normalize_legacy_glslang_spirv_instruction(
                parts, id_namespace=id_namespace
            )
            if instruction is not None:
                normalized_lines.append(instruction)

        return "\n".join(normalized_lines)

    def legacy_glslang_spirv_line_has_instruction_shape(self, line):
        opcode_pattern = "|".join(
            sorted(
                set(self.LEGACY_GLSLANG_SPIRV_ID_OPERANDS)
                | self.LEGACY_GLSLANG_SPIRV_GLOBAL_OPCODES
            )
        )
        return bool(
            re.match(r"\d+(?:\([^)]*\))?:\s+", line)
            or re.match(rf"(?:{opcode_pattern})\b", line)
        )

    def can_skip_unparseable_legacy_glslang_spirv_line(self, line):
        return bool(
            re.match(
                r"(?:\d+(?:\([^)]*\))?:\s+)?(?:Source|SourceContinued|String)\b",
                line,
            )
        )

    def normalize_legacy_glslang_spirv_instruction(self, parts, id_namespace=None):
        if not parts:
            return None

        result_id = None
        if self.is_legacy_glslang_spirv_result_token(parts[0]):
            result_id = self.legacy_glslang_spirv_id(
                parts.pop(0), namespace=id_namespace
            )
        if not parts:
            return None

        known_opcodes = (
            set(self.LEGACY_GLSLANG_SPIRV_ID_OPERANDS)
            | self.LEGACY_GLSLANG_SPIRV_GLOBAL_OPCODES
        )
        if result_id is None and parts[0] not in known_opcodes:
            return None

        if result_id is not None and parts[0] in known_opcodes:
            opcode = parts.pop(0)
            operands = [
                self.normalize_legacy_glslang_spirv_operand(
                    opcode, index, operand, id_namespace=id_namespace
                )
                for index, operand in enumerate(parts)
            ]
        elif result_id is not None and len(parts) >= 2 and parts[1] in known_opcodes:
            result_type_token = parts.pop(0)
            result_type = self.legacy_glslang_spirv_id(
                result_type_token, namespace=id_namespace
            )
            opcode = parts.pop(0)
            operands = [
                result_type,
                *[
                    self.normalize_legacy_glslang_spirv_operand(
                        opcode, index + 1, operand, id_namespace=id_namespace
                    )
                    for index, operand in enumerate(parts)
                ],
            ]
            operands = self.normalize_legacy_glslang_float_constant_operands(
                opcode, result_type_token, operands
            )
        else:
            opcode = parts.pop(0)
            operands = [
                self.normalize_legacy_glslang_spirv_operand(
                    opcode, index, operand, id_namespace=id_namespace
                )
                for index, operand in enumerate(parts)
            ]

        if not opcode.startswith("Op"):
            opcode = f"Op{opcode}"
        instruction = f"{result_id} = {opcode}" if result_id is not None else opcode
        if operands:
            instruction = f"{instruction} {' '.join(operands)}"
        return instruction

    def normalize_legacy_glslang_float_constant_operands(
        self, opcode, result_type_token, operands
    ):
        if opcode not in {"Constant", "SpecConstant"} or len(operands) < 2:
            return operands

        type_label = self.legacy_glslang_spirv_type_label(result_type_token)
        literal_widths = {
            "float16_t": 16,
            "float": 32,
            "float32_t": 32,
            "double": 64,
            "float64_t": 64,
        }
        width = literal_widths.get(type_label)
        if width is None:
            return operands

        word_count = 2 if width == 64 else 1
        literal_words = operands[1 : 1 + word_count]
        if len(literal_words) != word_count:
            return operands

        literal = self.legacy_glslang_float_literal_from_words(literal_words, width)
        if literal is None:
            return operands
        return [operands[0], literal, *operands[1 + word_count :]]

    def legacy_glslang_spirv_type_label(self, token):
        match = re.match(r"\d+\(([^)]*)\):?$", str(token))
        return match.group(1) if match else None

    def legacy_glslang_float_literal_from_words(self, words, width):
        try:
            integer_words = [int(str(word), 0) for word in words]
        except (TypeError, ValueError):
            return None

        if width == 16:
            data = (integer_words[0] & 0xFFFF).to_bytes(2, "little")
            value = struct.unpack("<e", data)[0]
        elif width == 32:
            data = (integer_words[0] & 0xFFFFFFFF).to_bytes(4, "little")
            value = struct.unpack("<f", data)[0]
        elif width == 64:
            data = b"".join(
                (word & 0xFFFFFFFF).to_bytes(4, "little") for word in integer_words
            )
            value = struct.unpack("<d", data)[0]
        else:
            return None
        return repr(value)

    def is_legacy_glslang_spirv_result_token(self, token):
        return bool(re.match(r"\d+(?:\([^)]*\))?:$", str(token)))

    def legacy_glslang_spirv_id(self, token, namespace=None):
        namespace = namespace or ""
        match = re.match(r"(\d+)(?:\([^)]*\))?:?$", str(token))
        if match:
            return f"%{namespace}{match.group(1)}"
        if re.match(r"\d+$", str(token)):
            return f"%{namespace}{token}"
        return token

    def normalize_legacy_glslang_spirv_operand(
        self, opcode, operand_index, operand, id_namespace=None
    ):
        if self.is_legacy_glslang_spirv_id_operand(opcode, operand_index):
            return self.legacy_glslang_spirv_id(operand, namespace=id_namespace)
        return operand

    def is_legacy_glslang_spirv_id_operand(self, opcode, operand_index):
        id_operands = self.LEGACY_GLSLANG_SPIRV_ID_OPERANDS.get(opcode)
        if id_operands == "all":
            return True
        if id_operands == "entry_point":
            return operand_index == 1 or operand_index >= 3
        if id_operands == "execution_mode_id":
            return operand_index == 0 or operand_index >= 2
        return operand_index in (id_operands or set())

    def spirv_assembly_tokens(self, code):
        tokens = []
        index = 0
        line_number = 1
        length = len(code)

        while index < length:
            char = code[index]
            if char.isspace():
                if char == "\n":
                    line_number += 1
                index += 1
                continue
            if char == ";":
                while index < length and code[index] != "\n":
                    index += 1
                continue
            if char == '"':
                token, index, line_number = self.spirv_assembly_string_token(
                    code, index, line_number
                )
                tokens.append(token)
                continue

            start = index
            token_line_number = line_number
            while index < length and not code[index].isspace() and code[index] != ";":
                index += 1
            tokens.append((code[start:index], False, token_line_number))

        return tokens

    def spirv_assembly_string_token(self, code, index, line_number):
        token_line_number = line_number
        value = []
        index += 1
        length = len(code)

        while index < length:
            char = code[index]
            if char == '"':
                token = ("".join(value), True, token_line_number)
                return token, index + 1, line_number
            if char == "\\":
                index += 1
                if index >= length:
                    break
                char = code[index]
            value.append(char)
            if char == "\n":
                line_number += 1
            index += 1

        raise SpirvSyntaxError(
            "Invalid SPIR-V assembly syntax on line "
            f"{token_line_number}: No closing quotation"
        )

    def next_spirv_assembly_instruction_index(self, tokens, index):
        while index < len(tokens):
            if self.is_spirv_assembly_instruction_start(tokens, index):
                return index
            index += 1
        return len(tokens)

    def is_spirv_assembly_instruction_start(self, tokens, index):
        if self.is_spirv_assembly_result_instruction_start(tokens, index):
            return True
        text, quoted, _line_number = tokens[index]
        if quoted:
            return False
        return text.startswith("Op")

    def is_spirv_assembly_result_instruction_start(self, tokens, index):
        if index + 2 >= len(tokens):
            return False
        text, quoted, _line_number = tokens[index]
        if quoted:
            return False
        equals, equals_quoted, _equals_line = tokens[index + 1]
        opcode, opcode_quoted, _opcode_line = tokens[index + 2]
        return (
            text.startswith("%")
            and equals == "="
            and not equals_quoted
            and opcode.startswith("Op")
            and not opcode_quoted
        )

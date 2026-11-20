"""Expression pieces of the Slang generator: call arguments, literals, precedence, bit queries and synchronization calls."""

import re

from ..ast import (
    BinaryOpNode,
    FunctionCallNode,
)
from ..refusals import SlangRefusal
from .array_utils import split_array_type_suffix
from .slang_exact_lookups import (
    SLANG_BITCAST_INTRINSICS,
    slang_foreign_call_reason,
)


class SlangExpressionPartsMixin:
    """Expression-part members of ``SlangCodeGen``."""

    def struct_member_type(self, object_type, member_name):
        if object_type is None or member_name is None:
            return None
        object_type = self.type_name_string(self.member_lookup_type_name(object_type))
        if not object_type:
            return None
        object_type, _array_suffix = split_array_type_suffix(object_type)
        member_type = self.struct_member_types.get(object_type, {}).get(member_name)
        if member_type is not None:
            return member_type
        if "<" in object_type and object_type.endswith(">"):
            base_type = object_type.split("<", 1)[0].strip()
            if base_type:
                return self.struct_member_types.get(base_type, {}).get(member_name)
        return None

    def struct_constructor_argument_types(self, struct_name):
        member_types = self.struct_member_types.get(struct_name)
        if not member_types:
            return []
        return list(member_types.values())

    def generate_struct_constructor_arguments(self, struct_name, args):
        expected_types = self.struct_constructor_argument_types(struct_name)
        return self.generate_call_arguments(args, expected_types)

    def user_function_argument_types(self, function_name):
        return self.user_function_parameter_types.get(function_name) or []

    def generate_user_function_arguments(self, function_name, args):
        expected_types = self.user_function_argument_types(function_name)
        generated_args = self.generate_call_argument_list(args, expected_types)
        existing_param_names = set()
        for parameter in self.required_slang_stage_parameters(function_name):
            name = getattr(parameter, "name", None)
            if not name or name in existing_param_names:
                continue
            generated_args.append(name)
        return ", ".join(generated_args)

    def slang_bitcast_intrinsic_callee(self, callee):
        if not isinstance(callee, str) or callee in self.user_function_names:
            return None
        return SLANG_BITCAST_INTRINSICS.get(callee)

    def slang_callee_name(self, callee):
        """The Slang name of a call to ``callee``, which the program does not declare; refuses a foreign built-in."""
        if callee in self.function_map:
            return self.function_map[callee]
        reason = slang_foreign_call_reason(callee, self.function_map)
        if reason is not None:
            return self.refuse_lowering(reason, callee)
        return callee

    def slang_bit_query_result_type(self, func_name, args):
        # Slang's countbits, firstbitlow and firstbithigh return uint components for any integer operand; CrossGL's
        # bitCount, findLSB and findMSB return int components (GLSL 4.60 section 8.8).
        result_scalar = {
            "bitCount": "int",
            "findLSB": "int",
            "findMSB": "int",
            "countbits": "uint",
            "firstbitlow": "uint",
            "firstbithigh": "uint",
        }.get(func_name)
        if result_scalar is None or len(args or ()) != 1:
            return None

        arg_type = self.expression_result_type(args[0])
        if arg_type is None:
            return None

        mapped = self.convert_type(arg_type)
        integer_scalars = {
            "int",
            "uint",
            "int8_t",
            "uint8_t",
            "int16_t",
            "uint16_t",
            "int64_t",
            "uint64_t",
        }
        if mapped in integer_scalars:
            return result_scalar
        for scalar in sorted(integer_scalars, key=len, reverse=True):
            if mapped.startswith(scalar):
                suffix = mapped[len(scalar) :]
                if suffix in {"2", "3", "4"}:
                    return f"{result_scalar}{suffix}"
        vector_match = re.fullmatch(
            r"vector<\s*([A-Za-z0-9_]+)\s*,\s*([234])\s*>", mapped
        )
        if vector_match and vector_match.group(1) in integer_scalars:
            return f"{result_scalar}{vector_match.group(2)}"
        return None

    def generate_slang_bit_query_call(self, func_name, args, generated_args):
        """``bitCount``, ``findLSB`` or ``findMSB`` as the Slang intrinsic converted to CrossGL's ``int`` result.

        The conversion keeps -1 for a scan that finds no bit, where the bare intrinsic would give 0xFFFFFFFF.
        Returns ``None`` for any other call.
        """
        if func_name not in {"bitCount", "findLSB", "findMSB"}:
            return None
        result_type = self.slang_bit_query_result_type(func_name, args)
        if result_type is None:
            return self.refuse_lowering(
                f"Slang {func_name}: the operand is not a known integer scalar or vector",
                "0",
            )
        return f"{result_type}({self.function_map[func_name]}({generated_args}))"

    def glsl_bitcast_result_type(self, func_name, args):
        component_type = {
            "floatBitsToInt": "int",
            "floatBitsToUint": "uint",
            "intBitsToFloat": "float",
            "uintBitsToFloat": "float",
        }.get(func_name)
        if component_type is None or not args:
            return None

        arg_type = self.expression_result_type(args[0])
        if arg_type is None:
            return None

        arg_info = self.vector_value_info(arg_type)
        if arg_info is not None:
            return f"{component_type}{arg_info['size']}"
        if self.is_scalar_value_type(arg_type):
            return component_type
        return None

    def generate_call_arguments(self, args, expected_types):
        return ", ".join(self.generate_call_argument_list(args, expected_types))

    def generate_call_argument_list(self, args, expected_types):
        generated_args = []
        for index, arg in enumerate(args or []):
            expected_type = (
                expected_types[index] if index < len(expected_types) else None
            )
            if expected_type is None:
                generated_args.append(self.generate_expression(arg))
            else:
                generated_args.append(
                    self.generate_expression_with_expected(arg, expected_type)
                )
        return generated_args

    def generate_literal(self, node):
        value = node.value
        literal_type = getattr(getattr(node, "literal_type", None), "name", None)

        if isinstance(value, bool):
            return "true" if value else "false"
        if literal_type == "bool" and isinstance(value, str):
            lower_value = value.lower()
            if lower_value in {"true", "false"}:
                return lower_value
        if literal_type == "char":
            escaped = self.escape_literal(value, quote="'")
            return f"'{escaped}'"
        if (
            literal_type == "uint"
            and isinstance(value, int)
            and not isinstance(value, bool)
        ):
            return f"{value}u"
        if isinstance(value, str):
            escaped = self.escape_literal(value, quote='"')
            return f'"{escaped}"'
        return str(value)

    def escape_literal(self, value, quote):
        text = str(value)
        escaped = []
        for index, char in enumerate(text):
            if char == "\n":
                escaped.append("\\n")
            elif char == "\r":
                escaped.append("\\r")
            elif char == "\t":
                escaped.append("\\t")
            elif char == quote and (index == 0 or text[index - 1] != "\\"):
                escaped.append("\\" + char)
            else:
                escaped.append(char)
        return "".join(escaped)

    def binary_precedence(self, op):
        return self.BINARY_PRECEDENCE.get(op, 0)

    def binary_child_needs_parentheses(self, parent_op, child, is_right_child=False):
        if not isinstance(child, BinaryOpNode):
            return False

        parent_precedence = self.binary_precedence(parent_op)
        child_op = getattr(child, "op", getattr(child, "operator", ""))
        child_precedence = self.binary_precedence(child_op)
        if child_precedence < parent_precedence:
            return True
        if child_precedence > parent_precedence:
            return False
        return is_right_child and (
            parent_op not in self.ASSOCIATIVE_BINARY_OPS or child_op != parent_op
        )

    def slang_helper_invocation_expression(self):
        """A read of ``gl_HelperInvocation`` as Slang's ``IsHelperLane()``, unless a variable of that name is in scope.

        Slang declares no ``gl_HelperInvocation``; ``IsHelperLane()`` is available in fragment stages only, so it is
        refused in another stage's entry point. A helper function is left to slangc, which checks the stages that
        call it.
        """
        if "gl_HelperInvocation" in self.variable_types:
            return "gl_HelperInvocation"
        shader_type = self.current_shader_type
        if (
            shader_type is not None
            and self.slang_shader_stage_name(shader_type) != "fragment"
        ):
            self.refuse_lowering(
                "Slang gl_HelperInvocation: lowered to IsHelperLane() only in a "
                f"fragment stage, not in a {shader_type} entry point"
            )
        return "IsHelperLane()"

    def function_call_simple_callee_name(self, node):
        if not isinstance(node, FunctionCallNode):
            return None

        func_expr = getattr(node, "function", None)
        if func_expr is None:
            func_expr = getattr(node, "name", None)
        if hasattr(func_expr, "name") and getattr(func_expr, "name", None):
            return func_expr.name
        if isinstance(func_expr, str):
            return func_expr
        return None

    def generate_slang_synchronization_statement(self, node):
        callee = self.function_call_simple_callee_name(node)
        call = self.generate_slang_synchronization_call(
            callee, getattr(node, "args", []), statement_context=True
        )
        if call is None:
            return None
        return f"{call};"

    def generate_slang_synchronization_call(
        self, callee, args, statement_context=False
    ):
        if not isinstance(callee, str) or callee in self.user_function_names:
            return None

        intrinsic = self.slang_synchronization_intrinsic_name(callee)
        if intrinsic is None:
            return None
        if args:
            raise SlangRefusal(
                f"Slang synchronization builtin '{callee}' requires 0 "
                f"argument(s), got {len(args)}"
            )
        if not statement_context:
            raise SlangRefusal(
                f"Slang synchronization builtin '{callee}' is statement-only "
                "and cannot be used as a value"
            )
        return f"{intrinsic}()"

    def generate_slang_geometry_stream_member_call(
        self, func_expr, args, statement_context=False
    ):
        receiver = getattr(func_expr, "object", getattr(func_expr, "object_expr", None))
        receiver_type = self.expression_result_type(receiver)
        if not self.is_slang_geometry_stream_type(receiver_type):
            return None

        member = str(getattr(func_expr, "member", ""))
        self.validate_slang_geometry_stream_call(member, args, receiver_type)
        shader_stage = self.slang_shader_stage_name(self.current_shader_type)
        if self.current_shader_type is not None and shader_stage != "geometry":
            raise SlangRefusal(
                f"Slang {self.current_shader_type} stage cannot call geometry "
                f"stream {member}; geometry stream methods are only valid in "
                "geometry stages"
            )
        if not statement_context:
            raise SlangRefusal(
                f"Slang geometry stream {member} is statement-only and cannot "
                "be used as a value"
            )

        receiver_expr = self.generate_expression(receiver)
        args_expr = ", ".join(self.generate_expression(arg) for arg in args)
        return f"{receiver_expr}.{member}({args_expr})"

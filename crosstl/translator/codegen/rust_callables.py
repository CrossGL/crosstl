"""Rust lambdas, callable types and calls of user functions, struct and enum constructors and vector and matrix constructors."""

import re

from ..ast import (
    BlockNode,
    IdentifierNode,
    LambdaNode,
    LiteralNode,
)
from .rust_exact_lookups import RUST_PRIMITIVE_SOURCE_TYPES
from .rust_gpu_spelling import GLAM_AXES
from .rust_refusals import RustEnumEmptyBraceError

# A matrix as the generator spells it before glam names it (``Mat2<f64>``), or glam's own name (``DMat2``).
_RUST_MATRIX_TYPE = re.compile(r"D?Mat[234](?:x[234])?(?:<\w+>)?")


class RustCallableMixin:
    """Callable and call members of ``RustCodeGen``."""

    def generate_lambda_expression(self, args):
        """Render CrossGL's compact pseudo-lambda as a native Rust closure."""
        if not args:
            return "|| ()"

        params = ", ".join(self.generate_lambda_parameter(arg) for arg in args[:-1])
        body = self.generate_lambda_body(args[-1])
        return f"|{params}| {body}"

    def generate_lambda_node_expression(self, node, target_type=None):
        """Render a first-class LambdaNode as a native Rust closure."""
        saved_variable_types = self.variable_types.copy()
        saved_local_variable_names = self.local_variable_names.copy()
        try:
            params = []
            parameter_target_types = self.callable_parameter_target_types(target_type)
            for index, param in enumerate(getattr(node, "parameters", []) or []):
                param_type = self.function_parameter_type(param)
                if self.is_inferred_declaration_type(param_type):
                    if index < len(parameter_target_types):
                        param_type = parameter_target_types[index]
                self.register_variable_type(param.name, param_type, scope="local")
                params.append(self.generate_lambda_node_parameter(param, param_type))

            return_type = self.lambda_node_return_target_type(node, target_type)
            body = self.generate_lambda_node_body(
                getattr(node, "body", None),
                return_type=return_type,
            )
            move_prefix = "move " if self.force_move_lambda_depth > 0 else ""
            return f"{move_prefix}|{', '.join(params)}| {body}"
        finally:
            self.variable_types = saved_variable_types
            self.local_variable_names = saved_local_variable_names

    def generate_lambda_node_parameter(self, param, param_type):
        if self.is_inferred_declaration_type(param_type):
            return param.name
        return f"{param.name}: {self.map_type(param_type)}"

    def lambda_node_return_target_type(self, node, target_type):
        explicit_type = getattr(node, "return_type", None)
        if explicit_type is None and hasattr(node, "get_annotation"):
            explicit_type = node.get_annotation("return_type")
        if explicit_type is not None:
            return self.type_name_string(explicit_type)

        return self.callable_return_target_type(target_type)

    def callable_return_target_type(self, type_name):
        type_name = self.type_name_string(type_name)
        if type_name is None:
            return None

        return_type = self.split_callable_return_type(type_name)
        if return_type is None:
            return None
        return self.source_type_from_rust_type(return_type)

    def callable_parameter_target_types(self, type_name):
        signature = self.callable_signature_parts(type_name)
        if signature is None:
            return []

        _callable_trait, params_text, _return_type = signature
        if not params_text:
            return []

        return [
            self.source_type_from_rust_type(param_type.strip())
            for param_type in self.split_top_level_list(params_text)
            if param_type.strip()
        ]

    def callable_type_uses_trait(self, type_name, trait_name):
        signature = self.callable_signature_parts(type_name)
        if signature is None:
            return False

        callable_trait, _params_text, _return_type = signature
        return callable_trait == trait_name

    def callable_signature_parts(self, type_name):
        type_name = self.type_name_string(type_name)
        if type_name is None:
            return None

        type_name = type_name.strip()
        match = re.match(r"^(?:dyn\s+|impl\s+)?(FnOnce|FnMut|Fn)\s*\(", type_name)
        if match is None:
            return None

        open_index = type_name.find("(", match.start())
        close_index = self.find_matching_delimiter(type_name, open_index, "(", ")")
        if close_index is None:
            return None

        params_text = type_name[open_index + 1 : close_index].strip()
        suffix = type_name[close_index + 1 :].strip()
        return_type = "()"
        if suffix.startswith("->"):
            return_type = suffix[2:].strip()

        return match.group(1), params_text, return_type

    def split_callable_return_type(self, type_name):
        paren_depth = 0
        bracket_depth = 0
        angle_depth = 0
        index = 0
        while index < len(type_name) - 1:
            char = type_name[index]
            next_char = type_name[index + 1]
            if (
                char == "-"
                and next_char == ">"
                and paren_depth == 0
                and bracket_depth == 0
                and angle_depth == 0
            ):
                return type_name[index + 2 :].strip()

            if char == "(":
                paren_depth += 1
            elif char == ")":
                paren_depth = max(0, paren_depth - 1)
            elif char == "[":
                bracket_depth += 1
            elif char == "]":
                bracket_depth = max(0, bracket_depth - 1)
            elif char == "<":
                angle_depth += 1
            elif char == ">":
                angle_depth = max(0, angle_depth - 1)
            index += 1
        return None

    def type_name_string(self, type_name):
        if type_name is None:
            return None
        if self.is_type_node(type_name):
            return self.convert_type_node_to_string(type_name)
        return str(type_name)

    def is_type_node(self, value):
        return (
            value.__class__.__name__ in {
                "ReferenceType",
                "PointerType",
                "FunctionType",
                "ArrayType",
            }
            or hasattr(value, "name")
            or hasattr(value, "element_type")
        )

    def source_type_from_rust_type(self, type_name):
        type_name = self.type_name_string(type_name)
        if type_name is None:
            return None
        type_name = type_name.strip()

        array_parts = self.rust_array_type_parts(type_name)
        if array_parts is not None:
            base_type, sizes = array_parts
            return self.format_c_array_type(base_type, sizes)

        source_type = RUST_PRIMITIVE_SOURCE_TYPES.get(type_name)
        if source_type is not None:
            return source_type
        # A declared structure, enum or generic type is spelled the same in Rust and CrossGL.
        return type_name

    def rust_array_type_parts(self, type_name):
        type_name = self.type_name_string(type_name)
        if type_name is None:
            return None
        type_name = type_name.strip()
        if not (type_name.startswith("[") and type_name.endswith("]")):
            return None

        body = type_name[1:-1].strip()
        separator = self.find_top_level_separator(body, ";")
        if separator is None:
            return None

        element_type = body[:separator].strip()
        size_text = body[separator + 1 :].strip()
        try:
            size = int(size_text)
        except ValueError:
            size = None

        nested_parts = self.rust_array_type_parts(element_type)
        if nested_parts is not None:
            base_type, nested_sizes = nested_parts
            return base_type, [size, *nested_sizes]
        return self.source_type_from_rust_type(element_type), [size]

    def find_top_level_separator(self, text, separator):
        paren_depth = 0
        bracket_depth = 0
        angle_depth = 0
        for index, char in enumerate(text):
            if char == separator and all(
                depth == 0 for depth in (paren_depth, bracket_depth, angle_depth)
            ):
                return index
            if char == "(":
                paren_depth += 1
            elif char == ")":
                paren_depth = max(0, paren_depth - 1)
            elif char == "[":
                bracket_depth += 1
            elif char == "]":
                bracket_depth = max(0, bracket_depth - 1)
            elif char == "<":
                angle_depth += 1
            elif char == ">":
                angle_depth = max(0, angle_depth - 1)
        return None

    def find_matching_delimiter(self, text, open_index, open_char, close_char):
        if open_index < 0 or open_index >= len(text) or text[open_index] != open_char:
            return None

        depth = 0
        for index in range(open_index, len(text)):
            char = text[index]
            if char == open_char:
                depth += 1
            elif char == close_char:
                depth -= 1
                if depth == 0:
                    return index
        return None

    def split_top_level_list(self, text, separator=","):
        parts = []
        current = []
        paren_depth = 0
        bracket_depth = 0
        angle_depth = 0
        for char in text:
            if char == separator and all(
                depth == 0 for depth in (paren_depth, bracket_depth, angle_depth)
            ):
                parts.append("".join(current))
                current = []
                continue
            current.append(char)

            if char == "(":
                paren_depth += 1
            elif char == ")":
                paren_depth = max(0, paren_depth - 1)
            elif char == "[":
                bracket_depth += 1
            elif char == "]":
                bracket_depth = max(0, bracket_depth - 1)
            elif char == "<":
                angle_depth += 1
            elif char == ">":
                angle_depth = max(0, angle_depth - 1)

        if current:
            parts.append("".join(current))
        return parts

    def generate_lambda_node_body(self, body, return_type=None):
        saved_return_type = self.current_return_type
        saved_output_parameters = self.current_rust_output_parameters
        self.current_return_type = return_type
        self.current_rust_output_parameters = []
        try:
            if isinstance(body, BlockNode):
                return self.generate_block_expression(
                    body,
                    target_type=return_type,
                    return_context=True,
                )

            return self.generate_return_branch_expression_with_type(body, return_type)
        finally:
            self.current_return_type = saved_return_type
            self.current_rust_output_parameters = saved_output_parameters

    def generate_lambda_parameter(self, arg):
        raw = self.lambda_raw_argument_text(arg).strip()
        typed_param = self.split_lambda_typed_parameter(raw)
        if typed_param is None:
            return raw or "_"

        type_name, param_name = typed_param
        return f"{self.rust_identifier(param_name)}: {self.map_type(type_name)}"

    def generate_lambda_body(self, arg):
        raw = self.lambda_raw_argument_text(arg).strip()
        if raw:
            return raw
        return self.generate_expression(arg)

    def lambda_raw_argument_text(self, arg):
        if isinstance(arg, IdentifierNode):
            return arg.name
        if isinstance(arg, str):
            return arg
        return self.generate_expression(arg)

    def split_lambda_typed_parameter(self, raw):
        if not raw or ":" in raw:
            return None
        if any(char in raw for char in "{}()"):
            return None

        parts = raw.rsplit(None, 1)
        if len(parts) != 2:
            return None

        type_name, param_name = parts
        if not param_name.isidentifier():
            return None
        if type_name in {"mut", "ref"}:
            return None
        return type_name, param_name

    def generate_bool_mix_expression(self, args):
        condition_type = self.expression_result_type(args[2])
        condition_info = self.vector_type_info(condition_type)
        if condition_info is not None:
            if condition_info["component_type"] != "bool":
                return None
            return self.generate_bool_vector_ternary_expression(
                args[2], args[1], args[0]
            )

        if self.normalize_scalar_type(condition_type) != "bool":
            return None

        result_type = self.promoted_bool_mix_scalar_type(args[0], args[1])
        if result_type is None:
            return None

        condition = self.generate_condition_expression(args[2])
        true_value = self.generate_expression_with_type(args[1], result_type)
        false_value = self.generate_expression_with_type(args[0], result_type)
        true_value = self.normalize_typed_expression_value(
            args[1], true_value, result_type
        )
        false_value = self.normalize_typed_expression_value(
            args[0], false_value, result_type
        )
        return f"(if {condition} {{ {true_value} }} else {{ {false_value} }})"

    def generate_function_call_with_target(self, expr, target_type):
        func_expr = getattr(expr, "function", getattr(expr, "name", "unknown"))
        func_name = getattr(func_expr, "name", func_expr)
        if not isinstance(func_name, str):
            return None

        args = getattr(expr, "arguments", getattr(expr, "args", []))
        enum_variant = self.generate_enum_variant_call_with_typed_args(
            func_name,
            args,
            target_type=target_type,
        )
        if enum_variant is not None:
            return enum_variant

        if self.is_user_defined_function(func_name):
            return None

        struct_constructor = self.generate_struct_new_call_with_typed_args(
            func_name,
            args,
        )
        if struct_constructor is not None:
            return struct_constructor

        vector_info = self.vector_type_info(func_name)
        if vector_info is None:
            return None

        target_rust_type = self.vector_constructor_target_type(
            target_type,
            vector_info,
        )
        if target_rust_type is None:
            return None

        generated_args, temp_bindings = self.generate_vector_constructor_args(
            vector_info,
            args,
        )
        return self.generate_constructor_call(
            target_rust_type,
            generated_args,
            temp_bindings,
        )

    def generate_struct_new_call_with_typed_args(self, func_name, args):
        type_name = self.struct_constructor_call_type_name(func_name)
        if type_name is None:
            return None

        member_types = self.resolve_struct_positional_member_types(type_name)
        if not member_types:
            return None

        generated_args = self.generate_constructor_call_args_with_types(
            args,
            member_types,
        )
        return f"{self.struct_new_call_path(type_name)}({', '.join(generated_args)})"

    def vector_constructor_target_type(self, target_type, source_vector_info):
        if target_type is None:
            return None

        target_info = self.vector_type_info(target_type)
        if target_info is None:
            return None
        if target_info["size"] != source_vector_info["size"]:
            return None

        return self.map_type(target_type)

    def promoted_bool_mix_scalar_type(self, false_expr, true_expr):
        false_type = self.expression_result_type(false_expr)
        true_type = self.expression_result_type(true_expr)
        if self.vector_type_info(false_type) or self.vector_type_info(true_type):
            return None
        return self.promoted_scalar_type(false_type, true_type)

    def struct_new_call_type_name(self, func_name):
        if not isinstance(func_name, str) or not func_name.endswith("::new"):
            return None

        type_name = func_name[: -len("::new")]
        return type_name.replace("::<", "<", 1)

    def struct_constructor_call_type_name(self, func_name):
        type_name = self.struct_new_call_type_name(func_name)
        if type_name is not None:
            return type_name

        type_name = self.normalize_turbofish_type_name(func_name)
        if (
            not isinstance(type_name, str)
            or "::" in type_name
            or not self.resolve_struct_positional_member_types(type_name)
        ):
            return None
        return type_name

    def struct_new_call_path(self, type_name):
        rust_type = self.map_type(type_name)
        return f"{self.rust_constructor_path(rust_type)}::new"

    def enum_variant_call_info(self, func_name, target_type=None):
        if not isinstance(func_name, str) or "::" not in func_name:
            return None

        enum_path, variant_name = func_name.rsplit("::", 1)
        if not enum_path or not variant_name:
            return None

        enum_type = self.normalize_turbofish_type_name(enum_path)
        enum_base, enum_args = self.generic_type_parts(enum_type)
        target_type = self.normalize_turbofish_type_name(target_type)
        target_base, target_args = (
            self.generic_type_parts(target_type) if target_type else (None, [])
        )
        if target_base == enum_base and target_args and not enum_args:
            enum_type = target_type

        enum_base, _enum_args = self.generic_type_parts(enum_type)
        variants = self.enum_variant_names.get(enum_base, [])
        if variant_name not in variants:
            return None

        payload_types = self.resolve_enum_variant_payload_types(
            enum_type,
            variant_name,
        )
        return enum_type, variant_name, payload_types

    def enum_variant_call_result_type(self, func_name):
        variant_info = self.enum_variant_call_info(func_name)
        if variant_info is None:
            return None
        enum_type, _variant_name, _payload_types = variant_info
        return enum_type

    def resolve_enum_variant_payload_types(self, enum_type, variant_name):
        if enum_type is None or variant_name is None:
            return []

        enum_type = self.normalize_turbofish_type_name(enum_type)
        exact_variants = self.enum_variant_payload_type_map.get(enum_type)
        if exact_variants is not None:
            return list(exact_variants.get(variant_name, []))

        enum_base, generic_args = self.generic_type_parts(enum_type)
        variant_payloads = self.enum_variant_payload_type_map.get(enum_base, {})
        payload_types = variant_payloads.get(variant_name)
        if payload_types is None:
            return []

        generic_params = self.enum_generic_params.get(
            enum_base,
            self.struct_generic_params.get(enum_base, []),
        )
        substitutions = dict(zip(generic_params, generic_args))
        return [
            self.substitute_type_parameters(payload_type, substitutions)
            for payload_type in payload_types
        ]

    def resolve_enum_variant_field_types(self, enum_type, variant_name):
        if enum_type is None or variant_name is None:
            return {}

        enum_type = self.normalize_turbofish_type_name(enum_type)
        exact_variants = self.enum_variant_field_types.get(enum_type)
        if exact_variants is not None:
            return dict(exact_variants.get(variant_name, {}))

        enum_base, generic_args = self.generic_type_parts(enum_type)
        variant_fields = self.enum_variant_field_types.get(enum_base, {}).get(
            variant_name,
        )
        if not variant_fields:
            return {}

        generic_params = self.enum_generic_params.get(
            enum_base,
            self.struct_generic_params.get(enum_base, []),
        )
        substitutions = dict(zip(generic_params, generic_args))
        return {
            field_name: self.substitute_type_parameters(field_type, substitutions)
            for field_name, field_type in variant_fields.items()
        }

    def enum_variant_struct_constructor_info(self, type_name, target_type=None):
        if not isinstance(type_name, str) or "::" not in type_name:
            return None

        enum_path, variant_name = self.split_variant_path(
            self.normalize_turbofish_type_name(type_name)
        )
        if enum_path is None or variant_name is None:
            return None

        enum_type = enum_path
        enum_base, enum_args = self.generic_type_parts(enum_type)
        target_type = self.normalize_turbofish_type_name(target_type)
        target_base, target_args = (
            self.generic_type_parts(target_type) if target_type else (None, [])
        )
        if target_base == enum_base and target_args and not enum_args:
            enum_type = target_type
            enum_base, enum_args = self.generic_type_parts(enum_type)

        variants = self.enum_variant_names.get(enum_base, [])
        if variant_name not in variants:
            return None

        field_types = self.resolve_enum_variant_field_types(enum_type, variant_name)
        if not field_types:
            return None

        return enum_type, variant_name, field_types

    def rust_empty_brace_enum_initializer(self, target_type):
        """``Mode::First`` for ``Mode m = {};`` when a plain variant's discriminant is 0.

        Returns ``None`` when ``target_type`` is not an enumeration. Raises
        :class:`RustEnumEmptyBraceError` when none of the variants is the value 0, or
        when the variants carry data, so that the enumeration has no zero value.
        """
        type_name = self.type_name_string(target_type)
        if not type_name:
            return None
        base_name = type_name.split("<", 1)[0].strip()
        enum_node = self.enum_nodes_by_name.get(base_name)
        if enum_node is None:
            return None

        variants = getattr(enum_node, "variants", []) or []
        if any(self.enum_variant_data(variant) for variant in variants):
            raise RustEnumEmptyBraceError(enum_node.name, "its variants carry data")

        next_value = 0
        for variant in variants:
            value = next_value
            if getattr(variant, "value", None) is not None:
                value = self.literal_int_value(variant.value)
            if value == 0:
                return self.enum_variant_call_path(enum_node.name, variant.name)
            next_value = None if value is None else value + 1

        raise RustEnumEmptyBraceError(enum_node.name, "no variant has the value 0")

    def rust_empty_brace_enum_array_element(self, element_type):
        """The zero value of an element of ``element_type``, an enumeration or an array of one.

        An element left out of a brace list is value-initialized, as ``{}`` is: ``M::B`` for
        ``enum M { A = 2, B = 0 }``, and ``[M::B; 3]`` for an ``M[3]`` element. Returns ``None``
        for any other type; raises :class:`RustEnumEmptyBraceError` as
        :meth:`rust_empty_brace_enum_initializer` does.
        """
        if self.is_array_type_name(element_type):
            inner_type, size = self.c_array_outer_element_type_and_size(element_type)
            if size is None:
                return None
            inner = self.rust_empty_brace_enum_array_element(inner_type)
            return None if inner is None else f"[{inner}; {size}]"
        return self.rust_empty_brace_enum_initializer(element_type)

    def enum_variant_call_path(self, enum_type, variant_name):
        rust_type = self.map_type(enum_type)
        return f"{self.rust_constructor_path(rust_type)}::{variant_name}"

    def normalize_turbofish_type_name(self, type_name):
        if type_name is None:
            return None
        if hasattr(type_name, "name") or hasattr(type_name, "element_type"):
            type_name = self.convert_type_node_to_string(type_name)
        else:
            type_name = str(type_name)
        return type_name.replace("::<", "<", 1)

    def generate_user_function_call_args(self, func_name, args):
        later_roots = self.return_call_arg_later_roots(args)
        lambda_conflict_roots = self.return_call_arg_lambda_conflict_roots(args)
        generated_args = []
        for index, arg in enumerate(args):
            param_type = self.user_function_param_type(func_name, index)
            target_type = self.user_function_argument_target_type(
                func_name,
                arg,
                param_type,
            )
            if target_type is None:
                generated_args.append(
                    self.generate_with_lambda_capture_move_blocks(
                        arg,
                        later_roots[index],
                        lambda: self.generate_expression(arg),
                        lambda_conflict_roots[index],
                    )
                )
                continue

            arg_expr = self.generate_with_lambda_capture_move_blocks(
                arg,
                later_roots[index],
                lambda: self.generate_expression_with_type(arg, target_type),
                lambda_conflict_roots[index],
            )
            arg_expr = self.normalize_user_function_call_arg(
                arg,
                arg_expr,
                target_type,
            )
            generated_args.append(arg_expr)
        self.apply_rust_output_arguments(func_name, args, generated_args)
        generated_args.extend(self.generate_captured_function_call_args(func_name))
        return generated_args

    def generate_captured_function_call_args(self, func_name):
        captured_params = self.rust_function_capture_params.get(func_name, [])
        return [
            self.generate_expression_with_type(
                IdentifierNode(param.name), param.param_type
            )
            for param in captured_params
        ]

    def user_function_argument_target_type(self, func_name, arg, param_type):
        if not isinstance(arg, LambdaNode):
            return param_type

        callable_type = self.callable_parameter_target_type(func_name, param_type)
        return callable_type or param_type

    def callable_parameter_target_type(self, func_name, param_type):
        if self.callable_return_target_type(param_type) is not None:
            return param_type

        param_name = self.type_name_string(param_type)
        if not param_name:
            return None

        callee = self.user_function_nodes.get(func_name)
        if callee is None:
            return None

        constraints = self.combined_function_generic_constraints(callee)
        for bound in constraints.get(param_name, []):
            if self.callable_return_target_type(bound) is not None:
                return bound
        return None

    def normalize_user_function_call_arg(self, arg, generated_arg, param_type):
        return self.normalize_typed_expression_value(arg, generated_arg, param_type)

    def user_function_param_type(self, func_name, index):
        param_types = self.user_function_param_types.get(func_name, [])
        return param_types[index] if index < len(param_types) else None

    def normalize_vector_typed_expression(self, expr, target_type):
        target_info = self.vector_type_info(target_type)
        source_info = self.vector_type_info(self.expression_result_type(expr))
        if target_info is None or source_info is None:
            return None
        if target_info["size"] != source_info["size"]:
            return None
        if target_info["component_type"] == source_info["component_type"]:
            return None

        temp_bindings = []
        lanes = self.vector_argument_lane_expressions(
            expr, source_info, temp_bindings, target_info["component_type"]
        )
        lanes = lanes[: target_info["size"]]
        return self.generate_constructor_call(
            self.map_type(target_type), lanes, temp_bindings
        )

    def normalize_matrix_typed_expression(self, expr, target_type):
        target_info = self.matrix_type_info(target_type)
        source_info = self.matrix_type_info(self.expression_result_type(expr))
        if target_info is None or source_info is None:
            return None
        if (
            target_info["columns"] != source_info["columns"]
            or target_info["rows"] != source_info["rows"]
        ):
            return None
        if target_info["component_type"] == source_info["component_type"]:
            return None

        temp_bindings = []
        lanes = self.matrix_argument_lane_expressions(
            expr, source_info, temp_bindings, target_info["component_type"]
        )
        return self.generate_constructor_call(
            self.map_type(target_type), lanes, temp_bindings
        )

    def generate_scalar_constructor_call(self, func_name, args):
        rust_type = self.scalar_constructor_type(func_name)
        if rust_type is None or len(args) != 1:
            return None

        arg = args[0]
        arg_expr = self.generate_expression(arg)

        if rust_type == "bool":
            arg_type = self.expression_result_type(arg)
            if arg_type == "bool":
                return arg_expr
            zero_literal = "0.0" if arg_type in {"float", "double", "half"} else "0"
            return f"({arg_expr} != {zero_literal})"

        if (
            rust_type in {"f16", "f32", "f64"}
            and self.normalize_scalar_type(self.expression_result_type(arg)) == "bool"
        ):
            # Rust casts bool only to integers (E0606), and true becomes 1.
            return f"(({arg_expr} as u32) as {rust_type})"
        return f"({arg_expr} as {rust_type})"

    def generate_vector_constructor_call(self, func_name, vector_info, args):
        rust_type = self.map_type(func_name)
        generated_args, temp_bindings = self.generate_vector_constructor_args(
            vector_info, args
        )
        return self.generate_constructor_call(rust_type, generated_args, temp_bindings)

    def generate_matrix_constructor_call(self, func_name, args):
        rust_type = self.map_type(func_name)
        matrix_info = self.matrix_type_info(func_name)
        generated_args, temp_bindings = self.generate_matrix_constructor_args(
            matrix_info, args
        )
        # glam builds a matrix from its components in column order (Mat3::from_cols_array).
        constructor = (
            f"{self.rust_constructor_path(rust_type)}::from_cols_array("
            f"&[{', '.join(generated_args)}])"
        )
        if not temp_bindings:
            return constructor
        bindings = " ".join(
            self.generate_temp_binding_statement(binding) for binding in temp_bindings
        )
        return f"{{ {bindings} {constructor} }}"

    def generate_constructor_call(self, rust_type, generated_args, temp_bindings):
        args_str = ", ".join(generated_args)
        if _RUST_MATRIX_TYPE.fullmatch(str(rust_type)):
            # glam has no lane-wise matrix ``new``; it builds a matrix from its components in column order.
            constructor = f"{self.rust_constructor_path(rust_type)}::from_cols_array(&[{args_str}])"
        else:
            constructor = f"{self.rust_constructor_path(rust_type)}::new({args_str})"
        if not temp_bindings:
            return constructor

        bindings = " ".join(
            self.generate_temp_binding_statement(binding) for binding in temp_bindings
        )
        return f"{{ {bindings} {constructor} }}"

    def generate_temp_binding_statement(self, binding):
        name, expr, *type_hint = binding
        if type_hint and type_hint[0]:
            return f"let {name}: {type_hint[0]} = {expr};"
        return f"let {name} = {expr};"

    def generate_matrix_constructor_args(self, matrix_info, args):
        generated_args = []
        temp_bindings = []
        expected_count = matrix_info["columns"] * matrix_info["rows"]
        component_type = matrix_info["component_type"]

        if len(args) == 1:
            arg_type = self.expression_result_type(args[0])
            arg_matrix_info = self.matrix_type_info(arg_type)
            if arg_matrix_info is not None:
                return self.generate_matrix_conversion_constructor_args(
                    args[0], arg_matrix_info, matrix_info, component_type
                )

            if self.normalize_scalar_type(arg_type) is not None:
                scalar_expr = self.generate_expression_with_type(
                    args[0], component_type
                )
                scalar_expr = self.normalize_scalar_assignment_value(
                    args[0], scalar_expr, arg_type, component_type
                )
                zero = (
                    self.rust_scalar_default_literal(self.map_type(component_type))
                    or "0"
                )
                for column in range(matrix_info["columns"]):
                    for row in range(matrix_info["rows"]):
                        generated_args.append(scalar_expr if row == column else zero)
                return generated_args, temp_bindings

        for arg in args:
            arg_info = self.vector_type_info(self.expression_result_type(arg))
            if arg_info is None:
                arg_expr = self.generate_expression(arg)
                arg_expr = self.normalize_constructor_scalar_lane(
                    arg,
                    arg_expr,
                    self.expression_result_type(arg),
                    component_type,
                )
                generated_args.append(arg_expr)
            else:
                generated_args.extend(
                    self.vector_argument_lane_expressions(
                        arg, arg_info, temp_bindings, component_type
                    )
                )

            if len(generated_args) >= expected_count:
                return generated_args[:expected_count], temp_bindings

        return generated_args, temp_bindings

    def generate_matrix_conversion_constructor_args(
        self, arg, source_info, target_info, component_type
    ):
        temp_bindings = []
        source_expr = self.generate_expression(arg)
        source_expr = self.generate_matrix_lane_source(arg, source_expr, temp_bindings)
        source_expr = self.rust_member_base_expression(source_expr)
        components = ("x", "y", "z", "w")
        zero = self.rust_scalar_default_literal(self.map_type(component_type)) or "0"
        one = (
            "1.0"
            if self.normalize_scalar_type(component_type) in {"f16", "f32", "f64"}
            else "1"
        )
        generated_args = []
        for column in range(target_info["columns"]):
            for row in range(target_info["rows"]):
                if column < source_info["columns"] and row < source_info["rows"]:
                    generated_args.append(
                        self.normalize_constructor_scalar_lane(
                            None,
                            f"{source_expr}.{GLAM_AXES[column]}.{components[row]}",
                            source_info["component_type"],
                            component_type,
                        )
                    )
                else:
                    generated_args.append(one if column == row else zero)
        return generated_args, temp_bindings

    def generate_vector_constructor_args(self, vector_info, args):
        component_type = vector_info["component_type"]
        if len(args) == 1:
            arg_type = self.expression_result_type(args[0])
            if arg_type is not None and not self.vector_type_info(arg_type):
                temp_bindings = []
                arg_expr = self.generate_expression(args[0])
                arg_expr = self.normalize_constructor_scalar_lane(
                    args[0], arg_expr, arg_type, component_type
                )
                if not self.is_repeat_safe_expression(args[0]):
                    temp_name = self.next_vector_arg_temp_name()
                    temp_bindings.append((temp_name, arg_expr))
                    arg_expr = temp_name
                return [arg_expr] * vector_info["size"], temp_bindings

        generated_args = []
        temp_bindings = []
        for arg in args:
            arg_info = self.vector_type_info(self.expression_result_type(arg))
            if arg_info is None:
                arg_expr = self.generate_expression(arg)
                arg_expr = self.normalize_constructor_scalar_lane(
                    arg,
                    arg_expr,
                    self.expression_result_type(arg),
                    component_type,
                )
                generated_args.append(arg_expr)
            else:
                generated_args.extend(
                    self.vector_argument_lane_expressions(
                        arg, arg_info, temp_bindings, component_type
                    )
                )

            if len(generated_args) >= vector_info["size"]:
                return generated_args[: vector_info["size"]], temp_bindings

        return generated_args, temp_bindings

    def vector_argument_lane_expressions(
        self, arg, arg_info, temp_bindings, target_component_type=None
    ):
        swizzle_components = self.member_swizzle_components(arg)
        if swizzle_components is not None:
            object_expr = getattr(arg, "object_expr", getattr(arg, "object", None))
            object_value = self.generate_vector_lane_source(
                object_expr, self.generate_expression(object_expr), temp_bindings
            )
            object_value = self.rust_member_base_expression(object_value)
            return [
                self.normalize_constructor_scalar_lane(
                    None,
                    f"{object_value}.{component}",
                    arg_info["component_type"],
                    target_component_type or arg_info["component_type"],
                )
                for component in swizzle_components
            ]

        arg_expr = self.generate_expression(arg)
        arg_expr = self.generate_vector_lane_source(arg, arg_expr, temp_bindings)
        arg_expr = self.rust_member_base_expression(arg_expr)
        components = ("x", "y", "z", "w")[: arg_info["size"]]
        return [
            self.normalize_constructor_scalar_lane(
                None,
                f"{arg_expr}.{component}",
                arg_info["component_type"],
                target_component_type or arg_info["component_type"],
            )
            for component in components
        ]

    def normalize_constructor_scalar_lane(
        self, expr, generated_expr, source_type, target_type
    ):
        source_type = self.normalize_scalar_type(source_type)
        target_type = self.normalize_scalar_type(target_type)
        if (
            isinstance(expr, LiteralNode)
            and isinstance(expr.value, float)
            and source_type in {"f32", "f64"}
            and target_type in {"f32", "f64"}
        ):
            return generated_expr
        return self.normalize_scalar_assignment_value(
            expr, generated_expr, source_type, target_type
        )

    def matrix_argument_lane_expressions(
        self, arg, matrix_info, temp_bindings, target_component_type=None
    ):
        arg_expr = self.generate_expression(arg)
        arg_expr = self.generate_matrix_lane_source(arg, arg_expr, temp_bindings)
        arg_expr = self.rust_member_base_expression(arg_expr)
        target_component_type = target_component_type or matrix_info["component_type"]
        components = ("x", "y", "z", "w")[: matrix_info["rows"]]
        lanes = []
        for column in range(matrix_info["columns"]):
            for component in components:
                lanes.append(
                    self.normalize_constructor_scalar_lane(
                        None,
                        f"{arg_expr}.{GLAM_AXES[column]}.{component}",
                        matrix_info["component_type"],
                        target_component_type,
                    )
                )
        return lanes

    def generate_matrix_lane_source(self, expr, generated_expr, temp_bindings):
        if self.is_repeat_safe_expression(expr):
            return generated_expr
        temp_name = self.next_matrix_arg_temp_name()
        temp_bindings.append((temp_name, generated_expr))
        return temp_name

    def generate_vector_lane_source(self, expr, generated_expr, temp_bindings):
        if self.is_repeat_safe_expression(expr):
            return generated_expr
        temp_name = self.next_vector_arg_temp_name()
        type_hint = self.rust_temp_binding_type_hint(expr)
        if type_hint is None:
            temp_bindings.append((temp_name, generated_expr))
        else:
            temp_bindings.append((temp_name, generated_expr, type_hint))
        return temp_name

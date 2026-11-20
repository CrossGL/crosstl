"""Typed return and argument expressions for the Rust generator, including moves out of places and lambda captures."""

from ..ast import (
    ArrayAccessNode,
    ArrayLiteralNode,
    BinaryOpNode,
    BlockNode,
    ConstructorNode,
    FunctionCallNode,
    IdentifierNode,
    LambdaNode,
    MatchNode,
    MemberAccessNode,
    TernaryOpNode,
    VariableNode,
)


class RustReturnValueMixin:
    """Typed return-value members of ``RustCodeGen``."""

    def generate_expression_with_type(self, expr, target_type, static_context=False):
        if self.reference_type_parts_for_type(target_type) is not None:
            if isinstance(expr, TernaryOpNode):
                return self.generate_ternary_expression(expr, target_type)
            if isinstance(expr, MatchNode):
                return self.generate_match_expression(expr, target_type=target_type)
            if isinstance(expr, BlockNode):
                return self.generate_block_expression(expr, target_type=target_type)
            generated = self.generate_borrow_operand_expression(expr)
            return self.normalize_reference_typed_expression(
                expr,
                generated,
                target_type,
            )
        if isinstance(expr, LambdaNode):
            return self.generate_lambda_node_expression(expr, target_type=target_type)
        if isinstance(expr, BlockNode):
            return self.generate_block_expression(expr, target_type=target_type)
        if isinstance(expr, MatchNode):
            return self.generate_match_expression(expr, target_type=target_type)
        if isinstance(expr, ArrayLiteralNode):
            return self.generate_array_literal_expression(
                expr, target_type, static_context=static_context
            )
        if isinstance(expr, ConstructorNode):
            return self.generate_constructor_expression(expr, target_type)
        if isinstance(expr, BinaryOpNode):
            return self.generate_binary_expression(expr, target_type)
        if isinstance(expr, TernaryOpNode):
            return self.generate_ternary_expression(expr, target_type)
        if isinstance(expr, FunctionCallNode):
            function_call = self.generate_function_call_with_target(expr, target_type)
            if function_call is not None:
                return function_call
        return self.generate_expression(expr)

    def generate_return_expression_with_type(self, expr, target_type):
        direct_move = self.generate_direct_return_move_expression(expr)
        if direct_move is not None:
            return direct_move
        member_move = self.generate_return_member_access_expression(expr)
        if member_move is not None:
            return member_move
        if isinstance(expr, TernaryOpNode):
            return self.generate_return_ternary_expression(expr, target_type)
        if isinstance(expr, MatchNode):
            return self.generate_match_expression(
                expr,
                target_type=target_type,
                return_context=True,
            )
        if isinstance(expr, LambdaNode):
            return self.generate_lambda_node_expression(expr, target_type=target_type)
        if isinstance(expr, BlockNode):
            return self.generate_block_expression(
                expr,
                target_type=target_type,
                return_context=True,
            )
        if isinstance(expr, ConstructorNode):
            return self.generate_return_constructor_expression(expr, target_type)
        if isinstance(expr, FunctionCallNode):
            function_call = self.generate_return_function_call_expression(
                expr,
                target_type,
            )
            if function_call is not None:
                return function_call
        return self.generate_expression_with_type(expr, target_type)

    def generate_return_branch_expression_with_type(self, expr, target_type):
        direct_move = self.generate_direct_return_move_expression(expr)
        if direct_move is not None:
            return direct_move
        member_move = self.generate_return_member_access_expression(expr)
        if member_move is not None:
            return member_move
        if isinstance(expr, TernaryOpNode):
            return self.generate_return_ternary_expression(expr, target_type)
        if isinstance(expr, MatchNode):
            return self.generate_match_expression(
                expr,
                target_type=target_type,
                return_context=True,
            )
        if isinstance(expr, LambdaNode):
            return self.generate_lambda_node_expression(expr, target_type=target_type)
        if isinstance(expr, BlockNode):
            return self.generate_block_expression(
                expr,
                target_type=target_type,
                return_context=True,
            )
        if isinstance(expr, ConstructorNode):
            return self.generate_return_constructor_expression(expr, target_type)
        if isinstance(expr, FunctionCallNode):
            function_call = self.generate_return_function_call_expression(
                expr,
                target_type,
            )
            if function_call is not None:
                return function_call
        return self.generate_expression_with_type(expr, target_type)

    def generate_return_ternary_expression(self, expr, target_type):
        condition_expr = getattr(expr, "condition", "")
        true_expr = getattr(expr, "true_expr", "")
        false_expr = getattr(expr, "false_expr", "")

        bool_vector_ternary = self.generate_bool_vector_ternary_expression(
            condition_expr,
            true_expr,
            false_expr,
            target_type,
        )
        if bool_vector_ternary is not None:
            return bool_vector_ternary

        condition = self.generate_condition_expression(condition_expr)
        branch_type = target_type or self.expression_result_type(expr)
        true_value = self.generate_return_branch_expression_with_type(
            true_expr,
            branch_type,
        )
        false_value = self.generate_return_branch_expression_with_type(
            false_expr,
            branch_type,
        )
        true_value = self.normalize_typed_expression_value(
            true_expr,
            true_value,
            branch_type,
        )
        false_value = self.normalize_typed_expression_value(
            false_expr,
            false_value,
            branch_type,
        )
        return f"(if {condition} {{ {true_value} }} else {{ {false_value} }})"

    def generate_return_function_call_expression(
        self,
        expr,
        target_type=None,
        move_counts=None,
    ):
        func_expr = getattr(expr, "function", getattr(expr, "name", "unknown"))
        func_name = self.function_call_name(func_expr)
        if not isinstance(func_name, str):
            return None

        args = getattr(expr, "arguments", getattr(expr, "args", [])) or []
        callee = func_name

        enum_variant = self.generate_enum_variant_call_with_typed_args(
            func_name,
            args,
            target_type=target_type,
            return_context=True,
            move_counts=move_counts,
        )
        if enum_variant is not None:
            return enum_variant

        if self.is_user_defined_function(func_name):
            param_types = self.user_function_param_types.get(func_name, [])
            generated_args = self.generate_return_call_args_with_types(
                args,
                param_types,
                func_name=func_name,
                move_counts=move_counts,
            )
            self.apply_rust_output_arguments(func_name, args, generated_args)
            generated_args.extend(self.generate_captured_function_call_args(func_name))
            return f"{self.rust_callable_name(callee)}({', '.join(generated_args)})"

        struct_constructor = self.generate_return_struct_new_call_with_typed_args(
            func_name,
            args,
            move_counts=move_counts,
        )
        if struct_constructor is not None:
            return struct_constructor

        return None

    def generate_return_constructor_expression(
        self,
        expr,
        target_type=None,
        move_counts=None,
    ):
        type_name = self.normalize_turbofish_type_name(
            self.convert_type_node_to_string(expr.constructor_type)
        )
        rust_type = self.map_type(type_name)

        named_arguments = getattr(expr, "named_arguments", {}) or {}
        if named_arguments:
            variant_info = self.enum_variant_struct_constructor_info(
                type_name,
                target_type,
            )
            if variant_info is not None:
                enum_type, variant_name, field_types = variant_info
                constructor_path = self.enum_variant_call_path(enum_type, variant_name)
                fields = self.generate_return_named_constructor_fields(
                    named_arguments,
                    lambda name: field_types.get(name),
                    move_counts=move_counts,
                )
                return f"{constructor_path} {{ {', '.join(fields)} }}"

            constructor_path = self.rust_constructor_path(rust_type)
            fields = self.generate_return_named_constructor_fields(
                named_arguments,
                lambda name: self.resolve_struct_member_type(type_name, name),
                move_counts=move_counts,
            )
            return f"{constructor_path} {{ {', '.join(fields)} }}"

        arguments = getattr(expr, "arguments", []) or []
        member_types = self.resolve_struct_positional_member_types(type_name)
        args = self.generate_return_call_args_with_types(
            arguments,
            member_types,
            move_counts=move_counts,
        )
        return f"{self.rust_constructor_path(rust_type)}::new({', '.join(args)})"

    def generate_return_struct_new_call_with_typed_args(
        self,
        func_name,
        args,
        move_counts=None,
    ):
        type_name = self.struct_constructor_call_type_name(func_name)
        if type_name is None:
            return None

        member_types = self.resolve_struct_positional_member_types(type_name)
        if not member_types:
            return None

        generated_args = self.generate_return_call_args_with_types(
            args,
            member_types,
            move_counts=move_counts,
        )
        return f"{self.struct_new_call_path(type_name)}({', '.join(generated_args)})"

    def generate_return_named_constructor_fields(
        self,
        named_arguments,
        field_type_for_name,
        move_counts=None,
    ):
        argument_values = list(named_arguments.values())
        local_move_counts = (
            move_counts
            if move_counts is not None
            else self.return_move_place_counts(argument_values)
        )
        later_roots = self.return_call_arg_later_roots(argument_values)
        lambda_conflict_roots = self.return_call_arg_lambda_conflict_roots(
            argument_values
        )
        fields = []
        for index, (name, value) in enumerate(named_arguments.items()):
            field_type = field_type_for_name(name)
            field_value = self.generate_with_lambda_capture_move_blocks(
                value,
                later_roots[index],
                lambda value=value, field_type=field_type: (
                    self.generate_return_argument_with_type(
                        value,
                        field_type,
                        local_move_counts,
                    )
                ),
                lambda_conflict_roots[index],
            )
            fields.append(f"{self.rust_identifier(name)}: {field_value}")
        return fields

    def generate_enum_variant_call_with_typed_args(
        self,
        func_name,
        args,
        target_type=None,
        return_context=False,
        move_counts=None,
    ):
        variant_info = self.enum_variant_call_info(func_name, target_type)
        if variant_info is None:
            return None

        enum_type, variant_name, payload_types = variant_info
        if len(args) == 0 and not payload_types:
            return self.enum_variant_call_path(enum_type, variant_name)
        if len(args) > 0 and not payload_types:
            return None

        if return_context:
            generated_args = self.generate_return_call_args_with_types(
                args,
                payload_types,
                move_counts=move_counts,
            )
        else:
            generated_args = self.generate_constructor_call_args_with_types(
                args,
                payload_types,
            )

        return (
            f"{self.enum_variant_call_path(enum_type, variant_name)}"
            f"({', '.join(generated_args)})"
        )

    def generate_return_call_args_with_types(
        self,
        args,
        target_types,
        func_name=None,
        move_counts=None,
    ):
        if move_counts is None:
            move_counts = self.return_move_place_counts(args)
        later_roots = self.return_call_arg_later_roots(args)
        lambda_conflict_roots = self.return_call_arg_lambda_conflict_roots(args)
        generated_args = []
        for index, arg in enumerate(args):
            target_type = target_types[index] if index < len(target_types) else None
            if func_name is not None:
                target_type = self.user_function_argument_target_type(
                    func_name,
                    arg,
                    target_type,
                )
            generated_args.append(
                self.generate_with_lambda_capture_move_blocks(
                    arg,
                    later_roots[index],
                    lambda: self.generate_return_argument_with_type(
                        arg,
                        target_type,
                        move_counts,
                    ),
                    lambda_conflict_roots[index],
                )
            )
        return generated_args

    def generate_return_argument_with_type(self, arg, target_type, move_counts):
        move_key = self.return_move_place_key(arg)
        if move_key is not None and move_key in move_counts:
            move_counts[move_key] -= 1
            if move_counts[move_key] == 0:
                direct_move = self.generate_direct_return_move_expression(arg)
                if direct_move is not None:
                    return direct_move
                member_move = self.generate_return_member_access_expression(arg)
                if member_move is not None:
                    return self.normalize_typed_expression_value(
                        arg,
                        member_move,
                        target_type,
                    )

        if isinstance(arg, FunctionCallNode):
            function_call = self.generate_return_function_call_expression(
                arg,
                target_type,
                move_counts=move_counts,
            )
            if function_call is not None:
                return self.normalize_typed_expression_value(
                    arg,
                    function_call,
                    target_type,
                )

        if isinstance(arg, ConstructorNode):
            constructor = self.generate_return_constructor_expression(
                arg,
                target_type,
                move_counts=move_counts,
            )
            return self.normalize_typed_expression_value(
                arg,
                constructor,
                target_type,
            )

        if isinstance(arg, BlockNode):
            arg_expr = self.generate_return_branch_expression_with_type(
                arg,
                target_type,
            )
        else:
            arg_expr = self.generate_expression_with_type(arg, target_type)
        return self.normalize_typed_expression_value(arg, arg_expr, target_type)

    def return_move_place_counts(self, args):
        records = []
        for index, arg in enumerate(args):
            records.extend(self.return_move_place_records(arg, (index,)))

        if not records:
            return {}

        eligible_keys = set()
        roots = {record["root"] for record in records}
        for root in roots:
            root_records = [record for record in records if record["root"] == root]
            blocking_records = [
                record
                for record in root_records
                if record["kind"] in {"indexed", "lambda_capture"}
            ]
            movable_records = [
                record
                for record in root_records
                if record["kind"] in {"identifier", "member"}
            ]
            if blocking_records and movable_records:
                continue
            root_records = movable_records
            if not root_records:
                continue
            direct_records = [
                record for record in root_records if record["kind"] == "identifier"
            ]
            member_records = [
                record for record in root_records if record["kind"] == "member"
            ]

            if direct_records and member_records:
                last_direct_index = max(record["index"] for record in direct_records)
                last_member_index = max(record["index"] for record in member_records)
                if last_direct_index > last_member_index:
                    eligible_keys.update(record["key"] for record in direct_records)
                else:
                    member_keys = {record["key"] for record in member_records}
                    for member_key in member_keys:
                        last_key_index = max(
                            record["index"]
                            for record in member_records
                            if record["key"] == member_key
                        )
                        if last_key_index > last_direct_index:
                            eligible_keys.add(member_key)
            else:
                eligible_keys.update(record["key"] for record in root_records)

        counts = {}
        for record in records:
            move_key = record["key"]
            if move_key not in eligible_keys:
                continue
            counts[move_key] = counts.get(move_key, 0) + 1
        return counts

    def return_call_arg_later_roots(self, args):
        records_by_index = []
        for index, arg in enumerate(args):
            records_by_index.append(self.return_move_place_records(arg, (index,)))

        later_roots_by_index = []
        later_roots = set()
        for records in reversed(records_by_index):
            later_roots_by_index.append(set(later_roots))
            later_roots.update(
                record["root"] for record in records if record["root"] is not None
            )

        return list(reversed(later_roots_by_index))

    def return_call_arg_lambda_conflict_roots(self, args):
        capture_roots_by_index = []
        capture_root_counts = {}
        for index, arg in enumerate(args):
            capture_roots = {
                record["root"]
                for record in self.return_move_place_records(arg, (index,))
                if record["kind"] == "lambda_capture" and record["root"] is not None
            }
            capture_roots_by_index.append(capture_roots)
            for root in capture_roots:
                capture_root_counts[root] = capture_root_counts.get(root, 0) + 1

        return [
            {root for root in capture_roots if capture_root_counts[root] > 1}
            for capture_roots in capture_roots_by_index
        ]

    def generate_with_lambda_capture_move_blocks(
        self,
        arg,
        later_roots,
        generate_arg,
        lambda_conflict_roots=None,
    ):
        capture_roots = self.return_argument_lambda_capture_root_names(arg)
        candidate_roots = capture_roots & (lambda_conflict_roots or set())
        if self.loop_depth > 0 and isinstance(arg, LambdaNode):
            candidate_roots |= capture_roots
        if self.nested_lambda_capture_preclone_depth > 0:
            candidate_roots |= capture_roots & self.return_move_blocked_roots
        preclone_places = self.lambda_preclone_capture_roots(
            arg,
            candidate_roots,
        )
        if preclone_places:
            return self.generate_with_precloned_lambda_captures(
                arg,
                preclone_places,
                generate_arg,
            )

        blocked_roots = capture_roots & later_roots
        if lambda_conflict_roots is not None:
            blocked_roots |= capture_roots & lambda_conflict_roots
        if not blocked_roots:
            return generate_arg()

        saved_blocked_roots = self.return_move_blocked_roots
        self.return_move_blocked_roots = saved_blocked_roots | blocked_roots
        enable_nested_preclone = not isinstance(arg, LambdaNode)
        if enable_nested_preclone:
            self.nested_lambda_capture_preclone_depth += 1
        try:
            return generate_arg()
        finally:
            if enable_nested_preclone:
                self.nested_lambda_capture_preclone_depth -= 1
            self.return_move_blocked_roots = saved_blocked_roots

    def lambda_preclone_capture_roots(self, arg, candidate_roots):
        if not isinstance(arg, LambdaNode) or not candidate_roots:
            return set()

        saved_blocked_roots = self.return_move_blocked_roots
        self.return_move_blocked_roots = saved_blocked_roots - set(candidate_roots)
        try:
            body_records = self.return_move_place_records(
                getattr(arg, "body", None),
                (),
            )
        finally:
            self.return_move_blocked_roots = saved_blocked_roots

        preclone_places = []
        for root in sorted(candidate_roots):
            if self.is_static_reference(root):
                continue
            root_type = self.variable_types.get(root)
            if root_type is None or self.type_is_copy_derivable(
                root_type,
                self.current_generic_param_names,
            ):
                continue

            root_records = [
                record
                for record in body_records
                if record["root"] == root and record["kind"] in {"identifier", "member"}
            ]
            if len(root_records) == 1:
                record = root_records[0]
                key = record["key"]
                if record["kind"] == "member":
                    key = key[1]
                place_type = self.return_place_key_type(key)
                if place_type is None or self.type_is_copy_derivable(
                    place_type,
                    self.current_generic_param_names,
                ):
                    continue
                preclone_places.append(
                    {
                        "key": key,
                        "root": root,
                        "type": place_type,
                    }
                )
        return preclone_places

    def generate_with_precloned_lambda_captures(
        self,
        arg,
        preclone_places,
        generate_arg,
    ):
        saved_variable_types = self.variable_types.copy()
        saved_local_variable_names = self.local_variable_names.copy()
        aliases = {}
        place_aliases = {}
        bindings = []
        for place in preclone_places:
            key = place["key"]
            place_type = place["type"]
            place_value = self.return_place_key_expression(key)
            if place_type is None or place_value is None:
                continue
            alias = self.next_lambda_capture_temp_name(
                self.lambda_capture_place_label(key)
            )
            if key[0] == "identifier":
                aliases[place["root"]] = alias
            else:
                place_aliases[key] = alias
            self.register_variable_type(alias, place_type, scope="local")
            bindings.append(f"let {alias} = {self.clone_value_expression(place_value)}")

        if not aliases and not place_aliases:
            self.variable_types = saved_variable_types
            self.local_variable_names = saved_local_variable_names
            return generate_arg()

        self.lambda_capture_alias_stack.append(aliases)
        self.lambda_capture_place_alias_stack.append(place_aliases)
        self.force_move_lambda_depth += 1
        try:
            closure = generate_arg()
        finally:
            self.force_move_lambda_depth -= 1
            self.lambda_capture_place_alias_stack.pop()
            self.lambda_capture_alias_stack.pop()
            self.variable_types = saved_variable_types
            self.local_variable_names = saved_local_variable_names

        return f"{{ {'; '.join(bindings)}; {closure} }}"

    def lambda_capture_place_label(self, key):
        if key[0] == "identifier":
            return key[1]
        if key[0] == "member":
            return f"{self.lambda_capture_place_label(key[1])}_{key[2]}"
        return "capture"

    def lambda_capture_place_alias_name(self, key):
        if key is None:
            return None
        for aliases in reversed(self.lambda_capture_place_alias_stack):
            alias = aliases.get(key)
            if alias is not None:
                return alias
        return None

    def return_place_key_type(self, key):
        if key is None:
            return None
        if key[0] == "identifier":
            return self.variable_types.get(key[1])
        if key[0] == "member":
            object_type = self.return_place_key_type(key[1])
            if object_type is None:
                return None
            return self.resolve_struct_member_type(object_type, key[2])
        return None

    def return_place_key_expression(self, key):
        if key is None:
            return None
        if key[0] == "identifier":
            return self.lazy_static_identifier_expression(key[1])
        if key[0] == "member":
            object_expr = self.return_place_key_expression(key[1])
            if object_expr is None:
                return None
            return f"{object_expr}.{key[2]}"
        return None

    def lambda_capture_alias_name(self, name):
        for aliases in reversed(self.lambda_capture_alias_stack):
            alias = aliases.get(name)
            if alias is not None:
                return alias
        return None

    def return_argument_lambda_capture_root_names(self, expr):
        return {
            record["root"]
            for record in self.return_move_place_records(expr, ())
            if record["kind"] == "lambda_capture" and record["root"] is not None
        }

    def lambda_capture_root_names(self, expr):
        if not isinstance(expr, LambdaNode):
            return set()

        roots = set()
        for capture in getattr(expr, "captures", []) or []:
            if isinstance(capture, str):
                name = capture if capture.isidentifier() else None
            else:
                name = self.return_place_root_name(capture)
            if name is not None:
                roots.add(name)
        return roots

    def return_move_place_records(self, expr, index_path):
        if isinstance(expr, LambdaNode):
            return [
                {
                    "index": index_path + (capture_index,),
                    "key": ("lambda_capture", name),
                    "kind": "lambda_capture",
                    "root": name,
                }
                for capture_index, name in enumerate(
                    sorted(self.lambda_capture_root_names(expr))
                )
            ]

        if isinstance(expr, ArrayAccessNode):
            root_name = self.return_place_root_name(expr)
            if root_name is not None and root_name in self.variable_types:
                return [
                    {
                        "index": index_path,
                        "key": ("indexed", root_name),
                        "kind": "indexed",
                        "root": root_name,
                    }
                ]

        move_key = self.return_move_place_key(expr)
        if move_key is not None:
            kind = move_key[0]
            root_name = (
                move_key[1]
                if kind == "identifier"
                else self.return_place_root_name(expr)
            )
            return [
                {
                    "index": index_path,
                    "key": move_key,
                    "kind": kind,
                    "root": root_name,
                }
            ]

        child_args = self.return_consumed_child_arguments(expr)
        records = []
        for child_index, child in enumerate(child_args):
            records.extend(
                self.return_move_place_records(
                    child,
                    index_path + (child_index,),
                )
            )
        return records

    def return_consumed_child_arguments(self, expr):
        if isinstance(expr, ConstructorNode):
            named_arguments = getattr(expr, "named_arguments", {}) or {}
            if named_arguments:
                return list(named_arguments.values())
            return list(getattr(expr, "arguments", []) or [])

        if not isinstance(expr, FunctionCallNode):
            return []

        func_expr = getattr(expr, "function", getattr(expr, "name", "unknown"))
        func_name = self.function_call_name(func_expr)
        if not isinstance(func_name, str):
            return []

        args = list(getattr(expr, "arguments", getattr(expr, "args", [])) or [])
        if self.is_user_defined_function(func_name):
            return args
        if self.enum_variant_call_info(func_name) is not None:
            return args

        type_name = self.struct_constructor_call_type_name(func_name)
        if type_name is not None and self.resolve_struct_positional_member_types(
            type_name
        ):
            return args

        return []

    def return_move_place_key(self, expr):
        name = self.simple_identifier_expression_name(expr)
        if name is not None and self.should_move_direct_return_identifier(name):
            return ("identifier", name)

        if self.should_move_member_return_place(expr):
            place_key = self.return_place_expression_key(expr)
            if place_key is not None:
                return ("member", place_key)

        return None

    def return_place_expression_key(self, expr):
        if isinstance(expr, IdentifierNode):
            return ("identifier", expr.name)
        if isinstance(expr, VariableNode):
            return ("identifier", expr.name)
        if isinstance(expr, str):
            return ("identifier", expr) if expr.isidentifier() else None
        if isinstance(expr, MemberAccessNode):
            object_expr = getattr(expr, "object_expr", getattr(expr, "object", None))
            object_key = self.return_place_expression_key(object_expr)
            if object_key is None:
                return None
            member = getattr(expr, "member", None)
            return ("member", object_key, member)
        return None

    def generate_constructor_call_args_with_types(self, args, target_types):
        later_roots = self.return_call_arg_later_roots(args)
        lambda_conflict_roots = self.return_call_arg_lambda_conflict_roots(args)
        generated_args = []
        for index, arg in enumerate(args):
            target_type = target_types[index] if index < len(target_types) else None
            generated_args.append(
                self.generate_with_lambda_capture_move_blocks(
                    arg,
                    later_roots[index],
                    lambda arg=arg, target_type=target_type: (
                        self.generate_constructor_argument_with_type(
                            arg,
                            target_type,
                        )
                    ),
                    lambda_conflict_roots[index],
                )
            )
        return generated_args

    def generate_constructor_argument_with_type(self, arg, target_type):
        arg_expr = self.generate_expression_with_type(arg, target_type)
        return self.normalize_typed_expression_value(arg, arg_expr, target_type)

    def generate_return_member_access_expression(self, expr):
        place_alias = self.lambda_capture_place_alias_name(
            self.return_place_expression_key(expr)
        )
        if place_alias is not None:
            return self.lazy_static_identifier_expression(place_alias)

        if not self.should_move_member_return_place(expr):
            return None

        swizzle_components = self.member_swizzle_components(expr)
        if swizzle_components is not None:
            return None

        obj_expr = getattr(expr, "object_expr", getattr(expr, "object", ""))
        member = getattr(expr, "member", "")
        self.member_object_depth += 1
        try:
            obj = self.generate_expression(obj_expr)
        finally:
            self.member_object_depth -= 1
        obj = self.lazy_static_object_expression(obj_expr, obj)
        return f"{obj}.{self.rust_identifier(member)}"

    def should_move_member_return_place(self, expr):
        if not isinstance(expr, MemberAccessNode):
            return False

        member_type = self.expression_result_type(expr)
        if member_type is None or self.type_is_copy_derivable(
            member_type,
            self.current_generic_param_names,
        ):
            return False

        object_expr = getattr(expr, "object_expr", getattr(expr, "object", None))
        if self.return_place_contains_index_access(object_expr):
            return False

        root_name = self.return_place_root_name(object_expr)
        if root_name is None or self.is_static_reference(root_name):
            return False
        if root_name in self.return_move_blocked_roots:
            return False

        return root_name in self.variable_types

    def return_place_contains_index_access(self, expr):
        if isinstance(expr, ArrayAccessNode):
            return True
        if isinstance(expr, MemberAccessNode):
            object_expr = getattr(expr, "object_expr", getattr(expr, "object", None))
            return self.return_place_contains_index_access(object_expr)
        return False

    def return_place_root_name(self, expr):
        if isinstance(expr, IdentifierNode):
            return expr.name
        if isinstance(expr, VariableNode):
            return expr.name
        if isinstance(expr, str):
            return expr if expr.isidentifier() else None
        if isinstance(expr, MemberAccessNode):
            object_expr = getattr(expr, "object_expr", getattr(expr, "object", None))
            return self.return_place_root_name(object_expr)
        if isinstance(expr, ArrayAccessNode):
            array_expr = getattr(expr, "array_expr", getattr(expr, "array", None))
            return self.return_place_root_name(array_expr)
        return None

    def generate_direct_return_move_expression(self, expr):
        name = self.simple_identifier_expression_name(expr)
        if name is None:
            return None
        move_name = self.lambda_capture_alias_name(name) or name
        if not self.should_move_direct_return_identifier(move_name):
            return None
        return self.lazy_static_identifier_expression(move_name)

    def should_move_direct_return_identifier(self, name):
        if name in self.return_move_blocked_roots:
            return False
        if self.is_static_reference(name):
            return False
        value_type = self.variable_types.get(name)
        if value_type is None or self.type_is_copy_derivable(
            value_type,
            self.current_generic_param_names,
        ):
            return False
        return True

    def simple_identifier_expression_name(self, expr):
        if isinstance(expr, IdentifierNode):
            return expr.name
        if isinstance(expr, VariableNode):
            return expr.name
        if isinstance(expr, str) and expr.isidentifier():
            return expr
        return None

"""Resource-query return analysis of the CUDA generator: local aliases and their source descriptors."""

from ..ast import (
    ArrayAccessNode,
    AssignmentNode,
    BinaryOpNode,
    ExpressionStatementNode,
    FunctionCallNode,
    IdentifierNode,
    IfNode,
    LiteralNode,
    MemberAccessNode,
    PointerAccessNode,
    ReturnNode,
    TernaryOpNode,
    UnaryOpNode,
    VariableNode,
)


class CudaQueryReturnMixin:
    """Query return members of ``CudaCodeGen``."""

    def collect_function_resource_aliases(self, func):
        """Collect local queryable resource declarations and initializers."""
        local_names = set()
        alias_sources = {}
        body = getattr(func, "body", [])

        for node in self.query_walk_nodes(body):
            if not isinstance(node, VariableNode):
                continue
            name = getattr(node, "name", None)
            var_type = self.get_variable_node_type(node)
            if not name or not self.is_queryable_resource_type(var_type):
                continue
            local_names.add(name)
            initial_value = getattr(node, "initial_value", getattr(node, "value", None))
            if initial_value is not None:
                alias_sources[name] = initial_value

        return local_names, alias_sources

    def collect_function_resource_assignments(self, func, local_resource_names):
        """Collect simple assignments into local queryable resource variables."""
        assignment_sources = {}
        body = getattr(func, "body", [])

        for node in self.query_walk_nodes(body):
            if not isinstance(node, AssignmentNode):
                continue
            if getattr(node, "operator", "=") != "=":
                continue
            target = getattr(node, "target", None)
            if not isinstance(target, (IdentifierNode, VariableNode, str)):
                continue
            target_name = self.get_expression_name(target)
            if target_name not in local_resource_names:
                continue
            value = getattr(node, "value", None)
            if value is not None:
                assignment_sources.setdefault(target_name, []).append(value)

        return assignment_sources

    def is_query_return_alias_type(self, type_name):
        """Return whether a local type can participate in resource-return tracing."""
        if self.is_queryable_resource_type(type_name):
            return True

        type_name = self.type_name_string(type_name)
        if not type_name:
            return False

        if self.struct_member_lookup_type(type_name) in self.struct_member_types:
            return True

        indirect_info = self.cuda_indirect_type_info(type_name)
        if indirect_info is None:
            return False
        pointee_type = self.struct_member_lookup_type(indirect_info["pointee_type"])
        return pointee_type in self.struct_member_types

    def collect_query_return_local_alias_sources(self, statements):
        """Collect simple local aliases before a traceable resource return."""
        local_alias_types = {}
        local_alias_sources = {}
        assigned_names = set()

        for statement in statements:
            if isinstance(statement, VariableNode):
                var_type = self.get_variable_node_type(statement)
                if not self.is_query_return_alias_type(var_type):
                    return None
                name = getattr(statement, "name", None)
                if not name or name in local_alias_types:
                    return None

                local_alias_types[name] = var_type
                initial_value = getattr(
                    statement,
                    "initial_value",
                    getattr(statement, "value", None),
                )
                if initial_value is not None:
                    local_alias_sources[name] = initial_value
                    assigned_names.add(name)
                continue

            if isinstance(statement, IfNode):
                branch_alias = self.query_return_if_alias_source(
                    statement,
                    local_alias_types,
                    assigned_names,
                )
                if branch_alias is None:
                    return None
                target_name, value = branch_alias
                local_alias_sources[target_name] = value
                assigned_names.add(target_name)
                continue

            assignment = self.query_return_simple_assignment(statement)
            if assignment is None:
                return None
            target_name, value = assignment
            if target_name not in local_alias_types or target_name in assigned_names:
                return None

            local_alias_sources[target_name] = value
            assigned_names.add(target_name)

        return local_alias_sources

    def query_return_simple_assignment(self, statement):
        """Return simple assignment target/source for query-return aliases."""
        assignment = statement
        if isinstance(statement, ExpressionStatementNode):
            assignment = getattr(statement, "expression", None)
        if not isinstance(assignment, AssignmentNode):
            return None
        if getattr(assignment, "operator", "=") != "=":
            return None

        target = getattr(assignment, "target", None)
        if not isinstance(target, (IdentifierNode, VariableNode, str)):
            return None
        target_name = self.get_expression_name(target)
        value = getattr(assignment, "value", None)
        if not target_name or value is None:
            return None
        return target_name, value

    def query_return_if_alias_source(
        self,
        statement,
        local_alias_types,
        assigned_names,
    ):
        """Collect a deterministic if/else alias assignment chain."""
        branch_assignments = []
        current_if = statement
        else_assignment = None

        while isinstance(current_if, IfNode):
            condition = getattr(
                current_if,
                "condition",
                getattr(current_if, "if_condition", None),
            )
            then_body = getattr(
                current_if,
                "then_branch",
                getattr(current_if, "if_body", None),
            )
            else_body = getattr(
                current_if,
                "else_branch",
                getattr(current_if, "else_body", None),
            )
            if condition is None or then_body is None or else_body is None:
                return None

            then_assignment = self.query_return_body_alias_assignment(then_body)
            if then_assignment is None:
                return None
            branch_assignments.append((condition, then_assignment))

            else_if_conditions = list(
                getattr(current_if, "else_if_conditions", None) or []
            )
            else_if_bodies = list(getattr(current_if, "else_if_bodies", None) or [])
            if len(else_if_conditions) != len(else_if_bodies):
                return None

            for else_if_condition, else_if_body in zip(
                else_if_conditions,
                else_if_bodies,
            ):
                if else_if_condition is None:
                    return None
                else_if_assignment = self.query_return_body_alias_assignment(
                    else_if_body
                )
                if else_if_assignment is None:
                    return None
                branch_assignments.append((else_if_condition, else_if_assignment))

            if isinstance(else_body, IfNode):
                current_if = else_body
                continue

            else_assignment = self.query_return_body_alias_assignment(else_body)
            if else_assignment is None:
                return None
            break

        if else_assignment is None:
            return None

        target_name = branch_assignments[0][1][0]
        if target_name not in local_alias_types or target_name in assigned_names:
            return None
        if any(
            branch_assignment[0] != target_name
            for _, branch_assignment in branch_assignments
        ):
            return None

        else_name, else_value = else_assignment
        if else_name != target_name:
            return None

        value = else_value
        for branch_condition, (_, branch_value) in reversed(branch_assignments):
            value = TernaryOpNode(branch_condition, branch_value, value)
        return target_name, value

    def query_return_body_alias_assignment(self, body):
        """Return a branch body's only alias assignment, if it is simple."""
        statements = self.statement_list(body)
        if len(statements) != 1:
            return None
        return self.query_return_simple_assignment(statements[0])

    def collect_simple_query_return_sources(self, root):
        """Collect direct resource returns that can reuse caller-side metadata."""
        return_sources = {}
        functions = self.query_collect_functions(root)
        global_resource_names = {
            getattr(var, "name", None)
            for var in getattr(root, "global_variables", [])
            if self.is_queryable_resource_type(self.get_variable_node_type(var))
        }
        global_resource_names = {name for name in global_resource_names if name}
        global_variable_types = {
            getattr(var, "name", None): self.type_name_string(
                self.get_variable_node_type(var)
            )
            for var in getattr(root, "global_variables", [])
        }
        global_variable_types = {
            name: type_name
            for name, type_name in global_variable_types.items()
            if name and type_name
        }

        function_infos = []
        for func in functions:
            func_name = getattr(func, "name", None)
            return_type = getattr(func, "return_type", None)
            if not func_name or not self.is_queryable_resource_type(return_type):
                continue

            statements = self.statement_list(getattr(func, "body", []))
            if not statements or not isinstance(statements[-1], ReturnNode):
                continue

            local_alias_sources = self.collect_query_return_local_alias_sources(
                statements[:-1]
            )
            if local_alias_sources is None:
                continue

            return_expr = getattr(statements[-1], "value", None)
            params = getattr(func, "parameters", getattr(func, "params", []))
            resource_param_indices = {
                getattr(param, "name", None): index
                for index, param in enumerate(params)
                if self.is_queryable_resource_type(self.get_parameter_type(param))
            }
            resource_param_indices = {
                name: index for name, index in resource_param_indices.items() if name
            }
            all_param_indices = {
                getattr(param, "name", None): index
                for index, param in enumerate(params)
            }
            all_param_indices = {
                name: index for name, index in all_param_indices.items() if name
            }
            param_types = {
                getattr(param, "name", None): self.type_name_string(
                    self.get_parameter_type(param)
                )
                for param in params
            }
            param_types = {
                name: type_name
                for name, type_name in param_types.items()
                if name and type_name
            }
            function_infos.append(
                (
                    func_name,
                    return_expr,
                    resource_param_indices,
                    all_param_indices,
                    param_types,
                    local_alias_sources,
                )
            )

        changed = True
        while changed:
            changed = False
            for (
                func_name,
                return_expr,
                resource_param_indices,
                all_param_indices,
                param_types,
                local_alias_sources,
            ) in function_infos:
                if func_name in return_sources:
                    continue
                return_source = self.query_return_source_descriptor(
                    return_expr,
                    resource_param_indices,
                    global_resource_names,
                    all_param_indices,
                    return_sources,
                    local_alias_sources,
                    param_types=param_types,
                    global_variable_types=global_variable_types,
                )
                if return_source is not None:
                    return_sources[func_name] = return_source
                    changed = True

        return return_sources

    def query_return_source_descriptor(
        self,
        return_expr,
        resource_param_indices,
        global_resource_names,
        all_param_indices,
        known_return_sources=None,
        local_alias_sources=None,
        local_visited=None,
        param_types=None,
        global_variable_types=None,
    ):
        """Describe a traceable resource-return expression for caller metadata."""
        if known_return_sources is None:
            known_return_sources = {}
        if local_alias_sources is None:
            local_alias_sources = {}
        if local_visited is None:
            local_visited = set()
        if param_types is None:
            param_types = {}
        if global_variable_types is None:
            global_variable_types = {}

        if isinstance(return_expr, TernaryOpNode):
            true_source = self.query_return_source_descriptor(
                return_expr.true_expr,
                resource_param_indices,
                global_resource_names,
                all_param_indices,
                known_return_sources,
                local_alias_sources,
                local_visited.copy(),
                param_types=param_types,
                global_variable_types=global_variable_types,
            )
            false_source = self.query_return_source_descriptor(
                return_expr.false_expr,
                resource_param_indices,
                global_resource_names,
                all_param_indices,
                known_return_sources,
                local_alias_sources,
                local_visited.copy(),
                param_types=param_types,
                global_variable_types=global_variable_types,
            )
            if true_source is None or false_source is None:
                return None
            return {
                "kind": "ternary",
                "condition": return_expr.condition,
                "true_source": true_source,
                "false_source": false_source,
                "param_indices": all_param_indices,
            }

        if isinstance(return_expr, FunctionCallNode):
            callee_name = self.raw_function_call_name(return_expr)
            return_source = known_return_sources.get(callee_name)
            if return_source is None:
                return None
            raw_args = getattr(
                return_expr,
                "arguments",
                getattr(return_expr, "args", []),
            )
            return self.inline_query_return_source_descriptor(
                return_source,
                raw_args,
                resource_param_indices,
                global_resource_names,
                all_param_indices,
                known_return_sources,
                local_alias_sources,
                param_types,
                global_variable_types,
            )

        return_base_expr = return_expr
        return_indices = []
        if isinstance(return_expr, ArrayAccessNode):
            return_base_expr, return_indices = self.query_array_access_parts(
                return_expr
            )
        elif isinstance(return_expr, (MemberAccessNode, PointerAccessNode)):
            return_base_expr = return_expr
        elif not isinstance(return_expr, (IdentifierNode, VariableNode, str)):
            return None

        if isinstance(return_base_expr, (MemberAccessNode, PointerAccessNode)):
            return self.query_return_member_source_descriptor(
                return_base_expr,
                return_indices,
                all_param_indices,
                param_types,
                global_variable_types,
                local_alias_sources,
                local_visited,
            )

        return_name = self.get_expression_name(return_base_expr)
        if not return_name:
            return None
        if any(
            not self.is_safe_query_return_index(index, all_param_indices)
            for index in return_indices
        ):
            return None
        if return_name in local_alias_sources:
            if return_name in local_visited:
                return None
            source_descriptor = self.query_return_source_descriptor(
                local_alias_sources[return_name],
                resource_param_indices,
                global_resource_names,
                all_param_indices,
                known_return_sources,
                local_alias_sources,
                local_visited | {return_name},
                param_types=param_types,
                global_variable_types=global_variable_types,
            )
            if source_descriptor is None:
                return None
            return self.append_query_return_indices(
                source_descriptor,
                return_indices,
                all_param_indices,
            )
        if return_name in resource_param_indices:
            return {
                "kind": "parameter",
                "index": resource_param_indices[return_name],
                "indices": return_indices,
                "param_indices": all_param_indices,
            }
        if return_name in global_resource_names:
            return {
                "kind": "global",
                "name": return_name,
                "indices": return_indices,
                "param_indices": all_param_indices,
            }
        return None

    def substitute_query_return_local_aliases(
        self,
        expr,
        local_alias_sources,
        local_visited=None,
    ):
        """Replace helper-local aliases in a return-source expression."""
        if local_visited is None:
            local_visited = set()

        expr_name = self.get_expression_name(expr)
        if isinstance(expr, (IdentifierNode, VariableNode, str)):
            if expr_name in local_alias_sources:
                if expr_name in local_visited:
                    return None
                return self.substitute_query_return_local_aliases(
                    local_alias_sources[expr_name],
                    local_alias_sources,
                    local_visited | {expr_name},
                )
            return expr

        if isinstance(expr, BinaryOpNode):
            left = self.substitute_query_return_local_aliases(
                expr.left,
                local_alias_sources,
                local_visited,
            )
            right = self.substitute_query_return_local_aliases(
                expr.right,
                local_alias_sources,
                local_visited,
            )
            if left is None or right is None:
                return None
            return BinaryOpNode(left, expr.operator, right)

        if isinstance(expr, UnaryOpNode):
            operand = self.substitute_query_return_local_aliases(
                expr.operand,
                local_alias_sources,
                local_visited,
            )
            if operand is None:
                return None
            return UnaryOpNode(expr.operator, operand, expr.is_postfix)

        if isinstance(expr, TernaryOpNode):
            condition = self.substitute_query_return_local_aliases(
                expr.condition,
                local_alias_sources,
                local_visited,
            )
            true_expr = self.substitute_query_return_local_aliases(
                expr.true_expr,
                local_alias_sources,
                local_visited,
            )
            false_expr = self.substitute_query_return_local_aliases(
                expr.false_expr,
                local_alias_sources,
                local_visited,
            )
            if condition is None or true_expr is None or false_expr is None:
                return None
            return TernaryOpNode(condition, true_expr, false_expr)

        if isinstance(expr, ArrayAccessNode):
            array_expr = self.substitute_query_return_local_aliases(
                getattr(expr, "array_expr", getattr(expr, "array", None)),
                local_alias_sources,
                local_visited,
            )
            index_expr = self.substitute_query_return_local_aliases(
                getattr(expr, "index_expr", getattr(expr, "index", None)),
                local_alias_sources,
                local_visited,
            )
            if array_expr is None or index_expr is None:
                return None
            return ArrayAccessNode(array_expr, index_expr)

        if isinstance(expr, MemberAccessNode):
            object_expr = self.substitute_query_return_local_aliases(
                getattr(expr, "object_expr", getattr(expr, "object", None)),
                local_alias_sources,
                local_visited,
            )
            if object_expr is None:
                return None
            return MemberAccessNode(object_expr, expr.member)

        if isinstance(expr, PointerAccessNode):
            pointer_expr = self.substitute_query_return_local_aliases(
                expr.pointer_expr,
                local_alias_sources,
                local_visited,
            )
            if pointer_expr is None:
                return None
            return PointerAccessNode(pointer_expr, expr.member)

        if isinstance(expr, FunctionCallNode):
            arguments = []
            for argument in getattr(expr, "arguments", getattr(expr, "args", [])):
                substituted = self.substitute_query_return_local_aliases(
                    argument,
                    local_alias_sources,
                    local_visited,
                )
                if substituted is None:
                    return None
                arguments.append(substituted)
            return FunctionCallNode(
                expr.function,
                arguments,
                getattr(expr, "generic_args", []),
            )

        return expr

    def query_return_member_source_descriptor(
        self,
        member_expr,
        return_indices,
        all_param_indices,
        param_types,
        global_variable_types,
        local_alias_sources=None,
        local_visited=None,
    ):
        """Describe a storage-image struct member returned by a helper."""
        if local_alias_sources is None:
            local_alias_sources = {}
        if local_visited is None:
            local_visited = set()

        if isinstance(member_expr, PointerAccessNode):
            object_node = getattr(member_expr, "pointer_expr", None)
        else:
            object_node = getattr(
                member_expr,
                "object_expr",
                getattr(member_expr, "object", None),
            )
        member = getattr(member_expr, "member", None)
        member_name = getattr(member, "name", member)
        if object_node is None or not isinstance(member_name, str):
            return None

        object_name = self.get_expression_name(object_node)

        object_kind = None
        object_type = None
        descriptor = {
            "kind": "member",
            "member": member_name,
            "indices": return_indices,
            "param_indices": all_param_indices,
            "object_access": (
                "->" if isinstance(member_expr, PointerAccessNode) else "."
            ),
        }
        if object_name in local_alias_sources:
            if object_name in local_visited:
                return None
            object_kind = "expression"
            object_node = local_alias_sources[object_name]
            object_type = self.query_return_expression_type(
                object_node,
                param_types,
                global_variable_types,
                local_alias_sources,
                local_visited | {object_name},
            )
        elif object_name in param_types:
            object_kind = "parameter"
            object_type = param_types[object_name]
            descriptor["object_index"] = all_param_indices.get(object_name)
        elif object_name in global_variable_types:
            object_kind = "global"
            object_type = global_variable_types[object_name]
            descriptor["object_name"] = object_name
        else:
            object_kind = "expression"
            object_type = self.query_return_expression_type(
                object_node,
                param_types,
                global_variable_types,
                local_alias_sources,
                local_visited,
            )
        if isinstance(member_expr, PointerAccessNode):
            pointer_info = self.cuda_indirect_type_info(object_type)
            if pointer_info is None:
                return None
            object_type = pointer_info["pointee_type"]

        object_type = self.struct_member_lookup_type(object_type)
        member_type = self.struct_member_types.get(object_type, {}).get(member_name)
        if not self.is_storage_image_type(member_type):
            return None
        if any(
            not self.is_safe_query_return_index(index, all_param_indices)
            for index in return_indices
        ):
            return None

        descriptor["object_kind"] = object_kind
        descriptor["object_type"] = object_type
        object_expr = self.substitute_query_return_local_aliases(
            object_node,
            local_alias_sources,
        )
        if object_expr is not None:
            descriptor["object_expr"] = object_expr
        return descriptor

    def query_return_expression_type(
        self,
        expr,
        param_types,
        global_variable_types,
        local_alias_sources=None,
        local_visited=None,
    ):
        """Infer expression types for returned-resource tracing without scope."""
        if local_alias_sources is None:
            local_alias_sources = {}
        if local_visited is None:
            local_visited = set()

        expr_name = self.get_expression_name(expr)
        if expr_name in local_alias_sources:
            if expr_name in local_visited:
                return None
            return self.query_return_expression_type(
                local_alias_sources[expr_name],
                param_types,
                global_variable_types,
                local_alias_sources,
                local_visited | {expr_name},
            )
        if expr_name in param_types:
            return param_types[expr_name]
        if expr_name in global_variable_types:
            return global_variable_types[expr_name]

        if isinstance(expr, TernaryOpNode):
            true_type = self.query_return_expression_type(
                expr.true_expr,
                param_types,
                global_variable_types,
                local_alias_sources,
                local_visited.copy(),
            )
            false_type = self.query_return_expression_type(
                expr.false_expr,
                param_types,
                global_variable_types,
                local_alias_sources,
                local_visited.copy(),
            )
            if self.type_name_string(true_type) == self.type_name_string(false_type):
                return true_type
            return None

        if isinstance(expr, PointerAccessNode):
            pointer_expr = getattr(expr, "pointer_expr", None)
            pointer_type = self.query_return_expression_type(
                pointer_expr,
                param_types,
                global_variable_types,
                local_alias_sources,
                local_visited,
            )
            pointer_info = self.cuda_indirect_type_info(pointer_type)
            if pointer_info is None:
                return None
            object_type = self.struct_member_lookup_type(pointer_info["pointee_type"])
            return self.struct_member_types.get(object_type, {}).get(
                getattr(expr, "member", "")
            )

        if isinstance(expr, MemberAccessNode):
            object_node = getattr(
                expr,
                "object_expr",
                getattr(expr, "object", None),
            )
            object_type = self.struct_member_lookup_type(
                self.query_return_expression_type(
                    object_node,
                    param_types,
                    global_variable_types,
                    local_alias_sources,
                    local_visited,
                )
            )
            return self.struct_member_types.get(object_type, {}).get(
                getattr(expr, "member", "")
            )

        if isinstance(expr, ArrayAccessNode):
            array_expr = getattr(expr, "array_expr", getattr(expr, "array", None))
            array_type = self.query_return_expression_type(
                array_expr,
                param_types,
                global_variable_types,
                local_alias_sources,
                local_visited,
            )
            return self.array_access_element_type(array_type)

        return self.expression_result_type(expr)

    def inline_query_return_source_descriptor(
        self,
        return_source,
        raw_args,
        resource_param_indices,
        global_resource_names,
        all_param_indices,
        known_return_sources,
        local_alias_sources=None,
        param_types=None,
        global_variable_types=None,
    ):
        """Inline a traceable callee resource return into the caller context."""
        if local_alias_sources is None:
            local_alias_sources = {}
        if param_types is None:
            param_types = {}
        if global_variable_types is None:
            global_variable_types = {}
        source_param_indices = return_source.get("param_indices", {})
        kind = return_source["kind"]

        if kind == "global":
            indices = self.substitute_query_return_indices(
                return_source.get("indices", []),
                source_param_indices,
                raw_args,
                all_param_indices,
            )
            if indices is None:
                return None
            return {
                "kind": "global",
                "name": return_source["name"],
                "indices": indices,
                "param_indices": all_param_indices,
            }

        if kind == "parameter":
            index = return_source["index"]
            if index >= len(raw_args):
                return None
            base_source = self.query_return_source_descriptor(
                raw_args[index],
                resource_param_indices,
                global_resource_names,
                all_param_indices,
                known_return_sources,
                local_alias_sources,
                param_types=param_types,
                global_variable_types=global_variable_types,
            )
            if base_source is None:
                return None
            indices = self.substitute_query_return_indices(
                return_source.get("indices", []),
                source_param_indices,
                raw_args,
                all_param_indices,
            )
            if indices is None:
                return None
            return self.append_query_return_indices(
                base_source,
                indices,
                all_param_indices,
            )

        if kind == "ternary":
            condition = self.substitute_query_return_expr(
                return_source["condition"],
                source_param_indices,
                raw_args,
            )
            if condition is None:
                return None
            true_source = self.inline_query_return_source_descriptor(
                return_source["true_source"],
                raw_args,
                resource_param_indices,
                global_resource_names,
                all_param_indices,
                known_return_sources,
                local_alias_sources,
                param_types,
                global_variable_types,
            )
            false_source = self.inline_query_return_source_descriptor(
                return_source["false_source"],
                raw_args,
                resource_param_indices,
                global_resource_names,
                all_param_indices,
                known_return_sources,
                local_alias_sources,
                param_types,
                global_variable_types,
            )
            if true_source is None or false_source is None:
                return None
            return {
                "kind": "ternary",
                "condition": condition,
                "true_source": true_source,
                "false_source": false_source,
                "param_indices": all_param_indices,
            }

        if kind == "member":
            indices = self.substitute_query_return_indices(
                return_source.get("indices", []),
                source_param_indices,
                raw_args,
                all_param_indices,
            )
            if indices is None:
                return None
            inlined = dict(return_source)
            inlined["indices"] = indices
            inlined["param_indices"] = all_param_indices
            return inlined

        return None

    def append_query_return_indices(self, return_source, indices, param_indices):
        """Append array indices to a flattened resource-return descriptor."""
        if not indices:
            return return_source
        if return_source["kind"] not in {"global", "parameter", "member"}:
            return None
        if any(
            not self.is_safe_query_return_index(index, param_indices)
            for index in indices
        ):
            return None
        combined = dict(return_source)
        combined["indices"] = list(return_source.get("indices", [])) + list(indices)
        combined["param_indices"] = param_indices
        return combined

    def substitute_query_return_indices(
        self,
        indices,
        param_indices,
        raw_args,
        target_param_indices,
    ):
        """Substitute callee index parameters with caller-context expressions."""
        substituted_indices = []
        for index_expr in indices:
            substituted = self.substitute_query_return_expr(
                index_expr,
                param_indices,
                raw_args,
            )
            if substituted is None or not self.is_safe_query_return_index(
                substituted,
                target_param_indices,
            ):
                return None
            substituted_indices.append(substituted)
        return substituted_indices

    def substitute_query_return_expr(self, expr, param_indices, raw_args):
        """Replace callee parameters in a return-source expression."""
        expr_name = self.get_expression_name(expr)
        if isinstance(expr, (IdentifierNode, VariableNode, str)):
            if expr_name in param_indices:
                param_index = param_indices[expr_name]
                if param_index >= len(raw_args):
                    return None
                return raw_args[param_index]
            return expr

        if isinstance(expr, BinaryOpNode):
            left = self.substitute_query_return_expr(
                expr.left,
                param_indices,
                raw_args,
            )
            right = self.substitute_query_return_expr(
                expr.right,
                param_indices,
                raw_args,
            )
            if left is None or right is None:
                return None
            return BinaryOpNode(left, expr.operator, right)

        if isinstance(expr, UnaryOpNode):
            operand = self.substitute_query_return_expr(
                expr.operand,
                param_indices,
                raw_args,
            )
            if operand is None:
                return None
            return UnaryOpNode(expr.operator, operand, expr.is_postfix)

        if isinstance(expr, TernaryOpNode):
            condition = self.substitute_query_return_expr(
                expr.condition,
                param_indices,
                raw_args,
            )
            true_expr = self.substitute_query_return_expr(
                expr.true_expr,
                param_indices,
                raw_args,
            )
            false_expr = self.substitute_query_return_expr(
                expr.false_expr,
                param_indices,
                raw_args,
            )
            if condition is None or true_expr is None or false_expr is None:
                return None
            return TernaryOpNode(condition, true_expr, false_expr)

        if isinstance(expr, ArrayAccessNode):
            array_expr = self.substitute_query_return_expr(
                expr.array_expr,
                param_indices,
                raw_args,
            )
            index_expr = self.substitute_query_return_expr(
                expr.index_expr,
                param_indices,
                raw_args,
            )
            if array_expr is None or index_expr is None:
                return None
            return ArrayAccessNode(array_expr, index_expr)

        if isinstance(expr, MemberAccessNode):
            object_expr = self.substitute_query_return_expr(
                expr.object_expr,
                param_indices,
                raw_args,
            )
            if object_expr is None:
                return None
            return MemberAccessNode(object_expr, expr.member)

        if isinstance(expr, PointerAccessNode):
            pointer_expr = self.substitute_query_return_expr(
                expr.pointer_expr,
                param_indices,
                raw_args,
            )
            if pointer_expr is None:
                return None
            return PointerAccessNode(pointer_expr, expr.member)

        if isinstance(expr, FunctionCallNode):
            arguments = []
            for argument in getattr(expr, "arguments", getattr(expr, "args", [])):
                substituted = self.substitute_query_return_expr(
                    argument,
                    param_indices,
                    raw_args,
                )
                if substituted is None:
                    return None
                arguments.append(substituted)
            return FunctionCallNode(
                expr.function,
                arguments,
                getattr(expr, "generic_args", []),
            )

        return expr

    def query_array_access_parts(self, expr):
        """Return the base expression and ordered indices for nested array access."""
        indices = []
        current = expr
        while isinstance(current, ArrayAccessNode):
            index_node = getattr(
                current,
                "index_expr",
                getattr(current, "index", None),
            )
            array_node = getattr(
                current,
                "array_expr",
                getattr(current, "array", None),
            )
            if index_node is None or array_node is None:
                return None, []
            indices.insert(0, index_node)
            current = array_node
        return current, indices

    def is_safe_query_return_index(self, index_expr, param_indices):
        """Return whether a returned-array index can be rendered in the caller."""
        if isinstance(index_expr, LiteralNode) or isinstance(index_expr, int):
            return True
        if isinstance(index_expr, BinaryOpNode):
            operator = getattr(index_expr, "operator", getattr(index_expr, "op", None))
            return (
                operator in self.query_return_index_binary_ops
                and self.is_safe_query_return_index(index_expr.left, param_indices)
                and self.is_safe_query_return_index(index_expr.right, param_indices)
            )
        index_name = self.get_expression_name(index_expr)
        if index_name:
            return index_name in param_indices
        if isinstance(index_expr, str):
            return index_expr.lstrip("-").isdigit()
        return False

"""Resource-query metadata of the CUDA generator: requirements, snapshots and safe return expressions."""

from ..ast import (
    ArrayAccessNode,
    AssignmentNode,
    BinaryOpNode,
    FunctionCallNode,
    IdentifierNode,
    LiteralNode,
    MemberAccessNode,
    PointerAccessNode,
    TernaryOpNode,
    UnaryOpNode,
    VariableNode,
)
from .resource_query import ResourceQueryMixin


class CudaQueryMetadataMixin:
    """Query metadata members of ``CudaCodeGen``."""

    def collect_resource_query_requirements(self, node):
        """Collect query metadata needs, resolving local CUDA resource aliases."""
        global_names, function_params = (
            ResourceQueryMixin.collect_resource_query_requirements(self, node)
        )

        functions = self.query_collect_functions(node)
        functions_by_name = {getattr(func, "name", None): func for func in functions}
        functions_by_name = {
            name: func for name, func in functions_by_name.items() if name
        }
        function_parameter_names = {
            func_name: {
                getattr(param, "name", None)
                for param in getattr(func, "parameters", getattr(func, "params", []))
            }
            for func_name, func in functions_by_name.items()
        }
        function_parameter_names = {
            func_name: {name for name in names if name}
            for func_name, names in function_parameter_names.items()
        }

        local_resource_names = {}
        local_alias_sources = {}
        local_assignment_sources = {}
        for func_name, func in functions_by_name.items():
            names, aliases = self.collect_function_resource_aliases(func)
            local_resource_names[func_name] = names
            local_alias_sources[func_name] = aliases
            local_assignment_sources[func_name] = (
                self.collect_function_resource_assignments(func, names)
            )
        local_metadata_snapshots = {func_name: set() for func_name in functions_by_name}

        global_resource_names = {
            getattr(var, "name", None)
            for var in getattr(node, "global_variables", [])
            if self.is_queryable_resource_type(self.get_variable_node_type(var))
        }
        global_resource_names = {name for name in global_resource_names if name}

        def mark_resolved_resource(func_name, resource_expr, visited=None):
            if visited is None:
                visited = set()
            if isinstance(resource_expr, TernaryOpNode):
                if not self.is_safe_query_return_actual_index(resource_expr.condition):
                    return False
                true_changed = mark_resolved_resource(
                    func_name,
                    resource_expr.true_expr,
                    visited.copy(),
                )
                false_changed = mark_resolved_resource(
                    func_name,
                    resource_expr.false_expr,
                    visited.copy(),
                )
                return true_changed or false_changed

            if isinstance(resource_expr, FunctionCallNode):
                callee_name = self.raw_function_call_name(resource_expr)
                return_source = self.query_return_sources.get(callee_name)
                if return_source is None:
                    return False
                raw_args = getattr(
                    resource_expr,
                    "arguments",
                    getattr(resource_expr, "args", []),
                )
                return mark_return_source_resources(
                    func_name,
                    return_source,
                    raw_args,
                    visited,
                )

            resource_name = self.get_expression_name(resource_expr)
            if not resource_name:
                return False
            if resource_name in visited:
                return False
            visited.add(resource_name)

            aliases = local_alias_sources.get(func_name, {})
            assignments = local_assignment_sources.get(func_name, {})
            if resource_name in local_resource_names.get(func_name, set()):
                if resource_name in assignments:
                    snapshot_names = local_metadata_snapshots.setdefault(
                        func_name,
                        set(),
                    )
                    before = len(snapshot_names)
                    snapshot_names.add(resource_name)
                    changed = len(snapshot_names) != before
                    sources = []
                    if resource_name in aliases:
                        sources.append(aliases[resource_name])
                    sources.extend(assignments.get(resource_name, []))
                    for source in sources:
                        changed = (
                            mark_resolved_resource(
                                func_name,
                                source,
                                visited.copy(),
                            )
                            or changed
                        )
                    return changed
                if resource_name in aliases:
                    return mark_resolved_resource(
                        func_name,
                        aliases[resource_name],
                        visited,
                    )
                return False

            if resource_name in function_parameter_names.get(func_name, set()):
                before = len(function_params.setdefault(func_name, set()))
                function_params[func_name].add(resource_name)
                return len(function_params[func_name]) != before

            before = len(global_names)
            global_names.add(resource_name)
            return len(global_names) != before

        def mark_return_source_resources(func_name, return_source, raw_args, visited):
            kind = return_source["kind"]
            if kind == "global":
                before = len(global_names)
                global_names.add(return_source["name"])
                return len(global_names) != before
            if kind == "parameter":
                index = return_source["index"]
                if index >= len(raw_args):
                    return False
                return mark_resolved_resource(func_name, raw_args[index], visited)
            if kind == "ternary":
                true_changed = mark_return_source_resources(
                    func_name,
                    return_source["true_source"],
                    raw_args,
                    visited.copy(),
                )
                false_changed = mark_return_source_resources(
                    func_name,
                    return_source["false_source"],
                    raw_args,
                    visited.copy(),
                )
                return true_changed or false_changed
            return False

        struct_metadata_member_names = {
            member
            for members in self.struct_query_metadata_members.values()
            for member in members
        }

        def mark_struct_constructor_resources(func_name, call):
            struct_name = self.raw_function_call_name(call)
            metadata_members = self.struct_query_metadata_members.get(struct_name)
            if not metadata_members:
                return False
            raw_args = getattr(call, "arguments", getattr(call, "args", []))
            changed = False
            member_types = list(self.struct_member_types.get(struct_name, {}).items())
            for index, (member_name, _) in enumerate(member_types):
                if member_name not in metadata_members or index >= len(raw_args):
                    continue
                changed = (
                    mark_resolved_resource(
                        func_name,
                        raw_args[index],
                    )
                    or changed
                )
            return changed

        def metadata_assignment_member_name(target):
            while isinstance(target, ArrayAccessNode):
                target = getattr(
                    target,
                    "array_expr",
                    getattr(target, "array", None),
                )
            if isinstance(target, (MemberAccessNode, PointerAccessNode)):
                member = getattr(target, "member", None)
                return getattr(member, "name", member)
            return None

        for func_name, func in functions_by_name.items():
            for node in self.query_walk_nodes(getattr(func, "body", [])):
                if isinstance(node, FunctionCallNode):
                    changed = mark_struct_constructor_resources(func_name, node)
                    if changed:
                        continue
                if not isinstance(node, AssignmentNode):
                    continue
                member_name = metadata_assignment_member_name(
                    getattr(node, "target", None)
                )
                if member_name not in struct_metadata_member_names:
                    continue
                value = getattr(node, "value", None)
                if value is not None:
                    mark_resolved_resource(func_name, value)

        for func_name, func in functions_by_name.items():
            for call in self.query_walk_nodes(getattr(func, "body", [])):
                if not isinstance(call, FunctionCallNode):
                    continue
                func_call_name = self.raw_function_call_name(call)
                raw_args = getattr(call, "arguments", getattr(call, "args", []))
                if func_call_name in self.query_function_names and raw_args:
                    mark_resolved_resource(func_name, raw_args[0])

        changed = True
        while changed:
            changed = False
            for caller_name, caller in functions_by_name.items():
                for call in self.query_walk_nodes(getattr(caller, "body", [])):
                    if not isinstance(call, FunctionCallNode):
                        continue
                    callee_name = self.raw_function_call_name(call)
                    callee = functions_by_name.get(callee_name)
                    if callee is None:
                        continue

                    callee_required = function_params.get(callee_name, set())
                    if not callee_required:
                        continue

                    callee_params = getattr(
                        callee, "parameters", getattr(callee, "params", [])
                    )
                    raw_args = getattr(call, "arguments", getattr(call, "args", []))
                    for index, param in enumerate(callee_params):
                        if index >= len(raw_args):
                            continue
                        param_name = getattr(param, "name", None)
                        if param_name not in callee_required:
                            continue
                        changed = (
                            mark_resolved_resource(caller_name, raw_args[index])
                            or changed
                        )

        for names in local_resource_names.values():
            global_names.difference_update(names - global_resource_names)

        self.query_local_resource_names_by_function = {
            func_name: names
            for func_name, names in local_resource_names.items()
            if names
        }
        self.query_metadata_snapshot_locals_by_function = {
            func_name: names
            for func_name, names in local_metadata_snapshots.items()
            if names
        }
        return (
            global_names,
            {func_name: names for func_name, names in function_params.items() if names},
        )

    def register_query_metadata_alias(self, name, type_name, source_node=None):
        """Track local resource aliases that can reuse an existing metadata sidecar."""
        if not self.current_function_name or not name:
            return
        if not self.is_queryable_resource_type(type_name) or source_node is None:
            self.query_metadata_aliases.pop(name, None)
            return

        metadata_expr = self.query_metadata_expression(source_node)
        if metadata_expr is None:
            self.query_metadata_aliases.pop(name, None)
            return
        self.query_metadata_aliases[name] = metadata_expr

    def is_query_metadata_snapshot_local(self, name, type_name=None):
        """Return whether a local resource needs its own mutable metadata sidecar."""
        if not self.current_function_name or not name:
            return False
        snapshot_names = self.query_metadata_snapshot_locals_by_function.get(
            self.current_function_name,
            set(),
        )
        if name not in snapshot_names:
            return False
        if type_name is not None and not self.is_queryable_resource_type(type_name):
            return False
        return True

    def query_metadata_snapshot_declaration(self, name, type_name, initial_value=None):
        """Return a local metadata sidecar declaration for reassigned resources."""
        if not self.is_query_metadata_snapshot_local(name, type_name):
            return None

        declarator = self.format_typed_declarator(
            self.query_metadata_type(type_name),
            self.query_metadata_name(name),
            dynamic_array_as_pointer=False,
        )
        metadata_expr = None
        if initial_value is not None:
            metadata_expr = self.query_metadata_expression(initial_value)
        if metadata_expr is None:
            metadata_expr = "{}"
        return f"{declarator} = {metadata_expr}"

    def struct_member_has_query_metadata(self, struct_type, member_name):
        """Return whether a struct member has an embedded metadata sidecar."""
        struct_type = self.struct_member_lookup_type(struct_type)
        return member_name in self.struct_query_metadata_members.get(struct_type, set())

    def can_reuse_struct_metadata_object_expression(
        self, object_node, allow_function_call_object=False
    ):
        """Return whether a direct member sidecar read avoids duplicating calls."""
        if object_node is None:
            return False
        if isinstance(object_node, FunctionCallNode):
            return allow_function_call_object
        if isinstance(object_node, ArrayAccessNode):
            array_node = getattr(
                object_node,
                "array_expr",
                getattr(object_node, "array", None),
            )
            index_node = getattr(
                object_node,
                "index_expr",
                getattr(object_node, "index", None),
            )
            return self.can_reuse_struct_metadata_object_expression(
                array_node, allow_function_call_object
            ) and self.is_safe_query_return_actual_index(index_node)
        if isinstance(object_node, MemberAccessNode):
            return self.can_reuse_struct_metadata_object_expression(
                getattr(
                    object_node,
                    "object_expr",
                    getattr(object_node, "object", None),
                ),
                allow_function_call_object,
            )
        if isinstance(object_node, PointerAccessNode):
            return self.can_reuse_struct_metadata_object_expression(
                getattr(object_node, "pointer_expr", None),
                allow_function_call_object,
            )
        return True

    def query_struct_member_metadata_expression(
        self, resource_expr, allow_function_call_object=False
    ):
        """Return metadata paired with a struct resource-member expression."""
        if isinstance(resource_expr, PointerAccessNode):
            object_node = getattr(resource_expr, "pointer_expr", None)
            member_name = getattr(resource_expr, "member", None)
            object_type = self.resource_expression_type(object_node)
            pointer_info = self.cuda_indirect_type_info(object_type)
            if pointer_info is None:
                return None
            struct_type = self.struct_member_lookup_type(pointer_info["pointee_type"])
            object_access = "->"
        elif isinstance(resource_expr, MemberAccessNode):
            object_node = getattr(
                resource_expr,
                "object_expr",
                getattr(resource_expr, "object", None),
            )
            member_name = getattr(resource_expr, "member", None)
            struct_type = self.struct_member_lookup_type(
                self.resource_expression_type(object_node)
            )
            object_access = "."
        else:
            return None

        if object_node is None or not isinstance(member_name, str):
            return None
        if not self.can_reuse_struct_metadata_object_expression(
            object_node, allow_function_call_object
        ):
            return None
        if not self.struct_member_has_query_metadata(struct_type, member_name):
            return None

        object_expr = self.visit(object_node)
        if not object_expr:
            return None
        return f"{object_expr}{object_access}{self.query_metadata_name(member_name)}"

    def is_struct_member_metadata_assignment_target(self, target_node):
        """Return whether an assignment target is a struct resource member."""
        if isinstance(target_node, ArrayAccessNode):
            target_node = getattr(
                target_node,
                "array_expr",
                getattr(target_node, "array", None),
            )
        return isinstance(target_node, (MemberAccessNode, PointerAccessNode))

    def format_query_metadata_assignment(self, node):
        """Return a metadata sidecar update for a reassigned local resource."""
        if getattr(node, "operator", "=") != "=":
            return None

        target_node = getattr(node, "target", None)
        if isinstance(target_node, (IdentifierNode, VariableNode, str)):
            target_name = self.get_expression_name(target_node)
            target_type = self.variable_types.get(target_name)
            if not self.is_query_metadata_snapshot_local(target_name, target_type):
                return None

            metadata_expr = self.query_metadata_expression(getattr(node, "value", None))
            if metadata_expr is None:
                resource_type = self.resource_base_type(target_type) or target_type
                resource_type = resource_type or "resource"
                metadata_expr = self.refuse_lowering(
                    f"CUDA resource query: metadata unavailable for {resource_type} assignment",
                    "CglResourceQueryInfo{}",
                )
            return f"{self.query_metadata_name(target_name)} = {metadata_expr}"

        if not self.is_struct_member_metadata_assignment_target(target_node):
            return None
        target_metadata = self.query_metadata_expression(target_node)
        if target_metadata is None:
            return None
        target_type = self.resource_expression_type(target_node)
        if not self.is_queryable_resource_type(target_type):
            return None
        metadata_expr = self.query_metadata_expression(getattr(node, "value", None))
        if metadata_expr is None:
            resource_type = self.resource_base_type(target_type) or target_type
            resource_type = resource_type or "resource"
            metadata_expr = self.refuse_lowering(
                f"CUDA resource query: metadata unavailable for {resource_type} assignment",
                "CglResourceQueryInfo{}",
            )
        return f"{target_metadata} = {metadata_expr}"

    def query_metadata_expression(
        self, resource_expr, allow_function_call_object=False
    ):
        """Return CUDA query metadata paired with a resource expression."""
        if isinstance(resource_expr, TernaryOpNode):
            if not self.is_safe_query_return_actual_index(resource_expr.condition):
                return None
            true_type = self.resource_base_type(
                self.resource_expression_type(resource_expr.true_expr)
            )
            false_type = self.resource_base_type(
                self.resource_expression_type(resource_expr.false_expr)
            )
            if true_type is None or false_type is None or true_type != false_type:
                return None
            true_metadata = self.query_metadata_expression(
                resource_expr.true_expr, allow_function_call_object
            )
            false_metadata = self.query_metadata_expression(
                resource_expr.false_expr, allow_function_call_object
            )
            if true_metadata is None or false_metadata is None:
                return None
            condition = self.visit(resource_expr.condition)
            return f"({condition} ? {true_metadata} : {false_metadata})"

        if isinstance(resource_expr, FunctionCallNode):
            callee_name = self.raw_function_call_name(resource_expr)
            return_source = self.query_return_sources.get(callee_name)
            if return_source is None:
                return None
            raw_args = getattr(
                resource_expr,
                "arguments",
                getattr(resource_expr, "args", []),
            )
            return self.query_return_source_metadata_expression(
                return_source,
                raw_args,
                allow_function_call_object,
            )

        if isinstance(resource_expr, ArrayAccessNode):
            array_node = getattr(
                resource_expr,
                "array_expr",
                getattr(resource_expr, "array", None),
            )
            index_node = getattr(
                resource_expr,
                "index_expr",
                getattr(resource_expr, "index", None),
            )
            if index_node is None or not self.is_safe_query_return_actual_index(
                index_node
            ):
                return None
            base_expr = self.query_metadata_expression(
                array_node, allow_function_call_object
            )
            if base_expr is None:
                return None
            return f"{base_expr}[{self.visit(index_node)}]"

        member_metadata = self.query_struct_member_metadata_expression(
            resource_expr, allow_function_call_object
        )
        if member_metadata is not None:
            return member_metadata

        resource_name = self.get_expression_name(resource_expr)
        if not resource_name:
            return None

        alias_expr = self.query_metadata_aliases.get(resource_name)
        if alias_expr is not None:
            return alias_expr

        current_function = self.current_function_name
        local_names = self.query_local_resource_names_by_function.get(
            current_function,
            set(),
        )
        if resource_name in local_names:
            return None

        if current_function:
            query_params = self.query_metadata_function_params.get(
                current_function,
                set(),
            )
            if resource_name in query_params:
                return self.query_metadata_name(resource_name)

        if resource_name in self.query_resource_names:
            return self.query_metadata_name(resource_name)
        return None

    def query_return_source_metadata_expression(
        self, return_source, raw_args, allow_function_call_object=False
    ):
        """Render query metadata for a resource returned by a user function."""
        if return_source["kind"] == "ternary":
            condition = self.format_query_return_safe_expression(
                return_source["condition"],
                return_source.get("param_indices", {}),
                raw_args,
            )
            if condition is None:
                return None
            true_metadata = self.query_return_source_metadata_expression(
                return_source["true_source"],
                raw_args,
                allow_function_call_object,
            )
            false_metadata = self.query_return_source_metadata_expression(
                return_source["false_source"],
                raw_args,
                allow_function_call_object,
            )
            if true_metadata is None or false_metadata is None:
                return None
            return f"({condition} ? {true_metadata} : {false_metadata})"

        if return_source["kind"] == "global":
            metadata_expr = self.query_metadata_name(return_source["name"])
        elif return_source["kind"] == "parameter":
            index = return_source["index"]
            if index >= len(raw_args):
                return None
            metadata_expr = self.query_metadata_expression(
                raw_args[index], allow_function_call_object
            )
        elif return_source["kind"] == "member":
            member = return_source.get("member")
            object_type = return_source.get("object_type")
            if not self.struct_member_has_query_metadata(object_type, member):
                return None
            object_expr = return_source.get("object_expr")
            if object_expr is None:
                return None
            metadata_object = self.format_query_return_metadata_object_expression(
                object_expr,
                return_source.get("param_indices", {}),
                raw_args,
            )
            if metadata_object is None:
                return None
            metadata_expr = (
                f"{metadata_object}"
                f"{return_source.get('object_access', '.')}"
                f"{self.query_metadata_name(member)}"
            )
        else:
            return None
        if metadata_expr is None:
            return None

        param_indices = return_source.get("param_indices", {})
        for index_expr in return_source.get("indices", []):
            rendered_index = self.format_query_return_index(
                index_expr,
                param_indices,
                raw_args,
            )
            if rendered_index is None:
                return None
            metadata_expr = f"{metadata_expr}[{rendered_index}]"
        return metadata_expr

    def format_query_return_metadata_object_expression(
        self,
        expr,
        param_indices,
        raw_args,
    ):
        """Render a returned struct object expression for member sidecar access."""
        if isinstance(expr, FunctionCallNode):
            func_name = self.raw_function_call_name(expr)
            if not func_name:
                return None
            arguments = []
            for argument in getattr(expr, "arguments", getattr(expr, "args", [])):
                argument_expr = self.format_query_return_safe_expression(
                    argument,
                    param_indices,
                    raw_args,
                    allow_free_identifiers=True,
                )
                if argument_expr is None:
                    return None
                arguments.append(argument_expr)
            return f"{func_name}({', '.join(arguments)})"

        return self.format_query_return_safe_expression(
            expr,
            param_indices,
            raw_args,
            allow_free_identifiers=True,
        )

    def format_query_return_safe_expression(
        self,
        expr,
        param_indices,
        raw_args,
        allow_free_identifiers=False,
    ):
        """Render a side-effect-safe returned-resource selector expression."""
        if isinstance(expr, BinaryOpNode):
            operator = getattr(expr, "operator", getattr(expr, "op", None))
            if operator not in (
                self.query_return_index_binary_ops
                | {"&&", "||", "==", "!=", "<", "<=", ">", ">="}
            ):
                return None
            left = self.format_query_return_safe_expression(
                expr.left,
                param_indices,
                raw_args,
                allow_free_identifiers=allow_free_identifiers,
            )
            right = self.format_query_return_safe_expression(
                expr.right,
                param_indices,
                raw_args,
                allow_free_identifiers=allow_free_identifiers,
            )
            if left is None or right is None:
                return None
            return f"({left} {operator} {right})"

        if isinstance(expr, UnaryOpNode):
            operator = getattr(expr, "operator", getattr(expr, "op", None))
            if getattr(expr, "is_postfix", False) or operator not in {
                "!",
                "+",
                "-",
                "~",
            }:
                return None
            operand = self.format_query_return_safe_expression(
                expr.operand,
                param_indices,
                raw_args,
                allow_free_identifiers=allow_free_identifiers,
            )
            if operand is None:
                return None
            return f"{operator}{operand}"

        if isinstance(expr, TernaryOpNode):
            condition = self.format_query_return_safe_expression(
                expr.condition,
                param_indices,
                raw_args,
                allow_free_identifiers=allow_free_identifiers,
            )
            true_expr = self.format_query_return_safe_expression(
                expr.true_expr,
                param_indices,
                raw_args,
                allow_free_identifiers=allow_free_identifiers,
            )
            false_expr = self.format_query_return_safe_expression(
                expr.false_expr,
                param_indices,
                raw_args,
                allow_free_identifiers=allow_free_identifiers,
            )
            if condition is None or true_expr is None or false_expr is None:
                return None
            return f"({condition} ? {true_expr} : {false_expr})"

        if isinstance(expr, ArrayAccessNode):
            array_node = getattr(
                expr,
                "array_expr",
                getattr(expr, "array", None),
            )
            index_node = getattr(
                expr,
                "index_expr",
                getattr(expr, "index", None),
            )
            if array_node is None or index_node is None:
                return None
            array_expr = self.format_query_return_safe_expression(
                array_node,
                param_indices,
                raw_args,
                allow_free_identifiers=allow_free_identifiers,
            )
            index_expr = self.format_query_return_safe_expression(
                index_node,
                param_indices,
                raw_args,
                allow_free_identifiers=allow_free_identifiers,
            )
            if array_expr is None or index_expr is None:
                return None
            return f"{array_expr}[{index_expr}]"

        if isinstance(expr, MemberAccessNode):
            object_node = getattr(
                expr,
                "object_expr",
                getattr(expr, "object", None),
            )
            member = getattr(expr, "member", None)
            member_name = getattr(member, "name", member)
            if object_node is None or not isinstance(member_name, str):
                return None
            object_expr = self.format_query_return_safe_expression(
                object_node,
                param_indices,
                raw_args,
                allow_free_identifiers=allow_free_identifiers,
            )
            if object_expr is None:
                return None
            return f"{object_expr}.{member_name}"

        if isinstance(expr, PointerAccessNode):
            pointer_node = getattr(expr, "pointer_expr", None)
            member = getattr(expr, "member", None)
            member_name = getattr(member, "name", member)
            if pointer_node is None or not isinstance(member_name, str):
                return None
            pointer_expr = self.format_query_return_safe_expression(
                pointer_node,
                param_indices,
                raw_args,
                allow_free_identifiers=allow_free_identifiers,
            )
            if pointer_expr is None:
                return None
            return f"{pointer_expr}->{member_name}"

        expr_name = self.get_expression_name(expr)
        if expr_name in param_indices:
            param_index = param_indices[expr_name]
            if param_index >= len(raw_args):
                return None
            return self.format_query_return_safe_expression(
                raw_args[param_index],
                {},
                [],
                allow_free_identifiers=True,
            )
        if allow_free_identifiers and expr_name:
            return self.visit(expr)
        if isinstance(expr, LiteralNode):
            return self.visit(expr)
        if isinstance(expr, bool):
            return "true" if expr else "false"
        if isinstance(expr, (int, float)):
            return str(expr)
        if isinstance(expr, str):
            stripped = expr.strip()
            if stripped in {"true", "false"} or stripped.lstrip("-").isdigit():
                return stripped
            if allow_free_identifiers and stripped.isidentifier():
                return stripped
        return None

    def format_query_return_index(self, index_expr, param_indices, raw_args):
        """Render a returned-array index in the caller's argument context."""
        if isinstance(index_expr, BinaryOpNode):
            operator = getattr(index_expr, "operator", getattr(index_expr, "op", None))
            if operator not in self.query_return_index_binary_ops:
                return None
            left = self.format_query_return_index(
                index_expr.left,
                param_indices,
                raw_args,
            )
            right = self.format_query_return_index(
                index_expr.right,
                param_indices,
                raw_args,
            )
            if left is None or right is None:
                return None
            return f"({left} {operator} {right})"
        index_name = self.get_expression_name(index_expr)
        if index_name in param_indices:
            param_index = param_indices[index_name]
            if param_index >= len(raw_args):
                return None
            actual_expr = raw_args[param_index]
            if not self.is_safe_query_return_actual_index(actual_expr):
                return None
            return self.visit(actual_expr)
        if isinstance(index_expr, LiteralNode):
            return self.visit(index_expr)
        if isinstance(index_expr, int):
            return str(index_expr)
        if isinstance(index_expr, str) and index_expr.lstrip("-").isdigit():
            return index_expr
        return None

    def is_safe_query_return_actual_index(self, actual_expr):
        """Return whether a caller argument can be reused as a metadata index."""
        if isinstance(actual_expr, LiteralNode) or isinstance(actual_expr, int):
            return True
        if isinstance(actual_expr, (IdentifierNode, VariableNode)):
            return True
        if isinstance(actual_expr, str):
            actual_expr = actual_expr.strip()
            return actual_expr.lstrip("-").isdigit() or actual_expr.isidentifier()
        if isinstance(actual_expr, BinaryOpNode):
            operator = getattr(
                actual_expr, "operator", getattr(actual_expr, "op", None)
            )
            return (
                operator in self.query_return_index_binary_ops
                and self.is_safe_query_return_actual_index(actual_expr.left)
                and self.is_safe_query_return_actual_index(actual_expr.right)
            )
        if isinstance(actual_expr, UnaryOpNode):
            operator = getattr(
                actual_expr, "operator", getattr(actual_expr, "op", None)
            )
            return (
                not getattr(actual_expr, "is_postfix", False)
                and operator in {"+", "-", "~"}
                and self.is_safe_query_return_actual_index(actual_expr.operand)
            )
        if isinstance(actual_expr, TernaryOpNode):
            return (
                self.is_safe_query_return_actual_index(actual_expr.condition)
                and self.is_safe_query_return_actual_index(actual_expr.true_expr)
                and self.is_safe_query_return_actual_index(actual_expr.false_expr)
            )
        if isinstance(actual_expr, ArrayAccessNode):
            array_node = getattr(
                actual_expr,
                "array_expr",
                getattr(actual_expr, "array", None),
            )
            index_node = getattr(
                actual_expr,
                "index_expr",
                getattr(actual_expr, "index", None),
            )
            return (
                array_node is not None
                and index_node is not None
                and self.is_safe_query_return_actual_index(array_node)
                and self.is_safe_query_return_actual_index(index_node)
            )
        if isinstance(actual_expr, MemberAccessNode):
            object_expr = getattr(
                actual_expr,
                "object_expr",
                getattr(actual_expr, "object", None),
            )
            return object_expr is not None and self.is_safe_query_return_actual_index(
                object_expr
            )
        return False

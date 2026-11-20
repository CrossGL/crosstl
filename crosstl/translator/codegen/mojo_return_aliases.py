"""Function return types and the resource aliases that function returns carry in the Mojo generator."""

import re

from ..ast import (
    ArrayLiteralNode,
    ConstantNode,
    ConstructorNode,
    FunctionCallNode,
    MatchNode,
    ReturnNode,
    TernaryOpNode,
)


class MojoReturnAliasMixin:
    """Return-alias members of ``MojoCodeGen``."""

    def generate_constants(self, ast):
        code = ""
        for node in getattr(ast, "constants", []) or []:
            if not isinstance(node, ConstantNode):
                continue
            name = getattr(node, "name", None)
            if not name:
                continue
            const_type = self.convert_type_node_to_string(
                getattr(node, "const_type", None)
            )
            value = getattr(node, "value", None)
            self.register_variable_type(name, const_type)
            value_code = self.generate_expression(value, const_type, f"constant {name}")
            code += f"comptime {self.mojo_identifier(name)} = {value_code}\n"
        return f"{code}\n" if code else ""

    def collect_function_return_types(self, ast):
        functions = list(getattr(ast, "functions", []))
        stages = getattr(ast, "stages", {})
        if stages:
            for stage in stages.values():
                entry_point = getattr(stage, "entry_point", None)
                if entry_point is not None:
                    functions.append(entry_point)
                functions.extend(self.stage_local_functions_to_emit(stage))

        for func in functions:
            self.register_function_return_type(func)

        for _ in range(max(1, len(functions))):
            changed = False
            for func in functions:
                changed = (
                    self.register_function_return_resource_aliases(func) or changed
                )
            if not changed:
                break

    def register_function_return_type(self, func):
        if not hasattr(func, "name"):
            return

        if hasattr(func, "return_type"):
            return_type = self.convert_type_node_to_string(func.return_type)
        else:
            return_type = "void"
        parameter_types = [
            self.function_parameter_type_name(param)
            for param in getattr(func, "parameters", getattr(func, "params", []))
        ]
        self.function_return_types[func.name] = return_type
        self.function_parameter_types[func.name] = parameter_types
        signature = {
            "return_type": return_type,
            "parameter_types": parameter_types,
        }
        overloads = self.function_overload_signatures.setdefault(func.name, [])
        if not any(
            existing["return_type"] == return_type
            and existing["parameter_types"] == parameter_types
            for existing in overloads
        ):
            overloads.append(signature)

    def register_function_return_resource_aliases(self, func):
        if not hasattr(func, "name"):
            return False

        return_type = self.function_return_types.get(func.name)
        if not self.function_return_resource_aliasable_type(return_type):
            return False

        params = getattr(func, "parameters", getattr(func, "params", []))
        param_indices = {
            getattr(param, "name", None): index
            for index, param in enumerate(params)
            if getattr(param, "name", None)
        }
        aliases = []
        static_aliases = []
        field_aliases = []
        static_field_aliases = {}
        for return_node in self.return_nodes(getattr(func, "body", [])):
            value = getattr(return_node, "value", None)
            for index in self.return_resource_alias_param_indices(
                value, param_indices, target_type=return_type
            ):
                if index not in aliases:
                    aliases.append(index)
            for ref in self.return_resource_alias_field_paths(
                value, param_indices, target_type=return_type
            ):
                if ref not in field_aliases:
                    field_aliases.append(ref)
            for candidate in self.return_resource_static_alias_candidates(
                value, param_indices, target_type=return_type
            ):
                if candidate not in static_aliases:
                    static_aliases.append(candidate)
            static_field_aliases = self.merge_resource_field_alias_maps(
                static_field_aliases,
                self.return_resource_static_field_alias_candidates(
                    value, param_indices, target_type=return_type
                ),
            )
        if (
            aliases == self.function_return_resource_aliases.get(func.name)
            and static_aliases == self.function_return_resource_static_aliases.get(
                func.name
            )
            and field_aliases == self.function_return_resource_field_aliases.get(
                func.name
            )
            and static_field_aliases
            == self.function_return_resource_static_field_aliases.get(func.name)
        ):
            return False
        if aliases:
            self.function_return_resource_aliases[func.name] = aliases
        else:
            self.function_return_resource_aliases.pop(func.name, None)
        if field_aliases:
            self.function_return_resource_field_aliases[func.name] = field_aliases
        else:
            self.function_return_resource_field_aliases.pop(func.name, None)
        if static_aliases:
            self.function_return_resource_static_aliases[func.name] = static_aliases
        else:
            self.function_return_resource_static_aliases.pop(func.name, None)
        if static_field_aliases:
            self.function_return_resource_static_field_aliases[func.name] = (
                static_field_aliases
            )
        else:
            self.function_return_resource_static_field_aliases.pop(func.name, None)
        return True

    def return_nodes(self, body):
        nodes = []
        for stmt in self.body_statements(body):
            self.collect_return_nodes(stmt, nodes)
        return nodes

    def collect_return_nodes(self, node, nodes):
        if node is None:
            return
        if isinstance(node, ReturnNode):
            nodes.append(node)
            return
        for child in self.node_children(node):
            self.collect_return_nodes(child, nodes)

    def function_return_resource_aliasable_type(self, return_type):
        return self.resource_access_aliasable_type(return_type)

    def return_resource_alias_param_indices(
        self, expr, param_indices, target_type=None
    ):
        if isinstance(expr, list):
            return []

        root_name = self.resource_access_root_name(expr)
        if root_name in param_indices:
            return [param_indices[root_name]]

        if isinstance(expr, TernaryOpNode):
            indices = []
            for branch in (expr.true_expr, expr.false_expr):
                for index in self.return_resource_alias_param_indices(
                    branch, param_indices, target_type=target_type
                ):
                    if index not in indices:
                        indices.append(index)
            return indices

        if isinstance(expr, MatchNode):
            indices = []
            for arm in getattr(expr, "arms", []) or []:
                try:
                    arm_expr, _ = self.match_arm_value_expression(arm)
                except ValueError:
                    continue
                for index in self.return_resource_alias_param_indices(
                    arm_expr, param_indices, target_type=target_type
                ):
                    if index not in indices:
                        indices.append(index)
            return indices

        if isinstance(expr, ArrayLiteralNode):
            element_type = self.array_element_type(target_type)
            indices = []
            for element in getattr(expr, "elements", []) or []:
                for index in self.return_resource_alias_param_indices(
                    element, param_indices, target_type=element_type
                ):
                    if index not in indices:
                        indices.append(index)
            return indices

        if isinstance(expr, FunctionCallNode):
            return self.return_resource_alias_param_indices_from_call(
                expr, param_indices, target_type=target_type
            )

        return []

    def return_resource_alias_param_indices_from_call(
        self, expr, param_indices, target_type=None
    ):
        func_name = self.function_call_name(expr)
        fields = self.generic_enum_variant_fields_for_type(func_name, target_type)
        if fields is not None and len(fields) == len(expr.args):
            indices = []
            for arg, (_field_name, field_type) in zip(expr.args, fields):
                if not self.resource_access_aliasable_type(field_type):
                    continue
                for index in self.return_resource_alias_param_indices(
                    arg, param_indices, target_type=field_type
                ):
                    if index not in indices:
                        indices.append(index)
            return indices

        alias_indices = self.function_return_resource_aliases.get(func_name)
        if not alias_indices:
            return []

        indices = []
        for alias_index in alias_indices:
            if alias_index >= len(expr.args):
                continue
            for index in self.return_resource_alias_param_indices(
                expr.args[alias_index], param_indices
            ):
                if index not in indices:
                    indices.append(index)
        return indices

    def return_resource_alias_field_paths(self, expr, param_indices, target_type=None):
        if isinstance(expr, list):
            return []

        target_type = self.type_name(target_type)
        root_name = self.resource_access_root_name(expr)
        if root_name in param_indices:
            return [
                (path, param_indices[root_name], path)
                for path in self.resource_aliasable_field_paths_for_type(target_type)
            ]

        if isinstance(expr, TernaryOpNode):
            refs = []
            for branch in (expr.true_expr, expr.false_expr):
                for ref in self.return_resource_alias_field_paths(
                    branch, param_indices, target_type=target_type
                ):
                    if ref not in refs:
                        refs.append(ref)
            return refs

        if isinstance(expr, MatchNode):
            refs = []
            for arm in getattr(expr, "arms", []) or []:
                try:
                    arm_expr, _ = self.match_arm_value_expression(arm)
                except ValueError:
                    continue
                for ref in self.return_resource_alias_field_paths(
                    arm_expr, param_indices, target_type=target_type
                ):
                    if ref not in refs:
                        refs.append(ref)
            return refs

        if isinstance(expr, ArrayLiteralNode):
            element_type = self.array_element_type(target_type)
            refs = []
            for element in getattr(expr, "elements", []) or []:
                for ref in self.return_resource_alias_field_paths(
                    element, param_indices, target_type=element_type
                ):
                    if ref not in refs:
                        refs.append(ref)
            return refs

        if isinstance(expr, ConstructorNode):
            return self.return_constructor_node_resource_field_alias_paths(
                expr, param_indices, target_type
            )

        if isinstance(expr, FunctionCallNode):
            return self.return_resource_alias_field_paths_from_call(
                expr, param_indices, target_type=target_type
            )

        return []

    def return_resource_alias_field_paths_from_call(
        self, expr, param_indices, target_type=None
    ):
        func_name = self.function_call_name(expr)
        refs = []

        fields = self.generic_enum_variant_fields_for_type(func_name, target_type)
        if fields is not None and len(fields) == len(expr.args):
            for arg, (field_name, field_type) in zip(expr.args, fields):
                if not self.resource_access_aliasable_type(field_type):
                    continue
                for (
                    return_path,
                    alias_index,
                    arg_path,
                ) in self.return_resource_alias_field_paths(
                    arg, param_indices, target_type=field_type
                ):
                    path = f"{field_name}.{return_path}" if return_path else field_name
                    ref = (path, alias_index, arg_path)
                    if ref not in refs:
                        refs.append(ref)
            return refs

        struct_refs = self.return_struct_constructor_resource_field_alias_paths(
            expr, param_indices, target_type
        )
        if struct_refs:
            return struct_refs

        callee_refs = self.function_return_resource_field_aliases.get(func_name) or []
        parameter_types = self.function_parameter_types.get(func_name, [])
        for return_path, alias_index, callee_arg_path in callee_refs:
            if alias_index >= len(expr.args):
                continue
            arg_type = (
                parameter_types[alias_index]
                if alias_index < len(parameter_types)
                else None
            )
            for (
                current_index,
                current_arg_path,
            ) in self.return_resource_alias_field_paths_for_relative_path(
                expr.args[alias_index],
                param_indices,
                callee_arg_path,
                arg_type,
            ):
                ref = (return_path, current_index, current_arg_path)
                if ref not in refs:
                    refs.append(ref)
        return refs

    def return_struct_constructor_resource_field_alias_paths(
        self, expr, param_indices, target_type=None
    ):
        func_name = self.function_call_name(expr)
        target_base, _ = self.resource_base_type_and_count(self.type_name(target_type))
        struct_type = target_base if target_base in self.struct_types else func_name
        if struct_type not in self.struct_types or func_name != struct_type:
            return []

        refs = []
        field_items = list(self.struct_types.get(struct_type, {}).items())
        for arg, (field_name, field_type) in zip(expr.args, field_items):
            if not self.resource_access_aliasable_type(field_type):
                continue
            for (
                return_path,
                alias_index,
                arg_path,
            ) in self.return_resource_alias_field_paths(
                arg, param_indices, target_type=field_type
            ):
                path = f"{field_name}.{return_path}" if return_path else field_name
                ref = (path, alias_index, arg_path)
                if ref not in refs:
                    refs.append(ref)
        return refs

    def constructor_node_struct_type(self, expr, target_type=None):
        struct_type, _ = self.constructor_node_struct_type_and_fields(expr, target_type)
        return struct_type

    def constructor_node_field_bindings(self, expr, struct_type, fields=None):
        fields = (
            fields if fields is not None else self.struct_types.get(struct_type, {})
        )
        field_items = list(fields.items())
        for index, arg in enumerate(getattr(expr, "arguments", []) or []):
            if index >= len(field_items):
                continue
            field_name, field_type = field_items[index]
            yield field_name, field_type, arg
        for field_name, arg in (getattr(expr, "named_arguments", {}) or {}).items():
            field_type = fields.get(field_name)
            if field_type is None:
                continue
            yield field_name, field_type, arg

    def return_constructor_node_resource_field_alias_paths(
        self, expr, param_indices, target_type=None
    ):
        struct_type, fields = self.constructor_node_struct_type_and_fields(
            expr, target_type
        )
        if struct_type is None:
            return []

        refs = []
        for field_name, field_type, arg in self.constructor_node_field_bindings(
            expr, struct_type, fields
        ):
            if not self.resource_access_aliasable_type(field_type):
                continue
            for (
                return_path,
                alias_index,
                arg_path,
            ) in self.return_resource_alias_field_paths(
                arg, param_indices, target_type=field_type
            ):
                path = f"{field_name}.{return_path}" if return_path else field_name
                ref = (path, alias_index, arg_path)
                if ref not in refs:
                    refs.append(ref)
        return refs

    def return_resource_alias_field_paths_for_relative_path(
        self, expr, param_indices, relative_path, target_type=None
    ):
        if not relative_path:
            return [
                (alias_index, arg_path)
                for return_path, alias_index, arg_path in (
                    self.return_resource_alias_field_paths(
                        expr, param_indices, target_type=target_type
                    )
                )
                if not return_path
            ]

        if isinstance(expr, TernaryOpNode):
            refs = []
            for branch in (expr.true_expr, expr.false_expr):
                for ref in self.return_resource_alias_field_paths_for_relative_path(
                    branch, param_indices, relative_path, target_type
                ):
                    if ref not in refs:
                        refs.append(ref)
            return refs

        if isinstance(expr, MatchNode):
            refs = []
            for arm in getattr(expr, "arms", []) or []:
                try:
                    arm_expr, _ = self.match_arm_value_expression(arm)
                except ValueError:
                    continue
                for ref in self.return_resource_alias_field_paths_for_relative_path(
                    arm_expr, param_indices, relative_path, target_type
                ):
                    if ref not in refs:
                        refs.append(ref)
            return refs

        if isinstance(expr, ArrayLiteralNode):
            element_type = self.array_element_type(target_type)
            refs = []
            for element in getattr(expr, "elements", []) or []:
                for ref in self.return_resource_alias_field_paths_for_relative_path(
                    element, param_indices, relative_path, element_type
                ):
                    if ref not in refs:
                        refs.append(ref)
            return refs

        root_name = self.resource_access_root_name(expr)
        if root_name in param_indices:
            return [(param_indices[root_name], relative_path)]

        if isinstance(expr, ConstructorNode):
            struct_type, fields = self.constructor_node_struct_type_and_fields(
                expr, target_type
            )
            if struct_type is not None:
                first, _, rest = relative_path.partition(".")
                for field_name, field_type, arg in self.constructor_node_field_bindings(
                    expr, struct_type, fields
                ):
                    if field_name == first:
                        return self.return_resource_alias_field_paths_for_relative_path(
                            arg, param_indices, rest, field_type
                        )

        if isinstance(expr, FunctionCallNode):
            func_name = self.function_call_name(expr)
            target_base, _ = self.resource_base_type_and_count(
                self.type_name(target_type)
            )
            struct_type = target_base if target_base in self.struct_types else func_name
            if struct_type in self.struct_types and func_name == struct_type:
                first, _, rest = relative_path.partition(".")
                field_items = list(self.struct_types.get(struct_type, {}).items())
                for arg, (field_name, field_type) in zip(expr.args, field_items):
                    if field_name == first:
                        return self.return_resource_alias_field_paths_for_relative_path(
                            arg, param_indices, rest, field_type
                        )

            refs = []
            callee_refs = self.function_return_resource_field_aliases.get(func_name)
            parameter_types = self.function_parameter_types.get(func_name, [])
            for return_path, alias_index, callee_arg_path in callee_refs or []:
                if alias_index >= len(expr.args):
                    continue

                if relative_path == return_path:
                    next_relative_path = callee_arg_path
                elif return_path and relative_path.startswith(f"{return_path}."):
                    remaining_path = relative_path[len(return_path) + 1 :]
                    next_relative_path = (
                        f"{callee_arg_path}.{remaining_path}"
                        if callee_arg_path
                        else remaining_path
                    )
                elif not return_path:
                    next_relative_path = relative_path
                else:
                    continue

                arg_type = (
                    parameter_types[alias_index]
                    if alias_index < len(parameter_types)
                    else None
                )
                for ref in self.return_resource_alias_field_paths_for_relative_path(
                    expr.args[alias_index],
                    param_indices,
                    next_relative_path,
                    arg_type,
                ):
                    if ref not in refs:
                        refs.append(ref)
            if refs:
                return refs

        return []

    def return_resource_static_alias_candidates(
        self, expr, param_indices, target_type=None
    ):
        if isinstance(expr, list):
            return []

        root_name = self.resource_access_root_name(expr)
        if root_name in param_indices:
            return []

        if isinstance(expr, TernaryOpNode):
            candidates = []
            for branch in (expr.true_expr, expr.false_expr):
                candidates.extend(
                    self.return_resource_static_alias_candidates(
                        branch, param_indices, target_type=target_type
                    )
                )
            return self.unique_resource_access_qualifiers(candidates)

        if isinstance(expr, MatchNode):
            candidates = []
            for arm in getattr(expr, "arms", []) or []:
                try:
                    arm_expr, _ = self.match_arm_value_expression(arm)
                except ValueError:
                    continue
                candidates.extend(
                    self.return_resource_static_alias_candidates(
                        arm_expr, param_indices, target_type=target_type
                    )
                )
            return self.unique_resource_access_qualifiers(candidates)

        if isinstance(expr, ArrayLiteralNode):
            element_type = self.array_element_type(target_type)
            candidates = []
            for element in getattr(expr, "elements", []) or []:
                candidates.extend(
                    self.return_resource_static_alias_candidates(
                        element, param_indices, target_type=element_type
                    )
                )
            return self.unique_resource_access_qualifiers(candidates)

        if isinstance(expr, FunctionCallNode):
            return self.return_resource_static_alias_candidates_from_call(
                expr, param_indices, target_type=target_type
            )

        candidates = []
        access, resource_name = self.direct_resource_access_qualifier(expr)
        if access is not None and resource_name is not None:
            candidates.append((access, resource_name))
        if root_name is not None:
            candidates.extend(self.resource_access_qualifier_aliases.get(root_name, []))
        return self.unique_resource_access_qualifiers(candidates)

    def return_resource_static_field_alias_candidates(
        self, expr, param_indices, target_type=None
    ):
        aliases = {}
        for path, candidates in self.resource_access_field_alias_candidates(
            expr, target_type=target_type
        ).items():
            for candidate in candidates:
                if self.resource_access_candidate_references_param(
                    candidate, param_indices
                ):
                    continue
                aliases.setdefault(path, [])
                if candidate not in aliases[path]:
                    aliases[path].append(candidate)
        return aliases

    def resource_access_candidate_references_param(self, candidate, param_indices):
        _, resource_name = candidate
        if not isinstance(resource_name, str):
            return False
        match = re.match(r"\s*([A-Za-z_]\w*)", resource_name)
        return match is not None and match.group(1) in param_indices

    def return_resource_static_alias_candidates_from_call(
        self, expr, param_indices, target_type=None
    ):
        func_name = self.function_call_name(expr)
        fields = self.generic_enum_variant_fields_for_type(func_name, target_type)
        if fields is not None and len(fields) == len(expr.args):
            candidates = []
            for arg, (_field_name, field_type) in zip(expr.args, fields):
                if not self.resource_access_aliasable_type(field_type):
                    continue
                candidates.extend(
                    self.return_resource_static_alias_candidates(
                        arg, param_indices, target_type=field_type
                    )
                )
            return self.unique_resource_access_qualifiers(candidates)

        candidates = list(
            self.function_return_resource_static_aliases.get(func_name, [])
        )
        alias_indices = self.function_return_resource_aliases.get(func_name) or []
        parameter_types = self.function_parameter_types.get(func_name, [])
        for alias_index in alias_indices:
            if alias_index >= len(expr.args):
                continue
            alias_type = (
                parameter_types[alias_index]
                if alias_index < len(parameter_types)
                else None
            )
            candidates.extend(
                self.return_resource_static_alias_candidates(
                    expr.args[alias_index], param_indices, target_type=alias_type
                )
            )
        return self.unique_resource_access_qualifiers(candidates)

    def is_user_defined_function(self, func_name):
        return isinstance(func_name, str) and func_name in self.function_return_types

    def function_parameter_type_name(self, param):
        if hasattr(param, "param_type"):
            return self.explicit_image_format_resource_type(
                self.convert_type_node_to_string(param.param_type), param
            )
        if hasattr(param, "vtype"):
            return self.explicit_image_format_resource_type(param.vtype, param)
        return "float"

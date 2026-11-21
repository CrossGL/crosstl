"""Texture usage analysis in the HLSL to CrossGL converter: direct, parameter and member texture uses, their propagation through functions, shadow textures and the variable type map."""

from ...common_ast import (
    ArrayAccessNode,
    TextureSampleNode,
)
from ..DirectxAst import (
    FunctionCallNode,
    MemberAccessNode,
    StructNode,
    VariableNode,
)
from ..raw_gather_textures import raw_gather_texture_type


class TextureUsageMixin:
    """Texture usage members of ``HLSLToCrossGLConverter``."""

    def iter_ast_children(self, node):
        if node is None or isinstance(node, (str, int, float, bool)):
            return
        if isinstance(node, dict):
            yield from node.values()
            return
        if isinstance(node, (list, tuple, set)):
            yield from node
            return
        yield from getattr(node, "__dict__", {}).values()

    def collect_direct_texture_method_usage_names(self, root):
        comparison_names = set()
        regular_names = set()

        def visit(node):
            if node is None or isinstance(node, (str, int, float, bool)):
                return
            if isinstance(node, TextureSampleNode):
                texture_name = self.expression_base_name(node.texture)
                if texture_name:
                    regular_names.add(texture_name)
            if isinstance(node, FunctionCallNode) and isinstance(
                node.name, MemberAccessNode
            ):
                texture_name = self.expression_base_name(node.name.object)
                descriptor = self.resource_method_descriptor(
                    node.name.member, len(getattr(node, "args", []) or [])
                )
                usage = (
                    descriptor["usage"]
                    if descriptor and descriptor["resource"] == "texture"
                    else None
                )
                if usage == "comparison":
                    if texture_name:
                        comparison_names.add(texture_name)
                elif usage == "regular":
                    if texture_name:
                        regular_names.add(texture_name)
            for child in self.iter_ast_children(node):
                visit(child)

        visit(root)
        return comparison_names, regular_names

    def collect_function_texture_parameter_usage(self, root):
        function_usage = {}
        for func in getattr(root, "functions", []) or []:
            param_by_name = {
                getattr(param, "name", None): index
                for index, param in enumerate(getattr(func, "params", []) or [])
            }
            param_by_name.pop(None, None)
            comparison_names, regular_names = (
                self.collect_direct_texture_method_usage_names(
                    getattr(func, "body", [])
                )
            )
            function_usage[getattr(func, "name", None)] = {
                "comparison": {
                    param_by_name[name]
                    for name in comparison_names
                    if name in param_by_name
                },
                "regular": {
                    param_by_name[name]
                    for name in regular_names
                    if name in param_by_name
                },
            }
        function_usage.pop(None, None)
        return function_usage

    def collect_nonparameter_texture_usage_names(self, root):
        comparison_names = set()
        regular_names = set()

        for node in getattr(root, "global_variables", []) or []:
            direct_comparison, direct_regular = (
                self.collect_direct_texture_method_usage_names(node)
            )
            comparison_names.update(direct_comparison)
            regular_names.update(direct_regular)

        for func in getattr(root, "functions", []) or []:
            param_names = {
                getattr(param, "name", None)
                for param in getattr(func, "params", []) or []
            }
            param_names.discard(None)
            direct_comparison, direct_regular = (
                self.collect_direct_texture_method_usage_names(
                    getattr(func, "body", [])
                )
            )
            comparison_names.update(direct_comparison - param_names)
            regular_names.update(direct_regular - param_names)

        return comparison_names, regular_names

    def collect_variable_types_from_nodes(self, nodes, type_map):
        for node in nodes or []:
            if isinstance(node, VariableNode) and getattr(node, "name", None):
                type_map[node.name] = getattr(node, "vtype", None)
            for child in self.iter_ast_children(node):
                self.collect_variable_types_from_nodes([child], type_map)

    def expression_raw_type_from_map(self, expr, variable_types):
        if isinstance(expr, MemberAccessNode):
            object_type = self.expression_raw_type_from_map(expr.object, variable_types)
            member_type = self.member_access_raw_type(object_type, expr.member)
            return member_type if member_type is not None else object_type
        if isinstance(expr, ArrayAccessNode):
            return self.expression_raw_type_from_map(expr.array, variable_types)
        if isinstance(expr, VariableNode):
            return variable_types.get(expr.name)
        if isinstance(expr, str):
            return variable_types.get(expr)
        return None

    def expression_struct_member_key(self, expr, variable_types):
        while isinstance(expr, ArrayAccessNode):
            expr = expr.array
        if not isinstance(expr, MemberAccessNode):
            return None
        owner_type = self.expression_raw_type_from_map(expr.object, variable_types)
        owner_base = self.raw_type_base(owner_type)
        if owner_base not in self.struct_member_types:
            return None
        return owner_base, expr.member

    def collect_direct_texture_member_usage_keys(self, root, variable_types):
        comparison_keys = set()
        regular_keys = set()

        def visit(node):
            if node is None or isinstance(node, (str, int, float, bool)):
                return
            if isinstance(node, FunctionCallNode) and isinstance(
                node.name, MemberAccessNode
            ):
                resource_type = self.expression_raw_type_from_map(
                    node.name.object, variable_types
                )
                descriptor = self.resource_method_descriptor(
                    node.name.member,
                    len(getattr(node, "args", []) or []),
                    resource_type,
                )
                usage = (
                    descriptor["usage"]
                    if descriptor and descriptor["resource"] == "texture"
                    else None
                )
                member_key = self.expression_struct_member_key(
                    node.name.object, variable_types
                )
                if member_key and usage == "comparison":
                    comparison_keys.add(member_key)
                elif member_key and usage == "regular":
                    regular_keys.add(member_key)
            for child in self.iter_ast_children(node):
                visit(child)

        visit(root)
        return comparison_keys, regular_keys

    def collect_texture_member_usage_keys(self, root):
        comparison_keys = set()
        regular_keys = set()
        global_variable_types = {
            getattr(node, "name", None): getattr(node, "vtype", None)
            for node in getattr(root, "global_variables", []) or []
        }
        global_variable_types.pop(None, None)

        for node in getattr(root, "global_variables", []) or []:
            direct_comparison, direct_regular = (
                self.collect_direct_texture_member_usage_keys(
                    node, global_variable_types
                )
            )
            comparison_keys.update(direct_comparison)
            regular_keys.update(direct_regular)

        for func in getattr(root, "functions", []) or []:
            function_variable_types = dict(global_variable_types)
            for param in getattr(func, "params", []) or []:
                if getattr(param, "name", None):
                    function_variable_types[param.name] = getattr(param, "vtype", None)
            self.collect_variable_types_from_nodes(
                getattr(func, "body", []), function_variable_types
            )
            direct_comparison, direct_regular = (
                self.collect_direct_texture_member_usage_keys(
                    getattr(func, "body", []), function_variable_types
                )
            )
            comparison_keys.update(direct_comparison)
            regular_keys.update(direct_regular)

        return comparison_keys, regular_keys

    def function_variable_type_maps(self, root):
        global_variable_types = {
            getattr(node, "name", None): getattr(node, "vtype", None)
            for node in getattr(root, "global_variables", []) or []
        }
        global_variable_types.pop(None, None)
        type_maps = {}

        for func in getattr(root, "functions", []) or []:
            function_variable_types = dict(global_variable_types)
            for param in getattr(func, "params", []) or []:
                if getattr(param, "name", None):
                    function_variable_types[param.name] = getattr(param, "vtype", None)
            self.collect_variable_types_from_nodes(
                getattr(func, "body", []), function_variable_types
            )
            type_maps[getattr(func, "name", None)] = function_variable_types

        type_maps.pop(None, None)
        return type_maps

    def propagate_function_texture_usage(
        self, root, function_usage, comparison_names, regular_names
    ):
        changed = False

        def apply_call_usage(node, caller_function_name, caller_param_by_name):
            nonlocal changed
            if not isinstance(node, FunctionCallNode) or not isinstance(node.name, str):
                return
            callee_usage = function_usage.get(node.name)
            if not callee_usage:
                return
            for usage_kind, usage_names in (
                ("comparison", comparison_names),
                ("regular", regular_names),
            ):
                for param_index in callee_usage[usage_kind]:
                    if param_index >= len(node.args):
                        continue
                    arg_name = self.expression_base_name(node.args[param_index])
                    if not arg_name:
                        continue
                    caller_param_index = caller_param_by_name.get(arg_name)
                    if caller_param_index is not None:
                        caller_usage = function_usage.get(caller_function_name)
                        if (
                            caller_usage is not None
                            and caller_param_index not in caller_usage[usage_kind]
                        ):
                            caller_usage[usage_kind].add(caller_param_index)
                            changed = True
                    elif arg_name not in usage_names:
                        usage_names.add(arg_name)
                        changed = True

        def visit(node, caller_function_name, caller_param_by_name):
            if node is None or isinstance(node, (str, int, float, bool)):
                return
            apply_call_usage(node, caller_function_name, caller_param_by_name)
            for child in self.iter_ast_children(node):
                visit(child, caller_function_name, caller_param_by_name)

        for func in getattr(root, "functions", []) or []:
            param_by_name = {
                getattr(param, "name", None): index
                for index, param in enumerate(getattr(func, "params", []) or [])
            }
            param_by_name.pop(None, None)
            visit(getattr(func, "body", []), getattr(func, "name", None), param_by_name)

        return changed

    def propagate_function_texture_member_usage(
        self, root, function_usage, member_comparison_keys, member_regular_keys
    ):
        type_maps = self.function_variable_type_maps(root)
        changed = False

        def apply_call_usage(node, caller_function_name):
            nonlocal changed
            if not isinstance(node, FunctionCallNode) or not isinstance(node.name, str):
                return
            callee_usage = function_usage.get(node.name)
            if not callee_usage:
                return
            variable_types = type_maps.get(caller_function_name, {})
            for usage_kind, usage_keys in (
                ("comparison", member_comparison_keys),
                ("regular", member_regular_keys),
            ):
                for param_index in callee_usage[usage_kind]:
                    if param_index >= len(node.args):
                        continue
                    member_key = self.expression_struct_member_key(
                        node.args[param_index], variable_types
                    )
                    if member_key and member_key not in usage_keys:
                        usage_keys.add(member_key)
                        changed = True

        def visit(node, caller_function_name):
            if node is None or isinstance(node, (str, int, float, bool)):
                return
            apply_call_usage(node, caller_function_name)
            for child in self.iter_ast_children(node):
                visit(child, caller_function_name)

        for func in getattr(root, "functions", []) or []:
            visit(getattr(func, "body", []), getattr(func, "name", None))

        return changed

    def collect_shadow_texture_names(self, root):
        comparison_names, regular_names = self.collect_nonparameter_texture_usage_names(
            root
        )
        member_comparison_keys, member_regular_keys = (
            self.collect_texture_member_usage_keys(root)
        )
        function_usage = self.collect_function_texture_parameter_usage(root)
        while self.propagate_function_texture_usage(
            root, function_usage, comparison_names, regular_names
        ):
            pass
        while self.propagate_function_texture_member_usage(
            root, function_usage, member_comparison_keys, member_regular_keys
        ):
            pass

        shadow_names = comparison_names - regular_names
        self.shadow_texture_declaration_ids = set()
        for node in getattr(root, "global_variables", []) or []:
            if getattr(node, "name", None) in shadow_names:
                self.shadow_texture_declaration_ids.add(id(node))

        for func in getattr(root, "functions", []) or []:
            usage = function_usage.get(getattr(func, "name", None), {})
            shadow_indices = usage.get("comparison", set()) - usage.get(
                "regular", set()
            )
            for index, param in enumerate(getattr(func, "params", []) or []):
                if index in shadow_indices:
                    self.shadow_texture_declaration_ids.add(id(param))

        shadow_member_keys = member_comparison_keys - member_regular_keys
        for struct in getattr(root, "structs", []) or []:
            if not isinstance(struct, StructNode):
                continue
            for member in getattr(struct, "members", []) or []:
                if (struct.name, getattr(member, "name", None)) in shadow_member_keys:
                    self.shadow_texture_declaration_ids.add(id(member))

        return shadow_names

    def map_variable_type(self, node):
        legacy_texture_type = self.legacy_bound_texture_types.get(
            getattr(node, "name", None)
        )
        if legacy_texture_type:
            return legacy_texture_type

        hlsl_type = getattr(node, "vtype", None)
        if id(node) in self.raw_gather_texture_declaration_ids:
            raw_gather_type = raw_gather_texture_type(self, hlsl_type)
            if raw_gather_type is not None:
                return raw_gather_type
        if id(node) not in self.shadow_texture_declaration_ids:
            return self.map_type(hlsl_type)
        type_name = hlsl_type
        if type_name and "<" in type_name and type_name.endswith(">"):
            type_name = type_name.split("<", 1)[0]
        return self.shadow_texture_type_map.get(type_name, self.map_type(hlsl_type))

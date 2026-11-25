"""Resource object type hints in the CUDA to CrossGL converter: global, local and struct-member hints and the declared names and types they are collected from."""

from ..CudaAst import (
    ArrayAccessNode,
    CastNode,
    MemberAccessNode,
    UnaryOpNode,
    VariableNode,
)


class CudaImportResourceHintMixin:
    """Resource hint members of ``CudaToCrossGLConverter``."""

    def add_resource_object_type_hint(self, hints, name, resource_type):
        if not name or not resource_type:
            return
        if name in hints and hints[name] != resource_type:
            hints[name] = None
            return
        hints[name] = resource_type

    def collect_global_resource_object_type_hints(self, node):
        hints = {}
        global_names = self.collect_global_declared_variable_names(node)
        for function in getattr(node, "functions", []):
            self.collect_global_resource_object_type_hints_from_function(
                function, global_names, hints
            )
        for kernel in getattr(node, "kernels", []):
            self.collect_global_resource_object_type_hints_from_function(
                kernel, global_names, hints
            )
        return hints

    def collect_global_declared_variable_names(self, node):
        return {
            var.name
            for var in getattr(node, "global_variables", [])
            if isinstance(var, VariableNode)
        }

    def collect_global_declared_variable_types(self, node):
        return {
            var.name: var.vtype
            for var in getattr(node, "global_variables", [])
            if isinstance(var, VariableNode)
        }

    def collect_struct_member_types(self, node):
        return {
            struct.name: {
                member.name: member.vtype
                for member in getattr(struct, "members", [])
                if isinstance(member, VariableNode)
            }
            for struct in getattr(node, "structs", [])
            if struct.name
        }

    def collect_declared_variable_types(self, node):
        types = {
            param.name: param.vtype
            for param in getattr(node, "params", [])
            if isinstance(param, VariableNode)
        }

        for current in self.walk_ast_values(getattr(node, "body", [])):
            if isinstance(current, VariableNode):
                types[current.name] = current.vtype
        return types

    def add_struct_resource_member_hint(
        self, hints, struct_name, member_name, resource_type
    ):
        key = (struct_name, member_name)
        if key in hints and hints[key] != resource_type:
            hints[key] = None
            return
        hints[key] = resource_type

    def resource_member_access_target(self, expression):
        if isinstance(expression, MemberAccessNode):
            return expression
        if isinstance(expression, ArrayAccessNode):
            return self.resource_member_access_target(expression.array)
        if isinstance(expression, CastNode):
            return self.resource_member_access_target(expression.expression)
        if isinstance(expression, UnaryOpNode):
            return self.resource_member_access_target(expression.operand)
        return None

    def struct_type_for_resource_member_object(
        self, expression, variable_types, struct_names, struct_member_types
    ):
        if isinstance(expression, str):
            return self.normalized_struct_type_name(
                variable_types.get(expression), struct_names
            )
        if isinstance(expression, ArrayAccessNode):
            return self.struct_type_for_resource_member_object(
                expression.array, variable_types, struct_names, struct_member_types
            )
        if isinstance(expression, CastNode):
            cast_struct = self.normalized_struct_type_name(
                expression.target_type, struct_names
            )
            if cast_struct is not None:
                return cast_struct
            return self.struct_type_for_resource_member_object(
                expression.expression, variable_types, struct_names, struct_member_types
            )
        if isinstance(expression, UnaryOpNode):
            return self.struct_type_for_resource_member_object(
                expression.operand, variable_types, struct_names, struct_member_types
            )
        if isinstance(expression, MemberAccessNode):
            owner_struct = self.struct_type_for_resource_member_object(
                expression.object,
                variable_types,
                struct_names,
                struct_member_types,
            )
            if owner_struct is None:
                return None
            member_type = struct_member_types.get(owner_struct, {}).get(
                expression.member
            )
            return self.normalized_struct_type_name(member_type, struct_names)
        return None

    def normalized_struct_type_name(self, type_name, struct_names):
        if not type_name:
            return None
        type_name = self.strip_type_qualifiers(type_name)
        type_name = type_name.split("[", 1)[0].replace("*", "").strip()
        return type_name if type_name in struct_names else None

    def is_user_defined_function(self, func_name):
        return isinstance(func_name, str) and func_name in self.user_function_names

    def is_cuda_intrinsic_override_function(self, func_name):
        if not isinstance(func_name, str):
            return False
        normalized_name = func_name[2:] if func_name.startswith("::") else func_name
        return normalized_name in self.CUDA_INTRINSIC_OVERRIDE_NAMES

    def push_resource_object_hint_scope(self, hints):
        self.resource_object_hint_scopes.append(hints)

    def pop_resource_object_hint_scope(self):
        if self.resource_object_hint_scopes:
            self.resource_object_hint_scopes.pop()

    def lookup_resource_object_type_hint(self, name):
        for scope in reversed(self.resource_object_hint_scopes):
            if name in scope:
                return scope[name]
        return self.global_resource_object_type_hints.get(name)

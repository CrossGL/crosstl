"""Identifier metadata and value types in the WGSL generator: struct and resource paths, scalar, vector and matrix shapes, and resource aliases."""

import re

from ..ast import (
    ArrayAccessNode,
    ArrayType,
    IdentifierNode,
    MemberAccessNode,
    NamedType,
    VectorType,
)
from ..refusals import WGSLRefusal
from .wgsl_read_only_globals import validate_writable_module_root


class WGSLValueTypeMixin:
    """Value type members of ``WGSLCodeGen``."""

    def collect_identifier_metadata(
        self,
        structs,
        cbuffers,
        constants,
        global_variables,
        stage_resource_parameters,
        helper_functions,
    ):
        type_names = [
            getattr(node, "name", "") for node in list(structs) + list(cbuffers)
        ]
        module_names = [
            getattr(node, "name", "")
            for node in (
                list(constants)
                + list(global_variables)
                + list(stage_resource_parameters)
            )
        ]
        function_groups = self.function_groups_by_name(helper_functions)
        self._function_overloads_by_name = function_groups
        function_names = []
        function_name_keys = {}
        used_function_names = {name for name in type_names + module_names if name}
        for function_name, functions in function_groups.items():
            if len(functions) == 1:
                emitted_name = self.unique_function_module_name(
                    function_name, used_function_names
                )
                function_names.append(emitted_name)
                function_name_keys[self.function_signature_key(functions[0])] = (
                    emitted_name
                )
                continue
            for function in functions:
                overload_name = (
                    f"{function_name}_{self.function_overload_suffix(function)}"
                )
                emitted_name = self.unique_function_module_name(
                    overload_name, used_function_names
                )
                function_names.append(emitted_name)
                function_name_keys[self.function_signature_key(function)] = emitted_name
        module_scope_names = self.wgsl_identifier_map(
            type_names + function_names + module_names
        )
        self._type_identifier_names = {
            name: module_scope_names[name] for name in type_names if name
        }
        self._function_identifier_names = {
            name: module_scope_names[name] for name in function_names if name
        }
        self._function_signature_identifier_names = {
            signature_key: module_scope_names[function_name]
            for signature_key, function_name in function_name_keys.items()
            if function_name
        }
        self._module_identifier_names = {
            name: module_scope_names[name] for name in module_names if name
        }
        self._struct_member_identifier_names = {}
        for struct in structs:
            self.register_struct_member_identifier_metadata(struct)

    def constant_literal_values(self, constants):
        values = {}
        for constant in constants:
            name = getattr(constant, "name", "")
            value = getattr(getattr(constant, "value", None), "value", None)
            if not name:
                continue
            try:
                values[name] = int(value)
            except (TypeError, ValueError):
                continue
        return values

    def collect_cbuffer_member_identifier_metadata(self, cbuffers):
        for cbuffer in cbuffers:
            self.register_struct_member_identifier_metadata(cbuffer)

    def register_struct_member_identifier_metadata(self, struct):
        struct_name = getattr(struct, "name", "")
        member_names = [
            getattr(member, "name", "")
            for member in getattr(struct, "members", []) or []
            if getattr(member, "name", "")
        ]
        member_map = self.wgsl_identifier_map(member_names)
        for source_name, emitted_name in member_map.items():
            self._struct_member_identifier_names[(struct_name, source_name)] = (
                emitted_name
            )

    def collect_struct_type_metadata(self, structs):
        self._structs_by_name = {struct.name: struct for struct in structs}
        self._struct_member_types = {}
        self._struct_resource_paths = {}
        for struct in structs:
            for member in getattr(struct, "members", []) or []:
                if not hasattr(member, "member_type"):
                    continue
                self._struct_member_types[(struct.name, member.name)] = (
                    member.member_type
                )
        for struct in structs:
            self._struct_resource_paths[struct.name] = tuple(
                self.resource_paths_for_type(NamedType(struct.name))
            )

    def resource_paths_for_type(self, vtype, prefix=(), seen=()):
        vtype = self.array_element_type(vtype)
        struct_name = self.struct_type_name(vtype)
        if not struct_name or struct_name in seen:
            return ()
        struct = self._structs_by_name.get(struct_name)
        if struct is None:
            return ()
        if getattr(struct, "generic_params", None):
            return ()

        paths = []
        for member in getattr(struct, "members", []) or []:
            if not hasattr(member, "member_type"):
                continue
            member_type = getattr(member, "member_type", None)
            member_path = tuple(prefix) + (member.name,)
            resource_type_name = self.struct_member_resource_type_name(member)
            if resource_type_name:
                if not self.supported_struct_resource_member(member):
                    continue
                info = {
                    "owner_struct": struct.name,
                    "member": member,
                    "member_name": member.name,
                    "member_type": member_type,
                    "path": member_path,
                    "resource_type_name": resource_type_name,
                }
                paths.append(info)
                continue
            paths.extend(
                self.resource_paths_for_type(
                    member_type, prefix=member_path, seen=tuple(seen) + (struct_name,)
                )
            )
        return tuple(paths)

    def struct_type_name(self, vtype):
        option_type = self.builtin_option_specialized_type_name(vtype)
        if option_type is not None:
            return option_type
        if isinstance(vtype, NamedType) and not vtype.generic_args:
            return str(vtype.name)
        if (
            isinstance(vtype, NamedType)
            and vtype.generic_args
            and vtype.name in self._generic_struct_definitions
        ):
            return self.generic_struct_specialized_type_name(vtype)
        if isinstance(vtype, str):
            return vtype
        return None

    def array_element_type(self, vtype):
        while isinstance(vtype, ArrayType):
            vtype = vtype.element_type
        return vtype

    def resource_member_binding_name(self, root_name, path):
        return "_".join((root_name, *path))

    def resource_member_binding_for_access(self, expr):
        access_path = self.expression_access_path(expr)
        if access_path is None:
            return None
        root_name, path = access_path
        if not path:
            return None

        alias = self.resource_alias_binding(root_name, path)
        if alias is not None:
            return alias

        binding = self._module_resource_bindings.get((root_name, tuple(path)))
        if binding is not None:
            return binding
        return None

    def expression_access_path(self, expr):
        if isinstance(expr, IdentifierNode):
            return expr.name, ()
        if isinstance(expr, MemberAccessNode):
            parent = self.expression_access_path(expr.object_expr)
            if parent is None:
                return None
            root_name, path = parent
            return root_name, tuple(path) + (expr.member,)
        if isinstance(expr, ArrayAccessNode):
            return self.expression_access_path(expr.array_expr)
        return None

    def _common_value_type(self, left_type, right_type):
        """The type two WGSL bitwise operands or select branches share, or None."""
        left_vector = self.vector_shape(left_type)
        right_vector = self.vector_shape(right_type)
        left_scalar = self.scalar_type_name(left_type)
        right_scalar = self.scalar_type_name(right_type)
        if left_vector is not None and left_vector == right_vector:
            return left_type
        if left_vector is not None and right_scalar is not None:
            return left_type
        if right_vector is not None and left_scalar is not None:
            return right_type
        if left_scalar is not None and left_scalar == right_scalar:
            return left_scalar
        return None

    def scalar_type_name(self, vtype):
        try:
            type_name = self.type_name_string(vtype) if vtype is not None else None
        except ValueError:
            return None
        if type_name in {"bool", "f32", "i32", "u32"}:
            return type_name
        return None

    def is_bool_type(self, vtype):
        return self.scalar_type_name(vtype) == "bool"

    def uniform_bool_member_type(self, expr):
        """The ``bool`` or ``vecN<bool>`` type of a uniform block member that ``expr`` accesses, else ``None``."""
        if not isinstance(expr, MemberAccessNode):
            return None
        object_type = self.array_element_type(self.expression_type(expr.object_expr))
        struct_name = self.struct_type_name(object_type)
        if struct_name not in self._uniform_buffer_struct_names:
            return None
        member_type = self._struct_member_types.get((struct_name, expr.member))
        return member_type if self.uniform_bool_lanes(member_type) is not None else None

    def uniform_bool_lanes(self, vtype):
        """The lane count of a ``bool`` or ``vecN<bool>`` type, which a uniform block stores as ``u32`` lanes.

        WGSL's ``bool`` is not host-shareable (WGSL section 6.2.1), so a uniform block member of a boolean type
        is a ``u32`` or ``vecN<u32>``, as std140 lays out GLSL's ``bool`` and ``bvecN``. Returns ``None`` for
        any other type.
        """
        if self.is_bool_type(vtype):
            return 1
        shape = self.vector_shape(vtype)
        if shape is not None and shape[0] == "bool":
            return shape[1]
        return None

    def uniform_bool_value(self, access, vtype):
        """The boolean value of uniform member ``access``, which holds ``u32`` lanes, or ``None``."""
        lanes = self.uniform_bool_lanes(vtype)
        if lanes is None:
            return None
        if lanes == 1:
            return f"({access} != 0u)"
        return f"vec{lanes}<bool>({access})"

    def vector_shape(self, vtype):
        try:
            type_name = self.type_name_string(vtype) if vtype is not None else None
        except ValueError:
            return None
        if type_name is None:
            return None
        match = re.fullmatch(r"vec([234])<([^>]+)>", type_name)
        if match:
            return match.group(2), int(match.group(1))
        return None

    def matrix_shape(self, vtype):
        try:
            type_name = self.type_name_string(vtype) if vtype is not None else None
        except ValueError:
            return None
        if type_name is None:
            return None
        match = re.fullmatch(r"mat([234])x([234])<([^>]+)>", type_name)
        if match:
            return match.group(3), int(match.group(1)), int(match.group(2))
        return None

    def integer_scalar_type(self, vtype):
        try:
            type_name = self.type_name_string(vtype) if vtype is not None else None
        except ValueError:
            return None
        if type_name in {"i32", "u32"}:
            return type_name
        return None

    def vector_component_type(self, vtype):
        if isinstance(vtype, VectorType):
            return self.type_name_string(vtype.element_type)
        if isinstance(vtype, str):
            match = re.fullmatch(r"vec[234]<([^>]+)>", self.map_type_name(vtype))
            if match:
                return match.group(1)
        return None

    def vector_component_count(self, vtype):
        if isinstance(vtype, VectorType):
            return vtype.size
        if isinstance(vtype, str):
            match = re.fullmatch(r"vec([234])<[^>]+>", self.map_type_name(vtype))
            if match:
                return int(match.group(1))
        return None

    def vector_type_name(self, vtype):
        if self.vector_component_count(vtype) is None:
            return None
        if isinstance(vtype, str):
            return self.map_type_name(vtype)
        return self.type_name_string(vtype)

    def is_vector_member(self, member):
        return bool(member) and all(component in "xyzwrgbastpq" for component in member)

    def value_type(self, name):
        for scope in reversed(self._value_type_scopes):
            if name in scope:
                return scope[name]
        return None

    def register_value_type(self, name, vtype):
        if self._value_type_scopes and name:
            self._value_type_scopes[-1][name] = vtype

    def register_parameter_value_types(self, function):
        for parameter in getattr(function, "parameters", []) or []:
            value_type = (
                self.stage_resource_module_type(parameter)
                if self.is_stage_resource_parameter(parameter)
                else parameter.param_type
            )
            self.register_value_type(parameter.name, value_type)
            self.register_parameter_resource_aliases(parameter)

    def register_parameter_resource_aliases(self, parameter):
        for info in self.resource_paths_for_type(parameter.param_type):
            self.register_resource_alias(
                parameter.name,
                info["path"],
                {
                    **info,
                    "binding_name": self.resource_member_binding_name(
                        parameter.name, info["path"]
                    ),
                },
            )

    def register_resource_aliases(self, name, vtype, initializer=None):
        if initializer is None:
            return
        initializer_access = self.expression_access_path(initializer)
        if initializer_access is None:
            return
        _initializer_root, initializer_path = initializer_access
        for info in self.resource_paths_for_type(vtype):
            binding = self.resource_binding_for_access_path(
                initializer_access[0], tuple(initializer_path) + tuple(info["path"])
            )
            if binding is not None:
                self.register_resource_alias(name, info["path"], binding)

    def register_resource_alias(self, root_name, path, binding):
        if self._resource_alias_scopes and root_name:
            self._resource_alias_scopes[-1].setdefault(root_name, {})[
                tuple(path)
            ] = binding

    def resource_alias_binding(self, root_name, path):
        for scope in reversed(self._resource_alias_scopes):
            binding = scope.get(root_name, {}).get(tuple(path))
            if binding is not None:
                return binding
        return None

    def resource_binding_for_access_path(self, root_name, path):
        alias = self.resource_alias_binding(root_name, path)
        if alias is not None:
            return alias
        return self._module_resource_bindings.get((root_name, tuple(path)))

    def storage_access_mode(self, name):
        return self._module_storage_access_modes.get(name)

    def validate_storage_assignment_target(self, target):
        validate_writable_module_root(self, target)
        access_path = self.expression_access_path(target)
        if access_path is None:
            return
        root_name, _path = access_path
        if self.storage_access_mode(root_name) == "read":
            raise WGSLRefusal(
                "WGSL target cannot write read-only GLSL buffer block resource "
                f"{root_name}"
            )

    def resource_member_parameter_declarations(self, root_name, root_type):
        declarations = []
        for info in self.resource_paths_for_type(root_type):
            binding_name = self.resource_member_binding_name(root_name, info["path"])
            sampled_texture_type = self.sampled_texture_type(info["member_type"])
            if sampled_texture_type is not None:
                declarations.append(f"{binding_name}: {sampled_texture_type}")
                declarations.append(
                    f"{self.texture_sampler_name(binding_name)}: "
                    f"{self.companion_sampler_type(info['member_type'])}"
                )
            elif self.is_sampler_type(info["member_type"]):
                declarations.append(
                    f"{binding_name}: {self.sampler_type(info['member_type'])}"
                )
        return declarations

    def resource_member_call_arguments(self, arg, resource_paths):
        arguments = []
        access = self.expression_access_path(arg)
        if access is None:
            raise WGSLRefusal(
                "WGSL target cannot forward resource-bearing struct expression "
                f"{self.generate_expression(arg)}; pass a named struct value"
            )
        root_name, root_path = access
        for info in resource_paths:
            binding = self.resource_binding_for_access_path(
                root_name, tuple(root_path) + tuple(info["path"])
            )
            if binding is None:
                raise WGSLRefusal(
                    "WGSL target cannot forward resource member "
                    f"{'.'.join((root_name, *root_path, *info['path']))}; "
                    "no module binding or resource alias is available"
                )
            arguments.append(binding["binding_name"])
            if self.sampled_texture_type(info["member_type"]) is not None:
                arguments.append(self.texture_sampler_name(binding["binding_name"]))
        return arguments

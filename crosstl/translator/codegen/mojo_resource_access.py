"""Resource access metadata and alias tracking in the Mojo generator: read and write qualifiers, path aliases and buffer value types."""

import re

from ..ast import (
    ArrayAccessNode,
    ArrayLiteralNode,
    ConstructorNode,
    FunctionCallNode,
    MatchNode,
    MemberAccessNode,
    TernaryOpNode,
    VariableNode,
)
from ..refusals import MojoRefusal
from .enum_utils import generic_enum_specialized_fields
from .mojo_tables import (
    MOJO_BYTE_ADDRESS_BUFFER_TYPES,
    MOJO_CONCRETE_RESOURCE_TYPES,
    MOJO_DTYPE_INFO,
)


class MojoResourceAccessMixin:
    """Resource-access members of ``MojoCodeGen``."""

    def generate_resource_metadata_comment(self, node, type_name, kind=None):
        name = getattr(node, "name", None)
        resource_kind = self.mojo_resource_kind(type_name, kind, node)
        if not name or resource_kind is None:
            return ""

        self.register_resource_access_metadata(node, type_name)
        _, count = self.resource_base_type_and_count(type_name)
        namespace = self.mojo_resource_binding_namespace_with_class(resource_kind, node)
        set_index = self.explicit_resource_set_index(node)
        binding = self.explicit_resource_binding_index(node)
        source_binding = None
        if binding is None:
            binding = self.next_available_resource_binding(namespace, set_index, count)
            binding_source = "automatic"
            self.reserve_resource_binding(namespace, set_index, binding, count, name)
        else:
            binding_source = "explicit"
            source_binding = binding
            uses_source_descriptor = self.uses_source_descriptor_array_binding(
                node, count, resource_kind
            )
            if self.explicit_binding_is_target_native(node, count, resource_kind):
                self.reserve_resource_binding(
                    namespace, set_index, binding, count, name
                )
            else:
                self.reserve_source_resource_binding(
                    namespace, set_index, source_binding, 1, name
                )
                if self.resource_binding_family_conflicts(
                    namespace, set_index, binding, count
                ):
                    binding = self.next_available_resource_binding(
                        namespace, set_index, count
                    )
                self.reserve_resource_binding(
                    namespace, set_index, binding, count, name
                )
            if not uses_source_descriptor:
                source_binding = None

        parts = [
            "# CrossGL resource metadata:",
            f"name={name}",
            f"kind={resource_kind}",
            f"set={set_index}",
            f"binding={binding}",
            f"binding_source={binding_source}",
        ]
        if count != 1:
            parts.append(f"count={count}")
        if source_binding is not None and source_binding != binding:
            parts.append(f"source_binding={source_binding}")
        register_metadata = self.resource_register_metadata(node)
        if register_metadata:
            parts.append(f"register={register_metadata}")
        layout = self.glsl_buffer_block_layout(node)
        if layout:
            parts.append(f"layout={layout}")
        access = self.normalized_resource_access_metadata(node)
        if access:
            parts.append(f"access={access}")
        memory_qualifiers = self.resource_memory_qualifiers(node)
        if memory_qualifiers:
            parts.append(f"memory={','.join(memory_qualifiers)}")
        self.mojo_resource_binding_initializers[name] = {
            "kind": resource_kind,
            "set": set_index,
            "binding": binding,
        }
        return " ".join(parts) + "\n"

    def register_resource_access_metadata(self, node, type_name):
        name = getattr(node, "name", None)
        if not name or self.mojo_resource_kind(type_name, node=node) is None:
            return
        access = self.normalized_resource_access_metadata(node)
        if access:
            self.resource_access_qualifiers[name] = access

    def resource_binding_initial_value(self, name, mapped_type, fallback_type):
        if mapped_type not in MOJO_CONCRETE_RESOURCE_TYPES:
            return self.zero_value_for_type(fallback_type)

        initializer = self.mojo_resource_binding_initializers.get(name)
        if not isinstance(initializer, dict):
            return self.zero_value_for_type(fallback_type)

        set_index = initializer.get("set", 0)
        binding = initializer.get("binding", 0)
        if mapped_type == "Texture2D":
            return f"Texture2D({set_index}, {binding})"
        if mapped_type == "Sampler":
            return f"Sampler({set_index}, {binding})"
        return self.zero_value_for_type(fallback_type)

    def register_resource_access_alias_metadata(self, name, type_name, initializer):
        if not name or initializer is None:
            return
        type_name = self.type_name(type_name)
        if not self.resource_access_aliasable_type(type_name):
            return

        candidates = self.resource_access_qualifier_candidates(
            initializer, target_type=type_name
        )
        if candidates:
            self.resource_access_qualifier_aliases[name] = candidates
        else:
            self.resource_access_qualifier_aliases.pop(name, None)

        self.clear_resource_access_path_aliases(name)
        for path, path_candidates in self.resource_access_field_alias_candidates(
            initializer, target_type=type_name
        ).items():
            if not path:
                continue
            self.resource_access_path_qualifier_aliases[f"{name}.{path}"] = (
                path_candidates
            )

    def clear_resource_access_path_aliases(self, name):
        prefix = f"{name}."
        for path in list(self.resource_access_path_qualifier_aliases):
            if path == name or path.startswith(prefix):
                self.resource_access_path_qualifier_aliases.pop(path, None)

    def register_resource_assignment_alias_metadata(self, target, value, operator):
        if self.map_operator(operator) != "=":
            return

        name = self.direct_assignment_target_name(target)
        if name is None:
            return

        type_name = self.expression_result_type(target) or self.variable_types.get(name)
        self.register_resource_access_alias_metadata(name, type_name, value)

    def direct_assignment_target_name(self, target):
        if isinstance(target, str):
            return target
        if isinstance(target, VariableNode) and hasattr(target, "name"):
            return target.name
        if hasattr(target, "__class__") and "Identifier" in str(target.__class__):
            return getattr(target, "name", None)
        return None

    def register_struct_resource_access_metadata(
        self, struct_name, member_name, member, member_type
    ):
        access = self.normalized_resource_access_metadata(member)
        if access is None:
            return

        base_type, _ = self.resource_base_type_and_count(member_type)
        if self.mojo_resource_kind(base_type, node=member) is None:
            return

        self.struct_resource_access_qualifiers.setdefault(struct_name, {})[
            member_name
        ] = access

    def resource_access_root_name(self, expr):
        if isinstance(expr, ArrayAccessNode):
            return self.resource_access_root_name(expr.array)
        if isinstance(expr, MemberAccessNode):
            return self.resource_access_root_name(expr.object)
        name = getattr(expr, "name", None)
        if name is not None:
            if not isinstance(name, str):
                return None
            replacement = self.expression_identifier_replacements.get(name)
            if isinstance(replacement, str):
                match = re.match(r"\s*([A-Za-z_]\w*)", replacement)
                if match is not None:
                    return match.group(1)
            return name
        return None

    def resource_access_path_name(self, expr):
        if isinstance(expr, ArrayAccessNode):
            return self.resource_access_path_name(expr.array)
        if isinstance(expr, MemberAccessNode):
            obj_path = self.resource_access_path_name(expr.object)
            if obj_path:
                return f"{obj_path}.{expr.member}"
            return expr.member
        name = getattr(expr, "name", None)
        if name is not None:
            if not isinstance(name, str):
                return None
            replacement = self.expression_identifier_replacements.get(name)
            if isinstance(replacement, str):
                return replacement.strip()
            return name
        return None

    def resource_access_source_path_name(self, expr):
        if isinstance(expr, ArrayAccessNode):
            return self.resource_access_source_path_name(expr.array)
        if isinstance(expr, MemberAccessNode):
            obj_path = self.resource_access_source_path_name(expr.object)
            if obj_path:
                return f"{obj_path}.{expr.member}"
            return expr.member
        if isinstance(expr, str):
            return expr
        name = getattr(expr, "name", None)
        if isinstance(name, str):
            return name
        return None

    def resource_access_diagnostic_name(self, expr, resource_name):
        source_name = self.resource_access_source_path_name(expr)
        if resource_name and source_name != resource_name:
            return resource_name
        return source_name or resource_name

    def struct_field_resource_access(self, expr):
        if isinstance(expr, ArrayAccessNode):
            return self.struct_field_resource_access(expr.array)
        if isinstance(expr, MemberAccessNode):
            obj_type = self.expression_result_type(expr.object)
            access = self.struct_resource_access_qualifiers.get(obj_type, {}).get(
                expr.member
            )
            if access is not None:
                return access, self.resource_access_source_path_name(expr)
            return self.struct_field_resource_access(expr.object)
        return None

    def direct_resource_access_qualifier(self, expr):
        root_name = self.resource_access_root_name(expr)
        if root_name is not None:
            access = self.resource_access_qualifiers.get(root_name)
            if access is not None:
                return access, root_name

        field_access = self.struct_field_resource_access(expr)
        if field_access is not None:
            return field_access
        return None, root_name

    def resource_access_qualifier_candidates(self, expr, target_type=None):
        if isinstance(expr, ArrayAccessNode):
            return self.resource_access_qualifier_candidates(expr.array)

        if isinstance(expr, TernaryOpNode):
            candidates = []
            candidates.extend(
                self.resource_access_qualifier_candidates(
                    expr.true_expr, target_type=target_type
                )
            )
            candidates.extend(
                self.resource_access_qualifier_candidates(
                    expr.false_expr, target_type=target_type
                )
            )
            return self.unique_resource_access_qualifiers(candidates)

        if isinstance(expr, MatchNode):
            candidates = []
            for arm in getattr(expr, "arms", []) or []:
                arm_expr, _ = self.match_arm_value_expression(arm)
                candidates.extend(
                    self.resource_access_qualifier_candidates(
                        arm_expr, target_type=target_type
                    )
                )
            return self.unique_resource_access_qualifiers(candidates)

        path_name = self.resource_access_path_name(expr)
        if path_name is not None:
            candidates = self.resource_access_path_qualifier_aliases.get(path_name)
            if candidates:
                return list(candidates)

        if isinstance(expr, FunctionCallNode):
            candidates = self.function_return_resource_alias_candidates(expr)
            if candidates:
                return candidates
            candidates = self.generic_enum_resource_alias_candidates(expr, target_type)
            if candidates:
                return candidates

        candidates = []
        access, resource_name = self.direct_resource_access_qualifier(expr)
        if access is not None and resource_name is not None:
            candidates.append((access, resource_name))

        root_name = self.resource_access_root_name(expr)
        if root_name is not None:
            candidates.extend(self.resource_access_qualifier_aliases.get(root_name, []))
        return self.unique_resource_access_qualifiers(candidates)

    def generic_enum_resource_alias_candidates(self, expr, target_type):
        if target_type is None:
            return []

        func_name = self.function_call_name(expr)
        fields = self.generic_enum_variant_fields_for_type(func_name, target_type)
        if fields is None or len(fields) != len(expr.args):
            return []

        candidates = []
        for arg, (_field_name, field_type) in zip(expr.args, fields):
            if not self.resource_access_aliasable_type(field_type):
                continue
            candidates.extend(self.resource_access_qualifier_candidates(arg))
        return self.unique_resource_access_qualifiers(candidates)

    def resource_access_aliasable_type(self, type_name):
        return self._resource_access_aliasable_type(self.type_name(type_name), set())

    def _resource_access_aliasable_type(self, type_name, seen):
        if self.is_array_type_name(type_name):
            base_type, _ = self.parse_array_type_name(type_name)
            return self._resource_access_aliasable_type(base_type, seen)

        generic_fields = self.generic_struct_fields_for_type(type_name)
        if generic_fields is not None:
            if type_name in seen:
                return False
            next_seen = {*seen, type_name}
            return any(
                self._resource_access_aliasable_type(field_type, next_seen)
                for field_type in generic_fields.values()
            )

        base_type, _ = self.resource_base_type_and_count(type_name)
        if self.mojo_resource_kind(base_type) is not None:
            return True

        if base_type in seen:
            return False

        if base_type in self.struct_types:
            next_seen = {*seen, base_type}
            return any(
                self._resource_access_aliasable_type(field_type, next_seen)
                for field_type in self.struct_types.get(base_type, {}).values()
            )

        specialization = self.generic_enum_specialization_for_type(type_name)
        if specialization is None:
            return False

        next_seen = {*seen, type_name}
        return any(
            self._resource_access_aliasable_type(field_type, next_seen)
            for _field_name, field_type in generic_enum_specialized_fields(
                self, specialization
            )
        )

    def resource_aliasable_field_paths_for_type(self, type_name, seen=None):
        if seen is None:
            seen = set()

        type_name = self.type_name(type_name)
        base_type, _ = self.resource_base_type_and_count(type_name)
        if self.mojo_resource_kind(base_type) is not None:
            return [""]

        generic_fields = self.generic_struct_fields_for_type(type_name)
        if generic_fields is not None:
            if type_name in seen:
                return []
            paths = []
            next_seen = {*seen, type_name}
            for field_name, field_type in generic_fields.items():
                for subpath in self.resource_aliasable_field_paths_for_type(
                    field_type, next_seen
                ):
                    paths.append(f"{field_name}.{subpath}" if subpath else field_name)
            return paths

        if base_type in seen:
            return []

        if base_type in self.struct_types:
            paths = []
            next_seen = {*seen, base_type}
            for field_name, field_type in self.struct_types.get(base_type, {}).items():
                for subpath in self.resource_aliasable_field_paths_for_type(
                    field_type, next_seen
                ):
                    paths.append(f"{field_name}.{subpath}" if subpath else field_name)
            return paths

        specialization = self.generic_enum_specialization_for_type(type_name)
        if specialization is None or type_name in seen:
            return []

        paths = []
        next_seen = {*seen, type_name}
        for field_name, field_type in generic_enum_specialized_fields(
            self, specialization
        ):
            for subpath in self.resource_aliasable_field_paths_for_type(
                field_type, next_seen
            ):
                paths.append(f"{field_name}.{subpath}" if subpath else field_name)
        return paths

    def resource_access_field_alias_candidates(self, expr, target_type=None):
        target_type = self.type_name(target_type)
        if not target_type or not self.resource_access_aliasable_type(target_type):
            return {}

        field_aliases = self.resource_access_path_aliases_for_expression(expr)
        if field_aliases:
            return field_aliases

        if isinstance(expr, TernaryOpNode):
            return self.merge_resource_field_alias_maps(
                self.resource_access_field_alias_candidates(
                    expr.true_expr, target_type=target_type
                ),
                self.resource_access_field_alias_candidates(
                    expr.false_expr, target_type=target_type
                ),
            )

        if isinstance(expr, MatchNode):
            merged = {}
            for arm in getattr(expr, "arms", []) or []:
                try:
                    arm_expr, _ = self.match_arm_value_expression(arm)
                except ValueError:
                    continue
                merged = self.merge_resource_field_alias_maps(
                    merged,
                    self.resource_access_field_alias_candidates(
                        arm_expr, target_type=target_type
                    ),
                )
            return merged

        if isinstance(expr, ArrayLiteralNode):
            element_type = self.array_element_type(target_type)
            merged = {}
            for element in getattr(expr, "elements", []) or []:
                merged = self.merge_resource_field_alias_maps(
                    merged,
                    self.resource_access_field_alias_candidates(
                        element, target_type=element_type
                    ),
                )
            return merged

        if isinstance(expr, FunctionCallNode):
            aliases = self.generic_enum_field_alias_candidates(expr, target_type)
            if aliases:
                return aliases
            aliases = self.struct_constructor_field_alias_candidates(expr, target_type)
            if aliases:
                return aliases
            return self.function_return_resource_field_alias_candidates(
                expr, target_type
            )

        if isinstance(expr, ConstructorNode):
            aliases = self.constructor_node_field_alias_candidates(expr, target_type)
            if aliases:
                return aliases

        candidates = self.resource_access_qualifier_candidates(
            expr, target_type=target_type
        )
        return {"": candidates} if candidates else {}

    def resource_access_path_aliases_for_expression(self, expr):
        path_name = self.resource_access_path_name(expr)
        if not path_name:
            return {}

        aliases = {}
        exact = self.resource_access_path_qualifier_aliases.get(path_name)
        if exact:
            aliases[""] = list(exact)

        prefix = f"{path_name}."
        for (
            alias_path,
            candidates,
        ) in self.resource_access_path_qualifier_aliases.items():
            if alias_path.startswith(prefix):
                aliases[alias_path[len(prefix) :]] = list(candidates)
        return aliases

    def merge_resource_field_alias_maps(self, *maps):
        merged = {}
        for alias_map in maps:
            for path, candidates in alias_map.items():
                current = merged.setdefault(path, [])
                for candidate in candidates:
                    if candidate not in current:
                        current.append(candidate)
        return merged

    def resource_access_alias_state(self):
        return (
            {
                name: list(candidates)
                for name, candidates in self.resource_access_qualifier_aliases.items()
            },
            {
                path: list(candidates)
                for path, candidates in (
                    self.resource_access_path_qualifier_aliases.items()
                )
            },
        )

    def restore_resource_access_alias_state(self, state):
        qualifier_aliases, path_aliases = state
        self.resource_access_qualifier_aliases = {
            name: list(candidates) for name, candidates in qualifier_aliases.items()
        }
        self.resource_access_path_qualifier_aliases = {
            path: list(candidates) for path, candidates in path_aliases.items()
        }

    def merge_resource_access_alias_states(self, *states):
        qualifier_maps = [state[0] for state in states]
        path_maps = [state[1] for state in states]
        return (
            self.merge_resource_access_candidate_maps(*qualifier_maps),
            self.merge_resource_access_candidate_maps(*path_maps),
        )

    def merge_resource_access_candidate_maps(self, *maps):
        merged = {}
        for candidate_map in maps:
            for name, candidates in candidate_map.items():
                current = merged.setdefault(name, [])
                for candidate in candidates:
                    if candidate not in current:
                        current.append(candidate)
        return merged

    def generic_enum_field_alias_candidates(self, expr, target_type):
        func_name = self.function_call_name(expr)
        fields = self.generic_enum_variant_fields_for_type(func_name, target_type)
        if fields is None or len(fields) != len(expr.args):
            return {}

        aliases = {}
        for arg, (field_name, field_type) in zip(expr.args, fields):
            if not self.resource_access_aliasable_type(field_type):
                continue
            nested = self.resource_access_field_alias_candidates(
                arg, target_type=field_type
            )
            for path, candidates in nested.items():
                field_path = f"{field_name}.{path}" if path else field_name
                aliases[field_path] = candidates
        return aliases

    def struct_constructor_field_alias_candidates(self, expr, target_type):
        func_name = self.function_call_name(expr)
        target_base, _ = self.resource_base_type_and_count(self.type_name(target_type))
        struct_type, fields = self.struct_type_name_and_fields(target_type)
        if struct_type is None:
            struct_type, fields = self.struct_type_name_and_fields(target_base)
        if struct_type is None:
            struct_type, fields = self.struct_type_name_and_fields(func_name)
        if struct_type is None or func_name != struct_type:
            return {}

        aliases = {}
        field_items = list(fields.items())
        for arg, (field_name, field_type) in zip(expr.args, field_items):
            if not self.resource_access_aliasable_type(field_type):
                continue
            nested = self.resource_access_field_alias_candidates(
                arg, target_type=field_type
            )
            for path, candidates in nested.items():
                field_path = f"{field_name}.{path}" if path else field_name
                aliases[field_path] = candidates
        return aliases

    def constructor_node_field_alias_candidates(self, expr, target_type):
        struct_type, fields = self.constructor_node_struct_type_and_fields(
            expr, target_type
        )
        if struct_type is None:
            return {}

        aliases = {}
        for field_name, field_type, arg in self.constructor_node_field_bindings(
            expr, struct_type, fields
        ):
            if not self.resource_access_aliasable_type(field_type):
                continue
            nested = self.resource_access_field_alias_candidates(
                arg, target_type=field_type
            )
            for path, candidates in nested.items():
                field_path = f"{field_name}.{path}" if path else field_name
                aliases[field_path] = candidates
        return aliases

    def function_return_resource_field_alias_candidates(self, expr, target_type=None):
        func_name = self.function_call_name(expr)
        refs = self.function_return_resource_field_aliases.get(func_name) or []
        aliases = {
            path: list(candidates)
            for path, candidates in (
                self.function_return_resource_static_field_aliases.get(func_name) or {}
            ).items()
        }
        parameter_types = self.function_parameter_types.get(func_name, [])
        for return_path, alias_index, arg_path in refs:
            if alias_index >= len(expr.args):
                continue
            arg_type = (
                parameter_types[alias_index]
                if alias_index < len(parameter_types)
                else None
            )
            candidates = self.resource_access_candidates_for_relative_path(
                expr.args[alias_index], arg_path, arg_type
            )
            if candidates:
                aliases.setdefault(return_path, [])
                for candidate in candidates:
                    if candidate not in aliases[return_path]:
                        aliases[return_path].append(candidate)
        return aliases

    def resource_access_candidates_for_relative_path(
        self, expr, relative_path, target_type=None
    ):
        if not relative_path:
            return self.resource_access_qualifier_candidates(
                expr, target_type=target_type
            )

        path_name = self.resource_access_path_name(expr)
        if path_name:
            exact_path = f"{path_name}.{relative_path}"
            candidates = self.resource_access_path_qualifier_aliases.get(exact_path)
            if candidates:
                return list(candidates)

        field_aliases = self.resource_access_field_alias_candidates(
            expr, target_type=target_type
        )
        return list(field_aliases.get(relative_path, []))

    def function_return_resource_alias_candidates(self, expr):
        func_name = self.function_call_name(expr)
        candidates = list(
            self.function_return_resource_static_aliases.get(func_name, [])
        )
        alias_indices = self.function_return_resource_aliases.get(func_name)
        if not alias_indices:
            return self.unique_resource_access_qualifiers(candidates)

        for alias_index in alias_indices:
            if alias_index >= len(expr.args):
                continue
            candidates.extend(
                self.resource_access_qualifier_candidates(expr.args[alias_index])
            )
        return self.unique_resource_access_qualifiers(candidates)

    def unique_resource_access_qualifiers(self, candidates):
        unique = []
        seen = set()
        for access, resource_name in candidates:
            key = (access, resource_name)
            if key in seen:
                continue
            seen.add(key)
            unique.append((access, resource_name))
        return unique

    def resource_access_qualifier(self, expr):
        candidates = self.resource_access_qualifier_candidates(expr)
        if candidates:
            return candidates[0]
        return self.direct_resource_access_qualifier(expr)

    def expression_has_resource_type(self, expr):
        expr_type = self.expression_result_type(expr)
        if expr_type is None:
            return True
        base_type, _ = self.resource_base_type_and_count(expr_type)
        if self.mojo_resource_kind(base_type) is not None:
            return True
        access, _ = self.direct_resource_access_qualifier(expr)
        return access is not None

    def validate_resource_read_access(self, expr, operation):
        if not self.expression_has_resource_type(expr):
            return
        for access, resource_name in self.resource_access_qualifier_candidates(expr):
            if access == "writeonly":
                diagnostic_name = self.resource_access_diagnostic_name(
                    expr, resource_name
                )
                raise MojoRefusal(
                    f"Unsupported {operation} for Mojo codegen; "
                    f"resource '{diagnostic_name}' is writeonly"
                )

    def validate_resource_write_access(self, expr, operation):
        if not self.expression_has_resource_type(expr):
            return
        for access, resource_name in self.resource_access_qualifier_candidates(expr):
            if access == "readonly":
                diagnostic_name = self.resource_access_diagnostic_name(
                    expr, resource_name
                )
                raise MojoRefusal(
                    f"Unsupported {operation} for Mojo codegen; "
                    f"resource '{diagnostic_name}' is readonly"
                )

    def validate_resource_read_write_access(self, expr, operation):
        self.validate_resource_read_access(expr, operation)
        self.validate_resource_write_access(expr, operation)

    def byte_address_buffer_value_type(self, buffer_type):
        if buffer_type in MOJO_BYTE_ADDRESS_BUFFER_TYPES:
            return "uint"
        return None

    def buffer_helper_element_type(self, buffer_type, element_type):
        return element_type or self.byte_address_buffer_value_type(buffer_type)

    def validate_buffer_operation(self, operation, info, allowed_buffer_types):
        if info is None:
            return
        buffer_type, _ = info
        if buffer_type not in allowed_buffer_types:
            raise MojoRefusal(
                f"Unsupported {operation} for Mojo codegen; "
                f"{buffer_type} is not valid for this operation"
            )

    def byte_address_load_result_type(self, width):
        if width == 1:
            return "uint"
        return f"uvec{width}"

    def byte_address_vector_mojo_type(self, width, dtype="DType.uint32"):
        source_width = int(width)
        storage_width = 4 if source_width == 3 else source_width
        return f"SIMD[{dtype}, {storage_width}]"

    def byte_address_vector_zero_value(self, width, dtype="DType.uint32"):
        mojo_type = self.byte_address_vector_mojo_type(width, dtype)
        zero = MOJO_DTYPE_INFO[dtype][2]
        storage_width = 4 if int(width) == 3 else int(width)
        return f"{mojo_type}({', '.join([zero] * storage_width)})"

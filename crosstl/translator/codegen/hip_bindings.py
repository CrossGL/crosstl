"""Resource bindings and GLSL buffer blocks of the HIP generator."""

from ..ast import (
    ArrayAccessNode,
    IdentifierNode,
    MemberAccessNode,
    VariableNode,
)
from ..refusals import HipRefusal


class HipBindingMixin:
    """Binding members of ``HipCodeGen``."""

    def attribute_value_to_string(self, value):
        if value is None:
            return None
        if isinstance(value, str):
            return value
        if hasattr(value, "name") and value.name is not None:
            return str(value.name)
        if hasattr(value, "value") and value.value is not None:
            return str(value.value).strip('"')
        return str(value)

    def attribute_arguments(self, attr):
        return getattr(attr, "arguments", getattr(attr, "args", [])) or []

    def resource_binding_index_value(self, value, prefixes=()):
        raw_value = self.attribute_value_to_string(value)
        if raw_value is None:
            return None
        raw_value = str(raw_value).strip().lower()
        if raw_value.isdigit():
            return int(raw_value)
        for prefix in prefixes:
            if raw_value.startswith(prefix) and raw_value[len(prefix) :].isdigit():
                return int(raw_value[len(prefix) :])
        return None

    def resource_register_space_value(self, value):
        raw_value = self.attribute_value_to_string(value)
        if raw_value is None:
            return None
        raw_value = str(raw_value).strip().lower()
        if raw_value.isdigit():
            return int(raw_value)
        if raw_value.startswith("space") and raw_value[5:].isdigit():
            return int(raw_value[5:])
        return None

    def explicit_hip_resource_binding_index(self, node):
        for attr in getattr(node, "attributes", []) or []:
            attr_name = str(getattr(attr, "name", "")).lower()
            arguments = self.attribute_arguments(attr)
            if not arguments:
                continue
            if attr_name in {"binding", "buffer", "sampler", "texture", "uav"}:
                binding = self.resource_binding_index_value(
                    arguments[0], ("b", "s", "t", "u")
                )
            elif attr_name == "register":
                binding = self.resource_binding_index_value(
                    arguments[0], ("b", "s", "t", "u")
                )
            else:
                binding = None
            if binding is not None:
                return binding
        return None

    def explicit_hip_resource_set_index(self, node):
        for attr in getattr(node, "attributes", []) or []:
            attr_name = str(getattr(attr, "name", "")).lower()
            arguments = self.attribute_arguments(attr)
            if attr_name in {"set", "group"} and arguments:
                set_index = self.resource_binding_index_value(arguments[0])
                if set_index is not None:
                    return set_index
            if attr_name == "space" and arguments:
                set_index = self.resource_register_space_value(arguments[0])
                if set_index is not None:
                    return set_index
            if attr_name == "register":
                for argument in arguments[1:]:
                    set_index = self.resource_register_space_value(argument)
                    if set_index is not None:
                        return set_index
        return 0

    def hip_resource_register_metadata(self, node):
        for attr in getattr(node, "attributes", []) or []:
            if str(getattr(attr, "name", "")).lower() != "register":
                continue
            values = [
                self.attribute_value_to_string(argument)
                for argument in self.attribute_arguments(attr)
            ]
            values = [value for value in values if value]
            if values:
                return ",".join(values)
        return None

    def hip_explicit_resource_binding_attribute(self, node):
        for attr in getattr(node, "attributes", []) or []:
            attr_name = str(getattr(attr, "name", "")).lower()
            arguments = self.attribute_arguments(attr)
            if not arguments:
                continue
            if attr_name in {"binding", "buffer", "sampler", "texture", "uav"}:
                binding = self.resource_binding_index_value(
                    arguments[0], ("b", "s", "t", "u")
                )
            elif attr_name == "register":
                binding = self.resource_binding_index_value(
                    arguments[0], ("b", "s", "t", "u")
                )
            else:
                binding = None
            if binding is not None:
                return attr_name
        return None

    def hip_resource_base_type_and_count(self, type_name):
        type_name = self.type_name_string(type_name)
        if not type_name:
            return type_name, 1
        if "[" not in type_name or "]" not in type_name:
            return type_name, 1
        base_type = type_name.split("[", 1)[0].strip()
        suffix = type_name[type_name.find("[") :]
        count = 1
        while suffix.startswith("["):
            close_bracket = suffix.find("]")
            if close_bracket < 0:
                break
            size_text = suffix[1:close_bracket].strip()
            if size_text.isdigit():
                count *= int(size_text)
            suffix = suffix[close_bracket + 1 :]
        return base_type, count

    def hip_resource_kind(self, type_name, node=None, forced_kind=None):
        if forced_kind is not None:
            return forced_kind
        if node is not None and self.is_glsl_buffer_block_node(node):
            return "glsl_buffer_block"

        base_type, _count = self.hip_resource_base_type_and_count(type_name)
        if not base_type:
            return None
        generic_parts = self.generic_type_parts(base_type)
        base_name = generic_parts[0] if generic_parts is not None else base_type
        base_name = base_name.rsplit("::", 1)[-1]

        if (
            self.structured_buffer_type_parts(base_type) is not None
            or self.byte_address_buffer_base_type(base_type) is not None
        ):
            return "buffer"

        canonical = self.canonical_resource_type(base_name) or base_name
        canonical = canonical.rsplit("::", 1)[-1]
        # Classification only: a type type_map does not rename is classified by its own spelling.
        mapped_type = (
            self.type_map[canonical] if canonical in self.type_map else canonical
        )
        mapped_base = mapped_type.split("<", 1)[0].rsplit("::", 1)[-1]
        if mapped_base == "CglRayTracingAccelerationStructure":
            return "acceleration_structure"
        if canonical in {"sampler", "SamplerState"}:
            return "sampler"
        if canonical.startswith("sampler"):
            return "texture"
        if canonical.startswith(("image", "iimage", "uimage")):
            return "image"
        return None

    def hip_resource_binding_namespace(self, kind):
        if kind in {"cbuffer", "glsl_buffer_block"}:
            return "buffer"
        return kind

    def hip_resource_binding_validation_count(self, node, resource_kind, count):
        if (
            resource_kind in {"image", "sampler", "texture"}
            and count != 1
            and self.hip_explicit_resource_binding_attribute(node) == "binding"
        ):
            return 1
        return count

    def hip_resource_binding_range_conflicts(self, key, binding, count):
        end = binding + count - 1
        for used_start, used_end, _used_name in self.hip_used_resource_bindings.get(
            key, []
        ):
            if binding <= used_end and used_start <= end:
                return True
        return False

    def next_available_hip_resource_binding(self, namespace, set_index, count):
        key = (namespace, set_index)
        binding = self.hip_resource_binding_cursors.get(key, 0)
        while self.hip_resource_binding_range_conflicts(key, binding, count):
            binding += 1
        self.hip_resource_binding_cursors[key] = binding + count
        return binding

    def reserve_hip_resource_binding(self, namespace, set_index, binding, count, name):
        key = (namespace, set_index)
        end = binding + count - 1
        ranges = self.hip_used_resource_bindings.setdefault(key, [])
        for used_start, used_end, used_name in ranges:
            if binding <= used_end and used_start <= end:
                if used_start == binding and used_end == end and used_name == name:
                    return
                raise HipRefusal(
                    "Conflicting HIP resource binding for "
                    f"'{name}': {namespace} set {set_index} binding "
                    f"{binding}-{end} overlaps '{used_name}' binding "
                    f"{used_start}-{used_end}"
                )
        ranges.append((binding, end, name))
        self.hip_resource_binding_cursors[key] = max(
            self.hip_resource_binding_cursors.get(key, 0), end + 1
        )

    def hip_resource_metadata_comment(self, node, type_name, kind=None):
        name = getattr(node, "name", None)
        resource_kind = self.hip_resource_kind(type_name, node=node, forced_kind=kind)
        if not name or resource_kind is None:
            return ""

        _base_type, count = self.hip_resource_base_type_and_count(type_name)
        validation_count = self.hip_resource_binding_validation_count(
            node, resource_kind, count
        )
        namespace = self.hip_resource_binding_namespace(resource_kind)
        set_index = self.explicit_hip_resource_set_index(node)
        binding = self.explicit_hip_resource_binding_index(node)
        if binding is None:
            binding = self.next_available_hip_resource_binding(
                namespace, set_index, validation_count
            )
            binding_source = "automatic"
            self.reserve_hip_resource_binding(
                namespace, set_index, binding, validation_count, name
            )
        else:
            binding_source = "explicit"
            self.reserve_hip_resource_binding(
                namespace, set_index, binding, validation_count, name
            )

        parts = [
            "// CrossGL resource metadata:",
            f"name={name}",
            f"kind={resource_kind}",
        ]
        if resource_kind == "glsl_buffer_block":
            layout = self.glsl_buffer_block_layout(node)
            if layout:
                parts.append(f"layout={layout}")
            access = self.explicit_resource_access(node)
            if access:
                parts.append(f"access={access}")
        parts.extend(
            [
                f"set={set_index}",
                f"binding={binding}",
                f"binding_source={binding_source}",
            ]
        )
        if count != 1:
            parts.append(f"count={count}")
        register_metadata = self.hip_resource_register_metadata(node)
        if register_metadata:
            parts.append(f"register={register_metadata}")
        return " ".join(parts)

    def reserve_explicit_hip_resource_binding(self, node, type_name, kind=None):
        name = getattr(node, "name", None)
        resource_kind = self.hip_resource_kind(type_name, node=node, forced_kind=kind)
        binding = self.explicit_hip_resource_binding_index(node)
        if not name or resource_kind is None or binding is None:
            return
        _base_type, count = self.hip_resource_base_type_and_count(type_name)
        validation_count = self.hip_resource_binding_validation_count(
            node, resource_kind, count
        )
        namespace = self.hip_resource_binding_namespace(resource_kind)
        set_index = self.explicit_hip_resource_set_index(node)
        self.reserve_hip_resource_binding(
            namespace, set_index, binding, validation_count, name
        )

    def reserve_explicit_hip_resource_bindings(self, ast):
        for node in getattr(ast, "global_variables", []) or []:
            type_name = self.get_variable_node_type(node) or "float"
            self.reserve_explicit_hip_resource_binding(node, type_name)
        for cbuffer in getattr(ast, "cbuffers", []) or []:
            self.reserve_explicit_hip_resource_binding(
                cbuffer, getattr(cbuffer, "name", None), kind="cbuffer"
            )
        stages = getattr(ast, "stages", {}) or {}
        for stage in stages.values():
            for node in getattr(stage, "local_variables", []) or []:
                type_name = self.get_variable_node_type(node) or "float"
                self.reserve_explicit_hip_resource_binding(node, type_name)
            for cbuffer in getattr(stage, "local_cbuffers", []) or []:
                self.reserve_explicit_hip_resource_binding(
                    cbuffer, getattr(cbuffer, "name", None), kind="cbuffer"
                )

    def explicit_resource_access(self, node):
        if node is None:
            return None

        access_names = {
            "read": "readonly",
            "readonly": "readonly",
            "write": "writeonly",
            "writeonly": "writeonly",
            "read_write": "readwrite",
            "readwrite": "readwrite",
            "access::read": "readonly",
            "access::write": "writeonly",
            "access::read_write": "readwrite",
        }
        for qualifier in getattr(node, "qualifiers", []) or []:
            access = access_names.get(str(qualifier).lower())
            if access is not None:
                return access

        for attr in getattr(node, "attributes", []) or []:
            attr_name = str(getattr(attr, "name", "")).lower()
            if attr_name == "access":
                arguments = getattr(attr, "arguments", []) or []
                if not arguments:
                    continue
                raw_access = self.attribute_value_to_string(arguments[0])
                access = access_names.get(str(raw_access).lower())
            else:
                access = access_names.get(attr_name)
            if access is not None:
                return access
        return None

    def is_glsl_buffer_block_node(self, node):
        attributes = {
            str(getattr(attr, "name", "")).lower()
            for attr in getattr(node, "attributes", []) or []
        }
        qualifiers = {
            str(qualifier).lower()
            for qualifier in getattr(node, "qualifiers", []) or []
        }
        if "glsl_buffer_block" in attributes:
            return True
        return "buffer" in qualifiers and bool(
            attributes & {"std140", "std430", "scalar"}
        )

    def glsl_buffer_block_layout(self, node):
        for attr in getattr(node, "attributes", []) or []:
            attr_name = str(getattr(attr, "name", "")).lower()
            if attr_name == "glsl_buffer_block":
                arguments = getattr(attr, "arguments", []) or []
                if arguments:
                    return self.attribute_value_to_string(arguments[0])
            if attr_name in {"std140", "std430", "scalar"}:
                return attr_name
        return None

    def glsl_buffer_block_declaration_type(self, type_name, node):
        type_name = self.type_name_string(type_name)
        if not self.is_glsl_buffer_block_node(node):
            return type_name
        if not type_name or "[" not in type_name or "]" not in type_name:
            return type_name
        open_bracket = type_name.find("[")
        base_type = type_name[:open_bracket].strip()
        array_suffix = type_name[open_bracket:]
        if array_suffix != "[]":
            return type_name
        if self.explicit_resource_access(node) == "readonly":
            return f"const {base_type}{array_suffix}"
        return type_name

    def glsl_buffer_block_metadata_comment(self, node, type_name):
        name = getattr(node, "name", None)
        if not name or not self.is_glsl_buffer_block_node(node):
            return ""
        parts = [
            "// CrossGL resource metadata:",
            f"name={name}",
            "kind=glsl_buffer_block",
        ]
        layout = self.glsl_buffer_block_layout(node)
        if layout:
            parts.append(f"layout={layout}")
        access = self.explicit_resource_access(node)
        if access:
            parts.append(f"access={access}")
        return " ".join(parts)

    def glsl_buffer_block_root_name(self, expr):
        if isinstance(expr, ArrayAccessNode):
            array_expr = getattr(expr, "array", getattr(expr, "array_expr", None))
            return self.glsl_buffer_block_root_name(array_expr)
        if isinstance(expr, MemberAccessNode):
            object_expr = getattr(expr, "object", getattr(expr, "object_expr", None))
            return self.glsl_buffer_block_root_name(object_expr)
        if isinstance(expr, IdentifierNode):
            return expr.name
        if isinstance(expr, VariableNode):
            return expr.name
        if isinstance(expr, str):
            return expr
        return getattr(expr, "name", None)

    def glsl_buffer_block_access(self, expr):
        root_name = self.glsl_buffer_block_root_name(expr)
        if root_name is None:
            return None, None
        return self.glsl_buffer_block_accesses.get(root_name), root_name

    def glsl_buffer_block_diagnostic(self, operation, expr, reason, fallback):
        _, resource_name = self.glsl_buffer_block_access(expr)
        resource_name = resource_name or "unknown"
        return self.refuse_lowering(
            f"{self.resource_backend_name()} GLSL buffer block {operation}: resource '{resource_name}' {reason}",
            fallback,
        )

    def glsl_buffer_block_read_diagnostic(self, expr, operation):
        access, _ = self.glsl_buffer_block_access(expr)
        if access != "writeonly":
            return None
        fallback = self.diagnostic_zero_value_for_type(
            self.expression_result_type(expr)
        )
        return self.glsl_buffer_block_diagnostic(
            operation, expr, "is writeonly", fallback
        )

    def glsl_buffer_block_write_diagnostic(self, expr, operation):
        access, _ = self.glsl_buffer_block_access(expr)
        if access != "readonly":
            return None
        return self.glsl_buffer_block_diagnostic(
            operation, expr, "is readonly", "((void)0)"
        )

    def glsl_buffer_block_read_write_diagnostic(self, expr, operation, fallback):
        access, _ = self.glsl_buffer_block_access(expr)
        if access == "readonly":
            return self.glsl_buffer_block_diagnostic(
                operation, expr, "is readonly", fallback
            )
        if access == "writeonly":
            return self.glsl_buffer_block_diagnostic(
                operation, expr, "is writeonly", fallback
            )
        return None

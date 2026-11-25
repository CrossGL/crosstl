"""Expression result types of the HIP generator and canonical resource type names."""

from .resource_diagnostics import ResourceDiagnosticMixin


class HipExpressionTypeMixin:
    """Expression type members of ``HipCodeGen``."""

    def step_result_type(self, raw_args):
        edge_type = self.expression_result_type(raw_args[0])
        value_type = self.expression_result_type(raw_args[1])
        value_info = self.vector_type_info(value_type)
        if value_info is not None:
            return value_type
        edge_info = self.vector_type_info(edge_type)
        if edge_info is not None:
            return edge_type
        return self.step_scalar_type(raw_args)

    def struct_member_lookup_type(self, type_name):
        """Return the struct key after HIP pointer/reference wrappers."""
        type_name = self.type_name_string(type_name)
        if not type_name:
            return None
        indirect_info = self.hip_indirect_type_info(type_name)
        if indirect_info is not None:
            return indirect_info["pointee_type"]
        return type_name

    def member_access_member_type(self, node):
        """Return the member type for object.field expressions."""
        object_node = getattr(
            node,
            "object_expr",
            getattr(node, "object", None),
        )
        object_type = self.struct_member_lookup_type(
            self.expression_result_type(object_node)
        )
        return self.struct_member_types.get(object_type, {}).get(
            getattr(node, "member", "")
        )

    def pointer_access_member_type(self, node):
        """Return the member type for ptr->field expressions."""
        pointer_expr = getattr(node, "pointer_expr", None)
        pointer_info = self.hip_indirect_type_info(
            self.expression_result_type(pointer_expr)
        )
        if pointer_info is None:
            return None

        struct_type = self.struct_member_lookup_type(pointer_info["pointee_type"])
        return self.struct_member_types.get(struct_type, {}).get(
            getattr(node, "member", "")
        )

    def buffer_call_result_type(self, node):
        """Infer result type for structured and byte-address buffer read calls."""
        function_expr = getattr(node, "function", getattr(node, "name", None))
        raw_args = getattr(node, "arguments", getattr(node, "args", []))

        byte_member_call = self.byte_address_buffer_member_call(function_expr)
        if byte_member_call is not None:
            _, operation, _ = byte_member_call
            if operation == "GetDimensions":
                return "uint"
            if operation in self.byte_address_buffer_atomic_operations():
                return "uint"
            if operation.startswith("Load") and raw_args:
                component_count = self.byte_address_buffer_component_count(operation)
                if component_count is not None:
                    return self.byte_address_buffer_value_type(component_count)
            return None

        member_call = self.structured_buffer_member_call(function_expr)
        if member_call is not None:
            _, operation, buffer_type = member_call
            if operation == "Load" and raw_args:
                return self.structured_buffer_type_parts(buffer_type)[1]
            if operation == "Consume":
                return self.structured_buffer_type_parts(buffer_type)[1]
            if operation == "GetDimensions" and not raw_args:
                return "uint"
            return None

        func_name = getattr(function_expr, "name", function_expr)
        if func_name == "buffer_load" and raw_args:
            buffer_type = self.expression_result_type(raw_args[0])
            if self.byte_address_buffer_base_type(buffer_type) is not None:
                return "uint"
            parts = self.structured_buffer_type_parts(buffer_type)
            if parts is not None:
                return parts[1]
        if func_name == "buffer_consume" and raw_args:
            buffer_type = self.expression_result_type(raw_args[0])
            parts = self.structured_buffer_type_parts(buffer_type)
            if parts is not None:
                return parts[1]
        if func_name == "buffer_dimensions" and raw_args:
            buffer_type = self.expression_result_type(raw_args[0])
            if self.buffer_type_supports_length(buffer_type):
                return "uint"
        if func_name in self.structured_buffer_atomic_operations() and raw_args:
            target = self.structured_buffer_atomic_target(raw_args[0])
            if target is not None:
                return target["target_type"]
            target_type = self.expression_result_type(raw_args[0])
            if self.hip_atomic_scalar_kind(target_type) is not None:
                return target_type
        return None

    def canonical_sampled_resource_type(self, type_name):
        """Return the sampler spelling for HLSL-style sampled resources."""
        if not isinstance(type_name, str):
            return None
        base_type = type_name.split("[", 1)[0].split("<", 1)[0].strip()
        return self.sampled_resource_type_aliases.get(base_type)

    def canonical_storage_resource_type(self, type_name):
        """Return the image spelling for HLSL-style writable resources."""
        if not isinstance(type_name, str):
            return None

        base_type = type_name.split("[", 1)[0].strip()
        base_name = base_type.split("<", 1)[0].strip()
        image_type = self.storage_resource_type_aliases.get(base_name)
        if image_type is None:
            return None

        if "<" not in base_type or ">" not in base_type:
            return image_type

        value_type = base_type.split("<", 1)[1].rsplit(">", 1)[0].strip()
        value_type = value_type.split(",", 1)[0].strip().lower()
        if value_type in {"int", "i32"}:
            return f"i{image_type}"
        if value_type in {"uint", "u32"}:
            return f"u{image_type}"
        return image_type

    def canonical_resource_type(self, type_name):
        """Return the canonical sampler/image resource type for an alias."""
        if isinstance(type_name, str):
            base_type = type_name.split("[", 1)[0].split("<", 1)[0].strip()
            if base_type in {
                "accelerationStructureEXT",
                "AccelerationStructure",
                "acceleration_structure",
                "RaytracingAccelerationStructure",
                "RayTracingAccelerationStructure",
            }:
                return "RayTracingAccelerationStructure"
            sampler_type = self.sampler_state_type_aliases.get(base_type)
            if sampler_type is not None:
                return sampler_type
        return self.canonical_sampled_resource_type(
            type_name
        ) or self.canonical_storage_resource_type(type_name)

    def resource_base_type(self, type_name):
        """Normalize resource aliases before resource dispatch decisions."""
        base_type = ResourceDiagnosticMixin.resource_base_type(self, type_name)
        return self.canonical_resource_type(base_type) or base_type

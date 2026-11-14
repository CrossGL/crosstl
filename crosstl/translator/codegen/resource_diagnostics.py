"""Shared diagnostics mixin for reporting resource translation issues."""


class ResourceDiagnosticMixin:
    """Refusals for resource operations an emitter cannot lower.

    Each ``unsupported_*`` helper records the operation through the emitter's ``refuse_lowering``
    (:class:`~.lowering_refusals.LoweringRefusalMixin`), so the program is refused with a typed diagnostic. The
    neutral value it returns only lets generation reach the program's other refusals and never reaches output.
    """

    resource_diagnostic_backend = None
    zero_float2 = "make_float2(0.0f, 0.0f)"
    zero_float4 = "make_float4(0.0f, 0.0f, 0.0f, 0.0f)"

    def resource_backend_name(self):
        """Return the backend name used in refusal reasons."""
        return self.resource_diagnostic_backend or self.__class__.__name__

    def resource_base_type(self, type_name):
        """Strip array suffixes from a resource type string."""
        if not isinstance(type_name, str):
            return None
        return type_name.split("[", 1)[0]

    def is_multisample_resource_type(self, type_name):
        """Return whether a resource type denotes a multisample resource."""
        base_type = self.resource_base_type(type_name)
        return isinstance(base_type, str) and "MS" in base_type

    def is_shadow_resource_type(self, type_name):
        """Return whether a resource type denotes a shadow sampler."""
        base_type = self.resource_base_type(type_name)
        return isinstance(base_type, str) and "Shadow" in base_type

    def image_value_type(self, image_type):
        """Return the scalar/vector value type loaded from an image resource."""
        base_type = self.resource_base_type(image_type)
        if isinstance(base_type, str) and base_type.startswith("iimage"):
            return "int"
        if isinstance(base_type, str) and base_type.startswith("uimage"):
            return "uint"
        return "float4"

    def coord_component(self, coord, component):
        """Format one component access on a coordinate expression."""
        return f"{coord}.{component}"

    def surface_x_offset(self, coord, value_type):
        """Format a byte offset expression for surface x coordinates."""
        return f"{self.coord_component(coord, 'x')} * sizeof({value_type})"

    def zero_value_for_type(self, type_name):
        """Return a neutral generated expression for a simple type."""
        if type_name == "uint":
            return "0u"
        if type_name == "int":
            return "0"
        if type_name == "float4":
            return self.zero_float4
        return "0"

    def multisample_zero_value(self, func_name, resource_type):
        """Return a neutral fallback value for unsupported multisample calls."""
        if func_name == "imageStore":
            return "((void)0)"
        if func_name == "imageLoad":
            return self.zero_value_for_type(self.image_value_type(resource_type))
        return self.zero_float4

    def unsupported_multisample_resource_call(self, func_name, resource_type, _args):
        """Refuse a multisample operation the target cannot express."""
        return self.refuse_lowering(
            f"{self.resource_backend_name()} multisample resource call: "
            f"{func_name} on {resource_type}",
            self.multisample_zero_value(func_name, resource_type),
        )

    def unsupported_resource_query_call(self, func_name, resource_type, _args):
        """Refuse a resource query the target cannot express."""
        return self.refuse_lowering(
            f"{self.resource_backend_name()} resource query: {func_name} on {resource_type}",
            self.zero_float2,
        )

    def shadow_zero_value(self, func_name):
        """Return a neutral fallback value for unsupported shadow operations."""
        if func_name in {
            "textureGather",
            "textureGatherCompare",
            "textureGatherCompareOffset",
        }:
            return self.zero_float4
        return "0.0f"

    def unsupported_shadow_resource_call(self, func_name, resource_type, _args):
        """Refuse a shadow-sampler operation the target cannot express."""
        return self.refuse_lowering(
            f"{self.resource_backend_name()} shadow resource call: "
            f"{func_name} on {resource_type}",
            self.shadow_zero_value(func_name),
        )

    def unsupported_sampled_resource_call(self, func_name, resource_type, _args):
        """Refuse a sampled-resource operation the target cannot express."""
        return self.refuse_lowering(
            f"{self.resource_backend_name()} sampled resource call: "
            f"{func_name} on {resource_type}",
            self.zero_float4,
        )

    def image_atomic_zero_value(self, resource_type):
        """Return a neutral fallback value for image atomic operations."""
        base_type = self.resource_base_type(resource_type)
        if isinstance(base_type, str) and base_type.startswith("uimage"):
            return "0u"
        return "0"

    def unsupported_image_atomic_resource_call(self, func_name, resource_type, _args):
        """Refuse an image atomic the target cannot express."""
        resource_type = resource_type or "unknown resource"
        return self.refuse_lowering(
            f"{self.resource_backend_name()} image atomic resource call: "
            f"{func_name} on {resource_type}",
            self.image_atomic_zero_value(resource_type),
        )

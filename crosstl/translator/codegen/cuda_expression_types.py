"""Expression result types of the CUDA generator and canonical resource type names."""

from .cuda_subscripts import matrix_column_type
from .cuda_texel_buffers import TEXEL_BUFFER_QUERY_SPEC, sampled_texel_buffer_type
from .resource_diagnostics import ResourceDiagnosticMixin
from .resource_query import ResourceQueryMixin


class CudaExpressionTypeMixin:
    """Expression type members of ``CudaCodeGen``."""

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

    def array_access_element_type(self, type_name):
        """Return the element type of one subscript; a matrix subscript yields its column vector."""
        column_type = matrix_column_type(type_name)
        if column_type is not None:
            return column_type
        return super().array_access_element_type(type_name)

    def struct_member_lookup_type(self, type_name):
        """Return the struct key after CUDA pointer/reference wrappers."""
        return self.strip_cuda_indirect_type_qualifiers(type_name)

    def member_access_member_type(self, node):
        """Return the member type for ``object.field`` expressions."""
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
        """Return the member type for ``ptr->field`` expressions."""
        pointer_expr = getattr(node, "pointer_expr", None)
        pointer_type = self.expression_result_type(pointer_expr)
        pointer_info = self.cuda_indirect_type_info(pointer_type)
        if pointer_info is None:
            return None

        struct_type = self.struct_member_lookup_type(pointer_info["pointee_type"])
        return self.struct_member_types.get(struct_type, {}).get(
            getattr(node, "member", "")
        )

    def cuda_shadow_scalar_diagnostic_calls(self):
        """Return shadow compare calls that CUDA lowers to scalar diagnostics."""
        return {
            "textureCompareProj",
            "textureCompareProjOffset",
            "textureCompareProjLod",
            "textureCompareProjLodOffset",
            "textureCompareProjGrad",
            "textureCompareProjGradOffset",
        }

    def cuda_sampled_diagnostic_float4_calls(self):
        """Return sampled-resource calls that CUDA lowers to float4 diagnostics."""
        return {
            "textureOffset",
            "textureLodOffset",
            "textureGradOffset",
            "textureProj",
            "textureProjOffset",
            "textureProjLod",
            "textureProjLodOffset",
            "textureProjGrad",
            "textureProjGradOffset",
            "texelFetchOffset",
        }

    def cuda_sampled_vector_result_calls(self):
        """Return sampled texture calls whose value type follows the resource."""
        return {
            "texture",
            "textureLod",
            "textureGrad",
            "textureOffset",
            "textureLodOffset",
            "textureGradOffset",
            "textureProj",
            "textureProjOffset",
            "textureProjLod",
            "textureProjLodOffset",
            "textureProjGrad",
            "textureProjGradOffset",
            "texelFetch",
            "texelFetchOffset",
        }

    def cuda_image_atomic_calls(self):
        """Return image atomic calls that CUDA lowers to scalar diagnostics."""
        return {
            "imageAtomicAdd",
            "imageAtomicMin",
            "imageAtomicMax",
            "imageAtomicAnd",
            "imageAtomicOr",
            "imageAtomicXor",
            "imageAtomicExchange",
            "imageAtomicCompSwap",
        }

    def cuda_image_atomic_result_type(self, resource_type):
        """Return the scalar result type for CUDA image atomic diagnostics."""
        base_type = self.resource_base_type(resource_type)
        if not isinstance(base_type, str):
            return None
        if base_type.startswith("uimage"):
            return "uint"
        return "int"

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
            if self.cuda_atomic_scalar_kind(target_type) is not None:
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

    def canonical_ray_resource_type(self, type_name):
        if not isinstance(type_name, str):
            return None
        base_type = type_name.split("[", 1)[0].split("<", 1)[0].strip()
        if base_type in {
            "accelerationStructureEXT",
            "AccelerationStructure",
            "acceleration_structure",
            "RaytracingAccelerationStructure",
            "RayTracingAccelerationStructure",
        }:
            return "RayTracingAccelerationStructure"
        return None

    def resource_base_type(self, type_name):
        """Normalize resource aliases before resource dispatch decisions."""
        type_name = self.strip_cuda_indirect_type_qualifiers(type_name)
        base_type = ResourceDiagnosticMixin.resource_base_type(self, type_name)
        return (
            self.canonical_sampled_resource_type(base_type)
            or self.canonical_storage_resource_type(base_type)
            or self.canonical_ray_resource_type(base_type)
            or base_type
        )

    def query_type_name(self, type_name):
        """Return a CUDA-aware type string for resource query decisions."""
        return self.type_name_string(type_name)

    def dimension_query_spec(self, type_name):
        """Return CUDA resource query metadata for supported image shapes."""
        type_name = self.resource_base_type(type_name)
        if sampled_texel_buffer_type(type_name) is not None:
            return dict(TEXEL_BUFFER_QUERY_SPEC)
        if isinstance(type_name, str) and type_name.startswith(
            ("isampler", "usampler")
        ):
            # An integer texture has the dimensions of the float texture of its shape.
            type_name = self.sampled_texture_shape_type(type_name)
        spec = ResourceQueryMixin.dimension_query_spec(self, type_name)
        if spec is not None:
            return spec

        specs = {
            "image1D": (("width",), False, False),
            "iimage1D": (("width",), False, False),
            "uimage1D": (("width",), False, False),
            "image1DArray": (("width", "elements"), False, False),
            "iimage1DArray": (("width", "elements"), False, False),
            "uimage1DArray": (("width", "elements"), False, False),
            "iimageCube": (("width", "height"), False, False),
            "uimageCube": (("width", "height"), False, False),
            "imageCubeArray": (("width", "height", "elements"), False, False),
            "iimageCubeArray": (("width", "height", "elements"), False, False),
            "uimageCubeArray": (("width", "height", "elements"), False, False),
        }
        spec = specs.get(type_name)
        if spec is None:
            return None
        dimensions, mip, samples = spec
        return {"dimensions": dimensions, "mip": mip, "samples": samples}

    def is_queryable_resource_type(self, type_name):
        """Return whether a resource type can use CUDA query metadata."""
        if type_name is None:
            return False
        resource_type = self.resource_base_type(self.query_type_name(type_name))
        return self.dimension_query_spec(resource_type) is not None

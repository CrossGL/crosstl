"""Parameter classification for ``VulkanSPIRVCodeGen.process_function_node``.

Each source parameter of a CrossGL function becomes one of: a mesh output, a tessellation patch interface
variable, an entry-point output variable, or a runtime parameter whose SPIR-V type may be a pointer. The
classification reads the facts of the function and fills the lists, sets and maps of one
``SpirvFunctionParameterContext``.
"""

from dataclasses import dataclass


@dataclass
class SpirvFunctionParameterContext:
    """The facts every parameter of one function reads, and the lists, sets and maps its parameters fill."""

    execution_model_hint: object
    is_entry_point: bool
    entry_point_uses_void_signature: bool
    mesh_parameter_execution_model: object
    param_type_hints: dict
    storage_image_pointer_params: object
    mesh_output_parameters: dict
    mesh_output_parameter_indices: set
    patch_interface_parameters: list
    output_parameters: list
    runtime_parameters: list
    param_types: list
    param_value_types: list
    resource_array_param_indices: set
    value_array_param_indices: set


class SpirvFunctionParameterMixin:
    """Per-parameter classification of ``VulkanSPIRVCodeGen.process_function_node``."""

    def classify_spirv_function_parameter(self, signature, source_param_index, param):
        """Classify one source parameter as a mesh output, a patch interface variable, an entry-point output or a runtime parameter, and record its SPIR-V types and array and resource facts in ``signature``."""
        param_type_source = getattr(param, "param_type", getattr(param, "vtype", None))
        param_name = getattr(param, "name", None)
        mesh_output_info = self.mesh_output_parameter_info(
            param, signature.mesh_parameter_execution_model
        )
        if mesh_output_info is not None:
            signature.mesh_output_parameters[param_name] = mesh_output_info
            signature.mesh_output_parameter_indices.add(source_param_index)
            return

        patch_info = self.patch_parameter_info(
            param, signature.execution_model_hint, signature.is_entry_point
        )
        if patch_info is not None:
            signature.patch_interface_parameters.append((param, patch_info))
            return

        if param_name in signature.param_type_hints:
            param_type_source = signature.param_type_hints[param_name]
        if signature.is_entry_point and self.is_storage_resource_parameter(param):
            param_type = self.register_primitive_type("void")
            signature.param_value_types.append(param_type)
            signature.runtime_parameters.append((param, param_type, param_type))
            return
        if param_type_source is not None:
            param_type = self.map_resource_type_with_format(param_type_source, param)
        else:
            param_type = self.map_crossgl_type("float")

        if self.is_graphics_output_parameter(
            param, signature.execution_model_hint, signature.is_entry_point
        ):
            signature.output_parameters.append((param, param_type))
            return

        signature.param_value_types.append(param_type)
        param_resource_metadata = self.resource_type_metadata.get(param_type.id)
        is_stage_object_pointer_param = bool(
            getattr(param, "_spirv_stage_object_pointer", False)
        )
        is_storage_image_param = (
            param_resource_metadata is not None
            and param_resource_metadata.get("kind") == "storage_image"
            and param_name in signature.storage_image_pointer_params
        )
        is_value_array_param = self.array_type_info_from_type(param_type) is not None
        if (
            self.is_resource_array_type(param_type)
            or is_storage_image_param
            or is_value_array_param
            or is_stage_object_pointer_param
        ):
            signature.resource_array_param_indices.add(len(signature.param_types))
            if is_value_array_param and not (
                self.is_resource_array_type(param_type)
                or is_storage_image_param
                or is_stage_object_pointer_param
                or self.function_parameter_is_reference_like(param)
            ):
                signature.value_array_param_indices.add(len(signature.param_types))
            storage_class = (
                "UniformConstant"
                if self.is_resource_array_type(param_type) or is_storage_image_param
                else "Function"
            )
            param_type = self.register_pointer_type(param_type, storage_class)

        if not signature.entry_point_uses_void_signature:
            signature.param_types.append(param_type)
        signature.runtime_parameters.append(
            (param, param_type, signature.param_value_types[-1])
        )

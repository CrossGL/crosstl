"""Parts of a Metal function that ``MetalCodeGen.generate_function`` renders in turn: the parameter scope, the parameter declarations, the stage-specific entry parameters, the return type and the signature."""

import re

from ..ast import (
    PointerType,
)
from ..refusals import (
    MetalRefusal,
)
from .array_utils import (
    format_c_style_array_declaration,
)
from .image_access_contracts import (
    record_explicit_image_metadata,
)
from .metal_entry_return import (
    metal_entry_return_semantic_key,
    refuse_unreturnable_metal_entry_semantic,
)
from .metal_post_tessellation import (
    MetalPostTessellationError,
    post_tessellation_plan,
)
from .metal_raster_order_groups import metal_raster_order_group_attribute
from .metal_varyings import (
    metal_varying_attribute,
)
from .stage_utils import (
    requests_early_fragment_tests,
)


class MetalFunctionPartsMixin:
    """Function-rendering members of ``MetalCodeGen.generate_function``."""

    def prepare_metal_function_parameter_scope(
        self,
        func,
        shader_type,
        param_list,
        stage_output_parameters,
        vertex_stage_input_parameters,
    ):
        """Collect the lowered GLSL buffer-block parameters, check the stage resource bindings and record the local types of the stage input interfaces."""
        (
            self.current_glsl_buffer_block_parameters,
            self.current_glsl_buffer_block_parameter_failures,
            self.current_glsl_buffer_block_parameter_struct_failures,
        ) = self.collect_lowered_glsl_buffer_block_parameters(param_list)
        for param in param_list or []:
            block = self.current_glsl_buffer_block_parameters.get(
                getattr(param, "name", None)
            )
            if block is not None:
                block["memory"] = self.metal_parameter_memory_qualifier_prefix(
                    param, getattr(param, "param_type", None), "device"
                )
        if shader_type in {"vertex", "fragment", "compute", "ray_generation"}:
            self.validate_stage_parameter_resource_bindings(
                param_list, self.current_function_name
            )
        if shader_type in {"vertex", "fragment"}:
            self.validate_graphics_builtin_parameter_types(param_list, shader_type)
            self.validate_metal_fragment_invocation_density(func, shader_type)
        if shader_type == "compute":
            self.validate_compute_builtin_parameter_types(param_list)
        if shader_type in self.METAL_DERIVATIVE_FREE_FUNCTION_KINDS:
            self.validate_metal_kernel_derivatives(func, shader_type)
        self.current_unsupported_glsl_buffer_block_parameters = (
            self.collect_unsupported_glsl_buffer_block_parameter_names(param_list)
        )
        self.current_unsupported_glsl_buffer_block_local_variables = set()
        unsupported_metal_ray_function_table_parameter_diagnostics = []
        self.current_structured_buffer_length_parameters = {}
        self.current_structured_buffer_counter_parameters = {}
        self.current_metal_mesh_payload_parameter = None
        self.current_metal_mesh_payload_type = None
        self.current_metal_mesh_output_accumulators = {}
        self.current_metal_non_thread_payload_parameters = set()
        self.current_readonly_metal_mesh_payload_parameters = set()
        self.current_readonly_metal_mesh_payload_reasons = {}
        self.current_readonly_raw_buffer_parameters = set()
        self.current_readonly_metal_parameters = set()
        self.current_readonly_metal_parameter_reasons = {}
        self.current_metal_wave_lane_index_parameter = None
        self.current_metal_wave_lane_count_parameter = None
        self.current_metal_graphics_builtin_parameter_names = {}
        self.current_metal_compute_builtin_parameter_names = {}
        self.local_variable_types = {}
        self.current_metal_value_array_locals = self.metal_value_array_returned_locals(
            func
        )
        self.current_address_space_variables = {}
        self.current_metal_native_width_local_array_declaration_ids = (
            self.metal_native_width_local_array_declaration_ids(func)
        )
        if shader_type in {"vertex", "fragment"}:
            for interface in self.spirv_stage_input_layouts(shader_type):
                name = getattr(interface, "variable_name", None)
                raw_type = getattr(interface, "data_type", None)
                if not name or not raw_type or "[" in str(name):
                    continue
                mapped_type = self.map_type(raw_type)
                if shader_type == "vertex":
                    vertex_stage_input_parameters.append(
                        {
                            "name": name,
                            "raw_type": raw_type,
                            "mapped_type": mapped_type,
                            "attribute": self.spirv_stage_input_attribute(interface),
                            "semantic": self.spirv_stage_input_semantic(interface),
                        }
                    )
                    self.local_variable_types[name] = self.type_name_string(raw_type)

            for interface in self.spirv_stage_output_layouts(shader_type):
                name = getattr(interface, "variable_name", None)
                raw_type = getattr(interface, "data_type", None)
                if not name or not raw_type or "[" in str(name):
                    continue
                stage_output_parameters.append(
                    {
                        "name": name,
                        "raw_type": raw_type,
                        "mapped_type": self.map_type(raw_type),
                        "semantic": self.spirv_stage_output_semantic(interface),
                        "attribute": self.spirv_stage_output_attribute(interface),
                    }
                )
                self.local_variable_types[name] = self.type_name_string(raw_type)
        return unsupported_metal_ray_function_table_parameter_diagnostics

    def metal_function_parameter_declarations(
        self,
        func,
        shader_type,
        fragment_stage_input_parameters,
        image_format_parameters,
        param_list,
        params,
        reserved_parameter_names,
        sampler_parameter_array_sizes,
        sampler_parameters,
        stage_builtin_parameter_alias_declarations,
        stage_output_parameters,
        texture_parameter_array_sizes,
        texture_parameter_raw_types,
        texture_parameters,
        unsupported_metal_ray_function_table_parameter_diagnostics,
        value_array_parameter_copy_statements,
        vertex_stage_input_parameters,
    ):
        """Declare each parameter the function keeps, recording the sampler, texture, buffer and builtin facts the body needs."""
        skipped_parameter_indices = self.skipped_function_parameter_indices(
            self.current_function_name
        )
        for index, p in enumerate(param_list):
            if index in skipped_parameter_indices:
                continue
            if hasattr(p, "param_type"):
                raw_param_type = (
                    self.type_name_string(p.param_type)
                    if getattr(p.param_type, "generic_args", None)
                    else p.param_type
                )
            elif hasattr(p, "vtype"):
                raw_param_type = p.vtype
            else:
                raw_param_type = "float"
            self.local_variable_types[p.name] = self.type_name_string(raw_param_type)
            address_space = self.parameter_variable_address_space(
                raw_param_type, p, shader_type
            )
            if address_space is not None:
                self.current_address_space_variables[p.name] = address_space
            value_array_dimensions = None
            if shader_type is None:
                value_array_dimensions = self.metal_value_array_parameter_dimensions(
                    raw_param_type,
                    p,
                    self.map_resource_type_with_format(raw_param_type, p),
                )
                if (
                    value_array_dimensions is not None
                    and not self.metal_value_array_parameter_is_mutated(func, p.name)
                ):
                    value_array_dimensions = None
            if (
                value_array_dimensions is None
                and self.is_readonly_raw_buffer_parameter(
                    raw_param_type, p, shader_type
                )
            ):
                self.current_readonly_raw_buffer_parameters.add(p.name)
            readonly_parameter_reason = (
                None
                if value_array_dimensions is not None
                else self.readonly_metal_parameter_reason(
                    raw_param_type, p, shader_type
                )
            )
            if readonly_parameter_reason is not None:
                self.current_readonly_metal_parameters.add(p.name)
                self.current_readonly_metal_parameter_reasons[p.name] = (
                    readonly_parameter_reason
                )

            if self.is_metal_mesh_output_parameter(shader_type, p):
                continue

            table_array_kind = self.metal_ray_function_table_array_parameter_kind(
                raw_param_type, p
            )
            if table_array_kind is not None:
                self.unsupported_metal_ray_function_table_array_variables[p.name] = (
                    table_array_kind
                )
                unsupported_metal_ray_function_table_parameter_diagnostics.append(
                    self.unsupported_metal_ray_function_table_array_diagnostic(
                        table_array_kind, p.name
                    ).rstrip()
                )
                continue

            acceleration_structure_array = (
                self.metal_acceleration_structure_array_parameter_kind(
                    raw_param_type, p
                )
            )
            if acceleration_structure_array is not None:
                self.unsupported_metal_acceleration_structure_array_variables[
                    p.name
                ] = acceleration_structure_array
                unsupported_metal_ray_function_table_parameter_diagnostics.append(
                    self.unsupported_metal_acceleration_structure_array_diagnostic(
                        p.name
                    ).rstrip()
                )
                continue

            if self.is_sampler_type(raw_param_type):
                sampler_parameters.add(p.name)
                resource_array = self.resource_array_parameter(raw_param_type, p)
                if resource_array is not None:
                    _, array_size = resource_array
                    sampler_parameter_array_sizes[p.name] = array_size
            elif self.is_texture_or_image_resource_type(raw_param_type):
                texture_parameters[p.name] = self.map_resource_type_with_format(
                    self.resource_base_type(raw_param_type), p
                )
                texture_parameter_raw_types[p.name] = self.resource_base_type(
                    raw_param_type
                )
                resource_array = self.resource_array_parameter(raw_param_type, p)
                if resource_array is not None:
                    _, array_size = resource_array
                    texture_parameter_array_sizes[p.name] = array_size
                record_explicit_image_metadata(
                    p.name,
                    p,
                    self.attribute_value_to_string,
                    image_formats=image_format_parameters,
                )
            param_type = self.map_resource_type_with_format(raw_param_type, p)

            semantic = self.semantic_from_node(p)
            if self.is_graphics_stage_output_parameter(p, shader_type):
                stage_output_parameters.append(
                    {
                        "name": p.name,
                        "raw_type": raw_param_type,
                        "mapped_type": param_type,
                        "semantic": semantic,
                        "attribute": self.metal_stage_output_parameter_attribute(
                            p, shader_type
                        ),
                    }
                )
                continue
            if self.is_metal_mesh_payload_parameter(shader_type, p):
                self.current_metal_mesh_payload_parameter = p.name
                self.current_metal_mesh_payload_type = self.map_type(raw_param_type)
                self.current_address_space_variables[p.name] = "object_data"
                readonly_reason = self.readonly_metal_mesh_payload_parameter_reason(
                    shader_type, p
                )
                if readonly_reason is not None:
                    self.current_readonly_metal_mesh_payload_parameters.add(p.name)
                    self.current_readonly_metal_mesh_payload_reasons[p.name] = (
                        readonly_reason
                    )
            ray_payload_declaration = self.metal_ray_payload_parameter_declaration(
                param_type, p.name, p, shader_type
            )
            if ray_payload_declaration is not None:
                self.current_metal_non_thread_payload_parameters.add(p.name)
                ray_payload_address_space = self.metal_ray_payload_address_space(
                    shader_type
                )
                if ray_payload_address_space is not None:
                    self.current_address_space_variables[p.name] = (
                        ray_payload_address_space
                    )

            builtin_lowering = self.metal_graphics_builtin_parameter_lowering(
                raw_param_type,
                param_type,
                p,
                shader_type,
                reserved_parameter_names
                | set(params)
                | self.metal_function_local_variable_names(func),
            )
            if builtin_lowering is not None:
                params.append(
                    f"{builtin_lowering['type']} {builtin_lowering['name']} "
                    f"[[{builtin_lowering['attribute']}]]"
                )
                reserved_parameter_names.add(builtin_lowering["name"])
                stage_builtin_parameter_alias_declarations.append(
                    f"{param_type} {p.name} = {param_type}({builtin_lowering['name']});"
                )
                continue

            if self.is_plain_metal_vertex_stage_input_parameter(
                raw_param_type, param_type, p, shader_type
            ):
                vertex_stage_input_parameters.append(
                    {
                        "name": p.name,
                        "raw_type": raw_param_type,
                        "mapped_type": param_type,
                        "attribute": self.metal_vertex_input_location_attribute(p),
                        "semantic": semantic,
                    }
                )
                continue
            if self.is_plain_metal_fragment_stage_input_parameter(
                raw_param_type, param_type, p, shader_type
            ):
                fragment_stage_input_parameters.append(
                    {
                        "name": p.name,
                        "raw_type": raw_param_type,
                        "mapped_type": param_type,
                        "attribute": metal_varying_attribute(self, p, semantic),
                        "semantic": semantic,
                        "preserve_user_attribute": True,
                    }
                )
                continue

            param_attr = (
                self.metal_vertex_input_location_attribute(p)
                if shader_type == "vertex"
                else None
            )
            if param_attr is None and self.metal_vertex_input_takes_next_attribute(
                raw_param_type, shader_type, p, semantic
            ):
                # The [[stage_in]] structure gives a user-semantic input the next free [[attribute(n)]].
                param_attr = ""
            if param_attr is None:
                param_attr = self.parameter_attribute(
                    raw_param_type, semantic, shader_type, p
                )
            param_attr += self.metal_conditional_resource_attribute(p)
            if value_array_dimensions is not None:
                input_name = self.metal_value_array_input_name(
                    p.name,
                    reserved_parameter_names
                    | self.metal_function_local_variable_names(func),
                )
                reserved_parameter_names.add(input_name)
                declaration = format_c_style_array_declaration(param_type, input_name)
                value_array_parameter_copy_statements.extend(
                    self.metal_value_array_copy_statements(
                        param_type, p.name, input_name, value_array_dimensions
                    )
                )
            else:
                declaration = self.format_parameter_declaration(
                    raw_param_type, param_type, p.name, p, shader_type
                )
            if any(
                str(getattr(attribute, "name", "")).lower() == "maybe_unused"
                for attribute in getattr(p, "attributes", []) or []
            ):
                declaration = self.format_unused_metal_declaration(declaration)
            if self.should_wrap_metal_vertex_stage_input_parameter(
                raw_param_type, shader_type, p
            ):
                vertex_stage_input_parameters.append(
                    {
                        "name": p.name,
                        "declaration": declaration,
                        "attribute": param_attr,
                        "semantic": semantic,
                    }
                )
                continue
            param_attr += metal_raster_order_group_attribute(
                p, shader_type, declaration
            )
            params.append(f"{declaration}{param_attr}")
            if self.is_structured_buffer_type(
                raw_param_type
            ) and self.structured_buffer_parameter_requires_length(
                self.current_function_name, p.name
            ):
                length_name = self.structured_buffer_length_parameter_name(p.name)
                self.current_structured_buffer_length_parameters[p.name] = length_name
                array_size = self.structured_buffer_parameter_array_size(
                    raw_param_type, p
                )
                params.append(
                    self.format_structured_buffer_length_parameter(p.name, array_size)
                )
            if self.structured_buffer_requires_counter(raw_param_type):
                counter_name = self.structured_buffer_counter_parameter_name(p.name)
                self.current_structured_buffer_counter_parameters[p.name] = counter_name
                array_size = self.structured_buffer_parameter_array_size(
                    raw_param_type, p
                )
                params.append(
                    self.format_structured_buffer_counter_parameter(p.name, array_size)
                )

    def metal_stage_entry_parameters(
        self,
        func,
        shader_type,
        param_list,
        params,
        reserved_parameter_names,
    ):
        """Add the stage inputs, builtins and outputs a vertex, fragment, compute or ray-generation entry point receives."""
        if shader_type == "vertex":
            explicit_stage_builtins = self.explicit_graphics_stage_builtin_parameters(
                param_list, shader_type
            )
            if explicit_stage_builtins.get("vertex_id"):
                self.current_metal_graphics_builtin_parameter_names[
                    "gl_VertexIndex"
                ] = explicit_stage_builtins["vertex_id"]
            if explicit_stage_builtins.get("instance_id"):
                self.current_metal_graphics_builtin_parameter_names[
                    "gl_InstanceIndex"
                ] = explicit_stage_builtins["instance_id"]
            reserved_builtin_names = set(reserved_parameter_names)
            reserved_builtin_names.update(
                self.metal_function_local_variable_names(func)
            )
            for (
                builtin_name,
                name,
                param_type,
                attribute,
            ) in self.required_metal_vertex_builtin_parameters(
                func, reserved_builtin_names, explicit_stage_builtins
            ):
                params.append(f"{param_type} {name} [[{attribute}]]")
                reserved_parameter_names.add(name)
                self.current_metal_graphics_builtin_parameter_names[builtin_name] = name

        if shader_type == "fragment":
            explicit_stage_builtins = self.explicit_graphics_stage_builtin_parameters(
                param_list, shader_type
            )
            if explicit_stage_builtins.get("position"):
                self.current_metal_graphics_builtin_parameter_names["gl_FragCoord"] = (
                    explicit_stage_builtins["position"]
                )
            reserved_builtin_names = set(reserved_parameter_names)
            reserved_builtin_names.update(
                self.metal_function_local_variable_names(func)
            )
            for (
                builtin_name,
                name,
                param_type,
                attribute,
            ) in self.required_metal_fragment_builtin_parameters(
                func, reserved_builtin_names, explicit_stage_builtins
            ):
                params.append(f"{param_type} {name} [[{attribute}]]")
                reserved_parameter_names.add(name)
                self.current_metal_graphics_builtin_parameter_names[builtin_name] = name

        if shader_type in {"compute", "ray_generation"}:
            existing_param_names = {getattr(p, "name", None) for p in param_list}
            explicit_stage_builtins = self.explicit_compute_stage_builtin_parameters(
                param_list
            )
            self.current_metal_compute_builtin_parameter_names.update(
                {
                    builtin_name: explicit_stage_builtins[attribute]
                    for builtin_name, _param_type, attribute in (
                        self.compute_builtin_parameter_specs()
                    )
                    if attribute in explicit_stage_builtins
                    and explicit_stage_builtins[attribute]
                }
            )
            self.current_metal_wave_lane_index_parameter = explicit_stage_builtins.get(
                "thread_index_in_simdgroup"
            )
            self.current_metal_wave_lane_count_parameter = explicit_stage_builtins.get(
                "threads_per_simdgroup"
            )
            reserved_builtin_names = set(existing_param_names)
            reserved_builtin_names.update(
                self.metal_function_local_variable_names(func)
            )
            for name, param_type, attribute in self.required_compute_builtin_parameters(
                func, reserved_builtin_names, explicit_stage_builtins
            ):
                if name not in existing_param_names:
                    params.append(f"{param_type} {name} [[{attribute}]]")
                    reserved_parameter_names.add(name)
                builtin_name = self.compute_builtin_name_for_metal_attribute(attribute)
                if builtin_name is not None:
                    self.current_metal_compute_builtin_parameter_names[builtin_name] = (
                        name
                    )
                if attribute == "thread_index_in_simdgroup":
                    self.current_metal_wave_lane_index_parameter = name
                elif attribute == "threads_per_simdgroup":
                    self.current_metal_wave_lane_count_parameter = name

    def metal_function_return_plan(
        self,
        func,
        shader_type,
        params,
        reserved_parameter_names,
        stage_output_parameters,
    ):
        """Join the parameters and choose the Metal return type, with the stage output struct an entry point returns."""
        params_str = ", ".join(params)
        if shader_type is None:
            params_str = self.append_required_stage_output_parameter(
                params_str, self.current_function_name
            )
            params_str = self.append_required_stage_parameter_parameters(
                params_str, self.current_function_name
            )
            params_str = self.append_required_cbuffer_parameters(
                params_str, self.current_function_name
            )
            params_str = self.append_required_thread_private_global_parameters(
                params_str, self.current_function_name
            )
            params_str = self.append_required_global_resource_parameters(
                params_str, self.current_function_name
            )
            params_str = self.append_metal_mesh_dispatch_context_parameters(
                params_str,
                func,
                reserved_parameter_names,
            )
            params_str = self.append_required_metal_wave_lane_parameters(
                params_str, self.current_function_name
            )
            for parameter in self.required_function_stage_parameters(
                self.current_function_name
            ):
                self.local_variable_types[parameter.name] = self.type_name_string(
                    self.parameter_raw_type(parameter)
                )
            stage_output_type = self.required_function_stage_output_type(
                self.current_function_name
            )
            if stage_output_type is not None:
                self.local_variable_types["output"] = self.type_name_string(
                    stage_output_type
                )

        if hasattr(func, "return_type"):
            effective_return_type = self.metal_effective_function_return_type(func)
            raw_return_type = self.type_name_string(effective_return_type)
            if "[" in raw_return_type:
                # Metal cannot return C arrays; ``array<T, N>`` keeps the
                # value semantics of a source ``metal::array`` result.
                return_type = self.metal_value_array_type(raw_return_type)
                if return_type is None:
                    raise MetalRefusal(
                        "Metal output does not support unsized C-style array "
                        "return types; wrap the array in a struct or use an "
                        "output parameter"
                    )
            elif isinstance(effective_return_type, PointerType):
                return_type = self.format_metal_pointer_return_type(
                    effective_return_type
                )
            else:
                return_type = self.map_type(raw_return_type)
        else:
            effective_return_type = "void"
            raw_return_type = "void"
            return_type = "void"
        if shader_type in {"vertex", "fragment"}:
            self.validate_function_return_semantic_type(
                func, raw_return_type, shader_type
            )
            if (
                return_type != "void"
                and str(raw_return_type) not in self.structs_by_name
            ):
                return_semantic = self.function_return_semantic(func)
                refuse_unreturnable_metal_entry_semantic(
                    self,
                    func,
                    shader_type,
                    return_semantic,
                    metal_entry_return_semantic_key(
                        self, func, return_semantic, shader_type
                    ),
                    return_type,
                    in_structure=bool(stage_output_parameters),
                )
            if stage_output_parameters and return_type != "void":
                stage_output_parameters.append(
                    self.metal_stage_return_value_output(
                        func,
                        shader_type,
                        raw_return_type,
                        return_type,
                        stage_output_parameters,
                    )
                )
        return params_str, effective_return_type, raw_return_type, return_type

    def metal_function_signature(
        self,
        func,
        shader_type,
        execution_config,
        entry_name,
        stage_node,
        code,
        fragment_stage_input_parameters,
        params_str,
        raw_return_type,
        reserved_parameter_names,
        return_type,
        stage_output_parameters,
        vertex_stage_input_parameters,
        fragment_stage_input_alias_declarations,
        stage_output_struct_name,
        vertex_stage_input_alias_declarations,
    ):
        """Write the stage attribute and the signature of the entry point or function."""
        if shader_type == "vertex":
            function_name = entry_name or f"vertex_{func.name}"
            patch_attribute = ""
            if post_tessellation_plan(func) is not None:
                if vertex_stage_input_parameters:
                    raise MetalPostTessellationError(
                        function_name,
                        "it reads stage inputs besides its patch, and a Metal post-tessellation vertex "
                        "function has one [[stage_in]] structure",
                    )
                structures, patch_parameters, patch_attribute, preamble = (
                    self.metal_post_tessellation_interface(
                        func, function_name, reserved_parameter_names
                    )
                )
                code += structures
                for declaration in reversed(patch_parameters):
                    params_str = (
                        f"{declaration}, {params_str}" if params_str else declaration
                    )
                vertex_stage_input_alias_declarations = preamble
            if vertex_stage_input_parameters:
                stage_input_struct_name = self.unique_vertex_stage_input_struct_name(
                    function_name
                )
                stage_input_parameter_name = self.unique_metal_generated_name(
                    "_crossglInput",
                    reserved_parameter_names
                    | self.metal_function_local_variable_names(func),
                )
                self.local_variable_types[stage_input_parameter_name] = (
                    stage_input_struct_name
                )
                code += self.generate_metal_vertex_stage_input_parameter_struct(
                    stage_input_struct_name, vertex_stage_input_parameters
                )
                params_str = self.append_parameter_declaration(
                    params_str,
                    f"{stage_input_struct_name} {stage_input_parameter_name} "
                    "[[stage_in]]",
                )
                vertex_stage_input_alias_declarations = (
                    self.generate_metal_vertex_stage_input_alias_declarations(
                        stage_input_parameter_name, vertex_stage_input_parameters
                    )
                )
            params_str = self.append_global_resource_parameters(
                params_str,
                self.current_function_name,
                func,
                filter_writable_textures=True,
            )
            if stage_output_parameters:
                stage_output_struct_name = self.unique_return_wrapper_struct_name(
                    function_name
                )
                code += self.generate_metal_stage_output_parameter_struct(
                    stage_output_struct_name, stage_output_parameters
                )
                return_type = stage_output_struct_name
                self.current_function_return_wrapper = None
            else:
                return_wrapper = self.function_return_semantic_wrapper(
                    func, raw_return_type, return_type, shader_type, function_name
                )
                if return_wrapper is not None:
                    code += self.generate_return_wrapper_struct(return_wrapper)
                    return_type = return_wrapper["struct_name"]
                self.current_function_return_wrapper = return_wrapper
            code += f"{patch_attribute}vertex {return_type} {function_name}({params_str}) {{\n"
        elif shader_type == "fragment":
            function_name = entry_name or f"fragment_{func.name}"
            if fragment_stage_input_parameters:
                stage_input_struct_name = self.unique_vertex_stage_input_struct_name(
                    function_name
                )
                stage_input_parameter_name = self.unique_metal_generated_name(
                    "_crossglInput",
                    reserved_parameter_names
                    | self.metal_function_local_variable_names(func),
                )
                self.local_variable_types[stage_input_parameter_name] = (
                    stage_input_struct_name
                )
                code += self.generate_metal_vertex_stage_input_parameter_struct(
                    stage_input_struct_name, fragment_stage_input_parameters
                )
                params_str = self.append_parameter_declaration(
                    params_str,
                    f"{stage_input_struct_name} {stage_input_parameter_name} "
                    "[[stage_in]]",
                )
                fragment_stage_input_alias_declarations = (
                    self.generate_metal_vertex_stage_input_alias_declarations(
                        stage_input_parameter_name, fragment_stage_input_parameters
                    )
                )
            params_str = self.append_global_resource_parameters(
                params_str, self.current_function_name, func, shader_type="fragment"
            )
            if stage_output_parameters:
                stage_output_struct_name = self.unique_return_wrapper_struct_name(
                    function_name
                )
                code += self.generate_metal_stage_output_parameter_struct(
                    stage_output_struct_name, stage_output_parameters
                )
                return_type = stage_output_struct_name
                self.current_function_return_wrapper = None
            else:
                return_wrapper = self.function_return_semantic_wrapper(
                    func, raw_return_type, return_type, shader_type, function_name
                )
                if return_wrapper is not None:
                    code += self.generate_return_wrapper_struct(return_wrapper)
                    return_type = return_wrapper["struct_name"]
                self.current_function_return_wrapper = return_wrapper
            # MSL's [[early_fragment_tests]] precedes the fragment function (MSL 4.1 section 5.1.2).
            early_tests = (
                "[[early_fragment_tests]] "
                if requests_early_fragment_tests(func, stage_node)
                else ""
            )
            code += f"{early_tests}fragment {return_type} {function_name}({params_str}) {{\n"
        elif shader_type in ["compute", "ray_generation"]:
            params_str = self.append_global_resource_parameters(
                params_str, self.current_function_name, func
            )
            function_name = entry_name or self.stage_entry_base_name(shader_type, func)
            self.validate_metal_kernel_return_type(func, shader_type, raw_return_type)
            code += f"kernel {return_type} {function_name}({params_str}) {{\n"
        elif shader_type in ["mesh", "object", "task", "amplification"]:
            stage_keyword = "mesh" if shader_type == "mesh" else "object"
            function_name = entry_name or f"{stage_keyword}_{func.name}"
            params_str = self.append_global_resource_parameters(
                params_str, self.current_function_name, func
            )
            stage_attribute = self.metal_mesh_stage_attribute(
                shader_type, func, execution_config
            )
            generated_mesh_payload_parameter = (
                self.generated_metal_mesh_payload_parameter(
                    shader_type, func, reserved_parameter_names
                )
            )
            if generated_mesh_payload_parameter is not None:
                payload_type, payload_name = generated_mesh_payload_parameter
                params_str = self.append_parameter_declaration(
                    params_str,
                    f"object_data {payload_type}& {payload_name} [[payload]]",
                )
                self.current_metal_mesh_payload_parameter = payload_name
                self.current_metal_mesh_payload_type = payload_type
                self.current_address_space_variables[payload_name] = "object_data"
            mesh_grid_properties_parameter = self.metal_mesh_grid_properties_parameter(
                shader_type, func, reserved_parameter_names
            )
            if mesh_grid_properties_parameter is not None:
                params_str = self.append_parameter_declaration(
                    params_str,
                    f"mesh_grid_properties {mesh_grid_properties_parameter}",
                )
                self.current_metal_mesh_grid_properties_parameter = (
                    mesh_grid_properties_parameter
                )
            mesh_output = self.metal_mesh_stage_output_config(
                shader_type, func, function_name, reserved_parameter_names
            )
            if mesh_output is not None:
                if mesh_output.get("generated_vertex_struct", False):
                    code += self.generate_metal_mesh_vertex_output_struct(mesh_output)
                params_str = self.append_parameter_declaration(
                    params_str,
                    self.metal_mesh_stage_output_parameter_declaration(mesh_output),
                )
                self.current_metal_mesh_output_parameter = mesh_output["parameter_name"]
                self.current_metal_mesh_output_config = mesh_output
                self.validate_metal_mesh_output_usage(func)
            code += (
                f"{stage_attribute} {return_type} {function_name}({params_str}) {{\n"
            )
        elif shader_type in [
            "ray_any_hit",
            "ray_closest_hit",
            "ray_miss",
            "ray_callable",
            "anyhit",
            "closesthit",
            "miss",
            "callable",
        ]:
            rt_stage_map = {
                "ray_any_hit": "anyhit",
                "ray_closest_hit": "closesthit",
                "ray_miss": "miss",
                "ray_callable": "callable",
                "anyhit": "anyhit",
                "closesthit": "closesthit",
                "miss": "miss",
                "callable": "callable",
            }
            stage_keyword = rt_stage_map[shader_type]
            function_name = entry_name or f"{stage_keyword}_{func.name}"
            code += f"[[visible]] {return_type} {function_name}({params_str}) {{\n"
        elif shader_type in ["ray_intersection", "intersection"]:
            function_name = entry_name or f"intersection_{func.name}"
            stage_attribute = self.metal_ray_intersection_stage_attribute(
                func, raw_return_type
            )
            code += (
                f"{stage_attribute} {return_type} "
                f"{function_name}({params_str}) {{\n"
            )
        else:
            semantic = self.semantic_from_node(func)
            function_name = entry_name or func.name
            semantic_attr = self.map_non_stage_function_semantic(semantic)
            # Lowered helpers can become unused after native operation selection.
            if (
                getattr(func, "linkage", None) == "internal"
                and getattr(func, "is_inline", False)
            ) or re.fullmatch(
                r"CrossGLMetalVectorIndex_[iu](?:8|16)vec[234]_set", function_name
            ):
                code += "__attribute__((unused))\n"
            code += (
                f"{self.metal_function_linkage_prefix(func)}"
                f"{return_type} {function_name}({params_str}){semantic_attr} {{\n"
            )
        return (
            function_name,
            code,
            vertex_stage_input_alias_declarations,
            stage_output_struct_name,
            return_type,
            fragment_stage_input_alias_declarations,
        )

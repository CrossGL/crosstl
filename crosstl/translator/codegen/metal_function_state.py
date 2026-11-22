"""Per-function state that ``MetalCodeGen.generate_function`` saves on entry and restores on exit.

Emitting a function rebinds many ``MetalCodeGen`` attributes (parameter maps, buffer-block facts, mesh and wave
builtin bindings). The attribute groups below record which ones a call saves, and when it restores them, so the
values of an enclosing emission survive a nested one.
"""

# Saved on entry and restored on every exit, including the declaration-only and refused exits.
METAL_FUNCTION_STATE_ATTRIBUTES = (
    "current_function_name",
    "current_metal_argument_table",
    "current_function_return_type",
    "current_function_return_wrapper",
    "current_metal_value_array_locals",
    "current_metal_value_array_names",
    "current_local_identifier_remaps",
    "local_variable_types",
    "current_address_space_variables",
    "current_generic_function_substitutions",
    "cbuffer_parameter_names",
    "cbuffer_member_references",
    "ambiguous_cbuffer_members",
    "current_glsl_buffer_block_parameters",
    "current_unsupported_glsl_buffer_block_parameters",
    "current_unsupported_glsl_buffer_block_local_variables",
    "current_glsl_buffer_block_parameter_failures",
    "current_glsl_buffer_block_parameter_struct_failures",
    "unsupported_metal_ray_function_table_array_variables",
    "unsupported_metal_acceleration_structure_array_variables",
    "current_structured_buffer_length_parameters",
    "current_structured_buffer_counter_parameters",
    "current_metal_mesh_output_config",
    "current_metal_mesh_output_parameter",
    "current_metal_mesh_grid_properties_parameter",
    "current_metal_mesh_payload_parameter",
    "current_metal_mesh_payload_type",
    "current_metal_mesh_output_accumulators",
    "current_metal_non_thread_payload_parameters",
    "current_readonly_metal_mesh_payload_parameters",
    "current_readonly_metal_mesh_payload_reasons",
    "current_readonly_raw_buffer_parameters",
    "current_readonly_metal_parameters",
    "current_readonly_metal_parameter_reasons",
    "current_metal_wave_lane_index_parameter",
    "current_metal_wave_lane_count_parameter",
    "current_metal_graphics_builtin_parameter_names",
    "current_metal_compute_builtin_parameter_names",
)

# Saved on entry and restored only after a body is emitted.
METAL_FUNCTION_BODY_STATE_ATTRIBUTES = (
    "current_metal_stage_output_return_active",
    "current_metal_native_width_local_array_declaration_ids",
)

# Saved just before the body is emitted, once the texture and sampler parameter maps are known.
METAL_FUNCTION_RESOURCE_PARAMETER_ATTRIBUTES = (
    "current_sampler_parameters",
    "current_sampler_parameter_array_sizes",
    "current_texture_parameters",
    "current_texture_parameter_raw_types",
    "current_texture_parameter_array_sizes",
    "current_texture_alias_sources",
    "current_image_format_parameters",
)

# Saved as a shallow copy, because emission mutates these dictionaries in place.
METAL_FUNCTION_COPIED_STATE_ATTRIBUTES = frozenset(
    {
        "unsupported_metal_acceleration_structure_array_variables",
        "unsupported_metal_ray_function_table_array_variables",
    }
)

# Saved with a default, because a code generator may not have set them yet.
METAL_FUNCTION_STATE_DEFAULTS = {"current_metal_stage_output_return_active": False}


def save_metal_function_state(codegen, attributes):
    """Return the current values of ``attributes`` on ``codegen`` in attribute order."""
    state = {}
    for name in attributes:
        if name in METAL_FUNCTION_STATE_DEFAULTS:
            value = getattr(codegen, name, METAL_FUNCTION_STATE_DEFAULTS[name])
        else:
            value = getattr(codegen, name)
        if name in METAL_FUNCTION_COPIED_STATE_ATTRIBUTES:
            value = dict(value)
        state[name] = value
    return state


def restore_metal_function_state(codegen, state):
    """Rebind every attribute saved by ``save_metal_function_state``."""
    for name, value in state.items():
        setattr(codegen, name, value)

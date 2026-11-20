"""Per-function state that ``HLSLCodeGen.generate_function`` saves on entry and restores on exit.

Emitting a function rebinds many ``HLSLCodeGen`` attributes (local types, identifier aliases, resource-pointer
bindings, buffer-block facts). The attribute groups below record which ones a call saves, and when it restores
them, so the values of an enclosing emission survive a nested one.
"""

from dataclasses import dataclass

# Saved on entry and restored on every exit, including the declaration-only and refused exits.
HLSL_FUNCTION_STATE_ATTRIBUTES = (
    "current_function_return_type",
    "current_hlsl_precise_return_type",
    "local_variable_types",
    "current_identifier_aliases",
    "current_identifier_reserved_names",
    "current_hlsl_stage_output_lowering",
    "current_hlsl_semantic_stage",
    "current_function_name",
    "current_hlsl_tail_recursive_call_ids",
    "current_hlsl_tail_recursive_function",
    "current_hlsl_resource_pointer_offsets",
    "current_hlsl_resource_pointer_aliases",
    "current_generic_function_substitutions",
    "current_glsl_buffer_block_parameters",
    "current_unsupported_glsl_buffer_block_parameters",
    "current_unsupported_glsl_buffer_block_local_variables",
    "current_glsl_buffer_block_parameter_failures",
    "current_glsl_buffer_block_parameter_struct_failures",
)

# Saved on entry and restored only after a body is emitted.
HLSL_FUNCTION_BODY_STATE_ATTRIBUTES = ("local_variable_source_types",)

# Saved just before the body is emitted, once the texture and sampler parameter maps are known.
HLSL_FUNCTION_RESOURCE_PARAMETER_ATTRIBUTES = (
    "current_sampler_parameters",
    "current_texture_parameters",
    "current_image_access_parameters",
    "current_image_format_parameters",
    "current_implicit_texture_samplers",
    "current_implicit_texture_regular_samplers",
    "current_implicit_texture_query_lod_samplers",
)


def save_hlsl_function_state(codegen, attributes):
    """Return the current values of ``attributes`` on ``codegen`` in attribute order."""
    return {name: getattr(codegen, name) for name in attributes}


def restore_hlsl_function_state(codegen, state):
    """Rebind every attribute saved by ``save_hlsl_function_state``."""
    for name, value in state.items():
        setattr(codegen, name, value)


@dataclass
class HLSLFunctionParameterContext:
    """The facts every parameter of one function reads, and the lists and maps its parameters fill.

    ``params`` collects the rendered parameter declarations; the prologue and epilogue lists collect the
    statements the body runs before and after its own statements. The sampler, texture and image maps record
    which parameters are resources, and ``emitted_param_names`` the names the signature already uses.
    """

    func: object
    param_list: list
    param_names: set
    emitted_param_names: set
    function_name: object
    effective_shader_type: object
    execution_config: object
    promoted_entry_resource_parameter_ids: set
    default_vertex_parameter_semantics: dict
    params: list
    parameter_prologue_statements: list
    parameter_epilogue_statements: list
    sampler_parameters: set
    texture_parameters: dict
    image_access_parameters: dict
    image_format_parameters: dict
    comparison_sampler_parameters: set
    implicit_texture_samplers: dict
    implicit_existing_comparison_samplers: set

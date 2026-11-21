"""Sections of a rendered Metal program, in output order: the prelude with includes, enum constants, constants and structs; the program-scope variables; the function and stage dependency tables; the functions and stage entry points; and the helpers they require."""

from ..ast import (
    FunctionCallNode,
)
from ..source_licenses import (
    source_license_comments,
)
from .constant_ordering import (
    partition_constants_by_struct_dependency,
)
from .enum_utils import (
    generate_enum_constants,
    generate_generic_enum_constants,
)
from .exponent_builtins import (
    emit_metal_exponent_helpers,
)
from .extended_arithmetic_builtins import (
    emit_metal_extended_arithmetic_helpers,
)
from .generic_emission_order import (
    mark_generic_function_emitted,
)
from .generic_function_utils import (
    generic_function_definition_name,
    generic_function_emission_list,
    generic_function_parameters,
)
from .metal_bit_scans import (
    emit_metal_bit_scan_helpers,
)
from .metal_buffer_coherence import (
    collect_inherited_buffer_contracts,
)
from .metal_constant_initializers import (
    folded_helper_is_unused,
)
from .metal_debug_printf import (
    uses_debug_printf,
)
from .metal_global_emission import (
    MetalGlobalEmission,
)
from .metal_hlsl_matrix import (
    metal_hlsl_matrix_prelude,
)
from .metal_safe_math import (
    METAL_SAFE_MATH_PRAGMA,
)
from .metal_texture_coherence import (
    collect_device_coherent_textures,
)
from .metal_wave_extended import (
    emit_metal_wave_extension_helpers,
)
from .stage_utils import (
    function_stage_name,
    normalize_stage_name,
    order_functions_by_dependencies,
    should_emit_qualified_function,
    stage_matches,
)
from .texture_residency import (
    emit_metal_texture_residency_helpers,
)


class MetalProgramSectionMixin:
    """Section-rendering members of ``MetalCodeGen.render_rewritten_program``."""

    def render_metal_program_prelude(
        self,
        ast,
        all_functions,
        global_vars,
        structs,
    ):
        """The program's prelude: license comments, includes, namespaces, enum constants, constants, structs and stage-IO helpers."""
        code = "\n"
        code += source_license_comments(ast, "metal")
        if getattr(self, "metal_safe_math", False):
            code += METAL_SAFE_MATH_PRAGMA + "\n"
        preprocessors = getattr(ast, "preprocessors", []) or []
        pre_lines = []
        for directive in preprocessors:
            line = self.generate_preprocessor_directive(directive)
            if line:
                pre_lines.append(line)
        if (
            self.requires_metal_system_thread_scope
            or self.requires_metal_resource_coherence
        ) and not any("#pragma metal internals" in line.lower() for line in pre_lines):
            code += "#pragma METAL internals : enable\n"
        if pre_lines:
            code += "\n".join(pre_lines) + "\n"
        if not any("metal_stdlib" in line for line in pre_lines):
            code += "#include <metal_stdlib>\n"
        if self.requires_metal_atomic_fence and not any(
            "metal_atomic" in line for line in pre_lines
        ):
            code += "#include <metal_atomic>\n"
        if uses_debug_printf(self, ast) and not any(
            "metal_logging" in line for line in pre_lines
        ):
            code += "#include <metal_logging>\n"
        if self.uses_cooperative_matrix(ast) and not any(
            "metal_simdgroup_matrix" in line for line in pre_lines
        ):
            code += "#include <metal_simdgroup_matrix>\n"
        code += "using namespace metal;\n"
        if self.uses_metal_raytracing_namespace(ast, global_vars, all_functions):
            code += "using namespace metal::raytracing;\n"
        code += "\n"
        self.metal_prelude_insertion_index = len(code)
        if self.requires_metal_system_thread_scope:
            code += self.generate_metal_system_thread_scope_support()
        code += generate_enum_constants(
            self,
            self.plain_enums + self.struct_payload_enums,
            qualifier=self.metal_unused_declaration_qualifier("constant"),
        )
        code += generate_generic_enum_constants(
            self,
            self.generic_enum_struct_definitions,
            qualifier=self.metal_unused_declaration_qualifier("constant"),
        )
        leading_constants, struct_dependent_constants = (
            partition_constants_by_struct_dependency(
                getattr(ast, "constants", []) or [], structs
            )
        )
        code += self.generate_constants(ast, leading_constants)
        code += self.generate_metal_builtin_limit_fallbacks(ast, all_functions)
        code += self.generate_metal_struct_declarations(structs)
        code += self.generate_constants(
            ast, struct_dependent_constants, include_function_constants=False
        )
        code += self.generate_metal_stage_io_array_helpers()
        code += self.generate_metal_enum_constructor_functions(
            self.struct_payload_enums
        )
        code += self.generate_metal_generic_enum_constructor_functions(
            self.generic_enum_specializations
        )
        return code

    def render_metal_program_scope_variables(
        self,
        code,
        global_vars,
    ):
        """Reserve the cbuffer and explicit bindings, then declare every program-scope variable after ``code``."""
        used_resource_bindings = {}
        source_resource_bindings = {}
        self.pre_reserve_cbuffer_bindings(
            used_resource_bindings, source_resource_bindings
        )
        self.reserve_explicit_global_resource_bindings(
            global_vars, used_resource_bindings, source_resource_bindings
        )
        self.reserve_cbuffer_bindings(
            used_resource_bindings,
            source_resource_bindings,
        )
        emission = MetalGlobalEmission(
            code,
            used_resource_bindings,
            source_resource_bindings,
        )
        for node in global_vars:
            self.declare_metal_program_scope_variable(emission, node)
        code = emission.code
        return code

    def collect_metal_program_function_dependencies(
        self,
        ast,
        target_stage,
        all_functions,
        code,
        generic_function_specializations,
        global_vars,
    ):
        """Collect each function's resource, stage-parameter and stage-output dependencies and the coherence contracts, and append the cbuffer and structured-element definitions."""
        self.function_global_resource_dependencies = (
            self.collect_function_global_resource_dependencies(
                [f for f in all_functions if not generic_function_parameters(f)]
                + list(generic_function_specializations.values())
            )
        )
        self.function_stage_parameter_dependencies = (
            self.collect_function_stage_parameter_dependencies(ast, target_stage)
        )
        self.function_stage_output_dependencies = (
            self.collect_function_stage_output_dependencies(ast, target_stage)
        )
        self.metal_device_coherent_textures = collect_device_coherent_textures(
            [entry[0] for entry in self.texture_variables],
            all_functions,
            is_storage_texture=self.is_metal_storage_texture_node,
            contracts_of=lambda node: self.resource_memory_qualifier_contracts(
                node, getattr(node, "param_type", getattr(node, "var_type", None))
            ),
            call_name=self.function_call_name,
            iter_nodes=self.iter_ast_nodes,
            refuse=self.refuse_lowering,
        )
        self.metal_inherited_buffer_contracts = collect_inherited_buffer_contracts(
            global_vars,
            all_functions,
            contracts_of=lambda node: self.resource_memory_qualifier_contracts(
                node, getattr(node, "param_type", getattr(node, "var_type", None))
            ),
            call_name=self.function_call_name,
            iter_nodes=self.iter_ast_nodes,
        )

        cbuffers = self.cbuffer_variables
        if cbuffers:
            code += "// Constant Buffers\n"
            code += self.generate_cbuffers(ast)
        code += self.metal_hlsl_structured_element_definitions()

        stage_entry_names = self.stage_entry_names(ast, target_stage)
        self.metal_compute_only_program = self.metal_program_is_compute_only(
            ast, target_stage
        )
        return code, stage_entry_names

    def metal_ordering_callee_name(self, call):
        """The function a call depends on for emission order.

        The parser keeps an explicit call's type arguments in the callee name (``twice<float>``), so the call names
        the generic definition ``twice`` once they are removed; otherwise a caller defined before the generic was
        emitted ahead of the specialization it calls.
        """
        return generic_function_definition_name(self, self.function_call_name(call))

    def render_metal_program_functions(
        self,
        ast,
        target_stage,
        stage_entry_names,
    ):
        """Generate the program's functions in dependency order and its stage entry points."""
        functions = order_functions_by_dependencies(
            getattr(ast, "functions", []),
            self.iter_ast_nodes,
            self.metal_ordering_callee_name,
            FunctionCallNode,
            include_overloads=True,
        )
        functions_code = ""
        for func in functions:
            qualifier_name = function_stage_name(func)

            if not should_emit_qualified_function(target_stage, qualifier_name):
                continue

            if generic_function_parameters(func):
                mark_generic_function_emitted(self, func)
                for specialized_func in generic_function_emission_list(self, func):
                    functions_code += self.generate_function(specialized_func)
                continue

            if qualifier_name == "vertex":
                functions_code += "// Vertex Shader\n"
                functions_code += self.generate_function(
                    func,
                    shader_type="vertex",
                    entry_name=stage_entry_names.get(id(func)),
                )
            elif qualifier_name == "fragment":
                functions_code += "// Fragment Shader\n"
                functions_code += self.generate_function(
                    func,
                    shader_type="fragment",
                    entry_name=stage_entry_names.get(id(func)),
                )
            elif qualifier_name == "compute":
                functions_code += "// Compute Shader\n"
                functions_code += self.generate_function(
                    func,
                    shader_type="compute",
                    entry_name=stage_entry_names.get(id(func)),
                    stage_local_variables=self.metal_stage_local_variables_with_lowered_program_groupshared(
                        func
                    ),
                )
            elif not folded_helper_is_unused(self, func):
                functions_code += self.generate_function(func)

        if hasattr(ast, "stages") and ast.stages:
            for stage_type, stage in ast.stages.items():
                stage_name = normalize_stage_name(stage_type)
                if not stage_matches(target_stage, stage_name):
                    continue

                for func in order_functions_by_dependencies(
                    getattr(stage, "local_functions", []) or [],
                    self.iter_ast_nodes,
                    self.metal_ordering_callee_name,
                    FunctionCallNode,
                ):
                    if generic_function_parameters(func):
                        mark_generic_function_emitted(self, func)
                        for specialized_func in generic_function_emission_list(
                            self,
                            func,
                        ):
                            functions_code += self.generate_function(specialized_func)
                    else:
                        functions_code += self.generate_function(func)

                if hasattr(stage, "entry_point"):
                    functions_code += f"// {stage_name.title()} Shader\n"
                    functions_code += self.generate_function(
                        stage.entry_point,
                        shader_type=stage_name,
                        execution_config=getattr(stage, "execution_config", None),
                        entry_name=stage_entry_names.get(id(stage.entry_point)),
                        stage_local_variables=self.metal_stage_local_variables_with_lowered_program_groupshared(
                            stage.entry_point,
                            getattr(stage, "local_variables", []),
                        ),
                        stage_node=stage,
                    )
        return functions_code

    def append_metal_program_helpers(
        self,
        code,
        functions_code,
    ):
        """Append the helpers the generated code requires, then the functions, and return the program."""
        code += self.generate_image_atomic_compare_helpers()
        code += self.metal_texel_buffers.code()
        code += self.generate_buffer_atomic_compare_helpers()
        code += self.generate_pointer_atomic_compare_helpers()
        code += self.generate_metal_wave_helpers()
        code += emit_metal_wave_extension_helpers(self)
        code += emit_metal_exponent_helpers(self)
        code += emit_metal_bit_scan_helpers(self)
        code += emit_metal_texture_residency_helpers(self)
        code += emit_metal_extended_arithmetic_helpers(self)
        code += self.generate_metal_ray_query_helpers()
        code += self.generate_glsl_buffer_aggregate_load_helpers()
        code += self.generate_metal_inverse_helpers()
        code += self.metal_matrix_builtins.code()
        code += self.generate_metal_integer_dot_helper()
        code += self.metal_array_arguments.code()
        code += functions_code
        if self.metal_emulated_matrix_shapes:
            index = self.metal_prelude_insertion_index
            code = (
                code[:index]
                + metal_hlsl_matrix_prelude(self.metal_emulated_matrix_shapes)
                + code[index:]
            )
        return code

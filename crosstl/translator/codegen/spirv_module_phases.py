"""Phases of ``VulkanSPIRVCodeGen._generate_module``.

The phases emit the module header and base types, declare the types and constants, collect the functions and
their metadata, declare the globals, and emit the helper and entry functions in order.
"""

from ..ast import ConstantNode, EnumNode, StructNode
from ..host_workgroup_size import function_entry_execution_config
from .array_utils import collect_literal_int_constants
from .enum_utils import (
    collect_generic_enum_specializations,
    collect_generic_enum_struct_definitions,
    collect_generic_parameter_names,
)
from .generic_function_utils import (
    generic_function_emission_list,
    generic_function_parameters,
    prepare_generic_function_specializations,
)
from .spirv_entry_workgroup_sizes import plan_entry_workgroup_sizes
from .spirv_mesh_builtins import is_glsl_mesh_builtin_output_declaration
from .spirv_specialized_workgroup_size import module_compute_stage_config


class SpirvModulePhaseMixin:
    """Module-emission phases of ``VulkanSPIRVCodeGen._generate_module``."""

    def begin_spirv_module(
        self,
        ast,
    ):
        """Emit the module header, capability, GLSL.std.450 import and memory model, register the base types, and collect the literal integer constants, struct and enum declarations and generic type facts of the program."""
        self.emit("; SPIR-V")
        self.emit("; Version: 1.0")
        self.emit("; Generator: CrossGL Vulkan SPIR-V Generator")
        self.emit("; Schema: 0")

        self.emit("OpCapability Shader")

        self.glsl_std450_id = self.get_id()
        self.emit(f'%{self.glsl_std450_id} = OpExtInstImport "GLSL.std.450"')

        self.emit("OpMemoryModel Logical GLSL450")

        self.register_primitive_type("void")
        self.register_primitive_type("bool")
        self.register_primitive_type("int")
        self.register_primitive_type("float")

        literal_integer_declarations = [
            constant
            for constant in getattr(ast, "constants", []) or []
            if not self.spirv_specialization_constant_attributes(constant)
        ]
        self.literal_int_constants = collect_literal_int_constants(
            literal_integer_declarations
        )

        float_type = self.primitive_types["float"]
        for i in range(2, 5):
            self.register_vector_type(float_type, i)

        struct_declarations = list(getattr(ast, "structs", []) or [])
        for stage in (getattr(ast, "stages", None) or {}).values():
            struct_declarations.extend(getattr(stage, "local_structs", []) or [])

        self.struct_declarations = {}
        self.enum_declarations = {}

        def collect_type_declarations(nodes):
            for node in nodes or []:
                if isinstance(node, StructNode):
                    self.struct_declarations.setdefault(node.name, node)
                    collect_type_declarations(getattr(node, "members", []) or [])
                elif isinstance(node, EnumNode):
                    self.enum_declarations.setdefault(node.name, node)

        collect_type_declarations(struct_declarations)
        # Generic parameters are unbound where a generic body is read; instances naming them stay generic.
        self.program_generic_parameter_names = frozenset(
            collect_generic_parameter_names(ast)
        ) - set(self.struct_declarations)
        self.generic_struct_instance_spellings = {}
        self.glsl_buffer_block_type_names = self.collect_glsl_buffer_block_type_names(
            ast
        )

        self.generic_enum_struct_definitions = collect_generic_enum_struct_definitions(
            struct_declarations
        )
        self.generic_enum_specializations = collect_generic_enum_specializations(
            ast,
            self.generic_enum_struct_definitions,
            self.type_name_string,
        )

    def declare_spirv_module_types_and_constants(
        self,
        ast,
    ):
        """Declare the specialization constants first, then the enum metadata, the structs of the program and of each stage, and the remaining named constants."""
        preprocessed_specialization_constant_declarations = set()
        for constant in getattr(ast, "constants", []) or []:
            if not isinstance(constant, ConstantNode):
                continue
            if not self.spirv_specialization_constant_attributes(constant):
                continue
            self.process_named_constant_declaration(constant)
            preprocessed_specialization_constant_declarations.add(id(constant))
        for var in getattr(ast, "global_variables", []) or []:
            declaration = self.spirv_specialization_constant_declaration_node(var)
            if declaration is None:
                continue
            self.process_named_constant_declaration(declaration)
            preprocessed_specialization_constant_declarations.add(id(declaration))

        self.collect_enum_metadata(ast.structs)
        for stage in (getattr(ast, "stages", None) or {}).values():
            self.collect_enum_metadata(getattr(stage, "local_structs", []) or [])
        self.register_generic_enum_metadata()

        self.process_enum_structs(ast.structs)
        for stage in (getattr(ast, "stages", None) or {}).values():
            self.process_enum_structs(getattr(stage, "local_structs", []) or [])

        for struct in ast.structs:
            if not isinstance(struct, StructNode):
                continue
            self.process_crossgl_struct(struct)
        for stage in (getattr(ast, "stages", None) or {}).values():
            for struct in getattr(stage, "local_structs", []) or []:
                if not isinstance(struct, StructNode):
                    continue
                self.process_crossgl_struct(struct)

        for constant in getattr(ast, "constants", []) or []:
            if (
                isinstance(constant, ConstantNode)
                and id(constant)
                not in preprocessed_specialization_constant_declarations
            ):
                self.process_named_constant_declaration(constant)

    def collect_spirv_module_functions(
        self,
        ast,
    ):
        """Record the source variable types, the global specialization-constant declarations and every function of the program, including generic specializations, and refuse the generic functions the target cannot express."""
        self.source_variable_type_names = self.collect_source_variable_type_names(ast)

        global_specialization_constant_declarations = {
            id(declaration)
            for var in getattr(ast, "global_variables", []) or []
            for declaration in [
                self.spirv_specialization_constant_declaration_node(var)
            ]
            if declaration is not None
        }

        all_functions = self.collect_ast_functions(ast)
        generic_function_specializations = prepare_generic_function_specializations(
            self,
            all_functions,
            global_types=self.source_variable_type_names,
            ast=ast,
        )
        if generic_function_specializations:
            all_functions = all_functions + list(
                generic_function_specializations.values()
            )
        self.reject_unsupported_generic_functions(ast)
        self.function_nodes_by_name = self.collect_functions_by_name(all_functions)
        return global_specialization_constant_declarations, all_functions

    def collect_spirv_function_metadata(
        self,
        ast,
        all_functions,
    ):
        """Collect the per-function resource-array, image-access, storage-buffer, execution-model, skipped-parameter, stage-object and storage-image facts, reserve explicit bindings and find the tessellation control stage."""
        self.function_resource_array_type_hints = (
            self.collect_resource_array_parameter_type_hints(ast)
        )
        self.collect_stage_local_resource_array_parameter_type_hints(ast)
        self.function_image_access_requirements = (
            self.collect_function_image_access_requirements_for_ast(ast)
        )
        self.collect_stage_local_function_image_access_metadata(ast)
        self.function_storage_buffer_access_requirements = (
            self.collect_function_storage_buffer_access_requirements_for_functions(
                all_functions
            )
        )
        self.collect_stage_local_function_storage_buffer_metadata(ast)
        self.inline_storage_buffer_functions = (
            self.collect_inline_storage_buffer_functions(all_functions)
        )
        self.function_execution_models = self.collect_function_execution_models(
            ast, all_functions
        )
        (
            self.spirv_skipped_function_parameter_indices,
            self.spirv_skipped_function_parameter_indices_by_id,
        ) = self.collect_unused_array_parameter_index_maps(all_functions)
        self.function_stage_input_dependencies = (
            self.collect_function_stage_object_dependencies(ast, None, "input")
        )
        self.function_stage_output_dependencies = (
            self.collect_function_stage_object_dependencies(ast, None, "output")
        )
        self.function_storage_image_pointer_params = (
            self.collect_storage_image_pointer_parameters(ast)
        )
        self.collect_stage_local_storage_image_pointer_metadata(ast)
        self.reserve_explicit_resource_bindings(ast)
        for stage_type, stage in (getattr(ast, "stages", None) or {}).items():
            if self.stage_key(stage_type) == "tessellation_control":
                self.tessellation_control_stage = stage
                break

    def declare_spirv_module_globals(
        self,
        ast,
        global_specialization_constant_declarations,
    ):
        """Declare the constant buffers, the default uniform block and the global variables that are not specialization constants."""
        for cbuffer in getattr(ast, "cbuffers", []) or []:
            self.process_cbuffer_declaration(cbuffer)
        self.declare_default_uniform_block(ast)

        for var in getattr(ast, "global_variables", []):
            declaration = self.spirv_specialization_constant_declaration_node(var)
            if (
                declaration is not None
                and id(declaration) in global_specialization_constant_declarations
            ):
                continue
            self.process_global_variable_declaration(var)
        if not getattr(ast, "stages", None):
            self.bind_default_uniform_block()

    def emit_spirv_helper_functions(
        self,
        ast,
    ):
        """Split the top-level functions into entry points and helpers, plan the compute workgroup sizes, find a tessellation control entry, and emit the helpers in dependency order; return the entry points with their qualifiers."""
        top_level_entries = []
        helper_functions = []
        for func in ast.functions:
            if getattr(func, "body", None) is None:
                continue
            if generic_function_parameters(func):
                helper_functions.extend(generic_function_emission_list(self, func))
                continue
            qualifier = self.get_function_qualifier(func)

            if func.name == "main" or qualifier in [
                "vertex",
                "fragment",
                "compute",
                "geometry",
                "tessellation_control",
                "tessellation_evaluation",
                "mesh",
                "task",
                "object",
                "amplification",
                "ray_generation",
                "ray_intersection",
                "ray_closest_hit",
                "ray_miss",
                "ray_any_hit",
                "ray_callable",
            ]:
                top_level_entries.append((func, qualifier))
            else:
                helper_functions.append(func)

        compute_entry_configs = [
            function_entry_execution_config(func)
            for func, qualifier in top_level_entries
            if qualifier == "compute"
        ]
        if compute_entry_configs and all(
            config == compute_entry_configs[0] for config in compute_entry_configs
        ):
            # Built-ins read outside a stage take the size of the module's compute entries.
            self.module_compute_entry_config = compute_entry_configs[0]
        elif not compute_entry_configs:
            self.module_compute_entry_config = module_compute_stage_config(self, ast)
        plan_entry_workgroup_sizes(self, ast, top_level_entries)

        if self.tessellation_control_stage is None:
            for func, qualifier in top_level_entries:
                if qualifier == "tessellation_control":
                    self.tessellation_control_stage = func
                    break

        for func in self.order_functions_by_dependencies(helper_functions):
            if self.function_has_storage_buffer_parameters(func):
                continue
            self.process_function_node(func)
        return top_level_entries

    def emit_spirv_entry_functions(
        self,
        ast,
        top_level_entries,
    ):
        """Declare the stage-local variables and functions and emit each stage's entry function, or emit the top-level entry functions of a program without stages; return the entry points to declare."""
        entry_points = []

        if getattr(ast, "stages", None):
            for stage_type, stage in ast.stages.items():
                previous_execution_model = self.current_execution_model
                previous_stage = self.current_stage
                self.current_stage = stage
                self.current_execution_model = self.spirv_execution_model(
                    self.stage_key(stage_type)
                )
                try:
                    for var in getattr(stage, "local_variables", []):
                        if is_glsl_mesh_builtin_output_declaration(self, var):
                            continue
                        if not self.global_variable_exists_for_current_stage(var.name):
                            self.process_global_variable_declaration(var)
                finally:
                    self.current_execution_model = previous_execution_model
                    self.current_stage = previous_stage

            self.bind_default_uniform_block()
            self.entry_point_resource_binding_baseline = self.resource_binding_state()

            processed_local_functions = set()
            for stage in ast.stages.values():
                local_functions = [
                    func
                    for func in getattr(stage, "local_functions", [])
                    if id(func) not in processed_local_functions
                    and getattr(func, "body", None) is not None
                ]
                for func in self.order_functions_by_dependencies(local_functions):
                    if id(func) not in processed_local_functions:
                        if generic_function_parameters(func):
                            for specialized_func in generic_function_emission_list(
                                self, func
                            ):
                                self.process_function_node(
                                    specialized_func, stage=stage
                                )
                            processed_local_functions.add(id(func))
                            continue
                        if self.function_has_storage_buffer_parameters(func):
                            processed_local_functions.add(id(func))
                            continue
                        self.process_function_node(func, stage=stage)
                        processed_local_functions.add(id(func))

            for stage_type, stage in ast.stages.items():
                entry_function = stage.entry_point
                function_id = self.process_function_node(entry_function, stage=stage)
                stage_name = self.stage_key(stage_type)
                execution_model = self.spirv_execution_model(stage_name)
                entry_points.append(
                    (execution_model, function_id, entry_function.name, stage)
                )
        else:
            self.entry_point_resource_binding_baseline = self.resource_binding_state()
            for func, qualifier in top_level_entries:
                function_id = self.process_function_node(func)
                execution_model = self.spirv_execution_model(qualifier)
                is_tessellation_entry = execution_model in {
                    "TessellationControl",
                    "TessellationEvaluation",
                }
                stage_metadata = (
                    func
                    if is_tessellation_entry or execution_model == "GLCompute"
                    else None
                )
                entry_points.append(
                    (execution_model, function_id, func.name, stage_metadata)
                )
        return entry_points

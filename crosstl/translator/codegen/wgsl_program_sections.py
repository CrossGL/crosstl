"""Steps of ``WGSLCodeGen.generate_program`` after the AST rewrites: resetting the per-program state, collecting the struct, constant, resource and function tables, recording module variable types and binding access modes, and rendering the module sections in output order."""

from ..ast import (
    NamedType,
)
from ..refusals import (
    WGSLRefusal,
)
from ..semantics.enum_values import (
    enum_values,
)
from .enum_utils import (
    collect_enum_type_names,
    collect_enum_variant_constants,
)
from .global_specialization_constants import (
    split_global_specialization_constants,
)
from .stage_utils import (
    normalize_stage_name,
)
from .wgsl_enums import (
    generate_enum_constants,
    split_enums,
)
from .wgsl_frexp_modf import (
    wgsl_frexp_modf_helpers,
)
from .wgsl_generic_functions import (
    WGSLGenericFunctionSpecializer,
)
from .wgsl_helper_stages import helper_stage_names
from .wgsl_matrix_inverse import (
    wgsl_matrix_inverse_helpers,
)
from .wgsl_matrix_resize import (
    wgsl_matrix_resize_helpers,
)
from .wgsl_primitive_index import (
    PRIMITIVE_INDEX_ENABLE_DIRECTIVE,
)
from .wgsl_read_only_globals import (
    read_only_module_names,
)
from .wgsl_specialization_constants import (
    WGSLSpecializationPlan,
)
from .wgsl_specialized_workgroup_size import (
    WorkgroupSizeOverrides,
)
from .wgsl_stage_io import (
    vertex_output_interpolation,
)
from .wgsl_uniform_registers import (
    plan_uniform_registers,
    wgsl_uniform_register_helpers,
)


class WGSLProgramSectionMixin:
    """Program-rendering members of ``WGSLCodeGen.generate_program``."""

    def reset_wgsl_program_state(
        self,
        ast,
    ):
        """Reset every per-program table before a new module is generated."""
        self._workgroup_size_overrides = WorkgroupSizeOverrides(ast)
        self._location_counters = {"in": 0, "out": 0, "generic": 0}
        self._next_implicit_binding_by_group = {}
        self._identifier_scopes = []
        self._identifier_alias_scopes = []
        self._pointer_identifier_scopes = []
        self._parameter_plan_stack = []
        self._value_type_scopes = []
        self._resource_alias_scopes = []
        self._function_return_types = {}
        self._function_return_types_by_signature = {}
        self._current_expression_expected_type = None
        self._current_function_return_type = None
        self._module_identifier_names = {}
        self._function_identifier_names = {}
        self._function_signature_identifier_names = {}
        self._function_overloads_by_name = {}
        self._type_identifier_names = {}
        self._struct_member_identifier_names = {}
        self._structs_by_name = {}
        self._struct_member_types = {}
        self._generic_struct_definitions = {}
        self._generic_struct_specializations = {}
        self._generic_struct_specialized_names = {}
        self._cbuffer_member_types = {}
        self._struct_resource_paths = {}
        self._glsl_buffer_block_struct_names = set()
        self._uniform_buffer_struct_names = set()
        self._uniform_scalar_array_wrappers = {}
        self._module_variable_types = {}
        self._module_storage_access_modes = {}
        self._used_access_mode_identifiers = set()
        self._read_only_module_names = set()
        self._module_storage_texture_access_modes = {}
        self._inferred_storage_texture_access_modes = {}
        self._module_resource_bindings = {}
        self._sampled_texture_array_bindings = {}
        self._sampled_texture_array_dispatch_helpers = {}
        self._frexp_modf_helpers = {}
        self._matrix_inverse_helpers = {}
        self._matrix_resize_helpers = {}
        self._uniform_registers = None
        self._half_conversion_helpers = {}
        self._constant_values = {}
        self._specialization_plan = WGSLSpecializationPlan({}, set(), set())
        self._specialization_default_values = {}
        self._function_resource_member_parameters = {}
        self._reserved_bindings_by_group = {}
        self._allocated_bindings_by_group = {}
        self._reserved_sampler_bindings = {}
        self._explicit_binding_owners = {}
        self._explicit_binding_register_classes = {}
        self._hlsl_register_binding_allocations = {}
        self._hlsl_register_source_owners = {}
        self._stage_output_lowerings = {}
        self._stage_interface_member_locations = {}
        self._vertex_output_interpolation = {}
        self._stage_interface_struct_names = set()
        self._fragment_output_struct_names = set()
        self._stage_interface_structs_by_name = {}
        self._entry_parameter_locations = {}
        self._stage_interface_builtin_members = {}
        self._builtin_option_specializations = {}
        self._generic_function_specializer = None
        self._uses_primitive_index = False
        self._int_builtin_members = {}

    def collect_wgsl_program_tables(
        self,
        ast,
        target_stage,
    ):
        """Collect the structs, enums, constants, constant buffers, globals, stage resources and functions, with the identifier and parameter metadata derived from them."""
        source_structs = self._collect_structs(ast, target_stage)
        self._generic_struct_definitions = self.collect_generic_struct_definitions(
            source_structs
        )
        structs = self.monomorphized_structs(ast, target_stage, source_structs)
        structs, enums = split_enums(structs)
        self._enum_type_names = collect_enum_type_names(enums)
        self._enum_variant_constants = collect_enum_variant_constants(enums)
        constants, _ = split_global_specialization_constants(ast)
        self._constant_values = self.constant_literal_values(constants)
        self._stage_output_lowerings = self.stage_output_lowerings(
            ast, target_stage, structs
        )
        if self._stage_output_lowerings:
            structs = [
                *structs,
                *(
                    lowering["struct"]
                    for lowering in self._stage_output_lowerings.values()
                ),
            ]
        self._stage_interface_structs_by_name = {
            getattr(struct, "name", ""): struct for struct in structs
        }
        self._stage_interface_builtin_members = self.stage_interface_builtin_members(
            ast,
            target_stage,
            self._stage_interface_structs_by_name,
        )
        self._stage_interface_member_locations = self.stage_interface_member_locations(
            ast, target_stage, structs
        )
        self.plan_specialization_constants(ast, structs)
        self._vertex_output_interpolation = vertex_output_interpolation(
            self, self._stage_nodes(ast, target_stage)
        )
        cbuffers = self._collect_cbuffers(ast, target_stage)
        global_variable_nodes = self._collect_global_variables(ast, target_stage)
        stage_resource_parameters = self._collect_stage_resource_parameters(
            ast, target_stage
        )
        helper_function_nodes = list(self._helper_functions(ast, target_stage))
        self._helper_stage_names = helper_stage_names(
            ast, self._stage_nodes(ast, target_stage)
        )
        all_function_nodes = list(helper_function_nodes)
        all_function_nodes.extend(
            stage_node.entry_point
            for stage_node in self._stage_nodes(ast, target_stage)
            if getattr(stage_node, "entry_point", None) is not None
        )
        self._builtin_option_specializations = (
            self.collect_builtin_option_specializations(ast)
        )
        generic_functions = WGSLGenericFunctionSpecializer(
            self, helper_function_nodes, ast
        )
        concrete_helper_function_nodes = [
            func
            for func in helper_function_nodes
            if not generic_functions.is_generic(func)
        ]

        self.collect_struct_type_metadata(structs)
        self._struct_member_types.update(
            self.builtin_option_specialization_member_types(
                self._builtin_option_specializations
            )
        )
        self._used_access_mode_identifiers = self.used_access_mode_identifiers(
            global_variable_nodes,
            stage_resource_parameters,
            all_function_nodes,
        )
        self.collect_identifier_metadata(
            structs,
            cbuffers,
            constants,
            global_variable_nodes,
            stage_resource_parameters,
            concrete_helper_function_nodes,
        )
        self.collect_cbuffer_member_identifier_metadata(cbuffers)
        self._cbuffer_member_accesses = self.cbuffer_member_accesses(cbuffers)
        self._cbuffer_member_types = self.cbuffer_member_types(cbuffers)
        self._function_texture_parameters = self.function_texture_parameters(
            ast, target_stage
        )
        self._function_pointer_parameters = self.function_buffer_pointer_parameters(
            ast, target_stage
        )
        self._function_resource_member_parameters = (
            self.function_resource_member_parameters(ast, target_stage)
        )
        return (
            structs,
            cbuffers,
            global_variable_nodes,
            stage_resource_parameters,
            all_function_nodes,
            generic_functions,
            concrete_helper_function_nodes,
            enums,
            constants,
        )

    def collect_wgsl_module_types_and_bindings(
        self,
        all_function_nodes,
        cbuffers,
        constants,
        global_variable_nodes,
        stage_resource_parameters,
        structs,
    ):
        """Record module variable and function return types, buffer and storage access modes, and the explicit resource bindings."""
        self._module_variable_types = {}
        for constant in constants:
            name = getattr(constant, "name", "")
            if not name:
                continue
            const_type = getattr(
                constant, "const_type", getattr(constant, "vtype", None)
            )
            if const_type is None and getattr(constant, "value", None) is not None:
                const_type = self.expression_type(constant.value)
            if const_type is not None:
                self._module_variable_types[name] = const_type
        for node in global_variable_nodes:
            name = getattr(node, "name", "")
            if not name:
                continue
            var_type = getattr(node, "var_type", None)
            if var_type is None and getattr(node, "initial_value", None) is not None:
                var_type = self.expression_type(node.initial_value)
                if var_type is None:
                    raise WGSLRefusal(
                        "WGSL target cannot infer the type of untyped global "
                        f"{name} from its initializer"
                    )
                node.var_type = var_type
            self._module_variable_types[name] = var_type
        self._module_variable_types.update(
            {
                getattr(parameter, "name", ""): self.stage_resource_module_type(
                    parameter
                )
                for parameter in stage_resource_parameters
                if getattr(parameter, "name", "")
            }
        )
        self._module_variable_types.update(
            {
                self.cbuffer_instance_name(cbuffer): NamedType(cbuffer.name)
                for cbuffer in cbuffers
            }
        )
        self._function_return_types = {}
        self._function_return_types_by_signature = {}
        for function in all_function_nodes:
            function_name = getattr(function, "name", "")
            if not function_name:
                continue
            signature_key = self.function_signature_key(function)
            self._function_return_types_by_signature[signature_key] = getattr(
                function, "return_type", None
            )
            if len(self._function_overloads_by_name.get(function_name, ())) <= 1:
                self._function_return_types[function_name] = getattr(
                    function, "return_type", None
                )
        self._glsl_buffer_block_struct_names = (
            self.collect_glsl_buffer_block_struct_names(
                global_variable_nodes, all_function_nodes
            )
        )
        # Blocks whose typed declaration WGSL would lay out elsewhere are read from registers
        # (wgsl_uniform_registers); only the typed blocks need uniform forms of their structs.
        self._uniform_registers = plan_uniform_registers(self, cbuffers, structs)
        register_blocks = (
            self._uniform_registers.blocks if self._uniform_registers else {}
        )
        typed_cbuffers = [
            cbuffer for cbuffer in cbuffers if cbuffer.name not in register_blocks
        ]
        self._uniform_buffer_struct_names = self.uniform_buffer_struct_names(
            typed_cbuffers, global_variable_nodes, stage_resource_parameters
        )
        self._uniform_scalar_array_wrappers = self.uniform_scalar_array_wrapper_names(
            typed_cbuffers, structs
        )
        self._module_storage_access_modes = self.module_storage_access_modes(
            global_variable_nodes, stage_resource_parameters
        )
        self._read_only_module_names = read_only_module_names(
            self, cbuffers, global_variable_nodes, stage_resource_parameters
        )
        self._inferred_storage_texture_access_modes = (
            self.inferred_storage_texture_access_modes(
                global_variable_nodes,
                stage_resource_parameters,
                all_function_nodes,
            )
        )
        self.reserve_explicit_resource_bindings(
            cbuffers, global_variable_nodes, stage_resource_parameters
        )

    def render_wgsl_module_declarations(
        self,
        ast,
        cbuffers,
        concrete_helper_function_nodes,
        constants,
        emitted_sections,
        enums,
        generic_functions,
        global_variable_nodes,
        stage_resource_parameters,
        structs,
    ):
        """Render the option, enum, struct, constant, constant buffer, resource, global and helper function sections."""
        uniform_scalar_array_wrappers = self.generate_uniform_scalar_array_wrappers()
        if uniform_scalar_array_wrappers:
            emitted_sections.append(uniform_scalar_array_wrappers)

        builtin_option_declarations = self.generate_builtin_option_declarations()
        if builtin_option_declarations:
            emitted_sections.append(builtin_option_declarations)

        if enums:
            emitted_sections.append(
                generate_enum_constants(self, enums, enum_values(ast))
            )

        if structs:
            emitted_sections.append(
                "\n\n".join(self.generate_struct(node) for node in structs)
            )

        constants = [self.generate_constant(node) for node in constants]
        if constants:
            emitted_sections.append("\n".join(constants))

        if cbuffers:
            emitted_sections.append(
                "\n\n".join(self.generate_cbuffer(node) for node in cbuffers)
            )

        stage_resources = [
            self.generate_stage_resource_parameter(node)
            for node in stage_resource_parameters
        ]
        if stage_resources:
            emitted_sections.append("\n".join(stage_resources))

        global_variables = [
            self.generate_global_variable(node) for node in global_variable_nodes
        ]
        if global_variables:
            emitted_sections.append("\n".join(global_variables))

        self._generic_function_specializer = generic_functions
        helper_functions = [
            self.generate_function(func)
            for func in concrete_helper_function_nodes
            if func
        ]
        helper_section_index = len(emitted_sections)
        if helper_functions:
            emitted_sections.append("\n\n".join(helper_functions))
        return helper_functions, helper_section_index

    def render_wgsl_stage_functions_and_helpers(
        self,
        ast,
        target_stage,
        emitted_sections,
        generic_functions,
        helper_functions,
        helper_section_index,
        lines,
    ):
        """Render the stage entry points, pending generic specializations and the helpers they need, then append the sections to the output lines."""
        stage_functions = []
        stage_name_counts = {}
        for stage_node in self._stage_nodes(ast, target_stage):
            stage_name = normalize_stage_name(getattr(stage_node, "stage", None))
            stage_index = stage_name_counts.get(stage_name, 0)
            stage_name_counts[stage_name] = stage_index + 1
            entry_name = f"{stage_name}_main"
            if stage_index:
                entry_name = f"{entry_name}_{stage_index}"
            stage_functions.append(self.generate_stage(stage_node, entry_name))
        specialized_functions = generic_functions.generate_pending()
        if specialized_functions:
            # Module-scope WGSL declarations are order independent, so the
            # specializations the bodies requested follow the helpers.
            emitted_sections.insert(
                helper_section_index + (1 if helper_functions else 0),
                "\n\n".join(specialized_functions),
            )
        if stage_functions:
            emitted_sections.append("\n\n".join(stage_functions))

        dispatch_helpers = self.generate_sampled_texture_array_dispatch_helpers()
        if dispatch_helpers:
            emitted_sections.append(dispatch_helpers)

        frexp_modf_helpers = wgsl_frexp_modf_helpers(self)
        if frexp_modf_helpers:
            emitted_sections.append(frexp_modf_helpers)

        matrix_inverse_helpers = wgsl_matrix_inverse_helpers(self)
        if matrix_inverse_helpers:
            emitted_sections.append(matrix_inverse_helpers)
        matrix_resize_helpers = wgsl_matrix_resize_helpers(self)
        if matrix_resize_helpers:
            emitted_sections.append(matrix_resize_helpers)
        uniform_register_helpers = wgsl_uniform_register_helpers(self)
        if uniform_register_helpers:
            emitted_sections.append(uniform_register_helpers)
        half_conversion_helpers = self.generate_half_conversion_helpers()
        if half_conversion_helpers:
            emitted_sections.append(half_conversion_helpers)

        if emitted_sections:
            lines.append("")
            lines.append(
                "\n\n".join(section for section in emitted_sections if section)
            )

        if self._uses_primitive_index:
            # An enable directive comes before every declaration.
            lines[1:1] = ["", PRIMITIVE_INDEX_ENABLE_DIRECTIVE]

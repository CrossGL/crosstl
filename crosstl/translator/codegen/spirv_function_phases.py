"""Phases of ``VulkanSPIRVCodeGen.process_function_node``.

The phases collect and classify the parameters, register the SPIR-V function, declare its runtime parameters,
bind the entry-point interface and emit the body.
"""

from .precise_dataflow import declaration_is_precise, precise_contributors
from .spirv_entry_bindings import reserve_entry_buffer_bindings
from .spirv_function_parameters import SpirvFunctionParameterContext
from .spirv_mesh_builtins import glsl_mesh_builtin_output_infos


class SpirvFunctionPhaseMixin:
    """Function-emission phases of ``VulkanSPIRVCodeGen.process_function_node``."""

    def collect_spirv_function_parameters(
        self,
        function_node,
        entry_point_uses_void_signature,
        execution_model_hint,
        is_entry_point,
    ):
        """Gather the parameters a function is emitted with (source parameters, minus the skipped ones, plus the stage-object pointers a helper needs) and classify each into a ``SpirvFunctionParameterContext``; also return whether any parameter is a mesh output."""
        mesh_output_parameters = {}
        output_parameters = []
        runtime_parameters = []
        param_types = []
        param_value_types = []
        resource_array_param_indices = set()
        value_array_param_indices = set()
        patch_interface_parameters = []
        param_type_hints = self.resolve_function_resource_array_type_hints(
            function_node.name
        )
        storage_image_pointer_params = (
            self.resolve_function_storage_image_pointer_params(function_node.name)
        )
        parameters = getattr(
            function_node, "parameters", getattr(function_node, "params", [])
        )
        parameters = list(parameters or [])
        if is_entry_point:
            reserve_entry_buffer_bindings(self, parameters)
        skipped_parameter_indices = (
            set()
            if is_entry_point
            else self.skipped_function_parameter_indices_for_node(function_node)
        )
        if skipped_parameter_indices:
            parameters = [
                param
                for index, param in enumerate(parameters)
                if index not in skipped_parameter_indices
            ]
        if not is_entry_point:
            required_stage_input = self.required_function_stage_input_type(
                function_node.name
            )
            required_stage_output = self.required_function_stage_output_type(
                function_node.name
            )
            if required_stage_input is not None:
                parameters.append(
                    self.stage_object_pointer_parameter("input", required_stage_input)
                )
            if required_stage_output is not None:
                parameters.append(
                    self.stage_object_pointer_parameter("output", required_stage_output)
                )
        has_mesh_output_parameters = any(
            self.is_mesh_output_parameter(param) for param in parameters
        )
        mesh_parameter_execution_model = execution_model_hint
        if mesh_parameter_execution_model is None and has_mesh_output_parameters:
            mesh_parameter_execution_model = "MeshEXT"
        mesh_output_parameter_indices = set()
        if is_entry_point and mesh_parameter_execution_model == "MeshEXT":
            mesh_output_parameters.update(glsl_mesh_builtin_output_infos(self))

        signature = SpirvFunctionParameterContext(
            execution_model_hint,
            is_entry_point,
            entry_point_uses_void_signature,
            mesh_parameter_execution_model,
            param_type_hints,
            storage_image_pointer_params,
            mesh_output_parameters,
            mesh_output_parameter_indices,
            patch_interface_parameters,
            output_parameters,
            runtime_parameters,
            param_types,
            param_value_types,
            resource_array_param_indices,
            value_array_param_indices,
        )
        for source_param_index, param in enumerate(parameters):
            self.classify_spirv_function_parameter(signature, source_param_index, param)
        return has_mesh_output_parameters, signature

    def register_spirv_function(
        self,
        function_node,
        stage,
        function_return_type,
        signature,
    ):
        """Declare the SPIR-V function for ``function_node`` and record its resource-array, mesh-output and value-array parameter indices, globally or for its stage."""
        had_global_function = function_node.name in self.functions
        previous_global_function = self.functions.get(function_node.name)
        previous_global_signature = self.function_signatures.get(function_node.name)
        function_id = self.create_function(
            function_node.name,
            function_return_type,
            signature.param_types,
            register_global=stage is None,
        )
        if stage is not None:
            self.register_stage_local_function(
                stage,
                function_node.name,
                function_id,
                function_return_type,
                signature.param_types,
            )
            if had_global_function:
                self.functions[function_node.name] = previous_global_function
                self.function_signatures[function_node.name] = previous_global_signature
            else:
                self.functions.pop(function_node.name, None)
                self.function_signatures.pop(function_node.name, None)
        self.function_resource_array_params[function_node.name] = (
            signature.resource_array_param_indices
        )
        self.function_mesh_output_parameter_indices[function_node.name] = (
            signature.mesh_output_parameter_indices
        )
        if stage is not None:
            key = self.stage_local_function_key(stage, function_node.name)
            self.stage_local_function_resource_array_params[key] = (
                signature.resource_array_param_indices
            )
            self.stage_local_function_mesh_output_parameter_indices[key] = (
                signature.mesh_output_parameter_indices
            )
        self.function_value_array_parameter_indices_by_id[function_id.id] = set(
            signature.value_array_param_indices
        )
        return function_id

    def declare_spirv_function_parameters(
        self,
        entry_point_uses_void_signature,
        signature,
    ):
        """Declare the runtime parameters of a function that keeps its signature, open its entry block, and copy by-value array parameters into function storage."""
        for i, (param, param_type, param_value_type) in enumerate(
            signature.runtime_parameters
        ):
            if entry_point_uses_void_signature:
                continue
            if hasattr(param, "name"):
                param_name = param.name
            else:
                param_name = f"param{i}"

            param_id = self.create_function_parameter(param_type, param_name)
            self.local_variables[param_name] = param_id
            if i in signature.resource_array_param_indices:
                self.variable_value_types[param_id.id] = param_value_type
            # A helper's image parameter is a value, not a memory object declaration, so it takes no
            # NonWritable or NonReadable decoration (SPIR-V 1.6 section 3.20); the caller's variable has them.
            self.register_declared_resource_metadata(
                param, param_id, param_value_type, decorate=False
            )

        self.begin_block()

        if not entry_point_uses_void_signature:
            for i, (param, _param_type, param_value_type) in enumerate(
                signature.runtime_parameters
            ):
                if i not in signature.value_array_param_indices:
                    continue
                if self.function_parameter_is_reference_like(param):
                    continue
                param_name = getattr(param, "name", f"param{i}")
                param_id = self.local_variables.get(param_name)
                if param_id is None:
                    continue
                local_copy = self.copy_array_pointer_to_function_storage(
                    param_id, param_value_type, name=param_name
                )
                if local_copy is not None:
                    self.local_variables[param_name] = local_copy

    def bind_spirv_function_interface(
        self,
        function_node,
        entry_point_uses_void_signature,
        execution_model_hint,
        function_id,
        has_direct_return_semantic,
        has_mesh_output_parameters,
        return_semantic,
        return_type,
        signature,
    ):
        """Set the current function and execution model, register the return and output interface variables of an entry point and its patch parameters, and initialize the entry-point parameters."""
        self.current_function_id = function_id.id
        if execution_model_hint is not None:
            self.current_execution_model = execution_model_hint
        elif has_mesh_output_parameters:
            self.current_execution_model = "MeshEXT"
        elif self.current_execution_model is None:
            self.current_execution_model = self.execution_model_hint_for_function(
                function_node
            )
        self.current_mesh_output_parameters = signature.mesh_output_parameters
        if has_direct_return_semantic:
            self.current_return_semantic_output = self.register_direct_return_output(
                function_node, return_semantic, return_type
            )
        elif entry_point_uses_void_signature and return_type.type.base_type != "void":
            self.current_entry_point_return_outputs = (
                self.register_entry_point_return_outputs(
                    function_node, return_type, execution_model_hint
                )
            )
        if entry_point_uses_void_signature:
            for param, param_value_type in signature.output_parameters:
                param_name = getattr(param, "name", None) or "param"
                variable = self.register_entry_point_interface_variable(
                    execution_model_hint,
                    "Output",
                    self.entry_point_interface_name(
                        execution_model_hint, "output", param_name
                    ),
                    param_value_type,
                    param,
                )
                self.local_variables[param_name] = variable
                self.register_declared_resource_metadata(
                    param, variable, param_value_type
                )
        for param, patch_info in signature.patch_interface_parameters:
            patch_variable = self.register_patch_parameter_interface_variable(
                param, patch_info
            )
            self.local_variables[patch_info["name"]] = patch_variable
            self.mark_function_interface_variable(patch_variable)

        if entry_point_uses_void_signature:
            self.initialize_entry_point_parameters(
                signature.runtime_parameters, execution_model_hint
            )

    def emit_spirv_function_body(
        self,
        function_node,
        entry_point_uses_void_signature,
    ):
        """Emit the function body (precise functions evaluate their floating-point arithmetic as written), the tessellation patch-constant call and a final return or unreachable, and close the function."""
        precise_function = self.has_attribute(
            function_node, "precise"
        ) or "precise" in {
            str(qualifier).lower()
            for qualifier in getattr(function_node, "qualifiers", None) or ()
        }
        self.precise_expression_depth += precise_function
        previous_precise_flow_names = self.precise_flow_names
        self.precise_flow_names = precise_contributors(
            function_node.body,
            seeds=sorted(
                set(self.precise_global_variables)
                | {
                    param.name
                    for param in getattr(function_node, "parameters", None) or ()
                    if isinstance(getattr(param, "name", None), str)
                    and declaration_is_precise(param)
                }
            ),
        )
        try:
            self.process_statements(function_node.body)
        finally:
            self.precise_expression_depth -= precise_function
            self.precise_flow_names = previous_precise_flow_names
        self.process_tessellation_patch_constant_function(function_node)

        if not self.current_block_has_terminator():
            if (
                entry_point_uses_void_signature
                or self.convert_type_node_to_string(function_node.return_type) == "void"
            ):
                self.create_return()
            else:
                self.create_unreachable()

        self.end_function()

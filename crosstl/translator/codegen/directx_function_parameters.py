"""HLSL renderings of the parameter forms that ``HLSLCodeGen.emit_hlsl_function_parameter`` handles: fixed arrays of storage pointers, workgroup pointers, compute and wave built-in inputs, and the final declaration with its semantic, resource-pointer alias and implicit samplers."""

from ..unhandled import (
    UNHANDLED,
)
from .array_utils import (
    format_c_style_array_declaration,
)
from .directx_values import (
    HLSL_NULL_WORKGROUP_POINTER_CONTRACT,
    DirectXResourcePointerArrayError,
)
from .hlsl_extension_builtins import (
    hlsl_builtin_input_parameter,
)
from .hlsl_narrow_system_values import (
    hlsl_narrow_compute_system_value_bridge,
)
from .stage_utils import (
    normalize_stage_name,
)


class HLSLFunctionParameterMixin:
    """Parameter-form members of ``HLSLCodeGen.emit_hlsl_function_parameter``."""

    def emit_hlsl_resource_pointer_array_parameter(
        self,
        signature,
        p,
    ):
        """Render a parameter that is a fixed array of storage pointers as its base and offset parameters, or return ``UNHANDLED``."""
        pointer_array_contract = self.hlsl_resource_pointer_array_parameter_contract(
            p, self.hlsl_function_declaration_name(signature.func)
        )
        if pointer_array_contract is not None:
            if (
                normalize_stage_name(signature.effective_shader_type)
                in self.stage_entry_types()
            ):
                raise DirectXResourcePointerArrayError(
                    "DirectX stage entry points cannot expose fixed arrays of "
                    "storage pointers directly",
                    function_name=self.hlsl_function_declaration_name(signature.func),
                    array_name=p.name,
                    address_space=pointer_array_contract["address_space"],
                    reason="entry-pointer-array-unsupported",
                    source_location=getattr(p, "source_location", None),
                )
            declaration_name = self.hlsl_declaration_identifier_name(p.name)
            signature.params.append(
                f"{pointer_array_contract['resource_type']} {declaration_name}"
            )
            signature.emitted_param_names.add(declaration_name)
            self.current_hlsl_parameter_resource_types[p.name] = pointer_array_contract[
                "resource_type"
            ]
            materialized_function_name = self.hlsl_function_declaration_name(
                signature.func
            )
            mutated = p.name in (
                self.function_hlsl_resource_pointer_array_mutated_parameters.get(
                    materialized_function_name, set()
                )
            )
            used_names = set(self.current_identifier_reserved_names)
            used_names.update(signature.emitted_param_names)
            offset_parameters = []
            for element_index in range(pointer_array_contract["extent"]):
                offset_name = self.hlsl_unique_local_identifier(
                    f"{p.name}_offset_{element_index}", used_names
                )
                used_names.add(offset_name)
                signature.emitted_param_names.add(offset_name)
                self.current_identifier_reserved_names.add(offset_name)
                self.local_variable_types[offset_name] = "int64_t"
                direction = "inout " if mutated else ""
                signature.params.append(f"{direction}int64_t {offset_name}")
                offset_parameters.append(offset_name)
            offset_array_name = self.hlsl_unique_local_identifier(
                f"{p.name}_offsets", used_names
            )
            used_names.add(offset_array_name)
            self.current_identifier_reserved_names.add(offset_array_name)
            self.local_variable_types[offset_array_name] = (
                f"int64_t[{pointer_array_contract['extent']}]"
            )
            # Metal array parameters decay to by-value pointers to their first
            # element.  Keep the raw logical offsets intact and carry that view
            # base separately so local pointer arithmetic and forwarding never
            # mutate the caller's pointer objects or eagerly index the array.
            base_parameter_name = self.hlsl_unique_local_identifier(
                f"{p.name}_base_offset", used_names
            )
            used_names.add(base_parameter_name)
            signature.emitted_param_names.add(base_parameter_name)
            self.current_identifier_reserved_names.add(base_parameter_name)
            self.local_variable_types[base_parameter_name] = "int64_t"
            signature.params.append(f"int64_t {base_parameter_name}")
            base_local_name = self.hlsl_unique_local_identifier(
                f"{p.name}_base", used_names
            )
            used_names.add(base_local_name)
            self.current_identifier_reserved_names.add(base_local_name)
            self.local_variable_types[base_local_name] = "int64_t"
            signature.parameter_prologue_statements.append(
                f"int64_t {offset_array_name}[{pointer_array_contract['extent']}];"
            )
            for element_index, offset_name in enumerate(offset_parameters):
                signature.parameter_prologue_statements.append(
                    f"{offset_array_name}[{element_index}] = {offset_name};"
                )
                if mutated:
                    signature.parameter_epilogue_statements.append(
                        f"{offset_name} = {offset_array_name}[{element_index}];"
                    )
            signature.parameter_prologue_statements.append(
                f"int64_t {base_local_name} = int64_t({base_parameter_name});"
            )
            array_binding = self.hlsl_narrow_pointer_parameter_alias_binding(
                pointer_array_contract,
                function_name=materialized_function_name,
                parameter_name=p.name,
                root=declaration_name,
                offset="0",
            )
            array_binding.update(
                {
                    "kind": "resource-pointer-array",
                    "array_name": p.name,
                    "parameter_name": p.name,
                    "extent": pointer_array_contract["extent"],
                    "required_access": pointer_array_contract["access"],
                    "offset_array": offset_array_name,
                    "offset_expressions": offset_parameters,
                    "base_parameter": base_parameter_name,
                    "base_index": base_local_name,
                    "parameter_binding": True,
                    "mutated": mutated,
                }
            )
            self.current_hlsl_resource_pointer_aliases[p.name] = array_binding
            return
        return UNHANDLED

    def emit_hlsl_workgroup_pointer_parameter(
        self,
        signature,
        p,
    ):
        """Render a workgroup-pointer parameter as an ``int`` base offset into its groupshared root, or return ``UNHANDLED``."""
        if self.hlsl_workgroup_pointer_declaration(p):
            root, extent = self.current_hlsl_workgroup_pointer_variant[p.name]
            base_name = self.current_hlsl_workgroup_pointer_base_names[p.name]
            signature.params.append(f"int {base_name}")
            signature.emitted_param_names.add(base_name)
            self.current_identifier_reserved_names.add(base_name)
            self.local_variable_types[base_name] = "int"
            qualifiers = {
                str(value).lower() for value in getattr(p, "qualifiers", []) or []
            }
            access = (
                "read"
                if qualifiers.intersection({"const", "constant", "readonly", "in"})
                else "read_write"
            )
            element_type = self.hlsl_workgroup_pointer_element_type(p)
            if (root, extent) == HLSL_NULL_WORKGROUP_POINTER_CONTRACT:
                self.current_hlsl_resource_pointer_aliases[p.name] = {
                    **self.hlsl_null_workgroup_pointer_binding(
                        element_type, offset=base_name
                    ),
                    "access": access,
                    "is_pointer_alias": True,
                }
                return
            self.current_hlsl_resource_pointer_aliases[p.name] = {
                "kind": "workgroup-pointer",
                "root": root,
                "root_kind": "concrete",
                "offset": base_name,
                "array_size": extent,
                "element_type": element_type,
                "source_element_type": element_type,
                "resource_type": "groupshared",
                "access": access,
                "is_pointer_alias": True,
            }
            return
        return UNHANDLED

    def emit_hlsl_builtin_input_parameter(
        self,
        signature,
        p,
        param_type,
        semantic,
    ):
        """Render a parameter that carries a stage built-in input, a compute dispatch-thread bridge or a compute wave or value built-in, or return ``UNHANDLED``."""
        builtin_input = hlsl_builtin_input_parameter(
            self,
            p,
            param_type,
            semantic,
            signature.effective_shader_type,
            signature.emitted_param_names,
        )
        if builtin_input is not None:
            declaration, prologue = builtin_input
            if declaration is not None:
                signature.params.append(declaration)
            signature.parameter_prologue_statements.append(prologue)
            return
        dispatch_thread_id_bridge = None
        if signature.effective_shader_type == "compute":
            dispatch_thread_id_bridge = (
                self.hlsl_compute_dispatch_thread_id_parameter_bridge(
                    p, param_type, semantic, signature.emitted_param_names
                )
            )
        if (
            dispatch_thread_id_bridge is None
            and signature.effective_shader_type == "compute"
        ):
            dispatch_thread_id_bridge = hlsl_narrow_compute_system_value_bridge(
                self, p, param_type, semantic, signature.emitted_param_names
            )
        if dispatch_thread_id_bridge is not None:
            declaration, local_alias = dispatch_thread_id_bridge
            semantic_attr = self.map_semantic(semantic)
            signature.params.append(
                f"{declaration} {semantic_attr}" if semantic_attr else declaration
            )
            signature.parameter_prologue_statements.append(local_alias)
            return

        if signature.effective_shader_type == "compute":
            wave_builtin_prologue = self.hlsl_compute_wave_builtin_parameter_prologue(
                p,
                param_type,
                semantic,
                signature.func,
                signature.execution_config,
                signature.param_list,
            )
            if wave_builtin_prologue is not None:
                signature.parameter_prologue_statements.append(wave_builtin_prologue)
                return

            # Value-builtins (gl_WorkGroupSize / gl_NumWorkGroups /
            # threads_per_grid) have no HLSL parameter system value even on a
            # stage entry. Lower to a body-local from the matching expression, or
            # drop the parameter when it is unrepresentable and unused — never
            # leak an invalid ": <builtin>" entry-point parameter semantic.
            value_builtin_prologue = self.hlsl_compute_value_builtin_parameter_prologue(
                p, param_type, semantic, signature.func, signature.execution_config
            )
            if value_builtin_prologue is self.HLSL_COMPUTE_DROP_PARAMETER:
                return
            if value_builtin_prologue is not None:
                signature.parameter_prologue_statements.append(value_builtin_prologue)
                return
        return UNHANDLED

    def emit_hlsl_parameter_declaration(
        self,
        signature,
        p,
        param_type,
        resource_pointer_contract,
        semantic,
    ):
        """Render the declaration and semantic of an ordinary parameter, and record its resource-pointer alias, private-pointer facts and implicit samplers."""
        ray_role = None
        if signature.effective_shader_type in self.hlsl_ray_stage_types():
            ray_role = self.hlsl_ray_semantic_role(p)
        if ray_role:
            param_type = self.apply_hlsl_ray_parameter_direction(param_type, ray_role)
        elif resource_pointer_contract is None:
            param_type = self.apply_hlsl_parameter_qualifiers(param_type, p)
            param_type = f"{self.hlsl_interpolation_modifier_prefix(p)}{param_type}"
        declaration = format_c_style_array_declaration(
            param_type, self.hlsl_declaration_identifier_name(p.name)
        )
        emit_semantic = semantic
        if (
            ray_role
            or self.hlsl_non_entry_compute_builtin_semantic(
                signature.effective_shader_type, semantic
            )
            or (
                signature.effective_shader_type == "compute"
                and self.hlsl_compute_value_builtin_semantic(semantic)
            )
        ):
            # Ray-payload parameters carry no semantic; GLSL compute/subgroup
            # builtins on a non-entry helper have no HLSL semantic form (the
            # value arrives as a plain parameter) and must not leak as ": gl_*".
            # A value-builtin reaching this point on a compute entry is one the
            # prologue could not express (no uniform) but the body still uses, so
            # it is kept as a plain parameter with its invalid semantic stripped.
            emit_semantic = None
        semantic_attr = self.map_semantic(emit_semantic)
        signature.params.append(
            f"{declaration} {semantic_attr}" if semantic_attr else declaration
        )
        if resource_pointer_contract is not None:
            used_names = set(self.current_identifier_reserved_names)
            used_names.update(signature.emitted_param_names)
            offset_name = self.hlsl_unique_local_identifier(
                f"{p.name}_offset", used_names
            )
            offset_direction = self.hlsl_resource_pointer_offset_parameter_direction(p)
            offset_declaration = f"int64_t {offset_name}"
            if offset_direction is not None:
                offset_declaration = f"{offset_direction} {offset_declaration}"
            signature.params.append(offset_declaration)
            signature.emitted_param_names.add(offset_name)
            self.current_identifier_reserved_names.add(offset_name)
            self.local_variable_types[offset_name] = "int64_t"
            self.current_hlsl_resource_pointer_aliases[p.name] = (
                self.hlsl_narrow_pointer_parameter_alias_binding(
                    resource_pointer_contract,
                    function_name=self.hlsl_function_declaration_name(signature.func),
                    parameter_name=p.name,
                    root=self.hlsl_declaration_identifier_name(p.name),
                    offset=offset_name,
                )
            )
        if self.hlsl_private_pointer_parameter(
            p
        ) and p.name not in self.function_private_pointer_scalar_parameters.get(
            signature.function_name, set()
        ):
            base_name = self.current_hlsl_private_pointer_base_names[p.name]
            signature.params.append(f"int {base_name}")
            signature.emitted_param_names.add(base_name)
            self.current_identifier_reserved_names.add(base_name)
            self.local_variable_types[base_name] = "int"
        if p.name in signature.implicit_texture_samplers:
            sampler_info = signature.implicit_texture_samplers[p.name]
            if sampler_info["synthetic"]:
                sampler_type = (
                    "SamplerComparisonState"
                    if sampler_info["comparison"]
                    else "SamplerState"
                )
                sampler_name = sampler_info["sampler_name"]
                signature.params.append(f"{sampler_type} {sampler_name}")
                signature.sampler_parameters.add(sampler_name)
            regular_sampler_name = self.implicit_regular_sampler_name(
                p.name, sampler_info
            )
            if (
                sampler_info.get("regular")
                and sampler_info.get("comparison")
                and regular_sampler_name != sampler_info["sampler_name"]
                and regular_sampler_name not in signature.param_names
            ):
                signature.params.append(f"SamplerState {regular_sampler_name}")
                signature.sampler_parameters.add(regular_sampler_name)
            query_sampler_name = self.implicit_query_lod_sampler_name(
                p.name, sampler_info
            )
            if (
                sampler_info.get("query_lod")
                and query_sampler_name != sampler_info["sampler_name"]
                and query_sampler_name not in signature.param_names
            ):
                signature.params.append(f"SamplerState {query_sampler_name}")
                signature.sampler_parameters.add(query_sampler_name)

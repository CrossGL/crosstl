"""Tessellation patch constants, interface widths and the function prepass of the SPIR-V generator: patch constant interface variables, inputs, semantics and functions, patch locations, mesh output parameter info, interpolation attributes, interface slots and component widths, per-vertex interfaces, global storage classes, type names, source locations, option payloads, expected types, the function prepass, function parameters, and storage buffer call signatures, argument scores and candidate matches."""

import re
from typing import (
    List,
    Optional,
)

from ..ast import VariableNode
from ..refusals import VulkanRefusal
from .array_utils import parse_array_type
from .ast_walk import walk_ast_values
from .enum_utils import (
    generic_type_parts,
    resolve_generic_enum_specialization,
    substitute_generic_type_name,
)
from .generic_concrete_call_results import concrete_call_overload
from .generic_function_utils import (
    generic_function_emission_list,
    generic_function_parameters,
)
from .spirv_entry_interfaces import ray_storage_class
from .spirv_generic_structs import generic_struct_instance_spelling
from .spirv_image_texels import image_load_in_context
from .spirv_ray_locations import ray_variable_location
from .spirv_storage_subscripts import reads_storage_buffer_element
from .spirv_values import (
    SpirvId,
    UnsupportedSPIRVFeatureError,
)


class SpirvFunctionPrepassMixin:
    """Function prepass members of ``VulkanSPIRVCodeGen``."""

    def register_tessellation_patch_constant_interface_variable(
        self,
        name: str,
        type_id: SpirvId,
        storage_class: str,
        source_node,
        preferred_location: Optional[int] = None,
    ):
        self.require_capability("Tessellation")
        self.validate_user_defined_interface_type(
            type_id, storage_class, name, source_node
        )
        variable = self.create_variable(type_id, storage_class, name)
        location = self.global_interface_location(
            source_node, storage_class, preferred_location, patch=True
        )
        self.decorations.append(f"OpDecorate %{variable.id} Location {location}")
        self.decorations.append(f"OpDecorate %{variable.id} Patch")
        self.decorate_global_interface_variable(source_node, variable)
        if storage_class == "Input":
            self.readonly_pointer_names[variable.id] = name
        self.mark_function_interface_variable(variable)
        return variable, location

    def store_tessellation_patch_constant_user_member(
        self,
        member_name: str,
        member_type: SpirvId,
        member_value: SpirvId,
        metadata: dict,
    ) -> bool:
        interface_key = self.tessellation_patch_constant_user_member_key(
            member_name, metadata
        )
        if interface_key is None:
            return False

        source_node = metadata.get("node")
        if source_node is None:
            return False

        variable = self.tessellation_patch_constant_output_variables.get(interface_key)
        if variable is None:
            variable, location = (
                self.register_tessellation_patch_constant_interface_variable(
                    member_name, member_type, "Output", source_node
                )
            )
            self.tessellation_patch_constant_output_variables[interface_key] = variable
            self.tessellation_patch_constant_interfaces[interface_key] = {
                "location": location,
                "name": member_name,
                "node": source_node,
                "type": member_type,
            }

        self.store_to_variable(
            variable, self.convert_value_to_type(member_value, member_type)
        )
        return True

    def ensure_tessellation_patch_constant_input(self, name: str) -> Optional[SpirvId]:
        if self.current_execution_model != "TessellationEvaluation":
            return None

        metadata = self.tessellation_patch_constant_interfaces.get(name)
        if metadata is None:
            return None

        variable = self.tessellation_patch_constant_input_variables.get(name)
        if variable is not None:
            self.mark_function_interface_variable(variable)
            return variable

        variable, _location = (
            self.register_tessellation_patch_constant_interface_variable(
                metadata["name"],
                metadata["type"],
                "Input",
                metadata["node"],
                metadata["location"],
            )
        )
        self.tessellation_patch_constant_input_variables[name] = variable
        return variable

    def store_tessellation_patch_constant_semantics(
        self, result: SpirvId, result_type: SpirvId, function_name: str
    ):
        struct_name = result_type.type.base_type
        members = self.current_struct_members.get(struct_name, [])
        metadata_by_member = self.struct_member_metadata.get(struct_name, {})
        stored_any = False

        for member_index, (member_type, member_name) in enumerate(members):
            metadata = metadata_by_member.get(member_name, {})
            builtin_name = self.tessellation_patch_constant_builtin_name(
                metadata.get("semantic")
            )
            member_value = self.composite_extract(result, member_type, member_index)

            if builtin_name is not None:
                for (
                    component_index,
                    component,
                ) in self.tessellation_patch_constant_member_components(
                    member_value, member_type
                ):
                    self.store_tessellation_patch_constant_builtin_component(
                        builtin_name, component_index, component
                    )
                    stored_any = True
                continue

            self.store_tessellation_patch_constant_user_member(
                member_name, member_type, member_value, metadata
            )

        if not stored_any:
            self.refuse_lowering(
                "SPIR-V tessellation_control patchconstantfunc "
                f"'{function_name}' returned no tessellation factor semantics"
            )

    def process_tessellation_patch_constant_function(self, function_node):
        if self.current_execution_model != "TessellationControl":
            return
        if not self.function_is_entry_point(function_node, self.current_stage):
            return
        if self.current_block_has_terminator():
            return

        function_name = self.tessellation_patch_constant_function_name(function_node)
        if function_name is None:
            return
        function_signature = self.resolve_function_signature(function_name)
        if function_signature is None:
            self.refuse_lowering(
                "SPIR-V tessellation_control patchconstantfunc "
                f"'{function_name}' does not reference a generated function"
            )
            return

        return_type, _param_types = function_signature
        if return_type.type.base_type == "void":
            self.refuse_lowering(
                "SPIR-V tessellation_control patchconstantfunc "
                f"'{function_name}' requires a non-void return type"
            )
            return

        patch_function = self.function_nodes.get(function_name)
        if patch_function is None:
            self.refuse_lowering(
                "SPIR-V tessellation_control patchconstantfunc "
                f"'{function_name}' has no available function definition"
            )
            return

        arguments = self.tessellation_patch_constant_call_arguments(patch_function)
        result = self.call_function(function_name, arguments)
        if result is None:
            return
        self.store_tessellation_patch_constant_semantics(
            result, return_type, function_name
        )

    def register_patch_parameter_interface_variable(self, param, patch_info: dict):
        self.require_capability("Tessellation")
        element_type = self.map_crossgl_type(patch_info["element_type_name"])
        array_type = self.register_array_type(
            element_type, patch_info["control_points"]
        )
        storage_class = patch_info["storage_class"]
        self.validate_user_defined_interface_type(
            array_type, storage_class, patch_info["name"], param
        )
        variable = self.create_variable(array_type, storage_class, patch_info["name"])
        location = self.global_interface_location(
            param,
            storage_class,
            self.matching_tessellation_output_patch_location(patch_info),
        )
        self.decorations.append(f"OpDecorate %{variable.id} Location {location}")
        self.decorate_global_interface_variable(param, variable)
        self.patch_parameter_metadata[variable.id] = patch_info
        self.record_tessellation_output_patch_location(patch_info, location)
        if storage_class == "Input":
            self.readonly_pointer_names[variable.id] = patch_info["name"]
        return variable

    def tessellation_patch_location_key(self, patch_info: dict) -> tuple:
        return (patch_info.get("element_type_name"), patch_info.get("control_points"))

    def record_tessellation_output_patch_location(
        self, patch_info: dict, location: int
    ):
        if (
            self.current_execution_model != "TessellationControl"
            or patch_info.get("patch_type") != "OutputPatch"
            or patch_info.get("storage_class") != "Output"
        ):
            return
        self.tessellation_output_patch_locations[
            self.tessellation_patch_location_key(patch_info)
        ] = location

    def matching_tessellation_output_patch_location(
        self, patch_info: dict
    ) -> Optional[int]:
        if (
            self.current_execution_model != "TessellationEvaluation"
            or patch_info.get("patch_type") != "OutputPatch"
            or patch_info.get("storage_class") != "Input"
        ):
            return None
        return self.tessellation_output_patch_locations.get(
            self.tessellation_patch_location_key(patch_info)
        )

    def mesh_output_parameter_info(self, param, execution_model: Optional[str]):
        if execution_model != "MeshEXT" or not self.is_mesh_output_parameter(param):
            return None

        param_name = getattr(param, "name", None)
        if not param_name:
            return None

        type_name = self.type_name_from_value(
            getattr(param, "param_type", getattr(param, "vtype", None))
        )
        array_info = self.split_outer_array_type(type_name)
        if array_info is None:
            self.refuse_lowering(
                f"SPIR-V mesh output parameter {param_name} must be " "an array type"
            )
            element_type_name = type_name
            element_count = None
        else:
            element_type_name, element_count = array_info

        element_type = self.map_crossgl_type(element_type_name)
        members = {}
        for index, (member_type, member_name) in enumerate(
            self.current_struct_members.get(element_type_name, [])
        ):
            metadata = self.struct_member_metadata.get(element_type_name, {}).get(
                member_name, {}
            )
            members[member_name] = {
                "index": index,
                "type": member_type,
                "semantic": metadata.get("semantic"),
            }

        return {
            "name": param_name,
            "role": self.mesh_output_parameter_role(param),
            "element_type_name": element_type_name,
            "element_type": element_type,
            "count": element_count,
            "members": members,
        }

    def validate_interface_interpolation_attributes(self, node: VariableNode):
        for first, second in (("flat", "noperspective"), ("centroid", "sample")):
            if self.has_attribute(node, first) and self.has_attribute(node, second):
                raise VulkanRefusal(
                    "SPIR-V interpolation attributes "
                    f"@{first} and @{second} cannot be combined"
                )

    def interface_slot_keys(
        self,
        node: VariableNode,
        storage_class: str,
        location: int,
        patch: bool = False,
    ) -> set:
        component = self.explicit_component_attribute(node)
        component_start = component if component is not None else 0
        index = self.explicit_interface_integer_attribute(node, "index") or 0
        scope = self.interface_location_scope()
        patch_key = "patch" if patch else "per_vertex"
        slots = set()

        for offset, component_width in enumerate(
            self.interface_location_component_widths(node)
        ):
            if component_start + component_width > 4:
                raise VulkanRefusal(
                    f"SPIR-V component range overflows location {location + offset}: "
                    f"{component_start}..{component_start + component_width - 1}"
                )

            slots.update(
                (scope, patch_key, location + offset, index, component)
                for component in range(
                    component_start, component_start + component_width
                )
            )

        return slots

    def interface_component_width(self, node: VariableNode) -> int:
        widths = self.interface_location_component_widths(node)
        if widths:
            return widths[0]
        return 4

    def interface_location_component_widths(self, node: VariableNode) -> List[int]:
        type_source = getattr(node, "var_type", getattr(node, "vtype", "float"))
        type_source = getattr(node, "member_type", type_source)
        type_name = self.type_name_from_value(type_source)
        if type_name is not None and self.is_per_vertex_interface(node):
            outer = self.split_outer_array_type(
                self.normalize_hlsl_matrix_type(
                    self.normalize_generic_vector_type(type_name)
                )
            )
            if outer is not None:
                type_name = outer[0]
        return self.interface_type_location_component_widths(type_name)

    # Arrayed interfaces whose outer dimension indexes vertices and takes no locations (Vulkan 1.3 section
    # 15.1.4): geometry inputs, tessellation inputs and non-patch control outputs, mesh outputs, pervertex inputs.
    PER_VERTEX_INTERFACES = {
        ("Geometry", "in"),
        ("TessellationControl", "in"),
        ("TessellationControl", "out"),
        ("TessellationEvaluation", "in"),
        ("MeshEXT", "out"),
        ("MeshNV", "out"),
    }

    def is_per_vertex_interface(self, node) -> bool:
        names = {
            str(getattr(attribute, "name", "")).lower()
            for attribute in getattr(node, "attributes", None) or []
        }
        names.update(str(q).lower() for q in getattr(node, "qualifiers", None) or [])
        if "patch" in names:
            return False
        if names & {"pervertex", "pervertexext", "pervertexnv"}:
            return True
        direction = (
            "in"
            if names & {"in", "input"}
            else "out" if names & {"out", "output"} else None
        )
        models = (
            {self.current_execution_model}
            if self.current_execution_model
            else getattr(self, "program_execution_models", set())
        )
        return len(models) == 1 and (*models, direction) in self.PER_VERTEX_INTERFACES

    def interface_type_location_component_widths(self, type_name: str) -> List[int]:
        if type_name is None:
            return [4]

        type_name = self.normalize_generic_vector_type(type_name)
        type_name = self.normalize_hlsl_matrix_type(type_name)
        array_type = self.split_outer_array_type(type_name)
        if array_type is not None:
            element_type_name, size = array_type
            element_widths = self.interface_type_location_component_widths(
                element_type_name
            )
            element_count = max(self.array_length_default_value(size), 1)
            return element_widths * element_count

        matrix_match = re.fullmatch(r"(d)?mat([234])(?:x([234]))?", type_name)
        if matrix_match:
            _is_double, cols, rows = matrix_match.groups()
            return [int(rows or cols)] * int(cols)

        vector_info = self.vector_component_type_and_count(type_name)
        if vector_info is not None:
            _, component_count = vector_info
            return [component_count]

        if self.normalize_primitive_name(type_name) in {
            "float",
            "double",
            "int",
            "uint",
            "bool",
        }:
            return [1]

        return [4]

    def infer_global_storage_class(
        self, node: VariableNode, default_storage_class: str, type_name: str = None
    ) -> str:
        attribute_names = {
            getattr(attribute, "name", "").lower()
            for attribute in getattr(node, "attributes", [])
        }
        qualifiers = {
            str(qualifier).lower() for qualifier in getattr(node, "qualifiers", [])
        }

        if self.is_task_payload_shared_node(node):
            return "TaskPayloadWorkgroupEXT"
        ray_storage = ray_storage_class(attribute_names | qualifiers)
        if ray_storage is not None:
            location = ray_variable_location(node)
            if location is not None:
                names = self.ray_storage_locations.setdefault(
                    (ray_storage, location), []
                )
                if node.name not in names:
                    names.append(node.name)
            if getattr(node, "initial_value", None) is not None:
                raise UnsupportedSPIRVFeatureError(
                    "ray tracing variable initializer",
                    f"SPIR-V {ray_storage} variable '{node.name}' cannot take an initializer",
                    missing_capabilities=("spirv.ray_tracing_storage.initializer",),
                    source_location=getattr(node, "source_location", None),
                )
            return ray_storage
        # Module- and stage-scope GLSL `shared`, HLSL `groupshared` and MSL `threadgroup` variables are one copy
        # per workgroup; Private storage would give every invocation its own copy.
        workgroup_names = {"shared", "groupshared", "threadgroup", "workgroup"}
        if attribute_names & workgroup_names or qualifiers & workgroup_names:
            if getattr(node, "initial_value", None) is not None:
                raise UnsupportedSPIRVFeatureError(
                    "workgroup variable initializer",
                    f"SPIR-V workgroup variable '{node.name}' cannot take an initializer",
                    missing_capabilities=("spirv.workgroup_storage.initializer",),
                    source_location=getattr(node, "source_location", None),
                )
            return "Workgroup"
        if attribute_names & {"input", "in"} or qualifiers & {"input", "in"}:
            return "Input"
        if attribute_names & {"output", "out"} or qualifiers & {"output", "out"}:
            return "Output"
        if type_name:
            base_type_name, _ = parse_array_type(type_name)
            if self.is_resource_type_name(base_type_name):
                return "UniformConstant"
        if attribute_names & {"uniform"} or qualifiers & {"uniform"}:
            base_type_name, _ = parse_array_type(type_name)
            if base_type_name in self.struct_types:
                return "Uniform"
        return default_storage_class

    def type_name_from_value(self, type_value) -> str:
        if type_value is None:
            return None
        if self.is_type_node_like(type_value):
            type_name = self.convert_type_node_to_string(type_value)
        else:
            type_name = str(type_value)

        substitutions = {}
        substitutions.update(
            getattr(self, "current_generic_type_substitutions", {}) or {}
        )
        substitutions.update(
            getattr(self, "current_generic_function_substitutions", {}) or {}
        )
        if substitutions:
            if type_name.startswith("&"):
                referenced = substitute_generic_type_name(
                    type_name[1:].strip(), substitutions
                )
                return f"&{referenced}"
            if type_name.endswith("*"):
                pointee = substitute_generic_type_name(
                    type_name[:-1].strip(), substitutions
                )
                return f"{pointee}*"
            return substitute_generic_type_name(type_name, substitutions)
        return type_name

    def type_name_string(self, type_value) -> str:
        return self.type_name_from_value(type_value)

    def expression_result_type(self, expr):
        inferred_type = self.infer_expression_result_type(expr)
        if inferred_type is None:
            return None
        return inferred_type.type.base_type

    def format_source_location(self, source_location) -> Optional[str]:
        if source_location is None:
            return None
        if isinstance(source_location, dict):
            line = source_location.get("line") or source_location.get("lineno")
            column = source_location.get("column") or source_location.get("col")
            if line is not None and column is not None:
                return f"line {line}, column {column}"
            if line is not None:
                return f"line {line}"
        if isinstance(source_location, (list, tuple)):
            if len(source_location) >= 2:
                return f"line {source_location[0]}, column {source_location[1]}"
            if len(source_location) == 1:
                return f"line {source_location[0]}"
        line = getattr(source_location, "line", None) or getattr(
            source_location, "lineno", None
        )
        column = getattr(source_location, "column", None) or getattr(
            source_location, "col", None
        )
        if line is not None and column is not None:
            return f"line {line}, column {column}"
        if line is not None:
            return f"line {line}"
        return str(source_location)

    def unsupported_generic_function_error(self, function_node, call_node=None):
        helper_name = getattr(function_node, "name", "<unknown>")
        generic_params = generic_function_parameters(function_node)
        params_label = ", ".join(generic_params) if generic_params else "<none>"
        source_location = getattr(call_node, "source_location", None) or getattr(
            function_node, "source_location", None
        )
        location_label = self.format_source_location(source_location)
        location_suffix = f" at {location_label}" if location_label else ""
        return UnsupportedSPIRVFeatureError(
            "generic-helper-specialization",
            "SPIR-V codegen does not support generic functions: "
            "unspecialized generic helper "
            f"'{helper_name}' with generic parameters ({params_label})"
            f"{location_suffix}; specialize the function before SPIR-V generation",
            missing_capabilities=("spirv.generic_function_specialization",),
            source_location=source_location,
        )

    def option_payload_type_name(self, type_value) -> Optional[str]:
        type_name = self.type_name_from_value(type_value)
        if not isinstance(type_name, str):
            return None

        base_name, generic_args = generic_type_parts(type_name.strip())
        if base_name.rsplit("::", 1)[-1] != "Option" or len(generic_args) != 1:
            return None
        return generic_args[0]

    def lowerable_option_payload_type_name(self, type_value) -> Optional[str]:
        type_name = self.type_name_from_value(type_value)
        payload_type = self.option_payload_type_name(type_name)
        if payload_type is None:
            return None
        if resolve_generic_enum_specialization(self, type_name) is not None:
            return None
        return payload_type

    def option_or_expected_payload_type_name(self) -> Optional[str]:
        expected_type = self.current_expression_expected_type
        payload_type = self.lowerable_option_payload_type_name(expected_type)
        if payload_type is not None:
            return payload_type
        if self.option_payload_type_name(expected_type) is not None:
            return None
        return self.type_name_from_value(expected_type)

    def option_none_default_value(self) -> Optional[SpirvId]:
        payload_type = self.lowerable_option_payload_type_name(
            self.current_expression_expected_type
        )
        if payload_type is None:
            return None
        return self.default_value_for_type(self.map_crossgl_type(payload_type))

    def expected_primitive_type_name(self) -> Optional[str]:
        expected_type = self.option_or_expected_payload_type_name()
        if expected_type is None:
            return None
        return self.normalize_primitive_name(expected_type)

    def is_type_node_like(self, value) -> bool:
        return any(
            hasattr(value, attribute)
            for attribute in (
                "name",
                "element_type",
                "pointee_type",
                "referenced_type",
                "return_type",
            )
        )

    def process_expression_with_expected_type(self, expr, expected_type):
        return self.process_expression_with_expected_type_and_precision(
            expr, expected_type, precise=False
        )

    def process_expression_with_expected_type_and_precision(
        self, expr, expected_type, precise: bool
    ):
        previous_expected_type = self.current_expression_expected_type
        self.current_expression_expected_type = self.type_name_string(expected_type)
        try:
            value = self.process_expression_with_precision(expr, precise)
            return image_load_in_context(
                self, expr, value, self.current_expression_expected_type
            )
        finally:
            self.current_expression_expected_type = previous_expected_type

    def collect_ast_functions(self, root):
        return [
            value
            for value in walk_ast_values(root)
            if hasattr(value, "body") and hasattr(value, "parameters")
        ]

    def concrete_function_prepass_list(self, root):
        """Return concrete functions, replacing generic definitions with emissions."""
        functions = []
        seen = set()
        for function_node in self.collect_ast_functions(root):
            candidates = (
                generic_function_emission_list(self, function_node)
                if generic_function_parameters(function_node)
                else [function_node]
            )
            for candidate in candidates:
                candidate_id = id(candidate)
                if candidate_id in seen:
                    continue
                seen.add(candidate_id)
                functions.append(candidate)
        return functions

    def run_function_prepass(self, function_node, operation):
        """Evaluate one function analysis with materialized generic substitutions."""
        previous_substitutions = self.current_generic_function_substitutions
        self.current_generic_function_substitutions = (
            getattr(function_node, "_generic_substitutions", {}) or {}
        )
        try:
            return operation()
        finally:
            self.current_generic_function_substitutions = previous_substitutions

    def collect_functions_by_name(self, functions):
        functions_by_name = {}
        for function_node in functions:
            function_name = self.function_source_name(function_node)
            if not function_name:
                continue
            functions_by_name.setdefault(function_name, []).append(function_node)
        return functions_by_name

    def function_source_name(self, function_node):
        return getattr(function_node, "_generic_source_name", None) or getattr(
            function_node, "name", None
        )

    def function_parameters(self, function_node):
        return list(
            getattr(function_node, "parameters", getattr(function_node, "params", []))
            or []
        )

    def function_parameter_type_name(self, param) -> Optional[str]:
        param_type = getattr(param, "param_type", getattr(param, "vtype", None))
        return self.type_name_from_value(param_type)

    def function_parameter_type_names(self, function_node):
        return [
            self.function_parameter_type_name(param) or "unknown"
            for param in self.function_parameters(function_node)
        ]

    def storage_buffer_effective_call_signature(self, function_node, call_args):
        parameters = self.function_parameters(function_node)
        args = list(call_args or [])
        skipped_indices = self.skipped_function_parameter_indices_for_node(
            function_node
        )
        if not skipped_indices:
            return parameters, args
        return (
            [
                param
                for index, param in enumerate(parameters)
                if index not in skipped_indices
            ],
            [arg for index, arg in enumerate(args) if index not in skipped_indices],
        )

    def storage_buffer_effective_arity_matches(self, function_node, call_args):
        parameters, args = self.storage_buffer_effective_call_signature(
            function_node, call_args
        )
        return len(parameters) == len(args)

    def storage_buffer_argument_match_score(
        self, param, arg, substitutions=None
    ) -> Optional[int]:
        storage_buffer_type_name = self.storage_buffer_parameter_type_name(param)
        if storage_buffer_type_name is not None:
            storage_buffer_type_name = self.substitute_generic_signature_type(
                storage_buffer_type_name, substitutions
            )
            actual_type_name = self.storage_buffer_expression_type_name(arg)
            if actual_type_name is None:
                return None
            if not self.storage_buffer_parameter_type_is_compatible(
                storage_buffer_type_name, actual_type_name
            ):
                return None
            score = 4
            if self.normalize_signature_type_name(
                storage_buffer_type_name
            ) == self.normalize_signature_type_name(actual_type_name):
                score += 1
            score += self.storage_buffer_argument_name_match_score(param, arg)
            return score

        declared_type_name = self.function_parameter_type_name(param)
        declared_type_name = self.substitute_generic_signature_type(
            declared_type_name, substitutions
        )
        actual_type_name = self.call_argument_type_name(arg)
        if not self.scalar_or_vector_type_compatible(
            declared_type_name, actual_type_name
        ):
            return None
        score = 0
        if actual_type_name is not None:
            score += 1
            if self.normalize_signature_type_name(
                declared_type_name
            ) == self.normalize_signature_type_name(actual_type_name):
                score += 1
        score += self.storage_buffer_argument_name_match_score(param, arg)
        return score

    def storage_buffer_argument_name_match_score(self, param, arg) -> int:
        param_name = getattr(param, "name", None)
        arg_name = self.expression_name(arg)
        if not param_name or not arg_name or param_name != arg_name:
            return 0

        if self.function_parameter_is_reference_like(param):
            return 4
        return 2

    def function_parameter_is_reference_like(self, param) -> bool:
        qualifiers = self.parameter_qualifier_names(param)
        if qualifiers & {"out", "inout"}:
            return True

        param_type = getattr(param, "param_type", getattr(param, "vtype", None))
        type_name = self.type_name_from_value(param_type)
        if type_name is None:
            return False

        type_name = str(type_name).strip()
        return type_name.startswith("&") or type_name.endswith("&")

    def function_parameter_requires_pointer_alias(self, param) -> bool:
        return self.function_parameter_is_reference_like(
            param
        ) and not self.is_mesh_output_parameter(param)

    def storage_buffer_argument_rejection_reason(
        self, param, arg, arg_index, substitutions=None
    ) -> str:
        param_name = getattr(param, "name", f"param{arg_index}")
        storage_buffer_type_name = self.storage_buffer_parameter_type_name(param)
        if storage_buffer_type_name is not None:
            storage_buffer_type_name = self.substitute_generic_signature_type(
                storage_buffer_type_name, substitutions
            )
            expected_class = self.storage_buffer_resource_class_label(
                storage_buffer_type_name
            )
            expression_label = self.expression_debug_name(arg)
            actual_type_name = self.storage_buffer_expression_type_name(arg)
            if actual_type_name is None:
                actual_type_name = self.call_argument_type_name(arg) or "unknown"
                return (
                    f"argument {arg_index + 1} expression {expression_label} "
                    f"does not carry storage-buffer provenance; expected "
                    f"{expected_class} for parameter {param_name} "
                    f"({storage_buffer_type_name})"
                    f", got {actual_type_name}"
                )
            return (
                f"argument {arg_index + 1} expression {expression_label} type "
                f"{actual_type_name} is incompatible with expected "
                f"{expected_class} parameter {param_name} "
                f"({storage_buffer_type_name})"
            )

        declared_type_name = self.substitute_generic_signature_type(
            self.function_parameter_type_name(param), substitutions
        )
        declared_type_name = declared_type_name or "unknown"
        actual_type_name = self.call_argument_type_name(arg) or "unknown"
        return (
            f"argument {arg_index + 1} type {actual_type_name} is incompatible "
            f"with parameter {param_name} ({declared_type_name})"
        )

    def storage_buffer_candidate_call_match(self, function_node, call_args):
        parameters = self.function_parameters(function_node)
        args = list(call_args or [])
        skipped_param_indices = self.skipped_function_parameter_indices_for_node(
            function_node
        )
        effective_parameters = [
            (index, param)
            for index, param in enumerate(parameters)
            if index not in skipped_param_indices
        ]
        substitutions = self.generic_storage_function_type_bindings(function_node, args)
        if substitutions is None:
            return None, "generic type arguments could not be resolved"

        reason = None
        incompatibility_reasons = []
        best_match = None

        def search(param_index, arg_index, score, bindings, skipped_arg_indices):
            nonlocal best_match, reason
            remaining_params = len(effective_parameters) - param_index
            remaining_args = len(args) - arg_index
            if remaining_args < remaining_params:
                reason = (
                    f"effective arity {len(effective_parameters)} exceeds remaining "
                    f"actual arguments {remaining_args}"
                )
                return
            if param_index == len(effective_parameters):
                skipped = skipped_arg_indices + list(range(arg_index, len(args)))
                candidate = (
                    score - len(skipped),
                    score,
                    -len(skipped),
                    bindings,
                    skipped,
                )
                if best_match is None or candidate[:3] > best_match[:3]:
                    best_match = candidate
                return
            if arg_index >= len(args):
                reason = (
                    f"effective arity {len(effective_parameters)} exceeds actual "
                    f"arity {len(args)}"
                )
                return

            original_param_index, param = effective_parameters[param_index]
            arg = args[arg_index]
            arg_score = self.storage_buffer_argument_match_score(
                param, arg, substitutions
            )
            if arg_score is not None:
                search(
                    param_index + 1,
                    arg_index + 1,
                    score + arg_score,
                    bindings + [(original_param_index, param, arg)],
                    skipped_arg_indices,
                )
            else:
                incompatibility_reasons.append(
                    self.storage_buffer_argument_rejection_reason(
                        param, arg, arg_index, substitutions
                    )
                )
            search(
                param_index,
                arg_index + 1,
                score,
                bindings,
                skipped_arg_indices + [arg_index],
            )

        search(0, 0, 0, [], [])
        if best_match is None:
            if incompatibility_reasons:
                reason = incompatibility_reasons[0]
            elif reason is None:
                reason = "no compatible ordered argument alignment"
            return None, reason

        rank, score, _skip_rank, bindings, skipped_arg_indices = best_match
        return {
            "rank_score": rank,
            "score": score,
            "bindings": bindings,
            "skipped_arg_indices": skipped_arg_indices,
            "effective_parameter_count": len(effective_parameters),
        }, None

    def storage_buffer_effective_call_bindings(self, function_node, call_args):
        match, _reason = self.storage_buffer_candidate_call_match(
            function_node, call_args
        )
        if match is not None:
            return match["bindings"]

        parameters = self.function_parameters(function_node)
        args = list(call_args or [])
        skipped_indices = self.skipped_function_parameter_indices_for_node(
            function_node
        )
        return [
            (index, param, args[index])
            for index, param in enumerate(parameters)
            if index not in skipped_indices and index < len(args)
        ]

    def format_function_candidate_signature(self, function_node) -> str:
        function_name = getattr(function_node, "name", "unknown")
        params = ", ".join(self.function_parameter_type_names(function_node))
        return f"{function_name}({params})"

    def normalize_signature_type_name(self, type_name) -> Optional[str]:
        if type_name is None:
            return None
        type_name = self.type_name_from_value(type_name)
        if type_name is None:
            return None
        type_name = self.normalize_reference_type_name(type_name)
        # Registered vector types are named ``v4float``; spell them as CrossGL
        # does so call keys and declared signatures compare equal.
        registered_vector = re.fullmatch(
            r"v([234])(float|double|int|uint|bool)", str(type_name).strip()
        )
        if registered_vector is not None:
            size, element = registered_vector.groups()
            type_name = f"vec{size}<{element}>"
        type_name = self.normalize_generic_vector_type(type_name)
        type_name = self.normalize_hlsl_matrix_type(type_name)
        primitive_name = self.normalize_primitive_name(type_name)
        if primitive_name == self.BFLOAT16_TYPE_NAME:
            type_name = primitive_name
        return re.sub(r"\s+", "", str(type_name))

    def call_argument_type_name(self, arg) -> Optional[str]:
        if reads_storage_buffer_element(self, arg):
            inferred_type = self.infer_expression_result_type(arg)
            return None if inferred_type is None else inferred_type.type.base_type
        storage_buffer_type = self.storage_buffer_expression_type_name(arg)
        if storage_buffer_type is not None:
            return storage_buffer_type
        inferred_type = self.infer_expression_result_type(arg)
        if inferred_type is None:
            return None
        base_type = inferred_type.type.base_type
        # A generic instance keeps its CrossGL spelling, so ``Box<T>`` binds ``T`` from ``Box_float``.
        return generic_struct_instance_spelling(self, base_type) or base_type

    def declared_concrete_call_result_type(self, call) -> Optional[SpirvId]:
        """The declared result type of the concrete overload ``call`` selects.

        Signatures are registered only as functions are emitted, and by name, so generic
        specialization could not type a helper call and a name lookup typed an overloaded
        helper by whichever overload was emitted last.
        """
        function_node = concrete_call_overload(
            self,
            call,
            lambda generator, arg: generator.call_argument_type_name(arg),
            lambda generator, type_name: generator.normalize_signature_type_name(
                type_name
            ),
        )
        if function_node is None:
            return None
        return self.map_crossgl_type(function_node.return_type)

    def call_argument_type_names(self, args):
        return [self.call_argument_type_name(arg) or "unknown" for arg in args]

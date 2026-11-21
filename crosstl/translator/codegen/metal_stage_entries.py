"""Stage entries of the Metal generator: mesh and ray payload parameters, resource parameters and their array sizes, stage entry discovery, default stage I/O semantics, vertex output position fallbacks, stage attributes and threadgroup limits, tessellation domains, and mesh stage output configuration and flow states."""

import re

from ..ast import (
    RayQueryOpNode,
    RayTracingOpNode,
)
from ..host_workgroup_size import HOST_WORKGROUP_SIZE_KEY, has_explicit_workgroup_size
from ..refusals import MetalRefusal
from .array_utils import (
    parse_array_type,
    split_array_type_suffix,
)
from .metal_conservative_depth import metal_conservative_depth_semantic
from .metal_exact_lookups import metal_semantic_index
from .metal_fragment_outputs import metal_fragment_output_color_semantic
from .metal_stage_refusals import refuse_metal_stages
from .resource_array_bounds import DynamicIndexSizes
from .resource_arrays import collect_resource_array_size_hints
from .stage_utils import (
    assign_stage_entry_names,
    collect_stage_entry_records,
    collect_stage_entry_reserved_function_names,
    collect_stage_local_variables,
    compute_local_size,
    function_stage_name,
    normalize_stage_name,
    stage_matches,
)


class MetalStageEntryMixin:
    """Stage entry members of ``MetalCodeGen``."""

    def readonly_metal_mesh_payload_parameter_reason(self, shader_type, node):
        if shader_type == "mesh":
            return "mesh stage payload"
        qualifiers = self.parameter_qualifier_names(node)
        if qualifiers & {"const", "constant"}:
            return "const-qualified payload parameter"
        return None

    def is_metal_mesh_payload_parameter(self, shader_type, node):
        if shader_type not in {"object", "task", "amplification", "mesh"}:
            return False
        return self.is_metal_mesh_payload_semantic(self.semantic_from_node(node))

    def is_metal_mesh_payload_semantic(self, semantic):
        if semantic is None:
            return False
        normalized = str(semantic).strip().lower().replace("-", "_")
        if normalized.startswith("metal_") or normalized.startswith("msl_"):
            normalized = normalized.split("_", 1)[1]
        return normalized in {
            "payload",
            "mesh_payload",
            "hlsl_mesh_payload",
            "task_payload",
            "taskpayloadsharedext",
        }

    def is_explicit_metal_mesh_payload_semantic(self, semantic):
        if semantic is None:
            return False
        normalized = str(semantic).strip().lower().replace("-", "_")
        if normalized.startswith("metal_") or normalized.startswith("msl_"):
            normalized = normalized.split("_", 1)[1]
        return normalized in {
            "mesh_payload",
            "hlsl_mesh_payload",
            "task_payload",
            "taskpayloadsharedext",
        }

    def metal_ray_payload_parameter_declaration(
        self, mapped_type, name, node=None, shader_type=None
    ):
        address_space = self.metal_ray_payload_address_space(shader_type)
        if address_space is None:
            return None
        ray_semantic_role = self.metal_ray_semantic_role(node, shader_type)
        if ray_semantic_role != "payload" and not (
            shader_type in {"ray_callable", "callable"}
            and ray_semantic_role == "callable_data"
        ):
            return None

        return f"{address_space} {mapped_type}& {name}"

    def metal_ray_payload_address_space(self, shader_type):
        if shader_type == "ray_generation":
            return "device"
        if shader_type in {
            "ray_intersection",
            "ray_any_hit",
            "ray_closest_hit",
            "ray_miss",
            "ray_callable",
            "intersection",
            "anyhit",
            "closesthit",
            "miss",
            "callable",
        }:
            return "ray_data"
        return None

    def structured_buffer_parameter_array_size(self, vtype, node=None):
        param_name = getattr(node, "name", None)
        function_hints = self.function_resource_array_size_hints.get(
            self.current_function_name, {}
        )
        if hasattr(vtype, "element_type") and str(type(vtype)).find("ArrayType") != -1:
            return (
                self.safe_expression_to_string(vtype.size)
                if vtype.size is not None
                else function_hints.get(param_name, "")
            )

        if hasattr(vtype, "name") or hasattr(vtype, "element_type"):
            return None

        type_string = str(vtype)
        if "[" not in type_string or "]" not in type_string:
            return None
        _, array_size = parse_array_type(type_string)
        return function_hints.get(param_name, "") if array_size is None else array_size

    def glsl_buffer_block_parameter_array_size(self, vtype, node=None):
        param_name = getattr(node, "name", None)
        function_hints = self.function_resource_array_size_hints.get(
            self.current_function_name, {}
        )
        if hasattr(vtype, "element_type") and str(type(vtype)).find("ArrayType") != -1:
            return (
                self.safe_expression_to_string(vtype.size)
                if vtype.size is not None
                else function_hints.get(param_name, "")
            )

        if hasattr(vtype, "name") or hasattr(vtype, "element_type"):
            return None

        type_string = str(vtype)
        if "[" not in type_string or "]" not in type_string:
            return None
        _, array_size = parse_array_type(type_string)
        return function_hints.get(param_name, "") if array_size is None else array_size

    def format_resource_parameter(self, resource_type, name, array_size):
        if array_size is not None:
            array_size = array_size or "1"
            return f"array<{resource_type}, {array_size}> {name}"
        return f"{resource_type} {name}"

    def resource_array_parameter(self, vtype, node=None):
        if hasattr(vtype, "element_type") and str(type(vtype)).find("ArrayType") != -1:
            base_type = self.convert_type_node_to_string(vtype.element_type)
            if not self.is_resource_parameter_type(base_type):
                return None
            array_size = (
                self.safe_expression_to_string(vtype.size)
                if vtype.size is not None
                else self.function_resource_array_size_hints.get(
                    self.current_function_name, {}
                ).get(node.name, "")
            )
            return self.map_resource_type_with_format(base_type, node), array_size

        if hasattr(vtype, "name") or hasattr(vtype, "element_type"):
            return None

        type_string = str(vtype)
        if "[" not in type_string or "]" not in type_string:
            return None
        base_type, array_size = parse_array_type(type_string)
        if not self.is_resource_parameter_type(base_type):
            return None
        return self.map_resource_type_with_format(base_type, node), (
            self.function_resource_array_size_hints.get(
                self.current_function_name, {}
            ).get(node.name, "")
            if array_size is None
            else array_size
        )

    def collect_resource_array_size_hints(self, ast):
        global_arrays = self.collect_unsized_resource_globals(ast)
        function_arrays = self.collect_unsized_resource_parameters(ast)
        fixed_global_array_sizes = self.collect_fixed_resource_global_sizes(ast)
        fixed_function_array_sizes = self.collect_fixed_resource_parameter_sizes(ast)
        if not (
            global_arrays
            or function_arrays
            or fixed_global_array_sizes
            or fixed_function_array_sizes
        ):
            return {}, {}

        functions = self.all_functions(ast)
        return collect_resource_array_size_hints(
            global_arrays=global_arrays,
            function_arrays=function_arrays,
            fixed_global_array_sizes=fixed_global_array_sizes,
            fixed_function_array_sizes=fixed_function_array_sizes,
            functions=functions,
            walk_nodes=self.iter_ast_nodes,
            expression_name=self.expression_name,
            literal_int_value=self.literal_int_value,
            visible_literal_int_constants=self.visible_literal_int_constants,
            function_call_name=self.function_call_name,
            initial_size=1,
            format_size=str,
            initial_literal_int_constants=self.initial_literal_int_constants,
            required_sizes=DynamicIndexSizes(
                global_arrays=global_arrays,
                function_arrays=function_arrays,
                functions=functions,
                constants=self.visible_literal_int_constants,
                target="metal",
            ),
        )

    def collect_unsized_resource_globals(self, ast):
        globals_by_name = {}
        for node in [
            *(getattr(ast, "global_variables", []) or []),
            *collect_stage_local_variables(ast),
        ]:
            name = getattr(node, "name", getattr(node, "variable_name", None))
            vtype = getattr(node, "var_type", getattr(node, "vtype", None))
            if name and self.is_unsized_resource_array_type(vtype):
                globals_by_name[name] = vtype
        return globals_by_name

    def collect_fixed_resource_global_sizes(self, ast):
        global_arrays = {}
        for node in [
            *(getattr(ast, "global_variables", []) or []),
            *collect_stage_local_variables(ast),
        ]:
            name = getattr(node, "name", getattr(node, "variable_name", None))
            vtype = getattr(node, "var_type", getattr(node, "vtype", None))
            size = self.fixed_resource_array_size(vtype)
            if name and size is not None:
                global_arrays[name] = size
        return global_arrays

    def collect_unsized_resource_parameters(self, ast):
        function_arrays = {}
        for func in self.all_functions(ast):
            func_name = getattr(func, "name", None)
            if not func_name:
                continue
            for param in getattr(func, "parameters", getattr(func, "params", [])):
                vtype = getattr(param, "param_type", getattr(param, "vtype", None))
                if self.is_unsized_resource_array_type(vtype):
                    function_arrays.setdefault(func_name, {})[param.name] = vtype
        return function_arrays

    def collect_fixed_resource_parameter_sizes(self, ast):
        function_arrays = {}
        for func in self.all_functions(ast):
            func_name = getattr(func, "name", None)
            if not func_name:
                continue
            for param in getattr(func, "parameters", getattr(func, "params", [])):
                size = self.fixed_resource_array_size(
                    getattr(param, "param_type", getattr(param, "vtype", None))
                )
                if size is not None:
                    function_arrays.setdefault(func_name, {})[param.name] = size
        return function_arrays

    def fixed_resource_array_size(self, vtype):
        if hasattr(vtype, "element_type") and str(type(vtype)).find("ArrayType") != -1:
            if vtype.size is None:
                return None
            base_type = self.convert_type_node_to_string(vtype.element_type)
            if not self.is_resource_array_hint_type(base_type):
                return None
            size = self.literal_int_value(vtype.size, self.literal_int_constants)
            return size if size is not None and size > 0 else None
        if hasattr(vtype, "name") or hasattr(vtype, "element_type"):
            return None
        type_string = str(vtype)
        if "[" not in type_string or "]" not in type_string:
            return None
        base_type, size = parse_array_type(type_string)
        if size is None or not self.is_resource_array_hint_type(base_type):
            return None
        return max(size, 1)

    def is_unsized_resource_array_type(self, vtype):
        if hasattr(vtype, "element_type") and str(type(vtype)).find("ArrayType") != -1:
            if vtype.size is not None:
                return False
            base_type = self.convert_type_node_to_string(vtype.element_type)
            return self.is_resource_array_hint_type(base_type)
        if hasattr(vtype, "name") or hasattr(vtype, "element_type"):
            return False
        type_string = str(vtype)
        if "[" not in type_string or "]" not in type_string:
            return False
        base_type, size = parse_array_type(type_string)
        return size is None and self.is_resource_array_hint_type(base_type)

    def is_resource_array_hint_type(self, vtype):
        return (
            self.is_resource_parameter_type(vtype)
            or str(self.resource_base_type(vtype))
            in self.glsl_buffer_block_struct_names
        )

    def validate_supported_stage_types(self, ast, target_stage=None):
        """Refuse the requested stages Metal cannot write as entry points (:mod:`.metal_stage_refusals`)."""
        stage_names = {
            normalize_stage_name(stage_type)
            for stage_type in getattr(ast, "stages", {}) or {}
        }
        for func in getattr(ast, "functions", []) or []:
            qualifiers = getattr(func, "qualifiers", None)
            if qualifiers is None:
                qualifiers = [getattr(func, "qualifier", None)]
            stage_names.update(normalize_stage_name(q) for q in qualifiers if q)
        refuse_metal_stages(
            name for name in stage_names if stage_matches(target_stage, name)
        )

    def stage_entry_types(self):
        return {
            "vertex",
            "fragment",
            "compute",
            "geometry",
            "tessellation_control",
            "tessellation_evaluation",
            "mesh",
            "task",
            "amplification",
            "object",
            "ray_generation",
            "ray_intersection",
            "ray_closest_hit",
            "ray_any_hit",
            "ray_miss",
            "ray_callable",
            "intersection",
            "anyhit",
            "closesthit",
            "miss",
            "callable",
        }

    def function_has_explicit_stage_entry_attribute(self, func):
        if getattr(func, "preserve_stage_entry_name", False):
            return True
        return any(
            str(getattr(attr, "name", "")).lower() == "stage_entry"
            for attr in getattr(func, "attributes", []) or []
        )

    def function_stage_qualifier(self, func):
        qualifiers = getattr(func, "qualifiers", None)
        if qualifiers:
            return qualifiers[0]

        qualifier = getattr(func, "qualifier", None)
        if qualifier:
            return qualifier

        stage_entry_types = self.stage_entry_types()
        for attr in getattr(func, "attributes", []) or []:
            stage_name = normalize_stage_name(getattr(attr, "name", None))
            if stage_name in stage_entry_types:
                return stage_name
        return None

    def stage_entry_names(self, ast, target_stage=None):
        stage_entry_types = self.stage_entry_types()
        entries = collect_stage_entry_records(ast, target_stage, stage_entry_types)
        used_names = collect_stage_entry_reserved_function_names(
            ast, target_stage, stage_entry_types
        )
        return assign_stage_entry_names(entries, used_names, self.stage_entry_base_name)

    def metal_stage_entry_functions(self, ast, expected_stage_name, target_stage=None):
        if not stage_matches(target_stage, expected_stage_name):
            return []

        functions = []
        for func in getattr(ast, "functions", []) or []:
            if function_stage_name(func) == expected_stage_name:
                functions.append(func)

        for stage_type, stage in getattr(ast, "stages", {}).items():
            if normalize_stage_name(stage_type) != expected_stage_name:
                continue
            entry_point = getattr(stage, "entry_point", None)
            if entry_point is not None:
                functions.append(entry_point)
        return functions

    def collect_metal_stage_entry_parameter_struct_names(
        self, ast, target_stage, expected_stage_name
    ):
        struct_names = set()
        for func in self.metal_stage_entry_functions(
            ast, expected_stage_name, target_stage
        ):
            for parameter in (
                getattr(func, "parameters", getattr(func, "params", [])) or []
            ):
                struct_name = self.metal_parameter_user_struct_type(parameter)
                if struct_name is not None:
                    struct_names.add(struct_name)
        return struct_names

    def collect_metal_stage_entry_return_struct_names(
        self, ast, target_stage, expected_stage_name
    ):
        struct_names = set()
        for func in self.metal_stage_entry_functions(
            ast, expected_stage_name, target_stage
        ):
            struct_node = self.metal_return_struct(func)
            if struct_node is not None:
                struct_names.add(struct_node.name)
        return struct_names

    def collect_metal_mesh_output_struct_names(self, ast, target_stage, role):
        struct_names = set()
        for func in self.metal_stage_entry_functions(ast, "mesh", target_stage):
            for parameter in getattr(func, "parameters", None) or []:
                if self.metal_mesh_output_parameter_role(parameter) != role:
                    continue
                raw_type = getattr(parameter, "param_type", None)
                element = getattr(raw_type, "element_type", raw_type)
                struct_name = self.type_name_string(element)
                if struct_name in self.structs_by_name:
                    struct_names.add(struct_name)
        return struct_names

    def metal_parameter_user_struct_type(self, parameter):
        param_type = getattr(parameter, "param_type", getattr(parameter, "vtype", None))
        type_name = self.type_name_string(param_type)
        if not type_name:
            return None
        mapped_type = self.map_type(type_name)
        base_type, _array_suffix = split_array_type_suffix(str(mapped_type))
        struct_name = base_type.split("<", 1)[0].strip()
        if struct_name not in self.structs_by_name:
            return None
        return struct_name

    def metal_return_struct(self, func):
        return_type = self.type_name_string(
            getattr(func, "return_type", getattr(func, "vtype", None))
        )
        if not return_type:
            return None
        base_type = return_type.split("<", 1)[0].split("[", 1)[0].strip()
        return self.structs_by_name.get(base_type)

    def metal_default_struct_member_semantics(self, struct_node):
        struct_name = getattr(struct_node, "name", None)
        defaults = {}
        if struct_name in self.metal_vertex_entry_input_struct_names:
            defaults.update(
                self.metal_default_vertex_input_member_semantics(struct_node)
            )
        elif struct_name in self.metal_vertex_entry_output_struct_names:
            defaults.update(
                self.metal_default_vertex_output_member_semantics(struct_node)
            )
        elif struct_name in self.metal_fragment_entry_output_struct_names:
            defaults.update(
                self.metal_default_fragment_output_member_semantics(struct_node)
            )
        if struct_name in self.metal_stage_io_struct_names:
            defaults.update(
                self.metal_default_user_stage_io_member_semantics(struct_node, defaults)
            )
        return defaults

    def metal_default_vertex_input_member_semantics(self, struct_node):
        used_attributes = set()
        for member in getattr(struct_node, "members", []) or []:
            semantic = self.metal_vertex_input_location_semantic(member)
            if semantic is None:
                semantic = self.semantic_from_node(member)
            attribute_index = self.metal_attribute_index_from_semantic(semantic)
            if attribute_index is not None:
                span = self.metal_stage_io_member_attribute_span(member)
                used_attributes.update(range(attribute_index, attribute_index + span))

        defaults = {}
        next_attribute = 0
        for member in getattr(struct_node, "members", []) or []:
            member_name = getattr(member, "name", None)
            if not member_name:
                continue
            location_semantic = self.metal_vertex_input_location_semantic(member)
            if location_semantic is not None:
                defaults[member_name] = location_semantic
                continue
            if self.semantic_from_node(member) is not None:
                continue
            if not self.metal_can_default_stage_io_semantic(
                self.struct_member_raw_type(member)
            ):
                continue

            while next_attribute in used_attributes:
                next_attribute += 1
            defaults[member_name] = f"attribute({next_attribute})"
            span = self.metal_stage_io_member_attribute_span(member)
            used_attributes.update(range(next_attribute, next_attribute + span))
            next_attribute += span
        return defaults

    def explicit_location_attribute_index(self, node):
        for attr in getattr(node, "attributes", []) or []:
            if str(getattr(attr, "name", "")).lower() != "location":
                continue
            arguments = getattr(attr, "arguments", []) or []
            if len(arguments) != 1:
                continue
            location = self.binding_index_value(arguments[0])
            if location is not None:
                return location
        return None

    def metal_vertex_input_location_semantic(self, node):
        location = self.explicit_location_attribute_index(node)
        if location is None:
            return None
        return f"attribute({location})"

    def metal_vertex_input_location_attribute(self, node):
        semantic = self.metal_vertex_input_location_semantic(node)
        if semantic is None:
            return None
        return self.map_semantic(semantic)

    def metal_default_user_stage_io_member_semantics(
        self, struct_node, existing_defaults=None
    ):
        defaults = {}
        existing_defaults = existing_defaults or {}
        used_attributes = set()
        for member in getattr(struct_node, "members", []) or []:
            semantic = existing_defaults.get(getattr(member, "name", None))
            if semantic is None:
                semantic = self.semantic_from_node(member)
            attribute_index = self.metal_attribute_index_from_semantic(semantic)
            if attribute_index is not None:
                span = self.metal_stage_io_member_attribute_span(member)
                used_attributes.update(range(attribute_index, attribute_index + span))

        next_attribute = 0
        for member in getattr(struct_node, "members", []) or []:
            member_name = getattr(member, "name", None)
            if not member_name or member_name in existing_defaults:
                continue
            if not self.is_metal_user_stage_io_semantic(
                self.semantic_from_node(member)
            ):
                continue
            if not self.metal_can_default_stage_io_semantic(
                self.struct_member_raw_type(member)
            ):
                continue

            while next_attribute in used_attributes:
                next_attribute += 1
            defaults[member_name] = f"attribute({next_attribute})"
            span = self.metal_stage_io_member_attribute_span(member)
            used_attributes.update(range(next_attribute, next_attribute + span))
            next_attribute += span
        return defaults

    def metal_struct_member_effective_semantic(
        self, member, default_member_semantics, struct_name=None
    ):
        if struct_name in self.metal_fragment_entry_output_struct_names:
            depth = metal_conservative_depth_semantic(member)
            if depth is not None:
                return depth
            color = metal_fragment_output_color_semantic(self, member)
            if color is not None:
                return color
        semantic = self.semantic_from_node(member)
        if (
            struct_name in self.metal_union_layouts
            and semantic == "union_member_layout"
        ):
            semantic = None
        default_semantic = (default_member_semantics or {}).get(
            getattr(member, "name", None)
        )
        if semantic is None:
            return default_semantic
        if (
            struct_name in getattr(self, "metal_stage_io_struct_names", set())
            and default_semantic is not None
            and self.is_metal_user_stage_io_semantic(semantic)
        ):
            return default_semantic
        return semantic

    def is_metal_user_stage_io_semantic(self, semantic):
        if semantic is None:
            return False
        if self.metal_attribute_index_from_semantic(semantic) is not None:
            return False
        canonical = self.canonical_metal_semantic(semantic)
        if canonical is None:
            return False
        canonical = str(canonical)
        if canonical.startswith("attribute("):
            return False
        if canonical.startswith("color(") or canonical.startswith("depth("):
            return False
        return canonical not in {
            "amplification_id",
            "barycentric_coord",
            "clip_distance",
            "front_facing",
            "instance_id",
            "payload",
            "point_coord",
            "point_size",
            "position",
            "primitive_id",
            "render_target_array_index",
            "sample_id",
            "sample_mask",
            "stencil",
            "thread_index_in_threadgroup",
            "thread_position_in_grid",
            "thread_position_in_threadgroup",
            "threadgroup_position_in_grid",
            "threads_per_threadgroup",
            "vertex_id",
            "viewport_array_index",
        }

    def metal_stage_io_member_attribute_span(self, member):
        array_info = self.metal_stage_io_member_array_info(member)
        if array_info is not None:
            _element_type, size = array_info
            return size

        mapped_type = self.map_type(self.struct_member_raw_type(member))
        dimensions = self.metal_matrix_dimensions(mapped_type)
        if dimensions is not None:
            _component_prefix, columns, _rows = dimensions
            return columns
        return 1

    def metal_default_vertex_output_member_semantics(self, struct_node):
        defaults = {}
        for member in getattr(struct_node, "members", []) or []:
            member_name = getattr(member, "name", None)
            normalized_name = str(member_name).replace("_", "").lower()
            if normalized_name not in {
                "position",
                "clipposition",
                "glposition",
            }:
                continue
            if self.semantic_from_node(member) is not None:
                continue
            if self.map_type(self.struct_member_raw_type(member)) != "float4":
                continue
            defaults[member_name] = "position"
        return defaults

    def metal_vertex_output_position_fallback_member_name(self, struct_node):
        struct_name = getattr(struct_node, "name", None)
        if struct_name not in self.metal_vertex_entry_output_struct_names:
            return None
        if self.metal_vertex_output_has_position_semantic(struct_node):
            return None

        existing_names = {
            getattr(member, "name", None)
            for member in getattr(struct_node, "members", []) or []
        }
        return self.unique_metal_generated_name("__crossgl_position", existing_names)

    def metal_vertex_output_has_position_semantic(self, struct_node):
        default_semantics = self.metal_default_vertex_output_member_semantics(
            struct_node
        )
        for member in getattr(struct_node, "members", []) or []:
            semantic = self.semantic_from_node(member)
            if semantic is None:
                semantic = default_semantics.get(getattr(member, "name", None))
            if self.canonical_metal_semantic(semantic) == "position":
                return True
        return False

    def metal_vertex_output_position_fallback_member_for_type(self, return_type):
        base_type = (
            self.type_name_string(return_type).split("<", 1)[0].split("[", 1)[0].strip()
        )
        struct_node = self.structs_by_name.get(base_type)
        if struct_node is None:
            return None
        return self.metal_vertex_output_position_fallback_member_name(struct_node)

    def generate_metal_vertex_output_position_fallback_return(
        self, return_expr, indent_str
    ):
        fallback_member = self.metal_vertex_output_position_fallback_member_for_type(
            self.current_function_return_type
        )
        if fallback_member is None:
            return None

        return_type = self.map_type(self.current_function_return_type)
        reserved_names = set(self.local_variable_types)
        result_name = self.unique_metal_generated_name("_crossglReturn", reserved_names)
        return (
            f"{indent_str}{return_type} {result_name} = {return_expr};\n"
            f"{indent_str}{result_name}.{fallback_member} = "
            "float4(0.0, 0.0, 0.0, 1.0);\n"
            f"{indent_str}return {result_name};\n"
        )

    def metal_default_fragment_output_member_semantics(self, struct_node):
        defaults = {}
        used_color_indices = set()
        for member in getattr(struct_node, "members", []) or []:
            semantic = self.semantic_from_node(member)
            color_index = self.metal_color_index_from_semantic(semantic)
            if color_index is not None:
                used_color_indices.add(color_index)

        next_color_index = 0
        for member in getattr(struct_node, "members", []) or []:
            member_name = getattr(member, "name", None)
            if not member_name or self.semantic_from_node(member) is not None:
                continue
            mapped_type = self.map_type(self.struct_member_raw_type(member))
            normalized_name = str(member_name).replace("_", "").lower()
            if normalized_name in {"depth", "fragdepth", "glfragdepth"}:
                if mapped_type in {"float", "half"}:
                    defaults[member_name] = "depth(any)"
                continue
            if mapped_type != "float4":
                continue
            while next_color_index in used_color_indices:
                next_color_index += 1
            defaults[member_name] = f"color({next_color_index})"
            used_color_indices.add(next_color_index)
            next_color_index += 1
        return defaults

    def metal_can_default_stage_io_semantic(self, member_type):
        type_name = self.type_name_string(member_type)
        if not type_name:
            return False

        base_type, _array_size = parse_array_type(str(type_name))
        mapped_type = self.map_type(base_type)
        if mapped_type in self.structs_by_name:
            return False
        return not self.is_resource_parameter_type(mapped_type)

    def metal_color_index_from_semantic(self, semantic):
        return metal_semantic_index(self, semantic, "color")

    def metal_attribute_index_from_semantic(self, semantic):
        return metal_semantic_index(self, semantic, "attribute")

    def metal_attribute_index_from_attribute(self, attribute):
        if not attribute:
            return None
        match = re.search(r"\[\[\s*attribute\((\d+)\)\s*\]\]", str(attribute))
        return int(match.group(1)) if match else None

    def uses_metal_raytracing_namespace(self, ast, global_vars, functions):
        for stage_type in getattr(ast, "stages", {}) or {}:
            if normalize_stage_name(stage_type) in {
                "ray_generation",
                "ray_intersection",
                "ray_any_hit",
                "ray_closest_hit",
                "ray_miss",
                "ray_callable",
                "intersection",
                "anyhit",
                "closesthit",
                "miss",
                "callable",
            }:
                return True

        for variable in global_vars:
            vtype, _ = self.global_resource_shape(variable)
            if (
                self.is_acceleration_structure_type(vtype)
                or self.is_visible_function_table_type(vtype)
                or self.is_intersection_function_table_type(vtype)
            ):
                return True

        return any(
            self.metal_render_walks.body_nodes_of_type(
                func, (RayTracingOpNode, RayQueryOpNode)
            )
            for func in functions
        )

    def metal_mesh_stage_attribute(self, shader_type, func, execution_config=None):
        stage_keyword = "mesh" if shader_type == "mesh" else "object"
        attributes = [stage_keyword]
        threadgroup_limit = self.metal_stage_threadgroup_limit(
            func, shader_type, execution_config
        )
        if threadgroup_limit:
            attributes.append(f"max_total_threads_per_threadgroup({threadgroup_limit})")
        grid_limit = self.metal_stage_attribute_arguments(
            func, "max_total_threadgroups_per_mesh_grid"
        )
        if stage_keyword == "object" and grid_limit:
            if len(grid_limit) != 1:
                raise MetalRefusal(
                    "Metal object stage max_total_threadgroups_per_mesh_grid requires exactly one argument"
                )
            attributes.append(
                "max_total_threadgroups_per_mesh_grid("
                f"{self.attribute_value_to_string(grid_limit[0])})"
            )
        return f"[[{', '.join(attributes)}]]"

    def metal_ray_intersection_stage_attribute(self, func, raw_return_type):
        primitive_type = self.metal_ray_intersection_primitive_type(
            func, raw_return_type
        )
        return f"[[intersection({primitive_type})]]"

    def metal_ray_intersection_primitive_type(self, func, raw_return_type):
        for attr in getattr(func, "attributes", []) or []:
            attr_name = getattr(attr, "name", None)
            if not attr_name:
                continue
            normalized = str(attr_name).strip().lower().replace("-", "_")
            arguments = getattr(attr, "arguments", []) or []
            if normalized in {"bounding_box", "boundingbox"}:
                return "bounding_box"
            if normalized in {"triangle", "triangles"}:
                return "triangle"
            if normalized in {"intersection", "primitive_type"} and arguments:
                requested_value = self.attribute_value_to_string(arguments[0])
                if requested_value is None:
                    continue
                requested = str(requested_value).strip().lower().replace("-", "_")
                if requested in {"bounding_box", "boundingbox"}:
                    return "bounding_box"
                if requested in {"triangle", "triangles"}:
                    return "triangle"

        return "triangle" if raw_return_type == "bool" else "bounding_box"

    def metal_stage_threadgroup_limit(self, func, shader_type, execution_config=None):
        if shader_type not in {"mesh", "object", "task", "amplification"}:
            return None

        arguments = self.metal_stage_attribute_arguments(
            func, "max_total_threads_per_threadgroup"
        )
        if arguments:
            if len(arguments) != 1:
                raise MetalRefusal(
                    f"Metal {shader_type} stage max_total_threads_per_threadgroup "
                    "requires exactly one argument"
                )
            return self.attribute_value_to_string(arguments[0])

        if not execution_config:
            return None
        if execution_config.get(
            HOST_WORKGROUP_SIZE_KEY
        ) and not has_explicit_workgroup_size(execution_config):
            # A Metal mesh or object function has no threadgroup size of its own: the host passes
            # threadsPerObjectThreadgroup and threadsPerMeshThreadgroup when it draws. A stage imported from such a
            # function keeps the host's choice, so no limit is declared.
            return None
        return self.metal_local_size_threadgroup_total(execution_config)

    def metal_stage_attribute_arguments(self, func, attribute_name):
        for attr in getattr(func, "attributes", []) or []:
            if self.metal_stage_control_attribute_name(attr) == attribute_name:
                return getattr(attr, "arguments", []) or []
        return []

    def metal_parameter_qualifiers(self, parameter):
        qualifiers = []
        allowed_qualifiers = {
            "const",
            "in",
            "out",
            "inout",
            "point",
            "line",
            "triangle",
            "lineadj",
            "triangleadj",
        }
        for qualifier in getattr(parameter, "qualifiers", []) or []:
            normalized = str(qualifier).lower()
            if normalized in allowed_qualifiers:
                qualifiers.append(normalized)

        for attr in getattr(parameter, "attributes", []) or []:
            if getattr(attr, "name", None) != "primitive":
                continue
            arguments = getattr(attr, "arguments", []) or []
            if not arguments:
                continue
            primitive = self.attribute_value_to_string(arguments[0])
            normalized = str(primitive).lower()
            if normalized in allowed_qualifiers:
                qualifiers.append(normalized)
        return qualifiers

    def metal_parameter_is_array(self, parameter):
        raw_type = self.parameter_raw_type(parameter)
        if (
            hasattr(raw_type, "element_type")
            and str(type(raw_type)).find("ArrayType") != -1
        ):
            return True
        type_name = self.type_name_string(raw_type)
        return bool(type_name and "[" in type_name and "]" in type_name)

    def metal_parameter_array_count(self, parameter):
        raw_type = self.parameter_raw_type(parameter)
        if (
            hasattr(raw_type, "element_type")
            and str(type(raw_type)).find("ArrayType") != -1
        ):
            size = getattr(raw_type, "size", None)
            if size is None:
                return None
            return self.literal_int_value(size, self.literal_int_constants)

        type_name = self.type_name_string(raw_type)
        if not type_name or "[" not in type_name or "]" not in type_name:
            return None
        _base_type, array_size = parse_array_type(type_name)
        if array_size is None:
            return None
        return self.literal_int_value(array_size, self.literal_int_constants)

    def metal_tessellation_stage_attribute_value(self, func, attribute_name):
        arguments = self.metal_stage_attribute_arguments(func, attribute_name)
        if not arguments:
            return None
        if len(arguments) != 1:
            raise MetalRefusal(
                f"Metal tessellation stage attribute {attribute_name} requires "
                "exactly one argument"
            )
        return self.attribute_value_to_string(arguments[0])

    def metal_tessellation_domain(self, func):
        value = self.metal_tessellation_stage_attribute_value(func, "domain")
        if value is None:
            return None
        domain_name = str(value).strip().strip('"').lower()
        domain_map = {
            "tri": "triangle",
            "triangle": "triangle",
            "triangles": "triangle",
            "quad": "quad",
            "quads": "quad",
            "isoline": "isoline",
            "isolines": "isoline",
        }
        mapped = domain_map.get(domain_name)
        if mapped is None:
            raise MetalRefusal(
                f"Metal tessellation domain cannot be represented: {value}"
            )
        return mapped

    def validate_metal_domain_location_type(self, parameters):
        for parameter in parameters:
            if not self.metal_parameter_has_semantic(parameter, "SV_DomainLocation"):
                continue

            base_type, array_suffix = self.metal_parameter_mapped_base_and_array_suffix(
                parameter
            )
            if base_type is None:
                continue

            floating_scalar_bases = ("float", "half", "double")
            is_floating_type = False
            for scalar_base in floating_scalar_bases:
                if not base_type.startswith(scalar_base):
                    continue
                suffix = base_type[len(scalar_base) :]
                if suffix in {"", "2", "3", "4"}:
                    is_floating_type = True
                break

            if array_suffix or not is_floating_type:
                raise MetalRefusal(
                    "Metal tessellation_evaluation stage SV_DomainLocation "
                    f"parameter '{parameter.name}' must be a floating-point "
                    "scalar or vector"
                )

    def validate_metal_domain_location_components(self, func, parameters):
        domain = self.metal_tessellation_domain(func)
        expected_counts = {
            "triangle": 3,
            "quad": 2,
            "isoline": 2,
        }
        expected_count = expected_counts.get(domain)
        if expected_count is None:
            return

        for parameter in parameters:
            if not self.metal_parameter_has_semantic(parameter, "SV_DomainLocation"):
                continue

            component_count = self.metal_parameter_component_count(parameter)
            if component_count is None:
                continue
            if component_count != expected_count:
                raise MetalRefusal(
                    "Metal tessellation_evaluation stage SV_DomainLocation "
                    f"parameter '{parameter.name}' must have {expected_count} "
                    f"component(s) for {domain} domains, got {component_count}"
                )

    def validate_metal_scalar_integer_semantic_type(
        self, parameters, shader_type, semantic
    ):
        for parameter in parameters:
            if not self.metal_parameter_has_semantic(parameter, semantic):
                continue

            base_type, array_suffix = self.metal_parameter_mapped_base_and_array_suffix(
                parameter
            )
            if base_type is None:
                continue
            if array_suffix or base_type not in {"int", "uint"}:
                raise MetalRefusal(
                    f"Metal {shader_type} stage {semantic} parameter "
                    f"'{parameter.name}' must be scalar int or uint"
                )

    def validate_metal_scalar_integer_semantic_types(
        self, parameters, shader_type, semantics
    ):
        for semantic in semantics:
            self.validate_metal_scalar_integer_semantic_type(
                parameters, shader_type, semantic
            )

    def metal_parameter_has_semantic(self, parameter, expected_semantic):
        semantic = self.semantic_from_node(parameter)
        if semantic is None:
            return False
        return str(semantic).lower() == expected_semantic.lower()

    def metal_parameter_mapped_base_and_array_suffix(self, parameter):
        raw_type = self.parameter_raw_type(parameter)
        type_name = self.type_name_string(raw_type)
        if not type_name:
            return None, None
        mapped_type = self.map_type(type_name)
        return split_array_type_suffix(str(mapped_type))

    def metal_parameter_component_count(self, parameter):
        base_type, _array_suffix = self.metal_parameter_mapped_base_and_array_suffix(
            parameter
        )
        if base_type is None:
            return None

        scalar_bases = (
            "float",
            "half",
            "double",
            "int",
            "uint",
            "bool",
        )
        for scalar_base in scalar_bases:
            if not base_type.startswith(scalar_base):
                continue
            suffix = base_type[len(scalar_base) :]
            if suffix in {"2", "3", "4"}:
                return int(suffix)
            if suffix == "":
                return 1
        return None

    def metal_local_size_threadgroup_total(self, execution_config):
        values = [str(value).strip() for value in compute_local_size(execution_config)]
        literal_product = 1
        for value in values:
            try:
                literal_product *= int(value, 0)
            except ValueError:
                return " * ".join(f"({item})" for item in values)
        return str(literal_product)

    def metal_mesh_stage_output_config(
        self, shader_type, func, function_name, reserved_parameter_names
    ):
        if shader_type != "mesh":
            return None

        role_parameters = self.metal_mesh_output_role_parameters(func)
        if role_parameters:
            return self.explicit_metal_mesh_stage_output_config(
                func, function_name, reserved_parameter_names, role_parameters
            )

        max_vertices = self.metal_single_stage_attribute_argument(func, "max_vertices")
        max_primitives = self.metal_single_stage_attribute_argument(
            func, "max_primitives"
        )
        topology = self.metal_single_stage_attribute_argument(func, "outputtopology")
        if not (max_vertices and max_primitives and topology):
            return None

        vertex_type = self.unique_metal_generated_name(
            f"_CrossGLMetalMeshVertex_{function_name}",
            set(getattr(self, "structs_by_name", {})),
        )
        parameter_name = self.unique_metal_generated_name(
            "_crossglMeshOut", reserved_parameter_names
        )
        return {
            "parameter_name": parameter_name,
            "vertex_type": vertex_type,
            "max_vertices": max_vertices,
            "max_primitives": max_primitives,
            "topology": self.metal_mesh_topology(topology),
            "primitive_type": "void",
            "output_parameters": {},
            "generated_vertex_struct": True,
        }

    def explicit_metal_mesh_stage_output_config(
        self, func, function_name, reserved_parameter_names, role_parameters
    ):
        missing_roles = {"vertices", "indices"} - set(role_parameters)
        if missing_roles:
            missing = ", ".join(sorted(missing_roles))
            raise MetalRefusal(
                "Metal mesh output parameters require explicit "
                f"{missing} output role(s)"
            )

        topology = self.metal_single_stage_attribute_argument(func, "outputtopology")
        if not topology:
            raise MetalRefusal(
                "Metal mesh output parameters require an outputtopology attribute"
            )
        topology = self.metal_mesh_topology(topology)

        vertices = self.metal_mesh_output_array_parameter_info(
            role_parameters["vertices"], "vertices"
        )
        indices = self.metal_mesh_output_array_parameter_info(
            role_parameters["indices"], "indices"
        )
        primitives = None
        if "primitives" in role_parameters:
            primitives = self.metal_mesh_output_array_parameter_info(
                role_parameters["primitives"], "primitives"
            )
            if primitives["count"] != indices["count"]:
                raise MetalRefusal(
                    "Metal mesh primitives output count must match indices output count"
                )

        self.validate_metal_mesh_indices_output_type(indices, topology)
        parameter_name = self.unique_metal_generated_name(
            "_crossglMeshOut", reserved_parameter_names
        )
        output_parameters = {
            vertices["name"]: {**vertices, "role": "vertices"},
            indices["name"]: {**indices, "role": "indices"},
        }
        primitive_type = "void"
        if primitives is not None:
            primitive_type = primitives["element_type"]
            output_parameters[primitives["name"]] = {
                **primitives,
                "role": "primitives",
            }

        return {
            "parameter_name": parameter_name,
            "vertex_type": vertices["element_type"],
            "primitive_type": primitive_type,
            "max_vertices": vertices["count"],
            "max_primitives": indices["count"],
            "topology": topology,
            "output_parameters": output_parameters,
            "generated_vertex_struct": False,
            "function_name": function_name,
        }

    def metal_mesh_output_role_parameters(self, func):
        role_parameters = {}
        for parameter in getattr(func, "parameters", []) or []:
            role = self.metal_mesh_output_parameter_role(parameter)
            if role is None:
                continue
            if role in role_parameters:
                raise MetalRefusal(
                    f"Metal mesh output role '{role}' can be used by only one parameter"
                )
            role_parameters[role] = parameter
        return role_parameters

    def is_metal_mesh_output_parameter(self, shader_type, parameter):
        return (
            shader_type == "mesh"
            and self.metal_mesh_output_parameter_role(parameter) is not None
        )

    def metal_mesh_output_parameter_role(self, parameter):
        for attr in getattr(parameter, "attributes", []) or []:
            attr_name = getattr(attr, "name", None)
            if not attr_name:
                continue
            normalized = str(attr_name).lower()
            if normalized.startswith("metal_") or normalized.startswith("msl_"):
                normalized = normalized.split("_", 1)[1]
            if normalized in {"vertices", "indices", "primitives"}:
                return normalized
        return None

    def metal_mesh_output_array_parameter_info(self, parameter, role):
        raw_type = getattr(parameter, "param_type", getattr(parameter, "vtype", None))
        if hasattr(raw_type, "element_type") and hasattr(raw_type, "size"):
            element_type = self.map_type(self.type_name_string(raw_type.element_type))
            count = self.metal_mesh_output_array_size(raw_type.size)
        else:
            type_name = self.type_name_string(raw_type)
            element_type, array_suffix = split_array_type_suffix(type_name)
            element_type = self.map_type(element_type)
            count = self.metal_mesh_output_array_suffix_size(array_suffix)

        if not element_type or count is None:
            raise MetalRefusal(f"Metal mesh {role} output must be a fixed-size array")
        return {
            "name": parameter.name,
            "element_type": element_type,
            "count": count,
        }

    def metal_mesh_output_array_size(self, size):
        if size is None:
            return None
        if isinstance(size, int):
            return str(size)
        return self.attribute_value_to_string(size)

    def metal_mesh_output_array_suffix_size(self, array_suffix):
        if not array_suffix:
            return None
        if not (array_suffix.startswith("[") and array_suffix.endswith("]")):
            return None
        size = array_suffix[1:-1].strip()
        return size or None

    def validate_metal_mesh_indices_output_type(self, indices, topology):
        expected_width = {"point": 1, "line": 2, "triangle": 3}[topology]
        expected_type = "uint" if expected_width == 1 else f"uint{expected_width}"
        actual_type = indices["element_type"]
        if actual_type != expected_type:
            raise MetalRefusal(
                "Metal mesh indices output for "
                f"{topology} topology requires {expected_type}, got {actual_type}"
            )

    def validate_metal_mesh_output_usage(self, func):
        mesh_output = self.current_metal_mesh_output_config
        if mesh_output is None:
            return

        body = getattr(func, "body", [])
        statements = getattr(body, "statements", body if isinstance(body, list) else [])
        self.validate_metal_mesh_output_statement_sequence(
            statements,
            self.metal_mesh_output_flow_state(),
        )

    def metal_mesh_output_flow_state(
        self,
        counts_seen=False,
        counts=None,
        falls_through=True,
        terminator=None,
        exits=None,
    ):
        normalized_counts = None
        if counts is not None:
            normalized_counts = {
                "vertices": counts.get("vertices"),
                "primitives": counts.get("primitives"),
            }
        return {
            "counts_seen": counts_seen,
            "counts": normalized_counts,
            "falls_through": falls_through,
            "terminator": None if falls_through else terminator,
            "exits": list(exits or []),
        }

    def copy_metal_mesh_output_flow_state(
        self, state, falls_through=None, terminator=None, exits=None
    ):
        copied_falls_through = (
            state.get("falls_through", True) if falls_through is None else falls_through
        )
        return self.metal_mesh_output_flow_state(
            counts_seen=state.get("counts_seen", False),
            counts=state.get("counts"),
            falls_through=copied_falls_through,
            terminator=(
                None
                if copied_falls_through
                else (state.get("terminator") if terminator is None else terminator)
            ),
            exits=(list(state.get("exits", [])) if exits is None else exits),
        )

    def metal_mesh_output_exit_state(self, state, terminator=None):
        return self.metal_mesh_output_flow_state(
            counts_seen=state.get("counts_seen", False),
            counts=state.get("counts"),
            falls_through=False,
            terminator=(
                terminator if terminator is not None else state.get("terminator")
            ),
        )

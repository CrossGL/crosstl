"""Ray tracing in the Metal generator: trace ray and call shader, ray stage signatures and intersection results, acceleration structure and function table arguments with their unsupported array forms, atomic function calls and mesh output calls."""

from ..ast import (
    ArrayAccessNode,
    BinaryOpNode,
    MemberAccessNode,
    PointerType,
    ReferenceType,
    UnaryOpNode,
)
from ..refusals import MetalRefusal
from .array_utils import split_array_type_suffix
from .metal_ray_flags import metal_trace_ray_flag_statements
from .stage_utils import normalize_stage_name


class MetalRayTracingMixin:
    """Ray tracing members of ``MetalCodeGen``."""

    def generate_metal_trace_ray(self, raw_args, rendered_args):
        if len(rendered_args) != 11:
            return None

        acceleration_structure_argument = (
            self.metal_trace_ray_acceleration_structure_argument(
                raw_args[0], rendered_args[0]
            )
        )
        if acceleration_structure_argument["reason"] is not None:
            return self.unsupported_metal_ray_tracing_intrinsic(
                "TraceRay acceleration structure",
                acceleration_structure_argument["reason"],
            )

        acceleration_structure = acceleration_structure_argument["argument"]
        instance_mask = rendered_args[2]
        origin = rendered_args[6]
        min_distance = rendered_args[7]
        direction = rendered_args[8]
        max_distance = rendered_args[9]
        acceleration_structure_type = acceleration_structure_argument["type"]
        intersection_function_table = self.default_intersection_function_table(
            acceleration_structure_type
        )
        intersector_tags = (
            intersection_function_table["tags"]
            if intersection_function_table is not None
            else self.default_metal_intersector_tags(acceleration_structure_type)
        )
        ray_name = self.next_metal_temp_variable("ray")
        intersector_name = self.next_metal_temp_variable("intersector")
        intersection_name = self.next_metal_temp_variable("intersection")
        intersect_args = [ray_name, acceleration_structure]
        if acceleration_structure_type != "primitive_acceleration_structure":
            intersect_args.append(instance_mask)
        if intersection_function_table is not None:
            intersect_args.append(intersection_function_table["name"])
        payload_diagnostic = None
        payload_argument = self.metal_trace_ray_payload_argument(
            raw_args[10], rendered_args[10], intersection_function_table
        )
        if payload_argument["argument"] is not None:
            intersect_args.append(payload_argument["argument"])
        elif payload_argument["reason"] is not None:
            payload_diagnostic = self.unsupported_metal_ray_tracing_intrinsic(
                "TraceRay payload", payload_argument["reason"]
            )
        intersector_type = f"intersector<{intersector_tags}>"
        intersection_type = f"intersection_result<{intersector_tags}>"
        flag_statements, flags_reason = metal_trace_ray_flag_statements(
            intersector_name,
            raw_args[1],
            rendered_args[1],
            self.literal_int_constants,
            lambda: self.next_metal_temp_variable("ray_flags"),
        )
        if flags_reason is not None:
            return self.unsupported_metal_ray_tracing_intrinsic(
                "TraceRay ray flags", flags_reason
            )

        lines = [
            f"ray {ray_name}",
            f"{ray_name}.origin = {origin}",
            f"{ray_name}.direction = {direction}",
            f"{ray_name}.min_distance = {min_distance}",
            f"{ray_name}.max_distance = {max_distance}",
            f"{intersector_type} {intersector_name}",
            *flag_statements,
            (
                f"{intersection_type} {intersection_name} = "
                f"{intersector_name}.intersect({', '.join(intersect_args)})"
            ),
        ]
        if payload_diagnostic is not None:
            lines.append(payload_diagnostic)
        lines.append(f"(void){intersection_name}")
        return "\n".join(lines)

    def metal_trace_ray_payload_argument(
        self, raw_payload_arg, rendered_payload_arg, intersection_function_table
    ):
        if self.is_metal_trace_ray_null_payload(raw_payload_arg):
            return {"argument": None, "reason": None}
        if intersection_function_table is None:
            return {
                "argument": None,
                "reason": (
                    "payload forwarding requires a compatible "
                    "intersection_function_table"
                ),
            }
        payload_display_name = (
            self.assignment_target_display_name(raw_payload_arg)
            or self.expression_name(raw_payload_arg)
            or "<expr>"
        )
        payload_root_name = self.assignment_target_root_name(raw_payload_arg)
        if not payload_root_name or payload_root_name not in self.local_variable_types:
            return {
                "argument": None,
                "reason": "payload forwarding requires a thread-local payload lvalue",
            }
        if payload_root_name in self.current_metal_non_thread_payload_parameters:
            return {
                "argument": None,
                "reason": (
                    "payload forwarding requires a thread-local payload lvalue; "
                    "ray payload parameters use a non-thread address space"
                ),
            }
        payload_address_space = self.argument_address_space(raw_payload_arg)
        if payload_address_space not in {None, "thread"}:
            return {
                "argument": None,
                "reason": (
                    "payload forwarding requires a thread-local payload lvalue; "
                    f"payload '{payload_display_name}' uses {payload_address_space} "
                    "address space"
                ),
            }
        if payload_root_name in self.current_readonly_metal_parameters:
            readonly_reason = self.current_readonly_metal_parameter_reasons.get(
                payload_root_name, "readonly"
            )
            return {
                "argument": None,
                "reason": (
                    "payload forwarding requires a mutable thread-local struct "
                    f"payload lvalue; payload '{payload_display_name}' is "
                    f"{readonly_reason}"
                ),
            }
        payload_type = self.expression_result_type(raw_payload_arg)
        if self.metal_type_is_pointer_like(payload_type):
            return {
                "argument": None,
                "reason": (
                    "payload forwarding requires a thread-local struct payload "
                    f"lvalue; payload '{payload_display_name}' has pointer or "
                    "array type"
                ),
            }
        payload_struct_type = self.metal_payload_struct_type_name(payload_type)
        if payload_struct_type is None:
            return {
                "argument": None,
                "reason": (
                    "payload forwarding requires a thread-local struct payload "
                    "lvalue"
                ),
            }
        if (
            self.metal_ray_payload_parameter_types
            and payload_struct_type not in self.metal_ray_payload_parameter_types
        ):
            expected_label = " or ".join(sorted(self.metal_ray_payload_parameter_types))
            return {
                "argument": None,
                "reason": (
                    f"ray payload argument '{payload_display_name}' has type "
                    f"'{payload_struct_type}' but declared ray payload interface "
                    f"expects '{expected_label}'"
                ),
            }
        return {"argument": rendered_payload_arg, "reason": None}

    def is_metal_trace_ray_null_payload(self, payload_arg):
        literal_value = self.literal_int_value(payload_arg, self.literal_int_constants)
        return literal_value == 0

    def collect_metal_ray_payload_parameter_types(self, stages):
        payload_types = set()
        if not isinstance(stages, dict):
            return payload_types

        for stage_type, stage in stages.items():
            stage_name = normalize_stage_name(stage_type)
            if stage_name not in {
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
                continue

            entry_point = getattr(stage, "entry_point", None)
            if entry_point is not None:
                parameters = getattr(
                    entry_point, "parameters", getattr(entry_point, "params", [])
                )
                for parameter in parameters or []:
                    if self.metal_ray_semantic_role(parameter, stage_name) != "payload":
                        continue
                    payload_type = self.metal_parameter_payload_struct_type(parameter)
                    if payload_type is not None:
                        payload_types.add(payload_type)

            for local_var in getattr(stage, "local_variables", []) or []:
                if self.metal_ray_semantic_role(local_var, stage_name) != "payload":
                    continue
                payload_type = self.metal_parameter_payload_struct_type(local_var)
                if payload_type is not None:
                    payload_types.add(payload_type)

        return payload_types

    def metal_ray_semantic_role(self, node, shader_type=None):
        semantic = self.semantic_from_node(node)
        if not semantic:
            return None
        compact = "".join(ch for ch in str(semantic).lower() if ch.isalnum())
        if compact in {
            "payload",
            "raypayload",
            "raypayloadext",
            "raypayloadin",
            "raypayloadinext",
        }:
            return "payload"
        if compact in {"hitattribute", "hitattributeext"}:
            return "hit_attribute"
        if compact in {
            "callabledata",
            "callabledataext",
            "callabledatain",
            "callabledatainext",
        }:
            return "callable_data"
        if compact in {"ray", "rayext", "objectray", "objectrayext"}:
            return "ray"
        return None

    def is_metal_ray_stage(self, shader_type):
        return shader_type in {
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
        }

    def validate_metal_ray_stage_signature(self, func, shader_type, raw_return_type):
        if not self.is_metal_ray_stage(shader_type):
            return

        self.validate_metal_ray_hit_attribute_parameters(func, shader_type)
        if shader_type == "ray_generation":
            self.validate_metal_ray_generation_parameters(func, shader_type)
        if shader_type in {"ray_intersection", "intersection"}:
            self.validate_metal_ray_intersection_parameters(func, shader_type)
            self.validate_metal_ray_intersection_return_type(func, raw_return_type)

    def validate_metal_ray_generation_parameters(self, func, shader_type):
        parameters = getattr(func, "parameters", getattr(func, "params", [])) or []
        for parameter in parameters:
            parameter_name = getattr(parameter, "name", "<unnamed>")
            if self.metal_ray_semantic_role(parameter, shader_type) == "ray":
                raise MetalRefusal(
                    f"Metal ray_generation parameter '{parameter_name}' cannot "
                    "use @ray; construct Metal ray values inside the kernel"
                )

            raw_type = self.parameter_raw_type(parameter)
            if not self.is_plain_metal_ray_value_parameter_type(raw_type):
                continue
            type_name = self.type_name_string(raw_type) or str(raw_type)
            raise MetalRefusal(
                f"Metal ray_generation parameter '{parameter_name}' has "
                f"unsupported ray value type '{type_name}'; construct ray "
                "values inside the kernel or pass data through an explicitly "
                "bound buffer"
            )

    def validate_metal_ray_intersection_parameters(self, func, shader_type):
        parameters = getattr(func, "parameters", getattr(func, "params", [])) or []
        for parameter in parameters:
            ray_semantic_role = self.metal_ray_semantic_role(parameter, shader_type)
            if ray_semantic_role == "payload":
                continue
            if ray_semantic_role != "ray":
                if self.is_metal_ray_intersection_builtin_parameter(parameter):
                    continue
                if self.is_metal_ray_intersection_buffer_parameter(
                    parameter, shader_type
                ):
                    continue
                parameter_name = getattr(parameter, "name", "<unnamed>")
                raise MetalRefusal(
                    f"Metal ray_intersection parameter '{parameter_name}' is not "
                    "a supported intersection parameter; use @payload, a Metal "
                    "intersection builtin semantic, or an explicitly bound "
                    "device/constant buffer"
                )
                continue
            parameter_name = getattr(parameter, "name", "<unnamed>")
            raise MetalRefusal(
                f"Metal ray_intersection parameter '{parameter_name}' cannot use "
                "@ray; Metal intersection functions do not accept ray input "
                "parameters"
            )

    def is_plain_metal_ray_value_parameter_type(self, raw_type):
        if isinstance(
            raw_type, (PointerType, ReferenceType)
        ) or self.is_array_type_node(raw_type):
            return False

        type_name = self.type_name_string(raw_type)
        if type_name is None:
            return False
        type_name = str(type_name).strip()
        if not type_name:
            return False
        return (
            type_name == "ray"
            or self.requires_metal_builtin_ray_desc(type_name)
            or self.is_metal_ray_query_type_name(type_name)
        )

    def is_metal_ray_intersection_builtin_parameter(self, parameter):
        semantic = self.semantic_from_node(parameter)
        metal_semantic = self.canonical_metal_semantic(semantic)
        if metal_semantic is None:
            return False
        normalized = str(metal_semantic).strip().lower()
        return normalized in {
            "origin",
            "direction",
            "min_distance",
            "max_distance",
            "geometry_id",
            "primitive_id",
            "instance_id",
            "world_space_origin",
            "world_space_direction",
            "barycentric_coord",
            "object_to_world_transform",
            "world_to_object_transform",
        }

    def is_metal_ray_intersection_buffer_parameter(self, parameter, shader_type):
        raw_type = self.parameter_raw_type(parameter)
        if not (
            isinstance(raw_type, (PointerType, ReferenceType))
            or self.is_array_type_node(raw_type)
        ):
            return False
        if self.explicit_buffer_binding_index(parameter) is None:
            return False
        address_space = self.normalized_address_space(
            self.effective_parameter_address_space(
                raw_type,
                parameter,
                shader_type,
                default_for_stage_binding=False,
            )
        )
        return address_space in {"device", "constant"}

    def validate_metal_kernel_return_type(self, func, shader_type, raw_return_type):
        if self.map_type(raw_return_type) == "void":
            return
        function_name = getattr(func, "name", "<entry>")
        raise MetalRefusal(
            f"Metal {shader_type} entry point '{function_name}' must return void; "
            "kernel functions cannot return values"
        )

    def validate_metal_ray_hit_attribute_parameters(self, func, shader_type):
        allowed_stages = {"ray_any_hit", "ray_closest_hit", "anyhit", "closesthit"}
        parameters = getattr(func, "parameters", getattr(func, "params", [])) or []
        for parameter in parameters:
            if not self.is_metal_hit_attribute_parameter(parameter):
                continue
            if shader_type in allowed_stages:
                continue
            parameter_name = getattr(parameter, "name", "<unnamed>")
            raise MetalRefusal(
                "Metal @hit_attribute parameter "
                f"'{parameter_name}' is only valid on any-hit or closest-hit stages"
            )

    def validate_metal_ray_intersection_return_type(self, func, raw_return_type):
        primitive_type = self.metal_ray_intersection_primitive_type(
            func, raw_return_type
        )
        mapped_return_type = self.map_type(raw_return_type)
        if primitive_type == "triangle" and mapped_return_type == "bool":
            return

        members = self.metal_intersection_result_members(raw_return_type)
        if members is None and mapped_return_type == "void":
            # GL_EXT_ray_tracing and SPIR-V intersection shaders report hits
            # through calls; a Metal intersection function returns its result.
            self.refuse_lowering(
                f"{primitive_type} intersection stage: a void entry that reports "
                "hits with ReportHit has no lowering to a Metal intersection "
                "function result yet"
            )
            return
        if members is None:
            if primitive_type == "triangle":
                raise MetalRefusal(
                    "Metal triangle intersection stage must return bool or a "
                    "result struct"
                )
            raise MetalRefusal(
                "Metal "
                f"{primitive_type} intersection stage must return a result struct"
            )

        self.require_metal_intersection_result_member(
            members,
            "accept_intersection",
            "bool",
            primitive_type,
            raw_return_type,
        )
        if primitive_type == "bounding_box":
            self.require_metal_intersection_result_member(
                members,
                "distance",
                "float",
                primitive_type,
                raw_return_type,
            )

    def require_metal_intersection_result_member(
        self, members, role, expected_type, primitive_type, raw_return_type
    ):
        for member in members:
            if self.metal_intersection_result_member_role(member) != role:
                continue
            member_type = self.type_name_string(
                getattr(
                    member,
                    "member_type",
                    getattr(member, "vtype", getattr(member, "var_type", None)),
                )
            )
            if self.map_type(member_type) == expected_type:
                return

        raise MetalRefusal(
            "Metal "
            f"{primitive_type} intersection result '{raw_return_type}' requires "
            f"a {expected_type} @{role} member"
        )

    def metal_intersection_result_members(self, raw_return_type):
        struct_type = self.metal_payload_struct_type_name(raw_return_type)
        if struct_type is None:
            return None
        struct_node = self.structs_by_name.get(struct_type)
        if struct_node is None:
            return None
        return getattr(struct_node, "members", []) or []

    def metal_intersection_result_member_role(self, member):
        semantic = self.semantic_from_node(member)
        if not semantic:
            return None
        compact = "".join(ch for ch in str(semantic).lower() if ch.isalnum())
        if compact in {"acceptintersection", "acceptintersectionext"}:
            return "accept_intersection"
        if compact in {"distance", "hitdistance", "intersectiondistance"}:
            return "distance"
        return None

    def is_metal_hit_attribute_parameter(self, parameter):
        return self.metal_ray_semantic_role(parameter) == "hit_attribute"

    def metal_parameter_payload_struct_type(self, parameter):
        type_name = self.type_name_string(
            getattr(
                parameter,
                "param_type",
                getattr(parameter, "var_type", getattr(parameter, "vtype", None)),
            )
        )
        return self.metal_payload_struct_type_name(type_name)

    def metal_acceleration_structure_argument_type(self, expr):
        name = self.expression_name(expr)
        local_type = self.local_variable_types.get(name)
        if local_type and self.is_acceleration_structure_type(local_type):
            return self.map_resource_type_with_format(local_type)
        for (
            acceleration_structure_variable,
            _,
            mapped_type,
            _,
        ) in self.acceleration_structure_variables:
            if getattr(acceleration_structure_variable, "name", None) == name:
                return mapped_type
        return "instance_acceleration_structure"

    def metal_trace_ray_acceleration_structure_argument(self, raw_arg, rendered_arg):
        display_name = (
            self.assignment_target_display_name(raw_arg)
            or self.expression_name(raw_arg)
            or rendered_arg
            or "<expr>"
        )

        unsupported_array = self.unsupported_metal_acceleration_structure_array_reason(
            raw_arg
        )
        if unsupported_array is not None:
            return {
                "argument": None,
                "type": None,
                "reason": (
                    "acceleration structure argument "
                    f"'{display_name}' uses an acceleration_structure array; "
                    f"{unsupported_array}; TraceRay requires a single "
                    "acceleration_structure resource"
                ),
            }

        result_type = self.expression_result_type(raw_arg)
        if result_type is not None:
            return self.metal_trace_ray_acceleration_structure_argument_from_type(
                raw_arg, rendered_arg, display_name, result_type
            )

        name = self.expression_name(raw_arg)
        local_type = self.local_variable_types.get(name)
        if local_type is not None:
            return self.metal_trace_ray_acceleration_structure_argument_from_type(
                raw_arg, rendered_arg, display_name, local_type
            )

        for (
            acceleration_structure_variable,
            _,
            mapped_type,
            array_size,
        ) in self.acceleration_structure_variables:
            if getattr(acceleration_structure_variable, "name", None) != name:
                continue
            if array_size is not None and not isinstance(raw_arg, ArrayAccessNode):
                return {
                    "argument": None,
                    "type": None,
                    "reason": (
                        "acceleration structure argument "
                        f"'{display_name}' has pointer or array type; "
                        "TraceRay requires an acceleration_structure element"
                    ),
                }
            return {"argument": rendered_arg, "type": mapped_type, "reason": None}

        return {
            "argument": None,
            "type": None,
            "reason": (
                "acceleration structure argument "
                f"'{display_name}' must resolve to an acceleration_structure "
                "resource"
            ),
        }

    def metal_trace_ray_acceleration_structure_argument_from_type(
        self, raw_arg, rendered_arg, display_name, raw_type
    ):
        type_name = self.type_name_string(raw_type) or "<unknown>"
        if not self.is_acceleration_structure_type(raw_type):
            return {
                "argument": None,
                "type": None,
                "reason": (
                    "acceleration structure argument "
                    f"'{display_name}' has type '{type_name}' but TraceRay "
                    "requires an acceleration_structure resource"
                ),
            }

        if self.metal_type_is_pointer_like(raw_type):
            return {
                "argument": None,
                "type": None,
                "reason": (
                    "acceleration structure argument "
                    f"'{display_name}' has pointer or array type; TraceRay "
                    "requires an acceleration_structure element"
                ),
            }

        return {
            "argument": rendered_arg,
            "type": self.map_resource_type_with_format(raw_type),
            "reason": None,
        }

    def default_metal_intersector_tags(self, acceleration_structure_type):
        if acceleration_structure_type == "primitive_acceleration_structure":
            return "triangle_data"
        return "triangle_data, instancing"

    def default_intersection_function_table(self, acceleration_structure_type):
        candidates = []
        for (
            intersection_function_table_variable,
            _,
            intersection_function_table_type,
            _,
        ) in self.intersection_function_table_variables:
            name = getattr(intersection_function_table_variable, "name", None)
            tags = self.intersection_function_table_tags(
                intersection_function_table_type
            )
            if (
                name
                and tags
                and self.intersection_function_table_tags_match_acceleration_structure(
                    tags, acceleration_structure_type
                )
            ):
                candidates.append({"name": name, "tags": ", ".join(tags)})
        if len(candidates) == 1:
            return candidates[0]
        return None

    def intersection_function_table_tags(self, table_type):
        type_name = str(table_type)
        if "<" not in type_name or not type_name.endswith(">"):
            return []
        tags = type_name.split("<", 1)[1][:-1].strip()
        if not tags:
            return []
        return [tag.strip() for tag in tags.split(",") if tag.strip()]

    def intersection_function_table_tags_match_acceleration_structure(
        self, tags, acceleration_structure_type
    ):
        has_instancing = "instancing" in tags
        if acceleration_structure_type == "primitive_acceleration_structure":
            return not has_instancing
        return has_instancing

    def generate_metal_call_shader(self, raw_args, rendered_args):
        if len(raw_args) == 2:
            table = self.default_visible_function_table()
            if table is None:
                return None
            table_name = table["name"]
            table_type = table["type"]
            shader_index, callable_data = rendered_args
            raw_callable_data = raw_args[1]
        elif len(raw_args) == 3:
            unsupported_array = self.unsupported_metal_ray_function_table_array_reason(
                raw_args[0]
            )
            if unsupported_array is not None:
                return self.unsupported_metal_ray_tracing_intrinsic(
                    "CallShader", unsupported_array
                )
            table_name = rendered_args[0]
            table_type = self.visible_function_table_argument_type(raw_args[0])
            if not self.is_visible_function_table_type(table_type):
                table_display_name = (
                    self.assignment_target_display_name(raw_args[0])
                    or self.expression_name(raw_args[0])
                    or table_name
                    or "<expr>"
                )
                return self.unsupported_metal_ray_tracing_intrinsic(
                    "CallShader",
                    "explicit table argument "
                    f"'{table_display_name}' must be a "
                    "visible_function_table resource",
                )
            shader_index = rendered_args[1]
            callable_data = rendered_args[2]
            raw_callable_data = raw_args[2]
        else:
            return None

        callable_data_argument = self.metal_call_shader_callable_data_argument(
            raw_callable_data, callable_data, table_type
        )
        if callable_data_argument["reason"] is not None:
            return self.unsupported_metal_ray_tracing_intrinsic(
                "CallShader callable data", callable_data_argument["reason"]
            )

        callable_data = callable_data_argument["argument"]
        return f"{table_name}[{shader_index}]({callable_data})"

    def default_visible_function_table(self):
        tables = [
            {"name": getattr(variable, "name", None), "type": table_type}
            for variable, _, table_type, _ in self.visible_function_table_variables
            if getattr(variable, "name", None)
        ]
        if len(tables) == 1:
            return tables[0]
        return None

    def default_visible_function_table_name(self):
        table = self.default_visible_function_table()
        if table is None:
            return None
        return table["name"]

    def visible_function_table_argument_type(self, table_arg):
        table_name = self.expression_name(table_arg)
        for variable, _, table_type, _ in self.visible_function_table_variables:
            if getattr(variable, "name", None) == table_name:
                return table_type
        if table_name in self.local_variable_types:
            return self.local_variable_types[table_name]
        return self.expression_result_type(table_arg)

    def metal_call_shader_callable_data_argument(
        self, raw_callable_data, rendered_callable_data, table_type
    ):
        callable_data_name = (
            self.assignment_target_display_name(raw_callable_data) or "<expr>"
        )
        root_name = self.assignment_target_root_name(raw_callable_data)
        if not root_name or root_name not in self.local_variable_types:
            return {
                "argument": None,
                "reason": (
                    "callable data forwarding requires a thread-local "
                    "callable-data lvalue"
                ),
            }

        callable_data_address_space = self.argument_address_space(raw_callable_data)
        if callable_data_address_space not in {None, "thread"}:
            return {
                "argument": None,
                "reason": (
                    "callable data forwarding requires a thread-local "
                    "callable-data lvalue; callable data "
                    f"'{callable_data_name}' uses {callable_data_address_space} "
                    "address space"
                ),
            }

        if root_name in self.current_readonly_metal_parameters:
            readonly_reason = self.current_readonly_metal_parameter_reasons.get(
                root_name, "readonly"
            )
            return {
                "argument": None,
                "reason": (
                    "callable data forwarding requires a mutable thread-local "
                    "callable-data lvalue; callable data "
                    f"'{callable_data_name}' is {readonly_reason}"
                ),
            }

        callable_data_raw_type = self.expression_result_type(raw_callable_data)
        if self.metal_type_is_pointer_like(callable_data_raw_type):
            return {
                "argument": None,
                "reason": (
                    "callable data forwarding requires a thread-local struct "
                    f"callable-data lvalue; callable data '{callable_data_name}' "
                    "has pointer or array type"
                ),
            }

        callable_data_type = self.metal_payload_struct_type_name(callable_data_raw_type)
        if callable_data_type is None:
            return {
                "argument": None,
                "reason": (
                    "callable data forwarding requires a thread-local struct "
                    "callable-data lvalue"
                ),
            }

        table_payload_type = self.visible_function_table_payload_type(table_type)
        if table_payload_type is not None and table_payload_type != callable_data_type:
            return {
                "argument": None,
                "reason": (
                    f"visible_function_table expects callable data "
                    f"'{table_payload_type}' but argument '{callable_data_name}' "
                    f"has '{callable_data_type}'"
                ),
            }

        return {"argument": rendered_callable_data, "reason": None}

    def visible_function_table_payload_type(self, table_type):
        type_name = self.type_name_string(table_type)
        if not type_name or "<" not in type_name or not type_name.endswith(">"):
            return None
        table_kind, payload_type = type_name.split("<", 1)
        if table_kind not in {"visible_function_table", "visibleFunctionTable"}:
            return None
        payload_type = payload_type[:-1].strip()
        if payload_type.startswith("void(") and payload_type.endswith(")"):
            payload_type = payload_type[5:-1].strip()
            if not payload_type:
                return None
            payload_type = payload_type.split(",", 1)[0].strip()
        return self.metal_payload_struct_type_name(payload_type)

    def metal_payload_struct_type_name(self, payload_type):
        type_name = self.type_name_string(payload_type)
        if not type_name:
            return None
        type_name = self.strip_metal_type_qualifiers(type_name)
        if "[" in type_name and "]" in type_name:
            type_name, _ = split_array_type_suffix(type_name)
        return type_name if type_name in self.structs_by_name else None

    def strip_metal_type_qualifiers(self, type_name):
        type_name = str(type_name).strip()
        qualifiers = {
            "const",
            "constant",
            "device",
            "object_data",
            "ray_data",
            "thread",
            "threadgroup",
        }
        changed = True
        while changed:
            changed = False
            parts = type_name.split(None, 1)
            if len(parts) == 2 and parts[0] in qualifiers:
                type_name = parts[1].strip()
                changed = True
        while type_name.endswith("&") or type_name.endswith("*"):
            type_name = type_name[:-1].strip()
        return type_name

    def unsupported_metal_ray_tracing_intrinsic(self, operation, reason):
        return self.refuse_lowering(
            f"ray tracing intrinsic: {operation} - {reason}", "(void)0"
        )

    def unsupported_metal_acceleration_structure_array_diagnostic(self, name):
        return self.refuse_lowering(
            "ray tracing resource: arrays of acceleration_structure are not valid "
            f"Metal buffer parameters ({name})"
        )

    def unsupported_metal_acceleration_structure_array_alias_diagnostic(
        self, name, source_name
    ):
        return self.refuse_lowering(
            f"ray tracing resource: local acceleration_structure alias '{name}' "
            "cannot be initialized from unsupported acceleration_structure array "
            f"'{source_name}'"
        )

    def unsupported_metal_acceleration_structure_array_alias_source(self, expr):
        source_name = self.expression_name(expr)
        if source_name in self.unsupported_metal_acceleration_structure_array_variables:
            return source_name
        return None

    def unsupported_metal_acceleration_structure_array_reason(self, expr):
        resource_name = self.expression_name(expr)
        if (
            resource_name
            not in self.unsupported_metal_acceleration_structure_array_variables
        ):
            return None
        return (
            "arrays of acceleration_structure are not valid Metal buffer "
            f"parameters ({resource_name})"
        )

    def unsupported_metal_acceleration_structure_array_expression(self, expr):
        reason = self.unsupported_metal_acceleration_structure_array_reason(expr)
        if reason is None:
            return None
        return self.refuse_lowering(f"ray tracing resource: {reason}", "0")

    def metal_acceleration_structure_array_parameter_kind(self, vtype, node=None):
        if not self.is_acceleration_structure_type(vtype):
            return None
        array_resource = self.resource_array_parameter(vtype, node)
        if array_resource is None:
            return None
        resource_type, _array_size = array_resource
        return resource_type

    def unsupported_metal_ray_function_table_array_diagnostic(self, table_kind, name):
        return self.refuse_lowering(
            f"ray tracing resource: arrays of {table_kind} are not valid Metal "
            f"buffer parameters ({name})"
        )

    def unsupported_metal_ray_function_table_array_alias_diagnostic(
        self, name, source_name
    ):
        table_kind = self.unsupported_metal_ray_function_table_array_variables.get(
            source_name, "ray function table"
        )
        return self.refuse_lowering(
            f"ray tracing resource: local {table_kind} alias '{name}' cannot be "
            f"initialized from unsupported {table_kind} array '{source_name}'"
        )

    def metal_ray_function_table_kind(self, vtype):
        if self.is_visible_function_table_type(vtype):
            return "visible_function_table"
        if self.is_intersection_function_table_type(vtype):
            return "intersection_function_table"
        return None

    def metal_ray_function_table_array_parameter_kind(self, vtype, node=None):
        table_kind = self.metal_ray_function_table_kind(vtype)
        if table_kind is None:
            return None
        if self.resource_array_parameter(vtype, node) is None:
            return None
        return table_kind

    def unsupported_metal_ray_function_table_array_alias_source(self, expr):
        source_name = self.expression_name(expr)
        if source_name in self.unsupported_metal_ray_function_table_array_variables:
            return source_name
        return None

    def unsupported_metal_ray_function_table_array_reason(self, expr):
        table_name = self.expression_name(expr)
        table_kind = self.unsupported_metal_ray_function_table_array_variables.get(
            table_name
        )
        if table_kind is None:
            return None
        return (
            f"arrays of {table_kind} are not valid Metal buffer parameters "
            f"({table_name})"
        )

    def unsupported_metal_ray_function_table_array_expression(self, expr):
        reason = self.unsupported_metal_ray_function_table_array_reason(expr)
        if reason is None:
            return None
        return self.refuse_lowering(f"ray tracing resource: {reason}", "0")

    def unsupported_metal_ray_function_table_array_member_call(self, func_expr):
        if not isinstance(func_expr, MemberAccessNode):
            return None
        object_expr = getattr(
            func_expr, "object_expr", getattr(func_expr, "object", None)
        )
        return self.unsupported_metal_ray_function_table_array_expression(object_expr)

    def generate_atomic_function_call(self, func_name, args):
        if func_name in self.user_function_names:
            return None
        if not self.is_metal_atomic_function_name(func_name) or not args:
            return None

        args = list(args)
        rendered_args = [self.generate_expression(arg) for arg in args]
        args, rendered_args = self.strip_metal_atomic_memory_scope_argument(
            func_name, args, rendered_args
        )
        if self.metal_atomic_target_needs_address(args[0], rendered_args[0]):
            rendered_args[0] = f"&{rendered_args[0]}"
        if self.is_metal_atomic_compare_exchange_name(func_name) and len(args) >= 2:
            expected_diagnostic = (
                self.metal_compare_exchange_expected_address_space_diagnostic(args[1])
            )
            if expected_diagnostic is not None:
                return expected_diagnostic
            if self.metal_compare_exchange_expected_needs_address(
                args[1], rendered_args[1]
            ):
                rendered_args[1] = f"&{rendered_args[1]}"
        return (
            f"{self.metal_atomic_intrinsic_name(func_name)}({', '.join(rendered_args)})"
        )

    def unsupported_metal_buffer_resource_atomic_call(self, func_name, target, reason):
        if func_name in {"atomicLoad", "atomicStore", "atomicCompareExchangeWeak"}:
            raise MetalRefusal(f"Metal {func_name} {reason}")
        return_type = (
            self.expression_result_type(target) or self.current_expression_expected_type
        )
        zero_value = self.diagnostic_zero_value_for_type(return_type or "int")
        return self.refuse_lowering(f"buffer atomic: {func_name} {reason}", zero_value)

    def generate_metal_expected_compare_exchange(
        self, atomic_target, args, scalar_type, *, bitwise_float=False
    ):
        expected_type = self.map_type(self.expression_result_type(args[1]))
        if expected_type != scalar_type or self.argument_address_space(args[1]) not in {
            None,
            "thread",
        }:
            raise MetalRefusal(
                "Metal atomicCompareExchangeWeak requires matching thread expected storage"
            )
        expected = self.generate_expression(args[1])
        desired = self.generate_expression_with_expected(args[2], scalar_type)
        expected_pointer = f"&({expected})"
        if bitwise_float:
            expected_pointer = f"reinterpret_cast<thread uint*>({expected_pointer})"
            desired = f"as_type<uint>({desired})"
        return (
            f"atomic_compare_exchange_weak_explicit({atomic_target}, {expected_pointer}, "
            f"{desired}, memory_order_relaxed, memory_order_relaxed)"
        )

    def strip_metal_atomic_memory_scope_argument(self, func_name, args, rendered_args):
        expected_count = self.metal_atomic_expected_argument_count(func_name)
        if (
            expected_count is not None
            and len(args) == expected_count + 1
            and self.is_metal_atomic_memory_scope_argument(args[-1], rendered_args[-1])
        ):
            return args[:-1], rendered_args[:-1]
        return args, rendered_args

    def metal_atomic_expected_argument_count(self, func_name):
        if self.is_metal_atomic_compare_exchange_name(func_name):
            return 5
        if func_name == "atomic_load_explicit":
            return 2
        if func_name in {
            "atomic_fetch_add_explicit",
            "atomic_fetch_sub_explicit",
            "atomic_fetch_min_explicit",
            "atomic_fetch_max_explicit",
            "atomic_fetch_and_explicit",
            "atomic_fetch_or_explicit",
            "atomic_fetch_xor_explicit",
            "atomic_store_explicit",
            "atomic_exchange_explicit",
        }:
            return 3
        return None

    def is_metal_atomic_memory_scope_argument(self, arg, rendered_arg):
        scope_names = {
            "memory_scope_device",
            "memory_scope_grid",
            "memory_scope_invocation",
            "memory_scope_queuefamily",
            "memory_scope_queue_family",
            "memory_scope_simdgroup",
            "memory_scope_subgroup",
            "memory_scope_system",
            "memory_scope_thread",
            "memory_scope_threadgroup",
            "memory_scope_workgroup",
        }
        for candidate in (self.expression_name(arg), rendered_arg):
            if candidate is None:
                continue
            normalized = str(candidate).strip().lower()
            normalized = normalized.replace(" ", "")
            normalized = normalized.replace("::", "_").replace(".", "_")
            if normalized.startswith("metal_") or normalized.startswith("msl_"):
                normalized = normalized.split("_", 1)[1]
            if normalized in scope_names:
                return True
        return False

    def metal_atomic_target_needs_address(self, arg, rendered_arg):
        if self.is_metal_address_expression(arg, rendered_arg):
            return False
        target_type = self.expression_result_type(arg)
        if self.metal_type_is_pointer_like(target_type):
            return False
        return self.is_metal_atomic_value_type(target_type)

    def metal_compare_exchange_expected_needs_address(self, arg, rendered_arg):
        if self.is_metal_address_expression(arg, rendered_arg):
            return False
        expected_type = self.expression_result_type(arg)
        if self.metal_type_is_pointer_like(expected_type):
            return False
        if isinstance(arg, (ArrayAccessNode, MemberAccessNode)):
            return True
        return self.expression_name(arg) is not None

    def metal_compare_exchange_expected_address_space_diagnostic(self, arg):
        member_address_space = self.address_space_qualified_member_address_space(arg)
        if member_address_space is not None:
            address_space = member_address_space
            storage_name = self.assignment_target_display_name(arg)
        else:
            storage_name = self.atomic_expected_storage_root_name(arg)
            if storage_name is None:
                return None
            address_space = self.current_address_space_variables.get(storage_name)
            if address_space is None and storage_name in self.local_variable_types:
                address_space = "thread"
        if address_space in {None, "thread"}:
            return None
        return self.refuse_lowering(
            "atomic compare-exchange expected pointer: expected storage "
            f"'{storage_name}' uses {address_space} address space; Metal requires "
            "thread storage",
            "false",
        )

    def atomic_expected_storage_root_name(self, arg):
        if isinstance(arg, UnaryOpNode) and getattr(arg, "operator", None) == "&":
            return self.atomic_expected_storage_root_name(arg.operand)
        if isinstance(arg, BinaryOpNode) and getattr(arg, "operator", None) in {
            "+",
            "-",
        }:
            return self.atomic_expected_storage_root_name(
                getattr(arg, "left", None)
            ) or self.atomic_expected_storage_root_name(getattr(arg, "right", None))
        return self.assignment_target_root_name(arg)

    def is_metal_address_expression(self, arg, rendered_arg):
        if isinstance(arg, UnaryOpNode) and getattr(arg, "operator", None) == "&":
            return True
        return str(rendered_arg).lstrip().startswith("&")

    def metal_type_is_pointer_like(self, vtype):
        if self.is_structured_buffer_type(vtype):
            return True
        type_name = self.type_name_string(vtype)
        if not type_name:
            return False
        type_name = str(type_name).strip()
        return type_name.endswith("*") or "[" in type_name

    def is_metal_atomic_compare_exchange_name(self, func_name):
        return func_name in {
            "atomic_compare_exchange_weak_explicit",
            "atomic_compare_exchange_strong_explicit",
        }

    def metal_atomic_intrinsic_name(self, func_name):
        if func_name == "atomic_compare_exchange_strong_explicit":
            return "atomic_compare_exchange_weak_explicit"
        return func_name

    def is_metal_atomic_function_name(self, func_name):
        return func_name in {
            "atomic_fetch_add_explicit",
            "atomic_fetch_sub_explicit",
            "atomic_fetch_min_explicit",
            "atomic_fetch_max_explicit",
            "atomic_fetch_and_explicit",
            "atomic_fetch_or_explicit",
            "atomic_fetch_xor_explicit",
            "atomic_load_explicit",
            "atomic_store_explicit",
            "atomic_exchange_explicit",
            "atomic_compare_exchange_weak_explicit",
            "atomic_compare_exchange_strong_explicit",
        }

    def generate_metal_mesh_output_call(self, func_name, args):
        mesh_output = self.current_metal_mesh_output_config
        output_parameter = self.current_metal_mesh_output_parameter
        if not mesh_output or not output_parameter:
            return None

        if func_name == "SetVertex" and len(args) == 2:
            index = self.generate_expression(args[0])
            value = self.metal_mesh_vertex_output_value(args[1], mesh_output)
            return f"{output_parameter}.set_vertex({index}, {value})"

        if func_name == "SetPrimitive" and len(args) == 2:
            primitive_output = self.metal_mesh_primitive_output_call(
                output_parameter, args[0], args[1], mesh_output
            )
            if primitive_output is not None:
                return primitive_output
            return self.metal_mesh_index_output_write(
                output_parameter, func_name, args[0], args[1]
            )

        if func_name == "SetIndex" and len(args) == 2:
            return self.metal_mesh_index_output_write(
                output_parameter, func_name, args[0], args[1]
            )

        return None

    def generate_metal_mesh_output_assignment(
        self, target, rendered_value, operator, value_expr
    ):
        target_info = self.metal_mesh_output_assignment_target(target)
        if target_info is None:
            # A write through a member's component or element (``v[i].uv.x = ...``): the mesh object sets a
            # vertex or primitive as a whole, and the accumulators carry whole members only.
            inner = target
            while isinstance(inner, (MemberAccessNode, ArrayAccessNode)):
                inner = (
                    inner.object
                    if isinstance(inner, MemberAccessNode)
                    else getattr(inner, "array", None)
                )
                enclosing = self.metal_mesh_output_assignment_target(inner)
                if enclosing is not None:
                    return self.unsupported_metal_mesh_output_assignment_diagnostic(
                        enclosing,
                        "component writes of an output member are not supported",
                    )
            return None

        compound_member_assignment = (
            operator != "="
            and target_info["member"] is not None
            and target_info["role"] in {"vertices", "primitives"}
        )
        if operator != "=" and not compound_member_assignment:
            return self.unsupported_metal_mesh_output_assignment_diagnostic(
                target_info, "compound mesh output assignments are not supported"
            )

        role = target_info["role"]
        index_expr = target_info["index"]
        if role == "vertices":
            if target_info["member"] is not None:
                return self.generate_metal_mesh_single_member_output_assignment(
                    target_info, rendered_value, "set_vertex", operator
                )
            value = self.metal_mesh_vertex_output_value_from_rendered(
                value_expr, rendered_value, self.current_metal_mesh_output_config
            )
            index = self.generate_expression(index_expr)
            return f"{self.current_metal_mesh_output_parameter}.set_vertex({index}, {value})"

        if role == "indices":
            if target_info["member"] is not None:
                return self.unsupported_metal_mesh_output_assignment_diagnostic(
                    target_info, "index vector components must be written as a whole"
                )
            return self.metal_mesh_index_output_write(
                self.current_metal_mesh_output_parameter,
                "SetPrimitive",
                index_expr,
                value_expr,
            )

        if role == "primitives":
            if target_info["member"] is not None:
                return self.generate_metal_mesh_single_member_output_assignment(
                    target_info, rendered_value, "set_primitive", operator
                )
            index = self.generate_expression(index_expr)
            return (
                f"{self.current_metal_mesh_output_parameter}"
                f".set_primitive({index}, {rendered_value})"
            )

        return None

    def generate_metal_mesh_single_member_output_assignment(
        self, target_info, rendered_value, setter, operator="="
    ):
        output_info = target_info["output"]
        member_name = target_info["member"]
        element_type = output_info["element_type"]
        member_types = self.struct_member_types.get(element_type, {})
        if operator != "=" or list(member_types) != [member_name]:
            accumulator = self.metal_mesh_output_accumulator(target_info)
            if accumulator is None:
                reason = (
                    "compound member writes require an output accumulator"
                    if operator != "="
                    else "partial member writes require an output accumulator"
                )
                return self.unsupported_metal_mesh_output_assignment_diagnostic(
                    target_info, reason
                )
            index = self.generate_expression(target_info["index"])
            temp_name = accumulator["name"]
            return (
                f"{temp_name}.{member_name} {operator} {rendered_value}\n"
                f"{self.current_metal_mesh_output_parameter}"
                f".{setter}({index}, {temp_name})"
            )

        index = self.generate_expression(target_info["index"])
        value = f"{element_type}{{{rendered_value}}}"
        return (
            f"{self.current_metal_mesh_output_parameter}" f".{setter}({index}, {value})"
        )

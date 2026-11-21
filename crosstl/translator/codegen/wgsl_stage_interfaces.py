"""Stage interfaces of the WGSL generator: stage support checks, stage emission, stage output lowering, interface member locations and built-ins, vertex position outputs and workgroup size attributes."""

import copy
import re

from ..ast import (
    IdentifierNode,
    LiteralNode,
    MemberAccessNode,
    NamedType,
    UnaryOpNode,
    VariableNode,
)
from ..host_workgroup_size import require_static_workgroup_size
from ..refusals import WGSLRefusal
from .fragment_output_targets import frag_data_attribute_index
from .host_workgroup_size_ids import is_host_sized_workgroup
from .stage_utils import normalize_stage_name
from .wgsl_primitive_index import refuse_misplaced_primitive_index
from .wgsl_specialized_workgroup_size import (
    specialized_workgroup_size,
    workgroup_size_u32_values,
)
from .wgsl_stage_io import (
    entry_parameter_locations,
    refuse_unmapped_system_semantic,
)


class WGSLStageInterfaceMixin:
    """Stage interface members of ``WGSLCodeGen``."""

    def validate_wgsl_stage_support(self, ast, target_stage=None):
        stages = set()
        normalized_target_stage = normalize_stage_name(target_stage)
        if normalized_target_stage:
            stages.add(normalized_target_stage)

        for func in getattr(ast, "functions", []) or []:
            qualifier = (
                func.qualifiers[0]
                if getattr(func, "qualifiers", None)
                else getattr(func, "qualifier", None)
            )
            stage_name = normalize_stage_name(qualifier)
            if stage_name:
                stages.add(stage_name)
        for stage_type, stage_node in getattr(ast, "stages", {}).items():
            stage_name = normalize_stage_name(stage_type)
            if stage_name:
                stages.add(stage_name)
            entry_point = getattr(stage_node, "entry_point", None)
            qualifier = (
                entry_point.qualifiers[0]
                if getattr(entry_point, "qualifiers", None)
                else None
            )
            entry_stage = normalize_stage_name(qualifier)
            if entry_stage:
                stages.add(entry_stage)

        unsupported = sorted(
            stage
            for stage in stages
            if stage in self.UNSUPPORTED_STAGE_NAMES
            or (stage and stage not in self.SUPPORTED_STAGE_NAMES)
        )
        if unsupported:
            raise WGSLRefusal(
                "WGSL target does not support shader stage(s): "
                + ", ".join(unsupported)
            )

    def generate_stage(self, stage_node, entry_name=None):
        stage_name = normalize_stage_name(getattr(stage_node, "stage", None))
        if stage_name not in self.SUPPORTED_STAGE_NAMES:
            raise WGSLRefusal(
                f"WGSL target does not support shader stage: {stage_name}"
            )

        entry_point = self.stage_output_lowered_entry_point(
            stage_node, stage_node.entry_point
        )
        refuse_misplaced_primitive_index(self, entry_point, stage_name)
        previous_stage = self._current_stage_name
        previous_workgroup_size = self._current_workgroup_size
        previous_function_return_type = self._current_function_return_type
        self._current_stage_name = stage_name
        self._current_workgroup_size = (
            workgroup_size_u32_values(
                self.stage_workgroup_size_config(stage_node),
                self.stage_workgroup_size_defaults(stage_node),
                self.workgroup_size_constants(),
            )
            if stage_name == "compute"
            else None
        )
        self._current_function_return_type = getattr(entry_point, "return_type", None)
        try:
            attributes = [f"@{stage_name}"]
            if stage_name == "compute":
                attributes.append(self.generate_workgroup_size_attribute(stage_node))
            stage_resource_parameter_ids = {
                id(parameter)
                for parameter in getattr(entry_point, "parameters", [])
                if self.is_stage_resource_parameter(parameter)
            }
            self.push_identifier_scope(
                getattr(param, "name", "")
                for param in getattr(entry_point, "parameters", [])
            )
            self.register_parameter_value_types(entry_point)
            self.push_pointer_identifier_scope(
                self.buffer_pointer_parameter_names(entry_point)
            )
            previous_parameter_locations = self._entry_parameter_locations
            self._entry_parameter_locations = entry_parameter_locations(
                self, entry_point, stage_name, stage_resource_parameter_ids
            )
            plan_pushed = False
            try:
                self.push_parameter_plan(entry_point, entry_point=True)
                plan_pushed = True
                signature = self.generate_function_signature(
                    entry_point,
                    name=entry_name or f"{stage_name}_main",
                    return_attributes=self.stage_return_attributes(
                        stage_name, entry_point
                    ),
                    leading_parameters=self.stage_implicit_builtin_parameters(
                        entry_point, stage_name
                    ),
                    skip_parameter_ids=stage_resource_parameter_ids,
                    parameter_attributes=True,
                )
                body = self.generate_block(entry_point.body, indent=0)
                body = self.add_generated_match_fallback_return(
                    body,
                    getattr(entry_point, "body", None),
                    getattr(entry_point, "return_type", None),
                )
                body = self.prepend_parameter_copies(body, entry_point)
            finally:
                self._entry_parameter_locations = previous_parameter_locations
                if plan_pushed:
                    self.pop_parameter_plan()
                self.pop_pointer_identifier_scope()
                self.pop_identifier_scope()
            overrides = (
                self.stage_workgroup_size_overrides(stage_node)
                if stage_name == "compute"
                else []
            )
            return "\n".join(overrides + attributes + [f"{signature} {body}"])
        finally:
            self._current_stage_name = previous_stage
            self._current_workgroup_size = previous_workgroup_size
            self._current_function_return_type = previous_function_return_type

    def module_declaration_names(self, ast):
        """Names declared at module scope in any stage, which a generated structure must not reuse."""
        names = set()
        containers = [ast, *(getattr(ast, "stages", None) or {}).values()]
        fields = (
            "structs",
            "functions",
            "global_variables",
            "constants",
            "cbuffers",
            "local_variables",
            "local_functions",
            "local_structs",
            "local_cbuffers",
        )
        for container in containers:
            for field in fields:
                for node in getattr(container, field, None) or []:
                    name = getattr(node, "name", None)
                    if isinstance(name, str):
                        names.add(name)
            entry_point = getattr(container, "entry_point", None)
            if isinstance(getattr(entry_point, "name", None), str):
                names.add(entry_point.name)
        return names

    def unique_stage_output_struct_name(self, preferred, used_names):
        if preferred not in used_names:
            return preferred
        base = "SPIRVVertexOutput"
        if base not in used_names:
            return base
        index = 2
        while f"{base}{index}" in used_names:
            index += 1
        return f"{base}{index}"

    def unique_stage_output_local_name(self, entry_point):
        used_names = {
            getattr(parameter, "name", "")
            for parameter in getattr(entry_point, "parameters", []) or []
            if getattr(parameter, "name", "")
        }
        body = getattr(entry_point, "body", None)
        if body is not None and hasattr(body, "walk"):
            used_names.update(
                getattr(node, "name", "")
                for node in body.walk()
                if isinstance(node, VariableNode) and getattr(node, "name", "")
            )
        if "output" not in used_names:
            return "output"
        base = "spirvPositionOutput"
        if base not in used_names:
            return base
        index = 2
        while f"{base}{index}" in used_names:
            index += 1
        return f"{base}{index}"

    def stage_interface_member_locations(self, ast, target_stage, structs):
        structs_by_name = {
            getattr(struct, "name", ""): struct
            for struct in structs
            if getattr(struct, "name", "")
        }
        self._fragment_output_struct_names = set()
        interface_struct_names = self.stage_interface_struct_names(
            ast, target_stage, structs_by_name
        )
        self._stage_interface_struct_names = set(interface_struct_names)
        locations = {}
        for struct in structs:
            struct_name = getattr(struct, "name", "")
            if struct_name not in interface_struct_names:
                continue

            used_locations = set()
            unbound_members = []
            for member in getattr(struct, "members", []) or []:
                if (struct_name, member.name) in self._stage_interface_builtin_members:
                    continue
                binding_kind, location = self.member_stage_binding(
                    member, self.struct_member_direction(struct_name)
                )
                slot_count = self.stage_interface_location_slot_count(
                    getattr(member, "member_type", None)
                )
                if binding_kind == "location":
                    if location is not None:
                        used_locations.update(range(location, location + slot_count))
                    continue
                if binding_kind == "builtin":
                    continue
                refuse_unmapped_system_semantic(
                    self, member, f"stage interface member {struct_name}.{member.name}"
                )
                unbound_members.append(member)

            next_location = 0
            for member in unbound_members:
                slot_count = self.stage_interface_location_slot_count(
                    getattr(member, "member_type", None)
                )
                while any(
                    location in used_locations
                    for location in range(next_location, next_location + slot_count)
                ):
                    next_location += 1
                locations[(struct_name, member.name)] = next_location
                used_locations.update(range(next_location, next_location + slot_count))
                next_location += slot_count
        return locations

    def stage_interface_location_slot_count(self, member_type):
        info = self.stage_interface_array_member_info_for_type(member_type)
        if info is not None:
            return info["size"]
        info = self.stage_interface_matrix_member_info_for_type(member_type)
        if info is not None:
            return info["columns"]
        return 1

    def stage_interface_builtin_members(self, ast, target_stage, structs_by_name):
        builtin_members = {}
        for stage_node in self._stage_nodes(ast, target_stage):
            stage_name = normalize_stage_name(getattr(stage_node, "stage", None))
            if stage_name != "vertex":
                continue
            entry_point = self.stage_output_lowered_entry_point(
                stage_node, getattr(stage_node, "entry_point", None)
            )
            if entry_point is None:
                continue
            struct_name = self.stage_interface_struct_type_name(
                getattr(entry_point, "return_type", None), structs_by_name
            )
            if not struct_name:
                continue
            struct = structs_by_name.get(struct_name)
            if struct is None:
                continue
            for member in getattr(struct, "members", []) or []:
                binding_kind, _location = self.member_stage_binding(member)
                if binding_kind is not None:
                    continue
                if self.is_vertex_position_member_name(getattr(member, "name", "")):
                    builtin_members[(struct_name, member.name)] = "position"
        return builtin_members

    def is_vertex_position_member_name(self, name):
        return self.semantic_key(name) in {"position", "clip_position", "clipposition"}

    def stage_interface_struct_names(self, ast, target_stage, structs_by_name):
        names = set()
        for stage_node in self._stage_nodes(ast, target_stage):
            stage_name = normalize_stage_name(getattr(stage_node, "stage", None))
            if stage_name not in {"vertex", "fragment"}:
                continue
            entry_point = self.stage_output_lowered_entry_point(
                stage_node, getattr(stage_node, "entry_point", None)
            )
            if entry_point is None:
                continue

            for parameter in getattr(entry_point, "parameters", []) or []:
                if self.is_stage_resource_parameter(parameter):
                    continue
                struct_name = self.stage_interface_struct_type_name(
                    getattr(parameter, "param_type", None), structs_by_name
                )
                if struct_name:
                    names.add(struct_name)

            struct_name = self.stage_interface_struct_type_name(
                getattr(entry_point, "return_type", None), structs_by_name
            )
            if struct_name:
                names.add(struct_name)
                if stage_name == "fragment":
                    self._fragment_output_struct_names.add(struct_name)
        return names

    def stage_interface_struct_type_name(self, type_node, structs_by_name):
        struct_name = self.struct_type_name(type_node)
        if struct_name in structs_by_name:
            return struct_name
        return None

    def member_stage_binding(self, member, direction="generic"):
        for attr in getattr(member, "attributes", []) or []:
            semantic = str(getattr(attr, "name", attr))
            key = self.semantic_key(semantic)
            if key == "builtin":
                return "builtin", None
            if key == "location":
                return "location", self.attribute_integer_argument(attr)
            if self.BUILTIN_SEMANTICS.get(key):
                return "builtin", None
            location = self.attribute_location(attr, semantic, direction)
            if location is not None:
                return "location", location
        return None, None

    def struct_member_direction(self, struct_name):
        """``"target"`` for the members of a structure a fragment entry point returns, else ``"generic"``."""
        if struct_name in self._fragment_output_struct_names:
            return "target"
        return "generic"

    def attribute_location(self, attr, semantic, direction):
        if self.semantic_key(semantic) == "gl_fragdata":
            # gl_FragData[n] names colour attachment n.
            attachment = frag_data_attribute_index(attr)
            if attachment is not None:
                return attachment
        return self.semantic_location(semantic, direction)

    def attribute_integer_argument(self, attr):
        arguments = getattr(attr, "arguments", []) or []
        if len(arguments) != 1:
            return None
        return self.integer_literal_value(arguments[0])

    def integer_literal_value(self, expr):
        if isinstance(expr, LiteralNode):
            value = str(expr.value)
            match = re.fullmatch(r"([+-]?(?:0[xX][0-9a-fA-F]+|\d+))[uUlL]*", value)
            if match:
                try:
                    return int(match.group(1), 0)
                except ValueError:
                    return None
        if (
            isinstance(expr, UnaryOpNode)
            and expr.operator == "+"
            and not expr.is_postfix
        ):
            return self.integer_literal_value(expr.operand)
        return None

    def vertex_position_output_variable(self, stage_node):
        if normalize_stage_name(getattr(stage_node, "stage", None)) != "vertex":
            return None
        for variable in getattr(stage_node, "local_variables", []) or []:
            if self.is_vertex_position_output_variable(variable):
                return variable
        return None

    def is_vertex_position_output_variable(self, variable):
        attribute_keys = {
            self.semantic_key(str(getattr(attribute, "name", attribute)))
            for attribute in getattr(variable, "attributes", []) or []
        }
        if "output" not in attribute_keys:
            return False
        return any(
            self.BUILTIN_SEMANTICS.get(attribute_key) == "position"
            for attribute_key in attribute_keys
        )

    def is_void_type(self, vtype):
        return self.semantic_key(str(getattr(vtype, "name", vtype))) == "void"

    def stage_output_lowered_entry_point(self, stage_node, entry_point):
        lowering = self._stage_output_lowerings.get(id(stage_node))
        if lowering is None:
            return entry_point
        if "entry_point" in lowering:
            return lowering["entry_point"]
        lowered = copy.copy(entry_point)
        lowered.return_type = NamedType(lowering["struct_name"])
        lowered.body = self.rewrite_stage_output_body(entry_point.body, lowering)
        return lowered

    def rewrite_stage_output_assignment(self, assignment, lowering):
        if not self.is_stage_output_assignment_target(assignment.target, lowering):
            return assignment
        rewritten = copy.copy(assignment)
        rewritten.target = MemberAccessNode(
            IdentifierNode(lowering["local_name"]), lowering["field_name"]
        )
        rewritten.left = rewritten.target
        return rewritten

    def is_stage_output_assignment_target(self, target, lowering):
        return (
            isinstance(target, IdentifierNode)
            and target.name == lowering["source_name"]
        )

    def generate_workgroup_size_attribute(self, stage_node):
        return (
            "@workgroup_size("
            + ", ".join(self.stage_workgroup_size_values(stage_node))
            + ")"
        )

    def stage_workgroup_size_config(self, stage_node):
        """The stage's execution metadata, specialized through overrides when the host chooses its size."""
        config = getattr(stage_node, "execution_config", {}) or {}
        return self._workgroup_size_overrides.effective_config(config)

    def stage_workgroup_size_values(self, stage_node):
        arguments, _ = specialized_workgroup_size(
            self.stage_workgroup_size_config(stage_node),
            self.stage_workgroup_size_defaults(stage_node),
            self.workgroup_size_constants(),
        )
        return arguments

    def stage_workgroup_size_overrides(self, stage_node):
        """Module-scope overrides of the workgroup dimensions the stage specializes that no entry declared yet."""
        return self._workgroup_size_overrides.new_declarations(
            self.stage_workgroup_size_config(stage_node),
            self.stage_workgroup_size_defaults(stage_node),
            self.workgroup_size_constants(),
        )

    def workgroup_size_constants(self):
        """``{id: (WGSL name, type)}`` of the program's own ``@id`` overrides, which a specialized dimension names."""
        return {
            spec_id: (self.module_identifier_name(name), type_name)
            for spec_id, (
                name,
                type_name,
            ) in self._specialization_plan.identified_overrides().items()
        }

    def stage_workgroup_size_defaults(self, stage_node):
        config = getattr(stage_node, "execution_config", {}) or {}
        if is_host_sized_workgroup(config):
            # The pipeline must set every dimension: its overrides have no initializer.
            return (None, None, None)
        require_static_workgroup_size(config)
        values = config.get("numthreads") or [
            config.get("local_size_x", "1"),
            config.get("local_size_y", "1"),
            config.get("local_size_z", "1"),
        ]
        if isinstance(values, str):
            values = [part.strip() for part in values.split(",")]
        values = list(values)[:3]
        while len(values) < 3:
            values.append("1")
        return tuple(str(value) for value in values)

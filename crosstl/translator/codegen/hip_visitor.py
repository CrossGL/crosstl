"""Includes, emission and struct ordering of the HIP generator, plus default values and stage-entry naming."""

from ..ast import (
    ArrayNode,
    EnumNode,
    FunctionNode,
    IdentifierNode,
    ShaderNode,
    StructNode,
    VariableNode,
)
from ..refusals import HipRefusal
from .array_utils import split_array_type_suffix
from .cuda_function_prototypes import (
    emission_order,
    late_called_functions,
    prototype_of,
)
from .cuda_shared_memory import helper_shared_variables
from .cuda_subgroup_inputs import is_subgroup_builtin_input
from .enum_utils import (
    enum_struct_fields,
    generate_enum_constants,
    generate_enum_constructor_functions,
    generate_enum_structs,
    generate_generic_enum_constants,
    generate_generic_enum_constructor_functions,
    generate_generic_enum_structs,
    generic_enum_specialized_fields,
)
from .generic_function_utils import (
    generic_function_emission_list,
    generic_function_parameters,
)
from .generic_struct_utils import (
    generate_generic_structs,
    generic_struct_specialized_fields,
    generic_struct_specialized_type_name,
)
from .stage_local_structs import module_structs
from .stage_utils import normalize_stage_name


class HipVisitorMixin:
    """Visitor members of ``HipCodeGen``."""

    def add_includes(self):
        """Emit the standard HIP runtime include block."""
        self.code_lines.extend(
            [
                "#include <hip/hip_runtime.h>",
                "#include <hip/hip_runtime_api.h>",
                "#include <hip/math_functions.h>",
                "#include <hip/device_functions.h>",
                "#include <hip/hip_fp16.h>",
                "",
            ]
        )

    def indent(self) -> str:
        """Return whitespace for the current indentation level."""
        return "    " * self.indent_level

    def add_line(self, line: str = ""):
        """Append one HIP output line using the current indentation level."""
        if line:
            self.code_lines.append(self.indent() + line)
        else:
            self.code_lines.append("")

    def add_generated_code(self, code: str):
        """Append pre-indented generated code to the output stream."""
        for line in code.rstrip("\n").splitlines():
            self.code_lines.append(line.rstrip())

    def ordered_generic_struct_specializations(self):
        specializations = self.generic_struct_specializations or {}
        if not specializations:
            return {}

        type_text_by_struct_name = {
            specialization["struct_name"]: type_text
            for type_text, specialization in specializations.items()
        }
        dependencies = {type_text: set() for type_text in specializations}
        for type_text, specialization in specializations.items():
            for _field_name, field_type in generic_struct_specialized_fields(
                self.type_name_string,
                specialization,
            ):
                dependency_name = generic_struct_specialized_type_name(
                    self,
                    field_type,
                )
                dependency_type = type_text_by_struct_name.get(dependency_name)
                if dependency_type and dependency_type != type_text:
                    dependencies[type_text].add(dependency_type)

        ordered = {}
        visiting = set()
        visited = set()
        type_order = {
            type_text: index for index, type_text in enumerate(specializations)
        }

        def visit(type_text):
            if type_text in visited or type_text in visiting:
                return
            visiting.add(type_text)
            for dependency_type in sorted(
                dependencies[type_text],
                key=lambda item: type_order[item],
            ):
                visit(dependency_type)
            visiting.remove(type_text)
            visited.add(type_text)
            ordered[type_text] = specializations[type_text]

        for type_text in specializations:
            visit(type_text)
        return ordered

    def mapped_type_dependency_name(self, type_value):
        mapped_type = self.map_type(type_value)
        base_type, _array_suffix = split_array_type_suffix(str(mapped_type or ""))
        base_type = base_type.strip()
        if base_type in {"int", "unsigned int", "float", "double", "bool", "void"}:
            return None
        return base_type or None

    def struct_node_dependency_names(self, node):
        dependencies = []
        for member in getattr(node, "members", []) or []:
            if isinstance(member, ArrayNode):
                member_type = getattr(
                    member,
                    "element_type",
                    getattr(member, "vtype", "float"),
                )
            elif hasattr(member, "member_type"):
                member_type = self.type_name_string(member.member_type)
            else:
                member_type = getattr(member, "vtype", "float")
            dependency = self.mapped_type_dependency_name(member_type)
            if dependency is not None:
                dependencies.append(dependency)
        return dependencies

    def order_struct_declaration_entries(self, entries):
        by_name = {}
        for entry in entries:
            by_name.setdefault(entry["name"], entry)

        order_index = {entry["name"]: index for index, entry in enumerate(entries)}
        dependencies = {
            entry["name"]: [
                dependency
                for dependency in entry["dependencies"]
                if dependency in by_name and dependency != entry["name"]
            ]
            for entry in entries
        }

        ordered = []
        visiting = set()
        visited = set()

        def visit(name):
            if name in visited or name in visiting:
                return
            visiting.add(name)
            for dependency in sorted(
                dependencies.get(name, []),
                key=lambda item: order_index[item],
            ):
                visit(dependency)
            visiting.remove(name)
            visited.add(name)
            ordered.append(by_name[name])

        for entry in entries:
            visit(entry["name"])
        return ordered

    def generate_struct_declaration_code(self, node):
        saved_lines = self.code_lines
        saved_indent = self.indent_level
        self.code_lines = []
        self.indent_level = 0
        try:
            self.visit_StructNode(node)
            return "\n".join(self.code_lines).rstrip() + "\n\n"
        finally:
            self.code_lines = saved_lines
            self.indent_level = saved_indent

    def generate_ordered_data_struct_declarations(self, data_struct_nodes):
        entries = []
        for node in data_struct_nodes:
            entries.append(
                {
                    "name": node.name,
                    "dependencies": self.struct_node_dependency_names(node),
                    "code": self.generate_struct_declaration_code(node),
                }
            )

        for specialization in self.ordered_generic_struct_specializations().values():
            entries.append(
                {
                    "name": specialization["struct_name"],
                    "dependencies": [
                        self.mapped_type_dependency_name(field_type)
                        for _field_name, field_type in (
                            generic_struct_specialized_fields(
                                self.type_name_string,
                                specialization,
                            )
                        )
                    ],
                    "code": generate_generic_structs(
                        self,
                        {specialization["type_name"]: specialization},
                    ),
                }
            )

        for enum in self.struct_payload_enums:
            fields = enum_struct_fields(enum) or []
            entries.append(
                {
                    "name": enum.name,
                    "dependencies": [
                        self.mapped_type_dependency_name(field_type)
                        for _field_name, field_type in fields
                    ],
                    "code": generate_enum_structs(self, [enum]),
                }
            )

        for type_text, specialization in self.generic_enum_specializations.items():
            entries.append(
                {
                    "name": specialization["struct_name"],
                    "dependencies": [
                        self.mapped_type_dependency_name(field_type)
                        for _field_name, field_type in generic_enum_specialized_fields(
                            self,
                            specialization,
                        )
                    ],
                    "code": generate_generic_enum_structs(
                        self,
                        {type_text: specialization},
                    ),
                }
            )

        code = ""
        for entry in self.order_struct_declaration_entries(entries):
            code += entry["code"]
        return code

    def default_value_expression_for_type(self, type_value):
        mapped_type = self.map_type(type_value)
        base_type, array_suffix = split_array_type_suffix(str(mapped_type or ""))
        if array_suffix:
            return None

        vector_info = self.vector_type_info(mapped_type)
        if vector_info is not None:
            scalar = (
                "0.0" if vector_info["component_type"] in {"float", "double"} else "0"
            )
            args = ", ".join([scalar] * len(vector_info["components"]))
            return f"{vector_info['constructor']}({args})"

        if base_type in self.struct_member_types:
            defaults = [
                self.default_value_expression_for_type(field_type)
                or self.primitive_default_value_expression(field_type)
                for field_type in self.struct_member_types[base_type].values()
            ]
            return f"{base_type}{{{', '.join(defaults)}}}"

        return None

    def primitive_default_value_expression(self, type_value):
        mapped_type = self.map_type(type_value)
        if mapped_type == "bool":
            return "false"
        if mapped_type in {"float", "double"}:
            return "0.0"
        if mapped_type in {"int", "unsigned int"}:
            return "0"
        return f"{mapped_type}{{}}"

    def is_hip_data_struct_node(self, node):
        if getattr(node, "is_cbuffer", False):
            return False
        for member in getattr(node, "members", []) or []:
            if isinstance(member, (EnumNode, FunctionNode, StructNode)):
                return False
        return True

    def visit_ShaderNode(self, node: ShaderNode) -> str:
        """Render a full shader/program AST as a HIP translation unit."""
        structs = module_structs(node, "HIP")
        self.add_generated_code(
            generate_enum_constants(
                self,
                self.plain_enums + self.struct_payload_enums,
                qualifier="static const",
            )
        )
        self.add_generated_code(
            generate_generic_enum_constants(
                self,
                self.generic_enum_struct_definitions,
                qualifier="static const",
            )
        )

        data_struct_nodes = []
        for struct in structs:
            if isinstance(struct, EnumNode):
                if struct in self.plain_enums:
                    self.visit(struct)
                continue
            if not isinstance(struct, StructNode):
                self.visit(struct)
                continue
            if struct.name in self.generic_enum_struct_definitions:
                continue
            if struct.name in self.generic_struct_definitions:
                continue
            if self.is_hip_data_struct_node(struct):
                data_struct_nodes.append(struct)
            elif getattr(struct, "is_cbuffer", False):
                self.visit(struct)

        self.add_generated_code(
            self.generate_ordered_data_struct_declarations(data_struct_nodes)
        )
        self.add_generated_code(
            generate_enum_constructor_functions(self, self.struct_payload_enums)
        )
        self.add_generated_code(
            generate_generic_enum_constructor_functions(
                self,
                self.generic_enum_specializations,
            )
        )

        for constant in getattr(node, "constants", []) or []:
            self.visit(constant)

        global_vars = getattr(node, "global_variables", [])
        for var in global_vars:
            self.visit(var)

        emitted_stage_local_variables = {
            getattr(var, "name", None): self.type_name_string(
                self.get_variable_node_type(var)
            )
            for var in global_vars
            if getattr(var, "name", None)
        }
        for stage_type, stage in (getattr(node, "stages", {}) or {}).items():
            stage_name = (
                str(stage_type).split(".")[-1].lower()
                if hasattr(stage_type, "name")
                else str(stage_type).lower()
            )
            self.file_scope_shared_variable_ids.update(
                id(var)
                for var in helper_shared_variables(stage, self.is_hip_shared_variable)
            )
            for var in getattr(stage, "local_variables", []) or []:
                if (
                    self.is_hip_shared_variable(var)
                    and id(var) not in self.file_scope_shared_variable_ids
                ):
                    continue
                if self.hip_stage_local_builtin_role(
                    stage_name, var
                ) is not None or is_subgroup_builtin_input(var):
                    # A stage-local input that names a compute or subgroup
                    # built-in is the built-in itself, not a variable to declare.
                    continue
                var_name = getattr(var, "name", None)
                var_type = self.type_name_string(self.get_variable_node_type(var))
                if not var_name:
                    continue
                previous_type = emitted_stage_local_variables.get(var_name)
                if previous_type is not None:
                    if previous_type != var_type:
                        raise HipRefusal(
                            "Conflicting HIP stage-local declaration for "
                            f"'{var_name}': {previous_type} differs from {var_type}"
                        )
                    self.register_variable_type(var_name, var_type, var)
                    continue
                self.visit(var)
                emitted_stage_local_variables[var_name] = var_type

        cbuffers = getattr(node, "cbuffers", [])
        for cbuffer in cbuffers:
            self.visit(cbuffer)

        functions = getattr(node, "functions", [])
        self.emit_late_function_prototypes(functions)
        for func in functions:
            if generic_function_parameters(func):
                for specialized_func in generic_function_emission_list(self, func):
                    self.visit(specialized_func)
                continue
            self.visit(func)

        # Handle shader stages (new AST structure)
        if hasattr(node, "stages") and node.stages:
            emitted_local_functions = set()
            stage_entry_name_counts = self.stage_entry_name_counts(node.stages)
            for stage_type, stage in node.stages.items():
                if hasattr(stage, "entry_point"):
                    # Set the stage type context for proper qualifier handling
                    stage_name = (
                        str(stage_type).split(".")[-1].lower()
                        if hasattr(stage_type, "name")
                        else str(stage_type).lower()
                    )

                    # Temporarily set qualifier for compute stages
                    if stage_name == "compute" or "compute" in stage_name:
                        # Set the function qualifier to compute for proper __global__ generation
                        if hasattr(stage.entry_point, "qualifiers"):
                            if "compute" not in stage.entry_point.qualifiers:
                                stage.entry_point.qualifiers.append("compute")
                        else:
                            stage.entry_point.qualifiers = ["compute"]

                    local_captures = self.collect_stage_local_function_captures(stage)
                    self.hip_function_capture_params.update(local_captures)
                    saved_stage_name = self.current_stage_name
                    saved_override = getattr(
                        self, "current_stage_entry_function_name", None
                    )
                    saved_stage_local_variables = getattr(
                        self, "current_stage_local_variables", []
                    )
                    try:
                        self.current_stage_name = saved_stage_name
                        self.emit_late_function_prototypes(
                            func
                            for func in getattr(stage, "local_functions", [])
                            if id(func) not in emitted_local_functions
                        )
                        for func in getattr(stage, "local_functions", []):
                            if id(func) in emitted_local_functions:
                                continue
                            if generic_function_parameters(func):
                                for specialized_func in generic_function_emission_list(
                                    self,
                                    func,
                                ):
                                    self.visit(specialized_func)
                                emitted_local_functions.add(id(func))
                                continue
                            self.visit(func)
                            emitted_local_functions.add(id(func))

                        self.current_stage_name = stage_name
                        self.current_stage_entry_function_name = (
                            self.stage_entry_function_name(
                                stage_name, stage.entry_point, stage_entry_name_counts
                            )
                        )
                        self.current_stage_local_variables = (
                            getattr(stage, "local_variables", []) or []
                        )
                        self.visit(stage.entry_point)
                    finally:
                        self.current_stage_entry_function_name = saved_override
                        self.current_stage_name = saved_stage_name
                        self.current_stage_local_variables = saved_stage_local_variables
                        for captured_name in local_captures:
                            self.hip_function_capture_params.pop(captured_name, None)

        return ""

    def emit_late_function_prototypes(self, functions):
        """Declare each function of ``functions`` that an earlier one calls (:mod:`.cuda_function_prototypes`)."""
        late = late_called_functions(
            *emission_order(
                functions,
                generic_function_parameters,
                lambda func: generic_function_emission_list(self, func),
            )
        )
        for func in late:
            self.visit(prototype_of(func))
        if late:
            self.add_line()

    def stage_entry_name_counts(self, stages):
        counts = {}
        for stage in stages.values():
            entry_point = getattr(stage, "entry_point", None)
            name = getattr(entry_point, "name", None)
            if name:
                counts[name] = counts.get(name, 0) + 1
        return counts

    def stage_entry_function_name(self, stage_name, entry_point, name_counts):
        name = getattr(entry_point, "name", None)
        if not name:
            return name
        if normalize_stage_name(stage_name) == "compute" and name == "main":
            return "compute_main"
        if stage_name in self.hip_ray_stage_names() and name == "main":
            return f"{stage_name}_{name}"
        if (
            stage_name in {
                "geometry",
                "tessellation_control",
                "tessellation_evaluation",
            } | self.hip_mesh_task_stage_names()
            and name == "main"
        ):
            return f"{stage_name}_{name}"
        if name_counts.get(name, 0) > 1:
            return f"{stage_name}_{name}"
        return name

    def collect_stage_local_function_captures(self, stage):
        entry_point = getattr(stage, "entry_point", None)
        entry_params = list(
            getattr(entry_point, "parameters", getattr(entry_point, "params", []))
        )
        entry_params_by_name = {
            getattr(param, "name", None): param for param in entry_params
        }
        entry_params_by_name = {
            name: param for name, param in entry_params_by_name.items() if name
        }
        if not entry_params_by_name:
            return {}

        captures = {}
        for function in getattr(stage, "local_functions", []) or []:
            if getattr(function, "body", None) is None:
                continue
            used_names = self.collect_function_free_identifier_names(function)
            captured_params = [
                entry_params_by_name[name]
                for name in entry_params_by_name
                if name in used_names
            ]
            if captured_params:
                captures[getattr(function, "name", None)] = captured_params
        return {name: params for name, params in captures.items() if name}

    def is_hip_shared_variable(self, node):
        """Return whether a stage-local variable maps to HIP shared memory."""
        return any(
            "shared" in qualifier_name or "workgroup" in qualifier_name
            for qualifier_name in (
                str(qualifier).lower()
                for qualifier in getattr(node, "qualifiers", []) or []
            )
        )

    def collect_function_free_identifier_names(self, function):
        params = getattr(function, "parameters", getattr(function, "params", []))
        bound_names = {getattr(param, "name", None) for param in params}
        bound_names = {name for name in bound_names if name}
        body = getattr(function, "body", None)
        local_names = {
            getattr(node, "name", None)
            for node in self.query_walk_nodes(body)
            if isinstance(node, VariableNode)
        }
        bound_names.update(name for name in local_names if name)
        return {
            node.name
            for node in self.query_walk_nodes(body)
            if isinstance(node, IdentifierNode) and node.name not in bound_names
        }

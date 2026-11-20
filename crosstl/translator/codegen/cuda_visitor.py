"""Visitor dispatch, emission and struct-metadata collection of the CUDA generator, plus the struct, enum and constant declarations."""

from ..ast import (
    AssignmentNode,
    BreakNode,
    ConstantNode,
    ContinueNode,
    FunctionCallNode,
    IdentifierNode,
    ReturnNode,
    VariableNode,
)
from ..refusals import CudaRefusal
from .cuda_constant_folding import device_constant_declaration
from .gpu_device_globals import at_file_scope
from .resource_arrays import format_array_declarator
from .stage_utils import normalize_stage_name


class CudaVisitorMixin:
    """Visitor members of ``CudaCodeGen``."""

    def visit(self, node):
        """Dispatch an AST node to its CUDA visitor method."""
        method_name = f"visit_{type(node).__name__}"
        visitor = getattr(self, method_name, self.generic_visit)
        return visitor(node)

    def generic_visit(self, node):
        """Fallback visitor for primitive values, lists, and unknown nodes."""
        if isinstance(node, str):
            builtin_alias = self.cuda_stage_builtin_alias_expression(node)
            if builtin_alias is not None:
                return builtin_alias
            if self.cuda_identifier_shadows_compute_builtin(node):
                return node
            ray_builtin = self.cuda_ray_builtin_expression(node)
            if ray_builtin is not None and node not in self.variable_types:
                return ray_builtin
            return self.cuda_identifier_spelling(node)
        elif isinstance(node, list):
            return [self.visit(item) for item in node]
        else:
            return str(node)

    def emit(self, code):
        """Append a line of CUDA output using the current indentation level."""
        if code.strip():
            self.output.append("    " * self.indent_level + code)
        else:
            self.output.append("")

    def emit_generated_code(self, code):
        """Append pre-indented generated code to the output stream."""
        for line in code.rstrip("\n").splitlines():
            self.output.append(line.rstrip())

    def emit_statement(self, node):
        """Render and append one statement node when it produces code."""
        if node is None:
            return

        if isinstance(node, AssignmentNode):
            self.emit_assignment_statement(node)
            return

        result = self.visit(node)
        if isinstance(result, str) and result.strip():
            self.emit(f"{result};")

    def emit_body(self, body):
        """Render a list-like or block-like function body."""
        if isinstance(body, list):
            for stmt in body:
                self.emit_statement(stmt)
        elif hasattr(body, "statements"):
            for stmt in body.statements:
                self.emit_statement(stmt)
        else:
            self.emit_statement(body)

    def statement_list(self, body):
        """Normalize block-like statement containers to a list."""
        if body is None:
            return []
        if isinstance(body, list):
            return body
        if hasattr(body, "statements"):
            return body.statements
        return [body]

    def collect_struct_member_metadata(self, root):
        """Collect struct member types and storage-image access before emission."""
        member_types_by_struct = {}
        member_accesses_by_struct = {}
        structs = list(getattr(root, "structs", []) or [])
        for stage in (getattr(root, "stages", {}) or {}).values():
            structs.extend(getattr(stage, "local_structs", []) or [])
        for struct in structs:
            struct_name = getattr(struct, "name", None)
            if not struct_name:
                continue
            member_types = {}
            member_accesses = {}
            for member in getattr(struct, "members", []) or []:
                member_name = getattr(member, "name", None)
                if not member_name:
                    continue
                if hasattr(member, "member_type"):
                    member_type = member.member_type
                elif hasattr(member, "vtype"):
                    member_type = member.vtype
                else:
                    member_type = "float"
                member_type = self.resource_type_with_access(member_type, member)
                member_types[member_name] = member_type
                member_access = self.explicit_resource_access(member)
                if member_access is not None and (
                    self.is_storage_image_type(member_type)
                    or self.is_buffer_resource_type(member_type)
                ):
                    member_accesses[member_name] = member_access
            member_types_by_struct[struct_name] = member_types
            member_accesses_by_struct[struct_name] = member_accesses
        return member_types_by_struct, member_accesses_by_struct

    def collect_struct_query_metadata_members(self, root):
        """Collect struct resource members that need embedded query sidecars."""
        has_resource_query = False
        for node in self.query_walk_nodes(root):
            if not isinstance(node, FunctionCallNode):
                continue
            if self.raw_function_call_name(node) in self.query_function_names:
                has_resource_query = True
                break
        if not has_resource_query:
            return {}

        metadata_members = {}
        for struct_name, member_types in self.struct_member_types.items():
            for member_name, member_type in member_types.items():
                if not self.is_queryable_resource_type(member_type):
                    continue
                metadata_members.setdefault(struct_name, set()).add(member_name)
        return metadata_members

    def statement_body_terminates(self, body):
        """Return true when the body already exits the active control flow."""
        statements = self.statement_list(body)
        if not statements:
            return False
        return isinstance(statements[-1], (BreakNode, ContinueNode, ReturnNode))

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
        if stage_name in self.cuda_ray_stage_names() and name == "main":
            return f"{stage_name}_{name}"
        if (
            stage_name in {
                "geometry",
                "tessellation_control",
                "tessellation_evaluation",
            } | self.cuda_mesh_task_stage_names()
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

    def is_cuda_shared_variable(self, node):
        """Return whether a stage-local variable maps to CUDA shared memory."""
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

    def visit_StructNode(self, node):
        self.emit(f"struct {node.name} {{")
        self.indent_level += 1

        members = getattr(node, "members", [])
        member_types = {}
        member_semantics = {}
        member_image_accesses = {}
        query_metadata_members = self.struct_query_metadata_members.get(
            node.name, set()
        )
        for member in members:
            if hasattr(member, "member_type"):
                member_type = member.member_type
            elif hasattr(member, "vtype"):
                member_type = member.vtype
            else:
                member_type = "float"

            member_type = self.resource_type_with_access(member_type, member)
            member_types[member.name] = member_type
            member_access = self.explicit_resource_access(member)
            if member_access is not None and (
                self.is_storage_image_type(member_type)
                or self.is_buffer_resource_type(member_type)
            ):
                member_image_accesses[member.name] = member_access
            semantic = self.semantic_from_node(member)
            if semantic:
                member_semantics[member.name] = semantic
                self.validate_cuda_builtin_semantic_type(
                    semantic, member_type, "struct member semantic"
                )
            semantic_comment = (
                f" // CrossGL semantic: {self.map_semantic(semantic)}"
                if semantic
                else ""
            )
            self.emit(
                f"{self.format_typed_declarator(member_type, member.name)};"
                f"{semantic_comment}"
            )
            if member.name in query_metadata_members:
                self.resource_query_info_required = True
                metadata_declarator = self.format_typed_declarator(
                    self.query_metadata_type(member_type),
                    self.query_metadata_name(member.name),
                    dynamic_array_as_pointer=False,
                )
                self.emit(f"{metadata_declarator};")

        self.struct_member_types[node.name] = member_types
        self.struct_member_semantics[node.name] = member_semantics
        self.struct_member_image_accesses[node.name] = member_image_accesses
        self.indent_level -= 1
        self.emit("};")

    def visit_EnumNode(self, node):
        self.emit(f"enum {node.name} {{")
        self.indent_level += 1

        variants = getattr(node, "variants", [])
        for index, variant in enumerate(variants):
            suffix = "," if index < len(variants) - 1 else ""
            value = getattr(variant, "value", None)
            if value is not None:
                self.emit(f"{variant.name} = {self.visit(value)}{suffix}")
            else:
                self.emit(f"{variant.name}{suffix}")

        self.indent_level -= 1
        self.emit("};")

    def visit_ConstantNode(self, node: ConstantNode):
        const_type = self.type_name_string(getattr(node, "const_type", "auto"))
        value = getattr(node, "value", None)
        self.register_variable_type(node.name, const_type, node, value)
        declaration_type = self.glsl_buffer_block_declaration_type(const_type, node)
        declaration = self.format_typed_declarator(declaration_type, node.name)
        if value is not None:
            declaration += f" = {self.visit(value)}"
        declaration = f"const {declaration}"
        if at_file_scope(self):
            declaration = device_constant_declaration(
                self, node.name, const_type, value, declaration, CudaRefusal
            )
        self.emit(f"{declaration};")

    def format_typed_declarator(self, type_name, name, dynamic_array_as_pointer=True):
        type_name = self.type_name_string(type_name)

        if "[" not in type_name or "]" not in type_name:
            return f"{self.convert_crossgl_type_to_cuda(type_name)} {name}"

        open_bracket = type_name.find("[")
        base_type = type_name[:open_bracket]
        array_suffix = type_name[open_bracket:]
        mapped_base = self.convert_crossgl_type_to_cuda(base_type)

        return format_array_declarator(
            mapped_base,
            name,
            array_suffix,
            dynamic_array_as_pointer=dynamic_array_as_pointer,
        )

    def format_array_size(self, size):
        if size is None:
            return ""
        if isinstance(size, int):
            return str(size)
        return self.visit(size)

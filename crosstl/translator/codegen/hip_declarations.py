"""Declarations of the HIP generator: structs, constant buffers, variables, constants and memory qualifiers."""

from ..ast import (
    CbufferNode,
    ConstantNode,
    MatchNode,
    StructNode,
    VariableNode,
)
from ..refusals import HipRefusal
from .cuda_constant_folding import (
    device_constant_declaration,
    device_variable_declaration,
    is_device_variable,
)
from .gpu_device_globals import (
    at_file_scope,
    constant_in_device_memory,
    file_scope_placement,
)
from .match_utils import (
    generate_match_expression_assignment,
    infer_match_expression_result_type,
)


class HipDeclarationMixin:
    """Declaration members of ``HipCodeGen``."""

    def visit_StructNode(self, node: StructNode) -> str:
        if getattr(node, "is_cbuffer", False):
            return self.emit_cbuffer_members_as_constant_memory(node)
        self.add_line(f"{self.native_unions.struct_keyword(node)} {node.name}")
        self.add_line("{")
        self.indent_level += 1

        members = getattr(node, "members", [])
        member_types = {}
        member_semantics = {}
        for member in members:
            if hasattr(member, "member_type"):
                member_type = member.member_type
            elif hasattr(member, "vtype"):
                member_type = member.vtype
            elif hasattr(member, "var_type"):
                member_type = member.var_type
            else:
                member_type = "float"

            member_type = self.resource_type_with_access(member_type, member)
            member_types[member.name] = member_type
            semantic = self.semantic_from_node(member)
            if semantic:
                member_semantics[member.name] = semantic
                self.validate_hip_builtin_semantic_type(
                    semantic, member_type, "struct member semantic"
                )
            semantic_comment = (
                f" // CrossGL semantic: {self.map_semantic(semantic)}"
                if semantic
                else ""
            )
            declaration = self.format_typed_declarator(member_type, member.name)
            declaration = self.apply_resource_memory_qualifiers_to_declaration(
                declaration, member.name, member
            )
            self.add_line(f"{declaration};{semantic_comment}")

        self.struct_member_types[node.name] = member_types
        self.struct_member_semantics[node.name] = member_semantics
        self.indent_level -= 1
        self.add_line("};")
        layout_assertion = self.native_unions.layout_assertion(node)
        if layout_assertion:
            self.add_line(layout_assertion)
        self.add_line()
        return ""

    def cbuffer_member_type(self, member):
        if hasattr(member, "member_type"):
            return member.member_type
        if hasattr(member, "vtype"):
            return member.vtype
        if hasattr(member, "var_type"):
            return member.var_type
        return "int"

    def emit_cbuffer_members_as_constant_memory(self, node) -> str:
        metadata_comment = self.hip_resource_metadata_comment(
            node, getattr(node, "name", None), kind="cbuffer"
        )
        if metadata_comment:
            self.add_line(metadata_comment)
        self.add_line(f"// Constant buffer: {node.name}")
        for member in getattr(node, "members", []) or []:
            member_type = self.cbuffer_member_type(member)
            declaration = self.format_typed_declarator(member_type, member.name)
            self.add_line(f"__constant__ {declaration};")
            # Record the member's type so expressions on it, such as vector
            # arithmetic, lower through the same helpers as other vectors.
            self.register_variable_type(
                member.name, self.type_name_string(member_type), member
            )
        self.add_line()
        return ""

    def visit_VariableNode(self, node: VariableNode) -> str:
        var_type = self.get_variable_node_type(node)
        initial_value = getattr(node, "initial_value", getattr(node, "value", None))
        if isinstance(initial_value, MatchNode):
            self.emit_match_expression_variable(node, initial_value, var_type)
            return ""

        metadata_comment = self.hip_resource_metadata_comment(node, var_type)
        if metadata_comment:
            self.add_line(metadata_comment)
        place = file_scope_placement(self, node, metadata_comment)
        declaration = self.format_variable_declaration(node)
        if place is constant_in_device_memory:
            declaration = device_constant_declaration(
                self, node.name, var_type, initial_value, declaration, HipRefusal
            )
        elif at_file_scope(self) and is_device_variable(node):
            declaration = device_variable_declaration(
                self, node.name, var_type, initial_value, declaration, HipRefusal
            )
        self.add_line(f"{place(declaration)};")
        metadata_declaration = self.query_metadata_declaration(node.name, var_type)
        if metadata_declaration:
            self.add_line(f"{place(metadata_declaration)};")
        length_declaration = self.structured_buffer_length_declaration(
            node.name, var_type
        )
        if length_declaration:
            self.add_line(f"{place(length_declaration)};")
        counter_declaration = self.structured_buffer_counter_declaration(
            node.name, var_type
        )
        if counter_declaration:
            self.add_line(f"{place(counter_declaration)};")
        return ""

    def visit_ConstantNode(self, node: ConstantNode) -> str:
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
                self, node.name, const_type, value, declaration, HipRefusal
            )
        self.add_line(f"{declaration};")
        return ""

    def variable_memory_qualifiers(self, node: VariableNode, declaration=""):
        qualifiers = []
        for qualifier in getattr(node, "qualifiers", []) or []:
            qualifier_name = str(qualifier).lower()
            if qualifier_name == "const":
                # Resource pointers already carry const from
                # apply_resource_memory_qualifiers_to_declaration.
                if not declaration.startswith("const "):
                    qualifiers.append("const")
            elif qualifier_name == "static":
                qualifiers.append("static")
            elif "workgroup" in qualifier_name or "shared" in qualifier_name:
                qualifiers.append("__shared__")
            elif "uniform" in qualifier_name:
                qualifiers.append("__constant__")
            elif qualifier_name == "device" and at_file_scope(self):
                # A program-scope variable in device memory, which every thread reads and writes.
                qualifiers.append("__device__")
        return qualifiers

    def declaration_qualifier_names(self, node):
        """Return normalized declaration qualifier and attribute names."""
        names = {
            str(qualifier).lower()
            for qualifier in getattr(node, "qualifiers", []) or []
        }
        for attr in getattr(node, "attributes", []) or []:
            attr_name = str(getattr(attr, "name", "")).lower()
            if attr_name:
                names.add(attr_name)
        return names

    def apply_resource_memory_qualifiers_to_declaration(self, declaration, name, node):
        """Apply HIP-native pointer qualifiers for resource-like declarations."""
        if "*" not in declaration or not name:
            return declaration

        names = self.declaration_qualifier_names(node)
        if not names:
            return declaration

        if (
            names & {"const", "constant", "readonly"}
            and "writeonly" not in names
            and not declaration.startswith("const ")
        ):
            declaration = f"const {declaration}"

        if "volatile" in names:
            if declaration.startswith("const "):
                declaration = f"const volatile {declaration[len('const '):]}"
            elif not declaration.startswith("volatile "):
                declaration = f"volatile {declaration}"

        if "restrict" in names and "__restrict__" not in declaration:
            name_token = f" {name}"
            if name_token in declaration:
                declaration = declaration.replace(
                    name_token, f" __restrict__{name_token}", 1
                )

        return declaration

    def emit_match_expression_variable(self, node, match_node, var_type):
        if var_type is None:
            var_type = infer_match_expression_result_type(self, match_node)
        var_type = var_type or "auto"
        self.register_variable_type(node.name, var_type, node)

        declaration = self.format_typed_declarator(var_type, node.name)
        declaration = self.apply_resource_memory_qualifiers_to_declaration(
            declaration, node.name, node
        )
        # The match arms assign the variable after this declaration, so it
        # cannot be const.
        qualifiers = [
            qualifier
            for qualifier in self.variable_memory_qualifiers(node, declaration)
            if qualifier != "const"
        ]
        if qualifiers:
            declaration = f"{' '.join(qualifiers)} {declaration}"

        self.add_line(f"{declaration};")
        self.add_generated_code(
            generate_match_expression_assignment(
                self,
                match_node,
                node.name,
                var_type,
                self.indent_level,
                "HIP",
            )
        )

    def visit_CbufferNode(self, node: CbufferNode) -> str:
        return self.emit_cbuffer_members_as_constant_memory(node)
